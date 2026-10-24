"""Reactive "follow the gap" driver: the starting point for every member.

Per lidar scan:
  1. clean the scan (inf -> max range, clip)
  2. zero a safety bubble around the nearest return so no target is chosen
     through a gap that passes too close to a wall
  3. aim into the widest remaining gap: a blend of its deepest beam and its middle
  4. pick a speed from the room ahead (a low percentile of the ranges inside the car's
     own corridor) and from how hard we are steering (a slow steering estimate)
  5. publish the speed and the steering angle on /drive

Everything that can jump beam-to-beam is rate-limited or low-passed, because the bridge
can run as slow as 10 Hz and raw gap targets weave full lock.

Inputs: /scan and /odom (the car's own speed). Output: /drive (AckermannDriveStamped,
speed in m/s, steering_angle in rad, at most +-0.5236). The same topics on the real car.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE,
                 history=QoSHistoryPolicy.KEEP_LAST, depth=10)
WHEELBASE = 0.324      # m
HALF_WIDTH = 0.135     # m
WHEEL_RADIUS = 0.059   # m
MAX_STEER = 0.5236     # rad, the simulator's steering limit
MAX_STEER_RATE = 3.2   # rad/s, the simulator's steering actuator
STEER_DELAY = 0.25     # s, command-to-wheel lag measured on this car


class ReactiveDriver(Node):
    def __init__(self, name="reactive_driver"):
        super().__init__(name)
        defaults = dict(bubble=0.40, gap_fov=1.60, range_cap=6.0, steer_gain=0.7, deep_weight=0.6,
                        max_speed=2.5, min_speed=0.9, accel_limit=4.0, decel_limit=6.0, lat_accel=6.0,
                        brake_margin=0.40, clear_pct=3.0, clear_tau=0.30, curv_tau=0.35, goal_tau=0.30, steer_tau=0.20,
                        follow_gap=0.0)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {k: self.get_parameter(k).value for k in defaults}

        self.speed = 0.0            # m/s, from /odom
        self.steer = 0.0            # steering command, rad
        self.steer_slow = 0.0       # slow steering estimate for the cornering speed
        self.clearance = None       # low-passed range ahead
        self.v_goal = 0.0           # low-passed speed goal
        self.speed_cmd = 0.0        # accel/decel-limited speed target
        self.last_scan_t = None

        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", QOS)
        self.create_subscription(LaserScan, "/scan", self.on_scan, QOS)
        self.create_subscription(Odometry, "/odom", self.on_odom, QOS)
        self.get_logger().info("reactive driver up")

    # ---------------------------------------------------------------- sensing

    def on_odom(self, msg):
        """Speed as the car reports it: the wheel odometry on the real car, the same thing
        dead-reckoned from the encoders in the simulator."""
        self.speed = msg.twist.twist.linear.x

    # ---------------------------------------------------------------- driving

    def on_scan(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:  # unstamped scan (replays, tests): fall back to the node clock
            t = self.get_clock().now().nanoseconds * 1e-9
        dt = t - self.last_scan_t if self.last_scan_t else 0.0
        self.last_scan_t = t
        if dt <= 0.0:
            return
        dt = min(dt, 0.5)   # a long gap (stalled simulator) still gets a command; never go blind

        ranges = np.asarray(msg.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = msg.range_max
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # follow mode (the house racer): a car within follow_gap straight ahead is not an obstacle
        # to steer around but something to sit behind. Its beams are filled with the corridor seen
        # beside it, so the line stays the corridor's, and the speed target keeps the distance.
        v_follow = None
        if self.p["follow_gap"] > 0.0:
            cone = np.abs(angles) <= math.radians(15.0)
            ahead = float(ranges[cone].min())
            if ahead < self.p["follow_gap"]:
                edges = np.flatnonzero(cone)
                ranges[cone] = max(ranges[edges[0]], ranges[edges[-1]])
                v_follow = max(0.5, 2.0 * (ahead - 0.5))

        target = self.gap_target(np.minimum(ranges, self.p["range_cap"]), angles)
        # the car keeps turning at the current steering for one actuator delay (plus the time
        # until the next scan) before this command acts: take that heading change off the target
        target -= self.speed * math.tan(self.steer) / WHEELBASE * (STEER_DELAY + dt)
        wanted = float(np.clip(self.p["steer_gain"] * target, -MAX_STEER, MAX_STEER))
        step = (wanted - self.steer) * lowpass(dt, self.p["steer_tau"])
        self.steer += float(np.clip(step, -MAX_STEER_RATE * dt, MAX_STEER_RATE * dt))

        ahead = self.room_ahead(ranges, angles, dt)
        self.steer_slow += (self.steer - self.steer_slow) * lowpass(dt, self.p["curv_tau"])
        curvature = abs(math.tan(self.steer_slow)) / WHEELBASE
        v_corner = math.sqrt(self.p["lat_accel"] / curvature) if curvature > 1e-3 else self.p["max_speed"]
        v_brake = math.sqrt(2.0 * self.p["decel_limit"] * max(0.0, ahead - self.p["brake_margin"]))
        goal = max(self.p["min_speed"], min(self.p["max_speed"], v_corner, v_brake))
        if v_follow is not None:
            goal = min(goal, v_follow)
        self.v_goal += (goal - self.v_goal) * lowpass(dt, self.p["goal_tau"])
        self.speed_cmd += float(np.clip(self.v_goal - self.speed_cmd, -self.p["decel_limit"] * dt, self.p["accel_limit"] * dt))

        self.publish(self.speed_cmd, self.steer)

    def gap_target(self, ranges, angles):
        """Bearing to aim at: inside the widest gap left after the safety bubble, between
        its deepest beam (deep_weight) and its middle."""
        fov = np.abs(angles) <= self.p["gap_fov"]
        nearest = int(np.argmin(np.where(fov, ranges, np.inf)))
        half_angle = math.atan2(self.p["bubble"], max(ranges[nearest], 0.05))
        free = np.where(fov & (np.abs(angles - angles[nearest]) >= half_angle), ranges, 0.0)
        good = np.flatnonzero(free > 0.0)
        if good.size == 0:
            return 0.0
        groups = np.split(good, np.flatnonzero(np.diff(good) > 1) + 1)
        gap = max(groups, key=len)
        deep = gap[int(np.argmax(free[gap]))]
        mid = gap[len(gap) // 2]
        return self.p["deep_weight"] * angles[deep] + (1.0 - self.p["deep_weight"]) * angles[mid]

    def room_ahead(self, ranges, angles, dt):
        """Low percentile of the forward range inside the car's own corridor, low-passed;
        a genuinely close obstacle passes straight through so braking is immediate."""
        x, y = ranges * np.cos(angles), ranges * np.sin(angles)
        corridor = (np.abs(y) < HALF_WIDTH + 0.10) & (x > 0.0)
        raw = float(np.percentile(x[corridor], self.p["clear_pct"])) if corridor.any() else self.p["range_cap"]
        self.clearance = raw if self.clearance is None else self.clearance + (raw - self.clearance) * lowpass(dt, self.p["clear_tau"])
        return min(raw, self.clearance) if raw < self.p["brake_margin"] + 0.3 else self.clearance

    def publish(self, speed, steer):
        """The contract: a speed in m/s and a steering angle in rad on /drive. The sim adapter
        (or the VESC driver on the car) turns them into throttle and servo."""
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed, cmd.drive.steering_angle = float(speed), float(steer)
        self.pub_drive.publish(cmd)


def lowpass(dt, tau):
    """Gain of a first-order low-pass with time constant tau at step dt."""
    return 1.0 - math.exp(-dt / max(tau, 1e-3))


def spin(node):
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(0.0, 0.0)
        node.destroy_node()


def main():
    rclpy.init()
    spin(ReactiveDriver())


if __name__ == "__main__":
    main()
