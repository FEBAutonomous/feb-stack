"""feb_reactive_competitor: a reactive walls driver meant to be clean on tight tracks and smooth
in corners, where the plain gap follower weaves. No map, no memory: every scan is judged alone.

Per scan:
  1. clean the scan, cap the ranges, smooth them a little
  2. disparity extension: at every jump between neighbouring beams the closer side is painted
     over the farther side for half a car width plus a margin, so a gap narrower than the car,
     a doorway, a cone's shadow, never looks like a path (the F1TENTH "disparity extender")
  3. target: the farthest reachable beam of the extended scan inside the field of view, with
     ties broken toward the previous target so the aim slides round a corner instead of jumping
     between walls
  4. steering: pure pursuit toward that point (geometry instead of a gain on the bearing), the
     actuator delay predicted away, then a side-clearance nudge, a low-pass and the actuator's
     rate limit
  5. speed: the lowest of the cap, what the steering curvature allows, what the room along the
     car's own corridor allows, and what the target's distance allows, slewed within accel and
     decel limits

Inputs: /scan and /odom. Output: /drive (AckermannDriveStamped). Same on the real car.
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
SCAN_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=1)        # the newest scan only, never a backlog
WHEELBASE = 0.324      # m
HALF_WIDTH = 0.135     # m
MAX_STEER = 0.5236     # rad
MAX_STEER_RATE = 3.2   # rad/s, the steering actuator


class Competitor(Node):
    def __init__(self, name="reactive_driver"):
        super().__init__(name)
        defaults = dict(range_cap=8.0, smooth=3, fov=1.40, disparity=0.30, margin=0.30,
                        tie=0.92, opening=0.6, center_weight=0.5,
                        lookahead_min=0.8, lookahead_gain=0.30, lookahead_max=1.8,
                        steer_delay=0.25, side_min=0.30, side_gain=0.8, steer_tau=0.12,
                        max_speed=3.0, min_speed=0.8, accel_limit=3.5, decel_limit=5.5, brake_decel=3.0, lat_accel=3.5,
                        brake_margin=0.45, clear_pct=3.0, corridor_extra=0.10, curv_tau=0.25, goal_tau=0.15,
                        stuck_after=1.0, reverse_speed=0.6, reverse_for=1.2, creep_speed=0.5)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {k: self.get_parameter(k).value for k in defaults}

        self.speed = 0.0
        self.steer = 0.0
        self.steer_slow = 0.0
        self.target_angle = 0.0
        self.v_goal = 0.0
        self.speed_cmd = 0.0
        self.last_scan_t = None
        self.blocked_since = None      # for the unstick: stopped against something
        self.reverse_until = None
        self.reverse_steer = 0.0

        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", QOS)
        self.create_subscription(LaserScan, "/scan", self.on_scan, SCAN_QOS)
        self.create_subscription(Odometry, "/odom", self.on_odom, QOS)
        self.get_logger().info("reactive competitor up")

    def on_odom(self, msg):
        self.speed = msg.twist.twist.linear.x

    # ---------------------------------------------------------------- the scan

    def on_scan(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        dt = t - self.last_scan_t if self.last_scan_t else 0.0
        self.last_scan_t = t
        if dt <= 0.0:
            return
        dt = min(dt, 0.5)

        ranges = np.asarray(msg.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = msg.range_max
        ranges = np.clip(ranges, msg.range_min, min(msg.range_max, self.p["range_cap"]))
        n = int(self.p["smooth"])
        if n > 1:
            kernel = np.ones(n) / n
            ranges = np.convolve(ranges, kernel, mode="same")
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        inc = msg.angle_increment

        ext = self.extend(ranges, inc)
        bearing, reach = self.target(ext, angles)
        ahead = self.room_along(ranges, angles, 0.0)
        ahead_bearing = self.room_along(ranges, angles, bearing)

        # stuck against something: stopped with nothing in front for a second -> back out, wheels
        # turned the other way from the target, then carry on
        if self.reverse_until is not None:
            if t < self.reverse_until:
                self.speed_cmd = -self.p["reverse_speed"]
                self.publish(self.speed_cmd, self.reverse_steer)
                return
            self.reverse_until = None
            self.speed_cmd = 0.0
        if min(ahead, ahead_bearing) < self.p["brake_margin"] and abs(self.speed) < 0.15:
            self.blocked_since = self.blocked_since or t
            if t - self.blocked_since > self.p["stuck_after"]:
                self.reverse_until = t + self.p["reverse_for"]
                self.reverse_steer = 0.0                   # straight back: the flanks are usually the problem
                self.blocked_since = None
                self.get_logger().warn("stuck: backing out for %.1f s" % self.p["reverse_for"])
                self.publish(-self.p["reverse_speed"], self.reverse_steer)
                return
        else:
            self.blocked_since = None

        # pure pursuit on the target point: the lookahead grows with speed, never past the target
        ld = float(np.clip(self.p["lookahead_min"] + self.p["lookahead_gain"] * self.speed,
                           self.p["lookahead_min"], self.p["lookahead_max"]))
        ld = min(ld, max(reach, 0.3))
        # the car keeps turning at the current steering for one delay before this command acts
        alpha = bearing - self.speed * math.tan(self.steer) / WHEELBASE * (self.p["steer_delay"] + dt)
        wanted = math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)
        wanted += self.side_nudge(ranges, angles)
        wanted = float(np.clip(wanted, -MAX_STEER, MAX_STEER))
        step = (wanted - self.steer) * lowpass(dt, self.p["steer_tau"])
        self.steer += float(np.clip(step, -MAX_STEER_RATE * dt, MAX_STEER_RATE * dt))

        # speed: the lowest of what the current steering allows, what the corner the target asks
        # for allows (the bend is seen before the wheel turns), the room straight ahead and the
        # target's own distance, each with the actuator delay and a realistic braking rate
        self.steer_slow += (self.steer - self.steer_slow) * lowpass(dt, self.p["curv_tau"])
        curvature = abs(math.tan(self.steer_slow)) / WHEELBASE
        v_corner = math.sqrt(self.p["lat_accel"] / curvature) if curvature > 1e-3 else self.p["max_speed"]
        k_target = abs(2.0 * math.sin(bearing) / max(ld, 0.3))
        v_bend = math.sqrt(self.p["lat_accel"] / k_target) if k_target > 1e-3 else self.p["max_speed"]
        v_brake = self.stoppable(ahead)
        v_reach = self.stoppable(reach)
        goal = max(self.p["min_speed"], min(self.p["max_speed"], v_corner, v_bend, v_brake, v_reach))
        if ahead < self.p["brake_margin"]:
            # the nose is on something: stop, unless the way the wheels point is still open, then creep round it
            goal = 0.0 if ahead_bearing < self.p["brake_margin"] else min(goal, self.p["creep_speed"])
        self.v_goal += (goal - self.v_goal) * lowpass(dt, self.p["goal_tau"])
        self.speed_cmd += float(np.clip(self.v_goal - self.speed_cmd, -self.p["decel_limit"] * dt, self.p["accel_limit"] * dt))
        self.publish(self.speed_cmd, self.steer)

    def extend(self, r, inc):
        """Disparity extension: at each jump between neighbours, paint the near side over the far
        side for half a car width plus the margin, nearest jumps first."""
        out = r.copy()
        d = np.diff(r)
        jumps = np.flatnonzero(np.abs(d) > self.p["disparity"])
        if jumps.size == 0:
            return out
        width = HALF_WIDTH + self.p["margin"]
        near_r = np.where(d[jumps] > 0, r[jumps], r[jumps + 1])
        for j, rn in sorted(zip(jumps, near_r), key=lambda x: x[1]):
            k = int(math.ceil(math.atan2(width, max(rn, 0.05)) / inc))
            if d[j] > 0:                   # the far side is to the right (higher index): paint forward
                lo, hi = j + 1, min(len(out), j + 1 + k)
            else:                          # the far side is to the left: paint backward
                lo, hi = max(0, j + 1 - k), j + 1
            out[lo:hi] = np.minimum(out[lo:hi], rn)
        return out

    def target(self, ext, angles):
        """Where to aim: between the farthest reachable beam inside the field of view (near-ties go
        to the beam closest to the previous target, so the aim slides rather than jumps) and the
        middle of the opening around it (the beams that still reach `opening` of its distance), so
        the car runs the middle of the corridor instead of hugging the wall the far beam grazes."""
        fov = np.abs(angles) <= self.p["fov"]
        cand = np.where(fov, ext, 0.0)
        best = float(cand.max())
        if best <= 0.0:
            return 0.0, 0.0
        tied = np.flatnonzero(cand >= self.p["tie"] * best)
        i = int(tied[int(np.argmin(np.abs(angles[tied] - self.target_angle)))])
        lo = hi = i
        floor = self.p["opening"] * best
        while lo > 0 and cand[lo - 1] >= floor:
            lo -= 1
        while hi < len(cand) - 1 and cand[hi + 1] >= floor:
            hi += 1
        middle = 0.5 * (angles[lo] + angles[hi])
        w = self.p["center_weight"]
        self.target_angle = float((1.0 - w) * angles[i] + w * middle)
        reach = float(min(ext[i], cand[lo:hi + 1].min())) if hi > lo else float(ext[i])
        return self.target_angle, max(reach, float(ext[int(np.argmin(np.abs(angles - self.target_angle)))]))

    def stoppable(self, d):
        """The speed from which the car stops within d minus the margin, braking at brake_decel
        after the actuator delay: v*delay + v^2/(2a) = d - margin."""
        a, lag = self.p["brake_decel"], self.p["steer_delay"]
        room = max(0.0, d - self.p["brake_margin"])
        return -a * lag + math.sqrt(a * a * lag * lag + 2.0 * a * room)

    def side_nudge(self, ranges, angles):
        """A wall closer than side_min on one flank pushes the steering the other way."""
        side = (np.abs(angles) > 0.5) & (np.abs(angles) < 1.6)
        left = ranges[side & (angles > 0)]
        right = ranges[side & (angles < 0)]
        dl = float(left.min()) if left.size else math.inf
        dr = float(right.min()) if right.size else math.inf
        nudge = 0.0
        if dl < self.p["side_min"]:
            nudge -= self.p["side_gain"] * (self.p["side_min"] - dl)
        if dr < self.p["side_min"]:
            nudge += self.p["side_gain"] * (self.p["side_min"] - dr)
        return nudge

    def room_along(self, ranges, angles, heading):
        """A low percentile of the range inside a car-wide corridor pointing along `heading`."""
        x, y = ranges * np.cos(angles - heading), ranges * np.sin(angles - heading)
        corridor = (np.abs(y) < HALF_WIDTH + self.p["corridor_extra"]) & (x > 0.0)
        return float(np.percentile(x[corridor], self.p["clear_pct"])) if corridor.any() else self.p["range_cap"]

    def publish(self, speed, steer):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed, cmd.drive.steering_angle = float(speed), float(steer)
        self.pub_drive.publish(cmd)


def lowpass(dt, tau):
    return 1.0 - math.exp(-dt / max(tau, 1e-3))


def main():
    rclpy.init()
    node = Competitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(0.0, 0.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
