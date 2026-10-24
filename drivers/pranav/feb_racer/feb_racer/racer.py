"""feb_racer: the memory racer for walled tracks, on the car's contract.

Three phases, like the RoboRacer League 6 entry this descends from:
  WARMUP   one lap driven by the gap-follower law at a moderate speed while the car's own scan
           matcher builds an occupancy map in the start frame, correcting its odometry as it goes
  COMPUTE  the lap closes: the map is frozen and a minimum-curvature raceline with a friction-
           ellipse speed profile is built in a thread (the car keeps driving reactively)
  RACE     every scan is matched against the map; a pure-pursuit tracker follows the raceline at
           the profile's speed. When the match score drops the car hands back to the gap follower
           at a recovery speed until the map matches again: the reactive law has no pose to lose.

Inputs: /scan, /odom (the dead-reckoned prior and the speed), /feb/state (the dead-man).
Output: /drive, a speed in m/s and a steering angle in rad. Nothing simulator-specific.
"""
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

from feb_racer.localizer import ScanMatcher, compose, correction_for
from feb_racer.raceline_gen import build_raceline

QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE,
                 history=QoSHistoryPolicy.KEEP_LAST, depth=10)
SCAN_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=1)       # newest scan only: never steer on a stale one
WHEELBASE = 0.324
HALF_WIDTH = 0.14
MAX_STEER = 0.5236
MAX_STEER_RATE = 3.2
STEER_DELAY = 0.25


def lowpass(dt, tau):
    return 1.0 - math.exp(-dt / max(tau, 1e-3))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def bounded_correction(pose, d, max_th=0.26):
    """Map correction with the heading part clamped: a scan match may rotate the estimate a few
    degrees, never more; the odometry heading is the better long-term reference."""
    c = correction_for(pose, d)
    th = float(np.clip(c[2], -max_th, max_th))
    ct, st = math.cos(th), math.sin(th)
    return (pose[0] - (ct * d[0] - st * d[1]), pose[1] - (st * d[0] + ct * d[1]), th)


class Racer(Node):
    def __init__(self):
        super().__init__("racer")
        defaults = dict(
            # the gap-follower law (warm-up and recovery): same meaning as feb_driver's numbers
            bubble=0.45, gap_fov=1.60, range_cap=6.0, steer_gain=0.7, deep_weight=0.6, steer_tau=0.20,
            warmup_speed=1.5, recover_speed=1.3, min_speed=0.8, accel_limit=4.0, decel_limit=6.0,
            lat_accel=5.0, brake_margin=0.5, clear_pct=3.0, clear_tau=0.30, curv_tau=0.35,
            # the map and the match
            map_res=0.05, map_size_m=60.0, lidar_x=0.27, range_max=9.5, min_lap_m=20.0, close_radius_m=1.5,
            lost_score=0.45, good_score=0.60, lost_handoff_s=0.3,
            # the raceline and its speed profile (the physics the profile assumes)
            v_max=3.0, ay_max=4.0, a_acc=3.0, a_dec=5.0, wall_margin_m=0.25, v_narrow=2.5, clear_full_m=0.6,
            raceline_iters=300, speed_scale=1.0, mapping=True,
            # pure pursuit
            lookahead_min=0.9, lookahead_max=2.4, lookahead_gain=0.55, speed_horizon_m=1.5,
        )
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in defaults}
        self.sm = None                 # the scan matcher, made on the first scan (needs the scan's geometry)
        self.mode = "WARMUP"
        self.corr = (0.0, 0.0, 0.0)    # map pose = corr (+) odom pose
        self.odom = None               # (x, y, yaw) from /odom
        self.speed = 0.0
        self.pose = (0.0, 0.0, 0.0)
        self.score = 0.0
        self.path = []                 # map-frame poses driven on the warm-up lap, every 0.2 m
        self.dist = 0.0
        self.start = None
        self.rl = None                 # raceline: xy (n,2), v (n,), closed
        self.lost_since = None
        self.need_reloc = False
        self.live = False
        self.steer = 0.0
        self.steer_slow = 0.0
        self.clearance = None
        self.v_goal = 0.0
        self.speed_cmd = 0.0
        self.last_t = None
        self.last_log = 0.0
        self.pub = self.create_publisher(AckermannDriveStamped, "/drive", QOS)
        self.create_subscription(LaserScan, "/scan", self.on_scan, SCAN_QOS)
        self.create_subscription(Odometry, "/odom", self.on_odom, QOS)
        self.create_subscription(Imu, "/imu", self.on_imu, QOS)
        self.imu_yaw = None            # absolute heading when the IMU gives one (the simulator does; the car may not)
        self.t_imu = 0.0
        self.imu_offset = None         # imu yaw minus odom yaw, so the two agree at the start
        self.create_subscription(String, "/feb/state", self.on_state, 10)
        self.get_logger().info("racer up: warm-up lap maps, then the raceline")

    # ---------------------------------------------------------------- inputs

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odom = (p.x, p.y, yaw_of(q))
        self.speed = msg.twist.twist.linear.x
        self.yaw_rate = msg.twist.twist.angular.z
        self.t_odom = time.monotonic()           # arrival time: stamps differ between sim time and wall time

    def on_imu(self, msg):
        q = msg.orientation
        if abs(q.x) + abs(q.y) + abs(q.z) + abs(q.w) < 0.5:
            return                     # no orientation estimate in this IMU
        self.imu_yaw = yaw_of(q)
        self.t_imu = time.monotonic()

    def odom_at(self, t):
        """The odometry advanced to now with its own velocity: at the scored 10 Hz the last odometry
        can be 0.1 s old, which at 4 m/s is the whole search radius of the matcher. Measured from the
        messages' arrival times, which are comparable whatever clock the stamps use."""
        lag = float(np.clip(time.monotonic() - getattr(self, "t_odom", time.monotonic()), 0.0, 0.15))
        x, y, yaw = self.odom
        # heading: the IMU's absolute yaw when it is fresh (integrated gyro drifts at 10 Hz), on the odom's frame
        if self.imu_yaw is not None and time.monotonic() - self.t_imu < 0.5:
            if self.imu_offset is None:
                self.imu_offset = wrap(self.imu_yaw - yaw)
            yaw = wrap(self.imu_yaw - self.imu_offset)
        if lag <= 0.0:
            return self.odom, lag
        yaw2 = yaw + self.yaw_rate * lag
        return (x + self.speed * lag * math.cos(yaw2), y + self.speed * lag * math.sin(yaw2), yaw2), lag

    def on_state(self, msg):
        live = msg.data.startswith("AUTONOMOUS")
        if live and not self.live and self.mode == "RACE":
            self.need_reloc = True        # the car was carried or driven by hand: find ourselves on the map again
        self.live = live

    def throttle_refresh(self, sm, every=0.4):
        """The matcher rebuilds its blurred score grid after every insert, ~50 ms on a 60 m map; at
        10 Hz that starves the control loop. Rebuild at most every `every` seconds instead; a map
        a few scans stale matches just as well."""
        original = sm.refresh
        state = {"t": 0.0}

        def refresh():
            if sm.dirty and time.time() - state["t"] > every:
                state["t"] = time.time()
                original()
        sm.refresh = refresh

    # ---------------------------------------------------------------- the loop

    def on_scan(self, msg):
        t_wall = time.time()
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        dt = min(t - self.last_t, 0.5) if self.last_t else 0.0
        self.last_t = t
        if self.odom is None:
            return
        ranges = np.asarray(msg.ranges, np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, self.p["range_cap"])
        angles = msg.angle_min + msg.angle_increment * np.arange(len(ranges))
        if self.sm is None:
            sub = max(4, len(ranges) // 270)
            self.sm = ScanMatcher(msg.angle_min, msg.angle_increment, len(ranges), res=self.p["map_res"],
                                  size_m=self.p["map_size_m"], lidar_x=self.p["lidar_x"], range_max=self.p["range_max"], sub=sub)
            self.throttle_refresh(self.sm)
            self.start = self.odom
            self.corr = correction_for((0.0, 0.0, 0.0), self.odom)     # the map frame starts at the car
        odom, lag = self.odom_at(t)
        self.odom_now = odom
        prior = compose(self.corr, odom)

        if not self.p["mapping"]:                 # a test switch: the reactive law alone
            self.pose = prior
            self.reactive(ranges, angles, dt, self.p["warmup_speed"])
        elif self.mode == "WARMUP":
            self.warmup(ranges, angles, prior, dt)
        elif self.mode == "COMPUTE":
            self.pose = prior
            self.reactive(ranges, angles, dt, self.p["warmup_speed"])
        else:
            self.race(ranges, angles, prior, dt)

        self.ms_max = max(getattr(self, "ms_max", 0.0), (time.time() - t_wall) * 1000.0)
        if t - self.last_log > 2.0:
            self.last_log = t
            self.get_logger().info("%s score %.2f pose (%.1f, %.1f, %.0f deg) v %.1f lap %.0f m  loop max %.0f ms" % (
                self.mode, self.score, self.pose[0], self.pose[1], math.degrees(self.pose[2]), self.speed, self.dist, self.ms_max))
            self.ms_max = 0.0

    def warmup(self, ranges, angles, prior, dt):
        if self.sm.n_inserted >= 5:
            score, pose = self.sm.match(ranges, prior, span_xy=0.30)
            self.score = score
            if score > self.p["lost_score"]:
                self.corr = bounded_correction(pose, self.odom_now)
                prior = compose(self.corr, self.odom_now)
        self.pose = prior
        li = self.sm.last_insert
        if li is None or math.hypot(self.pose[0] - li[0], self.pose[1] - li[1]) > 0.05 or abs(wrap(self.pose[2] - li[2])) > 0.02:
            self.sm.insert(ranges, self.pose)
        if not self.path or math.hypot(self.pose[0] - self.path[-1][0], self.pose[1] - self.path[-1][1]) >= 0.1:
            if self.path:
                self.dist += math.hypot(self.pose[0] - self.path[-1][0], self.pose[1] - self.path[-1][1])
            self.path.append((self.pose[0], self.pose[1]))
        if self.dist > self.p["min_lap_m"] and math.hypot(self.pose[0], self.pose[1]) < self.p["close_radius_m"] and math.cos(self.pose[2]) > 0.3:
            self.mode = "COMPUTE"
            self.get_logger().info("lap closed after %.1f m: building the raceline" % self.dist)
            threading.Thread(target=self.compute, daemon=True).start()
        self.reactive(ranges, angles, dt, self.p["warmup_speed"])

    def compute(self):
        t0 = time.time()
        try:
            occ = self.sm.occupied()
            free = (self.sm.miss >= 2.0) & ~occ
            P = {k: self.p[k] for k in ("v_max", "a_acc", "a_dec", "wall_margin_m", "v_narrow", "clear_full_m")}
            P["ay_98pct_mps2"] = self.p["ay_max"]
            P["L0_m"] = WHEELBASE
            rl = build_raceline(free, self.sm.res, (self.sm.origin, self.sm.origin), list(self.path), P, iters=int(self.p["raceline_iters"]))
            if rl["self_distance_m"] < 0.5:
                raise RuntimeError("raceline folded (self-distance %.2f m)" % rl["self_distance_m"])
            self.rl = (np.asarray(rl["points"], float), np.asarray(rl["v"], float))
            self.get_logger().info("RACE: raceline %d points, lap %.1f m, planned %.1f s (built in %.1f s)" % (
                len(rl["points"]), rl["lap_length_m"], rl["est_lap_s"], time.time() - t0))
            self.mode = "RACE"
        except Exception as e:                       # keep driving reactively; the next lap closes again
            self.get_logger().error("raceline build failed: %r; staying on the warm-up law" % (e,))
            self.path = []
            self.dist = 0.0
            self.mode = "WARMUP"

    def race(self, ranges, angles, prior, dt):
        if self.need_reloc:
            score, pose = self.sm.global_localize(ranges, yaw_prior=prior[2])
            self.need_reloc = False
            self.get_logger().info("relocalised: score %.2f" % score)
            if score > self.p["lost_score"]:
                self.corr = correction_for(pose, self.odom_now)
                prior = pose
        span = float(np.clip(0.30 + 0.6 * self.speed * max(dt, 0.1), 0.30, 0.8))
        span_th = math.radians(10.0) + abs(getattr(self, "yaw_rate", 0.0)) * max(dt, 0.1)
        score, pose = self.sm.match(ranges, prior, span_xy=span, span_th=span_th)
        if score < self.p["lost_score"]:          # lost this scan: one wider look before trusting dead reckoning
            s2, p2 = self.sm.match(ranges, prior, span_xy=1.0, step_xy=0.08, span_th=math.radians(25.0), step_th=math.radians(2.5))
            if s2 > self.p["good_score"]:
                score, pose = s2, p2
        self.score = score
        if score > self.p["lost_score"]:
            self.corr = bounded_correction(pose, self.odom_now)
            self.pose = compose(self.corr, self.odom_now)
            self.lost_since = None
        else:
            self.pose = prior
            if self.lost_since is None:
                self.lost_since = self.last_t
        lost = self.lost_since is not None and self.last_t - self.lost_since > self.p["lost_handoff_s"]
        if lost and self.last_t - self.lost_since > 2.0 and self.last_t - getattr(self, "t_reloc", -1e9) > 3.0:
            self.t_reloc = self.last_t
            self.need_reloc = True            # next scan: global search; a car moved by hand or by a hit
        if lost:
            self.reactive(ranges, angles, dt, self.p["recover_speed"])
            return
        self.pursue(ranges, angles, dt)

    # ---------------------------------------------------------------- the two laws

    def pursue(self, ranges, angles, dt):
        xy, vprof = self.rl
        # the command acts one actuator delay plus one scan later: aim from where the car will be then
        T = STEER_DELAY + dt
        x, y, yaw = self.pose
        yaw_dot = self.speed * math.tan(self.steer) / WHEELBASE
        yaw = yaw + yaw_dot * T
        x += self.speed * T * math.cos(yaw)
        y += self.speed * T * math.sin(yaw)
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        i0 = int(np.argmin(d))
        n = len(xy)
        step = 0.20
        L = float(np.clip(self.p["lookahead_gain"] * self.speed + self.p["lookahead_min"], self.p["lookahead_min"], self.p["lookahead_max"]))
        j = (i0 + int(L / step)) % n
        tx, ty = xy[j]
        alpha = wrap(math.atan2(ty - y, tx - x) - yaw)
        Ld = max(math.hypot(tx - x, ty - y), 0.3)
        wanted = float(np.clip(math.atan2(2.0 * WHEELBASE * math.sin(alpha), Ld), -MAX_STEER, MAX_STEER))
        self.slew_steer(wanted, dt)
        # the profile's speed over a short horizon ahead, so braking starts before the corner
        k = int(self.p["speed_horizon_m"] / step) + 1
        idx = (i0 + np.arange(k)) % n
        v_plan = float(vprof[idx].min()) * self.p["speed_scale"]
        # the gap follower's clearance brake stays armed underneath: a wall the map did not know about
        ahead = self.room_ahead(ranges, angles, dt)
        v_brake = math.sqrt(2.0 * self.p["a_dec"] * max(0.0, ahead - self.p["brake_margin"]))
        self.slew_speed(max(self.p["min_speed"], min(v_plan, v_brake)), dt)
        self.publish(self.speed_cmd, self.steer)

    def reactive(self, ranges, angles, dt, v_cap):
        """feb_driver's law: the widest gap after a safety bubble, speed from the room ahead."""
        r = np.minimum(ranges, self.p["range_cap"])
        target = self.gap_target(r, angles)
        target -= self.speed * math.tan(self.steer) / WHEELBASE * (STEER_DELAY + dt)
        wanted = float(np.clip(self.p["steer_gain"] * target, -MAX_STEER, MAX_STEER))
        self.slew_steer(wanted, dt)
        ahead = self.room_ahead(ranges, angles, dt)
        self.steer_slow += (self.steer - self.steer_slow) * lowpass(dt, self.p["curv_tau"])
        curvature = abs(math.tan(self.steer_slow)) / WHEELBASE
        v_corner = math.sqrt(self.p["lat_accel"] / curvature) if curvature > 1e-3 else v_cap
        v_brake = math.sqrt(2.0 * self.p["decel_limit"] * max(0.0, ahead - self.p["brake_margin"]))
        self.slew_speed(max(self.p["min_speed"], min(v_cap, v_corner, v_brake)), dt)
        self.publish(self.speed_cmd, self.steer)

    def slew_steer(self, wanted, dt):
        step = (wanted - self.steer) * lowpass(dt, self.p["steer_tau"])
        self.steer += float(np.clip(step, -MAX_STEER_RATE * dt, MAX_STEER_RATE * dt))

    def slew_speed(self, goal, dt):
        self.v_goal += (goal - self.v_goal) * lowpass(dt, 0.2)
        up, down = self.p["accel_limit"] * dt, self.p["decel_limit"] * dt
        self.speed_cmd += float(np.clip(self.v_goal - self.speed_cmd, -down, up))

    def gap_target(self, ranges, angles):
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
        x, y = ranges * np.cos(angles), ranges * np.sin(angles)
        corridor = (np.abs(y) < HALF_WIDTH + 0.10) & (x > 0.0)
        raw = float(np.percentile(x[corridor], self.p["clear_pct"])) if corridor.any() else self.p["range_cap"]
        self.clearance = raw if self.clearance is None else self.clearance + (raw - self.clearance) * lowpass(dt, self.p["clear_tau"])
        return min(raw, self.clearance) if raw < self.p["brake_margin"] + 0.3 else self.clearance

    def publish(self, speed, steer):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed, cmd.drive.steering_angle = float(speed), float(steer)
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = Racer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
