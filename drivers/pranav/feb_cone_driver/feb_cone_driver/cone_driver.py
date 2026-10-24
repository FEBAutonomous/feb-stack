"""Cone-track driver: 2D lidar + camera fusion, cone ordering, centreline, pure pursuit.

The simplest complete version of the pipeline the FSAE car runs, on the RoboRacer car:

  1. lidar    -> clusters of returns small enough to be a cone (position, no colour)
  2. camera   -> blue and yellow blobs by colour threshold (colour, no range)
  3. fusion   -> each lidar cone gets the colour of the blob at the same bearing
  4. memory   -> a cone's colour is carried from scan to scan (positions stay fresh from the
                 lidar), so cones beside the car, out of the camera's view, keep their colour
  5. ordering -> blue cones bound the left, yellow the right; pair them, midpoints = centreline
  6. control  -> pure pursuit on the centreline and a slow speed law, published on /drive

The same stages as the team's FSAE stack (perception, tracking, cone ordering, path, control)
in their simplest form: no neural network, no map, no graph optimisation. It drives the cone tracks at
1 to 1.5 m/s and publishes what it sees so Foxglove can show it. Speed, lookahead and
thresholds are parameters (config/cone_driver.yaml).

Inputs: /scan, /camera/color/image_raw, /imu, /odom. Output: /drive (speed m/s, steering angle rad).
Debug: /feb/cones (PoseArray, x forward, orientation.w encodes colour), /feb/centreline (Path).
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, LaserScan
QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE,
                 history=QoSHistoryPolicy.KEEP_LAST, depth=10)
WHEELBASE = 0.324
WHEEL_RADIUS = 0.059
MAX_STEER = 0.5236
MAX_STEER_RATE = 3.2

BLUE, YELLOW, ORANGE = 1.0, 2.0, 3.0   # colour codes carried in PoseArray orientation.w; orange = start-line cones


class ConeDriver(Node):
    def __init__(self):
        super().__init__("cone_driver")
        defaults = dict(max_speed=1.2, min_speed=0.8, lookahead=1.0, track_width=2.2, max_range=6.0,
                        cluster_gap=0.15, cone_max_width=0.35, cone_max_points=80, match_px_frac=0.04,
                        min_blob_px=3, camera_hfov_deg=57.1, camera_ahead=-0.66, camera_lateral=-0.033, camera_col_bias=-1.0, camera_lag_s=0.073,
                        hsv_blue=[105, 220, 8, 135, 255, 255], hsv_yellow=[18, 150, 8, 40, 255, 255], hsv_orange=[0, 150, 15, 15, 255, 255],
                        band_top=0.20, band_bottom=0.36, band_floor=0.21, orange_min_px=35, track_gate=0.4, track_memory=40, chain_step=1.8, side_override_x=1.5, side_override_y=0.6, side_guess_range=2.5, avoid_range=0.9, stall_time=2.0, reverse_time=1.2,
                        reverse_speed=0.5, steer_tau=0.15)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {k: self.get_parameter(k).value for k in defaults}

        self.bridge = CvBridge()
        self.blobs = []              # [(colour, u_centre_px, area)] from the latest image
        self.image_width = None
        self.image_height = None
        self.speed = 0.0             # m/s, from /odom
        self.yaw = None
        self.yaw_rate = 0.0
        self.prev_yaw = None
        self.prev = []               # last scan's cones: (x, y, colour, scans since the camera last saw it)
        self.steer = 0.0
        self.v_cmd = 0.0             # the speed we last asked for
        self.last_t = None
        self.prev_clusters = None    # last scan's cone clusters, to tell whether the world moves past us
        self.stalled_since = None    # sim time the lidar scene stopped changing although we were driving
        self.reverse_until = None    # sim time until which we back away from whatever we are stuck on
        self.target_t = -1e9         # sim time of the last scan that gave a target

        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", QOS)
        self.pub_cones = self.create_publisher(PoseArray, "/feb/cones", 10)
        self.pub_line = self.create_publisher(Path, "/feb/centreline", 10)
        self.pub_target = self.create_publisher(PoseStamped, "/feb/target", 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, QOS)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_image, QOS)
        self.create_subscription(Imu, "/imu", self.on_imu, QOS)
        self.create_subscription(Odometry, "/odom", self.on_odom, QOS)
        self.get_logger().info("cone driver: lidar + camera fusion, pure pursuit at %.1f m/s" % self.p["max_speed"])

    # ------------------------------------------------------------------ camera: colour blobs

    def on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.image_height, self.image_width = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cones are darker than their nominal colours at range, and the sky is a desaturated
        # blue: saturation separates a blue cone (S > 150) from sky (S ~ 100)
        # HSV bounds per colour (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) and the band of rows where
        # cones live, calibrated on this camera against ground truth: the cones are dark in this
        # scene, so the value floor is low and colour comes from hue and saturation; the sky is
        # a desaturated blue, so blue needs high saturation
        bounds = {BLUE: self.p["hsv_blue"], YELLOW: self.p["hsv_yellow"], ORANGE: self.p["hsv_orange"]}
        blobs = []
        h, w = img.shape[:2]
        for colour, b in bounds.items():
            mask = cv2.inRange(hsv, tuple(int(v) for v in b[:3]), tuple(int(v) for v in b[3:]))
            mask[: int(self.p["band_top"] * h), :] = 0       # sky
            mask[int(self.p["band_bottom"] * h):, :] = 0     # the car's own body and the floor
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not self.p["min_blob_px"] <= area <= 0.02 * w * h:   # too big is sky or bodywork
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if y + bh < self.p["band_floor"] * h:            # entirely above the horizon: sky
                    continue
                label = YELLOW if colour == ORANGE and area < self.p["orange_min_px"] else colour   # small orange = shaded yellow
                blobs.append((label, x + bw / 2.0, area))
        self.blobs = blobs

    # ------------------------------------------------------------------ lidar: cone clusters

    def cone_clusters(self, msg):
        """Small tight clusters of returns within max_range: cone candidates (x forward, y left)."""
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        ok = np.isfinite(ranges) & (ranges > 0.3) & (ranges < self.p["max_range"])   # < 0.3 m is the car itself
        xs, ys = ranges * np.cos(angles), ranges * np.sin(angles)
        cones = []
        cluster = []
        for i in np.flatnonzero(ok):
            if cluster and math.hypot(xs[i] - xs[cluster[-1]], ys[i] - ys[cluster[-1]]) > self.p["cluster_gap"]:
                cones.append(cluster)
                cluster = []
            cluster.append(i)
        if cluster:
            cones.append(cluster)
        out = []
        for c in cones:
            if len(c) > self.p["cone_max_points"]:
                continue                                  # a wall or a car, not a cone
            px, py = xs[c], ys[c]
            if math.hypot(px.max() - px.min(), py.max() - py.min()) > self.p["cone_max_width"]:
                continue
            out.append((float(px.mean()), float(py.mean())))
        return out

    # ------------------------------------------------------------------ fusion: colour by bearing

    def pixel_column(self, x, y):
        """Image column where a point (x forward, y left, lidar frame) lands; NaN if behind the
        camera. The camera's place relative to the lidar, its focal length, a column bias and the
        lag of the image behind the scan (the car keeps turning in between) were all measured on
        this car against ground truth; the numbers are parameters."""
        w = self.image_width
        fx = (w / 2.0) / math.tan(math.radians(self.p["camera_hfov_deg"]) / 2.0)   # pinhole focal length in pixels
        cx, cy = x - self.p["camera_ahead"], y - self.p["camera_lateral"]      # into the camera's frame
        if cx < 0.2:
            return float("nan")
        bearing = math.atan2(cy, cx) + self.yaw_rate * self.p["camera_lag_s"]   # where it was when the image was taken
        return w / 2.0 - fx * math.tan(bearing) + self.p["camera_col_bias"]     # +y (left) maps to smaller column

    def camera_colours(self, clusters):
        """One-to-one matching of lidar cones to camera blobs by image column, closest pairs
        first, so two cones cannot claim the same blob. Returns one colour (or None) per cluster."""
        n = len(clusters)
        colour = [None] * n
        if not self.blobs or self.image_width is None:
            return colour
        cols = [self.pixel_column(x, y) for x, y in clusters]
        tol = self.p["match_px_frac"] * self.image_width
        pairs = sorted((abs(ub - cols[i]), i, k) for i in range(n) if not math.isnan(cols[i])
                       for k, (c, ub, area) in enumerate(self.blobs) if abs(ub - cols[i]) < tol)
        done, taken = set(), set()
        for d, i, k in pairs:
            if i in done or k in taken:
                continue
            colour[i] = self.blobs[k][0]
            done.add(i)
            taken.add(k)
        return colour

    # ------------------------------------------------------------------ memory: carry colour between scans

    def on_imu(self, msg):
        q = msg.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.yaw_rate = msg.angular_velocity.z

    def remembered_colour(self, x, y):
        """Colour of the cone seen at this place in the previous scan, if any. Cone positions
        come fresh from every scan (the lidar is exact); only the colour is carried, so cones
        beside the car keep their colour after leaving the camera's view."""
        best, best_d = None, self.p["track_gate"]
        for px, py, colour, age in self.prev:
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best, best_d = (colour, age), d
        return best

    def predict_previous(self, dt):
        """Move last scan's cones into this scan's car frame using the motion since then."""
        dyaw = 0.0
        if self.yaw is not None and self.prev_yaw is not None:
            dyaw = math.atan2(math.sin(self.yaw - self.prev_yaw), math.cos(self.yaw - self.prev_yaw))
        self.prev_yaw = self.yaw
        ds = self.speed * dt
        c, s_ = math.cos(dyaw), math.sin(dyaw)
        moved = []
        for x, y, colour, age in self.prev:
            x0, y0 = x - ds, y
            moved.append((c * x0 + s_ * y0, -s_ * x0 + c * y0, colour, age + 1))
        self.prev = moved

    def fuse(self, clusters, dt):
        """Each lidar cone gets the camera's colour if a blob matches, else the colour it had in
        the previous scan (for a limited number of scans), else none."""
        self.predict_previous(dt)
        out = []
        camera = self.camera_colours(clusters)
        for (x, y), colour in zip(clusters, camera):
            age = 0
            if colour is None:
                remembered = self.remembered_colour(x, y)
                if remembered is not None and remembered[1] < self.p["track_memory"]:
                    colour, age = remembered
            if colour is None and 0.0 < x < self.p["side_guess_range"] and abs(y) < self.p["track_width"]:
                colour, age = (BLUE if y > 0.0 else YELLOW), self.p["track_memory"] - 1   # a guess; expires fast
            # a cone right beside the car is that side's boundary as long as the car is between the
            # lines: its side beats a colour remembered from when it was far ahead
            if colour is not None and x < self.p["side_override_x"] and self.p["side_override_y"] < abs(y) < self.p["track_width"]:
                colour = BLUE if y > 0.0 else YELLOW
            out.append((x, y, colour, age))
        self.prev = out
        return [(x, y, colour) for x, y, colour, age in out if colour is not None]

    def chain_step(self, cones):
        """Largest gap between consecutive cones of one colour when ordering them: twice the
        spacing of the cones in view (one may be missing), never below the configured minimum.
        Tracks are laid out with anything from 1 m to 5 m between cones."""
        gaps = []
        for i, (x, y, c) in enumerate(cones):
            same = [math.hypot(x - u, y - v) for j, (u, v, d) in enumerate(cones) if j != i and d == c]
            if same:
                gaps.append(min(same))
        if len(gaps) < 2:
            return self.p["chain_step"]
        return float(np.clip(2.0 * np.median(gaps), self.p["chain_step"], 6.0))

    def chain(self, cones, max_step):
        """Order one colour's cones along the track: start at the cone nearest the car, then
        repeatedly take the unused cone within max_step that continues the chain's direction
        best. Greedy, and enough for a corridor of evenly spaced cones."""
        if not cones:
            return []
        rest = list(cones)
        rest.sort(key=lambda c: math.hypot(c[0], c[1]))
        out = [rest.pop(0)]
        heading = (1.0, 0.0)                              # the car's own forward as the first direction
        while rest and len(out) < 8:
            last = out[-1]
            best, best_score = None, -1.0
            for c in rest:
                dx, dy = c[0] - last[0], c[1] - last[1]
                d = math.hypot(dx, dy)
                if d > max_step or d < 0.05:
                    continue
                score = (dx * heading[0] + dy * heading[1]) / d   # cos of the turn; > 0 means onward
                if score > best_score:
                    best, best_score = c, score
            if best is None or best_score < 0.0:
                break
            dx, dy = best[0] - last[0], best[1] - last[1]
            d = math.hypot(dx, dy)
            heading = (dx / d, dy / d)
            out.append(best)
            rest.remove(best)
        return out

    def centreline(self, cones):
        """Midpoints between the ordered chains. A blue chain cone pairs with the yellow cone that
        lies across the track to its RIGHT (right of the chain's direction of travel), never the
        nearest yellow regardless of side: at a hairpin the nearest yellow is often on the far
        branch and the midpoint would cut through the inner cones and turn the car around.
        Unpaired cones are offset by half the track width along their chain's normal."""
        # the orange start cones stand on the boundary lines: they count as the side they are on
        cones = [(x, y, (BLUE if y > 0.0 else YELLOW) if c == ORANGE else c) for x, y, c in cones]
        step = self.chain_step(cones)
        left = self.chain([c for c in cones if c[2] == BLUE and c[0] > -1.0], step)
        right = self.chain([c for c in cones if c[2] == YELLOW and c[0] > -1.0], step)
        half = self.p["track_width"] / 2.0

        def tangents(chain):
            out = []
            for i, c in enumerate(chain):
                a = chain[max(i - 1, 0)]
                b = chain[min(i + 1, len(chain) - 1)]
                tx, ty = b[0] - a[0], b[1] - a[1]
                n = math.hypot(tx, ty)
                out.append((tx / n, ty / n) if n > 1e-6 else (1.0, 0.0))
            return out

        lt, rt = tangents(left), tangents(right)
        points, used = [], set()
        for i, l in enumerate(left):
            nx, ny = lt[i][1], -lt[i][0]                       # right-hand normal of the blue chain
            best, best_score = None, None
            for j, r in enumerate(right):
                dx, dy = r[0] - l[0], r[1] - l[1]
                d = math.hypot(dx, dy)
                across = dx * nx + dy * ny                       # how far to the right
                along = abs(dx * lt[i][0] + dy * lt[i][1])       # how far along the track
                if j in used or d > 1.3 * self.p["track_width"] or across < 0.4 * d:
                    continue
                score = along + 0.5 * d
                if best_score is None or score < best_score:
                    best, best_score = j, score
            if best is None:
                points.append((l[0] + half * nx, l[1] + half * ny))
            else:
                used.add(best)
                points.append(((l[0] + right[best][0]) / 2.0, (l[1] + right[best][1]) / 2.0))
        for j, r in enumerate(right):
            if j not in used:
                nx, ny = -rt[j][1], rt[j][0]                     # left-hand normal of the yellow chain
                points.append((r[0] + half * nx, r[1] + half * ny))
        return sorted([q for q in points if q[0] > 0.0], key=lambda q: math.hypot(*q))

    # ------------------------------------------------------------------ control

    def on_scan(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = min(max(t - self.last_t, 0.01), 0.5) if self.last_t else 0.1
        self.last_t = t

        clusters = self.cone_clusters(msg)
        moving = self.scene_moving(clusters, dt)
        cones = self.fuse(clusters, dt)
        line = self.centreline(cones)
        self.publish_debug(msg.header, cones, line)
        self.get_logger().info("clusters %d blobs %d cones %d line %d speed %.2f cmd %.2f" % (
            len(clusters), len(self.blobs), len(cones), len(line), self.speed, self.v_cmd), throttle_duration_sec=2.0)

        # pure pursuit toward the centreline point nearest the lookahead distance; with no
        # cones at all, hold the wheel straight and crawl
        target = None
        for p in line:
            if math.hypot(*p) >= self.p["lookahead"]:
                target = p
                break
        if target is None and line:
            target = line[-1]
        if target is not None:
            self.target_t = t
            ld = max(math.hypot(*target), 0.3)
            alpha = math.atan2(target[1], target[0])
            wanted = math.atan(2.0 * WHEELBASE * math.sin(alpha) / ld)
        elif t - self.target_t < 1.0:
            # nothing ahead for a scan or two (the big start cones fill the view, a cone hidden
            # for a moment): hold the wheel and crawl before giving up
            target = (self.p["lookahead"] * math.cos(self.steer), self.p["lookahead"] * math.sin(self.steer))
            wanted = self.steer
        else:
            wanted = 0.0
        # any cone (whatever its colour) closer than avoid_range ahead pushes the wheel away from it
        for x, y in clusters:
            if 0.0 < x < self.p["avoid_range"] and abs(y) < 0.5:
                wanted -= math.copysign(0.6, y) * (1.0 - x / self.p["avoid_range"])
        wanted = float(np.clip(wanted, -MAX_STEER, MAX_STEER))
        tp = PoseStamped(header=msg.header)
        if target is not None:
            tp.pose.position.x, tp.pose.position.y = target
        tp.pose.orientation.w = 1.0 if target is not None else 0.0
        self.pub_target.publish(tp)
        # stuck on a cone (the wheels may still spin, so we watch the lidar scene instead of the
        # encoders): back away for a moment with the wheel turned the other way, then carry on
        if self.reverse_until is not None:
            if t < self.reverse_until:
                self.publish(-self.p["reverse_speed"], -self.steer)
                return
            self.reverse_until, self.stalled_since = None, None
            self.v_cmd = 0.0
        if moving and target is not None:
            self.stalled_since = None
        elif self.v_cmd > 0.05 or target is None:
            # stuck on a cone, or stopped with nothing to follow (nosed out of the corridor at a
            # hairpin): both end the same way, back up and look again
            self.stalled_since = self.stalled_since or t      # sticky: a throttle dip does not reset it
            if t - self.stalled_since > self.p["stall_time"]:
                self.get_logger().warn("stuck, backing up" if target is not None else "nothing to follow, backing up")
                self.reverse_until = t + self.p["reverse_time"]
        step = (wanted - self.steer) * min(1.0, dt / self.p["steer_tau"])
        self.steer += float(np.clip(step, -MAX_STEER_RATE * dt, MAX_STEER_RATE * dt))

        # slower the harder we steer; with nothing to follow, stop rather than wander off
        v_goal = max(self.p["min_speed"], self.p["max_speed"] * (1.0 - 0.7 * abs(self.steer) / MAX_STEER))
        self.v_cmd = 0.0 if target is None else v_goal
        self.publish(self.v_cmd, self.steer)

    def scene_moving(self, clusters, dt):
        """True when the world moves past the car: cone clusters matched to last scan's by
        nearest neighbour must have moved forward about speed * dt (or at least a little while
        the throttle is on and the wheels are blocked). Stuck on a cone, the steering rocks the
        body but nothing moves forward."""
        cur = np.array(clusters, float).reshape(-1, 2)
        prev, self.prev_clusters = self.prev_clusters, cur
        if prev is None or len(prev) < 3 or len(cur) < 3:
            return True
        d = np.linalg.norm(cur[:, None, :] - prev[None, :, :], axis=2)
        j = d.argmin(axis=1)
        near = d.min(axis=1) < 0.5
        if near.sum() < 3:
            return True
        forward = np.median(np.abs(cur[near, 0] - prev[j[near], 0]))
        expected = max(self.speed * dt, 0.03 if self.v_cmd > 0.1 else 0.0)
        return bool(expected <= 0.0 or forward > 0.4 * expected)

    def on_odom(self, msg):
        """Speed as the car reports it (wheel odometry on the car, the same dead-reckoned in the sim)."""
        self.speed = msg.twist.twist.linear.x

    def publish(self, speed, steer):
        """The contract: speed in m/s and steering angle in rad on /drive."""
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed, cmd.drive.steering_angle = float(speed), float(steer)
        self.pub_drive.publish(cmd)

    # ------------------------------------------------------------------ debug topics

    def publish_debug(self, header, cones, line):
        arr = PoseArray(header=header)
        for x, y, colour in cones:
            pose = Pose()
            pose.position.x, pose.position.y = x, y
            pose.orientation.w = colour
            arr.poses.append(pose)
        self.pub_cones.publish(arr)
        path = Path(header=header)
        for x, y in line:
            ps = PoseStamped(header=header)
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_line.publish(path)


def main():
    rclpy.init()
    node = ConeDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(0.0, 0.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
