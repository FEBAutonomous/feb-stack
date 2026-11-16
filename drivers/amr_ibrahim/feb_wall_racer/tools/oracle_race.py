"""The racing controller on the simulator with the truth for a map and (optionally) for the pose
and the speed: a test rig, not a driver. It answers one question at a time.

    python3 oracle_race.py <track.json> <out.csv> [seconds] [speed=encoder|truth] [-p name:=value ...]

  * the raceline is built from the track file's cones, so there is no mapping lap to wait for
    and no map error;
  * the pose is the simulator's own;
  * the speed is the simulator's own body velocity (truth) or the racer's estimate from the
    wheel encoders (encoder).

If the car laps cleanly here, the controller and the vehicle model are right, and whatever goes
wrong in a real run is the estimate of where the car is. Every tick is written to the csv.
"""
import math
import sys

sys.path.insert(0, "/home/autodrive_devkit/src/stack/.pydeps")

import numpy as np                                                   # noqa: E402
import rclpy                                                         # noqa: E402
from nav_msgs.msg import Odometry                                    # noqa: E402
from rclpy.node import Node                                          # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy                  # noqa: E402
from sensor_msgs.msg import JointState                               # noqa: E402
from std_msgs.msg import Float32, Int32                              # noqa: E402

from feb_wall_racer.benchmarks.controller import load_track                                   # noqa: E402
from feb_wall_racer.core.controller import RaceController                          # noqa: E402
from feb_wall_racer.core.mpc import KinematicMPC                                  # noqa: E402
from feb_wall_racer.benchmarks.controller import BENCH_DEFAULTS as RACE                                     # noqa: E402
from feb_wall_racer.core.vehicle import AXLE_TO_CENTRE, MAX_STEER, WHEEL_RADIUS, BodySpeed  # noqa: E402

NS = "/autodrive/roboracer_1/"
AXLE_TO_IPS = 0.08


class Oracle(Node):
    def __init__(self, track, out, seconds, speed, p):
        super().__init__("oracle_race")
        self.speed_source, self.seconds = speed, seconds
        _, self.cones, line, v = load_track(track, p)
        p["v_cap"] = min(p["v_cap"], float(v.max()) + 0.3)
        self.ctl = RaceController(p, KinematicMPC(p))
        self.ctl.set_raceline(line, v)
        self.get_logger().info("raceline %.1f m, v %.1f..%.1f m/s, the plan is a %.2f s lap"
                               % (self.ctl.line.length, v.min(), v.max(), float(np.sum(np.diff(self.ctl.line.s) / v))))
        q = QoSProfile(depth=1)
        q.reliability = ReliabilityPolicy.RELIABLE
        self.pub_t = self.create_publisher(Float32, NS + "throttle_command", q)
        self.pub_s = self.create_publisher(Float32, NS + "steering_command", q)
        self.delta, self.enc, self.enc_prev, self.t_prev = 0.0, {}, None, None
        self.hits, self.laps, self.last_lap = 0, 0, 0.0
        self.obs = BodySpeed()
        self.t0 = None
        self.out = open(out, "w")
        self.out.write("t,x,y,yaw,v,vy,r,v_est,wheel,steer_fb,steer_cmd,throttle,s,off,v_plan,v_ref,force,clear,solve_ms,ok,hits,laps,last_lap\n")
        self.create_subscription(Float32, NS + "steering", lambda m: setattr(self, "delta", float(m.data)), q)
        self.create_subscription(JointState, NS + "left_encoder", lambda m: self.enc.__setitem__("l", float(m.position[0])), q)
        self.create_subscription(JointState, NS + "right_encoder", lambda m: self.enc.__setitem__("r", float(m.position[0])), q)
        self.create_subscription(Int32, NS + "collision_count", lambda m: setattr(self, "hits", int(m.data)), q)
        self.create_subscription(Int32, NS + "lap_count", lambda m: setattr(self, "laps", int(m.data)), q)
        self.create_subscription(Float32, NS + "last_lap_time", lambda m: setattr(self, "last_lap", float(m.data)), q)
        self.create_subscription(Odometry, NS + "odom", self.on_odom, q)

    def on_odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        o = m.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        x = m.pose.pose.position.x - AXLE_TO_IPS * math.cos(yaw)
        y = m.pose.pose.position.y - AXLE_TO_IPS * math.sin(yaw)
        v_true, r = m.twist.twist.linear.x, m.twist.twist.angular.z
        wheel = float("nan")
        if len(self.enc) == 2:
            angle = 0.5 * (self.enc["l"] + self.enc["r"])
            if self.enc_prev is not None and t > self.t_prev:
                wheel = (angle - self.enc_prev) / (t - self.t_prev) * WHEEL_RADIUS
                self.obs.update(wheel, t - self.t_prev, r, self.delta)
            self.enc_prev = angle
        self.t_prev = t
        v = v_true if self.speed_source == "truth" else self.obs.v
        if t - self.t0 > self.seconds:
            self.pub_t.publish(Float32(data=0.0))
            self.pub_s.publish(Float32(data=0.0))
            self.out.close()
            raise SystemExit
        steer, thr = self.ctl.step(t, x, y, yaw, max(v, 0.0), self.delta)
        self.pub_s.publish(Float32(data=float(steer / MAX_STEER)))
        self.pub_t.publish(Float32(data=float(thr)))
        i = self.ctl.info
        centre = np.array([x + AXLE_TO_CENTRE * math.cos(yaw), y + AXLE_TO_CENTRE * math.sin(yaw)])
        clear = float(np.min(np.linalg.norm(self.cones - centre, axis=1)))
        self.out.write("%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.2f,%.3f,%.2f,%.2f,%.2f,%.3f,%.1f,%d,%d,%d,%.2f\n"
                       % (t - self.t0, x, y, yaw, v_true, m.twist.twist.linear.y, r, self.obs.v, wheel, self.delta, steer, thr,
                          i["s"], i["off"], i["v_plan"], i["v_ref"], i["force"], clear, self.ctl.solve_ms, int(i["ok"]), self.hits, self.laps, self.last_lap))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-p") and ":=" not in a]
    track, out = args[0], args[1]
    seconds = float(args[2]) if len(args) > 2 else 60.0
    speed = args[3] if len(args) > 3 else "encoder"
    p = dict(RACE)
    for a in sys.argv[1:]:
        if ":=" in a:
            k, val = a.split(":=")
            p[k] = type(RACE[k])(float(val))
    rclpy.init()
    node = Oracle(track, out, seconds, speed, p)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.get_logger().info("done")


if __name__ == "__main__":
    main()
