"""Dynamics identification, everything on the simulator's own clock, one row per bridge tick:
commanded throttle/steer, the throttle/steer the simulator says it applied, wheel encoder angles,
true body velocity (vx, vy), yaw rate, true position/yaw, IMU acceleration."""
import rclpy, numpy as np, time, pickle, sys, math
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy

rclpy.init(); n = Node("dynid")
q = QoSProfile(depth=10); q.reliability = ReliabilityPolicy.RELIABLE
pub_t = n.create_publisher(Float32, "/autodrive/roboracer_1/throttle_command", q)
pub_s = n.create_publisher(Float32, "/autodrive/roboracer_1/steering_command", q)
pub_r = n.create_publisher(Bool, "/autodrive/reset_command", q)
L = {"thr_fb": 0.0, "str_fb": 0.0, "encL": 0.0, "encR": 0.0, "imu": None}
n.create_subscription(Float32, "/autodrive/roboracer_1/throttle", lambda m: L.update(thr_fb=m.data), q)
n.create_subscription(Float32, "/autodrive/roboracer_1/steering", lambda m: L.update(str_fb=m.data), q)
n.create_subscription(JointState, "/autodrive/roboracer_1/left_encoder", lambda m: L.update(encL=m.position[0]), q)
n.create_subscription(JointState, "/autodrive/roboracer_1/right_encoder", lambda m: L.update(encR=m.position[0]), q)
n.create_subscription(Imu, "/autodrive/roboracer_1/imu", lambda m: L.update(imu=m), q)
CMD = {"tau": 0.0, "steer": 0.0, "reset": False}
rows = []
REC = {"on": False, "tag": ""}
def on_odom(m):
    # odom is published after throttle/steering feedback, encoders, ips and imu of the same tick
    t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    o = m.pose.pose.orientation
    yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
    imu = L["imu"]
    if REC["on"]:
        rows.append((REC["tag"], t, time.time(), CMD["tau"], CMD["steer"], L["thr_fb"], L["str_fb"], L["encL"], L["encR"],
                     m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z,
                     m.pose.pose.position.x, m.pose.pose.position.y, yaw,
                     imu.linear_acceleration.x if imu else 0.0, imu.linear_acceleration.y if imu else 0.0))
    G["t"] = t; G["v"] = m.twist.twist.linear.x
    # answer the tick at once, like a controller would
    pub_r.publish(Bool(data=CMD["reset"])); pub_s.publish(Float32(data=float(CMD["steer"]))); pub_t.publish(Float32(data=float(CMD["tau"])))
G = {"t": 0.0, "v": 0.0}
n.create_subscription(Odometry, "/autodrive/roboracer_1/odom", on_odom, q)

def hold(sec, tau, steer=0.0, reset=False):
    """hold a command for `sec` of SIMULATED time"""
    CMD.update(tau=tau, steer=steer, reset=reset)
    t0 = G["t"]; w0 = time.time()
    while G["t"] - t0 < sec and time.time() - w0 < 4 * sec + 5:
        rclpy.spin_once(n, timeout_sec=0.02)

def restart():
    REC["on"] = False
    hold(0.8, 0.0, reset=True); hold(1.5, 0.0)

def run(tag, schedule):
    restart(); REC["tag"] = tag; REC["on"] = True
    for seg in schedule:
        hold(*seg)
    REC["on"] = False
    print(tag, "done, rows", len(rows), flush=True)

# wait for the first tick
while G["t"] == 0.0: rclpy.spin_once(n, timeout_sec=0.1)
which = sys.argv[2] if len(sys.argv) > 2 else "long"
if which == "long":
    run("steps_small", [(1.5, 0.12), (1.2, 0.18), (1.2, 0.12), (1.2, 0.18), (1.0, 0.10)])
    run("steps_mid",   [(1.5, 0.15), (1.2, 0.25), (1.5, 0.12), (1.0, 0.22)])
    run("lift_big",    [(2.0, 0.22), (1.5, 0.05), (1.0, 0.20)])
    run("zero_brake",  [(2.0, 0.22), (1.5, 0.0)])
    run("neg_small",   [(2.0, 0.22), (1.2, -0.05), (0.5, 0.1)])
    run("neg_mid",     [(2.0, 0.22), (1.0, -0.15), (0.5, 0.1)])
    run("launch_006",  [(3.0, 0.06)])
    run("launch_010",  [(3.0, 0.10)])
    run("launch_015",  [(3.0, 0.15)])
    run("launch_030",  [(2.5, 0.30)])
    run("ramp",        [(0.25, 0.04 + 0.01 * i) for i in range(24)])
    run("prbs",        [(1.5, 0.18)] + [(0.3, 0.18 + (0.03 if (i * 7919 % 5) < 2 else -0.03)) for i in range(14)])
else:
    # lateral: hold a speed, step the steering (normalised command, +1 = full lock)
    for tau, tag in ((0.10, "v25"), (0.15, "v37"), (0.20, "v49")):
        for s in (0.15, 0.3, 0.5, 0.8):
            run("steer_%s_%.2f" % (tag, s), [(1.6, tau, 0.0), (1.2, tau, s), (1.0, tau, -s), (0.6, tau, 0.0)])
hold(0.5, 0.0)
pickle.dump(rows, open(sys.argv[1], "wb")); print("saved", len(rows))
