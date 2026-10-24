# Adapted from the FEB IROS 2026 league stack (tools/sysid_fit.py).
"""Fit car dynamics from a sysid rosbag (run inside the api container).
Longitudinal: dv/dt = a*thr - b*v - c*(thr<0.01)   [c = idle brake]
Lags: cross-correlation delay throttle->accel, steer_cmd->steer_fb.
Lateral: yaw_rate = v * tan(steer_fb) / L_eff  ->  L_eff by least squares,
         and lateral accel a_y = v*yaw_rate vs steer to find the slide limit.
usage: python3 sysid_fit.py <bag_dir>"""
import sys, json, math
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32

bag = sys.argv[1]
r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('', ''))
T = {'/autodrive/roboracer_1/left_encoder': JointState, '/autodrive/roboracer_1/right_encoder': JointState,
     '/autodrive/roboracer_1/imu': Imu, '/autodrive/roboracer_1/throttle': Float32,
     '/autodrive/roboracer_1/steering': Float32, '/autodrive/roboracer_1/throttle_command': Float32,
     '/autodrive/roboracer_1/steering_command': Float32}
D = {k: [] for k in T}
while r.has_next():
    topic, data, ts = r.read_next()
    if topic in T: D[topic].append((ts * 1e-9, deserialize_message(data, T[topic])))

def series(topic, f): 
    a = np.array([(t, f(m)) for t, m in D[topic]]); return a[:, 0], a[:, 1]
# speed from left encoder position (cumulative rad), teleport-aware
t_e, p = series('/autodrive/roboracer_1/left_encoder', lambda m: m.position[0])
dp = np.diff(p) * 0.059; dt = np.diff(t_e); ok = (dp >= 0) & (dt > 1e-4) & (dp / np.maximum(dt, 1e-4) < 27)
v = np.where(ok, dp / np.maximum(dt, 1e-4), np.nan); tv = t_e[1:]
v = np.interp(tv, tv[~np.isnan(v)], v[~np.isnan(v)])
t_th, thr = series('/autodrive/roboracer_1/throttle', lambda m: m.data)
t_st, st = series('/autodrive/roboracer_1/steering', lambda m: m.data)
t_im, yr = series('/autodrive/roboracer_1/imu', lambda m: m.angular_velocity.z)
thr_v = np.interp(tv, t_th, thr); st_v = np.interp(tv, t_st, st); yr_v = np.interp(tv, t_im, yr)
# smooth v, differentiate for accel
k = 5; vs = np.convolve(v, np.ones(k) / k, mode='same'); acc = np.gradient(vs, tv)
# --- longitudinal LS: acc = a*thr - b*v - c*[thr<0.01]
brake = (thr_v < 0.01).astype(float); A = np.c_[thr_v, -vs, -brake]; m = (vs > 0.3)
x, *_ = np.linalg.lstsq(A[m], acc[m], rcond=None); a, b, c = x
vmax_at = lambda u: (a * u) / b
# --- delay throttle -> accel by cross-correlation (lag in samples of median dt)
md = float(np.median(np.diff(tv)))
def lag(xs, ys, maxs=12):
    xs = xs - xs.mean(); ys = ys - ys.mean(); best = (0, -1)
    for s in range(0, maxs):
        c_ = float(np.dot(xs[:len(xs)-s], ys[s:]) / (np.linalg.norm(xs[:len(xs)-s]) * np.linalg.norm(ys[s:]) + 1e-9))
        if c_ > best[1]: best = (s, c_)
    return best[0] * md, best[1]
d_thr, c1 = lag(thr_v, acc)
t_sc, sc = series('/autodrive/roboracer_1/steering_command', lambda m: m.data)
sc_v = np.interp(tv, t_sc, sc); d_st, c2 = lag(sc_v, st_v)
# --- lateral: yaw_rate = v*tan(steer)/L  ->  L
mm = (vs > 0.8) & (np.abs(st_v) > 0.05)
L_eff = float(np.sum(vs[mm] * np.tan(st_v[mm]) * yr_v[mm]) / (np.sum(yr_v[mm] ** 2) + 1e-9))
ay = vs * yr_v; ay_max = float(np.percentile(np.abs(ay[vs > 1.0]), 98))
out = dict(long_a=float(a), long_b=float(b), idle_brake_c=float(c), v_ss_per_throttle=float(a / b),
           v_ss_at_0_25=float(vmax_at(0.25)), delay_throttle_to_accel_s=float(d_thr), delay_steer_cmd_to_fb_s=float(d_st),
           L_eff_m=L_eff, ay_98pct_mps2=ay_max, samples=int(len(tv)), sample_dt_s=md)
print(json.dumps(out, indent=1)); json.dump(out, open('/tmp/sysid_params.json', 'w'), indent=1)
