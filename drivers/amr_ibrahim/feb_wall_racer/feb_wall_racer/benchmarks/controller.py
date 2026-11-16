"""Closed-loop bench for the racing controller, without the simulator.

    python3 -m feb_wall_racer.benchmarks.controller tracks/loop_cones/track.json [--laps 3] [-p a_lat:=5.0 ...]
    ros2 run feb_wall_racer mpc_bench tracks/loop_cones/track.json

The controller under test is the modular core (core/controller.py, core/mpc.py); what it drives is the car as
measured in the simulator (vehicle.Plant: wheel speed set by the throttle, the tyre curve, the
cornering limit, the servo's slew rate, the actuation delay), ticked at the bridge's uneven
10 Hz. The raceline comes from the track file through the racer's own raceline and speed plan.

It reports lap times against the plan, how far the car runs from the line and from the cones,
how well the speed plan is kept, and how hard the controls are worked: a throttle that hunts
shows as reversals and swing, a steering that fights shows as reversals and peak rate.
"""
import argparse
import json
import math
import sys

import numpy as np

from ..core.controller import RaceController
from ..core.mpc import KinematicMPC
from ..control_params import RACE
from .cone_track import min_curvature, speed_plan
from ..core.track import resample_closed
from ..core.vehicle import AXLE_TO_CENTRE, MAX_STEER, WHEEL_RADIUS, BodySpeed, Plant

BENCH_DEFAULTS = dict(RACE, margin=.7, curvature_reg=.01, corner_scale=1.)

TICKS = (0.10, 0.05, 0.10, 0.10)          # the bridge's tick pattern with a window open


def load_track(path, p):
    """The raceline and speed plan for a track file, from the truth: the file's centreline, as
    wide at each point as its nearest cones allow, through the racer's own raceline and speed
    plan. (The racer itself gets its centreline from the map and the cone ordering.)"""
    t = json.load(open(path))
    cones = np.array([[c["x"], c["y"]] for c in t["cones"]], float)
    centre, s = resample_closed(np.array(t["centreline"], float).reshape(-1, 2), p["sample_step"])
    tang = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None] + 1e-9
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    half = np.min(np.linalg.norm(centre[:, None, :] - cones[None, :, :], axis=2), axis=1)
    half = np.minimum(half, float(np.median(half)))
    line, _ = min_curvature(centre, normal, half, p["car_half_width"], p["margin"], p["curvature_reg"], cones=cones)
    v, _ = speed_plan(line, p["v_max"] * p["race_speed_scale"], p["a_lat"] * p["corner_scale"] ** 2, p["a_acc"], p["a_brake"],
                      curv_window=p["curv_window"])
    return t, cones, line, v


def run(path, p, laps=3, speed="encoder", pose_noise=0.0, pose_lag=0.0, delay=None, seed=1, verbose=False, trace=None):
    rng = np.random.default_rng(seed)
    t_json, cones, line, v_plan = load_track(path, p)
    p = dict(p)
    p["v_cap"] = min(p["v_cap"], float(v_plan.max()) + 0.3)
    mpc = KinematicMPC(p)
    if not mpc.ok:
        sys.exit("CasADi is missing: install it (stack/feb_wall_racer/install_deps.sh)")
    ctl = RaceController(p, mpc)
    ctl.set_raceline(line, v_plan)
    plan_lap = float(np.sum(np.diff(ctl.line.s) / np.maximum(v_plan, 0.2)))
    # start on the line, at rest, as the racer does after its mapping lap (it starts slow)
    x0, y0, h0, _ = ctl.line.at(0.0)
    car = Plant(float(x0) - AXLE_TO_CENTRE * math.cos(h0), float(y0) - AXLE_TO_CENTRE * math.sin(h0), float(h0), 1.0)
    obs = BodySpeed()
    obs.v = car.v
    true_delay = p["act_delay"] if delay is None else delay
    last_angle, last_t = car.wheel_angle, 0.0
    s_prev, s_total, lap_t0, lap_times = None, 0.0, 0.0, []
    rec = dict(t=[], off=[], clear=[], v=[], v_plan=[], thr=[], steer=[], ay=[], solve=[], ok=[], v_est=[])
    tick = 0
    history = []                                       # true poses, for a lagging measurement
    while len(lap_times) < laps and car.t < 3.0 * laps * plan_lap + 20.0:
        dt = TICKS[tick % len(TICKS)]
        tick += 1
        car.step(dt)
        t = car.t
        wheel = (car.wheel_angle - last_angle) / (t - last_t) * WHEEL_RADIUS
        last_angle, last_t = car.wheel_angle, t
        v_est = obs.update(wheel, dt, car.yaw_rate, car.delta) if speed == "encoder" else (wheel if speed == "wheel" else car.v)
        history.append((t, car.x, car.y, car.psi))
        hx, hy, hpsi = car.x, car.y, car.psi
        if pose_lag > 0.0:
            old = [h for h in history if h[0] <= t - pose_lag]
            if old:
                _, hx, hy, hpsi = old[-1]
        mx, my = hx + rng.normal(0.0, pose_noise), hy + rng.normal(0.0, pose_noise)
        steer, thr = ctl.step(t, mx, my, hpsi, v_est, car.delta)
        car.command(steer, thr, delay=max(true_delay + rng.uniform(-0.02, 0.02), 0.0))
        # progress and laps, from the true centre of the car
        centre = car.point(AXLE_TO_CENTRE)
        s, off = ctl.line.project(centre, s_prev)
        if s_prev is not None:
            ds = s - s_prev
            if ds < -0.5 * ctl.line.length:
                ds += ctl.line.length
                lap_times.append(t - lap_t0)
                lap_t0 = t
            s_total += ds
        s_prev = s
        rec["t"].append(t); rec["off"].append(off); rec["v"].append(car.v); rec["v_plan"].append(float(ctl.line.at(s)[3]))
        rec["clear"].append(float(np.min(np.linalg.norm(cones - centre, axis=1))))
        rec["thr"].append(thr); rec["steer"].append(steer); rec["ay"].append(car.v * car.yaw_rate)
        rec["solve"].append(ctl.solve_ms); rec["ok"].append(ctl.info.get("ok", False)); rec["v_est"].append(v_est)
        if verbose and tick % 2 == 0:
            print("%6.2f s=%5.1f off %.2f v %.2f (plan %.2f, est %.2f) thr %.3f steer %+.2f ay %+.1f %s"
                  % (t, s, off, car.v, rec["v_plan"][-1], v_est, thr, steer, rec["ay"][-1], "" if rec["ok"][-1] else ctl.info.get("status", "")))
        if off > 3.0:
            break
    r = {k: np.array(val, float) for k, val in rec.items()}
    if trace:
        np.savez(trace, line=line, plan=v_plan, cones=cones, **r)
    skip = r["t"] > (lap_times[0] if lap_times else 0.0)          # the first lap starts from a crawl
    if not skip.any():
        skip = r["t"] >= 0.0
    dthr, dst = np.diff(r["thr"][skip]), np.diff(r["steer"][skip])
    dur = max(float(r["t"][skip][-1] - r["t"][skip][0]), 1e-6)
    rev = lambda d: float(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0) / dur)     # noqa: E731
    return dict(track=t_json.get("name", path), plan_lap=plan_lap, laps=lap_times, finished=len(lap_times) >= laps,
                off_rms=float(np.sqrt(np.mean(r["off"][skip] ** 2))), off_max=float(r["off"][skip].max()),
                clear_min=float(r["clear"][skip].min()) - p["car_half_width"],
                v_err=float(np.sqrt(np.mean((r["v"][skip] - r["v_plan"][skip]) ** 2))),
                v_over=float(np.max(r["v"][skip] - r["v_plan"][skip])),
                v_est_err=float(np.sqrt(np.mean((r["v_est"][skip] - r["v"][skip]) ** 2))),
                thr_rev=rev(dthr), thr_step=float(np.sqrt(np.mean(dthr ** 2))), thr_min=float(r["thr"][skip].min()),
                steer_rev=rev(dst), steer_rate=float(np.max(np.abs(dst)) / 0.0875),
                ay_max=float(np.max(np.abs(r["ay"][skip]))), sliding=car.sliding,
                solve_ms=float(np.median(r["solve"][skip])), failed=int(np.sum(r["ok"][skip] < 0.5)))


def report(r):
    laps = " ".join("%.2f" % x for x in r["laps"]) or "none"
    print("%s: plan %.2f s, laps %s%s" % (r["track"], r["plan_lap"], laps, "" if r["finished"] else "  (DID NOT FINISH)"))
    print("  line    %.2f m rms, %.2f m worst from the raceline; nearest cone %.2f m from the car's side" % (r["off_rms"], r["off_max"], r["clear_min"]))
    print("  speed   %.2f m/s rms from the plan, at most %.2f over it; estimate %.2f m/s rms from the truth" % (r["v_err"], r["v_over"], r["v_est_err"]))
    print("  throttle  %.1f reversals/s, %.4f rms step, lowest %.3f" % (r["thr_rev"], r["thr_step"], r["thr_min"]))
    print("  steering  %.1f reversals/s, peak rate %.2f rad/s; cornering up to %.1f m/s^2, %.2f s sliding" % (r["steer_rev"], r["steer_rate"], r["ay_max"], r["sliding"]))
    print("  solver  %.1f ms median, %d ticks without a solution" % (r["solve_ms"], r["failed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--speed", default="encoder", choices=["encoder", "wheel", "truth"], help="where the controller's speed comes from")
    ap.add_argument("--pose-noise", type=float, default=0.0, help="m of noise on the measured position")
    ap.add_argument("--pose-lag", type=float, default=0.0, help="s by which the measured pose is old")
    ap.add_argument("--delay", type=float, default=None, help="the car's real actuation delay, if not the one the controller assumes")
    ap.add_argument("--trace", default=None, help="save the run (npz)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-p", action="append", default=[], help="parameter override, name:=value")
    a = ap.parse_args()
    p = dict(BENCH_DEFAULTS)
    for kv in a.p:
        k, val = kv.split(":=")
        p[k] = type(BENCH_DEFAULTS[k])(float(val)) if not isinstance(BENCH_DEFAULTS[k], str) else val
    report(run(a.track, p, a.laps, a.speed, a.pose_noise, a.pose_lag, a.delay, verbose=a.verbose, trace=a.trace))


if __name__ == "__main__":
    main()
