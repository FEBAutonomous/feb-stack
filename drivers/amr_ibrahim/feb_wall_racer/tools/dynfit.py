"""Check the vehicle model (feb_wall_racer/vehicle.py) against what tools/dynid.py recorded.

    python3 tools/dynfit.py dyn_long.pkl dyn_lat.pkl

Two questions, each answered against the simulator's own body velocity:
  1. the plant: replay the throttle and steering the simulator applied, does the model's car
     do what the real one did?
  2. the speed observer: from the wheel encoders alone, how close is the body speed?
"""
import collections
import pickle
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
from feb_wall_racer.core.vehicle import BodySpeed, Plant, WHEEL_RADIUS   # noqa: E402

COLS = "t wall tau steer thr_fb str_fb encL encR vx vy r x y yaw ax ay".split()


def load(path):
    by = collections.OrderedDict()
    for row in pickle.load(open(path, "rb")):
        by.setdefault(row[0], []).append(row[1:])
    return {k: np.array(v, float) for k, v in by.items()}


def main():
    plant_err, obs_err, enc_err, yaw_err = [], [], [], []
    print("%-18s %9s %9s %9s %9s" % ("run", "plant v", "observer", "encoder", "yaw rate"))
    for path in sys.argv[1:]:
        for tag, a in load(path).items():
            t = a[:, 0]
            car = Plant(v=a[0, 8])
            obs = BodySpeed()
            obs.v = a[0, 8]
            enc = (a[:, 6] + a[:, 7]) / 2.0
            e_p, e_o, e_e, e_r = [], [], [], []
            for i in range(1, len(t)):
                dt = t[i] - t[i - 1]
                if dt <= 0.0:
                    continue
                # the feedback at tick i is what the simulator applied during the interval before it
                car.steer_cmd, car.throttle = a[i, 5], a[i, 4]
                car.delta = a[i, 5]
                car.step(dt)
                wheel = (enc[i] - enc[i - 1]) / dt * WHEEL_RADIUS
                obs.update(wheel, dt, a[i, 10], a[i, 5])
                e_p.append(car.v - a[i, 8])
                e_o.append(obs.v - a[i, 8])
                e_e.append(wheel - a[i, 8])
                e_r.append(car.yaw_rate - a[i, 10])
            rms = lambda e: float(np.sqrt(np.mean(np.square(e))))      # noqa: E731
            print("%-18s %9.3f %9.3f %9.3f %9.3f" % (tag, rms(e_p), rms(e_o), rms(e_e), rms(e_r)))
            plant_err += e_p; obs_err += e_o; enc_err += e_e; yaw_err += e_r
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))              # noqa: E731
    print("\nall runs, rms error against the simulator's body velocity:")
    print("  model car replaying the applied commands  %.3f m/s" % rms(plant_err))
    print("  body speed observed from the encoders      %.3f m/s" % rms(obs_err))
    print("  raw encoder speed                          %.3f m/s" % rms(enc_err))
    print("  model yaw rate                             %.3f rad/s" % rms(yaw_err))


if __name__ == "__main__":
    main()
