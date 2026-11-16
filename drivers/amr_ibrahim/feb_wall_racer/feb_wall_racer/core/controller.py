"""The racing controller: raceline in, steering and throttle out. No ROS in here, so the same
code drives the car in the simulator and in the offline controller benchmark.

Each tick:
  1. the car's state is carried forward to the moment this tick's command will take effect
     (ACT_DELAY later), through the commands already sent and still on their way;
  2. a reference is rolled out along the raceline from there, starting at the car's own speed
     and closing on the speed plan no faster than the tyres allow, so it is a trajectory the
     car can drive and not a point running away from it;
  3. the MPC tracks it;
  4. the plan's first tyre force becomes a throttle through the measured tyre curve, and the
     steering angle at the end of the first stage becomes the steering command.
"""
import math
import time

import numpy as np

from .vehicle import Plant, VehicleModel


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Raceline:
    """A closed line with a speed for every point, read by arc length."""

    def __init__(self, line, speed):
        self.line = np.asarray(line, float)
        self.speed = np.asarray(speed, float)
        seg = np.linalg.norm(np.roll(self.line, -1, axis=0) - self.line, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])          # N + 1 entries, the last is the lap length
        self.length = float(self.s[-1])
        d = np.roll(self.line, -1, axis=0) - np.roll(self.line, 1, axis=0)
        self.heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        turns = round((self.heading[-1] - self.heading[0]) / (2.0 * math.pi))    # a lap turns once, one way or the other
        self._x = np.append(self.line[:, 0], self.line[0, 0])
        self._y = np.append(self.line[:, 1], self.line[0, 1])
        self._v = np.append(self.speed, self.speed[0])
        self._h = np.append(self.heading, self.heading[0] + 2.0 * math.pi * turns)

    def at(self, s):
        s = np.mod(s, self.length)
        return (np.interp(s, self.s, self._x), np.interp(s, self.s, self._y),
                np.interp(s, self.s, self._h), np.interp(s, self.s, self._v))

    def project(self, point, hint=None, back=1.5, ahead=6.0):
        """Arc length of the point of the line nearest to `point`, searched round `hint` (the
        last answer), everywhere without one or when that finds nothing within 2 m."""
        n = len(self.line)
        if hint is None:
            idx = np.arange(n)
        else:
            i0 = int(np.searchsorted(self.s, np.mod(hint, self.length))) % n
            step = self.length / n
            idx = np.arange(i0 - int(back / step) - 1, i0 + int(ahead / step) + 2) % n
        best = self._nearest(point, idx)
        if hint is not None and best[0] > 2.0:
            best = self._nearest(point, np.arange(n))
        return best[1], best[0]

    def _nearest(self, point, idx):
        a = self.line[idx]
        b = self.line[(idx + 1) % len(self.line)]
        ab = b - a
        t = np.clip(np.sum((point - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(a + t[:, None] * ab - point, axis=1)
        k = int(np.argmin(d))
        return float(d[k]), float(self.s[idx[k]] + t[k] * np.linalg.norm(ab[k]))


class RaceController:
    def __init__(self, p, mpc):
        self.p, self.mpc = p, mpc
        self.vehicle = VehicleModel.from_parameters(p)
        self.line = None
        self.sent = []                   # (time sent, steering rad, throttle): commands on their way to the car
        self.s = None                    # progress along the raceline, m
        self.plan = None                 # (time of the plan's first stage, U, Z): the last good solution
        self.uprev = np.zeros(2)
        self.failures = 0
        self.solve_ms = 0.0
        self.speed_scale = 1.0           # share of the plan's speed to ask for (less while the car drives blind)
        self.speed_cap = 99.0            # m/s, the most to ask for whatever the plan says (the start of the race)
        self.info = {}

    def set_raceline(self, line, speed):
        self.line = Raceline(line, speed)
        self.s, self.plan, self.failures = None, None, 0
        if self.mpc is not None:
            self.mpc.warm = None

    def reset(self):
        self.sent, self.s, self.plan, self.failures = [], None, None, 0
        self.uprev = np.zeros(2)
        if self.mpc is not None:
            self.mpc.warm = None

    # ------------------------------------------------------------ the three steps

    def predict(self, t, x, y, psi, v, delta, state_age=0.0):
        """Carry timestamped state through command history to now + actuation delay."""
        if not math.isfinite(state_age) or state_age < 0.0:
            raise ValueError("state age must be finite and nonnegative")
        car = Plant(x, y, psi, v, model=self.vehicle)
        car.delta = delta
        car.t = t - state_age
        delay = self.vehicle.act_delay
        self.sent = [c for c in self.sent if car.t - c[0] < delay + 0.5]
        active = [c for c in self.sent if c[0] + delay <= car.t]
        if active:
            car.steer_cmd, car.throttle = active[-1][1], active[-1][2]
        else:
            car.steer_cmd, car.throttle = delta, v / self.vehicle.speed_per_throttle
        car.queue = [(c[0] + delay, c[1], c[2]) for c in self.sent if c[0] + delay > car.t]
        horizon = state_age + delay
        if horizon > 0.0:
            car.step(horizon, h=horizon / max(1, math.ceil(horizon / .004)))
        return np.array([car.x, car.y, car.psi, max(car.v, 0.0), car.delta])

    def record_command(self, t, steer, throttle):
        """Record when a command was actually published, on the prediction clock."""
        self.sent.append((float(t), float(steer), float(throttle)))

    def reference(self, s0, v0):
        """A drivable trajectory along the raceline from s0: x, y, heading, speed and the tyre
        force that produces it, one column per stage."""
        p, N, DT = self.p, self.mpc.N, self.mpc.DT
        ref = np.zeros((5, N))
        s, v = s0, v0
        for k in range(N):
            v_plan = min(self.speed_scale * float(self.line.at(s + max(v, 0.5) * DT)[3]), self.speed_cap)
            v_next = float(np.clip(v_plan, v - p["mpc_a_brake"] * DT, v + p["mpc_a_acc"] * DT))
            v_next = max(v_next, 0.3)
            s += 0.5 * (v + v_next) * DT
            x, y, h, _ = self.line.at(s)
            ref[:, k] = (x, y, h, v_next, (v_next - v) / DT + self.vehicle.drag * 0.5 * (v + v_next))
            v = v_next
        return ref

    def step(self, t, x, y, psi, v, delta, state_age=0.0, record_command=True):
        """One control tick. (x, y) is the rear axle, v the body speed, delta the steering angle
        the car reports. Returns (steering command rad, throttle)."""
        p = self.p
        z0 = self.predict(t, x, y, psi, v, delta, state_age)
        centre = z0[:2] + self.vehicle.axle_to_centre * np.array([math.cos(z0[2]), math.sin(z0[2])])
        s0, off = self.line.project(centre, self.s)
        self.s = s0
        if self.mpc is None:
            return self.fallback(t, z0, s0, record_command)
        ref = self.reference(s0, z0[3])
        ref[2] = z0[2] + np.array([wrap(a - z0[2]) for a in ref[2]])        # headings unwrapped round the car's
        t0 = time.time()
        out = self.mpc.solve(z0, ref, self.uprev)
        self.solve_ms = 0.8 * self.solve_ms + 0.2 * 1000.0 * (time.time() - t0)
        if out is not None:
            U, Z = out
            self.plan, self.failures = (t, U, Z), 0
            k = 0
        else:
            # no solution this tick: the last plan is still a plan, a stage or two further on
            self.failures += 1
            if self.plan is None or self.failures > 3:
                return self.fallback(t, z0, s0, record_command)
            t_plan, U, Z = self.plan
            k = int(min(round((t - t_plan) / self.mpc.DT), self.mpc.N - 1))
        force, omega = float(U[0, k]), float(U[1, k])
        self.uprev = np.array([force, omega])
        steer = float(np.clip(Z[4, k + 1], -self.vehicle.max_steer, self.vehicle.max_steer))
        throttle = self.vehicle.command_for_force(0.5 * (Z[3, k] + Z[3, k + 1]), force)
        throttle = float(np.clip(throttle, p["throttle_min"], p["throttle_max"]))
        if record_command:
            self.record_command(t, steer, throttle)
        self.info = dict(s=s0, off=off, v_ref=float(ref[3, 0]), v_plan=float(self.line.at(s0)[3]), force=force,
                         z0=z0, Z=Z, ref=ref, ok=out is not None, status=self.mpc.status)
        return steer, throttle

    def fallback(self, t, z0, s0, record_command=True):
        """Pure pursuit on the raceline at three quarters of the plan's speed, for the ticks the
        optimiser gives nothing. It is a way to stay on the track, not a way to race."""
        v = max(z0[3], 0.5)
        look = self.p["pursuit_lookahead"] + 0.25 * v
        tx, ty, _, v_plan = self.line.at(s0 + look)
        dx, dy = tx - z0[0], ty - z0[1]
        c, s = math.cos(-z0[2]), math.sin(-z0[2])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        steer = float(np.clip(math.atan(2.0 * self.vehicle.wheelbase * ly / max(lx * lx + ly * ly, 0.09)),
                              -self.vehicle.max_steer, self.vehicle.max_steer))
        v_goal = min(0.75 * self.speed_scale * float(v_plan), self.speed_cap)
        force = float(np.clip((v_goal - z0[3]) / 0.4, -self.p["mpc_a_brake"], self.p["mpc_a_acc"])) + self.vehicle.drag * z0[3]
        throttle = float(np.clip(self.vehicle.command_for_force(z0[3], force), self.p["throttle_min"], self.p["throttle_max"]))
        self.uprev = np.array([force, 0.0])
        if record_command:
            self.record_command(t, steer, throttle)
        self.info = dict(s=s0, off=0.0, v_ref=v_goal, v_plan=float(v_plan), force=force, z0=z0, Z=None, ref=None, ok=False, status="pursuit")
        return steer, throttle
