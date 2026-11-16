"""Conservative wall-clearance raceline and speed profile."""
from __future__ import annotations

import numpy as np


def curvature(path):
    path = np.asarray(path, dtype=float)
    nxt, prev = np.roll(path, -1, axis=0), np.roll(path, 1, axis=0)
    d1 = 0.5 * (nxt - prev)
    d2 = nxt - 2.0 * path + prev
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return cross / np.maximum(np.sum(d1 * d1, axis=1) ** 1.5, 1e-8)


def optimize(track, p):
    centre = np.asarray(track["centre"], dtype=float)
    normal = np.asarray(track["normal"], dtype=float)
    half = np.asarray(track["half_width"], dtype=float)
    room = np.maximum(half - p["car_half_width"] - p["wall_clearance"], 0.0)
    kappa = curvature(centre)
    # Offset toward the inside of a bend, with a strict measured-corridor clearance bound.
    alpha = np.clip(0.35 * kappa, -room, room)
    alpha = _smooth_closed(alpha, 5)
    alpha = np.clip(alpha, -room, room)
    line = centre + alpha[:, None] * normal
    return line


def speed_plan(path, p):
    line = np.asarray(path, dtype=float)
    ds = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
    window = max(3, int(round(float(p.get("curv_window",.8)) / max(float(np.mean(ds)), 1e-3))))
    # Preserve peaks and signs cannot cancel. The speed must be feasible on the
    # exact line supplied to the controller, including the cyclic seam.
    raw = np.abs(curvature(line))
    pad = window // 2
    k = np.maximum.reduce([np.roll(raw,i) for i in range(-pad,pad+1)])
    speed = np.minimum(float(p["v_max"]), np.sqrt(float(p["a_lat"]) / np.maximum(k, 1e-4)))
    speed *= float(p.get("race_speed_scale",1.0))
    for _ in range(2):
        for i in range(len(speed)):
            j = (i + 1) % len(speed)
            speed[j] = min(speed[j], np.sqrt(speed[i] ** 2 + 2.0 * p["a_acc"] * ds[i]))
        for i in range(len(speed) - 1, -1, -1):
            j = (i - 1) % len(speed)
            speed[j] = min(speed[j], np.sqrt(speed[i] ** 2 + 2.0 * p["a_brake"] * ds[j]))
    return speed


def _smooth_closed(values, window):
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return np.asarray(values, dtype=float).copy()
    pad = window // 2
    extended = np.r_[values[-pad:], values, values[:pad]]
    return np.convolve(extended, np.ones(window) / window, mode="valid")
