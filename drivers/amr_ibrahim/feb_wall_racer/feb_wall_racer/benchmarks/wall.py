"""Small, ROS-independent smoke bench for wall perception and track planning."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from ..params import DEFAULTS
from ..core.perception import observe
from ..core.raceline import optimize, speed_plan
from ..core.track import build_track


def synthetic_corridor_scan(width=2.2, length=9.0, beams=1080):
    angle_min = -3.0 * math.pi / 4.0
    angle_increment = (3.0 * math.pi / 2.0) / (beams - 1)
    angles = angle_min + np.arange(beams) * angle_increment
    ranges = np.full(beams, np.inf)
    half = width / 2.0
    for wall_y in (-half, half):
        valid = np.abs(np.sin(angles)) > 1e-6
        distance = wall_y / np.sin(angles[valid])
        x = distance * np.cos(angles[valid])
        keep = (distance > 0.3) & (x > 0.25) & (x < length)
        indices = np.flatnonzero(valid)[keep]
        ranges[indices] = np.minimum(ranges[indices], distance[keep])
    return SimpleNamespace(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_max=10.0,
    )


def run():
    p = dict(DEFAULTS)
    obs = observe(synthetic_corridor_scan(), p)
    if len(obs.centre) < 6 or obs.confidence < 0.4:
        raise RuntimeError("synthetic wall corridor was not reliably perceived")
    if not np.isfinite(obs.width) or abs(obs.width - p["track_width"]) > 0.25:
        raise RuntimeError("synthetic corridor width estimate was %.3f m" % obs.width)

    count = 360
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    path = np.column_stack((7.0 * np.cos(theta), 5.0 * np.sin(theta)))
    widths = np.full(count, p["track_width"])
    track, reason = build_track(path, widths, p)
    if track is None:
        raise RuntimeError("synthetic closed track rejected: %s" % reason)
    line = optimize(track, p)
    speeds = speed_plan(line, p)
    if line.shape != track["centre"].shape or speeds.shape != (len(line),):
        raise RuntimeError("raceline planner returned inconsistent array dimensions")
    if not np.all(np.isfinite(line)) or not np.all(np.isfinite(speeds)):
        raise RuntimeError("raceline planner returned non-finite values")
    offset = np.linalg.norm(line - track["centre"], axis=1)
    available = track["half_width"] - p["car_half_width"] - p["wall_clearance"]
    if np.any(offset > available + 1e-6):
        raise RuntimeError("raceline exceeds the measured wall-clearance bound")
    print("perception: %d corridor samples, width %.2f m, confidence %.2f"
          % (len(obs.centre), obs.width, obs.confidence))
    print("track: %.1f m, %d samples; raceline and speed plan finite and within clearance"
          % (track["length"], len(line)))


def main():
    run()


if __name__ == "__main__":
    main()
