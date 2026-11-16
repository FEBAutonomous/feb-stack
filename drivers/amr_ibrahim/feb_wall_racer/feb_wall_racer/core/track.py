"""Closed corridor construction and path projection."""
from __future__ import annotations

import numpy as np


def resample_closed(points, step):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    closed = np.vstack((points, points[:1]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    if arc[-1] <= 0.0:
        return np.empty((0, 2)), np.empty(0)
    count = max(16, int(round(arc[-1] / step)))
    stations = np.linspace(0.0, arc[-1], count, endpoint=False)
    sampled = np.column_stack((np.interp(stations, arc, closed[:, 0]),
                               np.interp(stations, arc, closed[:, 1])))
    return sampled, stations


def build_track(path, widths, p):
    """Build a closed, locally measured corridor from the mapping-lap vehicle path."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    widths = np.asarray(widths, dtype=float).reshape(-1)
    if len(path) < 20 or len(widths) != len(path) or not np.all(np.isfinite(path)):
        return None, "insufficient mapping path"
    valid_width = np.isfinite(widths) & (widths >= p["track_width_min"]) & (widths <= p["track_width_max"])
    if float(np.mean(valid_width)) < p["min_width_coverage"]:
        return None, "insufficient measured corridor width coverage"
    raw_ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(raw_ds)))
    length = float(arc[-1])
    if not p["min_track_length"] <= length <= p["max_track_length"]:
        return None, "implausible track length %.1f m" % length
    gap = float(np.linalg.norm(path[-1] - path[0]))
    if gap > p["loop_radius"]:
        return None, "mapping path did not close (%.2f m gap)" % gap
    if np.any(raw_ds > 2.0):
        return None, "mapping path has an odometry gap"

    # Spread closure drift over the lap instead of creating a discontinuity at the seam.
    corrected = path.copy()
    correction = path[-1] - path[0]
    corrected -= (arc / max(length, 1e-9))[:, None] * correction
    corrected[-1] = corrected[0]
    source_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(corrected, axis=0), axis=1))))
    unique = np.r_[True, np.diff(source_arc) > 1e-4]
    corrected, source_arc = corrected[unique], source_arc[unique]
    widths = widths[unique]
    valid_width = valid_width[unique]
    if len(corrected) < 20:
        return None, "mapping path collapsed during closure correction"
    measured = np.flatnonzero(valid_width)
    if len(measured) < 2:
        return None, "insufficient measured corridor width samples"
    widths = np.interp(np.arange(len(widths)), measured, widths[measured])

    step = float(p["sample_step"])
    total = float(source_arc[-1])
    count = max(32, int(round(total / step)))
    stations = np.linspace(0.0, total, count, endpoint=False)
    ext = np.vstack((corrected, corrected[0]))
    ext_arc = np.r_[source_arc, total + step]
    centre = np.column_stack((np.interp(stations, ext_arc, ext[:, 0]),
                              np.interp(stations, ext_arc, ext[:, 1])))
    if _has_self_intersection(centre):
        return None, "centerline self-intersection"
    half = np.interp(stations, source_arc, widths)
    half = np.clip(half / 2.0, p["track_width_min"] / 2.0, p["track_width_max"] / 2.0)
    half = _smooth_closed(half, 7)
    minimum = p["car_half_width"] + p["wall_clearance"] + 0.05
    if float(np.min(half)) < minimum:
        return None, "corridor too narrow for car and clearance"
    tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-5):
        return None, "degenerate centerline tangent"
    tangent /= norm[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = centre + half[:, None] * normal
    right = centre - half[:, None] * normal
    return {"centre": centre, "half_width": half, "normal": normal, "left": left, "right": right,
            "s": stations, "length": float(total)}, ""


def _has_self_intersection(path):
    count = len(path)
    if count < 4:
        return False
    cell_size = 0.5
    cells = {}
    for i in range(count):
        a, b = path[i], path[(i + 1) % count]
        low = np.floor(np.minimum(a, b) / cell_size).astype(int)
        high = np.floor(np.maximum(a, b) / cell_size).astype(int)
        candidates = set()
        for gx in range(low[0], high[0] + 1):
            for gy in range(low[1], high[1] + 1):
                candidates.update(cells.get((gx, gy), ()))
        for j in candidates:
            separation = min((i - j) % count, (j - i) % count)
            if separation > 2 and _segments_intersect(a, b, path[j], path[(j + 1) % count]):
                return True
        for gx in range(low[0], high[0] + 1):
            for gy in range(low[1], high[1] + 1):
                cells.setdefault((gx, gy), []).append(i)
    return False


def _segments_intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    eps = 1e-8
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True

    def on_segment(p, q, r):
        return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
                and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)

    return ((abs(o1) <= eps and on_segment(a, c, b))
            or (abs(o2) <= eps and on_segment(a, d, b))
            or (abs(o3) <= eps and on_segment(c, a, d))
            or (abs(o4) <= eps and on_segment(c, b, d)))


def _smooth_closed(values, window):
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return values.copy()
    pad = window // 2
    extended = np.r_[values[-pad:], values, values[:pad]]
    return np.convolve(extended, np.ones(window) / window, mode="valid")


def project(path, point, hint=None, search_distance=8.0):
    """Project a point onto a closed sampled path, returning station and lateral error."""
    path = np.asarray(path, dtype=float)
    delta = np.roll(path, -1, axis=0) - path
    seg_len = np.linalg.norm(delta, axis=1)
    ds = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    q = np.asarray(point, dtype=float)
    t = np.clip(np.sum((q - path) * delta, axis=1) / ds, 0.0, 1.0)
    proj = path + t[:, None] * delta
    dist = np.linalg.norm(proj - q, axis=1)
    if hint is not None:
        stations = np.r_[0.0, np.cumsum(seg_len)]
        length = stations[-1]
        mid = (stations[:-1] + 0.5 * seg_len) % length
        arc_error = np.abs((mid - hint + 0.5 * length) % length - 0.5 * length)
        mask = arc_error <= search_distance
        if np.any(mask):
            dist[~mask] = np.inf
    i = int(np.argmin(dist))
    station = float(np.sum(seg_len[:i]) + t[i] * seg_len[i])
    return station, float(dist[i]), i


def sample(path, stations, station):
    path = np.asarray(path, dtype=float)
    stations = np.asarray(stations, dtype=float)
    length = float(stations[-1] + np.linalg.norm(path[0] - path[-1]))
    s = float(station % length)
    extended_s = np.r_[stations, length]
    extended_path = np.vstack((path, path[0]))
    return np.array([np.interp(s, extended_s, extended_path[:, 0]),
                     np.interp(s, extended_s, extended_path[:, 1])])
