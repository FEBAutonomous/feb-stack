"""Bounded 2D lidar processing for wall corridors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WallObservation:
    points: np.ndarray
    left: np.ndarray
    right: np.ndarray
    centre: np.ndarray
    width: float
    confidence: float
    segments: int
    residual: float
    front_clearance: float
    diagnostics: str = ""
    heading: float = 0.0
    map_vote_score: float = 0.0
    map_vote_matches: int = 0


def corridor_is_usable(observation, p):
    return (
        len(observation.centre) >= 1
        and observation.confidence >= p["min_corridor_confidence"]
        and np.isfinite(observation.width)
        and p["track_width_min"] <= observation.width <= p["track_width_max"]
    )


def scan_points(scan, p):
    """Convert finite, in-range LaserScan returns to lidar-frame XY points."""
    ranges = np.asarray(scan.ranges, dtype=float)
    angles = float(scan.angle_min) + np.arange(len(ranges)) * float(scan.angle_increment)
    valid = np.isfinite(ranges) & (ranges >= p["min_range"]) & (ranges <= min(p["max_range"], scan.range_max))
    r = ranges[valid]
    a = angles[valid]
    return np.column_stack((r * np.cos(a), r * np.sin(a))), a, ranges, angles


def lidar_points_to_map(points, pose, yaw, sensor_forward):
    """Transform lidar-frame XY points into the racer's estimated map frame."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    pose = np.asarray(pose, dtype=float).reshape(2)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    lidar_origin = pose + sensor_forward * np.array([c, s])
    return points @ rotation.T + lidar_origin


def _runs(points, indices, gap_limit, angle_increment):
    if len(indices) == 0:
        return []
    breaks = []
    for k in range(1, len(indices)):
        i0, i1 = int(indices[k - 1]), int(indices[k])
        if i1 != i0 + 1:
            breaks.append(k)
            continue
        d = float(np.linalg.norm(points[k] - points[k - 1]))
        adaptive = max(gap_limit, 2.0 * max(np.linalg.norm(points[k]), np.linalg.norm(points[k - 1])) * angle_increment)
        if d > adaptive:
            breaks.append(k)
    return [r for r in np.split(points, breaks) if len(r)]


def _split(points, tolerance, min_points):
    """Iterative split-and-merge; returns endpoint pairs and maximum-fit residuals."""
    pending = [(0, len(points) - 1)]
    output = []
    while pending:
        first, last = pending.pop()
        if last - first + 1 < min_points:
            continue
        a, b = points[first], points[last]
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1e-8:
            continue
        rel = points[first:last + 1] - a
        cross = np.abs(delta[0] * rel[:, 1] - delta[1] * rel[:, 0]) / length
        split_at = int(np.argmax(cross)) + first
        error = float(cross[split_at - first])
        if error > tolerance and split_at - first >= min_points - 1 and last - split_at >= min_points - 1:
            pending.append((split_at, last))
            pending.append((first, split_at))
        else:
            output.append((points[first:last + 1], error, length))
    return output


def observe(scan, p, heading_hint=0.0, wall_prior=None, with_corridor=True):
    """Extract local wall polylines and a conservative forward corridor estimate."""
    points, _, ranges, angles = scan_points(scan, p)
    empty = np.empty((0, 2), dtype=float)
    scan_info = _scan_diagnostics(scan, ranges, angles, p)
    if len(points) < p["min_segment_points"]:
        return WallObservation(
            points, empty, empty, empty, float("nan"), 0.0, 0, float("inf"),
            _front_clearance(ranges, angles),
            "%s; too few in-range returns for fitting" % scan_info,
        )

    valid_indices = np.flatnonzero(np.isfinite(ranges) & (ranges >= p["min_range"]) &
                                   (ranges <= min(p["max_range"], scan.range_max)))
    runs = _runs(points, valid_indices, p["adjacent_gap"], float(scan.angle_increment))
    line_parts = []
    fit_errors = []
    short_runs = 0
    short_segments = 0
    for run in runs:
        if len(run) < p["min_segment_points"]:
            short_runs += 1
            continue
        for segment, error, length in _split(run, p["split_tolerance"], p["min_segment_points"]):
            if length < p["min_segment_length"]:
                short_segments += 1
                continue
            line_parts.append(segment)
            fit_errors.append(error)
            if len(line_parts) >= int(p["max_line_segments"]):
                break
        if len(line_parts) >= int(p["max_line_segments"]):
            break

    if not line_parts:
        longest_run = max((len(run) for run in runs), default=0)
        return WallObservation(
            points, empty, empty, empty, float("nan"), 0.0, 0, float("inf"),
            _front_clearance(ranges, angles),
            "%s; runs=%d longest_run=%d short_runs=%d segments_below_length=%d"
            % (scan_info, len(runs), longest_run, short_runs, short_segments),
        )

    fitted = np.vstack(line_parts)
    if not with_corridor:
        # Racing needs fitted returns for map matching, not mapping's axis search.
        return WallObservation(
            points, fitted[fitted[:, 1] >= 0.0], fitted[fitted[:, 1] < 0.0],
            empty, float("nan"), 0.0, len(line_parts), float(np.median(fit_errors)),
            _front_clearance(ranges, angles), "%s; racing wall fits" % scan_info)
    left = fitted[fitted[:, 1] > 0.12]
    right = fitted[fitted[:, 1] < -0.12]
    centre, widths, bin_info, heading, support, labelled_left, labelled_right, map_vote = (
        _corridor_analysis(fitted, p, heading_hint, wall_prior)
    )
    width = float(np.median(widths)) if widths else float("nan")
    edge_balance = min(support) / max(max(support), 1)
    coverage = min(len(centre) / 6.0, 1.0)
    fit_quality = float(np.exp(-np.median(fit_errors) / max(p["split_tolerance"], 1e-3)))
    confidence = float(np.clip(coverage * (0.45 + 0.55 * edge_balance) * fit_quality, 0.0, 1.0))
    residual = float(np.median(fit_errors)) if fit_errors else float("inf")
    left_span = _x_span(left)
    right_span = _x_span(right)
    diagnostics = (
        "%s; runs=%d longest_run=%d short_runs=%d segments=%d short_segments=%d "
        "sides=L%d[x=%s]/R%d[x=%s] corridor_axis=%.1fdeg "
        "map_vote=%.2f/%d corridor_bins=%s"
        % (scan_info, len(runs), max((len(run) for run in runs), default=0),
           short_runs, len(line_parts), short_segments, len(left), left_span,
           len(right), right_span, np.rad2deg(heading), map_vote[0], map_vote[1],
           ",".join(bin_info) or "none")
    )
    return WallObservation(
        points, labelled_left, labelled_right, centre, width, confidence,
        len(line_parts), residual, _front_clearance(ranges, angles),
        diagnostics, heading, map_vote[0], map_vote[1],
    )


def _corridor_analysis(points, p, heading_hint=0.0, wall_prior=None):
    candidates = []
    step = float(p["corridor_step"])
    max_heading = float(p.get("corridor_heading_max", 75.0))
    heading_step = max(float(p.get("corridor_heading_step", 5.0)), 1.0)
    heading_window = float(p.get("corridor_heading_window", 30.0))
    hint_degrees = float(np.rad2deg(heading_hint))
    low = max(-max_heading, hint_degrees - heading_window)
    high = min(max_heading, hint_degrees + heading_window)
    headings_degrees = np.arange(low, high + heading_step * 0.5, heading_step)
    headings_degrees = np.unique(np.append(
        headings_degrees, np.clip(hint_degrees, -max_heading, max_heading)))
    headings = np.deg2rad(headings_degrees)
    stations = np.arange(float(p["corridor_x_min"]), float(p["corridor_x_max"]), step)

    for heading in headings:
        forward = np.array([np.cos(heading), np.sin(heading)])
        normal = np.array([-forward[1], forward[0]])
        longitudinal = points @ forward
        lateral = points @ normal
        candidate_region = (
            (longitudinal >= max(float(p["corridor_x_min"]) - 0.5, 0.25))
            & (longitudinal <= float(p["corridor_x_max"]) + step)
        )
        candidate_left = points[candidate_region & (lateral > 0.12)]
        candidate_right = points[candidate_region & (lateral < -0.12)]
        centres, widths, bin_info, support = [], [], [], [0, 0]
        paired_bins = 0
        for station in stations:
            in_bin = (longitudinal >= station) & (longitudinal < station + step)
            left_values = lateral[in_bin & (lateral > 0.12)]
            right_values = lateral[in_bin & (lateral < -0.12)]
            if len(left_values) < 2 or len(right_values) < 2:
                bin_info.append("%.1f:L%d/R%d" % (station, len(left_values), len(right_values)))
                continue

            paired_bins += 1
            support[0] += len(left_values)
            support[1] += len(right_values)
            left_edge = float(np.quantile(left_values, 0.75))
            right_edge = float(np.quantile(right_values, 0.25))
            width = left_edge - right_edge
            if p["track_width_min"] <= width <= p["track_width_max"]:
                centre_station = station + 0.5 * step
                centre_offset = 0.5 * (left_edge + right_edge)
                centres.append(centre_station * forward + centre_offset * normal)
                widths.append(width)
                bin_info.append("%.1f:L%d/R%d/W%.2f" %
                                (station, len(left_values), len(right_values), width))
            else:
                bin_info.append("%.1f:L%d/R%d/W%.2f-out" %
                                (station, len(left_values), len(right_values), width))

        width_spread = float(np.std(widths)) if widths else float("inf")
        geometry_score = (len(centres), paired_bins, min(support),
                          -width_spread, -abs(float(heading)))
        candidates.append((
            geometry_score,
            np.asarray(centres, dtype=float).reshape(-1, 2),
            widths,
            bin_info,
            float(heading),
            support,
            candidate_left,
            candidate_right,
        ))

    if not candidates:
        empty = np.empty((0, 2), dtype=float)
        return empty, [], [], 0.0, (0, 0), empty, empty, (0.0, 0)

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if wall_prior is not None:
        candidates = candidates[:max(1, int(p.get("corridor_map_candidate_count", 5)))]

    best = None
    for candidate in candidates:
        map_vote = (0.0, 0)
        if wall_prior is not None:
            map_vote = wall_prior(candidate[6], candidate[7])
        prior_score = map_vote[0] if map_vote[1] >= int(
            p.get("corridor_map_min_votes", 8)) else 0.0
        score = (prior_score,) + candidate[0]
        if best is None or score > best[0]:
            best = (score, candidate, map_vote)
    candidate = best[1]
    return (candidate[1], candidate[2], candidate[3], candidate[4],
            candidate[5], candidate[6], candidate[7], best[2])


def _scan_diagnostics(scan, ranges, angles, p):
    valid = np.isfinite(ranges) & (ranges >= p["min_range"]) & (
        ranges <= min(p["max_range"], scan.range_max))
    valid_ranges = ranges[valid]
    if len(valid_ranges):
        range_span = "%.2f..%.2fm" % (float(np.min(valid_ranges)), float(np.max(valid_ranges)))
    else:
        range_span = "none"
    angle_end = float(scan.angle_min) + max(len(ranges) - 1, 0) * float(scan.angle_increment)
    return (
        "scan=%d beams valid=%d range=%s limit=%.2f..%.2fm "
        "angle=%.1f..%.1fdeg increment=%.3fdeg sensor_range=%.2f..%.2fm"
        % (len(ranges), int(np.count_nonzero(valid)), range_span,
           p["min_range"], min(p["max_range"], scan.range_max),
           np.rad2deg(float(scan.angle_min)), np.rad2deg(angle_end),
           np.rad2deg(float(scan.angle_increment)),
           getattr(scan, "range_min", 0.0), scan.range_max)
    )


def _x_span(points):
    if not len(points):
        return "none"
    return "%.1f..%.1fm" % (float(np.min(points[:, 0])), float(np.max(points[:, 0])))


def _front_clearance(ranges, angles):
    front = np.abs(angles) <= np.deg2rad(16.0)
    good = np.isfinite(ranges) & (ranges > 0.0) & front
    return float(np.min(ranges[good])) if np.any(good) else float("inf")
