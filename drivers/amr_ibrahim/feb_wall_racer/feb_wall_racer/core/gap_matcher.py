"""Rank connected, observed openings by the arcs the car can actually drive."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GapMatch:
    steering: float
    distance: float
    path: np.ndarray
    target: tuple


def _observed(points, returns, axle):
    """Verify points against contiguous measured rays, including their range."""
    bearings = np.arctan2(returns[:, 1], returns[:, 0])
    order = np.argsort(bearings)
    bearings = bearings[order]
    ranges = np.linalg.norm(returns[order], axis=1)
    spacing = np.diff(bearings)
    if not np.any(spacing > 0.):
        return np.zeros(points.shape[:-1], dtype=bool)
    max_gap = min(.025, 1.75 * np.median(spacing[spacing > 0.]))
    query = points - [axle, 0.]
    angles = np.arctan2(query[..., 1], query[..., 0])
    right = np.searchsorted(bearings, angles, side='right')
    i = np.clip(right, 1, len(bearings) - 1)
    known = ((right > 0) & (right < len(bearings)) & (spacing[i - 1] <= max_gap)
             & (np.linalg.norm(query, axis=-1) + .02 < np.minimum(ranges[i - 1], ranges[i])))
    return known


def match_gap(points, params, speed, previous):
    """Evaluate short forward arcs and keep only their continuously verified prefix."""
    returns = np.asarray(points, dtype=float).reshape(-1, 2)
    returns = returns[np.all(np.isfinite(returns), axis=1)]
    if len(returns) < 8:
        return None
    p = params
    wheelbase, lidar = p['wheelbase'], p['lidar_forward']
    margin = p['car_half_width'] + p['wall_clearance']
    step = min(.10, p['map_path_step'])
    stations = np.arange(step, p['map_path_horizon'] + 1e-8, step)
    if not len(stations):
        return None
    limit = p['max_steer']
    steering = np.unique(np.r_[np.linspace(-limit, limit, 21), np.clip(previous, -limit, limit)])
    curvature = np.tan(steering) / wheelbase
    angles = curvature[:, None] * stations
    tangent = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    # sinc form is continuous at zero curvature.
    rear = np.stack((stations * np.sinc(angles / np.pi),
                     stations * np.sin(angles / 2.) * np.sinc(angles / (2. * np.pi))), axis=-1)
    axles = rear[:, :, None, :] + tangent[:, :, None, :] * np.linspace(0., wheelbase, 3)[None, None, :, None]
    clearance = cKDTree(returns + [lidar, 0.]).query(axles.reshape(-1, 2))[0].reshape(axles.shape[:-1])
    clearance = clearance.min(axis=-1)
    # Verify coverage of the swept body. Measured obstacles still have to
    # clear the larger configured wall margin, including near the start.
    body = p['car_half_width'] + min(.04, p['wall_clearance'])
    checks = axles[:, :, :, None, :] + normal[:, :, None, None, :] * np.array(
        [-body, 0., body])[None, None, None, :, None]
    initial = checks - np.stack((np.clip(checks[..., 0], 0., wheelbase),
                                 np.zeros(checks.shape[:-1])), axis=-1)
    # The initial near-field buffer can be behind the lidar even at rest.
    inside = np.linalg.norm(initial, axis=-1) <= margin + .01
    visible = np.all(inside | _observed(checks, returns, lidar), axis=(-2, -1))
    valid = np.cumprod(visible & (clearance > margin + step / 2.), axis=1, dtype=bool)
    lengths = np.count_nonzero(valid, axis=1)
    distance = lengths * step
    required = step + p['map_path_reaction_distance'] + max(0., speed)**2 / (2. * p['map_a_brake'])
    feasible = np.flatnonzero(distance >= required)
    if not len(feasible):
        return None

    preview = min(len(stations), math.ceil(.8 / step))
    room = np.minimum(clearance[:, :preview].min(axis=1), 1.)
    score = np.minimum(distance, 1.8) + .65 * room - 1.15 * np.abs(steering - previous)
    score -= .20 * np.abs(steering)
    best = feasible[np.argmax(score[feasible])]
    path = rear[best, :lengths[best]]
    target_index = min(len(path) - 1, math.ceil(p['lookahead'] / step) - 1)
    target = tuple(map(float, path[target_index]))
    return GapMatch(float(steering[best]), float(distance[best]), path - [lidar, 0.], target)
