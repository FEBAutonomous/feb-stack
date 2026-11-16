"""Pure-pursuit guidance for the mapping-lap corridor."""
from __future__ import annotations

import math

import numpy as np

from .perception import corridor_is_usable
from .gap_matcher import match_gap


def mapping_steering(centre, p, heading=0.0):
    """Return steering, curvature, and lookahead target in rear-axle coordinates.

    Corridor samples are expressed relative to the lidar. Stop at the first missing
    forward bin rather than steering across an unobserved stretch of track.
    """
    points = np.asarray(centre, dtype=float).reshape(-1, 2)
    forward = np.array([math.cos(heading), math.sin(heading)])
    points = points[np.all(np.isfinite(points), axis=1)].copy()
    points = points[points @ forward > 0.25]
    if not len(points):
        return None

    points[:, 0] += float(p["lidar_forward"])
    if len(points) > 1:
        max_gap = 1.5 * float(p["corridor_step"])
        gap = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > max_gap)
        if len(gap):
            points = points[:int(gap[0]) + 1]

    lookahead = max(float(p["lookahead"]), 0.1)
    target = _lookahead_target(points, lookahead)
    x, y = float(target[0]), float(target[1])
    distance_squared = max(x * x + y * y, 0.1)
    curvature = 2.0 * y / distance_squared
    steering = math.atan(float(p["wheelbase"]) * curvature)
    steering = float(np.clip(steering, -p["max_steer"], p["max_steer"]))
    return steering, curvature, (x, y)


class GuidancePlanner:
    """Stateful guidance selection; command/force conversion belongs to control."""
    def __init__(self,p, *, use_gap_matcher=False):
        self.p=p
        self.last=None
        self.use_gap_matcher=use_gap_matcher
        self.gap_yaw = None
        self.gap_turn = 0.

    def mapping(self,observation,speed,now,cycle,stamp,yaw=None):
        p=self.p
        usable=corridor_is_usable(observation,p)
        guidance=mapping_steering(observation.centre,p,observation.heading) if usable else None
        if self.use_gap_matcher:
            return self.match_mapping(observation, guidance, speed, now, cycle, stamp, yaw)
        if guidance is not None:
            steer,curvature,target=guidance
            path=np.asarray(observation.centre).copy()
            self.last=(now,cycle,stamp,steer,target,path)
            safe,collision,_=path_clearance_speed(observation.points,curvature,p)
            curve=np.sqrt(float(p['map_a_lat'])/max(abs(curvature),1e-6))
            wanted=min(float(p['map_speed']),float(curve),safe,max(float(speed)+.4,.7))
            return dict(mode='MAPPING',path=path,steering=steer,curvature=curvature,
                        target=target,target_speed=wanted,collision_distance=collision,
                        emergency=wanted<=.05 or mapping_emergency_stop(speed,collision,p),
                        degraded=False,valid=True,reason='',guidance_cycle=cycle,
                        guidance_stamp=stamp,guidance_age=0.)
        if not usable and self.last is not None:
            at,good_cycle,good_stamp,steer,target,path=self.last
            age=now-at
            if age<=p['mapping_fallback_duration']:
                curvature=math.tan(steer)/float(p['wheelbase'])
                safe,collision,_=path_clearance_speed(observation.points,curvature,p)
                wanted=min(float(p['mapping_fallback_speed']),safe)
                return dict(mode='DEGRADED_MAPPING',path=path,steering=float(steer),curvature=curvature,
                    target=target,target_speed=wanted,collision_distance=collision,
                    emergency=wanted<=.05 or mapping_emergency_stop(speed,collision,p),
                    degraded=True,valid=True,reason='local corridor unavailable',
                    guidance_cycle=good_cycle,guidance_stamp=good_stamp,guidance_age=age)
        return dict(mode='DEGRADED_MAPPING',path=np.empty((0,2)),steering=0.,curvature=0.,
                    target=(0.,0.),target_speed=0.,collision_distance=float('inf'),
                    emergency=True,degraded=True,valid=True,reason='mapping guidance unavailable or expired',
                    guidance_cycle=0,guidance_stamp=0.,guidance_age=0.)

    def match_mapping(self, observation, guidance, speed, now, cycle, stamp, yaw):
        p = self.p
        previous = self.last[3] if self.last is not None else 0.
        if guidance is not None and len(observation.centre) >= 3 and observation.confidence >= .50:
            steer, _, target = guidance
            curvature = math.tan(steer) / p['wheelbase']
            safe, collision, _ = path_clearance_speed(observation.points, curvature, p)
            if safe > .05 and not mapping_emergency_stop(speed, collision, p):
                self.gap_yaw, self.gap_turn = None, 0.
                path = observation.centre
                self.last = (now, cycle, stamp, steer, target, path)
                wanted = min(p['map_speed'], safe, math.sqrt(p['map_a_lat'] / max(abs(curvature), 1e-6)),
                             max(speed + .4, .7))
                return dict(mode='MAPPING', path=path, steering=steer, curvature=curvature,
                            target=target, target_speed=wanted, collision_distance=collision,
                            emergency=False, degraded=False, valid=True, reason='paired corridor',
                            guidance_cycle=cycle, guidance_stamp=stamp, guidance_age=0.)
        if yaw is not None:
            if self.gap_yaw is not None:
                change = yaw - self.gap_yaw
                self.gap_turn += math.atan2(math.sin(change), math.cos(change))
            self.gap_yaw = yaw
        # A gap can carry a hairpin, but cannot authorize another circuit
        # around open space without a corridor to re-establish track direction.
        gap = match_gap(observation.points, p, speed, previous) if abs(self.gap_turn) < 1.5 * math.pi else None
        if gap is None:
            return dict(mode='DEGRADED_MAPPING', path=np.empty((0, 2)), steering=previous,
                        curvature=0., target=(0., 0.), target_speed=0., collision_distance=0.,
                        emergency=True, degraded=True, valid=True, reason='no verified braking arc',
                        guidance_cycle=cycle, guidance_stamp=stamp, guidance_age=0.)
        steer = gap.steering
        curvature = math.tan(steer) / p['wheelbase']
        braking_room = gap.distance - p['map_path_reaction_distance'] - min(.10, p['map_path_step'])
        stopping_speed = math.sqrt(2 * p['map_a_brake'] * max(0., braking_room))
        corner_speed = math.sqrt(p['map_a_lat'] / max(abs(curvature), 1e-6))
        speed_target = min(p['mapping_fallback_speed'], stopping_speed, corner_speed, max(speed + .4, .7))
        self.last = (now, cycle, stamp, steer, gap.target, gap.path)
        return dict(mode='MAPPING', path=gap.path, steering=steer, curvature=curvature,
                    target=gap.target, target_speed=speed_target, collision_distance=gap.distance,
                    emergency=False, degraded=False, valid=True,
                    reason='verified gap', guidance_cycle=cycle,
                    guidance_stamp=stamp, guidance_age=0.)


def path_clearance_speed(scan_points, curvature, p):
    """Limit speed when lidar points intrude into the predicted rear-axle path."""
    points = np.asarray(scan_points, dtype=float).reshape(-1, 2)
    if not len(points):
        return float(p["map_speed"]), float("inf"), float("inf")

    points = points[np.all(np.isfinite(points), axis=1)].copy()
    points[:, 0] += float(p["lidar_forward"])
    points = points[points[:, 0] >= 0.0]
    horizon = float(p["map_path_horizon"])
    points = points[np.linalg.norm(points, axis=1) <= horizon + 1.0]
    if not len(points):
        return float(p["map_speed"]), float("inf"), float("inf")

    step = float(p["map_path_step"])
    stations = np.arange(step, horizon + step * 0.5, step)
    curvature = float(curvature)
    if abs(curvature) < 1e-5:
        path = np.column_stack((stations, np.zeros_like(stations)))
    else:
        path = np.column_stack((
            np.sin(curvature * stations) / curvature,
            (1.0 - np.cos(curvature * stations)) / curvature,
        ))

    distances = np.linalg.norm(path[:, None, :] - points[None, :, :], axis=2)
    closest_by_station = np.min(distances, axis=1)
    minimum_clearance = float(np.min(closest_by_station))
    required_clearance = float(p["car_half_width"] + p["wall_clearance"])
    collision = np.flatnonzero(closest_by_station <= required_clearance)
    if not len(collision):
        return float(p["map_speed"]), float("inf"), minimum_clearance

    collision_distance = float(stations[int(collision[0])])
    braking_distance = max(
        collision_distance - float(p["map_path_reaction_distance"]), 0.0)
    speed_cap = math.sqrt(2.0 * float(p["map_a_brake"]) * braking_distance)
    return min(float(p["map_speed"]), speed_cap), collision_distance, minimum_clearance


def mapping_emergency_stop(speed, collision_distance, p):
    """Keep hard braking for a wall inside the current stopping-distance envelope."""
    braking = max(float(p["map_a_brake"]), 1e-3)
    distance = float(p["map_path_reaction_distance"]) + max(float(speed), 0.0) ** 2 / (2.0 * braking)
    return bool(np.isfinite(collision_distance) and collision_distance <= distance)


def _lookahead_target(points, lookahead):
    first = points[0]
    first_distance = float(np.linalg.norm(first))
    if first_distance >= lookahead or len(points) == 1:
        return first

    radius_squared = lookahead * lookahead
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        a = float(np.dot(delta, delta))
        if a < 1e-12:
            continue
        b = 2.0 * float(np.dot(start, delta))
        c = float(np.dot(start, start) - radius_squared)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = math.sqrt(discriminant)
        for fraction in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
            if 0.0 <= fraction <= 1.0:
                return start + fraction * delta
    return points[-1]
