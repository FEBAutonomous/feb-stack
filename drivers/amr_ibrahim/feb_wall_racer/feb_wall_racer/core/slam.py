"""Bounded wall-point map, scan-to-map translation correction, and loop evidence."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Alignment:
    pose: np.ndarray
    yaw: float
    matches: int = 0
    residual: float = float("inf")
    rank: int = 0
    accepted: bool = False


class WallMap:
    def __init__(self, p):
        self.voxel = float(p["map_voxel"])
        self.max_points = int(p["max_map_points"])
        self.min_matches = 8
        self.match_gate = min(float(p.get("map_match_gate",.55)),float(p["track_width"])/4.0)
        self._voxels: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        self._side_voxels = {"left": {}, "right": {}}
        self.label_gate = self.match_gate
        self.min_label_votes = 2.0
        self.start_scan = np.empty((0, 2), dtype=float)
        self.frozen = False
        self.p = p
        self._registration_cache = None

    @property
    def points(self):
        if not self._voxels:
            return np.empty((0, 2), dtype=float)
        return np.asarray([entry[0] for entry in self._voxels.values()], dtype=float)

    def add(self, local_points, pose, yaw, sensor_forward, travelled,
            left_points=None, right_points=None, label_weight=1.0):
        if self.frozen or len(local_points) == 0:
            return
        self._registration_cache = None
        c, s = math.cos(yaw), math.sin(yaw)
        offset = np.array([sensor_forward * c, sensor_forward * s])
        world = np.asarray(local_points) @ np.array([[c, s], [-s, c]]) + np.asarray(pose) + offset
        step = max(1, len(world) // 180)
        for q in world[::step]:
            key = (int(math.floor(q[0] / self.voxel)), int(math.floor(q[1] / self.voxel)))
            self._voxels.setdefault(key, (q.copy(), float(travelled)))
        if len(self._voxels) > self.max_points:
            # Keep the map bounded and spatially representative by deterministic subsampling.
            stride = int(math.ceil(len(self._voxels) / self.max_points))
            self._voxels = dict(list(self._voxels.items())[::stride])
        self._add_side_points("left", left_points, pose, yaw, sensor_forward,
                              travelled, label_weight)
        self._add_side_points("right", right_points, pose, yaw, sensor_forward,
                              travelled, label_weight)

    def _add_side_points(self, side, local_points, pose, yaw, sensor_forward,
                         travelled, label_weight):
        if local_points is None or len(local_points) == 0:
            return
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        origin = np.asarray(pose, dtype=float) + sensor_forward * np.array([c, s])
        world = np.asarray(local_points, dtype=float) @ rotation.T + origin
        stride = max(1, int(math.ceil(len(world) / 180)))
        weight = max(float(label_weight), 0.0)
        table = self._side_voxels[side]
        batch = {}
        for point in world[::stride]:
            key = (int(math.floor(point[0] / self.voxel)),
                   int(math.floor(point[1] / self.voxel)))
            update = batch.get(key)
            if update is None:
                batch[key] = [point.copy(), 1]
            else:
                update[0] += point
                update[1] += 1

        for key, (point_sum, point_count) in batch.items():
            point = point_sum / point_count
            entry = table.get(key)
            if entry is None:
                table[key] = [point.copy(), weight, float(travelled), 1]
            else:
                count = entry[3]
                entry[0] += (point - entry[0]) / (count + 1)
                entry[1] += weight
                entry[2] = max(entry[2], float(travelled))
                entry[3] += 1

        if len(table) > self.max_points:
            stride = int(math.ceil(len(table) / self.max_points))
            self._side_voxels[side] = dict(list(table.items())[::stride])

    def score_side_labels(self, left_points, right_points, pose, yaw, sensor_forward):
        """Score candidate local wall labels against repeatedly observed map labels.

        Returns a signed agreement score (-1..1) and the number of map-supported
        samples. A side prior is deliberately unavailable until cells have repeat votes.
        """
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        origin = np.asarray(pose, dtype=float) + sensor_forward * np.array([c, s])
        expected = []
        for side, local in (("left", left_points), ("right", right_points)):
            if local is None or len(local) == 0:
                continue
            points = np.asarray(local, dtype=float)
            stride = max(1, int(math.ceil(len(points) / 60)))
            expected.append((side, points[::stride] @ rotation.T + origin))
        if not expected:
            return 0.0, 0

        radius = max(1, int(math.ceil(self.label_gate / self.voxel)))
        score_sum = 0.0
        evidence = 0
        for side, points in expected:
            opposite = "right" if side == "left" else "left"
            for point in points:
                key_x = int(math.floor(point[0] / self.voxel))
                key_y = int(math.floor(point[1] / self.voxel))
                nearest = None
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        key = (key_x + dx, key_y + dy)
                        for label in (side, opposite):
                            entry = self._side_voxels[label].get(key)
                            if entry is None or entry[1] < self.min_label_votes:
                                continue
                            distance = float(np.linalg.norm(point - entry[0]))
                            if distance > self.label_gate:
                                continue
                            if nearest is None or distance < nearest[0]:
                                nearest = (distance, label, entry[1])
                if nearest is None:
                    continue
                distance, label, _ = nearest
                score_sum += (1.0 if label == side else -1.0) * (
                    1.0 - distance / self.label_gate)
                evidence += 1
        if evidence == 0:
            return 0.0, 0
        return float(score_sum / evidence), evidence

    def _registration_map(self):
        if self._registration_cache is None:
            points = self.points
            tree = cKDTree(points)
            distances, indices = tree.query(points, k=min(5,len(points)))
            neighbors = points[indices]
            centered = neighbors-neighbors.mean(axis=1,keepdims=True)
            covariance = np.einsum('nki,nkj->nij',centered,centered)/neighbors.shape[1]
            eigenvalues, vectors = np.linalg.eigh(covariance)
            good = ((distances[:,-1] < 1.0) & (eigenvalues[:,1] > .0025)
                    & (eigenvalues[:,0] < .15*eigenvalues[:,1]))
            self._registration_cache = points,tree,vectors[:,:,0],good
        return self._registration_cache

    def align_pose(self, local_points, pose, yaw, sensor_forward):
        """Robust planar point-to-line ICP; retain prediction in weak directions.

        Rotation is scaled by scan radius before the information eigenanalysis,
        so angular and translation columns have comparable units. A rank-two
        corridor update can correct yaw/lateral position without moving along it.
        This never modifies the map or its caller's prediction on rejection.
        """
        origin = np.asarray(pose,dtype=float).copy()
        rejected = Alignment(origin,yaw)
        local = np.asarray(local_points,dtype=float).reshape(-1,2)
        local = local[np.all(np.isfinite(local),axis=1)]
        if len(local)<12 or len(self._voxels)<12 or not np.all(np.isfinite(origin)) or not math.isfinite(yaw):
            return rejected
        local = local[::max(1,int(math.ceil(len(local)/240)))].copy()
        local[:,0] += sensor_forward
        points,tree,normals,good = self._registration_map()
        guess, angle = origin.copy(),float(yaw)
        lever = max(1.,float(np.sqrt(np.mean(np.sum(local**2,axis=1)))))
        rank = 0
        def correspond(at,theta):
            c,s = math.cos(theta),math.sin(theta)
            rotated = local@np.array([[c,s],[-s,c]])
            world = rotated+at
            distances,indices = tree.query(world)
            keep = (distances<self.match_gate)&good[indices]
            return rotated,world,indices,keep
        for _ in range(8):
            rotated,world,indices,keep = correspond(guess,angle)
            if np.count_nonzero(keep)<max(12,.35*len(local)):
                return rejected
            n = normals[indices[keep]]
            residual = np.sum(n*(world[keep]-points[indices[keep]]),axis=1)
            spin = np.column_stack((-rotated[keep,1],rotated[keep,0]))
            jacobian = np.column_stack((n,np.sum(n*spin,axis=1)/lever))
            weight = np.minimum(1.,.08/np.maximum(np.abs(residual),1e-9))
            hessian = (jacobian.T*weight)@jacobian
            values,vectors = np.linalg.eigh(hessian)
            observed = values>max(1e-6,values[-1]*self.p.get('scan_match_min_eigen_ratio',.03))
            rank = int(np.count_nonzero(observed))
            if rank<2:
                return rejected
            basis = vectors[:,observed]
            step = -basis@((basis.T@(jacobian.T@(weight*residual)))/values[observed])
            guess += step[:2]
            angle += step[2]/lever
            if (np.linalg.norm(guess-origin)>self.p.get('scan_match_max_translation',.15)
                    or abs(angle-yaw)>self.p.get('scan_match_max_yaw',.06)):
                return rejected
            if np.linalg.norm(step)<1e-4:
                break
        _,world,indices,keep = correspond(guess,angle)
        matched = int(np.count_nonzero(keep))
        if matched<max(12,.35*len(local)):
            return rejected
        errors = np.abs(np.sum(normals[indices[keep]]*(world[keep]-points[indices[keep]]),axis=1))
        residual = float(np.median(errors))
        if residual>self.p.get('scan_match_max_residual',.08):
            return rejected
        return Alignment(guess,math.atan2(math.sin(angle),math.cos(angle)),matched,residual,rank,True)

    def align(self, local_points, pose, yaw, sensor_forward):
        """Correct translation against nearby map points; preserve IMU heading."""
        if len(local_points) < self.min_matches or len(self._voxels) < self.min_matches:
            return np.asarray(pose, dtype=float).copy(), 0, float("inf")
        map_points = self.points
        nearby = np.linalg.norm(map_points - np.asarray(pose)[None, :], axis=1) < 9.0
        map_points = map_points[nearby]
        if len(map_points) < self.min_matches:
            return np.asarray(pose, dtype=float).copy(), 0, float("inf")
        local = np.asarray(local_points, dtype=float)
        if len(local) > 240:
            local = local[::int(math.ceil(len(local) / 240))]
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        lidar_origin = np.asarray(pose, dtype=float) + sensor_forward * np.array([c, s])
        guess = np.asarray(pose, dtype=float).copy()
        residual = float("inf")
        matched = 0
        for _ in range(3):
            world = local @ rot.T + lidar_origin + (guess - np.asarray(pose, dtype=float))
            diff = world[:, None, :] - map_points[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", diff, diff)
            nearest = np.argmin(dist2, axis=1)
            distance = np.sqrt(dist2[np.arange(len(world)), nearest])
            keep = distance < self.match_gate
            matched = int(np.count_nonzero(keep))
            if matched < self.min_matches:
                return np.asarray(pose, dtype=float).copy(), matched, float("inf")
            offsets = map_points[nearest[keep]] - world[keep]
            shift = np.median(offsets, axis=0)
            residual = float(np.median(distance[keep]))
            norm = float(np.linalg.norm(shift))
            if norm > 0.12:
                shift *= 0.12 / norm
            guess += shift
            if norm < 0.01:
                break
        return guess, matched, residual

    def loop_match(self, local_points, pose, start_pose, yaw, start_yaw, travelled, p):
        if travelled < p["min_loop_distance"] or len(self.start_scan) < 12 or len(local_points) < 12:
            return False, 0.0
        displacement = np.asarray(pose) - np.asarray(start_pose)
        position_error = float(np.linalg.norm(displacement))
        yaw_error = abs(math.atan2(math.sin(yaw - start_yaw), math.cos(yaw - start_yaw)))
        if position_error > p["loop_radius"] or yaw_error > 0.55:
            return False, float("inf")
        along, across = np.array([[math.cos(start_yaw), math.sin(start_yaw)],
                                  [-math.sin(start_yaw), math.cos(start_yaw)]]) @ displacement
        # A high-overlap scan of parallel walls cannot identify the start
        # station. Require the integrated pose to return to it independently.
        if abs(along) > .30 or abs(across) > .55:
            return False, float("inf")
        current = np.asarray(local_points, dtype=float)
        if len(current) > 250:
            current = current[::int(math.ceil(len(current) / 250))]
        first = self.start_scan
        if len(first) > 250:
            first = first[::int(math.ceil(len(first) / 250))]
        dist = np.linalg.norm(current[:, None, :] - first[None, :, :], axis=2)
        nearest = dist.min(axis=1)
        inliers = nearest < 0.30
        ratio = float(np.mean(inliers))
        residual = float(np.median(nearest[inliers])) if np.any(inliers) else float("inf")
        return ratio >= 0.45 and np.count_nonzero(inliers) >= 12 and residual < 0.20, residual

    def close_loop(self, correction, travelled):
        """Distribute a validated end-pose correction along mapped observations."""
        if travelled <= 0.0:
            return
        self._registration_cache = None
        delta = np.asarray(correction, dtype=float)
        for key, (point, at) in list(self._voxels.items()):
            fraction = float(np.clip(at / travelled, 0.0, 1.0))
            self._voxels[key] = (point - fraction * delta, at)
        rebuilt = {}
        for point, at in self._voxels.values():
            key = (int(math.floor(point[0] / self.voxel)), int(math.floor(point[1] / self.voxel)))
            rebuilt.setdefault(key, (point, at))
        self._voxels = rebuilt
        for side, table in self._side_voxels.items():
            corrected = {}
            for point, votes, at, count in table.values():
                fraction = float(np.clip(at / travelled, 0.0, 1.0))
                point = point - fraction * delta
                key = (int(math.floor(point[0] / self.voxel)),
                       int(math.floor(point[1] / self.voxel)))
                entry = corrected.get(key)
                if entry is None:
                    corrected[key] = [point, votes, at, count]
                else:
                    total = entry[3] + count
                    entry[0] += (point - entry[0]) * (count / total)
                    entry[1] += votes
                    entry[2] = max(entry[2], at)
                    entry[3] = total
            self._side_voxels[side] = corrected
        self.frozen = True
