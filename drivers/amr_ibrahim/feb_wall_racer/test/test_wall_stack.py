import math
import unittest
from types import SimpleNamespace

import numpy as np

from feb_wall_racer.benchmarks.wall import synthetic_corridor_scan
from feb_wall_racer.core.slam import WallMap
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.pathing import mapping_steering
from feb_wall_racer.core.pathing import path_clearance_speed
from feb_wall_racer.core.perception import (
    WallObservation, corridor_is_usable, lidar_points_to_map, observe, scan_points,
)
from feb_wall_racer.params import DEFAULTS as WALL_DEFAULTS
from feb_wall_racer.core.raceline import optimize, speed_plan
from feb_wall_racer.core.track import build_track
from feb_wall_racer.control_params import RACE as CONE_RACE


def _scan_from_wall_segments(segments, beams=1081):
    angle_min = -3.0 * math.pi / 4.0
    angle_increment = (3.0 * math.pi / 2.0) / (beams - 1)
    angles = angle_min + np.arange(beams) * angle_increment
    ranges = np.full(beams, np.inf)
    for index, angle in enumerate(angles):
        direction = np.array([math.cos(angle), math.sin(angle)])
        nearest = float("inf")
        for start, end in segments:
            edge = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
            denominator = direction[0] * edge[1] - direction[1] * edge[0]
            if abs(denominator) < 1e-10:
                continue
            start = np.asarray(start, dtype=float)
            distance = (start[0] * edge[1] - start[1] * edge[0]) / denominator
            fraction = (start[0] * direction[1] - start[1] * direction[0]) / denominator
            if distance > 0.0 and 0.0 <= fraction <= 1.0:
                nearest = min(nearest, distance)
        ranges[index] = nearest
    return SimpleNamespace(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.06,
        range_max=10.0,
    )


def _sharp_right_turn_scan(half_width=2.06):
    segments = [
        ((-5.0, half_width), (1.0, half_width)),
        ((1.0, half_width), (1.0 + half_width, 0.0)),
        ((1.0 + half_width, 0.0), (1.0 + half_width, -7.0)),
        ((-5.0, -half_width), (1.0, -half_width)),
        ((1.0, -half_width), (1.0 - half_width, 0.0)),
        ((1.0 - half_width, 0.0), (1.0 - half_width, -7.0)),
    ]
    return _scan_from_wall_segments(segments)


class WallStackTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(DEFAULTS)

    def test_perception_estimates_parallel_walls(self):
        result = observe(synthetic_corridor_scan(), self.p)
        self.assertGreaterEqual(len(result.centre), 6)
        self.assertGreater(result.confidence, 0.4)
        self.assertAlmostEqual(result.width, self.p["track_width"], delta=0.25)

    def test_perception_finds_corridor_through_sharp_right_turn(self):
        half_width = 2.06
        result = observe(_sharp_right_turn_scan(half_width), self.p)
        self.assertGreater(len(result.centre), 0, result.diagnostics)
        self.assertTrue(corridor_is_usable(result, self.p), result.diagnostics)
        self.assertGreater(result.width, 2.5, result.diagnostics)
        self.assertLess(result.width, 2.0 * half_width, result.diagnostics)
        self.assertLess(result.heading, 0.0, result.diagnostics)
        self.assertLess(result.heading, math.radians(30.0))
        self.assertLess(result.centre[-1, 1], result.centre[0, 1])
        steering, _, _ = mapping_steering(result.centre, self.p, result.heading)
        self.assertLess(steering, 0.0)

    def test_invalid_scan_has_no_false_corridor(self):
        scan = SimpleNamespace(
            ranges=np.full(1080, np.nan),
            angle_min=-3.0 * math.pi / 4.0,
            angle_increment=(3.0 * math.pi / 2.0) / 1079,
            range_max=10.0,
        )
        result = observe(scan, self.p)
        self.assertEqual(len(result.centre), 0)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("valid=0", result.diagnostics)

    def test_radian_heading_hint_searches_near_both_rotated_corridors(self):
        from feb_wall_racer.core.perception import _corridor_analysis
        x = np.linspace(0.0, 9.0, 350)
        walls = np.vstack((np.column_stack((x, np.full_like(x, 1.1))),
                           np.column_stack((x, np.full_like(x, -1.1)))))
        p = dict(self.p, corridor_heading_window=5.0, corridor_heading_step=1.0)
        for degrees in (-60.0, 60.0):
            with self.subTest(degrees=degrees):
                angle = math.radians(degrees)
                rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]])
                result = _corridor_analysis(walls @ rotation.T, p, heading_hint=angle)
                self.assertGreaterEqual(len(result[0]), 5)
                self.assertLessEqual(abs(math.degrees(result[3]) - degrees), 5.01)

    def test_lidar_points_transform_to_map_with_pose_heading_and_mount_offset(self):
        points = np.array([[1.0, 0.0], [0.0, 1.0]])
        pose = np.array([2.0, 3.0])
        transformed = lidar_points_to_map(points, pose, math.pi / 2.0, 0.5)
        np.testing.assert_allclose(transformed, [[2.0, 4.5], [1.0, 3.5]], atol=1e-12)

    def test_wide_corridor_reports_rejected_bin_widths(self):
        result = observe(synthetic_corridor_scan(width=5.0), dict(self.p, track_width_max=4.0))
        self.assertEqual(len(result.centre), 0)
        self.assertIn("W5.00-out", result.diagnostics)
        self.assertIn("corridor_bins=", result.diagnostics)

    def test_vegas_width_is_accepted_with_configured_headroom(self):
        p = dict(self.p, track_width_min=1.2, track_width_max=4.5)
        result = observe(synthetic_corridor_scan(width=4.12), p)
        self.assertGreater(len(result.centre), 0)
        self.assertTrue(corridor_is_usable(result, p))
        self.assertAlmostEqual(result.width, 4.12, delta=0.1)

    def test_mapping_pure_pursuit_tracks_straight_corridor(self):
        x = np.arange(0.675, 3.0, 0.35)
        centre = np.column_stack((x, np.zeros_like(x)))
        steer, curvature, target = mapping_steering(centre, self.p)
        self.assertAlmostEqual(steer, 0.0, places=8)
        self.assertAlmostEqual(curvature, 0.0, places=8)
        self.assertAlmostEqual(np.linalg.norm(target), self.p["lookahead"], places=6)
        self.assertGreater(target[0], centre[0, 0])

    def test_mapping_pure_pursuit_uses_rear_axle_and_turns_toward_path(self):
        x = np.arange(0.675, 3.0, 0.35)
        centre = np.column_stack((x, 0.08 * x * x))
        steer, curvature, target = mapping_steering(centre, self.p)
        self.assertGreater(steer, 0.0)
        self.assertGreater(curvature, 0.0)
        self.assertAlmostEqual(np.linalg.norm(target), self.p["lookahead"], places=6)
        self.assertGreater(target[0], 0.675 + self.p["lidar_forward"])

    def test_mapping_pure_pursuit_does_not_bridge_missing_corridor_bins(self):
        centre = np.array([[0.675, 0.2], [3.0, -0.5], [4.0, -1.0]])
        _, _, target = mapping_steering(centre, self.p)
        expected_first = np.array([0.675 + self.p["lidar_forward"], 0.2])
        np.testing.assert_allclose(target, expected_first)

    def test_mapping_pure_pursuit_respects_rotated_path_order(self):
        heading = math.radians(35.0)
        forward = np.array([math.cos(heading), math.sin(heading)])
        centre = np.arange(0.675, 3.0, 0.35)[:, None] * forward
        steer, curvature, target = mapping_steering(centre, self.p, heading)
        self.assertGreater(steer, 0.0)
        self.assertGreater(curvature, 0.0)
        self.assertGreater(float(np.dot(target, forward)), 0.0)
        self.assertAlmostEqual(np.linalg.norm(target), self.p["lookahead"], places=6)

    def test_map_wall_side_votes_prefer_repeated_classification(self):
        wall_map = WallMap(self.p)
        x = np.linspace(0.4, 4.5, 50)
        left = np.column_stack((x, np.full_like(x, 2.0)))
        right = np.column_stack((x, np.full_like(x, -2.0)))
        points = np.vstack((left, right))
        agreement, agreement_count = wall_map.score_side_labels(
            left, right, np.zeros(2), 0.0, 0.0)
        self.assertEqual(agreement_count, 0)
        wall_map.add(points, np.zeros(2), 0.0, 0.0, 1.0,
                     left_points=left, right_points=right)
        _, one_vote_count = wall_map.score_side_labels(
            left, right, np.zeros(2), 0.0, 0.0)
        self.assertEqual(one_vote_count, 0)
        wall_map.add(points, np.zeros(2), 0.0, 0.0, 1.4,
                     left_points=left, right_points=right)
        agreement, agreement_count = wall_map.score_side_labels(
            left, right, np.zeros(2), 0.0, 0.0)
        disagreement, disagreement_count = wall_map.score_side_labels(
            right, left, np.zeros(2), 0.0, 0.0)
        self.assertGreaterEqual(agreement_count, self.p["corridor_map_min_votes"])
        self.assertEqual(agreement_count, disagreement_count)
        self.assertGreater(agreement, 0.8)
        self.assertLess(disagreement, -0.8)

    def test_perception_uses_repeated_map_wall_labels_as_prior(self):
        scan = synthetic_corridor_scan()
        first = observe(scan, self.p)
        wall_map = WallMap(self.p)
        points = np.vstack((first.left, first.right))
        for travelled in (0.4, 0.8):
            wall_map.add(
                points, np.zeros(2), 0.0, self.p["lidar_forward"], travelled,
                left_points=first.left, right_points=first.right,
            )
        second = observe(
            scan, self.p, first.heading,
            wall_prior=lambda left, right: wall_map.score_side_labels(
                left, right, np.zeros(2), 0.0, self.p["lidar_forward"]),
        )
        self.assertGreater(second.map_vote_matches, 0, second.diagnostics)
        self.assertGreater(second.map_vote_score, 0.0, second.diagnostics)
        self.assertLessEqual(
            abs(second.heading - first.heading),
            math.radians(self.p["corridor_heading_step"]),
        )

    def test_repeated_map_votes_improve_sharp_turn_axis_choice(self):
        scan = _sharp_right_turn_scan()
        baseline = observe(scan, self.p)
        points, _, _, _ = scan_points(scan, self.p)
        expected_heading = math.radians(-45.0)
        forward = np.array([math.cos(expected_heading), math.sin(expected_heading)])
        normal = np.array([-forward[1], forward[0]])
        longitudinal = points @ forward
        lateral = points @ normal
        in_corridor = (
            (longitudinal >= self.p["corridor_x_min"] - 0.5)
            & (longitudinal <= self.p["corridor_x_max"] + self.p["corridor_step"])
        )
        left = points[in_corridor & (lateral > 0.12)]
        right = points[in_corridor & (lateral < -0.12)]
        wall_map = WallMap(self.p)
        for travelled in (0.4, 0.8):
            wall_map.add(
                points, np.zeros(2), 0.0, self.p["lidar_forward"], travelled,
                left_points=left, right_points=right,
            )
        guided = observe(
            scan, self.p, baseline.heading,
            wall_prior=lambda candidate_left, candidate_right: wall_map.score_side_labels(
                candidate_left, candidate_right, np.zeros(2), 0.0,
                self.p["lidar_forward"]),
        )
        self.assertGreaterEqual(guided.map_vote_matches, self.p["corridor_map_min_votes"])
        self.assertLess(
            abs(guided.heading - expected_heading),
            abs(baseline.heading - expected_heading),
            guided.diagnostics,
        )
        self.assertTrue(corridor_is_usable(guided, self.p), guided.diagnostics)

    def test_predicted_path_clearance_reduces_speed_for_wall_ahead(self):
        clear_points = np.array([[1.0, 2.0], [1.0, -2.0]])
        clear_speed, clear_collision, _ = path_clearance_speed(
            clear_points, 0.0, self.p)
        blocked_points = np.array([[0.3, 0.0]])
        blocked_speed, blocked_collision, _ = path_clearance_speed(
            blocked_points, 0.0, self.p)
        self.assertEqual(clear_speed, self.p["map_speed"])
        self.assertTrue(math.isinf(clear_collision))
        self.assertLess(blocked_speed, clear_speed)
        self.assertTrue(math.isfinite(blocked_collision))

    def test_predicted_path_clearance_checks_steered_arc(self):
        curvature = -0.5
        station = 0.6
        obstacle_rear = np.array([
            math.sin(curvature * station) / curvature,
            (1.0 - math.cos(curvature * station)) / curvature,
        ])
        obstacle_lidar = obstacle_rear - np.array([self.p["lidar_forward"], 0.0])
        speed, collision_distance, _ = path_clearance_speed(
            obstacle_lidar.reshape(1, 2), curvature, self.p)
        self.assertLess(speed, self.p["map_speed"])
        self.assertLess(collision_distance, station)
        self.assertGreaterEqual(
            collision_distance,
            station - self.p["car_half_width"] - self.p["wall_clearance"]
            - self.p["map_path_step"],
        )

    def test_single_measured_corridor_sample_is_usable(self):
        sample = WallObservation(
            points=np.empty((0, 2)),
            left=np.empty((0, 2)),
            right=np.empty((0, 2)),
            centre=np.array([[0.675, 0.05]]),
            width=2.43,
            confidence=0.16,
            segments=13,
            residual=0.01,
            front_clearance=2.19,
        )
        self.assertTrue(corridor_is_usable(sample, self.p))
        sample.width = float("nan")
        self.assertFalse(corridor_is_usable(sample, self.p))

    def test_racing_speed_limits_match_cone_defaults(self):
        for key in ("v_max", "race_speed_scale", "a_lat", "a_acc", "a_brake",
                    "mpc_a_lat", "mpc_a_acc", "mpc_a_brake", "throttle_max"):
            self.assertEqual(WALL_DEFAULTS[key], CONE_RACE[key], key)

        angle = np.linspace(0.0, 2.0 * math.pi, 252, endpoint=False)
        circle = 10.0 * np.column_stack((np.cos(angle), np.sin(angle)))
        speeds = speed_plan(circle, self.p)
        self.assertAlmostEqual(float(np.max(speeds)), CONE_RACE["v_max"], places=6)

    def test_closed_track_and_raceline_stay_inside_clearance(self):
        angle = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        path = np.column_stack((7.0 * np.cos(angle), 5.0 * np.sin(angle)))
        track, reason = build_track(path, np.full(len(path), self.p["track_width"]), self.p)
        self.assertIsNotNone(track, reason)
        line = optimize(track, self.p)
        speeds = speed_plan(line, self.p)
        self.assertEqual(line.shape, track["centre"].shape)
        self.assertEqual(speeds.shape, (len(line),))
        self.assertTrue(np.all(np.isfinite(line)))
        self.assertTrue(np.all(np.isfinite(speeds)))
        self.assertTrue(np.all(
            np.linalg.norm(line - track["centre"], axis=1)
            <= track["half_width"] - self.p["car_half_width"] - self.p["wall_clearance"] + 1e-6
        ))

    def test_track_rejects_unmeasured_width(self):
        angle = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        path = np.column_stack((7.0 * np.cos(angle), 5.0 * np.sin(angle)))
        track, reason = build_track(path, np.full(len(path), np.nan), self.p)
        self.assertIsNone(track)
        self.assertIn("width coverage", reason)

    def test_track_rejects_self_intersection(self):
        angle = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        path = np.column_stack((8.0 * np.sin(angle), 5.0 * np.sin(2.0 * angle)))
        track, reason = build_track(path, np.full(len(path), self.p["track_width"]), self.p)
        self.assertIsNone(track)
        self.assertIn("self-intersection", reason)

    def test_map_is_bounded_and_loop_map_freezes(self):
        p = dict(self.p, max_map_points=20, map_voxel=0.05)
        wall_map = WallMap(p)
        x = np.linspace(0.0, 5.0, 200)
        points = np.column_stack((x, np.ones_like(x)))
        wall_map.add(points, np.zeros(2), 0.0, 0.0, 1.0)
        self.assertLessEqual(len(wall_map.points), p["max_map_points"])
        wall_map.close_loop(np.array([0.1, 0.0]), 1.0)
        before = wall_map.points.copy()
        wall_map.add(points, np.zeros(2), 0.0, 0.0, 2.0)
        np.testing.assert_array_equal(wall_map.points, before)


if __name__ == "__main__":
    unittest.main()
