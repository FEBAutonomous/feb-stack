"""Mapping longitudinal control, adapter contract and delayed-plant regressions."""
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import rclpy
except ImportError:
    rclpy = None

if rclpy is not None:
    import numpy as np
    import yaml
    from ackermann_msgs.msg import AckermannDriveStamped
    from feb_tools.sim_adapter import SimAdapter
    from feb_wall_racer.benchmarks.wall import synthetic_corridor_scan
    from feb_wall_racer.core.perception import observe
    from mapping_harness import MappingHarness
    from feb_wall_racer.core.vehicle import BodySpeed, Plant, WHEEL_SPEED_PER_THROTTLE


class Capture:
    def __init__(self):
        self.values = []

    def publish(self, message):
        self.values.append(message)


@unittest.skipIf(rclpy is None, "requires the FEB ROS workspace")
class MappingSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.racer = MappingHarness()
        self.racer.last_fit = observe(synthetic_corridor_scan(), self.racer.p)

    def tearDown(self):
        self.racer.destroy_node()

    def test_verified_gap_commands_positive_throttle_with_one_wall(self):
        from test_gap_matcher import scan_fixture

        self.racer.local.guidance.use_gap_matcher = True
        obs = self.racer.last_fit
        obs.points = scan_fixture(one_wall=True)
        obs.centre = np.empty((0, 2))
        obs.confidence = 0.
        for _ in range(4):
            _, throttle = self.racer.run(obs, .25)
            self.assertGreater(throttle, 0.)
            self.assertEqual(self.racer.refs[-1].reason, 'verified gap')

    def _adapter(self):
        adapter = SimAdapter()
        self.addCleanup(adapter.destroy_node)
        card = yaml.safe_load((Path(__file__).resolve().parents[1] / "driver.yaml").read_text())
        adapter.p.update(card.get("adapter", {}))
        adapter.pub_throttle, adapter.pub_steering = Capture(), Capture()
        return adapter

    def test_small_overspeed_change_does_not_drop_wheel_demand_to_zero(self):
        requests = [self.racer._local_follow(self.racer.last_fit.centre, speed, 1.8)[1]
                    * WHEEL_SPEED_PER_THROTTLE for speed in (1.9, 1.9001)]
        self.assertGreater(min(requests), 0.0)
        self.assertLess(abs(requests[1] - requests[0]), 0.01)

    def test_overspeed_uses_positive_lower_wheel_demand_for_braking(self):
        _, throttle = self.racer._local_follow(self.racer.last_fit.centre, 2.2, 1.8)
        self.assertGreater(throttle, 0.0)
        self.assertLess(throttle * WHEEL_SPEED_PER_THROTTLE, 2.2)

    def test_fallback_overspeed_brakes_without_locking_wheels(self):
        self.racer._local_follow(self.racer.last_fit.centre, 1.8, 1.8)
        _, throttle, fallback, _ = self.racer._mapping_fallback(1.5)
        self.assertTrue(fallback)
        self.assertGreater(throttle, 0.0)
        self.assertLess(throttle * WHEEL_SPEED_PER_THROTTLE, 1.5)

    def test_collision_inside_reaction_distance_keeps_hard_stop(self):
        self.racer.last_fit.points = np.array([[0.05, 0.0], [0.10, 0.0]])
        _, throttle = self.racer._local_follow(self.racer.last_fit.centre, 1.8, 1.8)
        self.assertEqual(throttle, 0.0)

    def test_expired_guidance_keeps_zero_fallback(self):
        self.racer._local_follow(self.racer.last_fit.centre, 1.0, 1.8)
        self.racer.last_mapping_guidance_wall_time = time.monotonic() - 2.0
        _, throttle, fallback, _ = self.racer._mapping_fallback(1.0)
        self.assertFalse(fallback)
        self.assertEqual(throttle, 0.0)

    def test_wall_inside_stopping_distance_keeps_emergency_brake(self):
        self.racer.last_fit.points = np.array([[0.7, 0.0]])
        _, throttle = self.racer._local_follow(self.racer.last_fit.centre, 3.0, 1.8)
        self.assertEqual(throttle, 0.0)
        _, throttle, fallback, _ = self.racer._mapping_fallback(3.0)
        self.assertTrue(fallback)
        self.assertEqual(throttle, 0.0)

    def test_distant_wall_does_not_turn_nominal_overspeed_into_wheel_lock(self):
        self.racer.last_fit.points = np.array([[2.0, 0.0]])
        _, throttle = self.racer._local_follow(self.racer.last_fit.centre, 1.91, 1.8)
        self.assertGreater(throttle, 0.0)

    def test_halted_racer_does_not_advertise_idle_readiness(self):
        self.racer.pub_drive = Capture()
        self.racer.control.fail("test fault")
        self.racer.pub_drive.values.clear()
        self.racer._idle_heartbeat()
        self.assertEqual(self.racer.pub_drive.values, [])

    def test_driver_adapter_preserves_requested_wheel_speed(self):
        adapter = self._adapter()
        command = AckermannDriveStamped()
        command.drive.speed = 1.5
        adapter.on_cmd(command)
        self.assertAlmostEqual(adapter.throttle * WHEEL_SPEED_PER_THROTTLE, 1.5, places=6)

    def test_delayed_mapping_model_has_no_repeated_wheel_lock_brakes(self):
        adapter = self._adapter()
        clock = [1.0]
        adapter.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=int(clock[0] * 1e9)))
        car, body = Plant(), BodySpeed()
        history = [(0.0, 0.0)]
        command = AckermannDriveStamped()
        zero_requests = 0
        for tick in range(1000):  # 20 s, 10 Hz mapping, 50 Hz adapter, .12 s plant delay.
            t = tick * 0.02
            clock[0] = 1.0 + t
            window = adapter.p["speed_window"]
            samples = [wheel for stamp, wheel in history if stamp >= t - window - 1e-8]
            adapter.speed = float(np.mean(samples))
            if tick % 5 == 0:
                estimated = body.update(adapter.speed, 0.1)
                steer, throttle = self.racer._local_follow(self.racer.last_fit.centre, estimated, 1.8)
                command.drive.steering_angle = steer
                command.drive.speed = min(5.0, throttle * WHEEL_SPEED_PER_THROTTLE)
                if t >= 5.0 and command.drive.speed == 0.0:
                    zero_requests += 1
            adapter.on_cmd(command)
            car.command(command.drive.steering_angle, adapter.throttle)
            car.step(0.02)
            history.append((t + 0.02, car.throttle * WHEEL_SPEED_PER_THROTTLE))
            history = history[-15:]
        self.assertEqual(zero_requests, 0)
        self.assertGreater(car.v, 1.5)
        self.assertLess(car.v, 1.95)


if __name__ == "__main__":
    unittest.main()
