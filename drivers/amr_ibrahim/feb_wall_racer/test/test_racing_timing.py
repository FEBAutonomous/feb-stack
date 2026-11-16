"""Timing-only regressions: no simulator and no changed controller gains."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.controller import RaceController
from feb_wall_racer.core.vehicle import Plant, WHEEL_SPEED_PER_THROTTLE
from feb_wall_racer.benchmarks.wall import synthetic_corridor_scan
from feb_wall_racer.core.perception import observe


class RacingTimingTests(unittest.TestCase):
    def setUp(self):
        self.controller = RaceController(dict(DEFAULTS), None)

    def test_prediction_includes_state_age_and_delayed_history(self):
        c = self.controller
        c.sent = [(9.7, .05, 3.2/WHEEL_SPEED_PER_THROTTLE),
                  (9.95, .1, 4.0/WHEEL_SPEED_PER_THROTTLE)]
        expected = Plant(0.,0.,0.,3.)
        expected.t = 9.8
        expected.steer_cmd, expected.throttle = 0., 3./DEFAULTS["speed_per_throttle"]
        expected.queue = [(t+DEFAULTS["act_delay"],s,h) for t,s,h in c.sent]
        expected.step(.32,h=.004)
        result = c.predict(10.,0.,0.,0.,3.,0.,state_age=.2)
        np.testing.assert_allclose(result,[expected.x,expected.y,expected.psi,expected.v,expected.delta],atol=1e-7)

    def test_invalid_state_age_is_rejected(self):
        for age in (-.1,float("nan"),float("inf")):
            with self.subTest(age=age), self.assertRaises(ValueError):
                self.controller.predict(10.,0.,0.,0.,3.,0.,state_age=age)

    def test_external_publication_records_actual_command_time_once(self):
        theta = np.linspace(0,2*np.pi,100,endpoint=False)
        self.controller.set_raceline(np.column_stack((10*np.sin(theta),10*(1-np.cos(theta)))),np.ones(100)*3.)
        steer,throttle = self.controller.step(10.,0.,0.,0.,2.,0.,state_age=.1,record_command=False)
        self.assertEqual(self.controller.sent,[])
        record = getattr(self.controller,"record_command",None)
        self.assertTrue(callable(record),"controller must accept actual publication timestamps")
        record(10.04,steer,throttle)
        self.assertEqual(self.controller.sent,[(10.04,steer,throttle)])

    def test_racing_observation_skips_corridor_search_retains_walls_and_stop_range(self):
        scan = synthetic_corridor_scan()
        original = observe(scan,DEFAULTS)
        with patch("feb_wall_racer.core.perception._corridor_analysis",side_effect=AssertionError("mapping-only search")):
            fast = observe(scan,DEFAULTS,with_corridor=False)
        self.assertGreater(len(fast.left)+len(fast.right),12)
        self.assertEqual(fast.segments,original.segments)
        self.assertEqual(fast.front_clearance,original.front_clearance)
        np.testing.assert_array_equal(fast.points,original.points)
        self.assertEqual(len(fast.centre),0)

    def test_sensor_queue_keeps_only_latest_sample(self):
        try:
            from feb_wall_racer.qos import SENSOR_QOS
        except ImportError:
            self.skipTest("ROS QoS checked inside FEB box")
        self.assertEqual(SENSOR_QOS.depth,1)

    def test_slow_or_paused_ros_clock_never_adds_unscaled_wall_seconds(self):
        try:
            from feb_wall_racer.nodes.mpc_controller_node import MPCControllerNode as WallRacer
        except ImportError:
            self.skipTest("ROS clock checked in FEB box")
        clock = SimpleNamespace(nanoseconds=10_050_000_000)
        node = SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:clock),
                               _control_anchor=(10.,100.))
        with patch("time.monotonic",return_value=100.2):
            self.assertAlmostEqual(WallRacer._control_time(node),10.05)
        node._control_anchor = (10.05,100.2)
        with patch("time.monotonic",return_value=100.4):
            self.assertAlmostEqual(WallRacer._control_time(node),10.05)


if __name__ == "__main__":
    unittest.main()
