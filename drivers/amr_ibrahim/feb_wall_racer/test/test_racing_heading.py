"""Heading-only tests; sensor orientation stays in the public ROS contract."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
try:
    from feb_wall_racer.core.heading import HeadingTracker, quaternion_yaw
except ImportError:
    HeadingTracker = quaternion_yaw = None


class RacingHeadingTests(unittest.TestCase):
    def tracker(self):
        self.assertIsNotNone(HeadingTracker,"public IMU heading tracker must exist")
        return HeadingTracker()

    def test_initial_frame_offset_and_later_odom_drift(self):
        h = self.tracker()
        h.observe(math.pi/2,1.,0.)
        yaw,active = h.resolve(0.,1.)
        self.assertTrue(active)
        self.assertAlmostEqual(yaw,0.)
        h.observe(math.pi/2+.2,2.,.2)
        yaw,active = h.resolve(.3,2.)
        self.assertAlmostEqual(yaw,.2)
        self.assertTrue(active)
        self.assertAlmostEqual(h.correction,-.1)

    def test_wrap_and_interpolation_use_short_rotation(self):
        h = self.tracker()
        h.observe(math.radians(179),1.,0.)
        h.resolve(0.,1.)
        h.observe(math.radians(-179),1.1,0.)
        yaw,active = h.resolve(math.radians(1),1.05)
        self.assertTrue(active)
        self.assertAlmostEqual(yaw,math.radians(1))

    def test_stale_orientation_carries_last_correction_without_jump(self):
        h = self.tracker()
        h.observe(1.,1.,0.)
        h.resolve(0.,1.)
        h.observe(1.2,2.,0.)
        h.resolve(.3,2.)
        yaw,active = h.resolve(.6,3.)
        self.assertFalse(active)
        self.assertAlmostEqual(yaw,.5)

    def test_frame_reset_is_rejected(self):
        h = self.tracker()
        h.observe(0.,1.,0.)
        h.resolve(0.,1.)
        h.observe(math.pi/2,1.1,0.)
        yaw,active = h.resolve(.01,1.1)
        self.assertFalse(active)
        self.assertAlmostEqual(yaw,.01)

    def test_invalid_and_unavailable_quaternions_are_not_headings(self):
        self.assertTrue(callable(quaternion_yaw),"validate orientation before using it")
        for q,cov in [((0,0,0,0),0.),((0,0,float("nan"),1),0.),((0,0,0,1),-1.),((0,0,0,1),float("nan"))]:
            self.assertIsNone(quaternion_yaw(q,cov))
        self.assertAlmostEqual(quaternion_yaw((0,0,math.sin(.2),math.cos(.2)),0.),.4)

    def test_run_reset_accepts_a_new_source_clock_epoch(self):
        h = self.tracker()
        h.observe(0.,100.,0.)
        h.resolve(0.,100.)
        h.reset_reference()
        h.observe(.5,1.,0.)
        yaw,active = h.resolve(0.,1.)
        self.assertTrue(active)
        self.assertAlmostEqual(yaw,0.)

    def test_odom_displacement_rotates_into_heading_corrected_map_frame(self):
        try:
            from feb_wall_racer.core.mapping import MappingOperations as WallRacer
        except ImportError:
            self.skipTest("ROS method checked in FEB box")
        node = SimpleNamespace(odom_position=np.array([math.cos(.1),math.sin(.1)]),
            last_odom_position=np.zeros(2),pose=np.zeros(2),travelled=0.,
            heading_correction=-.1,last_integrated_heading_correction=-.1)
        WallRacer._integrate(node,1.,1.,True)
        np.testing.assert_allclose(node.pose,[1.,0.],atol=1e-12)


if __name__ == "__main__":
    unittest.main()
