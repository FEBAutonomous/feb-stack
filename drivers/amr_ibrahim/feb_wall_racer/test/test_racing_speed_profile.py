"""Speed-profile-only feasibility regressions on the exact supplied path."""
import math
import unittest
import numpy as np
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.raceline import curvature, speed_plan


class RacingSpeedProfileTests(unittest.TestCase):
    def test_signed_curvature_cancellation_cannot_hide_peak(self):
        theta = np.linspace(0,2*math.pi,200,endpoint=False)
        line = np.column_stack((6*np.cos(theta),4*np.sin(theta)))
        line[20] += [0.,.18]
        original = line.copy()
        v = speed_plan(line,DEFAULTS)
        self.assertTrue(np.all(v*v*abs(curvature(line)) <= DEFAULTS["a_lat"]+1e-8))
        np.testing.assert_array_equal(line,original)

    def test_minimum_speed_does_not_override_tight_corner_limit(self):
        theta = np.linspace(0,2*math.pi,100,endpoint=False)
        line = .05*np.column_stack((np.cos(theta),np.sin(theta)))
        v = speed_plan(line,DEFAULTS)
        self.assertLess(float(v.max()),.7)
        self.assertTrue(np.all(v*v*abs(curvature(line)) <= DEFAULTS["a_lat"]+1e-8))

    def test_cyclic_acceleration_and_braking_limits_survive_peak_caps(self):
        theta = np.linspace(0,2*math.pi,160,endpoint=False)
        line = np.column_stack((8*np.cos(theta),3*np.sin(theta)))
        line[0] += [.12,-.2]  # a restrictive point at the wrap seam
        v = speed_plan(line,DEFAULTS)
        ds = np.linalg.norm(np.roll(line,-1,axis=0)-line,axis=1)
        dv2 = np.roll(v,-1)**2-v**2
        self.assertTrue(np.all(dv2 <= 2*DEFAULTS["a_acc"]*ds+1e-8))
        self.assertTrue(np.all(-dv2 <= 2*DEFAULTS["a_brake"]*ds+1e-8))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(np.all(v>0))


if __name__ == "__main__":
    unittest.main()
