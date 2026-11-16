"""Physical overrides must change prediction/control, not just parameter services."""
import math
import unittest
import numpy as np
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.controller import RaceController
from feb_wall_racer.core.vehicle import Plant, BodySpeed, throttle_for
try:
    from feb_wall_racer.core.vehicle import VehicleModel
except ImportError:
    VehicleModel = None


class VehicleConfigurationTests(unittest.TestCase):
    def model(self,**values):
        self.assertIsNotNone(VehicleModel,"per-car vehicle model is required")
        return VehicleModel.from_parameters(dict(DEFAULTS,**values))

    def test_independent_models_change_wheelbase_servo_and_delay(self):
        slow = self.model(wheelbase=.6,steer_rate=.5,act_delay=.3)
        fast = self.model(wheelbase=.3,steer_rate=3.,act_delay=.1)
        a,b = Plant(v=2.,model=slow),Plant(v=2.,model=fast)
        for car in (a,b):
            car.command(.4,2./25.11)
            car.step(.2)
        self.assertAlmostEqual(a.delta,0.)
        self.assertGreater(b.delta,0.)
        self.assertGreater(b.psi,a.psi)
        self.assertEqual(slow.wheelbase,.6)

    def test_controller_prediction_uses_configured_wheelbase(self):
        short = RaceController(dict(DEFAULTS,wheelbase=.3),None)
        long = RaceController(dict(DEFAULTS,wheelbase=.6),None)
        a = short.predict(10.,0.,0.,0.,2.,.2)
        b = long.predict(10.,0.,0.,0.,2.,.2)
        self.assertGreater(a[2],1.8*b[2])

    def test_controller_fallback_honors_smaller_steer_limit(self):
        p = dict(DEFAULTS,max_steer=.1)
        c = RaceController(p,None)
        theta = np.linspace(0,2*math.pi,100,endpoint=False)
        c.set_raceline(np.column_stack((2*np.sin(theta),2*(1-np.cos(theta)))),np.ones(100))
        steer,_ = c.fallback(10.,np.array([0.,-1.,0.,1.,0.]),0.)
        self.assertLessEqual(abs(steer),.1+1e-12)

    def test_car_centre_offset_changes_actual_path_projection(self):
        theta = np.linspace(0,2*math.pi,300,endpoint=False)
        line = np.column_stack((20*np.sin(theta),20*(1-np.cos(theta))))
        controllers = [RaceController(dict(DEFAULTS,axle_to_centre=offset),None) for offset in (0.,.3)]
        for controller in controllers:
            controller.set_raceline(line,np.ones(len(line))*3.)
            controller.step(10.,0.,0.,0.,2.,0.)
        self.assertGreater(controllers[1].s-controllers[0].s,.2)

    def test_mpc_dynamics_use_configured_geometry_and_drag(self):
        from feb_wall_racer.core.mpc import KinematicMPC,ca
        if ca is None:
            self.skipTest("CasADi model checked with .pydeps in the FEB box")
        a,b = KinematicMPC.__new__(KinematicMPC),KinematicMPC.__new__(KinematicMPC)
        a.vehicle = self.model(wheelbase=.3,drag=.1,corner_drag=0.)
        b.vehicle = self.model(wheelbase=.6,drag=.5,corner_drag=0.)
        z,u = ca.DM([0.,0.,0.,2.,.2]),ca.DM([1.,0.])
        first,second = np.array(a.dynamics(z,u)).ravel(),np.array(b.dynamics(z,u)).ravel()
        self.assertAlmostEqual(first[2],2*second[2])
        self.assertGreater(first[3],second[3])

    def test_default_force_conversion_matches_working_calibration(self):
        model = self.model()
        # Characterization samples evaluated using the saved pre-refactor code.
        for speed,force,expected in ((0.,3.,.007168458781362007),
                                     (2.,-3.,.07483738218505244),
                                     (4.,5.,.17533961679720342)):
            self.assertEqual(model.command_for_force(speed,force),expected)
            self.assertEqual(throttle_for(speed,force),expected)

    def test_force_conversion_uses_each_models_calibration(self):
        a = self.model(speed_per_throttle=10.,mu_peak=7.,slip_peak=.1)
        b = self.model(speed_per_throttle=20.,mu_peak=7.,slip_peak=.1)
        self.assertAlmostEqual(a.command_for_force(2.,3.)*10.,b.command_for_force(2.,3.)*20.)
        self.assertGreater(a.command_for_force(2.,3.),b.command_for_force(2.,3.))

    def test_invalid_physical_values_fail_clearly(self):
        for override in (dict(wheelbase=0.),dict(max_steer=.8),dict(steer_rate=-1.),
                         dict(act_delay=float("nan")),dict(speed_per_throttle=0.),
                         dict(slip_peak=.4,slip_slide=.3)):
            with self.subTest(override=override),self.assertRaises(ValueError):
                self.model(**override)


if __name__ == "__main__":
    unittest.main()
