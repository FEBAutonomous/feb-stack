"""Guide contract: base -> car-only differences -> explicit override, with caps."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.raceline import speed_plan
from feb_wall_racer.core.slam import WallMap
try:
    from feb_wall_racer.configuration import effective_parameters, parameter_files, mode_speed_limit
except ImportError:
    effective_parameters = parameter_files = mode_speed_limit = None


class ParameterContractTests(unittest.TestCase):
    def test_base_yaml_documents_every_live_startup_tuning_name(self):
        import yaml
        path = Path(__file__).resolve().parents[1]/"config/driver.yaml"
        values = yaml.safe_load(path.read_text())["wall_racer"]["ros__parameters"]
        self.assertEqual(set(values),set(DEFAULTS)-{"mode_speed_limit"})

    def test_race_speed_scale_changes_the_actual_profile(self):
        angle = np.linspace(0,2*np.pi,160,endpoint=False)
        line = 10*np.column_stack((np.cos(angle),np.sin(angle)))
        base = speed_plan(line,DEFAULTS)
        reduced = speed_plan(line,dict(DEFAULTS,race_speed_scale=.5))
        np.testing.assert_allclose(reduced,.5*base)

    def test_nominal_width_bounds_map_associations(self):
        narrow = WallMap(dict(DEFAULTS,track_width=1.))
        normal = WallMap(DEFAULTS)
        self.assertLess(narrow.match_gate,normal.match_gate)

    def test_mode_caps_mapping_and_racing_without_changing_geometry(self):
        self.assertTrue(callable(effective_parameters))
        p = effective_parameters(dict(DEFAULTS,mode_speed_limit=.8,wheelbase=.41))
        self.assertEqual(p["wheelbase"],.41)
        self.assertLessEqual(p["map_speed"],.8)
        self.assertLessEqual(p["v_max"],.8)
        self.assertLessEqual(p["blind_speed"],.8)

    def test_car_overlay_only_on_car_and_extra_is_last(self):
        self.assertTrue(callable(parameter_files))
        with tempfile.TemporaryDirectory() as temp:
            base,car,extra = [Path(temp)/name for name in ("driver.yaml","driver.car.yaml","override.yaml")]
            for path in (base,car,extra):
                path.write_text("wall_racer:\n  ros__parameters:\n    wheelbase: 0.4\n")
            self.assertEqual(parameter_files(base,car,extra,False),[str(base),str(extra)])
            self.assertEqual(parameter_files(base,car,extra,True),[str(base),str(car),str(extra)])

    def test_unknown_parameter_or_wrong_node_selector_is_not_silently_ignored(self):
        self.assertTrue(callable(parameter_files))
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/"driver.yaml"
            for text in ("racer:\n  ros__parameters:\n    wheelbase: 0.4\n",
                         "wall_racer:\n  ros__parameters:\n    wheeelbase: 0.4\n",
                         "wall_racer:\n  ros__parameters: {}\nracer:\n  ros__parameters:\n    wheelbase: 0.45\n"):
                base.write_text(text)
                with self.assertRaises(ValueError):
                    parameter_files(base,None,None,False)

    def test_card_mode_is_validated(self):
        self.assertTrue(callable(mode_speed_limit))
        with tempfile.TemporaryDirectory() as temp:
            card = Path(temp)/"driver.yaml"
            card.write_text("modes:\n  inspection: {max_speed: 0.8}\n")
            self.assertEqual(mode_speed_limit(card,"inspection"),.8)
            with self.assertRaises(ValueError):
                mode_speed_limit(card,"typo")

    def test_launcher_owned_clock_is_not_silently_overridden_in_yaml(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"clock.yaml"
            path.write_text("wall_racer:\n  ros__parameters:\n    use_sim_time: false\n")
            with self.assertRaises(ValueError):
                parameter_files(path,None,None,False)

    def test_encoder_radius_is_owned_upstream_of_public_odometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"radius.yaml"
            path.write_text("wall_racer:\n  ros__parameters:\n    wheel_radius: 0.08\n")
            with self.assertRaises(ValueError):
                parameter_files(path,None,None,False)

    def test_invalid_range_or_time_constants_are_rejected(self):
        self.assertTrue(callable(effective_parameters))
        for override in (dict(sensor_timeout=0.),dict(race_speed_scale=2.),
                         dict(lost_after=4.,stop_after=1.),dict(throttle_min=.6,throttle_max=.4)):
            with self.subTest(override=override),self.assertRaises(ValueError):
                effective_parameters(dict(DEFAULTS,**override))

    def test_unsafe_dimensions_and_empty_sampling_domains_fail_at_startup(self):
        self.assertTrue(callable(effective_parameters))
        for override in (dict(car_half_width=-.1),dict(wall_clearance=-.1),
                         dict(map_path_step=0.),dict(map_path_horizon=0.),
                         dict(map_path_reaction_distance=-.1),
                         dict(corridor_x_min=3.,corridor_x_max=2.),
                         dict(min_track_length=50.,max_track_length=20.)):
            with self.subTest(override=override),self.assertRaises(ValueError):
                effective_parameters(dict(DEFAULTS,**override))


if __name__ == "__main__":
    unittest.main()
