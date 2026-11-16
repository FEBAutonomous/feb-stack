"""Validate startup tuning and the member guide's configuration layering."""
import math
from pathlib import Path
import yaml
from .params import DEFAULTS
from .core.vehicle import VehicleModel


def _document(path):
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document,dict):
        raise ValueError("configuration must be a YAML mapping: %s" % path)
    return document


def parameter_files(base, car=None, extra=None, on_car=False):
    paths = [base]+([car] if on_car else [])+([extra] if extra else [])
    for path in paths:
        if path is None or not Path(path).is_file():
            raise ValueError("wall racer parameter file is missing: %s" % path)
        document = _document(path)
        unexpected = set(document)-{"/**","wall_racer","/wall_racer"}
        if unexpected:
            raise ValueError("unsupported node selectors in %s: %s" % (path,", ".join(sorted(map(str,unexpected)))))
        found = False
        for selector in ("/**","wall_racer","/wall_racer"):
            if selector not in document:
                continue
            block = document[selector]
            values = block.get("ros__parameters") if isinstance(block,dict) else None
            if not isinstance(values,dict):
                raise ValueError("%s needs a ros__parameters mapping: %s" % (selector,path))
            found = True
            unknown = set(values)-set(DEFAULTS)-{"use_sim_time"}
            if unknown:
                raise ValueError("unknown wall_racer parameters in %s: %s" % (path,", ".join(sorted(unknown))))
            if {"mode_speed_limit","use_sim_time"} & set(values):
                raise ValueError("mode_speed_limit/use_sim_time are set by the driver card and FEB_CAR, not parameter YAML")
        if not found:
            raise ValueError("parameter file must select wall_racer: %s" % path)
    return [str(path) for path in paths]


def mode_speed_limit(card,mode):
    document = _document(card)
    try:
        cap = document["modes"][mode]["max_speed"]
    except (KeyError,TypeError):
        raise ValueError("unknown or incomplete wall racer mode: %s" % mode) from None
    if isinstance(cap,bool) or not isinstance(cap,(int,float)) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("mode max_speed must be a finite positive m/s value")
    return float(cap)


def effective_parameters(values):
    p = dict(values)
    for key,value in p.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool) and not math.isfinite(value):
            raise ValueError("%s must be finite" % key)
    VehicleModel.from_parameters(p)
    positive = ("sensor_timeout","status_period","mode_speed_limit","map_speed","v_max","v_cap",
                "a_lat","a_acc","a_brake","map_a_lat","map_a_brake","mpc_a_lat","mpc_a_acc",
                "mpc_a_brake","mpc_steer_rate","mpc_dt","mpc_horizon","mpc_substeps","mpc_max_iter",
                "corridor_step","map_voxel","sample_step","lookahead","pursuit_lookahead",
                "track_width","track_width_min","track_width_max","map_match_gate","curv_window",
                "car_half_width","map_path_step","map_path_horizon","obstacle_stop_range",
                "adjacent_gap","split_tolerance","min_segment_length","min_segment_points",
                "max_line_segments","keyframe_distance","path_sample_distance","min_loop_distance",
                "loop_radius","min_track_length","max_track_length","max_map_points",
                "corridor_heading_step","corridor_map_min_votes","corridor_map_candidate_count",
                "corridor_map_seed_samples","scan_match_min_eigen_ratio",
                "scan_match_max_translation","scan_match_max_yaw","scan_match_max_residual")
    for key in positive:
        if p[key] <= 0:
            raise ValueError("%s must be positive" % key)
    if p["scan_match_min_eigen_ratio"] >= 1 or p["scan_match_max_yaw"] > .35:
        raise ValueError("scan matcher eigen ratio must be below 1 and yaw bound at most .35 rad")
    if not 0 < p["race_speed_scale"] <= 1:
        raise ValueError("race_speed_scale must be in (0, 1]; raise physical limits explicitly for higher speeds")
    if not 0 <= p["throttle_min"] <= p["throttle_max"] <= 1:
        raise ValueError("throttle bounds must satisfy 0 <= min <= max <= 1")
    if not 0 <= p["lost_after"] < p["stop_after"]:
        raise ValueError("stop_after must be greater than nonnegative lost_after")
    if not 0 <= p["min_range"] < p["max_range"] or p["track_width_min"] >= p["track_width_max"]:
        raise ValueError("range/track-width minimum must be below its maximum")
    nonnegative = ("wall_clearance","map_path_reaction_distance","mapping_fallback_duration",
                   "mapping_fallback_speed","corridor_x_min","corridor_heading_max",
                   "corridor_heading_window","max_loop_correction")
    for key in nonnegative+tuple(key for key in p if key.startswith("w_")):
        if p[key] < 0:
            raise ValueError("%s must be nonnegative" % key)
    if p["map_path_horizon"] < p["map_path_step"] or p["corridor_x_max"] <= p["corridor_x_min"]:
        raise ValueError("sampling horizon must contain a forward sample")
    if p["max_track_length"] < p["min_track_length"]:
        raise ValueError("max_track_length must be at least min_track_length")
    for key in ("min_corridor_confidence","corridor_map_seed_confidence","min_width_coverage"):
        if not 0 <= p[key] <= 1:
            raise ValueError("%s must be in [0, 1]" % key)
    p["v_max"] = min(p["v_max"],p["mode_speed_limit"])
    p["map_speed"] = min(p["map_speed"],p["mode_speed_limit"])
    p["blind_speed"] = min(p["blind_speed"],p["mode_speed_limit"])
    p["v_cap"] = min(p["v_cap"],p["v_max"]*p["race_speed_scale"]+.3)
    p["mpc_steer_rate"] = min(p["mpc_steer_rate"],p["steer_rate"])
    for key in ("a_acc","a_brake","map_a_brake","mpc_a_acc","mpc_a_brake"):
        p[key] = min(p[key],p["mu_peak"])
    for key in ("a_lat","map_a_lat","mpc_a_lat"):
        p[key] = min(p[key],p["lat_peak"])
    return p


# Shared calibration is intentional: every process validates the same physical
# envelope. Algorithm settings below are passed only to their consuming nodes.
NODE_NAMES = ('perception','motion_estimator','slam','global_planner',
              'local_planner','mpc_controller','run_manager','visualization')
_COMMON = set(VehicleModel.defaults())-{'wheel_radius'} | set('''
mode_speed_limit sensor_timeout status_period scan_topic odom_topic imu_topic
drive_topic state_topic debug track_width track_width_min track_width_max
wall_clearance car_half_width lidar_forward map_speed v_max blind_speed
lost_after stop_after min_corridor_confidence corridor_map_seed_samples
corridor_map_seed_confidence path_sample_distance sample_step
'''.split())
_OWNED = {
    'perception': set('''min_range max_range min_segment_points min_segment_length
adjacent_gap split_tolerance max_line_segments corridor_x_min corridor_x_max
corridor_step corridor_heading_max corridor_heading_step corridor_heading_window
corridor_map_min_votes corridor_map_candidate_count obstacle_stop_range
map_voxel map_match_gate'''.split()),
    'motion_estimator': set(),
    'slam': set('''min_loop_distance loop_radius max_loop_correction keyframe_distance
map_voxel max_map_points map_match_gate scan_match_min_eigen_ratio
scan_match_max_translation scan_match_max_yaw scan_match_max_residual obstacle_stop_range'''.split()),
    'global_planner': set('''min_track_length max_track_length min_width_coverage
a_lat a_acc a_brake race_speed_scale curv_window'''.split()),
    'local_planner': set('''lookahead map_a_lat map_a_brake map_path_horizon
map_path_step map_path_reaction_distance mapping_fallback_duration
mapping_fallback_speed obstacle_stop_range corridor_step'''.split()),
    'mpc_controller': set('''mpc_horizon mpc_dt mpc_substeps mpc_max_iter mpc_a_acc
mpc_a_brake mpc_a_lat mpc_steer_rate v_cap w_lat w_lon w_head w_speed w_terminal
w_force w_omega w_dforce w_domega w_slack throttle_min throttle_max
pursuit_lookahead obstacle_stop_range mapping_fallback_duration'''.split()),
    'run_manager': set(), 'visualization': set(),
}


def node_parameters(p, node):
    """Return the declared consumers, failing closed when tuning gains no owner."""
    if node not in _OWNED:
        raise ValueError('unknown pipeline node: '+node)
    coverage = _COMMON | set().union(*_OWNED.values())
    if coverage != set(DEFAULTS):
        raise ValueError('parameter ownership mismatch: '+str(coverage ^ set(DEFAULTS)))
    return {k:v for k,v in p.items() if k in _COMMON | _OWNED[node]}


def configuration_id(p):
    """Stable identity for the complete effective startup configuration."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(p,sort_keys=True,allow_nan=False,
                                     separators=(',',':')).encode()).hexdigest()[:16]


NODE_SETTINGS={name:{} for name in NODE_NAMES}
NODE_SETTINGS['run_manager']={'reset_timeout':5.,'planning_timeout':2.}
NODE_SETTINGS['global_planner']={'planning_timeout':2.}
NODE_SETTINGS['mpc_controller']={'worker_timeout':.5}


def load_driver_parameters(base,car=None,extra=None,on_car=False,mode_limit=5.):
    """Normalize existing wall_racer selectors before distributing owned subsets."""
    result=dict(DEFAULTS)
    for path in parameter_files(base,car,extra,on_car):
        document=_document(path)
        for selector in ('/**','wall_racer','/wall_racer'):
            result.update(document.get(selector,{}).get('ros__parameters',{}))
    result['mode_speed_limit']=mode_limit
    return effective_parameters(result)


def validate_node_settings(values,node):
    if node not in NODE_SETTINGS or set(values)-set(NODE_SETTINGS[node]):
        raise ValueError('unknown node runtime settings: '+node)
    result=dict(NODE_SETTINGS[node],**values)
    for key,value in result.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(node+'.'+key+' must be finite and positive')
        result[key]=float(value)
    return result


def load_node_settings(paths):
    if isinstance(paths,(str,Path)):paths=[paths]
    result={name:dict(values) for name,values in NODE_SETTINGS.items()}
    for path in paths:
        document=_document(path)
        if set(document)-set(NODE_NAMES):
            raise ValueError('unknown pipeline nodes in '+str(path))
        for node,values in document.items():
            if not isinstance(values,dict):raise ValueError('node settings must be a mapping: '+node)
            result[node]=validate_node_settings(dict(result[node],**values),node)
    return result


def pipeline_configuration_id(p,settings):
    return configuration_id({'driver':p,'nodes':settings})
