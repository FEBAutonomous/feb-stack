import math
from types import SimpleNamespace

import numpy as np
import rclpy
from feb_wall_racer_msgs.msg import LocalizedState, RunContext, WallObservation

from feb_wall_racer.interfaces import header
from feb_wall_racer.nodes.mapping_debug_node import MappingDebugNode, scan_geometry


def test_scan_geometry_uses_the_lidar_offset_in_both_frames():
    walls = SimpleNamespace(points=[.2, .4, .8, -.9])
    state = SimpleNamespace(pose=[3., 5.], yaw=math.pi / 2.)
    local, world, clearance = scan_geometry(walls, state, .25, .34)
    np.testing.assert_allclose(local, [[.45, .4], [1.05, -.9]])
    np.testing.assert_allclose(world, [[2.6, 5.45], [3.9, 6.05]])
    assert clearance == math.hypot(.45 - .34, .4)


def test_empty_scan_has_no_clearance_measurement():
    walls = SimpleNamespace(points=[])
    state = SimpleNamespace(pose=[0., 0.], yaw=0.)
    local, world, clearance = scan_geometry(walls, state, .25, .34)
    assert local.shape == world.shape == (0, 2)
    assert math.isinf(clearance)


def test_viewer_pairs_valid_observations_and_clears_them_on_reset():
    rclpy.init(args=[])
    try:
        node = MappingDebugNode()
        try:
            meta = header('run', 1, 2, 1., 'laser', 'config')
            node.on_context(RunContext(meta=meta, phase='MAPPING'))
            walls = WallObservation(meta=meta, valid=True)
            state = LocalizedState(meta=meta, valid=True)
            node.join_input('walls', walls)
            assert node.sample is None
            node.join_input('state', state)
            assert node.sample[:2] == (walls, state)
            reset = header('run', 2, 0, 2., 'map', 'config')
            node.on_context(RunContext(meta=reset, phase='ARMING'))
            assert node.sample is None
            node.join_input('walls', walls)
            node.join_input('state', state)
            assert node.sample is None
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
