"""Pinned public loop inputs and 9192d76 golden outputs; no simulator truth."""
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rosidl_runtime_py.set_message import set_message_fields
from sensor_msgs.msg import LaserScan,Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String,Bool
from feb_wall_racer_msgs.msg import RunContext
from feb_wall_racer.configuration import node_parameters,pipeline_configuration_id,NODE_SETTINGS
from feb_wall_racer.interfaces import header,points,decode_observation,ros_time
from feb_wall_racer.nodes.perception_node import PerceptionNode
from feb_wall_racer.nodes.motion_estimator_node import MotionEstimatorNode
from feb_wall_racer.nodes.slam_node import SlamNode
from feb_wall_racer.nodes.local_planner_node import LocalPlannerNode
from feb_wall_racer.nodes.mpc_controller_node import MPCControllerNode


def restore(kind,values):
    values=dict(values)
    if kind is LaserScan:
        values['ranges']=[float('inf') if v is None else v for v in values['ranges']]
    msg=kind();set_message_fields(msg,values)
    return msg


@pytest.mark.parametrize('use_gap_matcher', [False, True], ids=['legacy', 'gap-matcher'])
def test_recorded_mapping_matches_frozen_oracle_per_selected_scan(use_gap_matcher):
    track = 'loop'
    with gzip.open(Path(__file__).parent/'fixtures/recorded_mapping.json.gz','rt') as source:
        artifact=json.load(source)
    assert artifact['oracle_commit']=='9192d76' and artifact['numerical_tolerance']==1e-8
    window=artifact['windows'][track];p=window['parameters']
    config_id=pipeline_configuration_id(p,NODE_SETTINGS)
    def options(name):
        values=node_parameters(p,name);values['configuration_id']=config_id
        return dict(parameter_overrides=[Parameter(k,value=v) for k,v in values.items()])
    rclpy.init(args=[])
    nodes=[]
    try:
        perception=PerceptionNode(**options('perception'));nodes.append(perception)
        motion=MotionEstimatorNode(**options('motion_estimator'));nodes.append(motion)
        slam=SlamNode(**options('slam'));nodes.append(slam)
        local=LocalPlannerNode(**options('local_planner'));nodes.append(local)
        with patch('feb_wall_racer.nodes.mpc_controller_node.KinematicMPC',return_value=SimpleNamespace(ok=False)):
            control=MPCControllerNode(**options('mpc_controller'));nodes.append(control)
        observations=[];commands=[];references=[];clock=[window['frames'][0]['clock']];mono=[100.]
        control._control_time=lambda:clock[0]
        control.pub_drive=SimpleNamespace(publish=commands.append)
        def reference(msg):
            references.append(msg)
            control.input('reference',msg)
        local.pub_reference=SimpleNamespace(publish=reference)
        def state(msg):
            control.input('state',msg);local.input('state',msg)
        slam.pub_state=SimpleNamespace(publish=state)
        slam.pub_context=SimpleNamespace(publish=perception.on_prior)
        motion.pub_motion=SimpleNamespace(publish=lambda msg:slam.input('motion',msg))
        def walls(msg):
            observations.append(msg)
            control.on_walls(msg)
            slam.input('walls',msg);local.input('walls',msg);motion.on_walls(msg)
        perception.pub_walls=SimpleNamespace(publish=walls)
        context=RunContext(meta=header('fixture-'+track,1,0,clock[0],'map',config_id),phase='ARMING',live=True)
        with patch('time.monotonic',side_effect=lambda:mono[0]):
            for node in nodes:node.on_context(context)
            local.guidance.use_gap_matcher = use_gap_matcher
            control.on_state(String(data='AUTONOMOUS: fixture'));control.on_mission_live(Bool(data=True))
            context=RunContext(meta=context.meta,phase='MAPPING',live=True)
            for i,frame in enumerate(window['frames']):
                clock[0]=frame['clock'];mono[0]=frame['monotonic']
                for node in nodes:node.on_context(context)
                odom=restore(Odometry,frame['inputs']['odom'])
                imu=restore(Imu,frame['inputs']['imu'])
                scan=restore(LaserScan,frame['inputs']['scan'])
                motion.sensor('odom',odom);motion.sensor('imu',imu)
                perception.on_odom(odom);perception.on_imu(imu)
                motion.core.commanded_steer=frame['steer']
                perception.on_scan(scan)
                expected=frame['expected'];label=track+':'+str(i)
                assert bool(control.fault)==expected['halted'],label
                np.testing.assert_allclose(slam.core.pose,expected['pose'],atol=1e-8,rtol=0,err_msg=label)
                np.testing.assert_allclose([slam.core.yaw,motion.core.body.v,slam.core.travelled],
                    [expected['yaw'],expected['body_speed'],expected['travelled']],atol=1e-8,rtol=0,err_msg=label)
                if use_gap_matcher:
                    assert 0. <= commands[-1].drive.speed <= control.p['mode_speed_limit'],label
                    assert abs(commands[-1].drive.steering_angle) <= control.p['max_steer'],label
                    if references[-1].emergency:
                        assert commands[-1].drive.speed == 0.,label
                else:
                    np.testing.assert_allclose([commands[-1].drive.speed,commands[-1].drive.steering_angle],
                        [expected['wheel_demand'],expected['steering']],atol=1e-8,rtol=0,err_msg=label)
                if expected['observation'] is not None:
                    actual=decode_observation(observations[-1])
                    for key,value in expected['observation'].items():
                        result=getattr(actual,key)
                        if value is None:assert not np.isfinite(result),label+'/'+key
                        else:
                            target=np.asarray(value).reshape(-1,2) if key in ('left','right','centre') else value
                            np.testing.assert_allclose(result,target,atol=1e-8,rtol=0,err_msg=label+'/'+key)
        np.testing.assert_allclose(slam.core.slam.points,np.asarray(window['final_map']).reshape(-1,2),atol=1e-8,rtol=0)
        if use_gap_matcher:
            assert any(msg.reason in ('paired corridor', 'verified gap') for msg in references)
    finally:
        for node in reversed(nodes):node.destroy_node()
        rclpy.shutdown()
