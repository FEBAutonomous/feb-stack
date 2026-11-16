import importlib
import math
import numpy as np
import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from feb_wall_racer.configuration import effective_parameters
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.vehicle import BodySpeed,VehicleModel


def engine():
    m=importlib.import_module('feb_wall_racer.core.estimation')
    return m.MotionEstimator(effective_parameters(DEFAULTS))


def feed(n,t=1.,speed=1.):
    odom=Odometry()
    odom.header.stamp.sec=int(t)
    odom.header.stamp.nanosec=round((t-int(t))*1e9)
    odom.pose.pose.orientation.w=1.
    odom.twist.twist.linear.x=speed
    imu=Imu()
    imu.header.stamp=odom.header.stamp
    imu.orientation.w=1.
    n.on_odom(odom)
    n.on_imu(imu)


def test_observer_advances_once_per_scan_including_unusable_corridor():
    n=engine()
    feed(n)
    oracle=BodySpeed(n.vehicle)
    expected=oracle.update(1.,.1,0.,0.)
    first=n.step(1.)
    assert first['dt']==.1
    assert first['body_speed']==pytest.approx(expected,abs=1e-8)
    # Input callbacks alone must not advance body-speed dynamics.
    feed(n,1.005)
    second=n.step(1.005)
    assert second['dt']==.01
    assert second['body_speed']==pytest.approx(oracle.update(1.,.01,0.,0.),abs=1e-8)


def test_invalid_time_and_stale_sensors_cannot_produce_fresh_motion():
    n=engine()
    feed(n)
    n.step(1.)
    with pytest.raises(ValueError,match='timestamp'):
        n.step(1.)
    feed(n,2.)
    with pytest.raises(ValueError,match='timestamp'):
        n.step(2.)


def test_cumulative_travel_survives_dropped_downstream_snapshots():
    n=engine()
    outputs=[]
    for i in range(4):
        feed(n,1.+i*.1)
        outputs.append(n.step(1.+i*.1))
    assert outputs[0]['travelled']==0.
    assert outputs[-1]['travelled']==pytest.approx(sum(o['body_speed']*o['dt'] for o in outputs[1:]),abs=1e-8)
    n.reset()
    feed(n,100.)
    assert n.step(100.)['dt']==.1


def test_node_uses_selected_held_samples_even_when_newer_raw_data_arrived():
    import rclpy
    from types import SimpleNamespace
    from feb_wall_racer.nodes.motion_estimator_node import MotionEstimatorNode
    from feb_wall_racer_msgs.msg import RunContext,WallObservation
    from feb_wall_racer.interfaces import header,ros_time
    rclpy.init(args=[])
    node=MotionEstimatorNode()
    try:
        meta=header('test',1,0,1.,'map',node.config_id)
        node.on_context(RunContext(meta=meta,phase='ARMING',live=True))
        node.on_context(RunContext(meta=meta,phase='MAPPING',live=True))
        out=[]
        node.pub_motion=SimpleNamespace(publish=out.append)
        for t,speed in ((1.,1.),(1.1,5.)):
            odom=Odometry();odom.header.stamp=ros_time(t)
            odom.pose.pose.orientation.w=1.
            odom.twist.twist.linear.x=speed
            imu=Imu();imu.header.stamp=ros_time(t);imu.orientation.w=1.
            imu.angular_velocity.z=.1 if t==1. else .9
            node.sensor('odom',odom);node.sensor('imu',imu)
        meta=header('test',1,1,1.,'laser',node.config_id)
        meta.odom_stamp=ros_time(1.);meta.imu_stamp=ros_time(1.)
        if hasattr(meta,'odom_available'):
            meta.odom_available=meta.imu_available=True
        obs=WallObservation(meta=meta,valid=True)
        node.on_walls(obs);node.on_walls(obs)
        assert len(out)==1 and out[0].valid
        assert out[0].wheel_speed==1.
        assert out[0].body_speed==pytest.approx(BodySpeed(node.core.vehicle).update(1.,.1,0.,0.),abs=1e-8)
        invalid=Imu();invalid.header.stamp=ros_time(1.2)
        invalid.angular_velocity.z=float('nan')
        node.sensor('imu',invalid)
        meta=header('test',1,2,1.1,'laser',node.config_id)
        meta.odom_stamp=ros_time(1.1);meta.imu_stamp=ros_time(1.2)
        if hasattr(meta,'odom_available'):
            meta.odom_available=meta.imu_available=True
        node.on_walls(WallObservation(meta=meta,valid=True))
        assert out[-1].yaw_rate==.9,'invalid gyro must hold the latest raw valid rate, not the previous selected rate'
    finally:
        node.destroy_node();rclpy.shutdown()


def test_startup_cycle_without_selected_inputs_does_not_block_later_cycles():
    import rclpy
    from types import SimpleNamespace
    from feb_wall_racer.nodes.motion_estimator_node import MotionEstimatorNode
    from feb_wall_racer_msgs.msg import RunContext,WallObservation
    from feb_wall_racer.interfaces import header
    rclpy.init(args=[]);n=MotionEstimatorNode()
    try:
        meta=header('test',1,0,1.,'map',n.config_id)
        n.on_context(RunContext(meta=meta,phase='ARMING',live=True))
        n.on_context(RunContext(meta=meta,phase='MAPPING',live=True))
        out=[];n.pub_motion=SimpleNamespace(publish=out.append)
        n.on_walls(WallObservation(meta=header('test',1,1,1.,'laser',n.config_id),valid=True))
        assert out and not out[0].valid and not n.fault
    finally:
        n.destroy_node();rclpy.shutdown()
