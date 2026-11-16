import importlib
from types import SimpleNamespace
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import LaserScan,Imu
from nav_msgs.msg import Odometry
from feb_wall_racer_msgs.msg import RunContext
from feb_wall_racer.interfaces import header,decode_observation
from feb_wall_racer.benchmarks.wall import synthetic_corridor_scan
from feb_wall_racer.core.perception import observe


@pytest.fixture
def perception_node():
    cls=importlib.import_module('feb_wall_racer.nodes.perception_node').PerceptionNode
    rclpy.init(args=[])
    n=cls()
    meta=header('test',1,0,1.,'map',n.config_id)
    n.on_context(RunContext(meta=meta,phase='ARMING',live=True))
    n.on_context(RunContext(meta=meta,phase='MAPPING',live=True))
    output=[]
    n.pub_walls=SimpleNamespace(publish=output.append)
    yield n,output
    n.destroy_node(); rclpy.shutdown()


def scan(t=1.):
    raw=synthetic_corridor_scan()
    msg=LaserScan()
    msg.header.stamp.sec=int(t)
    msg.header.stamp.nanosec=round((t-int(t))*1e9)
    msg.header.frame_id='laser'
    for k in ('angle_min','angle_increment'):
        setattr(msg,k,float(getattr(raw,k)))
    msg.range_min=.06
    msg.range_max=12.
    msg.ranges=np.asarray(raw.ranges,dtype=np.float32).tolist()
    return msg


def test_scan_output_matches_unchanged_kernel_without_feedback(perception_node):
    n,out=perception_node
    msg=scan()
    expected=observe(msg,n.p)
    n.on_scan(msg)
    assert len(out)==1 and out[0].valid
    actual=decode_observation(out[0])
    np.testing.assert_allclose(actual.centre,expected.centre,atol=1e-8,rtol=0)
    assert actual.heading==expected.heading


def test_selected_cycle_carries_held_sensor_stamps_and_racing_fast_path(perception_node):
    n,out=perception_node
    odom=Odometry(); odom.header.stamp.sec=7
    imu=Imu(); imu.header.stamp.sec=8
    n.on_odom(odom); n.on_imu(imu)
    n.ctx.phase='RACING'
    n.on_scan(scan())
    assert out[0].meta.odom_stamp.sec==7 and out[0].meta.imu_stamp.sec==8
    assert not out[0].centre
    assert len(out[0].left)>0


def test_backwards_scan_is_explicitly_invalid(perception_node):
    n,out=perception_node
    n.on_scan(scan())
    n.on_scan(scan(.9))
    assert n.fault and not out[-1].valid
    assert out[-1].meta.cycle==2


def test_exact_source_stamp_survives_double_precision_epoch_conversion(perception_node):
    n,out=perception_node
    msg=scan()
    msg.header.stamp.sec=1793853000;msg.header.stamp.nanosec=123456789
    n.on_scan(msg)
    assert out[0].meta.header.stamp==msg.header.stamp
