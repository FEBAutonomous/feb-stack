import importlib
from std_msgs.msg import Bool, String
import pytest
import rclpy
import time


@pytest.fixture
def manager():
    m=importlib.import_module('feb_wall_racer.nodes.run_manager_node')
    rclpy.init(args=[])
    node=m.RunManagerNode()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_both_permissions_reset_once_and_do_not_require_a_racing_plan(manager):
    n=manager
    n.on_mission_live(Bool(data=True))
    assert not n.live
    n.on_state(String(data='AUTONOMOUS: test'))
    assert n.live and n.generation==1 and n.phase=='ARMING'
    n.on_mission_live(Bool(data=True))
    n.on_state(String(data='AUTONOMOUS: test'))
    assert n.generation==1
    for name in n.critical_nodes:
        status=n.status_message(name,ready=True)
        n.on_health(status)
    n.tick()
    assert n.phase=='MAPPING'


def test_stop_cannot_be_cleared_by_automatic_rearm_or_mission_heartbeat(manager):
    n=manager
    n.on_state(String(data='AUTONOMOUS: test'))
    n.on_mission_live(Bool(data=True))
    n.on_state(String(data='STOP: released'))
    n.on_state(String(data='AUTONOMOUS: ready'))
    n.on_mission_live(Bool(data=True))
    assert n.halted and n.generation==1
    n.on_state(String(data='IDLE: test'))
    n.on_state(String(data='AUTONOMOUS: autonomous button'))
    n.on_mission_live(Bool(data=True))
    assert not n.halted and n.generation==2


def test_reset_barrier_rejects_old_generation_health(manager):
    n=manager
    n.on_state(String(data='AUTONOMOUS: test'))
    n.on_mission_live(Bool(data=True))
    for name in n.critical_nodes:
        msg=n.status_message(name,ready=True)
        msg.meta.generation=0
        n.on_health(msg)
    n.tick()
    assert n.phase=='ARMING'


def test_stale_scan_fault_is_latched_before_supervisor_idle(manager):
    n=manager
    n.on_state(String(data='AUTONOMOUS: test'))
    n.on_mission_live(Bool(data=True))
    n.phase='MAPPING'
    n.have_scan=True
    n.last_scan_wall_time=time.monotonic()-1.
    for name in n.critical_nodes:
        n.on_health(n.status_message(name,ready=True))
    n.on_state(String(data='IDLE: drive timeout'))
    assert n.halted and 'lidar' in n.stop_reason


def test_plan_arriving_before_its_map_is_reconciled_and_waits_for_controller(manager):
    from feb_wall_racer_msgs.msg import RacePlan,TrackMap
    n=manager
    n.on_state(String(data='AUTONOMOUS: test'))
    n.on_mission_live(Bool(data=True))
    n.phase='MAPPING'
    meta=n.status_message('global_planner').meta
    meta.map_id=1;meta.plan_id=1
    n.on_plan(RacePlan(meta=meta,valid=True))
    n.on_map(TrackMap(meta=meta,valid=True,finalized=True))
    assert n.phase=='FINALIZING' and n.plan_id==1
    for name in n.critical_nodes:
        msg=n.status_message(name,ready=True)
        msg.meta.plan_id=1 if name=='mpc_controller' else 0
        msg.meta.map_id=1 if name=='mpc_controller' else 0
        n.on_health(msg)
    n.tick()
    assert n.phase=='FINALIZING'
    local=n.status_message('local_planner',ready=True)
    local.meta.plan_id=1;local.meta.map_id=1
    n.on_health(local)
    n.tick()
    assert n.phase=='RACING'


def test_aggregate_status_preserves_mapping_progress_and_localization_details(manager):
    from types import SimpleNamespace
    from feb_wall_racer_msgs.msg import MotionEstimate,LocalizedState,WallObservation,TrackMap
    n=manager
    meta=n.status_message('slam').meta
    n.on_localized(LocalizedState(meta=meta,valid=True,travelled=8.3,matches=12,residual=.02,
        localized=True,motion=MotionEstimate(body_speed=1.2)))
    n.on_walls(WallObservation(meta=meta,valid=True,segments=3,centre=[1.,0.,2.,0.],confidence=.8))
    n.on_map(TrackMap(meta=meta,valid=True,trail=[0.,0.,1.,0.,2.,0.],walls=[0.,1.]*8))
    out=[];n.pub_status=SimpleNamespace(publish=out.append)
    n.phase='MAPPING';n.last_status=-1e9;n.emit()
    assert '1.2 m/s' in out[-1].data and '12 matches/0.02 m' in out[-1].data
    assert 'map 8.3 m, 3 path samples, 8 wall points' in out[-1].data


def test_duplicate_fault_health_does_not_feed_back_repeated_contexts(manager):
    from types import SimpleNamespace
    n=manager;n.live=True;n.phase='MAPPING'
    out=[];n.pub_context=SimpleNamespace(publish=out.append)
    fault=n.status_message('mpc_controller',ready=True,reason='test fault');fault.fault=True
    n.on_health(fault);n.on_health(fault)
    assert n.halted and len(out)==1
