import importlib
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import rclpy
from std_msgs.msg import Bool,String
from feb_wall_racer_msgs.msg import RunContext,WallObservation
from feb_wall_racer.interfaces import header
from feb_wall_racer.configuration import effective_parameters
from feb_wall_racer.params import DEFAULTS


@pytest.fixture
def controller_node():
    m=importlib.import_module('feb_wall_racer.nodes.mpc_controller_node')
    rclpy.init(args=[])
    with patch.object(m,'KinematicMPC',return_value=SimpleNamespace(ok=False)):
        n=m.MPCControllerNode()
    output=[]
    n.pub_drive=SimpleNamespace(publish=output.append)
    yield n,output
    n.destroy_node();rclpy.shutdown()


def authorize(n):
    meta=header('test',1,0,n._control_time(),'map',n.config_id)
    n.on_context(RunContext(meta=meta,phase='ARMING',live=True))
    n.on_state(String(data='AUTONOMOUS: test'))
    n.on_mission_live(Bool(data=True))
    n.on_context(RunContext(meta=meta,phase='MAPPING',live=True))


def test_mapping_overspeed_keeps_bounded_positive_wheel_demand(controller_node):
    n,out=controller_node
    authorize(n)
    throttle=n._throttle(1.8,2.2)
    n._command(.1,throttle)
    assert 0.<out[-1].drive.speed<2.2


def test_inactive_clipping_nonfinite_and_local_fault_latch(controller_node):
    n,out=controller_node
    n._command(.5,.3)
    assert out[-1].drive.speed==0.
    authorize(n)
    n._command(1.,1.)
    assert out[-1].drive.speed==n.p['mode_speed_limit']
    assert out[-1].drive.steering_angle==n.p['max_steer']
    n._command(float('nan'),.1)
    assert n.fault and out[-1].drive.speed==0.
    n.on_context(n.ctx)
    assert n.fault
    n._command(.1,.1)
    assert out[-1].drive.speed==0.


def test_direct_stop_does_not_wait_for_manager_and_suppresses_idle_readiness(controller_node):
    n,out=controller_node
    authorize(n)
    n._command(.1,.1)
    n.on_state(String(data='STOP: released'))
    assert n.fault and out[-1].drive.speed==0.
    count=len(out)
    n.tick()
    assert len(out)==count


def test_direct_front_obstacle_stops_without_waiting_for_local_planner(controller_node):
    n,out=controller_node
    authorize(n)
    meta=header('test',1,1,n._control_time(),'laser',n.config_id)
    n.on_walls(WallObservation(meta=meta,valid=True,front_clearance=.1))
    assert n.fault and out[-1].drive.speed==0.


def race_inputs(n, wall_points=(4., 2., 4., -2.)):
    import copy,time
    from feb_wall_racer.core.planning import plan_track
    from feb_wall_racer_msgs.msg import RacePlan,MotionEstimate,LocalizedState,LocalReference
    from feb_wall_racer.interfaces import flat,ros_time
    authorize(n)
    theta=np.linspace(0,2*np.pi,100,endpoint=False)
    data=plan_track(np.c_[8*np.sin(theta),8*(1-np.cos(theta))],np.full(100,2.2),n.p)
    meta=header('test',1,1,n._control_time(),'map',n.config_id)
    meta.map_id=1;meta.plan_id=1;meta.pose_revision=1
    track=data['track']
    plan=RacePlan(meta=meta,valid=True,line=flat(data['line']),speeds=data['speeds'].tolist(),
        left=flat(track['left']),right=flat(track['right']),centre=flat(track['centre']),
        half_width=track['half_width'].tolist(),length=float(track['length']),arc_length=data['arc_length'].tolist())
    n.on_plan(plan)
    n.on_context(RunContext(meta=copy.deepcopy(meta),phase='RACING',live=True))
    stamp=n._control_time()
    meta.header.stamp=ros_time(stamp)
    motion=MotionEstimate(meta=meta,odom_stamp=ros_time(stamp),body_speed=2.,valid=True)
    state=LocalizedState(meta=meta,motion=motion,pose=[0.,0.],yaw=0.,valid=True,
                         finalized=True,last_localization_stamp=stamp)
    reference=LocalReference(meta=copy.deepcopy(meta),mode='RACING',target_speed=5.,valid=True)
    n.on_walls(WallObservation(meta=meta,valid=True,front_clearance=.45,points=wall_points))
    n.clearance_input('state',state)
    return reference,state


def test_racing_prediction_cannot_publish_drive_through_a_live_wall(controller_node):
    n, out = controller_node
    ref, state = race_inputs(n, wall_points=(.43, .12))
    # The old forward cone does not stop for this range. The next swept
    # footprint, including the front axle, reaches the return during braking.
    def solve(*args, **kwargs):
        z = np.zeros((5, 9))
        z[0] = np.linspace(0., 1.6, 9)
        z[3] = 2.
        n.controller.info = dict(Z=z, status='Solve_Succeeded', ok=True)
        return 0., .1

    n.controller.step = solve
    n.process(ref, state)
    n.future.result(timeout=.3)
    n.poll_solver()
    assert out[-1].drive.speed == 0.
    assert 'clearance' in n.solver_diagnostic


def test_racing_fallback_checks_the_delayed_braking_sweep(controller_node):
    n, out = controller_node
    race_inputs(n, wall_points=(.43, .12))
    reason = n.checked_drive(0., .1)
    assert out[-1].drive.speed == 0.
    assert reason == 'lidar clearance braking'


def test_racing_guard_uses_full_deceleration_inside_the_stopping_envelope(controller_node):
    n, out = controller_node
    race_inputs(n, wall_points=(.96, .12))
    assert n.checked_drive(0., .1) == 'lidar clearance braking'
    expected=n.vehicle.command_for_force(2.,n.vehicle.drag*2.-n.p['a_brake'])
    assert out[-1].drive.speed == pytest.approx(expected*n.vehicle.speed_per_throttle)
    assert 0. < n.clearance_speed_cap < 2.


def test_racing_clearance_recovers_at_low_speed_with_a_clear_alternate_arc(controller_node):
    n, out = controller_node
    _, state = race_inputs(n, wall_points=(.026, .349))
    state.motion.body_speed = 0.
    state.motion.steering = .087
    assert n.checked_drive(.2, .1) == 'lidar clearance recovery'
    assert out[-1].drive.steering_angle < .2
    assert 0. < out[-1].drive.speed <= n.p['mapping_fallback_speed']


def test_racing_clearance_can_leave_the_buffer_without_moving_closer(controller_node):
    n, out = controller_node
    _, state = race_inputs(n, wall_points=(.16, -.266))
    state.motion.body_speed = 0.
    state.motion.steering = .48
    assert n.checked_drive(.48, .1) == 'lidar clearance recovery'
    assert 0. < out[-1].drive.speed <= n.p['mapping_fallback_speed']


def test_racing_clearance_does_not_crawl_when_already_inside_body_clearance(controller_node):
    n, out = controller_node
    _, state = race_inputs(n, wall_points=(.026, .15))
    state.motion.body_speed = 0.
    assert n.checked_drive(.2, .1) == 'lidar clearance braking'
    assert out[-1].drive.speed == 0.


@pytest.mark.parametrize('phase', ['MAPPING', 'RACING'])
def test_clearance_aligns_steering_at_rest_before_crawling(controller_node, phase):
    n, out = controller_node
    _, state = race_inputs(n, wall_points=(-.1902, .3407))
    n.ctx.phase = phase
    state.motion.body_speed = 0.
    state.motion.steering = .34
    assert n.checked_drive(.34, .1) == 'lidar clearance aligning'
    assert out[-1].drive.speed == 0.
    state.motion.steering = out[-1].drive.steering_angle
    assert n.checked_drive(.34, .1) == 'lidar clearance recovery'
    assert out[-1].drive.speed > 0.


def test_pause_and_resume_discards_an_in_flight_result(controller_node):
    from threading import Event
    n,out=controller_node
    ref,state=race_inputs(n)
    entered,release=Event(),Event()
    def slow(*args,**kwargs):
        entered.set();release.wait(2.);return .1,.1
    n.controller.step=slow
    try:
        n.process(ref,state)
        assert entered.wait(.3)
        n.on_mission_live(Bool(data=False))
        n.on_mission_live(Bool(data=True))
        release.set();n.future.result(timeout=1.)
        n.poll_solver()
        assert out and all(msg.drive.speed==0. for msg in out)
    finally:
        release.set()


def test_stale_state_is_rejected_before_scheduling_a_solve(controller_node):
    from feb_wall_racer.interfaces import ros_time
    n,out=controller_node
    ref,state=race_inputs(n)
    state.motion.odom_stamp=ros_time(n._control_time()-1.)
    n.process(ref,state)
    assert n.fault and n.future is None and out[-1].drive.speed==0.


def test_delayed_reset_barrier_does_not_consume_sensor_startup_grace(controller_node):
    import time
    n,_=controller_node
    authorize(n)
    n.on_context(RunContext(meta=n.ctx.meta,phase='ARMING',live=True))
    n.active_since=time.monotonic()-2.
    n.on_context(RunContext(meta=n.ctx.meta,phase='MAPPING',live=True))
    n.tick()
    assert not n.fault


def test_history_records_the_clipped_published_command_once(controller_node):
    n,out=controller_node
    race_inputs(n)
    n._control_time=lambda:10.04
    n._command(1.,1.)
    assert n.history==[(10.04,n.p['max_steer'],n.p['mode_speed_limit']/n.vehicle.speed_per_throttle)]
    assert n.controller.sent==n.history
    assert out[-1].drive.speed==n.p['mode_speed_limit']


def test_manager_death_stops_and_never_repeats_idle_readiness(controller_node):
    import time
    n,out=controller_node
    authorize(n)
    n.context_received=time.monotonic()-1.
    n.tick()
    assert n.fault and out[-1].drive.speed==0.
    count=len(out);n.tick()
    assert len(out)==count


def test_reset_during_solve_discards_old_result(controller_node):
    import copy
    from threading import Event
    n,out=controller_node
    ref,state=race_inputs(n)
    entered,release=Event(),Event()
    def slow(*args,**kwargs):
        entered.set();release.wait(2.);return .1,.1
    n.controller.step=slow
    try:
        n.process(ref,state);assert entered.wait(.3)
        meta=copy.deepcopy(n.ctx.meta);meta.generation=2
        n.on_context(RunContext(meta=meta,phase='ARMING',live=True))
        release.set();n.future.result(timeout=1.)
        n.poll_solver()
        assert not any(msg.drive.speed>0. for msg in out)
        assert n.loaded_plan_id==0 and not n.history
    finally:
        release.set()


def test_feedback_is_bounded_and_never_replaces_published_request_history(controller_node):
    from ackermann_msgs.msg import AckermannDriveStamped
    n,_=controller_node
    authorize(n);n._command(.1,.1)
    for _ in range(100):
        msg=AckermannDriveStamped();msg.drive.steering_angle=-.3;msg.drive.speed=3.
        n.on_feedback(msg)
    assert len(n.feedback)==64 and n.commanded_steer==.1 and not n.history
    assert n.feedback_diagnostic and not n.fault


def test_initial_missing_sensor_state_uses_startup_grace_not_a_zero_stamp_fault(controller_node):
    from feb_wall_racer_msgs.msg import LocalizedState,LocalReference
    n,out=controller_node
    authorize(n)
    meta=header('test',1,1,n._control_time(),'map',n.config_id)
    state=LocalizedState(meta=meta,valid=False,reason='selected-cycle sensors unavailable')
    ref=LocalReference(meta=meta,mode='STOP',valid=False,emergency=True,reason=state.reason)
    n.process(ref,state)
    assert not n.fault and out[-1].drive.speed==0.


def test_mapping_path_emergency_keeps_its_selected_steering(controller_node):
    from feb_wall_racer_msgs.msg import MotionEstimate,LocalizedState,LocalReference
    from feb_wall_racer.interfaces import ros_time
    n,out=controller_node;authorize(n)
    stamp=n._control_time();meta=header('test',1,1,stamp,'map',n.config_id)
    state=LocalizedState(meta=meta,valid=True,motion=MotionEstimate(odom_stamp=ros_time(stamp),body_speed=2.))
    ref=LocalReference(meta=meta,mode='MAPPING',valid=True,steering=.2,emergency=True)
    n.process(ref,state)
    assert out[-1].drive.speed==0. and out[-1].drive.steering_angle==.2 and not n.fault
