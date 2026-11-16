"""Sole command owner, with direct stops and a nonblocking single MPC worker."""
import copy
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool,String
from feb_wall_racer_msgs.msg import (RunContext,WallObservation,LocalizedState,
                                    RacePlan,LocalReference)
from .base import PipelineNode,spin
from ..core.controller import RaceController
from ..core.mpc import KinematicMPC
from ..core.vehicle import VehicleModel
from ..core.safety import (braking_path,clear_crawl_steering,clip_command,
                           first_blocked_distance,mapping_throttle)
from ..core.perception import lidar_points_to_map
from ..core.synchronization import CycleJoin
from ..interfaces import seconds,ros_time,validate_plan,flat,points
from ..qos import SENSOR_QOS,QOS,LATCHED
from .. import topics


class MPCControllerNode(PipelineNode):
    def __init__(self,**kwargs):
        super().__init__('mpc_controller',**kwargs)
        self.vehicle=VehicleModel.from_parameters(self.p)
        self.mpc=None
        try:
            candidate=KinematicMPC(self.p)
            if candidate.ok:
                self.mpc=candidate
            else:
                self.get_logger().warning('CasADi unavailable; using pure pursuit')
        except Exception as exc:
            self.get_logger().warning('MPC unavailable (%s); using pure pursuit'%exc)
        self.controller=RaceController(self.p,self.mpc)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='mpc')
        self.future=self.job=self.queued=None
        self.join=CycleJoin(('reference','state'))
        self.clearance_join=CycleJoin(('walls','state'))
        self.clearance_sample=None
        self.clearance_speed_cap=self.p['mode_speed_limit']
        self.plan=None
        self.loaded_map_id=0
        self.history=[]
        self.published=deque(maxlen=64)
        self.feedback=deque(maxlen=64)
        self.diagnostic=''
        self.feedback_diagnostic=''
        self.solver_diagnostic='solver warming up'
        self.last_feedback=None
        self.last_scan=None
        self.last_reference=None
        self.active_since=None
        self.supervisor_live=None
        self.mission_live=False
        self.operator_restart=False
        self.pending_context=None
        self.context_was_live=False
        self.authorization_revision=0
        self.context_identity=None
        self.guidance_deadline=None
        self.guidance_cycle=None
        self.commanded_steer=0.
        self.reset_solver=True
        self.pub_drive=self.create_publisher(AckermannDriveStamped,self.p['drive_topic'],QOS)
        self.pub_prediction=self.create_publisher(LocalReference,topics.PREDICTION,SENSOR_QOS)
        self.create_subscription(String,self.p['state_topic'],self.on_state,QOS)
        self.create_subscription(Bool,'/feb/mission/live',self.on_mission_live,QOS)
        self.create_subscription(WallObservation,topics.WALLS,self.on_walls,SENSOR_QOS)
        self.create_subscription(LocalizedState,topics.STATE,lambda m:self.input('state',m),SENSOR_QOS)
        self.create_subscription(LocalReference,topics.REFERENCE,lambda m:self.input('reference',m),SENSOR_QOS)
        self.create_subscription(RacePlan,topics.PLAN,self.on_plan,LATCHED)
        self.create_subscription(AckermannDriveStamped,'/ackermann_cmd',self.on_feedback,QOS)
        self.create_timer(.02,self.tick,clock=self.steady)
        self.last_idle=-1e9

    def on_context(self,msg):
        new_epoch=self.ctx is not None and (msg.meta.run_id,msg.meta.generation)!=(self.ctx.meta.run_id,self.ctx.meta.generation)
        if new_epoch and self.fault and not self.operator_restart:
            self.pending_context=msg
            return
        PipelineNode.on_context(self,msg)

    def on_reset(self):
        self.authorization_revision+=1
        self.controller=RaceController(self.p,self.mpc)
        self.join=CycleJoin(('reference','state'))
        self.clearance_join=CycleJoin(('walls','state'))
        self.clearance_sample=None
        self.clearance_speed_cap=self.p['mode_speed_limit']
        self.plan=None
        self.loaded_map_id=0
        self.history=[]
        self.feedback.clear()
        self.published.clear()
        self.diagnostic=self.feedback_diagnostic=''
        self.solver_diagnostic='solver warming up'
        self.last_feedback=self.last_scan=self.last_reference=None
        self.active_since=None
        self.queued=None
        self.guidance_deadline=self.guidance_cycle=None
        self.commanded_steer=0.
        self.reset_solver=True
        self.operator_restart=False
        self.pending_context=None

    def context_changed(self):
        identity=(self.ctx.meta.run_id,self.ctx.meta.generation,self.ctx.phase,self.ctx.live,self.ctx.halted)
        previous_phase=self.context_identity[2] if self.context_identity else None
        if identity!=self.context_identity:
            self.authorization_revision+=1
            self.context_identity=identity
        if not self.ctx.live:
            self.active_since=None
        elif self.ctx.phase in ('MAPPING','FINALIZING','RACING') and (
                self.active_since is None or previous_phase not in ('MAPPING','FINALIZING','RACING')):
            self.active_since=time.monotonic()
        if self.ctx.halted:
            self.fail('manager: '+self.ctx.reason)
        elif not self.ctx.live and self.context_was_live and not self.fault:
            self._command(0.,0.)
        self.context_was_live=self.ctx.live

    def on_state(self,msg):
        authorized=msg.data.startswith('AUTONOMOUS')
        if not authorized:
            self._watchdog()
        previous=self.supervisor_live
        if authorized!=previous:
            self.authorization_revision+=1
        if msg.data.partition(':')[0].strip()=='STOP':
            self.fail('supervisor '+msg.data)
        if authorized and previous is not True and msg.data=='AUTONOMOUS: autonomous button':
            self.operator_restart=True
        self.supervisor_live=authorized
        if not authorized:
            self.mission_live=False
            self.active_since=None
            if previous is not False and not self.fault:
                self._command(0.,0.)
        if self.operator_restart and self.pending_context is not None:
            self.on_context(self.pending_context)

    def on_mission_live(self,msg):
        if not msg.data:
            self._watchdog()
        previous=self.mission_live
        self.mission_live=bool(msg.data) and self.supervisor_live is not False
        if previous!=self.mission_live:
            self.authorization_revision+=1
        if previous and not self.mission_live and not self.fault:
            self._command(0.,0.)
        if not self.mission_live:
            self.active_since=None
        elif self.active() and not previous:
            self.active_since=time.monotonic()

    def authorized(self):
        return self.active() and self.supervisor_live is True and self.mission_live

    def on_walls(self,msg):
        if not self.accept(msg.meta) or not self.active():
            return
        self.last_scan=time.monotonic()
        if not msg.valid:
            self.fail(msg.reason or 'invalid lidar observation')
        elif msg.front_clearance<self.p['obstacle_stop_range']:
            self.fail('obstacle inside stop distance')
        else:
            self.clearance_input('walls',msg)

    def clearance_input(self,kind,msg):
        try:
            for walls,state in self.clearance_join.add(kind,msg):
                if state.valid:
                    self.clearance_sample=(walls,state)
        except ValueError as exc:
            self.fail(str(exc))

    def on_feedback(self,msg):
        self.feedback.append((time.monotonic(),msg))
        self.last_feedback=time.monotonic()
        if not math.isfinite(msg.drive.speed) or not math.isfinite(msg.drive.steering_angle):
            self.feedback_diagnostic='invalid supervisor feedback'
        elif self.authorized() and self.published:
            now=self._control_time()
            matching=any(now-at<=self.p['sensor_timeout']
                and abs(speed-msg.drive.speed)<1e-6 and abs(steer-msg.drive.steering_angle)<1e-6
                for at,steer,speed in self.published)
            self.feedback_diagnostic='' if matching else 'supervisor feedback differs from recent requests'
        else:
            self.feedback_diagnostic=''

    def on_plan(self,msg):
        if not self.accept(msg.meta) or not msg.valid:
            return
        if self.ctx.meta.map_id not in (0,msg.meta.map_id):
            return
        if self.loaded_plan_id==msg.meta.plan_id and self.loaded_map_id==msg.meta.map_id:
            return
        try:
            line,speeds=validate_plan(msg)
        except ValueError as exc:
            self.fail(str(exc));return
        self.controller.reset()
        self.controller.set_raceline(line,speeds)
        self.controller.speed_cap=self.p['mode_speed_limit']
        self.plan=msg
        self.loaded_plan_id=msg.meta.plan_id
        self.loaded_map_id=msg.meta.map_id
        self.history=[]
        self.publish_health()

    def input(self,kind,msg):
        if not self.authorized() or not self.accept(msg.meta):
            return
        if kind=='state':
            self.clearance_input(kind,msg)
        try:
            for reference,state in self.join.add(kind,msg):
                self.process(reference,state)
        except ValueError as exc:
            self.fail(str(exc))

    def process(self,reference,state):
        if not self.authorized():
            return
        if reference.meta.map_id!=state.meta.map_id or reference.meta.pose_revision!=state.meta.pose_revision:
            self.fail('reference/state map or pose revision mismatch');return
        if not reference.valid or not state.valid:
            if self.active_since is not None and time.monotonic()-self.active_since<=self.p['sensor_timeout']:
                self._command(0.,0.)
            else:
                self.fail(reference.reason or state.reason or 'control state unavailable')
            return
        now=self._control_time()
        age=now-seconds(state.motion.odom_stamp)
        scan_age=now-seconds(reference.meta.header.stamp)
        if not math.isfinite(age) or age<-.05 or age>self.p['sensor_timeout'] or scan_age<-.05 or scan_age>self.p['sensor_timeout']:
            self.fail('state or reference timestamp stale or in future');return
        if not np.all(np.isfinite([*state.pose,state.yaw,state.motion.body_speed,state.motion.steering,
                                   reference.steering,reference.target_speed,reference.guidance_age,*reference.target])):
            self.fail('non-finite state or reference');return
        if reference.target_speed<0 or reference.guidance_age<0:
            self.fail('invalid reference speed or guidance age');return
        mapping=reference.mode in ('MAPPING','DEGRADED_MAPPING') and self.ctx.phase in ('MAPPING','FINALIZING')
        racing=(reference.mode=='RACING' and self.ctx.phase=='RACING' and self.plan is not None
                and reference.meta.plan_id==self.loaded_plan_id and state.meta.plan_id==self.loaded_plan_id
                and state.meta.map_id==self.loaded_map_id and self.ctx.meta.plan_id==self.loaded_plan_id)
        if not mapping and not racing:
            return
        self.last_reference=time.monotonic()
        if reference.emergency:
            self._command(reference.steering,0.);return
        if reference.mode in ('MAPPING','DEGRADED_MAPPING'):
            if self.ctx.phase not in ('MAPPING','FINALIZING'):
                return
            if reference.degraded:
                remaining=self.p['mapping_fallback_duration']-reference.guidance_age
                deadline=time.monotonic()+remaining
                if self.guidance_cycle!=reference.guidance_cycle:
                    self.guidance_deadline=deadline
                    self.guidance_cycle=reference.guidance_cycle
                else:
                    self.guidance_deadline=min(self.guidance_deadline,deadline)
                if time.monotonic()>self.guidance_deadline:
                    self._command(0.,0.);return
            else:
                self.guidance_cycle=reference.guidance_cycle
                self.guidance_deadline=time.monotonic()+self.p['mapping_fallback_duration']
            throttle=self._throttle(reference.target_speed,state.motion.body_speed)
            if reference.reason in ('paired corridor','verified gap'):
                self.checked_drive(reference.steering,throttle)
            else:
                self._command(reference.steering,throttle)
        elif reference.mode=='RACING':
            if (self.ctx.phase!='RACING' or not self.plan or reference.meta.plan_id!=self.loaded_plan_id
                    or state.meta.plan_id!=self.loaded_plan_id or state.meta.map_id!=self.loaded_map_id
                    or self.ctx.meta.plan_id!=self.loaded_plan_id):
                return
            lost=seconds(state.meta.header.stamp)-state.last_localization_stamp
            if lost>self.p['stop_after']:
                self.fail('wall-map localization lost');return
            reference=copy.deepcopy(reference)
            reference.target_speed=min(reference.target_speed,self.p['mode_speed_limit'],
                self.p['blind_speed'] if lost>self.p['lost_after'] else self.p['mode_speed_limit'])
            self.queued=(reference,state)
            self.poll_solver()
        else:
            self.fail('unknown local reference mode '+reference.mode)

    def _throttle(self,target_speed,speed):
        return mapping_throttle(target_speed,speed,self.p,self.vehicle)

    def checked_drive(self,steer,throttle,prediction=None,prediction_time=0.):
        """Use the latest matched lidar and pose before committing a command."""
        steer,throttle=clip_command(steer,throttle,self.p,self.authorized())
        if self.clearance_sample is None:
            self._command(steer,0.)
            return 'clearance awaiting matched scan'
        walls,state=self.clearance_sample
        scan_stamp=seconds(walls.meta.header.stamp)
        age=self._control_time()-scan_stamp
        same_map=self.ctx.phase!='RACING' or state.meta.map_id==self.loaded_map_id
        if not same_map or not -.05<=age<=self.p['sensor_timeout'] or not len(walls.points):
            self._command(steer,0.)
            return 'clearance awaiting fresh scan'
        pose=(*state.pose,state.yaw)
        speed=max(0.,float(state.motion.body_speed))
        measured=state.motion.steering
        if not np.all(np.isfinite([*pose,state.motion.body_speed,measured])):
            self.fail('invalid state for lidar clearance')
            return 'clearance state invalid'
        obstacle=lidar_points_to_map(points(walls.points),state.pose,state.yaw,self.p['lidar_forward'])
        brake=self.p['a_brake']
        delay=self.vehicle.act_delay+max(age,0.)
        reaction=max(delay+self.p['mpc_dt'],self.p['map_path_reaction_distance']/max(speed,.6))
        margin=self.p['car_half_width']+self.p['wall_clearance']
        # Steering in flight and a pure-pursuit result both need a braking
        # check even when no fresh MPC trajectory accompanies the command.
        path=braking_path(pose,speed,measured,steer,delay,reaction,brake,self.vehicle)
        distance=first_blocked_distance(path,obstacle,self.vehicle.wheelbase,margin)
        if prediction is not None:
            stages=prediction_time+self.vehicle.act_delay+np.arange(prediction.shape[1])*self.p['mpc_dt']
            upcoming=prediction[:3,stages>=scan_stamp].T
            if len(upcoming):
                planned=np.vstack((pose,upcoming))
                distance=min(distance,first_blocked_distance(planned,obstacle,self.vehicle.wheelbase,margin))
        reserve=max(self.p['map_path_reaction_distance'],speed*(delay+self.p['mpc_dt']))
        safe_speed=math.sqrt(2*brake*max(0.,distance-reserve))
        self.clearance_speed_cap=min(self.clearance_speed_cap,safe_speed)
        crawl_speed=min(self.p['mapping_fallback_speed'],self.p['mode_speed_limit'])
        if self.clearance_speed_cap<crawl_speed and speed<=crawl_speed:
            # Keep crawl speed through the first movement after aligning,
            # even if the new pose and steering clear the nominal check.
            crawl_reaction=delay+self.p['mpc_dt']+reserve/crawl_speed
            crawl_steer=clear_crawl_steering(pose,obstacle,crawl_speed,measured,steer,
                delay,crawl_reaction,brake,self.vehicle,self.p['car_half_width'],margin)
            if crawl_steer is not None:
                self.clearance_speed_cap=crawl_speed
                self._command(crawl_steer,crawl_speed/self.vehicle.speed_per_throttle)
                return 'lidar clearance recovery'
            if speed<=.05:
                aligned_steer=clear_crawl_steering(pose,obstacle,crawl_speed,None,steer,
                    delay,crawl_reaction,brake,self.vehicle,self.p['car_half_width'],margin)
                if aligned_steer is not None:
                    self._command(aligned_steer,0.)
                    return 'lidar clearance aligning'
            safe_speed=0.
        if math.isfinite(safe_speed):
            if safe_speed<=.05:
                limited=0.
            elif speed>safe_speed:
                # The speed cap assumes this deceleration. A gradual speed
                # controller could otherwise spend the remaining braking room.
                limited=self.vehicle.command_for_force(speed,self.vehicle.drag*speed-brake)
            else:
                limited=self._throttle(safe_speed,speed)
            if limited<throttle:
                self._command(steer,limited)
                return 'lidar clearance braking'
        else:
            # Once the sweep is clear, restore speed at the normal rate rather
            # than alternating between a high-speed solve and hard braking.
            self.clearance_speed_cap=min(self.p['mode_speed_limit'],self.clearance_speed_cap
                                        + self.p['a_acc']*self.p['mpc_dt'])
        self._command(steer,throttle)
        return ''

    def _command(self,steer,throttle):
        try:
            steer,throttle=clip_command(steer,throttle,self.p,self.authorized())
        except ValueError as exc:
            self.fail(str(exc));return
        now=self._control_time()
        msg=AckermannDriveStamped()
        msg.header.stamp=ros_time(now)
        msg.header.frame_id='base_link'
        msg.drive.speed=throttle*self.vehicle.speed_per_throttle
        msg.drive.steering_angle=steer
        self.pub_drive.publish(msg)
        self.published.append((now,steer,msg.drive.speed))
        self.commanded_steer=steer
        if self.ctx and self.ctx.phase=='RACING':
            self.history.append((now,steer,throttle))
            self.history=[h for h in self.history if now-h[0]<self.vehicle.act_delay+.5]
            if self.future is None:
                self.controller.sent=list(self.history)

    def fail(self,reason):
        if self.fault:
            return
        self.authorization_revision+=1
        PipelineNode.fail(self,reason)
        if hasattr(self,'pub_drive'):
            self._command(0.,0.)

    def _control_time(self):
        return self.get_clock().now().nanoseconds*1e-9

    def _watchdog(self):
        if (self.fault or not self.ctx or not self.ctx.live
                or self.supervisor_live is not True or not self.mission_live):
            return
        now=time.monotonic()
        if self.context_received is None or now-self.context_received>self.p['sensor_timeout']:
            self.fail('run manager unavailable or stale');return
        if self.ctx.phase not in ('MAPPING','FINALIZING','RACING'):
            return
        if self.last_scan is not None and now-self.last_scan>.5:
            self.fail('lidar stream stale');return
        if self.active_since is not None and now-self.active_since>self.p['sensor_timeout']:
            if self.last_scan is None or self.last_reference is None or now-self.last_reference>self.p['sensor_timeout']:
                self.fail('control inputs unavailable or stale');return
        if self.future is not None and now-self.job['started']>self.settings['worker_timeout']:
            self.fail('MPC worker deadline exceeded')

    @staticmethod
    def solve(controller,state,reference,now,age):
        controller.speed_scale=1.
        controller.speed_cap=reference.target_speed
        return controller.step(now,float(state.pose[0]),float(state.pose[1]),state.yaw,
            max(state.motion.body_speed,0.),state.motion.steering,state_age=max(age,0.),record_command=False)

    def poll_solver(self):
        if self.future is not None and self.future.done():
            future,job=self.future,self.job
            self.future=self.job=None
            ref,state=job['inputs']
            if (self.authorized() and self.accept(ref.meta) and self.ctx.phase=='RACING'
                    and ref.meta.plan_id==self.loaded_plan_id
                    and job['authorization_revision']==self.authorization_revision
                    and -.05<=self._control_time()-seconds(state.motion.odom_stamp)<=self.p['sensor_timeout']):
                try:
                    steer,throttle=future.result()
                    info=job['controller'].info
                    prediction=info.get('Z')
                    guard=self.checked_drive(float(steer),float(throttle),prediction if info.get('ok') else None,
                                             job['time'])
                    self.solver_diagnostic='solver %.1f ms (%s)'%(
                        job['controller'].solve_ms,info.get('status','warming up'))
                    if guard:
                        self.solver_diagnostic+=' | '+guard
                    if prediction is not None:
                        meta=copy.deepcopy(state.meta);meta.header.stamp=ros_time(job['time'])
                        self.pub_prediction.publish(LocalReference(meta=meta,mode='RACING',valid=True,
                            path=flat(np.asarray(prediction[:2,1:]).T)))
                except Exception as exc:
                    self.fail('controller failed: '+str(exc))
        if self.future is None and self.queued is not None and self.authorized():
            reference,state=self.queued
            self.queued=None
            reference.target_speed=min(reference.target_speed,self.clearance_speed_cap)
            now=self._control_time()
            age=now-seconds(state.motion.odom_stamp)
            if age<-.05 or age>self.p['sensor_timeout']:
                self.fail('racing state timestamp stale or in future');return
            if self.reset_solver:
                if self.mpc is not None:
                    self.mpc.warm=None
                self.reset_solver=False
            self.controller.sent=list(self.history)
            self.job=dict(inputs=(reference,state),started=time.monotonic(),time=now,controller=self.controller,
                          authorization_revision=self.authorization_revision)
            self.future=self.pool.submit(self.solve,self.controller,state,reference,now,age)

    def tick(self):
        self._watchdog()
        self.poll_solver()
        if self.authorized() and self.active_since is not None and time.monotonic()-self.active_since>self.p['sensor_timeout']:
            if self.last_feedback is None or time.monotonic()-self.last_feedback>self.p['sensor_timeout']:
                self.feedback_diagnostic='supervisor feedback unavailable'
        self.diagnostic='; '.join(v for v in (self.solver_diagnostic,self.feedback_diagnostic) if v)
        if not self.fault and not self.authorized() and time.monotonic()-self.last_idle>=.1:
            self.last_idle=time.monotonic()
            self._command(0.,0.)

    def destroy_node(self):
        if hasattr(self,'pool'):
            self.pool.shutdown(wait=True,cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    spin(MPCControllerNode,args)
