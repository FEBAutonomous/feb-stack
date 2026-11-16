"""Supervisor/mission lifecycle and versioned reset/plan coordination."""
import time
import uuid
from std_msgs.msg import Bool, String
from sensor_msgs.msg import LaserScan
from feb_wall_racer_msgs.msg import (RunContext,NodeStatus,TrackMap,RacePlan,
                                    LocalizedState,WallObservation)
from .base import PipelineNode, spin
from ..core.lifecycle import Lifecycle
from ..interfaces import header
from ..configuration import NODE_NAMES
from ..qos import QOS, LATCHED
from .. import topics


class RunManagerNode(Lifecycle,PipelineNode):
    critical_nodes=tuple(n for n in NODE_NAMES if n not in ('run_manager','visualization'))

    def __init__(self, **kwargs):
        super().__init__('run_manager',manage_context=False,**kwargs)
        self.run_id=uuid.uuid4().hex
        self.generation=0
        self.phase='IDLE'
        self.resume_phase='MAPPING'
        self.mode='IDLE'
        self.live=False
        self.live_since=None
        self.halted=False
        self.stop_reason=''
        self.supervisor_live=None
        self.mission_live=False
        self.autonomous_start_pending=False
        self.health={}
        self.reset_started=None
        self.map_id=0
        self.plan_id=0
        self.early_plans={}
        self.finalizing_started=None
        self.last_status=-1e9
        self.latest_state=self.latest_walls=None
        self.map_samples=self.map_points=0
        self.plan_length=0.
        self.have_scan=False
        self.last_scan_wall_time=None
        self.ready=True
        self.pub_context=self.create_publisher(RunContext,topics.CONTEXT,LATCHED)
        self.pub_status=self.create_publisher(String,'/feb/status',QOS)
        self.create_subscription(String,self.p['state_topic'],self.on_state,QOS)
        self.create_subscription(Bool,'/feb/mission/live',self.on_mission_live,QOS)
        self.create_subscription(LaserScan,self.p['scan_topic'],self.on_scan,topics.SENSOR_QOS)
        for name in self.critical_nodes:
            self.create_subscription(NodeStatus,topics.health(name),self.on_health,QOS)
        self.create_subscription(TrackMap,topics.MAP,self.on_map,LATCHED)
        self.create_subscription(RacePlan,topics.PLAN,self.on_plan,LATCHED)
        self.create_subscription(LocalizedState,topics.STATE,self.on_localized,topics.SENSOR_QOS)
        self.create_subscription(WallObservation,topics.WALLS,self.on_walls,topics.SENSOR_QOS)
        self.create_timer(.1,self.tick,clock=self.steady)
        self.emit()

    def _reset_run(self):
        self.generation+=1
        self.halted=False
        self.stop_reason=''
        self.phase='ARMING'
        self.health={}
        self.map_id=self.plan_id=0
        self.early_plans={}
        self.finalizing_started=None
        self.reset_started=time.monotonic()
        self.latest_state=self.latest_walls=None
        self.map_samples=self.map_points=0
        self.plan_length=0.
        self.last_status=-1e9
        self.have_scan=False
        self.last_scan_wall_time=None

    def on_scan(self,msg):
        if self.live:
            self.have_scan=True
            self.last_scan_wall_time=time.monotonic()

    def _update_live(self,reason):
        previous=self.live
        Lifecycle._update_live(self,reason)
        if self.live and not previous and self.phase=='PAUSED':
            self.phase=self.resume_phase
        self.emit()

    def _command(self,steer,throttle):
        # This owner publishes authorization only; MPC alone publishes /drive.
        if self.halted:
            self.phase='STOPPED'
        elif not self.live and self.phase not in ('IDLE','PAUSED'):
            self.resume_phase=self.phase
            self.phase='PAUSED'
        self.emit()

    def status_message(self,name,ready=False,reason=''):
        return NodeStatus(meta=header(self.run_id,self.generation,0,0.,'map',self.config_id),
                          node=name,ready=ready,critical=True,reason=reason)

    def on_health(self,msg):
        if not self.accept(msg.meta) or msg.node not in self.critical_nodes:
            return
        self.health[msg.node]=(time.monotonic(),msg)
        if self.live and not self.halted and msg.fault:
            self._controlled_stop(msg.node+': '+msg.reason)

    def on_map(self,msg):
        if not self.accept(msg.meta) or self.halted:
            return
        if msg.valid:
            self.map_samples=len(msg.trail)//2
            self.map_points=len(msg.walls)//2
        if not msg.finalized:return
        if not msg.valid:
            self._controlled_stop('invalid finalized map: '+msg.reason)
        elif self.phase in ('MAPPING','FINALIZING') and msg.meta.map_id>=self.map_id:
            if msg.meta.map_id!=self.map_id:
                self.finalizing_started=time.monotonic()
            self.map_id=msg.meta.map_id
            self.phase='FINALIZING'
            early=self.early_plans.pop(self.map_id,None)
            if early is not None:
                self.on_plan(early)
        self.emit()

    def on_plan(self,msg):
        if not self.accept(msg.meta) or not msg.meta.map_id:
            return
        if not self.map_id:
            if len(self.early_plans)<8:
                self.early_plans[msg.meta.map_id]=msg
            return
        if msg.meta.map_id!=self.map_id:
            return
        if not msg.valid:
            self._controlled_stop('global planning failed: '+msg.reason)
        else:
            self.plan_id=msg.meta.plan_id
            self.plan_length=msg.length
        self.emit()

    def on_localized(self,msg):
        if self.accept(msg.meta) and msg.valid:
            if self.latest_state is None or msg.meta.cycle>=self.latest_state.meta.cycle:
                self.latest_state=msg

    def on_walls(self,msg):
        if self.accept(msg.meta):
            if self.latest_walls is None or msg.meta.cycle>=self.latest_walls.meta.cycle:
                self.latest_walls=msg

    def status_text(self):
        state,fit=self.latest_state,self.latest_walls
        speed=state.motion.body_speed if state else 0.
        matches=state.matches if state else 0
        residual=state.residual if state else float('inf')
        perception='fit=%d seg/%d center/%.2f conf'%(
            fit.segments if fit else 0,len(fit.centre)//2 if fit else 0,fit.confidence if fit else 0.)
        if self.phase in ('MAPPING','FINALIZING'):
            progress='map %.1f m, %d path samples, %d wall points'%(
                state.travelled if state else 0.,self.map_samples,self.map_points)
        elif self.phase=='RACING':
            health=self.health.get('mpc_controller')
            progress='line %.1f m, %s'%(self.plan_length,health[1].reason if health else 'solver warming up')
            perception='local wall-map match'
        else:
            progress=self.stop_reason or 'wall racer'
        return '%s: %.1f m/s, %s, loc=%d matches/%.2f m, %s; %s'%(
            self.phase,speed,perception,matches,residual,
            'localized' if state and state.localized else 'fallback',progress)

    def _watchdog(self):
        if not self.live or self.halted:
            return
        now=time.monotonic()
        if self.have_scan and self.last_scan_wall_time is not None and now-self.last_scan_wall_time>.5:
            self._controlled_stop('lidar stream stale')
            return
        if self.phase=='ARMING':
            if self.reset_started is not None and now-self.reset_started>self.settings['reset_timeout']:
                self._controlled_stop('reset readiness deadline exceeded')
            return
        if self.phase=='FINALIZING' and self.finalizing_started is not None and now-self.finalizing_started>self.settings['planning_timeout']:
            self._controlled_stop('global planning handoff deadline exceeded')
            return
        for name in self.critical_nodes:
            receipt,msg=self.health.get(name,(0.,None))
            if now-receipt>self.p['sensor_timeout'] or msg is None or not msg.ready:
                self._controlled_stop(name+' unavailable or stale')
                return

    def tick(self):
        self._watchdog()
        if self.live and not self.halted:
            if self.phase=='ARMING' and all(
                    n in self.health and self.health[n][1].ready
                    and not self.health[n][1].fault
                    and time.monotonic()-self.health[n][0]<=self.p['sensor_timeout']
                    for n in self.critical_nodes):
                self.phase='MAPPING'
            if self.phase=='FINALIZING' and self.plan_id:
                if all(name in self.health
                       and self.health[name][1].meta.plan_id==self.plan_id
                       and self.health[name][1].meta.map_id==self.map_id
                       for name in ('mpc_controller','local_planner')):
                    self.phase='RACING'
        self.emit()

    def emit(self):
        if not hasattr(self,'pub_context'):
            return
        meta=header(self.run_id,self.generation,0,self.get_clock().now().nanoseconds*1e-9,
                    'map',self.config_id)
        meta.map_id=self.map_id
        meta.plan_id=self.plan_id
        msg=RunContext(meta=meta,phase=self.phase,live=self.live,halted=self.halted,
                       resetting=self.phase=='ARMING',reason=self.stop_reason)
        self.ctx=msg
        self.context_received=time.monotonic()
        self.pub_context.publish(msg)
        now=self.get_clock().now().nanoseconds*1e-9
        if now-self.last_status>=self.p['status_period']:
            self.last_status=now
            self.pub_status.publish(String(data=self.status_text()))


def main(args=None):
    spin(RunManagerNode,args)
