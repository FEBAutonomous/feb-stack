"""Choose current guidance; the controller owns delay-aware race projection."""
import copy
import time
from feb_wall_racer_msgs.msg import WallObservation,LocalizedState,RacePlan,LocalReference
from .base import PipelineNode,spin
from ..core.pathing import GuidancePlanner
from ..core.synchronization import CycleJoin
from ..interfaces import decode_observation,flat,seconds,ros_time,validate_plan
from ..qos import SENSOR_QOS,LATCHED
from .. import topics


class LocalPlannerNode(PipelineNode):
    def __init__(self,**kwargs):
        super().__init__('local_planner',**kwargs)
        self.guidance=GuidancePlanner(self.p, use_gap_matcher=True)
        self.join=CycleJoin(('walls','state'))
        self.plan=None
        self.loaded_map_id=0
        self.pub_reference=self.create_publisher(LocalReference,topics.REFERENCE,SENSOR_QOS)
        self.create_subscription(WallObservation,topics.WALLS,lambda m:self.input('walls',m),SENSOR_QOS)
        self.create_subscription(LocalizedState,topics.STATE,lambda m:self.input('state',m),SENSOR_QOS)
        self.create_subscription(RacePlan,topics.PLAN,self.on_plan,LATCHED)
        self.create_timer(.1,self.watchdog,clock=self.steady)

    def on_reset(self):
        self.guidance=GuidancePlanner(self.p, use_gap_matcher=True)
        self.join=CycleJoin(('walls','state'))
        self.plan=None
        self.loaded_map_id=0
        self.pub_reference.publish(LocalReference(meta=self.meta(),mode='STOP',valid=False,
                                                   emergency=True,reason='reset'))

    def on_plan(self,msg):
        if not self.accept(msg.meta) or not msg.valid:
            return
        if self.ctx.meta.map_id not in (0,msg.meta.map_id):
            return
        try:
            validate_plan(msg)
        except ValueError as exc:
            self.fail(str(exc))
            return
        self.plan=msg
        self.loaded_plan_id=msg.meta.plan_id
        self.loaded_map_id=msg.meta.map_id
        self.publish_health()

    def input(self,kind,msg):
        if not self.active() or not self.accept(msg.meta):
            return
        try:
            for obs,state in self.join.add(kind,msg):
                self.process(obs,state)
        except ValueError as exc:
            self.fail(str(exc))

    def process(self,obs,state):
        meta=copy.deepcopy(state.meta)
        if not obs.valid or not state.valid:
            self.pub_reference.publish(LocalReference(meta=meta,mode='STOP',valid=False,
                emergency=True,reason=obs.reason or state.reason))
            return
        if self.ctx.phase=='RACING':
            if (self.plan is None or state.meta.map_id!=self.plan.meta.map_id
                    or state.meta.plan_id!=self.plan.meta.plan_id
                    or self.ctx.meta.plan_id!=self.plan.meta.plan_id):
                return  # a bounded handoff wait; controller independently expires data
            lost=seconds(meta.header.stamp)-state.last_localization_stamp
            if lost>self.p['stop_after']:
                self.fail('wall-map localization lost')
                return
            cap=min(self.p['mode_speed_limit'],
                    self.p['blind_speed'] if lost>self.p['lost_after'] else self.p['mode_speed_limit'])
            self.pub_reference.publish(LocalReference(meta=meta,mode='RACING',valid=True,
                path=self.plan.line,target_speed=float(cap),collision_distance=float('inf')))
        else:
            chosen=self.guidance.mapping(decode_observation(obs),state.motion.body_speed,
                                         time.monotonic(),meta.cycle,seconds(meta.header.stamp),yaw=state.yaw)
            meta.header.frame_id=obs.meta.header.frame_id
            chosen['path']=flat(chosen['path'])
            chosen['target']=[float(v) for v in chosen['target']]
            chosen['guidance_stamp']=ros_time(chosen['guidance_stamp'])
            self.pub_reference.publish(LocalReference(meta=meta,**chosen))

    def watchdog(self):
        if self.active() and self.join.stale(self.p['sensor_timeout']):
            self.fail('matching observation/localized state unavailable')


def main(args=None):
    spin(LocalPlannerNode,args)
