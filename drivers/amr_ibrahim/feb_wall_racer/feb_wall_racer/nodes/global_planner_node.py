"""Build each finalized global plan once without blocking lifecycle callbacks."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from feb_wall_racer_msgs.msg import TrackMap,RacePlan
from .base import PipelineNode,spin
from ..core.planning import plan_track
from ..interfaces import points,flat
from ..qos import LATCHED
from .. import topics


class GlobalPlannerNode(PipelineNode):
    def __init__(self,**kwargs):
        super().__init__('global_planner',**kwargs)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='global_plan')
        self.future=None
        self.job=None
        self.queued=None
        self.completed=None
        self.plan_counter=0
        self.activity='IDLE'
        self.pub_plan=self.create_publisher(RacePlan,topics.PLAN,LATCHED)
        self.create_subscription(TrackMap,topics.MAP,self.on_map,LATCHED)
        self.create_timer(.02,self.poll,clock=self.steady)

    def on_reset(self):
        self.queued=self.completed=None
        self.plan_counter=0
        self.activity='IDLE'
        self.pub_plan.publish(RacePlan(meta=self.meta(),valid=False,reason='reset'))

    @staticmethod
    def key(msg):
        return (msg.meta.run_id,msg.meta.generation,msg.meta.map_id,msg.meta.config_id)

    def on_map(self,msg):
        if not self.ready or not self.accept(msg.meta) or not msg.finalized or not msg.meta.map_id:
            return
        key=self.key(msg)
        if key==self.completed or self.job and key==self.key(self.job[0]):
            return
        if not msg.valid:
            self.fail('invalid finalized map: '+msg.reason)
            return
        if self.ctx.phase=='RACING' and msg.meta.map_id!=self.ctx.meta.map_id:
            return
        self.queued=msg
        self.poll()

    def poll(self):
        if self.future is not None and self.future.done():
            source,_=self.job
            future=self.future
            self.future=self.job=None
            if self.accept(source.meta) and not self.fault:
                meta=copy.deepcopy(source.meta)
                try:
                    result=future.result()
                    self.plan_counter+=1
                    meta.plan_id=self.plan_counter
                    track=result['track']
                    msg=RacePlan(meta=meta,line=flat(result['line']),speeds=result['speeds'].tolist(),
                        left=flat(track['left']),right=flat(track['right']),centre=flat(track['centre']),
                        half_width=track['half_width'].tolist(),length=float(track['length']),
                        arc_length=result['arc_length'].tolist(),valid=True)
                    self.pub_plan.publish(msg)
                    self.completed=self.key(source)
                    self.activity='READY'
                except Exception as exc:
                    self.fail('global planning failed: '+str(exc))
                    self.pub_plan.publish(RacePlan(meta=meta,valid=False,reason=self.fault))
        if self.future is not None and time.monotonic()-self.job[1]>self.settings['planning_timeout']:
            self.fail('global planning deadline exceeded')
        if self.future is None and self.queued is not None and not self.fault:
            source=self.queued
            self.queued=None
            if self.accept(source.meta):
                try:
                    trail=points(source.trail);widths=list(source.widths)
                    self.future=self.pool.submit(plan_track,trail,widths,dict(self.p),walls=points(source.walls))
                    self.job=(source,time.monotonic())
                    self.activity='PLANNING'
                except ValueError as exc:
                    self.fail(str(exc))

    def destroy_node(self):
        if hasattr(self,'pool'):
            self.pool.shutdown(wait=True,cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    spin(GlobalPlannerNode,args)
