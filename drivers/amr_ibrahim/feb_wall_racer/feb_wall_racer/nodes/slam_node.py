"""Own the wall map, estimated pose, closure and estimated TF."""
import copy
import math
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from feb_wall_racer_msgs.msg import (WallObservation,MotionEstimate,LocalizedState,
                                    TrackMap,WallContext)
from .base import PipelineNode,spin
from ..core.mapping import SlamEngine
from ..core.synchronization import CycleJoin
from ..interfaces import decode_observation,flat,seconds
from ..qos import SENSOR_QOS,LATCHED
from .. import topics


class SlamNode(PipelineNode):
    def __init__(self,**kwargs):
        super().__init__('slam',**kwargs)
        self.core=SlamEngine(self.p,self.get_logger())
        self.join=CycleJoin(('walls','motion'))
        self.latest=None
        self.last_map_stamp=-1e9
        self.published_map_id=0
        self.pub_state=self.create_publisher(LocalizedState,topics.STATE,SENSOR_QOS)
        self.pub_map=self.create_publisher(TrackMap,topics.MAP,LATCHED)
        self.pub_context=self.create_publisher(WallContext,topics.WALL_CONTEXT,SENSOR_QOS)
        self.tf=TransformBroadcaster(self)
        self.create_subscription(WallObservation,topics.WALLS,lambda m:self.input('walls',m),SENSOR_QOS)
        self.create_subscription(MotionEstimate,topics.MOTION,lambda m:self.input('motion',m),SENSOR_QOS)
        self.create_timer(.1,self.watchdog,clock=self.steady)

    def on_reset(self):
        self.core.reset()
        self.join=CycleJoin(('walls','motion'))
        self.latest=None
        self.last_map_stamp=-1e9
        self.published_map_id=0
        self.publish_map(self.meta())

    def input(self,kind,msg):
        if not self.active() or not self.accept(msg.meta):
            return
        if kind=='walls':
            self.latest=msg
        try:
            for obs,motion in self.join.add(kind,msg):
                self.process(obs,motion)
        except ValueError as exc:
            self.fail(str(exc))

    def process(self,obs,motion):
        stamp=seconds(obs.meta.header.stamp)
        localized=False
        valid=obs.valid and motion.valid
        if valid:
            if not np.all(np.isfinite([*motion.odom_position,motion.yaw,motion.heading_correction,
                                      motion.body_speed,motion.dt,motion.travelled])):
                self.fail('non-finite motion state')
                valid=False
            else:
                localized=self.core.process(motion,decode_observation(obs),stamp)
        if self.core.stop_reason:
            self.fail(self.core.stop_reason)
            valid=False
        meta=copy.deepcopy(obs.meta)
        meta.header.frame_id='map'
        meta.map_id=self.core.map_id
        meta.map_revision=self.core.map_revision
        meta.pose_revision=self.core.pose_revision
        meta.plan_id=self.ctx.meta.plan_id if meta.map_id==self.ctx.meta.map_id else 0
        state=LocalizedState(meta=meta,motion=motion,pose=self.core.pose.tolist(),yaw=self.core.yaw,
            lidar_heading_offset=self.core.lidar_heading_offset,travelled=self.core.travelled,
            last_localization_stamp=self.core.last_loc_t,matches=self.core.last_matches,
            residual=self.core.last_residual,valid=valid,localized=localized,
            finalized=self.core.slam.frozen,reason=self.fault or obs.reason or motion.reason)
        self.pub_state.publish(state)
        if valid:
            self.publish_tf(state)
            self.publish_prior(state)
            if ((self.core.map_id and self.published_map_id!=self.core.map_id)
                    or not self.core.slam.frozen and stamp-self.last_map_stamp>=.2):
                self.publish_map(meta)
                self.last_map_stamp=stamp
                self.published_map_id=self.core.map_id

    def publish_map(self,meta):
        self.pub_map.publish(TrackMap(meta=meta,walls=flat(self.core.slam.points),
            trail=flat(self.core.path_points),widths=list(self.core.path_widths),
            finalized=self.core.slam.frozen,valid=not bool(self.core.stop_reason),reason=self.core.stop_reason))

    def publish_prior(self,state):
        msg=WallContext(meta=state.meta,pose=state.pose,yaw=state.yaw,
                        odom_position=state.motion.odom_position,
                        heading_correction=state.motion.heading_correction+self.core.lidar_heading_offset)
        for side in ('left','right'):
            entries=[e for e in self.core.slam._side_voxels[side].values()
                     if e[1]>=self.core.slam.min_label_votes]
            setattr(msg,side,flat([e[0] for e in entries]))
            setattr(msg,side+'_votes',[float(e[1]) for e in entries])
        self.pub_context.publish(msg)

    def publish_tf(self,state):
        tf=TransformStamped()
        tf.header=state.meta.header
        tf.child_frame_id='feb_wall_racer_base'
        tf.transform.translation.x=float(state.pose[0])
        tf.transform.translation.y=float(state.pose[1])
        tf.transform.rotation.z=math.sin(state.yaw/2.)
        tf.transform.rotation.w=math.cos(state.yaw/2.)
        self.tf.sendTransform(tf)

    def watchdog(self):
        if self.active() and self.join.stale(self.p['sensor_timeout']):
            self.fail('matching motion snapshot unavailable')
            if self.latest:
                self.pub_state.publish(LocalizedState(meta=self.latest.meta,valid=False,reason=self.fault))


def main(args=None):
    spin(SlamNode,args)
