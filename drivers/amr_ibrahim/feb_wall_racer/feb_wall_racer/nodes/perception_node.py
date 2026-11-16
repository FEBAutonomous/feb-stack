"""Fit walls and select scan cycles using bounded previous-map context."""
import math
import copy
import time
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan,Imu
from feb_wall_racer_msgs.msg import WallObservation,WallContext
from .base import PipelineNode,spin
from ..core.perception import observe,corridor_is_usable
from ..core.slam import WallMap
from ..interfaces import encode_observation,points,seconds
from ..qos import SENSOR_QOS
from .. import topics


class PerceptionNode(PipelineNode):
    def __init__(self,**kwargs):
        super().__init__('perception',**kwargs)
        self.cycle=0
        self.last_stamp=None
        self.last_scan=None
        self.hint=0.
        self.odom=None
        self.imu=None
        self.prior=None
        self.pub_walls=self.create_publisher(WallObservation,topics.WALLS,SENSOR_QOS)
        self.create_subscription(LaserScan,self.p['scan_topic'],self.on_scan,SENSOR_QOS)
        self.create_subscription(Odometry,self.p['odom_topic'],self.on_odom,SENSOR_QOS)
        self.create_subscription(Imu,self.p['imu_topic'],self.on_imu,SENSOR_QOS)
        self.create_subscription(WallContext,topics.WALL_CONTEXT,self.on_prior,SENSOR_QOS)
        self.create_timer(.1,self.watchdog,clock=self.steady)

    def on_reset(self):
        self.cycle=0
        self.last_stamp=self.last_scan=None
        self.hint=0.
        self.prior=None

    def on_odom(self,msg):
        self.odom=msg

    def on_imu(self,msg):
        self.imu=msg

    def on_prior(self,msg):
        if not self.accept(msg.meta):
            return
        try:
            left,right=points(msg.left),points(msg.right)
            if len(left)!=len(msg.left_votes) or len(right)!=len(msg.right_votes):
                return
            if not np.all(np.isfinite([*msg.pose,msg.yaw,*msg.odom_position,msg.heading_correction])):
                return
            wall_map=WallMap(self.p)
            for side,xy,votes in (('left',left,msg.left_votes),('right',right,msg.right_votes)):
                for point,vote in zip(xy,votes):
                    if not math.isfinite(vote) or vote<0:
                        return
                    key=tuple(np.floor(point/wall_map.voxel).astype(int))
                    wall_map._side_voxels[side][key]=[point,float(vote),0.,1]
            self.prior=(time.monotonic(),msg,wall_map)
        except ValueError:
            return

    def prior_for(self,stamp):
        if not self.prior or self.odom is None:
            return None
        received,ctx,wall_map=self.prior
        age=stamp-seconds(ctx.meta.header.stamp)
        if (time.monotonic()-received>self.p['sensor_timeout'] or age<0 or age>self.p['sensor_timeout']
                or ctx.meta.map_id!=self.ctx.meta.map_id):
            return None
        raw=self.odom.pose.pose
        if not np.all(np.isfinite([raw.position.x,raw.position.y,*np.asarray(ctx.pose)])):
            return None
        correction=ctx.heading_correction
        c,s=math.cos(correction),math.sin(correction)
        delta=np.array([raw.position.x,raw.position.y])-np.asarray(ctx.odom_position)
        pose=np.asarray(ctx.pose)+np.array([c*delta[0]-s*delta[1],s*delta[0]+c*delta[1]])
        q=raw.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))+correction
        return lambda left,right:wall_map.score_side_labels(left,right,pose,yaw,self.p['lidar_forward'])

    def on_scan(self,msg):
        if not self.active():
            return
        stamp=seconds(msg.header.stamp)
        self.cycle+=1
        meta=self.meta(self.cycle,stamp,msg.header.frame_id)
        meta.header=copy.deepcopy(msg.header)
        meta.map_id=self.ctx.meta.map_id
        if self.odom is not None:
            meta.odom_stamp=self.odom.header.stamp
            meta.odom_available=True
        if self.imu is not None:
            meta.imu_stamp=self.imu.header.stamp
            meta.imu_available=True
        interval=None if self.last_stamp is None else stamp-self.last_stamp
        self.last_stamp=stamp
        self.last_scan=time.monotonic()
        invalid=WallObservation(meta=meta,valid=False,front_clearance=float('inf'))
        if interval is not None and (interval<=0. or interval>.5):
            invalid.reason='invalid lidar timestamp interval %.3f s'%interval
            self.pub_walls.publish(invalid)
            self.fail(invalid.reason)
            return
        mapping=self.ctx.phase!='RACING'
        try:
            observation=observe(msg,self.p,self.hint if mapping else 0.,
                                wall_prior=self.prior_for(stamp) if mapping else None,
                                with_corridor=mapping)
            out=encode_observation(observation,meta)
            if mapping and corridor_is_usable(observation,self.p):
                self.hint=observation.heading
            self.pub_walls.publish(out)
        except (ValueError,ArithmeticError) as exc:
            invalid.reason='perception failed: '+str(exc)
            self.pub_walls.publish(invalid)
            self.fail(invalid.reason)

    def watchdog(self):
        if self.active() and self.last_scan is not None and time.monotonic()-self.last_scan>.5:
            self.fail('lidar stream stale')


def main(args=None):
    spin(PerceptionNode,args)
