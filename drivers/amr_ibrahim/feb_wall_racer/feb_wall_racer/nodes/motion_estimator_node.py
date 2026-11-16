"""Advance the vehicle observer on selected scan cycles, never on a timer."""
import copy
import math
import time
from collections import OrderedDict,deque
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from feb_wall_racer_msgs.msg import WallObservation,MotionEstimate
from .base import PipelineNode,spin
from ..core.estimation import MotionEstimator,SensorState,SensorUnavailable
from ..interfaces import seconds
from ..qos import SENSOR_QOS,QOS
from .. import topics


def stamp_key(stamp):
    return stamp.sec*1000000000+stamp.nanosec


class MotionEstimatorNode(PipelineNode):
    def __init__(self,**kwargs):
        super().__init__('motion_estimator',**kwargs)
        self.core=MotionEstimator(self.p)
        self.raw=SensorState()
        self.history={'odom':OrderedDict(),'imu':OrderedDict()}
        self.pending=OrderedDict()
        self.last_cycle=-1
        self.active_since=None
        self.previous_phase=None
        self.feedback=None
        self.pub_motion=self.create_publisher(MotionEstimate,topics.MOTION,SENSOR_QOS)
        self.create_subscription(Odometry,self.p['odom_topic'],lambda m:self.sensor('odom',m),SENSOR_QOS)
        self.create_subscription(Imu,self.p['imu_topic'],lambda m:self.sensor('imu',m),SENSOR_QOS)
        self.create_subscription(WallObservation,topics.WALLS,self.on_walls,SENSOR_QOS)
        self.create_subscription(AckermannDriveStamped,self.p['drive_topic'],self.on_drive,QOS)
        self.create_subscription(AckermannDriveStamped,'/ackermann_cmd',self.on_feedback,QOS)
        self.create_timer(.05,self.drain,clock=self.steady)

    def on_reset(self):
        self.core.reset()
        self.raw.heading.reset_reference()
        self.pending.clear()
        self.last_cycle=-1
        self.active_since=None
        self.previous_phase=None

    def context_changed(self):
        if not self.ctx.live:
            self.active_since=None
        elif self.ctx.phase in ('MAPPING','FINALIZING','RACING') and (
                self.active_since is None or self.previous_phase not in ('MAPPING','FINALIZING','RACING')):
            self.active_since=time.monotonic()
        self.previous_phase=self.ctx.phase

    def sensor(self,kind,msg):
        history=self.history[kind]
        getattr(self.raw,'on_'+kind)(msg)
        fields=(('odom_ready','odom_position','odom_yaw','wheel_speed','last_odom_wall_time','last_odom_stamp')
                if kind=='odom' else ('imu_ready','yaw_rate','last_imu_wall_time','last_imu_source'))
        snapshot={name:copy.deepcopy(getattr(self.raw,name)) for name in fields}
        history[stamp_key(msg.header.stamp)]=(time.monotonic(),msg,snapshot)
        while len(history)>128:
            history.popitem(last=False)
        self.drain()

    def on_drive(self,msg):
        if math.isfinite(msg.drive.steering_angle):
            self.core.commanded_steer=float(msg.drive.steering_angle)

    def on_feedback(self,msg):
        # Supervisor feedback is diagnostic, not a changed prediction input.
        self.feedback=msg

    def on_walls(self,msg):
        if not self.active() or not self.accept(msg.meta) or msg.meta.cycle<=self.last_cycle:
            return
        self.pending[msg.meta.cycle]=(time.monotonic(),msg)
        while len(self.pending)>16:
            self.pending.popitem(last=False)
        self.drain()

    def drain(self):
        if not self.active():
            return
        for cycle,(received,obs) in list(self.pending.items()):
            if not obs.meta.odom_available or not obs.meta.imu_available:
                self.pending.pop(cycle)
                try:
                    self.core.interval(seconds(obs.meta.header.stamp))
                except ValueError as exc:
                    self.fail(str(exc))
                self.pub_motion.publish(MotionEstimate(meta=copy.deepcopy(obs.meta),valid=False,
                    reason='selected-cycle sensors unavailable'))
                self.last_cycle=cycle
                if self.active_since is not None and time.monotonic()-self.active_since>self.p['sensor_timeout']:
                    self.fail('selected-cycle sensors unavailable')
                continue
            odom=self.history['odom'].get(stamp_key(obs.meta.odom_stamp))
            imu=self.history['imu'].get(stamp_key(obs.meta.imu_stamp))
            if odom is None or imu is None:
                if time.monotonic()-received>self.p['sensor_timeout']:
                    self.fail('selected-cycle sensor snapshot unavailable')
                return
            self.pending.pop(cycle)
            for key,value in dict(odom[2],**imu[2]).items():
                setattr(self.core,key,copy.deepcopy(value))
            self.core.heading.samples=deque((s for s in self.raw.heading.samples
                if s[0]<=seconds(obs.meta.imu_stamp)+1e-9),maxlen=64)
            msg=MotionEstimate(meta=copy.deepcopy(obs.meta),odom_stamp=odom[1].header.stamp,
                               imu_stamp=self.core.last_imu_source or imu[1].header.stamp)
            msg.meta.header.frame_id='map'
            try:
                state=self.core.step(seconds(obs.meta.header.stamp))
                for key,value in state.items():
                    setattr(msg,key,value.tolist() if key=='odom_position' else value)
                msg.valid=True
            except SensorUnavailable as exc:
                msg.reason=str(exc)
                if self.active_since is None or time.monotonic()-self.active_since>self.p['sensor_timeout']:
                    self.fail(str(exc))
            except ValueError as exc:
                msg.reason=str(exc)
                self.fail(str(exc))
            self.last_cycle=cycle
            self.pub_motion.publish(msg)


def main(args=None):
    spin(MotionEstimatorNode,args)
