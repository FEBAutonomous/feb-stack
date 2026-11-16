"""Selected-scan vehicle observer and public sensor validation."""
import math
import time
import numpy as np
from .heading import HeadingTracker,quaternion_yaw
from .vehicle import VehicleModel,BodySpeed


def _yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))


class SensorInputs:
    def on_imu(self, msg):
        yaw_rate = float(msg.angular_velocity.z)
        if math.isfinite(yaw_rate):
            self.yaw_rate = yaw_rate
            self.imu_ready = True
            self.last_imu_wall_time = time.monotonic()
            self.last_imu_source = msg.header.stamp
            q = msg.orientation
            yaw = quaternion_yaw((q.x,q.y,q.z,q.w),msg.orientation_covariance[0])
            if yaw is not None:
                stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                self.heading.observe(yaw,stamp,yaw_rate)

    def on_odom(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        speed = float(msg.twist.twist.linear.x)
        quaternion_norm = math.sqrt(
            orientation.x * orientation.x + orientation.y * orientation.y
            + orientation.z * orientation.z + orientation.w * orientation.w)
        if (not all(math.isfinite(value) for value in (
                position.x, position.y, speed, quaternion_norm))
                or quaternion_norm <= 0.5 or not -5.0 < speed < 25.0):
            self.odom_ready = False
            return
        self.odom_position = np.array([position.x, position.y], dtype=float)
        self.odom_yaw = _yaw(orientation)
        self.wheel_speed = speed
        self.odom_ready = True
        self.last_odom_wall_time = time.monotonic()
        self.last_odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class SensorUnavailable(ValueError):
    pass


class SensorState(SensorInputs):
    """Raw held inputs, separate from selected-cycle observer replay."""
    def __init__(self):
        self.heading=HeadingTracker()
        self.odom_ready=self.imu_ready=False
        self.odom_position=None
        self.odom_yaw=self.yaw_rate=self.wheel_speed=0.
        self.last_odom_wall_time=self.last_imu_wall_time=None
        self.last_odom_stamp=None
        self.last_imu_source=None


class MotionEstimator(SensorState):
    def __init__(self,p):
        super().__init__()
        self.p=p
        self.vehicle=VehicleModel.from_parameters(p)
        self.reset()

    def reset(self):
        self.heading.reset_reference()
        self.body=BodySpeed(self.vehicle)
        self.delta=self.commanded_steer=0.
        self.last_scan_t=None
        self.started=False
        self.travelled=0.

    def interval(self,stamp):
        raw_dt=None if self.last_scan_t is None else stamp-self.last_scan_t
        self.last_scan_t=stamp
        if raw_dt is not None and (raw_dt<=0. or raw_dt>.5):
            raise ValueError('invalid lidar timestamp interval %.3f s'%raw_dt)
        return .1 if raw_dt is None else float(np.clip(raw_dt,.01,.5))

    def step(self,stamp,now=None):
        now=time.monotonic() if now is None else now
        dt=self.interval(stamp)
        if not self.odom_ready or self.last_odom_wall_time is None or now-self.last_odom_wall_time>self.p['sensor_timeout']:
            raise SensorUnavailable('odometry unavailable or stale')
        if not self.imu_ready or self.last_imu_wall_time is None or now-self.last_imu_wall_time>self.p['sensor_timeout']:
            raise SensorUnavailable('IMU unavailable or stale')
        yaw,active=self.heading.resolve(self.odom_yaw,self.last_odom_stamp)
        self.delta+=float(np.clip(self.commanded_steer-self.delta,
                                  -self.vehicle.steer_rate*dt,self.vehicle.steer_rate*dt))
        speed=self.body.update(self.wheel_speed,dt,self.yaw_rate,self.delta)
        if self.started:
            self.travelled+=max(float(speed),0.)*dt
        self.started=True
        return dict(odom_position=self.odom_position.copy(),odom_yaw=self.odom_yaw,
                    yaw=yaw,heading_correction=self.heading.correction,heading_active=active,
                    body_speed=float(speed),wheel_speed=self.wheel_speed,yaw_rate=self.yaw_rate,
                    steering=self.delta,dt=dt,travelled=self.travelled)
