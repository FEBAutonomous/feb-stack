"""Lifecycle fixture over real owners; only context/health dispatch is deterministic.

Sensors, perception, motion, SLAM, references and /drive use real ROS transport.
The separate installed-launch tests exercise context/health over DDS too.
"""
from types import SimpleNamespace
from unittest.mock import patch
import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan,Imu
from nav_msgs.msg import Odometry
from feb_wall_racer.benchmarks.wall import synthetic_corridor_scan
from feb_wall_racer.nodes.run_manager_node import RunManagerNode
from feb_wall_racer.nodes.perception_node import PerceptionNode
from feb_wall_racer.nodes.motion_estimator_node import MotionEstimatorNode
from feb_wall_racer.nodes.slam_node import SlamNode
from feb_wall_racer.nodes.global_planner_node import GlobalPlannerNode
from feb_wall_racer.nodes.local_planner_node import LocalPlannerNode
from feb_wall_racer.nodes.mpc_controller_node import MPCControllerNode


class LifecycleHarness:
    def __init__(self):
        self.manager=RunManagerNode()
        self.perception=PerceptionNode()
        self.motion=MotionEstimatorNode()
        self.mapping=SlamNode()
        self.global_planner=GlobalPlannerNode()
        self.local=LocalPlannerNode()
        with patch('feb_wall_racer.nodes.mpc_controller_node.KinematicMPC',return_value=SimpleNamespace(ok=False)):
            self.control=MPCControllerNode()
        self.peers=[self.perception,self.motion,self.mapping,self.global_planner,self.local,self.control]
        self.inputs=Node('lifecycle_public_sensor_fixture')
        self.nodes=[self.manager,*self.peers,self.inputs]
        self.manager.pub_context=SimpleNamespace(publish=self.context)
        for node in self.peers:
            node.pub_health=SimpleNamespace(publish=self.manager.on_health)
        self.scan_pub=self.inputs.create_publisher(LaserScan,'/scan',10)
        self.odom_pub=self.inputs.create_publisher(Odometry,'/odom',10)
        self.imu_pub=self.inputs.create_publisher(Imu,'/imu',10)
        self.sample=synthetic_corridor_scan()
        self.inputs.create_timer(.1,self.publish_inputs)
        self.manager.emit()

    def context(self,msg):
        for node in self.peers:node.on_context(msg)

    def publish_inputs(self):
        stamp=self.inputs.get_clock().now().to_msg()
        odom=Odometry();odom.header.stamp=stamp;odom.pose.pose.orientation.w=1.
        imu=Imu();imu.header.stamp=stamp;imu.orientation.w=1.
        scan=LaserScan();scan.header.stamp=stamp;scan.header.frame_id='laser'
        scan.angle_min=float(self.sample.angle_min);scan.angle_increment=float(self.sample.angle_increment)
        scan.range_min=.05;scan.range_max=float(self.sample.range_max)
        scan.ranges=np.asarray(self.sample.ranges,dtype=np.float32).tolist()
        self.odom_pub.publish(odom);self.imu_pub.publish(imu);self.scan_pub.publish(scan)

    def on_state(self,msg):
        self.control.on_state(msg);self.manager.on_state(msg);self.manager.tick()

    def on_mission_live(self,msg):
        self.control.on_mission_live(msg);self.manager.on_mission_live(msg);self.manager.tick()

    def _command(self,steer,throttle):self.control._command(steer,throttle)
    def _controlled_stop(self,reason):self.control.fail(reason)
    def _idle_heartbeat(self):self.control.tick()
    def _watchdog(self):self.control._watchdog();self.manager._watchdog()

    @property
    def slam(self):return self.mapping.core.slam
    @property
    def controller(self):return self.control.controller
    @property
    def live(self):return self.manager.live
    @property
    def live_since(self):return self.manager.live_since
    @property
    def halted(self):return self.manager.halted or bool(self.control.fault)
    @property
    def commanded_steer(self):return self.control.commanded_steer
    @property
    def travelled(self):return self.mapping.core.travelled
    @travelled.setter
    def travelled(self,value):self.mapping.core.travelled=value
    @property
    def have_scan(self):return self.manager.have_scan
    @have_scan.setter
    def have_scan(self,value):
        self.manager.have_scan=value
        self.control.last_scan=time.monotonic() if value else None
    @property
    def last_scan_wall_time(self):return self.manager.last_scan_wall_time
    @last_scan_wall_time.setter
    def last_scan_wall_time(self,value):
        self.manager.last_scan_wall_time=value;self.control.last_scan=value
    @property
    def pub_drive(self):return self.control.pub_drive
    @pub_drive.setter
    def pub_drive(self,value):self.control.pub_drive=value

    def destroy_node(self):
        for node in reversed(self.nodes):node.destroy_node()
