"""Mapping planner/controller boundary fixture, with original physical assertions."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import time
import numpy as np
from std_msgs.msg import Bool,String
from feb_wall_racer_msgs.msg import RunContext,MotionEstimate,LocalizedState
from feb_wall_racer.interfaces import header,encode_observation,ros_time
from feb_wall_racer.nodes.local_planner_node import LocalPlannerNode
from feb_wall_racer.nodes.mpc_controller_node import MPCControllerNode


class MappingHarness:
    def __init__(self, *, use_gap_matcher=False):
        with patch('feb_wall_racer.nodes.mpc_controller_node.KinematicMPC',return_value=SimpleNamespace(ok=False)):
            self.control=MPCControllerNode()
        self.local=LocalPlannerNode()
        self.p=self.control.p
        self.source=10.;self.cycle=0;self.last_fit=None
        self.control._control_time=lambda:self.source
        self.refs=[];self.commands=[]
        self.local.pub_reference=SimpleNamespace(publish=self.refs.append)
        self.control.pub_drive=SimpleNamespace(publish=self.commands.append)
        self.meta=header('mapping-test',1,0,self.source,'map',self.control.config_id)
        for node in (self.local,self.control):
            node.on_context(RunContext(meta=self.meta,phase='ARMING',live=True))
            node.on_context(RunContext(meta=self.meta,phase='MAPPING',live=True))
        self.local.guidance.use_gap_matcher = use_gap_matcher
        self.control.on_state(String(data='AUTONOMOUS: test'))
        self.control.on_mission_live(Bool(data=True))

    def run(self,observation,speed):
        self.cycle+=1;self.source+=.1
        meta=header('mapping-test',1,self.cycle,self.source,'map',self.control.config_id)
        self.control.on_context(RunContext(meta=self.meta,phase='MAPPING',live=True))
        state=LocalizedState(meta=meta,valid=True,
            motion=MotionEstimate(meta=meta,odom_stamp=ros_time(self.source),body_speed=float(speed),valid=True))
        obs=encode_observation(observation,meta)
        if self.local.guidance.use_gap_matcher:
            self.control.on_walls(obs)
            self.control.clearance_input('state',state)
        self.local.process(obs,state)
        self.control.process(self.refs[-1],state)
        command=self.commands[-1]
        return command.drive.steering_angle,command.drive.speed/self.control.vehicle.speed_per_throttle

    def _local_follow(self,centre,speed,target_speed):
        assert target_speed==self.p['map_speed']
        return self.run(replace(self.last_fit,centre=np.asarray(centre,dtype=float)),speed)

    def _mapping_fallback(self,speed):
        observation=replace(self.last_fit,centre=np.empty((0,2)),confidence=0.)
        steer,throttle=self.run(observation,speed)
        ref=self.refs[-1]
        return steer,throttle,bool(ref.guidance_cycle),ref.guidance_age

    @property
    def last_mapping_guidance_wall_time(self):return self.local.guidance.last[0]
    @last_mapping_guidance_wall_time.setter
    def last_mapping_guidance_wall_time(self,value):
        self.local.guidance.last=(value,*self.local.guidance.last[1:])
    @property
    def pub_drive(self):return self.control.pub_drive
    @pub_drive.setter
    def pub_drive(self,value):self.control.pub_drive=value
    def _idle_heartbeat(self):self.control.tick()
    def destroy_node(self):self.local.destroy_node();self.control.destroy_node()
