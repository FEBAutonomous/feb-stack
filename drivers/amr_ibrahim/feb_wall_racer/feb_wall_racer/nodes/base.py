"""Common epoch, configuration and liveness transport; no driving algorithm."""
from __future__ import annotations
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rcl_interfaces.msg import ParameterDescriptor
from feb_wall_racer_msgs.msg import RunContext, NodeStatus
from ..configuration import (effective_parameters, node_parameters,
                             pipeline_configuration_id,NODE_SETTINGS,validate_node_settings)
from ..params import DEFAULTS
from ..interfaces import header, same_run
from .. import topics
from ..qos import QOS, LATCHED


class PipelineNode(Node):
    def __init__(self, name, *, manage_context=True, **kwargs):
        super().__init__(name+'_node', **kwargs)
        self.stage=name
        try:
            owned=node_parameters(DEFAULTS,name)
            for key,value in owned.items():
                self.declare_parameter(key,value,ParameterDescriptor(read_only=True))
            for key,value in NODE_SETTINGS[name].items():
                self.declare_parameter(key,value,ParameterDescriptor(read_only=True))
            self.settings=validate_node_settings({key:self.get_parameter(key).value
                                                  for key in NODE_SETTINGS[name]},name)
            self.declare_parameter('configuration_id','',ParameterDescriptor(read_only=True))
            unknown=set(self._parameter_overrides)-set(owned)-set(self.settings)-{'configuration_id','use_sim_time'}
            if unknown:
                raise ValueError('unowned node parameters: '+str(sorted(unknown)))
            requested=dict(DEFAULTS)
            requested.update({key:self.get_parameter(key).value for key in owned})
            self.p=effective_parameters(requested)
            settings={n:dict(v) for n,v in NODE_SETTINGS.items()}
            settings[name]=self.settings
            self.config_id=self.get_parameter('configuration_id').value or pipeline_configuration_id(self.p,settings)
            self.get_logger().info(
                'effective config: wheelbase=%.4f m steer=%.4f rad map=%.2f m/s race=%.2f m/s mode cap=%.2f m/s'
                %(self.p['wheelbase'],self.p['max_steer'],self.p['map_speed'],self.p['v_max'],self.p['mode_speed_limit']))
        except Exception:
            self.destroy_node()
            raise
        self.ctx=None
        self.context_received=None
        self.retired_epochs=set()
        self.ready=False
        self.fault=''
        self.loaded_plan_id=0
        self.steady=Clock(clock_type=ClockType.STEADY_TIME)
        self.pub_health=self.create_publisher(NodeStatus,topics.health(name),QOS)
        if manage_context:
            self.create_subscription(RunContext,topics.CONTEXT,self.on_context,LATCHED)
        self.create_timer(.1,self.publish_health,clock=self.steady)

    def meta(self, cycle=0, stamp=None, frame='map'):
        stamp=self.get_clock().now().nanoseconds*1e-9 if stamp is None else stamp
        return header(self.ctx.meta.run_id if self.ctx else '',
                      self.ctx.meta.generation if self.ctx else 0,cycle,stamp,frame,self.config_id)

    def accept(self, meta):
        return bool(self.ctx and same_run(meta,self.ctx.meta.run_id,self.ctx.meta.generation)
                    and meta.config_id==self.config_id)

    def on_context(self, msg):
        epoch=(msg.meta.run_id,msg.meta.generation)
        if epoch in self.retired_epochs or not epoch[0]:
            return
        if self.ctx and msg.meta.run_id==self.ctx.meta.run_id and msg.meta.generation<self.ctx.meta.generation:
            return
        if msg.meta.config_id!=self.config_id:
            self.fail('configuration identity mismatch')
            return
        old=None if self.ctx is None else (self.ctx.meta.run_id,self.ctx.meta.generation)
        self.ctx=msg
        self.context_received=time.monotonic()
        if old!=epoch:
            if old:
                self.retired_epochs.add(old)
            self.fault=''
            self.loaded_plan_id=0
            self.on_reset()
            # A retained racing context cannot initialize a newly restarted driver.
            self.ready=msg.phase in ('IDLE','ARMING')
        self.context_changed()
        self.publish_health()

    def on_reset(self):
        pass

    def context_changed(self):
        pass

    def active(self):
        return bool(self.ready and self.ctx and self.ctx.live and not self.ctx.halted
                    and not self.fault and self.ctx.phase in ('MAPPING','FINALIZING','RACING')
                    and self.context_received is not None
                    and time.monotonic()-self.context_received<=self.p['sensor_timeout'])

    def fail(self, reason):
        if not self.fault:
            self.get_logger().error(reason)
        self.fault=reason
        self.publish_health()

    def publish_health(self):
        msg=NodeStatus(meta=self.meta(),node=self.stage,ready=self.ready,
                       fault=bool(self.fault),critical=self.stage!='visualization',
                       phase=getattr(self,'activity',self.ctx.phase if self.ctx else 'WAITING'),
                       reason=self.fault or getattr(self,'diagnostic',''))
        msg.meta.plan_id=self.loaded_plan_id
        msg.meta.map_id=getattr(self,'loaded_map_id',self.ctx.meta.map_id if self.ctx else 0)
        self.pub_health.publish(msg)


def spin(node_type,args=None):
    rclpy.init(args=args)
    node=None
    try:
        node=node_type()
        rclpy.spin(node)
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
