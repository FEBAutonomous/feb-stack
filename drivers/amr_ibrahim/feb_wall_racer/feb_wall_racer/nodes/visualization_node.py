"""Preserve display topics independently of control and estimated TF."""
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from feb_wall_racer_msgs.msg import WallObservation,LocalizedState,RacePlan,LocalReference
from .base import PipelineNode,spin
from ..core.visualization import path_message,pose_message,car_marker,Trajectory
from ..core.perception import lidar_points_to_map
from ..core.synchronization import CycleJoin
from ..interfaces import points,seconds,validate_plan
from ..qos import SENSOR_QOS,QOS,LATCHED
from .. import topics


class VisualizationNode(PipelineNode):
    def __init__(self,**kwargs):
        super().__init__('visualization',**kwargs)
        self.trajectory=Trajectory(self.p['path_sample_distance'])
        self.join=CycleJoin(('walls','state'))
        self.last_local=self.last_live=self.last_visual=-1e9
        self.plan=self.pending_plan=None
        self.paths={}
        for name in ('wall/left','wall/right','wall/live/left','wall/live/right',
                     'wall/map/left','wall/map/right','raceline','trajectory','mpc_prediction'):
            self.paths[name]=self.create_publisher(Path,'/feb/'+name,
                LATCHED if name in ('wall/map/left','wall/map/right','raceline') else QOS)
        self.pub_pose=self.create_publisher(PoseStamped,'/feb/pose',QOS)
        self.pub_marker=self.create_publisher(Marker,'/feb/car_marker',QOS)
        self.create_subscription(WallObservation,topics.WALLS,self.on_walls,SENSOR_QOS)
        self.create_subscription(LocalizedState,topics.STATE,self.on_state,SENSOR_QOS)
        self.create_subscription(RacePlan,topics.PLAN,self.on_plan,LATCHED)
        self.create_subscription(LocalReference,topics.PREDICTION,self.on_prediction,SENSOR_QOS)

    def on_context(self,msg):
        pending=self.pending_plan
        PipelineNode.on_context(self,msg)
        if self.accept(msg.meta):
            self.ready=True  # a restarted viewer may attach during racing
            if pending is not None and self.accept(pending.meta):
                self.on_plan(pending)

    def on_reset(self):
        self.trajectory=Trajectory(self.p['path_sample_distance'])
        self.join=CycleJoin(('walls','state'))
        self.last_local=self.last_live=self.last_visual=-1e9
        self.plan=None
        stamp=self.get_clock().now().to_msg()
        for name,publisher in self.paths.items():
            publisher.publish(path_message([],stamp,'laser' if name in ('wall/left','wall/right') else 'map'))
        marker=car_marker([0.,0.],0.,stamp);marker.action=Marker.DELETE
        self.pub_marker.publish(marker)

    def on_walls(self,msg):
        if not self.accept(msg.meta):return
        stamp=msg.meta.header.stamp
        t=seconds(stamp)
        try:
            if t-self.last_local>=.18:
                self.last_local=t
                for side in ('left','right'):
                    self.paths['wall/'+side].publish(path_message(points(getattr(msg,side)),stamp,msg.meta.header.frame_id))
            self.joined('walls',msg)
        except ValueError as exc:self.fail(str(exc))

    def on_state(self,msg):
        if not self.accept(msg.meta) or not msg.valid:return
        stamp=msg.meta.header.stamp
        self.pub_pose.publish(pose_message(msg.pose,msg.yaw,stamp))
        self.trajectory.append(msg.pose)
        t=seconds(stamp)
        if t-self.last_visual>=.2:
            self.last_visual=t
            self.paths['trajectory'].publish(path_message(self.trajectory.points,stamp,'map'))
            self.pub_marker.publish(car_marker(msg.pose,msg.yaw,stamp))
        self.joined('state',msg)

    def joined(self,kind,msg):
        try:
            for obs,state in self.join.add(kind,msg):
                t=seconds(obs.meta.header.stamp)
                if not state.valid or t-self.last_live<.18:continue
                self.last_live=t
                for side in ('left','right'):
                    world=lidar_points_to_map(points(getattr(obs,side)),state.pose,state.yaw,self.p['lidar_forward'])
                    self.paths['wall/live/'+side].publish(path_message(world,obs.meta.header.stamp,'map'))
        except ValueError as exc:self.fail(str(exc))

    def on_plan(self,msg):
        if not self.accept(msg.meta):
            self.pending_plan=msg;return
        if not msg.valid:return
        try:validate_plan(msg)
        except ValueError as exc:self.fail(str(exc));return
        self.plan=msg;self.pending_plan=None
        for side in ('left','right'):
            self.paths['wall/map/'+side].publish(path_message(points(getattr(msg,side)),msg.meta.header.stamp,'map'))
        self.paths['raceline'].publish(path_message(points(msg.line),msg.meta.header.stamp,'map'))

    def on_prediction(self,msg):
        if not self.accept(msg.meta) or not msg.valid or self.plan is None:return
        if msg.meta.map_id!=self.plan.meta.map_id or msg.meta.plan_id!=self.plan.meta.plan_id:return
        try:self.paths['mpc_prediction'].publish(path_message(points(msg.path),msg.meta.header.stamp,'map'))
        except ValueError as exc:self.fail(str(exc))


def main(args=None):
    spin(VisualizationNode,args)
