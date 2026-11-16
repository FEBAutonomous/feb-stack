"""Bounded display data and ROS message conversion; no control authority."""
import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

MAX_TRAJECTORY_POSES=2000


def path_message(points,stamp,frame):
    msg=Path()
    msg.header.stamp=stamp;msg.header.frame_id=frame
    for x,y in np.asarray(points,dtype=float).reshape(-1,2):
        pose=PoseStamped();pose.header=msg.header
        pose.pose.position.x=float(x);pose.pose.position.y=float(y)
        pose.pose.orientation.w=1.
        msg.poses.append(pose)
    return msg


def pose_message(position,yaw,stamp):
    msg=PoseStamped();msg.header.stamp=stamp;msg.header.frame_id='map'
    msg.pose.position.x=float(position[0]);msg.pose.position.y=float(position[1])
    msg.pose.orientation.z=math.sin(yaw/2.);msg.pose.orientation.w=math.cos(yaw/2.)
    return msg


def car_marker(position,yaw,stamp):
    msg=Marker();msg.header.stamp=stamp;msg.header.frame_id='map'
    msg.ns='feb_wall_racer';msg.id=0;msg.type=Marker.CUBE;msg.action=Marker.ADD
    msg.pose.position.x=float(position[0]+.16*math.cos(yaw))
    msg.pose.position.y=float(position[1]+.16*math.sin(yaw));msg.pose.position.z=.06
    msg.pose.orientation.z=math.sin(yaw/2.);msg.pose.orientation.w=math.cos(yaw/2.)
    msg.scale.x=.42;msg.scale.y=.24;msg.scale.z=.12
    msg.color.r=.95;msg.color.g=.55;msg.color.b=.12;msg.color.a=1.
    msg.lifetime.sec=1
    return msg


class Trajectory:
    def __init__(self,spacing):
        self.spacing=spacing;self.points=[]

    def append(self,position):
        point=np.asarray(position,dtype=float)
        if not self.points or np.linalg.norm(point-self.points[-1])>=self.spacing:
            self.points.append(point.copy())
            if len(self.points)>MAX_TRAJECTORY_POSES:
                del self.points[:-MAX_TRAJECTORY_POSES]
