import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="feb_mission", executable="mission", name="mission",
             parameters=[{"use_sim_time": os.environ.get("FEB_CAR") != "1"}], output="screen", emulate_tty=True),
    ])
