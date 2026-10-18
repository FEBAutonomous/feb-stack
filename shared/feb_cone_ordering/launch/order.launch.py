from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="feb_cone_ordering", executable="cone_ordering_node", name="cone_ordering",
             parameters=[{"use_sim_time": True}], output="screen", emulate_tty=True),
    ])
