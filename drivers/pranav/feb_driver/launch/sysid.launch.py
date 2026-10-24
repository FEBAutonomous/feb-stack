from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

TOPICS = ["/autodrive/roboracer_1/" + t for t in
          ("left_encoder", "right_encoder", "imu", "throttle", "steering", "throttle_command", "steering_command")]


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=["ros2", "bag", "record", "-o", "/home/autodrive_devkit/src/stack/sysid_bag"] + TOPICS, output="screen"),
        Node(package="feb_driver", executable="sysid_probe", name="sysid_probe", parameters=[{"use_sim_time": True}], output="screen", emulate_tty=True),
    ])
