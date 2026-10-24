import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from ament_index_python.packages import PackageNotFoundError
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# mode is the driver's mode from driver.yaml (trackdrive, inspection, ...). This driver behaves the
# same in every mode; the supervisor's speed cap is what changes. Read it here if yours differs.
# On the car (FEB_CAR=1) the clock is the wall clock and config/<name>.car.yaml, if present, is
# loaded on top of the base config: only the numbers that differ on the real car go there.


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("feb_cone_driver"), "config")
    on_car = os.environ.get("FEB_CAR") == "1"
    params = [os.path.join(cfg, "cone_driver.yaml")]
    if on_car and os.path.exists(os.path.join(cfg, "cone_driver.car.yaml")):
        params.append(os.path.join(cfg, "cone_driver.car.yaml"))      # later files win
    actions = [
        DeclareLaunchArgument("mode", default_value="trackdrive"),
        Node(package="feb_cone_driver", executable="cone_driver", name="cone_driver",
             parameters=params + [{"use_sim_time": not on_car}], output="screen", emulate_tty=True),
    ]
    try:    # the mission node (feb_mission, in the starter kit) when it is in the workspace: go/stop hooks from /feb/state
        share = get_package_share_directory("feb_mission")
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, "launch", "mission.launch.py"))))
    except PackageNotFoundError:
        pass
    return LaunchDescription(actions)
