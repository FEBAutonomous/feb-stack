"""Stable driver-card entry point for the modular wall-racing graph."""
from ament_index_python.packages import get_package_share_directory
from feb_wall_racer.launching import driver_launch


def generate_launch_description():
    return driver_launch(get_package_share_directory('feb_wall_racer'),
                         get_package_share_directory)
