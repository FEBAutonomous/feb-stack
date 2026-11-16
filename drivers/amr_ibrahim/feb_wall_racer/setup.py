import os
from glob import glob
from setuptools import find_packages, setup

package_name = "feb_wall_racer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test", "test.*")),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "driver.yaml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FEB Autonomous",
    maintainer_email="pranavbhatttheonlyone@gmail.com",
    description="Lidar wall-track racing stack with mapping, raceline, and MPC",
    license="MIT",
    entry_points={"console_scripts": [
        "perception_node = feb_wall_racer.nodes.perception_node:main",
        "motion_estimator_node = feb_wall_racer.nodes.motion_estimator_node:main",
        "slam_node = feb_wall_racer.nodes.slam_node:main",
        "global_planner_node = feb_wall_racer.nodes.global_planner_node:main",
        "local_planner_node = feb_wall_racer.nodes.local_planner_node:main",
        "mpc_controller_node = feb_wall_racer.nodes.mpc_controller_node:main",
        "run_manager_node = feb_wall_racer.nodes.run_manager_node:main",
        "visualization_node = feb_wall_racer.nodes.visualization_node:main",
        "mapping_debug_node = feb_wall_racer.nodes.mapping_debug_node:main",
        "wall_bench = feb_wall_racer.benchmarks.wall:main",
        "mpc_bench = feb_wall_racer.benchmarks.controller:main",
    ]},
)
