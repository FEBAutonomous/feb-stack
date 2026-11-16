"""Real driver launch tests with synthetic sensors, never a simulator or hardware.

Run inside the FEB box after building/sourcing the workspace, on an isolated
ROS_DOMAIN_ID. The process group created by each test is its only shutdown target.
"""
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tempfile
import time
import unittest

try:
    import rclpy
except ImportError:
    rclpy = None

if rclpy is not None:
    import numpy as np
    from ament_index_python.packages import get_package_share_directory
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from rcl_interfaces.srv import GetParameters
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Bool, String
    from feb_wall_racer.benchmarks.wall import synthetic_corridor_scan


@unittest.skipIf(rclpy is None, "requires a built ROS workspace inside the FEB box")
class MissionLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])
        cls.launch_file = Path(os.environ.get(
            "FEB_MISSION_TEST_LAUNCH",
            str(Path(get_package_share_directory("feb_wall_racer")) / "launch/drive.launch.py")))

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wall-mission-test-")
        self.log = open(Path(self.temp.name) / "launch.log", "w+")
        self.process = None
        self.probe = Node("mission_launch_probe")
        self.commands, self.statuses, self.mission = [], [], []
        self.probe.create_subscription(
            AckermannDriveStamped, "/drive", self.commands.append, 20)
        self.probe.create_subscription(String, "/feb/status", self.statuses.append, 20)
        self.probe.create_subscription(Bool, "/feb/mission/live", self.mission.append, 20)
        self.state_pub = self.probe.create_publisher(String, "/feb/state", 10)
        self.clock_pub = self.probe.create_publisher(Clock, "/clock", 10)
        self.scan_pub = self.probe.create_publisher(LaserScan, "/scan", 10)
        self.odom_pub = self.probe.create_publisher(Odometry, "/odom", 10)
        self.imu_pub = self.probe.create_publisher(Imu, "/imu", 10)
        self.state = "IDLE: launch test"
        self.started = time.monotonic()
        self.probe.create_timer(0.1, self._publish_inputs)

    def tearDown(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3.0)
        self.probe.destroy_node()
        self.log.close()
        self.temp.cleanup()

    def _output(self):
        self.log.flush()
        self.log.seek(0)
        return self.log.read()

    def _wait(self, predicate, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self.probe, timeout_sec=0.02)
            if predicate():
                return
        self.fail("launch condition timed out:\n" + self._output())

    def _publish_inputs(self):
        now = 10.0 + time.monotonic() - self.started
        clock = Clock()
        clock.clock.sec = int(now)
        clock.clock.nanosec = int((now - int(now)) * 1e9)
        self.clock_pub.publish(clock)
        odom = Odometry()
        odom.header.stamp = clock.clock
        odom.pose.pose.orientation.w = 1.0
        imu = Imu()
        imu.header.stamp = clock.clock
        scan = LaserScan()
        sample = synthetic_corridor_scan()
        scan.header.stamp, scan.header.frame_id = clock.clock, "laser"
        scan.angle_min = sample.angle_min
        scan.angle_increment = sample.angle_increment
        scan.angle_max = sample.angle_min + (len(sample.ranges) - 1) * sample.angle_increment
        scan.range_min, scan.range_max = 0.05, sample.range_max
        # Give the synthetic launch test measured coverage out to the range
        # limit. The gap matcher treats missing measurements as unknown.
        scan.ranges = np.where(np.isposinf(sample.ranges), sample.range_max,
                               sample.ranges).astype(np.float32).tolist()
        self.odom_pub.publish(odom)
        self.imu_pub.publish(imu)
        self.scan_pub.publish(scan)
        self.state_pub.publish(String(data=self.state))

    def _launch(self, on_car=False, env=None, mode="trackdrive", car_overlay=None):
        child_env = dict(os.environ, FEB_CAR="1" if on_car else "0")
        # Tests never inherit a user's tuning override.
        child_env.pop("FEB_RACER_PARAMS", None)
        child_env.pop("FEB_WALL_RACER_PARAMS", None)
        child_env.update(env or {})
        target = self.launch_file
        if car_overlay is not None:
            # Real child processes, temporary configuration; no hardware drivers.
            original_share = Path(get_package_share_directory("feb_wall_racer"))
            share = Path(self.temp.name)/"install/feb_wall_racer/share/feb_wall_racer"
            shutil.copytree(original_share/"config",share/"config")
            shutil.copy2(original_share/"driver.yaml",share/"driver.yaml")
            (share/"config/driver.car.yaml").write_text(car_overlay)
            target = Path(self.temp.name)/"layers.launch.py"
            target.write_text(
                "import importlib.util\n"
                "from ament_index_python.packages import get_package_share_directory as real_share\n"
                f"spec=importlib.util.spec_from_file_location('wall_launch',{str(self.launch_file)!r})\n"
                "module=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n"
                f"module.get_package_share_directory=lambda p: {str(share)!r} if p=='feb_wall_racer' else real_share(p)\n"
                "def generate_launch_description():\n    return module.generate_launch_description()\n")
        self.process = subprocess.Popen(
            ["ros2", "launch", str(target), "mode:="+mode], env=child_env,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def _parameter(self, node_name, name):
        client = self.probe.create_client(GetParameters, "/%s/get_parameters" % node_name)
        try:
            self._wait(client.service_is_ready)
            future = client.call_async(GetParameters.Request(names=[name]))
            self._wait(future.done)
            return future.result().values[0]
        finally:
            self.probe.destroy_client(client)

    def test_sim_launch_drives_stops_restarts_and_exits_with_mission(self):
        self._launch(car_overlay="wall_racer:\n  ros__parameters:\n    wheelbase: 0.45\n")
        self._wait(lambda: self.probe.get_node_names().count("mpc_controller_node") == 1
                   and self.probe.get_node_names().count("mission") == 1, timeout=45.0)
        self.assertTrue(self._parameter("mpc_controller_node", "use_sim_time").bool_value)
        self.assertTrue(self._parameter("mission", "use_sim_time").bool_value)
        self.assertAlmostEqual(self._parameter("mpc_controller_node", "map_speed").double_value, 1.8)
        self.assertAlmostEqual(self._parameter("mpc_controller_node", "wheelbase").double_value, .339)
        self._wait(lambda: self.commands and self.mission)
        self.assertFalse(self.mission[-1].data)
        self.assertTrue(all(m.drive.speed == 0.0 for m in self.commands))

        self.state = "AUTONOMOUS: launch test"
        self.commands.clear()
        self._wait(lambda: any(m.drive.speed > 0.0 for m in self.commands))
        self._wait(lambda: any("MAPPING:" in m.data for m in self.statuses))
        self.assertTrue(self.mission[-1].data)

        self.state = "STOP: launch test"
        self._wait(lambda: self.mission and not self.mission[-1].data)
        # The stop sends zero; a halted driver then suppresses readiness traffic.
        self._wait(lambda: self.commands and self.commands[-1].drive.speed == 0.0)
        self.commands.clear()
        self.state = "AUTONOMOUS: stack is driving"
        self._wait(lambda: self.mission[-1].data)
        deadline = time.monotonic() + 0.6
        self._wait(lambda: time.monotonic() >= deadline)
        self.assertTrue(all(m.drive.speed == 0.0 for m in self.commands))
        self.state = "IDLE: released"
        self._wait(lambda: not self.mission[-1].data)
        self.state = "AUTONOMOUS: autonomous button"
        self.commands.clear()
        self._wait(lambda: any(m.drive.speed > 0.0 for m in self.commands))

        # Only terminate the child identified in this test's own launch log.
        mission_pids = re.findall(r"\[mission-\d+\]: process started with pid \[(\d+)\]",
                                  self._output())
        self.assertEqual(len(mission_pids), 1, self._output())
        mission_pid = int(mission_pids[0])
        self.assertIn(b"/lib/feb_mission/mission",
                      Path("/proc/%d/cmdline" % mission_pid).read_bytes())
        os.kill(mission_pid, signal.SIGTERM)
        self._wait(lambda: self.process.poll() is not None, timeout=8.0)
        self._wait(lambda: "mpc_controller_node" not in self.probe.get_node_names())

    def test_car_clock_and_parameter_override_are_preserved_without_hardware(self):
        override = Path(self.temp.name) / "override.yaml"
        override.write_text("wall_racer:\n  ros__parameters:\n    map_speed: 1.3\n    wheelbase: 0.5\n")
        self._launch(on_car=True, env={"FEB_WALL_RACER_PARAMS": str(override)},
                     car_overlay="wall_racer:\n  ros__parameters:\n    wheelbase: 0.45\n    max_steer: 0.15\n")
        self._wait(lambda: "mpc_controller_node" in self.probe.get_node_names()
                   and "mission" in self.probe.get_node_names(), timeout=45.0)
        self.assertFalse(self._parameter("mpc_controller_node", "use_sim_time").bool_value)
        self.assertFalse(self._parameter("mission", "use_sim_time").bool_value)
        self.assertAlmostEqual(self._parameter("mpc_controller_node", "map_speed").double_value, 1.3)
        self.assertAlmostEqual(self._parameter("mpc_controller_node", "wheelbase").double_value, .5)
        self.assertAlmostEqual(self._parameter("mpc_controller_node", "max_steer").double_value, .15)
        self.assertIn("wheelbase=0.5000 m steer=0.1500 rad",self._output())
        self._wait(lambda: bool(self.commands))
        self.assertTrue(all(m.drive.speed == 0.0 for m in self.commands))

    def test_inspection_mode_caps_the_planner_and_commands(self):
        self._launch(mode="inspection")
        self._wait(lambda: "mpc_controller_node" in self.probe.get_node_names(),timeout=45.)
        self.assertAlmostEqual(self._parameter("mpc_controller_node","mode_speed_limit").double_value,.8)
        self.assertIn("map=0.80 m/s race=0.80 m/s mode cap=0.80 m/s",self._output())
        self.state = "AUTONOMOUS: inspection test"
        self._wait(lambda:any(m.drive.speed>0 for m in self.commands))
        self.assertTrue(all(m.drive.speed <= .8+1e-8 for m in self.commands))

    def test_missing_mission_fails_launch_clearly(self):
        # Hide only prefixes that provide mission; leave wall/ROS installed.
        prefixes = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
                    if p and not (Path(p) / "share/ament_index/resource_index/packages/feb_mission").exists()]
        self._launch(env={"AMENT_PREFIX_PATH": os.pathsep.join(prefixes)})
        self._wait(lambda: self.process.poll() is not None, timeout=8.0)
        self.assertNotEqual(self.process.returncode, 0)
        self.assertIn("feb_wall_racer requires feb_mission", self._output())


if __name__ == "__main__":
    unittest.main()
