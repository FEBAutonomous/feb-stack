"""Lifecycle regressions; run in ROS with an isolated ROS_DOMAIN_ID.

The manager/controller/motion/SLAM/planning nodes and public data transport are
real. Only MPC construction and deterministic context/health dispatch are fixtures;
the installed graph tests cover those two transports too.
"""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import rclpy
except ImportError:
    rclpy = None

if rclpy is not None:
    from ackermann_msgs.msg import AckermannDriveStamped
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
    from node_harness import LifecycleHarness


@unittest.skipIf(rclpy is None, "requires ROS 2; run inside the FEB box")
class MissionLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # Solver construction is unrelated to authorization and reset semantics.
        self.racer = LifecycleHarness()
        self.probe = Node("mission_lifecycle_probe")
        self.commands = []
        self.probe.create_subscription(
            AckermannDriveStamped, "/drive", self.commands.append, 10)
        self.mission_pub = self.probe.create_publisher(Bool, "/feb/mission/live", 10)

    def tearDown(self):
        self.racer.destroy_node()
        self.probe.destroy_node()

    def _mission(self, live):
        callback = getattr(self.racer, "on_mission_live", None)
        self.assertTrue(callable(callback), "wall racer must consume mission live")
        callback(Bool(data=live))

    def _start(self):
        self.racer.on_state(String(data="AUTONOMOUS: test"))
        self._mission(True)
        self.assertTrue(self.racer.live)

    def _spin_until(self, predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for node in self.racer.nodes:
                rclpy.spin_once(node, timeout_sec=0.001)
            rclpy.spin_once(self.probe, timeout_sec=0.01)
            if predicate():
                return
        self.fail("timed out waiting for lifecycle ROS messages")

    def test_supervisor_alone_waits_for_mission(self):
        original_map = self.racer.slam
        self.racer.on_state(String(data="AUTONOMOUS: test"))
        self.assertFalse(self.racer.live, "supervisor alone must not start the racer")
        self.assertIsNone(self.racer.live_since)
        self.assertIs(self.racer.slam, original_map)
        self.racer._command(0.2, 0.1)
        self.assertEqual(self.racer.commanded_steer, 0.0)

    def test_mission_before_supervisor_starts_once(self):
        original_map = self.racer.slam
        self._mission(True)
        self.assertFalse(self.racer.live)
        self.racer.on_state(String(data="AUTONOMOUS: test"))
        self.assertTrue(self.racer.live)
        self.assertIsNotNone(self.racer.live_since)
        self.assertIsNot(self.racer.slam, original_map)

    def test_supervisor_before_mission_starts_once(self):
        original_map = self.racer.slam
        self.racer.on_state(String(data="AUTONOMOUS: test"))
        self._mission(True)
        self.assertTrue(self.racer.live)
        self.assertIsNot(self.racer.slam, original_map)

    def test_repeated_live_messages_preserve_mapping_and_controller_memory(self):
        self._start()
        current_map, controller = self.racer.slam, self.racer.controller
        started = self.racer.live_since
        self.racer.travelled = 12.5
        for _ in range(3):
            self._mission(True)
            self.racer.on_state(String(data="AUTONOMOUS: another announcement"))
        self.assertEqual(self.racer.travelled, 12.5)
        self.assertIs(self.racer.slam, current_map)
        self.assertIs(self.racer.controller, controller)
        self.assertEqual(self.racer.live_since, started)

    def test_mission_false_stops_and_publishes_zero(self):
        self._start()
        self._spin_until(lambda: self.mission_pub.get_subscription_count() == 2)
        self.racer._command(0.2, 0.1)
        self._spin_until(lambda: any(m.drive.speed > 0.0 for m in self.commands))
        self.commands.clear()
        self._mission(False)
        self.assertFalse(self.racer.live)
        self.assertIsNone(self.racer.live_since)
        self.assertEqual(self.racer.commanded_steer, 0.0)
        self._spin_until(lambda: any(m.drive.speed == 0.0 for m in self.commands))

    def test_delayed_mission_edges_do_not_reset_twice_in_one_supervisor_session(self):
        self._start()
        self.racer.on_state(String(data="IDLE: released"))
        self.racer.on_state(String(data="AUTONOMOUS: new session"))
        self._mission(True)  # Can be queued from just before mission observed IDLE.
        current_map, controller = self.racer.slam, self.racer.controller
        self.racer.travelled = 4.5
        self._mission(False)  # Delayed IDLE edge from mission's own topic queue.
        self.assertFalse(self.racer.live)
        self._mission(True)
        self.assertTrue(self.racer.live)
        self.assertIs(self.racer.slam, current_map)
        self.assertIs(self.racer.controller, controller)
        self.assertEqual(self.racer.travelled, 4.5)

    def test_direct_supervisor_stop_dominates_late_mission_true(self):
        for state in ("STOP: test", "MANUAL: test", "IDLE: test"):
            with self.subTest(state=state):
                self._start()
                self.racer._command(0.2, 0.1)
                self.racer.on_state(String(data=state))
                self.assertFalse(self.racer.live)
                self.assertEqual(self.racer.commanded_steer, 0.0)
                self._mission(True)
                self.assertFalse(self.racer.live)
                self.racer.on_state(String(data="AUTONOMOUS: autonomous button"))
                self.assertFalse(self.racer.live, "old mission permission must be discarded")

    def test_new_autonomous_session_resets_once_after_mission_confirmation(self):
        self._start()
        previous_map = self.racer.slam
        self.racer.travelled = 25.0
        self.racer._controlled_stop("test obstacle")
        self.racer.on_state(String(data="STOP: supervisor"))
        self.racer.on_state(String(data="AUTONOMOUS: autonomous button"))
        self.assertFalse(self.racer.live)
        self.assertTrue(self.racer.halted)
        self._mission(True)
        self.assertTrue(self.racer.live)
        self.assertFalse(self.racer.halted)
        self.assertEqual(self.racer.travelled, 0.0)
        self.assertIsNot(self.racer.slam, previous_map)
        current_map = self.racer.slam
        self._mission(True)
        self.assertIs(self.racer.slam, current_map)

    def test_automatic_supervisor_reentry_cannot_clear_local_fault(self):
        self._start()
        current_map = self.racer.slam
        self.racer.travelled = 12.5
        self.racer._controlled_stop("odometry stale")
        self.racer.on_state(String(data="IDLE: released"))
        self.racer.on_state(String(data="AUTONOMOUS: stack is driving"))
        self._mission(True)
        self.assertTrue(self.racer.halted)
        self.assertFalse(self.racer.live)
        self.assertIs(self.racer.slam, current_map)
        self.assertEqual(self.racer.travelled, 12.5)

    def test_supervisor_stop_latches_even_before_local_watchdog(self):
        self._start()
        current_map = self.racer.slam
        self.racer.on_state(String(data="STOP: no lidar"))
        self.assertTrue(self.racer.halted)
        self.racer.on_state(String(data="IDLE: recovered"))
        self.racer.on_state(String(data="AUTONOMOUS: stack is driving"))
        self._mission(True)
        self.assertFalse(self.racer.live)
        self.assertTrue(self.racer.halted)
        self.assertIs(self.racer.slam, current_map)

    def test_held_autonomous_button_cannot_clear_local_fault(self):
        self._start()
        self.racer._controlled_stop("IMU stale")
        self.racer.on_state(String(data="AUTONOMOUS: autonomous button"))
        self._mission(False)
        self._mission(True)
        self.assertTrue(self.racer.halted)
        self.assertFalse(self.racer.live)

    def test_halted_racer_settles_with_stock_supervisor_without_restarting(self):
        # Load the actual image supervisor, with deterministic time and in-process
        # transport. Other tests above exercise the real ROS topic connections.
        from feb_supervisor.supervisor import Supervisor
        supervisor = Supervisor()
        self.addCleanup(supervisor.destroy_node)
        clock = [10.0]
        supervisor.now = lambda: clock[0]
        supervisor.p["require_deadman"] = False
        wheel_commands, drive_commands = [], []
        supervisor.pub_cmd = SimpleNamespace(publish=wheel_commands.append)

        def state(msg):
            self.racer.on_state(msg)
            self._mission(msg.data.startswith("AUTONOMOUS"))

        def drive(msg):
            drive_commands.append(msg)
            supervisor.on_drive(msg)

        supervisor.pub_state = SimpleNamespace(publish=state)
        self.racer.pub_drive = SimpleNamespace(publish=drive)
        supervisor.scan_t = clock[0]
        self.racer._idle_heartbeat()
        supervisor.tick()
        self.assertTrue(self.racer.live)
        self.racer._command(0.0, 0.1)
        supervisor.tick()
        self.assertGreater(wheel_commands[-1].drive.speed, 0.0)
        current_map = self.racer.slam
        self.racer.have_scan = True
        self.racer.last_scan_wall_time = clock[0]
        self.racer._controlled_stop("odometry stale")
        for _ in range(40):
            clock[0] += 0.1
            supervisor.scan_t = clock[0]
            with patch("time.monotonic", return_value=clock[0]):
                self.racer._watchdog()
            supervisor.tick()
            self.racer._idle_heartbeat()
        self.assertEqual(supervisor.state, "IDLE")
        self.assertTrue(self.racer.halted)
        self.assertFalse(self.racer.live)
        self.assertIs(self.racer.slam, current_map)
        self.assertTrue(all(msg.drive.speed == 0.0 for msg in wheel_commands[2:]))
        count = len(drive_commands)
        self.racer._idle_heartbeat()
        self.assertEqual(len(drive_commands), count)

        # A fresh operator autonomous edge still works with stock button semantics.
        from sensor_msgs.msg import Joy
        supervisor.on_joy(Joy(buttons=[0, 0, 0, 0, 0, 1, 0]))
        supervisor.tick()
        self.assertTrue(self.racer.live)
        self.assertFalse(self.racer.halted)
        self.assertIsNot(self.racer.slam, current_map)

    def test_stock_idle_before_watchdog_cannot_erase_stale_scan_fault(self):
        self._start()
        current_map = self.racer.slam
        self.racer.have_scan = True
        self.racer.last_scan_wall_time = 10.0
        # Stock command timeout may beat the racer's 250 ms timer by one tick.
        with patch("time.monotonic", return_value=10.51):
            self.racer.on_state(String(data="IDLE: released"))
            self.racer._idle_heartbeat()
            self.racer.on_state(String(data="AUTONOMOUS: stack is driving"))
            self._mission(True)
        self.assertTrue(self.racer.halted)
        self.assertFalse(self.racer.live)
        self.assertIs(self.racer.slam, current_map)

    def test_mission_stop_before_stock_idle_preserves_stale_scan_fault(self):
        self._start()
        current_map = self.racer.slam
        self.racer.have_scan = True
        self.racer.last_scan_wall_time = 10.0
        with patch("time.monotonic", return_value=10.51):
            self._mission(False)  # Mission may see the supervisor state first.
            self.racer.on_state(String(data="IDLE: released"))
            self.racer._idle_heartbeat()
            self.racer.on_state(String(data="AUTONOMOUS: stack is driving"))
            self._mission(True)
        self.assertTrue(self.racer.halted)
        self.assertFalse(self.racer.live)
        self.assertIs(self.racer.slam, current_map)

    def test_mission_heartbeat_or_toggle_cannot_clear_latched_safety_stop(self):
        self._start()
        current_map = self.racer.slam
        self.racer._controlled_stop("test obstacle")
        self._mission(True)
        self.racer.on_state(String(data="AUTONOMOUS: still the same session"))
        self.assertTrue(self.racer.halted)
        self._mission(False)
        self._mission(True)
        self.assertTrue(self.racer.halted)
        self.assertFalse(self.racer.live)
        self.assertIs(self.racer.slam, current_map)

    def test_idle_heartbeat_keeps_automatic_startup_possible(self):
        self._spin_until(lambda: self.probe.count_publishers("/drive") == 1)
        self.racer._idle_heartbeat()
        self._spin_until(lambda: bool(self.commands))
        self.assertTrue(all(m.drive.speed == 0.0 for m in self.commands))
        self.assertFalse(self.racer.live)

    def test_live_topic_is_connected_to_the_lifecycle_callback(self):
        self._spin_until(lambda: self.mission_pub.get_subscription_count() == 2)
        self.racer.on_state(String(data="AUTONOMOUS: transport test"))
        self.mission_pub.publish(Bool(data=True))
        self._spin_until(lambda: self.racer.live)
        self.mission_pub.publish(Bool(data=False))
        self._spin_until(lambda: not self.racer.live)


if __name__ == "__main__":
    unittest.main()
