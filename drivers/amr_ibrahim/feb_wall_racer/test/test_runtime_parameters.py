"""Synthetic ROS nodes only: calibrated values must reach live commands/reset."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
try:
    import rclpy
except ImportError:
    rclpy = None
if rclpy is not None:
    from rclpy.parameter import Parameter
    from feb_wall_racer.nodes.mpc_controller_node import MPCControllerNode
    from feb_wall_racer.nodes.motion_estimator_node import MotionEstimatorNode
    from feb_wall_racer.nodes.slam_node import SlamNode
    from test_controller_node import authorize


@unittest.skipIf(rclpy is None,"requires ROS in FEB box")
class RuntimeParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def node(self,**values):
        with patch("feb_wall_racer.nodes.mpc_controller_node.KinematicMPC",return_value=SimpleNamespace(ok=False)):
            node = MPCControllerNode(parameter_overrides=[Parameter(k,value=v) for k,v in values.items()])
        self.addCleanup(node.destroy_node)
        return node

    def test_live_commands_use_configured_geometry_conversion_and_mode_limit(self):
        node = self.node(wheelbase=.45,max_steer=.15,speed_per_throttle=10.,mode_speed_limit=.8)
        commands = []
        node.pub_drive = SimpleNamespace(publish=commands.append)
        authorize(node)
        node._command(.4,.2)
        self.assertAlmostEqual(commands[-1].drive.steering_angle,.15)
        self.assertAlmostEqual(commands[-1].drive.speed,.8)
        self.assertAlmostEqual(node.controller.vehicle.wheelbase,.45)
        node.on_reset()
        motion=MotionEstimatorNode(parameter_overrides=[Parameter(k,value=v) for k,v in
            dict(wheelbase=.45,max_steer=.15,speed_per_throttle=10.,mode_speed_limit=.8).items()])
        self.addCleanup(motion.destroy_node)
        motion.on_reset()
        self.assertIs(motion.core.body.model,motion.core.vehicle)
        self.assertAlmostEqual(motion.core.vehicle.wheelbase,node.controller.vehicle.wheelbase)
        self.assertAlmostEqual(node.controller.vehicle.max_steer,.15)

    def test_runtime_edit_is_rejected_instead_of_becoming_ineffective(self):
        node = self.node(wheelbase=.45)
        result = node.set_parameters([Parameter("wheelbase",value=.5)])
        self.assertFalse(result[0].successful)
        self.assertAlmostEqual(node.get_parameter("wheelbase").value,.45)

    def test_mapping_alignment_updates_persistent_heading_and_reset_clears_it(self):
        import math
        import numpy as np
        from feb_wall_racer.core.slam import Alignment
        owner=SlamNode()
        self.addCleanup(owner.destroy_node)
        node=owner.core
        node.start_pose = np.zeros(2)
        node.start_yaw = 0.
        node.travelled = .5
        node.yaw = .02
        node.last_scan_t = 1.
        node.slam.add(np.column_stack((np.linspace(-5,5,50),np.ones(50))),np.zeros(2),0,0,0)
        points = np.column_stack((np.linspace(-3,3,20),np.ones(20)))
        observation = SimpleNamespace(left=points,right=points*[1,-1],centre=np.array([[1.,0.],[2.,0.],[3.,0.]]),confidence=1.,width=2.2)
        result = Alignment(np.array([.01,.02]),.01,30,.01,2,True)
        with patch.object(node.slam,'align_pose',return_value=result):
            node._mapping_step(1.,observation)
        self.assertAlmostEqual(node.lidar_heading_offset,-.01)
        self.assertAlmostEqual(node.heading_correction,-.01)
        self.assertAlmostEqual(node.last_integrated_heading_correction,-.01)
        self.assertAlmostEqual(node.yaw,.01)
        node.odom_position = np.array([1.,0.])
        node.last_odom_position = np.zeros(2)
        before = node.pose.copy()
        node._integrate(1.,1.,True)
        np.testing.assert_allclose(node.pose-before,[math.cos(.01),-math.sin(.01)],atol=1e-12)
        owner.on_reset()
        self.assertEqual(node.lidar_heading_offset,0.)

    def test_parameter_descriptors_keep_their_own_names_and_types(self):
        node = self.node(wheelbase=.45)
        geometry,debug = node.describe_parameters(["wheelbase","debug"])
        self.assertEqual(geometry.name,"wheelbase")
        self.assertEqual(debug.name,"debug")
        self.assertNotEqual(geometry.type,debug.type)


if __name__ == "__main__":
    unittest.main()
