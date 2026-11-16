import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool,String
from feb_wall_racer.nodes.run_manager_node import RunManagerNode
from feb_wall_racer.qos import QOS,LATCHED
from feb_wall_racer_msgs.msg import RunContext


def test_manager_receives_stock_volatile_supervisor_and_mission_publishers():
    rclpy.init(args=[])
    manager=RunManagerNode();probe=Node('stock_contract_probe')
    try:
        state=probe.create_publisher(String,'/feb/state',QOS)
        mission=probe.create_publisher(Bool,'/feb/mission/live',QOS)
        deadline=time.monotonic()+2.
        while time.monotonic()<deadline and not manager.live:
            state.publish(String(data='AUTONOMOUS: test'))
            mission.publish(Bool(data=True))
            rclpy.spin_once(manager,timeout_sec=.01)
            rclpy.spin_once(probe,timeout_sec=.01)
        assert manager.live,'stock VOLATILE topics must be compatible with manager subscriptions'
    finally:
        manager.destroy_node();probe.destroy_node();rclpy.shutdown()


def test_internal_context_is_delivered_to_a_late_subscriber():
    rclpy.init(args=[])
    manager=RunManagerNode();probe=Node('late_context_probe')
    try:
        manager.emit()
        received=[]
        probe.create_subscription(RunContext,'/feb/run/context',received.append,LATCHED)
        deadline=time.monotonic()+2.
        while not received and time.monotonic()<deadline:
            rclpy.spin_once(probe,timeout_sec=.02)
        assert received and received[0].meta.run_id==manager.run_id
    finally:
        manager.destroy_node();probe.destroy_node();rclpy.shutdown()
