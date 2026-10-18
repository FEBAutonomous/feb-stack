"""Mission: the part of a stack that listens to the car.

The supervisor publishes /feb/state: IDLE, MANUAL, AUTONOMOUS or STOP. Your stack is running the
whole time, also while a human drives with the gamepad, so anything that remembers (an MPC's
warm start, a localiser, integrators) is stale by the time the autonomous button goes down.
This node turns the state into two moments, on_go and on_stop, and publishes /feb/mission/live
(Bool) for nodes that just want a flag.

The gap follower needs none of this: it re-plans from the scan every tick. The starter drivers'
launch files include this package when it is in the workspace. A stack with memory copies this
package and fills in the two hooks, or subscribes to /feb/mission/live from the node that has
the memory and resets there. The rule: on AUTONOMOUS, start clean.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class Mission(Node):
    def __init__(self):
        super().__init__("mission")
        self.live = False
        self.pub_live = self.create_publisher(Bool, "/feb/mission/live", 10)
        self.create_subscription(String, "/feb/state", self.on_state, 10)
        self.create_timer(0.1, lambda: self.pub_live.publish(Bool(data=self.live)))
        self.get_logger().info("mission: waiting for AUTONOMOUS on /feb/state")

    def on_state(self, msg):
        live = msg.data.startswith("AUTONOMOUS")
        if live and not self.live:
            self.on_go()
        elif self.live and not live:
            self.on_stop(msg.data)
        self.live = live

    def on_go(self):
        """The autonomous button just went down. Reset what remembers: warm starts, integrators,
        the pose estimate from the current odometry. Nothing to do for the gap follower."""
        self.get_logger().info("go: the stack is driving")

    def on_stop(self, why):
        """The button was released, or the supervisor stopped us. The stack keeps running but its
        commands go nowhere; freeze what should not drift while a human drives."""
        self.get_logger().info("not driving: %s" % why)


def main():
    rclpy.init()
    node = Mission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
