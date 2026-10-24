"""System-identification probe: drives with the reactive steering so the car stays on
the track, while the throttle follows a fixed schedule of steps, a coast and a chirp.
Record a bag alongside (launch/sysid.launch.py does) and fit it with tools/sysid_fit.py
to get the steady-state gain, the idle-brake coefficient and the throttle lag.
"""
import math
import time

import rclpy
from std_msgs.msg import Float32

from feb_driver.reactive_driver import ReactiveDriver, spin


def schedule(t):
    """Throttle at time t (s) and the phase number, for the analysis."""
    if t < 6: return 0.08, 1      # steady-state gain: three throttle steps
    if t < 12: return 0.14, 1
    if t < 18: return 0.20, 1
    if t < 22: return 0.00, 2     # coast: idle brake
    if t < 30: return 0.16, 2     # step: rise time
    if t < 70:                    # chirp 0.05..0.6 Hz around 0.14
        f = 0.05 + 0.55 * (t - 30) / 40.0
        return 0.14 + 0.06 * math.sin(2 * math.pi * f * (t - 30)), 3
    k = int((t - 70) // 3)        # pseudo-random steps held 3 s
    return 0.08 + 0.12 * ((k * 7919) % 13) / 12.0, 4


class SysIdProbe(ReactiveDriver):
    def __init__(self):
        super().__init__("sysid_probe")
        self.declare_parameter("duration", 130.0)
        self.t0 = None
        self.pub_phase = self.create_publisher(Float32, "/feb/sysid_phase", 10)

    def throttle_for(self, v_target, dt):
        if self.t0 is None:
            self.t0 = time.time()
        t = time.time() - self.t0
        if t > self.get_parameter("duration").value:
            self.get_logger().info("schedule finished; stop the bag and run tools/sysid_fit.py")
            raise KeyboardInterrupt
        throttle, phase = schedule(t)
        self.pub_phase.publish(Float32(data=float(phase)))
        return throttle


def main():
    rclpy.init()
    spin(SysIdProbe())


if __name__ == "__main__":
    main()
