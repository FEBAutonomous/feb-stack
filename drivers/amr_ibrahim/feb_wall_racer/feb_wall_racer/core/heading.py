"""Public IMU heading in the odometry/map frame, with continuous fallback."""
from bisect import bisect_left
from collections import deque
import math


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(quaternion, covariance_first=0.0):
    if not math.isfinite(covariance_first) or covariance_first < 0.0 or not all(math.isfinite(v) for v in quaternion):
        return None
    norm = math.sqrt(sum(v*v for v in quaternion))
    if not .5 <= norm <= 1.5:
        return None
    x,y,z,w = (v/norm for v in quaternion)
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


class HeadingTracker:
    def __init__(self):
        self.samples = deque(maxlen=64)
        self.reset_reference()

    def reset_reference(self):
        self.samples.clear()
        self.offset = None
        self.correction = 0.0

    def observe(self, yaw, stamp, rate):
        if not all(math.isfinite(v) for v in (yaw,stamp,rate)):
            return
        if self.samples:
            if stamp <= self.samples[-1][0]:
                return
            previous = self.samples[-1][1]
            yaw = previous + wrap(yaw-previous)
        self.samples.append((stamp,yaw,rate))

    def resolve(self, odom_yaw, stamp, max_age=.15):
        fallback = wrap(odom_yaw+self.correction)
        if not self.samples or not math.isfinite(stamp):
            return fallback,False
        samples = list(self.samples)
        i = bisect_left([s[0] for s in samples],stamp)
        if i < len(samples) and abs(samples[i][0]-stamp) < 1e-8:
            measured = samples[i][1]
        elif i == 0 or i == len(samples):
            at,yaw,rate = samples[0 if i == 0 else -1]
            if abs(stamp-at) > max_age:
                return fallback,False
            measured = yaw+rate*(stamp-at)
        else:
            before,after = samples[i-1],samples[i]
            if max(stamp-before[0],after[0]-stamp) > max_age:
                return fallback,False
            fraction = (stamp-before[0])/(after[0]-before[0])
            measured = before[1]+fraction*(after[1]-before[1])
        if self.offset is None:
            self.offset = odom_yaw-measured
        corrected = wrap(measured+self.offset)
        # An orientation frame reset must not rotate an already-built map.
        if abs(wrap(corrected-fallback)) > .35:
            return fallback,False
        self.correction = wrap(corrected-odom_yaw)
        return corrected,True
