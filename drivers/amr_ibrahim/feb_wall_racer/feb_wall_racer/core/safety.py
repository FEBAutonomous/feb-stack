"""Final physical command boundary and mapping tyre-force conversion."""
import math
import numpy as np

SWEEP_PADDING = .025


def clip_command(steer,throttle,p,authorized):
    if not authorized:
        return 0.,0.
    if not math.isfinite(steer) or not math.isfinite(throttle):
        raise ValueError('non-finite command')
    return (float(np.clip(steer,-p['max_steer'],p['max_steer'])),
            float(np.clip(throttle,0.,min(1.,p['mode_speed_limit']/p['speed_per_throttle']))))


def mapping_throttle(target_speed,speed,p,vehicle):
    force=float(np.clip((target_speed-speed)/.4,-p['mpc_a_brake'],p['mpc_a_acc']))+vehicle.drag*speed
    return float(np.clip(vehicle.command_for_force(speed,force),p['throttle_min'],p['throttle_max']))


def braking_path(pose, speed, current_steer, steer, delay, reaction, brake, vehicle):
    """Sample a delayed steering transition followed by braking."""
    x, y, yaw = pose
    dt = .01
    path = [(x, y, yaw)]
    # Check an initial move from rest as well as a moving stop envelope.
    speed = max(speed, .30)
    for i in range(math.ceil((reaction + speed / brake) / dt)):
        elapsed = i * dt
        if elapsed >= delay:
            current_steer += float(np.clip(steer - current_steer, -vehicle.steer_rate * dt,
                                          vehicle.steer_rate * dt))
        distance = speed * dt
        turn = distance * math.tan(current_steer) / vehicle.wheelbase
        x += distance * math.cos(yaw + turn / 2.)
        y += distance * math.sin(yaw + turn / 2.)
        yaw += turn
        if elapsed >= reaction:
            speed = max(0., speed - brake * dt)
        path.append((x, y, yaw))
    return np.asarray(path)


def first_blocked_distance(path, obstacles, wheelbase, margin):
    """Find the first measured return in the swept axle capsule."""
    path = np.asarray(path, dtype=float).copy()
    path[:, 2] = np.unwrap(path[:, 2])
    delta = np.diff(path, axis=0)
    distances = np.linalg.norm(delta[:, :2], axis=1)
    stations = np.r_[0., np.cumsum(distances)]
    # Include motion at the front axle when choosing interpolation intervals.
    counts = np.maximum(1, np.ceil((distances + wheelbase * abs(delta[:, 2])) / .05)).astype(int)
    fractions = np.concatenate([np.arange(n) / n for n in counts] + [np.array([0.])])
    starts = np.repeat(np.arange(len(delta)), counts)
    dense = np.vstack((path[starts] + fractions[:-1, None] * delta[starts], path[-1]))
    arc = np.r_[stations[starts] + fractions[:-1] * distances[starts], stations[-1]]
    blocked = np.flatnonzero(axle_clearance_squared(dense, obstacles, wheelbase) <= (margin + SWEEP_PADDING)**2)
    return float(arc[blocked[0]]) if len(blocked) else math.inf


def axle_clearance_squared(poses, obstacles, wheelbase):
    """Squared distance to the closest lidar return along the axle segment."""
    offset = obstacles[None, :, :] - poses[:, None, :2]
    c, s = np.cos(poses[:, 2, None]), np.sin(poses[:, 2, None])
    longitudinal = offset[..., 0] * c + offset[..., 1] * s
    lateral = -offset[..., 0] * s + offset[..., 1] * c
    excess = longitudinal - np.clip(longitudinal, 0., wheelbase)
    return np.min(excess**2 + lateral**2, axis=1, initial=math.inf)


def clear_crawl_steering(pose, obstacles, speed, current_steer, desired_steer,
                        delay, reaction, brake, vehicle, half_width, margin):
    """Find the closest clear steering, assuming pre-alignment if current_steer is None."""
    current_clearance = axle_clearance_squared(np.asarray([pose]), obstacles, vehicle.wheelbase)[0]
    if current_clearance <= (half_width + SWEEP_PADDING)**2:
        return None
    target_clearance = (margin + SWEEP_PADDING)**2
    # From inside the extra wall buffer, only allow escape paths that keep
    # at least the current separation and end outside the full buffer.
    minimum_clearance = min(current_clearance, target_clearance)
    candidates = np.r_[desired_steer, np.linspace(-vehicle.max_steer, vehicle.max_steer, 13)]
    for candidate in candidates[np.argsort(np.abs(candidates - desired_steer), kind='stable')]:
        initial_steer = candidate if current_steer is None else current_steer
        path = braking_path(pose, speed, initial_steer, candidate, delay, reaction, brake, vehicle)
        clearance = axle_clearance_squared(path, obstacles, vehicle.wheelbase)
        if np.min(clearance) >= minimum_clearance and clearance[-1] > target_clearance:
            return float(candidate)
    return None
