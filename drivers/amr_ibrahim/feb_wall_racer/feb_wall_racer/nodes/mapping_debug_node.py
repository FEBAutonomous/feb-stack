"""Read-only desktop view of the wall racer's mapping and clearance inputs."""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from feb_wall_racer_msgs.msg import LocalizedState, LocalReference, RacePlan, RunContext, TrackMap, WallObservation

from .. import topics
from ..core.perception import lidar_points_to_map
from ..core.safety import SWEEP_PADDING, axle_clearance_squared
from ..core.synchronization import CycleJoin, identity
from ..interfaces import points
from ..qos import LATCHED, QOS, SENSOR_QOS


def scan_geometry(walls, state, lidar_forward, wheelbase):
    scan = points(walls.points)
    local = scan + [lidar_forward, 0.]
    world = lidar_points_to_map(scan, state.pose, state.yaw, lidar_forward)
    distance = axle_clearance_squared(np.zeros((1, 3)), local, wheelbase)[0]
    return local, world, math.sqrt(distance)


def epoch(meta):
    return meta.run_id, meta.generation, meta.config_id


class MappingDebugNode(Node):
    def __init__(self):
        super().__init__('mapping_debug_node')
        self.lidar = self.declare_parameter('lidar_forward', .2733).value
        self.wheelbase = self.declare_parameter('wheelbase', .339).value
        self.half_width = self.declare_parameter('car_half_width', .135).value
        self.margin = self.half_width + self.declare_parameter('wall_clearance', .18).value + SWEEP_PADDING
        self.run_context = None
        self.reset()
        self.create_subscription(RunContext, topics.CONTEXT, self.on_context, LATCHED)
        self.create_subscription(WallObservation, topics.WALLS, lambda msg: self.join_input('walls', msg), SENSOR_QOS)
        self.create_subscription(LocalizedState, topics.STATE, lambda msg: self.join_input('state', msg), SENSOR_QOS)
        self.create_subscription(TrackMap, topics.MAP, lambda msg: self.cache('wall_map', msg), LATCHED)
        self.create_subscription(RacePlan, topics.PLAN, lambda msg: self.cache('plan', msg), LATCHED)
        self.create_subscription(LocalReference, topics.REFERENCE, lambda msg: self.cache('reference', msg), SENSOR_QOS)
        self.create_subscription(LocalReference, topics.PREDICTION, lambda msg: self.cache('prediction', msg), SENSOR_QOS)
        self.create_subscription(String, '/feb/status', self.on_status, QOS)

    def reset(self):
        self.join = CycleJoin(('walls', 'state'))
        self.sample = self.wall_map = self.plan = self.reference = self.prediction = None
        self.status = ''

    def on_context(self, msg):
        if self.run_context is not None and epoch(self.run_context.meta) != epoch(msg.meta):
            self.reset()
        self.run_context = msg

    def accepts(self, msg):
        # Latched data may arrive before the first context when attaching mid-run.
        return self.run_context is None or epoch(msg.meta) == epoch(self.run_context.meta)

    def cache(self, name, msg):
        if self.accepts(msg):
            setattr(self, name, msg)

    def join_input(self, kind, msg):
        if self.run_context is None or not self.accepts(msg):
            return
        try:
            for walls, state in self.join.add(kind, msg):
                if walls.valid and state.valid:
                    self.sample = walls, state, time.monotonic()
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=2.)

    def on_status(self, msg):
        self.status = msg.data


class Plot:
    def __init__(self, canvas, centre, span):
        self.canvas = canvas
        self.width, self.height = canvas.winfo_width(), canvas.winfo_height()
        self.scale = max(1., min(self.width, self.height) - 30) / span
        self.centre = np.asarray(centre)
        canvas.delete('all')

    def pixels(self, xy):
        return (xy - self.centre) * [self.scale, -self.scale] + [self.width / 2, self.height / 2]

    def dots(self, xy, color):
        stride = max(1, math.ceil(len(xy) / 1800))
        for x, y in self.pixels(xy[::stride]):
            self.canvas.create_rectangle(x-1, y-1, x+1, y+1, fill=color, width=0)

    def line(self, xy, color, width=2, **options):
        if len(xy) > 1:
            self.canvas.create_line(*self.pixels(xy).ravel(), fill=color, width=width, **options)

    def axes(self):
        origin = self.pixels(np.zeros(2))
        x, y = origin
        self.canvas.create_line(0, y, self.width, y, fill='#29374b')
        self.canvas.create_line(x, 0, x, self.height, fill='#29374b')
        self.canvas.create_text(self.width-10, y-10, text='+X', fill='#a4b5ca', anchor='e')
        self.canvas.create_text(x+10, 10, text='+Y', fill='#a4b5ca', anchor='nw')
        self.canvas.create_line(16, self.height-18, 16+self.scale, self.height-18, fill='#d1d9e4', width=2)
        self.canvas.create_text(16, self.height-23, text='1 m', fill='#d1d9e4', anchor='sw')


def run_window(tk, root, node):
    root.title('FEB Mapping Debug')
    root.geometry('1260x740')
    root.configure(bg='#111927')
    summary = tk.Label(root, text='Waiting for the wall racer…', bg='#111927', fg='white', anchor='w', justify='left')
    summary.pack(fill='x', padx=12, pady=8)
    tk.Label(root, text='Grey  SLAM map   •   Blue  driven trail   •   White  live scan   •   Green/red  left/right'
             '   •   Yellow  guidance or raceline   •   Purple dashed  race-plan boundaries   •   Cyan  MPC',
             bg='#111927', fg='#c3cedb', anchor='w').pack(fill='x', padx=12)
    panels = tk.Frame(root, bg='#111927')
    panels.pack(fill='both', expand=True, padx=8, pady=8)
    canvases = []
    for title in ('Estimated map  /  fixed frame map', 'Car view  /  rear axle origin, +X forward'):
        frame = tk.Frame(panels, bg='#111927')
        frame.pack(side='left', fill='both', expand=True, padx=4)
        tk.Label(frame, text=title, bg='#111927', fg='white').pack(anchor='w')
        canvas = tk.Canvas(frame, width=580, height=550, bg='#182334', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        canvases.append(canvas)

    def redraw():
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.)
        if not rclpy.ok():
            root.quit()
            return
        if node.sample is None:
            for canvas in canvases:
                canvas.delete('all')
            summary.configure(text=node.status or 'Waiting for a matching lidar and pose cycle…')
        else:
            draw_frame(canvases, summary, node)
        root.after(150, redraw)

    root.after(0, redraw)
    root.mainloop()


def draw_frame(canvases, summary, node):
    walls, state, received = node.sample
    local, world, distance = scan_geometry(walls, state, node.lidar, node.wheelbase)
    track = node.wall_map
    mapped = points(track.walls) if track and track.valid and epoch(track.meta) == epoch(state.meta) and track.meta.map_id == state.meta.map_id else np.empty((0, 2))
    trail = points(track.trail) if len(mapped) else np.empty((0, 2))
    if len(mapped):
        lo, hi = np.min(mapped, axis=0), np.max(mapped, axis=0)
        centre, span = (lo+hi)/2., max(4., float(np.max(hi-lo)) + 2.)
    else:
        centre, span = state.pose, 8.
    overview, close = Plot(canvases[0], centre, span), Plot(canvases[1], [1.5, 0.], 7.)
    overview.axes()
    close.axes()
    axle = np.array([[0., 0.], [node.wheelbase, 0.]])
    color = '#873f48' if distance <= node.margin else '#43576c'
    close.line(axle, color, 2*node.margin*close.scale, capstyle='round')
    close.line(axle, '#edaa55', 2*node.half_width*close.scale, capstyle='round')
    overview.dots(mapped, '#8695aa')
    overview.line(trail, '#4e9fff')
    overview.dots(world, '#e5edf8')
    close.dots(local, '#e5edf8')
    to_map = lambda xy: lidar_points_to_map(xy, state.pose, state.yaw, node.lidar)
    for side, color in (('left', '#48dda5'), ('right', '#ff797a')):
        classified = points(getattr(walls, side))
        overview.dots(to_map(classified), color)
        close.dots(classified + [node.lidar, 0.], color)
    plan = node.plan
    if plan and plan.valid and epoch(plan.meta) == epoch(state.meta) and plan.meta.map_id == state.meta.map_id:
        for side in ('left', 'right'):
            overview.line(points(getattr(plan, side)), '#bba1ed', dash=(4, 4))
        overview.line(points(plan.line), '#ffda69')
    reference = node.reference
    if reference and reference.valid and identity(reference.meta) == identity(state.meta) and reference.meta.header.frame_id == walls.meta.header.frame_id:
        guidance = points(reference.path)
        overview.line(to_map(guidance), '#ffda69')
        close.line(guidance + [node.lidar, 0.], '#ffda69')
    prediction = node.prediction
    if (prediction and prediction.valid and plan and plan.valid and epoch(prediction.meta) == epoch(state.meta)
            and prediction.meta.map_id == state.meta.map_id and prediction.meta.plan_id == plan.meta.plan_id
            and abs(prediction.meta.cycle-state.meta.cycle) <= 10):
        overview.line(points(prediction.path), '#56e2f2')
    close.dots(np.array([[node.lidar, 0.]]), 'white')
    world_axle = lidar_points_to_map(axle, state.pose, state.yaw, 0.)
    overview.line(world_axle, '#edaa55', 4, arrow='last')
    phase = node.run_context.phase if node.run_context else 'WAITING'
    age = time.monotonic()-received
    summary.configure(text=f'{phase}   cycle {state.meta.cycle}   map revision {state.meta.map_revision}   '
        f'pose {state.pose[0]:.2f}, {state.pose[1]:.2f} m   yaw {math.degrees(state.yaw):.1f}°   '
        f'{state.motion.body_speed:.2f} m/s   paired data received {age:.2f} s ago\n'
        f'Axle clearance {distance:.2f} m / {node.margin:.2f} m guard   •   front cone {walls.front_clearance:.2f} m   '
        f'•   corridor confidence {walls.confidence:.2f}   •   scan match {state.matches} returns, {state.residual:.2f} m residual\n'
        f'{node.status}')


def main(args=None):
    try:
        import tkinter as tk
    except ImportError:
        print('Mapping debug window requires Python Tk (python3-tk).')
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f'Mapping debug desktop unavailable: {exc}')
        return
    rclpy.init(args=args)
    node = MappingDebugNode()
    try:
        run_window(tk, root, node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
