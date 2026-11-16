"""One coherent global track, optimized line and cyclic speed profile."""
import numpy as np
from scipy.spatial import cKDTree
from .track import build_track
from .raceline import optimize,speed_plan


def _clears_measured_walls(line, walls, margin):
    """Check every segment, including the closing seam, against lidar returns."""
    ends = np.roll(line, -1, axis=0)
    segments = ends - line
    lengths = np.linalg.norm(segments, axis=1)
    # Any return within margin of a segment lies inside this midpoint search.
    midpoints = (line + ends) / 2.
    neighbors = walls.query_ball_point(midpoints, margin + lengths / 2.)
    for start, segment, length, candidates in zip(line, segments, lengths, neighbors):
        if not candidates:
            continue
        offsets = walls.data[candidates] - start
        fraction = np.clip(offsets @ segment / max(length**2, 1e-12), 0., 1.)
        if np.any(np.linalg.norm(offsets - fraction[:, None] * segment, axis=1) < margin):
            return False
    return True


def plan_track(trail,widths,p,*,walls=None):
    track,why=build_track(trail,widths,p)
    if track is None:
        raise ValueError('closed lap failed track validation: '+why)
    line=optimize(track,p)
    if walls is not None:
        measured = np.asarray(walls, dtype=float).reshape(-1, 2)
        if not len(measured):
            raise ValueError('no measured walls for raceline clearance validation')
        wall_tree = cKDTree(measured)
        margin = p['car_half_width'] + p['wall_clearance']
        if not _clears_measured_walls(line, wall_tree, margin):
            line = track['centre'].copy()
            if not _clears_measured_walls(line, wall_tree, margin):
                raise ValueError('raceline and mapping trail violate measured wall clearance')
    speeds=speed_plan(line,p)
    if not np.all(np.isfinite(line)) or not np.all(np.isfinite(speeds)):
        raise ValueError('track planner produced non-finite raceline or speed profile')
    lengths=np.linalg.norm(np.roll(line,-1,axis=0)-line,axis=1)
    return dict(track=track,line=line,speeds=speeds,arc_length=np.r_[0.,np.cumsum(lengths)])
