"""Typed transport conversions. Algorithms retain NumPy values, not ROS objects."""
import math
import numpy as np
from builtin_interfaces.msg import Time
from feb_wall_racer_msgs.msg import PipelineHeader, WallObservation as ObservationMsg
from .core.perception import WallObservation


def seconds(stamp):
    return stamp.sec+stamp.nanosec*1e-9


def ros_time(value):
    if not math.isfinite(value) or value<0:
        raise ValueError('source time must be finite and nonnegative')
    n=int(round(value*1e9))
    return Time(sec=n//1000000000,nanosec=n%1000000000)


def header(run_id,generation,cycle,stamp,frame,config_id):
    h=PipelineHeader(run_id=run_id,generation=generation,cycle=cycle,config_id=config_id)
    h.header.stamp=ros_time(stamp)
    h.header.frame_id=frame
    return h


def same_run(meta,run_id,generation):
    return meta.run_id==run_id and meta.generation==generation


def flat(value):
    a=np.asarray(value,dtype=float).reshape(-1,2)
    if not np.all(np.isfinite(a)):
        raise ValueError('geometry must be finite')
    return a.ravel().tolist()


def points(value):
    return np.asarray(flat(value),dtype=float).reshape(-1,2)


def encode_observation(obs,meta):
    msg=ObservationMsg(meta=meta,valid=True)
    for name in ('points','left','right','centre'):
        setattr(msg,name,flat(getattr(obs,name)))
    for name in ('width','confidence','segments','residual','front_clearance',
                 'diagnostics','heading','map_vote_score','map_vote_matches'):
        setattr(msg,name,getattr(obs,name))
    return msg


def decode_observation(msg):
    return WallObservation(**{name:(points(getattr(msg,name)) if name in
        ('points','left','right','centre') else getattr(msg,name)) for name in WallObservation.__dataclass_fields__})


def validate_plan(msg):
    """Validate the complete retained snapshot before either consumer loads it."""
    if not msg.valid or not msg.meta.map_id or not msg.meta.plan_id:
        raise ValueError('invalid plan identity or validity')
    line=points(msg.line)
    n=len(line)
    speed=np.asarray(msg.speeds,dtype=float)
    if n<3 or speed.shape!=(n,) or not np.all(np.isfinite(speed)) or np.any(speed<0):
        raise ValueError('invalid plan geometry/speed cardinality')
    for name in ('left','right','centre'):
        if len(points(getattr(msg,name)))!=n:
            raise ValueError('invalid plan boundary cardinality')
    half=np.asarray(msg.half_width,dtype=float)
    arc=np.asarray(msg.arc_length,dtype=float)
    if (half.shape!=(n,) or not np.all(np.isfinite(half)) or np.any(half<=0)
            or arc.shape!=(n+1,) or not np.all(np.isfinite(arc))
            or abs(arc[0])>1e-8 or np.any(np.diff(arc)<=0)
            or not math.isfinite(msg.length) or msg.length<=0):
        raise ValueError('invalid plan widths or arc-length basis')
    expected=np.r_[0.,np.cumsum(np.linalg.norm(np.roll(line,-1,axis=0)-line,axis=1))]
    if not np.allclose(arc,expected,atol=1e-8,rtol=0):
        raise ValueError('plan arc-length basis does not match geometry')
    return line,speed
