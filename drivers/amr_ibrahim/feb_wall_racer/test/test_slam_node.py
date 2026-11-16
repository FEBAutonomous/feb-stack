import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from feb_wall_racer.configuration import effective_parameters
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.perception import WallObservation


def engine():
    return importlib.import_module('feb_wall_racer.core.mapping').SlamEngine(effective_parameters(DEFAULTS))


def motion(x,correction=0.,travelled=0.,odom_stamp=(1,0)):
    return SimpleNamespace(odom_position=np.array([x,0.]),yaw=correction,
        heading_correction=correction,body_speed=1.,dt=.1,travelled=travelled,
        odom_stamp=SimpleNamespace(sec=odom_stamp[0],nanosec=odom_stamp[1]))


def empty():
    a=np.empty((0,2))
    return WallObservation(a,a,a,a,float('nan'),0.,0,float('inf'),float('inf'))


def test_dropped_snapshots_keep_displacement_and_mapping_distance():
    n=engine()
    n.process(motion(0.),empty(),1.)
    n.process(motion(2.,.2,2.),empty(),1.2)
    np.testing.assert_allclose(n.pose,[2*np.cos(.1),2*np.sin(.1)],atol=1e-8,rtol=0)
    assert n.travelled==2.
    assert len(n.slam.points)==0  # unusable corridor never inserts a map keyframe


def test_mapping_trail_keeps_positions_between_paired_wall_observations():
    n=engine()
    n.process(motion(0.),empty(),1.)
    n.path_points=[n.pose.copy()]
    n.path_widths=[2.2]
    for i in range(1,7):
        x=i*.4
        n.process(motion(x,travelled=x),empty(),1.+i*.1)
    assert max(np.linalg.norm(np.diff(n.path_points,axis=0),axis=1))<.5
    assert len(n.path_points)==len(n.path_widths)==7
    assert np.isnan(n.path_widths[1:]).all()
    assert len(n.slam.points)==0


def test_frozen_map_matches_only_once_per_fresh_odometry_stamp():
    n=engine()
    n.slam.frozen=True
    n.mode='RACING'
    predictions=[]

    def align(_points,pose,_yaw,_offset):
        predictions.append(pose.copy())
        return pose+[-.07,0.],20,.08

    n.slam.align=align
    assert n.process(motion(0.),empty(),1.)
    assert n.process(motion(0.),empty(),1.03)
    assert n.last_loc_t==1.
    np.testing.assert_allclose(n.pose,[-.07,0.])
    assert n.process(motion(.2,odom_stamp=(1,100_000_000)),empty(),1.1)
    np.testing.assert_allclose(predictions,[[0.,0.],[.13,0.]])
    np.testing.assert_allclose(n.pose,[.06,0.])
    assert n.last_loc_t==1.1


def test_duplicate_odometry_cannot_retry_a_rejected_match_or_extend_localization():
    n=engine()
    n.slam.frozen=True
    n.mode='RACING'
    attempts=[]

    def align(_points,pose,_yaw,_offset):
        attempts.append(pose.copy())
        return pose,0,float('inf')

    n.slam.align=align
    n.last_loc_t=1.
    assert not n.process(motion(0.),empty(),1.1)
    assert not n.process(motion(0.),empty(),1.2)
    assert not n.process(motion(0.),empty(),1.+n.p['stop_after']+.01)
    assert len(attempts)==1
    assert n.last_loc_t==1.
    assert n.stop_reason=='wall-map localization lost'


def test_closure_correction_freezes_map_and_preserves_trail_weighting():
    n=engine()
    n.path_points=[np.array([0.,0.]),np.array([1.,1.]),np.array([.5,.2])]
    n.path_widths=[2.2]*3
    n.pose=np.array([.5,.2]);n.start_pose=np.zeros(2);n.start_yaw=0.
    n.travelled=20.;n.last_scan_t=10.
    n._finish_mapping(np.array([.5,.2]))
    np.testing.assert_allclose(n.path_points[1],[.75,.9],atol=1e-8)
    np.testing.assert_allclose(n.pose,[0.,0.],atol=1e-8)
    assert n.slam.frozen and n.map_id==1 and n.mode=='FINALIZING'
    n.reset()
    assert not n.slam.frozen and n.map_id==0 and not n.path_points


def test_join_requires_exact_cycle_and_source_stamp_and_is_bounded():
    from feb_wall_racer.interfaces import header
    cls=importlib.import_module('feb_wall_racer.core.synchronization').CycleJoin
    join=cls(('walls','motion'),limit=3)
    a=SimpleNamespace(meta=header('r',1,4,2.,'laser','c'))
    b=SimpleNamespace(meta=header('r',1,4,2.,'map','c'))
    assert join.add('walls',a)==[]
    assert join.add('motion',b)==[(a,b)]
    assert join.add('motion',b)==[]
    for cycle in range(5,15):
        a=SimpleNamespace(meta=header('r',1,cycle,2.+cycle*.1,'laser','c'))
        join.add('walls',a)
    assert join.size<=3
    b=SimpleNamespace(meta=header('r',1,14,999.,'map','c'))
    with pytest.raises(ValueError,match='identity'):
        join.add('motion',b)
