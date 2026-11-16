import math
import numpy as np
import pytest
from feb_wall_racer.core.slam import WallMap
from feb_wall_racer.params import DEFAULTS


def walls(corner=False):
    x = np.linspace(-7, 7, 400)
    result = np.vstack([np.column_stack([x, np.full_like(x,y)]) for y in [-1.1,1.1]])
    if corner:
        y = np.linspace(-1.1,1.1,100)
        result = np.vstack([result,np.column_stack([np.full_like(y,6),y])])
    return result


def matcher(points):
    m = WallMap(DEFAULTS)
    m.add(points,np.zeros(2),0,0,0)
    return m


def test_loop_closure_rejects_an_overlapping_scan_two_metres_before_the_start():
    m = matcher(walls())
    m.start_scan = walls()
    approach = m.start_scan + [2.2, 0.]
    assert not m.loop_match(approach, [-2.2, 0.], [0., 0.], 0., 0., 30., DEFAULTS)[0]
    assert m.loop_match(m.start_scan, [0., 0.], [0., 0.], 0., 0., 30., DEFAULTS)[0]


def test_straight_walls_correct_yaw_and_lateral_but_preserve_longitudinal_prediction():
    cloud=walls()
    m=matcher(cloud)
    r=m.align_pose(cloud,np.array([.24,.12]),math.radians(2),0)
    assert r.accepted
    assert r.rank == 2
    assert abs(r.pose[0]-.24)<.015
    assert abs(r.pose[1])<.025
    assert abs(r.yaw)<math.radians(.3)


def test_corner_recovers_translation_and_heading():
    cloud=walls(True)
    r=matcher(cloud).align_pose(cloud,np.array([.10,.08]),math.radians(1.5),0)
    assert r.accepted
    assert r.rank == 3
    assert np.linalg.norm(r.pose)<.035
    assert abs(r.yaw)<math.radians(.3)


def test_rotated_corridor_keeps_weak_direction():
    a=.7
    rot=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    cloud=walls()@rot.T
    tangent=rot[:,0]
    prediction=rot@np.array([.24,.10])
    r=matcher(cloud).align_pose(cloud,prediction,.025,0)
    assert r.accepted
    assert abs(np.dot(r.pose-prediction,tangent))<.02
    assert abs(np.dot(r.pose,rot[:,1]))<.025


def test_missing_overlap_does_not_move_prediction():
    m=matcher(walls())
    r=m.align_pose(walls(),np.array([0.,10.]),0,0)
    assert not r.accepted
    np.testing.assert_array_equal(r.pose,[0,10])
    assert r.yaw==0


def test_outliers_and_noise_do_not_dominate_alignment():
    rng=np.random.default_rng(42)
    cloud=walls(True)
    noisy=cloud+rng.normal(0,.008,cloud.shape)
    noisy=np.vstack([noisy,rng.uniform([-6,-4],[6,4],(100,2))])
    r=matcher(cloud).align_pose(noisy,np.array([.08,.08]),.02,0)
    assert r.accepted
    assert np.linalg.norm(r.pose)<.04
    assert abs(r.yaw)<.008


def test_mount_offset_is_rotated_and_map_is_not_modified_by_alignment():
    cloud=walls(True)
    m=WallMap(DEFAULTS)
    m.add(cloud,np.zeros(2),0,.27,0)
    before=m.points.copy()
    r=m.align_pose(cloud,np.array([.05,.05]),.02,.27)
    assert r.accepted
    assert np.linalg.norm(r.pose)<.025
    assert abs(r.yaw)<.005
    np.testing.assert_array_equal(before,m.points)


def test_excessive_correction_is_rejected_without_partial_update():
    cloud=walls(True)
    m=matcher(cloud)
    prior=np.array([0.,.3])
    r=m.align_pose(cloud,prior,0,0)
    assert not r.accepted
    np.testing.assert_array_equal(r.pose,prior)


def test_registration_configuration_rejects_unusable_bounds():
    from feb_wall_racer.configuration import effective_parameters
    for key,value in [('scan_match_min_eigen_ratio',1.),('scan_match_max_yaw',.5),
                      ('scan_match_max_translation',0.),('scan_match_max_residual',float('nan'))]:
        with pytest.raises(ValueError):
            effective_parameters(dict(DEFAULTS,**{key:value}))
