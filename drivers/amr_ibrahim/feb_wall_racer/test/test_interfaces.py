"""Typed transport must preserve geometry, identity and validity."""
import importlib
import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('feb_wall_racer.interfaces')
    except ImportError:
        pytest.fail('typed wall-racer transport has not been implemented')


def test_identity_rejects_previous_run_and_wrong_map():
    a=api()
    h=a.header('run-a',2,10,1.5,'laser','config')
    assert a.same_run(h,'run-a',2)
    assert not a.same_run(h,'run-a',3)
    assert not a.same_run(h,'run-b',2)


def test_empty_and_sparse_geometry_round_trip_without_shape_loss():
    a=api()
    for points in [np.empty((0,2)),np.array([[1.,2.]]),np.array([[1.,2.],[3.,4.]])]:
        np.testing.assert_array_equal(a.points(a.flat(points)),points)
    with pytest.raises(ValueError):
        a.points([1.,2.,3.])


def test_wall_observation_transport_preserves_all_fields():
    a=api()
    from feb_wall_racer.core.perception import WallObservation
    p=np.array([[1.,2.],[3.,4.]])
    obs=WallObservation(p,p,p,p,2.2,.8,7,.02,float('inf'),'sparse',.3,.4,9)
    msg=a.encode_observation(obs,a.header('run',1,4,2.,'laser','config'))
    result=a.decode_observation(msg)
    for name in vars(obs):
        if isinstance(getattr(obs,name),np.ndarray):
            np.testing.assert_array_equal(getattr(result,name),getattr(obs,name))
        else:
            assert getattr(result,name)==getattr(obs,name)


def test_nonfinite_geometry_and_odd_arrays_are_rejected():
    a=api()
    with pytest.raises(ValueError):
        a.points([1.,float('nan')])
    with pytest.raises(ValueError):
        a.flat([[float('inf'),0.]])


def test_malformed_plan_cannot_reach_control():
    from feb_wall_racer_msgs.msg import RacePlan
    a=api()
    assert hasattr(a,'validate_plan'),'consumer plan validation is missing'
    with pytest.raises(ValueError,match='plan'):
        a.validate_plan(RacePlan(valid=True))
