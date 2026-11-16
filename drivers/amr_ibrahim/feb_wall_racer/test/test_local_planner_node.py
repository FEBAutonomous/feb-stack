import importlib
import numpy as np
import pytest
from feb_wall_racer.configuration import effective_parameters
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.perception import observe,WallObservation
from feb_wall_racer.benchmarks.wall import synthetic_corridor_scan
from feb_wall_racer.core.pathing import mapping_steering,path_clearance_speed,mapping_emergency_stop


def planner():
    cls=importlib.import_module('feb_wall_racer.core.pathing').GuidancePlanner
    return cls(effective_parameters(DEFAULTS))


def test_normal_guidance_preserves_the_selected_geometry_and_speed_envelope():
    n=planner();obs=observe(synthetic_corridor_scan(),n.p)
    steer,curvature,target=mapping_steering(obs.centre,n.p,obs.heading)
    safety,collision,_=path_clearance_speed(obs.points,curvature,n.p)
    expected=min(n.p['map_speed'],np.sqrt(n.p['map_a_lat']/max(abs(curvature),1e-6)),safety,1.4)
    ref=n.mapping(obs,1.,10.,1,1.)
    assert ref['steering']==steer and ref['curvature']==curvature
    np.testing.assert_allclose(ref['target'],target,atol=1e-8,rtol=0)
    assert ref['target_speed']==pytest.approx(expected,abs=1e-8)
    assert ref['emergency']==mapping_emergency_stop(1.,collision,n.p)


def test_degraded_guidance_cannot_renew_the_original_age():
    n=planner();obs=observe(synthetic_corridor_scan(),n.p)
    normal=n.mapping(obs,1.,10.,1,1.)
    a=np.empty((0,2))
    lost=WallObservation(a,a,a,a,float('nan'),0.,0,float('inf'),float('inf'))
    for i,now in enumerate((10.1,10.3,10.49)):
        ref=n.mapping(lost,1.,now,i+2,1.+now-10.)
        assert ref['guidance_cycle']==1
        assert ref['guidance_age']==pytest.approx(now-10.)
        assert ref['target_speed']<=.6
        assert ref['steering']==normal['steering']
    assert n.mapping(lost,1.,10.51,8,1.51)['emergency']


def test_finalizing_uses_fresh_mapping_guidance_without_a_new_hard_stop():
    n=planner();obs=observe(synthetic_corridor_scan(),n.p)
    ref=n.mapping(obs,1.,10.,1,1.)
    assert ref['valid'] and not ref['emergency'] and ref['target_speed']>0.
