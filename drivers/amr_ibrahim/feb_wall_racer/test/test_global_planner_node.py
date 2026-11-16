import importlib
import numpy as np
import pytest
from feb_wall_racer.configuration import effective_parameters
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.core.track import build_track
from feb_wall_racer.core.raceline import optimize,speed_plan


def test_complete_plan_matches_existing_track_line_and_speed_kernels():
    plan=importlib.import_module('feb_wall_racer.core.planning').plan_track
    p=effective_parameters(DEFAULTS)
    theta=np.linspace(0,2*np.pi,200,endpoint=False)
    trail=np.c_[8*np.cos(theta),8*np.sin(theta)]
    widths=np.full(len(trail),2.2)
    track,_=build_track(trail,widths,p)
    expected_line=optimize(track,p)
    actual=plan(trail,widths,p)
    for field in ('left','right','centre','half_width'):
        np.testing.assert_allclose(actual['track'][field],track[field],atol=1e-8,rtol=0)
    np.testing.assert_allclose(actual['line'],expected_line,atol=1e-8,rtol=0)
    np.testing.assert_allclose(actual['speeds'],speed_plan(expected_line,p),atol=1e-8,rtol=0)
    assert len(actual['arc_length'])==len(actual['line'])+1


def test_invalid_closed_track_cannot_produce_a_plan():
    plan=importlib.import_module('feb_wall_racer.core.planning').plan_track
    with pytest.raises(ValueError,match='track validation'):
        plan(np.zeros((3,2)),np.full(3,2.2),effective_parameters(DEFAULTS))


def test_race_plan_falls_back_to_trail_when_inside_offset_hits_measured_wall():
    from feb_wall_racer.core.planning import plan_track
    p = effective_parameters(DEFAULTS)
    theta = np.linspace(0, 2*np.pi, 201)
    trail = np.c_[8*np.cos(theta), 8*np.sin(theta)]
    widths = np.full(len(trail), 2.2)
    track, _ = build_track(trail, widths, p)
    wall = track['centre'][0] + .34 * track['normal'][0]
    actual = plan_track(trail, widths, p, walls=[wall])
    np.testing.assert_allclose(actual['line'], track['centre'])


def test_measured_wall_on_trail_rejects_the_plan():
    from feb_wall_racer.core.planning import plan_track
    p = effective_parameters(DEFAULTS)
    theta = np.linspace(0, 2*np.pi, 201)
    trail = np.c_[8*np.cos(theta), 8*np.sin(theta)]
    with pytest.raises(ValueError, match='measured wall'):
        plan_track(trail, np.full(len(trail), 2.2), p, walls=[trail[0]])
