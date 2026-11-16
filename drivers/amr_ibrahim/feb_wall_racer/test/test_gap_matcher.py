import gzip
import json
from pathlib import Path

import numpy as np

from feb_wall_racer.core.gap_matcher import match_gap
from feb_wall_racer.params import DEFAULTS


def scan_fixture(*, one_wall=False, occluded=False):
    angles = np.linspace(-2.35, 2.35, 1081)
    forward, lateral = np.cos(angles), np.sin(angles)
    ranges = np.full(len(angles), 7.)
    right = lateral < -.01
    ranges[right] = np.minimum(ranges[right], -1. / lateral[right])
    if not one_wall:
        left = lateral > .01
        ranges[left] = np.minimum(ranges[left], 1. / lateral[left])
    keep = np.abs(angles) > .35 if occluded else np.ones(len(angles), dtype=bool)
    return np.column_stack((ranges * forward, ranges * lateral))[keep]


def test_one_wall_still_has_a_reachable_gap():
    gap = match_gap(scan_fixture(one_wall=True), DEFAULTS, speed=0., previous=0.)
    assert gap is not None and gap.distance > 1.5
    assert abs(gap.steering) < .12


def test_unobserved_forward_opening_cannot_start_a_gap():
    assert match_gap(scan_fixture(occluded=True), DEFAULTS, speed=0., previous=0.) is None


def test_braking_requirement_filters_short_candidates():
    scan = scan_fixture()
    gap = match_gap(scan, DEFAULTS, speed=0., previous=0.)
    assert gap is not None
    assert match_gap(scan, DEFAULTS, speed=7., previous=0.) is None


def test_porto_corner_has_an_observed_footprint_even_when_the_buffer_is_occluded():
    with gzip.open(Path(__file__).parent / 'fixtures' / 'porto_gap_corner.json.gz', 'rt') as stream:
        scan = json.load(stream)
    gap = match_gap(scan['points'], DEFAULTS, speed=0., previous=0.)
    assert gap is not None and gap.distance >= .8 and gap.steering > .2


def test_short_verified_gap_allows_progress_when_there_is_room_to_brake():
    with gzip.open(Path(__file__).parent / 'fixtures' / 'loop_gap_corner.json.gz', 'rt') as stream:
        scan = json.load(stream)
    gap = match_gap(scan['points'], DEFAULTS, speed=0., previous=.052)
    assert gap is not None and .45 <= gap.distance <= .65
    assert match_gap(scan['points'], DEFAULTS, speed=2., previous=.052) is None


def test_reliable_paired_corridor_reclaims_steering_after_a_gap_turn():
    from feb_wall_racer.core.pathing import GuidancePlanner
    from feb_wall_racer.core.perception import WallObservation

    points = scan_fixture()
    centre = np.column_stack((np.linspace(.55, 3., 8), np.zeros(8)))
    empty = np.empty((0, 2))
    obs = WallObservation(points, empty, empty, centre, 2., .95, 2, 0., 7.)
    guidance = GuidancePlanner(DEFAULTS, use_gap_matcher=True)
    guidance.last = (1., 1, 1., .4, (0., 0.), centre)
    ref = guidance.mapping(obs, .6, 1.1, 2, 1.1)
    assert ref['reason'] == 'paired corridor'
    assert abs(ref['steering']) < .01


def test_gap_without_paired_corridor_cannot_continue_a_full_turn():
    from feb_wall_racer.core.pathing import GuidancePlanner
    from feb_wall_racer.core.perception import WallObservation

    empty = np.empty((0, 2))
    obs = WallObservation(scan_fixture(one_wall=True), empty, empty, empty, 2., 0., 1, 0., 7.)
    guidance = GuidancePlanner(DEFAULTS, use_gap_matcher=True)
    refs = [guidance.mapping(obs, .6, i, i, i, yaw=np.arctan2(np.sin(a), np.cos(a)))
            for i, a in enumerate(np.linspace(0., 5., 10), 1)]
    assert not refs[0]['emergency']
    assert refs[-1]['emergency']
