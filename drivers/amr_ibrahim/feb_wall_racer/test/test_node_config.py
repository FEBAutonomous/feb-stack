import importlib
import pytest
from feb_wall_racer.params import DEFAULTS
from feb_wall_racer.configuration import effective_parameters


def registry():
    m=importlib.import_module('feb_wall_racer.configuration')
    assert hasattr(m,'node_parameters'),'node parameter ownership is not implemented'
    return m


def test_all_tuning_has_an_explicit_node_consumer():
    m=registry()
    p=effective_parameters(DEFAULTS)
    coverage=set().union(*(m.node_parameters(p,n) for n in m.NODE_NAMES))
    assert coverage==set(DEFAULTS)


def test_vehicle_calibration_and_mode_limit_are_consistent_across_nodes():
    m=registry()
    p=effective_parameters(dict(DEFAULTS,wheelbase=.45,mode_speed_limit=.8))
    for node in m.NODE_NAMES:
        selected=m.node_parameters(p,node)
        assert selected['wheelbase']==.45
        assert selected['mode_speed_limit']==.8
    assert m.configuration_id(p)==m.configuration_id(dict(reversed(list(p.items()))))
    assert m.configuration_id(p)!=m.configuration_id(effective_parameters(DEFAULTS))


def test_unknown_node_fails_instead_of_silently_getting_defaults():
    m=registry()
    with pytest.raises(ValueError):
        m.node_parameters(DEFAULTS,'typo')
