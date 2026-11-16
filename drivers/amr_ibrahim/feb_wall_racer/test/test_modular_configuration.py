from pathlib import Path
import tempfile
import pytest
from feb_wall_racer import configuration as config


def test_legacy_selector_layers_are_loaded_once_for_all_node_consumers():
    assert hasattr(config,'load_driver_parameters'),'canonical modular configuration loader is missing'
    with tempfile.TemporaryDirectory() as folder:
        base,car,extra=[Path(folder)/name for name in ('base.yaml','car.yaml','extra.yaml')]
        base.write_text('wall_racer:\n  ros__parameters:\n    wheelbase: 0.4\n')
        car.write_text('wall_racer:\n  ros__parameters:\n    wheelbase: 0.45\n')
        extra.write_text('/wall_racer:\n  ros__parameters:\n    lidar_forward: 0.3\n')
        p=config.load_driver_parameters(base,car,extra,True,.8)
        assert p['wheelbase']==.45 and p['lidar_forward']==.3
        assert p['map_speed']<=.8 and p['v_max']<=.8
        for node in config.NODE_NAMES:
            assert config.node_parameters(p,node)['wheelbase']==.45


def test_node_settings_reject_unknown_nodes_and_nonpositive_deadlines():
    assert hasattr(config,'load_node_settings'),'node runtime settings loader is missing'
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'nodes.yaml'
        for text in ('typo: {}\n','run_manager: {reset_timeout: 0.0}\n',
                     'mpc_controller: {worker_timeout: true}\n'):
            path.write_text(text)
            with pytest.raises(ValueError):config.load_node_settings(path)


def test_eight_node_executables_and_required_interfaces_are_packaged():
    import ast
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'setup.py').read_text())
    text=(root/'setup.py').read_text()
    for node in config.NODE_NAMES:
        assert node+'_node = feb_wall_racer.nodes.'+node+'_node:main' in text
    assert (root/'launch/modular.launch.py').is_file()
    assert 'feb_wall_racer_msgs' in (root/'package.xml').read_text()


def test_modular_graph_is_the_only_driver_runtime():
    root=Path(__file__).resolve().parents[1]
    package=root/'feb_wall_racer'
    setup=(root/'setup.py').read_text()
    launch=(package/'launching.py').read_text()
    assert 'wall_racer = feb_wall_racer.racer:main' not in setup
    assert not (package/'racer.py').exists()
    assert 'implementation' not in launch
    for name in ('controller','heading','mpc','perception','pathing',
                 'raceline','slam','track','vehicle','bench','cone_bench','controller_raceline'):
        assert not (package/(name+'.py')).exists()


def test_benchmark_code_is_separate_from_runtime_defaults():
    from feb_wall_racer.control_params import RACE
    from feb_wall_racer.benchmarks.controller import BENCH_DEFAULTS
    for name in ('margin','curvature_reg','corner_scale'):
        assert name not in RACE and name in BENCH_DEFAULTS
