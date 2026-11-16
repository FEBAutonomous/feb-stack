"""Real installed eight-process graph on synthetic public inputs only."""
import os
import re
import signal
import time
import unittest
from pathlib import Path
import rclpy
from ament_index_python.packages import get_package_share_directory
import test_mission_launch as launch_fixture
from feb_wall_racer.configuration import NODE_NAMES


class ModularStackLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])
        cls.launch_file=Path(get_package_share_directory('feb_wall_racer'))/'launch/modular.launch.py'

    @classmethod
    def tearDownClass(cls):rclpy.shutdown()

    # Reuse only the unchanged public-input/process-ownership harness, not tests.
    setUp=launch_fixture.MissionLaunchTests.setUp
    tearDown=launch_fixture.MissionLaunchTests.tearDown
    _wait=launch_fixture.MissionLaunchTests._wait
    _publish_inputs=launch_fixture.MissionLaunchTests._publish_inputs
    _output=launch_fixture.MissionLaunchTests._output
    _parameter=launch_fixture.MissionLaunchTests._parameter

    def _launch(self,**kwargs):
        env=dict(kwargs.pop('env',{}))
        launch_fixture.MissionLaunchTests._launch(self,env=env,**kwargs)

    def _ready(self):
        expected={name+'_node' for name in NODE_NAMES}|{'mission'}
        self._wait(lambda:expected.issubset(self.probe.get_node_names()),timeout=45.)
        self._wait(lambda:self.commands and self.mission,timeout=20.)
        publishers=self.probe.get_publishers_info_by_topic('/drive')
        self.assertEqual([p.node_name for p in publishers],['mpc_controller_node'])
        ids=[self._parameter(name+'_node','configuration_id').string_value for name in NODE_NAMES]
        self.assertTrue(ids[0] and len(set(ids))==1)

    def test_graph_start_direct_stop_rearm_reset_and_critical_exit(self):
        self._launch();self._ready()
        self.assertTrue(all(m.drive.speed==0. for m in self.commands))
        self.state='AUTONOMOUS: launch test'
        self.commands.clear()
        self._wait(lambda:any(m.drive.speed>0. for m in self.commands),timeout=15.)
        self._wait(lambda:any('MAPPING:' in m.data for m in self.statuses))
        self.state='STOP: launch test'
        self._wait(lambda:self.mission and not self.mission[-1].data)
        self._wait(lambda:self.commands[-1].drive.speed==0.)
        self.commands.clear();self.state='AUTONOMOUS: stack is driving'
        self._wait(lambda:self.mission[-1].data)
        deadline=time.monotonic()+.6
        self._wait(lambda:time.monotonic()>=deadline)
        self.assertTrue(all(m.drive.speed==0. for m in self.commands))
        self.state='IDLE: released';self._wait(lambda:not self.mission[-1].data)
        self.state='AUTONOMOUS: autonomous button';self.commands.clear()
        self._wait(lambda:any(m.drive.speed>0. for m in self.commands),timeout=15.)
        pids=re.findall(r'\[perception_node-\d+\]: process started with pid \[(\d+)\]',self._output())
        self.assertEqual(len(pids),1,self._output())
        pid=int(pids[0])
        self.assertIn(b'/lib/feb_wall_racer/perception_node',Path('/proc/%d/cmdline'%pid).read_bytes())
        os.kill(pid,signal.SIGTERM)
        self._wait(lambda:self.process.poll() is not None,timeout=10.)
        self._wait(lambda:'mpc_controller_node' not in self.probe.get_node_names())

    def test_inspection_clock_and_viewer_exit_do_not_change_command_authority(self):
        self._launch(mode='inspection');self._ready()
        for name in NODE_NAMES:
            self.assertTrue(self._parameter(name+'_node','use_sim_time').bool_value)
            self.assertAlmostEqual(self._parameter(name+'_node','mode_speed_limit').double_value,.8)
        self.state='AUTONOMOUS: inspection test';self.commands.clear()
        self._wait(lambda:any(m.drive.speed>0. for m in self.commands),timeout=15.)
        self.assertTrue(all(m.drive.speed<=.8+1e-8 for m in self.commands))
        pid=int(re.findall(r'\[visualization_node-\d+\]: process started with pid \[(\d+)\]',self._output())[0])
        self.assertIn(b'/lib/feb_wall_racer/visualization_node',Path('/proc/%d/cmdline'%pid).read_bytes())
        os.kill(pid,signal.SIGTERM)
        self._wait(lambda:'visualization_node' not in self.probe.get_node_names())
        self.commands.clear();self._wait(lambda:any(m.drive.speed>0. for m in self.commands))
        self.assertIsNone(self.process.poll())

    def test_car_layers_reach_every_node_without_hardware(self):
        extra=Path(self.temp.name)/'override.yaml'
        extra.write_text('wall_racer:\n  ros__parameters:\n    wheelbase: 0.5\n')
        self._launch(on_car=True,env={'FEB_WALL_RACER_PARAMS':str(extra)},
            car_overlay='wall_racer:\n  ros__parameters:\n    wheelbase: 0.45\n    max_steer: 0.15\n')
        self._ready()
        for name in NODE_NAMES:
            self.assertFalse(self._parameter(name+'_node','use_sim_time').bool_value)
            self.assertAlmostEqual(self._parameter(name+'_node','wheelbase').double_value,.5)
            self.assertAlmostEqual(self._parameter(name+'_node','max_steer').double_value,.15)
        self.assertTrue(all(m.drive.speed==0. for m in self.commands))
