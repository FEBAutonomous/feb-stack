"""Installed launch wiring shared by the stable and modular entry points."""
import os
from pathlib import Path
from ament_index_python.packages import PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,SetEnvironmentVariable,Shutdown,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from .configuration import (mode_speed_limit,load_driver_parameters,
    load_node_settings,node_parameters,pipeline_configuration_id,NODE_NAMES)


def driver_launch(share,locate):
    share=Path(share)
    try:locate('feb_mission')
    except PackageNotFoundError as exc:
        raise RuntimeError('feb_wall_racer requires feb_mission; build and source the workspace '
                           'including the existing feb_mission package') from exc
    on_car=os.environ.get('FEB_CAR')=='1'
    base,car=share/'config/driver.yaml',share/'config/driver.car.yaml'
    if not base.is_file():raise RuntimeError('installed wall racer parameters are missing: '+str(base))
    extra=os.environ.get('FEB_WALL_RACER_PARAMS',os.environ.get('FEB_RACER_PARAMS',''))
    pydeps=Path(os.environ.get('FEB_PYDEPS',share.parents[3]/'.pydeps'))
    python_path=os.environ.get('PYTHONPATH','')
    if pydeps.is_dir():python_path=str(pydeps)+(os.pathsep+python_path if python_path else '')

    def nodes(context):
        mode=LaunchConfiguration('mode').perform(context)
        cap=mode_speed_limit(share/'driver.yaml',mode)
        actions=[SetEnvironmentVariable('PYTHONPATH',python_path),
            Node(package='feb_mission',executable='mission',name='mission',
                 parameters=[{'use_sim_time':not on_car}],output='screen',emulate_tty=True,
                 on_exit=[Shutdown(reason='mission exited; stopping the wall driver')])]
        try:locate('feb_wall_racer_msgs')
        except PackageNotFoundError as exc:
            raise RuntimeError('feb_wall_racer requires generated feb_wall_racer_msgs; build both member packages') from exc
        p=load_driver_parameters(base,car,extra,on_car,cap)
        node_files=[share/'config/nodes.yaml']
        if os.environ.get('FEB_WALL_RACER_NODES'):
            node_files.append(os.environ['FEB_WALL_RACER_NODES'])
        settings=load_node_settings(node_files)
        config_id=pipeline_configuration_id(p,settings)
        for name in NODE_NAMES:
            parameters=node_parameters(p,name)
            parameters.update(settings[name])
            parameters.update(configuration_id=config_id,use_sim_time=not on_car)
            options=dict(package='feb_wall_racer',executable=name+'_node',name=name+'_node',
                parameters=[parameters],output='screen',emulate_tty=True)
            if name!='visualization':
                options['on_exit']=[Shutdown(reason=name+' exited; stopping the wall driver')]
            actions.append(Node(**options))
        debug = LaunchConfiguration('mapping_debug').perform(context)
        display = os.environ.get('DISPLAY', '')
        display_socket = display.split('.')[0].removeprefix(':')
        desktop = display.startswith(':') and Path('/tmp/.X11-unix', f'X{display_socket}').exists()
        if not on_car and (debug == 'true' or (debug == 'auto' and desktop)):
            geometry = {name: p[name] for name in ('lidar_forward', 'wheelbase', 'car_half_width', 'wall_clearance')}
            actions.append(Node(package='feb_wall_racer', executable='mapping_debug_node',
                                parameters=[geometry], output='screen'))
        return actions
    return LaunchDescription([
        DeclareLaunchArgument('mode',default_value='trackdrive'),
        DeclareLaunchArgument('mapping_debug', default_value='auto', choices=['auto', 'true', 'false']),
        OpaqueFunction(function=nodes)])
