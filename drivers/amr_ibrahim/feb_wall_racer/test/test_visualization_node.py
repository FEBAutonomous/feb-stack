import importlib
import math
import numpy as np
from builtin_interfaces.msg import Time


def test_display_converters_keep_source_frame_stamp_and_marker_geometry():
    m=importlib.import_module('feb_wall_racer.core.visualization')
    stamp=Time(sec=123,nanosec=456)
    path=m.path_message([[1.,2.]],stamp,'laser')
    assert path.header.stamp==stamp and path.header.frame_id=='laser'
    assert path.poses[0].pose.position.x==1.
    marker=m.car_marker([3.,4.],math.pi/2,stamp)
    assert marker.pose.position.x==3.
    assert abs(marker.pose.position.y-4.16)<1e-8
    assert (marker.scale.x,marker.scale.y,marker.scale.z)==(.42,.24,.12)


def test_display_trajectory_has_the_original_bound_and_sampling_rule():
    cls=importlib.import_module('feb_wall_racer.core.visualization').Trajectory
    t=cls(.15)
    for i in range(2005):t.append([i*.2,0.])
    assert len(t.points)==2000
    np.testing.assert_allclose(t.points[0],[1.,0.],atol=1e-8)
    t.append([400.81,0.])
    assert len(t.points)==2000 and t.points[-1][0]==400.8


def test_restarted_viewer_accepts_racing_context_and_owns_no_drive_or_tf():
    import rclpy,time
    from feb_wall_racer.nodes.visualization_node import VisualizationNode
    from feb_wall_racer_msgs.msg import RunContext
    from feb_wall_racer.interfaces import header
    rclpy.init(args=[]);n=VisualizationNode()
    try:
        meta=header('running',2,0,1.,'map',n.config_id)
        n.on_context(RunContext(meta=meta,phase='RACING',live=True))
        assert n.ready and n.accept(meta)
        deadline=time.monotonic()+1.
        names=[]
        while not names and time.monotonic()<deadline:
            names=[name for name,_ in n.get_publisher_names_and_types_by_node(n.get_name(),n.get_namespace())]
            rclpy.spin_once(n,timeout_sec=.01)
        assert '/feb/pose' in names and '/drive' not in names and '/tf' not in names
    finally:
        n.destroy_node();rclpy.shutdown()
