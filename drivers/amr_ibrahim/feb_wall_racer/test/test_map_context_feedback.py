import numpy as np
import pytest
from nav_msgs.msg import Odometry
from feb_wall_racer_msgs.msg import WallContext
from feb_wall_racer.interfaces import header,flat,ros_time
from feb_wall_racer.core.slam import WallMap
from test_perception_node import perception_node


def test_exported_vote_context_preserves_the_original_prior_score(perception_node):
    n,_=perception_node
    wall_map=WallMap(n.p)
    x=np.linspace(.5,5.,40)
    left=np.c_[x,np.full_like(x,1.1)]
    right=np.c_[x,np.full_like(x,-1.1)]
    for _ in range(3):
        wall_map.add(np.vstack((left,right)),np.zeros(2),0.,n.p['lidar_forward'],0.,
                     left_points=left,right_points=right)
    ctx=WallContext(meta=header('test',1,0,.95,'map',n.config_id),pose=[0.,0.],
                    yaw=0.,odom_position=[0.,0.],heading_correction=0.)
    for side in ('left','right'):
        entries=list(wall_map._side_voxels[side].values())
        setattr(ctx,side,flat([v[0] for v in entries]))
        setattr(ctx,side+'_votes',[v[1] for v in entries])
    odom=Odometry();odom.pose.pose.orientation.w=1.
    n.on_odom(odom)
    n.on_prior(ctx)
    actual=n.prior_for(1.)(left,right)
    expected=wall_map.score_side_labels(left,right,np.zeros(2),0.,n.p['lidar_forward'])
    assert actual==pytest.approx(expected,abs=1e-8)
    ctx.meta.header.stamp=ros_time(0.)
    n.on_prior(ctx)
    assert n.prior_for(1.) is None
