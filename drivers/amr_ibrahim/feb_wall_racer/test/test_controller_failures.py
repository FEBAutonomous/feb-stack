"""Exercise existing recovery branches explicitly after moving the controller."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from feb_wall_racer.core.controller import RaceController
from feb_wall_racer.params import DEFAULTS


def controller(mpc):
    c=RaceController(dict(DEFAULTS),mpc)
    theta=np.linspace(0,2*np.pi,100,endpoint=False)
    c.set_raceline(np.c_[8*np.sin(theta),8*(1-np.cos(theta))],np.full(100,3.))
    return c


def test_three_plan_reuses_then_pursuit_and_solver_recovery():
    U=np.zeros((2,20));Z=np.zeros((5,21));Z[3]=2.;Z[4]=.123
    solver=SimpleNamespace(N=20,DT=.1,status='forced',warm=None)
    solver.solve=lambda *args:(U,Z)
    c=controller(solver)
    c.step(10.,0.,0.,0.,2.,0.,record_command=False)
    assert c.failures==0 and c.plan is not None
    solver.solve=lambda *args:None
    for i in (1,2,3):
        steer,_=c.step(10.+i*.1,0.,0.,0.,2.,0.,record_command=False)
        assert c.failures==i and c.info['Z'] is not None and steer==.123
    c.step(10.4,0.,0.,0.,2.,0.,record_command=False)
    assert c.failures==4 and c.info['status']=='pursuit'
    solver.solve=lambda *args:(U,Z)
    c.step(10.5,0.,0.,0.,2.,0.,record_command=False)
    assert c.failures==0 and c.info['Z'] is not None
    c.reset()
    assert c.plan is None and c.sent==[] and solver.warm is None


def test_first_failure_and_missing_solver_use_pursuit():
    for solver in (None,SimpleNamespace(N=20,DT=.1,status='forced',warm=None,solve=lambda *a:None)):
        c=controller(solver)
        _,throttle=c.step(10.,0.,0.,0.,2.,0.,record_command=False)
        assert c.info['status']=='pursuit' and throttle>0.


def test_solver_constructor_exception_does_not_prevent_node_startup():
    import rclpy
    from feb_wall_racer.nodes.mpc_controller_node import MPCControllerNode
    rclpy.init(args=[])
    n=None
    try:
        with patch('feb_wall_racer.nodes.mpc_controller_node.KinematicMPC',side_effect=RuntimeError('forced')):
            n=MPCControllerNode()
        assert n.mpc is None and not n.fault
    finally:
        if n:n.destroy_node()
        rclpy.shutdown()
