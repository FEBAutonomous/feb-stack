"""Parameters of the racing half (raceline, speed plan, MPC), with the reason for each value.
Kept free of ROS so the offline bench reads the same defaults as the node."""
from .core.vehicle import ACT_DELAY, MAX_STEER

RACE = dict(
    # raceline
    sample_step=0.25,         # m between raceline points
    curv_window=0.75,         # m of raceline the signed curvature is averaged over before it limits the speed:
                              # enough to drop point-to-point noise, short enough to keep a corner's peak (over
                              # 2 m the loop's plan asked 5.5 m/s^2 of the tyres where it meant 4.5)
    car_half_width=0.135,
    # speed plan: what the tyres are asked for. Measured limits are 6.3 m/s^2 sideways and a
    # tyre force of 5.4 m/s^2 along the car that is there even when the tyre slides
    v_max=5.0,                # m/s. 5.9 and 7.0 were each run over all seventeen cone tracks: both clean, 7.0 is 3 to 12% faster
    a_lat=5.0,
    a_acc=4.6,
    a_brake=4.6,
    race_speed_scale=1.0,     # scales v_max; an organiser's knob for a slower field
    # mpc
    mpc_horizon=20,           # stages of mpc_dt: 1.2 s of plan
    mpc_dt=0.1,
    mpc_substeps=2,
    mpc_max_iter=100,
    mpc_a_acc=5.4,            # m/s^2 of tyre force the optimiser may ask for, driving and braking
    mpc_a_brake=5.4,
    mpc_a_lat=5.8,            # m/s^2 of cornering it may plan (soft)
    mpc_steer_rate=2.9,       # rad/s, nine tenths of the servo's
    max_steer=MAX_STEER,
    v_cap=12.0,               # m/s ceiling on the planned speed; lowered at startup to v_max plus a margin
    w_lat=12.0,               # error across the raceline
    w_lon=1.0,                # error along it
    w_head=4.0,
    w_speed=5.0,
    w_terminal=3.0,           # the last stage counts this many times
    w_force=0.05,             # tyre force against the reference's
    w_omega=0.3,              # steering rate
    w_dforce=0.001,           # how fast the force changes (jerk): at 0.004 the car braked too gently and
                              # ran 0.6 m/s over the plan into corners, at 0.001 it is 0.3 and as smooth
    w_domega=0.004,           # how fast the steering rate changes
    w_slack=50.0,
    act_delay=ACT_DELAY,
    throttle_min=0.02,        # never zero while racing: zero locks the wheels
    throttle_max=0.45,        # 11 m/s of wheel speed
    pursuit_lookahead=0.9,
)
