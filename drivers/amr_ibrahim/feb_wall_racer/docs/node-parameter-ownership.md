# Node parameter ownership

`config/driver.yaml` remains the canonical physical/algorithm tuning file. Its
existing `wall_racer`, `/wall_racer` and `/**` selectors are normalized once at
launch, followed by the car-only overlay and the explicit override. Card mode
caps and `FEB_CAR` determine speed limits and clock selection. The physical values
in the supplied driver YAML have not changed.

Every process receives its declared consumers from `configuration.node_parameters`.
Shared calibration/time/geometry is intentional. `configuration._OWNED` enumerates
algorithm ownership and fails if a DEFAULTS key gains no owner:

- Perception: range filtering, line/corridor fitting and map-vote decoding. Voxel
  size and association gate are shared with SLAM so exported votes retain meaning.
- Motion: vehicle dynamics, held odometry/IMU inputs, heading and scan-driven observer.
- SLAM: registration, voxel map, keyframes, trail, closure and localization.
- Global planning: track validation, raceline and cyclic speed profile.
- Local planning: mapping pursuit geometry, clearance/curvature speed limits and
  degraded guidance. Corridor bin size is shared with perception.
- Controller: MPC/pursuit, tyre-force conversion, delay/history and physical clipping.
  Obstacle limits and degraded lifetime are checked independently at this boundary.
- Manager: lifecycle, health, reset/plan barriers and public status.
- Visualization: frames, rendering, sampling and bounded trajectory.

`config/nodes.yaml` holds process coordination deadlines. An optional
`FEB_WALL_RACER_NODES` file overlays these settings; unknown nodes/settings and
nonpositive deadlines fail launch. These are startup-only ROS parameters, just
like car tuning. All nodes receive the same hash of the complete effective car configuration and
node settings. A mismatch cannot join an active run.

The launch passes effective capped values to ROS parameter services. Runtime
parameter changes are rejected. Restart the driver after editing YAML.
