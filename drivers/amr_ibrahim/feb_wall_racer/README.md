# FEB wall racer

`feb_wall_racer` is a lidar-based racing stack for closed wall corridors. ROS owners
live in `feb_wall_racer/nodes/`; reusable algorithms live in `feb_wall_racer/core/`. Mission
integration and its checks are described in
[`docs/2026-10-30-mission-integration.md`](docs/2026-10-30-mission-integration.md).
See [`PLAN.md`](./PLAN.md) for the original stack plan and simulator validation gates.

The driver-card launch uses the eight-node modular graph as the sole runtime.
The historical node-split and parity plans describe how that graph was extracted;
they are no longer instructions to retain or launch the monolithic driver.

The modular graph contains perception, motion estimation, SLAM, global planning,
local planning, MPC control, run management and visualization. MPC alone publishes
`/drive`; SLAM alone publishes the estimated `map -> feb_wall_racer_base` transform.
The stock supervisor and existing mission remain responsible for their original
roles. There is no separate command-gate node. Both member packages,
`feb_wall_racer` and `feb_wall_racer_msgs`, must be built.

Use `ros2 launch feb_wall_racer drive.launch.py` for the standard driver-card
entry point or `ros2 launch feb_wall_racer modular.launch.py` directly. Parameter
ownership and process deadlines are described in
[`docs/node-parameter-ownership.md`](docs/node-parameter-ownership.md).
The final source layout and publisher/subscriber connections are in
[`docs/organization.md`](docs/organization.md).

Configuration and the actual mapper/localizer are documented in
[`docs/configuration-and-localization.md`](docs/configuration-and-localization.md).
All supported startup tuning is explicit in `config/driver.yaml`. The car-only
`config/driver.car.yaml` loads on top with `FEB_CAR=1` and is currently empty:
no different real-car calibration values were supplied. An optional
`FEB_WALL_RACER_PARAMS` file (fallback `FEB_RACER_PARAMS`) overrides those files.
Edit YAML and restart; live parameter edits are rejected rather than silently
leaving cached control values unchanged. Unknown/misspelled settings fail early.
Mode caps from the package-root driver card bound both planning and published
commands. Startup logs list effective values whenever another limit clips them.

## Run

Run the registered wall driver through the current FEB simulator box from the repository root:

```sh
./feb up --driver feb_wall_racer --track loop
```

The root [driver.yaml](./driver.yaml) registers this ROS package as a `walls` driver, selects
`drive.launch.py`, and defines the `trackdrive` (5.0 m/s) and `inspection` (0.8 m/s) supervisor
caps. To test under the member-guide dead-man rules, run
`./feb up --deadman --driver feb_wall_racer --mode-name inspection --track loop` and hold SPACE
to drive. `./feb practice --driver feb_wall_racer` runs the qualifying wall-track practice set.
The package launch reads `config/driver.yaml`, overlays `config/driver.car.yaml` only when
`FEB_CAR=1`; that car file is required in car mode. It launches the eight stack nodes
and the existing `feb_mission` node with the same clock setting. A missing mission
package fails launch. A required node or mission exit stops the driver;
visualization is noncritical. The house supervisor retains its command timeout and stop behavior.

If a build reports that the removed `launch/race.launch.py` is missing, or launch warns that the
installed `config/driver.yaml` is missing, remove only this package's stale build/install outputs
and rebuild it from the current source inside the FEB box:

```sh
./feb shell
cd /home/feb/FEB/feb-racing/stack
rm -rf build/feb_wall_racer install/feb_wall_racer
source /opt/ros/humble/setup.bash
colcon build --packages-up-to feb_wall_racer --symlink-install
source install/setup.bash
test -f install/feb_wall_racer/share/feb_wall_racer/config/driver.yaml
ros2 pkg prefix feb_wall_racer
```

Only retry `./feb up --driver feb_wall_racer` after the package build succeeds and the parameter
file check passes. The current `feb up` build wrapper can continue launching house nodes after a
colcon failure, so the later launch log alone does not confirm that the package rebuilt.

The FEB workspace exports `.pydeps` through `FEB_PYDEPS`; the launch adds that directory to
`PYTHONPATH` when present. CasADi is optional and the racer falls back to pure pursuit when it
is unavailable.

The driver card disables the simulator adapter's PI trim and uses fast throttle slew, because
this racer already computes tyre slip and requested wheel speed. Preserve that `adapter:` block
when copying the driver; applying a second speed loop changes its requested braking force.

The supervisor and platform launch helpers remain stock. Fault-restart protection lives in
this driver: a local safety stop or supervisor `STOP` latches the racer, suppresses idle
readiness commands, and prevents automatic supervisor re-entry from resetting its map.
The scan watchdog is also checked when authorization is revoked, covering a stock command
timeout that reaches `IDLE` just before the driver's next watchdog tick. A latched watchdog
does not keep publishing zero commands that would refresh automatic readiness.

After a fault, end the run and start `feb up` again. An explicit operator restart also works:
release autonomous, wait for supervisor recovery/`IDLE` and healthy sensors, then press the
autonomous button again. The driver recognizes the stock `AUTONOMOUS: autonomous button`
state on a fresh edge; a held button, automatic `stack is driving` state, or mission toggle
cannot clear the latch. Do not source the archived `.feb-house.disabled-*` overlay.

The racer waits for both `/feb/state` to report `AUTONOMOUS` and `/feb/mission/live` to be true.
While waiting, it publishes zero-speed `/drive` heartbeats so the simulator supervisor can
detect the driver and enter autonomous mode. Once both signals authorize a new supervisor
autonomous session, it resets mapping, localization, and controller memory once before driving.
Repeated state/live messages preserve that memory; a delayed mission false/true pair within
the same supervisor session pauses/resumes without another reset. A direct supervisor stop immediately commands zero without waiting for
the mission heartbeat, and revokes the old mission authorization. Mission false also stops.
A latched safety stop requires a fresh explicit operator autonomous session or a new driver
process before reset; mission heartbeats or a mission false/true toggle alone cannot clear it.
It starts in `MAPPING`, follows the
local lidar corridor conservatively, and builds a bounded wall-point map from `/odom` motion and
wall observations. It enters `RACING` only after scan overlap indicates a lap closure and the
measured corridor passes track/raceline checks. If local width evidence is insufficient during
mapping, it may briefly reuse recent steering at a reduced speed, then stops rather than
guessing a corridor. If a detected loop fails global track validation, it stops rather than
freezing an invalid map.

## Simulator interface

- `/scan` (`sensor_msgs/LaserScan`): 1080 beams over 270 degrees; sensor frame is normally `laser`.
- `/odom` (`nav_msgs/Odometry`): public dead-reckoned vehicle pose and speed.
- `/imu` (`sensor_msgs/Imu`): gyro input used by the speed estimate.
- `/feb/state` (`std_msgs/String`): the racer drives only while the supervisor reports
  `AUTONOMOUS`.
- `/feb/mission/live` (`std_msgs/Bool`): published by `feb_mission`; confirms mission start
  and stops the racer when false. It supplements the direct supervisor stop path.
- `/drive` (`ackermann_msgs/AckermannDriveStamped`): requested wheel speed in m/s and steering
  angle in radians. The supervisor applies its mode speed cap; steering is bounded to +/-0.5236
  rad.

Sensor subscriptions use best-effort QoS for compatibility with sensor publishers. The racer
stops if odometry, IMU, or lidar becomes stale. It does not use simulator-private
`/autodrive/...` topics.

The standard launch uses `/feb/state` for both nodes. If you customize the racer's `state_topic`,
also remap `/feb/state` to that topic on the mission node in `drive.launch.py`; otherwise the
mission gate will correctly remain closed. Keep the mission and racer on the same ROS domain.

During `MAPPING`, perception searches local travel-axis orientations and pairs left/right wall
returns in forward-distance bins along each candidate axis. After at least three center samples
and adequate confidence, those wall labels are accumulated in separate left/right map vote grids
(at most one vote per voxel per scan). Repeated votes can prefer a candidate orientation consistent
with the walls mapped earlier; before enough repeat evidence exists, current scan geometry alone
chooses the corridor. Per-scan classification still requires a plausible width and paired wall
support.

The mapping follower uses ordered center samples for pure-pursuit steering, accounting for the
lidar offset and preserving sample order around corners. A missing path bin ends the usable path
rather than connecting across an unobserved gap. It also checks lidar points against the
short-horizon path predicted from the requested steering curvature. If that path approaches a
wall inside the configured car-width and clearance margin, it lowers the speed target according
to available braking distance; when recent wall evidence indicates a collision is too close to
brake for, it commands zero throttle. If a corridor is briefly lost, it can reuse the last
steering command for at most 0.5 s at a 0.6 m/s cap, subject to the same path-clearance check;
without recent valid guidance, it stops. Mapping speed is also reduced for high estimated
curvature. Set `debug` to log the selected corridor axis, map-vote evidence, lookahead target,
curvature, steering, clearance, collision distance, and safety speed cap.

Routine overspeed and fallback slowing use the bounded tyre-force controller rather than
switching to zero wheel demand at a small speed error. Hard stops remain for missing/expired
guidance, a zero safety cap, and a wall inside reaction distance plus the current braking
distance. Heading hints remain radians at the perception API and are converted to degrees
only for the candidate-axis search.

Racing uses the newest sensor samples (queue depth one) and skips mapping-only corridor-axis
search while retaining fitted wall returns and the front-stop check. The controller predicts
from the odometry source timestamp through recorded published commands to the expected
actuation time. Prediction/history use ROS clock seconds throughout, including when the
simulator runs slower than real time or pauses. Source time can remain fixed within a scan
callback in the single-threaded executor; shorter racing processing reduces that remaining
callback/solver delay rather than assuming wall seconds equal simulated seconds.

Valid public IMU orientation supplies heading after calibration into the initial odometry/map
frame. Heading samples are interpolated at the odometry timestamp; extrapolation is limited
to 0.15 s. Invalid/unavailable orientation, stale samples or a large orientation-frame jump
retain the last correction and fall back to odometry without rotating the map abruptly.
Odometry displacements receive the same heading correction. A new run clears calibration
and sample history. Gyro-only IMUs remain usable through that fallback.

The racing speed profile uses an absolute-curvature peak envelope, followed by cyclic
acceleration/braking passes. Opposite signed bends cannot cancel, and no lower-speed floor
overrides a tight corner's limit. Path geometry and MPC gains are unchanged; this can slow
corners substantially on a noisy line. It does not make an unreachable geometric turn
drivable or add the separate racing swept-path clearance guard.

After entering `RACING`, steering and planned speed come from the frozen raceline. Lidar wall
returns are still used for map localization and the forward obstacle stop check, but local
corridor center estimates do not gate or steer racing. The speed plan currently uses a 5 m/s
maximum, lateral/longitudinal acceleration settings,
MPC limits, and throttle limit. Actual requested speed still follows the curvature-limited
raceline profile; degraded localization scales speed down and eventually stops the car.

Parameters are in `config/driver.yaml`; overrides can be supplied through
`FEB_WALL_RACER_PARAMS` (or the existing `FEB_RACER_PARAMS` compatibility variable).

## Debug topics

- `/feb/status`: mode, speed, local fit confidence, scan-map match count/residual, fallback state.
- `/feb/pose`: estimated rear-axle pose in the node's `map` frame.
- `/feb/wall/left`, `/feb/wall/right`: scan-local fitted wall paths in the scan's header frame
  (normally `laser`).
- `/feb/wall/live/left`, `/feb/wall/live/right`: current fitted walls transformed into the
  estimated `map` frame, available during mapping and racing.
- `/feb/wall/map/left`, `/feb/wall/map/right`: latched mapped corridor boundaries after track build.
- `/feb/raceline`: latched planned line after track build.
- `/feb/trajectory`: bounded recent trail of the estimated car position in `map`.
- `/feb/car_marker`: car-shaped cube marker at the estimated pose in `map`.
- `/feb/mpc_prediction`: latest MPC prediction when the optional solver returns one.

For one view of the racer outputs, set the Foxglove 3D panel's fixed frame to `map`. These
estimated visualization topics intentionally do not use the simulator's separate ground-truth
`world` TF tree; do not add an identity `map`-to-`world` transform.
The scan-local `/feb/wall/left` and `/feb/wall/right` topics use `laser`; use the
`/feb/wall/live/*` topics in the `map` view instead.
The wall racer publishes `map` → `feb_wall_racer_base` from its estimated pose from the first
usable sensor cycle, so `map` is available as a TF root before global track closure.

The race-time node uses public `/odom` and `/imu` state, but does not subscribe to simulator
truth, simulator-private topics, lap counters, or collision counters.

## Offline checks

Run the ROS-independent smoke bench in an environment with NumPy available:

```sh
ros2 run feb_wall_racer wall_bench
```

The unit tests in `test/test_wall_stack.py` cover synthetic corridors, invalid scans, closed-track
raceline clearance, and bounded/frozen wall-map behavior. These checks do not replace the FEB box
build, live scan inspection, or scored simulator runs.
# Live mapping debug window

Starting the wall racer in the simulator's GUI devkit opens **FEB Mapping
Debug** on the devkit desktop. View it at
[the devkit desktop](http://localhost:6080/vnc.html?autoconnect=1&resize=scale)
or open the desktop from a second terminal with `./feb shell --gui`. The
window can be closed while the sim is running. For a direct ROS launch, set
`mapping_debug:=false` to disable it or `mapping_debug:=true` to use your
current display. The viewer uses Python Tk and is disabled on the car.

The left pane overlays the SLAM wall points, driven trail and same-cycle
lidar returns in the estimated `map` frame. The purple dashed boundaries are
the same race-plan boundaries that Foxglove displays on
`/feb/wall/map/left` and `/feb/wall/map/right`. They appear after mapping
and are reconstructed from the driven trail and corridor widths.

The right pane uses the rear axle as its origin and shows the raw scan,
classified walls and mapping guidance. The outer capsule marks the
controller's padded clearance threshold. The readout reports distance from
the axle segment to the closest lidar return, alongside the separate
forward-cone distance and scan-match residual.
