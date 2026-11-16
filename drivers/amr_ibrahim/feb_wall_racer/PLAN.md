Run with: ./feb up --driver feb_wall_racer --track loop

## Goal ##

The goal is to develop code to allow autonomous racing within walls instead of cones, using feb_cone_racer code as a reference for message types, car parameters, track parameters, and physics models (ex: kinematic bicycle). Some stuff can also be reuse, such as mpc, and some will also be removed.

This document contains all recurring project info, if you have anything to add about stack changes or new rules, please add them with a human's consent.

Future improvements: Clean up params (there are 4 files that have similar info), Full on graphslam pose-graph optimization for first lap, improved path planning based on FEB planning, fix path discontinuity near start of lap, fix oscillations in path following.

## Implementation ##

1. Make sure to read the rules below

2. Make sure you understand the goal stack below

3. Make sure you understand the repo structure, and the runtime environment. This is very important to keep the same, do not deviate much from the current setup

4. You will be developing new code side by side the old code, NOT a replacement. Make sure to understand this

5. Make sure all code is feasible, that is it accounts for real environments (like outliers), physical constraints, and most importantly, runtime complexity. Code needs to run quickly during runtime to optimize the racer and avoid complications

6. Make sure your code uses the very same ros messages, to keep this project standard and make debugging  easily. It should also broadcast and receieve messages at similar times, such as map messages after lap closure

7. Also adopt similar debugging statements throughout the code, keeping it to a similar level as the feb_cone_racer code (defined below)

8. Report any mismatches/issues you find in the plan

## Rules: ##

1. Do not modify files in `feb_cone_racer` unless explicitly told to. The wall stack now lives
directly in `feb_wall_racer/`. The cone-only runtime modules were removed; only shared
vehicle/controller support and offline bench helpers remain here under root-level names
(`vehicle.py`, `controller.py`, `control_params.py`, `controller_raceline.py`, and `cone_bench.py`).

2. Make sure to not change anything outside of stack/

3. Never make any assumptions

## Stack ##

1. The Perception Stack (Point Cloud to Polylines)

The goal of this layer is to ingest raw 2D/3D LiDAR scans and output ordered lists of vertices representing the inner and outer track walls.
 [ Raw LiDAR Scan ] ──> Downsampling & Filtering ──> Clustering (Euclidean) ──> Line Extraction (Split-and-Merge) ──> Ordered Polylines
• Downsampling & Outlier Removal: Raw scans are passed through a pass-through filter to drop any points outside the track boundary or too close to the car's chassis.
• Segmentation / Clustering: Point clouds are split into distinct clusters using Euclidean Clustering or Density-Based Clustering (DBSCAN). This separates the left wall, right wall, and any dynamic obstacles (opponent cars).
• Line Extraction via Split-and-Merge (Iterative End-Point Fit):
	1. For a given wall cluster, a straight line is drawn from the first point to the last point.
	2. The algorithm finds the point in the cluster furthest from that line.
	3. If that distance exceeds a small threshold (e.g., \(2\text{ cm}\)), the line is "split" at that point into two segments.
	4. This repeats recursively until all points are tightly bounded by a chain of interconnected lines.
• Output: Two clean arrays of ordered 2D coordinates: Left_Wall_Polyline = [[x1, y1], [x2, y2], ...] and Right_Wall_Polyline = [[x3, y3], [x4, y4], ...].

2. The SLAM Stack (Localization & Map Maintenance)

Traditional SLAM tracks thousands of points. Polyline SLAM instead treats the track map as a graph of line features.
• Feature-Based Scan Matching: Instead of matching raw points, the front-end aligns the currently observed polylines with the previously saved polylines in the global map. This is achieved by minimizing the perpendicular distance between observed segment endpoints and the map lines.
• Tracking Track Curvature (Odometry Fusion): To resolve longitudinal slippage along featureless walls, a high-rate Extended Kalman Filter (EKF) fuses wheel odometry and IMU data. This EKF provides a highly accurate "prior" pose prediction, ensuring the scan-matching algorithm doesn't latch onto the wrong segment of a repeating wall pattern.
• Loop Closure via Vertex Graph: When the vehicle completes a lap, it detects that its current local polyline overlaps heavily with the starting segments of the global map. A Pose Graph Optimization (PGO) step runs in the background, distributing any accumulated drift evenly backwards across the lap, instantly snapping the global racetrack polyline into alignment.

3. The Pathing Stack (Planning & Control)

With the track reduced to left and right bounding lines, path planning becomes a highly efficient geometric and optimization problem.
 [ Left & Right Polylines ] ──> Midpoint Interpolation ──> Centerline / Frenet Frame
                                                                   │
 [ Optimum Racing Line Optimization ] <────────────────────────────┘
         │
         ▼
 [ Local MPC Solver ] ──> Hard Inequality Constraints (Don't cross polylines) ──> Throttle & Steering
• Centerline Extraction (The Topological Path): By computing the geometric midpoints between corresponding vertices on the left and right polylines, the system generates a track centerline.
• Global Planning (Minimum Curvature / Maximum Speed): The global planner fits a smooth cubic spline through this centerline. It then computes the Optimal Racing Line by solving a quadratic programming problem that trades off path length against curvature (maximizing speed through corners while clipping the apexes).
• Local Planning (Frenet Frame Formulation): During the race, the car's state is converted from Cartesian \((X, Y)\) coordinates to Frenet coordinates \((s, d)\), where \(s\) is the distance along the optimal racing line and \(d\) is the lateral deviation from it.
• Collision Avoidance & Control via Model Predictive Control (MPC):
	• The MPC looks ahead \(1.0\) to \(1.5\text{ seconds}\) (predicting the car's future trajectory across 10–15 steps).
	• The left and right polylines are fed into the optimizer as Hard Spatial Constraints:
\(\text{Wall}_{\text{left}}(s)\le d_{\text{car}}(s)\le \text{Wall}_{\text{right}}(s)\)
	• Because checking the distance to a line segment requires a trivial mathematical dot product, the MPC optimization problem can solve in under \(5\text{ ms}\), outputting the perfect steering angle and motor throttle for the F1TENTH platform.

4. Control

Use the same MPC strategy as cone stack, inputted paths should follow the same format as the cone racing stack

Recommended Open-Source Packages for this Architecture

If you are building this in ROS 2 (the standard for F1TENTH), you do not need to code everything from scratch:
1. Perception: Use the laser_line_extraction ROS 2 package to instantly convert your raw 2D sensor_msgs/msg/LaserScan data into ordered line segments.
2. Localization: For a pre-mapped track, look into AMCL (Adaptive Monte Carlo Localization) configured for line features, or use a lightweight 2D NDT (Normal Distributions Transform) matcher.
3. Pathing: The F1TENTH Development System repository contains reference implementations for Frenet Frame Planners and optimization scripts to calculate minimum curvature paths given inner and outer track boundaries.

## Implementation Plan: `feb_wall_racer`

This section records the repository-specific implementation and its validation gates. The wall
stack now exists directly under `feb_wall_racer/`; it is an initial implementation, not yet
proven against the FEB box or simulator.

### Scope and project constraints

- Add wall racing alongside `feb_cone_racer`; do not replace or edit files in the cone package.
- Keep the wall implementation and its executable entry point at the package root so names such
  as `feb_wall_racer.racer` and `feb_wall_racer.perception` remain simple.
- Retain only controller/vehicle support needed by the wall racer and offline cone bench. Keep
  cone-specific helpers separately named (`control_params.py`, `controller_raceline.py`,
  `cone_bench.py`) to avoid colliding with wall defaults and planning modules.
- Keep all implementation changes inside `stack/`. This plan is in that scope.
- Use the current FEB box/simulator workflow and the standard RoboRacer ROS interface:
  `/scan`, `/odom`, `/imu`, `/drive`, and `/feb/state`.
- At race time use lidar, dead-reckoned odometry, IMU gyro data, supervisor run state, and the
  racer's own map estimates. Do not subscribe to simulator-private `/autodrive/...` topics,
  ground truth, lap/collision counters, or reset commands.
- Keep per-scan work bounded and keep driving safely if mapping, localization, or optimization
  is uncertain. Log and publish the reason for a fallback; never turn a failed fit into a
  success-shaped map or command.

### Onboarding findings and current package state

The package has a ROS 2 Python-package skeleton (`setup.py`, `package.xml`, root `driver.yaml`,
`config/driver.yaml`, `launch/drive.launch.py`, `install_deps.sh`). The wall implementation is at
the package root.
`controller.py`, `mpc.py`, `vehicle.py`, and `control_params.py` provide the shared controller,
vehicle model, and tuned race defaults. `controller_raceline.py` and `cone_bench.py` preserve
cone-geometry-only offline planning/benchmark helpers under distinct names; they are not used by
the wall node.
- The wall launch starts only the wall node and does not start or require `feb_cone_ordering`.
- The wall launch, executable, node, and YAML selector use `wall_racer`.
- The FEB driver card registers `feb_wall_racer` as kind `walls`, with `trackdrive` and
  `inspection` modes and their supervisor speed caps. Its `packages` list includes the package
  itself for FEB submission.
- The wall runtime uses `config/driver.yaml`; the launch overlays `config/driver.car.yaml` only
  on-car when present, and does not need a cone-racer YAML or camera
  calibration package.
- The copied mapping-lap closure code uses a candidate pose before checking whether the candidate
  is `None`. Do not port that defect: wall closure candidates must be validated before use.

### Verified runtime interface

The member guide (release 2) defines the public interfaces as `/scan` (`sensor_msgs/LaserScan`,
1080 beams over 270 degrees, frame `laser`), `/odom` (`nav_msgs/Odometry`, dead-reckoned pose
and speed), and `/imu` (`sensor_msgs/Imu`). Commands go to `/drive`
(`ackermann_msgs/AckermannDriveStamped`): `drive.speed` is in m/s and
`drive.steering_angle` is in radians, limited to +/-0.5236 rad. `/feb/state` is a
`std_msgs/String` from the supervisor; only `AUTONOMOUS` is a live driving state. Simulator
`/autodrive/...` topics remain debug-only and are not available on the car. Use best-effort
sensor subscriptions so both reliable and best-effort sensor publishers can connect. Sensor
delivery may vary; use scan timestamps and stop on stale odometry, IMU, or lidar rather than
depending on a nominal scan rate.

The inputs are 2D scans, not `PointCloud2`. Do not make an external point-cloud or line-extraction
package a prerequisite until its availability in the current box, interface, runtime cost, and
value have been verified. Start with a small deterministic NumPy implementation over scan
ranges. Use public odometry increments for translation and orientation, IMU gyro data for the
body-speed estimate, and wall matching for map-localization corrections; do not use simulator
truth or private bridge topics.

### Proposed runtime and data flow

Implement one ROS node, `wall_racer`, following the existing node's integrated architecture:
on each fresh autonomous run, the first lap is a low-speed mapping lap; no preloaded map or
track-specific ground truth is required. Freeze and publish a map
only after a validated loop closure and track build. The release-2 member guide does not promise
an unscored warm-up lap, so verify practice/competition timing before relying on mapping being
complete before scoring begins. Integration must verify that the node is alive and receives
scans before the first lap begins; if it does not finish mapping successfully, it must stay in
the safe mapping/fallback mode rather than treating an incomplete map as race-ready.

1. **Sense:** receive lidar scans, odometry, IMU updates, and `/feb/state`. Transform valid scan
   returns from the `laser` sensor frame to the vehicle pose using the configured mount offset
   and current public state estimate.
2. **Perceive:** remove invalid/out-of-range and chassis returns, form scan-contiguous segments,
   robustly fit/split line segments, and assemble locally ordered left/right wall polylines.
   Carry confidence and scan time with observations. Use geometric side, local heading, and
   continuity; do not assume left and right are independent Euclidean clusters, or that their
   vertices correspond by array index.
3. **Map while driving:** follow a conservative local corridor estimate while integrating
   odometry motion and IMU gyro data. Add bounded keyframes of wall segments/features, align scans
   to the local map, and detect loop closure from a unique, consistent return to mapped geometry.
   A failed/ambiguous match must not move the pose estimate or close the lap.
4. **Build and validate the track:** after a complete loop, freeze a map snapshot and build
   ordered, closed wall boundaries. Establish centerline samples and local half-width by
   projecting across the corridor, not by pairing arbitrary left/right vertices. Reject
   self-intersections, gaps, implausible width changes, inconsistent orientation, and
   insufficient coverage before racing.
5. **Plan and race:** optimize a raceline within the measured wall corridor with car-width and
   clearance margins, compute the speed plan, then use the existing vehicle model/controller
   where tests confirm it is suitable. After mapping, localize scans against the frozen wall map
   and drive the raceline; do not require local corridor-center estimates for racing steering.
   Wall returns remain inputs to localization and forward obstacle detection. Reduce speed or
   stop on stale localization; never silently continue at racing speed.
6. **Observe:** publish standard ROS debug messages and a throttled human-readable status showing
   mode, fit confidence, map/closure progress, localization residual, control fallback, speed,
   and solver time.

Keep the mapping follower independent of map closure: it needs a local wall-based corridor and
safe pure-pursuit or equivalent fallback from the first usable scan. Pair wall returns at shared
forward-distance bins along candidate local travel axes to make ordered center samples; select an
axis by corridor support and plausible, consistent width. Choose a lookahead target on the
observed path and compute steering curvature from the rear axle, accounting for the lidar's
forward mounting offset. Do not bridge missing forward bins. Accumulate left/right labels in
separate map vote grids only after a well-supported local corridor classification; use
repeated map votes as a prior, not as an unconditional replacement for current scan geometry.
Briefly reuse the last valid steering at a reduced speed only when a recent mapping path
exists; otherwise stop rather than guessing a direction. Check the predicted short-horizon
steering arc against current lidar wall returns and reduce speed by available braking distance,
stopping if the path is already too close to a wall. Reduce mapping speed further when
estimated curvature requires it.

### File-by-file work

| File | Planned work |
|---|---|
| `feb_wall_racer/racer.py` | Implement `MAPPING`, `RACING`, and controlled-stop behavior using `/scan`, `/odom`, `/imu`, `/feb/state`, and `/drive`. Reset mapping/localization/controller state on each transition into `AUTONOMOUS`. Keep bounded per-scan work; do not import camera or cone-ordering runtime code. |
| `feb_wall_racer/perception.py` | Filter LaserScan returns, segment adjacent beams, split line runs, assign wall sides, estimate local paired corridors, and report confidence/residuals. Bound point/segment counts and avoid quadratic all-pairs work over all 1080 beams. |
| `feb_wall_racer/pathing.py` | Search local scan headings for supported corridor bins and convert the ordered center samples into rear-axle pure-pursuit guidance without bridging gaps. |
| `feb_wall_racer/slam.py` | Maintain a bounded voxel wall-point map plus repeat-voted left/right wall labels; perform heading-preserving translation matching and gate loop closure by travel, pose, heading, scan overlap, and correction size. This is not an EKF or full pose-graph optimizer. |
| `feb_wall_racer/params.py` | Provide wall racer defaults while adapting the shared racing/control values in `control_params.py`. Keep wall perception, mapping, closure, corridor validation, and localization settings together here and in the wall YAML. Do not add ROS dependencies to this module. |
| `driver.yaml`, `config/driver.yaml` | Register the package as a `walls` driver with `trackdrive` and `inspection` modes; document each tunable and keep the node's parameter selector as `wall_racer`. Keep `config/racer.yaml` unchanged as reference. |
| `feb_wall_racer/track.py` | Build a closed sampled track from mapping-path samples and measured corridor widths; reject implausible length, closure gaps, odometry gaps, centerline self-intersections, inadequate width coverage, and too-narrow corridors. Return center, normal, half-width, and arc length for raceline planning. |
| `feb_wall_racer/raceline.py` | Create a bounded smoothed centerline offset and lateral-acceleration speed profile. Test that the raceline stays within the measured half-width after vehicle and wall-clearance margins. |
| `feb_wall_racer/controller.py`, `mpc.py`, `vehicle.py`, `control_params.py` | Reuse the measured vehicle model and controller where suitable. Verify path projection, speed estimates, 10 Hz response, and solver failure behavior on wall tracks. The reused MPC does not consume walls directly: enforce clearance in raceline construction and treat direct online wall constraints as separate scope requiring explicit implementation and tests. |
| `launch/drive.launch.py` | Follow the FEB driver launch contract: declare `mode`, choose sim time except on-car, overlay optional car and environment override parameters, include `feb_mission` when present, and start only the wall node. |
| `setup.py`, `package.xml`, `install_deps.sh` | Register the root driver card for FEB discovery/submission, wall racer and offline bench entry points, discover the package, and declare ROS runtime dependencies. Use the workspace `.pydeps` path exported by FEB. No wall calibration executable is included. |
| `feb_wall_racer/bench.py` and `test/` | Provide a ROS-independent synthetic smoke bench and unit coverage for parallel-wall perception, invalid scans, measured-width rejection, closed-track/raceline clearance, and bounded/frozen map behavior. Additional simulator replay and FEB box checks remain required. |
| package-local `README.md` | Document launch, two-phase behavior, debug topics, tuning, safety limits, and the distinction between local unit checks and uncompleted simulator validation. |

### Current implementation status

The ROS node, wall scan processing, bounded point map, loop candidate gate, track/raceline
construction, mapping-only local follower, controller integration, YAML, launch wiring, smoke
bench, and focused unit tests are present. Racing follows the built raceline; wall returns are
still used for localization and the forward obstacle stop check, but local corridor-center
confidence no longer gates racing or substitutes a local steering command. The wall stack now
uses the cone stack's 7 m/s speed-plan maximum and its acceleration, MPC, and throttle limits;
curvature still lowers the actual planned speed, and localization degradation scales it down.
The offline checks currently cover synthetic parallel walls, all-invalid scans, measured-width
rejection, closed-track planning/clearance, bounded/frozen map behavior, and equality with the
cone stack's speed settings.

The estimator is deliberately simpler than the architecture sketch: it uses the copied body-speed
observer, integrated IMU heading, dead-reckoned translation, and translation-only scan matching.
It is not a full EKF or pose-graph optimizer. The map is a bounded wall-point map rather than a
topological line graph, global track boundaries are inferred from the mapping path and measured
widths, and the MPC does not enforce wall inequalities online. Repeated-track ambiguity checks,
scan replay on recorded simulator data, ROS box build/launch, and scored wall-track runs are
still required. Do not treat the offline tests as evidence that the next live run is validated.

### ROS messages and observability

Use existing standard messages; do not introduce custom interfaces for the initial version.
Use `LaserScan` `/scan`, `Odometry` `/odom`, `Imu` `/imu`, `String` `/feb/state`, and
`AckermannDriveStamped` `/drive` (`speed` in m/s, `steering_angle` in radians). Do not publish
the old simulator-private normalized `Float32` steering/throttle commands. Keep
`/feb/status`, `/feb/pose`, `/feb/raceline`, and
`/feb/mpc_prediction` compatible with their current standard types. Publish the current local
wall fits as `nav_msgs/Path` on `/feb/wall/left` and `/feb/wall/right` in `lidar`, and publish
transformed live wall fits as `/feb/wall/live/left` and `/feb/wall/live/right` in the estimated
`map` frame during mapping and racing. Publish a bounded recent `/feb/trajectory` path and
`/feb/car_marker` in `map` for the driven path and car body. Broadcast the estimated
`map` → `feb_wall_racer_base` transform from the first usable sensor cycle so Foxglove can use
`map` as a fixed frame before loop closure; do not connect it to simulator `world`. Publish frozen ordered
map boundaries as `nav_msgs/Path` on
`/feb/wall/map/left` and `/feb/wall/map/right` only after a validated loop closure and track
build; use transient-local QoS so late subscribers receive the completed map. Latch the completed
raceline as the cone racer does. Document these topics and distinguish scan-local from frozen
global geometry. Source the scan-local wall frame from `LaserScan.header.frame_id` (normally
`laser`). Use `map` as the Foxglove fixed frame for these racer outputs; keep the
simulator's ground-truth `world` TF tree separate rather than publishing an unverified
`map`-to-`world` transform. Do not reuse `/feb/cones` or cone-order topics for wall data.

Status/debug output should make these transitions and failure cases visible: too few valid
returns, ambiguous wall side, excessive fit residual, insufficient map coverage, rejected loop
closure, invalid corridor, stale map match, MPC unavailable/failed, and controlled stop. Avoid
per-beam/per-scan log floods; use throttled status and concise transition logs.

### Implementation sequence and exit criteria

1. **Characterize the real inputs.** In the FEB box, inspect scan timing, ranges, invalid values,
   frame, scan geometry, and wall visibility on a wall track. Record representative bags and
   compare against generated track geometry offline. Confirm which public tracks have duct walls.
   No algorithm or dependency choice is considered validated solely from the architecture sketch.
2. **Implement and test perception offline.** Cover empty/all-invalid scans, range limits,
   isolated outliers, missing returns, parallel walls, corners, curved walls, and nearby
   obstacles. Require deterministic output, bounded allocations/runtime, correct side/order, and
   reported confidence. Establish thresholds from recorded scans rather than assume the
   suggested 2 cm split tolerance is valid at all ranges.
3. **Implement local driving and safe degradation.** Drive the first mapping lap from locally
   detected wall boundaries. Test both sides of bends and short occlusions. Verify no target,
   invalid corridor, or stale scan produces a steering guess; it must reduce speed or stop.
4. **Implement wall mapping and closure.** Add keyframe selection, motion prior, scan matching,
   robust outlier rejection, and a unique loop-closure gate. Replay bags deterministically and
   compare estimated wall-map alignment with generated geometry offline. Cap map size/search
   work and reject low-confidence solves.
5. **Implement track and raceline construction.** Test closure, orientation, width and
   self-intersection validation on straight, curved, hairpin, and narrow track fixtures. Verify
   every raceline sample respects both boundaries with at least the configured vehicle-width
   and clearance margin; verify speed-plan acceleration, braking, and cornering limits.
6. **Wire race-time localization/control.** Reuse the copied controller/vehicle code only after
   bench testing the wall-derived path. Test normal localization, adjacent-track ambiguity,
   gradual drift, sudden mismatch, unavailable CasADi, solver failure, and 10 Hz scan delivery.
   Preserve a low-speed fallback and stop after the configured loss timeout.
7. **Integrate and score in simulation.** Build with the FEB box's normal `colcon` startup and run
   `./feb up --driver feb_wall_racer --track loop`. First check an inspection-mode
   `./feb up --deadman --driver feb_wall_racer --mode-name inspection`; then exercise tight/long
   wall tracks and variable scan rates. Check Foxglove topics and logs; inspect bags and truth
   only offline. Run `./feb practice --driver feb_wall_racer` on the qualifying wall-track set.

Completion gates:

- Package builds and launches from the normal FEB box entrypoint without `feb_cone_ordering`.
- A full mapping lap closes only on validated geometry, and produces a closed track/raceline
  with all samples within the measured corridor and clearance.
- The car completes at least 10 scored laps on each targeted qualifying wall track with zero
  wall contacts; report any excluded or failing tracks rather than treating a partial run as
  success.
- No race-time subscription to restricted ground-truth/scoring topics.
- Establish the scan-time budget from measurements in the target FEB box; perception and map
  matching must stay within the actual simulator and practice/scoring scan intervals. Log
  measured p50/p95 instead of claiming an unmeasured solver deadline.

### Mismatches and decisions to resolve while implementing

1. The initial architecture describes ordered left/right polylines, but it does not specify how
   an individual return is assigned to a wall at crossings, hairpins, occlusions, or where
   separate wall returns are scan-contiguous. Implement and validate side assignment and
   continuity before relying on midpoint interpolation.
2. The architecture's point/segment association and vertex midpoint rules assume useful
   correspondences. Left and right polylines will generally have different sample counts and
   sample locations; establish common centerline stations through local projection and
   orientation/width checks rather than index pairing.
3. The suggested EKF and background pose-graph optimization are not present in the copied
   package. The current graph solver is a sparse linear least-squares graph over cone landmarks
   with IMU heading supplied externally. First implement the simplest robust wall-feature
   estimator that passes replay and runtime tests; add a larger EKF/PGO architecture only if
   evidence justifies it.
4. `laser_line_extraction`, AMCL-for-lines, NDT, and an F1TENTH planner are suggestions, not
   established dependencies or configured components in this Humble package. Check package
   availability and actual 2D scan compatibility before adopting any; the default plan does not
   require them.
5. The architecture claims hard wall inequalities inside MPC. The copied `mpc.py` tracks a
   reference and enforces vehicle/control limits; it does not take wall polylines as constraints.
   Initially enforce clearance in track/raceline construction and validate it after optimization.
   Do not describe runtime wall constraints as implemented until the MPC formulation and
   infeasible-solve fallback have been implemented and tested.
6. Camera calibration, HSV classification, cone colours, and the cone-ordering service do not
   apply to the duct-wall sensor path. Remove these from the wall node/launch. No calibration
   node is needed unless measured lidar/IMU alignment errors demonstrate otherwise.
7. Resolved integration mismatch: the node, launch executable, and YAML selector all use
   `wall_racer`, and loop-closure candidates are checked before use. Preserve these guards during
   future launch or closure changes.
8. The release-2 guide names the lidar frame `laser` but does not give its transform to the car
   frame. `lidar_forward` is retained from the prior sim calibration and must be confirmed against
   the updated sim before treating the estimated pose and wall clearances as calibrated.
