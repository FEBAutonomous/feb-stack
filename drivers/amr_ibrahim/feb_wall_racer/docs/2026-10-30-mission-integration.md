# Mission integration design and implementation checklist

Follow-up (2026-10-31): the user requested stock supervisor/platform behavior.
The wall racer now latches local faults and supervisor STOP, rejects automatic
re-entry while halted, and checks stale lidar before either authorization input
is revoked. A latch clears only through a fresh stock autonomous-button session
with mission confirmation, or a new driver process. Normal initial automatic
startup and one reset per healthy autonomous session remain supported. The
original design and verification record below are historical.

## Intent

Connect the existing `feb_mission` node to `feb_wall_racer` start/reset behavior,
while preserving direct supervisor stops and the currently working
`./feb up --driver feb_wall_racer --track loop` workflow. The user selected this
approach on 2026-10-30. Driving algorithms, tuning, map closure and sensor safety
are outside this change.

## Approaches

- Launch-only: guaranteeing the existing mission process would leave its live
  signal unused by the racer. This was not the selected behavior.
- Mission-only gate: consuming only `/feb/mission/live` would delay direct stops
  until the periodic mission publication. This does not preserve the requested
  direct supervisor stop path.
- Selected: require both supervisor AUTONOMOUS and mission live for starting;
  either false stops the racer. The supervisor remains the immediate stop path.

## Proposed wiring

- Launch the existing `feb_mission` package's `mission` executable exactly once
  alongside `wall_racer`, with the same sim/car clock setting. Declare the ROS
  runtime dependency and report a clear launch error if the package is missing.
  Mission process exit shuts down the driver launch, rather than leaving an
  orphaned racer. The shared mission implementation remains unchanged.
- The racer subscribes to `/feb/mission/live` (`std_msgs/Bool`) in addition to
  its existing configured supervisor state input. Starts require both signals.
  Retain idle zero-speed heartbeats so automatic simulator startup can proceed.
- Keep supervisor authorization and mission authorization separately. A
  supervisor stop invalidates the old mission authorization. Late mission true
  messages cannot restart while the supervisor is not AUTONOMOUS. Accept either
  startup delivery order; repeated live heartbeats never reset a running map.
- On the first confirmed start of a new supervisor autonomous session, call the
  existing `_reset_run` exactly once and record `live_since`. Mission reauthorization
  within that same session resumes without another reset. On active-to-inactive, immediately publish
  zero and clear live state. Existing sensor watchdogs and latched STOPPED behavior
  remain intact; heartbeats while continuously active must not clear a stop.
- Keep standard topic names and the user's normal command. Document joint topic
  remapping if a custom supervisor topic is used, so mission and racer see the
  same supervisor. Preserve parameter-file layering and dependency PYTHONPATH.

## Implementation checklist (execute inline after approval)

- [x] Add ROS lifecycle tests for both startup orders, repeated live messages,
  mission stop, direct supervisor stop, stale mission true after stop, restart
  reset, latched safety stop, and idle zero heartbeats. Observe the new integration
  assertions fail against the old node. Use real ROS nodes in an isolated domain
  where available; avoid a physical simulator or live-driving session.
- [x] Add the smallest lifecycle wiring to `feb_wall_racer/racer.py`; make mission
  mandatory in `launch/drive.launch.py`; add `feb_mission` to `package.xml`.
- [x] Build the packages in the existing FEB box, then test the real launch in
  an isolated ROS domain with synthetic supervisor state/clock/sensors. Verify
  both nodes and their subscriptions, startup, stop and re-entry, and mission
  exit handling. Do not stop or replace any existing driving session.
- [x] Run the entire wall test suite and offline bench; check installed launch
  arguments/config and sim/car clock selection. No steering/throttle tuning.
- [x] Update package README and local reference notes, refresh changed source
  hashes, and report exact validation and any remaining live-simulator limits.

## Acceptance

The normal command launches mission and racer together; mission live participates
in start/reset; direct supervisor stops still immediately command zero; state
delivery order and repeated heartbeats do not repeatedly reset the map; one new
autonomous session resets once; existing geometry checks remain passing. Live
lap performance is not claimed without an actual simulator comparison.

Review clarification: reset ownership is one reset per supervisor autonomous
session after mission confirmation. Independent topic queues can deliver a
mission false/true pair after a new session has already started; that pair
pauses/resumes authorization but must not discard the new session's map again.

## Verification record

- FEB box build: `colcon build --packages-up-to feb_wall_racer --symlink-install`;
  both feb_mission and feb_wall_racer completed successfully.
- Final complete package suite: `python3 -B -m pytest stack/feb_wall_racer/test
  -q -p no:cacheprovider`: 35 passed in 45.05 s, isolated ROS domain 165.
- Includes 21 existing geometry tests, 11 lifecycle regressions, and 3 real
  installed-launch tests with synthetic inputs. Tests exercise the real mission
  executable; only optional MPC construction is stubbed in lifecycle unit tests.
- Both sim/car clocks, base/override parameter loading, mapping commands, STOP,
  re-entry, missing-package error and mission-child exit shutdown passed.
- Before/after evidence: original racer fails mission gating tests; original
  launch fails missing-mission and mission-exit checks; initial integration fails
  the delayed-edge reset regression; final implementation passes all.
- Offline smoke bench: 13 corridor samples, width 2.20 m, confidence 1.00;
  37.9 m / 151-sample synthetic track, finite line/speeds within clearance.
- One independent code review; its ordering finding was reproduced and fixed
  with a regression before the final suite. No simulator laps or hardware drive.
