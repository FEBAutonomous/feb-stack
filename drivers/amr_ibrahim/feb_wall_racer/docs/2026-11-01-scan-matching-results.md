# Planar scan-matching experiment results

Branch: `codex/planar-scan-matching`, starting from stack commit `fd82e33`.
Existing strategy retained: bounded voxel wall map, mapping keyframes, start-scan
loop detection, distributed translation closure, frozen-map racing localization.
Mapping registration now estimates X/Y/yaw with robust point-to-line residuals.
Weak information directions retain odometry prediction. Heading correction
persists through subsequent IMU updates and odometry integration and clears on
reset. Racing still uses its original translation matcher.

`slam.py` replaces the previous mapping-module filename. Imports, documentation
file references and tests were updated; obsolete tracked bytecode for the old
module was removed. Platform/supervisor/mission implementation is unchanged.

## Verification

Clean baseline: 91 tests passed. Final modified suite: **100 passed in 65.88 s**.
Colcon build succeeded. Independent review found no important correctness issue;
its additional 400 noisy straight-corridor cases retained rank two and had at
most 2.83 mm unintended longitudinal correction.

Tests cover straight/rotated corridors, corners, noise/outliers, overlap loss,
mount offset, map immutability during registration, excessive observable
correction rejection, invalid settings, heading-offset integration and reset.

## Recorded realistic scans: controlled registration comparison

Input: `runs/20261101-120138-Amr_Ibrahim/bag`, 10 Hz practice recording.
114 selected scans from the first 350 scan messages, each tested under four
conditions: heading-only or combined position/heading prediction error, with
or without 15 mm point noise and 5% perturbed returns. Total 456 paired trials.

Reference local submaps use the preceding six recorded scans positioned using
diagnostic simulator truth. Neither matcher sees that truth as a live input.
This is an isolated registration benchmark, **not an end-to-end SLAM replay**.
The benchmark uses filtered raw lidar returns; runtime uses fitted wall returns.
Rejected predictions are included in error totals, not excluded.

- Heading-only, clean: mean heading error 1.719 -> 0.382 degrees;
  mean position error 4.06 -> 2.50 cm (baseline -> planar).
- Heading-only, noisy: 1.719 -> 0.420 degrees; 4.02 -> 2.76 cm.
- Combined, clean: 1.432 -> 0.790 degrees; 8.20 -> 5.96 cm.
- Combined, noisy: 1.432 -> 0.803 degrees; 8.29 -> 6.20 cm.
- Planar accepted 454/456 trials. At least 75/114 in each condition were rank two.
- Planar registration p95 runtime 2.91–3.26 ms, including local-map indexing.

Limit: combined-error heading tails did not uniformly improve. In the noisy
case, p95 heading error was 1.638 degrees versus the 1.432-degree injected prior.
Mean improvement does not guarantee improvement on each scan.

Reproduce inside the ROS box, with ROS sourced:

```bash
python3 -B stack/feb_wall_racer/tools/replay_scan_matching.py \
  runs/20261101-120138-Amr_Ibrahim/bag \
  --baseline .agent-reference/slam-experiment/baseline/feb_wall_racer/graphslam.py \
  --output .agent-reference/slam-experiment/replay.json
```

The baseline argument intentionally names archived historical source outside the
stack repository. Current package code has no imports of that module.

## Native closed-loop comparison

Same normal stock launcher, loop track, default tuning, headless, 110-second
bounds and 100-second passive recordings. Truth collected only for diagnosis.
Baseline source is an immutable snapshot of the initial stack commit, selected
only for the wall-racer executable; the modified run uses installed current code.

Baseline: `runs/slam-test-baseline-20261101-124854`.
Planar: `runs/slam-test-current-20261101-125203`.

- Both reached lap 6, zero collisions, no recorded safety stops.
- Loop-closure translation discrepancy: 0.774 -> 0.944 m.
- Mapping callback p95: 77.6 -> 88.3 ms (includes perception/control, not just ICP).
- Racing estimated path error median: 2.64 -> 2.74 cm.
- Racing estimated path error p95: 9.60 -> 14.12 cm; max 28.90 -> 36.75 cm.
- Both generated finite closed paths and respected the configured lateral-speed
  envelope; no planned point was below the inferred wall clearance.

These are single paired runs with natural scheduling/trajectory variation.
They **do not demonstrate better overall mapping or racing** on the clean loop.
Loop discrepancy is not a complete measure of map distortion, and estimated
path error is not independent ground-truth localization accuracy.

## Interpretation

The incremental registration improves recovery from controlled prediction errors
on recorded geometry and behaves as intended in straight corridors. It works
through mapping-to-racing transition without a collision in the tested run.
The clean-simulator comparison is mixed and contains worse metrics; keep this
as an experimental branch rather than claim an unqualified upgrade over main.
No global pose graph or rotational loop correction was added. Full map accuracy,
real hardware performance, other tracks and full qualifying runs remain unproven.

Raw local evidence: `.agent-reference/slam-experiment/replay.json`,
`native-comparison.json`, baseline/current `driver.jsonl`, `passive.json`, and
`analysis.json`. No live driving processes were left running by these tests.
