# Planar scan matching experiment

Implements the incremental mapping strategy approved in the conversation. Keep
the existing bounded wall map, driving, loop detection and closure distribution.
Replace mapping-lap translation matching with robust point-to-line registration
of x/y/yaw against the existing map before inserting the scan. A normal-based
information matrix projects away weak directions: parallel walls must not invent
motion along the corridor. Retain odometry/IMU prediction in those directions.
The lidar yaw correction persists as a map-frame offset to the sensor heading;
reset clears it. Racing retains its current translation matcher for this first
lap experiment. No new SLAM package, graph optimization or platform changes.

Point-to-point rigid ICP was considered, but discrete wall sampling can create
false longitudinal constraints. Point-to-line residuals directly represent wall
geometry and allow observability checks. Reject insufficient overlap, poor wall
neighborhoods and excessive corrections; robust weights reduce outlier influence.
Bound runtime by the existing scan/map sampling limits and spatial indexing.

Checklist (inline implementation, as in prior tasks):
- Capture baseline sources and run existing suite.
- Add failing tests for yaw correction, rotated straight-corridor null direction,
  corner translation, insufficient matches, outliers, and persistent/reset heading.
- Rename the old mapping module to slam.py and update all package references.
- Implement registration and mapping integration with configurable validated bounds.
- Run full tests; compare identical realistic scans with injected prediction errors.
- Run baseline and modified native simulator laps under identical settings; record
  mapping/closure, collisions and timing. Report regressions as well as improvements.
- Review complete diff and document measured results and remaining limitations.

No accuracy or collision improvement is assumed before testing. Truth may be
recorded for evaluation only and must never enter runtime localization.

Completed: baseline snapshot and suite, failing-then-passing matcher tests,
rename/integration, build and 100-test suite, recorded-scan comparison, paired
native loop runs, independent review. Results are mixed; see
[measured results](2026-11-01-scan-matching-results.md).
