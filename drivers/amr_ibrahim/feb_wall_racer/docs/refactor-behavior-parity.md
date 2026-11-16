# Refactor implementation evidence

Baseline runtime: 9192d76. Implementation branch: stack-refactor. The existing
100-test suite is an oracle, not sufficient evidence that new nodes are correct.
Large replay/native evidence is retained outside the member source package.

Status: the modular graph is now the sole packaged runtime. This document
records the historical parity gates and results.

The node-by-node implementation plan retains the original B01–B20 acceptance
register. The final runtime layout is documented in [organization.md](organization.md).

The 2026-11-06 cleanup removes monolithic entry points, compatibility wrappers and
tracked bytecode. Offline benches now live under `benchmarks/` and use the modular
core. Driver tuning and the supervisor/platform packages are unchanged.

The committed replay fixture covers 64 public loop scans. Its expected outputs
were generated from `9192d76`; it exercises only the modular owners during tests.
The incomplete Porto replay placeholder was removed because that earlier bag
contained no usable command history. No Porto replay pass is claimed.

Historical evidence before this cleanup: the ported suite passed 160 cases;
the initial modular native loop run reached lap 5 with zero collisions and no
controller faults. Those results do not constitute a fresh ROS test of later edits.
Current cleanup validation (2026-11-06): 75 local numerical/configuration tests
passed, with three ROS/CasADi-dependent checks skipped. All 77 Python sources
parsed; 83 relative package imports resolved. The wall smoke benchmark passed and
the controller benchmark CLI loaded successfully. The full ROS suite/build could
not be rerun because Docker Desktop failed to start on its inaccessible
`dockerInference` runtime socket. No fresh simulator or ROS pass is claimed.
