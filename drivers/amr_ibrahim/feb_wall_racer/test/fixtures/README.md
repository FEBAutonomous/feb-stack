# Recorded regression input

`recorded_mapping.json.gz` contains 64 selected public loop scans, their held
odometry/IMU inputs and command history, and the frozen `9192d76` expected outputs.
`test_refactor_parity.py` runs the modular owners against that data at `1e-8`.
It contains no simulator truth. Unavailable range values are stored as null.

The fixture currently covers loop only. The earlier Porto recording began after
the driver stopped and supplied no usable command history; no Porto replay result
is claimed. Native simulator tests are separate from this deterministic fixture.
The one-time archived-oracle export script remains in local reference storage,
outside the modular source package.

`porto_gap_corner.json.gz` contains the public lidar points captured at the
26.59 m standstill in the initial gap-matcher Porto practice attempt. It guards
the distinction between visibility of the swept body and the larger clearance
required from measured walls.

`loop_gap_corner.json.gz` contains the public scan at the 18.05 m standstill
in the first corrected Loop practice attempt. It has a short verified arc with
room to brake from a crawl, but insufficient room at mapping speed.
