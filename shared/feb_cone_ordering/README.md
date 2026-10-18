# feb_cone_ordering

The team's cone ordering, unchanged, as a simulator package. `src/algorithms/` is a verbatim
copy of `cone_ordering/src/algorithms` from feb-system-integration (commit e048173, Akhil
Agarwal), minus the raylib visualiser; `delaunator.h` is the Mapbox Delaunator port (MIT).

What it does, in order: a Delaunay triangulation over all cones keeps the same-colour edges
under 10 m; a wall-following walk from the edge nearest the car extracts the boundary polygon
of each colour; a parity test says whether the car is inside a closed loop; every boundary edge
contributes a "force" rotated by 5/8 pi (blue one way, yellow the other), falling off with the
cube of the distance, and integrating that slope field from the car gives a path down the
middle of the corridor; every second path point is projected sideways onto the two boundaries,
rotated when a rung would cross the previous one and snapped to the polygon, giving aligned
lists of blue and yellow points, one rung per position along the track.

Use it live (`ros2 launch feb_cone_ordering order.launch.py`: rungs on `/feb/cone_order/blue`
and `/feb/cone_order/yellow` from `/feb/map` and `/feb/pose`), on demand through the
`/feb/order_cones` service (the cone racer does this once its map closes), or on a text file:

```bash
printf 'car 0 0 0\nb 0 1\nb 2 1\ny 0 -1\ny 2 -1\n' | ros2 run feb_cone_ordering cone_order_cli
```

`src/ordering_robust.h` is the one addition around the algorithm: on a closed map a single cone
of the wrong colour can collapse the ordering to a couple of rungs (13 of the 183 cones on the
FS Germany map do that when flipped), so when that happens the most suspicious cones, those
nearer to the other colour than to their own, are flipped one at a time and the ordering run
again; 12 of those 13 cases recover with one to three flips, in about 10 ms each. The node also
retries from a pose a little behind or ahead of the car. Nothing in `src/algorithms` changed.

The constants (edge limit, step, rotation, falloff) are in `src/algorithms/util.h`.
