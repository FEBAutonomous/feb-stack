"""Tracking MPC on a kinematic bicycle, the formulation of the team's own controller.

State  z = [x, y, psi, v, delta]     rear axle position, heading, body speed, steering angle
Input  u = [a, omega]                tyre force along the car per unit mass (m/s^2), steering rate

    dx = v cos(psi)     dy = v sin(psi)     dpsi = v tan(delta) / L
    dv = a - drag v - corner_drag v^2 delta tan(delta) / L
    ddelta = omega

Why kinematic: measured on this car, the yaw rate is v tan(delta) / L to within 1% and the rear
slip angle at 6.2 m/s^2 of cornering is 0.2 degrees, right up to the point the tyres let go.
A dynamic model with tyres soft enough to integrate at this step describes a car that leans on
its tyres, which this one does not; the limit itself is kept as a constraint instead.

Why these inputs: the steering servo is a rate, and the throttle sets the wheel speed, which
the tyre turns into a force on the body. Asking the optimiser for a force and a steering rate
keeps both inside what the car can do, and the force is turned into a throttle afterwards by
the measured tyre curve (vehicle.throttle_for), so the wheels are never spun or locked.

Cost, per stage, against a reference the car can actually drive (controller.py rolls it out
from the car's own speed): the error across the raceline in full, the error along it lightly,
heading, speed, the inputs against the reference's, and how fast the inputs change. Direct
multiple shooting, RK4, IPOPT, warm started from the last solution shifted one stage.
"""
import numpy as np

try:
    import casadi as ca
except ImportError:                  # the node reports this clearly at startup
    ca = None

from .vehicle import VehicleModel

NZ, NU, NREF = 5, 2, 5               # state, input, reference columns (x, y, heading, speed, tyre force)


class KinematicMPC:
    def __init__(self, p):
        self.p = p
        self.vehicle = VehicleModel.from_parameters(p)
        self.N, self.DT = int(p["mpc_horizon"]), float(p["mpc_dt"])
        self.ok = ca is not None
        self.warm = None
        self.status = ""
        if self.ok:
            self.build()

    # ------------------------------------------------------------ model

    def dynamics(self, z, u):
        v, d = z[3], z[4]
        return ca.vertcat(v * ca.cos(z[2]),
                          v * ca.sin(z[2]),
                          v * ca.tan(d) / self.vehicle.wheelbase,
                          u[0] - self.vehicle.drag * v - self.vehicle.corner_drag * v * v * d * ca.tan(d) / self.vehicle.wheelbase,
                          u[1])

    def step(self, z, u):
        n = int(self.p["mpc_substeps"])
        h = self.DT / n
        for _ in range(n):
            k1 = self.dynamics(z, u)
            k2 = self.dynamics(z + h / 2 * k1, u)
            k3 = self.dynamics(z + h / 2 * k2, u)
            k4 = self.dynamics(z + h * k3, u)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    # ------------------------------------------------------------ problem

    def build(self):
        p, N, DT = self.p, self.N, self.DT
        Z = ca.SX.sym("Z", NZ, N + 1)
        U = ca.SX.sym("U", NU, N)
        S = ca.SX.sym("S", N)                     # slack on the cornering limit, one per stage
        z0 = ca.SX.sym("z0", NZ)
        ref = ca.SX.sym("ref", NREF, N)
        uprev = ca.SX.sym("uprev", NU)
        g, lbg, ubg = [Z[:, 0] - z0], [0.0] * NZ, [0.0] * NZ
        cost = 0
        for k in range(N):
            g.append(Z[:, k + 1] - self.step(Z[:, k], U[:, k]))
            lbg += [0.0] * NZ
            ubg += [0.0] * NZ
            zk = Z[:, k + 1]
            # the point that follows the line is the middle of the car, not its rear axle
            ex = zk[0] + self.vehicle.axle_to_centre * ca.cos(zk[2]) - ref[0, k]
            ey = zk[1] + self.vehicle.axle_to_centre * ca.sin(zk[2]) - ref[1, k]
            c, s = ca.cos(ref[2, k]), ca.sin(ref[2, k])
            last = 1.0 if k < N - 1 else p["w_terminal"]
            cost += last * p["w_lat"] * (-ex * s + ey * c) ** 2
            cost += last * p["w_lon"] * (ex * c + ey * s) ** 2
            cost += last * p["w_head"] * (1 - ca.cos(zk[2] - ref[2, k]))
            cost += last * p["w_speed"] * (zk[3] - ref[3, k]) ** 2
            cost += p["w_force"] * (U[0, k] - ref[4, k]) ** 2
            cost += p["w_omega"] * U[1, k] ** 2
            prev = uprev if k == 0 else U[:, k - 1]
            cost += p["w_dforce"] * ((U[0, k] - prev[0]) / DT) ** 2
            cost += p["w_domega"] * ((U[1, k] - prev[1]) / DT) ** 2
            # cornering limit on the stage's end state, softened so that a car already past it
            # (a slide, a bad measurement) still gets a plan: the slack is what it costs
            ay = zk[3] ** 2 * ca.tan(zk[4]) / self.vehicle.wheelbase
            g += [ay - S[k], ay + S[k]]
            lbg += [-ca.inf, -p["mpc_a_lat"]]
            ubg += [p["mpc_a_lat"], ca.inf]
            cost += p["w_slack"] * S[k] ** 2
        w = ca.vertcat(ca.vec(Z), ca.vec(U), S)
        nlp = {"x": w, "f": cost, "g": ca.vertcat(*g), "p": ca.vertcat(z0, ca.vec(ref), uprev)}
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": int(p["mpc_max_iter"]),
                "ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-2, "ipopt.acceptable_iter": 3,
                "ipopt.warm_start_init_point": "yes", "ipopt.mu_strategy": "adaptive", "ipopt.sb": "yes"}
        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self.lbg, self.ubg = np.array(lbg, float), np.array(ubg, float)
        lim = 0.98 * p["max_steer"]
        lbz = np.tile([-np.inf, -np.inf, -np.inf, 0.0, -lim], N + 1)
        ubz = np.tile([np.inf, np.inf, np.inf, p["v_cap"], lim], N + 1)
        # the first column is the car as it is, pinned by the first equality: no bounds on it,
        # or a car measured over the speed cap makes the whole problem infeasible
        lbz[:NZ], ubz[:NZ] = -np.inf, np.inf
        steer_rate = min(p["mpc_steer_rate"],self.vehicle.steer_rate)
        lbu = np.tile([-p["mpc_a_brake"], -steer_rate], N)
        ubu = np.tile([p["mpc_a_acc"], steer_rate], N)
        self.lbx = np.concatenate([lbz, lbu, np.zeros(N)])
        self.ubx = np.concatenate([ubz, ubu, np.full(N, np.inf)])

    def solve(self, z0, ref, uprev):
        """z0 (5,), ref (5, N), uprev (2,). Returns (U (2, N), Z (5, N+1)) or None."""
        N = self.N
        z0, ref = np.asarray(z0, float), np.asarray(ref, float)
        if self.warm is None:
            Zg = np.repeat(z0[:, None], N + 1, axis=1)
            Zg[2, 1:], Zg[3, 1:] = ref[2], ref[3]
            Zg[0, 1:] = ref[0] - self.vehicle.axle_to_centre * np.cos(ref[2])
            Zg[1, 1:] = ref[1] - self.vehicle.axle_to_centre * np.sin(ref[2])
            Ug = np.zeros((NU, N))
            Ug[0] = ref[4]
            x0 = np.concatenate([Zg.flatten(order="F"), Ug.flatten(order="F"), np.zeros(N)])
        else:
            x0 = self.warm
        pvec = np.concatenate([z0, ref.flatten(order="F"), np.asarray(uprev, float)])
        try:
            sol = self.solver(x0=x0, p=pvec, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        except Exception as e:                       # noqa: BLE001
            self.warm, self.status = None, "exception: %s" % e
            return None
        self.status = self.solver.stats().get("return_status", "")
        if self.status not in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):      # an unfinished iterate is not a plan
            self.warm = None
            return None
        w = np.asarray(sol["x"]).flatten()
        Z = w[: NZ * (N + 1)].reshape((NZ, N + 1), order="F")
        U = w[NZ * (N + 1): NZ * (N + 1) + NU * N].reshape((NU, N), order="F")
        if not np.all(np.isfinite(w)):
            self.warm = None
            return None
        Zs = np.hstack([Z[:, 1:], Z[:, -1:]])
        Us = np.hstack([U[:, 1:], U[:, -1:]])
        self.warm = np.concatenate([Zs.flatten(order="F"), Us.flatten(order="F"), np.zeros(N)])
        return U, Z
