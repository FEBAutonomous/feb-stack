"""Raceline: minimum-curvature path inside the track, then a speed profile.

Minimum curvature (the usual first-order approximation of minimum time on a narrow track):
every centreline sample i may move sideways by alpha_i along its normal, within the track less
a margin for the car's half width. Curvature is approximated by the second difference of the
positions, which is linear in alpha, so the objective is a quadratic in alpha and the problem
is a box-constrained QP. CasADi's qrqp solves it in milliseconds; ipopt is the fallback.

Speed profile: the lateral-acceleration limit caps speed at each point from the curvature;
a forward pass limits acceleration and a backward pass limits braking, both run twice around
the loop so the closure is consistent.
"""
import numpy as np


def lateral_bounds(centre, normal, half_width, cones, clearance):
    """How far the raceline may move left (upper) and right (lower) of the centreline at each
    sample: the boundary less the clearance, and never within `clearance` of any cone. A cone
    sitting near the centreline (a stray landmark, a mislabelled cone the boundary missed)
    would otherwise end up under the car."""
    N = len(centre)
    lo, hi = -np.maximum(half_width - clearance, 0.03), np.maximum(half_width - clearance, 0.03)
    if cones is None or len(cones) == 0:
        return lo, hi
    cones = np.asarray(cones, float).reshape(-1, 2)
    tang = np.column_stack([normal[:, 1], -normal[:, 0]])          # normal is left of travel
    for c in cones:
        d = c - centre
        along = np.abs(d[:, 0] * tang[:, 0] + d[:, 1] * tang[:, 1])
        near = along < clearance
        if not np.any(near):
            continue
        side = d[near, 0] * normal[near, 0] + d[near, 1] * normal[near, 1]      # signed lateral offset of the cone
        idx = np.flatnonzero(near)
        # boundary cones are already the half width (with their scatter); only a cone well
        # inside the corridor tightens the bounds, otherwise every cone leaves a step
        inside = np.abs(side) < half_width[idx] - 0.3
        left = (side > 0) & inside
        right = (side <= 0) & inside
        hi[idx[left]] = np.minimum(hi[idx[left]], side[left] - clearance)
        lo[idx[right]] = np.maximum(lo[idx[right]], side[right] + clearance)
    # a cone near the centreline leaves no room between the bounds: pass it on the roomier side
    # with a real interval to move in, so the solution bends rather than jumps
    for i in np.flatnonzero(hi - lo < 0.15):
        if hi[i] + lo[i] > 0:
            lo[i] = hi[i] - 0.15
        else:
            hi[i] = lo[i] + 0.15
    # bounds may only change slowly along the track (5 cm per sample), so the line inside them
    # can be smooth: a tighter spot is approached gradually from both sides
    step = 0.05
    for arr, sign in ((hi, 1.0), (lo, -1.0)):
        v = sign * arr
        for _ in range(2):
            for i in range(1, N):
                v[i] = min(v[i], v[i - 1] + step)
            for i in range(N - 2, -1, -1):
                v[i] = min(v[i], v[i + 1] + step)
            v[0] = min(v[0], v[-1] + step)
            v[-1] = min(v[-1], v[0] + step)
        arr[:] = sign * v
    # the limiter can push the two bounds past each other where a cone sits in a narrowing:
    # open a real interval around their midpoint (the midpoint of two slope-limited curves is
    # slope-limited, and so are these), which is the least-bad line past that cone
    mid = 0.5 * (hi + lo)
    hi, lo = np.maximum(hi, mid + 0.075), np.minimum(lo, mid - 0.075)
    return lo, hi


def min_curvature(centre, normal, half_width, car_half_width=0.135, margin=0.12, reg=0.02, cones=None):
    """Minimum-curvature offsets alpha along the normals, a box-constrained QP. The Hessian is
    pentadiagonal (second differences), so it is built sparse: a 1300-point track solves in
    well under a second instead of minutes. CasADi's qrqp first, scipy's L-BFGS-B as fallback.
    With `cones`, every cone is kept at least car half width + margin away."""
    import scipy.sparse as sps
    N = len(centre)
    lo, hi = lateral_bounds(centre, normal, half_width, cones, car_half_width + margin)
    idx = np.arange(N)
    ds = float(np.median(np.linalg.norm(np.roll(centre, -1, axis=0) - centre, axis=1)))
    # second difference divided by ds^2 approximates curvature, so the objective (and reg) mean
    # the same thing whatever the sampling
    D = sps.csr_matrix((np.concatenate([-2.0 * np.ones(N), np.ones(N), np.ones(N)]) / (ds * ds),
                        (np.concatenate([idx, idx, idx]), np.concatenate([idx, (idx + 1) % N, (idx - 1) % N]))), shape=(N, N))
    Cx, Cy = D @ centre[:, 0], D @ centre[:, 1]
    Nx, Ny = D @ sps.diags(normal[:, 0]), D @ sps.diags(normal[:, 1])
    H = (2.0 * (Nx.T @ Nx + Ny.T @ Ny) + 2.0 * reg * sps.eye(N)).tocsc()
    H.sum_duplicates()
    H.sort_indices()                      # CasADi requires sorted compressed-column storage
    g = 2.0 * (Nx.T @ Cx + Ny.T @ Cy)
    alpha = None
    try:
        import casadi as ca
        a = ca.MX.sym("a", N)
        Hc = ca.DM(H)
        qp = {"x": a, "f": 0.5 * ca.dot(a, ca.mtimes(Hc, a)) + ca.dot(ca.DM(g), a)}
        solver = ca.qpsol("qp", "qrqp", qp, {"print_iter": False, "print_header": False, "error_on_fail": False})
        sol = solver(lbx=lo, ubx=hi, x0=np.clip(np.zeros(N), lo, hi))
        alpha = np.asarray(sol["x"]).flatten()
        if not np.all(np.isfinite(alpha)):
            alpha = None
    except Exception:
        alpha = None
    if alpha is None:
        from scipy.optimize import minimize
        res = minimize(lambda x: 0.5 * x @ (H @ x) + g @ x, np.zeros(N), jac=lambda x: H @ x + g,
                       method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": 500})
        alpha = res.x
    return centre + alpha[:, None] * normal, alpha


def curvature(path):
    nxt, prv = np.roll(path, -1, axis=0), np.roll(path, 1, axis=0)
    d1 = (nxt - prv) / 2.0
    d2 = nxt - 2.0 * path + prv
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-9
    return num / den


def smooth_loop(x, n):
    """Moving average round a closed path."""
    n = int(n)
    if n < 3:
        return x
    if n % 2 == 0:
        n += 1
    return np.convolve(np.r_[x[-n:], x, x[:n]], np.ones(n) / n, mode="same")[n:-n]


def speed_profile(path, v_max, a_lat, a_acc, a_brake, v_min=0.5, curv_window=2.0):
    ds = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    # Curvature is a second difference of points a quarter of a metre apart, so a centimetre of
    # map error reads as a tight corner. Left raw it puts a slow-down every metre and a half:
    # twenty-four separate brake-and-accelerate events a lap on the loop, and three centimetres
    # of map error turns an 8.9 s plan into a 17 s one. Averaging the signed curvature over
    # about a car and a half leaves ten, and 9.2 s. Signed matters: averaging the magnitude
    # rectifies the error instead of cancelling it and changes almost nothing.
    kappa = np.abs(smooth_loop(curvature(path), round(curv_window / max(float(np.mean(ds)), 1e-6))))
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-6)))
    N = len(v)
    for _ in range(2):
        for i in range(N):                                   # forward: acceleration limit
            j = (i + 1) % N
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_acc * ds[i]))
        for i in range(N - 1, -1, -1):                       # backward: braking limit
            j = (i - 1) % N
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_brake * ds[j]))
    return np.maximum(v, v_min), kappa


def speed_plan(path, v_max, a_lat, a_acc, a_brake, curv_window=2.0, v_min=0.5):
    """Speed along a closed path from what the tyres can do: `a_lat` of cornering, and a tyre
    force of `a_acc` driving and `a_brake` braking (m/s^2). The body's drag and the drag of
    cornering are the car's own (vehicle.py): they take from the acceleration and add to the
    braking, so the plan brakes later and accelerates less at speed, as the car does."""
    from ..core.vehicle import CORNER_DRAG, DRAG, WHEELBASE
    ds = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    kappa = np.abs(smooth_loop(curvature(path), round(curv_window / max(float(np.mean(ds)), 1e-6))))
    turn = CORNER_DRAG * kappa * np.arctan(WHEELBASE * kappa)        # cornering drag per v^2
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-6)))
    N = len(v)
    for _ in range(2):
        for i in range(N):                                   # forward: what the tyres can add
            j = (i + 1) % N
            a = a_acc - DRAG * v[i] - turn[i] * v[i] ** 2
            v[j] = min(v[j], np.sqrt(max(v[i] ** 2 + 2.0 * a * ds[i], v_min ** 2)))
        for i in range(N - 1, -1, -1):                       # backward: what they can take off
            j = (i - 1) % N
            a = a_brake + DRAG * v[i] + turn[i] * v[i] ** 2
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a * ds[j]))
    return np.maximum(v, v_min), kappa


def heading_along(path):
    nxt, prv = np.roll(path, -1, axis=0), np.roll(path, 1, axis=0)
    d = nxt - prv
    return np.arctan2(d[:, 1], d[:, 0])
