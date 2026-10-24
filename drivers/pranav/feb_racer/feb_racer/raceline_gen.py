"""Raceline generation: free-space grid + driven lap path -> minimum-curvature line + speed profile.
Importable (race_online builds the line in-process after the warm-up lap); tools/raceline.py is the CLI.
"""
import sys, json, math
import numpy as np
import yaml
from scipy import ndimage
from PIL import Image

def load_map(yaml_path):
    meta = yaml.safe_load(open(yaml_path))
    img = np.array(Image.open(yaml_path.rsplit('/', 1)[0] + '/' + meta['image']))
    free = (img >= 250)[::-1]                              # only truly-free cells are free (unknown = wall); y-up
    return free, meta['resolution'], np.array(meta['origin'][:2])

def centreline(free, res):
    dist = ndimage.distance_transform_edt(free) * res
    # keep the biggest connected free region (the track), then its ridge
    lbl, n = ndimage.label(free)
    if n > 1:
        big = np.argmax(ndimage.sum(free, lbl, range(1, n + 1))) + 1
        dist[lbl != big] = 0
    ridge = (dist == ndimage.maximum_filter(dist, size=5)) & (dist > 0.25)
    pts = np.argwhere(ridge)[:, ::-1].astype(float)      # (x, y) in cells
    return pts, dist

def order_loop(pts, res):
    # greedy nearest-neighbour chaining into a closed loop
    left = list(range(len(pts))); cur = left.pop(0); order = [cur]
    while left:
        d = np.linalg.norm(pts[left] - pts[cur], axis=1); j = int(np.argmin(d))
        if d[j] > 1.5 / res: break                          # gap: stop the chain
        cur = left.pop(j); order.append(cur)
    return pts[order]

def resample(loop, step_cells):
    seg = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]; total = s[-1]
    n = max(50, int(total / step_cells)); t = np.linspace(0, total, n, endpoint=False)
    xy = np.vstack([loop, loop[:1]])
    return np.c_[np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])]

def _corridor(line, nrm, dist_cells, margin_cells):
    """Allowed lateral offset [lo, hi] (cells) of each point along its normal."""
    n = len(line); H, W = dist_cells.shape; offs = np.arange(0.0, 60.0, 0.5); lo = np.zeros(n); hi = np.zeros(n)
    for i in range(n):
        for sgn, arr in ((1, hi), (-1, lo)):
            p = line[i][None, :] + sgn * offs[:, None] * nrm[i]
            ok = dist_cells[np.clip(p[:, 1].astype(int), 0, H - 1), np.clip(p[:, 0].astype(int), 0, W - 1)] >= margin_cells
            k = int(np.argmin(ok)) if not ok.all() else len(ok)
            arr[i] = sgn * offs[max(k - 1, 0)]
    return lo, hi


def min_curvature(line, dist_cells, margin_cells, iters=300, alpha=0.25, kappa_max=None, ds_cells=None, w_k=50.0, rounds=6, clear_soft_cells=None, w_c=0.0, res=0.05):
    """Minimum-curvature line: optimise each point's lateral offset along the local
    normal inside the corridor, minimising sum |d2 P|^2 with an analytic gradient,
    RE-RESAMPLED to uniform spacing between rounds. Without the resampling the
    objective rewards cutting inside a hairpin: moving points inward shrinks the
    spacing ds and |d2| = kappa*ds^2 drops even though kappa rises (measured:
    the 'optimised' line had kappa 1.09 where the centreline had 0.85)."""
    from scipy.optimize import minimize
    if ds_cells is None:
        ds_cells = float(np.linalg.norm(np.roll(line, -1, 0) - line, axis=1).mean())
    # |d2| in cells = kappa[1/m] * res[m/cell] * ds_cells^2  (the old kappa*ds_cells^2 was 20x too
    # large in cell units, so the curvature cap never engaged)
    m_cap = (kappa_max * res * ds_cells ** 2) if kappa_max else None
    L = line.copy()
    for _ in range(rounds):
        L = resample(L, ds_cells); n = len(L)
        tang = np.roll(L, -1, 0) - np.roll(L, 1, 0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        nrm = np.c_[-tang[:, 1], tang[:, 0]]; lo, hi = _corridor(L, nrm, dist_cells, margin_cells)
        Hh, Ww = dist_cells.shape
        def clr(P):
            return dist_cells[np.clip(P[:, 1].astype(int), 0, Hh - 1), np.clip(P[:, 0].astype(int), 0, Ww - 1)]
        def fg(a):
            P = L + a[:, None] * nrm
            d2 = np.roll(P, -1, 0) - 2 * P + np.roll(P, 1, 0); gd = 2 * d2; f = float((d2 ** 2).sum())
            if m_cap is not None:
                mag = np.linalg.norm(d2, axis=1); ex = np.maximum(0.0, mag - m_cap)
                f += w_k * float((ex ** 2).sum()); gd = gd + w_k * 2 * (ex / np.maximum(mag, 1e-9))[:, None] * d2
            g = np.einsum('ij,ij->i', nrm, np.roll(gd, 1, 0) - 2 * gd + np.roll(gd, -1, 0))
            if clear_soft_cells and w_c > 0:      # soft wall clearance: trade a little curvature for room
                c = clr(P); short = np.maximum(0.0, clear_soft_cells - c)
                dc = (clr(P + 0.5 * nrm) - clr(P - 0.5 * nrm))     # d clearance / d offset (per cell)
                f += w_c * float((short ** 2).sum()); g = g - w_c * 2 * short * dc
            return f, g
        r = minimize(fg, np.zeros(n), jac=True, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'maxiter': iters, 'maxfun': iters * 3})
        L = L + r.x[:, None] * nrm
    return L


def curvature(line):
    p = np.roll(line, 1, 0); q = np.roll(line, -1, 0)
    a = np.linalg.norm(line - p, axis=1); b = np.linalg.norm(q - line, axis=1); c = np.linalg.norm(q - p, axis=1)
    area = 0.5 * np.abs((line[:, 0] - p[:, 0]) * (q[:, 1] - p[:, 1]) - (q[:, 0] - p[:, 0]) * (line[:, 1] - p[:, 1]))
    return 4 * area / np.maximum(a * b * c, 1e-9)

def speed_profile_capped(v, ds, a_acc, a_dec, kappa=None, ay_max=None):
    """Forward/backward accel-limit passes on an arbitrary per-point cap. With kappa/ay_max the
    friction ellipse applies here too: the hairpin EXIT ramped 1.0 -> 3.0 m/s in 0.4 m while
    still turning at kappa ~1 (G7 trace) and the car understeered into the outer wall."""
    v = v.copy(); n = len(v)
    def avail(a, vi, ki):
        if kappa is None: return a
        u = min(1.0, (vi * vi * abs(ki)) / max(ay_max, 1e-6))
        return a * math.sqrt(max(0.0, 1.0 - u * u)) + 0.3
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n; v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * avail(a_acc, v[i], kappa[i] if kappa is not None else 0.0) * ds))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n; v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * avail(a_dec, v[j], kappa[j] if kappa is not None else 0.0) * ds))
    return v


def speed_profile(kappa, ds, v_max, ay_max, a_acc, a_dec, ellipse=True):
    """Friction-ellipse profile: longitudinal accel available at a point shrinks with the
    lateral accel already used there, a_x = a * sqrt(1 - (v^2 kappa / ay_max)^2). Full
    throttle on a hairpin exit while the tyres are still loaded laterally is what ran the
    car wide into the S-bend (runs 10e-10i)."""
    ay_max = np.broadcast_to(np.asarray(ay_max, float), np.shape(kappa)).copy()
    v = np.minimum(v_max, np.sqrt(ay_max / np.maximum(kappa, 1e-6)))
    n = len(v)
    def avail(a, vi, ki, ayi):
        if not ellipse: return a
        u = min(1.0, (vi * vi * ki) / max(ayi, 1e-6))
        return a * math.sqrt(max(0.0, 1.0 - u * u)) + 0.3
    for _ in range(4):
        for i in range(n):            # forward: acceleration limit
            j = (i + 1) % n; v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * avail(a_acc, v[i], kappa[i], ay_max[i]) * ds))
        for i in range(n - 1, -1, -1):  # backward: braking limit
            j = (i + 1) % n; v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * avail(a_dec, v[j], kappa[j], ay_max[j]) * ds))
    return v

def loop_from_path(path_xy, dist_cells, res, origin):
    """Driven path (map frame, ordered by time) -> one closed loop of corridor-centre
    points in cells. The path gives the ORDER (the ridge chain folded); each point is
    then moved along its local normal to the corridor centre (argmax of the distance map)."""
    cells = (np.array(path_xy) - origin) / res
    seg = np.linalg.norm(np.diff(cells, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
    d0 = np.linalg.norm(cells - cells[0], axis=1); end = len(cells)
    near = np.where((s > 20.0 / res) & (d0 < 1.5 / res))[0]
    if len(near):
        first = near[0]; win = near[near < first + int(6.0 / res / max(seg.mean(), 1e-6))]
        end = int(win[np.argmin(d0[win])])
    loop = resample(cells[:end], 0.20 / res)
    n = len(loop); tang = np.roll(loop, -1, 0) - np.roll(loop, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True); nrm = np.c_[-tang[:, 1], tang[:, 0]]
    H, W = dist_cells.shape; offs = np.arange(-1.0 / res, 1.0 / res, 0.5); out = loop.copy()
    for i in range(n):
        p = loop[i][None, :] + offs[:, None] * nrm[i]
        d = dist_cells[np.clip(p[:, 1].astype(int), 0, H - 1), np.clip(p[:, 0].astype(int), 0, W - 1)]
        # nearest local maximum of the distance map (the corridor centre), not a far-away pocket
        k = int(np.argmax(d * np.exp(-np.abs(offs) / (2.0 / res))))
        out[i] = p[k]
    # light smoothing of the centre estimate (ordering noise from the driven path)
    for _ in range(5): out = 0.5 * out + 0.25 * (np.roll(out, 1, 0) + np.roll(out, -1, 0))
    return out


def clean_free(free):
    """Seal dotted walls and pad one cell so nothing leaks through a gap."""
    occ = ~free
    occ = ndimage.binary_closing(occ, iterations=2) | occ
    return ~ndimage.binary_dilation(occ, iterations=1)


def build_raceline(free, res, origin, path_xy, P, iters=300, step_m=0.20, kappa_max=None):
    """free: bool grid [row=y, col=x], y-up, cell (0,0) at origin (m). path_xy: driven
    lap in the same frame. Returns the raceline dict (points, kappa, v, lap_length_m)."""
    free = clean_free(free)
    pts, dist = centreline(free, res)
    loop = loop_from_path(path_xy, dist / res, res, np.asarray(origin, float))
    line = resample(loop, step_m / res)
    margin = (0.27 / 2 + P.get('wall_margin_m', 0.12)) / res
    race = resample(min_curvature(line, dist / res, margin, iters=iters, kappa_max=kappa_max or P.get('kappa_max'), ds_cells=step_m / res,
                                  clear_soft_cells=P.get('clear_soft_m', 0.0) / res, w_c=P.get('w_clear', 0.0), res=res), step_m / res)
    xy = race * res + np.asarray(origin, float)
    kap = curvature(xy)
    # clearance-aware speed cap: where the line runs close to a wall (the inside of an
    # S-bend apex) a tracking error of 0.3-0.5 m at 4 m/s is a hit - every collision of
    # runs 09b-09e was at such a spot. Full speed only with >= clear_full clearance.
    H, W = dist.shape; cx = np.clip(race[:, 0].astype(int), 0, W - 1); cy = np.clip(race[:, 1].astype(int), 0, H - 1)
    clr = dist[cy, cx]; c_lo = 0.27 / 2 + P.get('wall_margin_m', 0.12); c_hi = P.get('clear_full_m', 0.60)
    v_cap = P.get('v_narrow', 2.8) + (P.get('v_max', 5.0) - P.get('v_narrow', 2.8)) * np.clip((clr - c_lo) / max(c_hi - c_lo, 1e-3), 0.0, 1.0)
    # the cap is for curves (a tracking error there points INTO the wall); on a straight the
    # line may hug a wall harmlessly - W22 logs showed the start straight capped at 3.1 m/s
    k_cap = P.get('clear_cap_kappa', 0.12); w_curve = np.clip(np.abs(kap) / k_cap, 0.0, 1.0)
    v_cap = v_cap * w_curve + P.get('v_max', 5.0) * (1.0 - w_curve)
    # curvature-dependent lateral limit: hairpins (kappa >= 0.6) sit at the slip peak of the
    # steering, so keep ay_low there; gentle curves (kappa <= 0.3) use small angles and are
    # grip-limited (ay_high). 11a/11b (ay 7 everywhere): 21/18 collisions.
    ay_lo = P.get('ay_98pct_mps2', 6.0); ay_hi = P.get('ay_high_mps2', ay_lo)
    ay_k = ay_lo + (ay_hi - ay_lo) * np.clip((0.6 - kap) / 0.3, 0.0, 1.0)
    v = speed_profile(kap, step_m, P.get('v_max', 5.0), ay_k, P.get('a_acc', 3.0), P.get('a_dec', 5.0))
    # steering-angle (slip peak) cap with understeer: kappa * L0 * (1 + (v/v_ch)^2) <= tan(delta_peak)
    L0 = P.get('L0_m', 0.348); vch = P.get('v_ch_mps', 1e9); td = math.tan(P.get('delta_peak_rad', 0.30)) * P.get('delta_margin', 0.85)
    if vch < 100.0:                                 # only meaningful with a speed-dependent wheelbase
        v_steer = np.sqrt(np.maximum(0.0, (td / np.maximum(kap * L0, 1e-6) - 1.0))) * vch
        v_cap = np.minimum(v_cap, np.maximum(v_steer, 1.0))
    # empirical grip map: max curvature the car can actually hold at each speed (log-mined, table of
    # (v, kappa_max)); the speed at which kappa_max(v) == kappa*safety is the cap for that point
    gm = P.get('grip_map')
    if gm:
        gv = np.array([g[0] for g in gm]); gk = np.array([g[1] for g in gm]) * P.get('grip_safety', 0.9)
        vs_ = np.linspace(0.5, P.get('v_max', 5.0), 200); kmax_v = np.interp(vs_, gv, gk)   # decreasing in v
        v_grip = np.array([vs_[kmax_v >= k].max() if (kmax_v >= k).any() else 0.5 for k in np.abs(kap)])
        v_cap = np.minimum(v_cap, np.maximum(v_grip, 1.0))
    # hairpin approach: the steering actuator (0.25 s delay + 0.3 s swing) cannot follow a curvature
    # step; arriving at 2 m/s turns the car in 1 m early and clips the wall end (P2i1). Cap the speed
    # for hairpin_pre_m before any kappa >= hairpin_kmin so the lag costs ~0.5 m, not 1 m.
    hk = P.get('hairpin_kmin', 0.0)
    if hk > 0:
        n_ = len(kap); pre = int(round(P.get('hairpin_pre_m', 1.0) / step_m)); hv = P.get('hairpin_v', 1.3)
        tight = np.where(np.abs(kap) >= hk)[0]
        for i in tight:
            for j in range(1, pre + 1): v_cap[(i - j) % n_] = min(v_cap[(i - j) % n_], hv)
            v_cap[i] = min(v_cap[i], hv)
    v = speed_profile_capped(np.minimum(v, v_cap), step_m, P.get('a_acc', 3.0), P.get('a_dec', 5.0), kappa=kap, ay_max=ay_lo)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2); n_ = len(xy); ii = np.arange(n_)
    far = (np.abs(ii[:, None] - ii[None]) % n_ > 5) & (np.abs(ii[:, None] - ii[None]) % n_ < n_ - 5)
    self_d = float(D[far].min()) if far.any() else 9.9
    return {'resolution': step_m, 'points': xy.tolist(), 'kappa': kap.tolist(), 'v': v.tolist(),
            'lap_length_m': float(len(xy) * step_m), 'self_distance_m': self_d,
            'est_lap_s': float((step_m / np.maximum(v, 0.1)).sum())}
