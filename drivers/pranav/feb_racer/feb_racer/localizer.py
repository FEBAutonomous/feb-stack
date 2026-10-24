"""Own 2-D scan-matching mapper / localizer (numpy + scipy only, no ROS types).

Why not slam_toolbox: on this sim it produced smeared, ghosted maps even from a
clean collision-free bag with verified sensor conventions (lidar +angle = left,
IMU yaw and encoders agree with ground truth to 0.04 rad / 1%). This one is
deterministic, testable offline against the bag's ground truth, and rate-agnostic.

Model: pose = correction (+) dead_reckoning.  Dead reckoning = encoder distance
along the IMU heading (the sim IMU orientation is absolute, no drift).  Each scan
is matched to the map by a coarse-to-fine correlative search around the prior;
the result updates the correction.  Mapping mode also inserts the scan (hits at
the beam ends, misses along the beams) into the grid.
"""
import math
import numpy as np
from scipy import ndimage


def compose(c, d):
    """pose = c (+) d  (2-D rigid transforms as (x, y, yaw))."""
    ct, st = math.cos(c[2]), math.sin(c[2])
    return (c[0] + ct * d[0] - st * d[1], c[1] + st * d[0] + ct * d[1], c[2] + d[2])


def correction_for(pose, d):
    """c such that pose = c (+) d."""
    th = pose[2] - d[2]; ct, st = math.cos(th), math.sin(th)
    return (pose[0] - (ct * d[0] - st * d[1]), pose[1] - (st * d[0] + ct * d[1]), th)


class ScanMatcher:
    def __init__(self, angle_min, angle_inc, n_beams, res=0.05, size_m=80.0, lidar_x=0.2733,
                 range_min=0.10, range_max=9.5, sub=4):
        self.res = res; self.n = int(size_m / res); self.origin = -size_m / 2.0
        self.hits = np.zeros((self.n, self.n), np.float32)
        self.miss = np.zeros((self.n, self.n), np.float32)
        self.score = np.zeros((self.n, self.n), np.float32)
        self.angles = angle_min + angle_inc * np.arange(n_beams)
        self.cos = np.cos(self.angles); self.sin = np.sin(self.angles)
        self.lx = lidar_x; self.rmin = range_min; self.rmax = range_max; self.sub = sub
        self.n_inserted = 0; self.last_insert = None; self.dirty = False
        self.min_score = 0.15; self.prior_weight = 0.10   # per-point penalty at the window edge
        self.sigma_hit = 0.12; self.z_hit = 0.90; self.z_rand = 0.10; self.lf = None

    # ---- geometry -------------------------------------------------------------
    def base_points(self, ranges, sub=1):
        """Beam end points in the base_link frame (lidar offset applied)."""
        r = np.asarray(ranges, np.float32)[::sub]; c = self.cos[::sub]; s = self.sin[::sub]
        ok = np.isfinite(r) & (r > self.rmin) & (r < self.rmax)
        return np.c_[self.lx + r[ok] * c[ok], r[ok] * s[ok]]

    def to_cells(self, wx, wy):
        return (np.clip(((wx - self.origin) / self.res).astype(np.int32), 0, self.n - 1),
                np.clip(((wy - self.origin) / self.res).astype(np.int32), 0, self.n - 1))

    # ---- mapping --------------------------------------------------------------
    def insert(self, ranges, pose):
        r = np.asarray(ranges, np.float32); ct, st = math.cos(pose[2]), math.sin(pose[2])
        sx = pose[0] + self.lx * ct; sy = pose[1] + self.lx * st          # sensor origin
        fin = np.isfinite(r) & (r > self.rmin)
        hit = fin & (r < self.rmax); rr = np.where(fin, np.minimum(r, self.rmax), self.rmax)
        ex = sx + rr * np.cos(pose[2] + self.angles); ey = sy + rr * np.sin(pose[2] + self.angles)
        ix, iy = self.to_cells(ex[hit], ey[hit]); np.add.at(self.hits, (iy, ix), 1.0)
        # misses: sample every 2nd beam along the ray, one sample per cell
        k = np.maximum(((rr[::2] - 0.08) / self.res).astype(np.int32), 0)
        beam = np.repeat(np.arange(len(k)), k); t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + 0.5) * self.res
        a = pose[2] + self.angles[::2][beam]
        ix, iy = self.to_cells(sx + t * np.cos(a), sy + t * np.sin(a)); np.add.at(self.miss, (iy, ix), 1.0)
        self.n_inserted += 1; self.last_insert = pose; self.dirty = True

    def occupied(self):
        return (self.hits >= 2.0) & (self.hits >= 0.25 * self.miss)

    def refresh(self):
        if self.dirty:
            occ = self.occupied().astype(np.float32)
            self.score = ndimage.gaussian_filter(occ, 1.2); self.score /= max(float(self.score.max()), 1e-6)
            # to_cells clips out-of-grid points onto the border: a scan placed beyond the grid then
            # scores whatever wall sits on the edge (Phase 2: the track reaches x 19.9 m on a 40 m
            # grid and the estimate ran away to x 50 m with score 0.9). The border scores zero.
            self.score[0, :] = 0.0; self.score[-1, :] = 0.0; self.score[:, 0] = 0.0; self.score[:, -1] = 0.0
            # Likelihood field for the particle filter: distance to the nearest obstacle, turned
            # into p(z|x) the way AMCL's likelihood_field model does. Kept separate from
            # self.score so the matcher's behaviour is untouched.
            d = ndimage.distance_transform_edt(~self.occupied()).astype(np.float32) * self.res
            self.lf = (self.z_hit * np.exp(-(d * d) / (2.0 * self.sigma_hit ** 2)) + self.z_rand).astype(np.float32)
            self.lf[0, :] = self.z_rand; self.lf[-1, :] = self.z_rand; self.lf[:, 0] = self.z_rand; self.lf[:, -1] = self.z_rand
            self.dirty = False

    # ---- matching -------------------------------------------------------------
    def match(self, ranges, prior, span_xy=0.30, step_xy=0.05, span_th=math.radians(10.0), step_th=math.radians(1.25)):
        """Returns (score in [0,1], pose). Coarse search then a fine search around the best."""
        self.refresh(); pb = self.base_points(ranges, self.sub)
        if len(pb) < 20 or self.n_inserted == 0:
            return 0.0, prior
        s1, p1 = self._search(pb, prior, span_xy, step_xy, span_th, step_th)
        s2, p2 = self._search(pb, p1, step_xy, step_xy / 5.0, step_th, step_th / 4.0)
        if s2 < self.min_score:          # no evidence (empty/flat map here): trust dead reckoning
            return s2, prior
        return s2, p2

    def loglik_many(self, pb, xs, ys, ths):
        """Sum of log p(z|x) over beams for N poses. Beams are subsampled (as AMCL's max_beams
        does) so one scan cannot collapse the whole particle set onto a single sample."""
        ct = np.cos(ths)[:, None]; st = np.sin(ths)[:, None]
        wx = xs[:, None] + ct * pb[None, :, 0] - st * pb[None, :, 1]
        wy = ys[:, None] + st * pb[None, :, 0] + ct * pb[None, :, 1]
        ix, iy = self.to_cells(wx, wy)
        return np.log(self.lf[iy, ix]).sum(axis=1)

    def score_many(self, pb, xs, ys, ths):
        """Likelihood-field score of one scan at N poses at once: the sensor model a particle
        filter needs. self.score is a blurred occupancy grid, i.e. AMCL's likelihood_field."""
        ct = np.cos(ths)[:, None]; st = np.sin(ths)[:, None]
        wx = xs[:, None] + ct * pb[None, :, 0] - st * pb[None, :, 1]
        wy = ys[:, None] + st * pb[None, :, 0] + ct * pb[None, :, 1]
        ix, iy = self.to_cells(wx, wy)
        return self.score[iy, ix].mean(axis=1)

    def score_at(self, pb, pose):
        """Mean map score of the scan's base points placed at pose (no search)."""
        ct, st = math.cos(pose[2]), math.sin(pose[2])
        wx = pose[0] + ct * pb[:, 0] - st * pb[:, 1]; wy = pose[1] + st * pb[:, 0] + ct * pb[:, 1]
        ix, iy = self.to_cells(wx, wy)
        return float(self.score[iy, ix].mean())

    def along_track_observable(self, ranges, pose, shift=0.3, drop=0.05):
        """False when sliding the scan +-shift along the heading barely changes the score: a
        corridor of parallel walls (Phase 2: 20 m straight) leaves the along-track coordinate
        unobservable and the argmax random-walks by up to the window size per scan (measured
        +2-3 m vs ground truth), so the caller keeps dead reckoning for that coordinate."""
        pb = self.base_points(ranges, self.sub)
        if len(pb) < 20: return True
        s0 = self.score_at(pb, pose); c, s = math.cos(pose[2]), math.sin(pose[2])
        sf = self.score_at(pb, (pose[0] + shift * c, pose[1] + shift * s, pose[2]))
        sb = self.score_at(pb, (pose[0] - shift * c, pose[1] - shift * s, pose[2]))
        return (s0 - max(sf, sb)) >= drop

    def _search(self, pb, c, sxy, dxy, sth, dth):
        ths = c[2] + np.arange(-sth, sth + 1e-9, dth); offs = np.arange(-sxy, sxy + 1e-9, dxy)
        OX, OY = np.meshgrid(offs, offs, indexing='ij'); OX = OX.ravel(); OY = OY.ravel()
        S = self.score; best = -1.0; bp = c
        # prior-centred tie-break: a flat score must not send the pose to the window corner
        pen = self.prior_weight * (np.abs(OX) + np.abs(OY)) / max(2 * sxy, 1e-9)
        for th in ths:
            pen_th = self.prior_weight * 0.5 * abs(th - c[2]) / max(sth, 1e-9)
            ct, st = math.cos(th), math.sin(th)
            wx = c[0] + ct * pb[:, 0] - st * pb[:, 1]; wy = c[1] + st * pb[:, 0] + ct * pb[:, 1]
            ix, iy = self.to_cells(wx[None, :] + OX[:, None], wy[None, :] + OY[:, None])
            v = S[iy, ix].mean(1) - pen - pen_th; k = int(np.argmax(v))
            if v[k] > best:
                best = float(v[k]); bp = (c[0] + OX[k], c[1] + OY[k], th)
        return best, bp

    def global_localize(self, ranges, cell_step=4, n_head=36, yaw_prior=None, yaw_tol=0.6):
        """Search the whole known free space x headings for the best match, then
        refine locally. Used when the map frame is not the live frame (preloaded
        map) or after being lost (respawn somewhere else). ~0.3 s."""
        self.refresh(); pb = self.base_points(ranges, self.sub * 2)
        free = (self.miss >= 2.0) & ~self.occupied(); ys, xs = np.where(free)
        sel = (xs % cell_step == 0) & (ys % cell_step == 0); xs, ys = xs[sel], ys[sel]
        if len(xs) == 0 or len(pb) < 20:
            return 0.0, None
        cx = self.origin + (xs + 0.5) * self.res; cy = self.origin + (ys + 0.5) * self.res
        S = self.score; cands = []
        heads = np.linspace(-math.pi, math.pi, n_head, endpoint=False) if yaw_prior is None else yaw_prior + np.linspace(-yaw_tol, yaw_tol, 9)
        for th in heads:                # a symmetric track needs the heading prior to disambiguate
            ct, st = math.cos(th), math.sin(th)
            px = ct * pb[:, 0] - st * pb[:, 1]; py = st * pb[:, 0] + ct * pb[:, 1]
            ix, iy = self.to_cells(cx[:, None] + px[None, :], cy[:, None] + py[None, :])
            v = S[iy, ix].mean(1)
            for k in np.argsort(-v)[:3]:
                cands.append((float(v[k]), float(cx[k]), float(cy[k]), float(th)))
        # a symmetric track is ambiguous under rotation: refine the best few DISTINCT
        # coarse candidates and keep the best refined score
        cands.sort(reverse=True); picked = []
        for c in cands:
            if all(math.hypot(c[1] - q[1], c[2] - q[2]) > 0.8 or abs(math.atan2(math.sin(c[3] - q[3]), math.cos(c[3] - q[3]))) > 0.5 for q in picked):
                picked.append(c)
            if len(picked) >= 4:
                break
        pbf = self.base_points(ranges, self.sub); s2 = -1.0; p2 = None
        for (_, x, y, th) in picked:
            sa, pa = self._search(pbf, (x, y, th), 0.30, 0.05, math.radians(10.0), math.radians(1.5))
            sb, pb2 = self._search(pbf, pa, 0.05, 0.01, math.radians(1.5), math.radians(0.25))
            if sb > s2:
                s2, p2 = sb, pb2
        return s2, p2

    # ---- export ---------------------------------------------------------------
    def export(self, prefix):
        """map_server-compatible pgm/yaml cropped to the known area."""
        occ = self.occupied(); free = (self.miss >= 2.0) & ~occ
        img = np.full(occ.shape, 205, np.uint8); img[free] = 254; img[occ] = 0
        ys, xs = np.where(occ | free)
        if len(xs) == 0:
            return None
        x0, x1 = max(xs.min() - 10, 0), min(xs.max() + 10, self.n - 1); y0, y1 = max(ys.min() - 10, 0), min(ys.max() + 10, self.n - 1)
        crop = img[y0:y1 + 1, x0:x1 + 1]
        from PIL import Image
        Image.fromarray(crop[::-1]).save(prefix + '.pgm')
        open(prefix + '.yaml', 'w').write('image: %s.pgm\nresolution: %.3f\norigin: [%.4f, %.4f, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n'
                                          % (prefix.rsplit('/', 1)[-1], self.res, self.origin + x0 * self.res, self.origin + y0 * self.res))
        return prefix + '.yaml'


class DeadReckoning:
    """Encoder distance along IMU heading. Rate-agnostic; teleport-aware (encoder
    resets on collision teleport -> the step is rejected)."""
    def __init__(self, wheel_radius=0.0578, v_max=22.88):   # 0.059 nominal; GT calibration 0.0576-0.0580 (2.4% over-count = 0.45 m per 18 m straight)
        self.R = wheel_radius; self.v_max = v_max
        self.ref = {}; self.pending = 0.0; self.yaw = None; self.yaw_off = None
        self.x = 0.0; self.y = 0.0; self.dist = 0.0
        self.t_last = None; self.v = 0.0            # for extrapolation to a scan stamp
        self.gz = 0.0; self.t_imu = None
        self.gap = 0.1; self.dt_sim_max = 0.2       # max physics advance per frame gap (measured under GPU load)
        # wheel-slip rejection: AWD throttle bursts spin the wheels (Phase 2 GT run: encoder x ran
        # 2.2 m AHEAD of truth after the hairpin exit, on a 20 m straight with no lidar feature to
        # fix it). When the encoder speed departs from the IMU-predicted speed by more than
        # slip_thr, the IMU-integrated speed carries both the speed and the distance.
        self.ax = 0.0; self.brake_carry = 0.0; self.slip_thr = 0.35; self.n_slip = 0; self.slipping = False; self.teleported = False; self.braking = False

    def imu(self, yaw_abs, gz=0.0, t=None, ax=None):
        self.gz = float(gz); t_prev = self.t_imu; self.t_imu = t
        if ax is not None:
            # Low-pass the accelerometer over ~0.1 s. Telemetry arrives once per frame: at 10 Hz one
            # frame IS 0.1 s (this reduces to the old assignment), at 40 Hz the per-frame reading is
            # four times noisier (p10/p90 -6/+5 m/s^2 on a plain lap) and, fed raw into the slip
            # logic below, it sent the dead-reckoned speed to 13 m/s on the hairpin approach - the
            # matcher's prior then ran ahead of the car, the score fell from 0.95 to 0.4 and the car
            # never turned in (40 Hz test 2026-09-18: 31-42 hits in 130 s).
            dt = 0.1 if (t is None or t_prev is None) else float(np.clip(t - t_prev, 0.0, 0.2))
            self.ax += (getattr(self, 'ax_scale', 0.84) * float(ax) - self.ax) * min(1.0, dt / 0.1)   # 0.84: 'the sim IMU reads 1.19x' (GT regression at 10 Hz, 2026-09-14); the open-loop probe of 2026-09-18 found the x0.84 integral 15 % short of the true speed gain at BOTH rates
        if self.yaw_off is None:
            self.yaw_off = yaw_abs
        self.yaw = math.atan2(math.sin(yaw_abs - self.yaw_off), math.cos(yaw_abs - self.yaw_off))

    def encoder(self, side, t, pos):
        """Feed one encoder message; returns True when a pair was integrated."""
        if self.yaw is None:
            return False
        pr = self.ref.get(side)
        if pr is not None:
            dte = t - pr[0]; d = (pos - pr[1]) * self.R
            if dte > 1e-4 and d >= -0.6 and d / dte <= 1.2 * self.v_max:   # reverse (recovery) counts; a respawn's large negative jump is still rejected
                self.pending += 0.5 * d
        self.ref[side] = (t, pos)
        # Step the integrator on a fixed cadence (step_s, 0.095 s: one frame at 10 Hz, four at 40 Hz).
        # Every test below - slip, lock, brake carry, teleport - compares a per-step encoder speed
        # with a per-step IMU prediction, and their thresholds were set on 10 Hz frames. Fed 25 ms
        # frames they misfired and handed the distance to the IMU integral (which under-reads by
        # ~15 % at any rate): the estimate fell 2 m behind the car by the first hairpin at 40 Hz.
        if side == 'right' and self.t_last is not None and t - self.t_last < getattr(self, 'step_s', 0.095):
            return False
        if side == 'right':
            d = self.pending; self.pending = 0.0; d_raw_step = d; v_enc_step = float('nan')
            # Encoder speed over the last >= 0.1 s, not one frame: at 40 Hz a single frame's wheel
            # increment is quantised and physics-substep-jittered and made the slip tests fire at random.
            # At 10 Hz the window is one frame, i.e. the old per-frame value.
            self.dist_raw = getattr(self, 'dist_raw', 0.0) + d
            self.raw_hist = getattr(self, 'raw_hist', []); self.raw_hist.append((t, self.dist_raw))
            while len(self.raw_hist) > 2 and t - self.raw_hist[1][0] >= 0.1: self.raw_hist.pop(0)
            if self.t_last is not None and t - self.t_last > 1e-3:
                # Telemetry comes once per rendered frame. When the simulator's frame rate drops
                # (a screen recorder on the organisers' machine), wall-clock gaps of 0.4-0.6 s
                # advanced the physics by only ~0.2 s (measured from the wheel angles), so d/dt_wall
                # under-reads the speed 2-3x and the speed law throttles into the next hairpin.
                # Divide by the sim-time estimate instead; over-reading is the safe direction (brakes).
                self.gap = t - self.t_last
                dt_sim = min(self.gap, self.dt_sim_max)
                v_enc = d / dt_sim; v_pred = max(0.0, self.v + self.ax * dt_sim)
                t0_, d0_ = self.raw_hist[0]
                if t - t0_ >= 0.05: v_enc = (self.dist_raw - d0_) / min(t - t0_, self.dt_sim_max * max(1, len(self.raw_hist) - 1))
                v_enc_step = v_enc
                if getattr(self, 'mode', 'imu') == 'ratelimit':
                    # Rate-limited encoder speed instead of the IMU-carried branches below. Wheel spin
                    # over-reads the encoders (up to +1.7 m/s at 40 Hz under launch throttle) and wheel
                    # lock under-reads them; but the CAR cannot gain more than a_acc_max nor lose more
                    # than a_dec_max per second, so clip the encoder speed to that envelope around the
                    # last estimate. No IMU in the loop: the old branches integrated the accelerometer
                    # whenever the encoders disagreed with it, and at 40 Hz that integral ran 15 %
                    # slow, the disagreement never closed, and the estimate fell 2 m behind the car
                    # by the first hairpin (dr_log 2026-09-18). Rate- and calibration-independent.
                    v_lim = float(np.clip(v_enc, self.v - getattr(self, 'a_dec_max', 10.0) * dt_sim, self.v + getattr(self, 'a_acc_max', 6.0) * dt_sim))
                    self.slipping = abs(v_lim - v_enc) > 1e-6; self.n_slip += int(self.slipping)
                    self.teleported = self.v > 2.5 and v_enc < 0.2
                    self.v = max(0.0, v_lim) if v_enc >= 0.0 else v_enc          # reverse (recovery) passes through
                    d = self.v * dt_sim if self.slipping else d
                    self.x += d * math.cos(self.yaw); self.y += d * math.sin(self.yaw); self.dist += d
                    if getattr(self, 'log', None) is not None:
                        self.log.write('%.3f,%.4f,%.4f,%.3f,%.3f,%.3f,%d,%d,%.3f\n' % (t, d_raw_step, d, v_enc, self.v, self.ax, int(self.slipping), int(self.braking), self.gap)); self.log.flush()
                    self.t_last = t
                    return True
                # The IMU carry below is open loop: it ignores the encoders entirely. That is right
                # while the wheels are locked, and catastrophic otherwise - after a crash the car sat
                # against a wall with ax ~ 0 and the estimate held its pre-crash speed indefinitely.
                # Gate it on the accelerometer actually showing a deceleration, and cap how long one
                # braking event may run open loop.
                if self.braking and self.ax > -3.0:
                    self.brake_carry = 0.0
                elif self.braking:
                    self.brake_carry = getattr(self, 'brake_carry', 0.0) + dt_sim
                if getattr(self, 'carry_on', True) and self.braking and self.v > 0.8 and self.ax <= -3.0 and getattr(self, 'brake_carry', 0.0) <= 1.2:
                    # throttle 0 = hard brake (12 m/s^2): the wheels lock and the encoders under-count
                    # (GT trace: speed estimate 2.2/2.8/4.2/2.9/1.4 while the truth decays smoothly,
                    # position 2.4 m behind -> the car drove into the hairpin's end wall). While braking,
                    # speed AND distance come from the IMU (already scaled to GT).
                    self.v = v_pred; d = self.v * dt_sim
                    self.x += d * math.cos(self.yaw); self.y += d * math.sin(self.yaw); self.dist += d
                    if getattr(self, 'log', None) is not None:
                        self.log.write('%.3f,%.4f,%.4f,%.3f,%.3f,%.3f,%d,%d,%.3f\n' % (t, d_raw_step, d, v_enc, self.v, self.ax, 2, int(self.braking), self.gap)); self.log.flush()
                    self.t_last = t
                    return True
                # collision respawn: the sim zeroes the velocity in one frame (>25 m/s^2 - the brake
                # does 16). Flag it for the node (the encoders do NOT reset in this sim version).
                self.teleported = self.v > 2.5 and v_enc < 0.2        # a respawn zeroes the speed in ONE frame; hard braking (12 m/s^2) never drops 1.5 m/s to zero in 0.1 s
                spin = (v_enc - v_pred > self.slip_thr) and 0.0 <= self.ax <= 12.0 and v_pred <= 12.0   # wheels faster than the car can accelerate (impact spikes excluded); throttle >= 0.4 spins the AWD wheels 2-5x (probe 2026-09-15)
                lock = (v_pred - v_enc > self.slip_thr) and self.ax <= getattr(self, 'lock_ax', -6.0) and self.v > getattr(self, 'lock_v', 3.0)   # wheel lock-up only under HARD braking from speed (lock_ax/lock_v: at 40 Hz the encoders under-count under lift-off too - raw 2.45 m vs 2.91 true in the braking zone - and the estimate fell behind by the hairpin) (8 m/s -> hairpin: estimate fell 0.5 m behind); a loose version chattered at low speed
                if (spin or lock) and self.v > 0.3:
                    # wheels spinning (throttle burst) or locked (brake): the IMU-integrated speed
                    # carries speed AND distance until the encoders agree with it again.
                    # Under lift-off (lock with a mild deceleration) the wheels under-rotate only by
                    # 5-25 % (probe 2026-09-18) and the IMU integral over-counts, so blend the two
                    # there (lock_blend, 0.5): 40 Hz run LK_R0_1 over-counted 1.1 m per second of
                    # lift-off on the IMU alone after the lock gate was relaxed to catch the under-count.
                    self.slipping = True; self.n_slip += 1
                    kb = getattr(self, 'lock_blend', 0.0) if (lock and not spin) else 0.0
                    self.v = v_pred + kb * (v_enc - v_pred); d = self.v * dt_sim
                else:
                    self.slipping = False
                    self.v = 0.4 * self.v + 0.6 * v_enc   # fast: under 16 m/s^2 braking a slow filter overshoots the extrapolation
            self.x += d * math.cos(self.yaw); self.y += d * math.sin(self.yaw); self.dist += d
            if getattr(self, 'log', None) is not None:   # diagnostics: one line per integrator step
                self.log.write('%.3f,%.4f,%.4f,%.3f,%.3f,%.3f,%d,%d,%.3f\n' % (t, d_raw_step, d, v_enc_step, self.v, self.ax, int(self.slipping), int(self.braking), self.gap)); self.log.flush()
            self.t_last = t
            return True
        return False

    @property
    def pose(self):
        return (self.x, self.y, self.yaw if self.yaw is not None else 0.0)

    def pose_at(self, t):
        """Pose extrapolated to stamp t (the scan may arrive before this tick's
        encoders: at 10 Hz and 2.7 m/s a one-tick-old pose is 0.27 m behind, which
        turns the car in late and runs it wide in every curve)."""
        yaw = self.yaw if self.yaw is not None else 0.0
        if self.t_imu is not None:      # heading from the gyro too: 50 ms at 2 rad/s is 6 deg
            yaw += self.gz * float(np.clip(t - self.t_imu, 0.0, 0.15))
        if self.t_last is None:
            return (self.x, self.y, yaw)
        dt = float(np.clip(t - self.t_last, 0.0, 0.15))
        return (self.x + self.v * dt * math.cos(yaw), self.y + self.v * dt * math.sin(yaw), yaw)


class ParticleFilter:
    """Augmented Monte-Carlo localization over the likelihood field (Probabilistic Robotics 8.3).

    Why the single-hypothesis matcher fails here: it searches a fixed box around the dead-reckoned
    prior, so when the wheels lock under braking and the encoders under-count, the true pose leaves
    that box and is never evaluated again - the estimate is lost for good while the car drives on
    it (measured on this track: score 0.95 -> 0.27 through the hairpin, then a wall).

    Three things fix that here:
      * many hypotheses, so the truth does not have to be inside one box;
      * motion noise that GROWS with suspected wheel slip, so the cloud spreads exactly when the
        odometry is lying, and symmetrically - the sign of the odometry error is not assumed;
      * augmented MCL: when the short-term likelihood falls below the long-term average, random
        particles are injected along the racing line, which is the principled form of the
        'kidnapped robot' recovery this stack previously faked with teleport detection.

    Method after Probabilistic Robotics and ForzaETH SynPF (arXiv 2401.07658); code our own.
    """

    def __init__(self, sm, n=400, seed=12345, beams=72):
        self.sm = sm; self.n = int(n); self.rng = np.random.default_rng(seed); self.beams = int(beams)
        self.x = self.y = self.th = self.w = None; self.ready = False
        self.w_slow = 0.0; self.w_fast = 0.0
        # AMCL's 0.001/0.1 assume tens of thousands of updates; a race is ~1000, so the slow
        # average never converges and sits BELOW the fast one, which disables recovery
        # entirely (measured 2026-09-20: w_slow stuck at 0.13 while w_fast rode at 0.70).
        # Time constants sized to this problem: ~2.5 s long-term, ~0.1 s short-term at 40 Hz.
        self.a_slow = 0.01; self.a_fast = 0.25
        self.cand = None; self.n_inject = 0; self.p_rand = 0.0

    # ---- candidate poses for recovery -----------------------------------------
    def set_candidates(self, pts):
        """Racing-line points with their tangent heading: where a lost car plausibly is."""
        p = np.asarray(pts, np.float32)
        if len(p) < 8:
            self.cand = None; return
        d = np.roll(p, -1, axis=0) - p
        self.cand = np.c_[p[:, 0], p[:, 1], np.arctan2(d[:, 1], d[:, 0])].astype(np.float32)

    def init_at(self, pose, spread_xy=0.25, spread_th=0.03):
        n = self.n; r = self.rng
        self.x = pose[0] + r.normal(0.0, spread_xy, n)
        self.y = pose[1] + r.normal(0.0, spread_xy, n)
        self.th = pose[2] + r.normal(0.0, spread_th, n)
        self.w = np.full(n, 1.0 / n); self.ready = True
        self.w_slow = self.w_fast = 0.0

    def predict(self, d_fwd, d_lat, d_yaw, slip=0.0):
        n = self.n; r = self.rng
        s_fwd = 0.03 + 0.10 * abs(d_fwd) + 0.60 * slip * max(abs(d_fwd), 0.05)
        s_lat = 0.015 + 0.05 * abs(d_fwd)
        s_yaw = 0.004 + 0.06 * abs(d_yaw) + 0.05 * slip * abs(d_yaw)
        f = d_fwd + r.normal(0.0, s_fwd, n); l = d_lat + r.normal(0.0, s_lat, n)
        self.th = self.th + d_yaw + r.normal(0.0, s_yaw, n)
        ct = np.cos(self.th); st = np.sin(self.th)
        self.x = self.x + f * ct - l * st
        self.y = self.y + f * st + l * ct

    def update(self, pb):
        """Weight by p(z|x) and track the short/long-term averages that drive recovery."""
        sub = max(1, len(pb) // self.beams); pbs = pb[::sub]
        ll = self.sm.loglik_many(pbs, self.x, self.y, self.th)
        per_beam = ll / max(1, len(pbs))              # geometric mean likelihood per beam
        w = np.exp(ll - ll.max())
        tot = float(w.sum())
        self.w = np.full(self.n, 1.0 / self.n) if (not np.isfinite(tot) or tot <= 1e-300) else w / tot
        w_avg = float(np.exp(per_beam).mean())
        self.w_slow += self.a_slow * (w_avg - self.w_slow) if self.w_slow > 0 else w_avg
        self.w_fast += self.a_fast * (w_avg - self.w_fast) if self.w_fast > 0 else w_avg
        self.p_rand = 0.0 if self.w_slow <= 0 else max(0.0, 1.0 - self.w_fast / self.w_slow)
        return w_avg

    def ess(self):
        return 1.0 / float(np.sum(self.w ** 2))

    def needs_resample(self, frac=0.5):
        """Resample on a degenerate cloud OR on a collapse in likelihood. The second is the one
        that matters: when the car is kidnapped every particle is equally WRONG, so the weights
        stay uniform and the effective sample size stays at its maximum. Keying recovery off ESS
        alone means the injection can never fire in the one case it exists for."""
        return self.ess() < frac * self.n or self.p_rand > 0.05

    def resample(self, jitter_xy=0.02, jitter_th=0.004, inject_cap=0.30, yaw_prior=None, yaw_tol=0.6):
        """Low-variance resampling, with an augmented-MCL share drawn along the racing line."""
        n = self.n; r = self.rng
        n_inj = 0
        if self.cand is not None and self.p_rand > 0.0:
            n_inj = int(min(inject_cap, self.p_rand) * n)
        self.n_inject = n_inj
        keep = n - n_inj
        pos = (r.random() + np.arange(keep)) / max(1, keep)
        idx = np.clip(np.searchsorted(np.cumsum(self.w), pos), 0, n - 1)
        x = self.x[idx] + r.normal(0.0, jitter_xy, keep)
        y = self.y[idx] + r.normal(0.0, jitter_xy, keep)
        th = self.th[idx] + r.normal(0.0, jitter_th, keep)
        if n_inj > 0:
            cand = self.cand
            if yaw_prior is not None:
                # The two straights of this track are geometrically identical and are driven in
                # OPPOSITE directions, so a scan alone cannot tell them apart but the absolute IMU
                # heading can. Without this the filter happily recovers onto the wrong straight -
                # a perfect explanation of the scan, and a wall.
                dth = np.arctan2(np.sin(cand[:, 2] - yaw_prior), np.cos(cand[:, 2] - yaw_prior))
                sel = np.abs(dth) < yaw_tol
                if sel.sum() >= 8:
                    cand = cand[sel]
            j = r.integers(0, len(cand), n_inj)
            c = cand[j]
            x = np.r_[x, c[:, 0] + r.normal(0.0, 0.20, n_inj)]
            y = np.r_[y, c[:, 1] + r.normal(0.0, 0.20, n_inj)]
            th = np.r_[th, c[:, 2] + r.normal(0.0, 0.10, n_inj)]
            self.w_fast = self.w_slow          # injected: reset the trigger so it does not run away
        self.x, self.y, self.th = x, y, th
        self.w = np.full(n, 1.0 / n)

    def estimate(self):
        """Mean of the dominant cluster, not of everything: averaging across two hypotheses on
        opposite straights would return a pose on neither."""
        i = int(np.argmax(self.w))
        near = (np.hypot(self.x - self.x[i], self.y - self.y[i]) < 0.60) & \
               (np.abs(np.arctan2(np.sin(self.th - self.th[i]), np.cos(self.th - self.th[i]))) < 0.35)
        w = self.w * near
        tot = float(w.sum())
        if tot <= 1e-12:
            return (float(self.x[i]), float(self.y[i]), float(self.th[i]))
        w = w / tot
        return (float(np.sum(w * self.x)), float(np.sum(w * self.y)),
                math.atan2(float(np.sum(w * np.sin(self.th))), float(np.sum(w * np.cos(self.th)))))

    def spread(self):
        mx = float(np.sum(self.w * self.x)); my = float(np.sum(self.w * self.y))
        return float(np.sqrt(np.sum(self.w * ((self.x - mx) ** 2 + (self.y - my) ** 2))))
