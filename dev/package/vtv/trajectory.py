# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Parametric temporal trajectories of the global transform (concept sections 8-9).

The eight per-frame global parameters are

    ch 0..5  geometry : da11, a12, a21, da22 (Q16), tx, ty (Q6 pel)
    ch 6..7  photometry: dg (Q10 gain deviation), do (1/16 gray-level offset)

Over a *trajectory segment* of L frames (tau = 0..L-1) each channel is a
polynomial of order 0, 1 or 2 evaluated in pure integer arithmetic:

    v(tau) = ( (c0 << 8) + (c1 << 4) * tau + c2 * tau^2 + 128 ) >> 8

i.e. c1 has 1/16 and c2 has 1/256 unit resolution.  c0 is transmitted as a
delta against the last value of the previous segment (continuity prediction).

The encoder chooses, per segment, the *lowest* order that reproduces the
measured per-frame parameters within a displacement tolerance (in pixels,
measured at the image corners) and splits the segment recursively when no
order <= 2 suffices.  Segment length 1 is the "no trajectory" ablation: it
degenerates to per-frame parameters with the same tolerance-based snapping.
"""
import numpy as np

from .dct import qstep

NCH = 8
GEO = [0, 1, 2, 3, 4, 5]
PHOTO = [6, 7]
UNITS = np.array([65536.0, 65536.0, 65536.0, 65536.0, 64.0, 64.0, 1024.0, 16.0])


def theta_to_units(theta):
    """Float model (a11,a12,a21,a22,tx,ty,g,o) -> unrounded integer-unit targets."""
    a11, a12, a21, a22, tx, ty, g, o = theta
    return np.array([a11 - 1.0, a12, a21, a22 - 1.0, tx, ty, g - 1.0, o]) * UNITS


def eval_poly(c, order, L):
    tau = np.arange(L, dtype=np.int64)
    v = np.full(L, int(c[0]) << 8, dtype=np.int64)
    if order >= 1:
        v += (int(c[1]) << 4) * tau
    if order >= 2:
        v += int(c[2]) * tau * tau
    return (v + 128) >> 8


def eval_segment(c, og, op, geo, photo, L):
    """Integer parameter values (L, 8) of a trajectory segment (decoder path)."""
    vals = np.zeros((L, NCH), dtype=np.int64)
    if geo:
        for ch in GEO:
            vals[:, ch] = eval_poly(c[ch], og, L)
    if photo:
        for ch in PHOTO:
            vals[:, ch] = eval_poly(c[ch], op, L)
    return vals


def default_tolerances(qp):
    """(geometry tolerance in pixels, photometric tolerance in gray levels)."""
    q = qstep(qp)
    return (min(0.30, max(0.05, 0.02 + 0.0065 * q)),
            min(2.0, max(0.5, 0.4 + 0.02 * q)))


# ------------------------------------------------------------------ errors
def _geo_err(e, W, H):
    p = e / UNITS[:6]
    xs = np.array([-W / 2, W / 2, -W / 2, W / 2, 0.0])
    ys = np.array([-H / 2, -H / 2, H / 2, H / 2, 0.0])
    ex = p[:, 0:1] * xs + p[:, 1:2] * ys + p[:, 4:5]
    ey = p[:, 2:3] * xs + p[:, 3:4] * ys + p[:, 5:6]
    return float(np.sqrt(ex ** 2 + ey ** 2).max())


def _photo_err(e):
    p = e / UNITS[6:]
    return float((np.abs(p[:, 0]) * 128.0 + np.abs(p[:, 1])).max())


def _fit(y, order):
    L = len(y)
    order = min(order, L - 1)
    c = np.zeros(3, dtype=np.int64)
    if order == 0:
        c[0] = int(np.floor(y.mean() + 0.5))
        return c
    b = np.polynomial.polynomial.polyfit(np.arange(L, dtype=np.float64), y, order)
    c[0] = int(np.floor(b[0] + 0.5))
    c[1] = int(np.floor(b[1] * 16 + 0.5))
    if order >= 2:
        c[2] = int(np.floor(b[2] * 256 + 0.5))
    return c


def _min_len(order):
    return {0: 1, 1: 3, 2: 5}[order]


def _fit_group(Y, chans, prev_end, err_fn, tol):
    """Lowest order (0..2) whose integer reconstruction stays within `tol`."""
    L = Y.shape[0]
    tgt = Y[:, chans]
    for order in (0, 1, 2):
        if L < _min_len(order):
            break
        if order == 0:
            variants = [[int(prev_end[ch]) for ch in chans],
                        [0] * len(chans),
                        [int(np.floor(Y[:, ch].mean() + 0.5)) for ch in chans]]
            cands = []
            for v in variants:
                c = np.zeros((len(chans), 3), dtype=np.int64)
                c[:, 0] = v
                cands.append(c)
        else:
            cands = [np.array([_fit(Y[:, ch], order) for ch in chans])]
        for c in cands:
            vals = np.stack([eval_poly(c[k], order, L) for k in range(len(chans))], axis=1)
            if err_fn(vals - tgt) <= tol:
                return order, c, vals
    return None


def plan(targets, prev_end, geo, photo, tol_geo, tol_photo, W, H, max_len, allow_traj):
    """Split per-frame targets (n, 8) into trajectory segments.

    Returns a list of dicts: L, og, op, c (8,3) int, vals (L,8) int.
    """
    segs = []

    def rec(i, j, pe):
        L = j - i
        if (L > max_len) or (not allow_traj and L > 1):
            mid = (i + j) // 2
            return rec(mid, j, rec(i, mid, pe))
        Y = targets[i:j]
        c = np.zeros((NCH, 3), dtype=np.int64)
        vals = np.zeros((L, NCH), dtype=np.int64)
        og = op = 0
        ok = True
        if geo:
            r = _fit_group(Y, GEO, pe, lambda e: _geo_err(e, W, H), tol_geo)
            if r is None:
                ok = False
            else:
                og, cg, vg = r
                c[GEO] = cg
                vals[:, GEO] = vg
        if ok and photo:
            r = _fit_group(Y, PHOTO, pe, _photo_err, tol_photo)
            if r is None:
                ok = False
            else:
                op, cp, vp = r
                c[PHOTO] = cp
                vals[:, PHOTO] = vp
        if not ok:
            assert L > 1, "length-1 segments always fit"
            mid = (i + j) // 2
            return rec(mid, j, rec(i, mid, pe))
        segs.append(dict(L=L, og=og, op=op, c=c, vals=vals))
        return vals[-1].copy()

    if len(targets):
        rec(0, len(targets), np.asarray(prev_end, dtype=np.int64))
    return segs
