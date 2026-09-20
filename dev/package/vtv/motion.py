# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Encoder-side estimation of the global transform (non-normative, float).

Model (current pixel -> reference position, centre-origin, full-res units):
    u = cx + a11*(x-cx) + a12*(y-cy) + tx
    v = cy + a21*(x-cx) + a22*(y-cy) + ty
    cur ~= g * ref(u, v) + o
solved coarse-to-fine with robust (Cauchy-weighted) Gauss-Newton /
Levenberg-Marquardt.  Photometric parameters are estimated jointly so that
brightness changes are not mistaken for motion (concept section 11).
"""
import numpy as np

IDENT = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])


def _down2(a):
    H, W = a.shape
    h, w = H // 2, W // 2
    return a[: h * 2, : w * 2].reshape(h, 2, w, 2).mean(axis=(1, 3))


def _bilin(img, u, v):
    H, W = img.shape
    uc = np.clip(u, 0, W - 1.000001)
    vc = np.clip(v, 0, H - 1.000001)
    x0 = uc.astype(np.int64)
    y0 = vc.astype(np.int64)
    fx = uc - x0
    fy = vc - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _levels_for(shape):
    lv, m = 1, min(shape)
    while m >= 80 and lv < 4:
        m //= 2
        lv += 1
    return lv


def estimate_global(cur, ref, photo=True, init=None):
    """Return (theta[8], info) with info = dict(mad, mad0, valid).

    mad  : mean |g*ref(warp) + o - cur| over the valid area at full resolution
    mad0 : same for the identity transform (used for scene-cut detection)
    """
    cur = np.asarray(cur, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    H, W = cur.shape
    cx, cy = W // 2, H // 2
    L = _levels_for(cur.shape)
    pc, pr = [cur], [ref]
    for _ in range(L - 1):
        pc.append(_down2(pc[-1]))
        pr.append(_down2(pr[-1]))

    th = IDENT.copy() if init is None else np.array(init, dtype=np.float64)
    free = [0, 1, 2, 3, 4, 5] + ([6, 7] if photo else [])
    iters = {0: 6, 1: 8, 2: 10, 3: 10}

    def warp_res(th, lvl, want_grad):
        c, r = pc[lvl], pr[lvl]
        s = 2.0 ** lvl
        h, w = c.shape
        X = s * (np.arange(w)[None, :] + 0.5) - 0.5 - cx
        Y = s * (np.arange(h)[:, None] + 0.5) - 0.5 - cy
        Ux = cx + th[0] * X + th[1] * Y + th[4]
        Vy = cy + th[2] * X + th[3] * Y + th[5]
        u = (Ux + 0.5) / s - 0.5
        v = (Vy + 0.5) / s - 0.5
        m = (u >= 2) & (u <= w - 3) & (v >= 2) & (v <= h - 3)
        Iw = _bilin(r, u, v)
        res = th[6] * Iw + th[7] - c
        if not want_grad:
            return res, m, None
        gy, gx = np.gradient(r)
        Ix, Iy = _bilin(gx, u, v), _bilin(gy, u, v)
        g = th[6]
        J = np.stack([np.broadcast_to(g * Ix * X / s, res.shape),
                      np.broadcast_to(g * Ix * Y / s, res.shape),
                      np.broadcast_to(g * Iy * X / s, res.shape),
                      np.broadcast_to(g * Iy * Y / s, res.shape),
                      g * Ix / s, g * Iy / s, Iw, np.ones_like(Iw)], axis=-1)
        return res, m, J

    def robust_cost(res, m):
        r = res[m]
        if r.size == 0:
            return 1e30
        sig = max(1.4826 * np.median(np.abs(r)), 1.0)
        return float(np.mean(np.log1p((r / (2.0 * sig)) ** 2))) * sig * sig

    for lvl in range(L - 1, -1, -1):
        for _ in range(iters[lvl]):
            res, m, J = warp_res(th, lvl, True)
            r = res[m]
            if r.size < 64:
                break
            sig = max(1.4826 * np.median(np.abs(r)), 1.0)
            w = 1.0 / (1.0 + (r / (2.0 * sig)) ** 2)
            Jm = J[m][:, free]
            A = Jm.T @ (w[:, None] * Jm)
            b = Jm.T @ (w * r)
            sc = 1.0 / np.sqrt(np.diag(A) + 1e-9)
            A_s = A * sc[:, None] * sc[None, :] + 1e-3 * np.eye(len(free))
            try:
                d = -np.linalg.solve(A_s, b * sc) * sc
            except np.linalg.LinAlgError:
                break
            c0 = robust_cost(res, m)
            step = 1.0
            accepted = False
            for _b in range(4):
                th2 = th.copy()
                th2[free] += step * d
                res2, m2, _ = warp_res(th2, lvl, False)
                if robust_cost(res2, m2) < c0:
                    th = th2
                    accepted = True
                    break
                step *= 0.5
            if not accepted or np.abs(d).max() < 1e-5:
                break

    # full-resolution diagnostics; fall back to identity if estimation made things worse
    res, m, _ = warp_res(th, 0, False)
    res0, m0, _ = warp_res(IDENT, 0, False)
    mad = float(np.abs(res[m]).mean()) if m.any() else 1e9
    mad0 = float(np.abs(res0[m0]).mean()) if m0.any() else 1e9
    if robust_cost(res, m) > robust_cost(res0, m0):
        th, mad = IDENT.copy(), mad0
    return th, dict(mad=mad, mad0=mad0)
