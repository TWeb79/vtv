# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Fixed-point geometric transport (normative for the decoder).

Coordinate conventions
----------------------
* Luma sample (x, y) has its centre at integer coordinates.  Frame centre
  (cx, cy) = (W//2, H//2) is the origin of the affine model.
* Positions are Q6 (1/64 pel).  Matrix entries are Q16.  Gain is Q10.
* Global model, mapping a *current* pixel to a *reference* position:
      u = cx + a11*(x+dx-cx) + a12*(y+dy-cy) + tx
      v = cy + a21*(x+dx-cx) + a22*(y+dy-cy) + ty
  where (dx, dy) is the regional delta motion of the leaf containing the pixel
  (quarter-pel units), applied *before* the global transform:  T_i = T_g o dT_i.
* 4:2:0 chroma sits at the centre of each 2x2 luma block; chroma is predicted
  by the same luma-domain transform and converted to chroma coordinates.
* Out-of-frame reads use edge replication (coordinate clamping).
* Photometric model:  luma' = g*s + o ;  chroma' = 128 + g*(s-128).

All arithmetic is int64; no floating point is used anywhere in this module.
"""
import math
from fractions import Fraction

import numpy as np

from .config import INTERP_BILINEAR, INTERP_CATMULL

PAD = 4


def _catmull_table():
    tab = []
    for k in range(64):
        t = Fraction(k, 64)
        t2, t3 = t * t, t * t * t
        w = [(-t3 + 2 * t2 - t) / 2,
             (3 * t3 - 5 * t2 + 2) / 2,
             (-3 * t3 + 4 * t2 + t) / 2,
             (t3 - t2) / 2]
        ints = [math.floor(x * 128 + Fraction(1, 2)) for x in w]
        ints[1 if k < 32 else 2] += 128 - sum(ints)      # taps must sum to 128
        tab.append(ints)
    return np.array(tab, dtype=np.int64)


def _bilinear_table():
    return np.array([[0, 128 - 2 * k, 2 * k, 0] for k in range(64)], dtype=np.int64)


TAPS = {INTERP_BILINEAR: _bilinear_table(), INTERP_CATMULL: _catmull_table()}
assert all(int(t.sum()) == 128 for tab in TAPS.values() for t in tab)


def sample(ref, u6, v6, interp=INTERP_CATMULL):
    """Separable 4-tap interpolation of `ref` (H, W) at Q6 positions (u6, v6).

    Taps are Q7; the horizontal pass keeps full precision and the vertical pass
    rounds once:  out = (sum_j ty_j * (sum_i tx_i * p_ij) + 2^13) >> 14.
    """
    ref = np.asarray(ref)
    H, W = ref.shape
    refp = np.pad(ref.astype(np.int64), PAD, mode="edge")
    flat = refp.ravel()
    Wp = W + 2 * PAD
    x0 = u6 >> 6
    y0 = v6 >> 6
    tab = TAPS[interp]
    tx = tab[u6 & 63]
    ty = tab[v6 & 63]
    x0c = np.clip(x0, -PAD + 1, W + PAD - 3) + PAD
    y0c = np.clip(y0, -PAD + 1, H + PAD - 3) + PAD
    base = (y0c - 1) * Wp + (x0c - 1)
    taps = (1, 2) if interp == INTERP_BILINEAR else (0, 1, 2, 3)
    acc = 0
    for j in taps:
        row = 0
        bj = base + j * Wp
        for i in taps:
            row = row + tx[..., i] * flat[bj + i]
        acc = acc + ty[..., j] * row
    return np.clip((acc + 8192) >> 14, 0, 255)


def luma_ref_positions(H, W, gp, dxq=None, dyq=None):
    """Reference positions (Q6) of every luma pixel under params gp and deltas."""
    da11, a12, a21, da22, tx, ty = gp[:6]
    a11 = 65536 + da11
    a22 = 65536 + da22
    cx, cy = W // 2, H // 2
    x = np.arange(W, dtype=np.int64)[None, :]
    y = np.arange(H, dtype=np.int64)[:, None]
    dx6 = ((x - cx) << 6)
    dy6 = ((y - cy) << 6)
    if dxq is not None:
        dx6 = dx6 + dxq.astype(np.int64) * 16
        dy6 = dy6 + dyq.astype(np.int64) * 16
    else:
        dx6 = np.broadcast_to(dx6, (H, W))
        dy6 = np.broadcast_to(dy6, (H, W))
    u6 = (cx << 6) + tx + ((a11 * dx6 + a12 * dy6 + 32768) >> 16)
    v6 = (cy << 6) + ty + ((a21 * dx6 + a22 * dy6 + 32768) >> 16)
    return u6, v6


def chroma_ref_positions(Hc, Wc, gp, dxq=None, dyq=None):
    """Reference positions (Q6, in chroma-pel units) of every chroma pixel.

    `Hc, Wc` are the *chroma* plane dimensions.  dxq/dyq are luma-resolution
    delta maps (H, W) = (2*Hc, 2*Wc); the value at the top-left luma pixel of
    each 2x2 group is used.
    """
    da11, a12, a21, da22, tx, ty = gp[:6]
    a11 = 65536 + da11
    a22 = 65536 + da22
    W, H = 2 * Wc, 2 * Hc
    cx, cy = W // 2, H // 2
    xc = np.arange(Wc, dtype=np.int64)[None, :]
    yc = np.arange(Hc, dtype=np.int64)[:, None]
    dx6 = ((2 * xc - cx) << 6) + 32
    dy6 = ((2 * yc - cy) << 6) + 32
    if dxq is not None:
        dx6 = dx6 + dxq[::2, ::2].astype(np.int64) * 16
        dy6 = dy6 + dyq[::2, ::2].astype(np.int64) * 16
    else:
        dx6 = np.broadcast_to(dx6, (Hc, Wc))
        dy6 = np.broadcast_to(dy6, (Hc, Wc))
    u6 = (cx << 6) + tx + ((a11 * dx6 + a12 * dy6 + 32768) >> 16)
    v6 = (cy << 6) + ty + ((a21 * dx6 + a22 * dy6 + 32768) >> 16)
    return (u6 - 32) >> 1, (v6 - 32) >> 1


def photometric(s, gp, chroma):
    dg, do = gp[6], gp[7]
    if dg == 0 and do == 0:
        return s
    G = 1024 + dg
    if chroma:
        return np.clip(128 + (((s - 128) * G + 512) >> 10), 0, 255)
    return np.clip((s * G + do * 64 + 512) >> 10, 0, 255)


def predict_luma(ref_y, gp, dxq=None, dyq=None, interp=INTERP_CATMULL):
    H, W = ref_y.shape
    u6, v6 = luma_ref_positions(H, W, gp, dxq, dyq)
    return photometric(sample(ref_y, u6, v6, interp), gp, False)


def predict_chroma(ref_c, gp, dxq=None, dyq=None, interp=INTERP_CATMULL):
    Hc, Wc = ref_c.shape
    u6, v6 = chroma_ref_positions(Hc, Wc, gp, dxq, dyq)
    return photometric(sample(ref_c, u6, v6, interp), gp, True)


IDENTITY = (0, 0, 0, 0, 0, 0, 0, 0)
