# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Integer 8x8 transform, scan order and scalar quantization.

Everything on the *decoder* path here is pure integer arithmetic so that any
implementation (Python, WASM, native) reproduces identical pixels.

The transform is the HEVC 8-point core matrix scaled so that coefficients are
in orthonormal units (DC of a flat 8x8 block of value v is 8*v).  This makes
squared error in the coefficient domain ~= squared error in the pixel domain,
which the encoder's rate-distortion optimisation relies on.
"""
import numpy as np

M8 = np.array([
    [64,  64,  64,  64,  64,  64,  64,  64],
    [89,  75,  50,  18, -18, -50, -75, -89],
    [83,  36, -36, -83, -83, -36,  36,  83],
    [75, -18, -89, -50,  50,  89,  18, -75],
    [64, -64, -64,  64,  64, -64, -64,  64],
    [50, -89,  18,  75, -75, -18,  89, -50],
    [36, -83,  83, -36, -36,  83, -83,  36],
    [18, -50,  75, -89,  89, -75,  50, -18],
], dtype=np.int64)
M8T = np.ascontiguousarray(M8.T)


def fdct8(blocks):
    """Forward transform of (..., 8, 8) integer blocks -> int64 coefficients."""
    b = np.asarray(blocks, dtype=np.int64)
    t = (M8 @ b + 64) >> 7          # vertical
    return (t @ M8T + 128) >> 8     # horizontal


def idct8(coefs):
    """Inverse transform of (..., 8, 8) integer coefficients -> int64 samples."""
    c = np.asarray(coefs, dtype=np.int64)
    t = (M8T @ c + 64) >> 7
    return (t @ M8 + 128) >> 8


def blockify(plane):
    """(H, W) -> (H/8, W/8, 8, 8) view."""
    h, w = plane.shape
    return plane.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


def unblockify(blocks):
    nby, nbx = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(nby * 8, nbx * 8)


def _zigzag():
    order = []
    for s in range(15):
        if s % 2 == 0:
            rows = range(min(s, 7), max(0, s - 7) - 1, -1)
        else:
            rows = range(max(0, s - 7), min(s, 7) + 1)
        for r in rows:
            order.append(r * 8 + (s - r))
    return np.array(order, dtype=np.int64)


ZZ = _zigzag()                       # ZZ[k] = raster index of k-th coefficient in scan
ZZ_INV = np.argsort(ZZ)              # raster index -> scan position
assert sorted(ZZ.tolist()) == list(range(64))

# ---- quantisation --------------------------------------------------------
# Qstep(qp) = QBASE[qp % 6] * 2^(qp // 6) / 64   (same progression as HEVC:
# the step doubles every 6 QP).  Stored in Q6 fixed point.
QBASE = (40, 45, 51, 57, 64, 72)


def qstep_q6(qp):
    return QBASE[qp % 6] << (qp // 6)


def qstep(qp):
    return qstep_q6(qp) / 64.0


def quantize(coefs, qp, theta_num=1, theta_den=3):
    """Dead-zone scalar quantiser (encoder only).  level = floor(|c|/step + theta)."""
    step = qstep_q6(qp)
    a = np.abs(coefs)
    lv = (a * 64 + (step * theta_num) // theta_den) // step
    return np.where(coefs < 0, -lv, lv)


def dequantize(levels, qp):
    """Normative reconstruction: sign * ((|level| * step_q6 + 32) >> 6)."""
    step = qstep_q6(qp)
    a = np.abs(levels).astype(np.int64)
    r = (a * step + 32) >> 6
    return np.where(levels < 0, -r, r)
