r"""VTV - Vector-Time Video (beta)

# Placeholder
This is a temporary documentation block used while assembling the file.
"""

import io as _stdio
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import zlib
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

DOC = __doc__

# =============================================================================
# Configuration and constants
# Shared constants and the encoder configuration for VTV beta.
# =============================================================================
VERSION = (0, 2)          # bitstream version (major, minor)
CTU = 32                  # quadtree root size (luma pixels)
MIN_LEAF = 8              # smallest leaf; also the residual-transform block size
UNIT = 16                 # luma area that maps to one 8x8 chroma block

# header flag bits
FLAG_GEO = 1              # global affine trajectory present
FLAG_PHOTO = 2            # photometric (gain/offset) trajectory present
FLAG_INTERP_SHIFT = 2     # bits 2-3: interpolation filter id
INTERP_BILINEAR = 0
INTERP_CATMULL = 1


@dataclass
class EncoderConfig:
    qp: int = 32                  # base quantizer (P frames)
    qp_i_offset: int = -3         # I frames use qp + offset
    gop: int = 32                 # max frames per independently decodable segment
    use_global: bool = True       # global affine transform trajectory
    use_traj: bool = True         # polynomial temporal trajectories (False = per-frame params)
    use_photo: bool = True        # gain/offset trajectory
    interp: int = INTERP_CATMULL
    search_range: int = 8         # integer full-search range for regional deltas (pixels)
    subpel: bool = True           # 1/2 and 1/4 pel refinement
    traj_max_len: int = 16        # max frames per trajectory segment
    tol_geo: float = -1.0         # pixels; <0 = derived from qp
    tol_photo: float = -1.0       # gray levels; <0 = derived from qp
    lam_scale: float = 0.13       # lambda = lam_scale * qstep^2
    cut_mad: float = 18.0         # scene-cut threshold on compensated MAD
    rdoq: bool = True             # block-level zero-out decision
    stats: bool = True            # collect per-category bit statistics

    def qp_i(self) -> int:
        return max(0, min(51, self.qp + self.qp_i_offset))


# =============================================================================
# Transform / quantisation
# Integer 8x8 transform, scan order and scalar quantization.
#
# Everything on the *decoder* path here is pure integer arithmetic so that any
# implementation (Python, WASM, native) reproduces identical pixels.
#
# The transform is the HEVC 8-point core matrix scaled so that coefficients are
# in orthonormal units (DC of a flat 8x8 block of value v is 8*v).  This makes
# squared error in the coefficient domain ~= squared error in the pixel domain,
# which the encoder's rate-distortion optimisation relies on.
# =============================================================================
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


# =============================================================================
# Entropy coder
# Adaptive binary range coder (LZMA-style) with a *symmetric* API.
#
# `RangeEncoder` and `RangeDecoder` expose identical methods.  Every method takes
# the value to encode and *returns the value* (encoder: the value passed in,
# decoder: the value decoded, argument ignored).  This lets the bitstream syntax
# be written exactly once and used for both directions, which removes a whole
# class of encoder/decoder divergence bugs.
#
# Probabilities are 11-bit (P(bit==0) in 1..2047), adaptation shift 5, exactly as
# in LZMA, so the coder is fully integer and deterministic.
# =============================================================================
PROB_BITS = 11
PROB_INIT = 1 << (PROB_BITS - 1)
MOVE = 5
TOP = 1 << 24
MASK32 = 0xFFFFFFFF
MAX_PREFIX = 32


class BitstreamError(Exception):
    pass


# cost (bits) of coding a symbol with probability p/2048, for statistics only
_COST0 = [0.0] * (1 << PROB_BITS)
_COST1 = [0.0] * (1 << PROB_BITS)
for _p in range(1, 1 << PROB_BITS):
    _COST0[_p] = -math.log2(_p / (1 << PROB_BITS))
    _COST1[_p] = -math.log2(1.0 - _p / (1 << PROB_BITS))


def new_ctx(n):
    return [PROB_INIT] * n


class _Sym:
    """Binarisations shared by encoder and decoder (built on `bit`/`direct`)."""

    def bittree(self, ctx, nbits, v):
        idx = 1
        for k in range(nbits - 1, -1, -1):
            b = self.bit(ctx, idx, (v >> k) & 1)
            idx = (idx << 1) | b
        return idx - (1 << nbits)

    def ue(self, ctx, v, off=0):
        """Order-0 Exp-Golomb with adaptive prefix contexts ctx[off:]."""
        cap = len(ctx) - off - 1
        n = (v + 1).bit_length() - 1 if self.is_enc else 0
        k = 0
        while True:
            more = self.bit(ctx, off + (k if k < cap else cap), 1 if k < n else 0)
            if not more:
                break
            k += 1
            if k > MAX_PREFIX:
                raise BitstreamError("exp-golomb prefix too long")
        suffix = self.direct(k, (v + 1) - (1 << k)) if k else 0
        return (1 << k) + suffix - 1

    def se(self, ctx, v):
        """Signed value: ctx layout = [zero_flag, sign, prefix...]."""
        nz = self.bit(ctx, 0, 1 if v != 0 else 0)
        if not nz:
            return 0
        neg = self.bit(ctx, 1, 1 if v < 0 else 0)
        m = self.ue(ctx, abs(v) - 1, off=2) + 1
        return -m if neg else m


class RangeEncoder(_Sym):
    is_enc = True

    def __init__(self, stats=False):
        self.low = 0
        self.range = MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()
        self.stats = stats
        self.cat = "misc"
        self.bits = defaultdict(float)

    def _shift_low(self):
        low = self.low
        if low < 0xFF000000 or low >= 0x100000000:
            carry = low >> 32
            temp = self.cache
            out = self.out
            while True:
                out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (low & 0x00FFFFFF) << 8

    def bit(self, ctx, i, b):
        p = ctx[i]
        bound = (self.range >> PROB_BITS) * p
        if b:
            self.low += bound
            self.range -= bound
            ctx[i] = p - (p >> MOVE)
            if self.stats:
                self.bits[self.cat] += _COST1[p]
        else:
            self.range = bound
            ctx[i] = p + (((1 << PROB_BITS) - p) >> MOVE)
            if self.stats:
                self.bits[self.cat] += _COST0[p]
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self._shift_low()
        return b

    def direct(self, nbits, v):
        for k in range(nbits - 1, -1, -1):
            self.range >>= 1
            if (v >> k) & 1:
                self.low += self.range
            while self.range < TOP:
                self.range = (self.range << 8) & MASK32
                self._shift_low()
        if self.stats:
            self.bits[self.cat] += nbits
        return v

    def finish(self):
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder(_Sym):
    is_enc = False

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos
        self.range = MASK32
        self.code = 0
        self.cat = "misc"
        if len(data) - pos < 5:
            raise BitstreamError("range coder payload too short")
        if data[pos] != 0:
            raise BitstreamError("bad range coder start byte")
        for _ in range(5):
            self.code = ((self.code << 8) | data[self.pos]) & MASK32
            self.pos += 1

    def _next(self):
        p = self.pos
        self.pos = p + 1
        if p < len(self.data):
            return self.data[p]
        if p > len(self.data) + 16:
            raise BitstreamError("range decoder ran past end of payload")
        return 0

    def bit(self, ctx, i, _b=0):
        if self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self.code = ((self.code << 8) | self._next()) & MASK32
        p = ctx[i]
        bound = (self.range >> PROB_BITS) * p
        if self.code < bound:
            self.range = bound
            ctx[i] = p + (((1 << PROB_BITS) - p) >> MOVE)
            return 0
        self.code -= bound
        self.range -= bound
        ctx[i] = p - (p >> MOVE)
        return 1

    def direct(self, nbits, _v=0):
        v = 0
        for _ in range(nbits):
            if self.range < TOP:
                self.range = (self.range << 8) & MASK32
                self.code = ((self.code << 8) | self._next()) & MASK32
            self.range >>= 1
            if self.code >= self.range:
                self.code -= self.range
                v = (v << 1) | 1
            else:
                v <<= 1
        return v


# =============================================================================
# Geometric transport (warping)
# Fixed-point geometric transport (normative for the decoder).
#
# Coordinate conventions
# ----------------------
# * Luma sample (x, y) has its centre at integer coordinates.  Frame centre
#   (cx, cy) = (W//2, H//2) is the origin of the affine model.
# * Positions are Q6 (1/64 pel).  Matrix entries are Q16.  Gain is Q10.
# * Global model, mapping a *current* pixel to a *reference* position:
#       u = cx + a11*(x+dx-cx) + a12*(y+dy-cy) + tx
#       v = cy + a21*(x+dx-cx) + a22*(y+dy-cy) + ty
#   where (dx, dy) is the regional delta motion of the leaf containing the pixel
#   (quarter-pel units), applied *before* the global transform:  T_i = T_g o dT_i.
# * 4:2:0 chroma sits at the centre of each 2x2 luma block; chroma is predicted
#   by the same luma-domain transform and converted to chroma coordinates.
# * Out-of-frame reads use edge replication (coordinate clamping).
# * Photometric model:  luma' = g*s + o ;  chroma' = 128 + g*(s-128).
#
# All arithmetic is int64; no floating point is used anywhere in this module.
# =============================================================================
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


# =============================================================================
# Frames, colour conversion, y4m
# Frame containers, colour conversion and YUV4MPEG2 (.y4m) I/O.
#
# Internal frame representation: a tuple (Y, Cb, Cr) of int32 arrays, 4:2:0,
# 8-bit, full-range BT.601-style YCbCr, padded to a multiple of CTU.
# =============================================================================
def coded_size(width, height):
    c = CTU
    return ((width + c - 1) // c * c, (height + c - 1) // c * c)


def pad_planes(planes, width, height):
    """Edge-replicate visible planes up to the coded size."""
    Y, Cb, Cr = planes
    Wc, Hc = coded_size(width, height)
    Y = np.pad(Y, ((0, Hc - Y.shape[0]), (0, Wc - Y.shape[1])), mode="edge")
    ch, cw = Hc // 2, Wc // 2
    Cb = np.pad(Cb, ((0, ch - Cb.shape[0]), (0, cw - Cb.shape[1])), mode="edge")
    Cr = np.pad(Cr, ((0, ch - Cr.shape[0]), (0, cw - Cr.shape[1])), mode="edge")
    return Y.astype(np.int32), Cb.astype(np.int32), Cr.astype(np.int32)


def crop_planes(planes, width, height):
    Y, Cb, Cr = planes
    cw, ch = (width + 1) // 2, (height + 1) // 2
    return (Y[:height, :width].astype(np.uint8),
            Cb[:ch, :cw].astype(np.uint8),
            Cr[:ch, :cw].astype(np.uint8))


def rgb_to_planes(rgb):
    """uint8 (H, W, 3) RGB -> visible (Y, Cb, Cr) planes, 4:2:0 (2x2 box chroma)."""
    H, W, _ = rgb.shape
    assert H % 2 == 0 and W % 2 == 0, "even dimensions required"
    f = rgb.astype(np.float64)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128.0 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128.0 + 0.5 * R - 0.418688 * G - 0.081312 * B
    sub = lambda c: c.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    q = lambda a: np.clip(np.floor(a + 0.5), 0, 255).astype(np.uint8)
    return q(Y), q(sub(Cb)), q(sub(Cr))


def _up2(c):
    """'Fancy' 2x chroma upsampling (triangle filter), edge-replicated."""
    c = c.astype(np.float64)
    p = np.pad(c, ((0, 0), (1, 1)), mode="edge")
    out = np.empty((c.shape[0], c.shape[1] * 2))
    out[:, 0::2] = 0.75 * c + 0.25 * p[:, :-2]
    out[:, 1::2] = 0.75 * c + 0.25 * p[:, 2:]
    p = np.pad(out, ((1, 1), (0, 0)), mode="edge")
    res = np.empty((c.shape[0] * 2, out.shape[1]))
    res[0::2] = 0.75 * out + 0.25 * p[:-2]
    res[1::2] = 0.75 * out + 0.25 * p[2:]
    return res


def planes_to_rgb(planes):
    """Visible (Y, Cb, Cr) planes -> uint8 RGB (display only, not normative)."""
    Y, Cb, Cr = planes
    Yf = Y.astype(np.float64)
    cb = _up2(Cb)[: Y.shape[0], : Y.shape[1]] - 128.0
    cr = _up2(Cr)[: Y.shape[0], : Y.shape[1]] - 128.0
    R = Yf + 1.402 * cr
    G = Yf - 0.344136 * cb - 0.714136 * cr
    B = Yf + 1.772 * cb
    return np.clip(np.floor(np.stack([R, G, B], -1) + 0.5), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ y4m
def write_y4m(path, frames, fps=(25, 1)):
    """frames: iterable of visible (Y, Cb, Cr) uint8 planes."""
    frames = list(frames)
    H, W = frames[0][0].shape
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{fps[0]}:{fps[1]} Ip A1:1 C420jpeg\n".encode())
        for Y, Cb, Cr in frames:
            f.write(b"FRAME\n")
            f.write(np.ascontiguousarray(Y, dtype=np.uint8).tobytes())
            f.write(np.ascontiguousarray(Cb, dtype=np.uint8).tobytes())
            f.write(np.ascontiguousarray(Cr, dtype=np.uint8).tobytes())


def read_y4m(path, max_frames=None):
    """Return (frames, width, height, (fps_num, fps_den)); 4:2:0 only."""
    with open(path, "rb") as f:
        header = f.readline().decode().strip().split()
        assert header[0] == "YUV4MPEG2"
        W = H = None
        fps = (25, 1)
        for tok in header[1:]:
            if tok[0] == "W":
                W = int(tok[1:])
            elif tok[0] == "H":
                H = int(tok[1:])
            elif tok[0] == "F":
                a, b = tok[1:].split(":")
                fps = (int(a), int(b))
            elif tok[0] == "C" and not tok.startswith("C420"):
                raise ValueError(f"unsupported chroma format {tok}")
        ys, cs = W * H, ((W + 1) // 2) * ((H + 1) // 2)
        frames = []
        while max_frames is None or len(frames) < max_frames:
            line = f.readline()
            if not line:
                break
            assert line.startswith(b"FRAME")
            buf = f.read(ys + 2 * cs)
            if len(buf) < ys + 2 * cs:
                break
            a = np.frombuffer(buf, dtype=np.uint8)
            Y = a[:ys].reshape(H, W)
            Cb = a[ys:ys + cs].reshape((H + 1) // 2, (W + 1) // 2)
            Cr = a[ys + cs:].reshape((H + 1) // 2, (W + 1) // 2)
            frames.append((Y.copy(), Cb.copy(), Cr.copy()))
    return frames, W, H, fps


# =============================================================================
# Temporal trajectories
# Parametric temporal trajectories of the global transform (concept sections 8-9).
#
# The eight per-frame global parameters are
#
#     ch 0..5  geometry : da11, a12, a21, da22 (Q16), tx, ty (Q6 pel)
#     ch 6..7  photometry: dg (Q10 gain deviation), do (1/16 gray-level offset)
#
# Over a *trajectory segment* of L frames (tau = 0..L-1) each channel is a
# polynomial of order 0, 1 or 2 evaluated in pure integer arithmetic:
#
#     v(tau) = ( (c0 << 8) + (c1 << 4) * tau + c2 * tau^2 + 128 ) >> 8
#
# i.e. c1 has 1/16 and c2 has 1/256 unit resolution.  c0 is transmitted as a
# delta against the last value of the previous segment (continuity prediction).
#
# The encoder chooses, per segment, the *lowest* order that reproduces the
# measured per-frame parameters within a displacement tolerance (in pixels,
# measured at the image corners) and splits the segment recursively when no
# order <= 2 suffices.  Segment length 1 is the "no trajectory" ablation: it
# degenerates to per-frame parameters with the same tolerance-based snapping.
# =============================================================================
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


# =============================================================================
# Global motion estimation (encoder only)
# Encoder-side estimation of the global transform (non-normative, float).
#
# Model (current pixel -> reference position, centre-origin, full-res units):
#     u = cx + a11*(x-cx) + a12*(y-cy) + tx
#     v = cy + a21*(x-cx) + a22*(y-cy) + ty
#     cur ~= g * ref(u, v) + o
# solved coarse-to-fine with robust (Cauchy-weighted) Gauss-Newton /
# Levenberg-Marquardt.  Photometric parameters are estimated jointly so that
# brightness changes are not mistaken for motion (concept section 11).
# =============================================================================
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


# =============================================================================
# Bitstream syntax
# Bitstream syntax, written once and used for both writing and reading.
#
# Every function takes a range coder `rc` (RangeEncoder or RangeDecoder) and a
# `FrameSyntax` `fs`.  When encoding, `fs` holds the data to write; when decoding,
# `fs` starts zeroed and is filled in as symbols are decoded.  Values passed to
# `rc.*` are ignored by the decoder, so no separate "reader" exists.
#
# Frame structure
# ---------------
#   I frame : every 16x16 unit is intra; no partition syntax.
#   P frame : per 32x32 CTU a quadtree (leaves 32/16/8) with per-leaf mode
#             SKIP  (global transform only, no residual),
#             INTER (global transform + regional delta motion + residual),
#             INTRA (only leaves >= 16: "new information", spatial prediction).
#   Residual: 8x8 blocks, coded per CTU (luma), then chroma per 16x16 unit.
# =============================================================================
MODE_SKIP, MODE_INTER, MODE_INTRA = 0, 1, 2


class Contexts:
    """All adaptive probability models; reset at every segment start."""

    def __init__(self):
        self.split = new_ctx(2 * 3)
        self.skip = new_ctx(3 * 3)
        self.intra = new_ctx(3)
        self.mvd = [new_ctx(20), new_ctx(20)]
        self.imode = new_ctx(4)
        self.cbf = new_ctx(4 * 3)
        self.last = [new_ctx(64) for _ in range(4)]
        self.sig = new_ctx(4 * 64 * 3)
        self.gt1 = new_ctx(4 * 4)
        self.gt2 = new_ctx(4 * 4)
        self.rem = [new_ctx(20) for _ in range(4)]
        self.t_len = new_ctx(20)
        self.t_ord = new_ctx(4)
        self.t_coef = [[new_ctx(20) for _ in range(3)] for _ in range(NCH)]


class FrameSyntax:
    """Everything the decoder needs to reconstruct one frame (besides the reference)."""

    def __init__(self, ftype, Hc, Wc, qp):
        self.ftype = ftype                       # 'I' or 'P'
        self.qp = qp
        self.Hb, self.Wb = Hc // 8, Wc // 8      # luma 8x8 block grid
        Hb, Wb = self.Hb, self.Wb
        self.lsize = np.zeros((Hb, Wb), np.uint8)       # leaf size in pixels
        self.mode = np.zeros((Hb, Wb), np.uint8)
        self.dx = np.zeros((Hb, Wb), np.int32)          # regional delta, quarter-pel
        self.dy = np.zeros((Hb, Wb), np.int32)
        self.lev_y = np.zeros((Hb, Wb, 64), np.int32)   # quantised levels, scan order
        self.lev_cb = np.zeros((Hb // 2, Wb // 2, 64), np.int32)
        self.lev_cr = np.zeros((Hb // 2, Wb // 2, 64), np.int32)
        self.imode_y = np.zeros((Hb, Wb), np.uint8)
        self.imode_cb = np.zeros((Hb // 2, Wb // 2), np.uint8)
        self.imode_cr = np.zeros((Hb // 2, Wb // 2), np.uint8)


class _State:
    def __init__(self, fs):
        self.cbf_y = np.zeros((fs.Hb, fs.Wb), np.uint8)
        self.cbf_cb = np.zeros((fs.Hb // 2, fs.Wb // 2), np.uint8)
        self.cbf_cr = np.zeros((fs.Hb // 2, fs.Wb // 2), np.uint8)


# ------------------------------------------------------------ coefficient block
def code_block(rc, cx, ctype, nbr, lv):
    """One 8x8 block of quantised levels in scan order.

    lv: int array (64,) when encoding, None when decoding.
    Returns the list of 64 levels, or None for an all-zero block.
    """
    if rc.is_enc:
        lvl = lv.tolist()
        nz = [i for i, v in enumerate(lvl) if v]
        cbf_e = 1 if nz else 0
        last_e = nz[-1] if nz else 0
    else:
        lvl, cbf_e, last_e = None, 0, 0
    if not rc.bit(cx.cbf, ctype * 3 + nbr, cbf_e):
        return None
    last = rc.bittree(cx.last[ctype], 6, last_e)
    out = [0] * 64
    nnz = ngt1 = p1 = p2 = 0
    sig_base = ctype * 192
    for i in range(last + 1):
        v_e = lvl[i] if lvl is not None else 0
        if i < last:
            sig = rc.bit(cx.sig, sig_base + i * 3 + p1 + p2, 1 if v_e else 0)
        else:
            sig = 1
        if sig:
            a_e = abs(v_e)
            a = 1
            if rc.bit(cx.gt1, ctype * 4 + (nnz if nnz < 3 else 3), 1 if a_e > 1 else 0):
                a = 2
                if rc.bit(cx.gt2, ctype * 4 + (ngt1 if ngt1 < 3 else 3), 1 if a_e > 2 else 0):
                    a = 3 + rc.ue(cx.rem[ctype], a_e - 3)
                ngt1 += 1
            neg = rc.direct(1, 1 if v_e < 0 else 0)
            out[i] = -a if neg else a
            nnz += 1
        p2 = p1
        p1 = 1 if sig else 0
    return out


def _code_imode(rc, cx, pt, m):
    if rc.bit(cx.imode, pt * 2, 1 if m == 0 else 0):
        return 0
    return 1 if rc.bit(cx.imode, pt * 2 + 1, 1 if m == 1 else 0) else 2


# ------------------------------------------------------------ partition / modes
def _mv_pred(fs, y8, x8):
    def get(yy, xx):
        if yy < 0 or xx < 0:
            return (0, 0)
        return (int(fs.dx[yy, xx]), int(fs.dy[yy, xx]))
    l, t, tl = get(y8, x8 - 1), get(y8 - 1, x8), get(y8 - 1, x8 - 1)
    return (sorted((l[0], t[0], tl[0]))[1], sorted((l[1], t[1], tl[1]))[1])


def _quadtree(rc, cx, fs, x8, y8, s8, depth):
    size = s8 * 8
    if s8 > 1:
        left = int(fs.lsize[y8, x8 - 1]) if x8 > 0 else 0
        top = int(fs.lsize[y8 - 1, x8]) if y8 > 0 else 0
        inc = (1 if 0 < left < size else 0) + (1 if 0 < top < size else 0)
        split = rc.bit(cx.split, depth * 3 + inc, 1 if fs.lsize[y8, x8] < size else 0)
        if split:
            h = s8 // 2
            for oy in (0, h):
                for ox in (0, h):
                    _quadtree(rc, cx, fs, x8 + ox, y8 + oy, h, depth + 1)
            return
    lm = int(fs.mode[y8, x8 - 1]) if x8 > 0 else -1
    tm = int(fs.mode[y8 - 1, x8]) if y8 > 0 else -1
    inc = (1 if lm == MODE_SKIP else 0) + (1 if tm == MODE_SKIP else 0)
    rc.cat = "struct"
    mode_e = int(fs.mode[y8, x8])
    if rc.bit(cx.skip, depth * 3 + inc, 1 if mode_e == MODE_SKIP else 0):
        mode = MODE_SKIP
    else:
        is_intra = 0
        if size >= 16:
            inc2 = (1 if lm == MODE_INTRA else 0) + (1 if tm == MODE_INTRA else 0)
            is_intra = rc.bit(cx.intra, inc2, 1 if mode_e == MODE_INTRA else 0)
        mode = MODE_INTRA if is_intra else MODE_INTER
    dx = dy = 0
    if mode == MODE_INTER:
        rc.cat = "mvd"
        px, py = _mv_pred(fs, y8, x8)
        dx = rc.se(cx.mvd[0], int(fs.dx[y8, x8]) - px) + px
        dy = rc.se(cx.mvd[1], int(fs.dy[y8, x8]) - py) + py
        if abs(dx) > 1 << 12 or abs(dy) > 1 << 12:
            raise BitstreamError("delta motion out of range")
    sl = (slice(y8, y8 + s8), slice(x8, x8 + s8))
    fs.lsize[sl] = size
    fs.mode[sl] = mode
    fs.dx[sl] = dx
    fs.dy[sl] = dy


# ------------------------------------------------------------ residual
def _residual_ctu(rc, cx, fs, st, cy, cxx):
    enc = rc.is_enc
    for by in range(cy * 4, cy * 4 + 4):
        for bx in range(cxx * 4, cxx * 4 + 4):
            m = int(fs.mode[by, bx])
            if m == MODE_SKIP:
                continue
            intra = 1 if m == MODE_INTRA else 0
            rc.cat = "intra" if intra else "res_y"
            if intra:
                fs.imode_y[by, bx] = _code_imode(rc, cx, 0, int(fs.imode_y[by, bx]))
            nbr = (int(st.cbf_y[by, bx - 1]) if bx > 0 else 0) + (int(st.cbf_y[by - 1, bx]) if by > 0 else 0)
            out = code_block(rc, cx, intra, nbr, fs.lev_y[by, bx] if enc else None)
            if out is not None:
                st.cbf_y[by, bx] = 1
                if not enc:
                    fs.lev_y[by, bx, :] = out
    for uy in range(cy * 2, cy * 2 + 2):
        for ux in range(cxx * 2, cxx * 2 + 2):
            ms = fs.mode[2 * uy:2 * uy + 2, 2 * ux:2 * ux + 2]
            if (ms == MODE_INTRA).any():
                intra = 1
            elif (ms != MODE_SKIP).any():
                intra = 0
            else:
                continue
            rc.cat = "intra" if intra else "res_c"
            for lev, cbfm, imode, pt in ((fs.lev_cb, st.cbf_cb, fs.imode_cb, 1),
                                         (fs.lev_cr, st.cbf_cr, fs.imode_cr, 1)):
                if intra:
                    imode[uy, ux] = _code_imode(rc, cx, pt, int(imode[uy, ux]))
                nbr = (int(cbfm[uy, ux - 1]) if ux > 0 else 0) + (int(cbfm[uy - 1, ux]) if uy > 0 else 0)
                out = code_block(rc, cx, 2 + intra, nbr, lev[uy, ux] if enc else None)
                if out is not None:
                    cbfm[uy, ux] = 1
                    if not enc:
                        lev[uy, ux, :] = out


def code_frame(rc, cx, fs):
    """Code (or parse) one complete frame."""
    st = _State(fs)
    for cy in range(fs.Hb // 4):
        for cxx in range(fs.Wb // 4):
            if fs.ftype == "P":
                _quadtree(rc, cx, fs, cxx * 4, cy * 4, 4, 0)
            else:
                sl = (slice(cy * 4, cy * 4 + 4), slice(cxx * 4, cxx * 4 + 4))
                fs.lsize[sl] = 32
                fs.mode[sl] = MODE_INTRA
            _residual_ctu(rc, cx, fs, st, cy, cxx)


# ------------------------------------------------------------ trajectory block
def code_traj(rc, cx, seg, prev_end, geo, photo, frames_left):
    """Code one trajectory segment.  `seg` is a dict when encoding, None when decoding.

    Returns dict(L, og, op, c, vals).  `prev_end` (8,) are the parameter values of
    the last frame of the previous segment (delta base for c0).
    """
    enc = rc.is_enc
    rc.cat = "traj"
    L = rc.ue(cx.t_len, (seg["L"] - 1) if enc else 0) + 1
    if L > frames_left:
        raise BitstreamError("trajectory segment longer than remaining frames")
    og = op = 0
    c = np.zeros((NCH, 3), dtype=np.int64)
    for group, on, ob in ((GEO, geo, 0), (PHOTO, photo, 2)):
        if not on:
            continue
        o_e = (seg["og"] if group is GEO else seg["op"]) if enc else 0
        o = 0
        if rc.bit(cx.t_ord, ob, 1 if o_e > 0 else 0):
            o = 1 + rc.bit(cx.t_ord, ob + 1, 1 if o_e > 1 else 0)
        for ch in group:
            for j in range(o + 1):
                v_e = int(seg["c"][ch][j]) if enc else 0
                if j == 0:
                    c[ch][0] = rc.se(cx.t_coef[ch][0], v_e - int(prev_end[ch])) + int(prev_end[ch])
                else:
                    c[ch][j] = rc.se(cx.t_coef[ch][j], v_e)
        if group is GEO:
            og = o
        else:
            op = o
    vals = eval_segment(c, og, op, geo, photo, L)
    return dict(L=L, og=og, op=op, c=c, vals=vals)


# =============================================================================
# Shared reconstruction
# Frame reconstruction shared by the encoder (closed loop) and the decoder.
#
# Because both sides call exactly this code, encoder-side reconstruction and
# decoder output are bit-identical by construction (verified by the tests).
#
# Pass A  : inter/skip areas   = warp(reference) + dequantised residual
# Pass B  : intra ("new information") blocks, raster order, from neighbouring
#           reconstructed samples (DC / horizontal / vertical prediction).
# =============================================================================
def levels_to_residual(lev, qp):
    """(nby, nbx, 64) scan-order levels -> int64 residual plane."""
    nby, nbx = lev.shape[:2]
    coef = np.zeros((nby, nbx, 64), dtype=np.int64)
    coef[..., ZZ] = dequantize(lev, qp)
    return unblockify(idct8(coef.reshape(nby, nbx, 8, 8)))


def delta_maps(fs):
    m = fs.mode == MODE_INTER
    dx = np.where(m, fs.dx, 0)
    dy = np.where(m, fs.dy, 0)
    up = lambda a: np.repeat(np.repeat(a, 8, axis=0), 8, axis=1)
    return up(dx), up(dy)


def reconstruct_inter(fs, ref, gp, interp):
    dxp, dyp = delta_maps(fs)
    Yp = predict_luma(ref[0], gp, dxp, dyp, interp)
    Cbp = predict_chroma(ref[1], gp, dxp, dyp, interp)
    Crp = predict_chroma(ref[2], gp, dxp, dyp, interp)
    Y = np.clip(Yp + levels_to_residual(fs.lev_y, fs.qp), 0, 255)
    Cb = np.clip(Cbp + levels_to_residual(fs.lev_cb, fs.qp), 0, 255)
    Cr = np.clip(Crp + levels_to_residual(fs.lev_cr, fs.qp), 0, 255)
    return [Y.astype(np.int32), Cb.astype(np.int32), Cr.astype(np.int32)]


def intra_pred(plane, y0, x0, mode):
    top = plane[y0 - 1, x0:x0 + 8] if y0 > 0 else None
    left = plane[y0:y0 + 8, x0 - 1] if x0 > 0 else None
    if mode == 1 and left is not None:
        return np.repeat(left.astype(np.int64)[:, None], 8, axis=1)
    if mode == 2 and top is not None:
        return np.repeat(top.astype(np.int64)[None, :], 8, axis=0)
    if top is not None and left is not None:
        dc = (int(top.sum()) + int(left.sum()) + 8) >> 4
    elif top is not None:
        dc = (int(top.sum()) + 4) >> 3
    elif left is not None:
        dc = (int(left.sum()) + 4) >> 3
    else:
        dc = 128
    return np.full((8, 8), dc, dtype=np.int64)


def recon_intra_block(plane, y0, x0, mode, lv, qp):
    """Reconstruct one 8x8 intra block in place; `lv` = 64 scan-order levels or None."""
    pred = intra_pred(plane, y0, x0, mode)
    if lv is not None and np.any(lv):
        coef = np.zeros(64, dtype=np.int64)
        coef[ZZ] = dequantize(np.asarray(lv, dtype=np.int64), qp)
        pred = pred + idct8(coef.reshape(8, 8))
    plane[y0:y0 + 8, x0:x0 + 8] = np.clip(pred, 0, 255)


def reconstruct_intra_planes(fs, planes):
    for by, bx in zip(*np.nonzero(fs.mode == MODE_INTRA)):
        recon_intra_block(planes[0], by * 8, bx * 8, int(fs.imode_y[by, bx]), fs.lev_y[by, bx], fs.qp)
    unit = fs.mode[::2, ::2] == MODE_INTRA
    for uy, ux in zip(*np.nonzero(unit)):
        recon_intra_block(planes[1], uy * 8, ux * 8, int(fs.imode_cb[uy, ux]), fs.lev_cb[uy, ux], fs.qp)
        recon_intra_block(planes[2], uy * 8, ux * 8, int(fs.imode_cr[uy, ux]), fs.lev_cr[uy, ux], fs.qp)


def reconstruct_i(fs):
    Hc, Wc = fs.Hb * 8, fs.Wb * 8
    planes = [np.zeros((Hc, Wc), np.int32), np.zeros((Hc // 2, Wc // 2), np.int32),
              np.zeros((Hc // 2, Wc // 2), np.int32)]
    reconstruct_intra_planes(fs, planes)
    return planes


def reconstruct_p(fs, ref, gp, interp):
    planes = reconstruct_inter(fs, ref, gp, interp)
    reconstruct_intra_planes(fs, planes)
    return planes


# =============================================================================
# Container format
# VTV container: header | segments | index (index at the end, offset in header).
#
#     FILE_HEADER (40 bytes, little endian)
#       magic 'VTVB' | ver_major u8 | ver_minor u8 | flags u16 | width u16 | height u16
#       fps_num u32 | fps_den u32 | n_frames u32 | n_segments u32 | index_offset u64
#       chroma_format u8 (1 = 4:2:0) | bit_depth u8 | reserved u16
#     SEGMENT
#       n_frames u16 | qp_i u8 | qp_p u8 | payload_len u32 | payload | crc32 u32
#       (crc over header+payload).  Every segment starts with an I frame and fresh
#       entropy-coder contexts, so it decodes without any other segment.
#     INDEX  (n_segments entries)
#       frame_start u32 | n_frames u16 | offset u64 | size u32
#
# Writing the index last lets an encoder stream segments out in one pass; an HTTP
# client fetches the 40-byte header, then the index via a range request, then any
# segment it needs.
# =============================================================================
MAGIC = b"VTVB"
HDR = struct.Struct("<4sBBHHHIIIIQBBH")
SEG = struct.Struct("<HBBI")
IDX = struct.Struct("<IHQI")
CRC = struct.Struct("<I")

MAX_DIM = 8192
MAX_FRAMES = 1 << 24


def write_file(width, height, fps, flags, segments):
    """segments: list of (n_frames, qp_i, qp_p, payload_bytes).  Returns the file bytes."""
    n_frames = sum(s[0] for s in segments)
    body = bytearray()
    index = []
    start = 0
    for nf, qi, qp, payload in segments:
        off = HDR.size + len(body)
        rec = SEG.pack(nf, qi, qp, len(payload)) + payload
        rec += CRC.pack(zlib.crc32(rec) & 0xFFFFFFFF)
        index.append((start, nf, off, len(rec)))
        body += rec
        start += nf
    index_offset = HDR.size + len(body)
    hdr = HDR.pack(MAGIC, VERSION[0], VERSION[1], flags, width, height, fps[0], fps[1],
                   n_frames, len(segments), index_offset, 1, 8, 0)
    idx = b"".join(IDX.pack(*e) for e in index)
    return hdr + bytes(body) + idx


def read_header(data):
    if len(data) < HDR.size:
        raise BitstreamError("file too short")
    (magic, vmaj, vmin, flags, w, h, fn, fd, nfr, nseg, ioff, cf, bd, _r) = HDR.unpack_from(data, 0)
    if magic != MAGIC:
        raise BitstreamError("bad magic")
    if vmaj != VERSION[0]:
        raise BitstreamError(f"unsupported major version {vmaj}")
    if not (0 < w <= MAX_DIM and 0 < h <= MAX_DIM) or w % 2 or h % 2:
        raise BitstreamError("invalid dimensions")
    if cf != 1 or bd != 8:
        raise BitstreamError("only 4:2:0 8-bit supported in this version")
    if nfr > MAX_FRAMES or fd == 0 or nseg > nfr:
        raise BitstreamError("invalid frame/segment count")
    if ioff + nseg * IDX.size > len(data):
        raise BitstreamError("index outside file")
    return dict(version=(vmaj, vmin), flags=flags, width=w, height=h, fps=(fn, fd), n_frames=nfr,
                n_segments=nseg, index_offset=ioff,
                geo=bool(flags & FLAG_GEO), photo=bool(flags & FLAG_PHOTO),
                interp=(flags >> FLAG_INTERP_SHIFT) & 3)


def read_index(data, hdr):
    idx = []
    for i in range(hdr["n_segments"]):
        fs_, nf, off, size = IDX.unpack_from(data, hdr["index_offset"] + i * IDX.size)
        if off < HDR.size or off + size > hdr["index_offset"] or size < SEG.size + CRC.size:
            raise BitstreamError(f"segment {i}: bad index entry")
        idx.append(dict(frame_start=fs_, n_frames=nf, offset=off, size=size))
    return idx


def read_segment(data, entry):
    """Verify CRC and return (n_frames, qp_i, qp_p, payload).  Raises BitstreamError."""
    rec = data[entry["offset"]: entry["offset"] + entry["size"]]
    if len(rec) != entry["size"]:
        raise BitstreamError("truncated segment")
    body, (crc,) = rec[:-4], CRC.unpack(rec[-4:])
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise BitstreamError("segment CRC mismatch")
    nf, qi, qp, plen = SEG.unpack_from(body, 0)
    if plen != len(body) - SEG.size or nf == 0 or nf != entry["n_frames"] or qi > 51 or qp > 51:
        raise BitstreamError("inconsistent segment header")
    return nf, qi, qp, body[SEG.size:]


# =============================================================================
# Encoder
# VTV encoder (closed loop, rate-distortion optimised).
#
# Pipeline per GOP (= independently decodable segment)
#   1. I frame: intra DC/H/V prediction + 8x8 transform coding.
#   2. Global affine+photometric parameters are estimated between consecutive
#      *source* frames, then fitted as polynomial trajectories (trajectory.py).
#   3. Every P frame is predicted from the previous *reconstructed* frame with the
#      trajectory-defined global warp.  A quadtree of 32/16/8 leaves selects, per
#      region, SKIP / INTER (+ quarter-pel delta motion) / INTRA ("new
#      information") by minimising  J = D + lambda*R  with D measured in the
#      transform domain (orthonormal => equals pixel-domain SSE).
#   4. Residuals are transform-coded; a block is zeroed when that lowers J.
#   5. All syntax elements are coded with the adaptive range coder.
#
# The encoder never invents a decoder path: after choosing syntax it always
# reconstructs with the *shared* recon module, so its reference frames are
# bit-identical to what the decoder will produce.
# =============================================================================
# ---- rate proxy (bits) used only for decisions; calibrated against the real coder
B_BLOCK, B_NZ, B_LOG, B_LAST, B_ZERO = 2.0, 3.2, 1.6, 0.22, 0.25
THETA = (1, 3)        # quantiser rounding offset numerator/denominator (dead zone)
LAM_M = 0.36          # motion-search lambda factor (SAD domain): lam_m = LAM_M * qstep


def proxy_bits(lv):
    a = np.abs(lv)
    nzm = a > 0
    n = nzm.sum(1)
    lg = np.where(nzm, np.log2(np.maximum(a, 1)), 0.0).sum(1)
    last = 63 - np.argmax(nzm[:, ::-1], axis=1)
    bits = B_BLOCK + B_NZ * n + B_LOG * lg + B_LAST * last
    return np.where(n > 0, bits, B_ZERO)


def quantize_blocks(coefs, qp, lam, rdoq=True):
    """(..., 8, 8) coefficients -> (levels (N,64) in scan order, J (N,))."""
    c = np.asarray(coefs).reshape(-1, 64)[:, ZZ]
    lv = quantize(c, qp, THETA[0], THETA[1])
    deq = dequantize(lv, qp)
    Jc = ((c - deq) ** 2).sum(1) + lam * proxy_bits(lv)
    J0 = (c ** 2).sum(1) + lam * B_ZERO
    if rdoq:
        z = J0 <= Jc
        return np.where(z[:, None], 0, lv), np.minimum(J0, Jc)
    allz = (lv != 0).sum(1) == 0
    return lv, np.where(allz, J0, Jc)


# ---------------------------------------------------------------- intra
def encode_intra_blocks(cur, plane, blocks, qp, lam, imode_arr, lev_arr, rdoq):
    """Sequentially code 8x8 intra blocks (raster order), reconstructing into `plane`."""
    for by, bx in blocks:
        y0, x0 = by * 8, bx * 8
        X = cur[y0:y0 + 8, x0:x0 + 8].astype(np.int64)
        modes = [0] + ([1] if x0 > 0 else []) + ([2] if y0 > 0 else [])
        best = None
        for m in modes:
            p = intra_pred(plane, y0, x0, m)
            sad = int(np.abs(X - p).sum())
            if best is None or sad < best[0]:
                best = (sad, m, p)
        _, m, p = best
        lv, _ = quantize_blocks(fdct8(X - p)[None], qp, lam, rdoq)
        imode_arr[by, bx] = m
        lev_arr[by, bx, :] = lv[0]
        recon_intra_block(plane, y0, x0, m, lv[0], qp)


def encode_i_frame(cur, qp, cfg):
    Hc, Wc = cur[0].shape
    qs = qstep(qp)
    lam = cfg.lam_scale * qs * qs
    fs = FrameSyntax("I", Hc, Wc, qp)
    fs.mode[:] = MODE_INTRA
    fs.lsize[:] = 32
    tmp = [np.zeros((Hc, Wc), np.int32), np.zeros((Hc // 2, Wc // 2), np.int32),
           np.zeros((Hc // 2, Wc // 2), np.int32)]
    Hb, Wb = fs.Hb, fs.Wb
    encode_intra_blocks(cur[0], tmp[0], [(y, x) for y in range(Hb) for x in range(Wb)],
                        qp, lam, fs.imode_y, fs.lev_y, cfg.rdoq)
    cb = [(y, x) for y in range(Hb // 2) for x in range(Wb // 2)]
    encode_intra_blocks(cur[1], tmp[1], cb, qp, lam, fs.imode_cb, fs.lev_cb, cfg.rdoq)
    encode_intra_blocks(cur[2], tmp[2], cb, qp, lam, fs.imode_cr, fs.lev_cr, cfg.rdoq)
    return fs


# ---------------------------------------------------------------- inter
def _expand(a, bs):
    return np.repeat(np.repeat(a, bs, axis=0), bs, axis=1)


def _refine_subpel(Y, ref_y, gp, dxq, dyq, s8, cfg, lam_m):
    """Half then quarter-pel refinement of per-block deltas with the exact sampler."""
    bs = 8 * s8
    hb, wb = dxq.shape
    H, W = Y.shape

    def cost(ax, ay):
        pred = predict_luma(ref_y, gp, _expand(ax, bs), _expand(ay, bs), cfg.interp)
        sad = np.abs(Y - pred).reshape(hb, bs, wb, bs).sum(axis=(1, 3))
        return sad + lam_m * (2 * np.log2(1 + np.abs(ax)) + 2 * np.log2(1 + np.abs(ay)))

    bx, by = dxq.copy(), dyq.copy()
    bc = cost(bx, by)
    for step, offs in ((2, [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, 2), (2, -2), (-2, -2)]),
                       (1, [(1, 0), (-1, 0), (0, 1), (0, -1)])):
        for ox, oy in offs:
            cx_, cy_ = bx + ox, by + oy
            c = cost(cx_, cy_)
            better = c < bc
            bc = np.where(better, c, bc)
            bx = np.where(better, cx_, bx)
            by = np.where(better, cy_, by)
    return bx, by


def encode_p_frame(cur, ref, gp, cfg, qp):
    Y = cur[0].astype(np.int64)
    Hc, Wc = Y.shape
    Hb, Wb = Hc // 8, Wc // 8
    qs = qstep(qp)
    lam = cfg.lam_scale * qs * qs
    lam_m = LAM_M * qs
    interp = cfg.interp

    # -- base prediction: global transform only
    P0 = predict_luma(ref[0], gp, None, None, interp)

    # -- integer full search of regional deltas on the 8x8 grid
    R = cfg.search_range
    P0p = np.pad(P0, R, mode="edge")
    cands = [(dx, dy) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
    nc = len(cands)
    S = np.empty((nc, Hb, Wb), np.int64)
    for k, (dx, dy) in enumerate(cands):
        sh = P0p[R + dy:R + dy + Hc, R + dx:R + dx + Wc]
        S[k] = np.abs(Y - sh).reshape(Hb, 8, Wb, 8).sum(axis=(1, 3))
    cdx = np.array([c[0] for c in cands])
    cdy = np.array([c[1] for c in cands])
    pen = lam_m * (2 * np.log2(1 + 4 * np.abs(cdx)) + 2 * np.log2(1 + 4 * np.abs(cdy)))

    # -- hypotheses: SKIP, INTER per level, INTRA
    Jskip8 = ((Y - P0) ** 2).reshape(Hb, 8, Wb, 8).sum(axis=(1, 3)) + lam * 0.3
    blocks = blockify(Y)
    mean = (blocks.sum(axis=(2, 3)) + 32) >> 6
    _, JI = quantize_blocks(fdct8(blocks - mean[:, :, None, None]), qp, lam, cfg.rdoq)
    Jintra_u = (JI.reshape(Hb, Wb) + lam * 5.0).reshape(Hb // 2, 2, Wb // 2, 2).sum(axis=(1, 3))

    dec = {}
    Jprev = None
    for s8 in (1, 2, 4):
        bs, hb, wb = 8 * s8, Hb // s8, Wb // s8
        Ss = S.reshape(nc, hb, s8, wb, s8).sum(axis=(2, 4))
        k = (Ss + pen[:, None, None]).argmin(axis=0)
        dxq, dyq = 4 * cdx[k], 4 * cdy[k]
        if cfg.subpel:
            dxq, dyq = _refine_subpel(Y, ref[0], gp, dxq, dyq, s8, cfg, lam_m)
        predY = predict_luma(ref[0], gp, _expand(dxq, bs), _expand(dyq, bs), interp)
        _, J = quantize_blocks(fdct8(blockify(Y - predY)), qp, lam, cfg.rdoq)
        J8 = J.reshape(Hb, Wb)

        agg = lambda a: a.reshape(hb, s8, wb, s8).sum(axis=(1, 3))
        mvb = 2 * np.log2(1 + np.abs(dxq)) + 2 * np.log2(1 + np.abs(dyq)) + 1.0
        opts = [agg(Jskip8) + lam * 0.6, agg(J8) + lam * (2.0 + mvb)]
        if s8 >= 2:
            n = s8 // 2
            opts.append(Jintra_u.reshape(hb, n, wb, n).sum(axis=(1, 3)) + lam * 3.0)
        st = np.stack(opts)
        mode, Jleaf = st.argmin(axis=0), st.min(axis=0)
        if s8 == 1:
            split, Jb = np.zeros((hb, wb), bool), Jleaf
        else:
            Jch = Jprev.reshape(hb, 2, wb, 2).sum(axis=(1, 3)) + lam
            split = Jch < Jleaf
            Jb = np.where(split, Jch, Jleaf)
        dec[s8] = (mode, split, dxq, dyq)
        Jprev = Jb

    fs = FrameSyntax("P", Hc, Wc, qp)

    def assign(s8, iy, ix):
        mode, split, dxq, dyq = dec[s8]
        if s8 > 1 and split[iy, ix]:
            for oy in (0, 1):
                for ox in (0, 1):
                    assign(s8 // 2, 2 * iy + oy, 2 * ix + ox)
            return
        m = int(mode[iy, ix])
        sl = (slice(iy * s8, (iy + 1) * s8), slice(ix * s8, (ix + 1) * s8))
        fs.lsize[sl] = 8 * s8
        fs.mode[sl] = m
        if m == MODE_INTER:
            fs.dx[sl] = int(dxq[iy, ix])
            fs.dy[sl] = int(dyq[iy, ix])

    for cy in range(Hb // 4):
        for cx in range(Wb // 4):
            assign(4, cy, cx)

    # -- final residual for inter/skip areas
    dxp, dyp = delta_maps(fs)
    predY = predict_luma(ref[0], gp, dxp, dyp, interp)
    lv, _ = quantize_blocks(fdct8(blockify(Y - predY)), qp, lam, cfg.rdoq)
    fs.lev_y[:] = lv.reshape(Hb, Wb, 64)
    fs.lev_y[fs.mode != MODE_INTER] = 0
    for pi, lev in ((1, fs.lev_cb), (2, fs.lev_cr)):
        predC = predict_chroma(ref[pi], gp, dxp, dyp, interp)
        C = cur[pi].astype(np.int64)
        lvc, _ = quantize_blocks(fdct8(blockify(C - predC)), qp, lam, cfg.rdoq)
        lev[:] = lvc.reshape(Hb // 2, Wb // 2, 64)
    any_inter = (fs.mode.reshape(Hb // 2, 2, Wb // 2, 2) == MODE_INTER).any(axis=(1, 3))
    fs.lev_cb[~any_inter] = 0
    fs.lev_cr[~any_inter] = 0

    # -- intra ("new information") units, sequential because of neighbour dependence
    if (fs.mode == MODE_INTRA).any():
        planes = reconstruct_inter(fs, ref, gp, interp)
        encode_intra_blocks(cur[0], planes[0], list(zip(*np.nonzero(fs.mode == MODE_INTRA))),
                            qp, lam, fs.imode_y, fs.lev_y, cfg.rdoq)
        units = list(zip(*np.nonzero(fs.mode[::2, ::2] == MODE_INTRA)))
        encode_intra_blocks(cur[1], planes[1], units, qp, lam, fs.imode_cb, fs.lev_cb, cfg.rdoq)
        encode_intra_blocks(cur[2], planes[2], units, qp, lam, fs.imode_cr, fs.lev_cr, cfg.rdoq)
    return fs


# ---------------------------------------------------------------- top level
class Encoder:
    def __init__(self, cfg=None, width=0, height=0, fps=(25, 1)):
        self.cfg = cfg or EncoderConfig()
        self.width, self.height, self.fps = width, height, fps

    # -- analysis: per-frame global estimates and scene cuts
    def _analyse(self, planes, progress=None):
        cfg = self.cfg
        n = len(planes)
        est = [None] * n
        mad = [0.0] * n
        for t in range(1, n):
            th, info = estimate_global(planes[t][0], planes[t - 1][0], photo=True)
            est[t] = th
            mad[t] = info["mad"]
            if progress:
                progress(t, 2 * n - 1)
        return est, mad

    def _gops(self, n, mad):
        cfg = self.cfg
        gops, a = [], 0
        for t in range(1, n):
            if t - a >= cfg.gop or mad[t] > cfg.cut_mad:
                gops.append((a, t))
                a = t
        gops.append((a, n))
        return gops

    def encode(self, frames, progress=None):
        """frames: list of visible (Y, Cb, Cr) uint8 planes.  Returns (bytes, stats).

        progress(done, total) is called after every analysed / encoded frame."""
        cfg = self.cfg
        W, H = self.width, self.height
        planes = [pad_planes(f, W, H) for f in frames]
        n = len(planes)
        est, mad = self._analyse(planes, progress)
        done = n - 1
        gops = self._gops(n, mad)
        Hc, Wc = planes[0][0].shape
        tol_g, tol_p = default_tolerances(cfg.qp)
        if cfg.tol_geo >= 0:
            tol_g = cfg.tol_geo
        if cfg.tol_photo >= 0:
            tol_p = cfg.tol_photo

        segments, bits = [], {}
        n_traj = 0
        self.last_recon = []
        for (a, b) in gops:
            qp_i, qp_p = cfg.qp_i(), cfg.qp
            fs_i = encode_i_frame(planes[a], qp_i, cfg)
            ref = reconstruct_i(fs_i)
            self.last_recon.append(ref)
            done += 1
            if progress:
                progress(done, 2 * n - 1)
            targets = np.array([theta_to_units(est[t]) for t in range(a + 1, b)]).reshape(-1, 8)
            segs = plan(targets, np.zeros(8, dtype=np.int64), cfg.use_global, cfg.use_photo,
                        tol_g, tol_p, Wc, Hc, cfg.traj_max_len, cfg.use_traj)
            n_traj += len(segs)
            fs_lists, t = [], a + 1
            for seg in segs:
                lst = []
                for k in range(seg["L"]):
                    gp = tuple(int(v) for v in seg["vals"][k])
                    fs = encode_p_frame(planes[t], ref, gp, cfg, qp_p)
                    ref = reconstruct_p(fs, ref, gp, cfg.interp)
                    self.last_recon.append(ref)
                    lst.append(fs)
                    t += 1
                    done += 1
                    if progress:
                        progress(done, 2 * n - 1)
                fs_lists.append(lst)

            rc = RangeEncoder(stats=cfg.stats)
            cx = Contexts()
            code_frame(rc, cx, fs_i)
            prev_end, left = np.zeros(8, dtype=np.int64), b - a - 1
            for seg, lst in zip(segs, fs_lists):
                code_traj(rc, cx, seg, prev_end, cfg.use_global, cfg.use_photo, left)
                for fs in lst:
                    code_frame(rc, cx, fs)
                prev_end = seg["vals"][-1]
                left -= seg["L"]
            payload = rc.finish()
            segments.append((b - a, qp_i, qp_p, payload))
            for k, v in rc.bits.items():
                bits[k] = bits.get(k, 0.0) + v

        flags = ((FLAG_GEO if cfg.use_global else 0) | (FLAG_PHOTO if cfg.use_photo else 0)
                 | (cfg.interp << FLAG_INTERP_SHIFT))
        data = write_file(W, H, self.fps, flags, segments)
        stats = dict(bytes=len(data), bits=bits, n_segments=len(segments), n_traj_segments=n_traj,
                     gops=gops, mad=mad)
        return data, stats


def encode(frames, width, height, cfg=None, fps=(25, 1)):
    return Encoder(cfg, width, height, fps).encode(frames)


# =============================================================================
# Decoder
# Reference decoder.  Deterministic, integer-only reconstruction.
#
# `decode_segment` needs nothing but the bytes of one segment (random access).
# `decode` walks all segments and, if a segment is corrupt or missing, conceals
# it (repeat last good frame, or mid-gray at the start) and resumes at the next
# segment -- the resilience behaviour of concept section 31.
# =============================================================================
def decode_segment_payload(payload, n_frames, qp_i, qp_p, width, height, geo, photo, interp):
    """Decode one segment payload -> list of coded-size (Y, Cb, Cr) int32 planes."""
    Wc, Hc = coded_size(width, height)
    rc = RangeDecoder(payload)
    cx = Contexts()
    fs = FrameSyntax("I", Hc, Wc, qp_i)
    code_frame(rc, cx, fs)
    ref = reconstruct_i(fs)
    frames = [ref]
    prev_end = np.zeros(8, dtype=np.int64)
    done = 1
    while done < n_frames:
        seg = code_traj(rc, cx, None, prev_end, geo, photo, n_frames - done)
        for k in range(seg["L"]):
            gp = tuple(int(v) for v in seg["vals"][k])
            fs = FrameSyntax("P", Hc, Wc, qp_p)
            code_frame(rc, cx, fs)
            ref = reconstruct_p(fs, ref, gp, interp)
            frames.append(ref)
        prev_end = seg["vals"][-1]
        done += seg["L"]
    return frames


def decode_segment(data, hdr, entry):
    """Decode segment `entry` of a parsed file -> list of visible uint8 planes."""
    nf, qi, qp, payload = read_segment(data, entry)
    frames = decode_segment_payload(payload, nf, qi, qp, hdr["width"], hdr["height"],
                                    hdr["geo"], hdr["photo"], hdr["interp"])
    return [crop_planes(f, hdr["width"], hdr["height"]) for f in frames]


def decode(data, conceal=True):
    """Decode a whole file.  Returns (frames, info) with info['problems'] listing
    (segment_index, message) for every segment that had to be concealed."""
    hdr = read_header(data)
    index = read_index(data, hdr)
    W, H = hdr["width"], hdr["height"]
    frames, problems = [], []
    for i, e in enumerate(index):
        try:
            frames.extend(decode_segment(data, hdr, e))
        except Exception as ex:                         # noqa: BLE001 - untrusted input
            if not conceal:
                raise
            problems.append((i, f"{type(ex).__name__}: {ex}"))
            if frames:
                fill = frames[-1]
            else:
                fill = (np.full((H, W), 128, np.uint8),
                        np.full(((H + 1) // 2, (W + 1) // 2), 128, np.uint8),
                        np.full(((H + 1) // 2, (W + 1) // 2), 128, np.uint8))
            frames.extend([fill] * e["n_frames"])
    info = dict(header=hdr, problems=problems)
    return frames, info


# =============================================================================
# Video input, PNG output, metrics, demo clip  (everything here is outside the
# normative codec: it only feeds frames in and measures results)
# =============================================================================
def png_bytes(rgb):
    """uint8 (H, W, 3) -> PNG file bytes (no external imaging library needed)."""
    H, W, _ = rgb.shape
    raw = np.concatenate([np.zeros((H, 1), np.uint8), np.ascontiguousarray(rgb).reshape(H, W * 3)], axis=1)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def find_ffmpeg():
    """Path of an ffmpeg executable (PATH, or the imageio-ffmpeg wheel), else None."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                            # noqa: BLE001
        return None


def _probe(path, ff):
    """fps / duration of a video via `ffmpeg -i` (parses its banner)."""
    info = dict(fps=None, duration=None, src_w=None, src_h=None)
    try:
        r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=30)
        txt = r.stderr.decode("utf-8", "replace")
    except Exception:                                            # noqa: BLE001
        return info
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt) or re.search(r"(\d+(?:\.\d+)?)\s*tbr", txt)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", txt)
    if m:
        info["src_w"], info["src_h"] = int(m.group(1)), int(m.group(2))
    return info


def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _read_ppm(f):
    """Read one binary PPM (P6) image from a stream -> (H, W, 3) uint8 or None at EOF."""
    magic = f.read(2)
    if len(magic) < 2:
        return None
    if magic != b"P6":
        raise ValueError("unexpected data from ffmpeg")
    toks = []
    while len(toks) < 3:
        c = f.read(1)
        if not c:
            return None
        if c.isspace():
            continue
        if c == b"#":
            f.readline()
            continue
        tok = c
        while True:
            c = f.read(1)
            if not c or c.isspace():
                break
            tok += c
        toks.append(int(tok))
    W, H, _ = toks
    buf = _read_exact(f, W * H * 3)
    if buf is None:
        return None
    return np.frombuffer(buf, np.uint8).reshape(H, W, 3)


def read_video(path, width=256, max_frames=60, start=0.0, progress=None):
    """Read a video file -> (frames, W, H, fps, info).

    frames : list of (Y, Cb, Cr) uint8 planes (4:2:0, even size, full-range BT.601)
    Width is capped at `width` (aspect kept, even dimensions); at most `max_frames`
    frames are read.  Uses ffmpeg if available, else OpenCV; `.y4m` files are read as-is.
    """
    if path.lower().endswith(".y4m"):
        frames, W, H, fps = read_y4m(path, max_frames=max_frames)
        return frames, W, H, fps[0] / fps[1], dict(source="y4m")
    ff = find_ffmpeg()
    if ff:
        info = _probe(path, ff)
        cmd = [ff, "-v", "error", "-nostats"]
        if start:
            cmd += ["-ss", str(start)]
        cmd += ["-i", path, "-an", "-vf", f"scale=w='trunc(min(iw,{int(width)})/2)*2':h=-2:flags=area",
                "-frames:v", str(int(max_frames)), "-f", "image2pipe", "-c:v", "ppm", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = []
        try:
            while True:
                rgb = _read_ppm(proc.stdout)
                if rgb is None:
                    break
                frames.append(rgb_to_planes(rgb))
                if progress:
                    progress(len(frames), int(max_frames))
        finally:
            proc.stdout.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            proc.wait()
        if not frames:
            raise RuntimeError("ffmpeg could not decode any frame: " + err.strip()[-300:])
        H, W = frames[0][0].shape
        info.update(source="ffmpeg")
        return frames, W, H, info["fps"] or 25.0, info
    try:
        import cv2
    except ImportError:
        raise RuntimeError("No video reader found. Install ffmpeg (https://ffmpeg.org) or "
                           "`pip install opencv-python` (or `pip install imageio-ffmpeg`).")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    W = H = None
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if W is None:
            h0, w0 = bgr.shape[:2]
            W = max(2, (min(w0, width) // 2) * 2)
            H = max(2, int(round(h0 * W / w0 / 2)) * 2)
        if bgr.shape[1] != W or bgr.shape[0] != H:
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(rgb_to_planes(bgr[..., ::-1]))
        if progress:
            progress(len(frames), max_frames)
    cap.release()
    if not frames:
        raise RuntimeError("OpenCV could not decode any frame from this file")
    return frames, W, H, fps, dict(source="opencv", fps=fps,
                                   duration=None, src_w=None, src_h=None)


# ---- metrics ------------------------------------------------------------------
def psnr(a, b, peak=255.0):
    m = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(peak * peak / m))


def _box_mean(x, win):
    c = np.pad(np.cumsum(np.cumsum(x, axis=0), axis=1), ((1, 0), (1, 0)))
    return (c[win:, win:] - c[:-win, win:] - c[win:, :-win] + c[:-win, :-win]) / (win * win)


def ssim(a, b, win=7):
    """Mean SSIM of one plane (uniform 7x7 window; matches scikit-image's default)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if min(a.shape) < win:
        return 1.0 if np.array_equal(a, b) else 0.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = _box_mean(a, win), _box_mean(b, win)
    va, vb = _box_mean(a * a, win) - ma * ma, _box_mean(b * b, win) - mb * mb
    cov = _box_mean(a * b, win) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())


def sequence_quality(src, dec):
    """Per-clip PSNR-Y (mean/min), PSNR-YUV (6:1:1) and SSIM-Y."""
    py, pyuv, ss = [], [], []
    for s, d in zip(src, dec):
        p = [psnr(s[i], d[i]) for i in range(3)]
        py.append(p[0])
        pyuv.append((6 * p[0] + p[1] + p[2]) / 8)
        ss.append(ssim(s[0], d[0]))
    return dict(psnr_y=float(np.mean(py)), psnr_y_min=float(np.min(py)),
                psnr_yuv=float(np.mean(pyuv)), ssim_y=float(np.mean(ss)))


def bd_rate(ref_pts, test_pts):
    """Bjontegaard delta-rate in percent (negative = test needs fewer bits).
    pts: list of (bitrate, quality), >= 4 points each."""
    r1, q1 = np.log10([p[0] for p in ref_pts]), np.array([p[1] for p in ref_pts])
    r2, q2 = np.log10([p[0] for p in test_pts]), np.array([p[1] for p in test_pts])
    p1, p2 = np.polyfit(q1, r1, 3), np.polyfit(q2, r2, 3)
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        return float("nan")
    i1, i2 = np.polyint(p1), np.polyint(p2)
    a1 = (np.polyval(i1, hi) - np.polyval(i1, lo)) / (hi - lo)
    a2 = (np.polyval(i2, hi) - np.polyval(i2, lo)) / (hi - lo)
    return float((10 ** (a2 - a1) - 1) * 100)


# ---- demo clip (numpy only) ----------------------------------------------------
def demo_clip(W=192, H=128, N=32, seed=7):
    """Synthetic clip: textured world, camera pan with sub-pel motion + a moving disk.
    Returns a list of (Y, Cb, Cr) uint8 planes.  Handy for trying VTV without any video file."""
    rng = np.random.default_rng(seed)
    Ww, Hw = W + int(2 * N) + 48, H + int(N) + 48
    yy, xx = np.mgrid[0:Hw, 0:Ww].astype(np.float64)
    world = np.empty((Hw, Ww, 3))
    for c in range(3):
        acc = np.full((Hw, Ww), 125.0)
        for _ in range(26):
            f, th = rng.uniform(0.01, 0.28), rng.uniform(0, np.pi)
            acc += rng.uniform(5, 19) * np.sin(2 * np.pi * f * (np.cos(th) * xx + np.sin(th) * yy)
                                               + rng.uniform(0, 6.28))
        world[..., c] = acc
    for _ in range(28):
        cx, cy = rng.uniform(0, Ww), rng.uniform(0, Hw)
        col = rng.uniform(40, 220, 3)
        if rng.random() < 0.5:
            m = (np.abs(xx - cx) < rng.uniform(6, 30)) & (np.abs(yy - cy) < rng.uniform(5, 22))
        else:
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < rng.uniform(6, 22) ** 2
        world[m] = 0.35 * world[m] + 0.65 * col
    frames = []
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
    for t in range(N):
        ox, oy = 8 + 1.6 * t, 8 + 0.7 * t
        x0, y0 = int(ox), int(oy)
        fx, fy = ox - x0, oy - y0
        w = world[y0:y0 + H + 1, x0:x0 + W + 1]
        img = ((w[:-1, :-1] * (1 - fx) + w[:-1, 1:] * fx) * (1 - fy) +
               (w[1:, :-1] * (1 - fx) + w[1:, 1:] * fx) * fy)
        cxs, cys = 30 + 2.2 * t, 40 + 0.8 * t
        a = np.clip(0.5 - (np.sqrt((gx - cxs) ** 2 + (gy - cys) ** 2) - 12), 0, 1)[..., None]
        img = img * (1 - a) + np.array([235.0, 190.0, 60.0]) * a
        frames.append(rgb_to_planes(np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)))
    return frames


# =============================================================================
# Web app:  python vtv.py serve      (standard library HTTP server, no extra packages)
# =============================================================================
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VTV codec</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--mut:#667085;--line:#e3e6ec;--acc:#2f6feb;--acc2:#e8f0fe;--bar:#2f6feb;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2b3240;--acc:#6aa0ff;--acc2:#1e2a44;--bar:#6aa0ff;--warn:#f5b26b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:28px 20px 8px;max-width:960px;margin:0 auto}
h1{margin:0;font-size:28px;letter-spacing:-.01em}h1 small{font-weight:400;color:var(--mut);font-size:15px;margin-left:8px}
.sub{color:var(--mut);margin:.3em 0 0}
main{max-width:960px;margin:0 auto;padding:12px 20px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}
h2{margin:0 0 12px;font-size:18px}h3{margin:20px 0 6px;font-size:16px}h4{margin:16px 0 4px;font-size:14px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
label{display:block;font-size:13px;color:var(--mut);margin-bottom:3px}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;margin:10px 0}
.row>div{min-width:130px}
input[type=number],select,input[type=file]{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);max-width:100%}
input[type=range]{width:100%}
button{font:inherit;padding:8px 16px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer}
button.sec{background:transparent;color:var(--acc)}button:disabled{opacity:.45;cursor:not-allowed}
a.btn{display:inline-block;text-decoration:none;padding:8px 14px;border-radius:8px;border:1px solid var(--acc);color:var(--acc);margin-right:8px;font-size:14px}
progress{width:100%;height:10px;accent-color:var(--bar)}
#msg,#pmsg{color:var(--mut);font-size:13px;min-height:1.4em;margin-top:4px}
.err{color:#d92d20!important}
details{margin:8px 0}summary{cursor:pointer;color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.tile{background:var(--acc2);border-radius:10px;padding:10px 12px}
.tile .k{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:21px;font-weight:600;margin-top:2px}.tile .s{font-size:12px;color:var(--mut);margin-top:2px}
.bars .b{display:grid;grid-template-columns:210px 1fr 90px;gap:10px;align-items:center;margin:4px 0;font-size:13px}
.bars .t{height:12px;background:var(--acc2);border-radius:6px;overflow:hidden}.bars .f{height:100%;background:var(--bar)}
.note{font-size:13px;color:var(--mut);margin-top:8px}
.chk label{display:inline-flex;align-items:center;gap:6px;margin-right:16px;color:var(--fg);font-size:14px}
#stage{background:#000;border-radius:10px;padding:8px;text-align:center;overflow:auto}
canvas{max-width:100%;height:auto;background:#000}canvas.pix{image-rendering:pixelated}
.ctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:10px}
.ctl input[type=range]{flex:1;min-width:180px}
code,pre{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow:auto}
code{background:var(--acc2);padding:1px 5px;border-radius:4px}pre code{background:none;padding:0}
.docs ul{padding-left:20px}.docs p{margin:.5em 0}
</style></head><body>
<header><h1>VTV<small>Vector-Time Video &middot; beta __VERSION__</small></h1>
<p class="sub">Research video codec: global-motion trajectories + regional deltas + transform residual. Pure Python / NumPy, deterministic integer decoder.</p></header>
<main>

<section class="card" id="enc"><h2>1 &middot; Encode a video</h2>
<div class="row"><div style="flex:1"><label for="file">Video file (mp4, mov, mkv, webm &hellip;)</label><input id="file" type="file" accept="video/*,.mp4,.mov,.mkv,.webm,.avi,.y4m"></div></div>
<div class="row">
 <div style="flex:1;min-width:220px"><label for="qp">Quality (QP <b id="qpv">32</b>) &mdash; lower = better &amp; bigger</label><input id="qp" type="range" min="16" max="46" value="32"></div>
 <div><label for="width">Max width</label><select id="width"><option>160</option><option selected>256</option><option>320</option><option>480</option></select></div>
 <div><label for="frames">Max frames</label><input id="frames" type="number" min="1" max="600" value="60" style="width:90px"></div>
 <div><button id="go">Encode</button></div>
</div>
<details><summary>Advanced: codec tools (for experiments / ablations)</summary>
 <div class="row chk">
  <label><input type="checkbox" id="o_glob" checked> global affine motion</label>
  <label><input type="checkbox" id="o_traj" checked> polynomial trajectories</label>
  <label><input type="checkbox" id="o_photo" checked> photometric (fade) model</label>
  <label><input type="checkbox" id="o_bil"> bilinear instead of Catmull-Rom</label>
  <label>GOP <input id="gop" type="number" min="2" max="120" value="32" style="width:70px"></label>
 </div></details>
<progress id="bar" value="0" max="1" hidden></progress><div id="msg"></div>
<div class="note">Encoding runs in pure Python: expect roughly 0.2&ndash;0.5 s per frame at 256 px width. The video is downscaled and truncated to the limits above; audio is dropped.</div>
</section>

<section class="card" id="results" hidden><h2>Statistics</h2>
<div class="grid" id="tiles"></div>
<div id="bitsbox" hidden><h3>Where the bits go</h3><div class="bars" id="bits"></div><div class="note" id="bitnote"></div></div>
<div class="row" id="dl" style="margin-top:14px"></div></section>

<section class="card" id="player"><h2>2 &middot; Player</h2>
<div class="row"><div><label>Play an existing .vtv file</label><input id="vtvfile" type="file" accept=".vtv"></div>
 <div class="note" style="margin:0 0 6px">Or encode a video above &mdash; it loads here automatically. Frames are decoded by the Python VTV decoder on the server.</div></div>
<div id="playerbox" hidden>
 <div id="stage"><canvas id="cv" width="16" height="16"></canvas></div>
 <div class="ctl"><button id="play" disabled>&#9654; Play</button>
  <input id="seek" type="range" min="0" max="0" value="0" disabled><span id="time" style="min-width:150px;font-variant-numeric:tabular-nums"></span></div>
 <div class="ctl">
  <label style="margin:0">View <select id="view"><option value="vtv">VTV (decoded)</option><option value="orig">Original</option><option value="side">Side by side</option><option value="diff">Difference &times;4</option></select></label>
  <label style="margin:0">Speed <select id="speed"><option value=".25">0.25&times;</option><option value=".5">0.5&times;</option><option value="1" selected>1&times;</option><option value="2">2&times;</option></select></label>
  <label style="margin:0"><input type="checkbox" id="loop" checked> loop</label>
  <label style="margin:0"><input type="checkbox" id="pix"> pixelated</label>
  <span class="note" style="margin:0">Space = play/pause &middot; &larr; &rarr; = step</span></div>
</div><div id="pmsg"></div></section>

<section class="card docs" id="docs"><h2>3 &middot; About the codec &amp; how to use it</h2>__DOCS__</section>
</main>
<script>
"use strict";
const $=s=>document.querySelector(s);
const fmtB=b=>b<1024?b+" B":b<1048576?(b/1024).toFixed(1)+" KB":(b/1048576).toFixed(2)+" MB";
const fmtT=s=>s<60?s.toFixed(1)+" s":Math.floor(s/60)+"m "+Math.round(s%60)+"s";
$("#qp").oninput=()=>$("#qpv").textContent=$("#qp").value;

function upload(url,file,onprog){return new Promise((res,rej)=>{const x=new XMLHttpRequest();x.open("POST",url);
 x.setRequestHeader("Content-Type","application/octet-stream");
 x.upload.onprogress=e=>{if(e.lengthComputable&&onprog)onprog(e.loaded/e.total)};
 x.onload=()=>{let j;try{j=JSON.parse(x.responseText)}catch(e){return rej(new Error("HTTP "+x.status))}
  x.status<300?res(j):rej(new Error(j.error||x.statusText))};
 x.onerror=()=>rej(new Error("network error"));x.send(file)})}
async function poll(id,cb){for(;;){const j=await (await fetch("/api/job/"+id)).json();cb(j);
 if(j.status==="done"||j.status==="error")return j;await new Promise(r=>setTimeout(r,400))}}
function setMsg(t,err){const m=$("#msg");m.textContent=t;m.className=err?"err":""}

$("#go").onclick=async()=>{
 const f=$("#file").files[0];if(!f){setMsg("Choose a video file first.",true);return}
 const q=new URLSearchParams({name:f.name,qp:$("#qp").value,width:$("#width").value,frames:$("#frames").value,gop:$("#gop").value,
  glob:$("#o_glob").checked?1:0,traj:$("#o_traj").checked?1:0,photo:$("#o_photo").checked?1:0,bil:$("#o_bil").checked?1:0});
 const bar=$("#bar");bar.hidden=false;bar.value=0;$("#go").disabled=true;$("#results").hidden=true;
 try{
  const {id}=await upload("/api/encode?"+q,f,p=>{bar.value=p;setMsg("Uploading "+Math.round(p*100)+" %\u2026")});
  const j=await poll(id,s=>{bar.value=s.progress;setMsg(s.message||s.status,s.status==="error")});
  if(j.status==="error")throw new Error(j.error);
  bar.value=1;setMsg("Done.");showStats(j.stats,id);await loadPlayer(id,j.stats);
  $("#player").scrollIntoView({behavior:"smooth",block:"start"});
 }catch(e){setMsg("Error: "+e.message,true)}
 $("#go").disabled=false;
};

const LABELS={traj:"Global-motion trajectory",struct:"Partition & modes",mvd:"Regional motion deltas",intra:"Intra / new information",res_y:"Residual \u00b7 luma",res_c:"Residual \u00b7 chroma"};
function tile(k,v,s){return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>${s?`<div class="s">${s}</div>`:""}</div>`}
function showStats(s,id){
 const T=[];
 T.push(tile("Frames",s.frames,`${s.width}\u00d7${s.height} \u00b7 ${s.fps.toFixed(2)} fps \u00b7 ${fmtT(s.duration_s)}`));
 T.push(tile("VTV file size",fmtB(s.vtv_bytes),`${s.kbps.toFixed(1)} kbit/s \u00b7 ${s.bpp.toFixed(3)} bit/pixel`));
 T.push(tile("Raw size (YUV 4:2:0)",fmtB(s.raw_bytes),"uncompressed, same frames"));
 T.push(tile("Compression ratio",s.ratio.toFixed(1)+" : 1",`file is ${(100/s.ratio).toFixed(1)} % of raw`));
 if(s.psnr_y!=null)T.push(tile("Quality",s.psnr_y.toFixed(2)+" dB",`PSNR-Y avg \u00b7 min ${s.psnr_y_min.toFixed(2)} \u00b7 SSIM-Y ${s.ssim_y.toFixed(4)}`));
 if(s.uploaded_bytes)T.push(tile("Uploaded file",fmtB(s.uploaded_bytes),"source (other codec, full size) \u2013 not comparable"));
 if(s.enc_s!=null)T.push(tile("Encode time",fmtT(s.enc_s),`${(s.frames/s.enc_s).toFixed(1)} frames/s`));
 T.push(tile("Decode time",fmtT(s.dec_s),`${(s.frames/Math.max(s.dec_s,1e-6)).toFixed(1)} frames/s`));
 T.push(tile("Structure",`${s.segments} GOP`,s.traj_segments!=null?`${s.traj_segments} trajectory segments`:(s.tools||"")));
 $("#tiles").innerHTML=T.join("");$("#results").hidden=false;
 const bb=$("#bitsbox");
 if(s.bits){const tot=Object.values(s.bits).reduce((a,b)=>a+b,0)||1;
  $("#bits").innerHTML=Object.entries(s.bits).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
   `<div class="b"><span>${LABELS[k]||k}</span><div class="t"><div class="f" style="width:${(100*v/tot).toFixed(1)}%"></div></div><span>${(100*v/tot).toFixed(1)} % \u00b7 ${fmtB(Math.round(v/8))}</span></div>`).join("");
  const rs=((s.bits.res_y||0)+(s.bits.res_c||0))/tot;
  $("#bitnote").textContent=`Residual share: ${(100*rs).toFixed(0)} % of the stream. If this stays high, the motion/trajectory model explains little of the picture and the transform residual carries the information.`;
  bb.hidden=false}else bb.hidden=true;
 $("#dl").innerHTML=`<a class="btn" href="/api/vtv/${id}">Download .vtv</a><a class="btn" href="/api/y4m/${id}">Download decoded .y4m</a>`;
}

/* ---------------- player ---------------- */
let P=null;const cv=$("#cv"),ctx=cv.getContext("2d"),ta=document.createElement("canvas"),tb=document.createElement("canvas");
function loadImgs(id,n,src,onp){let k=0;return Promise.all(Array.from({length:n},(_,i)=>new Promise((res,rej)=>{
 const im=new Image();im.onload=()=>{onp&&onp(++k);res(im)};im.onerror=()=>rej(new Error("frame "+i+" failed"));
 im.src=`/api/frame/${id}/${i}.png${src?"?src=1":""}`})))}
function stop(){if(P){P.playing=false}$("#play").innerHTML="&#9654; Play"}
async function loadPlayer(id,m){
 stop();P={id,n:m.frames,fps:m.fps||25,W:m.width,H:m.height,hasSrc:!!m.has_source,dec:null,src:null,idx:0,playing:false};
 $("#playerbox").hidden=false;$("#pmsg").className="";
 try{P.dec=await loadImgs(id,P.n,false,k=>$("#pmsg").textContent=`Loading decoded frames ${k}/${P.n}\u2026`)}
 catch(e){$("#pmsg").textContent="Error: "+e.message;$("#pmsg").className="err";return}
 for(const o of $("#view").options)o.disabled=(o.value!=="vtv")&&!P.hasSrc;$("#view").value="vtv";
 $("#seek").max=P.n-1;$("#seek").disabled=false;$("#play").disabled=false;
 sizeCanvas();show(0);$("#pmsg").textContent=`${P.n} frames \u00b7 ${P.W}\u00d7${P.H} \u00b7 ${P.fps.toFixed(2)} fps`}
function sizeCanvas(){const v=$("#view").value;cv.width=v==="side"?2*P.W:P.W;cv.height=P.H;cv.className=$("#pix").checked?"pix":""}
function pix(im){ta.width=tb.width=P.W;ta.height=tb.height=P.H;return im}
function show(i){P.idx=i;$("#seek").value=i;$("#time").textContent=`frame ${i+1}/${P.n} \u00b7 ${(i/P.fps).toFixed(2)} s`;
 const v=$("#view").value;
 if(v==="vtv")ctx.drawImage(P.dec[i],0,0);
 else if(v==="orig")ctx.drawImage(P.src[i],0,0);
 else if(v==="side"){ctx.drawImage(P.src[i],0,0);ctx.drawImage(P.dec[i],P.W,0);ctx.fillStyle="rgba(0,0,0,.6)";ctx.fillRect(0,0,70,16);ctx.fillRect(P.W,0,50,16);
  ctx.fillStyle="#fff";ctx.font="11px sans-serif";ctx.fillText("original",4,12);ctx.fillText("VTV",P.W+4,12)}
 else if(v==="diff"){pix();const a=ta.getContext("2d"),b=tb.getContext("2d");a.drawImage(P.src[i],0,0);b.drawImage(P.dec[i],0,0);
  const A=a.getImageData(0,0,P.W,P.H),B=b.getImageData(0,0,P.W,P.H);
  for(let k=0;k<A.data.length;k+=4){for(let c=0;c<3;c++)A.data[k+c]=Math.min(255,Math.abs(A.data[k+c]-B.data[k+c])*4);A.data[k+3]=255}
  ctx.putImageData(A,0,0)}}
function tick(now){if(!P||!P.playing)return;
 const sp=parseFloat($("#speed").value);let i=Math.max(0,Math.floor(P.base+(now-P.t0)/1000*P.fps*sp));
 if(i>=P.n){if($("#loop").checked){P.t0=now;P.base=0;i=0}else{show(P.n-1);stop();return}}
 if(i!==P.idx)show(i);requestAnimationFrame(tick)}
function play(){if(!P)return;if(P.idx>=P.n-1)P.idx=0;P.playing=true;P.base=P.idx;P.t0=performance.now();$("#play").innerHTML="&#10074;&#10074; Pause";requestAnimationFrame(tick)}
async function needSrc(){if(P.hasSrc&&!P.src){$("#pmsg").textContent="Loading original frames\u2026";
 P.src=await loadImgs(P.id,P.n,true,k=>$("#pmsg").textContent=`Loading original frames ${k}/${P.n}\u2026`);
 $("#pmsg").textContent=`${P.n} frames \u00b7 ${P.W}\u00d7${P.H} \u00b7 ${P.fps.toFixed(2)} fps`}}
$("#play").onclick=()=>P&&(P.playing?stop():play());
$("#seek").oninput=e=>{if(!P)return;const i=+e.target.value;if(P.playing){P.base=i;P.t0=performance.now()}show(i)};
$("#view").onchange=async()=>{if(!P)return;if($("#view").value!=="vtv")await needSrc();sizeCanvas();show(P.idx)};
$("#pix").onchange=()=>{if(P)sizeCanvas(),show(P.idx)};
document.addEventListener("keydown",e=>{if(!P||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)&&e.target.type!=="range"&&e.target.type!=="checkbox")return;
 if(e.code==="Space"){e.preventDefault();$("#play").click()}
 else if(e.code==="ArrowRight"){stop();show(Math.min(P.n-1,P.idx+1))}else if(e.code==="ArrowLeft"){stop();show(Math.max(0,P.idx-1))}});
$("#vtvfile").onchange=async e=>{const f=e.target.files[0];if(!f)return;$("#pmsg").className="";$("#pmsg").textContent="Uploading\u2026";
 try{const {id}=await upload("/api/play?name="+encodeURIComponent(f.name),f);
  const j=await poll(id,s=>$("#pmsg").textContent=s.message||s.status);if(j.status==="error")throw new Error(j.error);
  showStats(j.stats,id);await loadPlayer(id,j.stats)}
 catch(err){$("#pmsg").textContent="Error: "+err.message;$("#pmsg").className="err"}};
</script></body></html>
"""


def md_to_html(md):
    """Tiny markdown subset (#/## headings, - lists, ``` code, `code`, **bold**) for the docs panel."""
    import html as _h

    def inline(s):
        s = _h.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    out, para, in_code, in_list = [], [], False, False

    def flush():
        nonlocal para, in_list
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []
        if in_list:
            out.append("</ul>")
            in_list = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            flush()
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(_h.escape(line))
            continue
        if line.startswith("## "):
            flush()
            out.append("<h4>" + inline(line[3:]) + "</h4>")
        elif line.startswith("# "):
            flush()
            out.append("<h3>" + inline(line[2:]) + "</h3>")
        elif line.startswith("- "):
            if para:
                out.append("<p>" + inline(" ".join(para)) + "</p>")
                para = []
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + inline(line[2:]) + "</li>")
        elif line.startswith("  ") and in_list and line.strip():
            out[-1] = out[-1][:-5] + " " + inline(line.strip()) + "</li>"
        elif not line.strip():
            flush()
        else:
            if in_list:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def y4m_bytes(frames, fps=(25, 1)):
    H, W = frames[0][0].shape
    out = [f"YUV4MPEG2 W{W} H{H} F{fps[0]}:{fps[1]} Ip A1:1 C420jpeg\n".encode()]
    for Y, Cb, Cr in frames:
        out += [b"FRAME\n", np.ascontiguousarray(Y).tobytes(), np.ascontiguousarray(Cb).tobytes(),
                np.ascontiguousarray(Cr).tobytes()]
    return b"".join(out)


class Job:
    def __init__(self, kind, name):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.name = kind, name
        self.status, self.progress, self.message, self.error = "queued", 0.0, "Queued\u2026", None
        self.stats = self.vtv = self.dec = self.src = self.tmp = None
        self.fps = (25, 1)
        self.png = {}

    def set(self, status, progress, message):
        self.status, self.progress, self.message = status, float(progress), message

    def public(self):
        return dict(id=self.id, status=self.status, progress=self.progress, message=self.message,
                    error=self.error, stats=self.stats)


JOBS, JOB_ORDER = {}, []
JOBS_LOCK = threading.Lock()
WORK_SEM = threading.Semaphore(1)       # one heavy job at a time: pure-Python codec is CPU bound
MAX_JOBS = 8
LIMITS = dict(max_upload=500 * 1024 * 1024)


def new_job(kind, name):
    j = Job(kind, name)
    with JOBS_LOCK:
        JOBS[j.id] = j
        JOB_ORDER.append(j.id)
        while len(JOB_ORDER) > MAX_JOBS:
            old = JOBS.pop(JOB_ORDER.pop(0), None)
            if old is not None and old.tmp:
                try:
                    os.remove(old.tmp)
                except OSError:
                    pass
    return j


def _fps_pair(fps):
    from fractions import Fraction as _F
    f = _F(float(fps)).limit_denominator(1001)
    return (max(1, f.numerator), max(1, f.denominator))


def _tools_label(hdr):
    t = [n for n, on in (("global affine", hdr["geo"]), ("photometric", hdr["photo"])) if on]
    t.append("bilinear" if hdr["interp"] == 0 else "Catmull-Rom")
    return ", ".join(t)


def run_encode_job(job, path, opts):
    with WORK_SEM:
        try:
            job.set("reading", 0.02, "Reading & scaling video\u2026")
            frames, W, H, fps, info = read_video(
                path, width=opts["width"], max_frames=opts["frames"],
                progress=lambda d, t: job.set("reading", 0.02 + 0.06 * d / t, f"Reading video\u2026 {d} frames"))
            n = len(frames)
            cfg = EncoderConfig(qp=opts["qp"], gop=opts["gop"], use_global=opts["glob"], use_traj=opts["traj"],
                                use_photo=opts["photo"], interp=0 if opts["bil"] else 1)
            job.fps = _fps_pair(fps)
            t0 = time.time()
            def prog(d, t):
                msg = (f"Analysing motion\u2026 {d}/{n - 1}" if d < n
                       else f"Encoding frame {d - n + 1}/{n}")
                job.set("encoding", 0.08 + 0.72 * d / t, msg)
            data, st = Encoder(cfg, W, H, job.fps).encode(frames, progress=prog)
            enc_s = time.time() - t0
            job.set("verifying", 0.82, "Decoding & measuring quality\u2026")
            t0 = time.time()
            dec, dinfo = decode(data, conceal=False)
            dec_s = time.time() - t0
            q = sequence_quality(frames, dec)
            hdr = read_header(data)
            fps_f = job.fps[0] / job.fps[1]
            dur = n / fps_f
            raw = W * H * 3 // 2 * n
            job.vtv, job.dec, job.src = data, dec, frames
            job.stats = dict(kind="encode", file=job.name, uploaded_bytes=os.path.getsize(path),
                             frames=n, width=W, height=H, fps=fps_f, duration_s=dur, raw_bytes=raw,
                             vtv_bytes=len(data), ratio=raw / len(data), kbps=len(data) * 8 / dur / 1000,
                             bpp=len(data) * 8 / (W * H * n), enc_s=enc_s, dec_s=dec_s,
                             segments=st["n_segments"], traj_segments=st["n_traj_segments"],
                             bits=st["bits"], has_source=True, tools=_tools_label(hdr), qp=cfg.qp, **q)
            job.set("done", 1.0, "Done")
        except Exception as e:                                       # noqa: BLE001
            job.error = f"{type(e).__name__}: {e}"
            job.set("error", job.progress, "Failed")
        finally:
            try:
                os.remove(path)
                job.tmp = None
            except OSError:
                pass


def run_play_job(job, data):
    with WORK_SEM:
        try:
            job.set("decoding", 0.1, "Decoding\u2026")
            hdr = read_header(data)
            t0 = time.time()
            frames, info = decode(data, conceal=True)
            dec_s = time.time() - t0
            n, W, H = len(frames), hdr["width"], hdr["height"]
            fps_f = hdr["fps"][0] / hdr["fps"][1]
            dur = n / fps_f
            raw = W * H * 3 // 2 * n
            job.vtv, job.dec = data, frames
            job.fps = hdr["fps"]
            job.stats = dict(kind="play", file=job.name, frames=n, width=W, height=H, fps=fps_f,
                             duration_s=dur, raw_bytes=raw, vtv_bytes=len(data), ratio=raw / len(data),
                             kbps=len(data) * 8 / dur / 1000, bpp=len(data) * 8 / (W * H * n),
                             dec_s=dec_s, segments=hdr["n_segments"], tools=_tools_label(hdr),
                             has_source=False, problems=len(info["problems"]))
            job.set("done", 1.0, "Done")
        except Exception as e:                                       # noqa: BLE001
            job.error = f"{type(e).__name__}: {e}"
            job.set("error", 0, "Failed")


class Handler(BaseHTTPRequestHandler):
    server_version = "VTV/" + ".".join(map(str, VERSION))

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _job(self, jid):
        with JOBS_LOCK:
            return JOBS.get(jid)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            page = PAGE.replace("__VERSION__", ".".join(map(str, VERSION))).replace(
                "__DOCS__", md_to_html(DOC))
            return self._send(200, page, "text/html; charset=utf-8")
        m = re.match(r"^/api/job/([0-9a-f]+)$", p)
        if m:
            j = self._job(m.group(1))
            return self._send(200, j.public()) if j else self._send(404, dict(error="unknown job"))
        m = re.match(r"^/api/frame/([0-9a-f]+)/(\d+)\.png$", p)
        if m:
            j = self._job(m.group(1))
            src = "src=1" in (u.query or "")
            seq = (j.src if src else j.dec) if j else None
            i = int(m.group(2))
            if not seq or i >= len(seq):
                return self._send(404, dict(error="no such frame"))
            key = (src, i)
            if key not in j.png:
                j.png[key] = png_bytes(planes_to_rgb(seq[i]))
            return self._send(200, j.png[key], "image/png", {"Cache-Control": "max-age=3600"})
        m = re.match(r"^/api/(vtv|y4m)/([0-9a-f]+)$", p)
        if m:
            j = self._job(m.group(2))
            if not j or not j.vtv:
                return self._send(404, dict(error="not available"))
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(j.name or "video")[0])[:60] or "video"
            if m.group(1) == "vtv":
                body, name = j.vtv, stem + ".vtv"
            else:
                body, name = y4m_bytes(j.dec, j.fps), stem + "_decoded.y4m"
            return self._send(200, body, "application/octet-stream",
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        self._send(404, dict(error="not found"))

    def _read_body(self, dest):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            self._send(400, dict(error="empty upload"))
            return False
        if n > LIMITS["max_upload"]:
            self._send(413, dict(error="file too large (limit %d MB)" % (LIMITS["max_upload"] >> 20)))
            return False
        left = n
        while left:
            chunk = self.rfile.read(min(1 << 20, left))
            if not chunk:
                self._send(400, dict(error="upload interrupted"))
                return False
            dest.write(chunk)
            left -= len(chunk)
        return True

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        def geti(k, d, lo, hi):
            try:
                return max(lo, min(hi, int(q.get(k, [d])[0])))
            except ValueError:
                return d
        name = (q.get("name", ["video"])[0] or "video")[:120]
        if u.path == "/api/encode":
            ext = re.sub(r"[^A-Za-z0-9]", "", os.path.splitext(name)[1])[:6]
            fd, path = tempfile.mkstemp(suffix="." + (ext or "bin"), prefix="vtv_up_")
            with os.fdopen(fd, "wb") as f:
                ok = self._read_body(f)
            if not ok:
                os.remove(path)
                return
            opts = dict(qp=geti("qp", 32, 10, 48), width=geti("width", 256, 64, 640),
                        frames=geti("frames", 60, 1, 600), gop=geti("gop", 32, 2, 120),
                        glob=bool(geti("glob", 1, 0, 1)), traj=bool(geti("traj", 1, 0, 1)),
                        photo=bool(geti("photo", 1, 0, 1)), bil=bool(geti("bil", 0, 0, 1)))
            job = new_job("encode", name)
            job.tmp = path
            threading.Thread(target=run_encode_job, args=(job, path, opts), daemon=True).start()
            return self._send(200, dict(id=job.id))
        if u.path == "/api/play":
            buf = _stdio.BytesIO()
            if not self._read_body(buf):
                return
            job = new_job("play", name)
            threading.Thread(target=run_play_job, args=(job, buf.getvalue()), daemon=True).start()
            return self._send(200, dict(id=job.id))
        self._send(404, dict(error="not found"))


def serve(host="127.0.0.1", port=8765, open_browser=True):
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("no free port found")
    httpd.daemon_threads = True
    url = f"http://{'localhost' if host in ('0.0.0.0', '') else host}:{p}/"
    print(f"VTV web app running at {url}   (Ctrl+C to stop)")
    print("ffmpeg:", find_ffmpeg() or "not found -> falling back to OpenCV if installed")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


# =============================================================================
# Self-test and command line
# =============================================================================
def selftest(verbose=True):
    """Fast end-to-end verification of this file (about 10-20 s).  Returns True if all pass."""
    results = []

    def check(name, cond):
        results.append(bool(cond))
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(1)
    x = rng.integers(-255, 256, (200, 8, 8))
    check("integer 8x8 transform round trip (max error <= 3 on worst-case noise)",
          np.abs(idct8(fdct8(x)) - x).max() <= 3)

    enc, ctx = RangeEncoder(), new_ctx(4)
    vals = [int(rng.integers(0, 1000)) for _ in range(3000)]
    e_ue, e_se = new_ctx(12), new_ctx(18)
    for v in vals:
        enc.bit(ctx, v % 4, v & 1)
        enc.ue(e_ue, v)
        enc.se(e_se, v - 500)
    dec_rc, ctx, e_ue, e_se = RangeDecoder(enc.finish()), new_ctx(4), new_ctx(12), new_ctx(18)
    good = True
    for v in vals:
        good &= dec_rc.bit(ctx, v % 4) == (v & 1) and dec_rc.ue(e_ue, 0) == v and dec_rc.se(e_se, 0) == v - 500
    check("range coder round trip (bits, Exp-Golomb, signed)", good)

    img = rng.integers(0, 256, (48, 64))
    check("warp: identity transform is an exact copy",
          np.array_equal(predict_luma(img, IDENTITY), img))
    check("warp: integer translation equals a pixel shift",
          np.array_equal(predict_luma(img, (0, 0, 0, 0, 3 * 64, 0, 0, 0))[:, :-4], img[:, 3:-1]))

    W, H, N = 96, 64, 10
    frames = demo_clip(W, H, N)
    variants = (("default", {}), ("no global/photometric", dict(use_global=False, use_photo=False)),
                ("no trajectories", dict(use_traj=False)), ("bilinear filter", dict(interp=INTERP_BILINEAR)))
    data0 = None
    for label, kw in variants:
        enc_ = Encoder(EncoderConfig(qp=30, gop=6, **kw), W, H)
        data, st = enc_.encode(frames)
        dec, info = decode(data, conceal=False)
        same = len(dec) == N and all(np.array_equal(a, b) for e, d in zip(enc_.last_recon, dec)
                                     for a, b in zip(crop_planes(e, W, H), d))
        check(f"encoder reconstruction == decoder output, bit-exact [{label}]", same and not info["problems"])
        if data0 is None:
            data0, dec0 = data, dec
    q = sequence_quality(frames, dec0)
    check(f"decoded quality is sane (PSNR-Y {q['psnr_y']:.1f} dB > 30)", q["psnr_y"] > 30)

    hdr = read_header(data0)
    idx = read_index(data0, hdr)
    seg1 = decode_segment(data0, hdr, idx[1])
    check("random access: segment 1 decodes from its own bytes alone",
          all(np.array_equal(a, b) for f1, f2 in zip(seg1, dec0[idx[1]["frame_start"]:]) for a, b in zip(f1, f2)))
    bad = bytearray(data0)
    bad[idx[0]["offset"] + 30] ^= 0xFF
    dec_b, info_b = decode(bytes(bad), conceal=True)
    check("corrupt segment is detected (CRC) and concealed; later segments still decode",
          len(dec_b) == N and [p[0] for p in info_b["problems"]] == [0] and
          all(np.array_equal(a, b) for f1, f2 in zip(dec_b[idx[1]["frame_start"]:], dec0[idx[1]["frame_start"]:])
              for a, b in zip(f1, f2)))
    try:
        decode(b"VTVB" + bytes(60), conceal=False)
        rejected = False
    except Exception:                                                    # noqa: BLE001
        rejected = True
    check("malformed file is rejected without crashing", rejected)
    ok = all(results)
    if verbose:
        print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


def _cli_progress(prefix):
    last = [-1]

    def cb(d, t):
        pct = int(100 * d / max(t, 1))
        if pct != last[0]:
            last[0] = pct
            print(f"\r{prefix} {pct:3d} %", end="", flush=True)
    return cb


def cmd_encode(a):
    if a.input == "demo":
        frames, W, H, fps = demo_clip(192, 128, min(a.frames, 64)), 192, 128, 25.0
        src_bytes = None
    else:
        frames, W, H, fps, _ = read_video(a.input, width=a.width, max_frames=a.frames, start=a.start,
                                          progress=None)
        src_bytes = os.path.getsize(a.input)
    n = len(frames)
    print(f"input : {n} frames, {W}x{H}, {fps:.2f} fps")
    cfg = EncoderConfig(qp=a.qp, gop=a.gop, use_global=not a.no_global, use_traj=not a.no_traj,
                        use_photo=not a.no_photo, interp=INTERP_BILINEAR if a.bilinear else INTERP_CATMULL)
    t0 = time.time()
    data, st = Encoder(cfg, W, H, _fps_pair(fps)).encode(frames, progress=_cli_progress("encoding"))
    enc_s = time.time() - t0
    print()
    with open(a.output, "wb") as f:
        f.write(data)
    raw = W * H * 3 // 2 * n
    dur = n / fps
    print(f"output: {a.output}")
    print(f"  size          {len(data):,} bytes   (raw YUV 4:2:0: {raw:,} bytes)")
    print(f"  compression   {raw / len(data):.1f} : 1   ({len(data) * 8 / dur / 1000:.1f} kbit/s, "
          f"{len(data) * 8 / (W * H * n):.3f} bit/pixel)")
    print(f"  structure     {st['n_segments']} GOP segment(s), {st['n_traj_segments']} trajectory segment(s)")
    print(f"  encode time   {enc_s:.1f} s  ({n / enc_s:.1f} frames/s)")
    if not a.no_verify:
        t0 = time.time()
        dec, _ = decode(data, conceal=False)
        q = sequence_quality(frames, dec)
        print(f"  quality       PSNR-Y {q['psnr_y']:.2f} dB (min {q['psnr_y_min']:.2f}), "
              f"PSNR-YUV {q['psnr_yuv']:.2f} dB, SSIM-Y {q['ssim_y']:.4f}   [decode {time.time() - t0:.1f} s]")
    tot = sum(st["bits"].values()) or 1
    print("  bits          " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in
                                        sorted(st["bits"].items(), key=lambda kv: -kv[1])))
    if src_bytes:
        print(f"  (source file is {src_bytes:,} bytes in its own codec at full size - not comparable)")


def cmd_decode(a):
    with open(a.input, "rb") as f:
        data = f.read()
    t0 = time.time()
    frames, info = decode(data, conceal=True)
    hdr = info["header"]
    for i, msg in info["problems"]:
        print(f"warning: segment {i} concealed ({msg})")
    fps = hdr["fps"]
    out = a.output
    if out.lower().endswith(".y4m"):
        with open(out, "wb") as f:
            f.write(y4m_bytes(frames, fps))
    elif out.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
        ff = find_ffmpeg()
        if not ff:
            raise SystemExit("ffmpeg is needed to write " + out + "; use a .y4m file or a directory instead")
        y4m = y4m_bytes(frames, fps)
        for codec in (["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"], ["-c:v", "mpeg4", "-q:v", "2"]):
            r = subprocess.run([ff, "-y", "-v", "error", "-f", "yuv4mpegpipe", "-i", "-"] + codec + [out],
                               input=y4m, capture_output=True)
            if r.returncode == 0:
                break
        else:
            raise SystemExit("ffmpeg failed: " + r.stderr.decode("utf-8", "replace")[-300:])
    else:
        os.makedirs(out, exist_ok=True)
        for i, fr in enumerate(frames):
            with open(os.path.join(out, f"frame_{i:05d}.png"), "wb") as f:
                f.write(png_bytes(planes_to_rgb(fr)))
    print(f"decoded {len(frames)} frames ({hdr['width']}x{hdr['height']}) in {time.time() - t0:.1f} s -> {out}")


def cmd_info(a):
    with open(a.input, "rb") as f:
        data = f.read()
    hdr = read_header(data)
    idx = read_index(data, hdr)
    fps = hdr["fps"][0] / hdr["fps"][1]
    n = hdr["n_frames"]
    print(f"file      {a.input}  ({len(data):,} bytes)")
    print(f"format    VTV bitstream v{hdr['version'][0]}.{hdr['version'][1]}, 4:2:0 8-bit")
    print(f"video     {hdr['width']}x{hdr['height']}, {n} frames @ {fps:.3f} fps = {n / fps:.2f} s")
    print(f"tools     {_tools_label(hdr)}")
    print(f"bitrate   {len(data) * 8 / (n / fps) / 1000:.1f} kbit/s, "
          f"{len(data) * 8 / (hdr['width'] * hdr['height'] * n):.3f} bit/pixel, "
          f"{W_RAW(hdr) / len(data):.1f}:1 vs raw")
    print(f"segments  {hdr['n_segments']}  (each starts with an I frame; independently decodable)")
    for i, e in enumerate(idx):
        try:
            nf, qi, qp, payload = read_segment(data, e)
            st = f"QP I/P {qi}/{qp}, CRC ok"
        except BitstreamError as ex:
            st = f"BAD: {ex}"
        print(f"  #{i:<3d} frames {e['frame_start']}-{e['frame_start'] + e['n_frames'] - 1}  "
              f"offset {e['offset']:>9,}  {e['size']:>9,} bytes  {st}")


def W_RAW(hdr):
    return hdr["width"] * hdr["height"] * 3 // 2 * hdr["n_frames"]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="vtv.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="VTV - Vector-Time Video, research codec (beta %d.%d).\n"
                    "Run `python vtv.py doc` for the full documentation." % VERSION,
        epilog="examples:\n  python vtv.py serve\n  python vtv.py encode clip.mp4 clip.vtv --qp 30 --width 256 --frames 60\n"
               "  python vtv.py encode demo demo.vtv\n  python vtv.py decode clip.vtv clip_decoded.y4m\n"
               "  python vtv.py info clip.vtv\n  python vtv.py selftest")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web app (upload, statistics, player)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--max-upload-mb", type=int, default=500)
    e = sub.add_parser("encode", help="encode a video (or `demo`) to .vtv")
    e.add_argument("input", help="video file (mp4, mov, mkv, webm, y4m ...) or the word `demo`")
    e.add_argument("output")
    e.add_argument("--qp", type=int, default=32, help="quantiser 10-48, lower = better/larger (default 32)")
    e.add_argument("--width", type=int, default=256, help="max width in pixels (default 256)")
    e.add_argument("--frames", type=int, default=60, help="max frames to encode (default 60)")
    e.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    e.add_argument("--gop", type=int, default=32, help="max frames per independent segment (default 32)")
    e.add_argument("--no-global", action="store_true", help="disable global affine motion")
    e.add_argument("--no-traj", action="store_true", help="disable polynomial trajectories")
    e.add_argument("--no-photo", action="store_true", help="disable photometric gain/offset model")
    e.add_argument("--bilinear", action="store_true", help="bilinear instead of Catmull-Rom interpolation")
    e.add_argument("--no-verify", action="store_true", help="skip decode + quality measurement")
    d = sub.add_parser("decode", help="decode .vtv to .y4m, .mp4 (needs ffmpeg) or a folder of PNGs")
    d.add_argument("input")
    d.add_argument("output")
    i = sub.add_parser("info", help="show header, segments and checksums of a .vtv")
    i.add_argument("input")
    sub.add_parser("selftest", help="verify this file works (bit-exact round trips etc.)")
    sub.add_parser("doc", help="print the full documentation")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        LIMITS["max_upload"] = a.max_upload_mb * 1024 * 1024
        serve(a.host, a.port, not a.no_browser)
    elif a.cmd == "encode":
        cmd_encode(a)
    elif a.cmd == "decode":
        cmd_decode(a)
    elif a.cmd == "info":
        cmd_info(a)
    elif a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif a.cmd == "doc":
        print(DOC)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
