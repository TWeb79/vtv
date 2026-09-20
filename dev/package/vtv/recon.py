# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Frame reconstruction shared by the encoder (closed loop) and the decoder.

Because both sides call exactly this code, encoder-side reconstruction and
decoder output are bit-identical by construction (verified by the tests).

Pass A  : inter/skip areas   = warp(reference) + dequantised residual
Pass B  : intra ("new information") blocks, raster order, from neighbouring
          reconstructed samples (DC / horizontal / vertical prediction).
"""
import numpy as np

from .dct import ZZ, dequantize, idct8, unblockify
from .syntax import MODE_INTER, MODE_INTRA
from .warp import predict_chroma, predict_luma


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
