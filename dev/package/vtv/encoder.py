# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""VTV encoder (closed loop, rate-distortion optimised).

Pipeline per GOP (= independently decodable segment)
  1. I frame: intra DC/H/V prediction + 8x8 transform coding.
  2. Global affine+photometric parameters are estimated between consecutive
     *source* frames, then fitted as polynomial trajectories (trajectory.py).
  3. Every P frame is predicted from the previous *reconstructed* frame with the
     trajectory-defined global warp.  A quadtree of 32/16/8 leaves selects, per
     region, SKIP / INTER (+ quarter-pel delta motion) / INTRA ("new
     information") by minimising  J = D + lambda*R  with D measured in the
     transform domain (orthonormal => equals pixel-domain SSE).
  4. Residuals are transform-coded; a block is zeroed when that lowers J.
  5. All syntax elements are coded with the adaptive range coder.

The encoder never invents a decoder path: after choosing syntax it always
reconstructs with the *shared* recon module, so its reference frames are
bit-identical to what the decoder will produce.
"""
import numpy as np

from .bitstream import write_file
from .config import (EncoderConfig, FLAG_GEO, FLAG_INTERP_SHIFT, FLAG_PHOTO)
from .dct import blockify, dequantize, fdct8, qstep, quantize, ZZ
from .io import pad_planes
from .motion import estimate_global
from .rangecoder import RangeEncoder
from .recon import (delta_maps, intra_pred, recon_intra_block, reconstruct_i,
                    reconstruct_inter, reconstruct_p)
from .syntax import (Contexts, FrameSyntax, MODE_INTER, MODE_INTRA, MODE_SKIP,
                     code_frame, code_traj)
from .trajectory import default_tolerances, plan, theta_to_units
from .warp import predict_chroma, predict_luma

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
