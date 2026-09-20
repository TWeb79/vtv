# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Bitstream syntax, written once and used for both writing and reading.

Every function takes a range coder `rc` (RangeEncoder or RangeDecoder) and a
`FrameSyntax` `fs`.  When encoding, `fs` holds the data to write; when decoding,
`fs` starts zeroed and is filled in as symbols are decoded.  Values passed to
`rc.*` are ignored by the decoder, so no separate "reader" exists.

Frame structure
---------------
  I frame : every 16x16 unit is intra; no partition syntax.
  P frame : per 32x32 CTU a quadtree (leaves 32/16/8) with per-leaf mode
            SKIP  (global transform only, no residual),
            INTER (global transform + regional delta motion + residual),
            INTRA (only leaves >= 16: "new information", spatial prediction).
  Residual: 8x8 blocks, coded per CTU (luma), then chroma per 16x16 unit.
"""
import numpy as np

from .rangecoder import new_ctx, BitstreamError
from .trajectory import GEO, PHOTO, NCH, eval_segment

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
