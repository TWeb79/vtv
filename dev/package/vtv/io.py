# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Frame containers, colour conversion and YUV4MPEG2 (.y4m) I/O.

Internal frame representation: a tuple (Y, Cb, Cr) of int32 arrays, 4:2:0,
8-bit, full-range BT.601-style YCbCr, padded to a multiple of CTU.
"""
import numpy as np

from .config import CTU


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
        f.write(f"YUV4MPEG2 W{W} H{H} F{fps[0]}:{fps[1]} Ip A1:1 C420mpeg2\n".encode())
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
