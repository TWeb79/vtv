# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Reference decoder.  Deterministic, integer-only reconstruction.

`decode_segment` needs nothing but the bytes of one segment (random access).
`decode` walks all segments and, if a segment is corrupt or missing, conceals
it (repeat last good frame, or mid-gray at the start) and resumes at the next
segment -- the resilience behaviour of concept section 31.
"""
import numpy as np

from .bitstream import read_header, read_index, read_segment
from .io import coded_size, crop_planes
from .rangecoder import BitstreamError, RangeDecoder
from .recon import reconstruct_i, reconstruct_p
from .syntax import Contexts, FrameSyntax, code_frame, code_traj


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
