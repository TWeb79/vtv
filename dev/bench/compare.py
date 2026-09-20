# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Matched-quality comparison of VTV against x264 (and optionally x265 / AV1) via ffmpeg.

    python bench/compare.py pan                       # a procedural clip (see vtv/synth.py)
    python bench/compare.py path/to/clip.y4m 32       # any 4:2:0 y4m, first 32 frames
    python bench/compare.py clip.y4m 32 --codecs x264-P,x264-full,x265,av1

Every codec gets the identical source planes.  Sizes count everything a decoder needs
(for VTV: header + index + CRC) but exclude the ~600-byte encoder-info SEI banner that
x264/x265 embed, which would otherwise bias small-clip comparisons.  BD-rate is computed
from rate-quality curves (positive = VTV needs more bits at equal quality).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from ext_codecs import PROFILES, encode_ext                          # noqa: E402
from rd import load_clip, vtv_curve                                   # noqa: E402
from vtv.io import rgb_to_planes, write_y4m                           # noqa: E402
from vtv.metrics import bd_rate, sequence_quality                     # noqa: E402
from vtv.synth import CLIPS                                           # noqa: E402


def clip_path(name):
    if name.endswith(".y4m"):
        return name
    d = os.path.join(HERE, "results", "clips")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".y4m")
    if not os.path.exists(p):
        write_y4m(p, [rgb_to_planes(f) for f in CLIPS[name]()], (25, 1))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("frames", type=int, nargs="?", default=32)
    ap.add_argument("--codecs", default="x264-P,x264-full")
    ap.add_argument("--qps", default="24,28,32,36,40")
    a = ap.parse_args()
    path = clip_path(a.clip)
    frames, W, H = load_clip(path, a.frames)
    dur = len(frames) / 25.0
    kbps = lambda nbytes: nbytes * 8 / dur / 1000
    t0 = time.time()

    vt = vtv_curve(path, a.frames, {}, tuple(int(q) for q in a.qps.split(",")))
    print(f"clip {os.path.basename(path)}  {W}x{H}  {len(frames)} frames")
    for p in vt:
        print(f"  VTV qp{p['qp']:2d}  {p['kbps']:7.1f} kbit/s  PSNR-Y {p['psnr_y']:.2f}  SSIM-Y {p['ssim_y']:.4f}")
    for prof in a.codecs.split(","):
        pts = []
        for q in PROFILES[prof][1]:
            size, dec = encode_ext(path, prof, q, n_expected=len(frames))
            qq = sequence_quality(frames, dec)
            pts.append(dict(kbps=kbps(size), psnr_y=qq["psnr_y"], ssim_y=qq["ssim_y"]))
        d_psnr = bd_rate([(p["kbps"], p["psnr_y"]) for p in pts], [(p["kbps"], p["psnr_y"]) for p in vt])
        d_ssim = bd_rate([(p["kbps"], p["ssim_y"]) for p in pts], [(p["kbps"], p["ssim_y"]) for p in vt])
        print(f"  BD-rate VTV vs {prof:10s}: {d_psnr:+7.1f}% (PSNR-Y)   {d_ssim:+7.1f}% (SSIM-Y)")
    print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
