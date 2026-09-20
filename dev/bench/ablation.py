# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Ablation ladder: which VTV idea earns its bits?   (BD-rate vs. the block-only baseline)

    A0 block-only      : quadtree + regional delta motion + transform residual (no global model)
    A1 +global affine  : per-frame affine parameters (trajectory segments of length 1)
    A2 +trajectory     : polynomial temporal trajectories of the global parameters
    A3 +photometric    : gain/offset trajectory ("colour" channel)

All rungs share the same entropy coder, quantiser, RDO and interpolation, so the
differences are attributable to the representation, not to tool maturity.

    python bench/ablation.py [--frames 16] [--clips pan,zoom,fade] [--real path.y4m ...]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rd import curve_bd, vtv_curve                                   # noqa: E402
from vtv.io import rgb_to_planes, write_y4m                          # noqa: E402
from vtv.synth import CLIPS                                          # noqa: E402

LADDER = [
    ("A0 block-only",     dict(use_global=False, use_traj=False, use_photo=False)),
    ("A1 +global affine", dict(use_global=True,  use_traj=False, use_photo=False)),
    ("A2 +trajectory",    dict(use_global=True,  use_traj=True,  use_photo=False)),
    ("A3 +photometric",   dict(use_global=True,  use_traj=True,  use_photo=True)),
]


def synth_path(name):
    d = os.path.join(HERE, "results", "clips")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".y4m")
    if not os.path.exists(p):
        write_y4m(p, [rgb_to_planes(f) for f in CLIPS[name]()], (25, 1))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--clips", default="pan,zoom,fade,noisy_pan")
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--qps", default="26,31,36,41")
    a = ap.parse_args()
    qps = tuple(int(x) for x in a.qps.split(","))
    paths = [(c, synth_path(c)) for c in a.clips.split(",") if c] + [(os.path.basename(r)[:-4], r) for r in a.real]

    print(f"{'clip':14s} " + " ".join(f"{n:>18s}" for n, _ in LADDER[1:]) + "   residual share A0/A3")
    for name, path in paths:
        curves = [vtv_curve(path, a.frames, kw, qps) for _, kw in LADDER]
        bd = [curve_bd(curves[0], c) for c in curves[1:]]
        share = lambda c: sum(p["bit_share"].get("res_y", 0) + p["bit_share"].get("res_c", 0) for p in c) / len(c)
        print(f"{name:14s} " + " ".join(f"{b:+17.1f}%" for b in bd) + f"   {share(curves[0]):.0%} / {share(curves[3]):.0%}", flush=True)


if __name__ == "__main__":
    main()
