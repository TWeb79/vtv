# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Command line interface:  python -m vtv {encode,decode,info} ..."""
import argparse
import sys
import time

from . import EncoderConfig, Encoder, decode
from .bitstream import read_header, read_index
from .io import write_y4m
from .metrics import sequence_quality
from .videoio import read_video


def _fps_pair(fps):
    from fractions import Fraction
    f = Fraction(float(fps[0]) / float(fps[1])).limit_denominator(1001)
    return (max(1, f.numerator), max(1, f.denominator))


def cmd_encode(a):
    frames, W, H, fps = read_video(a.input, width=a.width, max_frames=a.frames)
    cfg = EncoderConfig(qp=a.qp, gop=a.gop, use_global=not a.no_global, use_traj=not a.no_traj,
                        use_photo=not a.no_photo, interp=0 if a.bilinear else 1)
    t0 = time.time()
    data, st = Encoder(cfg, W, H, _fps_pair(fps)).encode(frames)
    enc_s = time.time() - t0
    open(a.output, "wb").write(data)
    raw = W * H * 3 // 2 * len(frames)
    print(f"{len(frames)} frames {W}x{H} -> {len(data)} bytes  ({raw / len(data):.1f}:1 vs raw 4:2:0, "
          f"{len(data) * 8 / (len(frames) / (fps[0] / fps[1])) / 1000:.1f} kbit/s)  "
          f"segments={st['n_segments']} trajectory-segments={st['n_traj_segments']}  {enc_s:.1f}s")
    tot = sum(st["bits"].values()) or 1
    print("bit budget: " + "  ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(st["bits"].items(), key=lambda kv: -kv[1])))
    if a.verify:
        dec, info = decode(data, conceal=False)
        q = sequence_quality(frames, dec)
        print(f"decoded {len(dec)} frames: PSNR-Y {q['psnr_y']:.2f} dB  SSIM-Y {q['ssim_y']:.4f}")


def cmd_decode(a):
    data = open(a.input, "rb").read()
    hdr = read_header(data)
    t0 = time.time()
    frames, info = decode(data, conceal=True)
    write_y4m(a.output, frames, hdr["fps"])
    print(f"decoded {len(frames)} frames in {time.time() - t0:.1f}s -> {a.output}"
          + (f"  (concealed segments: {[p[0] for p in info['problems']]})" if info["problems"] else ""))


def cmd_info(a):
    data = open(a.input, "rb").read()
    h = read_header(data)
    idx = read_index(data, h)
    print(f"VTV v{h['version'][0]}.{h['version'][1]}  {h['width']}x{h['height']}  {h['fps'][0]}/{h['fps'][1]} fps  "
          f"{h['n_frames']} frames  {h['n_segments']} segments  {len(data)} bytes")
    print(f"tools: global-affine={h['geo']} photometric={h['photo']} interpolation={'catmull-rom' if h['interp'] else 'bilinear'}")
    for i, e in enumerate(idx):
        print(f"  segment {i}: frames {e['frame_start']}..{e['frame_start'] + e['n_frames'] - 1}  offset {e['offset']}  {e['size']} bytes")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m vtv", description="VTV research codec")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="encode a video (mp4/mov/mkv/webm/y4m) to .vtv")
    e.add_argument("input"); e.add_argument("output")
    e.add_argument("--qp", type=int, default=32, help="quantiser 10-48, lower = better/bigger (default 32)")
    e.add_argument("--width", type=int, default=256, help="max width in pixels (default 256)")
    e.add_argument("--frames", type=int, default=60, help="max frames (default 60)")
    e.add_argument("--gop", type=int, default=32, help="max frames per independently decodable segment")
    e.add_argument("--no-global", action="store_true", help="ablation: disable global affine motion")
    e.add_argument("--no-traj", action="store_true", help="ablation: per-frame parameters instead of trajectories")
    e.add_argument("--no-photo", action="store_true", help="ablation: disable gain/offset model")
    e.add_argument("--bilinear", action="store_true", help="ablation: bilinear instead of Catmull-Rom")
    e.add_argument("--verify", action="store_true", help="decode afterwards and print PSNR/SSIM")
    e.set_defaults(fn=cmd_encode)
    d = sub.add_parser("decode", help="decode .vtv to .y4m (playable with ffplay/VLC/mpv)")
    d.add_argument("input"); d.add_argument("output"); d.set_defaults(fn=cmd_decode)
    i = sub.add_parser("info", help="print header and segment index"); i.add_argument("input"); i.set_defaults(fn=cmd_info)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
