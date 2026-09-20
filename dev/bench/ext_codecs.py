# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Reference codecs via ffmpeg (x264 / x265 / AV1 / VP9) for matched-quality comparison."""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vtv.io import read_y4m  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# name -> (ffmpeg video args template with {q}, quality parameter list)
PROFILES = {
    # closest structural match to VTV: P frames only, single reference, 32-frame GOP
    "x264-P":  ("-c:v libx264 -preset veryslow -tune psnr -crf {q} "
                "-x264-params bframes=0:ref=1:keyint=32:min-keyint=32:scenecut=40:threads=1", [18, 22, 26, 30, 34, 38]),
    # what people actually use (B-frames, multi-ref, deblocking, CABAC ...)
    "x264-full": ("-c:v libx264 -preset veryslow -tune psnr -crf {q} "
                  "-x264-params keyint=32:min-keyint=32:scenecut=40:threads=1", [18, 22, 26, 30, 34, 38]),
    "x265":    ("-c:v libx265 -preset medium -tune psnr -crf {q} "
                "-x265-params keyint=32:min-keyint=32:pools=1:frame-threads=1:log-level=error", [20, 24, 28, 32, 36, 40]),
    "av1":     ("-c:v libaom-av1 -cpu-used 4 -crf {q} -b:v 0 -g 32 -threads 1", [28, 34, 40, 46, 52, 58]),
}


def _strip_sei(data, hevc):
    """Drop SEI NAL units (x264/x265 write a ~600-byte encoder-info banner that is not
    video payload).  Everything else, including SPS/PPS and start codes, is counted."""
    pos = [m.start() for m in re.finditer(b"\x00\x00\x01", data)]
    keep = 0
    for i, p in enumerate(pos):
        end = pos[i + 1] if i + 1 < len(pos) else len(data)
        b = data[p + 3]
        t = ((b >> 1) & 0x3F) if hevc else (b & 0x1F)
        if (hevc and t in (39, 40)) or (not hevc and t == 6):
            continue
        keep += end - p
    return keep


def encode_ext(y4m_in, profile, q, workdir=None, n_expected=None):
    """Encode with an external codec; return (payload bytes, decoded frames).

    Payload bytes exclude encoder-info SEI banners and container framing so that the
    comparison with VTV (which counts header+index+CRC) is not biased against them."""
    args, _ = PROFILES[profile]
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        bs = os.path.join(td, "out.bin")
        fmt = "-f obu" if profile == "av1" else ("-f hevc" if profile == "x265" else "-f h264")
        cmd = f"{FFMPEG} -v error -y -i {y4m_in} {args.format(q=q)} -pix_fmt yuv420p {fmt} {bs}"
        subprocess.run(cmd.split(), check=True)
        size = os.path.getsize(bs)
        if profile != "av1":
            size = _strip_sei(open(bs, "rb").read(), hevc=(profile == "x265"))
        dec = os.path.join(td, "dec.y4m")
        subprocess.run(f"{FFMPEG} -v error -y -i {bs} -pix_fmt yuv420p {dec}".split(), check=True)
        frames, *_ = read_y4m(dec)
    if n_expected is not None:
        assert len(frames) == n_expected, f"{profile}: decoded {len(frames)} frames, expected {n_expected}"
    return size, frames
