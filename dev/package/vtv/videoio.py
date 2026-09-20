# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Reading ordinary video files (mp4, mov, mkv, webm ...) through ffmpeg.

The codec itself only knows raw 4:2:0 planes; this module is the convenience
layer used by the CLI and the web app.  audio is dropped, video is scaled
(never enlarged) to `width` keeping the aspect ratio, and truncated to
`max_frames`.
"""
import os
import shutil
import subprocess
import tempfile

from .io import read_y4m


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                                            # bundled binary of the imageio-ffmpeg package
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                               # noqa: BLE001
        return None


def read_video(path, width=256, max_frames=60):
    """Return (frames, W, H, (fps_num, fps_den)) as visible uint8 4:2:0 planes."""
    if path.lower().endswith(".y4m"):
        frames, W, H, fps = read_y4m(path, max_frames=max_frames)
        if not frames:
            raise ValueError("no frames in y4m file")
        return frames, W, H, fps
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found - install ffmpeg (or `pip install imageio-ffmpeg`) "
                           "to read mp4/mov/mkv/webm files, or provide a .y4m file")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "in.y4m")
        vf = f"scale='trunc(min({int(width)},iw)/2)*2':-2:flags=area"
        cmd = [ff, "-v", "error", "-y", "-i", path, "-an", "-vf", vf,
               "-frames:v", str(int(max_frames)), "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("ffmpeg could not read the video: " + (r.stderr.strip()[-300:] or "unknown error"))
        frames, W, H, fps = read_y4m(out, max_frames=max_frames)
    if not frames:
        raise RuntimeError("no video frames found in the file")
    return frames, W, H, fps
