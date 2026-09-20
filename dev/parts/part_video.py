# =============================================================================
# Video input, PNG output, metrics, demo clip  (everything here is outside the
# normative codec: it only feeds frames in and measures results)
# =============================================================================
def png_bytes(rgb):
    """uint8 (H, W, 3) -> PNG file bytes (no external imaging library needed)."""
    H, W, _ = rgb.shape
    raw = np.concatenate([np.zeros((H, 1), np.uint8), np.ascontiguousarray(rgb).reshape(H, W * 3)], axis=1)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def find_ffmpeg():
    """Path of an ffmpeg executable (PATH, or the imageio-ffmpeg wheel), else None."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                            # noqa: BLE001
        return None


def _probe(path, ff):
    """fps / duration of a video via `ffmpeg -i` (parses its banner)."""
    info = dict(fps=None, duration=None, src_w=None, src_h=None)
    try:
        r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=30)
        txt = r.stderr.decode("utf-8", "replace")
    except Exception:                                            # noqa: BLE001
        return info
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt) or re.search(r"(\d+(?:\.\d+)?)\s*tbr", txt)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", txt)
    if m:
        info["src_w"], info["src_h"] = int(m.group(1)), int(m.group(2))
    return info


def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _read_ppm(f):
    """Read one binary PPM (P6) image from a stream -> (H, W, 3) uint8 or None at EOF."""
    magic = f.read(2)
    if len(magic) < 2:
        return None
    if magic != b"P6":
        raise ValueError("unexpected data from ffmpeg")
    toks = []
    while len(toks) < 3:
        c = f.read(1)
        if not c:
            return None
        if c.isspace():
            continue
        if c == b"#":
            f.readline()
            continue
        tok = c
        while True:
            c = f.read(1)
            if not c or c.isspace():
                break
            tok += c
        toks.append(int(tok))
    W, H, _ = toks
    buf = _read_exact(f, W * H * 3)
    if buf is None:
        return None
    return np.frombuffer(buf, np.uint8).reshape(H, W, 3)


def read_video(path, width=256, max_frames=60, start=0.0, progress=None):
    """Read a video file -> (frames, W, H, fps, info).

    frames : list of (Y, Cb, Cr) uint8 planes (4:2:0, even size, full-range BT.601)
    Width is capped at `width` (aspect kept, even dimensions); at most `max_frames`
    frames are read.  Uses ffmpeg if available, else OpenCV; `.y4m` files are read as-is.
    """
    if path.lower().endswith(".y4m"):
        frames, W, H, fps = read_y4m(path, max_frames=max_frames)
        return frames, W, H, fps[0] / fps[1], dict(source="y4m")
    ff = find_ffmpeg()
    if ff:
        info = _probe(path, ff)
        cmd = [ff, "-v", "error", "-nostats"]
        if start:
            cmd += ["-ss", str(start)]
        cmd += ["-i", path, "-an", "-vf", f"scale=w='trunc(min(iw,{int(width)})/2)*2':h=-2:flags=area",
                "-frames:v", str(int(max_frames)), "-f", "image2pipe", "-c:v", "ppm", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = []
        try:
            while True:
                rgb = _read_ppm(proc.stdout)
                if rgb is None:
                    break
                frames.append(rgb_to_planes(rgb))
                if progress:
                    progress(len(frames), int(max_frames))
        finally:
            proc.stdout.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            proc.wait()
        if not frames:
            raise RuntimeError("ffmpeg could not decode any frame: " + err.strip()[-300:])
        H, W = frames[0][0].shape
        info.update(source="ffmpeg")
        return frames, W, H, info["fps"] or 25.0, info
    try:
        import cv2
    except ImportError:
        raise RuntimeError("No video reader found. Install ffmpeg (https://ffmpeg.org) or "
                           "`pip install opencv-python` (or `pip install imageio-ffmpeg`).")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    W = H = None
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if W is None:
            h0, w0 = bgr.shape[:2]
            W = max(2, (min(w0, width) // 2) * 2)
            H = max(2, int(round(h0 * W / w0 / 2)) * 2)
        if bgr.shape[1] != W or bgr.shape[0] != H:
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(rgb_to_planes(bgr[..., ::-1]))
        if progress:
            progress(len(frames), max_frames)
    cap.release()
    if not frames:
        raise RuntimeError("OpenCV could not decode any frame from this file")
    return frames, W, H, fps, dict(source="opencv", fps=fps,
                                   duration=None, src_w=None, src_h=None)


# ---- metrics ------------------------------------------------------------------
def psnr(a, b, peak=255.0):
    m = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(peak * peak / m))


def _box_mean(x, win):
    c = np.pad(np.cumsum(np.cumsum(x, axis=0), axis=1), ((1, 0), (1, 0)))
    return (c[win:, win:] - c[:-win, win:] - c[win:, :-win] + c[:-win, :-win]) / (win * win)


def ssim(a, b, win=7):
    """Mean SSIM of one plane (uniform 7x7 window; matches scikit-image's default)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if min(a.shape) < win:
        return 1.0 if np.array_equal(a, b) else 0.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = _box_mean(a, win), _box_mean(b, win)
    va, vb = _box_mean(a * a, win) - ma * ma, _box_mean(b * b, win) - mb * mb
    cov = _box_mean(a * b, win) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())


def sequence_quality(src, dec):
    """Per-clip PSNR-Y (mean/min), PSNR-YUV (6:1:1) and SSIM-Y."""
    py, pyuv, ss = [], [], []
    for s, d in zip(src, dec):
        p = [psnr(s[i], d[i]) for i in range(3)]
        py.append(p[0])
        pyuv.append((6 * p[0] + p[1] + p[2]) / 8)
        ss.append(ssim(s[0], d[0]))
    return dict(psnr_y=float(np.mean(py)), psnr_y_min=float(np.min(py)),
                psnr_yuv=float(np.mean(pyuv)), ssim_y=float(np.mean(ss)))


def bd_rate(ref_pts, test_pts):
    """Bjontegaard delta-rate in percent (negative = test needs fewer bits).
    pts: list of (bitrate, quality), >= 4 points each."""
    r1, q1 = np.log10([p[0] for p in ref_pts]), np.array([p[1] for p in ref_pts])
    r2, q2 = np.log10([p[0] for p in test_pts]), np.array([p[1] for p in test_pts])
    p1, p2 = np.polyfit(q1, r1, 3), np.polyfit(q2, r2, 3)
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        return float("nan")
    i1, i2 = np.polyint(p1), np.polyint(p2)
    a1 = (np.polyval(i1, hi) - np.polyval(i1, lo)) / (hi - lo)
    a2 = (np.polyval(i2, hi) - np.polyval(i2, lo)) / (hi - lo)
    return float((10 ** (a2 - a1) - 1) * 100)


# ---- demo clip (numpy only) ----------------------------------------------------
def demo_clip(W=192, H=128, N=32, seed=7):
    """Synthetic clip: textured world, camera pan with sub-pel motion + a moving disk.
    Returns a list of (Y, Cb, Cr) uint8 planes.  Handy for trying VTV without any video file."""
    rng = np.random.default_rng(seed)
    Ww, Hw = W + int(2 * N) + 48, H + int(N) + 48
    yy, xx = np.mgrid[0:Hw, 0:Ww].astype(np.float64)
    world = np.empty((Hw, Ww, 3))
    for c in range(3):
        acc = np.full((Hw, Ww), 125.0)
        for _ in range(26):
            f, th = rng.uniform(0.01, 0.28), rng.uniform(0, np.pi)
            acc += rng.uniform(5, 19) * np.sin(2 * np.pi * f * (np.cos(th) * xx + np.sin(th) * yy)
                                               + rng.uniform(0, 6.28))
        world[..., c] = acc
    for _ in range(28):
        cx, cy = rng.uniform(0, Ww), rng.uniform(0, Hw)
        col = rng.uniform(40, 220, 3)
        if rng.random() < 0.5:
            m = (np.abs(xx - cx) < rng.uniform(6, 30)) & (np.abs(yy - cy) < rng.uniform(5, 22))
        else:
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < rng.uniform(6, 22) ** 2
        world[m] = 0.35 * world[m] + 0.65 * col
    frames = []
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
    for t in range(N):
        ox, oy = 8 + 1.6 * t, 8 + 0.7 * t
        x0, y0 = int(ox), int(oy)
        fx, fy = ox - x0, oy - y0
        w = world[y0:y0 + H + 1, x0:x0 + W + 1]
        img = ((w[:-1, :-1] * (1 - fx) + w[:-1, 1:] * fx) * (1 - fy) +
               (w[1:, :-1] * (1 - fx) + w[1:, 1:] * fx) * fy)
        cxs, cys = 30 + 2.2 * t, 40 + 0.8 * t
        a = np.clip(0.5 - (np.sqrt((gx - cxs) ** 2 + (gy - cys) ** 2) - 12), 0, 1)[..., None]
        img = img * (1 - a) + np.array([235.0, 190.0, 60.0]) * a
        frames.append(rgb_to_planes(np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)))
    return frames
