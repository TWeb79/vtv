# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Procedural test clips (concept spec section 50 / experimental phase 0).

Every clip is deterministic (seeded) and returns a list of RGB uint8 frames.
Clips are rendered by sampling a large textured "world" with a *cubic* sampler
along a known camera path, then compositing moving sprites, gain changes and
noise.  Because the ground-truth camera path is known, global-motion
estimation and trajectory fitting can be validated exactly.
"""
import numpy as np
from scipy import ndimage

W_DEFAULT, H_DEFAULT, N_DEFAULT = 192, 128, 32


def make_world(seed, H=640, W=960):
    rng = np.random.default_rng(seed)

    def noise(sigma):
        n = ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma, mode="wrap")
        return n / n.std()

    tex = 0.55 * noise(1.1) + 0.9 * noise(2.5) + 1.1 * noise(7) + 1.6 * noise(26)
    tex /= tex.std()
    world = np.empty((H, W, 3))
    for c in range(3):
        world[..., c] = 122 + 34 * (0.75 * tex + 0.55 * noise(4 + 3 * c) + 0.25 * noise(1.5))
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(70):                                   # hard-edged shapes -> edges
        cx, cy = rng.integers(0, W), rng.integers(0, H)
        col = rng.uniform(30, 225, 3)
        if rng.random() < 0.5:
            hw, hh = rng.integers(8, 70), rng.integers(6, 50)
            m = (np.abs(xx - cx) < hw) & (np.abs(yy - cy) < hh)
        else:
            r = rng.integers(8, 45)
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        a = rng.uniform(0.55, 1.0)
        world[m] = (1 - a) * world[m] + a * col
    for _ in range(25):                                   # thin dark lines
        x0 = rng.integers(0, W - 100)
        y0 = rng.integers(0, H - 100)
        L = rng.integers(40, 160)
        ang = rng.uniform(0, np.pi)
        for t in np.linspace(0, L, L * 2):
            x, y = int(x0 + t * np.cos(ang)), int(y0 + t * np.sin(ang))
            if 0 <= x < W and 0 <= y < H:
                world[y, x] = 25
    world = ndimage.gaussian_filter(world, (0.6, 0.6, 0))
    return np.clip(world, 0, 255)


def render(world, W, H, cam):
    """cam = (tx, ty, scale, theta_rad) : output pixel -> world position."""
    tx, ty, s, th = cam
    Hw, Ww = world.shape[:2]
    cx0, cy0 = Ww / 2 + tx, Hw / 2 + ty
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - W / 2, yy - H / 2
    wx = cx0 + s * (np.cos(th) * dx - np.sin(th) * dy)
    wy = cy0 + s * (np.sin(th) * dx + np.cos(th) * dy)
    out = np.empty((H, W, 3))
    for c in range(3):
        out[..., c] = ndimage.map_coordinates(world[..., c], [wy, wx], order=3, mode="reflect")
    return out


def _sprite(frame, kind, cx, cy, size, color, W, H):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if kind == "disk":
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - size
        shade = 0.75 + 0.25 * np.clip((xx - cx) / max(size, 1), -1, 1)
    else:
        d = np.maximum(np.abs(xx - cx) - size * 1.3, np.abs(yy - cy) - size * 0.8)
        shade = 0.8 + 0.2 * np.sin((xx + yy) / 2.5)
    a = np.clip(0.5 - d, 0, 1)[..., None]                 # 1-px soft edge
    col = np.array(color)[None, None, :] * shade[..., None]
    return frame * (1 - a) + col * a


def _finish(frames):
    return [np.clip(np.floor(f + 0.5), 0, 255).astype(np.uint8) for f in frames]


# ------------------------------------------------------------------ clips
def clip_static_obj(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    world = make_world(11)
    bg = render(world, W, H, (0, 0, 1, 0))
    out = []
    for t in range(N):
        f = _sprite(bg, "disk", 30 + 2.3 * t, 40 + 0.9 * t, 13, (235, 190, 60), W, H)
        f = _sprite(f, "rect", W - 20 - 1.6 * t, 90 + 0.4 * t, 11, (60, 110, 220), W, H)
        out.append(f)
    return _finish(out)


def clip_pan(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    world = make_world(21)
    return _finish([render(world, W, H, (1.7 * t, 0.6 * t, 1, 0)) for t in range(N)])


def clip_pan_accel(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    world = make_world(22)
    return _finish([render(world, W, H, (0.5 * t + 0.04 * t * t, 0.25 * t, 1, 0)) for t in range(N)])


def clip_zoom(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    world = make_world(23)
    return _finish([render(world, W, H, (0.3 * t, 0, 1.0 - 0.005 * t, 0)) for t in range(N)])


def clip_rotate(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    world = make_world(24)
    return _finish([render(world, W, H, (0, 0, 1, np.deg2rad(0.2 * t))) for t in range(N)])


def clip_pan_objects(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    world = make_world(25)
    out = []
    for t in range(N):
        f = render(world, W, H, (1.2 * t, 0.3 * t, 1, 0))
        f = _sprite(f, "disk", 40 + 1.0 * t, 35 + 0.5 * t, 12, (240, 80, 70), W, H)
        f = _sprite(f, "rect", 150 - 2.2 * t, 95 - 0.6 * t, 10, (70, 220, 120), W, H)
        f = _sprite(f, "disk", W + 15 - 3.0 * t, 60 + 1.0 * t, 9, (250, 250, 90), W, H)   # enters
        out.append(f)
    return _finish(out)


def clip_fade(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    world = make_world(26)
    out = []
    for t in range(N):
        g = 1.0 - 0.55 * t / (N - 1)
        out.append(render(world, W, H, (0.5 * t, 0.1 * t, 1, 0)) * g)
    return _finish(out)


def clip_noisy_pan(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT, sigma=3.5):
    world = make_world(27)
    rng = np.random.default_rng(99)
    return _finish([render(world, W, H, (1.4 * t, 0.5 * t, 1, 0)) + rng.normal(0, sigma, (H, W, 3))
                    for t in range(N)])


def clip_cut(W=W_DEFAULT, H=H_DEFAULT, N=N_DEFAULT):
    a, b = make_world(28), make_world(29)
    out = []
    for t in range(N):
        if t < N // 2:
            out.append(render(a, W, H, (1.5 * t, 0.4 * t, 1, 0)))
        else:
            out.append(render(b, W, H, (-1.1 * (t - N // 2), 0.7 * (t - N // 2), 1, 0)))
    return _finish(out)


CLIPS = {
    "static_obj": clip_static_obj,
    "pan": clip_pan,
    "pan_accel": clip_pan_accel,
    "zoom": clip_zoom,
    "rotate": clip_rotate,
    "pan_objects": clip_pan_objects,
    "fade": clip_fade,
    "noisy_pan": clip_noisy_pan,
    "cut": clip_cut,
}
