# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Quality and rate metrics: PSNR, SSIM, Bjontegaard delta-rate."""
import numpy as np
from scipy import ndimage


def psnr(a, b, peak=255.0):
    m = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(peak * peak / m))


def psnr_yuv(fa, fb):
    """Per-plane PSNR of (Y, Cb, Cr) frames and the usual (6Y+U+V)/8 combination."""
    py, pu, pv = (psnr(fa[i], fb[i]) for i in range(3))
    return py, pu, pv, (6 * py + pu + pv) / 8


def ssim(a, b, win=7):
    """Mean SSIM (uniform window, Wang et al. constants) on one plane."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    f = lambda x: ndimage.uniform_filter(x, win, mode="reflect")
    ma, mb = f(a), f(b)
    va, vb = f(a * a) - ma * ma, f(b * b) - mb * mb
    cov = f(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    r = win // 2
    return float(s[r:-r, r:-r].mean())


def sequence_quality(src, dec):
    """Average PSNR-Y / PSNR-YUV / SSIM-Y over a clip."""
    py = [psnr(s[0], d[0]) for s, d in zip(src, dec)]
    pyuv = [psnr_yuv(s, d)[3] for s, d in zip(src, dec)]
    ss = [ssim(s[0], d[0]) for s, d in zip(src, dec)]
    return dict(psnr_y=float(np.mean(py)), psnr_yuv=float(np.mean(pyuv)), ssim_y=float(np.mean(ss)))


def bd_rate(ref_pts, test_pts):
    """Bjontegaard delta-rate in percent (negative = test needs fewer bits).

    pts: list of (bitrate, quality) with >= 4 points.  Uses cubic polynomial fits of
    log-rate vs quality integrated over the overlapping quality range.
    """
    r1, q1 = np.log10([p[0] for p in ref_pts]), np.array([p[1] for p in ref_pts])
    r2, q2 = np.log10([p[0] for p in test_pts]), np.array([p[1] for p in test_pts])
    p1, p2 = np.polyfit(q1, r1, 3), np.polyfit(q2, r2, 3)
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        return float("nan")
    i1, i2 = np.polyint(p1), np.polyint(p2)
    avg1 = (np.polyval(i1, hi) - np.polyval(i1, lo)) / (hi - lo)
    avg2 = (np.polyval(i2, hi) - np.polyval(i2, lo)) / (hi - lo)
    return float((10 ** (avg2 - avg1) - 1) * 100)
