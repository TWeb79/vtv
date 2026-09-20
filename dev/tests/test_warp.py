# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
import numpy as np
from scipy import ndimage
from vtv.warp import (sample, predict_luma, predict_chroma, luma_ref_positions,
                      TAPS, IDENTITY, photometric)
from vtv.config import INTERP_BILINEAR, INTERP_CATMULL


def _img(h=64, w=96, seed=0):
    rng = np.random.default_rng(seed)
    base = ndimage.gaussian_filter(rng.random((h, w)), 2.0)
    base = (base - base.min()) / (base.max() - base.min())
    return (base * 255).astype(np.int64)


def test_tap_tables_sum_to_unity():
    for tab in TAPS.values():
        assert (tab.sum(axis=1) == 128).all()
    assert list(TAPS[INTERP_CATMULL][0]) == [0, 128, 0, 0]      # phase 0 = copy


def test_identity_is_exact_copy():
    img = _img()
    for f in (INTERP_BILINEAR, INTERP_CATMULL):
        assert np.array_equal(predict_luma(img, IDENTITY, interp=f), img)
        c = img[::2, ::2]
        assert np.array_equal(predict_chroma(c, IDENTITY, interp=f), c)


def test_integer_translation_matches_shift():
    img = _img()
    gp = (0, 0, 0, 0, 3 * 64, -2 * 64, 0, 0)                    # tx=+3, ty=-2 pixels
    out = predict_luma(img, gp)
    H, W = img.shape
    ref = np.pad(img, 8, mode="edge")
    exp = ref[8 - 2: 8 - 2 + H, 8 + 3: 8 + 3 + W]
    assert np.array_equal(out, exp)


def test_subpel_close_to_scipy_cubic():
    img = _img()
    gp = (0, 0, 0, 0, int(0.37 * 64), int(-0.61 * 64), 0, 0)
    out = predict_luma(img, gp)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(float)
    ref = ndimage.map_coordinates(img.astype(float), [yy - 0.61 + 0, xx + 0.37], order=3, mode="nearest")
    inner = (slice(8, -8), slice(8, -8))
    assert np.abs(out[inner] - ref[inner]).mean() < 1.0


def test_catmull_sharper_than_bilinear_on_repeated_warp():
    img = _img()
    gp = (0, 0, 0, 0, 20, 13, 0, 0)          # ~0.3 px shift, repeated -> blur accumulates
    a = b = img.copy()
    for _ in range(10):
        a = predict_luma(a, gp, interp=INTERP_BILINEAR)
        b = predict_luma(b, gp, interp=INTERP_CATMULL)
    assert b[8:-8, 8:-8].std() > a[8:-8, 8:-8].std() * 1.02


def test_deterministic_and_dtype_independent():
    img = _img()
    gp = (300, -200, 150, -100, 77, -31, 0, 0)
    a = predict_luma(img, gp)
    b = predict_luma(img.astype(np.int32), gp)
    c = predict_luma(img.copy(), gp)
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_rotation_zoom_plausible():
    img = _img(96, 128)
    th = np.deg2rad(2.0)
    s = 1.02
    gp = (int(round((s * np.cos(th) - 1) * 65536)), int(round(-s * np.sin(th) * 65536)),
          int(round(s * np.sin(th) * 65536)), int(round((s * np.cos(th) - 1) * 65536)), 0, 0, 0, 0)
    out = predict_luma(img, gp)
    H, W = img.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u = cx + s * np.cos(th) * (xx - cx) - s * np.sin(th) * (yy - cy)
    v = cy + s * np.sin(th) * (xx - cx) + s * np.cos(th) * (yy - cy)
    ref = ndimage.map_coordinates(img.astype(float), [v, u], order=3, mode="nearest")
    assert np.abs(out[10:-10, 10:-10] - ref[10:-10, 10:-10]).mean() < 1.0


def test_photometric():
    s = np.array([[0, 100, 255]])
    assert np.array_equal(photometric(s, (0,) * 6 + (0, 0), False), s)
    out = photometric(s, (0,) * 6 + (-512, 0), False)            # gain 0.5
    assert list(out[0]) == [0, 50, 128]
    out = photometric(s, (0,) * 6 + (0, 32), False)              # offset +2 gray
    assert list(out[0]) == [2, 102, 255]
    c = photometric(np.array([[128, 228]]), (0,) * 6 + (-512, 0), True)
    assert list(c[0]) == [128, 178]


def test_out_of_frame_clamps():
    img = _img()
    gp = (0, 0, 0, 0, 500 * 64, 500 * 64, 0, 0)
    out = predict_luma(img, gp)
    assert (out == img[-1, -1]).all()
