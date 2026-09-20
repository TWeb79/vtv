# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
import numpy as np
from vtv.synth import CLIPS
from vtv.io import rgb_to_planes
from vtv.motion import estimate_global
from vtv.trajectory import (theta_to_units, plan, eval_poly, eval_segment, default_tolerances, UNITS)

_cache = {}
def _luma(name):
    if name not in _cache:
        _cache[name] = [rgb_to_planes(f)[0].astype(float) for f in CLIPS[name]()]
    return _cache[name]


def test_pan_estimate_matches_ground_truth():
    Y = _luma("pan")
    for t in (5, 15, 25):
        th, info = estimate_global(Y[t], Y[t - 1])
        assert abs(th[4] - 1.7) < 0.08 and abs(th[5] - 0.6) < 0.08, th
        assert abs(th[0] - 1) < 2e-3 and abs(th[3] - 1) < 2e-3
        assert info["mad"] < info["mad0"] * 0.5


def test_zoom_and_rotation_estimate():
    Y = _luma("zoom")
    t = 16
    th, _ = estimate_global(Y[t], Y[t - 1])
    s_true = (1 - 0.005 * t) / (1 - 0.005 * (t - 1))       # world scale ratio -> a11 = a22
    assert abs(th[0] - s_true) < 1.5e-3 and abs(th[3] - s_true) < 1.5e-3, (th, s_true)
    Y = _luma("rotate")
    th, _ = estimate_global(Y[10], Y[9])
    rot = np.deg2rad(0.2)
    assert abs(th[2] - rot) < 1.5e-3 and abs(th[1] + rot) < 1.5e-3, th


def test_fade_estimated_as_gain():
    Y = _luma("fade")
    th, _ = estimate_global(Y[10], Y[9], photo=True)
    g_true = (1 - 0.55 * 10 / 31) / (1 - 0.55 * 9 / 31)
    assert abs(th[6] - g_true) < 0.006, (th[6], g_true)


def test_poly_eval_and_plan_constant_velocity():
    n = 12
    tgt = np.zeros((n, 8))
    tgt[:, 4] = 1.7 * 64 + 0.0            # constant tx
    tgt[:, 5] = 0.6 * 64
    tg, tp = default_tolerances(32)
    segs = plan(tgt, np.zeros(8, int), True, True, tg, tp, 192, 128, 16, True)
    assert len(segs) == 1 and segs[0]["og"] == 0 and segs[0]["L"] == n
    assert np.abs(segs[0]["vals"][:, 4] - tgt[:, 4]).max() < 1


def test_plan_picks_order2_for_acceleration_and_splits_on_jumps():
    n = 16
    tau = np.arange(n)
    tgt = np.zeros((n, 8))
    tgt[:, 4] = (0.5 + 0.08 * tau) * 64          # linear velocity ramp = quadratic position
    tg, tp = default_tolerances(28)
    segs = plan(tgt, np.zeros(8, int), True, False, tg, tp, 192, 128, 16, True)
    assert len(segs) == 1 and segs[0]["og"] == 1        # slope-only fit is enough
    tgt[:, 4] = 64 * np.where(tau < 8, 1.0, 5.0)         # velocity jump
    segs = plan(tgt, np.zeros(8, int), True, False, tg, tp, 192, 128, 16, True)
    assert len(segs) >= 2
    off = 0
    for s in segs:
        err = np.abs(s["vals"][:, 4] - tgt[off:off + s["L"], 4])
        assert err.max() < tg * 64 * 1.6
        off += s["L"]
    assert off == n


def test_no_traj_mode_gives_unit_segments():
    n = 9
    tgt = np.zeros((n, 8)); tgt[:, 4] = np.arange(n) * 64.0
    tg, tp = default_tolerances(32)
    segs = plan(tgt, np.zeros(8, int), True, False, tg, tp, 192, 128, 16, allow_traj=False)
    assert [s["L"] for s in segs] == [1] * n


def test_eval_segment_matches_plan():
    rng = np.random.default_rng(0)
    tgt = np.cumsum(rng.normal(0, 3, (10, 8)), axis=0)
    tg, tp = default_tolerances(30)
    for s in plan(tgt, np.zeros(8, int), True, True, tg, tp, 192, 128, 16, True):
        v = eval_segment(s["c"], s["og"], s["op"], True, True, s["L"])
        assert np.array_equal(v, s["vals"])
