# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Encoder reconstruction must equal decoder output bit-for-bit, for every tool."""
import numpy as np
import pytest
from vtv import Encoder, EncoderConfig, decode
from vtv.io import crop_planes, rgb_to_planes
from vtv.synth import CLIPS

_CLIPS = {}


def _frames(name, n):
    """First n frames of a procedural clip (generated once per test session)."""
    if name not in _CLIPS:
        rgb = CLIPS[name]()
        _CLIPS[name] = ([rgb_to_planes(f) for f in rgb], rgb[0].shape[1], rgb[0].shape[0])
    planes, W, H = _CLIPS[name]
    return planes[:n], W, H


def _check(name, n, cfg):
    fr, W, H = _frames(name, n)
    enc = Encoder(cfg, W, H)
    data, st = enc.encode(fr)
    dec, info = decode(data, conceal=False)
    assert not info["problems"] and len(dec) == len(fr)
    for k, (e, d) in enumerate(zip(enc.last_recon, dec)):
        ev = crop_planes(e, W, H)
        for pe, pd in zip(ev, d):
            assert np.array_equal(pe, pd), f"{name}: frame {k} mismatch"
    return data, st


@pytest.mark.parametrize("name", ["pan", "pan_objects", "cut", "fade", "static_obj"])
def test_default_config(name):
    _check(name, 8, EncoderConfig(qp=30, gop=6))


@pytest.mark.parametrize("kw", [
    dict(use_global=False, use_photo=False),
    dict(use_traj=False),
    dict(interp=0),
    dict(subpel=False, rdoq=False),
])
def test_ablation_configs(kw):
    _check("pan_objects", 7, EncoderConfig(qp=34, gop=32, **kw))


def test_odd_size_padding():
    fr, W, H = _frames("pan", 4)
    fr = [(y[:120, :180], cb[:60, :90], cr[:60, :90]) for (y, cb, cr) in fr]
    enc = Encoder(EncoderConfig(qp=30), 180, 120)
    data, _ = enc.encode(fr)
    dec, info = decode(data, conceal=False)
    assert dec[0][0].shape == (120, 180) and dec[0][1].shape == (60, 90)
    for e, d in zip(enc.last_recon, dec):
        for pe, pd in zip(crop_planes(e, 180, 120), d):
            assert np.array_equal(pe, pd)


def test_scene_cut_creates_segment_and_intra_fallback():
    fr, W, H = _frames("cut", 20)
    # (a) automatic cut detection -> new independently decodable segment at frame 16
    data, st = _check("cut", 20, EncoderConfig(qp=30))
    assert st["n_segments"] == 2 and st["gops"][1][0] == 16
    # (b) cut detection disabled: the P frame must fall back to INTRA leaves ("new information")
    _, st_ref = _check("cut", 16, EncoderConfig(qp=30))           # no cut inside: intra bits = I frame only
    data2, st2 = _check("cut", 20, EncoderConfig(qp=30, cut_mad=1e9))
    assert st2["n_segments"] == 1
    assert st2["bits"]["intra"] > 1.3 * st_ref["bits"]["intra"]   # P frame coded intra leaves at the cut


def test_random_access_and_index():
    from vtv.bitstream import read_header, read_index
    from vtv.decoder import decode_segment
    fr, W, H = _frames("cut", 20)
    data, st = Encoder(EncoderConfig(qp=30), W, H).encode(fr)
    full, _ = decode(data)
    hdr = read_header(data)
    idx = read_index(data, hdr)
    assert [e["frame_start"] for e in idx] == [0, 16]
    seg1 = decode_segment(data, hdr, idx[1])          # needs nothing but segment 1's bytes
    assert len(seg1) == 4
    for a, b in zip(seg1, full[16:]):
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_corruption_is_contained_and_concealed():
    fr, W, H = _frames("cut", 20)
    data, _ = Encoder(EncoderConfig(qp=30), W, H).encode(fr)
    good, _ = decode(data)
    from vtv.bitstream import read_header, read_index
    idx = read_index(data, read_header(data))
    bad = bytearray(data)
    bad[idx[0]["offset"] + 60] ^= 0xFF                # flip bits inside segment 0
    dec, info = decode(bytes(bad), conceal=True)
    assert len(dec) == 20
    assert [p[0] for p in info["problems"]] == [0]    # only segment 0 is lost
    assert "CRC" in info["problems"][0][1]
    for a, b in zip(dec[16:], good[16:]):             # segment 1 unaffected
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.all(dec[0][0] == 128)                   # concealed with neutral gray
    with pytest.raises(Exception):
        decode(bytes(bad), conceal=False)


def test_garbage_input_never_crashes_or_hangs():
    import time
    rng = np.random.default_rng(5)
    fr, W, H = _frames("pan", 3)
    data, _ = Encoder(EncoderConfig(qp=30), W, H).encode(fr)
    from vtv.bitstream import read_header, read_index, HDR
    idx = read_index(data, read_header(data))
    t0 = time.time()
    for trial in range(25):                            # corrupt payload AND fix the CRC -> parser sees garbage
        import zlib, struct
        b = bytearray(data)
        e = idx[0]
        for _ in range(6):
            b[e["offset"] + 8 + int(rng.integers(0, e["size"] - 16))] = int(rng.integers(0, 256))
        body = bytes(b[e["offset"]: e["offset"] + e["size"] - 4])
        b[e["offset"] + e["size"] - 4: e["offset"] + e["size"]] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
        dec, info = decode(bytes(b), conceal=True)      # must return, never raise/hang
        assert len(dec) == 3
    assert time.time() - t0 < 60
    with pytest.raises(Exception):
        decode(b"VTVB" + bytes(60))
