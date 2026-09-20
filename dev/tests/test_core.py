# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
import numpy as np
from vtv.dct import fdct8, idct8, ZZ, quantize, dequantize, qstep, blockify, unblockify
from vtv.rangecoder import RangeEncoder, RangeDecoder, new_ctx


def test_transform_roundtrip_error_small():
    rng = np.random.default_rng(1)
    x = rng.integers(-255, 256, size=(500, 8, 8))
    y = idct8(fdct8(x))
    err = (y - x).astype(float)
    assert np.abs(err).max() <= 3                 # worst case, full-range white noise
    assert np.sqrt((err ** 2).mean()) < 0.8       # typical error well below 1 level


def test_transform_orthonormal_scale():
    x = np.full((1, 8, 8), 100)
    c = fdct8(x)
    assert abs(int(c[0, 0, 0]) - 800) <= 1          # DC = 8*v
    assert np.abs(c[0].ravel()[1:]).max() <= 1


def test_energy_preserved():
    rng = np.random.default_rng(2)
    x = rng.integers(-100, 100, size=(200, 8, 8))
    c = fdct8(x)
    ratio = (c.astype(float) ** 2).sum() / (x.astype(float) ** 2).sum()
    assert 0.98 < ratio < 1.02


def test_quant_dequant():
    c = np.array([-500, -30, -1, 0, 1, 30, 500])
    for qp in (10, 25, 40):
        lv = quantize(c, qp)
        r = dequantize(lv, qp)
        assert np.all(np.abs(r - c) <= qstep(qp) * 1.01 + 1)
        assert np.all(np.sign(r) * np.sign(c) >= 0)


def test_zigzag_permutation():
    assert sorted(ZZ.tolist()) == list(range(64))
    assert ZZ[0] == 0 and ZZ[1] == 1 and ZZ[2] == 8


def test_blockify_roundtrip():
    p = np.arange(16 * 24).reshape(16, 24)
    assert np.array_equal(unblockify(blockify(p)), p)


def test_range_coder_roundtrip_mixed():
    rng = np.random.default_rng(3)
    n = 20000
    kinds = rng.integers(0, 5, n)
    vals = []
    enc = RangeEncoder()
    c_bit = new_ctx(4); c_ue = new_ctx(12); c_se = new_ctx(18); c_bt = new_ctx(64)
    for k in kinds:
        if k == 0:
            v = int(rng.random() < 0.9); vals.append(v); enc.bit(c_bit, 0, v)
        elif k == 1:
            v = int(rng.geometric(0.2)) - 1; vals.append(v); enc.ue(c_ue, v)
        elif k == 2:
            v = int(rng.integers(-300, 300)); vals.append(v); enc.se(c_se, v)
        elif k == 3:
            v = int(rng.integers(0, 64)); vals.append(v); enc.bittree(c_bt, 6, v)
        else:
            v = int(rng.integers(0, 1 << 11)); vals.append(v); enc.direct(11, v)
    data = enc.finish()
    dec = RangeDecoder(data)
    c_bit = new_ctx(4); c_ue = new_ctx(12); c_se = new_ctx(18); c_bt = new_ctx(64)
    for k, v in zip(kinds, vals):
        if k == 0: got = dec.bit(c_bit, 0)
        elif k == 1: got = dec.ue(c_ue, 0)
        elif k == 2: got = dec.se(c_se, 0)
        elif k == 3: got = dec.bittree(c_bt, 6, 0)
        else: got = dec.direct(11)
        assert got == v


def test_range_coder_compresses_skewed_source():
    enc = RangeEncoder(stats=True)
    ctx = new_ctx(1)
    rng = np.random.default_rng(4)
    bits = (rng.random(50000) < 0.05).astype(int)
    for b in bits:
        enc.bit(ctx, 0, int(b))
    data = enc.finish()
    h = -(0.05 * np.log2(0.05) + 0.95 * np.log2(0.95)) * 50000 / 8
    assert len(data) < h * 1.06                      # within ~6% of entropy
    assert abs(enc.bits["misc"] / 8 - len(data)) < 0.03 * len(data) + 8   # stats ~ real size
