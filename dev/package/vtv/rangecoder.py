# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Adaptive binary range coder (LZMA-style) with a *symmetric* API.

`RangeEncoder` and `RangeDecoder` expose identical methods.  Every method takes
the value to encode and *returns the value* (encoder: the value passed in,
decoder: the value decoded, argument ignored).  This lets the bitstream syntax
be written exactly once and used for both directions, which removes a whole
class of encoder/decoder divergence bugs.

Probabilities are 11-bit (P(bit==0) in 1..2047), adaptation shift 5, exactly as
in LZMA, so the coder is fully integer and deterministic.
"""
import math
from collections import defaultdict

PROB_BITS = 11
PROB_INIT = 1 << (PROB_BITS - 1)
MOVE = 5
TOP = 1 << 24
MASK32 = 0xFFFFFFFF
MAX_PREFIX = 32


class BitstreamError(Exception):
    pass


# cost (bits) of coding a symbol with probability p/2048, for statistics only
_COST0 = [0.0] * (1 << PROB_BITS)
_COST1 = [0.0] * (1 << PROB_BITS)
for _p in range(1, 1 << PROB_BITS):
    _COST0[_p] = -math.log2(_p / (1 << PROB_BITS))
    _COST1[_p] = -math.log2(1.0 - _p / (1 << PROB_BITS))


def new_ctx(n):
    return [PROB_INIT] * n


class _Sym:
    """Binarisations shared by encoder and decoder (built on `bit`/`direct`)."""

    def bittree(self, ctx, nbits, v):
        idx = 1
        for k in range(nbits - 1, -1, -1):
            b = self.bit(ctx, idx, (v >> k) & 1)
            idx = (idx << 1) | b
        return idx - (1 << nbits)

    def ue(self, ctx, v, off=0):
        """Order-0 Exp-Golomb with adaptive prefix contexts ctx[off:]."""
        cap = len(ctx) - off - 1
        n = (v + 1).bit_length() - 1 if self.is_enc else 0
        k = 0
        while True:
            more = self.bit(ctx, off + (k if k < cap else cap), 1 if k < n else 0)
            if not more:
                break
            k += 1
            if k > MAX_PREFIX:
                raise BitstreamError("exp-golomb prefix too long")
        suffix = self.direct(k, (v + 1) - (1 << k)) if k else 0
        return (1 << k) + suffix - 1

    def se(self, ctx, v):
        """Signed value: ctx layout = [zero_flag, sign, prefix...]."""
        nz = self.bit(ctx, 0, 1 if v != 0 else 0)
        if not nz:
            return 0
        neg = self.bit(ctx, 1, 1 if v < 0 else 0)
        m = self.ue(ctx, abs(v) - 1, off=2) + 1
        return -m if neg else m


class RangeEncoder(_Sym):
    is_enc = True

    def __init__(self, stats=False):
        self.low = 0
        self.range = MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()
        self.stats = stats
        self.cat = "misc"
        self.bits = defaultdict(float)

    def _shift_low(self):
        low = self.low
        if low < 0xFF000000 or low >= 0x100000000:
            carry = low >> 32
            temp = self.cache
            out = self.out
            while True:
                out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (low & 0x00FFFFFF) << 8

    def bit(self, ctx, i, b):
        p = ctx[i]
        bound = (self.range >> PROB_BITS) * p
        if b:
            self.low += bound
            self.range -= bound
            ctx[i] = p - (p >> MOVE)
            if self.stats:
                self.bits[self.cat] += _COST1[p]
        else:
            self.range = bound
            ctx[i] = p + (((1 << PROB_BITS) - p) >> MOVE)
            if self.stats:
                self.bits[self.cat] += _COST0[p]
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self._shift_low()
        return b

    def direct(self, nbits, v):
        for k in range(nbits - 1, -1, -1):
            self.range >>= 1
            if (v >> k) & 1:
                self.low += self.range
            while self.range < TOP:
                self.range = (self.range << 8) & MASK32
                self._shift_low()
        if self.stats:
            self.bits[self.cat] += nbits
        return v

    def finish(self):
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder(_Sym):
    is_enc = False

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos
        self.range = MASK32
        self.code = 0
        self.cat = "misc"
        if len(data) - pos < 5:
            raise BitstreamError("range coder payload too short")
        if data[pos] != 0:
            raise BitstreamError("bad range coder start byte")
        for _ in range(5):
            self.code = ((self.code << 8) | data[self.pos]) & MASK32
            self.pos += 1

    def _next(self):
        p = self.pos
        self.pos = p + 1
        if p < len(self.data):
            return self.data[p]
        if p > len(self.data) + 16:
            raise BitstreamError("range decoder ran past end of payload")
        return 0

    def bit(self, ctx, i, _b=0):
        if self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self.code = ((self.code << 8) | self._next()) & MASK32
        p = ctx[i]
        bound = (self.range >> PROB_BITS) * p
        if self.code < bound:
            self.range = bound
            ctx[i] = p + (((1 << PROB_BITS) - p) >> MOVE)
            return 0
        self.code -= bound
        self.range -= bound
        ctx[i] = p - (p >> MOVE)
        return 1

    def direct(self, nbits, _v=0):
        v = 0
        for _ in range(nbits):
            if self.range < TOP:
                self.range = (self.range << 8) & MASK32
                self.code = ((self.code << 8) | self._next()) & MASK32
            self.range >>= 1
            if self.code >= self.range:
                self.code -= self.range
                v = (v << 1) | 1
            else:
                v <<= 1
        return v
