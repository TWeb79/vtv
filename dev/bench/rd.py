# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Rate-distortion measurement helpers (used by run_bench.py and the tuning scripts)."""
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from vtv import Encoder, EncoderConfig, decode          # noqa: E402
from vtv.io import read_y4m                             # noqa: E402
from vtv.metrics import bd_rate, sequence_quality       # noqa: E402

CACHE_PATH = os.path.join(HERE, "results", "cache.json")
_cache = None


def _load():
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    return _cache


def _save():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    json.dump(_cache, open(CACHE_PATH, "w"))


def code_fingerprint():
    """Hash of the codec sources: any change to the codec invalidates cached VTV results."""
    h = hashlib.sha1()
    d = os.path.join(HERE, "..", "vtv")
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()[:10]


def load_clip(path, n):
    frames, W, H, fps = read_y4m(path, max_frames=n)
    return frames, W, H


def vtv_point(clip_path, n, qp, cfg_kwargs, use_cache=True):
    cfg = EncoderConfig(qp=qp, **cfg_kwargs)
    key = json.dumps([os.path.basename(clip_path), n, asdict(cfg), code_fingerprint()], sort_keys=True)
    cache = _load()
    if use_cache and key in cache:
        return cache[key]
    frames, W, H = load_clip(clip_path, n)
    t0 = time.time()
    data, st = Encoder(cfg, W, H).encode(frames)
    t1 = time.time()
    dec, info = decode(data, conceal=False)
    t2 = time.time()
    assert len(dec) == len(frames) and not info["problems"]
    q = sequence_quality(frames, dec)
    dur = len(frames) / 25.0
    tot = sum(st["bits"].values()) or 1.0
    res = dict(qp=qp, bytes=len(data), kbps=len(data) * 8 / dur / 1000, enc_s=t1 - t0, dec_s=t2 - t1,
               n_traj=st["n_traj_segments"], n_seg=st["n_segments"],
               bit_share={k: v / tot for k, v in st["bits"].items()},
               bits={k: v for k, v in st["bits"].items()}, **q)
    cache[key] = res
    _save()
    return res


def vtv_curve(clip_path, n, cfg_kwargs, qps=(26, 31, 36, 41)):
    return [vtv_point(clip_path, n, qp, cfg_kwargs) for qp in qps]


def curve_bd(ref, test, metric="psnr_y"):
    return bd_rate([(p["kbps"], p[metric]) for p in ref], [(p["kbps"], p[metric]) for p in test])
