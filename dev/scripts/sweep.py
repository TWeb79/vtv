import sys, json, time, numpy as np
sys.path.insert(0,'/home/claude/vtv-beta')
import vtv.encoder as E
from vtv import Encoder, EncoderConfig, decode
from vtv.io import read_y4m
from vtv.metrics import sequence_quality, bd_rate

CLIPS = {'carphone':'/home/claude/work/real/carphone.y4m','bunny':'/home/claude/work/real/bunny.y4m','pan_objects':'/home/claude/work/synth/pan_objects.y4m'}
N=16; QPS=(26,31,36,41)
data = {k: read_y4m(v, max_frames=N) for k,v in CLIPS.items()}

def curve(clip, cfg_kw, patch):
    frames,W,H,_ = data[clip]
    old = {k: getattr(E,k) for k in patch}
    for k,v in patch.items(): setattr(E,k,v)
    pts=[]
    try:
        for qp in QPS:
            d,_ = Encoder(EncoderConfig(qp=qp, **cfg_kw), W, H).encode(frames)
            dec,_ = decode(d); q = sequence_quality(frames, dec)
            pts.append((len(d)*8/(len(frames)/25)/1000, q['psnr_y']))
    finally:
        for k,v in old.items(): setattr(E,k,v)
    return pts

CONFIGS = {
 'base': ({}, {}),
 'theta_1/4': ({}, {'THETA':(1,4)}),
 'theta_2/5': ({}, {'THETA':(2,5)}),
 'lam_0.09': ({'lam_scale':0.09}, {}),
 'lam_0.20': ({'lam_scale':0.20}, {}),
 'lamm_0.7': ({}, {'LAM_M':0.7}),
 'lamm_0.2': ({}, {'LAM_M':0.2}),
 'bnz_4.5': ({}, {'B_NZ':4.5, 'B_BLOCK':3.0}),
}
base = {c: curve(c, {}, {}) for c in CLIPS}
print("base done", flush=True)
for name,(kw,patch) in CONFIGS.items():
    if name=='base': continue
    t=time.time(); out={}
    for c in CLIPS: out[c] = bd_rate(base[c], curve(c, kw, patch))
    print(f"{name:10s} " + "  ".join(f"{c}:{v:+6.1f}%" for c,v in out.items()) + f"   mean {np.mean(list(out.values())):+6.1f}%   ({time.time()-t:.0f}s)", flush=True)
