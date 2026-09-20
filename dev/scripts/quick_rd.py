import sys, time, numpy as np
sys.path.insert(0,'/home/claude/vtv-beta'); sys.path.insert(0,'/home/claude/vtv-beta/bench')
from vtv import encode, decode, EncoderConfig
from vtv.io import read_y4m
from vtv.metrics import sequence_quality, bd_rate
from ext_codecs import encode_ext, PROFILES

name = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv)>2 else 32
import os; path = name if name.endswith('.y4m') else f'/home/claude/work/synth/{name}.y4m'
frames, W, H, fps = read_y4m(path, max_frames=N)
dur = len(frames)/25.0
def kbps(nbytes): return nbytes*8/dur/1000
res = {}
t0=time.time()
pts=[]
for qp in (24,28,32,36,40):
    t=time.time(); data, st = encode(frames, W, H, EncoderConfig(qp=qp)); te=time.time()-t
    dec,_ = decode(data); q = sequence_quality(frames, dec)
    pts.append((kbps(len(data)), q['psnr_y'])); 
    print(f"VTV qp{qp}: {kbps(len(data)):7.1f} kbps  PSNR-Y {q['psnr_y']:.2f}  SSIM {q['ssim_y']:.4f}  enc {te:.1f}s  bits%: "+ " ".join(f"{k}={v/sum(st['bits'].values())*100:.0f}" for k,v in sorted(st['bits'].items())))
res['VTV']=pts
for prof in ("x264-P","x264-full"):
    p=[]
    for q in PROFILES[prof][1]:
        size, dec = encode_ext(path, prof, q, n_expected=len(frames)); qq = sequence_quality(frames, dec)
        p.append((kbps(size), qq['psnr_y']))
    res[prof]=p
    print(prof, [(round(a,1), round(b,2)) for a,b in p])
for prof in ("x264-P","x264-full"):
    print(f"BD-rate VTV vs {prof}: {bd_rate(res[prof], res['VTV']):+.1f}%  (positive = VTV needs more bits)")
print(f"total {time.time()-t0:.0f}s")
