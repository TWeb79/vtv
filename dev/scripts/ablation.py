import sys, json, time, numpy as np
sys.path.insert(0, '/home/claude/final')          # the SINGLE-FILE codec, not the package
import vtv
print("using", vtv.__file__, flush=True)
CLIPS = {
  'pan':        '/home/claude/work/synth/pan.y4m',
  'pan_accel':  '/home/claude/work/synth/pan_accel.y4m',
  'zoom':       '/home/claude/work/synth/zoom.y4m',
  'rotate':     '/home/claude/work/synth/rotate.y4m',
  'fade':       '/home/claude/work/synth/fade.y4m',
  'pan_objects':'/home/claude/work/synth/pan_objects.y4m',
  'noisy_pan':  '/home/claude/work/synth/noisy_pan.y4m',
  'carphone':   '/home/claude/work/real/carphone.y4m',
  'bikes':      '/home/claude/work/real/bikes.y4m',
  'bunny':      '/home/claude/work/real/bunny.y4m',
}
LADDER = [
  ('A block-only',         dict(use_global=False, use_photo=False, use_traj=False)),
  ('B +global affine',     dict(use_global=True,  use_photo=False, use_traj=False)),
  ('C +trajectories',      dict(use_global=True,  use_photo=False, use_traj=True)),
  ('D +photometric',       dict(use_global=True,  use_photo=True,  use_traj=True)),
]
N = 24; QPS = (26, 31, 36, 41)
out = {}
for cname, path in CLIPS.items():
    frames, W, H, fps = vtv.read_y4m(path, max_frames=N)
    out[cname] = {}
    for lname, kw in LADDER:
        pts = []; traj_share = []; nseg = []
        for qp in QPS:
            data, st = vtv.Encoder(vtv.EncoderConfig(qp=qp, **kw), W, H).encode(frames)
            dec, _ = vtv.decode(data)
            q = vtv.sequence_quality(frames, dec)
            tot = sum(st['bits'].values())
            pts.append((len(data)*8/(len(frames)/25)/1000, q['psnr_y']))
            traj_share.append(st['bits'].get('traj',0)/tot); nseg.append(st['n_traj_segments'])
        out[cname][lname] = dict(pts=pts, traj_share=float(np.mean(traj_share)), n_traj=float(np.mean(nseg)))
    base = out[cname]['A block-only']['pts']
    row = "  ".join(f"{l[:1]}:{vtv.bd_rate(base, out[cname][l]['pts']):+6.1f}%" for l,_ in LADDER[1:])
    print(f"{cname:12s} BD-rate vs A (neg = fewer bits)  {row}   traj bits {out[cname]['D +photometric']['traj_share']*100:.1f}%  traj-segs {out[cname]['D +photometric']['n_traj']:.0f}/{N-1}", flush=True)
    json.dump(out, open('/home/claude/work/ablation.json','w'))
print("DONE", flush=True)
