# =============================================================================
# Web app:  python vtv.py serve      (standard library HTTP server, no extra packages)
# =============================================================================
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VTV codec</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--mut:#667085;--line:#e3e6ec;--acc:#2f6feb;--acc2:#e8f0fe;--bar:#2f6feb;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2b3240;--acc:#6aa0ff;--acc2:#1e2a44;--bar:#6aa0ff;--warn:#f5b26b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:28px 20px 8px;max-width:960px;margin:0 auto}
h1{margin:0;font-size:28px;letter-spacing:-.01em}h1 small{font-weight:400;color:var(--mut);font-size:15px;margin-left:8px}
.sub{color:var(--mut);margin:.3em 0 0}
main{max-width:960px;margin:0 auto;padding:12px 20px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}
h2{margin:0 0 12px;font-size:18px}h3{margin:20px 0 6px;font-size:16px}h4{margin:16px 0 4px;font-size:14px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
label{display:block;font-size:13px;color:var(--mut);margin-bottom:3px}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;margin:10px 0}
.row>div{min-width:130px}
input[type=number],select,input[type=file]{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);max-width:100%}
input[type=range]{width:100%}
button{font:inherit;padding:8px 16px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer}
button.sec{background:transparent;color:var(--acc)}button:disabled{opacity:.45;cursor:not-allowed}
a.btn{display:inline-block;text-decoration:none;padding:8px 14px;border-radius:8px;border:1px solid var(--acc);color:var(--acc);margin-right:8px;font-size:14px}
progress{width:100%;height:10px;accent-color:var(--bar)}
#msg,#pmsg{color:var(--mut);font-size:13px;min-height:1.4em;margin-top:4px}
.err{color:#d92d20!important}
details{margin:8px 0}summary{cursor:pointer;color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.tile{background:var(--acc2);border-radius:10px;padding:10px 12px}
.tile .k{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:21px;font-weight:600;margin-top:2px}.tile .s{font-size:12px;color:var(--mut);margin-top:2px}
.bars .b{display:grid;grid-template-columns:210px 1fr 90px;gap:10px;align-items:center;margin:4px 0;font-size:13px}
.bars .t{height:12px;background:var(--acc2);border-radius:6px;overflow:hidden}.bars .f{height:100%;background:var(--bar)}
.note{font-size:13px;color:var(--mut);margin-top:8px}
.chk label{display:inline-flex;align-items:center;gap:6px;margin-right:16px;color:var(--fg);font-size:14px}
#stage{background:#000;border-radius:10px;padding:8px;text-align:center;overflow:auto}
canvas{max-width:100%;height:auto;background:#000}canvas.pix{image-rendering:pixelated}
.ctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:10px}
.ctl input[type=range]{flex:1;min-width:180px}
code,pre{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow:auto}
code{background:var(--acc2);padding:1px 5px;border-radius:4px}pre code{background:none;padding:0}
.docs ul{padding-left:20px}.docs p{margin:.5em 0}
</style></head><body>
<header><h1>VTV<small>Vector-Time Video &middot; beta __VERSION__</small></h1>
<p class="sub">Research video codec: global-motion trajectories + regional deltas + transform residual. Pure Python / NumPy, deterministic integer decoder.</p></header>
<main>

<section class="card" id="enc"><h2>1 &middot; Encode a video</h2>
<div class="row"><div style="flex:1"><label for="file">Video file (mp4, mov, mkv, webm &hellip;)</label><input id="file" type="file" accept="video/*,.mp4,.mov,.mkv,.webm,.avi,.y4m"></div></div>
<div class="row">
 <div style="flex:1;min-width:220px"><label for="qp">Quality (QP <b id="qpv">32</b>) &mdash; lower = better &amp; bigger</label><input id="qp" type="range" min="16" max="46" value="32"></div>
 <div><label for="width">Max width</label><select id="width"><option>160</option><option selected>256</option><option>320</option><option>480</option></select></div>
 <div><label for="frames">Max frames</label><input id="frames" type="number" min="1" max="600" value="60" style="width:90px"></div>
 <div><button id="go">Encode</button></div>
</div>
<details><summary>Advanced: codec tools (for experiments / ablations)</summary>
 <div class="row chk">
  <label><input type="checkbox" id="o_glob" checked> global affine motion</label>
  <label><input type="checkbox" id="o_traj" checked> polynomial trajectories</label>
  <label><input type="checkbox" id="o_photo" checked> photometric (fade) model</label>
  <label><input type="checkbox" id="o_bil"> bilinear instead of Catmull-Rom</label>
  <label>GOP <input id="gop" type="number" min="2" max="120" value="32" style="width:70px"></label>
 </div></details>
<progress id="bar" value="0" max="1" hidden></progress><div id="msg"></div>
<div class="note">Encoding runs in pure Python: expect roughly 0.2&ndash;0.5 s per frame at 256 px width. The video is downscaled and truncated to the limits above; audio is dropped.</div>
</section>

<section class="card" id="results" hidden><h2>Statistics</h2>
<div class="grid" id="tiles"></div>
<div id="bitsbox" hidden><h3>Where the bits go</h3><div class="bars" id="bits"></div><div class="note" id="bitnote"></div></div>
<div class="row" id="dl" style="margin-top:14px"></div></section>

<section class="card" id="player"><h2>2 &middot; Player</h2>
<div class="row"><div><label>Play an existing .vtv file</label><input id="vtvfile" type="file" accept=".vtv"></div>
 <div class="note" style="margin:0 0 6px">Or encode a video above &mdash; it loads here automatically. Frames are decoded by the Python VTV decoder on the server.</div></div>
<div id="playerbox" hidden>
 <div id="stage"><canvas id="cv" width="16" height="16"></canvas></div>
 <div class="ctl"><button id="play" disabled>&#9654; Play</button>
  <input id="seek" type="range" min="0" max="0" value="0" disabled><span id="time" style="min-width:150px;font-variant-numeric:tabular-nums"></span></div>
 <div class="ctl">
  <label style="margin:0">View <select id="view"><option value="vtv">VTV (decoded)</option><option value="orig">Original</option><option value="side">Side by side</option><option value="diff">Difference &times;4</option></select></label>
  <label style="margin:0">Speed <select id="speed"><option value=".25">0.25&times;</option><option value=".5">0.5&times;</option><option value="1" selected>1&times;</option><option value="2">2&times;</option></select></label>
  <label style="margin:0"><input type="checkbox" id="loop" checked> loop</label>
  <label style="margin:0"><input type="checkbox" id="pix"> pixelated</label>
  <span class="note" style="margin:0">Space = play/pause &middot; &larr; &rarr; = step</span></div>
</div><div id="pmsg"></div></section>

<section class="card docs" id="docs"><h2>3 &middot; About the codec &amp; how to use it</h2>__DOCS__</section>
</main>
<script>
"use strict";
const $=s=>document.querySelector(s);
const fmtB=b=>b<1024?b+" B":b<1048576?(b/1024).toFixed(1)+" KB":(b/1048576).toFixed(2)+" MB";
const fmtT=s=>s<60?s.toFixed(1)+" s":Math.floor(s/60)+"m "+Math.round(s%60)+"s";
$("#qp").oninput=()=>$("#qpv").textContent=$("#qp").value;

function upload(url,file,onprog){return new Promise((res,rej)=>{const x=new XMLHttpRequest();x.open("POST",url);
 x.setRequestHeader("Content-Type","application/octet-stream");
 x.upload.onprogress=e=>{if(e.lengthComputable&&onprog)onprog(e.loaded/e.total)};
 x.onload=()=>{let j;try{j=JSON.parse(x.responseText)}catch(e){return rej(new Error("HTTP "+x.status))}
  x.status<300?res(j):rej(new Error(j.error||x.statusText))};
 x.onerror=()=>rej(new Error("network error"));x.send(file)})}
async function poll(id,cb){for(;;){const j=await (await fetch("/api/job/"+id)).json();cb(j);
 if(j.status==="done"||j.status==="error")return j;await new Promise(r=>setTimeout(r,400))}}
function setMsg(t,err){const m=$("#msg");m.textContent=t;m.className=err?"err":""}

$("#go").onclick=async()=>{
 const f=$("#file").files[0];if(!f){setMsg("Choose a video file first.",true);return}
 const q=new URLSearchParams({name:f.name,qp:$("#qp").value,width:$("#width").value,frames:$("#frames").value,gop:$("#gop").value,
  glob:$("#o_glob").checked?1:0,traj:$("#o_traj").checked?1:0,photo:$("#o_photo").checked?1:0,bil:$("#o_bil").checked?1:0});
 const bar=$("#bar");bar.hidden=false;bar.value=0;$("#go").disabled=true;$("#results").hidden=true;
 try{
  const {id}=await upload("/api/encode?"+q,f,p=>{bar.value=p;setMsg("Uploading "+Math.round(p*100)+" %\u2026")});
  const j=await poll(id,s=>{bar.value=s.progress;setMsg(s.message||s.status,s.status==="error")});
  if(j.status==="error")throw new Error(j.error);
  bar.value=1;setMsg("Done.");showStats(j.stats,id);await loadPlayer(id,j.stats);
  $("#player").scrollIntoView({behavior:"smooth",block:"start"});
 }catch(e){setMsg("Error: "+e.message,true)}
 $("#go").disabled=false;
};

const LABELS={traj:"Global-motion trajectory",struct:"Partition & modes",mvd:"Regional motion deltas",intra:"Intra / new information",res_y:"Residual \u00b7 luma",res_c:"Residual \u00b7 chroma"};
function tile(k,v,s){return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>${s?`<div class="s">${s}</div>`:""}</div>`}
function showStats(s,id){
 const T=[];
 T.push(tile("Frames",s.frames,`${s.width}\u00d7${s.height} \u00b7 ${s.fps.toFixed(2)} fps \u00b7 ${fmtT(s.duration_s)}`));
 T.push(tile("VTV file size",fmtB(s.vtv_bytes),`${s.kbps.toFixed(1)} kbit/s \u00b7 ${s.bpp.toFixed(3)} bit/pixel`));
 T.push(tile("Raw size (YUV 4:2:0)",fmtB(s.raw_bytes),"uncompressed, same frames"));
 T.push(tile("Compression ratio",s.ratio.toFixed(1)+" : 1",`file is ${(100/s.ratio).toFixed(1)} % of raw`));
 if(s.psnr_y!=null)T.push(tile("Quality",s.psnr_y.toFixed(2)+" dB",`PSNR-Y avg \u00b7 min ${s.psnr_y_min.toFixed(2)} \u00b7 SSIM-Y ${s.ssim_y.toFixed(4)}`));
 if(s.uploaded_bytes)T.push(tile("Uploaded file",fmtB(s.uploaded_bytes),"source (other codec, full size) \u2013 not comparable"));
 if(s.enc_s!=null)T.push(tile("Encode time",fmtT(s.enc_s),`${(s.frames/s.enc_s).toFixed(1)} frames/s`));
 T.push(tile("Decode time",fmtT(s.dec_s),`${(s.frames/Math.max(s.dec_s,1e-6)).toFixed(1)} frames/s`));
 T.push(tile("Structure",`${s.segments} GOP`,s.traj_segments!=null?`${s.traj_segments} trajectory segments`:(s.tools||"")));
 $("#tiles").innerHTML=T.join("");$("#results").hidden=false;
 const bb=$("#bitsbox");
 if(s.bits){const tot=Object.values(s.bits).reduce((a,b)=>a+b,0)||1;
  $("#bits").innerHTML=Object.entries(s.bits).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
   `<div class="b"><span>${LABELS[k]||k}</span><div class="t"><div class="f" style="width:${(100*v/tot).toFixed(1)}%"></div></div><span>${(100*v/tot).toFixed(1)} % \u00b7 ${fmtB(Math.round(v/8))}</span></div>`).join("");
  const rs=((s.bits.res_y||0)+(s.bits.res_c||0))/tot;
  $("#bitnote").textContent=`Residual share: ${(100*rs).toFixed(0)} % of the stream. If this stays high, the motion/trajectory model explains little of the picture and the transform residual carries the information.`;
  bb.hidden=false}else bb.hidden=true;
 $("#dl").innerHTML=`<a class="btn" href="/api/vtv/${id}">Download .vtv</a><a class="btn" href="/api/y4m/${id}">Download decoded .y4m</a>`;
}

/* ---------------- player ---------------- */
let P=null;const cv=$("#cv"),ctx=cv.getContext("2d"),ta=document.createElement("canvas"),tb=document.createElement("canvas");
function loadImgs(id,n,src,onp){let k=0;return Promise.all(Array.from({length:n},(_,i)=>new Promise((res,rej)=>{
 const im=new Image();im.onload=()=>{onp&&onp(++k);res(im)};im.onerror=()=>rej(new Error("frame "+i+" failed"));
 im.src=`/api/frame/${id}/${i}.png${src?"?src=1":""}`})))}
function stop(){if(P){P.playing=false}$("#play").innerHTML="&#9654; Play"}
async function loadPlayer(id,m){
 stop();P={id,n:m.frames,fps:m.fps||25,W:m.width,H:m.height,hasSrc:!!m.has_source,dec:null,src:null,idx:0,playing:false};
 $("#playerbox").hidden=false;$("#pmsg").className="";
 try{P.dec=await loadImgs(id,P.n,false,k=>$("#pmsg").textContent=`Loading decoded frames ${k}/${P.n}\u2026`)}
 catch(e){$("#pmsg").textContent="Error: "+e.message;$("#pmsg").className="err";return}
 for(const o of $("#view").options)o.disabled=(o.value!=="vtv")&&!P.hasSrc;$("#view").value="vtv";
 $("#seek").max=P.n-1;$("#seek").disabled=false;$("#play").disabled=false;
 sizeCanvas();show(0);$("#pmsg").textContent=`${P.n} frames \u00b7 ${P.W}\u00d7${P.H} \u00b7 ${P.fps.toFixed(2)} fps`}
function sizeCanvas(){const v=$("#view").value;cv.width=v==="side"?2*P.W:P.W;cv.height=P.H;cv.className=$("#pix").checked?"pix":""}
function pix(im){ta.width=tb.width=P.W;ta.height=tb.height=P.H;return im}
function show(i){P.idx=i;$("#seek").value=i;$("#time").textContent=`frame ${i+1}/${P.n} \u00b7 ${(i/P.fps).toFixed(2)} s`;
 const v=$("#view").value;
 if(v==="vtv")ctx.drawImage(P.dec[i],0,0);
 else if(v==="orig")ctx.drawImage(P.src[i],0,0);
 else if(v==="side"){ctx.drawImage(P.src[i],0,0);ctx.drawImage(P.dec[i],P.W,0);ctx.fillStyle="rgba(0,0,0,.6)";ctx.fillRect(0,0,70,16);ctx.fillRect(P.W,0,50,16);
  ctx.fillStyle="#fff";ctx.font="11px sans-serif";ctx.fillText("original",4,12);ctx.fillText("VTV",P.W+4,12)}
 else if(v==="diff"){pix();const a=ta.getContext("2d"),b=tb.getContext("2d");a.drawImage(P.src[i],0,0);b.drawImage(P.dec[i],0,0);
  const A=a.getImageData(0,0,P.W,P.H),B=b.getImageData(0,0,P.W,P.H);
  for(let k=0;k<A.data.length;k+=4){for(let c=0;c<3;c++)A.data[k+c]=Math.min(255,Math.abs(A.data[k+c]-B.data[k+c])*4);A.data[k+3]=255}
  ctx.putImageData(A,0,0)}}
function tick(now){if(!P||!P.playing)return;
 const sp=parseFloat($("#speed").value);let i=Math.max(0,Math.floor(P.base+(now-P.t0)/1000*P.fps*sp));
 if(i>=P.n){if($("#loop").checked){P.t0=now;P.base=0;i=0}else{show(P.n-1);stop();return}}
 if(i!==P.idx)show(i);requestAnimationFrame(tick)}
function play(){if(!P)return;if(P.idx>=P.n-1)P.idx=0;P.playing=true;P.base=P.idx;P.t0=performance.now();$("#play").innerHTML="&#10074;&#10074; Pause";requestAnimationFrame(tick)}
async function needSrc(){if(P.hasSrc&&!P.src){$("#pmsg").textContent="Loading original frames\u2026";
 P.src=await loadImgs(P.id,P.n,true,k=>$("#pmsg").textContent=`Loading original frames ${k}/${P.n}\u2026`);
 $("#pmsg").textContent=`${P.n} frames \u00b7 ${P.W}\u00d7${P.H} \u00b7 ${P.fps.toFixed(2)} fps`}}
$("#play").onclick=()=>P&&(P.playing?stop():play());
$("#seek").oninput=e=>{if(!P)return;const i=+e.target.value;if(P.playing){P.base=i;P.t0=performance.now()}show(i)};
$("#view").onchange=async()=>{if(!P)return;if($("#view").value!=="vtv")await needSrc();sizeCanvas();show(P.idx)};
$("#pix").onchange=()=>{if(P)sizeCanvas(),show(P.idx)};
document.addEventListener("keydown",e=>{if(!P||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)&&e.target.type!=="range"&&e.target.type!=="checkbox")return;
 if(e.code==="Space"){e.preventDefault();$("#play").click()}
 else if(e.code==="ArrowRight"){stop();show(Math.min(P.n-1,P.idx+1))}else if(e.code==="ArrowLeft"){stop();show(Math.max(0,P.idx-1))}});
$("#vtvfile").onchange=async e=>{const f=e.target.files[0];if(!f)return;$("#pmsg").className="";$("#pmsg").textContent="Uploading\u2026";
 try{const {id}=await upload("/api/play?name="+encodeURIComponent(f.name),f);
  const j=await poll(id,s=>$("#pmsg").textContent=s.message||s.status);if(j.status==="error")throw new Error(j.error);
  showStats(j.stats,id);await loadPlayer(id,j.stats)}
 catch(err){$("#pmsg").textContent="Error: "+err.message;$("#pmsg").className="err"}};
</script></body></html>
"""


def md_to_html(md):
    """Tiny markdown subset (#/## headings, - lists, ``` code, `code`, **bold**) for the docs panel."""
    import html as _h

    def inline(s):
        s = _h.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    out, para, in_code, in_list = [], [], False, False

    def flush():
        nonlocal para, in_list
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []
        if in_list:
            out.append("</ul>")
            in_list = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            flush()
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(_h.escape(line))
            continue
        if line.startswith("## "):
            flush()
            out.append("<h4>" + inline(line[3:]) + "</h4>")
        elif line.startswith("# "):
            flush()
            out.append("<h3>" + inline(line[2:]) + "</h3>")
        elif line.startswith("- "):
            if para:
                out.append("<p>" + inline(" ".join(para)) + "</p>")
                para = []
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + inline(line[2:]) + "</li>")
        elif line.startswith("  ") and in_list and line.strip():
            out[-1] = out[-1][:-5] + " " + inline(line.strip()) + "</li>"
        elif not line.strip():
            flush()
        else:
            if in_list:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def y4m_bytes(frames, fps=(25, 1)):
    H, W = frames[0][0].shape
    out = [f"YUV4MPEG2 W{W} H{H} F{fps[0]}:{fps[1]} Ip A1:1 C420jpeg\n".encode()]
    for Y, Cb, Cr in frames:
        out += [b"FRAME\n", np.ascontiguousarray(Y).tobytes(), np.ascontiguousarray(Cb).tobytes(),
                np.ascontiguousarray(Cr).tobytes()]
    return b"".join(out)


class Job:
    def __init__(self, kind, name):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.name = kind, name
        self.status, self.progress, self.message, self.error = "queued", 0.0, "Queued\u2026", None
        self.stats = self.vtv = self.dec = self.src = self.tmp = None
        self.fps = (25, 1)
        self.png = {}

    def set(self, status, progress, message):
        self.status, self.progress, self.message = status, float(progress), message

    def public(self):
        return dict(id=self.id, status=self.status, progress=self.progress, message=self.message,
                    error=self.error, stats=self.stats)


JOBS, JOB_ORDER = {}, []
JOBS_LOCK = threading.Lock()
WORK_SEM = threading.Semaphore(1)       # one heavy job at a time: pure-Python codec is CPU bound
MAX_JOBS = 8
LIMITS = dict(max_upload=500 * 1024 * 1024)


def new_job(kind, name):
    j = Job(kind, name)
    with JOBS_LOCK:
        JOBS[j.id] = j
        JOB_ORDER.append(j.id)
        while len(JOB_ORDER) > MAX_JOBS:
            old = JOBS.pop(JOB_ORDER.pop(0), None)
            if old is not None and old.tmp:
                try:
                    os.remove(old.tmp)
                except OSError:
                    pass
    return j


def _fps_pair(fps):
    from fractions import Fraction as _F
    f = _F(float(fps)).limit_denominator(1001)
    return (max(1, f.numerator), max(1, f.denominator))


def _tools_label(hdr):
    t = [n for n, on in (("global affine", hdr["geo"]), ("photometric", hdr["photo"])) if on]
    t.append("bilinear" if hdr["interp"] == 0 else "Catmull-Rom")
    return ", ".join(t)


def run_encode_job(job, path, opts):
    with WORK_SEM:
        try:
            job.set("reading", 0.02, "Reading & scaling video\u2026")
            frames, W, H, fps, info = read_video(
                path, width=opts["width"], max_frames=opts["frames"],
                progress=lambda d, t: job.set("reading", 0.02 + 0.06 * d / t, f"Reading video\u2026 {d} frames"))
            n = len(frames)
            cfg = EncoderConfig(qp=opts["qp"], gop=opts["gop"], use_global=opts["glob"], use_traj=opts["traj"],
                                use_photo=opts["photo"], interp=0 if opts["bil"] else 1)
            job.fps = _fps_pair(fps)
            t0 = time.time()
            def prog(d, t):
                msg = (f"Analysing motion\u2026 {d}/{n - 1}" if d < n
                       else f"Encoding frame {d - n + 1}/{n}")
                job.set("encoding", 0.08 + 0.72 * d / t, msg)
            data, st = Encoder(cfg, W, H, job.fps).encode(frames, progress=prog)
            enc_s = time.time() - t0
            job.set("verifying", 0.82, "Decoding & measuring quality\u2026")
            t0 = time.time()
            dec, dinfo = decode(data, conceal=False)
            dec_s = time.time() - t0
            q = sequence_quality(frames, dec)
            hdr = read_header(data)
            fps_f = job.fps[0] / job.fps[1]
            dur = n / fps_f
            raw = W * H * 3 // 2 * n
            job.vtv, job.dec, job.src = data, dec, frames
            job.stats = dict(kind="encode", file=job.name, uploaded_bytes=os.path.getsize(path),
                             frames=n, width=W, height=H, fps=fps_f, duration_s=dur, raw_bytes=raw,
                             vtv_bytes=len(data), ratio=raw / len(data), kbps=len(data) * 8 / dur / 1000,
                             bpp=len(data) * 8 / (W * H * n), enc_s=enc_s, dec_s=dec_s,
                             segments=st["n_segments"], traj_segments=st["n_traj_segments"],
                             bits=st["bits"], has_source=True, tools=_tools_label(hdr), qp=cfg.qp, **q)
            job.set("done", 1.0, "Done")
        except Exception as e:                                       # noqa: BLE001
            job.error = f"{type(e).__name__}: {e}"
            job.set("error", job.progress, "Failed")
        finally:
            try:
                os.remove(path)
                job.tmp = None
            except OSError:
                pass


def run_play_job(job, data):
    with WORK_SEM:
        try:
            job.set("decoding", 0.1, "Decoding\u2026")
            hdr = read_header(data)
            t0 = time.time()
            frames, info = decode(data, conceal=True)
            dec_s = time.time() - t0
            n, W, H = len(frames), hdr["width"], hdr["height"]
            fps_f = hdr["fps"][0] / hdr["fps"][1]
            dur = n / fps_f
            raw = W * H * 3 // 2 * n
            job.vtv, job.dec = data, frames
            job.fps = hdr["fps"]
            job.stats = dict(kind="play", file=job.name, frames=n, width=W, height=H, fps=fps_f,
                             duration_s=dur, raw_bytes=raw, vtv_bytes=len(data), ratio=raw / len(data),
                             kbps=len(data) * 8 / dur / 1000, bpp=len(data) * 8 / (W * H * n),
                             dec_s=dec_s, segments=hdr["n_segments"], tools=_tools_label(hdr),
                             has_source=False, problems=len(info["problems"]))
            job.set("done", 1.0, "Done")
        except Exception as e:                                       # noqa: BLE001
            job.error = f"{type(e).__name__}: {e}"
            job.set("error", 0, "Failed")


class Handler(BaseHTTPRequestHandler):
    server_version = "VTV/" + ".".join(map(str, VERSION))

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _job(self, jid):
        with JOBS_LOCK:
            return JOBS.get(jid)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            page = PAGE.replace("__VERSION__", ".".join(map(str, VERSION))).replace(
                "__DOCS__", md_to_html(DOC))
            return self._send(200, page, "text/html; charset=utf-8")
        m = re.match(r"^/api/job/([0-9a-f]+)$", p)
        if m:
            j = self._job(m.group(1))
            return self._send(200, j.public()) if j else self._send(404, dict(error="unknown job"))
        m = re.match(r"^/api/frame/([0-9a-f]+)/(\d+)\.png$", p)
        if m:
            j = self._job(m.group(1))
            src = "src=1" in (u.query or "")
            seq = (j.src if src else j.dec) if j else None
            i = int(m.group(2))
            if not seq or i >= len(seq):
                return self._send(404, dict(error="no such frame"))
            key = (src, i)
            if key not in j.png:
                j.png[key] = png_bytes(planes_to_rgb(seq[i]))
            return self._send(200, j.png[key], "image/png", {"Cache-Control": "max-age=3600"})
        m = re.match(r"^/api/(vtv|y4m)/([0-9a-f]+)$", p)
        if m:
            j = self._job(m.group(2))
            if not j or not j.vtv:
                return self._send(404, dict(error="not available"))
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(j.name or "video")[0])[:60] or "video"
            if m.group(1) == "vtv":
                body, name = j.vtv, stem + ".vtv"
            else:
                body, name = y4m_bytes(j.dec, j.fps), stem + "_decoded.y4m"
            return self._send(200, body, "application/octet-stream",
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        self._send(404, dict(error="not found"))

    def _read_body(self, dest):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            self._send(400, dict(error="empty upload"))
            return False
        if n > LIMITS["max_upload"]:
            self._send(413, dict(error="file too large (limit %d MB)" % (LIMITS["max_upload"] >> 20)))
            return False
        left = n
        while left:
            chunk = self.rfile.read(min(1 << 20, left))
            if not chunk:
                self._send(400, dict(error="upload interrupted"))
                return False
            dest.write(chunk)
            left -= len(chunk)
        return True

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        def geti(k, d, lo, hi):
            try:
                return max(lo, min(hi, int(q.get(k, [d])[0])))
            except ValueError:
                return d
        name = (q.get("name", ["video"])[0] or "video")[:120]
        if u.path == "/api/encode":
            ext = re.sub(r"[^A-Za-z0-9]", "", os.path.splitext(name)[1])[:6]
            fd, path = tempfile.mkstemp(suffix="." + (ext or "bin"), prefix="vtv_up_")
            with os.fdopen(fd, "wb") as f:
                ok = self._read_body(f)
            if not ok:
                os.remove(path)
                return
            opts = dict(qp=geti("qp", 32, 10, 48), width=geti("width", 256, 64, 640),
                        frames=geti("frames", 60, 1, 600), gop=geti("gop", 32, 2, 120),
                        glob=bool(geti("glob", 1, 0, 1)), traj=bool(geti("traj", 1, 0, 1)),
                        photo=bool(geti("photo", 1, 0, 1)), bil=bool(geti("bil", 0, 0, 1)))
            job = new_job("encode", name)
            job.tmp = path
            threading.Thread(target=run_encode_job, args=(job, path, opts), daemon=True).start()
            return self._send(200, dict(id=job.id))
        if u.path == "/api/play":
            buf = _stdio.BytesIO()
            if not self._read_body(buf):
                return
            job = new_job("play", name)
            threading.Thread(target=run_play_job, args=(job, buf.getvalue()), daemon=True).start()
            return self._send(200, dict(id=job.id))
        self._send(404, dict(error="not found"))


def serve(host="127.0.0.1", port=8765, open_browser=True):
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("no free port found")
    httpd.daemon_threads = True
    url = f"http://{'localhost' if host in ('0.0.0.0', '') else host}:{p}/"
    print(f"VTV web app running at {url}   (Ctrl+C to stop)")
    print("ffmpeg:", find_ffmpeg() or "not found -> falling back to OpenCV if installed")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
