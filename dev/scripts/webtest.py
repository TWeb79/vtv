import json, time, struct, urllib.request, urllib.error, sys
B = "http://127.0.0.1:8791"
def get(p):
    with urllib.request.urlopen(B + p) as r: return r.status, r.headers, r.read()
def post(p, body):
    req = urllib.request.Request(B + p, data=body, method="POST", headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
def wait(jid, tmax=240):
    t0 = time.time(); last = ""
    while time.time() - t0 < tmax:
        s, h, b = get("/api/job/" + jid); j = json.loads(b)
        if j["message"] != last: print("   ", j["status"], round(j["progress"], 2), j["message"]); last = j["message"]
        if j["status"] in ("done", "error"): return j
        time.sleep(0.5)
    raise SystemExit("timeout")

s, h, b = get("/")
page = b.decode()
print("GET /  ->", s, h["Content-Type"], len(page), "bytes;  docs rendered:", "<h3>" in page, "| placeholder left:", "__DOCS__" in page or "__VERSION__" in page)

mp4 = open("/home/claude/work/clips/carphone_pristine.mp4", "rb").read()
s, j = post("/api/encode?name=carphone.mp4&qp=32&width=176&frames=16&gop=32&glob=1&traj=1&photo=1&bil=0", mp4)
print("POST /api/encode ->", s, j)
job = wait(j["id"])
assert job["status"] == "done", job
st = job["stats"]
print("stats:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items() if k != "bits"})
print("bits :", {k: round(v/8) for k, v in st["bits"].items()})

s, h, png = get(f"/api/frame/{j['id']}/3.png")
w, hh = struct.unpack(">II", png[16:24])
print("frame png:", s, h["Content-Type"], png[:8] == b"\x89PNG\r\n\x1a\n", f"{w}x{hh}")
s, h, png2 = get(f"/api/frame/{j['id']}/3.png?src=1")
print("source frame png:", s, len(png2), "bytes (differs from decoded:", png2 != png, ")")
try: get(f"/api/frame/{j['id']}/999.png")
except urllib.error.HTTPError as e: print("out-of-range frame ->", e.code)

s, h, vtv = get(f"/api/vtv/{j['id']}")
print("download .vtv:", s, h["Content-Disposition"], len(vtv), "bytes == stats:", len(vtv) == st["vtv_bytes"])
s, h, y4m = get(f"/api/y4m/{j['id']}")
print("download .y4m:", s, h["Content-Disposition"], len(y4m), "bytes")
open("/tmp/web.vtv", "wb").write(vtv)

# play an existing .vtv
s, j2 = post("/api/play?name=web.vtv", vtv)
job2 = wait(j2["id"])
print("play job:", job2["status"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in job2["stats"].items()})
s, h, png3 = get(f"/api/frame/{j2['id']}/3.png")
print("play frame identical to encode-job decoded frame:", png3 == png)

# error handling
s, j3 = post("/api/encode?name=bad.mp4", b"this is not a video" * 100)
job3 = wait(j3["id"]); print("garbage upload ->", job3["status"], "|", job3["error"][:110])
s, j4 = post("/api/play?name=bad.vtv", b"VTVB" + bytes(100))
job4 = wait(j4["id"]); print("garbage .vtv   ->", job4["status"], "|", job4["error"])
try: get("/api/job/deadbeef")
except urllib.error.HTTPError as e: print("unknown job ->", e.code)
s, j5 = post("/api/encode?name=x.mp4", b"")
print("empty upload ->", s, j5)
