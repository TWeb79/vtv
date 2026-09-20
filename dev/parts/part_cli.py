# =============================================================================
# Self-test and command line
# =============================================================================
def selftest(verbose=True):
    """Fast end-to-end verification of this file (about 10-20 s).  Returns True if all pass."""
    results = []

    def check(name, cond):
        results.append(bool(cond))
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(1)
    x = rng.integers(-255, 256, (200, 8, 8))
    check("integer 8x8 transform round trip (max error <= 3 on worst-case noise)",
          np.abs(idct8(fdct8(x)) - x).max() <= 3)

    enc, ctx = RangeEncoder(), new_ctx(4)
    vals = [int(rng.integers(0, 1000)) for _ in range(3000)]
    e_ue, e_se = new_ctx(12), new_ctx(18)
    for v in vals:
        enc.bit(ctx, v % 4, v & 1)
        enc.ue(e_ue, v)
        enc.se(e_se, v - 500)
    dec_rc, ctx, e_ue, e_se = RangeDecoder(enc.finish()), new_ctx(4), new_ctx(12), new_ctx(18)
    good = True
    for v in vals:
        good &= dec_rc.bit(ctx, v % 4) == (v & 1) and dec_rc.ue(e_ue, 0) == v and dec_rc.se(e_se, 0) == v - 500
    check("range coder round trip (bits, Exp-Golomb, signed)", good)

    img = rng.integers(0, 256, (48, 64))
    check("warp: identity transform is an exact copy",
          np.array_equal(predict_luma(img, IDENTITY), img))
    check("warp: integer translation equals a pixel shift",
          np.array_equal(predict_luma(img, (0, 0, 0, 0, 3 * 64, 0, 0, 0))[:, :-4], img[:, 3:-1]))

    W, H, N = 96, 64, 10
    frames = demo_clip(W, H, N)
    variants = (("default", {}), ("no global/photometric", dict(use_global=False, use_photo=False)),
                ("no trajectories", dict(use_traj=False)), ("bilinear filter", dict(interp=INTERP_BILINEAR)))
    data0 = None
    for label, kw in variants:
        enc_ = Encoder(EncoderConfig(qp=30, gop=6, **kw), W, H)
        data, st = enc_.encode(frames)
        dec, info = decode(data, conceal=False)
        same = len(dec) == N and all(np.array_equal(a, b) for e, d in zip(enc_.last_recon, dec)
                                     for a, b in zip(crop_planes(e, W, H), d))
        check(f"encoder reconstruction == decoder output, bit-exact [{label}]", same and not info["problems"])
        if data0 is None:
            data0, dec0 = data, dec
    q = sequence_quality(frames, dec0)
    check(f"decoded quality is sane (PSNR-Y {q['psnr_y']:.1f} dB > 30)", q["psnr_y"] > 30)

    hdr = read_header(data0)
    idx = read_index(data0, hdr)
    seg1 = decode_segment(data0, hdr, idx[1])
    check("random access: segment 1 decodes from its own bytes alone",
          all(np.array_equal(a, b) for f1, f2 in zip(seg1, dec0[idx[1]["frame_start"]:]) for a, b in zip(f1, f2)))
    bad = bytearray(data0)
    bad[idx[0]["offset"] + 30] ^= 0xFF
    dec_b, info_b = decode(bytes(bad), conceal=True)
    check("corrupt segment is detected (CRC) and concealed; later segments still decode",
          len(dec_b) == N and [p[0] for p in info_b["problems"]] == [0] and
          all(np.array_equal(a, b) for f1, f2 in zip(dec_b[idx[1]["frame_start"]:], dec0[idx[1]["frame_start"]:])
              for a, b in zip(f1, f2)))
    try:
        decode(b"VTVB" + bytes(60), conceal=False)
        rejected = False
    except Exception:                                                    # noqa: BLE001
        rejected = True
    check("malformed file is rejected without crashing", rejected)
    ok = all(results)
    if verbose:
        print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


def _cli_progress(prefix):
    last = [-1]

    def cb(d, t):
        pct = int(100 * d / max(t, 1))
        if pct != last[0]:
            last[0] = pct
            print(f"\r{prefix} {pct:3d} %", end="", flush=True)
    return cb


def cmd_encode(a):
    if a.input == "demo":
        frames, W, H, fps = demo_clip(192, 128, min(a.frames, 64)), 192, 128, 25.0
        src_bytes = None
    else:
        frames, W, H, fps, _ = read_video(a.input, width=a.width, max_frames=a.frames, start=a.start,
                                          progress=None)
        src_bytes = os.path.getsize(a.input)
    n = len(frames)
    print(f"input : {n} frames, {W}x{H}, {fps:.2f} fps")
    cfg = EncoderConfig(qp=a.qp, gop=a.gop, use_global=not a.no_global, use_traj=not a.no_traj,
                        use_photo=not a.no_photo, interp=INTERP_BILINEAR if a.bilinear else INTERP_CATMULL)
    t0 = time.time()
    data, st = Encoder(cfg, W, H, _fps_pair(fps)).encode(frames, progress=_cli_progress("encoding"))
    enc_s = time.time() - t0
    print()
    with open(a.output, "wb") as f:
        f.write(data)
    raw = W * H * 3 // 2 * n
    dur = n / fps
    print(f"output: {a.output}")
    print(f"  size          {len(data):,} bytes   (raw YUV 4:2:0: {raw:,} bytes)")
    print(f"  compression   {raw / len(data):.1f} : 1   ({len(data) * 8 / dur / 1000:.1f} kbit/s, "
          f"{len(data) * 8 / (W * H * n):.3f} bit/pixel)")
    print(f"  structure     {st['n_segments']} GOP segment(s), {st['n_traj_segments']} trajectory segment(s)")
    print(f"  encode time   {enc_s:.1f} s  ({n / enc_s:.1f} frames/s)")
    if not a.no_verify:
        t0 = time.time()
        dec, _ = decode(data, conceal=False)
        q = sequence_quality(frames, dec)
        print(f"  quality       PSNR-Y {q['psnr_y']:.2f} dB (min {q['psnr_y_min']:.2f}), "
              f"PSNR-YUV {q['psnr_yuv']:.2f} dB, SSIM-Y {q['ssim_y']:.4f}   [decode {time.time() - t0:.1f} s]")
    tot = sum(st["bits"].values()) or 1
    print("  bits          " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in
                                        sorted(st["bits"].items(), key=lambda kv: -kv[1])))
    if src_bytes:
        print(f"  (source file is {src_bytes:,} bytes in its own codec at full size - not comparable)")


def cmd_decode(a):
    with open(a.input, "rb") as f:
        data = f.read()
    t0 = time.time()
    frames, info = decode(data, conceal=True)
    hdr = info["header"]
    for i, msg in info["problems"]:
        print(f"warning: segment {i} concealed ({msg})")
    fps = hdr["fps"]
    out = a.output
    if out.lower().endswith(".y4m"):
        with open(out, "wb") as f:
            f.write(y4m_bytes(frames, fps))
    elif out.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
        ff = find_ffmpeg()
        if not ff:
            raise SystemExit("ffmpeg is needed to write " + out + "; use a .y4m file or a directory instead")
        y4m = y4m_bytes(frames, fps)
        for codec in (["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"], ["-c:v", "mpeg4", "-q:v", "2"]):
            r = subprocess.run([ff, "-y", "-v", "error", "-f", "yuv4mpegpipe", "-i", "-"] + codec + [out],
                               input=y4m, capture_output=True)
            if r.returncode == 0:
                break
        else:
            raise SystemExit("ffmpeg failed: " + r.stderr.decode("utf-8", "replace")[-300:])
    else:
        os.makedirs(out, exist_ok=True)
        for i, fr in enumerate(frames):
            with open(os.path.join(out, f"frame_{i:05d}.png"), "wb") as f:
                f.write(png_bytes(planes_to_rgb(fr)))
    print(f"decoded {len(frames)} frames ({hdr['width']}x{hdr['height']}) in {time.time() - t0:.1f} s -> {out}")


def cmd_info(a):
    with open(a.input, "rb") as f:
        data = f.read()
    hdr = read_header(data)
    idx = read_index(data, hdr)
    fps = hdr["fps"][0] / hdr["fps"][1]
    n = hdr["n_frames"]
    print(f"file      {a.input}  ({len(data):,} bytes)")
    print(f"format    VTV bitstream v{hdr['version'][0]}.{hdr['version'][1]}, 4:2:0 8-bit")
    print(f"video     {hdr['width']}x{hdr['height']}, {n} frames @ {fps:.3f} fps = {n / fps:.2f} s")
    print(f"tools     {_tools_label(hdr)}")
    print(f"bitrate   {len(data) * 8 / (n / fps) / 1000:.1f} kbit/s, "
          f"{len(data) * 8 / (hdr['width'] * hdr['height'] * n):.3f} bit/pixel, "
          f"{W_RAW(hdr) / len(data):.1f}:1 vs raw")
    print(f"segments  {hdr['n_segments']}  (each starts with an I frame; independently decodable)")
    for i, e in enumerate(idx):
        try:
            nf, qi, qp, payload = read_segment(data, e)
            st = f"QP I/P {qi}/{qp}, CRC ok"
        except BitstreamError as ex:
            st = f"BAD: {ex}"
        print(f"  #{i:<3d} frames {e['frame_start']}-{e['frame_start'] + e['n_frames'] - 1}  "
              f"offset {e['offset']:>9,}  {e['size']:>9,} bytes  {st}")


def W_RAW(hdr):
    return hdr["width"] * hdr["height"] * 3 // 2 * hdr["n_frames"]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="vtv.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="VTV - Vector-Time Video, research codec (beta %d.%d).\n"
                    "Run `python vtv.py doc` for the full documentation." % VERSION,
        epilog="examples:\n  python vtv.py serve\n  python vtv.py encode clip.mp4 clip.vtv --qp 30 --width 256 --frames 60\n"
               "  python vtv.py encode demo demo.vtv\n  python vtv.py decode clip.vtv clip_decoded.y4m\n"
               "  python vtv.py info clip.vtv\n  python vtv.py selftest")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web app (upload, statistics, player)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--max-upload-mb", type=int, default=500)
    e = sub.add_parser("encode", help="encode a video (or `demo`) to .vtv")
    e.add_argument("input", help="video file (mp4, mov, mkv, webm, y4m ...) or the word `demo`")
    e.add_argument("output")
    e.add_argument("--qp", type=int, default=32, help="quantiser 10-48, lower = better/larger (default 32)")
    e.add_argument("--width", type=int, default=256, help="max width in pixels (default 256)")
    e.add_argument("--frames", type=int, default=60, help="max frames to encode (default 60)")
    e.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    e.add_argument("--gop", type=int, default=32, help="max frames per independent segment (default 32)")
    e.add_argument("--no-global", action="store_true", help="disable global affine motion")
    e.add_argument("--no-traj", action="store_true", help="disable polynomial trajectories")
    e.add_argument("--no-photo", action="store_true", help="disable photometric gain/offset model")
    e.add_argument("--bilinear", action="store_true", help="bilinear instead of Catmull-Rom interpolation")
    e.add_argument("--no-verify", action="store_true", help="skip decode + quality measurement")
    d = sub.add_parser("decode", help="decode .vtv to .y4m, .mp4 (needs ffmpeg) or a folder of PNGs")
    d.add_argument("input")
    d.add_argument("output")
    i = sub.add_parser("info", help="show header, segments and checksums of a .vtv")
    i.add_argument("input")
    sub.add_parser("selftest", help="verify this file works (bit-exact round trips etc.)")
    sub.add_parser("doc", help="print the full documentation")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        LIMITS["max_upload"] = a.max_upload_mb * 1024 * 1024
        serve(a.host, a.port, not a.no_browser)
    elif a.cmd == "encode":
        cmd_encode(a)
    elif a.cmd == "decode":
        cmd_decode(a)
    elif a.cmd == "info":
        cmd_info(a)
    elif a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif a.cmd == "doc":
        print(DOC)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
