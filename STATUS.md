# Status snapshot (status quo at commit time)

## Verified
- `python vtv.py selftest`: 12/12 checks pass on the committed file (bit-exact encoder/decoder
  for 4 tool configurations, random access per segment, CRC corruption concealment,
  malformed input rejected).
- Web app exercised over real HTTP (upload -> encode -> statistics -> frame PNGs -> downloads ->
  play an existing .vtv) and in a simulated browser (jsdom): 23 UI checks pass, no JS errors.
  The committed `vtv.py` is byte-identical to the build those tests ran against.
- CLI encode / info / decode verified on a real mp4 (carphone, 24 frames: 66:1 vs raw, PSNR-Y 34.1 dB).

## Measured (indicative, single runs, not a benchmark suite)
BD-rate on PSNR-Y, 32 frames, x264 `veryslow --tune psnr`, keyint 32, x264 encoder-info SEI
excluded, VTV counts header+index+CRC. Negative = VTV needs fewer bits.
- synthetic pan (ideal global motion, noise-free): -27 % vs x264 P-only, +36 % vs x264 with B-frames
- real clip "carphone" (talking head):            +53 % vs x264 P-only, +111 % vs x264 with B-frames
- Parameter sweep (`dev/results/sweep.log`): quantiser dead-zone, lambda and MV-cost changes moved
  BD-rate by less than ~3 % (mostly worse) -> the gap on real footage is structural
  (no loop filter, simple intra, no B-frames, simpler entropy contexts), not mistuning.

## Known issues / open
1. **Embedded documentation in `vtv.py` is still a placeholder** (`dev/parts/part_doc.txt`);
   `python vtv.py doc` and the docs panel in the web page show placeholder text.
2. **Photometric (gain/offset) estimation is degenerate.** On the demo pan the estimated gain/offset
   wander (g 1.03-1.09, o -4..-11) although there is no brightness change: the two are collinear.
   The trajectory planner's error metric (|dg|*128 + |do|) then rejects a single-segment fit, so
   trajectories fall back to one segment per frame (demo: 11 segments for 11 frames; carphone: 23 for 23)
   and spurious gain/offset may enter the prediction. Geometry alone fits as one segment. Not fixed here.
   Idea: regularise towards identity / parametrise around mean luma, and score error at mid-gray.
3. **Ablation ladder (block-only -> +global -> +trajectories -> +photometric) is not finished.**
   `dev/scripts/ablation.py` was started but did not complete; results are outstanding. A log with a
   different table format was found in the working directory whose origin I could not verify; it is
   deliberately not included in this commit.
4. `dev/tests` and `dev/scripts` contain absolute sandbox paths (`/home/claude/...`); adjust before running.
5. `dev/package/vtv/*.py` carry the header line `# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026`
   (bulk-added to the sources at 21:20:31, not by the build steps); the generated `vtv.py` does not include it.

6. The tree also contains files I did not write and have **not reviewed or run**: `dev/bench/ablation.py`,
   `dev/bench/compare.py`, `dev/package/vtv/__main__.py`, `dev/package/vtv/videoio.py`, `dev/tests/conftest.py`
   (they arrived with the bulk update of the source folder; committed as-is because this is a status-quo snapshot).
   `dev/bench/ablation.py` is probably the origin of the unverified log mentioned in item 3.

## Not implemented
Loop/deblocking filter, B-frames, rate control, multi-reference, richer intra modes, browser-side
(JS/WASM/WebGPU) decoder, low-rank colour basis, progressive quality layers, per-region trajectories.
