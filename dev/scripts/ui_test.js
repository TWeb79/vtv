const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs");
const BASE = "http://127.0.0.1:8791/";
const sleep = ms => new Promise(r => setTimeout(r, ms));
const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", e => errors.push("jsdomError: " + e.message));
vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));
let fails = 0;

(async () => {
  const dom = await JSDOM.fromURL(BASE, {
    runScripts: "dangerously", resources: "usable", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) {
      w.fetch = (u, o) => fetch(new URL(u, BASE), o);
      w.Element.prototype.scrollIntoView = function () {};
    }
  });
  const w = dom.window, d = w.document, $ = s => d.querySelector(s);
  await sleep(500);
  const check = (name, ok, extra = "") => { if (!ok) fails++; console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${extra ? "  " + extra : ""}`); };
  check("page loads, docs rendered", d.querySelectorAll("#docs h3").length > 0, `(${d.querySelectorAll("#docs h3").length} doc sections)`);
  check("results + player box initially hidden", $("#results").hidden && $("#playerbox").hidden);

  // pick a file and encode
  const mp4 = fs.readFileSync("/home/claude/work/clips/carphone_pristine.mp4");
  const file = new w.File([mp4], "carphone.mp4", { type: "video/mp4" });
  Object.defineProperty($("#file"), "files", { value: [file] });
  $("#frames").value = "10"; $("#width").value = "176"; $("#qp").value = "34"; $("#qp").dispatchEvent(new w.Event("input"));
  check("QP label follows slider", $("#qpv").textContent === "34");
  $("#go").click();
  await sleep(300);
  check("Encode button disabled while running", $("#go").disabled);
  const t0 = Date.now(); const seen = new Set();
  while ($("#results").hidden && Date.now() - t0 < 120000) { seen.add($("#msg").textContent.replace(/\d+/g, "N")); await sleep(300); }
  check("statistics card appears", !$("#results").hidden, `(${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  console.log("     progress messages seen:", [...seen].filter(Boolean).join(" | "));
  const tiles = [...d.querySelectorAll(".tile")].map(t => t.querySelector(".k").textContent + " = " + t.querySelector(".v").textContent);
  console.log("     tiles:", tiles.join(" ; "));
  const need = ["Frames", "VTV file size", "Raw size", "Compression ratio", "Quality", "Encode time", "Decode time"];
  check("all key statistics shown", need.every(k => tiles.some(t => t.startsWith(k))));
  check("bit-breakdown bars rendered", d.querySelectorAll("#bits .b").length >= 4, `(${d.querySelectorAll("#bits .b").length} categories)`);
  check("download links present", d.querySelectorAll("#dl a").length === 2 && $("#dl a").href.includes("/api/vtv/"));

  // player
  const t1 = Date.now();
  while ($("#play").disabled && Date.now() - t1 < 30000) await sleep(200);
  check("player loaded all frames, controls enabled", !$("#play").disabled && !$("#seek").disabled, `("${$("#pmsg").textContent}")`);
  const n = +$("#seek").max + 1;
  check("slider range = number of frames", n === 10, `(n=${n})`);
  const cv = $("#cv"), ctx = cv.getContext("2d");
  const variance = () => { const a = ctx.getImageData(0, 0, cv.width, cv.height).data; let s = 0, s2 = 0, k = 0; for (let i = 0; i < a.length; i += 4) { s += a[i]; s2 += a[i] * a[i]; k++ } const m = s / k; return s2 / k - m * m };
  check("canvas shows a decoded picture", cv.width === 176 && cv.height === 144 && variance() > 100, `(${cv.width}x${cv.height}, var ${variance().toFixed(0)})`);

  $("#loop").checked = false;
  const seenIdx = [];
  $("#play").click();
  for (let k = 0; k < 6; k++) { await sleep(60); seenIdx.push(+$("#seek").value); }
  check("playback advances monotonically", seenIdx[seenIdx.length - 1] > 0 && seenIdx.every((v, i) => i === 0 || v >= seenIdx[i - 1]), `(frames seen: ${seenIdx.join(",")})`);
  await sleep(600);
  check("playback stops at the last frame when not looping", +$("#seek").value === 9 && $("#play").textContent.includes("Play"), `(idx ${$("#seek").value})`);
  $("#loop").checked = true;
  $("#play").click(); await sleep(150);
  check("pressing play at the end restarts from frame 1", +$("#seek").value < 9);
  $("#play").click();   // pause the restarted playback
  check("pause works (button shows Play)", $("#play").textContent.includes("Play"));
  const beforeKey = +$("#seek").value;
  d.dispatchEvent(new w.KeyboardEvent("keydown", { code: "ArrowRight", bubbles: true }));
  check("arrow key steps one frame", +$("#seek").value === Math.min(9, beforeKey + 1), `(${beforeKey} -> ${$("#seek").value})`);
  $("#seek").value = 5; $("#seek").dispatchEvent(new w.Event("input"));
  check("seek slider jumps to frame 6", $("#time").textContent.startsWith("frame 6/10"));

  for (const v of ["orig", "side", "diff", "vtv"]) {
    $("#view").value = v; $("#view").dispatchEvent(new w.Event("change")); await sleep(2500);
    const okSize = v === "side" ? cv.width === 352 : cv.width === 176;
    check(`view "${v}" renders`, okSize && variance() > 1, `(${cv.width}x${cv.height}, var ${variance().toFixed(0)})`);
  }

  // open an existing .vtv file
  const vtv = fs.readFileSync("/tmp/web.vtv");
  Object.defineProperty($("#vtvfile"), "files", { value: [new w.File([vtv], "web.vtv")], configurable: true });
  $("#vtvfile").dispatchEvent(new w.Event("change"));
  await sleep(500);
  const t2 = Date.now();
  while (!/\d+ frames ·/.test($("#pmsg").textContent) && Date.now() - t2 < 60000) await sleep(300);
  await sleep(500);
  check("existing .vtv loads into the player", /16 frames/.test($("#pmsg").textContent), `("${$("#pmsg").textContent}")`);
  check("source-only views disabled for a bare .vtv", [...$("#view").options].filter(o => o.disabled).length === 3);

  console.log(errors.length ? "\nJS ERRORS:\n" + errors.join("\n") : "\n  no JavaScript errors reported");
  w.close(); process.exit(errors.length || fails ? 1 : 0);
})().catch(e => { console.error("TEST CRASH", e); process.exit(2); });
