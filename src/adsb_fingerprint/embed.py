"""adsb-embed: project a checkpoint's learned embeddings to 2-D with t-SNE.

(The plan says UMAP, but umap-learn needs numba, which has no Python 3.14
wheels yet — sklearn's Barnes-Hut t-SNE gives the same cluster-structure
diagnostic; swap back when numba catches up.)

Runs a trained run's model over its own session universe (or, with
--everything, the whole corpus), takes the pooled penultimate-layer
embedding per message, and writes two artifacts into the run directory:

  embedding.npz   — 48-dim embeddings, 2-D projected coords, and per-
                    message metadata (icao, session, split, prediction,
                    rssi)
  embedding.html  — a self-contained canvas-2D viewer (no server, no
                    WebGL, no network): color by aircraft / session /
                    rssi / correctness / prediction, hover for details,
                    click legend entries to isolate, drag/wheel to
                    pan/zoom.

The IQ is shown to the model in the checkpoint's own ablation variant —
the embedding should reflect what that model actually saw.

It also prints k-NN purity in the raw 48-dim space for held-out points:
the fraction of nearest neighbors sharing the aircraft vs sharing the
session. "Same aircraft, different session" is the fingerprint signal;
"same session" is the channel signature.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from adsb_fingerprint import dataset
from adsb_fingerprint.evaluate import load_run
from adsb_fingerprint.model import (
    iq_channels,
    pick_device,
)

SPLITS = ("train", "held-out", "other")


def embed_corpus(net, x, device, batch_size):
    """(embeddings, predicted class, confidence) for (n, 2, t) input."""
    net = net.to(device)
    outs, preds, confs = [], [], []
    with torch.no_grad():
        for lo in tqdm(
            range(0, len(x), batch_size),
            desc="embed",
            leave=False,
        ):
            batch = torch.from_numpy(x[lo : lo + batch_size]).to(device)
            emb = net.features(batch)
            prob = torch.softmax(net.classify(emb), dim=1)
            conf, pred = prob.max(dim=1)
            outs.append(emb.cpu().numpy())
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())
    return np.concatenate(outs), np.concatenate(preds), np.concatenate(confs)


def knn_purity(embeddings, icaos, sessions, queries, k):
    """Neighbor-agreement fractions for the query rows, cosine, chunked."""
    normed = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
    )
    same_aircraft = []
    same_session = []
    aircraft_other_session = []
    for lo in range(0, len(queries), 1024):
        rows = queries[lo : lo + 1024]
        sims = normed[rows] @ normed.T
        sims[np.arange(len(rows)), rows] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        icao_match = icaos[top] == icaos[rows, None]
        session_match = sessions[top] == sessions[rows, None]
        same_aircraft.append(icao_match.mean(axis=1))
        same_session.append(session_match.mean(axis=1))
        aircraft_other_session.append((icao_match & ~session_match).mean(axis=1))
    return (
        float(np.concatenate(same_aircraft).mean()),
        float(np.concatenate(same_session).mean()),
        float(np.concatenate(aircraft_other_session).mean()),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Write a UMAP embedding viewer (npz + self-contained html) into a run directory.",
    )
    parser.add_argument("run", help="Run directory (from adsb-train).")
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Which checkpoint to embed with.",
    )
    parser.add_argument(
        "--everything",
        action="store_true",
        help="Embed every indexed message (all sessions, all aircraft), "
        "not just the checkpoint's train/test universe.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Embedding batch size.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--metric", default="cosine", help="t-SNE metric.")
    parser.add_argument("--knn", type=int, default=10, help="k for the purity report.")
    parser.add_argument("--seed", type=int, default=0, help="t-SNE random_state.")
    args = parser.parse_args()

    run_dir = Path(args.run).expanduser()
    ckpt, net = load_run(run_dir, args.checkpoint)
    classes = ckpt["classes"]

    data = dataset.load_examples()
    split = np.full(len(data["icao"]), 2, dtype=int)
    in_class = np.isin(data["icao"], classes)
    split[in_class & np.isin(data["session"], ckpt["train_sessions"])] = 0
    split[in_class & np.isin(data["session"], ckpt["test_sessions"])] = 1
    keep = np.ones(len(split), bool) if args.everything else split < 2
    index = np.where(keep)[0]
    if not len(index):
        raise SystemExit("nothing to embed")

    iq = dataset.apply_variant(data["iq"][index], ckpt["variant"])
    device = pick_device()
    embeddings, pred, conf = embed_corpus(
        net,
        iq_channels(iq),
        device,
        args.batch_size,
    )

    icaos = data["icao"][index]
    sessions = data["session"][index]
    split = split[index]
    held_out = np.where(split == 1)[0]
    if len(held_out):
        aircraft, session, cross = knn_purity(
            embeddings,
            icaos,
            sessions,
            held_out,
            args.knn,
        )
        print(f"k-NN purity (k={args.knn}, cosine, {embeddings.shape[1]}-dim, {len(held_out)} held-out queries):")
        print(f"  same aircraft                : {aircraft:.3f}")
        print(f"  same aircraft, other session : {cross:.3f}   <- fingerprint signal")
        print(f"  same session                 : {session:.3f}   <- channel signature")

    print("fitting t-SNE...")
    xy = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        metric=args.metric,
        init="pca",
        random_state=args.seed,
    ).fit_transform(embeddings.astype(np.float64))

    times = np.array([t.strftime("%m-%d %H:%MZ") for t in data["captured_at"][index]])
    np.savez_compressed(
        run_dir / "embedding.npz",
        xy=xy.astype(np.float32),
        embedding=embeddings.astype(np.float32),
        icao=icaos,
        session=sessions,
        split=split,
        pred=np.array(classes)[pred],
        confidence=conf.astype(np.float32),
        rssi_db=data["rssi_db"][index],
        captured_at=times,
    )

    icao_list = sorted(set(icaos.tolist()))
    labels = dataset._labels(icao_list)
    session_list = sorted(set(sessions.tolist()))
    icao_index = {icao: i for i, icao in enumerate(icao_list)}
    session_index = {session: i for i, session in enumerate(session_list)}
    rssi = data["rssi_db"][index]
    payload = {
        "run": run_dir.name,
        "variant": ckpt["variant"],
        "checkpoint": args.checkpoint,
        "icaos": icao_list,
        "labels": [labels.get(icao, "") for icao in icao_list],
        "sessions": session_list,
        "xs": [round(float(v), 3) for v in xy[:, 0]],
        "ys": [round(float(v), 3) for v in xy[:, 1]],
        "ic": [icao_index[icao] for icao in icaos.tolist()],
        "se": [session_index[s] for s in sessions.tolist()],
        "sp": split.tolist(),
        "pr": [icao_index.get(classes[p], -1) for p in pred.tolist()],
        "ok": [
            -1 if s != 1 else int(classes[p] == icao)
            for s, p, icao in zip(split.tolist(), pred.tolist(), icaos.tolist())
        ],
        "cf": [round(float(v), 2) for v in conf.tolist()],
        "rs": [round(float(v), 1) if np.isfinite(v) else None for v in rssi.tolist()],
        "tm": times.tolist(),
    }
    html = _TEMPLATE.replace("__TITLE__", f"{run_dir.name} · embedding").replace(
        "__DATA__",
        json.dumps(payload, separators=(",", ":")),
    )
    (run_dir / "embedding.html").write_text(html)

    print(f"embedded {len(index)} messages ({', '.join(f'{(split == s).sum()} {name}' for s, name in enumerate(SPLITS) if (split == s).any())})")
    print(f"  {run_dir / 'embedding.npz'}")
    print(f"  {run_dir / 'embedding.html'}")


_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #14171c; color: #cfd6e0;
               font: 13px/1.4 -apple-system, system-ui, sans-serif; overflow: hidden; }
  #bar { position: fixed; top: 0; left: 0; right: 0; padding: 8px 12px; display: flex;
         gap: 14px; align-items: center; background: #1b2027; border-bottom: 1px solid #2a313b;
         flex-wrap: wrap; z-index: 2; }
  #bar b { color: #fff; font-weight: 600; }
  #bar .dim { color: #7d8794; }
  select, label { background: #232a33; color: #cfd6e0; border: 1px solid #384252;
                  border-radius: 4px; padding: 2px 6px; }
  label { border: none; background: none; }
  #legend { position: fixed; top: 46px; right: 0; bottom: 0; width: 240px; overflow-y: auto;
            padding: 10px 12px; background: rgba(27,32,39,.92); border-left: 1px solid #2a313b; z-index: 2; }
  .lrow { display: flex; gap: 8px; align-items: center; padding: 2px 4px; border-radius: 4px;
          cursor: pointer; user-select: none; }
  .lrow:hover { background: #232a33; }
  .lrow.off { opacity: .35; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }
  .lrow .n { margin-left: auto; color: #7d8794; }
  #reset { color: #6ea8fe; cursor: pointer; display: none; margin-bottom: 6px; }
  canvas { position: fixed; inset: 0; }
  #tip { position: fixed; pointer-events: none; background: #0d1013; border: 1px solid #384252;
         border-radius: 6px; padding: 8px 10px; display: none; z-index: 3; max-width: 320px; }
  #tip .t { color: #fff; font-weight: 600; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="bar">
  <b>__TITLE__</b>
  <span class="dim" id="meta"></span>
  <label>color by
    <select id="mode">
      <option value="aircraft">aircraft</option>
      <option value="session">session</option>
      <option value="rssi">rssi</option>
      <option value="correctness">correctness</option>
      <option value="prediction">prediction</option>
    </select>
  </label>
  <label><input type="checkbox" id="held"> held-out only</label>
  <span class="dim" id="count"></span>
  <span class="dim">drag to pan · wheel to zoom · double-click to re-fit · hover for detail · click legend to isolate</span>
</div>
<div id="legend"><div id="reset">reset isolation</div><div id="lrows"></div></div>
<div id="tip"></div>
<script>
const D = __DATA__;
const N = D.xs.length;
const SPLIT = ["train", "held-out", "other"];
const canvas = document.getElementById("c"), ctx = canvas.getContext("2d");
const tip = document.getElementById("tip");
let mode = "aircraft", heldOnly = false, iso = new Set();
let scale = 1, cx = 0, cy = 0, W = 0, H = 0, AX = 0, AY = 0;

function catColor(i) { return `hsl(${(i * 137.508) % 360}, 65%, ${48 + (i % 3) * 7}%)`; }
const CORRECT = { "-1": "#5a6472", 0: "#e05252", 1: "#3ecf8e", other: "#4a90d9" };

let rssiLo = Infinity, rssiHi = -Infinity;
for (const v of D.rs) if (v !== null) { rssiLo = Math.min(rssiLo, v); rssiHi = Math.max(rssiHi, v); }
function rssiColor(v) {
  if (v === null) return "#555";
  const t = Math.max(0, Math.min(1, (v - rssiLo) / (rssiHi - rssiLo || 1)));
  return `hsl(${220 - 210 * t}, 80%, 55%)`;
}

function key(i) {
  if (mode === "aircraft") return D.ic[i];
  if (mode === "session") return D.se[i];
  if (mode === "prediction") return D.pr[i];
  if (mode === "correctness") return D.sp[i] === 2 ? "other" : (D.sp[i] === 0 ? "train" : (D.ok[i] ? "correct" : "wrong"));
  return 0;
}
function colorOf(i) {
  if (mode === "aircraft") return catColor(D.ic[i]);
  if (mode === "session") return catColor(D.se[i]);
  if (mode === "prediction") return D.pr[i] < 0 ? "#555" : catColor(D.pr[i]);
  if (mode === "rssi") return rssiColor(D.rs[i]);
  const k = key(i);
  return k === "train" ? CORRECT["-1"] : k === "correct" ? CORRECT[1] : k === "wrong" ? CORRECT[0] : CORRECT.other;
}
function nameOf(k) {
  if (mode === "aircraft" || mode === "prediction")
    return (D.labels[k] || "").split(" ").slice(0, 2).join(" ") || D.icaos[k];
  if (mode === "session") return D.sessions[k];
  return String(k);
}
function passes(i) { return (!heldOnly || D.sp[i] === 1) && (iso.size === 0 || iso.has(key(i))); }
function visible(i) { return !heldOnly || D.sp[i] === 1; }

function anchor() {
  const bar = document.getElementById("bar").offsetHeight;
  const side = document.getElementById("legend").offsetWidth;
  AX = (W - side) / 2;
  AY = bar + (H - bar) / 2;
  return { w: W - side, h: H - bar };
}
function fit() {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < N; i++) {
    x0 = Math.min(x0, D.xs[i]); x1 = Math.max(x1, D.xs[i]);
    y0 = Math.min(y0, D.ys[i]); y1 = Math.max(y1, D.ys[i]);
  }
  const v = anchor();
  cx = (x0 + x1) / 2; cy = (y0 + y1) / 2;
  scale = 0.85 * Math.min(v.w / (x1 - x0 || 1), v.h / (y1 - y0 || 1));
}
function resize() {
  const dpr = window.devicePixelRatio || 1;
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  anchor();
  draw();
}
function sx(i) { return (D.xs[i] - cx) * scale + AX; }
function sy(i) { return (D.ys[i] - cy) * scale + AY; }

let raf = 0;
function draw() { if (!raf) raf = requestAnimationFrame(paint); }
function paint() {
  raf = 0;
  ctx.fillStyle = "#14171c"; ctx.fillRect(0, 0, W, H);
  const groups = new Map();
  let shown = 0;
  for (let i = 0; i < N; i++) {
    if (!visible(i)) continue;
    if (!passes(i)) {
      if (iso.size) { ctx.fillStyle = "#262c34"; ctx.fillRect(sx(i), sy(i), 2, 2); }
      continue;
    }
    shown++;
    const c = colorOf(i);
    let g = groups.get(c); if (!g) groups.set(c, g = []);
    g.push(i);
  }
  for (const [c, idxs] of groups) {
    ctx.fillStyle = c;
    for (const i of idxs) ctx.fillRect(sx(i) - 1, sy(i) - 1, 2.5, 2.5);
  }
  document.getElementById("count").textContent = `${shown.toLocaleString()} / ${N.toLocaleString()} shown`;
  legend();
}

function legend() {
  const rows = document.getElementById("lrows");
  document.getElementById("reset").style.display = iso.size ? "block" : "none";
  if (mode === "rssi") {
    rows.innerHTML = `<div class="dim">rssi ${rssiLo} dB → ${rssiHi} dB</div>` +
      [...Array(12)].map((_, j) => `<span class="swatch" style="display:inline-block;background:${rssiColor(rssiLo + (j / 11) * (rssiHi - rssiLo))}"></span>`).join("");
    return;
  }
  const counts = new Map();
  for (let i = 0; i < N; i++) {
    if (!visible(i)) continue;
    const k = key(i);
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 24);
  rows.innerHTML = top.map(([k, n]) => {
    const c = mode === "correctness"
      ? (k === "train" ? CORRECT["-1"] : k === "correct" ? CORRECT[1] : k === "wrong" ? CORRECT[0] : CORRECT.other)
      : catColor(k);
    const off = iso.size && !iso.has(k) ? " off" : "";
    return `<div class="lrow${off}" data-k="${k}"><span class="swatch" style="background:${c}"></span>` +
           `<span>${nameOf(k)}</span><span class="n">${n.toLocaleString()}</span></div>`;
  }).join("") + (counts.size > top.length ? `<div class="dim">… ${counts.size - top.length} more</div>` : "");
  for (const el of rows.querySelectorAll(".lrow")) {
    el.onclick = () => {
      const raw = el.dataset.k;
      const k = (mode === "correctness") ? raw : Number(raw);
      iso.has(k) ? iso.delete(k) : iso.add(k);
      draw();
    };
  }
}
document.getElementById("reset").onclick = () => { iso.clear(); draw(); };
document.getElementById("mode").onchange = (e) => { mode = e.target.value; iso.clear(); draw(); };
document.getElementById("held").onchange = (e) => { heldOnly = e.target.checked; draw(); };

let dragging = false, px = 0, py = 0, moved = 0;
canvas.onmousedown = (e) => { dragging = true; moved = 0; px = e.clientX; py = e.clientY; };
window.onmouseup = () => dragging = false;
window.onmousemove = (e) => {
  if (dragging) {
    cx -= (e.clientX - px) / scale; cy -= (e.clientY - py) / scale;
    moved += Math.abs(e.clientX - px) + Math.abs(e.clientY - py);
    px = e.clientX; py = e.clientY;
    tip.style.display = "none";
    draw();
    return;
  }
  let best = -1, bd = 100;
  for (let i = 0; i < N; i++) {
    if (!visible(i) || !passes(i)) continue;
    const dx = sx(i) - e.clientX, dy = sy(i) - e.clientY, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  if (best < 0) { tip.style.display = "none"; return; }
  const i = best;
  const label = D.labels[D.ic[i]] || D.icaos[i];
  const predName = D.pr[i] < 0 ? "—" : (D.labels[D.pr[i]] || D.icaos[D.pr[i]]).split(" ")[0] || D.icaos[D.pr[i]];
  tip.innerHTML = `<div class="t">${label}</div>` +
    `${D.icaos[D.ic[i]]} · ${SPLIT[D.sp[i]]}<br>` +
    `session ${D.sessions[D.se[i]]}<br>` +
    `${D.tm[i]} · rssi ${D.rs[i] === null ? "?" : D.rs[i] + " dB"}<br>` +
    `pred ${predName} (${D.cf[i]})` +
    (D.ok[i] === 1 ? " ✓" : D.ok[i] === 0 ? " ✗" : "");
  tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 14, W - 340) + "px";
  tip.style.top = Math.min(e.clientY + 14, H - 120) + "px";
};
canvas.onwheel = (e) => {
  e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0015);
  const wx = (e.clientX - AX) / scale + cx, wy = (e.clientY - AY) / scale + cy;
  scale *= f;
  cx = wx - (e.clientX - AX) / scale; cy = wy - (e.clientY - AY) / scale;
  draw();
};
canvas.ondblclick = () => { fit(); draw(); };

document.getElementById("meta").textContent =
  `${D.variant} · ${D.checkpoint}.pt · ${D.icaos.length} aircraft · ${D.sessions.length} sessions`;
window.onresize = resize;
resize(); fit(); draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
