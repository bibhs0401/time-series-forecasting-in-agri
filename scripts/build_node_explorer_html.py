"""Build a self-contained interactive HTML explorer for the crop graph nodes.

Every crop is one node. The page lets you pick a node and see its attributes,
its weekly Google Trends series, its degree on each multiplex layer, and its
neighbours on a chosen layer, with a clickable network diagram.

The output is a single HTML file with all data embedded, so it opens by
double-clicking. No server, no internet, no extra files.

Usage
    python scripts/build_node_explorer_html.py
    python scripts/build_node_explorer_html.py --train-end 400 --include-te
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.utils.io import read_panel
from src.graph.build_graph import build_multiplex_graph
from src.features.static_attributes import get_raw_attributes

DIRECTED_LAYERS = ("A_lag", "A_te")

LAYER_LABELS = {
    "A_part": "Partial correlation",
    "A_lag": "Lagged correlation",
    "A_te": "Transfer entropy",
    "A_agro": "Agronomic",
    "A_sem": "Semantic",
    "A_taxon": "Taxonomy",
    "A_subst": "Substitute",
    "A_compl": "Complement",
    "A_pheno": "Phenology",
}


def build_payload(graph, panel: pd.DataFrame, train_end: int, seed: int = 0) -> dict:
    crops = list(graph.crops)
    N = len(crops)
    attrs = get_raw_attributes(crops)

    # Shared layout from the union of undirected layers.
    union = np.zeros((N, N), dtype=float)
    for name, A in graph.layers.items():
        if name in DIRECTED_LAYERS:
            continue
        union = np.maximum(union, np.maximum(A, A.T))
    G = nx.from_numpy_array(union)
    raw_pos = nx.spring_layout(G, seed=seed, k=2.2 / np.sqrt(max(N, 1)), iterations=300)

    xs = np.array([raw_pos[i][0] for i in range(N)])
    ys = np.array([raw_pos[i][1] for i in range(N)])

    def rescale(v):
        lo, hi = float(v.min()), float(v.max())
        span = (hi - lo) or 1.0
        return (v - lo) / span

    xs, ys = rescale(xs), rescale(ys)
    pos = {crops[i]: [round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in range(N)}

    # Per-layer edge lists as index triples.
    edges = {}
    layer_meta = []
    for name in graph.layer_names:
        A = graph.layers[name]
        directed = name in DIRECTED_LAYERS
        lst = []
        if directed:
            for i in range(N):
                for j in range(N):
                    if i != j and A[i, j] > 0:
                        lst.append([i, j, round(float(A[i, j]), 4)])
        else:
            for i in range(N):
                for j in range(i + 1, N):
                    w = max(float(A[i, j]), float(A[j, i]))
                    if w > 0:
                        lst.append([i, j, round(w, 4)])
        edges[name] = lst
        possible = N * (N - 1) if directed else N * (N - 1) / 2
        layer_meta.append({
            "layer": name,
            "label": LAYER_LABELS.get(name, name),
            "kind": "directed" if directed else "undirected",
            "edges": len(lst),
            "density": round(len(lst) / possible, 4) if possible else 0.0,
        })

    attr_map = {}
    for c in crops:
        if c in attrs.index:
            attr_map[c] = {
                "family": str(attrs.loc[c, "family"]),
                "season": str(attrs.loc[c, "season_class"]),
                "perish": str(attrs.loc[c, "perishability"]),
            }
        else:
            attr_map[c] = {"family": "unknown", "season": "unknown", "perish": "unknown"}

    dates = [d.strftime("%Y-%m-%d") for d in panel.index]
    series = {}
    stats = {}
    for c in crops:
        v = panel[c].to_numpy(dtype=float)
        series[c] = [None if not np.isfinite(x) else round(float(x), 1) for x in v]
        fin = v[np.isfinite(v)]
        stats[c] = {
            "mean": round(float(fin.mean()), 1) if fin.size else 0.0,
            "max": round(float(fin.max()), 1) if fin.size else 0.0,
            "std": round(float(fin.std()), 1) if fin.size else 0.0,
            "zero_pct": round(100.0 * float(np.mean(np.nan_to_num(v) == 0)), 1),
        }

    return {
        "meta": {
            "n_crops": N,
            "train_start": str(panel.index[0].date()),
            "train_cut": str(panel.index[train_end - 1].date()),
            "panel_end": str(panel.index[-1].date()),
            "n_weeks": len(panel),
            "threshold_pct": 80.0,
        },
        "crops": crops,
        "pos": pos,
        "attrs": attr_map,
        "stats": stats,
        "layers": layer_meta,
        "edges": edges,
        "dates": dates,
        "series": series,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Crop graph node explorer</title>
<style>
  :root {
    --bg: #14161a; --panel: #1c1f26; --line: #2c313b; --txt: #e6e8ec;
    --muted: #9aa2b1; --accent: #5b9dff; --hot: #e5675f; --nbr: #55b98e;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--txt);
         font: 14px/1.5 "Segoe UI", system-ui, sans-serif; }
  header { padding: 18px 24px 12px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0 0 4px; font-size: 20px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 13px; }
  .wrap { display: grid; grid-template-columns: 220px 1fr; gap: 0; min-height: calc(100vh - 74px); }
  .side { border-right: 1px solid var(--line); padding: 14px; overflow-y: auto; max-height: calc(100vh - 74px); }
  .side input { width: 100%; padding: 7px 9px; margin-bottom: 10px; border-radius: 6px;
                border: 1px solid var(--line); background: var(--panel); color: var(--txt); }
  .croplist { display: flex; flex-direction: column; gap: 2px; }
  .croplist button { text-align: left; padding: 6px 9px; border: 0; border-radius: 6px;
                     background: transparent; color: var(--txt); cursor: pointer; font-size: 13px; }
  .croplist button:hover { background: var(--panel); }
  .croplist button.on { background: var(--accent); color: #fff; font-weight: 600; }
  main { padding: 18px 24px 40px; }
  .pills { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 16px; }
  .pill { background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
          padding: 3px 11px; font-size: 12px; color: var(--muted); }
  .stats { display: flex; gap: 22px; flex-wrap: wrap; margin-bottom: 18px; }
  .stat b { display: block; font-size: 22px; font-weight: 600; }
  .stat span { color: var(--muted); font-size: 12px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  .card h2 { margin: 0 0 10px; font-size: 13px; font-weight: 600; color: var(--muted);
             text-transform: uppercase; letter-spacing: .04em; }
  select { padding: 6px 9px; border-radius: 6px; border: 1px solid var(--line);
           background: var(--bg); color: var(--txt); font-size: 13px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .full { grid-column: 1 / -1; }
  .netnode { cursor: pointer; }
  .lbl { font-size: 8px; fill: var(--muted); pointer-events: none; }
  .lbl.on { fill: #fff; font-size: 10px; font-weight: 600; }
  .legend { color: var(--muted); font-size: 12px; margin-top: 8px; }
  .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin: 0 4px 0 12px; }
</style>
</head>
<body>
<header>
  <h1>Crop graph node explorer</h1>
  <div class="sub" id="meta"></div>
</header>
<div class="wrap">
  <aside class="side">
    <input id="search" placeholder="Search crop..." autocomplete="off">
    <div class="croplist" id="croplist"></div>
  </aside>
  <main>
    <h2 style="margin:0;font-size:24px" id="title"></h2>
    <div class="pills" id="pills"></div>
    <div class="stats" id="stats"></div>

    <div style="margin-bottom:14px">
      <label style="color:var(--muted);font-size:13px;margin-right:8px">Graph layer</label>
      <select id="layer"></select>
    </div>

    <div class="grid">
      <div class="card full">
        <h2>Weekly Google Trends series</h2>
        <svg id="ts" width="100%" height="200" preserveAspectRatio="none"></svg>
      </div>

      <div class="card">
        <h2>Network position <span id="netlayer" style="text-transform:none"></span></h2>
        <svg id="net" width="100%" height="440" viewBox="0 0 600 440"></svg>
        <div class="legend">
          <i style="background:var(--hot)"></i> selected
          <i style="background:var(--nbr)"></i> neighbour
          <i style="background:#5a6172"></i> other
        </div>
      </div>

      <div class="card">
        <h2>Neighbours on this layer</h2>
        <div id="nbrs"></div>
      </div>

      <div class="card full">
        <h2>Degree by layer</h2>
        <div id="degs"></div>
      </div>
    </div>
  </main>
</div>

<script>
const D = __DATA__;
const DIRECTED = new Set(["A_lag", "A_te"]);
let sel = D.crops.includes("Tomato") ? "Tomato" : D.crops[0];
let layer = D.layers.some(l => l.layer === "A_agro") ? "A_agro" : D.layers[0].layer;

const idx = {};
D.crops.forEach((c, i) => idx[c] = i);

document.getElementById("meta").textContent =
  D.meta.n_crops + " crop nodes | " + D.meta.n_weeks + " weekly rows "
  + D.meta.train_start + " to " + D.meta.panel_end
  + " | graph fitted on train weeks through " + D.meta.train_cut
  + " | edges kept at or above the " + D.meta.threshold_pct + "th percentile per layer";

const layerSel = document.getElementById("layer");
D.layers.forEach(l => {
  const o = document.createElement("option");
  o.value = l.layer;
  o.textContent = l.label + " (" + l.edges + " " + l.kind + " edges)";
  layerSel.appendChild(o);
});
layerSel.value = layer;
layerSel.onchange = () => { layer = layerSel.value; render(); };

const searchBox = document.getElementById("search");
searchBox.oninput = renderList;

function renderList() {
  const q = searchBox.value.trim().toLowerCase();
  const box = document.getElementById("croplist");
  box.innerHTML = "";
  D.crops.filter(c => c.toLowerCase().includes(q)).forEach(c => {
    const b = document.createElement("button");
    b.textContent = c;
    if (c === sel) b.className = "on";
    b.onclick = () => { sel = c; renderList(); render(); };
    box.appendChild(b);
  });
}

// Neighbours of `crop` on `layer`: [{crop, weight, dir}]
function neighbours(crop, lay) {
  const i = idx[crop], out = [];
  const directed = DIRECTED.has(lay);
  for (const [a, b, w] of D.edges[lay]) {
    if (a === i) out.push({ crop: D.crops[b], weight: w, dir: directed ? "out" : "-" });
    else if (b === i) out.push({ crop: D.crops[a], weight: w, dir: directed ? "in" : "-" });
  }
  out.sort((p, q) => q.weight - p.weight);
  return out;
}

function degree(crop, lay) {
  return new Set(neighbours(crop, lay).map(n => n.crop)).size;
}

function render() {
  const a = D.attrs[sel], s = D.stats[sel];
  document.getElementById("title").textContent = sel;
  document.getElementById("pills").innerHTML =
    ["family: " + a.family, "season: " + a.season, "perishability: " + a.perish]
      .map(t => '<span class="pill">' + t + "</span>").join("");

  const totalDeg = D.layers.reduce((acc, l) => acc + degree(sel, l.layer), 0);
  document.getElementById("stats").innerHTML = [
    ["Total degree", totalDeg], ["Mean GSV", s.mean], ["Peak GSV", s.max],
    ["Std GSV", s.std], ["Zero weeks", s.zero_pct + "%"]
  ].map(([k, v]) => '<div class="stat"><b>' + v + "</b><span>" + k + "</span></div>").join("");

  document.getElementById("netlayer").textContent =
    "— " + D.layers.find(l => l.layer === layer).label;

  drawSeries();
  drawNet();
  drawNeighbours();
  drawDegrees();
}

function drawSeries() {
  const svg = document.getElementById("ts");
  const W = svg.clientWidth || 900, H = 200, P = 26;
  const vals = D.series[sel];
  const fin = vals.filter(v => v !== null);
  const max = Math.max(1, ...fin);
  const n = vals.length;
  let d = "", started = false;
  vals.forEach((v, i) => {
    if (v === null) { started = false; return; }
    const x = P + (i / (n - 1)) * (W - P - 10);
    const y = H - P - (v / max) * (H - 2 * P);
    d += (started ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1) + " ";
    started = true;
  });
  const ticks = [0, Math.round(max / 2), Math.round(max)];
  const grid = ticks.map(t => {
    const y = H - P - (t / max) * (H - 2 * P);
    return '<line x1="' + P + '" x2="' + (W - 10) + '" y1="' + y + '" y2="' + y
      + '" stroke="#2c313b"/><text x="2" y="' + (y + 4) + '" fill="#9aa2b1" font-size="10">' + t + "</text>";
  }).join("");
  const first = D.dates[0], last = D.dates[n - 1];
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.innerHTML = grid
    + '<path d="' + d + '" fill="none" stroke="#5b9dff" stroke-width="1.4"/>'
    + '<text x="' + P + '" y="' + (H - 6) + '" fill="#9aa2b1" font-size="10">' + first + "</text>"
    + '<text x="' + (W - 70) + '" y="' + (H - 6) + '" fill="#9aa2b1" font-size="10">' + last + "</text>";
}

function drawNet() {
  const svg = document.getElementById("net");
  const W = 600, H = 440, P = 30;
  const nbrs = new Set(neighbours(sel, layer).map(n => n.crop));
  const px = c => P + D.pos[c][0] * (W - 2 * P);
  const py = c => P + (1 - D.pos[c][1]) * (H - 2 * P);

  let edgeSvg = "";
  const maxW = Math.max(...D.edges[layer].map(e => e[2]), 1);
  for (const [a, b, w] of D.edges[layer]) {
    const ca = D.crops[a], cb = D.crops[b];
    const touch = (ca === sel || cb === sel);
    edgeSvg += '<line x1="' + px(ca).toFixed(1) + '" y1="' + py(ca).toFixed(1)
      + '" x2="' + px(cb).toFixed(1) + '" y2="' + py(cb).toFixed(1)
      + '" stroke="' + (touch ? "#55b98e" : "#39404d") + '" stroke-opacity="'
      + (touch ? 0.95 : 0.30) + '" stroke-width="' + (touch ? 0.7 + 2.0 * (w / maxW) : 0.5).toFixed(2) + '"/>';
  }

  let nodeSvg = "";
  for (const c of D.crops) {
    const isSel = c === sel, isNbr = nbrs.has(c);
    const fill = isSel ? "#e5675f" : (isNbr ? "#55b98e" : "#5a6172");
    const r = isSel ? 8 : (isNbr ? 5.5 : 4);
    nodeSvg += '<circle class="netnode" cx="' + px(c).toFixed(1) + '" cy="' + py(c).toFixed(1)
      + '" r="' + r + '" fill="' + fill + '" stroke="#14161a" stroke-width="1"'
      + ' onclick="pick(\'' + c.replace(/'/g, "\\'") + '\')"><title>' + c + "</title></circle>";
    if (isSel || isNbr) {
      nodeSvg += '<text class="lbl' + (isSel ? " on" : "") + '" x="' + (px(c) + 8).toFixed(1)
        + '" y="' + (py(c) + 3).toFixed(1) + '">' + c + "</text>";
    }
  }
  svg.innerHTML = edgeSvg + nodeSvg;
}

function pick(c) { sel = c; renderList(); render(); }

function drawNeighbours() {
  const rows = neighbours(sel, layer);
  const directed = DIRECTED.has(layer);
  const el = document.getElementById("nbrs");
  if (!rows.length) {
    el.innerHTML = '<div style="color:var(--muted)">No retained edges for '
      + sel + " on this layer.</div>";
    return;
  }
  el.innerHTML = "<table><thead><tr><th>Neighbour</th>"
    + (directed ? "<th>Direction</th>" : "")
    + '<th class="num">Weight</th></tr></thead><tbody>'
    + rows.map(r => "<tr><td>" + r.crop + "</td>"
      + (directed ? "<td>" + r.dir + "</td>" : "")
      + '<td class="num">' + r.weight.toFixed(3) + "</td></tr>").join("")
    + "</tbody></table>";
}

function drawDegrees() {
  const data = D.layers.map(l => ({ label: l.label, lay: l.layer, v: degree(sel, l.layer) }));
  const max = Math.max(1, ...data.map(d => d.v));
  document.getElementById("degs").innerHTML =
    '<table><tbody>' + data.map(d =>
      "<tr><td style='width:170px'>" + d.label + "</td>"
      + "<td><div style='background:" + (d.lay === layer ? "#5b9dff" : "#3d4757")
      + ";height:11px;border-radius:3px;width:" + (100 * d.v / max).toFixed(1) + "%'></div></td>"
      + '<td class="num" style="width:46px">' + d.v + "</td></tr>").join("")
    + "</tbody></table>";
}

renderList();
render();
window.addEventListener("resize", drawSeries);
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-end", type=int, default=None,
                   help="build the graph on panel[:train_end] (default: all but last 52 weeks)")
    p.add_argument("--include-te", action="store_true", help="also build A_te (slow)")
    p.add_argument("--no-kg", action="store_true", help="drop the knowledge-graph layers")
    p.add_argument("--threshold-pct", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    panel = read_panel()
    crops = pd.read_csv(config.PANEL_DIR / "crop_list.csv")["crop"].tolist()
    crops = [c for c in crops if c in panel.columns]
    panel = panel[crops]

    train_end = args.train_end if args.train_end is not None else max(1, len(panel) - 52)
    print(f"Building graph on panel[:{train_end}], N={len(crops)} crops ...")
    graph = build_multiplex_graph(
        panel_train=panel.iloc[:train_end], crops=crops,
        include_te=args.include_te, include_kg=not args.no_kg,
        threshold_pct=args.threshold_pct, verbose=True,
    )

    payload = build_payload(graph, panel, train_end, seed=args.seed)
    payload["meta"]["threshold_pct"] = args.threshold_pct

    out = Path(args.out) if args.out else (config.FIGURES_DIR / "graph" / "node_explorer.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    out.write_text(html, encoding="utf-8")
    print(f"\nWrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print("Open it by double-clicking the file, or run:")
    print(f'  start "" "{out}"')


if __name__ == "__main__":
    main()
