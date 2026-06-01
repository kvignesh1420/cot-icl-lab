// The CoT-ICL Lab pipeline — the main animated figure at the top.
//
// The full learning loop, in three stages:
//   1. Sample K in-context examples from the generator: G (a graph), E (an
//      embedding matrix), and H (a small MLP).
//   2. Feed them to a transformer; its attention map animates from diffuse to a
//      structured pattern that recovers the directed edges of the DAG G.
//   3. Decode by argmax over the vocabulary to predict the held-out answer.
// Token colors are consistent everywhere: input = slate, reasoning = violet,
// answer = teal. Click to regenerate with a new DAG.

import { generateAdjList } from "../dag.js";
import { AbstractConstructor } from "../token-processor.js";
import { RNG, randomSeed } from "../rng.js";

const d3 = window.d3;
const W = 1000;
const H = 338;
// shift all pipeline content right by OX so the left/right margins balance
// (stage 1 is denser than stage 3, which otherwise leaves the right side emptier)
const OX = 36;
const N = 4;
const C = 3;
const L = N + C;
const K = 4;
const CELL = 18;
const ATTN_X = 402;
const ATTN_Y = 132;

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
const attnColor = d3.interpolateRgb("#eef2fb", "#1d4ed8");
const tokenLabel = (i) => (i < N ? `x${i + 1}` : `y${i - N + 1}`);

// move point `p` toward (cx, cy) by distance d — used to trim edges to node edges
const segTo = (p, cx, cy, d) => {
  const dx = cx - p[0], dy = cy - p[1], Ln = Math.hypot(dx, dy) || 1;
  return [p[0] + (dx / Ln) * d, p[1] + (dy / Ln) * d];
};

export function mountHeroAnimation(root) {
  root.innerHTML = `<svg class="hero-svg" viewBox="${-OX} 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="From synthetic data to a transformer prediction"></svg>`;
  const svg = d3.select(root).select("svg");

  let runId = 0;
  let seed = 23;
  let cells = [];
  let uniform = [];
  let target = [];

  function buildScenario(s) {
    const adjList = generateAdjList({ nInputs: N, nParents: 2, chainLength: C }, new RNG(s));
    const ctor = new AbstractConstructor({ vocabSize: 128, nDims: 10, seed: s + 4 });
    const ex = ctor.generateExample(adjList, N);
    uniform = Array.from({ length: L }, () => new Array(L).fill(0));
    target = Array.from({ length: L }, () => new Array(L).fill(0));
    for (let i = 0; i < L; i++) {
      for (let j = 0; j <= i; j++) uniform[i][j] = 1 / (i + 1);
      target[i][i] = 1;
      if (i >= N) for (const p of adjList[i - N]) target[i][p] += 3;
      let sum = 0;
      for (let j = 0; j <= i; j++) sum += target[i][j];
      for (let j = 0; j <= i; j++) target[i][j] = sum > 0 ? target[i][j] / sum : 0;
    }
    return { adjList, ex };
  }

  function updateAttention(t) {
    for (let i = 0; i < L; i++) {
      let rowMax = 1e-6;
      const row = [];
      for (let j = 0; j <= i; j++) {
        row[j] = (1 - t) * uniform[i][j] + t * target[i][j];
        if (row[j] > rowMax) rowMax = row[j];
      }
      for (let j = 0; j <= i; j++) cells[i][j].attr("fill", attnColor(row[j] / rowMax));
    }
  }

  function flowArrow(x1, x2, y) {
    return svg.append("path").attr("class", "flow-arrow").attr("d", `M${x1},${y} L${x2},${y}`).attr("opacity", 0);
  }
  function pulseAlong(x1, x2, y, gPulse) {
    gPulse.append("circle").attr("class", "flow-dot").attr("r", 3.5).attr("cx", x1).attr("cy", y)
      .transition().duration(680).attr("cx", x2).on("end", function () { d3.select(this).remove(); });
  }

  function drawMiniDAG(g2, adjList) {
    const gd = g2.append("g").attr("class", "minidag");
    // spread the nodes wide so the directed edges (and arrowheads) read clearly
    const xIn = (i) => 556 + i * 30;
    const xCh = (c) => 571 + c * 30;
    const topY = 156, botY = 220, r = 6;
    const pos = (id) => (id < N ? { x: xIn(id), y: topY } : { x: xCh(id - N), y: botY });
    gd.append("text").attr("class", "recover-title").attr("x", 601).attr("y", 142).attr("text-anchor", "middle").text("G, recovered");
    const edges = [];
    adjList.forEach((ps, c) => ps.forEach((p) => edges.push({ s: p, t: N + c })));
    const eSel = gd.selectAll("path.mini-edge").data(edges).join("path").attr("class", "mini-edge")
      .attr("marker-end", "url(#mini-arrow)")
      .attr("d", (e) => {
        const a = pos(e.s), b = pos(e.t);
        const sameRow = e.s >= N;
        const mx = (a.x + b.x) / 2;
        const my = sameRow ? Math.max(a.y, b.y) + 12 : (a.y + b.y) / 2;
        const s = segTo([a.x, a.y], mx, my, r + 1);
        const t = segTo([b.x, b.y], mx, my, r + 3.5);
        return `M${s[0]},${s[1]} Q${mx},${my} ${t[0]},${t[1]}`;
      })
      .attr("opacity", 0);
    for (let id = 0; id < L; id++) {
      const p = pos(id);
      const ng = gd.append("g").attr("transform", `translate(${p.x},${p.y})`);
      ng.append("circle").attr("class", `mini-node ${id < N ? "mini-in" : "mini-ch"}${id === L - 1 ? " mini-answer" : ""}`).attr("r", r);
      ng.append("text").attr("class", "mini-lbl").attr("text-anchor", "middle").attr("dy", "0.3em").attr("font-size", 6.5).text(tokenLabel(id));
    }
    return eSel;
  }

  function scaffold(scn) {
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    const addArrowMarker = (id, w, fill, refX) =>
      defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", refX).attr("refY", 5)
        .attr("markerUnits", "userSpaceOnUse").attr("markerWidth", w).attr("markerHeight", w).attr("orient", "auto")
        .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", fill);
    addArrowMarker("hero-arrow", 9, "#9aa0a6", 8); // flow arrows between stages (neutral gray)
    addArrowMarker("mini-arrow", 7, "#2563eb", 9); // recovered G edges (blue, directed)
    addArrowMarker("glyph-arrow", 5, "#2563eb", 9); // G-glyph edges in the generator panel

    // title and hint stay frame-relative (undo the OX pipeline shift)
    svg.append("text").attr("class", "hero-title").attr("x", W / 2 - OX).attr("y", 22).attr("text-anchor", "middle").attr("font-size", 14).attr("opacity", 0).text("FROM SYNTHETIC DATA TO A TRANSFORMER PREDICTION");
    svg.append("text").attr("class", "hero-hint").attr("x", W - 14 - OX).attr("y", 20).attr("text-anchor", "end").attr("font-size", 10).attr("opacity", 0).text("click to regenerate ↻");

    const stage = (x, num, txt) => svg.append("text").attr("class", "stage-label").attr("x", x).attr("y", 46).attr("text-anchor", "middle").attr("opacity", 0).html(`<tspan class="stage-num">${num} · </tspan>${txt}`);
    const s1 = stage(160, "1", "sample K examples from (G, E, H)");
    const s2 = stage(510, "2", "a transformer reads them; attention learns G");
    const s3 = stage(806, "3", "decode the held-out answer");

    // ---- stage 1: the generator, drawn as G (graph) · E (matrix) · H (MLP) ----
    const g1 = svg.append("g").attr("class", "stage1").attr("opacity", 0);
    const pX = 12, pW = 126, pY = 108, pH = 176;
    g1.append("text").attr("class", "gen-sub").attr("x", pX + pW / 2).attr("y", pY - 6).attr("text-anchor", "middle").text("the generator");
    g1.append("rect").attr("class", "gen-panel").attr("x", pX).attr("y", pY).attr("width", pW).attr("height", pH).attr("rx", 9);
    const gcx = pX + 80, letX = pX + 22;

    // G — the ACTUAL DAG (same scn.adjList the transformer recovers in stage 2),
    // drawn compactly: N inputs across the top, C chain nodes below, directed blue edges.
    const glyG = g1.append("g").attr("class", "gen-glyph glyph-G").attr("opacity", 0);
    glyG.append("text").attr("class", "glyph-letter g").attr("x", letX).attr("y", pY + 37).attr("text-anchor", "middle").text("G");
    {
      const topY = pY + 20, botY = pY + 44, gr = 3;
      const xIn = (i) => gcx - 26 + i * (52 / (N - 1));
      const xCh = (c) => gcx - 22 + c * (44 / (C - 1 || 1));
      const gpos = (id) => (id < N ? [xIn(id), topY] : [xCh(id - N), botY]);
      scn.adjList.forEach((ps, c) => ps.forEach((p) => {
        const a = gpos(p), b = gpos(N + c);
        const sameRow = p >= N;
        const mx = (a[0] + b[0]) / 2;
        const my = sameRow ? Math.max(a[1], b[1]) + 8 : (a[1] + b[1]) / 2;
        const s = segTo(a, mx, my, gr + 0.5);
        const t = segTo(b, mx, my, gr + 3);
        glyG.append("path").attr("class", "glyph-edge g").attr("d", `M${s[0]},${s[1]} Q${mx},${my} ${t[0]},${t[1]}`);
      }));
      for (let id = 0; id < L; id++) {
        const [x, y] = gpos(id);
        const cls = id < N ? "n-in" : id === L - 1 ? "n-ans" : "n-ch";
        glyG.append("circle").attr("class", `glyph-node ${cls}`).attr("cx", x).attr("cy", y).attr("r", gr);
      }
    }
    glyG.append("text").attr("class", "glyph-desc").attr("x", gcx).attr("y", pY + 56).attr("text-anchor", "middle").text("causal structure");

    // E — the embedding matrix (amber grid, token x dims)
    const glyE = g1.append("g").attr("class", "gen-glyph glyph-E").attr("opacity", 0);
    glyE.append("text").attr("class", "glyph-letter e").attr("x", letX).attr("y", pY + 89).attr("text-anchor", "middle").text("E");
    {
      const cols = 8, rows = 4, cw = 7, gap = 1.4, ox = gcx - 31, oy = pY + 72;
      const er = new RNG(seed + 17);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        glyE.append("rect").attr("class", "emat-cell").attr("x", ox + c * (cw + gap)).attr("y", oy + r * (cw + gap))
          .attr("width", cw).attr("height", cw).attr("rx", 1).attr("fill-opacity", 0.22 + 0.72 * er.next());
      }
      glyE.append("rect").attr("class", "emat-frame").attr("x", ox - 1.5).attr("y", oy - 1.5)
        .attr("width", cols * (cw + gap) + 1).attr("height", rows * (cw + gap) + 1).attr("rx", 1.5);
    }
    glyE.append("text").attr("class", "glyph-desc").attr("x", gcx).attr("y", pY + 113).attr("text-anchor", "middle").text("embedding matrix");

    // H — a small MLP (violet)
    const glyH = g1.append("g").attr("class", "gen-glyph glyph-H").attr("opacity", 0);
    glyH.append("text").attr("class", "glyph-letter h").attr("x", letX).attr("y", pY + 143).attr("text-anchor", "middle").text("H");
    {
      const cy = pY + 138, span = 28;
      const layers = [[3, gcx - 22], [3, gcx + 2], [2, gcx + 26]];
      const cols = layers.map(([n, x]) => Array.from({ length: n }, (_, i) => [x, n === 1 ? cy : cy - span / 2 + (i * span) / (n - 1)]));
      for (let l = 0; l < cols.length - 1; l++) cols[l].forEach((a) => cols[l + 1].forEach((b) => glyH.append("path").attr("class", "mlp-edge").attr("d", `M${a[0]},${a[1]} L${b[0]},${b[1]}`)));
      cols.flat().forEach((p) => glyH.append("circle").attr("class", "mlp-node").attr("cx", p[0]).attr("cy", p[1]).attr("r", 2.6));
    }
    glyH.append("text").attr("class", "glyph-desc").attr("x", gcx).attr("y", pY + 162).attr("text-anchor", "middle").text("token processor");

    // examples block geometry (shared chip size used by the held-out query too)
    const exX = 184, chip = 16, pitch = 18, exTop = 120, exPitch = 24;
    const exRight = exX + (L - 1) * pitch + chip;
    // arrow: generator -> the K example rows it produces (clear gaps on both ends)
    g1.append("path").attr("class", "flow-arrow").attr("d", `M${pX + pW + 6},164 L${exX - 8},164`).attr("opacity", 0.9);

    for (let k = 0; k < K; k++) {
      const row = g1.append("g").attr("class", `exrow exrow-${k}`).attr("opacity", 0).attr("transform", `translate(${exX}, ${exTop + k * exPitch})`);
      for (let j = 0; j < L; j++) {
        const cls = j < N ? "ex-in" : j === L - 1 ? "ex-ans" : "ex-think";
        row.append("rect").attr("class", cls).attr("x", j * pitch).attr("y", 0).attr("width", chip).attr("height", chip).attr("rx", 3);
      }
    }
    // role legend (below the example rows; starts a little left so all three
    // items fit in the gap before the transformer box)
    const leg = g1.append("g").attr("transform", `translate(${exX - 34}, ${exTop + K * exPitch + 6})`);
    const legItem = (x, cls, label) => {
      leg.append("rect").attr("class", cls).attr("x", x).attr("y", 0).attr("width", 10).attr("height", 10).attr("rx", 2);
      leg.append("text").attr("class", "role-legend").attr("x", x + 14).attr("y", 9).text(label);
    };
    legItem(0, "ex-in", "input");
    legItem(46, "ex-think", "reasoning");
    legItem(116, "ex-ans", "answer");

    // ---- stage 2: transformer box + attention heatmap + mini-DAG ----
    const g2 = svg.append("g").attr("class", "stage2").attr("opacity", 0);
    g2.append("rect").attr("class", "tf-box").attr("x", 336).attr("y", 66).attr("width", 348).attr("height", 232).attr("rx", 10);
    g2.append("text").attr("class", "tf-label").attr("x", 510).attr("y", 90).attr("text-anchor", "middle").text("Transformer");
    g2.append("text").attr("class", "tf-sub").attr("x", 510).attr("y", 104).attr("text-anchor", "middle").text("self-attention over the K-example prompt");

    const gAttn = g2.append("g");
    cells = Array.from({ length: L }, () => []);
    for (let i = 0; i < L; i++) {
      for (let j = 0; j <= i; j++) {
        cells[i][j] = gAttn.append("rect").attr("class", "attn-cell")
          .attr("x", ATTN_X + j * CELL).attr("y", ATTN_Y + i * CELL).attr("width", CELL).attr("height", CELL).attr("fill", attnColor(0.1));
      }
    }
    g2.append("rect").attr("class", "attn-frame").attr("x", ATTN_X).attr("y", ATTN_Y).attr("width", L * CELL).attr("height", L * CELL);
    for (let i = 0; i < L; i++) {
      g2.append("text").attr("class", "attn-axis").attr("x", ATTN_X - 5).attr("y", ATTN_Y + i * CELL + CELL / 2 + 3).attr("text-anchor", "end").text(tokenLabel(i));
      g2.append("text").attr("class", "attn-axis").attr("x", ATTN_X + i * CELL + CELL / 2).attr("y", ATTN_Y - 5).attr("text-anchor", "middle").text(tokenLabel(i));
    }
    const miniEdgeSel = drawMiniDAG(g2, scn.adjList);
    // the attention map's off-diagonal weights ARE the directed edges of G
    g2.append("path").attr("class", "recover-arrow").attr("data-el", "recarr").attr("d", "M530,188 L549,188").attr("opacity", 0);
    g2.append("text").attr("class", "attn-caption").attr("data-el", "caption").attr("x", 465).attr("y", ATTN_Y + L * CELL + 18).attr("text-anchor", "middle").text("");
    g2.append("text").attr("class", "counter").attr("data-el", "counter").attr("x", 465).attr("y", ATTN_Y + L * CELL + 34).attr("text-anchor", "middle").text("");
    svg.append("text").attr("class", "anno anno-attn").attr("x", 510).attr("y", 312).attr("text-anchor", "middle").attr("font-size", 11).attr("opacity", 0).text("attention concentrates on each token's causal parents");

    // ---- stage 3: held-out query + decode (argmax over vocab) + prediction ----
    // chips here match the example-row chip size exactly, so the rows line up
    const g3 = svg.append("g").attr("class", "stage3").attr("opacity", 0);
    const qx = 722, qy = 156, qp = pitch, qw = chip;
    const qRight = qx + (L - 1) * qp + qw, qMid = (qx + qRight) / 2, qCy = qy + qw / 2;
    g3.append("text").attr("class", "gen-sub").attr("x", qMid).attr("y", qy - 9).attr("text-anchor", "middle").text("held-out query");
    for (let j = 0; j < L; j++) {
      const isAns = j === L - 1;
      g3.append("rect").attr("class", isAns ? "pred-chip" : (j < N ? "ex-in" : "ex-think")).attr("data-q", isAns ? "ans" : "")
        .attr("x", qx + j * qp).attr("y", qy).attr("width", qw).attr("height", qw).attr("rx", 3);
    }
    const ansCx = qx + (L - 1) * qp + qw / 2;
    g3.append("text").attr("class", "pred-q").attr("data-el", "qmark").attr("x", ansCx).attr("y", qCy + 3.6).attr("text-anchor", "middle").text("?");
    // arrow: transformer -> the held-out query (clear gaps on both ends)
    g3.append("path").attr("class", "flow-arrow").attr("data-el", "arr3").attr("d", `M690,${qCy} L${qx - 7},${qCy}`).attr("opacity", 0);

    // decode: argmax over the vocabulary (bar aligns under the query chips)
    const barX = qx, barY = qy + 56, barW = (L - 1) * qp + qw, barH = 30;
    g3.append("text").attr("class", "decode-lbl").attr("x", barX).attr("y", barY - 6).text("decode: logits over 128 tokens → argmax");
    g3.append("rect").attr("class", "attn-frame").attr("x", barX - 1).attr("y", barY - 1).attr("width", barW + 2).attr("height", barH + 2);
    const logits = scn.ex.steps[C - 1].logits;
    const vmin = Math.min(...logits), vmax = Math.max(...logits), span = Math.max(1e-6, vmax - vmin);
    const answerTok = scn.ex.chainTokens[C - 1];
    const bw = barW / logits.length;
    const gbars = g3.append("g").attr("data-el", "vbars").attr("opacity", 0);
    logits.forEach((l, v) => {
      if (v === answerTok) return;
      const h = ((l - vmin) / span) * barH;
      gbars.append("rect").attr("class", "vbar").attr("x", barX + v * bw).attr("y", barY + barH - h).attr("width", Math.max(0.6, bw - 0.1)).attr("height", Math.max(0.6, h));
    });
    // the argmax spike (drawn wider, teal) + reveal handle
    g3.append("rect").attr("class", "vbar max").attr("data-el", "vmax").attr("x", barX + answerTok * bw - 1).attr("y", barY).attr("width", 3).attr("height", barH).attr("opacity", 0);
    g3.append("text").attr("class", "pred-ok").attr("data-el", "prednote").attr("x", qMid).attr("y", barY + barH + 16).attr("text-anchor", "middle").attr("opacity", 0).text("");

    const gPulse = svg.append("g");
    return { s1, s2, s3, g1, g2, g3, gPulse, miniEdgeSel, answerTok, ansCx, qy };
  }

  async function playOnce(myId) {
    const scn = buildScenario(seed);
    const parts = scaffold(scn);
    updateAttention(0);

    svg.select(".hero-title").transition().duration(450).attr("opacity", 1);
    svg.select(".hero-hint").transition().duration(450).attr("opacity", 0.7);
    if (myId !== runId) return;

    // stage 1: the panel appears, then its three components G -> E -> H,
    // which then emit the K example rows
    parts.s1.transition().duration(350).attr("opacity", 1);
    parts.g1.transition().duration(300).attr("opacity", 1);
    ["G", "E", "H"].forEach((c, i) => svg.select(`.glyph-${c}`).transition().delay(120 + i * 180).duration(320).attr("opacity", 1));
    for (let k = 0; k < K; k++) svg.select(`.exrow-${k}`).transition().delay(620 + k * 120).duration(300).attr("opacity", 1);
    await sleep(620 + K * 120 + 320);
    if (myId !== runId) return;

    // flow into transformer (arrow sits in the gap between examples and the box)
    parts.s2.transition().duration(350).attr("opacity", 1);
    parts.g2.transition().duration(450).attr("opacity", 1);
    flowArrow(314, 332, 164).transition().duration(300).attr("opacity", 1);
    for (let p = 0; p < 4; p++) { pulseAlong(314, 332, 164, parts.gPulse); await sleep(150); }
    if (myId !== runId) return;

    // stage 2: attention learns the DAG (stepped; edges reveal)
    const counterEl = svg.select('[data-el="counter"]');
    const captionEl = svg.select('[data-el="caption"]');
    const nEdges = parts.miniEdgeSel.size();
    const STEPS = 40;
    for (let st = 0; st <= STEPS; st++) {
      if (myId !== runId) return;
      const t = st / STEPS;
      updateAttention(t);
      const reveal = Math.floor(t * nEdges);
      parts.miniEdgeSel.each(function (d, i) { d3.select(this).attr("opacity", i < reveal ? 0.9 : 0); });
      svg.select('[data-el="recarr"]').attr("opacity", t > 0.5 ? Math.min(1, (t - 0.5) * 3) : 0);
      counterEl.text(`in-context examples seen: ${st}`);
      captionEl.html(t < 0.5 ? "attention: diffuse" : 'attention: diffuse → <tspan class="struct">recovers the directed edges of G</tspan>');
      if (t > 0.45) svg.select(".anno-attn").attr("opacity", Math.min(1, (t - 0.45) * 3));
      await sleep(62);
    }
    if (myId !== runId) return;

    // stage 3: decode + predict
    parts.s3.transition().duration(350).attr("opacity", 1);
    parts.g3.transition().duration(350).attr("opacity", 1);
    svg.select('[data-el="arr3"]').transition().duration(300).attr("opacity", 1);
    for (let p = 0; p < 3; p++) { pulseAlong(690, 713, 164, parts.gPulse); await sleep(150); }
    await sleep(150);
    if (myId !== runId) return;
    // reveal the logits, then the argmax spike
    svg.select('[data-el="vbars"]').transition().duration(350).attr("opacity", 1);
    await sleep(380);
    if (myId !== runId) return;
    svg.select('[data-el="vmax"]').transition().duration(350).attr("opacity", 1);
    // fill the held-out answer chip + highlight attention answer row + pulse DAG answer node
    svg.select('[data-el="qmark"]').transition().duration(250).attr("opacity", 0).remove();
    svg.selectAll('[data-q="ans"]').classed("hit", true);
    parts.g2.append("rect").attr("class", "attn-answer-row").attr("x", ATTN_X).attr("y", ATTN_Y + (L - 1) * CELL).attr("width", L * CELL).attr("height", CELL).attr("opacity", 0).transition().duration(300).attr("opacity", 1);
    parts.g2.select(".mini-answer").transition().duration(220).attr("r", 9).transition().duration(220).attr("r", 6);
    parts.g3.append("text").attr("class", "pred-tok").attr("x", parts.ansCx).attr("y", parts.qy + 11.3).attr("text-anchor", "middle").attr("font-size", 8.5).attr("opacity", 0).text(parts.answerTok)
      .transition().delay(150).duration(300).attr("opacity", 1);
    svg.select('[data-el="prednote"]').text(`predicted y${C} = ${parts.answerTok}   ✓ correct`).transition().duration(350).attr("opacity", 1);
    await sleep(900);
  }

  async function loop(myId) {
    while (myId === runId && document.body.contains(root)) {
      await playOnce(myId);
      if (myId !== runId) return;
      await sleep(2600);
      if (myId !== runId) return;
      svg.selectAll(".stage1, .stage2, .stage3, .anno, .flow-arrow").transition().duration(450).attr("opacity", 0);
      await sleep(550);
      seed = (seed + 101) % 100000;
    }
  }

  function start() { runId += 1; loop(runId); }
  svg.on("click", () => { seed = randomSeed() % 100000; start(); });
  start();
}
