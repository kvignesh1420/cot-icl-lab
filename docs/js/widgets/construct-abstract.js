// Interactive 2A — Track A abstract token construction.
//
// Shows how token embeddings propagate through the DAG and are decoded into an
// output token, and highlights the three pieces of HIDDEN GROUND TRUTH the model
// must infer from the in-context examples:
//   G — the causal structure (which parents feed each step)   [blue]
//   E — the embedding matrix (token <-> vector, and decoder Eᵀ) [amber]
//   H — the embedding transformation (how parents combine)     [violet]
//
// Faithful to token_processor.py + data.py get_output_token.

import { generateAdjList, clampNParents, nodeLabel } from "../dag.js";
import { AbstractConstructor } from "../token-processor.js";
import { RNG, randomSeed } from "../rng.js";

const d3 = window.d3;
const SW = 720; // svg width
const CARD_W = 58;
const CARD_H = 42;
const ROW_TOP = 104;
const SH = 178;

export function mountConstructAbstract(root) {
  const state = {
    vocabSize: 64,
    nDims: 16,
    activation: "leaky_relu",
    chainLength: 3,
    nInputs: 4,
    nParents: 2,
    seed: 11,
    step: 0,
  };
  let ctor = null;
  let example = null;
  let adjList = null;
  let nodeEmb = []; // embedding (E[token]) for every node position
  let playTimer = null;

  root.innerHTML = `
    <div class="widget track-shared">
      <div class="widget-head">
        <span class="track-badge">token construction</span>
        <span class="widget-title">Embeddings propagate through the DAG, then decode to a token</span>
      </div>

      <div class="controls controls-oneline">
        <div class="control">
          <label>vocab_size</label>
          <div class="segmented" data-seg="vocabSize">
            <button data-val="64">64</button><button data-val="128">128</button><button data-val="256">256</button>
          </div>
        </div>
        <div class="control">
          <label>n_dims · <span class="value-tag" data-v="nDims"></span></label>
          <input type="range" data-c="nDims" min="4" max="16" step="1">
        </div>
        <div class="control">
          <label>activation</label>
          <select data-c="activation">
            <option value="leaky_relu">leaky_relu</option>
            <option value="relu">relu</option>
            <option value="silu">silu</option>
            <option value="identity">identity</option>
          </select>
        </div>
        <div class="control">
          <label>n_inputs · <span class="value-tag" data-v="nInputs"></span></label>
          <input type="range" data-c="nInputs" min="2" max="6" step="1">
        </div>
        <div class="control">
          <label>n_parents · <span class="value-tag" data-v="nParents"></span></label>
          <input type="range" data-c="nParents" min="1" max="4" step="1">
        </div>
        <div class="control">
          <label>chain_length · <span class="value-tag" data-v="chainLength"></span></label>
          <input type="range" data-c="chainLength" min="1" max="4" step="1">
        </div>
        <div class="control">
          <label>&nbsp;</label>
          <button class="btn" data-act="regen">↻ New seed</button>
        </div>
      </div>

      <div class="abs-top">
        <svg class="prop-svg" viewBox="0 0 ${SW} ${SH}" preserveAspectRatio="xMidYMid meet" aria-label="Embedding propagation through the DAG"></svg>
        <div class="gt-callout">
          <div class="gt-callout-title">Hidden ground truth</div>
          <div class="gt-callout-sub">what the model must infer from the in-context examples alone:</div>
          <div class="gt-items">
            <div class="gt-item gt-G"><b>G</b> · causal structure<span class="gt-desc">which parents feed each step</span></div>
            <div class="gt-item gt-E"><b>E</b> · embedding matrix<span class="gt-desc">the token ↔ vector table</span></div>
            <div class="gt-item gt-H"><b>H</b> · transformation<span class="gt-desc">how parents combine</span></div>
          </div>
        </div>
      </div>
      <div class="muted-note" style="margin:0.5rem 0 0">Each node carries an embedding <b style="color:#d97706">E[token]</b> (amber bars); edges are the causal structure <b style="color:#2563eb">G</b>. As each node is computed, every parent embedding is transformed by <b style="color:#7c3aed">H</b> (shown just above the node) and the results are averaged.</div>

      <div data-el="decode" style="margin-top:0.9rem"></div>

      <div class="controls" style="margin-top:0.9rem;border-top:1px solid var(--rule);padding-top:0.9rem">
        <button class="btn" data-act="reset">↺ Reset</button>
        <button class="btn btn-accent" data-act="step">Step ▸</button>
        <button class="btn" data-act="play">▶ Play</button>
        <span class="muted-note" style="margin:0" data-el="progress"></span>
      </div>
      <p class="muted-note">The produced token id is <strong>opaque</strong>; succeeding requires inferring G, E, and H together. Difficulty is set by <code>vocab_size</code> and <code>activation</code> (the paper's TokenCoverage). This is exactly what the custom Llama models meta-train on.</p>
    </div>`;

  const svg = d3.select(root).select("svg.prop-svg");
  const decodeEl = root.querySelector('[data-el="decode"]');
  const progEl = root.querySelector('[data-el="progress"]');

  // ---- geometry ----
  function totalNodes() { return state.nInputs + state.chainLength; }
  function cardX(i) {
    const n = totalNodes();
    if (n === 1) return SW / 2 - CARD_W / 2;
    const margin = 40;
    const span = SW - 2 * margin - CARD_W;
    return margin + (i * span) / (n - 1);
  }
  const cardCx = (i) => cardX(i) + CARD_W / 2;

  function rebuild() {
    stopPlay();
    ctor = new AbstractConstructor({
      vocabSize: state.vocabSize,
      nDims: state.nDims,
      hLayers: 1,
      activation: state.activation,
      seed: state.seed,
    });
    adjList = generateAdjList(
      { nInputs: state.nInputs, nParents: state.nParents, chainLength: state.chainLength },
      new RNG(state.seed + 1)
    );
    example = ctor.generateExample(adjList, state.nInputs);
    nodeEmb = [];
    example.inputTokens.forEach((t) => nodeEmb.push(ctor.embedding(t)));
    example.chainTokens.forEach((t) => nodeEmb.push(ctor.embedding(t)));
    state.step = 0;
    drawGraph();
    render();
  }

  // ---- H transformation shown on the active edges: E[parent] --H--> H(E[parent]) ----
  function drawMiniSpark(g, vec, x, y, w, h, cls) {
    const maxAbs = Math.max(1e-6, ...vec.map((v) => Math.abs(v)));
    const bw = w / vec.length;
    vec.forEach((v, k) => {
      const bh = Math.max(0.8, (Math.abs(v) / maxAbs) * h);
      g.append("rect").attr("class", cls).attr("x", x + k * bw).attr("y", y + (h - bh)).attr("width", Math.max(0.5, bw - 0.2)).attr("height", bh);
    });
  }

  // Stacked just above the active node — one labeled box per parent edge,
  // E[parent] --H--> H(E[parent]) — so the boxes never overlap.
  function drawHTransforms(detail, activeIdx) {
    gHtransform.selectAll("*").remove();
    if (!detail || activeIdx < 0) return;
    const boxW = 104, boxH = 19, gap = 3;
    const M = detail.parentIndices.length;
    const cx = Math.max(boxW / 2 + 4, Math.min(SW - boxW / 2 - 4, cardCx(activeIdx)));
    gHtransform.append("line").attr("x1", cx).attr("y1", ROW_TOP - 12).attr("x2", cx).attr("y2", ROW_TOP).attr("stroke", "#c9ccd1").attr("stroke-width", 1);
    detail.parentIndices.forEach((pIdx, i) => {
      const top = ROW_TOP - 12 - boxH - (M - 1 - i) * (boxH + gap);
      const g = gHtransform.append("g").attr("transform", `translate(${cx - boxW / 2}, ${top})`);
      g.append("rect").attr("class", "ht-box").attr("x", 0).attr("y", 0).attr("width", boxW).attr("height", boxH).attr("rx", 3);
      g.append("text").attr("x", 5).attr("y", boxH / 2).attr("dy", "0.32em").attr("font-size", "8.5px").attr("font-family", "var(--mono)").attr("fill", "#475569").text(nodeLabel(pIdx, state.nInputs));
      drawMiniSpark(g, detail.parentEmbeddings[i], 24, 3, 13, 13, "ht-before");
      g.append("text").attr("class", "ht-arrow").attr("x", 43).attr("y", boxH / 2).attr("dy", "0.32em").attr("text-anchor", "middle").text("→");
      g.append("text").attr("x", 54).attr("y", boxH / 2).attr("dy", "0.32em").attr("text-anchor", "middle").attr("font-size", "9px").attr("font-weight", "700").attr("fill", "#7c3aed").text("H");
      g.append("text").attr("class", "ht-arrow").attr("x", 65).attr("y", boxH / 2).attr("dy", "0.32em").attr("text-anchor", "middle").text("→");
      drawMiniSpark(g, detail.Hout[i], 75, 3, 13, 13, "ht-after");
    });
  }

  // ---- static graph scaffold (built once per rebuild) ----
  let gEdges, gNodes, gPulse, gHtransform;
  function drawGraph() {
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "abs-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "#2563eb");

    gEdges = svg.append("g");
    gNodes = svg.append("g");
    gPulse = svg.append("g");
    gHtransform = svg.append("g");

    // edges parent -> child, arcing above the row
    const edges = [];
    adjList.forEach((parents, c) => parents.forEach((p) => edges.push({ source: p, target: state.nInputs + c, child: c })));
    gEdges
      .selectAll("path")
      .data(edges)
      .join("path")
      .attr("class", "dag2-edge")
      .attr("d", (e) => edgePath(e));

    // node cards
    const nodes = d3.range(totalNodes()).map((i) => ({ i, type: i < state.nInputs ? "input" : "chain" }));
    const cards = gNodes
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("class", (d) => `acard ${d.type}`)
      .attr("transform", (d) => `translate(${cardX(d.i)},${ROW_TOP})`);
    cards.append("rect").attr("width", CARD_W).attr("height", CARD_H).attr("rx", 7);
    cards.append("text").attr("class", "tokid").attr("x", CARD_W / 2).attr("y", 13).attr("text-anchor", "middle");
    cards.append("g").attr("class", "evec").attr("transform", "translate(6,18)");
    cards.append("text").attr("class", "nodename").attr("x", CARD_W / 2).attr("y", CARD_H + 13).attr("text-anchor", "middle").text((d) => nodeLabel(d.i, state.nInputs));

    // section labels
    svg.append("text").attr("class", "seclabel").attr("x", 8).attr("y", 14).text("inputs");
  }

  function edgePath(e) {
    const sx = cardCx(e.source), tx = cardCx(e.target);
    const sy = ROW_TOP, ty = ROW_TOP; // top edge of cards
    const mx = (sx + tx) / 2;
    const arc = Math.min(72, 26 + Math.abs(tx - sx) * 0.4);
    const my = ROW_TOP - arc;
    // trim endpoints to just above card tops so arrowhead is visible
    const ex = tx, ey = ty - 2;
    return `M${sx},${sy - 2} Q${mx},${my} ${ex},${ey}`;
  }
  function edgeApex(e) {
    const sx = cardCx(e.source), tx = cardCx(e.target);
    const mx = (sx + tx) / 2;
    const arc = Math.min(72, 26 + Math.abs(tx - sx) * 0.4);
    return { x: mx, y: ROW_TOP - arc + 6 };
  }

  // small amber embedding sparkbars inside a card's .evec group
  function fillEvec(sel, vec) {
    sel.selectAll("*").remove();
    const innerW = CARD_W - 12;
    const h = 18;
    const maxAbs = Math.max(1e-6, ...vec.map((v) => Math.abs(v)));
    const bw = innerW / vec.length;
    vec.forEach((v, k) => {
      const bh = Math.max(1, (Math.abs(v) / maxAbs) * h);
      sel
        .append("rect")
        .attr("class", v < 0 ? "neg" : "")
        .attr("x", k * bw)
        .attr("y", h - bh)
        .attr("width", Math.max(1, bw - 0.6))
        .attr("height", bh);
    });
  }

  // ---- per-state render ----
  function render(animate = false) {
    const done = state.step >= state.chainLength;
    const activeIdx = done ? -1 : state.nInputs + state.step;

    // node cards: inputs always filled; chain filled up to step; active highlighted; rest hidden
    gNodes.selectAll(".acard").each(function (d) {
      const g = d3.select(this);
      const revealed = d.type === "input" || d.i < state.nInputs + state.step;
      const isActive = d.i === activeIdx;
      const isAnswer = d.i === state.nInputs + state.chainLength - 1 && d.i < state.nInputs + state.step;
      g.classed("active", isActive).classed("answer", isAnswer);
      g.style("opacity", revealed ? 1 : isActive ? 1 : 0.18);
      g.select(".tokid").text(revealed ? tokenOf(d.i) : isActive ? "?" : "");
      const ev = g.select(".evec");
      if (revealed) fillEvec(ev, nodeEmb[d.i]);
      else ev.selectAll("*").remove();
    });

    // edges + H badges: highlight those feeding the active node
    gEdges.selectAll(".dag2-edge")
      .classed("active", (e) => e.target === activeIdx)
      .classed("dim", (e) => activeIdx >= 0 && e.target !== activeIdx && e.target >= state.nInputs + state.step);
    // H transformation shown above the active node
    const detail = done ? null : example.steps[state.step];
    drawHTransforms(detail, activeIdx);

    if (animate && activeIdx >= 0) pulseInto(activeIdx);

    renderDecode(done);
    progEl.textContent = done ? `done · ${state.chainLength}/${state.chainLength} chain tokens` : `computing ${nodeLabel(activeIdx, state.nInputs)} · step ${state.step + 1}/${state.chainLength}`;
    if (done) stopPlay();
  }

  function tokenOf(i) {
    return i < state.nInputs ? example.inputTokens[i] : example.chainTokens[i - state.nInputs];
  }

  // animate embeddings flowing along the active node's incoming edges
  function pulseInto(activeIdx) {
    gEdges.selectAll(".dag2-edge").filter((e) => e.target === activeIdx).each(function () {
      const node = this;
      const len = node.getTotalLength();
      const dot = gPulse.append("circle").attr("class", "prop-dot").attr("r", 4);
      dot
        .transition()
        .duration(620)
        .attrTween("transform", () => (t) => {
          const pt = node.getPointAtLength(t * len);
          return `translate(${pt.x},${pt.y})`;
        })
        .on("end", () => dot.remove());
    });
  }

  // ---- decode panel (the heart: parents -> H -> mean -> σ -> h -> decode argmax) ----
  function htmlVec(vec, cls = "") {
    const maxAbs = Math.max(1e-6, ...vec.map((v) => Math.abs(v)));
    const bars = vec.map((v) => `<span class="bar ${v < 0 ? "neg" : ""}" style="height:${Math.max(1, (Math.abs(v) / maxAbs) * 22)}px"></span>`).join("");
    return `<span class="vecstrip ${cls}">${bars}</span>`;
  }
  function logitsBar(logits, maxIdx) {
    const max = Math.max(...logits), min = Math.min(...logits), span = Math.max(1e-6, max - min);
    return `<div class="logits">${logits.map((l, i) => `<div class="lbar ${i === maxIdx ? "max" : ""}" style="height:${Math.max(1, ((l - min) / span) * 68)}px" title="token ${i}: ${l.toFixed(2)}"></div>`).join("")}</div>`;
  }
  // compact logits: max-pool the vocab into a few bins, highlight the argmax bin
  function compactLogits(logits, maxIdx) {
    const bins = 22, V = logits.length, per = Math.ceil(V / bins);
    const binned = [];
    for (let b = 0; b < bins; b++) {
      let m = -Infinity, hasMax = false;
      for (let k = b * per; k < Math.min(V, (b + 1) * per); k++) { if (logits[k] > m) m = logits[k]; if (k === maxIdx) hasMax = true; }
      binned.push({ v: m, isMax: hasMax });
    }
    const mn = Math.min(...binned.map((x) => x.v)), mx = Math.max(...binned.map((x) => x.v)), span = Math.max(1e-6, mx - mn);
    return `<div class="logits sm">${binned.map((x) => `<div class="lbar ${x.isMax ? "max" : ""}" style="height:${Math.max(2, ((x.v - mn) / span) * 38)}px"></div>`).join("")}</div>`;
  }

  function renderDecode(done) {
    if (done) {
      decodeEl.innerHTML = `<div class="example-block cot"><div class="example-tag cot">Chain complete · decoded output sequence</div>
        <div class="chips">${example.chainTokens.map((t, c) => `<span class="tok chain ${c === state.chainLength - 1 ? "answer" : ""}">${nodeLabel(state.nInputs + c, state.nInputs)} = ${t}</span>`).join("")}</div>
        <p class="muted-note" style="margin-top:0.5rem">Intermediate tokens are the "thoughts"; the last is the <strong>answer</strong>. Every value came from the same hidden (G, E, H).</p></div>`;
      return;
    }
    const s = state.step;
    const detail = example.steps[s];
    const name = nodeLabel(state.nInputs + s, state.nInputs);
    const parentNames = detail.parentIndices.map((p) => nodeLabel(p, state.nInputs)).join(", ");
    const isAns = s === state.chainLength - 1;
    const eVecs = detail.parentIndices.map((p, i) => `<div class="dvec-row"><span class="dvec-lbl">${nodeLabel(p, state.nInputs)}</span>${htmlVec(detail.parentEmbeddings[i], "e")}</div>`).join("");
    const hVecs = detail.parentIndices.map((p, i) => `<div class="dvec-row"><span class="dvec-lbl">${nodeLabel(p, state.nInputs)}</span>${htmlVec(detail.Hout[i], "h")}</div>`).join("");

    decodeEl.innerHTML = `
      <div class="example-tag cot">Decode ${name} · <span class="decode-tag g">G:</span> parents { ${parentNames} }</div>
      <div class="decode-flow">
        <div class="dstep">
          <div class="dstep-label">1 · <span class="decode-tag e">E</span>[parents]</div>
          <div class="dstep-body">${eVecs}</div>
        </div>
        <div class="dop"><span class="arrow">→</span><span class="decode-tag h">H</span>(·)</div>
        <div class="dstep">
          <div class="dstep-label">2 · transform</div>
          <div class="dstep-body">${hVecs}</div>
        </div>
        <div class="dop"><span class="arrow">→</span>average</div>
        <div class="dstep">
          <div class="dstep-label">3 · mean</div>
          <div class="dstep-body">${htmlVec(detail.meanVec, "h")}</div>
        </div>
        <div class="dop"><span class="arrow">→</span>${state.activation}</div>
        <div class="dstep">
          <div class="dstep-label">4 · vector h</div>
          <div class="dstep-body">${htmlVec(detail.actVec, "h")}</div>
        </div>
        <div class="dop"><span class="arrow">→</span><span class="decode-tag e">·Eᵀ</span>, argmax</div>
        <div class="dstep">
          <div class="dstep-label">5 · decode → ${name}</div>
          <div class="dstep-body dstep-decode">
            ${compactLogits(detail.logits, detail.token)}
            <span class="arrow">→</span>
            <span class="tok chain ${isAns ? "answer" : ""}">${detail.token}</span>
          </div>
        </div>
      </div>`;
  }

  // ---- controls ----
  function step() { if (state.step < state.chainLength) { state.step += 1; render(true); } }
  function reset() { stopPlay(); state.step = 0; render(); }
  function play() {
    if (playTimer) { stopPlay(); return; }
    if (state.step >= state.chainLength) reset();
    root.querySelector('[data-act="play"]').textContent = "⏸ Pause";
    render(true);
    playTimer = setInterval(() => {
      if (state.step >= state.chainLength) { stopPlay(); return; }
      step();
    }, 1500);
  }
  function stopPlay() {
    if (playTimer) clearInterval(playTimer);
    playTimer = null;
    const b = root.querySelector('[data-act="play"]');
    if (b) b.textContent = "▶ Play";
  }

  root.querySelector('[data-act="step"]').addEventListener("click", step);
  root.querySelector('[data-act="reset"]').addEventListener("click", reset);
  root.querySelector('[data-act="play"]').addEventListener("click", play);
  root.querySelector('[data-act="regen"]').addEventListener("click", () => { state.seed = randomSeed() % 1000; rebuild(); });
  root.querySelectorAll("[data-c]").forEach((inp) => {
    inp.addEventListener("input", () => {
      state[inp.dataset.c] = inp.type === "range" ? +inp.value : inp.value;
      if (inp.dataset.c === "nInputs") state.nParents = clampNParents(state.nInputs, state.nParents);
      syncControls();
      rebuild();
    });
  });
  root.querySelectorAll("[data-seg]").forEach((seg) => {
    seg.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => { state[seg.dataset.seg] = +b.dataset.val; syncControls(); rebuild(); });
    });
  });

  function syncControls() {
    root.querySelectorAll("[data-v]").forEach((s) => (s.textContent = state[s.dataset.v]));
    root.querySelectorAll("[data-c]").forEach((inp) => { if (inp.dataset.c in state) inp.value = state[inp.dataset.c]; });
    root.querySelectorAll("[data-seg]").forEach((seg) => seg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", +b.dataset.val === state[seg.dataset.seg])));
    // n_parents can be at most n_inputs
    root.querySelector('[data-c="nParents"]').max = Math.min(4, state.nInputs);
  }

  syncControls();
  rebuild();
}
