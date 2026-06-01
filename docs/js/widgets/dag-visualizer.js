// Figure 1 (§1) — the causal structure G.
// An interactive random DAG: inputs on top, chain ("thought") tokens below,
// directed edges parent -> child. Hover a thought to trace its full dependency
// cone and read its reasoning depth; drag nodes; read the adjacency list the
// dataset actually returns. This figure is purely about G; embeddings (E) and the
// token processor (H) enter in §2 and in the top pipeline figure.

import { generateAdjList, clampNParents, nodeLabel } from "../dag.js";
import { RNG, randomSeed } from "../rng.js";

const d3 = window.d3;
const SW = 480;
const SH = 180;
const TOP_Y = 34;
const BOT_Y = 146;
const R = 14;

export function mountDagVisualizer(root) {
  const state = { nInputs: 4, nParents: 2, chainLength: 3, seed: 7, sortParents: false };

  root.innerHTML = `
    <div class="widget track-shared">
      <div class="widget-head">
        <span class="track-badge">causal structure · G</span>
        <span class="widget-title">Random DAG · causal structure G</span>
      </div>
      <div class="controls">
        <div class="control">
          <label>n_inputs · <span class="value-tag" data-v="nInputs"></span></label>
          <input type="range" data-c="nInputs" min="2" max="8" step="1">
        </div>
        <div class="control">
          <label>n_parents · <span class="value-tag" data-v="nParents"></span></label>
          <input type="range" data-c="nParents" min="1" max="8" step="1">
        </div>
        <div class="control">
          <label>chain_length · <span class="value-tag" data-v="chainLength"></span></label>
          <input type="range" data-c="chainLength" min="1" max="6" step="1">
        </div>
        <div class="control">
          <label>&nbsp;</label>
          <button class="btn" data-act="regen">↻ New graph</button>
        </div>
      </div>
      <div class="gee-legend">
        <span class="gee g"><b>G</b> · causal structure (edges)</span>
        <span class="gee e later"><b>E</b> · embeddings <i>· in §2</i></span>
        <span class="gee h later"><b>H</b> · transform <i>· in §2</i></span>
      </div>
      <div class="dag-main">
        <svg class="graph-svg" viewBox="0 0 ${SW} ${SH}" preserveAspectRatio="xMidYMid meet" aria-label="DAG diagram"></svg>
        <div class="dag-table-wrap">
          <div class="example-tag">Adjacency list</div>
          <table class="adj-table dag-adj">
            <thead><tr><th>thought</th><th>parents</th><th>reads as</th></tr></thead>
            <tbody data-el="adjbody"></tbody>
          </table>
        </div>
      </div>
      <div class="dag-readout" data-el="readout"></div>
      <p class="muted-note">Each thought <em>y<sub>i</sub></em> samples <code>n_parents</code> distinct parents from the inputs <em>and</em> all earlier thoughts, so dependencies always point top-to-bottom. Hover a <span style="color:#2563eb;font-weight:600">y</span> node to light up its <b>dependency cone</b> (direct parents in dark blue, deeper ancestors in light blue) and read its <b>reasoning depth</b> above. The last thought is the <b>answer</b>; drag any node to rearrange.</p>
    </div>`;

  const svg = d3.select(root).select("svg");
  const adjBodyEl = root.querySelector('[data-el="adjbody"]');
  const readoutEl = root.querySelector('[data-el="readout"]');

  let nodes = [];

  function syncControls() {
    root.querySelectorAll("[data-v]").forEach((s) => (s.textContent = state[s.dataset.v]));
    root.querySelectorAll("[data-c]").forEach((inp) => {
      const key = inp.dataset.c;
      if (inp.type === "checkbox") inp.checked = state[key];
      else inp.value = state[key];
    });
    root.querySelector('[data-c="nParents"]').max = state.nInputs;
  }

  function computeDepthsAncestors(adjList) {
    const nIn = state.nInputs;
    const depth = new Array(nIn + state.chainLength).fill(0);
    const anc = {};
    for (let i = 0; i < nIn; i++) anc[i] = new Set();
    for (let c = 0; c < state.chainLength; c++) {
      const id = nIn + c;
      const set = new Set();
      let maxd = 0;
      for (const p of adjList[c]) {
        set.add(p);
        for (const a of anc[p] || []) set.add(a);
        maxd = Math.max(maxd, depth[p]);
      }
      anc[id] = set;
      depth[id] = maxd + 1;
    }
    return { depth, anc };
  }

  function readoutHTML(id, depth, anc, answerId) {
    const nIn = state.nInputs;
    const label = nodeLabel(id, nIn);
    if (id < nIn) return `<b>${label}</b> is an input token: a given, not computed.`;
    const ancestors = anc[id] || new Set();
    const inputs = [...ancestors].filter((a) => a < nIn).sort((a, b) => a - b).map((a) => nodeLabel(a, nIn));
    const thoughts = [...ancestors].filter((a) => a >= nIn && a !== id).sort((a, b) => a - b).map((a) => nodeLabel(a, nIn));
    const tag = id === answerId ? `<span class="ans-pill">answer</span>` : "";
    const thoughtClause = thoughts.length
      ? ` and ${thoughts.length} earlier thought${thoughts.length > 1 ? "s" : ""} (${thoughts.join(", ")})`
      : "";
    return `${tag}<b>${label}</b> sits at <b class="depth-num">reasoning depth ${depth[id]}</b>. Following its dependencies all the way back, it draws on <b>${inputs.length} of ${nIn}</b> inputs (${inputs.join(", ") || "none"})${thoughtClause}.`;
  }

  function computePositions() {
    const spread = (n, i) => (n === 1 ? SW / 2 : 42 + (i * (SW - 84)) / (n - 1));
    nodes = [];
    for (let i = 0; i < state.nInputs; i++) {
      nodes.push({ id: i, label: nodeLabel(i, state.nInputs), type: "input", x: spread(state.nInputs, i), y: TOP_Y });
    }
    for (let c = 0; c < state.chainLength; c++) {
      const id = state.nInputs + c;
      nodes.push({ id, label: nodeLabel(id, state.nInputs), type: "chain", x: spread(state.chainLength, c), y: BOT_Y });
    }
  }

  function render() {
    state.nParents = clampNParents(state.nInputs, state.nParents);
    const adjList = generateAdjList(
      { nInputs: state.nInputs, nParents: state.nParents, chainLength: state.chainLength, sortParents: state.sortParents },
      new RNG(state.seed)
    );
    computePositions();
    const edges = [];
    adjList.forEach((parents, c) => parents.forEach((p) => edges.push({ source: p, target: state.nInputs + c })));
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const { depth, anc } = computeDepthsAncestors(adjList);
    const answerId = state.nInputs + state.chainLength - 1;

    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    const addArrow = (id, color) =>
      defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 10).attr("refY", 5)
        .attr("markerWidth", 8).attr("markerHeight", 8).attr("orient", "auto")
        .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", color);
    addArrow("dag-arrow", "#9db8e8");
    addArrow("dag-arrow-hl", "#2563eb");

    const gEdges = svg.append("g");
    const gNodes = svg.append("g");

    function edgePath(e) {
      const s = byId[e.source];
      const t = byId[e.target];
      const sameRow = s.type === "chain" && t.type === "chain";
      const mx = (s.x + t.x) / 2;
      const my = sameRow ? Math.max(s.y, t.y) + 22 : (s.y + t.y) / 2;
      let dx = mx - s.x, dy = my - s.y, L = Math.hypot(dx, dy) || 1;
      const sx = s.x + (dx / L) * R, sy = s.y + (dy / L) * R;
      dx = mx - t.x; dy = my - t.y; L = Math.hypot(dx, dy) || 1;
      const ex = t.x + (dx / L) * R, ey = t.y + (dy / L) * R;
      return `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;
    }

    const edgeSel = gEdges.selectAll("path").data(edges).join("path").attr("class", "edge").attr("d", edgePath);

    const nodeSel = gNodes.selectAll("g").data(nodes, (d) => d.id).join("g")
      .attr("class", (d) => `node ${d.type}`)
      .attr("transform", (d) => `translate(${d.x},${d.y})`);
    nodeSel.append("circle").attr("r", R);
    const txt = nodeSel.append("text").attr("text-anchor", "middle").attr("dy", "0.34em");
    txt.append("tspan").text((d) => d.label[0]);
    txt.append("tspan").attr("baseline-shift", "sub").attr("font-size", "8px").text((d) => d.label.slice(1));
    nodeSel.classed("answer", (d) => d.id === answerId);

    function highlightCone(id) {
      const ancestors = anc[id] || new Set();
      const directParents = new Set(id >= state.nInputs ? adjList[id - state.nInputs] : []);
      edgeSel
        .classed("hl", (e) => e.target === id)
        .classed("anc", (e) => e.target !== id && ancestors.has(e.target))
        .classed("dim", (e) => !(e.target === id || ancestors.has(e.target)));
      nodeSel
        .classed("hl", (n) => n.id === id)
        .classed("parent", (n) => directParents.has(n.id))
        .classed("ancestor", (n) => ancestors.has(n.id) && !directParents.has(n.id))
        .classed("dim", (n) => !(n.id === id || ancestors.has(n.id)));
    }
    function clearCone() {
      edgeSel.classed("hl", false).classed("anc", false).classed("dim", false);
      nodeSel.classed("hl", false).classed("parent", false).classed("ancestor", false).classed("dim", false);
    }

    nodeSel.filter((d) => d.type === "chain").style("cursor", "grab")
      .on("mouseenter", function (event, d) {
        highlightCone(d.id);
        readoutEl.innerHTML = readoutHTML(d.id, depth, anc, answerId);
        highlightRow(d.id - state.nInputs);
      })
      .on("mouseleave", function () {
        clearCone();
        readoutEl.innerHTML = readoutHTML(answerId, depth, anc, answerId);
        highlightRow(-1);
      });

    nodeSel.call(
      d3.drag()
        .on("start", function () { d3.select(this).raise(); })
        .on("drag", function (event, d) {
          d.x = Math.max(14, Math.min(SW - 14, event.x));
          d.y = Math.max(14, Math.min(SH - 14, event.y));
          d3.select(this).attr("transform", `translate(${d.x},${d.y})`);
          edgeSel.attr("d", edgePath);
        })
    );

    renderTables(adjList);
    readoutEl.innerHTML = readoutHTML(answerId, depth, anc, answerId);
  }

  function renderTables(adjList) {
    const last = state.chainLength - 1;
    adjBodyEl.innerHTML = adjList
      .map((parents, c) => {
        const names = parents.map((p) => nodeLabel(p, state.nInputs)).join(", ");
        return `<tr data-row="${c}"><td class="chain-cell${c === last ? " ans" : ""}">y${c + 1}</td><td><code>[${parents.join(", ")}]</code></td><td>{ ${names} }</td></tr>`;
      })
      .join("");
  }

  function highlightRow(c) {
    adjBodyEl.querySelectorAll("tr").forEach((tr) => tr.classList.toggle("hl", +tr.dataset.row === c));
  }

  root.querySelectorAll("[data-c]").forEach((inp) => {
    inp.addEventListener("input", () => {
      const key = inp.dataset.c;
      state[key] = inp.type === "checkbox" ? inp.checked : +inp.value;
      if (key === "nInputs") state.nParents = clampNParents(state.nInputs, state.nParents);
      syncControls();
      render();
    });
  });
  root.querySelector('[data-act="regen"]').addEventListener("click", () => {
    state.seed = randomSeed() % 1000;
    render();
  });

  syncControls();
  render();
}
