// Figure 2.2 — the same chain, in words.
// Mirrors Figure 2.1: a word-DAG (same skeleton, blue G-edges, propagation),
// the string rule applied at each node, and the chat-template framing below.
// Each chain word = second-half of each parent word, concatenated, then
// Caesar-shifted by char_offset. Parents are sorted (as in prepare.py).

import { generateAdjList, nodeLabel } from "../dag.js";
import { createIclExample, buildConversation, secondHalf } from "../symbolic.js";
import { RNG, randomSeed } from "../rng.js";

const d3 = window.d3;
const SW = 720;
const SH = 150;
const CARD_W = 80;
const CARD_H = 26;
const ROW_TOP = 84;

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function chatmlHTML(conv) {
  return conv
    .map((m) => {
      const c = esc(m.content).replace(/\\boxed\{([^}]*)\}/g, '<span class="boxed">\\boxed{$1}</span>');
      return `<span class="im">&lt;|im_start|&gt;</span><span class="role-${m.role}">${m.role}</span>\n${c}<span class="im">&lt;|im_end|&gt;</span>`;
    })
    .join("\n");
}
function wordHTML(word) {
  const cut = Math.floor(word.length / 2);
  return `<span class="word">${word.slice(0, cut)}<span class="half">${word.slice(cut)}</span></span>`;
}
function fitFont(word) {
  return Math.max(8, Math.min(12.5, (CARD_W - 10) / (word.length * 0.62)));
}

export function mountConstructSymbolic(root) {
  const state = {
    wordLength: 8,
    charOffset: 1,
    chainLength: 3,
    nInputs: 4,
    nParents: 2,
    seed: 3,
    step: 0,
  };
  let example = null;
  let adjList = null;
  let playTimer = null;
  let gEdges, gNodes, gPulse, gOverlay;

  root.innerHTML = `
    <div class="widget track-shared">
      <div class="widget-head">
        <span class="track-badge">the same chain, in words</span>
        <span class="widget-title">Word construction · halves → concat → Caesar shift</span>
      </div>
      <div class="controls">
        <div class="control">
          <label>word_length · <span class="value-tag" data-v="wordLength"></span></label>
          <input type="range" data-c="wordLength" min="4" max="10" step="1">
        </div>
        <div class="control">
          <label>char_offset · <span class="value-tag" data-v="charOffset"></span></label>
          <input type="range" data-c="charOffset" min="0" max="25" step="1">
        </div>
        <div class="control">
          <label>chain_length · <span class="value-tag" data-v="chainLength"></span></label>
          <input type="range" data-c="chainLength" min="1" max="4" step="1">
        </div>
        <div class="control">
          <label>&nbsp;</label>
          <button class="btn" data-act="regen">↻ New words</button>
        </div>
      </div>

      <div class="abs-top">
        <svg class="prop-svg" viewBox="0 0 ${SW} ${SH}" preserveAspectRatio="xMidYMid meet" aria-label="Word propagation through the DAG"></svg>
        <div class="chat-panel">
          <div class="example-tag" style="color:#0f766e">Prompt framing · ChatML</div>
          <div class="chatml scrollbox" data-el="chat"></div>
        </div>
      </div>
      <div class="muted-note" style="margin:0.4rem 0 0">Each node holds a <b style="color:#0f766e">word</b>; edges are the same causal structure <b style="color:#2563eb">G</b> as Figure 2.1. The string rule below is applied to the parent words at each step; the panel on the right shows how the finished example is framed as a chat prompt (with <code>&lt;|im_start|&gt;think</code> and <code>\\boxed{}</code>) for a pretrained LLM.</div>

      <div data-el="stage" style="margin-top:0.7rem"></div>

      <div class="controls" style="margin-top:1rem;border-top:1px solid var(--rule);padding-top:0.9rem">
        <button class="btn" data-act="reset">↺ Reset</button>
        <button class="btn btn-accent" data-act="step">Step ▸</button>
        <button class="btn" data-act="play">▶ Play</button>
        <span class="muted-note" style="margin:0" data-el="progress"></span>
      </div>
      <p class="muted-note">The same construction as Figure 2.1, but every token is now a legible word, so a pretrained LLM (Qwen2.5) can be asked to infer the rule and apply it. (In this word version the parents of each step are <strong>sorted</strong>, making the rule order-independent.)</p>
    </div>`;

  const svg = d3.select(root).select("svg.prop-svg");
  const stageEl = root.querySelector('[data-el="stage"]');
  const chatEl = root.querySelector('[data-el="chat"]');
  const progEl = root.querySelector('[data-el="progress"]');

  // ---- geometry (shared with Fig 2.1) ----
  const totalNodes = () => state.nInputs + state.chainLength;
  function cardX(i) {
    const n = totalNodes();
    if (n === 1) return SW / 2 - CARD_W / 2;
    const margin = 40;
    const span = SW - 2 * margin - CARD_W;
    return margin + (i * span) / (n - 1);
  }
  const cardCx = (i) => cardX(i) + CARD_W / 2;
  const wordOf = (i) => (i < state.nInputs ? example.inputWords[i] : example.chainWords[i - state.nInputs]);

  function edgePath(e) {
    const sx = cardCx(e.source), tx = cardCx(e.target);
    const mx = (sx + tx) / 2;
    const arc = Math.min(58, 22 + Math.abs(tx - sx) * 0.36);
    const my = ROW_TOP - arc;
    return `M${sx},${ROW_TOP - 2} Q${mx},${my} ${tx},${ROW_TOP - 2}`;
  }

  function rebuild() {
    stopPlay();
    adjList = generateAdjList(
      { nInputs: state.nInputs, nParents: state.nParents, chainLength: state.chainLength, sortParents: true },
      new RNG(state.seed)
    );
    example = createIclExample(
      { wordLength: state.wordLength, numWords: state.nInputs, adjList, charOffset: state.charOffset },
      new RNG(state.seed + 100)
    );
    state.step = 0;
    chatEl.innerHTML = chatmlHTML(buildConversation([example], [true]));
    drawGraph();
    render();
  }

  function drawGraph() {
    svg.selectAll("*").remove();
    svg
      .append("defs")
      .append("marker")
      .attr("id", "sym-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "#2563eb");

    gEdges = svg.append("g");
    gNodes = svg.append("g");
    gPulse = svg.append("g");
    gOverlay = svg.append("g");

    const edges = [];
    adjList.forEach((parents, c) => parents.forEach((p) => edges.push({ source: p, target: state.nInputs + c, child: c })));
    gEdges.selectAll("path").data(edges).join("path").attr("class", "sym-edge").attr("d", (e) => edgePath(e));

    const nodes = d3.range(totalNodes()).map((i) => ({ i, type: i < state.nInputs ? "input" : "chain" }));
    const cards = gNodes
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("class", (d) => `wcard ${d.type}`)
      .attr("transform", (d) => `translate(${cardX(d.i)},${ROW_TOP})`);
    cards.append("rect").attr("width", CARD_W).attr("height", CARD_H).attr("rx", 7);
    cards.append("text").attr("class", "wtext").attr("x", CARD_W / 2).attr("y", CARD_H / 2).attr("dy", "0.34em").attr("text-anchor", "middle");
    cards.append("text").attr("class", "wname").attr("x", CARD_W / 2).attr("y", CARD_H + 13).attr("text-anchor", "middle").text((d) => nodeLabel(d.i, state.nInputs));

    svg.append("text").attr("class", "seclabel").attr("x", 8).attr("y", 14).text("inputs");
  }

  function pulseInto(activeIdx) {
    gEdges.selectAll(".sym-edge").filter((e) => e.target === activeIdx).each(function () {
      const node = this;
      const len = node.getTotalLength();
      const dot = gPulse.append("circle").attr("class", "sym-pulse").attr("r", 4);
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

  function render(animate = false) {
    const done = state.step >= state.chainLength;
    const activeIdx = done ? -1 : state.nInputs + state.step;
    const parents = done ? new Set() : new Set(example.steps[state.step].parents);

    gNodes.selectAll(".wcard").each(function (d) {
      const g = d3.select(this);
      const revealed = d.type === "input" || d.i < state.nInputs + state.step;
      const isActive = d.i === activeIdx;
      const isAnswer = d.i === state.nInputs + state.chainLength - 1 && d.i < state.nInputs + state.step;
      g.classed("active", isActive).classed("answer", isAnswer).classed("dim", !revealed && !isActive);
      g.classed("parent-ring", parents.has(d.i));
      const txt = g.select(".wtext");
      if (revealed) {
        const w = wordOf(d.i);
        txt.text(w).attr("font-size", fitFont(w));
      } else {
        txt.text(isActive ? "?" : "").attr("font-size", 12);
      }
      g.select("rect").attr("stroke", parents.has(d.i) ? "#0f766e" : null).attr("stroke-width", parents.has(d.i) ? 2.4 : null);
    });

    gEdges.selectAll(".sym-edge")
      .classed("active", (e) => e.target === activeIdx)
      .classed("dim", (e) => activeIdx >= 0 && e.target !== activeIdx && e.target >= state.nInputs + state.step);

    // rule tag above the active node
    gOverlay.selectAll("*").remove();
    if (!done) {
      const cx = Math.max(70, Math.min(SW - 70, cardCx(activeIdx)));
      gOverlay.append("line").attr("class", "rule-conn").attr("x1", cx).attr("y1", ROW_TOP - 13).attr("x2", cx).attr("y2", ROW_TOP);
      gOverlay.append("text").attr("class", "ruletag").attr("x", cx).attr("y", ROW_TOP - 17).attr("text-anchor", "middle").text(`halves → concat → +${state.charOffset}`);
    }

    if (animate && activeIdx >= 0) pulseInto(activeIdx);

    renderStage(done);
    progEl.textContent = done ? `done · ${state.chainLength}/${state.chainLength} chain words` : `step ${state.step + 1}/${state.chainLength}`;
    if (done) stopPlay();
  }

  function renderStage(done) {
    if (done) {
      stageEl.innerHTML = `<div class="example-block" style="border-left:3px solid #0f766e">
        <div class="example-tag" style="color:#0f766e">Chain complete</div>
        <div class="chips">${example.chainWords
          .map((w, c) => `<span class="word ${c === state.chainLength - 1 ? "answer" : "chain"}">${nodeLabel(state.nInputs + c, state.nInputs)} = ${w}</span>`)
          .join("")}</div>
        <p class="muted-note" style="margin-top:0.5rem">Intermediate words are the "thoughts"; the final word is <code>\\boxed{${example.answerWord}}</code>.</p></div>`;
      return;
    }
    const s = state.step;
    const detail = example.steps[s];
    const name = nodeLabel(state.nInputs + s, state.nInputs);
    const parentNames = detail.parents.map((p) => nodeLabel(p, state.nInputs)).join(", ");
    const isAns = s === state.chainLength - 1;
    const parentWordsHTML = detail.parentWords.map((w, i) => `<div class="dvec-row"><span class="dvec-lbl">${nodeLabel(detail.parents[i], state.nInputs)}</span>${wordHTML(w)}</div>`).join("");
    const halvesHTML = detail.parentWords.map((w) => `<div class="dvec-row"><span class="word chain">${secondHalf(w)}</span></div>`).join("");

    stageEl.innerHTML = `
      <div class="example-tag" style="color:#0f766e">Apply the rule at ${name} · parents { ${parentNames} }</div>
      <div class="decode-flow">
        <div class="dstep">
          <div class="dstep-label">1 · parent words</div>
          <div class="dstep-body">${parentWordsHTML}</div>
        </div>
        <div class="dop"><span class="arrow">→</span>second half</div>
        <div class="dstep">
          <div class="dstep-label">2 · keep halves</div>
          <div class="dstep-body">${halvesHTML}</div>
        </div>
        <div class="dop"><span class="arrow">→</span>concat</div>
        <div class="dstep">
          <div class="dstep-label">3 · concatenate</div>
          <div class="dstep-body"><span class="word">${detail.concat}</span></div>
        </div>
        <div class="dop"><span class="arrow">→</span>Caesar +${state.charOffset}</div>
        <div class="dstep">
          <div class="dstep-label">4 · shift letters → ${name}</div>
          <div class="dstep-body"><span class="word ${isAns ? "answer" : "chain"}">${detail.chainWord}</span></div>
        </div>
      </div>`;
  }

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
    }, 1300);
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
      state[inp.dataset.c] = +inp.value;
      syncControls();
      rebuild();
    });
  });

  function syncControls() {
    root.querySelectorAll("[data-v]").forEach((s) => (s.textContent = state[s.dataset.v]));
    root.querySelectorAll("[data-c]").forEach((inp) => (inp.value = state[inp.dataset.c]));
  }

  syncControls();
  rebuild();
}
