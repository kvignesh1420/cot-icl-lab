// Interactive 4 — Hybrid prompt viewer (framing + CoT/direct mixing).
// One K-example ICL prompt where each example is independently CoT or direct,
// controlled by cot_example_prob. Toggle between the two prompt framings:
//   Track A — integer special tokens (hybrid_special_token strategy)
//   Track B — chat template (ChatML)
// Counters mirror num_cot_examples / num_standard_examples in base.py.

import { generateAdjList, nodeLabel } from "../dag.js";
import { AbstractConstructor } from "../token-processor.js";
import { createIclExample, SYSTEM_PROMPT, prepareQuestion, prepareCotSolution, prepareDirectSolution } from "../symbolic.js";
import { RNG, randomSeed } from "../rng.js";

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

export function mountHybridPrompt(root) {
  const state = {
    framing: "A",
    prob: 0.5,
    K: 3,
    chainLength: 3,
    nInputs: 4,
    nParents: 2,
    seed: 5,
  };
  let abExamples = []; // Track A
  let symExamples = []; // Track B
  let draws = []; // per-example uniform draws

  root.innerHTML = `
    <div class="widget track-shared" data-el="shell">
      <div class="widget-head">
        <span class="track-badge" data-el="badge">framing</span>
        <span class="widget-title">Hybrid prompt · mixing CoT &amp; direct examples</span>
      </div>
      <div class="controls">
        <div class="control">
          <label>prompt framing</label>
          <div class="segmented" data-seg="framing">
            <button data-val="A">Tokenized framing</button>
            <button data-val="B">Chat framing</button>
          </div>
        </div>
        <div class="control">
          <label>cot_example_prob · <span class="value-tag" data-v="prob"></span></label>
          <input type="range" data-c="prob" min="0" max="1" step="0.05">
        </div>
        <div class="control">
          <label>K (examples)</label>
          <div class="stepper" data-step="K"><button data-d="-1">−</button><span class="num" data-v="K"></span><button data-d="1">+</button></div>
        </div>
        <div class="control">
          <label>&nbsp;</label>
          <button class="btn" data-act="resample">↻ Resample</button>
        </div>
      </div>

      <div class="counters">
        <span class="counter cot"><b data-el="ncot"></b> CoT</span>
        <span class="counter direct"><b data-el="nstd"></b> direct</span>
      </div>

      <div data-el="body" style="margin-top:0.8rem"></div>

      <p class="muted-note">Move <code>cot_example_prob</code> and watch the same prompt fill with CoT vs direct examples, then flip the framing to see the very same mixture expressed as integer special tokens or as a chat transcript. The CoT-Recipe (next) schedules this probability across training.</p>
    </div>`;

  const bodyEl = root.querySelector('[data-el="body"]');
  const shell = root.querySelector('[data-el="shell"]');
  const badge = root.querySelector('[data-el="badge"]');

  function rebuild() {
    const adjList = generateAdjList(
      { nInputs: state.nInputs, nParents: state.nParents, chainLength: state.chainLength },
      new RNG(state.seed)
    );
    const adjSorted = generateAdjList(
      { nInputs: state.nInputs, nParents: state.nParents, chainLength: state.chainLength, sortParents: true },
      new RNG(state.seed)
    );
    const ctor = new AbstractConstructor({ vocabSize: 64, nDims: 10, seed: state.seed + 9 });
    const symRng = new RNG(state.seed + 21);
    const drawRng = new RNG(state.seed + 77);

    abExamples = [];
    symExamples = [];
    draws = [];
    for (let i = 0; i < state.K; i++) {
      abExamples.push(ctor.generateExample(adjList, state.nInputs));
      symExamples.push(
        createIclExample(
          { wordLength: 8, numWords: state.nInputs, adjList: adjSorted, charOffset: 1 },
          symRng
        )
      );
      draws.push(drawRng.next());
    }
    render();
  }

  function cotFlags() {
    return draws.map((u) => u < state.prob);
  }

  function trackBExampleHTML(ex, isCot) {
    const turns = [
      { role: "user", content: prepareQuestion(ex.inputWords) },
      { role: "assistant", content: isCot ? prepareCotSolution(ex.intermediateWords, ex.answerWord) : prepareDirectSolution(ex.answerWord) },
    ];
    return `<div class="example-block ${isCot ? "cot" : "direct"}">
        <div class="example-tag ${isCot ? "cot" : ""}">${isCot ? "CoT example" : "direct example"}</div>
        <div class="chatml">${chatmlHTML(turns)}</div>
      </div>`;
  }

  function trackAExampleHTML(ex, isCot) {
    const ids = ex.inputTokens.map((t) => `<span class="tok input">${t}</span>`).join(" ");
    const intermediate = ex.chainTokens.slice(0, -1);
    const answer = ex.chainTokens[ex.chainTokens.length - 1];
    const thinkBlock = isCot
      ? `<span class="sp think">&lt;think_start&gt;</span> ${intermediate
          .map((t) => `<span class="tok chain">${t}</span>`)
          .join(" ")} <span class="sp think">&lt;think_end&gt;</span>`
      : "";
    return `<div class="example-block ${isCot ? "cot" : "direct"}">
        <div class="example-tag ${isCot ? "cot" : ""}">${isCot ? "CoT example" : "direct example"}</div>
        <div class="frame">
          <span class="sp inp">&lt;input_start&gt;</span> ${ids} <span class="sp inp">&lt;input_end&gt;</span>
          ${thinkBlock}
          <span class="sp ans">&lt;answer_start&gt;</span> <span class="tok chain answer">${answer}</span> <span class="sp ans">&lt;answer_end&gt;</span>
          <span class="sp eos">&lt;eos&gt;</span>
        </div>
      </div>`;
  }

  function render() {
    const flags = cotFlags();
    root.querySelector('[data-el="ncot"]').textContent = flags.filter(Boolean).length;
    root.querySelector('[data-el="nstd"]').textContent = flags.filter((f) => !f).length;

    badge.textContent = state.framing === "A" ? "integer special tokens" : "chat transcript";

    if (state.framing === "A") {
      bodyEl.innerHTML =
        `<div class="example-tag">Concatenated token sequence (one block per example)</div>` +
        abExamples.map((ex, i) => trackAExampleHTML(ex, flags[i])).join("");
    } else {
      const sys = `<div class="example-block" style="border-left:3px solid var(--faint)"><div class="example-tag">system prompt</div><div class="chatml">${chatmlHTML([{ role: "system", content: SYSTEM_PROMPT }])}</div></div>`;
      bodyEl.innerHTML =
        `<div class="example-tag">Chat transcript (one block per example)</div>` +
        sys +
        symExamples.map((ex, i) => trackBExampleHTML(ex, flags[i])).join("");
    }
  }

  // wiring
  root.querySelectorAll("[data-seg]").forEach((seg) => {
    seg.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        state[seg.dataset.seg] = b.dataset.val;
        syncControls();
        render();
      });
    });
  });
  root.querySelector('[data-c="prob"]').addEventListener("input", (e) => {
    state.prob = +e.target.value;
    syncControls();
    render();
  });
  root.querySelector('[data-step="K"]').addEventListener("click", (e) => {
    const d = e.target.dataset.d;
    if (!d) return;
    state.K = Math.max(2, Math.min(12, state.K + +d));
    syncControls();
    rebuild();
  });
  root.querySelector('[data-act="resample"]').addEventListener("click", () => {
    state.seed = randomSeed() % 1000;
    rebuild();
  });

  function syncControls() {
    root.querySelector('[data-v="prob"]').textContent = state.prob.toFixed(2);
    root.querySelector('[data-v="K"]').textContent = state.K;
    root.querySelector('[data-c="prob"]').value = state.prob;
    root.querySelectorAll("[data-seg] button").forEach((b) =>
      b.classList.toggle("active", b.dataset.val === state.framing)
    );
  }

  syncControls();
  rebuild();
}
