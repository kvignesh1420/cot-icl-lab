// Interactive 3 — the CoT-Recipe.
// The PowerLaw recipe assigns each prompt a cot_example_prob from its INDEX,
// then ALL prompts are shuffled. So alpha sets the overall fraction/mix of CoT
// supervision (~ 1/(alpha+1)), not an early->late curriculum. The widget shows
// both phases: the by-index assignment (a gradient), then the shuffle.

import { powerLawValue, curriculumCurve } from "../recipe.js";
import { RNG } from "../rng.js";

const d3 = window.d3;
const ALPHAS = [
  { label: "0", value: 0 },
  { label: "0.5", value: 0.5 },
  { label: "1", value: 1 },
  { label: "2", value: 2 },
  { label: "∞", value: Infinity },
];
const N_STREAM = 90;
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

export function mountRecipeScheduler(root) {
  const state = { alpha: 2 };
  let runId = 0;

  root.innerHTML = `
    <div class="widget track-shared">
      <div class="widget-head">
        <span class="track-badge">CoT-Recipe</span>
        <span class="widget-title">CoT-Recipe · how much CoT supervision, then shuffled</span>
      </div>
      <div class="controls">
        <div class="control">
          <label>alpha (recipe exponent)</label>
          <div class="segmented" data-seg="alpha"></div>
        </div>
        <div class="control">
          <label>&nbsp;</label>
          <button class="btn btn-accent" data-act="play">▶ Build &amp; shuffle</button>
        </div>
        <div class="control" style="margin-left:auto">
          <label>&nbsp;</label>
          <span class="recipe-frac" data-el="frac"></span>
        </div>
      </div>

      <div class="flex-split">
        <div>
          <svg class="recipe-plot" viewBox="0 0 480 240" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
        <div class="recipe-streams">
          <div class="example-tag">1 · cot_example_prob assigned by prompt index</div>
          <div class="stream compact" data-el="stream1"></div>
          <div class="shuffle-arrow">↓ &nbsp;then <b>shuffle all prompts</b>&nbsp; ↓</div>
          <div class="example-tag">2 · shuffled — the order training actually sees</div>
          <div class="stream compact" data-el="stream2"></div>
          <div class="stream-legend">
            <span><span class="sw cot"></span>CoT-heavy prompt</span>
            <span><span class="sw direct"></span>direct (answer-only)</span>
          </div>
        </div>
      </div>
      <p class="muted-note">The recipe sets each prompt's <code>cot_example_prob</code> from its <em>index</em> (left curve), but every prompt is then <strong>shuffled</strong>. There is no early-to-late curriculum: <strong>α</strong> simply controls the overall <em>fraction</em> of CoT supervision (≈ <code>1/(α+1)</code>). <strong>α = 0</strong> makes every prompt CoT (the always-CoT baseline that can <em>hurt</em> novel-task accuracy); <strong>α = ∞</strong> leaves almost none. Rationing this fraction is the recipe that lifts accuracy by up to 300% (synthetic) / ~130% (pretrained Qwen2.5), even with no CoT in-context at test time.</p>
    </div>`;

  const seg = root.querySelector('[data-seg="alpha"]');
  seg.innerHTML = ALPHAS.map((a) => `<button data-val="${a.value}">${a.label}</button>`).join("");
  const svg = d3.select(root).select("svg");
  const stream1El = root.querySelector('[data-el="stream1"]');
  const stream2El = root.querySelector('[data-el="stream2"]');
  const fracEl = root.querySelector('[data-el="frac"]');

  // fixed per-prompt uniform draws + a fixed shuffle permutation (deterministic)
  const drawRng = new RNG(2024);
  const draws = Array.from({ length: N_STREAM }, () => drawRng.next());
  const permRng = new RNG(99);
  const perm = (() => {
    const a = Array.from({ length: N_STREAM }, (_, i) => i);
    for (let i = N_STREAM - 1; i > 0; i--) { const j = Math.floor(permRng.next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  })();

  // ---- plot ----
  const M = { l: 40, r: 12, t: 12, b: 34 };
  const W = 480, H = 240;
  const x = d3.scaleLinear().domain([0, 1]).range([M.l, W - M.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - M.b, M.t]);
  const line = d3.line().x((d) => x(d.x)).y((d) => y(d.y));

  function drawStaticAxes() {
    svg.selectAll("*").remove();
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - M.b})`).call(d3.axisBottom(x).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(5));
    svg.append("text").attr("class", "axis-label").attr("x", (M.l + W - M.r) / 2).attr("y", H - 4).attr("text-anchor", "middle").text("prompt index");
    svg.append("text").attr("class", "axis-label").attr("transform", `translate(13,${(M.t + H - M.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("cot_example_prob");
    svg.append("path").attr("class", "curve");
    svg.append("line").attr("class", "playhead").attr("y1", M.t).attr("y2", H - M.b).style("opacity", 0);
  }

  const opts = () => ({ nPrompts: N_STREAM - 1, alpha: state.alpha, scale: 1, initialProb: 0, finalProb: 1 });
  const statusOf = (i) => draws[i] < powerLawValue(i, opts());

  function renderPlot() {
    svg.select(".curve").attr("d", line(curriculumCurve(opts(), 200)));
  }
  function setPlayhead(idx) {
    if (idx == null) { svg.select(".playhead").style("opacity", 0); return; }
    const px = x(idx / (N_STREAM - 1));
    svg.select(".playhead").attr("x1", px).attr("x2", px).style("opacity", 1);
  }

  function chipsHTML(order, reveal) {
    return order
      .map((idx, pos) => {
        if (pos >= reveal) return `<span class="ex pending"></span>`;
        return `<span class="ex ${statusOf(idx) ? "cot" : ""}"></span>`;
      })
      .join("");
  }
  const idxOrder = Array.from({ length: N_STREAM }, (_, i) => i);

  function renderStreams(r1, r2) {
    stream1El.innerHTML = chipsHTML(idxOrder, r1 == null ? N_STREAM : r1);
    stream2El.innerHTML = chipsHTML(perm, r2 == null ? N_STREAM : r2);
    const cot = draws.reduce((s, _, i) => s + (statusOf(i) ? 1 : 0), 0);
    fracEl.innerHTML = `≈ <b>${Math.round((cot / N_STREAM) * 100)}%</b> CoT`;
  }

  function renderAll() {
    renderPlot();
    renderStreams(null, null);
    setPlayhead(null);
  }

  async function play() {
    runId += 1;
    const myId = runId;
    const btn = root.querySelector('[data-act="play"]');
    btn.textContent = "▶ Building…";
    // phase 1: assign by prompt index (gradient), playhead sweeps the index axis
    stream2El.innerHTML = chipsHTML(perm, 0);
    for (let i = 1; i <= N_STREAM; i++) {
      if (myId !== runId) return;
      stream1El.innerHTML = chipsHTML(idxOrder, i);
      setPlayhead(i - 1);
      await sleep(22);
    }
    setPlayhead(null);
    await sleep(550);
    if (myId !== runId) return;
    // phase 2: shuffle — reveal the shuffled order
    btn.textContent = "▶ Shuffling…";
    for (let i = 1; i <= N_STREAM; i++) {
      if (myId !== runId) return;
      stream2El.innerHTML = chipsHTML(perm, i);
      await sleep(20);
    }
    btn.textContent = "▶ Build & shuffle";
  }

  seg.querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      runId += 1; // cancel any running animation
      state.alpha = b.dataset.val === "Infinity" ? Infinity : +b.dataset.val;
      seg.querySelectorAll("button").forEach((o) => o.classList.toggle("active", o === b));
      root.querySelector('[data-act="play"]').textContent = "▶ Build & shuffle";
      renderAll();
    });
  });
  root.querySelector('[data-act="play"]').addEventListener("click", play);

  seg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", +b.dataset.val === state.alpha));
  drawStaticAxes();
  renderAll();
}
