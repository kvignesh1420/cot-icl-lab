// Shared CoT-Recipe curriculum.
//
// Faithful port of `PowerLawRecipe.get_value` in
// src/tokenized_cot_icl/core/data/recipes.py:
//
//     x = prompt_index / n_prompts
//     y = min(scale * x**alpha + initial_prob, final_prob)
//
// alpha controls how CoT supervision is rationed across the meta-training stream:
//   alpha = 0    -> constant final_prob (always-CoT, the "excessive" baseline)
//   alpha = inf  -> ~0 until the very end, then jumps to final_prob (CoT only late)

export function powerLawValue(promptIndex, opts) {
  const {
    nPrompts,
    alpha,
    scale = 1.0,
    initialProb = 0.0,
    finalProb = 1.0,
  } = opts;
  const x = nPrompts > 0 ? promptIndex / nPrompts : 0;
  // JS quirk: Math.pow(1, Infinity) === NaN (Python's math.pow returns 1.0).
  // Guard so alpha = Infinity gives 0 below the right edge and final_prob at x = 1.
  const p = Math.pow(x, alpha);
  const pw = Number.isNaN(p) ? (x >= 1 ? 1 : 0) : p;
  return Math.min(scale * pw + initialProb, finalProb);
}

// Sample a whole curriculum curve over [0, nPrompts] for plotting.
export function curriculumCurve(opts, samples = 200) {
  const pts = [];
  for (let i = 0; i <= samples; i++) {
    const promptIndex = (i / samples) * opts.nPrompts;
    pts.push({ x: i / samples, y: powerLawValue(promptIndex, opts) });
  }
  return pts;
}
