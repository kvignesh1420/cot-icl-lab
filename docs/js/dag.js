// Shared causal structure: the random DAG (G).
//
// Faithful port of `RandomDAG.generate_adj_list` in
// src/tokenized_cot_icl/core/data/dag.py.
//
// Both construction tracks (abstract tokens and natural-language symbols) build
// on this same skeleton. The only divergence: the symbolic pipeline sorts each
// parent list (prepare.py: `adj_list = [sorted(p) for p in adj_list]`), exposed
// here via `sortParents`.

import { RNG } from "./rng.js";

export function clampNParents(nInputs, nParents) {
  // Mirrors data.py `_sample_params`: n_parents = min(n_parents, n_inputs).
  return Math.min(nParents, nInputs);
}

// Returns an adjacency list: one array of parent *indices* per chain token.
// Indices refer to positions in [inputs ... earlier-chain-tokens], NOT token ids.
export function generateAdjList({ nInputs, nParents, chainLength, sortParents = false }, rng) {
  const r = rng instanceof RNG ? rng : new RNG(rng ?? 42);
  const m = clampNParents(nInputs, nParents);
  const adjList = [];
  const available = [];
  for (let i = 0; i < nInputs; i++) available.push(i);

  for (let chainIdx = 0; chainIdx < chainLength; chainIdx++) {
    let parents = r.sampleWithoutReplacement(available, m);
    if (sortParents) parents = parents.slice().sort((a, b) => a - b);
    adjList.push(parents);
    // Each new chain token becomes available to later chain tokens.
    available.push(nInputs + chainIdx);
  }
  return adjList;
}

// Human-friendly node label for a position index given nInputs.
export function nodeLabel(idx, nInputs) {
  if (idx < nInputs) return `x${idx + 1}`;
  return `y${idx - nInputs + 1}`;
}
