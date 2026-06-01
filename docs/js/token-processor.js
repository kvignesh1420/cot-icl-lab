// Track A — abstract / tokenized construction.
//
// Faithful port of:
//   - src/tokenized_cot_icl/core/data/token_processor.py (TokenProcessor, activations)
//   - src/tokenized_cot_icl/core/data/data.py (get_cached_embeddings, get_output_token,
//     _generate_example)
//
// Tokens are abstract integer ids. Each chain token is produced by an opaque,
// controllable function H over fixed Gaussian embeddings, then projected back to
// the vocabulary by argmax. This is exactly the data the custom Llama models train on.

import { RNG } from "./rng.js";

// --- activations (token_processor.get_activation_fn) ---
export const ACTIVATIONS = {
  identity: (x) => x,
  relu: (x) => Math.max(0, x),
  leaky_relu: (x) => (x >= 0 ? x : 0.5 * x), // nn.LeakyReLU(0.5)
  silu: (x) => x / (1 + Math.exp(-x)),
};

function applyActivationVec(vec, fn) {
  return vec.map(fn);
}

// y = x @ W^T  (nn.Linear(d, d, bias=False)); W is d-by-d.
function linear(xRow, W) {
  const d = W.length;
  const y = new Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    let s = 0;
    const Wi = W[i];
    for (let j = 0; j < xRow.length; j++) s += xRow[j] * Wi[j];
    y[i] = s;
  }
  return y;
}

function randomMatrix(rows, cols, rng, std = 1) {
  const M = [];
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols);
    for (let j = 0; j < cols; j++) row[j] = rng.gaussian(0, std);
    M.push(row);
  }
  return M;
}

function argmaxSkippingReserved(logits, reserved) {
  let best = -Infinity;
  let bestIdx = 0;
  for (let v = 0; v < logits.length; v++) {
    if (reserved.has(v)) continue;
    if (logits[v] > best) {
      best = logits[v];
      bestIdx = v;
    }
  }
  return bestIdx;
}

export class AbstractConstructor {
  constructor({
    vocabSize = 128,
    nDims = 10,
    hLayers = 1,
    activation = "leaky_relu",
    std = 1.0, // data_initializer_range
    seed = 42,
    reserved = [],
  }) {
    this.vocabSize = vocabSize;
    this.nDims = nDims;
    this.hLayers = hLayers;
    this.activation = activation;
    this.std = std;
    this.reserved = new Set(reserved);
    this.rng = new RNG(seed);

    // Fixed embedding table E ~ N(0, std), shape V x d (get_cached_embeddings).
    this.E = randomMatrix(vocabSize, nDims, this.rng, std);
  }

  // One TokenProcessor = hLayers Gaussian d-by-d linear layers (no bias).
  _newProcessor() {
    const layers = [];
    for (let i = 0; i < this.hLayers; i++) {
      layers.push(randomMatrix(this.nDims, this.nDims, this.rng, 1.0));
    }
    return layers;
  }

  // TokenProcessor.forward: activation on all but the last layer.
  _runProcessor(layers, xRow) {
    let x = xRow;
    for (let i = 0; i < layers.length - 1; i++) {
      x = applyActivationVec(linear(x, layers[i]), ACTIVATIONS[this.activation]);
    }
    return linear(x, layers[layers.length - 1]);
  }

  _embed(token) {
    return this.E[token];
  }

  // Public accessor: the embedding vector E[token] that a token propagates as.
  embedding(token) {
    return this.E[token];
  }

  // Project a d-vector to vocab logits: logits = h @ E^T.
  _logits(h) {
    const logits = new Array(this.vocabSize).fill(0);
    for (let v = 0; v < this.vocabSize; v++) {
      let s = 0;
      const ev = this.E[v];
      for (let k = 0; k < this.nDims; k++) s += h[k] * ev[k];
      logits[v] = s;
    }
    return logits;
  }

  // Port of data.py get_output_token, returning intermediate tensors for display.
  _stepDetail(layers, availableTokens, parentIndices) {
    const parentTokens = parentIndices.map((idx) => availableTokens[idx]);
    const parentEmbeddings = parentTokens.map((t) => this._embed(t));
    const Hout = parentEmbeddings.map((emb) => this._runProcessor(layers, emb));

    // mean over the M parent rows
    const meanVec = new Array(this.nDims).fill(0);
    for (const row of Hout) for (let k = 0; k < this.nDims; k++) meanVec[k] += row[k];
    for (let k = 0; k < this.nDims; k++) meanVec[k] /= Hout.length;

    const actVec = applyActivationVec(meanVec, ACTIVATIONS[this.activation]);
    const logits = this._logits(actVec);
    const token = argmaxSkippingReserved(logits, this.reserved);

    return { parentIndices, parentTokens, parentEmbeddings, Hout, meanVec, actVec, logits, token };
  }

  // Generate a full example: random inputs then chain tokens via the DAG.
  // Mirrors _generate_example (one processor per chain index).
  generateExample(adjList, nInputs) {
    const effectiveVocab = this.vocabSize - this.reserved.size;
    const inputTokens = [];
    for (let i = 0; i < nInputs; i++) inputTokens.push(this.rng.randint(effectiveVocab));

    const available = inputTokens.slice();
    const chainTokens = [];
    const steps = [];

    adjList.forEach((parentIndices) => {
      const layers = this._newProcessor();
      const detail = this._stepDetail(layers, available, parentIndices);
      available.push(detail.token);
      chainTokens.push(detail.token);
      steps.push(detail);
    });

    return { inputTokens, chainTokens, adjList, steps };
  }
}
