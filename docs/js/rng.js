// Seeded PRNG + sampling helpers.
//
// These widgets reproduce the *logic* of the Python data pipeline, not numpy's
// exact bit stream. A small seeded generator (mulberry32) makes every
// "Regenerate" reproducible and shareable via a visible seed.

export class RNG {
  constructor(seed = 42) {
    this.seed = seed >>> 0;
    this._state = this.seed >>> 0;
  }

  // mulberry32: tiny, fast, good-enough statistical quality for visualization.
  next() {
    this._state = (this._state + 0x6d2b79f5) >>> 0;
    let t = this._state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Integer in [0, max).
  randint(max) {
    return Math.floor(this.next() * max);
  }

  choice(arr) {
    return arr[this.randint(arr.length)];
  }

  // Sample k distinct items, mirroring np.random.choice(..., replace=False).
  sampleWithoutReplacement(arr, k) {
    const pool = arr.slice();
    const out = [];
    for (let i = 0; i < k && pool.length > 0; i++) {
      const idx = this.randint(pool.length);
      out.push(pool[idx]);
      pool.splice(idx, 1);
    }
    return out;
  }

  // Standard normal via Box–Muller (used for Track A embeddings / FCN weights).
  gaussian(mean = 0, std = 1) {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    return mean + std * mag * Math.cos(2.0 * Math.PI * v);
  }
}

// Convenience: a fresh seed for "Regenerate" buttons.
export function randomSeed() {
  return Math.floor(Math.random() * 1e9);
}
