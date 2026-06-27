// random.js — random number generation with a seedable source.
//
// The lab demos draw from Math.random(); these generators accept an explicit
// source `rng` (a 0..1 supplier) so tests can seed them deterministically.
// `mulberry32` is a small, fast, well-distributed PRNG; `makeRng(seed)`
// returns a stateful source you can pass to any generator below. When no
// source is given, generators fall back to Math.random — matching the page.

// deterministic uniform(0,1) generator from a 32-bit seed
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// convenience alias: a fresh seeded uniform source
export const makeRng = (seed) => mulberry32(seed);

// standard normal via Box-Muller, drawing from `rng`
export function randn(rng = Math.random) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// exponential(rate=lambda) via inverse-CDF
export const randExp = (lambda, rng = Math.random) => -Math.log(1 - rng()) / lambda;

// gamma(shape=k, scale=1) via Marsaglia-Tsang, k > 0
export function randGamma(k, rng = Math.random) {
  if (k < 1) { const u = rng(); return randGamma(1 + k, rng) * Math.pow(u, 1 / k); }
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = randn(rng); v = 1 + c * x; } while (v <= 0);
    v = v * v * v; const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
