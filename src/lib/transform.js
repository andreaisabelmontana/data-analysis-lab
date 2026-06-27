// transform.js — feature scaling and binning.
//
// minMax rescales to [0,1]; zScore standardizes to zero mean and unit
// (sample) variance; binning buckets values into equal-width intervals. The
// histogram demos in the lab bucket values the same way: bin index =
// floor((v - lo) / (hi - lo) * bins), clamped to the valid range.

import { mean, std } from './summary.js';

// rescale to [0,1]; a constant series maps to all zeros
export function minMax(a) {
  if (!a.length) return [];
  const lo = Math.min(...a), hi = Math.max(...a), range = hi - lo;
  return a.map((x) => (range === 0 ? 0 : (x - lo) / range));
}

// standardize to mean 0, sample std 1; a constant series maps to all zeros
export function zScore(a) {
  if (!a.length) return [];
  const m = mean(a), s = std(a);
  return a.map((x) => (s === 0 || !Number.isFinite(s) ? 0 : (x - m) / s));
}

// assign one value to an equal-width bin index in [0, bins-1]
export function binIndex(v, lo, hi, bins) {
  if (hi <= lo) return 0;
  let b = Math.floor((v - lo) / (hi - lo) * bins);
  if (b < 0) b = 0;
  if (b >= bins) b = bins - 1;
  return b;
}

// equal-width histogram: returns counts per bin plus the bin edges.
// Range defaults to [min, max] of the data.
export function histogram(a, bins, lo, hi) {
  lo = lo === undefined ? Math.min(...a) : lo;
  hi = hi === undefined ? Math.max(...a) : hi;
  const counts = new Array(bins).fill(0);
  for (const v of a) counts[binIndex(v, lo, hi, bins)]++;
  const edges = [];
  for (let i = 0; i <= bins; i++) edges.push(lo + (i / bins) * (hi - lo));
  return { counts, edges };
}
