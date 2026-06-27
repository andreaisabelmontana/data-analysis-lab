// correlation.js — covariance and the Pearson correlation coefficient.
//
// Sample covariance uses the n-1 divisor, matching the sample variance in
// summary.js. Pearson r is the covariance normalized by the two standard
// deviations and always lies in [-1, 1]; it is +1 / -1 for perfectly
// (anti-)linear data.

import { mean } from './summary.js';

// sample covariance of two equal-length series (divides by n-1)
export function covariance(x, y) {
  if (x.length !== y.length) throw new Error('covariance: length mismatch');
  if (x.length < 2) return NaN;
  const mx = mean(x), my = mean(y);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (x[i] - mx) * (y[i] - my);
  return s / (x.length - 1);
}

// Pearson correlation coefficient in [-1, 1]
export function correlation(x, y) {
  if (x.length !== y.length) throw new Error('correlation: length mismatch');
  if (x.length < 2) return NaN;
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  const denom = Math.sqrt(sxx * syy);
  return denom === 0 ? NaN : sxy / denom;
}
