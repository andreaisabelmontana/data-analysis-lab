// regression.js — ordinary least-squares simple linear regression.
//
// Fits y = slope*x + intercept by minimizing squared residuals, and reports
// the coefficient of determination R². For an exact linear relationship the
// residuals vanish and R² = 1.

import { mean } from './summary.js';

// fit a straight line; returns { slope, intercept, r2 }
export function linearRegression(x, y) {
  if (x.length !== y.length) throw new Error('linearRegression: length mismatch');
  const n = x.length;
  if (n < 2) return { slope: NaN, intercept: NaN, r2: NaN };
  const mx = mean(x), my = mean(y);
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  const slope = sxx === 0 ? NaN : sxy / sxx;
  const intercept = my - slope * mx;
  // R² = explained / total sum of squares; equals 1 when y is constant only
  // if it is also exactly predicted (syy === 0 -> perfect, define r2 = 1).
  let r2;
  if (syy === 0) {
    r2 = 1;
  } else {
    let ssRes = 0;
    for (let i = 0; i < n; i++) { const e = y[i] - (slope * x[i] + intercept); ssRes += e * e; }
    r2 = 1 - ssRes / syy;
  }
  return { slope, intercept, r2 };
}

// predict y for a given x from a fitted model
export const predict = (model, x) => model.slope * x + model.intercept;
