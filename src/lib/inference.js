// inference.js — the test statistics the lab demos compute, as pure functions.
//
// Each routine mirrors the math drawn on the canvases: confidence-interval
// half-widths, one- and two-sample t/z tests, the paired t, one-way ANOVA's
// F = MSB/MSE, and the chi-square test of independence. Tail probabilities
// come from distributions.js.

import { mean, variance } from './summary.js';
import { normInv, normCdf, tInv, tTailTwo, fTail, chiSqTail } from './distributions.js';

// half-width of a mean CI: t_{alpha/2, n-1} * s / sqrt(n)
export function ciHalfWidth(s, n, conf) {
  const tc = tInv(1 - (1 - conf) / 2, Math.max(1, n - 1));
  return tc * s / Math.sqrt(n);
}

// one-sample z or t test of H0: mean = mu0. `kind` is 'z' or 't'; `alt` is
// 'two' | 'gt' | 'lt'. Returns the statistic, p-value and critical value.
export function oneSampleTest({ diff, s, n, alpha = 0.05, kind = 'z', alt = 'two' }) {
  const isZ = kind === 'z';
  const df = n - 1;
  const stat = diff / (s / Math.sqrt(n));
  const tailGt = isZ
    ? (x) => 1 - normCdf(x)
    : (x) => (x >= 0 ? 0.5 * tTailTwo(x, df) : 1 - 0.5 * tTailTwo(x, df));
  let crit, pval;
  if (alt === 'two') {
    crit = isZ ? normInv(1 - alpha / 2) : tInv(1 - alpha / 2, df);
    pval = isZ ? 2 * (1 - normCdf(Math.abs(stat))) : tTailTwo(stat, df);
  } else if (alt === 'gt') {
    crit = isZ ? normInv(1 - alpha) : tInv(1 - alpha, df);
    pval = tailGt(stat);
  } else {
    crit = isZ ? normInv(1 - alpha) : tInv(1 - alpha, df);
    pval = 1 - tailGt(stat);
  }
  return { stat, crit, pval, df, reject: pval < alpha };
}

// two-sample t-test, pooled (equal-variance) and Welch (Satterthwaite df).
// Inputs are the two groups' (mean, sd, n).
export function twoSampleTest(a, b) {
  const va = a.sd * a.sd, vb = b.sd * b.sd;
  const sp2 = ((a.n - 1) * va + (b.n - 1) * vb) / (a.n + b.n - 2);
  const tPool = (a.mean - b.mean) / Math.sqrt(sp2 * (1 / a.n + 1 / b.n));
  const dfPool = a.n + b.n - 2;
  const seW = Math.sqrt(va / a.n + vb / b.n);
  const tWelch = (a.mean - b.mean) / seW;
  const dfWelch = Math.pow(va / a.n + vb / b.n, 2) /
    (Math.pow(va / a.n, 2) / (a.n - 1) + Math.pow(vb / b.n, 2) / (b.n - 1));
  return { tPool, dfPool, tWelch, dfWelch, pWelch: tTailTwo(tWelch, dfWelch) };
}

// paired t-test on the within-pair differences after[i] - before[i]
export function pairedTest(before, after) {
  if (before.length !== after.length) throw new Error('pairedTest: length mismatch');
  const n = before.length;
  const diffs = before.map((b, i) => after[i] - b);
  const md = mean(diffs), sd = Math.sqrt(variance(diffs, md));
  const t = md / (sd / Math.sqrt(n));
  return { t, df: n - 1, p: tTailTwo(t, n - 1), meanDiff: md };
}

// one-way ANOVA over an array of numeric groups. F = MSB / MSE.
export function anova(groups) {
  const all = [].concat(...groups);
  const grand = mean(all);
  const groupMeans = groups.map((g) => mean(g));
  let ssb = 0, sse = 0;
  groups.forEach((g, i) => {
    ssb += g.length * (groupMeans[i] - grand) ** 2;
    g.forEach((x) => { sse += (x - groupMeans[i]) ** 2; });
  });
  const dfB = groups.length - 1, dfW = all.length - groups.length;
  const msb = ssb / dfB, mse = sse / dfW;
  const F = msb / mse;
  return { F, dfB, dfW, msb, mse, p: fTail(F, dfB, dfW) };
}

// chi-square test of independence on a contingency table of observed counts.
// Returns the statistic, df, expected table and p-value.
export function chiSquareIndependence(observed) {
  const rows = observed.length, cols = observed[0].length;
  const rowSum = observed.map((r) => r.reduce((a, b) => a + b, 0));
  const colSum = [];
  for (let j = 0; j < cols; j++) { let s = 0; for (let i = 0; i < rows; i++) s += observed[i][j]; colSum.push(s); }
  const total = rowSum.reduce((a, b) => a + b, 0);
  const expected = [];
  let chi = 0;
  for (let i = 0; i < rows; i++) {
    expected.push([]);
    for (let j = 0; j < cols; j++) {
      const e = total ? rowSum[i] * colSum[j] / total : 0;
      expected[i].push(e);
      if (e > 0) chi += (observed[i][j] - e) ** 2 / e;
    }
  }
  const df = (rows - 1) * (cols - 1);
  return { chi, df, expected, p: chiSqTail(chi, df) };
}
