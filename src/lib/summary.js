// summary.js — univariate summary statistics.
//
// `mean` and `variance` are the helpers the demos use (sample variance, the
// n-1 / Bessel-corrected form). The rest are the standard descriptive family:
// median, mode, quantiles, IQR and 1.5*IQR outlier flagging.
//
// Quantiles use linear interpolation between order statistics (the R-7 /
// NumPy default): q maps to position (n-1)*q in the sorted sample.

// arithmetic mean
export function mean(a) {
  if (!a.length) return NaN;
  let s = 0; for (const x of a) s += x; return s / a.length;
}

// sample variance (divides by n-1); pass a precomputed mean `m` to reuse it
export function variance(a, m) {
  if (a.length < 2) return NaN;
  m = m === undefined ? mean(a) : m;
  let s = 0; for (const x of a) s += (x - m) * (x - m);
  return s / (a.length - 1);
}

// sample standard deviation
export const std = (a, m) => Math.sqrt(variance(a, m));

// quantile via linear interpolation on the sorted sample, q in [0,1]
export function quantile(a, q) {
  if (!a.length) return NaN;
  if (q <= 0) return Math.min(...a);
  if (q >= 1) return Math.max(...a);
  const s = [...a].sort((x, y) => x - y);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  return s[lo] + (pos - lo) * (s[hi] - s[lo]);
}

// median = 50th percentile
export const median = (a) => quantile(a, 0.5);

// most frequent value(s); returns an array (ties preserved, in first-seen order)
export function mode(a) {
  const counts = new Map();
  let best = 0;
  for (const x of a) {
    const c = (counts.get(x) || 0) + 1;
    counts.set(x, c);
    if (c > best) best = c;
  }
  const out = [];
  for (const [v, c] of counts) if (c === best) out.push(v);
  return out;
}

// quartiles { q1, q2, q3 } and the interquartile range
export function quartiles(a) {
  return { q1: quantile(a, 0.25), q2: quantile(a, 0.5), q3: quantile(a, 0.75) };
}

export const iqr = (a) => quantile(a, 0.75) - quantile(a, 0.25);

// Tukey fences: values outside [Q1 - k*IQR, Q3 + k*IQR] are outliers (k=1.5)
export function outlierBounds(a, k = 1.5) {
  const q1 = quantile(a, 0.25), q3 = quantile(a, 0.75);
  const spread = q3 - q1;
  return { lower: q1 - k * spread, upper: q3 + k * spread };
}

// the subset of values flagged as outliers under the 1.5*IQR rule
export function outliers(a, k = 1.5) {
  const { lower, upper } = outlierBounds(a, k);
  return a.filter((x) => x < lower || x > upper);
}

// full descriptive summary in one pass-friendly object
export function summarize(a) {
  const m = mean(a);
  const { q1, q2, q3 } = quartiles(a);
  return {
    n: a.length,
    mean: m,
    median: q2,
    variance: variance(a, m),
    std: std(a, m),
    min: a.length ? Math.min(...a) : NaN,
    max: a.length ? Math.max(...a) : NaN,
    q1, q3,
    iqr: q3 - q1,
    outliers: outliers(a),
  };
}
