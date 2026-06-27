// distributions.js — probability densities, CDFs, inverses and tail
// probabilities for the Normal, Student-t, chi-square and F families.
//
// Extracted verbatim (no DOM, no globals) from the lab demos. All tails use
// standard numerical approximations and require no external libraries:
//   - Normal CDF via Abramowitz & Stegun 7.1.26 (erf), accurate ~1e-7
//   - Student-t / chi-square / F via the regularized incomplete beta/gamma.

// ---- standard normal ----------------------------------------------------
export const normPdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

export function erf(x) {
  // Abramowitz & Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}

export const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

// inverse normal cdf (Acklam / Beasley-Springer-Moro rational approximation)
export function normInv(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pl = 0.02425, ph = 1 - pl;
  let q, r;
  if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > ph) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  q = p - 0.5; r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---- gamma / beta machinery --------------------------------------------
// log-gamma (Lanczos)
export function logGamma(x) {
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// regularized lower incomplete gamma P(s,x)
export function gammaP(s, x) {
  if (x <= 0) return 0;
  if (x < s + 1) {
    let term = 1 / s, sum = term;
    for (let k = 1; k < 200; k++) { term *= x / (s + k); sum += term; if (term < sum * 1e-12) break; }
    return sum * Math.exp(-x + s * Math.log(x) - logGamma(s));
  }
  // continued fraction for Q, then 1-Q
  let b = x + 1 - s, c2 = 1e30, d = 1 / b, h = d;
  for (let i = 1; i < 200; i++) {
    const an = -i * (i - s);
    b += 2; d = an * d + b; if (Math.abs(d) < 1e-30) d = 1e-30;
    c2 = b + an / c2; if (Math.abs(c2) < 1e-30) c2 = 1e-30;
    d = 1 / d; const del = d * c2; h *= del; if (Math.abs(del - 1) < 1e-12) break;
  }
  return 1 - Math.exp(-x + s * Math.log(x) - logGamma(s)) * h;
}

// regularized incomplete beta I_x(a,b)
export function betaCf(x, a, b) {
  const fpmin = 1e-30; let qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < fpmin) d = fpmin; d = 1 / d; let h = d;
  for (let m = 1; m < 200; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < fpmin) d = fpmin; c = 1 + aa / c; if (Math.abs(c) < fpmin) c = fpmin; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < fpmin) d = fpmin; c = 1 + aa / c; if (Math.abs(c) < fpmin) c = fpmin; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-12) break;
  }
  return h;
}

export function betaI(x, a, b) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return bt * betaCf(x, a, b) / a;
  return 1 - bt * betaCf(1 - x, b, a) / b;
}

// ---- tails / inverses / densities --------------------------------------
export const chiSqTail = (x, df) => x <= 0 ? 1 : 1 - gammaP(df / 2, x / 2);          // P(X > x)

export function tTailTwo(t, df) {                                                     // P(|T| > |t|)
  const x = df / (df + t * t);
  return betaI(x, df / 2, 0.5);
}

export const tCdf = (t, df) => t >= 0 ? 1 - 0.5 * tTailTwo(t, df) : 0.5 * tTailTwo(t, df); // P(T <= t)

export const fTail = (f, d1, d2) => f <= 0 ? 1 : betaI(d2 / (d2 + d1 * f), d2 / 2, d1 / 2); // P(F > f)

// inverse Student-t critical value: returns t with P(T <= t) = p (bisection on tCdf)
export function tInv(p, df) {
  if (df > 1e5) return normInv(p);
  let lo = -100, hi = 100;
  for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; (tCdf(m, df) < p ? lo = m : hi = m); }
  return (lo + hi) / 2;
}

export function tPdf(t, df) {
  return Math.exp(logGamma((df + 1) / 2) - logGamma(df / 2)) / Math.sqrt(df * Math.PI) * Math.pow(1 + t * t / df, -(df + 1) / 2);
}
