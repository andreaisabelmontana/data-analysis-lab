// ============================================================
// data-analysis-lab — 12 visual demos across the inferential-statistics
// syllabus: sampling distributions & the CLT, point estimation (MoM & MLE),
// confidence intervals, one- and two-sample hypothesis tests, error/power,
// one-way ANOVA, and a chi-square test for categorical data.
//
// Every demo follows the same pattern as the rest of the *-lab series:
//   1. read control state through helpers that always return finite values
//   2. compute into a local buffer
//   3. render in a single idempotent `draw()` that resets the transform
//      and clears before drawing, so resizes and rapid input never compound.
//
// All probability tails use standard numerical approximations (no libraries):
//   - Normal CDF via Abramowitz & Stegun 7.1.26 (erf), accurate ~1e-7
//   - Student-t / chi-square / F tails via numerically integrated densities
//     and the regularized incomplete beta/gamma functions.
// ============================================================

// ---------- helpers ------------------------------------------------------
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function n(id, fallback) {
  const el = document.getElementById(id);
  const v = el ? +el.value : NaN;
  return Number.isFinite(v) ? v : fallback;
}
const $ = id => document.getElementById(id);
const setText = (id, t) => { const el = $(id); if (el) el.textContent = t; };
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');

// ---------- palette ------------------------------------------------------
const ACCENT = '#4338CA';
const ACCENT_S = 'rgba(67,56,202,0.16)';
const RULE  = '#E5E5EA';
const RULE_H = '#CDCDD4';
const INK   = '#15151A';
const INK_S = '#4B4B55';
const MUTED = '#8A8A92';
const GOOD  = '#16A34A';
const WARN  = '#F59E0B';
const BAD   = '#DC2626';

function fitCanvas(cv) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = cv.getBoundingClientRect();
  const cssW = Math.max(80, rect.width);
  const cssH = Math.max(80, parseInt(cv.getAttribute('height'), 10) || 280);
  cv.width  = Math.floor(cssW * dpr);
  cv.height = Math.floor(cssH * dpr);
  cv.style.height = cssH + 'px';
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '12px Inter, sans-serif';
  ctx.textBaseline = 'alphabetic';
  return { ctx, w: cssW, h: cssH };
}
// pointer position in CSS pixels relative to canvas
function ptr(cv, ev) {
  const r = cv.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}

// ============================================================
// SHARED STATISTICS — distributions, sampling, tail probabilities
// ============================================================

// standard normal pdf / cdf
const normPdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
function erf(x) {
  // Abramowitz & Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
// inverse normal cdf (Acklam / Beasley-Springer-Moro rational approximation)
function normInv(p) {
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

// log-gamma (Lanczos) and regularized incomplete gamma / beta
function logGamma(x) {
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
// regularized lower incomplete gamma P(s,x)
function gammaP(s, x) {
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
function betaCf(x, a, b) {
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
function betaI(x, a, b) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return bt * betaCf(x, a, b) / a;
  return 1 - bt * betaCf(1 - x, b, a) / b;
}

// two-sided / one-sided tail probabilities
const chiSqTail = (x, df) => x <= 0 ? 1 : 1 - gammaP(df / 2, x / 2);                 // P(X > x)
function tTailTwo(t, df) {                                                            // P(|T| > |t|)
  const x = df / (df + t * t);
  return betaI(x, df / 2, 0.5);
}
const tCdf = (t, df) => t >= 0 ? 1 - 0.5 * tTailTwo(t, df) : 0.5 * tTailTwo(t, df);   // P(T <= t)
const fTail = (f, d1, d2) => f <= 0 ? 1 : betaI(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);  // P(F > f)

// inverse Student-t critical value (two-sided level: returns t with P(T> t)=p), bisection on tCdf
function tInv(p, df) {                                                                // P(T <= t) = p
  if (df > 1e5) return normInv(p);
  let lo = -100, hi = 100;
  for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; (tCdf(m, df) < p ? lo = m : hi = m); }
  return (lo + hi) / 2;
}

// densities for plotting
function tPdf(t, df) {
  return Math.exp(logGamma((df + 1) / 2) - logGamma(df / 2)) / Math.sqrt(df * Math.PI) * Math.pow(1 + t * t / df, -(df + 1) / 2);
}

// random generators
function randn() { // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const randExp = (lambda) => -Math.log(1 - Math.random()) / lambda;
function randGamma(k) { // Marsaglia-Tsang, shape k (scale 1), k>0
  if (k < 1) { const u = Math.random(); return randGamma(1 + k) * Math.pow(u, 1 / k); }
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = randn(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v; const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// sample summaries
function mean(a) { let s = 0; for (const x of a) s += x; return s / a.length; }
function variance(a, m) { m = m === undefined ? mean(a) : m; let s = 0; for (const x of a) s += (x - m) * (x - m); return s / (a.length - 1); }

// ============================================================
// 1. CLT — sampling distribution of the mean
// ============================================================
(function clt() {
  const cv = $('cv-clt'); if (!cv) return;
  let means = [];
  // population: returns one draw, plus theoretical mu & sigma
  function popInfo() {
    switch ($('clt-pop').value) {
      case 'uniform': return { mu: 0.5, sigma: Math.sqrt(1 / 12), draw: () => Math.random(), lo: 0, hi: 1 };
      case 'exp':     return { mu: 1, sigma: 1, draw: () => randExp(1), lo: 0, hi: 5 };
      case 'normal':  return { mu: 0, sigma: 1, draw: () => randn(), lo: -3.5, hi: 3.5 };
      case 'bimodal': return { mu: 0, sigma: Math.sqrt(1 + 4), draw: () => (Math.random() < 0.5 ? -2 : 2) + randn() * 1, lo: -5, hi: 5 };
    }
    return { mu: 0, sigma: 1, draw: () => randn(), lo: -3.5, hi: 3.5 };
  }
  function addSamples(count) {
    const N = n('clt-n', 10), p = popInfo();
    for (let s = 0; s < count; s++) {
      let acc = 0; for (let i = 0; i < N; i++) acc += p.draw();
      means.push(acc / N);
    }
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const N = n('clt-n', 10), p = popInfo();
    setText('clt-nv', N);
    setText('clt-pop-ms', `${fmt(p.mu)}, ${fmt(p.sigma)}`);
    const se = p.sigma / Math.sqrt(N);
    setText('clt-se', fmt(se, 3));
    setText('clt-cnt', means.length);

    const pad = 36, x0 = pad, x1 = w - 12, y0 = 14, y1 = h - 26;
    // x-range centered on mu, +-4 SE of a moderate n so bars stay visible
    const half = Math.max(p.sigma * 1.2, 4 * se);
    const lo = p.mu - half, hi = p.mu + half;
    const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);

    // histogram of means
    const bins = 34;
    const counts = new Array(bins).fill(0);
    means.forEach(m => { const b = Math.floor((m - lo) / (hi - lo) * bins); if (b >= 0 && b < bins) counts[b]++; });
    const maxC = Math.max(1, ...counts);

    // theoretical normal curve N(mu, se^2) scaled to histogram area
    const binW = (hi - lo) / bins;
    ctx.fillStyle = ACCENT_S;
    counts.forEach((c, i) => {
      const bx = X(lo + i * binW), bw = (x1 - x0) / bins;
      const bh = (c / maxC) * (y1 - y0);
      ctx.fillRect(bx + 0.5, y1 - bh, bw - 1, bh);
    });
    // overlay theoretical density (scaled so peak ~ matches when well-sampled)
    if (means.length > 5 && se > 0) {
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.beginPath();
      const peakDens = normPdf(0) / se;
      // expected count at the mode bin = N_total * binW * peakDens
      const scale = (means.length * binW * peakDens) / maxC * (y1 - y0);
      for (let px = x0; px <= x1; px++) {
        const v = lo + (px - x0) / (x1 - x0) * (hi - lo);
        const dens = normPdf((v - p.mu) / se) / se;
        const yy = y1 - (dens / peakDens) * scale;
        px === x0 ? ctx.moveTo(px, yy) : ctx.lineTo(px, yy);
      }
      ctx.stroke();
    }
    // mu line
    ctx.strokeStyle = INK_S; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(p.mu), y0); ctx.lineTo(X(p.mu), y1); ctx.stroke(); ctx.setLineDash([]);
    // axes
    ctx.strokeStyle = RULE_H; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('μ', X(p.mu), y1 + 14);
    ctx.fillText('distribution of the sample mean  X̄', (x0 + x1) / 2, y0 + 4);
    ctx.textAlign = 'left';

    if (means.length) {
      const om = mean(means), os = means.length > 1 ? Math.sqrt(variance(means, om)) : 0;
      setText('clt-obsm', fmt(om, 3));
      setText('clt-obss', fmt(os, 3));
    } else { setText('clt-obsm', '—'); setText('clt-obss', '—'); }
  }
  $('clt-draw1').addEventListener('click', () => { addSamples(1); draw(); });
  $('clt-draw100').addEventListener('click', () => { addSamples(200); draw(); });
  $('clt-reset').addEventListener('click', () => { means = []; draw(); });
  $('clt-n').addEventListener('input', () => { means = []; draw(); });
  $('clt-pop').addEventListener('change', () => { means = []; draw(); });
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 2. MLE — log-likelihood curve with the analytic maximiser
// ============================================================
(function mle() {
  const cv = $('cv-mle'); if (!cv) return;
  let data = [];
  const SIGMA = 1; // known sigma for the normal-mean model
  function resample() {
    const N = n('mle-n', 40);
    data = [];
    if ($('mle-model').value === 'exp') { const lam = 1.5; for (let i = 0; i < N; i++) data.push(randExp(lam)); }
    else { const mu = 2; for (let i = 0; i < N; i++) data.push(mu + randn() * SIGMA); }
  }
  function paramRange() {
    return $('mle-model').value === 'exp' ? { lo: 0.2, hi: 4 } : { lo: -1, hi: 5 };
  }
  function logLik(theta) {
    if ($('mle-model').value === 'exp') {
      if (theta <= 0) return -Infinity;
      let s = 0; for (const x of data) s += Math.log(theta) - theta * x; return s;
    }
    let s = 0; for (const x of data) s += Math.log(normPdf((x - theta) / SIGMA) / SIGMA); return s;
  }
  function mleHat() {
    const m = mean(data);
    return $('mle-model').value === 'exp' ? 1 / m : m;
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const N = n('mle-n', 40);
    if (data.length !== N) resample();
    setText('mle-nv', N);
    const { lo, hi } = paramRange();
    const guess = lo + (n('mle-p', 50) / 100) * (hi - lo);
    setText('mle-pv', fmt(guess, 2));
    const hat = mleHat();
    setText('mle-hat', fmt(hat, 3));
    setText('mle-ll', fmt(logLik(guess), 1));

    const pad = 40, x0 = pad, x1 = w - 14, y0 = 16, y1 = h - 26;
    const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
    // sample the curve
    const pts = [];
    let maxL = -Infinity, minL = Infinity;
    for (let i = 0; i <= 240; i++) { const th = lo + (i / 240) * (hi - lo); const l = logLik(th); pts.push([th, l]); if (l > maxL) maxL = l; if (l < minL && Number.isFinite(l)) minL = l; }
    const Y = l => y1 - (l - minL) / (maxL - minL) * (y1 - y0);

    // curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    pts.forEach(([th, l]) => { if (!Number.isFinite(l)) { started = false; return; } const px = X(th), py = Y(l); started ? ctx.lineTo(px, py) : ctx.moveTo(px, py); started = true; });
    ctx.stroke();
    // MLE marker
    ctx.strokeStyle = GOOD; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(hat), y0); ctx.lineTo(X(hat), y1); ctx.stroke(); ctx.setLineDash([]);
    // guess marker
    ctx.strokeStyle = WARN; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(guess), y0); ctx.lineTo(X(guess), y1); ctx.stroke();
    ctx.fillStyle = WARN; ctx.beginPath(); ctx.arc(X(guess), Y(logLik(guess)), 4, 0, Math.PI * 2); ctx.fill();
    // axis
    ctx.strokeStyle = RULE_H; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText($('mle-model').value === 'exp' ? 'rate λ' : 'mean μ', (x0 + x1) / 2, y1 + 14);
    ctx.fillStyle = GOOD; ctx.fillText('MLE θ̂', X(hat), y0 + 2);
    ctx.fillStyle = WARN; ctx.fillText('your guess', X(guess), y1 - 4);
    ctx.fillStyle = MUTED; ctx.textAlign = 'left'; ctx.fillText('log-likelihood ℓ(θ)', x0, y0 + 2);
  }
  $('mle-p').addEventListener('input', draw);
  $('mle-n').addEventListener('input', () => { resample(); draw(); });
  $('mle-model').addEventListener('change', () => { resample(); draw(); });
  $('mle-resample').addEventListener('click', () => { resample(); draw(); });
  window.addEventListener('resize', draw);
  resample(); draw();
})();

// ============================================================
// 3. METHOD OF MOMENTS vs MLE
// ============================================================
(function momVsMle() {
  const cv = $('cv-mom'); if (!cv) return;
  // true params
  const TRUE = { gamma: 3.0 /* shape k, scale 1 */, exp: 1.5 /* rate */ };
  let data = [];
  let repsMom = [], repsMle = [];
  function model() { return $('mom-model').value; }
  function sampleOne(N) {
    const out = [];
    if (model() === 'gamma') for (let i = 0; i < N; i++) out.push(randGamma(TRUE.gamma));
    else for (let i = 0; i < N; i++) out.push(randExp(TRUE.exp));
    return out;
  }
  // estimators -> return parameter of interest
  function estimates(sample) {
    const m = mean(sample);
    if (model() === 'exp') return { mom: 1 / m, mle: 1 / m };  // coincide
    // gamma scale=1 assumed; estimate shape k
    const v = variance(sample, m);
    const momK = (m * m) / v;                                  // mean=k, var=k (scale1) -> use mean^2/var generally
    // MLE for gamma shape (scale 1): solve ln k - psi(k) = ln(mean of x) - mean of ln x  -> here scale free via Newton
    const logMean = Math.log(m);
    const meanLog = mean(sample.map(x => Math.log(x)));
    const s = logMean - meanLog;                               // >0
    let k = (3 - s + Math.sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s); // Minka init
    for (let it = 0; it < 40; it++) {
      const dpsi = digamma(k), tpsi = trigamma(k);
      const f = Math.log(k) - dpsi - s, fp = 1 / k - tpsi;
      const kn = k - f / fp; if (kn > 0 && Math.abs(kn - k) < 1e-9) { k = kn; break; } if (kn > 0) k = kn;
    }
    return { mom: momK, mle: k };
  }
  function digamma(x) { let r = 0; while (x < 6) { r -= 1 / x; x += 1; } const f = 1 / (x * x); return r + Math.log(x) + 1 / (2 * x) - f * (1 / 12 - f * (1 / 120 - f / 252)); }
  function trigamma(x) { let r = 0; while (x < 6) { r += 1 / (x * x); x += 1; } const f = 1 / (x * x); return r + 1 / x + f / 2 + f / x * (1 / 6 - f * (1 / 30 - f / 42)); }
  function resample() { data = sampleOne(n('mom-n', 30)); }
  function runReps() {
    repsMom = []; repsMle = []; const N = n('mom-n', 30);
    for (let r = 0; r < 300; r++) { const e = estimates(sampleOne(N)); repsMom.push(e.mom); repsMle.push(e.mle); }
  }
  function rmse(arr, truth) { if (!arr.length) return NaN; let s = 0; for (const x of arr) s += (x - truth) * (x - truth); return Math.sqrt(s / arr.length); }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const N = n('mom-n', 30); setText('mom-nv', N);
    if (data.length !== N) resample();
    const truth = model() === 'gamma' ? TRUE.gamma : TRUE.exp;
    const e = estimates(data);
    setText('mom-true', `${model() === 'gamma' ? 'k' : 'λ'} = ${fmt(truth)}`);
    setText('mom-mom', fmt(e.mom, 3));
    setText('mom-mle', fmt(e.mle, 3));
    setText('mom-rmm', repsMom.length ? fmt(rmse(repsMom, truth), 3) : '—');
    setText('mom-rml', repsMle.length ? fmt(rmse(repsMle, truth), 3) : '—');

    // plot: histogram of replicate estimates (MoM faint, MLE accent), true line
    const pad = 36, x0 = pad, x1 = w - 12, y0 = 16, y1 = h - 26;
    const all = repsMom.concat(repsMle);
    const lo = all.length ? Math.min(truth * 0.4, ...all) : truth * 0.4;
    const hi = all.length ? Math.max(truth * 1.8, ...all) : truth * 1.8;
    const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
    const bins = 30; const cm = new Array(bins).fill(0), cl = new Array(bins).fill(0);
    const bin = v => Math.floor((v - lo) / (hi - lo) * bins);
    repsMom.forEach(v => { const b = bin(v); if (b >= 0 && b < bins) cm[b]++; });
    repsMle.forEach(v => { const b = bin(v); if (b >= 0 && b < bins) cl[b]++; });
    const maxC = Math.max(1, ...cm, ...cl);
    const bw = (x1 - x0) / bins;
    for (let i = 0; i < bins; i++) {
      const bx = x0 + i * bw;
      if (cm[i]) { ctx.fillStyle = 'rgba(138,138,146,0.35)'; const bh = cm[i] / maxC * (y1 - y0); ctx.fillRect(bx, y1 - bh, bw - 1, bh); }
      if (cl[i]) { ctx.fillStyle = ACCENT_S; const bh = cl[i] / maxC * (y1 - y0); ctx.fillRect(bx, y1 - bh, bw - 1, bh); }
    }
    ctx.strokeStyle = GOOD; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(truth), y0); ctx.lineTo(X(truth), y1); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = RULE_H; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('estimate value', (x0 + x1) / 2, y1 + 14);
    ctx.fillStyle = GOOD; ctx.fillText('truth', X(truth), y0 + 2);
    ctx.textAlign = 'left';
    ctx.fillStyle = MUTED; ctx.fillText('grey = MoM   ·   indigo = MLE (over replicate samples)', x0, y0 + 2);
  }
  $('mom-n').addEventListener('input', () => { repsMom = []; repsMle = []; resample(); draw(); });
  $('mom-model').addEventListener('change', () => { repsMom = []; repsMle = []; resample(); draw(); });
  $('mom-resample').addEventListener('click', () => { resample(); draw(); });
  $('mom-run').addEventListener('click', () => { runReps(); draw(); });
  window.addEventListener('resize', draw);
  resample(); draw();
})();

// ============================================================
// 4. CONFIDENCE-INTERVAL COVERAGE
// ============================================================
(function coverage() {
  const cv = $('cv-cov'); if (!cv) return;
  const MU = 0, SIGMA = 1;
  let intervals = []; // {lo, hi, hit}
  let total = 0, hit = 0;
  function level() { return n('cov-c', 95) / 100; }
  function addOne() {
    const N = n('cov-n', 15);
    const xs = []; for (let i = 0; i < N; i++) xs.push(MU + randn() * SIGMA);
    const m = mean(xs), s = N > 1 ? Math.sqrt(variance(xs, m)) : SIGMA;
    const tc = tInv(1 - (1 - level()) / 2, N - 1);
    const half = tc * s / Math.sqrt(N);
    const lo = m - half, hi = m + half, h = lo <= MU && MU <= hi;
    intervals.push({ lo, hi, hit: h, m });
    total++; if (h) hit++;
    if (intervals.length > 60) intervals.shift();
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    setText('cov-cv', `${n('cov-c', 95)}%`);
    setText('cov-nv', n('cov-n', 15));
    const pad = 30, x0 = pad, x1 = w - pad, y0 = 14, y1 = h - 22;
    const span = 3.2; // x in [mu-span, mu+span]
    const X = v => (x0 + x1) / 2 + (v - MU) / span * ((x1 - x0) / 2);
    // true mean line
    ctx.strokeStyle = INK_S; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(MU), y0); ctx.lineTo(X(MU), y1); ctx.stroke(); ctx.setLineDash([]);
    const rows = intervals.length || 1;
    const rh = Math.min(16, (y1 - y0) / Math.max(rows, 1));
    intervals.forEach((iv, i) => {
      const yy = y0 + rh * (i + 0.5);
      ctx.strokeStyle = iv.hit ? GOOD : BAD; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(clamp(X(iv.lo), x0, x1), yy); ctx.lineTo(clamp(X(iv.hi), x0, x1), yy); ctx.stroke();
      ctx.fillStyle = iv.hit ? GOOD : BAD; ctx.beginPath(); ctx.arc(clamp(X(iv.m), x0, x1), yy, 1.8, 0, Math.PI * 2); ctx.fill();
    });
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('μ (true mean)', X(MU), y1 + 14);
    ctx.fillText('red intervals miss μ · last 60 shown', (x0 + x1) / 2, y0 + 2);
    ctx.textAlign = 'left';
    setText('cov-tot', total); setText('cov-hit', hit);
    setText('cov-rate', total ? `${(100 * hit / total).toFixed(1)}%` : '—');
  }
  $('cov-1').addEventListener('click', () => { addOne(); draw(); });
  $('cov-50').addEventListener('click', () => { for (let i = 0; i < 50; i++) addOne(); draw(); });
  $('cov-reset').addEventListener('click', () => { intervals = []; total = 0; hit = 0; draw(); });
  $('cov-c').addEventListener('input', () => { intervals = []; total = 0; hit = 0; draw(); });
  $('cov-n').addEventListener('input', () => { intervals = []; total = 0; hit = 0; draw(); });
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 5. CI WIDTH vs SAMPLE SIZE & CONFIDENCE
// ============================================================
(function ciWidth() {
  const cv = $('cv-ciw'); if (!cv) return;
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const conf = n('ciw-c', 95) / 100, s = n('ciw-s', 10) / 10, nHi = 120, nSel = n('ciw-n', 20);
    setText('ciw-cv', `${n('ciw-c', 95)}%`);
    setText('ciw-sv', fmt(s, 1));
    setText('ciw-nv', nSel);
    const halfWidth = (nn) => tInv(1 - (1 - conf) / 2, Math.max(1, nn - 1)) * s / Math.sqrt(nn);

    const pad = 42, x0 = pad, x1 = w - 14, y0 = 16, y1 = h - 26;
    const X = nn => x0 + (nn - 2) / (nHi - 2) * (x1 - x0);
    const maxHW = halfWidth(2);
    const Y = hw => y1 - clamp(hw / maxHW, 0, 1) * (y1 - y0);
    // grid
    ctx.strokeStyle = RULE; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) { const yy = y0 + g / 4 * (y1 - y0); ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke(); }
    // curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.beginPath();
    for (let nn = 2; nn <= nHi; nn++) { const px = X(nn), py = Y(halfWidth(nn)); nn === 2 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
    ctx.stroke();
    // highlight n
    const hw = halfWidth(nSel);
    ctx.strokeStyle = WARN; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(nSel), y0); ctx.lineTo(X(nSel), y1); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = WARN; ctx.beginPath(); ctx.arc(X(nSel), Y(hw), 4, 0, Math.PI * 2); ctx.fill();
    // axes labels
    ctx.strokeStyle = RULE_H; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('sample size n', (x0 + x1) / 2, y1 + 14);
    ctx.save(); ctx.translate(12, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('CI half-width', 0, 0); ctx.restore();
    ctx.textAlign = 'left';

    const tc = tInv(1 - (1 - conf) / 2, Math.max(1, nSel - 1));
    setText('ciw-t', fmt(tc, 3));
    setText('ciw-hw', fmt(hw, 3));
    setText('ciw-quad', `${nSel * 4} (≈4n)`);
  }
  ['ciw-c', 'ciw-s', 'ciw-n'].forEach(id => $(id).addEventListener('input', draw));
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 6. ONE-SAMPLE z / t TEST with rejection region
// ============================================================
(function oneSample() {
  const cv = $('cv-onez'); if (!cv) return;
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const isZ = $('oz-type').value === 'z';
    const alt = $('oz-alt').value;
    const diff = n('oz-d', 6) / 10, s = n('oz-s', 10) / 10, N = n('oz-n', 25), alpha = n('oz-a', 5) / 100;
    setText('oz-dv', fmt(diff, 1)); setText('oz-sv', fmt(s, 1)); setText('oz-nv', N); setText('oz-av', fmt(alpha, 2));
    const df = N - 1;
    const stat = diff / (s / Math.sqrt(N));
    const dens = isZ ? (x => normPdf(x)) : (x => tPdf(x, df));
    const tailGt = isZ ? (x => 1 - normCdf(x)) : (x => x >= 0 ? 0.5 * tTailTwo(x, df) : 1 - 0.5 * tTailTwo(x, df));
    // critical values & p-value
    let crit, pval, critLo, critHi;
    if (alt === 'two') {
      crit = isZ ? normInv(1 - alpha / 2) : tInv(1 - alpha / 2, df);
      critLo = -crit; critHi = crit;
      pval = isZ ? 2 * (1 - normCdf(Math.abs(stat))) : tTailTwo(stat, df);
      setText('oz-crit', `±${fmt(crit, 3)}`);
    } else if (alt === 'gt') {
      crit = isZ ? normInv(1 - alpha) : tInv(1 - alpha, df); critHi = crit;
      pval = tailGt(stat);
      setText('oz-crit', fmt(crit, 3));
    } else {
      crit = isZ ? normInv(1 - alpha) : tInv(1 - alpha, df); critLo = -crit;
      pval = 1 - tailGt(stat);
      setText('oz-crit', fmt(-crit, 3));
    }
    setText('oz-stat', `${isZ ? 'z' : 't'} = ${fmt(stat, 3)}`);
    setText('oz-p', fmt(pval, 4));
    const reject = pval < alpha;
    setText('oz-dec', reject ? 'reject H₀' : 'fail to reject');
    $('oz-dec').style.color = reject ? BAD : GOOD;

    // plot null density on [-4,4]
    const pad = 24, x0 = pad, x1 = w - pad, y0 = 18, y1 = h - 24;
    const lo = -4.2, hi = 4.2;
    const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
    const peak = dens(0);
    const Y = d => y1 - d / peak * (y1 - y0);
    // rejection shading
    const shade = (a, b) => {
      ctx.fillStyle = 'rgba(220,38,38,0.16)'; ctx.beginPath(); ctx.moveTo(X(a), y1);
      for (let v = a; v <= b; v += (b - a) / 60) ctx.lineTo(X(v), Y(dens(v)));
      ctx.lineTo(X(b), y1); ctx.closePath(); ctx.fill();
    };
    if (critHi !== undefined) shade(critHi, hi);
    if (critLo !== undefined) shade(lo, critLo);
    // density curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.beginPath();
    for (let px = x0; px <= x1; px++) { const v = lo + (px - x0) / (x1 - x0) * (hi - lo); const py = Y(dens(v)); px === x0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
    ctx.stroke();
    // statistic line
    const sx = clamp(X(stat), x0, x1);
    ctx.strokeStyle = reject ? BAD : INK; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(sx, y0); ctx.lineTo(sx, y1); ctx.stroke();
    ctx.fillStyle = reject ? BAD : INK; ctx.font = '600 11px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText(`${isZ ? 'z' : 't'}=${fmt(stat, 2)}`, sx, y0 + 2);
    // baseline
    ctx.strokeStyle = RULE_H; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter';
    ctx.fillText('0', X(0), y1 + 14);
    ctx.fillText('red = rejection region (area α)', (x0 + x1) / 2, y1 + 14);
    ctx.textAlign = 'left';
  }
  ['oz-d', 'oz-s', 'oz-n', 'oz-a'].forEach(id => $(id).addEventListener('input', draw));
  ['oz-type', 'oz-alt'].forEach(id => $(id).addEventListener('change', draw));
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 7. p-VALUE as a tail area on the standard normal
// ============================================================
(function pvalue() {
  const cv = $('cv-pval'); if (!cv) return;
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const z = n('pv-z', 196) / 100, alt = $('pv-alt').value;
    setText('pv-zv', fmt(z, 2));
    let p;
    if (alt === 'two') p = 2 * (1 - normCdf(Math.abs(z)));
    else if (alt === 'gt') p = 1 - normCdf(z);
    else p = normCdf(z);
    setText('pv-p', fmt(p, 4));
    setText('pv-sig', p < 0.05 ? 'significant' : 'not sig.'); $('pv-sig').style.color = p < 0.05 ? GOOD : MUTED;
    setText('pv-sig1', p < 0.01 ? 'significant' : 'not sig.'); $('pv-sig1').style.color = p < 0.01 ? GOOD : MUTED;

    const pad = 24, x0 = pad, x1 = w - pad, y0 = 18, y1 = h - 24;
    const lo = -4.2, hi = 4.2;
    const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
    const peak = normPdf(0); const Y = d => y1 - d / peak * (y1 - y0);
    const shade = (a, b) => {
      ctx.fillStyle = ACCENT_S; ctx.beginPath(); ctx.moveTo(X(a), y1);
      for (let v = a; v <= b; v += (b - a) / 80) ctx.lineTo(X(v), Y(normPdf(v)));
      ctx.lineTo(X(b), y1); ctx.closePath(); ctx.fill();
    };
    if (alt === 'two') { shade(Math.abs(z), hi); shade(lo, -Math.abs(z)); }
    else if (alt === 'gt') shade(z, hi);
    else shade(lo, z);
    // curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.beginPath();
    for (let px = x0; px <= x1; px++) { const v = lo + (px - x0) / (x1 - x0) * (hi - lo); const py = Y(normPdf(v)); px === x0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
    ctx.stroke();
    // z line(s)
    ctx.strokeStyle = INK; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(clamp(X(z), x0, x1), y0); ctx.lineTo(clamp(X(z), x0, x1), y1); ctx.stroke();
    ctx.strokeStyle = RULE_H; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = INK; ctx.font = '600 11px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText(`z=${fmt(z, 2)}`, clamp(X(z), x0, x1), y0 + 2);
    ctx.fillStyle = ACCENT; ctx.font = '600 13px Inter';
    ctx.fillText(`p = ${fmt(p, 3)}`, (x0 + x1) / 2, (y0 + y1) / 2);
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.fillText('shaded tail area = p-value', (x0 + x1) / 2, y1 + 14);
    ctx.textAlign = 'left';
  }
  $('pv-z').addEventListener('input', draw);
  $('pv-alt').addEventListener('change', draw);
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 8. TYPE I / II ERROR & POWER
// ============================================================
(function power() {
  const cv = $('cv-power'); if (!cv) return;
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const d = n('pw-d', 5) / 10, N = n('pw-n', 25), alpha = n('pw-a', 5) / 100, two = $('pw-two').checked;
    setText('pw-dv', fmt(d, 1)); setText('pw-nv', N); setText('pw-av', fmt(alpha, 2));
    // distributions of X̄ in SE units: H0 ~ N(0,1), H1 ~ N(mu1,1) where mu1 = d*sqrt(N)
    const mu1 = d * Math.sqrt(N);
    let alphaUsed, beta, powerVal, critHi, critLo;
    if (two) {
      const c = normInv(1 - alpha / 2); critHi = c; critLo = -c;
      alphaUsed = alpha;
      beta = normCdf(c - mu1) - normCdf(-c - mu1);
    } else {
      const c = normInv(1 - alpha); critHi = c;
      alphaUsed = alpha;
      beta = normCdf(c - mu1);
    }
    powerVal = 1 - beta;
    setText('pw-alpha', fmt(alphaUsed, 3));
    setText('pw-beta', fmt(beta, 3));
    setText('pw-power', fmt(powerVal, 3));

    const pad = 22, x0 = pad, x1 = w - pad, y0 = 18, y1 = h - 24;
    const lo = -4.2, hi = Math.max(4.2, mu1 + 4.2);
    const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
    const peak = normPdf(0); const Y = dd => y1 - dd / peak * (y1 - y0);
    const dens0 = v => normPdf(v), dens1 = v => normPdf(v - mu1);
    // beta region (under H1, left of critHi / between crit) — orange
    const shade = (fn, a, b, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(X(a), y1); for (let v = a; v <= b; v += (b - a) / 80) ctx.lineTo(X(v), Y(fn(v))); ctx.lineTo(X(b), y1); ctx.closePath(); ctx.fill(); };
    // Type II (beta): under H1 in the non-rejection region
    if (two) shade(dens1, critLo, critHi, 'rgba(245,158,11,0.22)');
    else shade(dens1, lo, critHi, 'rgba(245,158,11,0.22)');
    // alpha: under H0 in rejection region (red)
    shade(dens0, critHi, hi, 'rgba(220,38,38,0.20)');
    if (two) shade(dens0, lo, critLo, 'rgba(220,38,38,0.20)');
    // power: under H1 in rejection region (green) — drawn lightly
    shade(dens1, critHi, hi, 'rgba(22,163,74,0.18)');
    if (two) shade(dens1, lo, critLo, 'rgba(22,163,74,0.18)');
    // curves
    const curve = (fn, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let px = x0; px <= x1; px++) { const v = lo + (px - x0) / (x1 - x0) * (hi - lo); const py = Y(fn(v)); px === x0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); } ctx.stroke(); };
    curve(dens0, INK_S); curve(dens1, ACCENT);
    // crit lines
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    [critHi, critLo].forEach(c => { if (c !== undefined) { ctx.beginPath(); ctx.moveTo(X(c), y0); ctx.lineTo(X(c), y1); ctx.stroke(); } });
    ctx.setLineDash([]);
    ctx.strokeStyle = RULE_H; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = INK_S; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('H₀', X(0), y0 + 2);
    ctx.fillStyle = ACCENT; ctx.fillText('H₁', X(mu1), y0 + 2);
    ctx.fillStyle = MUTED; ctx.fillText('red α · orange β · green power', (x0 + x1) / 2, y1 + 14);
    ctx.textAlign = 'left';
  }
  ['pw-d', 'pw-n', 'pw-a'].forEach(id => $(id).addEventListener('input', draw));
  $('pw-two').addEventListener('change', draw);
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 9. TWO-SAMPLE t-TEST — pooled vs Welch
// ============================================================
(function twoSample() {
  const cv = $('cv-twot'); if (!cv) return;
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const ma = n('t2-ma', 50), sa = n('t2-sa', 10), na = n('t2-na', 20);
    const mb = n('t2-mb', 56), sb = n('t2-sb', 14), nb = n('t2-nb', 20);
    [['t2-ma-v', ma], ['t2-sa-v', sa], ['t2-na-v', na], ['t2-mb-v', mb], ['t2-sb-v', sb], ['t2-nb-v', nb]].forEach(([id, v]) => setText(id, v));
    const va = sa * sa, vb = sb * sb;
    // pooled
    const sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2);
    const tPool = (ma - mb) / Math.sqrt(sp2 * (1 / na + 1 / nb));
    const dfPool = na + nb - 2;
    // Welch
    const seW = Math.sqrt(va / na + vb / nb);
    const tW = (ma - mb) / seW;
    const dfW = Math.pow(va / na + vb / nb, 2) / (Math.pow(va / na, 2) / (na - 1) + Math.pow(vb / nb, 2) / (nb - 1));
    const pW = tTailTwo(tW, dfW);
    setText('t2-pool', `${fmt(tPool, 3)} (df ${dfPool})`);
    setText('t2-welch', `${fmt(tW, 3)} (df ${fmt(dfW, 1)})`);
    setText('t2-p', fmt(pW, 4));

    // draw two normal curves N(ma,sa^2/na?) -> show population-shaped curves of the data
    const pad = 24, x0 = pad, x1 = w - pad, y0 = 18, y1 = h - 28;
    const lo = Math.min(ma - 3 * sa, mb - 3 * sb), hi = Math.max(ma + 3 * sa, mb + 3 * sb);
    const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
    const pk = Math.max(normPdf(0) / sa, normPdf(0) / sb);
    const Y = dd => y1 - dd / pk * (y1 - y0);
    const curve = (m, s, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let px = x0; px <= x1; px++) { const v = lo + (px - x0) / (x1 - x0) * (hi - lo); const py = Y(normPdf((v - m) / s) / s); px === x0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); } ctx.stroke(); };
    curve(ma, sa, ACCENT); curve(mb, sb, WARN);
    // mean markers
    [[ma, ACCENT, 'A'], [mb, WARN, 'B']].forEach(([m, col, lab]) => {
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(X(m), y0); ctx.lineTo(X(m), y1); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col; ctx.font = '600 11px Inter'; ctx.textAlign = 'center'; ctx.fillText(lab, X(m), y0 + 2);
    });
    ctx.strokeStyle = RULE_H; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('indigo = group A · orange = group B (population shapes)', (x0 + x1) / 2, y1 + 16);
    ctx.textAlign = 'left';
  }
  ['t2-ma', 't2-sa', 't2-na', 't2-mb', 't2-sb', 't2-nb'].forEach(id => $(id).addEventListener('input', draw));
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 10. PAIRED vs UNPAIRED
// ============================================================
(function paired() {
  const cv = $('cv-paired'); if (!cv) return;
  let before = [], after = [];
  function resample() {
    const N = n('pd-n', 15), shift = n('pd-d', 30) / 10, between = n('pd-b', 12), within = n('pd-w', 4);
    before = []; after = [];
    for (let i = 0; i < N; i++) {
      const subj = 50 + randn() * between;   // subject baseline
      before.push(subj + randn() * within);
      after.push(subj + shift + randn() * within);
    }
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const N = n('pd-n', 15);
    setText('pd-dv', fmt(n('pd-d', 30) / 10, 1)); setText('pd-bv', n('pd-b', 12)); setText('pd-wv', n('pd-w', 4)); setText('pd-nv', N);
    if (before.length !== N) resample();
    // paired test on differences
    const diffs = before.map((b, i) => after[i] - b);
    const md = mean(diffs), sd = Math.sqrt(variance(diffs, md));
    const tPaired = md / (sd / Math.sqrt(N));
    const pPaired = tTailTwo(tPaired, N - 1);
    // unpaired (Welch) treating before/after as independent groups
    const mb = mean(before), ma = mean(after);
    const vb = variance(before, mb), vaa = variance(after, ma);
    const seU = Math.sqrt(vb / N + vaa / N);
    const tUn = (ma - mb) / seU;
    const dfU = Math.pow(vb / N + vaa / N, 2) / (Math.pow(vb / N, 2) / (N - 1) + Math.pow(vaa / N, 2) / (N - 1));
    const pUn = tTailTwo(tUn, dfU);
    setText('pd-pt', fmt(tPaired, 3)); setText('pd-pp', fmt(pPaired, 4));
    setText('pd-ut', fmt(tUn, 3)); setText('pd-up', fmt(pUn, 4));

    // slope chart: before -> after with connecting lines
    const pad = 30, x0 = pad + 18, x1 = w - 14, y0 = 16, y1 = h - 24;
    const all = before.concat(after);
    const lo = Math.min(...all), hi = Math.max(...all);
    const Y = v => y1 - (v - lo) / (hi - lo || 1) * (y1 - y0);
    const xB = x0 + (x1 - x0) * 0.25, xA = x0 + (x1 - x0) * 0.75;
    ctx.lineWidth = 1.2;
    before.forEach((b, i) => {
      ctx.strokeStyle = after[i] >= b ? 'rgba(67,56,202,0.5)' : 'rgba(220,38,38,0.5)';
      ctx.beginPath(); ctx.moveTo(xB, Y(b)); ctx.lineTo(xA, Y(after[i])); ctx.stroke();
      ctx.fillStyle = ACCENT; ctx.beginPath(); ctx.arc(xB, Y(b), 3, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = WARN; ctx.beginPath(); ctx.arc(xA, Y(after[i]), 3, 0, Math.PI * 2); ctx.fill();
    });
    // mean markers
    ctx.strokeStyle = INK; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xB - 14, Y(mb)); ctx.lineTo(xB + 14, Y(mb)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xA - 14, Y(ma)); ctx.lineTo(xA + 14, Y(ma)); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('before', xB, y1 + 14); ctx.fillText('after', xA, y1 + 14);
    ctx.fillText('lines connect the same subject — pairing removes between-subject spread', (x0 + x1) / 2, y0 + 2);
    ctx.textAlign = 'left';
  }
  ['pd-d', 'pd-b', 'pd-w', 'pd-n'].forEach(id => $(id).addEventListener('input', () => { resample(); draw(); }));
  $('pd-resample').addEventListener('click', () => { resample(); draw(); });
  window.addEventListener('resize', draw);
  resample(); draw();
})();

// ============================================================
// 11. ONE-WAY ANOVA — F = MSB / MSE
// ============================================================
(function anova() {
  const cv = $('cv-anova'); if (!cv) return;
  let groups = [];
  function resample() {
    const k = n('av-k', 3), N = n('av-n', 12), spread = n('av-sp', 6), within = n('av-w', 8);
    groups = [];
    for (let g = 0; g < k; g++) {
      const center = 50 + (g - (k - 1) / 2) * spread;
      const arr = []; for (let i = 0; i < N; i++) arr.push(center + randn() * within);
      groups.push(arr);
    }
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const k = n('av-k', 3), N = n('av-n', 12);
    setText('av-kv', k); setText('av-nv', N); setText('av-spv', n('av-sp', 6)); setText('av-wv', n('av-w', 8));
    if (groups.length !== k || groups[0].length !== N) resample();
    const all = [].concat(...groups);
    const grand = mean(all);
    const groupMeans = groups.map(g => mean(g));
    let ssb = 0, sse = 0;
    groups.forEach((g, i) => { ssb += g.length * (groupMeans[i] - grand) ** 2; g.forEach(x => sse += (x - groupMeans[i]) ** 2); });
    const dfB = k - 1, dfW = all.length - k;
    const msb = ssb / dfB, mse = sse / dfW;
    const F = msb / mse;
    const p = fTail(F, dfB, dfW);
    setText('av-df', `${dfB}, ${dfW}`);
    setText('av-ms', `${fmt(msb, 1)} / ${fmt(mse, 1)}`);
    setText('av-f', fmt(F, 3));
    setText('av-p', fmt(p, 4));
    $('av-f').style.color = p < 0.05 ? BAD : ACCENT;

    // strip plot per group with mean lines and grand mean
    const pad = 30, x0 = pad, x1 = w - 14, y0 = 16, y1 = h - 26;
    const lo = Math.min(...all), hi = Math.max(...all);
    const Y = v => y1 - (v - lo) / (hi - lo || 1) * (y1 - y0);
    const colW = (x1 - x0) / k;
    // grand mean line
    ctx.strokeStyle = INK_S; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(x0, Y(grand)); ctx.lineTo(x1, Y(grand)); ctx.stroke(); ctx.setLineDash([]);
    groups.forEach((g, i) => {
      const cx = x0 + colW * (i + 0.5);
      g.forEach((x, j) => {
        const jitter = (j % 7 - 3) * (colW * 0.05);
        ctx.fillStyle = ACCENT_S; ctx.beginPath(); ctx.arc(cx + jitter, Y(x), 3, 0, Math.PI * 2); ctx.fill();
      });
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(cx - colW * 0.32, Y(groupMeans[i])); ctx.lineTo(cx + colW * 0.32, Y(groupMeans[i])); ctx.stroke();
      ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'center';
      ctx.fillText('G' + (i + 1), cx, y1 + 14);
    });
    ctx.fillStyle = INK_S; ctx.font = '11px Inter'; ctx.textAlign = 'left';
    ctx.fillText('dashed = grand mean · bars = group means', x0, y0 + 2);
  }
  ['av-k', 'av-n', 'av-sp', 'av-w'].forEach(id => $(id).addEventListener('input', () => { resample(); draw(); }));
  $('av-resample').addEventListener('click', () => { resample(); draw(); });
  window.addEventListener('resize', draw);
  resample(); draw();
})();

// ============================================================
// 12. CHI-SQUARE TEST OF INDEPENDENCE
// ============================================================
(function chiSquare() {
  const cv = $('cv-chi'); if (!cv) return;
  const ROWS = 2, COLS = 3;
  let O = [[30, 20, 10], [10, 20, 30]];
  function buildTable() {
    const host = $('chi-table'); if (!host) return;
    let html = '<table class="grid"><tr><th></th>';
    for (let j = 0; j < COLS; j++) html += `<th>C${j + 1}</th>`;
    html += '</tr>';
    for (let i = 0; i < ROWS; i++) {
      html += `<tr><th>R${i + 1}</th>`;
      for (let j = 0; j < COLS; j++) html += `<td><input type="number" min="0" step="1" data-i="${i}" data-j="${j}" value="${O[i][j]}"/></td>`;
      html += '</tr>';
    }
    html += '</table>';
    host.innerHTML = html;
    host.querySelectorAll('input').forEach(inp => {
      inp.addEventListener('input', () => {
        const i = +inp.dataset.i, j = +inp.dataset.j; const v = Math.max(0, Math.round(+inp.value || 0));
        O[i][j] = v; draw();
      });
    });
  }
  function syncInputs() {
    const host = $('chi-table'); if (!host) return;
    host.querySelectorAll('input').forEach(inp => { inp.value = O[+inp.dataset.i][+inp.dataset.j]; });
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const rowSum = O.map(r => r.reduce((a, b) => a + b, 0));
    const colSum = [];
    for (let j = 0; j < COLS; j++) { let s = 0; for (let i = 0; i < ROWS; i++) s += O[i][j]; colSum.push(s); }
    const Ntot = rowSum.reduce((a, b) => a + b, 0);
    let chi = 0; const E = [];
    for (let i = 0; i < ROWS; i++) { E.push([]); for (let j = 0; j < COLS; j++) { const e = Ntot ? rowSum[i] * colSum[j] / Ntot : 0; E[i].push(e); if (e > 0) chi += (O[i][j] - e) ** 2 / e; } }
    const df = (ROWS - 1) * (COLS - 1);
    const p = chiSqTail(chi, df);
    setText('chi-stat', fmt(chi, 3));
    setText('chi-df', df);
    setText('chi-p', fmt(p, 4));
    setText('chi-dec', p < 0.05 ? 'reject independence' : 'no evidence');
    $('chi-dec').style.color = p < 0.05 ? BAD : GOOD;

    // draw observed (solid bars) vs expected (outline) grouped by cell
    const pad = 30, x0 = pad, x1 = w - 14, y0 = 28, y1 = h - 26;
    const cells = ROWS * COLS;
    const maxV = Math.max(1, ...O.flat(), ...E.flat());
    const cw = (x1 - x0) / cells;
    const Y = v => y1 - v / maxV * (y1 - y0);
    let idx = 0;
    ctx.textAlign = 'center'; ctx.font = '9px JetBrains Mono';
    for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) {
      const bx = x0 + cw * idx + cw * 0.15, bw = cw * 0.7;
      // observed
      ctx.fillStyle = ACCENT_S; ctx.fillRect(bx, Y(O[i][j]), bw, y1 - Y(O[i][j]));
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 1.4; ctx.strokeRect(bx, Y(O[i][j]), bw, y1 - Y(O[i][j]));
      // expected outline
      ctx.strokeStyle = WARN; ctx.lineWidth = 1.4; ctx.setLineDash([3, 2]);
      ctx.strokeRect(bx, Y(E[i][j]), bw, y1 - Y(E[i][j])); ctx.setLineDash([]);
      ctx.fillStyle = MUTED; ctx.fillText(`R${i + 1}C${j + 1}`, bx + bw / 2, y1 + 12);
      idx++;
    }
    ctx.fillStyle = INK_S; ctx.font = '11px Inter'; ctx.textAlign = 'left';
    ctx.fillText('solid = observed O · dashed = expected E (under independence)', x0, y0 - 12);
    ctx.fillStyle = MUTED; ctx.textAlign = 'right'; ctx.font = '10px JetBrains Mono';
    ctx.fillText(`N = ${Ntot}`, x1, y0 - 12);
    ctx.textAlign = 'left';
  }
  $('chi-indep').addEventListener('click', () => { O = [[25, 25, 25], [25, 25, 25]]; syncInputs(); draw(); });
  $('chi-assoc').addEventListener('click', () => { O = [[40, 25, 10], [10, 25, 40]]; syncInputs(); draw(); });
  window.addEventListener('resize', draw);
  buildTable(); draw();
})();
