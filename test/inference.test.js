import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ciHalfWidth, oneSampleTest, twoSampleTest, pairedTest, anova, chiSquareIndependence,
} from '../src/lib/inference.js';
import { makeRng, randn } from '../src/lib/random.js';

const approx = (a, b, eps) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b} (eps ${eps})`);

test('CI half-width = t * s / sqrt(n)', () => {
  // n=25, s=1, 95% -> t_{0.975,24}=2.0639, hw = 2.0639/5 = 0.41278
  approx(ciHalfWidth(1, 25, 0.95), 2.0639 / 5, 2e-3);
});

test('one-sample z-test rejects a clear effect', () => {
  // diff=1, s=1, n=25 -> z=5, two-sided p ~ 0
  const r = oneSampleTest({ diff: 1, s: 1, n: 25, alpha: 0.05, kind: 'z', alt: 'two' });
  approx(r.stat, 5, 1e-9);
  assert.ok(r.pval < 1e-5);
  assert.ok(r.reject);
});

test('one-sample t-test fails to reject a tiny effect', () => {
  const r = oneSampleTest({ diff: 0.05, s: 1, n: 25, alpha: 0.05, kind: 't', alt: 'two' });
  assert.ok(r.pval > 0.05);
  assert.ok(!r.reject);
});

test('two-sample pooled and Welch agree when sizes and variances match', () => {
  const r = twoSampleTest({ mean: 50, sd: 10, n: 20 }, { mean: 56, sd: 10, n: 20 });
  approx(r.tPool, r.tWelch, 1e-9);   // equal n and sd -> identical statistic
  assert.equal(r.dfPool, 38);
  assert.ok(r.pWelch > 0 && r.pWelch < 1);
});

test('paired t-test on a consistent shift is significant', () => {
  const before = [10, 12, 14, 16, 18, 20];
  const after = before.map((b) => b + 3); // exact +3 each, zero within-pair noise
  const r = pairedTest(before, after);
  approx(r.meanDiff, 3, 1e-12);
  assert.ok(!Number.isFinite(r.t) || r.t > 0); // sd of diffs is 0 -> t is +Inf
});

test('ANOVA F is large when group means are far apart', () => {
  const g1 = [1, 2, 3], g2 = [11, 12, 13], g3 = [21, 22, 23];
  const r = anova([g1, g2, g3]);
  assert.equal(r.dfB, 2);
  assert.equal(r.dfW, 6);
  assert.ok(r.F > 50);
  assert.ok(r.p < 0.001);
});

test('ANOVA on identical groups gives F near 0 and large p', () => {
  const g = [4, 5, 6, 7];
  const r = anova([g, [...g], [...g]]);
  approx(r.F, 0, 1e-9);
  approx(r.p, 1, 1e-9);
});

test('chi-square independence: associated table rejects, uniform does not', () => {
  const assoc = chiSquareIndependence([[40, 25, 10], [10, 25, 40]]);
  assert.equal(assoc.df, 2);
  assert.ok(assoc.chi > 0);
  assert.ok(assoc.p < 0.05);

  const indep = chiSquareIndependence([[25, 25, 25], [25, 25, 25]]);
  approx(indep.chi, 0, 1e-9); // observed == expected
  approx(indep.p, 1, 1e-9);
});

test('seeded RNG is deterministic and reproducible', () => {
  const a = makeRng(12345);
  const b = makeRng(12345);
  for (let i = 0; i < 5; i++) assert.equal(a(), b());
  // a different seed diverges
  const c = makeRng(999);
  assert.notEqual(makeRng(12345)(), c());
});

test('seeded randn has roughly zero mean over many draws', () => {
  const rng = makeRng(7);
  let s = 0; const N = 20000;
  for (let i = 0; i < N; i++) s += randn(rng);
  approx(s / N, 0, 0.05); // sample mean of standard normals near 0
});
