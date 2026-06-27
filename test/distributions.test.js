import test from 'node:test';
import assert from 'node:assert/strict';
import {
  normPdf, normCdf, normInv, tTailTwo, tInv, chiSqTail, fTail,
} from '../src/lib/distributions.js';

const approx = (a, b, eps) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b} (eps ${eps})`);

test('standard normal pdf and cdf hit known points', () => {
  approx(normPdf(0), 1 / Math.sqrt(2 * Math.PI), 1e-12);
  approx(normCdf(0), 0.5, 1e-7);
  approx(normCdf(1.96), 0.975, 1e-3);   // erf approx is ~1e-7 accurate
  approx(normCdf(-1.96), 0.025, 1e-3);
});

test('normInv inverts normCdf', () => {
  approx(normInv(0.975), 1.959963985, 1e-3);
  approx(normInv(0.5), 0, 1e-9);
  approx(normCdf(normInv(0.83)), 0.83, 1e-4);
});

test('two-sided t tail matches the normal in the large-df limit', () => {
  // P(|T|>1.96) -> ~0.05 as df grows
  approx(tTailTwo(1.96, 100000), 0.05, 2e-3);
});

test('tInv returns the textbook 95% critical value', () => {
  // t_{0.975, 10} = 2.228
  approx(tInv(0.975, 10), 2.228, 1e-2);
});

test('chi-square upper tail at known quantiles', () => {
  // P(X^2 > 3.841) = 0.05 for df=1
  approx(chiSqTail(3.841, 1), 0.05, 1e-3);
  // P(X^2 > 0) = 1
  approx(chiSqTail(0, 5), 1, 1e-12);
});

test('F upper tail is a probability in (0,1)', () => {
  const p = fTail(5, 3, 20);
  assert.ok(p > 0 && p < 1);
  approx(fTail(0, 3, 20), 1, 1e-12);
});
