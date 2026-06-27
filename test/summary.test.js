import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mean, variance, std, median, mode, quantile, quartiles, iqr,
  outliers, outlierBounds, summarize,
} from '../src/lib/summary.js';

const approx = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

// Hand-computed reference: data = [2,4,4,4,5,5,7,9], n = 8.
// mean = 40/8 = 5; squared devs from 5 sum to 32, sample var = 32/7.
const D = [2, 4, 4, 4, 5, 5, 7, 9];

test('mean matches hand value', () => {
  approx(mean(D), 5);
});

test('sample variance and std use the n-1 divisor', () => {
  approx(variance(D), 32 / 7);
  approx(std(D), Math.sqrt(32 / 7));
});

test('median is the R-7 interpolated 50th percentile', () => {
  // pos = 3.5 -> s[3] + 0.5*(s[4]-s[3]) = 4 + 0.5*(5-4) = 4.5
  approx(median(D), 4.5);
});

test('quartiles match hand-computed R-7 values', () => {
  const { q1, q2, q3 } = quartiles(D);
  approx(q1, 4);   // pos 1.75 -> 4 + 0.75*0
  approx(q2, 4.5);
  approx(q3, 5.5); // pos 5.25 -> 5 + 0.25*(7-5)
  approx(iqr(D), 1.5);
});

test('quantile endpoints return the extremes', () => {
  approx(quantile(D, 0), 2);
  approx(quantile(D, 1), 9);
});

test('mode returns the most frequent value', () => {
  assert.deepEqual(mode(D), [4]);
  assert.deepEqual(mode([1, 1, 2, 2, 3]).sort(), [1, 2]); // ties
});

test('1.5*IQR rule flags a known outlier and nothing else', () => {
  // tight cluster 20..27 (Q1=22, Q3=26, IQR=4 -> fences [16,32]) plus one spike
  const data = [20, 21, 22, 23, 24, 25, 26, 27, 80];
  const flagged = outliers(data);
  assert.deepEqual(flagged, [80]);
  const { lower, upper } = outlierBounds(data);
  assert.ok(80 > upper);
  // every non-outlier sits inside the fences
  for (const x of data.filter((v) => v !== 80)) assert.ok(x >= lower && x <= upper);
});

test('no outliers in tight, symmetric data', () => {
  assert.deepEqual(outliers([1, 2, 3, 4, 5, 6, 7]), []);
});

test('summarize bundles the descriptive set', () => {
  const s = summarize(D);
  approx(s.mean, 5);
  approx(s.median, 4.5);
  approx(s.iqr, 1.5);
  assert.equal(s.n, 8);
  assert.equal(s.min, 2);
  assert.equal(s.max, 9);
});
