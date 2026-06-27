import test from 'node:test';
import assert from 'node:assert/strict';
import { minMax, zScore, binIndex, histogram } from '../src/lib/transform.js';
import { mean, std } from '../src/lib/summary.js';

const approx = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

test('min-max scales into [0,1] with 0 and 1 attained', () => {
  const out = minMax([10, 20, 30, 40, 50]);
  for (const v of out) assert.ok(v >= 0 && v <= 1);
  approx(Math.min(...out), 0);
  approx(Math.max(...out), 1);
  approx(out[2], 0.5); // 30 is the midpoint of [10,50]
});

test('min-max maps a constant series to zeros', () => {
  assert.deepEqual(minMax([7, 7, 7]), [0, 0, 0]);
});

test('z-score yields zero mean and unit sample variance', () => {
  const z = zScore([2, 4, 4, 4, 5, 5, 7, 9]);
  approx(mean(z), 0, 1e-12);
  approx(std(z), 1, 1e-12); // sample std of standardized data is 1
});

test('binIndex buckets values into the correct equal-width bin', () => {
  // range [0,10], 5 bins of width 2: [0,2) [2,4) [4,6) [6,8) [8,10]
  assert.equal(binIndex(0, 0, 10, 5), 0);
  assert.equal(binIndex(1.9, 0, 10, 5), 0);
  assert.equal(binIndex(2, 0, 10, 5), 1);
  assert.equal(binIndex(5, 0, 10, 5), 2);
  assert.equal(binIndex(9.5, 0, 10, 5), 4);
  assert.equal(binIndex(10, 0, 10, 5), 4); // top edge clamps into last bin
});

test('histogram counts assign every value to a bin', () => {
  const { counts, edges } = histogram([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5, 0, 10);
  assert.deepEqual(counts, [2, 2, 2, 2, 2]);
  assert.equal(counts.reduce((a, b) => a + b, 0), 10);
  assert.equal(edges.length, 6);
  approx(edges[0], 0);
  approx(edges[5], 10);
});
