import test from 'node:test';
import assert from 'node:assert/strict';
import { correlation, covariance } from '../src/lib/correlation.js';
import { linearRegression, predict } from '../src/lib/regression.js';

const approx = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

test('correlation is +1 for perfectly linear data', () => {
  const x = [1, 2, 3, 4, 5];
  const y = x.map((v) => 3 * v + 7); // strictly increasing line
  approx(correlation(x, y), 1);
});

test('correlation is -1 for perfectly anti-linear data', () => {
  const x = [1, 2, 3, 4, 5];
  const y = x.map((v) => -2 * v + 10);
  approx(correlation(x, y), -1);
});

test('correlation stays within [-1, 1] for noisy data', () => {
  const x = [1, 2, 3, 4, 5, 6, 7];
  const y = [2, 1, 4, 3, 6, 5, 8];
  const r = correlation(x, y);
  assert.ok(r > -1 && r < 1);
});

test('covariance has the expected sign and value', () => {
  const x = [1, 2, 3, 4, 5];
  const y = [2, 4, 6, 8, 10]; // y = 2x, cov = 2 * var(x)
  // var(x) with n-1 over 1..5 = 2.5, so cov = 5
  approx(covariance(x, y), 5);
});

test('regression recovers a known slope, intercept and R^2 = 1', () => {
  const x = [0, 1, 2, 3, 4, 5];
  const slopeTrue = 2.5, interceptTrue = -1.0;
  const y = x.map((v) => slopeTrue * v + interceptTrue);
  const m = linearRegression(x, y);
  approx(m.slope, slopeTrue);
  approx(m.intercept, interceptTrue);
  approx(m.r2, 1);
  approx(predict(m, 10), slopeTrue * 10 + interceptTrue);
});

test('R^2 drops below 1 when residuals are nonzero', () => {
  const x = [1, 2, 3, 4, 5];
  const y = [1, 2, 1.3, 3.75, 2.25]; // scattered
  const m = linearRegression(x, y);
  assert.ok(m.r2 < 1 && m.r2 >= 0);
});
