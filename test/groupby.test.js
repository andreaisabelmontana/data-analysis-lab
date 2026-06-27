import test from 'node:test';
import assert from 'node:assert/strict';
import { groupBy, aggregate, reducers, sumBy, meanBy, countBy } from '../src/lib/groupby.js';

const approx = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

// constructed dataset: sales by region
const rows = [
  { region: 'N', amount: 10 },
  { region: 'S', amount: 20 },
  { region: 'N', amount: 30 },
  { region: 'S', amount: 40 },
  { region: 'N', amount: 50 },
];

test('groupBy partitions rows by key', () => {
  const g = groupBy(rows, (r) => r.region);
  assert.equal(g.get('N').length, 3);
  assert.equal(g.get('S').length, 2);
});

test('sum aggregation totals each group correctly', () => {
  const sums = sumBy(rows, (r) => r.region, (r) => r.amount);
  assert.equal(sums.N, 90); // 10+30+50
  assert.equal(sums.S, 60); // 20+40
});

test('mean aggregation averages each group correctly', () => {
  const means = meanBy(rows, (r) => r.region, (r) => r.amount);
  approx(means.N, 30); // 90/3
  approx(means.S, 30); // 60/2
});

test('count aggregation counts members per group', () => {
  const counts = countBy(rows, (r) => r.region);
  assert.equal(counts.N, 3);
  assert.equal(counts.S, 2);
});

test('aggregate works with min/max reducers', () => {
  const mins = aggregate(rows, (r) => r.region, (r) => r.amount, reducers.min);
  const maxs = aggregate(rows, (r) => r.region, (r) => r.amount, reducers.max);
  assert.equal(mins.N, 10);
  assert.equal(maxs.N, 50);
});
