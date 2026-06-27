// groupby.js — split-apply-combine helpers over arrays of records.
//
// `groupBy` partitions rows by a key function. `aggregate` applies a reducer
// to a chosen numeric field within each group, returning a plain object keyed
// by group. Convenience reducers cover the common cases: count, sum, mean.

import { mean as arrMean } from './summary.js';

// partition rows into a Map keyed by keyFn(row)
export function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const k = keyFn(row);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(row);
  }
  return groups;
}

// aggregate a numeric field per group with a reducer over the extracted values.
// reducer: (values: number[]) => number. Returns { [groupKey]: number }.
export function aggregate(rows, keyFn, valueFn, reducer) {
  const groups = groupBy(rows, keyFn);
  const out = {};
  for (const [k, members] of groups) out[k] = reducer(members.map(valueFn));
  return out;
}

// ready-made reducers
export const reducers = {
  count: (vs) => vs.length,
  sum: (vs) => vs.reduce((a, b) => a + b, 0),
  mean: (vs) => arrMean(vs),
  min: (vs) => Math.min(...vs),
  max: (vs) => Math.max(...vs),
};

// shorthands
export const sumBy = (rows, keyFn, valueFn) => aggregate(rows, keyFn, valueFn, reducers.sum);
export const meanBy = (rows, keyFn, valueFn) => aggregate(rows, keyFn, valueFn, reducers.mean);
export const countBy = (rows, keyFn) => aggregate(rows, keyFn, () => 1, reducers.count);
