import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const transpile = file => ts.transpileModule(readFileSync(new URL(file, import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
const constants = dataUrl(transpile('../src/lib/constants.ts'));
const { buildChartData, formatLoss } = await import(dataUrl(
  transpile('../src/lib/chart.ts').replace("'./constants'", `'${constants}'`),
));
const entry = (step, metrics) => ({ step, metrics, learning_rate: .001, grad_norm: null });

test('current metric-map payload produces finite charts and preserves zero', () => {
  const chart = buildChartData({ data: [entry(1, { 'loss/total': 1, 'loss/policy': .6, 'loss/value': .4 }),
    entry(2, { 'loss/total': 0, 'loss/policy': 0, 'loss/value': 0 })],
    chartWidth: 100, chartHeight: 80, includePoints: true, includeAvg100: true });
  assert.equal(chart.points.total[1].value, 0);
  assert.equal(formatLoss(undefined), '—');
  assert.notEqual(formatLoss(0), '—');
  assert.doesNotMatch(JSON.stringify(chart), /NaN|Infinity|null/);
});

test('missing loss metrics create gaps and never fabricate zeroes', () => {
  const chart = buildChartData({ data: [entry(1, { 'loss/total': 1, 'loss/policy': .8 }),
    entry(2, { 'loss/total': .5 }), entry(3, { 'loss/total': .2, 'loss/policy': .1 })],
    chartWidth: 100, chartHeight: 80, includePoints: true, includeAvg100: true });
  assert.equal(chart.paths.value, '');
  assert.equal(chart.points.value.length, 0);
  assert.deepEqual(chart.points.policy.map(p => p.step), [1, 3]);
  assert.equal((chart.paths.policy.match(/M/g) ?? []).length, 2);
  assert.doesNotMatch(JSON.stringify(chart), /NaN|Infinity|null/);
});

test('an unrelated algorithm metric does not become an invalid AlphaZero chart', () => {
  const chart = buildChartData({ data: [entry(1, { 'loss/td': .5 }), entry(2, { 'loss/td': .2 })],
    chartWidth: 100, chartHeight: 80, includePoints: true, includeAvg100: true });
  assert.deepEqual(chart.points, { total: [], policy: [], value: [] });
  assert.deepEqual(chart.paths, { total: '', policy: '', value: '', avg100: '' });
  assert.doesNotMatch(JSON.stringify(chart), /NaN|Infinity|null/);
});
