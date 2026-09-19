import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

// Compile the pure TS helpers with the project's existing compiler. No added
// runner dependency and no generated files in the source tree.
const source = readFileSync(new URL('../src/lib/analysis.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { percent, samePosition, sortedActions, targetMass, boundAnalysis } =
  await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

test('missing metrics are not zero; probabilities preserve their scale', () => {
  assert.equal(percent(null), '—');
  assert.equal(percent(NaN), '—');
  assert.equal(percent(0), '0.0%');
  assert.equal(percent(0.25), '25.0%');
});

test('session and revision both bind the analysis', () => {
  const state = { session_id: 'a', revision: 2 };
  assert.equal(samePosition(state, { ...state, session_id: 'b' }), false);
  assert.equal(samePosition(state, { ...state, revision: 3 }), false);
  assert.equal(boundAnalysis({ state, decision: { schema_version: 1, position: { ...state, revision: 1 } } }), null);
  assert.equal(boundAnalysis({ state, decision: { schema_version: 2, position: state } }), null);
  const decision = { schema_version: 1, position: state };
  assert.equal(boundAnalysis({ state, decision }), decision);
});

test('Generals source mass sums directions and excludes Wait', () => {
  const actions = [
    { action: 36, target: { kind: 'edge', from: 9, to: 1 } },
    { action: 37, target: { kind: 'edge', from: 9, to: 10 } },
    { action: 256, target: { kind: 'named', name: 'Wait' } },
  ];
  const stats = [
    { action: { kind: 'discrete', index: 36 }, visit_share: 0.2 },
    { action: { kind: 'discrete', index: 37 }, visit_share: 0.3 },
    { action: { kind: 'discrete', index: 256 }, visit_share: 0.5 },
  ];
  assert.equal(targetMass(actions, stats, 'visit_share', 'source', 9), 0.5);
  assert.equal(targetMass(actions, stats, 'visit_share', 'source', 10), null);
  assert.equal(targetMass(actions, stats, 'network_prior', 'source', 9), null);
});

test('columns and cell actions are not interchangeable; Pass has no cell', () => {
  const actions = [
    { action: 0, target: { kind: 'column', index: 0 } },
    { action: 64, target: { kind: 'named', name: 'Pass' } },
  ];
  const stats = [{ action: { kind: 'discrete', index: 0 }, visit_share: 0.4 }];
  assert.equal(targetMass(actions, stats, 'visit_share', 'column', 0), 0.4);
  assert.equal(targetMass(actions, stats, 'visit_share', 'cell', 0), null);
  assert.equal(targetMass(actions, stats, 'visit_share', 'cell', 64), null);
});

test('ranking switches metrics without mutating the recorded data', () => {
  const rows = [
    { action: { index: 1 }, network_prior: 0.8, visit_share: 0.2 },
    { action: { index: 2 }, network_prior: 0.2, visit_share: 0.8 },
  ];
  assert.equal(sortedActions(rows, 'network_prior')[0].action.index, 1);
  assert.equal(sortedActions(rows, 'visit_share')[0].action.index, 2);
  assert.equal(rows[0].action.index, 1);
});

test('non-board agent-zero Q data is not converted to a probability', () => {
  const state = { session_id: 'counter', revision: 1 };
  const decision = { schema_version: 1, position: state, actor: 0, source: 'dqn_q_values',
    actions: [{ action: { kind: 'discrete', index: 1 }, q_value: 2.5, network_prior: null, visit_share: null }], search: null };
  assert.equal(boundAnalysis({ state, decision }).actor, 0);
  assert.equal(boundAnalysis({ state, decision }).actions[0].q_value, 2.5);
  assert.equal(percent(decision.actions[0].network_prior), '—');
});
