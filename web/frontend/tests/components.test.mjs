import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';
import ts from 'typescript';
import { definitions, fixture } from './preview-fixtures.mjs';

// Exercise real component markup without a browser or generated source files.
// This complements, but does not replace, interactive browser testing.
const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const helpers = dataUrl(ts.transpileModule(
  readFileSync(new URL('../src/lib/analysis.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.ESNext } },
).outputText);
const componentUrls = new Map();
function componentUrl(name) {
  if (componentUrls.has(name)) return componentUrls.get(name);
  const { js } = compile(readFileSync(new URL(`../src/${name}.svelte`, import.meta.url), 'utf8'),
    { filename: `${name}.svelte`, generate: 'server' });
  const code = js.code.replace(/from '(svelte(?:\/internal\/server)?|\.\/lib\/analysis|\.\/\w+\.svelte)'/g,
    (_, module) => `from '${module.endsWith('.svelte') ? componentUrl(module.slice(2, -7)) :
      module === './lib/analysis' ? helpers : import.meta.resolve(module)}'`);
  const url = dataUrl(code);
  componentUrls.set(name, url);
  return url;
}
const Board = (await import(componentUrl('GenericBoard'))).default;
const Inspector = (await import(componentUrl('DecisionInspector'))).default;

function generalsBoard(overrides = {}) {
  const { info, state, history } = fixture('generals_8x8');
  return render(Board, { props: {
    cells: state.cells, legalMoves: state.legal_moves, gameOver: false,
    lastBotMove: null, gameInfo: info, currentPlayer: state.current_player,
    actionPresentations: state.actions, assessments: history.records[0].decision.actions,
    onCellClick() { assert.fail('render cannot play'); }, ...overrides,
  } }).body;
}

test('Generals renders terrain, armies, accessible tiles, and global source totals', () => {
  const html = generalsBoard();
  assert.equal([...html.matchAll(/aria-label="Tile /g)].length, 64);
  assert.match(html, /aria-label="Tile 1,1"[^>]*>.*?★.*?12/s);
  assert.match(html, /aria-label="Tile 3,3"[^>]*>.*?◉.*?20/s);
  assert.match(html, /aria-label="Tile 5,5"[^>]*>.*?▲/s);
  assert.match(html, /22\.2%/);
  assert.match(html, /57\.8%/);
  assert.equal([...html.matchAll(/class="probability /g)].length, 2);
  assert.doesNotMatch(html, /<button[^>]*generals-wait[^>]*disabled/);
});

for (const [label, overrides] of [
  ['read-only', { readOnly: true }],
  ['game over', { gameOver: true }],
  ['illegal Wait', { legalMoves: [36, 37] }],
  ['missing Wait presentation', { actionPresentations: [] }],
]) {
  test(`Generals disables Wait when ${label}`, () => {
    const html = generalsBoard(overrides);
    assert.match(html, /<button[^>]*generals-wait[^>]*disabled/);
    const disabledTiles = [...html.matchAll(/<button[^>]*disabled[^>]*aria-label="Tile /g)].length;
    assert.equal(disabledTiles, overrides.gameOver ? 64 : 0);
  });
}

test('Generals leaves missing probabilities absent and preserves genuine zeros', () => {
  assert.doesNotMatch(generalsBoard({ assessments: [] }), /class="probability /);
  const html = generalsBoard({ assessments: [{ action: { kind: 'discrete', index: 36 }, visit_share: 0 }] });
  assert.match(html, /0\.0%/);
  assert.equal([...html.matchAll(/class="probability /g)].length, 1);
});

for (const id of Object.keys(definitions)) {
  test(`${id}: recorded board and inspector render the same position`, () => {
    const { info, history } = fixture(id);
    const { state, decision } = history.records[0];
    const board = render(Board, { props: {
      cells: state.cells, legalMoves: state.legal_moves, gameOver: state.game_over,
      lastBotMove: null, gameInfo: info, currentPlayer: state.current_player,
      humanPlayer: state.human_player, actionPresentations: state.actions,
      assessments: decision.actions, readOnly: true, onCellClick() { assert.fail('render cannot play'); },
    } }).body;
    const inspector = render(Inspector, { props: { decision, position: state, presentations: state.actions } }).body;
    assert.match(inspector, /Search value/);
    assert.match(inspector, /not win percentages/);
    assert.match(inspector, /Agent 2/);
    assert.match(inspector, /Checkpoint aaaaaaaaaaaa/);
    assert.doesNotMatch(board + inspector, /NaN|undefined/);
    const probabilityLabels = [...board.matchAll(/class="probability /g)];
    assert.equal(probabilityLabels.length, id === 'othello' ? 0 : id === 'generals_8x8' ? 2 : state.actions.length);
    if (id === 'othello') assert.match(inspector, /Pass/);
    if (id === 'generals_8x8') assert.match(inspector, /Wait/);
  });
}

test('stale decision values never render on another position', () => {
  const { state, history } = fixture('tictactoe');
  const html = render(Inspector, { props: { decision: history.records[0].decision, position: state, presentations: state.actions } }).body;
  assert.match(html, /No recorded decision for position 2/);
  assert.doesNotMatch(html, /Search value|Checkpoint/);
});

test('non-board agent-zero Q estimates render without invented probabilities', () => {
  const position = { session_id: 'counter', revision: 3 };
  const decision = { schema_version: 1, position, actor: 0, source: 'dqn_q_values',
    selected_action: { kind: 'discrete', index: 1 },
    actions: [{ action: { kind: 'discrete', index: 1 }, q_value: 2.5 }],
    network_value: { value: 2.5, perspective_agent: 0, quantity: 'discounted_return', bounds: null },
    search_value: null, search: null, checkpoint: null };
  const html = render(Inspector, { props: { decision, position, presentations: [{ action: 1, label: 'Right', target: { kind: 'named', name: 'Right' } }] } }).body;
  assert.match(html, /Agent 0/);
  assert.match(html, /discounted return/);
  assert.match(html, /Right/);
  assert.match(html, /2\.500/);
  assert.doesNotMatch(html, /0\.0%|Overlay \/ rank|Search details/);
});

test('random selection is labelled honestly and has no model or values', () => {
  const { history } = fixture('tictactoe');
  const { state, decision } = history.records[0];
  decision.source = 'random';
  decision.network_value = decision.search_value = decision.search = decision.checkpoint = null;
  decision.actions = state.actions.map(a => ({ action: { kind: 'discrete', index: a.action }, selection_probability: 1 / state.actions.length }));
  const html = render(Inspector, { props: { decision, position: state, presentations: state.actions } }).body;
  assert.match(html, /uniform random selection probabilities/);
  assert.match(html, /12\.5%/);
  assert.doesNotMatch(html, /Network value|Search value|Checkpoint|Search details/);
});
