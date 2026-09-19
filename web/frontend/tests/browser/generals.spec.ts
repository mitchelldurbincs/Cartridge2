import { test, expect } from '@playwright/test';
import { fixture } from '../preview-fixtures.mjs';

test.beforeEach(async ({ page, request }) => {
  await request.post('/game/new', { data: { game: 'tictactoe' } });
  await page.goto('/');
  await expect(page.getByLabel('Select Game:')).toBeEnabled();
  await page.getByLabel('Select Game:').selectOption('generals_8x8');
  await expect(page.getByLabel('History position')).toBeEnabled();
});

test('Generals selects, cancels, and repicks sources before submitting a target or Wait', async ({ page }) => {
  const moves: unknown[] = [];
  await page.route('**/move', async route => {
    moves.push(route.request().postDataJSON());
    await route.fulfill({ json: { ...fixture('generals_8x8').state, bot_move: null } });
  });
  const source = page.getByRole('button', { name: 'Tile 1,1', exact: true });
  const otherSource = page.getByRole('button', { name: 'Tile 0,5', exact: true });
  const hint = page.locator('.generals-hint');

  await source.click();
  await expect(source).toHaveClass(/selected/);
  await expect(hint).toContainText('Now pick an adjacent tile');
  await source.click();
  await expect(hint).toContainText('Pick one of your tiles');
  await source.click();
  await otherSource.click();
  await expect(otherSource).toHaveClass(/selected/);
  await expect(source).not.toHaveClass(/selected/);
  await page.getByRole('button', { name: 'Tile 5,5', exact: true }).click();
  await expect(hint).toContainText('Pick one of your tiles');
  expect(moves).toEqual([]);

  await source.click();
  await page.getByRole('button', { name: 'Tile 2,1', exact: true }).click();
  await expect(page.getByLabel('History position')).toBeEnabled();
  await expect(hint).toContainText('Pick one of your tiles');
  await source.click();
  await page.getByRole('button', { name: 'Wait', exact: true }).click();
  await expect(page.getByLabel('History position')).toBeEnabled();
  await expect(hint).toContainText('Pick one of your tiles');
  expect(moves).toEqual([
    { position: 37, expected: { session_id: 'fixture-generals_8x8', revision: 2 } },
    { position: 256, expected: { session_id: 'fixture-generals_8x8', revision: 2 } },
  ]);
  await source.click();
  await page.getByRole('button', { name: 'New Game (You First)', exact: true }).click();
  await expect(page.getByLabel('History position')).toBeEnabled();
  await expect(hint).toContainText('Pick one of your tiles');
});

test('Generals uses advertised actions and recovers a failed move without retrying it', async ({ page }) => {
  const data = fixture('generals_8x8');
  // Deliberately decouple the action ID from the tile/direction formula.
  const state = { ...data.state, legal_moves: [201], actions: [
    { action: 201, label: 'Advertised edge', target: { kind: 'edge', from: 9, to: 10 } },
  ] };
  const history = { ...data.history, records: [{ state, decision: null }] };
  await page.route('**/games', route => route.fulfill({ json: { games: ['generals_8x8'] } }));
  await page.route('**/game/state', route => route.fulfill({ json: state }));
  await page.route('**/game/history?*', route => route.fulfill({ json: history }));
  const moves: unknown[] = [];
  await page.route('**/move', async route => {
    moves.push(route.request().postDataJSON());
    await route.fulfill({ status: 503, contentType: 'text/plain', body: 'Synthetic move failure' });
  });
  // Reload reads the synthetic position without starting a game.
  await page.reload();
  await expect(page.getByLabel('History position')).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Wait', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Tile 1,1', exact: true }).click();
  await page.getByRole('button', { name: 'Tile 2,1', exact: true }).click();
  await expect(page.locator('.game-section .error')).toContainText('Synthetic move failure');
  await expect(page.getByLabel('History position')).toBeEnabled();
  await expect(page.locator('.generals-hint')).toContainText('Pick one of your tiles');
  expect(moves).toEqual([
    { position: 201, expected: { session_id: 'fixture-generals_8x8', revision: 2 } },
  ]);
});

test('Generals inspection preserves global probabilities and never submits a move', async ({ page }) => {
  const moves: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname === '/move') moves.push(request.url()); });
  await page.getByRole('button', { name: 'Last bot decision', exact: true }).click();
  const source = page.getByRole('button', { name: 'Tile 1,1', exact: true });
  const directions = page.getByLabel('Directional action probabilities');
  await source.click();
  await expect(source.locator('.probability')).toHaveText('22.2%');
  await expect(directions.locator('strong')).toHaveText(['↑', '→', '↓', '←']);
  await expect(directions).toContainText('2.2%');
  await expect(directions).toContainText('8.9%');
  await expect(page.locator('.generals-wait')).toBeDisabled();
  await page.getByLabel('Overlay / rank').selectOption('network_prior');
  await expect(source.locator('.probability')).toHaveText('44.4%');
  await expect(directions).toContainText('11.1%');
  await source.click();
  await expect(directions).toHaveCount(0);
  const inspector = page.getByRole('region', { name: 'Decision inspector' });
  await inspector.getByRole('button', { name: /\(6, 1\) → \(6, 2\)/ }).click();
  await expect(page.getByRole('button', { name: 'Tile 0,5', exact: true })).toHaveClass(/selected/);
  await expect(directions.locator('.inspected')).toContainText('11.1%');
  await page.getByRole('button', { name: 'Live', exact: true }).click();
  await expect(directions).toHaveCount(0);
  await expect(page.locator('.generals-hint')).toContainText('Pick one of your tiles');
  expect(moves).toEqual([]);
});
