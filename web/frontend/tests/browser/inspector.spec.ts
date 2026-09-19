import { test, expect } from '@playwright/test';

// Exercise the real Svelte app with explicitly synthetic serving data. Rust
// integration tests own the engine and wire semantics; these tests own clicks,
// reactive updates, read-only history, and browser rendering.
test.beforeEach(async ({ page, request }) => {
  await request.post('/game/new', { data: { game: 'tictactoe' } });
  await page.goto('/');
  await expect(page.getByLabel('Select Game:')).toBeEnabled();
});

for (const [game, probabilities] of [['tictactoe', 8], ['connect4', 7], ['othello', 0], ['generals_8x8', 2]] as const) {
  test(`${game}: inspect a recorded decision, switch metrics, and return to live`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    let moves = 0;
    page.on('request', request => { if (new URL(request.url()).pathname === '/move') moves++; });
    await page.getByLabel('Select Game:').selectOption(game);
    await page.getByRole('button', { name: 'Last bot decision', exact: true }).click();
    const inspector = page.getByRole('region', { name: 'Decision inspector' });
    await expect(inspector.getByText('Search value', { exact: true })).toBeVisible();
    await expect(page.getByLabel('History position')).toHaveValue('1');
    await expect(page.locator('.probability')).toHaveCount(probabilities);
    await expect(inspector).toContainText('Agent 2');
    await expect(inspector).toContainText('Checkpoint aaaaaaaaaaaa');
    await page.getByLabel('Overlay / rank').selectOption('network_prior');
    await expect(page.locator('.inspection-label').first()).toContainText('network prior');
    if (game === 'othello') await expect(inspector.getByRole('button', { name: '✓ Pass', exact: true })).toBeVisible();
    if (game === 'generals_8x8') await expect(inspector.getByRole('button', { name: 'Wait', exact: true })).toBeVisible();
    await inspector.getByRole('button').first().click();
    await expect(page.getByLabel('History position')).toHaveValue('1');
    expect(moves).toBe(0);
    await page.getByRole('button', { name: 'Live', exact: true }).click();
    await expect(page.getByLabel('History position')).toHaveValue('');
    await expect(page.locator('.probability')).toHaveCount(0);
    await expect(inspector.getByText('Search value', { exact: true })).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}

test('history navigation and reset clear the selected analysis', async ({ page }) => {
  await page.getByRole('button', { name: 'Previous position', exact: true }).click();
  await expect(page.getByLabel('History position')).toHaveValue('1');
  await page.getByRole('button', { name: 'Next position', exact: true }).click();
  await expect(page.getByLabel('History position')).toHaveValue('2');
  await expect(page.getByRole('region', { name: 'Decision inspector' })).toContainText('No recorded decision');
  await page.getByRole('button', { name: 'Last bot decision', exact: true }).click();
  await page.getByRole('button', { name: 'New Game (Bot First)', exact: true }).click();
  await expect(page.getByLabel('History position')).toHaveValue('');
  await expect(page.locator('.probability')).toHaveCount(0);
});

test('sparse metric charts render, support hovering, and do not reset the game on return', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  let resets = 0;
  page.on('request', request => { if (new URL(request.url()).pathname === '/game/new') resets++; });
  await expect(page.locator('.stats-panel')).toContainText('0.0000');
  await page.getByRole('link', { name: '⛶' }).click();
  const chart = page.getByRole('img', { name: 'Loss over time chart with interactive hover' });
  await expect(chart).toBeVisible();
  await chart.hover({ position: { x: 300, y: 100 } });
  expect(await chart.evaluate(element => element.outerHTML)).not.toMatch(/NaN|Infinity|undefined/);
  await page.getByRole('link', { name: '← Back to Game' }).click();
  await expect(page.getByLabel('History position')).toBeEnabled();
  expect(resets).toBe(0);
  expect(errors).toEqual([]);
});
