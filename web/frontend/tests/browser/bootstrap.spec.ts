import { test, expect } from '@playwright/test';

test('shows a retryable bootstrap error when game state loading fails after health succeeds', async ({ page }) => {
  let healthRequests = 0;
  let stateRequests = 0;
  await page.route('**/health', async route => {
    healthRequests++;
    await route.fulfill({ json: { status: 'ok', version: 'fixture' } });
  });
  await page.route('**/game/state', async route => {
    stateRequests++;
    if (stateRequests === 1) {
      await route.fulfill({ status: 503, contentType: 'text/plain', body: 'synthetic bootstrap failure' });
    } else {
      await route.fallback();
    }
  });

  await page.goto('/');
  const alert = page.getByRole('alert');
  await expect(alert).toContainText('Could not load game: Error: Failed to get game state');
  await expect(alert.getByRole('button', { name: 'Retry' })).toBeEnabled();
  await expect(page.getByText('Loading game...', { exact: true })).toHaveCount(0);
  expect(healthRequests).toBe(1);

  await alert.getByRole('button', { name: 'Retry' }).click();
  await expect(page.getByLabel('History position')).toBeEnabled();
  expect(healthRequests).toBe(2);
  expect(stateRequests).toBe(2);
});

test('shows a retryable offline bootstrap error when health fails, then recovers', async ({ page }) => {
  let healthRequests = 0;
  await page.route('**/health', async route => {
    healthRequests++;
    if (healthRequests === 1) {
      await route.fulfill({ status: 503, contentType: 'text/plain', body: 'synthetic health failure' });
    } else {
      await route.fallback();
    }
  });

  await page.goto('/');
  const alert = page.getByRole('alert');
  await expect(alert).toContainText('Could not load game: Error: Health check failed');
  await expect(alert.getByRole('button', { name: 'Retry' })).toBeEnabled();
  await expect(page.getByText('Cannot connect to server.')).toBeVisible();
  expect(healthRequests).toBe(1);

  await alert.getByRole('button', { name: 'Retry' }).click();
  await expect(page.getByLabel('History position')).toBeEnabled();
  expect(healthRequests).toBe(2);
});
