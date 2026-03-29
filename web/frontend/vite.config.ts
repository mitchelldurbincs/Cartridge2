import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://localhost:8080',
      '/move': 'http://localhost:8080',
      '/stats': 'http://localhost:8080',
      '/model': 'http://localhost:8080',
      '/selfplay': 'http://localhost:8080',
      '/game': 'http://localhost:8080',
      '/actor-stats': 'http://localhost:8080',
      '/games': 'http://localhost:8080',
      '/game-info': 'http://localhost:8080',
    },
  },
});
