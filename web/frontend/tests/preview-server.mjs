// Start with `npm run preview:fixtures`. This uses local synthetic data only.
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { definitions, fixture } from './preview-fixtures.mjs';

let selected = 'connect4';
const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  configFile: fileURLToPath(new URL('../vite.config.ts', import.meta.url)),
  server: { host: '127.0.0.1', port: 5174, strictPort: true },
  plugins: [{ name: 'explicit-inspector-fixtures', configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      const url = new URL(req.url, 'http://localhost');
      const path = url.pathname;
      if (!['/health', '/games', '/game/state', '/game/history', '/game/new', '/move', '/stats', '/model'].includes(path) && !path.startsWith('/game-info/')) return next();
      res.setHeader('Content-Type', 'application/json');
      if (path === '/game/new') {
        let body = ''; for await (const chunk of req) body += chunk;
        const requested = JSON.parse(body || '{}');
        if (definitions[requested.game]) selected = requested.game;
      }
      const data = fixture(selected);
      if (path === '/health') return res.end(JSON.stringify({ status: 'ok', version: 'fixture' }));
      if (path === '/games') return res.end(JSON.stringify({ games: Object.keys(definitions) }));
      if (path.startsWith('/game-info/')) return res.end(JSON.stringify(fixture(path.split('/').pop()).info));
      if (path === '/game/state' || path === '/game/new') return res.end(JSON.stringify(data.state));
      if (path === '/game/history') return res.end(JSON.stringify(data.history));
      res.statusCode = 503;
      res.end(JSON.stringify({ error: 'Unavailable in UI-only fixture preview' }));
    });
  } }],
});
await server.listen();
server.printUrls();
