# Cartridge2 Web Interface

Minimal web interface for playing TicTacToe against the AI and monitoring training.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Svelte Frontend│────▶│  Axum Backend    │
│  (localhost:5173)     │  (localhost:8080)│
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              ┌──────────┐              ┌──────────────┐
              │ engine-  │              │ data/        │
              │ core     │              │ stats.json   │
              └──────────┘              └──────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/games` | GET | List available games |
| `/game-info/:id` | GET | Get metadata for a specific game |
| `/game/state` | GET | Get current board state |
| `/game/new` | POST | Start a new game |
| `/move` | POST | Make a move (player + bot response) |
| `/stats` | GET | Read training stats from stats.json |
| `/actor-stats` | GET | Read actor self-play stats |
| `/model` | GET | Get info about the loaded model |

## Quick Start

### 1. Start the Rust backend

```bash
cd web
cargo run
```

The server starts on `http://localhost:8080`.

### 2. Start the Svelte frontend

```bash
cd web/frontend
npm install
npm run dev
```

The dev server starts on `http://localhost:5173` with hot-reload.

### 3. Play!

Open http://localhost:5173 in your browser.

## Development

### Build for production

```bash
# Build frontend
cd frontend
npm run build

# Build backend
cargo build --release
```

The frontend builds to `frontend/dist/` which can be served by the Rust backend.

### Run tests

```bash
cargo test  # ~27 tests
```

## Configuration

The web server uses `config.toml` from the project root for centralized configuration. Settings can be overridden with environment variables:

- `CARTRIDGE_WEB_HOST` - Server bind address (default: `0.0.0.0`)
- `CARTRIDGE_WEB_PORT` - Server port (default: `8080`)
- `CARTRIDGE_COMMON_ENV_ID` - Default game environment (default: `tictactoe`)
- `CARTRIDGE_COMMON_DATA_DIR` - Base data directory for stats.json (default: `./data`)

For full configuration options, see `config.toml` and `config.defaults.toml`.

## API Examples

### Get game state
```bash
curl http://localhost:8080/game/state
```

### Start new game (player first)
```bash
curl -X POST http://localhost:8080/game/new \
  -H "Content-Type: application/json" \
  -d '{"first":"player"}'
```

### Make a move
```bash
curl -X POST http://localhost:8080/move \
  -H "Content-Type: application/json" \
  -d '{"position":4}'
```

### Get training stats
```bash
curl http://localhost:8080/stats
```
