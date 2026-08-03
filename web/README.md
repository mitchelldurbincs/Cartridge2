# Cartridge2 Web Interface

Serving host and Svelte UI for the selected algorithm/environment contract.
Startup resolves `[algorithm].id`, requires environment compatibility, and
dispatches the descriptor's serving component. The installed implementation is
`alphazero_mcts_web_v1` from `alphazero_board_v1`; it additionally requires the
environment's optional `metadata.board` profile and board presentation.

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
              │ engine-  │              │ selected runtime profile │
              │ core     │              │ model + stats            │
              └──────────┘              └──────────────────────────┘
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
| `/model` | GET | Get info about the loaded model |

## Quick Start

### 1. Start the Rust backend

```bash
cargo run --manifest-path web/Cargo.toml
```

The server starts on `http://localhost:8080`.

### 2. Start the Svelte frontend

```bash
npm --prefix web/frontend install
npm --prefix web/frontend run dev
```

The dev server starts on `http://localhost:5173` with hot-reload.

### 3. Play!

Open http://localhost:5173 in your browser.

## Development

### Build for production

```bash
# Build frontend
npm --prefix web/frontend run build

# Build backend
cargo build --release --manifest-path web/Cargo.toml
```

The frontend builds to `web/frontend/dist/`, which is served by **nginx** in
the frontend container. The Axum router registers API routes only and has no
static-file service.

### Run tests

```bash
cargo test --manifest-path web/Cargo.toml
```

## Configuration

The web server uses `config.toml` from the project root for centralized configuration. Settings can be overridden with environment variables:

- `CARTRIDGE_WEB_HOST` - Server bind address (default: `0.0.0.0`)
- `CARTRIDGE_WEB_PORT` - Server port (default: `8080`)
- `CARTRIDGE_COMMON_ENV_ID` - Default game environment (default: `tictactoe`)
- `CARTRIDGE_ALGORITHM_ID` - Algorithm cartridge (default: `alphazero_board_v1`)
- `CARTRIDGE_COMMON_DATA_DIR` - Runtime root (default: `./data`)
- `CARTRIDGE_STORAGE_MODEL_BACKEND` - `filesystem` or `s3`

The web host appends
`profiles/{algorithm_id}/{env_id}/v{env_contract_version}` to the runtime root.
Filesystem mode watches that profile's `models/channels/current.json`; S3 mode
polls the equivalent object key. This sole mutable `RunHeadV2` selects a fully
validated immutable RunCommit/checkpoint chain. Web uses
`ChampionOrLatest`: champion state from the latest RunCommit wins, with that
commit's latest checkpoint used before the first promotion. The accepted
RunHead generation advances even when the selected champion checkpoint remains
unchanged. RunHead, RunCommit lineage, manifest/blob digests and sizes, exact
five-field artifact identity, and policy/value tensor contract must all
validate. An absent RunHead permits random play, while present invalid authority
fails startup without replacing the last valid in-memory evaluator.

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
