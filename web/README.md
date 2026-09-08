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
| `/game/history` | GET | Read retained positions and their recorded decisions; never runs inference |
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
npm --prefix web/frontend test
npm --prefix web/frontend run check
```

### Decision inspector

Play a move, then choose **Last bot decision** to inspect the position the bot
actually searched. Previous/next and the position selector browse the retained
match. Historical boards are read-only; **Live** returns to play. Inspecting an
action highlights it without submitting a move.

The board-independent action table shows separate quantities:

| Metric | Meaning |
|--------|---------|
| Prior | Raw legal-masked network policy, before search pruning/noise |
| Visits % | Root child visit share, before temperature |
| Pick % | Distribution used to select the move, after temperature |
| N | Visits to this action |
| Q | Mean action value from the decision actor's perspective; unavailable if unvisited |
| Network / search value | Expected outcome in `[-1, 1]`, from the explicitly labelled agent's perspective, not win probability |

Search details include completed simulations, root visits (including the root
bootstrap), evaluated observations, elapsed search time, temperature and the
exact checkpoint used. Capturing diagnostics uses the same search and random
draws as ordinary play; opening history never re-evaluates a position. Without
a model, the inspector labels the bot's uniform random selection and leaves
learned values/search metrics absent rather than showing fabricated zeroes.

Environments optionally implement `describe_discrete_action`: cells for
TicTacToe/Othello, columns for Connect 4, edges for Generals, and named Pass/Wait
actions. Generic renderers use those targets for overlays; probabilities on a
Generals source sum its outgoing actions without renormalizing away Wait.
Missing presentation falls back to `Action N` in the table. New discrete games
can use the inspector before adding a spatial renderer.

The schema makes value quantity, perspective and optional metrics explicit.
It includes a Q-value-only/non-board agent-zero test, but this is **not DQN web
serving**: the installed server still only accepts `alphazero_mcts_web_v1` and a
board profile. A DQN provider, persistent replay import/export, whole-match value
charts and optional on-demand analysis of human positions remain follow-ups.

### Position identity and trace API

State responses include `session_id`, monotonic `revision`, and legal action
presentations. A `PositionRecord.decision` describes the move **from** that
record's state. Thus one `/move` response may advance two revisions while history
retains the intermediate pre-bot position. Reset creates a new session identity.

`GET /game/history?session_id=...&from_revision=0&limit=64` returns schema version
1 with `first_available_revision`, `current_revision`, `records` and
`next_revision`. The server retains at most 256 positions in memory for the
shared session; pages contain 1–64 records and default to the most recent 64.
Reset and process restart discard history. This is not a durable replay store.

The UI sends `expected: {session_id, revision}` with `/move` and `/game/new`.
Mismatches return HTTP 409 without changing the session. The field remains
optional for older API clients; clients omitting it do not receive stale-write
protection. A history request for another session also returns 409. After a
failed move, read `/game/state` before proceeding: the human move may have
committed before bot evaluation failed. Do not automatically retry the move.

### UI-only preview fixtures

```bash
npm --prefix web/frontend run preview:fixtures
```

This starts a localhost-only preview on port 5174 with explicit fixture data for
the four board games, including Pass and Wait. It does not simulate games or
evaluate a trained model. Helper and server-rendered component tests run with
`npm test`; interactive browser checks are still needed for input and animation.

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
