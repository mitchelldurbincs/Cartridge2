# Cartridge2 - Claude Code Guide

## Project Overview

Cartridge2 is a simplified AlphaZero training and visualization platform. It enables training neural network game agents via self-play and lets users play against trained models through a web interface.

**Target Games:** TicTacToe (complete), Connect 4 (complete), Othello (complete), Generals 8×8 (`generals_8x8` — engine/trainer complete; training does not yet beat random at local compute scale; not playable in the web UI, because `web/src/game.rs::parse_state` assumes a flat `[board][player][winner]` layout and generals uses a 12-byte header + 64×6-byte tiles)

**Key Difference from Cartridge1:** local processes over shared storage instead of microservices talking gRPC. K8s manifests (`k8s/`) and Terraform modules (`terraform/`) do exist for cloud deployment — what's avoided is service-to-service RPC, not orchestration.

**Where things are documented.** This file covers commands, conventions and
gotchas for working in the repo. It deliberately does not restate facts that
have an authoritative home elsewhere — every previous copy had drifted:

| For | Read |
|-----|------|
| Architecture, data flow, schemas, metrics | [`documentation/ARCHITECTURE.md`](documentation/ARCHITECTURE.md) |
| REST API reference | [`documentation/API.md`](documentation/API.md) |
| Deploying / running | [`documentation/DEPLOYMENT.md`](documentation/DEPLOYMENT.md) |
| Every config key and default | [`config.defaults.toml`](config.defaults.toml), [`engine/engine-config/SCHEMA.md`](engine/engine-config/SCHEMA.md) |
| Database schema | [`sql/schema.sql`](sql/schema.sql) |
| Test counts | `make test` — they change every commit |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Filesystem                         │
│  PostgreSQL (replay)   - Concurrent replay buffer           │
│  ./data/models/        - ONNX model files                   │
│  ./data/stats.json     - Training telemetry                 │
└─────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴────────┐  ┌───────┴───────┐  ┌────────┴────────┐
│  Web Server     │  │ Python Trainer│  │  Svelte Frontend│
│  (Axum :8080)   │  │ (Learner)     │  │  (Vite :5173)   │
│  - Engine lib   │  │ - PyTorch     │  │  - Play UI      │
│  - Game API     │  │ - PostgreSQL  │  │  - Stats display│
│  - Stats API    │  │ - ONNX export │  │                 │
└─────────────────┘  └───────────────┘  └─────────────────┘
```
## Programming
**IMPORTANT** Make sure to run tests / linters for the code changes you make.

### Rust (engine, actor, web)
```bash
# Format
cargo fmt --check --manifest-path engine/Cargo.toml
cargo fmt --check --manifest-path actor/Cargo.toml
cargo fmt --check --manifest-path web/Cargo.toml

# Lint
cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path actor/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path web/Cargo.toml --all-targets -- -D warnings

# Test
cargo test --manifest-path engine/Cargo.toml
cargo test --manifest-path actor/Cargo.toml
cargo test --manifest-path web/Cargo.toml
```

### Python (trainer)
```bash
cd trainer

# Lint
python -m ruff check src/
python -m black --check src/

# Auto-fix lint issues
python -m ruff check --fix src/
python -m black src/

# Test (requires deps: pip install -e ".[dev]")
python -m pytest tests/ -v --tb=short
```

### Frontend (web/frontend)
```bash
cd web/frontend
npm run check   # TypeScript/Svelte check
npm run build   # Build
``` 

## Components

### Engine (Rust Library) - `engine/`
**Status: COMPLETE**

Pure game logic library. No network I/O. Library-only design (no gRPC).

- `engine-core/` - Game trait, erased adapter, registry, EngineContext API, GameMetadata, LegalMask
- `engine-config/` - Centralized configuration loading from config.toml
- `engine-games/` - Registration of all bundled games; observation-layout invariants; the game-metadata manifest generator (`make game-manifest`) and its golden drift test
- `games-tictactoe/` - TicTacToe implementation
- `games-connect4/` - Connect 4 implementation
- `games-othello/` - Othello implementation
- `games-generals/` - Generals 8×8 implementation; see the crate's
  lib.rs for the ruleset (full-info, alternating turns, territory
  adjudication, parity-randomized ply cap) and `generals_obs:v1` layout
- `mcts/` - Monte Carlo Tree Search implementation; legal masks are
  dynamic-width (`LegalMask`) and read from the observation, never `info_bits`.
  Three diagnostic examples: `generals_policy_probe` (visit-distribution
  health), `generals_strength_probe` (MCTS+model vs random — the honest
  strength measure; the trainer's built-in eval is argmax-only and understates
  models), and `generals_search_diag` (branching factor vs search budget)
- `model-watcher/` - Shared model hot-reload utilities
- `metrics-common/` - Prometheus registration/encoding shared by actor and web

### Actor (Rust Binary) - `actor/`
**Status: COMPLETE**

Self-play episode runner using engine-core directly:
- Uses `EngineContext` for game simulation (no gRPC)
- PostgreSQL storage backend (local development or K8s)
- MCTS policy with ONNX neural network evaluation
- Hot-reloads model when `latest.onnx` changes (via model_watcher)
- Stores MCTS visit distributions as policy targets (raw tau=1, never sharpened
  by the play temperature)
- Game outcome backfill for value targets
- Auto-derives game configuration from GameMetadata (`game_config.rs` is a type
  alias over it)
- Episodes that time out are discarded whole — there is no outcome to backfill —
  and counted in `actor_stats.json` and `actor_episodes_abandoned_total`

### Web Server (Rust Binary) - `web/`
**Status: COMPLETE**

Axum HTTP server for frontend interaction:
- `/health` - Health check
- `/metrics` - Prometheus metrics
- `/games` - List available games
- `/game-info/:id` - Get game metadata
- `/game/new` - Start new game
- `/game/state` - Get current board state
- `/move` - Make player move + bot response
- `/stats` - Read training stats from stats.json
- `/actor-stats` - Read actor self-play stats
- `/model` - Get info about loaded model

#### API Endpoints

Full reference with request/response shapes: [`documentation/API.md`](documentation/API.md).
Routes are registered in `web/src/startup.rs`; response types live in
`web/src/types/responses.rs`.


### Web Frontend (Svelte + TypeScript) - `web/frontend/`
**Status: COMPLETE**

Svelte 5 frontend with Vite:
- Generic board display (TicTacToe, Connect 4, Othello)
- Play against the trained model (MCTS + ONNX; random moves until a model is loaded)
- Live training stats polling
- Responsive dark-mode UI

### Trainer (Python) - `trainer/`
**Status: COMPLETE**

PyTorch training with AlphaZero-style learning and orchestration:

**CLI Commands:**
- `python -m trainer train` - Train on replay buffer data
- `python -m trainer evaluate` - Evaluate model against random baseline
- `python -m trainer loop` - Synchronized AlphaZero training (actor + trainer + eval)
- `python -m trainer solver-eval` - Score Connect4 model moves against the bitbully perfect solver

**Features:**
- Reads transitions from PostgreSQL replay buffer
- MCTS policy distributions as soft targets
- Game outcome propagation for value targets
- MLP network for TicTacToe, ResNet for spatial games (Connect4, Othello, Generals)
- Game facts read from the engine-generated manifest (`game_metadata.json`);
  only network architecture is chosen trainer-side
- Exports ONNX models with atomic write-then-rename
- Writes `stats.json` and `eval_stats.json` telemetry
- Cosine annealing LR schedule with warmup
- Gradient clipping for stability
- Model evaluation against random baseline (enabled by default in loop)
- Orchestrator auto-resume from last completed iteration
- MCTS simulation ramping (start low, increase over iterations)
- Structured JSON logging for cloud deployments
- Prometheus metrics export

### crucible Dependency (Python) - sibling repo
**Status: EXTRACTED (github.com/mitchelldurbincs/crucible; not on PyPI — CI installs it from GitHub pinned to a commit, local dev uses the editable sibling checkout)**

The trainer's orchestration core lives in the sibling `crucible` repo
(`crucible` package): the synchronized loop (`Orchestrator`), the
`ActorRunner`/`EvalRunner` base classes, stats manager, promotion +
eval-reporting logic, `LoopConfig`, plus `wandb_logger`, `atomic_io`, and
`backoff`. Generation, training, and evaluation backends are injected
through the `typing.Protocol` seams in `crucible/protocols.py`.

**Dev setup** - install crucible editable before the trainer:
```bash
cd trainer
pip install -e ../../crucible   # sibling checkout, relative to trainer/
pip install -e ".[dev]"
```

**Composition root:** `trainer/src/trainer/orchestrator/orchestrator.py`
binds this repo's concrete pieces (storage.create_replay_buffer,
Trainer/TrainerConfig via TrainSpec, the shim ActorRunner/EvalRunner
subclasses, structured_logging tracing) into the core Orchestrator, keeping
the `Orchestrator(config)` signature unchanged for cli.py and callers.

**Shims:** `trainer/src/trainer/{wandb_logger,atomic_io,backoff}.py` and
`trainer/src/trainer/orchestrator/{config,stats_manager,eval_reporting,actor_runner,eval_runner}.py`
re-export from `crucible` (some restore repo-specific defaults) so
existing `trainer.*` imports keep working. Repo code imports through the
shims; only the composition root and tests reference `crucible.*`
directly.

## Directory Structure

```
cartridge2/
├── actor/                  # Rust actor binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Entry point
│       ├── actor.rs        # Episode runner using EngineContext
│       ├── config.rs       # CLI configuration (uses engine-config)
│       ├── game_config.rs  # Game-specific config derived from metadata
│       ├── mcts_policy.rs  # MCTS policy implementation
│       ├── health.rs       # Health check endpoint
│       ├── metrics.rs      # Prometheus metrics
│       ├── stats.rs        # Self-play statistics
│       └── storage/        # Storage backends (PostgreSQL)
├── engine/                 # Rust workspace
│   ├── Cargo.toml         # Workspace config
│   ├── engine-core/       # Core Game trait + EngineContext API
│   │   └── src/
│   │       ├── adapter.rs  # GameAdapter (typed -> erased)
│   │       ├── board_game.rs # TwoPlayerObs shared observation type
│   │       ├── context.rs  # EngineContext high-level API
│   │       ├── erased.rs   # ErasedGame trait
│   │       ├── game_utils.rs # Shared helpers for game implementations
│   │       ├── metadata.rs # GameMetadata for game configuration
│   │       ├── registry.rs # Static game registration
│   │       └── typed.rs    # Game trait definition
│   ├── engine-config/     # Centralized configuration (shared by actor/web)
│   │   ├── src/
│   │   │   ├── lib.rs      # Public API exports
│   │   │   ├── defaults.rs # Default configuration values
│   │   │   ├── structs.rs  # Config struct definitions
│   │   │   ├── loader.rs   # Loading logic + env overrides
│   │   │   └── tests.rs    # Unit tests
│   │   └── SCHEMA.md       # Configuration schema documentation
│   ├── engine-games/      # Registration + manifest generator + golden test
│   ├── metrics-common/    # Prometheus plumbing shared by actor and web
│   ├── games-tictactoe/   # TicTacToe implementation
│   ├── games-connect4/    # Connect 4 implementation
│   ├── games-othello/    # Othello implementation
│   ├── games-generals/   # Generals 8×8 implementation
│   ├── mcts/              # Monte Carlo Tree Search
│   │   └── src/
│   │       ├── config.rs   # MctsConfig
│   │       ├── evaluator.rs # Evaluator trait + UniformEvaluator
│   │       ├── node.rs     # MctsNode
│   │       ├── onnx.rs     # OnnxEvaluator (feature-gated)
│   │       ├── search.rs   # MCTS search algorithm
│   │       └── tree.rs     # MctsTree with arena allocation
│   └── model-watcher/     # Shared model hot-reload library
├── web/                    # Web server + frontend
│   ├── Cargo.toml         # Axum server
│   ├── src/
│   │   ├── main.rs        # Thin entry point
│   │   ├── startup.rs     # Router, AppState, CORS, shutdown
│   │   ├── game.rs        # Game session management
│   │   ├── metrics.rs     # Prometheus metrics
│   │   │   ├── handlers/      # Route handlers (game, health, stats)
│   │   └── types/         # Request/response types
│   ├── frontend/          # Svelte frontend
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── App.svelte
│   │   │   ├── GenericBoard.svelte  # Game board component
│   │   │   ├── LossChart.svelte     # Loss visualization chart
│   │   │   ├── LossOverTimePage.svelte # Training progress page
│   │   │   ├── Stats.svelte
│   │   │   ├── main.ts              # SPA routing (/loss-over-time)
│   │   │   └── lib/                 # Shared utilities
│   │   │       ├── api.ts           # API client library
│   │   │       ├── chart.ts         # Chart formatting utilities
│   │   │       └── constants.ts     # Polling intervals, etc.
│   │   └── vite.config.ts
│   └── README.md          # Run commands
├── trainer/               # Python training package
│   ├── pyproject.toml     # Package configuration
│   ├── tests/             # Pytest suite
│   └── src/trainer/
│       ├── __main__.py    # CLI entrypoint (train, evaluate, loop, solver-eval)
│       ├── trainer.py     # Training loop
│       ├── network.py     # Neural network (MLP, used for TicTacToe)
│       ├── resnet.py      # ResNet architecture (used for Connect4, Othello)
│       ├── evaluator.py   # Model evaluation
│       ├── solver_eval/   # Perfect-solver move-quality evaluation (Connect4)
│       ├── replay_setup.py # Buffer setup + engine/DB metadata cross-check
│       ├── game_metadata.json # GENERATED by `make game-manifest`
│       ├── wandb_logger.py # W&B wrapper (null-logger fallback, used by loop)
│       ├── game_config.py # Game-specific configs (auto-selects network type)
│       ├── stats.py       # Training statistics
│       ├── config.py      # TrainerConfig dataclass
│       ├── lr_scheduler.py # LR schedule (warmup + cosine annealing)
│       ├── checkpoint.py  # Checkpoint save/load utilities
│       ├── backoff.py     # Wait-with-backoff utilities
│       ├── central_config.py # Central config.toml loading
│       ├── metrics.py     # Prometheus metrics export
│       ├── logging_utils.py # Logging configuration
│       ├── structured_logging.py # JSON structured logging
│       ├── orchestrator/  # Synchronized AlphaZero training orchestrator
│       │   ├── orchestrator.py # Main loop coordinator
│       │   ├── cli.py     # CLI argument parsing for loop command
│       │   ├── config.py  # Orchestrator configuration
│       │   ├── actor_runner.py # Actor process management
│       │   ├── eval_runner.py  # Evaluation runner
│       │   └── stats_manager.py # Stats aggregation
│       ├── policies/      # Policy implementations
│       │   ├── random.py  # Random baseline policy
│       │   └── onnx.py    # ONNX model policy
│       ├── games/         # Pure Python game implementations (for evaluation)
│       │   ├── tictactoe.py
│       │   ├── connect4.py
│       │   └── generals.py # Mirrors engine/games-generals rules exactly
│       └── storage/       # Storage backends (PostgreSQL, S3, filesystem)
├── Dockerfile.alphazero   # Combined actor+trainer image for Docker
├── docker-compose.yml     # Local services (postgres, minio, training, web)
├── docker-compose.k8s.yml # Overlay for K8s-style backends (S3 models)
├── Makefile               # Convenience targets (setup, test, lint, train)
├── config.toml            # Central configuration file
├── config.defaults.toml   # Default values (single source of truth)
├── .github/workflows/
│   └── ci.yml             # CI pipeline (Rust fmt/clippy/test, Python lint/test, frontend build)
├── documentation/
│   ├── ARCHITECTURE.md    # Comprehensive architecture reference
│   └── API.md             # REST API documentation with examples
├── data/                  # Runtime data (gitignored)
│   ├── models/            # ONNX model files
│   └── stats.json         # Training telemetry (replay buffer lives in PostgreSQL)
└── CLAUDE.md              # This file
```

## Configuration

All settings live in `config.toml` at the project root, layered over the
checked-in defaults in `config.defaults.toml`. Both are read by every component
(actor, trainer, web).

**The key reference is not duplicated here.** `config.defaults.toml` lists every
key with its default and a comment, and is the single source of truth; the
schema reference is [`engine/engine-config/SCHEMA.md`](engine/engine-config/SCHEMA.md).

### Configuration Priority

Highest to lowest:

1. **CLI arguments** - direct flags
2. **Environment variables** - `CARTRIDGE_<SECTION>_<KEY>` (or legacy `ALPHAZERO_*`)
3. **`config.toml`** - your local overrides
4. **`config.defaults.toml`** - checked-in defaults

### Gotchas worth knowing

- **The two languages do not honour the same env vars.** Python parses
  `CARTRIDGE_<SECTION>_<KEY>` generically, so anything can be overridden. Rust
  matches an explicit list in `engine/engine-config/src/loader.rs`, so keys
  outside it — `logging.format`, the MCTS ramping keys, `num_actors`,
  `allowed_origins`, `health_port` — are honoured by the trainer but **ignored
  by the actor and web server**. Put those in `config.toml`.
- **`[wandb]` and the solver-eval keys are Python-only.** They have no
  counterpart in the Rust `CentralConfig`, so setting them does nothing for the
  actor or web server.
- **Empty `allowed_origins` does not mean "allow all".** `configure_cors` in
  `web/src/startup.rs` is deny-by-default: empty falls back to a localhost
  allowlist.
- **The trainer reads the replay DSN only from `CARTRIDGE_STORAGE_POSTGRES_URL`.**
  `storage.postgres_url` in `config.toml` is used by the Rust actor and web
  server, but *not* by the Python trainer.

## Quick Start

### Play TicTacToe in Browser

Terminal 1 - Start Rust backend:
```bash
cd web
cargo run
# Server starts on http://localhost:8080
```

Terminal 2 - Start Svelte frontend:
```bash
cd web/frontend
npm install
npm run dev
# Dev server starts on http://localhost:5173
```

Open http://localhost:5173 in your browser!

### Train with Docker (Easiest)

```bash
# Train using settings from config.toml
docker compose up alphazero

# Override game via environment variable
CARTRIDGE_COMMON_ENV_ID=connect4 docker compose up alphazero
# Or using legacy format:
ALPHAZERO_ENV_ID=connect4 docker compose up alphazero

# Run in background
docker compose up alphazero -d
docker compose logs -f alphazero  # Watch progress

# Run standalone evaluation
docker compose run --rm alphazero python -m trainer evaluate --model /app/data/models/latest.onnx

# Play against trained model (in another terminal)
docker compose up web frontend
# Open http://localhost in browser
```

**To customize training:** Edit `config.toml` before running, or use environment variable overrides. See the Configuration section above for all available settings.

## Commands

```bash
# Build engine
cd engine && cargo build --release

# Build actor
cd actor && cargo build --release

# Build web server
cd web && cargo build --release

# Run all tests (engine + actor + web + trainer)
make test

# Or individually
cargo test --manifest-path engine/Cargo.toml
cargo test --manifest-path actor/Cargo.toml
cargo test --manifest-path web/Cargo.toml
cd trainer && python -m pytest tests/ -v --tb=short

# Regenerate the game-metadata manifest after changing any game's metadata()
# (cargo test fails if the committed manifest is stale)
make game-manifest

# Format and lint
cd engine && cargo fmt && cargo clippy
cd actor && cargo fmt && cargo clippy
cd web && cargo fmt && cargo clippy

# Start web server
cd web && cargo run

# Start frontend dev server
cd web/frontend && npm run dev

# ======= RECOMMENDED: Synchronized AlphaZero Training =======
# Each iteration: clear buffer -> generate episodes -> train -> evaluate
# This ensures training data comes from the current model only
# Evaluation runs after each iteration by default!

# Install trainer package (required for local training)
cd trainer && pip install -e .

# PostgreSQL must be running, and the Python trainer reads the replay-buffer
# connection string ONLY from this env var — config.toml's storage.postgres_url
# is used by the Rust actor/web but NOT by the trainer:
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:cartridge@localhost:5432/cartridge

# Basic synchronized training (TicTacToe) with evaluation
python -m trainer loop --iterations 50 --episodes 200 --steps 500

# Connect4 with more data per iteration
python -m trainer loop --env-id connect4 --iterations 100 --episodes 500 --steps 1000

# With GPU (evaluation runs by default every iteration)
python -m trainer loop --device cuda --iterations 100

# Disable evaluation for faster training
python -m trainer loop --eval-interval 0 --iterations 50

# Resume from a specific iteration
python -m trainer loop --iterations 100 --start-iteration 25

# ======= Standalone Commands =======

# Train on existing replay buffer data
# (requires PostgreSQL + CARTRIDGE_STORAGE_POSTGRES_URL, see above)
python -m trainer train --steps 1000

# Evaluate model against random play
python -m trainer evaluate --model ./data/models/latest.onnx --games 100

# Score Connect4 model decisions against a perfect solver (bitbully)
# Metrics: value-optimal-move rate, blunder rate, exact-best rate
# (overall / by ply bucket / by seat); appends to data/solver_stats.json
python -m trainer solver-eval --model ./data/models/latest.onnx --games 100
python -m trainer solver-eval --all-checkpoints --games 100   # progression across checkpoints

# The loop runs solver eval automatically each evaluation (connect4) and can
# log everything to W&B; opt into solver-based gatekeeping with:
python -m trainer loop --env-id connect4 --wandb-enabled true --promotion-metric solver_optimal

# ======= Alternative: Continuous (non-synchronized) training =======
# Actor and trainer run concurrently - mixes data from multiple model versions
# Less correct for AlphaZero but simpler for quick experiments

# Run self-play to generate training data
cd actor && cargo run -- --env-id tictactoe --max-episodes 1000

# Train the model (in separate terminal)
python -m trainer train --steps 1000
```

## Current Status

- [x] Engine core abstractions (Game trait, adapter, registry, metadata)
- [x] EngineContext high-level API
- [x] TicTacToe game implementation
- [x] Connect 4 game implementation
- [x] Othello game implementation
- [x] Removed gRPC/proto dependencies (library-only)
- [x] Actor core (episode runner, pluggable storage backends)
- [x] MCTS integration in actor with ONNX evaluation
- [x] Model hot-reload via file watching (model-watcher crate)
- [x] Auto-derived game configuration from GameMetadata
- [x] Web server (Axum, game API)
- [x] Web frontend (Svelte, play UI, stats, loss visualization)
- [x] MCTS implementation
- [x] Python trainer (PyTorch, ONNX export, evaluator)
- [x] ResNet architecture for spatial games (Connect4, Othello)
- [x] MCTS policy targets + game outcome propagation
- [x] Storage backends (PostgreSQL, S3, filesystem)
- [x] CI pipeline (GitHub Actions)
- [x] Generals 8×8 game + dynamic-width legal masks
- [x] Perfect-solver evaluation + solver-based promotion (Connect 4)
- [x] Weights & Biases logging
- [x] Orchestration core extracted to the `crucible` sibling repo
- [x] Engine-generated game-metadata manifest (single source of truth)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/games` | GET | List available games |
| `/game-info/:id` | GET | Get game metadata |
| `/game/new` | POST | Start a new game |
| `/game/state` | GET | Get current board state |
| `/move` | POST | Make a move (player + bot) |
| `/stats` | GET | Read training stats |
| `/actor-stats` | GET | Read actor self-play stats |
| `/model` | GET | Get info about loaded model |

## Using the Engine

```rust
use engine_core::EngineContext;
use games_tictactoe::register_tictactoe;

// Register games at startup
register_tictactoe();

// Create a context for TicTacToe
let mut ctx = EngineContext::new("tictactoe").expect("game registered");

// Reset to initial state
let reset = ctx.reset(42, &[]).unwrap();
println!("Initial state: {} bytes", reset.state.len());

// Take a step (action = position 4 = center)
let action = 4u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();
println!("Reward: {}, Done: {}", step.reward, step.done);
```

## Game Trait Pattern

Games implement a typed trait that gets erased for runtime dispatch:

```rust
pub trait Game {
    type State;
    type Action;
    type Obs;

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8]) -> (State, Obs);
    fn step(&mut self, state: &mut State, action: Action, rng: &mut ChaCha20Rng)
        -> (Obs, f32, bool, u64);
    fn encode_state(state: &State, buf: &mut Vec<u8>) -> Result<(), Error>;
    fn decode_state(buf: &[u8]) -> Result<State, Error>;
    // ... similar for Action and Obs
}
```

## Adding a New Game

1. Create crate in `engine/games-{name}/`
2. Implement `Game` trait with State/Action/Obs types
3. Implement encode/decode for each type
4. Add a `register_{name}()` function that calls `register_game()`
5. Add tests for game logic + encoding round-trips

Example registration:
```rust
use engine_core::{register_game, GameAdapter};

pub fn register_connect4() {
    register_game("connect4".to_string(), || {
        Box::new(GameAdapter::new(Connect4::new()))
    });
}
```

## Differences from Cartridge1

| Aspect | Cartridge1 | Cartridge2 |
|--------|------------|------------|
| Architecture | 7 microservices | Monolith + Python |
| Communication | gRPC everywhere | Filesystem + HTTP |
| Replay Buffer | Go service + Redis | PostgreSQL |
| Model Storage | Go service + MinIO | Single ONNX file |
| Orchestration | K8s/Docker Compose | Python package |
| Complexity | Production-grade | MVP-focused |

## Using MCTS

```rust
use mcts::{MctsConfig, UniformEvaluator, run_mcts};
use engine_core::EngineContext;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

// Register game and create context
games_tictactoe::register_tictactoe();
let mut ctx = EngineContext::new("tictactoe").unwrap();
let reset = ctx.reset(42, &[]).unwrap();

// Set up MCTS with uniform evaluator (for testing)
let evaluator = UniformEvaluator::new();
let config = MctsConfig::for_training()
    .with_simulations(800)
    .with_temperature(1.0);

// Run search. The mask is a dynamic-width LegalMask read from the observation —
// never a u64 (that capped action spaces at 64) and never from info_bits.
let legal_mask = ctx.metadata().legal_mask_from_obs(&reset.obs);
let mut rng = ChaCha20Rng::seed_from_u64(42);
let result = run_mcts(
    &mut ctx, &evaluator, config, reset.state, reset.obs, legal_mask, &mut rng,
).unwrap();

println!("Best action: {}", result.action);
println!("Policy: {:?}", result.policy);  // tau=1 visit distribution
println!("Value: {}", result.value);
```

**`SearchResult.policy` is the training target and is always the raw tau=1 visit
distribution** — it is never sharpened by `MctsConfig::temperature`, which
controls only which action gets *played*. Sharpening the target destroys the
soft-target signal policy learning depends on (at tau=0.1 the visit counts are
raised to the 10th power, collapsing the target to near one-hot).

### MCTS Architecture

```
engine/mcts/src/
├── lib.rs          # Public API exports
├── config.rs       # MctsConfig (num_simulations, c_puct, temperature, etc.)
├── evaluator.rs    # Evaluator trait + UniformEvaluator
├── node.rs         # MctsNode (visit_count, value_sum, prior, children)
├── tree.rs         # MctsTree with arena allocation
├── search.rs       # Select, expand, backpropagate, run_search
├── sampling.rs     # Dirichlet noise + action sampling
├── types.rs        # SearchResult, SearchStats, errors
└── onnx.rs         # OnnxEvaluator (feature-gated)

engine/mcts/examples/   # generals_policy_probe, generals_strength_probe,
                        # generals_search_diag
engine/mcts/benches/    # search microbenchmarks
```

** Also, remember that if you are working on MCTS, that we have benchmarks for that. It may be a good idea to run those if you are making major changes to performance for it.**


### Key Types

- `MctsConfig` - Search parameters (simulations, c_puct, dirichlet noise, temperature)
- `Evaluator` trait - Provides policy priors and value estimates
- `UniformEvaluator` - Returns uniform policy (for testing without neural network)
- `SearchResult` - Contains best action, policy distribution, value estimate

## Known Gaps

- **Generals training does not yet beat random** at local compute scale. The
  measured shape of the problem: mean branching factor 37 (p90 71) against a
  50-250 simulation budget, and 92% of games decided by territory adjudication
  at the round cap. Use `cargo run -p mcts --example generals_search_diag
  --release` to re-measure.
- **Generals is not playable in the web UI** — `web/src/game.rs::parse_state`
  cannot decode its state layout (see Project Overview).
- **No Othello in the pure-Python game mirrors** (`trainer/src/trainer/games/`),
  so `create_game_state("othello")` raises. Othello is engine-side only for
  evaluation purposes.

## Reference

- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - Python AlphaZero reference
- [`documentation/ARCHITECTURE.md`](documentation/ARCHITECTURE.md) - full architecture reference
