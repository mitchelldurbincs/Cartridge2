# Cartridge2 - Claude Code Guide

## Project Overview

Cartridge2 is a simplified AlphaZero training and visualization platform. It enables training neural network game agents via self-play and lets users play against trained models through a web interface.

**Target Games:** TicTacToe (complete), Connect 4 (complete), Othello (complete)

**Key Difference from Cartridge1:** This is a monolithic/filesystem approach vs. Cartridge1's microservices architecture. No Kubernetes, no gRPC between services—just shared filesystem and local processes.

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

- `engine-core/` - Game trait, erased adapter, registry, EngineContext API, GameMetadata (70 tests)
- `engine-config/` - Centralized configuration loading from config.toml (19 tests)
- `games-tictactoe/` - TicTacToe implementation (26 tests)
- `games-connect4/` - Connect 4 implementation (20 tests)
- `games-othello/` - Othello implementation (25 tests)
- `mcts/` - Monte Carlo Tree Search implementation (22 tests)
- `model-watcher/` - Shared model hot-reload utilities (5 tests)

### Actor (Rust Binary) - `actor/`
**Status: COMPLETE (69 tests)**

Self-play episode runner using engine-core directly:
- Uses `EngineContext` for game simulation (no gRPC)
- PostgreSQL storage backend (local development or K8s)
- MCTS policy with ONNX neural network evaluation
- Hot-reloads model when `latest.onnx` changes (via model_watcher)
- Stores MCTS visit distributions as policy targets
- Game outcome backfill for value targets
- Auto-derives game configuration from GameMetadata

### Web Server (Rust Binary) - `web/`
**Status: COMPLETE (27 tests)**

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

### Web Frontend (Svelte + TypeScript) - `web/frontend/`
**Status: COMPLETE**

Svelte 5 frontend with Vite:
- TicTacToe board display
- Play against bot (random moves for now)
- Live training stats polling
- Responsive dark-mode UI

### Trainer (Python) - `trainer/`
**Status: COMPLETE (145 tests)**

PyTorch training with AlphaZero-style learning and orchestration:

**CLI Commands:**
- `python -m trainer train` - Train on replay buffer data
- `python -m trainer evaluate` - Evaluate model against random baseline
- `python -m trainer loop` - Synchronized AlphaZero training (actor + trainer + eval)

**Features:**
- Reads transitions from PostgreSQL replay buffer
- MCTS policy distributions as soft targets
- Game outcome propagation for value targets
- MLP network for TicTacToe, ResNet for spatial games (Connect4, Othello)
- Exports ONNX models with atomic write-then-rename
- Writes `stats.json` and `eval_stats.json` telemetry
- Cosine annealing LR schedule with warmup
- Gradient clipping for stability
- Model evaluation against random baseline (enabled by default in loop)
- Orchestrator auto-resume from last completed iteration
- MCTS simulation ramping (start low, increase over iterations)
- Structured JSON logging for cloud deployments
- Prometheus metrics export

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
│       ├── model_watcher.rs # ONNX model hot-reload via file watching
│       ├── health.rs       # Health check endpoint
│       ├── metrics.rs      # Prometheus metrics
│       ├── stats.rs        # Self-play statistics
│       └── storage/        # Storage backends (PostgreSQL)
├── engine/                 # Rust workspace
│   ├── Cargo.toml         # Workspace config
│   ├── engine-core/       # Core Game trait + EngineContext API
│   │   └── src/
│   │       ├── adapter.rs  # GameAdapter (typed -> erased)
│   │       ├── context.rs  # EngineContext high-level API
│   │       ├── erased.rs   # ErasedGame trait
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
│   ├── games-tictactoe/   # TicTacToe implementation
│   ├── games-connect4/    # Connect 4 implementation
│   ├── games-othello/    # Othello implementation
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
│   │   ├── main.rs        # HTTP server setup, routing (uses engine-config)
│   │   ├── game.rs        # Game session management
│   │   ├── metrics.rs     # Prometheus metrics
│   │   ├── model_watcher.rs # Model hot-reload for web
│   │   ├── handlers/      # Route handlers (game, health, stats)
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
│   ├── Dockerfile         # Trainer-only Docker image
│   └── src/trainer/
│       ├── __main__.py    # CLI entrypoint (train, evaluate, loop)
│       ├── trainer.py     # Training loop
│       ├── network.py     # Neural network (MLP, used for TicTacToe)
│       ├── resnet.py      # ResNet architecture (used for Connect4, Othello)
│       ├── replay.py      # Replay buffer interface
│       ├── evaluator.py   # Model evaluation
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
│       │   └── connect4.py
│       └── storage/       # Storage backends (PostgreSQL, S3, filesystem)
├── Dockerfile.alphazero   # Combined actor+trainer image for Docker
├── config.toml            # Central configuration file
├── .github/workflows/
│   └── ci.yml             # CI pipeline (Rust fmt/clippy/test, Python lint/test, frontend build)
├── documentation/
│   ├── MVP.md             # Design document
│   ├── ARCHITECTURE.md    # Comprehensive architecture reference
│   └── API.md             # REST API documentation with examples
├── data/                  # Runtime data (gitignored)
│   ├── replay.db          # (Legacy) SQLite replay buffer - now using PostgreSQL
│   ├── models/            # ONNX model files
│   └── stats.json         # Training telemetry
└── CLAUDE.md              # This file
```

## Configuration

All settings are centralized in `config.toml` at the project root. This single file controls all components (actor, trainer, web server).

### Configuration Priority

Settings are loaded with the following priority (highest to lowest):
1. **CLI arguments** - Direct command-line flags
2. **Environment variables** - `CARTRIDGE_*` or legacy `ALPHAZERO_*`
3. **config.toml** - Central configuration file
4. **Built-in defaults** - Hardcoded fallbacks

### config.toml Structure

```toml
[common]
data_dir = "./data"      # Base data directory
env_id = "tictactoe"     # Game: tictactoe, connect4, othello
log_level = "info"       # trace, debug, info, warn, error

[training]
iterations = 100         # Training iterations
start_iteration = 1      # For resuming training
episodes_per_iteration = 500
steps_per_iteration = 1000
batch_size = 64
learning_rate = 0.001
weight_decay = 0.0001    # L2 regularization
grad_clip_norm = 1.0     # Gradient clipping (0 to disable)
device = "auto"          # auto, cpu, cuda, mps
checkpoint_interval = 250
max_checkpoints = 10
num_actors = 6           # Parallel actor processes for self-play

[evaluation]
interval = 1             # Evaluate every N iterations (0=disable)
games = 50               # Games per evaluation
win_threshold = 0.55     # Win rate to become new best model
eval_vs_random = true    # Also evaluate against random baseline

[actor]
actor_id = "actor-1"
max_episodes = -1        # -1 for unlimited
episode_timeout_secs = 180
flush_interval_secs = 5
log_interval = 50

[web]
host = "0.0.0.0"
port = 8080
# allowed_origins = []   # CORS origins (empty = allow all in dev)

[mcts]
# Simulation ramping: starts low, increases over iterations
start_sims = 50          # Simulations for first iteration
max_sims = 250           # Maximum simulations after ramping
sim_ramp_rate = 10       # Simulations added per iteration
num_simulations = 200    # Legacy setting (used if ramping not configured)
c_puct = 1.0
temperature = 1.0
temp_threshold = 15      # Move number to reduce temperature (0 = disabled)
dirichlet_alpha = 0.4
dirichlet_weight = 0.25
eval_batch_size = 64     # Batch size for NN evaluation during MCTS
onnx_intra_threads = 1   # Threads for ONNX inference (1 = best for multi-actor)

[logging]
format = "text"          # "text" or "json" (structured for cloud)
include_timestamps = true
include_target = true

[storage]
model_backend = "filesystem"  # filesystem or s3
postgres_url = "postgresql://cartridge:cartridge@localhost:5432/cartridge"
pool_max_size = 16
pool_connect_timeout = 30
pool_idle_timeout = 300
# s3_bucket = "cartridge-models"      # For S3 backend
# s3_endpoint = "http://minio:9000"   # For MinIO
```

### Environment Variable Overrides

All components (actor, trainer, web) support the `CARTRIDGE_*` format:
```bash
CARTRIDGE_COMMON_ENV_ID=connect4
CARTRIDGE_COMMON_DATA_DIR=/data
CARTRIDGE_TRAINING_ITERATIONS=50
CARTRIDGE_EVALUATION_GAMES=100
CARTRIDGE_WEB_HOST=127.0.0.1
CARTRIDGE_WEB_PORT=3000
CARTRIDGE_STORAGE_MODEL_BACKEND=s3
CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://user:pass@host/db
```

Additional overrides:
```bash
CARTRIDGE_LOGGING_FORMAT=json        # Structured JSON logging (for cloud)
CARTRIDGE_MCTS_START_SIMS=100       # MCTS simulation ramping
CARTRIDGE_MCTS_MAX_SIMS=400
```

Legacy format (Python trainer only):
```bash
ALPHAZERO_ENV_ID=connect4
ALPHAZERO_ITERATIONS=50
ALPHAZERO_EVAL_GAMES=100
```

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
docker compose run --rm evaluator

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

# Run all tests
cd engine && cargo test   # 187 tests (70 + 19 + 26 + 20 + 25 + 22 + 5)
cd actor && cargo test    # 69 tests
cd web && cargo test      # 27 tests
cd trainer && python -m pytest tests/ -v --tb=short  # 145 tests

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

# Train on existing replay buffer data (requires PostgreSQL)
python -m trainer train --steps 1000

# Evaluate model against random play
python -m trainer evaluate --model ./data/models/latest.onnx --games 100

# ======= Alternative: Continuous (non-synchronized) training =======
# Actor and trainer run concurrently - mixes data from multiple model versions
# Less correct for AlphaZero but simpler for quick experiments

# Run self-play to generate training data
cd actor && cargo run -- --env-id tictactoe --max-episodes 1000

# Train the model (in separate terminal)
python -m trainer train --steps 1000
```

## Current Status

- [x] Engine core abstractions (Game trait, adapter, registry, metadata) - 70 tests
- [x] EngineContext high-level API
- [x] TicTacToe game implementation - 26 tests
- [x] Connect 4 game implementation - 20 tests
- [x] Othello game implementation - 25 tests
- [x] Removed gRPC/proto dependencies (library-only)
- [x] Actor core (episode runner, pluggable storage backends) - 69 tests
- [x] MCTS integration in actor with ONNX evaluation
- [x] Model hot-reload via file watching (model-watcher crate) - 5 tests
- [x] Auto-derived game configuration from GameMetadata
- [x] Web server (Axum, game API) - 27 tests
- [x] Web frontend (Svelte, play UI, stats, loss visualization)
- [x] MCTS implementation - 22 tests
- [x] Python trainer (PyTorch, ONNX export, evaluator) - 145 tests
- [x] ResNet architecture for spatial games (Connect4, Othello)
- [x] MCTS policy targets + game outcome propagation
- [x] Storage backends (PostgreSQL, S3, filesystem)
- [x] CI pipeline (GitHub Actions)

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

// Run search
let legal_mask = 0b111111111u64; // All 9 positions legal initially
let mut rng = ChaCha20Rng::seed_from_u64(42);
let result = run_mcts(&mut ctx, &evaluator, config, reset.state, legal_mask, &mut rng).unwrap();

println!("Best action: {}", result.action);
println!("Policy: {:?}", result.policy);
println!("Value: {}", result.value);
```

### MCTS Architecture

```
engine/mcts/src/
├── lib.rs          # Public API exports
├── config.rs       # MctsConfig (num_simulations, c_puct, temperature, etc.)
├── evaluator.rs    # Evaluator trait + UniformEvaluator
├── node.rs         # MctsNode (visit_count, value_sum, prior, children)
├── tree.rs         # MctsTree with arena allocation
└── search.rs       # Select, expand, backpropagate, run_search
```

** Also, remember that if you are working on MCTS, that we have benchmarks for that. It may be a good idea to run those if you are making major changes to performance for it.**


### Key Types

- `MctsConfig` - Search parameters (simulations, c_puct, dirichlet noise, temperature)
- `Evaluator` trait - Provides policy priors and value estimates
- `UniformEvaluator` - Returns uniform policy (for testing without neural network)
- `SearchResult` - Contains best action, policy distribution, value estimate

## Next Steps

1. **Analysis/Replay Mode** - Add a page to step through saved games and see MCTS visit counts

## Reference

- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - Python AlphaZero reference
- MVP.md in documentation/ - Full design document
