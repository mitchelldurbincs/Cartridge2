<p align="center">
  <img src="./logo.png" alt="Cartridge2 Logo" width="500">
</p>

# Cartridge2

A simplified AlphaZero training and visualization platform for board games. Train neural network agents via self-play and play against them through a web interface.

**Games:** TicTacToe (complete), Connect 4 (complete), Othello (complete)

**Why the name?**
I love history (Hannibal Barca is my goat) and it also just happens to be close to cartridge which is a goal of this project - being able to easily add new games.
You may have noticed that the C in the logo is the esteemed harbor of Carthage.

**Improvements over v1**
The original Cartridge used 7 microservices with gRPC, Go, and Kubernetes—great for production scale, but overkill for experimentation. Cartridge2 simplifies everything: just Rust + Python, filesystem-based storage, and a single `docker compose up` to start training. It also swaps PPO for AlphaZero, which produces higher-quality training data for board games.

## Architecture

```
+---------------------------------------------------------------+
|                    PostgreSQL + Filesystem                     |
|   PostgreSQL            - Replay buffer (transitions)          |
|   ./data/models/        - ONNX model files                     |
|   ./data/stats.json     - Training telemetry                   |
+---------------------------------------------------------------+
         |                       |                       |
         v                       v                       v
+-----------------+    +-----------------+    +------------------+
|   Web Server    |    | Python Trainer  |    | Svelte Frontend  |
|   (Axum :8080)  |    | (Learner)       |    | (Vite :5173)     |
|   - Engine lib  |    | - PyTorch       |    | - Play UI        |
|   - Game API    |    | - PostgreSQL    |    | - Stats display  |
|   - Stats API   |    | - ONNX export   |    |                  |
+-----------------+    +-----------------+    +------------------+
```

PostgreSQL is used for the replay buffer (storing training transitions) while models
are stored on the filesystem. For cloud deployments, S3/MinIO can be used for model
storage.

## Quick Start

### Option 1: Local Development (Recommended)

Local development offers better performance and faster iteration. PostgreSQL is required for the replay buffer.

**Terminal 0** - Start PostgreSQL:
```bash
docker compose up postgres  # Or use a local PostgreSQL installation
```

**Terminal 1** - Start the Rust backend:
```bash
cd web && cargo run
# Server starts on http://localhost:8080
```

**Terminal 2** - Start the Svelte frontend:
```bash
cd web/frontend && npm install && npm run dev
# Dev server starts on http://localhost:5173
```

**Terminal 3** - Train a model:
```bash
cd trainer && pip install -e .
python -m trainer loop --iterations 50 --episodes 200 --steps 500
```

Open http://localhost:5173 to play!

### Option 2: Docker

Docker is convenient for quick testing but may have performance overhead.

```bash
# Train a model using synchronized AlphaZero loop
docker compose up alphazero

# Train Connect4 instead of TicTacToe
CARTRIDGE_COMMON_ENV_ID=connect4 docker compose up alphazero

# Play against the trained model
docker compose up web frontend
# Open http://localhost in browser
```

## Project Structure

```
cartridge2/
|-- actor/                     # Self-play episode runner
|   |-- src/
|   |   |-- main.rs            # Entry point
|   |   |-- actor.rs           # Episode loop
|   |   |-- config.rs          # CLI configuration
|   |   |-- game_config.rs     # Game-specific config from metadata
|   |   |-- health.rs          # Health check endpoint
|   |   |-- mcts_policy.rs     # MCTS policy implementation
|   |   |-- metrics.rs         # Prometheus metrics
|   |   |-- model_watcher.rs   # ONNX model hot-reload
|   |   |-- stats.rs           # Self-play statistics
|   |   +-- storage/           # Storage backends (PostgreSQL)
|   +-- tests/
|
|-- engine/                    # Rust workspace
|   |-- engine-config/         # Centralized configuration loading
|   |-- engine-core/           # Game trait + registry
|   |   +-- src/
|   |       |-- typed.rs       # Game trait definition
|   |       |-- adapter.rs     # Type erasure adapter
|   |       |-- erased.rs      # Erased game interface
|   |       |-- context.rs     # EngineContext API
|   |       |-- metadata.rs    # GameMetadata for game configuration
|   |       +-- registry.rs    # Static game registry
|   |-- engine-games/          # Game registration utilities
|   |-- games-tictactoe/       # TicTacToe implementation
|   |-- games-connect4/        # Connect 4 implementation
|   |-- mcts/                  # Monte Carlo Tree Search
|   +-- model-watcher/         # Shared model hot-reload library
|
|-- web/                       # HTTP server + frontend
|   |-- src/
|   |   |-- main.rs            # Axum server setup
|   |   |-- game.rs            # Session management
|   |   |-- metrics.rs         # Prometheus metrics
|   |   |-- model_watcher.rs   # ONNX model hot-reload
|   |   |-- handlers/          # HTTP endpoint handlers
|   |   +-- types/             # Request/response types
|   +-- frontend/              # Svelte application
|       +-- src/
|           |-- App.svelte
|           |-- GenericBoard.svelte     # Game board component
|           |-- LossChart.svelte        # Loss visualization chart
|           |-- LossOverTimePage.svelte # Training progress page
|           +-- Stats.svelte
|
|-- trainer/                   # Python training
|   |-- pyproject.toml         # Package configuration
|   +-- src/trainer/
|       |-- __main__.py        # CLI entrypoint
|       |-- trainer.py         # Training loop
|       |-- network.py         # Neural network (MLP)
|       |-- resnet.py          # ResNet architecture
|       |-- evaluator.py       # Model evaluation
|       |-- game_config.py     # Game-specific configurations
|       |-- stats.py           # Training statistics
|       |-- config.py          # TrainerConfig dataclass
|       |-- checkpoint.py      # Checkpoint utilities
|       |-- central_config.py  # Central config.toml loading
|       |-- games/             # Game-specific neural network configs
|       |-- orchestrator/      # Synchronized AlphaZero orchestrator
|       |-- policies/          # Policy implementations (ONNX, random)
|       +-- storage/           # Storage backends (PostgreSQL, S3, filesystem)
|
|-- data/                      # Runtime data (gitignored)
|   |-- models/                # ONNX checkpoints
|   +-- stats.json             # Training telemetry
|
|-- config.toml                # Central configuration
|-- docker-compose.yml         # Local mode services
+-- docker-compose.k8s.yml     # K8s mode services (PostgreSQL + MinIO)
```

## Components

### Engine Core (`engine/engine-core/`)

Pure Rust game logic library:

- **Game Trait** - Type-safe interface for implementing games
- **Type Erasure** - Runtime polymorphism via trait objects
- **Registry** - Static game registration system
- **EngineContext** - High-level API for game simulation

```rust
use engine_core::EngineContext;
use games_tictactoe::register_tictactoe;

// Register games at startup
register_tictactoe();

// Create context and play
let mut ctx = EngineContext::new("tictactoe").expect("game registered");
let reset = ctx.reset(42, &[]).unwrap();

// Take action (position 4 = center square)
let action = 4u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();
```

### Actor (`actor/`)

Self-play episode generator:

- Runs game simulations using `EngineContext`
- MCTS with ONNX neural network evaluation
- Hot-reloads model when `latest.onnx` changes
- Stores transitions in PostgreSQL replay buffer
- MCTS visit distributions saved as policy targets
- Game outcomes backfilled to all positions

```bash
# Requires PostgreSQL running (use docker compose up postgres)
cargo run -- --env-id tictactoe --max-episodes 10000
```

### Web Server (`web/`)

Axum HTTP server with endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/games` | GET | List available games |
| `/game-info/:id` | GET | Get game metadata |
| `/game/new` | POST | Start a new game |
| `/game/state` | GET | Get current board state |
| `/move` | POST | Make player move + get bot response |
| `/stats` | GET | Read training telemetry |
| `/model` | GET | Get info about loaded model |

### Python Trainer (`trainer/`)

PyTorch training loop:

- Reads transitions from PostgreSQL replay buffer
- AlphaZero-style loss (policy cross-entropy + value MSE)
- MCTS visit distributions as soft policy targets
- Game outcomes propagated as value targets
- Exports ONNX models with atomic write-then-rename
- Cosine annealing LR schedule
- Gradient clipping for stability
- Checkpoint management
- Model evaluation against random baseline

#### Synchronized AlphaZero training loop

The Python package now includes an orchestrated, synchronous AlphaZero workflow
that coordinates the actor, trainer, and post-iteration evaluation. This
pipeline clears the replay buffer each iteration, generates fresh self-play
episodes, trains on that data, and then benchmarks the resulting model against
the random baseline.

Run locally (with defaults targeting TicTacToe):

```bash
# Using the subcommand interface
trainer loop --iterations 5 --episodes 200 --steps 500
# Or: python -m trainer loop --iterations 5 --episodes 200 --steps 500

# Legacy entry point also works
trainer-loop --iterations 5 --episodes 200 --steps 500
```

Configuration can be supplied via flags or environment variables (prefixed with
`ALPHAZERO_` or `CARTRIDGE_`). For example, to train Connect4 with GPU acceleration
and disable evaluation for speed:

```bash
ALPHAZERO_ENV_ID=connect4 ALPHAZERO_DEVICE=cuda ALPHAZERO_EVAL_INTERVAL=0 \
    trainer loop --iterations 20 --episodes 300 --steps 1000
```

Docker usage mirrors the same interface:

```bash
docker compose up alphazero
# Override parameters as needed
CARTRIDGE_COMMON_ENV_ID=connect4 docker compose up alphazero
```

See [Deployment Modes](#deployment-modes) for K8s-style backends with PostgreSQL and MinIO.

## Deployment Modes

### Default Mode (PostgreSQL + Filesystem)

Uses PostgreSQL for replay buffer and filesystem for model storage. Docker Compose automatically starts PostgreSQL.

```bash
# Start synchronized training (PostgreSQL starts automatically)
docker compose up alphazero

# Start web UI to play against trained model
docker compose up web frontend
```

### Cloud Mode (PostgreSQL + S3)

Uses PostgreSQL for replay buffer and S3/MinIO for model storage. Enables distributed deployments.

```bash
# Start with MinIO for model storage
docker compose -f docker-compose.yml -f docker-compose.k8s.yml up

# Scale actors horizontally (4 parallel self-play workers)
docker compose -f docker-compose.yml -f docker-compose.k8s.yml up --scale actor=4

# Access MinIO web console at http://localhost:9001
```

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `CARTRIDGE_STORAGE_MODEL_BACKEND` | `filesystem` or `s3` | `filesystem` |
| `CARTRIDGE_STORAGE_POSTGRES_URL` | PostgreSQL connection string | `postgresql://cartridge:cartridge@localhost:5432/cartridge` |
| `CARTRIDGE_STORAGE_S3_BUCKET` | S3 bucket for models | - |
| `CARTRIDGE_STORAGE_S3_ENDPOINT` | S3-compatible endpoint (MinIO) | - |

## Adding a New Game

1. Create a new crate in `engine/games-{name}/`
2. Implement the `Game` trait:

```rust
pub trait Game {
    type State;   // Game state (must be Copy-friendly)
    type Action;  // Action space
    type Obs;     // Observation for neural networks

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8]) -> (State, Obs);
    fn step(&mut self, state: &mut State, action: Action, rng: &mut ChaCha20Rng)
        -> (Obs, f32, bool, u64);  // obs, reward, done, info_bits

    fn encode_state(state: &State, buf: &mut Vec<u8>) -> Result<(), Error>;
    fn decode_state(buf: &[u8]) -> Result<State, Error>;
    // ... similar for Action and Obs
}
```

3. Register the game:

```rust
use engine_core::{register_game, GameAdapter};

pub fn register_connect4() {
    register_game("connect4".to_string(), || {
        Box::new(GameAdapter::new(Connect4::new()))
    });
}
```

4. Add tests for game logic and encoding round-trips

## Development

### Build

```bash
# Build all Rust components
cd engine && cargo build --release
cd ../actor && cargo build --release
cd ../web && cargo build --release

# Install Python dependencies
cd trainer && pip install -e .
```

### Test

```bash
cd engine && cargo test    # ~172 tests
cd actor && cargo test     # 69 tests
cd web && cargo test       # 27 tests
cd trainer && pytest       # Python tests
```

### Format & Lint

```bash
cd engine && cargo fmt && cargo clippy
cd actor && cargo fmt && cargo clippy
cd web && cargo fmt && cargo clippy
```

## Current Status

**Core:**
- [x] Engine core abstractions (Game trait, adapter, registry, metadata)
- [x] EngineContext high-level API
- [x] TicTacToe game implementation
- [x] Connect 4 game implementation
- [x] MCTS implementation with ONNX evaluation

**Training:**
- [x] Actor (episode runner, MCTS + ONNX, model hot-reload)
- [x] Python trainer (PyTorch, ONNX export, cosine LR)
- [x] Synchronized AlphaZero training loop (orchestrator)
- [x] MCTS policy targets + game outcome propagation
- [x] Model evaluation against random baseline

**Storage Backends:**
- [x] PostgreSQL replay buffer (default)
- [x] Filesystem model storage (default)
- [x] S3/MinIO model storage (cloud mode)

**Web:**
- [x] Web server (Axum, game API)
- [x] Web frontend (Svelte, play UI, stats)
- [x] Loss visualization chart

**Deployment:**
- [x] Docker Compose (PostgreSQL, optional MinIO for S3)
- [x] Horizontal actor scaling via `--scale`

**Planned:**
- [ ] Othello game
- [ ] Kubernetes manifests (Helm/Kustomize)

## Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Architecture | Monolith + Python | Simplicity for MVP, easy local development |
| Deployment | Docker Compose | Simple orchestration, PostgreSQL included |
| Language | Rust + Python | Type safety + ML ecosystem |
| Game Interface | Typed trait + erasure | Compile-time safety + runtime flexibility |
| Replay Storage | PostgreSQL | Concurrent access, scales with multiple actors |
| Model Storage | Filesystem / S3 | Filesystem for local, S3/MinIO for distributed |
| Model Format | ONNX | Framework-agnostic, production-ready |
| RNG | ChaCha20 | Deterministic, reproducible simulations |

## License

MIT
