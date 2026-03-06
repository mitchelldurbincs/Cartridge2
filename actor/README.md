# Actor

Self-play episode runner for Cartridge2. Generates game experience data by running continuous episodes and storing transitions in PostgreSQL for training.

## Overview

The Actor is a long-running Rust binary that:

1. Runs game episodes using the engine-core library directly (no gRPC)
2. Selects actions using MCTS with ONNX neural network evaluation
3. Hot-reloads the model when `latest.onnx` changes
4. Stores transitions with MCTS policy distributions and game outcomes in PostgreSQL
5. Supports graceful shutdown via Ctrl+C
6. Can run multiple instances in parallel for faster data generation

## Quick Start

```bash
# Run with defaults (tictactoe, unlimited episodes)
cargo run

# Run 100 episodes with debug logging
cargo run -- --max-episodes 100 --log-level debug

# Run multiple actors in parallel
cargo run -- --actor-id actor-1 &
cargo run -- --actor-id actor-2 &
cargo run -- --actor-id actor-3 &
```

## CLI Arguments

| Argument | Env Variable | Default | Description |
|----------|--------------|---------|-------------|
| `--actor-id` | `ACTOR_ACTOR_ID` | `actor-1` | Unique identifier for this actor |
| `--env-id` | `ACTOR_ENV_ID` | `tictactoe` | Game environment to run |
| `--max-episodes` | `ACTOR_MAX_EPISODES` | `-1` | Max episodes (-1 = unlimited) |
| `--episode-timeout-secs` | `ACTOR_EPISODE_TIMEOUT` | `30` | Per-episode timeout |
| `--flush-interval-secs` | `ACTOR_FLUSH_INTERVAL` | `5` | Buffer flush interval |
| `--log-level` | `ACTOR_LOG_LEVEL` | `info` | Logging level |
| `--data-dir` | `ACTOR_DATA_DIR` | `./data` | Data directory for models |
| `--postgres-url` | `CARTRIDGE_STORAGE_POSTGRES_URL` | `postgresql://cartridge:cartridge@localhost:5432/cartridge` | PostgreSQL connection string |

Configuration is loaded from `config.toml` at the project root, with environment variables taking precedence.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        ACTOR                             │
├─────────────────────────────────────────────────────────┤
│  main()                                                  │
│    ├─ Parse CLI args (clap)                             │
│    ├─ Load config from config.toml                       │
│    ├─ Validate config                                   │
│    ├─ Load GameConfig from metadata                     │
│    ├─ Init model watcher                                │
│    ├─ Init tracing                                      │
│    └─ Run actor.run() async loop                        │
│                                                          │
│  run() async loop                                        │
│    ├─ Check shutdown flag                               │
│    ├─ Check episode limit                               │
│    └─ Run episode:                                      │
│        ├─ engine.reset()                                │
│        └─ loop:                                         │
│            ├─ policy.select_action(obs)                 │
│            ├─ engine.step(state, action)                │
│            ├─ replay.store(transition)                  │
│            └─ break if done                             │
└─────────────────────────────────────────────────────────┘
                             │
                             ▼
               PostgreSQL (cartridge.transitions)
```

## Components

### Actor (`src/actor.rs`)

Orchestrates the self-play loop:

```rust
pub struct Actor {
    config: Config,
    engine: Mutex<EngineContext>,      // Game simulation
    mcts_policy: Mutex<MctsPolicy>,    // Action selection
    replay: Mutex<ReplayStore>,       // PostgreSQL storage
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
}

impl Actor {
    pub fn new(config: Config) -> Result<Self>;
    pub async fn run(&self) -> Result<()>;
    pub fn shutdown(&self);
    pub fn episode_count(&self) -> u32;
}
```

### MCTS Policy (`src/mcts_policy.rs`)

Action selection is handled by `MctsPolicy`, which wraps the MCTS searcher and
ONNX evaluator:

- Runs Monte Carlo Tree Search with a shared `OnnxEvaluator` that can be hot-reloaded
- Falls back to uniform random over legal moves until a model is available
- Returns visit count distributions as policy targets for training

Supported action space:
- **Discrete(n)** - Single integer 0..n (encoded as 4-byte u32)

### Model Watcher (`src/model_watcher.rs`)

Hot-reload for ONNX models:
- Watches `./data/models/latest.onnx` for changes
- Automatically reloads the model when updated
- Thread-safe access via Arc<RwLock>

### Game Config (`src/game_config.rs`)

Game-specific configuration derived from GameMetadata:
- Auto-derives observation size, action space, legal mask offset
- Supports TicTacToe and Connect 4
- No hardcoded game parameters

### Replay Buffer (`src/storage/postgres.rs`)

PostgreSQL-backed storage for transitions:

```rust
pub struct PostgresReplayStore {
    pool: deadpool_postgres::Pool,
}

impl ReplayStore for PostgresReplayStore {
    async fn store(&self, transition: &Transition) -> Result<()>;
    async fn store_batch(&self, transitions: &[Transition]) -> Result<()>;
    async fn count(&self) -> Result<usize>;
    async fn clear(&self) -> Result<()>;
    async fn store_metadata(&self, metadata: &GameMetadata) -> Result<()>;
}
```

### Transition

Data structure for a single step:

```rust
pub struct Transition {
    pub id: String,              // Unique ID
    pub env_id: String,          // Game environment
    pub episode_id: String,      // Episode grouping
    pub step_number: u32,        // Step within episode
    pub state: Vec<u8>,          // Serialized state
    pub action: Vec<u8>,         // Serialized action
    pub next_state: Vec<u8>,     // Next state
    pub observation: Vec<u8>,    // Current observation
    pub next_observation: Vec<u8>,
    pub reward: f32,
    pub done: bool,
    pub timestamp: u64,
    pub policy_probs: Vec<u8>,   // MCTS visit distribution (f32 array)
    pub mcts_value: f32,         // MCTS value estimate
    pub game_outcome: Option<f32>, // Final outcome (+1/-1/0), backfilled
}
```

## Database Schema

```sql
CREATE TABLE transitions (
    id TEXT PRIMARY KEY,
    env_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    state BLOB NOT NULL,
    action BLOB NOT NULL,
    next_state BLOB NOT NULL,
    observation BLOB NOT NULL,
    next_observation BLOB NOT NULL,
    reward REAL NOT NULL,
    done INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    policy_probs BLOB,           -- f32[num_actions] MCTS visit distribution
    mcts_value REAL DEFAULT 0.0, -- MCTS value estimate
    game_outcome REAL            -- Final outcome, backfilled after episode
);

CREATE INDEX idx_transitions_timestamp ON transitions(timestamp);
CREATE INDEX idx_transitions_episode ON transitions(episode_id);
```

## Examples

### Basic Usage

```bash
# Default configuration (requires PostgreSQL running)
cargo run

# Specific actor ID and episode limit
cargo run -- --actor-id actor-1 --max-episodes 1000

# Custom PostgreSQL connection
cargo run -- --postgres-url "postgresql://user:pass@host:5432/cartridge"
```

### Environment Variables

```bash
export ACTOR_ACTOR_ID=distributed-1
export ACTOR_ENV_ID=tictactoe
export ACTOR_MAX_EPISODES=10000
export ACTOR_LOG_LEVEL=info
export CARTRIDGE_STORAGE_POSTGRES_URL="postgresql://user:pass@localhost:5432/cartridge"
cargo run
```

### Docker Compose (Recommended)

```bash
# Start PostgreSQL
docker compose up postgres -d

# Run actor
cargo run
```

### Multiple Actors

```bash
# Run 4 actors in parallel
for i in {1..4}; do
  cargo run -- --actor-id "actor-$i" &
done
wait
```

### Integration with Trainer

```bash
# Terminal 1: Start PostgreSQL
docker compose up postgres -d

# Terminal 2: Actor generates data
cargo run -- --actor-id actor-1

# Terminal 3: Python trainer consumes data
cd ../trainer
# Uses CARTRIDGE_STORAGE_POSTGRES_URL from environment or config.toml
python -m trainer train
```

## Testing

```bash
# Run all tests (29 tests)
cargo test

# Run specific test with output
cargo test test_actor_run_single_episode -- --nocapture

# Run benchmarks
cargo bench
```

### Test Coverage

- **Game Config tests** - Game-specific configuration from metadata
- **Health tests** - Health state management and progress tracking
- **MCTS Policy tests** - MCTS-based action selection and random fallback
- **Metrics tests** - Prometheus metrics recording
- **Stats tests** - Episode statistics tracking

## Dependencies

| Crate | Purpose |
|-------|---------|
| `engine-core` | Game simulation API |
| `engine-config` | Centralized configuration |
| `engine-games` | Game registration |
| `mcts` | Monte Carlo Tree Search |
| `tokio-postgres` | PostgreSQL client |
| `deadpool-postgres` | PostgreSQL connection pool |
| `tokio` | Async runtime |
| `clap` | CLI argument parsing |
| `rand_chacha` | Deterministic RNG |
| `tracing` | Structured logging |
| `notify` | File watching for model hot-reload |
| `prometheus` | Metrics collection |

## Performance

- **Zero-copy buffers** - Efficient handling of state/action/observation data
- **Async I/O** - Non-blocking episode execution
- **Deterministic RNG** - ChaCha20Rng for reproducible randomness
- **Pooled PostgreSQL writes** - Concurrent actor-safe replay buffer ingestion

## Troubleshooting

### Common Issues

1. **Unknown env_id** - Ensure game is registered with engine-core
2. **PostgreSQL connection failed** - Check `CARTRIDGE_STORAGE_POSTGRES_URL` and that PostgreSQL is running
3. **Permission denied** - Check write permissions on data directory

### Debug Mode

```bash
# Enable debug logging
RUST_LOG=debug cargo run -- --log-level debug

# Check database connection
docker compose ps postgres
docker compose logs postgres
```

## Future Work

- [x] MCTS policy implementation
- [x] ONNX neural network policy
- [x] Model hot-reload via file watching
- [x] Auto-derived game configuration from GameMetadata
- [ ] Distributed actor coordination
- [ ] Priority experience replay
