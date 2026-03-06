# Cartridge2 Architecture Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Engine Component](#3-engine-component)
4. [Actor Component](#4-actor-component)
5. [Trainer Component](#5-trainer-component)
6. [Web Component](#6-web-component)
7. [Storage Backends](#7-storage-backends)
8. [Configuration System](#8-configuration-system)
9. [Data Flow](#9-data-flow)
10. [Deployment](#10-deployment)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Overview

Cartridge2 is a simplified AlphaZero training and visualization platform that enables training neural network game agents via self-play and lets users play against trained models through a web interface.

### Key Design Philosophy

- **Monolithic over Microservices**: Shared filesystem and local processes instead of Kubernetes/gRPC
- **Library-First**: Engine is a Rust library, not a service
- **Self-Describing**: Database stores game metadata for automatic configuration
- **Hot-Reloadable**: Models update without restarting services

### Target Games

| Game | Status | Board | Actions | Network |
|------|--------|-------|---------|---------|
| TicTacToe | Complete | 3x3 | 9 | MLP |
| Connect 4 | Complete | 7x6 | 7 | ResNet |
| Othello | Planned | 8x8 | 64 | ResNet |

### Technology Stack

| Layer | Technology |
|-------|------------|
| Game Engine | Rust (engine-core, games-*) |
| Self-Play | Rust (actor) |
| Training | Python (PyTorch, ONNX) |
| Backend API | Rust (Axum) |
| Frontend | Svelte 5 + TypeScript |
| Storage | PostgreSQL + Filesystem/S3 |

---

## 2. System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Shared Filesystem / Storage                        │
│  PostgreSQL                 - Replay buffer database                         │
│  ./data/models/latest.onnx - Current ONNX model (hot-reloaded)              │
│  ./data/stats.json         - Training telemetry for web UI                  │
└─────────────────────────────────────────────────────────────────────────────┘
         ▲                   ▲                   ▲                    ▲
         │                   │                   │                    │
┌────────┴────────┐  ┌───────┴───────┐  ┌────────┴────────┐  ┌────────┴────────┐
│     Actor       │  │    Trainer    │  │   Web Server    │  │    Frontend     │
│  (Rust Binary)  │  │   (Python)    │  │  (Axum :8080)   │  │  (Svelte :5173) │
│                 │  │               │  │                 │  │                 │
│  - Engine lib   │  │  - PyTorch    │  │  - Engine lib   │  │  - Play UI      │
│  - MCTS policy  │  │  - PostgreSQL │  │  - Game API     │  │  - Stats charts │
│  - Self-play    │  │  - ONNX export│  │  - Stats API    │  │  - Loss display │
│  - Model watch  │  │  - Evaluation │  │  - Model watch  │  │                 │
└─────────────────┘  └───────────────┘  └─────────────────┘  └─────────────────┘
```

### Component Interactions

```
                                     ┌──────────────┐
                                     │   Browser    │
                                     │   (User)     │
                                     └──────┬───────┘
                                            │ HTTP
                                     ┌──────▼───────┐
                                     │   Frontend   │
                                     │   (Svelte)   │
                                     └──────┬───────┘
                                            │ API calls
                                     ┌──────▼───────┐
                                     │  Web Server  │
                                     │   (Axum)     │
                                     └──────┬───────┘
                                            │ reads
                              ┌─────────────┼─────────────┐
                              │             │             │
                       ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
                       │ latest.onnx │ │stats.json│ │Engine (lib) │
                       └──────▲──────┘ └────▲────┘ └─────────────┘
                              │             │
              writes (atomic) │             │ writes (atomic)
                              │             │
                       ┌──────┴──────┐ ┌────┴─────────────┐
                       │   Trainer   │ │                  │
                       │  (Python)   │ │                  │
                       └──────▲──────┘ │                  │
                              │ samples│                  │
                       ┌──────┴──────┐ │                  │
                       │  PostgreSQL │◄┘                  │
                       │  (Database) │                    │
                       └──────▲──────┘                    │
                              │ stores transitions        │
                       ┌──────┴──────┐                    │
                       │    Actor    │────────────────────┘
                       │   (Rust)    │  uses Engine lib
                       └──────┬──────┘
                              │ reads (hot-reload)
                       ┌──────▼──────┐
                       │ latest.onnx │
                       └─────────────┘
```

### Training Modes

#### Synchronized AlphaZero (Recommended)

Each iteration follows this pattern:
```
┌──────────────────────────────────────────────────────────────┐
│ Iteration N                                                  │
├──────────────────────────────────────────────────────────────┤
│ 1. Clear replay buffer (fresh data from current model)       │
│ 2. Run Actor: Generate N episodes via self-play              │
│ 3. Run Trainer: Train for M steps on fresh data              │
│ 4. Run Evaluator: Test against best model + random           │
│ 5. Promote if win_rate > threshold                           │
│ 6. Export latest.onnx (actor hot-reloads)                    │
└──────────────────────────────────────────────────────────────┘
```

#### Continuous Mode (Alternative)

Actor and trainer run concurrently:
```
┌─────────────────┐     ┌─────────────────┐
│     Actor       │     │    Trainer      │
│  (continuous)   │     │  (continuous)   │
│                 │     │                 │
│  Generate ──────┼────►│  Sample ────────┤
│  episodes       │     │  batches        │
│                 │◄────┼──────────────── │
│  Hot-reload     │     │  Export ONNX    │
│  model          │     │                 │
└─────────────────┘     └─────────────────┘
```

---

## 3. Engine Component

### Directory Structure

```
engine/
├── Cargo.toml                 # Workspace root
├── engine-core/               # Core abstractions (Game trait, registry, context)
│   └── src/
│       ├── lib.rs             # Public API exports
│       ├── typed.rs           # Game trait (compile-time type safety)
│       ├── erased.rs          # ErasedGame trait (runtime polymorphism)
│       ├── adapter.rs         # GameAdapter (typed → erased conversion)
│       ├── context.rs         # EngineContext high-level API
│       ├── registry.rs        # Static game registration
│       ├── metadata.rs        # GameMetadata for UI/config
│       └── board_game.rs      # TwoPlayerObs generic type
├── games-tictactoe/           # TicTacToe implementation
├── games-connect4/            # Connect 4 implementation
├── mcts/                      # Monte Carlo Tree Search
│   └── src/
│       ├── config.rs          # MctsConfig parameters
│       ├── evaluator.rs       # Evaluator trait + UniformEvaluator
│       ├── node.rs            # MctsNode (visit stats)
│       ├── tree.rs            # Arena-allocated tree
│       ├── search.rs          # Select/expand/backprop algorithm
│       └── onnx.rs            # OnnxEvaluator (feature-gated)
└── model-watcher/             # ONNX hot-reload utility
```

### Core Abstractions

#### Game Trait (Typed)

The typed `Game` trait provides compile-time type safety:

```rust
pub trait Game: Send + Sync + Debug + 'static {
    type State;      // Game state (e.g., board configuration)
    type Action;     // Action type (e.g., position index)
    type Obs;        // Observation (neural network input)

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8])
        -> (Self::State, Self::Obs);
    fn step(&mut self, state: &mut Self::State, action: Self::Action,
        rng: &mut ChaCha20Rng) -> (Self::Obs, f32, bool, u64);

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<()>;
    fn decode_state(buf: &[u8]) -> Result<Self::State>;
    // ... encode/decode for Action and Obs
}
```

#### ErasedGame Trait (Runtime)

The erased trait enables runtime polymorphism with byte-only interface:

```rust
pub trait ErasedGame: Send + Sync + Debug + 'static {
    fn reset(&mut self, seed: u64, hint: &[u8],
        out_state: &mut Vec<u8>, out_obs: &mut Vec<u8>) -> Result<()>;
    fn step(&mut self, state: &[u8], action: &[u8],
        out_state: &mut Vec<u8>, out_obs: &mut Vec<u8>)
        -> Result<(f32, bool, u64)>;
}
```

#### GameAdapter Pattern

Converts typed games to erased interface:

```
Typed Game (State, Action, Obs)
        ↓
GameAdapter<T: Game>
    ├─ Wraps game instance
    ├─ Manages RNG (re-seeded on reset)
    └─ Handles encode/decode
        ↓
ErasedGame trait (bytes-only)
        ↓
Registry storage
```

#### Registry System

Static compile-time registration with runtime lookup:

```rust
// Registration (called at startup)
pub fn register_tictactoe() {
    register_game("tictactoe".to_string(), || {
        Box::new(GameAdapter::new(TicTacToe::new()))
    });
}

// Lookup (at runtime)
let game = create_game("tictactoe")?;
```

#### EngineContext API

High-level convenience wrapper:

```rust
let mut ctx = EngineContext::new("tictactoe")?;
let reset = ctx.reset(42, &[])?;           // seed=42
let step = ctx.step(&reset.state, &action)?;
println!("Reward: {}, Done: {}", step.reward, step.done);
```

### MCTS Implementation

#### Configuration

```rust
pub struct MctsConfig {
    pub num_simulations: u32,      // Default: 800
    pub c_puct: f32,               // UCB exploration (default: 1.25)
    pub dirichlet_alpha: f32,      // Root noise (default: 0.3)
    pub dirichlet_epsilon: f32,    // Noise weight (default: 0.25)
    pub temperature: f32,          // Action selection (default: 1.0)
}

// Presets
MctsConfig::for_training()    // With exploration noise
MctsConfig::for_evaluation()  // Greedy, no noise
MctsConfig::for_testing()     // Fast (50 sims)
```

#### Search Algorithm

Per simulation:
1. **Selection**: Traverse using UCB = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
2. **Expansion**: Add children with policy prior from neural network
3. **Evaluation**: Call evaluator.evaluate(obs) for policy/value
4. **Backpropagation**: Update visit counts and value sums

#### Evaluator Trait

```rust
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, obs: &[u8], legal_mask: u64, num_actions: usize)
        -> Result<EvalResult>;
}

pub struct EvalResult {
    pub policy: Vec<f32>,   // Probability distribution
    pub value: f32,         // -1 (loss) to +1 (win)
}
```

Implementations:
- `UniformEvaluator`: Equal probability for legal moves (testing)
- `OnnxEvaluator`: Neural network inference via ONNX Runtime

### Game Implementations

#### TicTacToe

| Property | Value |
|----------|-------|
| Board | 3x3 grid |
| Actions | 9 positions |
| Observation | 29 f32s (18 board + 9 legal + 2 player) |
| Network | MLP |
| Board Type | "grid" |

#### Connect 4

| Property | Value |
|----------|-------|
| Board | 7x6 grid (column-drop) |
| Actions | 7 columns |
| Observation | 93 f32s (84 board + 7 legal + 2 player) |
| Network | ResNet (4 blocks, 128 filters) |
| Board Type | "drop_column" |

### Model Watcher

Hot-reload system for ONNX models:

```
Trainer exports model:
  1. Write to latest.onnx.tmp
  2. Atomic rename to latest.onnx

ModelWatcher detects:
  1. inotify event (or polling fallback)
  2. Load new ONNX model
  3. Acquire write lock on evaluator
  4. Atomic swap
  5. Signal subscribers
```

Features:
- Dual strategy: inotify + polling (Docker compatibility)
- Atomic model loading (no partial loads)
- Concurrent-safe via Arc<RwLock<>>

---

## 4. Actor Component

### Directory Structure

```
actor/
├── Cargo.toml
└── src/
    ├── main.rs            # Entry point, CLI parsing
    ├── actor.rs           # Episode runner, main loop
    ├── mcts_policy.rs     # MCTS action selection
    ├── model_watcher.rs   # Re-export from engine
    ├── game_config.rs     # Auto-derived game config
    ├── central_config.rs  # config.toml loading
    └── storage/
        ├── mod.rs         # ReplayStore trait
        └── postgres.rs    # PostgreSQL backend
```

### Actor Struct

```rust
pub struct Actor {
    config: Config,
    game_config: GameConfig,
    engine: Mutex<EngineContext>,
    mcts_policy: Mutex<MctsPolicy>,
    replay: Arc<dyn ReplayStore>,
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
    model_watcher: ModelWatcher,
}
```

### Episode Execution Flow

```
actor.run_episode():
  1. Reset game with random seed
  2. Loop while !done:
     a. Lock policy
     b. Select action via MCTS (or random if no model)
     c. Unlock policy
     d. Lock engine
     e. Execute action, get next state
     f. Unlock engine
     g. Create Transition with MCTS policy
  3. Backfill game outcomes:
     For each transition:
       steps_from_end = total_steps - step_number - 1
       outcome = final_reward × (-1)^steps_from_end
  4. Batch store to replay buffer
  5. Return (steps, reward)
```

### MCTS Policy

```rust
pub struct MctsPolicy {
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,  // Hot-swappable
    config: MctsConfig,
    base_temperature: f32,
    late_temperature: f32,
    temp_threshold: u32,  // Move number to switch temps
}
```

Temperature schedule:
- Before threshold: `temperature = 1.0` (exploration)
- After threshold: `temperature = 0.1` (exploitation)

Fallback behavior:
- If no model loaded: Random legal action with uniform policy

### Transition Data

```rust
pub struct Transition {
    pub id: String,                // "{episode_id}-step-{n}"
    pub env_id: String,
    pub episode_id: String,
    pub step_number: u32,
    pub state: Vec<u8>,
    pub action: Vec<u8>,           // u32 little-endian
    pub next_state: Vec<u8>,
    pub observation: Vec<u8>,      // f32[obs_size]
    pub next_observation: Vec<u8>,
    pub reward: f32,
    pub done: bool,
    pub timestamp: u64,
    pub policy_probs: Vec<u8>,     // f32[num_actions] MCTS distribution
    pub mcts_value: f32,
    pub game_outcome: Option<f32>, // Backfilled after episode
}
```

### Game Configuration Auto-Derivation

```rust
pub struct GameConfig {
    pub num_actions: usize,
    pub obs_size: usize,
    pub legal_mask_offset: usize,
}

// Auto-derived from GameMetadata
let config = GameConfig::from_context(&ctx);
```

---

## 5. Trainer Component

### Directory Structure

```
trainer/
├── pyproject.toml
└── src/trainer/
    ├── __main__.py        # CLI (train, evaluate, loop)
    ├── trainer.py         # Training loop
    ├── orchestrator.py    # Synchronized AlphaZero
    ├── network.py         # MLP architecture
    ├── resnet.py          # ResNet architecture
    ├── evaluator.py       # Model evaluation
    ├── config.py          # TrainerConfig
    ├── game_config.py     # Game configurations
    ├── checkpoint.py      # ONNX + PyTorch save/load
    ├── stats.py           # Statistics tracking
    ├── lr_scheduler.py    # Warmup + cosine annealing
    ├── backoff.py         # Wait utilities
    ├── central_config.py  # config.toml loading
    └── storage/
        ├── base.py        # Abstract interfaces
        ├── factory.py     # Backend factory
        ├── postgres.py    # PostgreSQL implementation
        ├── s3.py          # S3 model storage
        └── filesystem.py  # Filesystem model storage
```

### CLI Commands

```bash
# Standalone training
python -m trainer train --steps 1000

# Model evaluation
python -m trainer evaluate --model ./data/models/latest.onnx --games 100

# Synchronized AlphaZero (recommended)
python -m trainer loop --iterations 50 --episodes 500 --steps 1000
```

### Network Architectures

#### MLP (TicTacToe)

```
Input (obs_size)
  → FC(128) → ReLU
  → FC(128) → ReLU
  → FC(64) → ReLU

  Policy Head: FC(num_actions)
  Value Head: FC(32) → ReLU → FC(1) → Tanh
```

#### ResNet (Connect4, Othello)

```
Input reshaped to (batch, channels, height, width)
  → Initial Conv(3x3) → BN → ReLU
  → N Residual Blocks:
      Conv(3x3) → BN → ReLU → Conv(3x3) → BN → + skip → ReLU

  Policy Head: Conv(1x1) → BN → ReLU → Flatten → Linear(num_actions)
  Value Head: Conv(1x1) → BN → ReLU → Flatten → Linear → ReLU → Linear(1) → Tanh
```

### Loss Function

AlphaZero combined loss:
```
Loss = value_weight × MSE(v, z) + policy_weight × CrossEntropy(p, π)

Where:
  v = predicted value
  z = target value (game outcome: +1/-1/0)
  p = predicted policy (softmax)
  π = target policy (MCTS visit distribution)
```

### Learning Rate Schedule

```
Phase 1: Warmup (linear ramp)
  LR: warmup_start_lr → target_lr

Phase 2: Cosine Annealing
  LR: target_lr → min_lr (following cosine curve)

LR
^
|        target_lr
|       /──────────.
|      /            '.
|     /               '..
|    /                   ''..._____ min_lr
|   /
+──┼─────────────────────────────>  step
   0   warmup                  total_steps
```

### Orchestrator (Synchronized Training)

```python
class LoopConfig:
    iterations: int = 100
    episodes_per_iteration: int = 500
    steps_per_iteration: int = 1000

    # MCTS simulation ramping
    mcts_start_sims: int = 200
    mcts_max_sims: int = 800
    mcts_sim_ramp_rate: int = 30

    # Evaluation gatekeeper
    eval_interval: int = 1
    eval_games: int = 50
    eval_win_threshold: float = 0.55
```

MCTS ramping formula:
```
sims = min(start_sims + (iter-1) × ramp_rate, max_sims)
```

### Checkpoint System

Two-file approach:
1. **ONNX** (`model_step_XXXXXX.onnx`): For actor inference
2. **PyTorch** (`latest.pt`): For training continuity

Atomic write-then-rename pattern prevents partial reads.

### Statistics Tracking

```python
@dataclass
class TrainerStats:
    step: int
    total_steps: int
    total_loss: float
    policy_loss: float
    value_loss: float
    learning_rate: float
    replay_buffer_size: int
    history: List[Dict]      # Per-interval entries (downsampled)
    eval_history: List[Dict] # Evaluation results
```

History downsampling:
- Recent (<1000 steps): Keep every entry
- Medium (1000-10000): Keep every 100th
- Old (>10000): Keep every 500th

---

## 6. Web Component

### Directory Structure

```
web/
├── Cargo.toml
├── src/
│   ├── main.rs            # Server setup, routing
│   ├── game.rs            # GameSession management
│   ├── central_config.rs  # Config loading
│   ├── model_watcher.rs   # Re-export
│   ├── handlers/
│   │   ├── game.rs        # Game endpoints
│   │   ├── health.rs      # Health check
│   │   └── stats.rs       # Training stats
│   └── types/
│       ├── requests.rs    # Request DTOs
│       └── responses.rs   # Response DTOs
└── frontend/
    ├── package.json
    └── src/
        ├── main.ts        # Router
        ├── App.svelte     # Main game page
        ├── GenericBoard.svelte    # Board rendering
        ├── Stats.svelte           # Training stats
        ├── LossChart.svelte       # Loss visualization
        └── lib/
            └── api.ts     # API client
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/games` | GET | List available games |
| `/game-info/:id` | GET | Game metadata |
| `/game/new` | POST | Start new game |
| `/game/state` | GET | Get current board |
| `/move` | POST | Make player move + bot response |
| `/stats` | GET | Training statistics |
| `/model` | GET | Model info |

### GameSession

```rust
pub struct GameSession {
    ctx: EngineContext,
    metadata: GameMetadata,
    state: Vec<u8>,
    obs: Vec<u8>,
    board: Vec<u8>,
    current_player: u8,
    winner: u8,
    human_player: u8,
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    mcts_config: MctsConfig,
}
```

Bot AI:
1. Extract legal moves from observation
2. If model loaded: Run MCTS (200 sims, temp=0.5)
3. If no model: Random legal move
4. Execute selected action

### Frontend Components

| Component | Purpose |
|-----------|---------|
| `App.svelte` | Main page with game + stats |
| `GenericBoard.svelte` | Renders grid or drop-column boards |
| `Stats.svelte` | Training metrics, model status |
| `LossChart.svelte` | Loss curve visualization |
| `LossOverTimePage.svelte` | Full-screen interactive charts |

Features:
- Auto-detect trained game from stats.json
- Game switcher dropdown
- 5-second stats polling
- Responsive dark theme

---

## 7. Storage Backends

### Replay Buffer (PostgreSQL)

```python
replay = create_replay_buffer(
    backend="postgres",
    connection_string="postgresql://user:pass@host:5432/db"
)
```

Schema:
```sql
CREATE TABLE transitions (
    id TEXT PRIMARY KEY,
    env_id TEXT,
    episode_id TEXT,
    step_number INTEGER,
    state BYTEA,
    action BYTEA,
    observation BYTEA,
    policy_probs BYTEA,
    mcts_value REAL,
    game_outcome REAL,
    ...
);

CREATE TABLE game_metadata (
    env_id TEXT PRIMARY KEY,
    num_actions INTEGER,
    obs_size INTEGER,
    ...
);
```

Supports concurrent writers from multiple actors.

### Model Storage

#### Filesystem (Local)

```
./data/models/
├── latest.onnx           # Current model (hot-reloaded)
├── best.onnx             # Best model (gatekeeper)
├── model_step_000100.onnx
├── model_step_000200.onnx
└── ...
```

#### S3 (Kubernetes)

```python
store = create_model_store(
    backend="s3",
    bucket="cartridge-models",
    endpoint="http://minio:9000"  # Optional, for MinIO
)
```

---

## 8. Configuration System

### Priority (Highest to Lowest)

1. **CLI arguments**: `--env-id connect4`
2. **Environment variables**: `CARTRIDGE_COMMON_ENV_ID=connect4`
3. **config.toml file**: `env_id = "tictactoe"`
4. **Built-in defaults**: `"tictactoe"`

### config.toml Structure

```toml
[common]
data_dir = "./data"
env_id = "tictactoe"
log_level = "info"

[training]
iterations = 100
episodes_per_iteration = 500
steps_per_iteration = 1000
batch_size = 64
learning_rate = 0.001
device = "cpu"

[evaluation]
interval = 1
games = 50
win_threshold = 0.55

[actor]
actor_id = "actor-1"
max_episodes = -1
log_interval = 50

[mcts]
num_simulations = 800
c_puct = 1.4
temperature = 1.0
temp_threshold = 15

[web]
host = "0.0.0.0"
port = 8080

[storage]
replay_backend = "postgres"
model_backend = "filesystem"
postgres_url = "postgresql://..."
# s3_bucket = "cartridge-models"
# s3_endpoint = "http://minio:9000"
```

### Environment Variable Format

New format (preferred):
```bash
CARTRIDGE_COMMON_ENV_ID=connect4
CARTRIDGE_TRAINING_ITERATIONS=50
CARTRIDGE_MCTS_NUM_SIMULATIONS=800
```

Legacy format (supported):
```bash
ALPHAZERO_ENV_ID=connect4
ALPHAZERO_ITERATIONS=50
```

### Search Paths

1. `$CARTRIDGE_CONFIG` (if set)
2. `./config.toml`
3. `../config.toml`
4. `/app/config.toml` (Docker)

---

## 9. Data Flow

### Self-Play Data Flow

```
Actor                    Storage                  Trainer
  │                         │                        │
  │  1. Run episode         │                        │
  │  (MCTS + model)         │                        │
  │                         │                        │
  │  2. Backfill outcomes   │                        │
  │                         │                        │
  │  3. Store transitions ──►│                       │
  │                         │                        │
  │                         │◄── 4. Sample batch ────│
  │                         │                        │
  │                         │                        │  5. Train step
  │                         │                        │
  │◄───────────────────────────── 6. Export ONNX ───│
  │  7. Hot-reload model    │                        │
  │                         │                        │
```

### Statistics Flow

```
Trainer                 Filesystem              Web Server            Frontend
   │                        │                       │                     │
   │  Write stats.json ────►│                       │                     │
   │  (atomic)              │                       │                     │
   │                        │◄── Poll (on request) ─│                     │
   │                        │                       │                     │
   │                        │    Read stats.json ──►│                     │
   │                        │                       │                     │
   │                        │                       │◄── GET /stats ──────│
   │                        │                       │                     │
   │                        │                       │    JSON response ───►│
   │                        │                       │                     │
   │                        │                       │                     │ Render
```

### Model Promotion Flow

```
Evaluator evaluates model
        │
        ▼
  win_rate > threshold?
        │
   Yes  │  No
        │   │
        ▼   │
Copy to best.onnx
        │   │
        │   ▼
        │  Keep previous best
        │
        ▼
Update best_model.json
```

---

## 10. Deployment

### Docker Compose Services

#### Local Development

```bash
# Synchronized AlphaZero training
docker compose up alphazero

# Play in browser
docker compose up web frontend
# Open http://localhost
```

#### Kubernetes Simulation

```bash
# Full K8s stack (PostgreSQL + MinIO)
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up

# Scale actors
docker compose --profile k8s up --scale actor-k8s=4
```

### Dockerfiles

| Image | Base | Purpose | Size |
|-------|------|---------|------|
| `alphazero` | Python 3.11 | Actor + Trainer | ~1.3GB |
| `actor` | Debian slim | Self-play only | ~150MB |
| `trainer` | Python 3.11 | Training only | ~1.2GB |
| `web` | Debian slim | API server | ~200MB |
| `frontend` | nginx unprivileged | Svelte UI | ~50MB |

### Feature Flags

Actor and web support optional features:
- `postgres`: PostgreSQL replay backend
- `s3`: S3 model storage

Build with features:
```bash
docker build --build-arg CARGO_FEATURES="postgres,s3" -t cartridge-actor-k8s ./actor
```

### Health Checks

| Service | Check |
|---------|-------|
| postgres | `pg_isready -U cartridge` |
| minio | `mc ready local` |
| web | `curl -f http://localhost:8080/health` |
| frontend | `wget -q --spider http://localhost:8080/` |

---

## 11. Testing Strategy

### Test Distribution

| Component | Tests | Coverage |
|-----------|-------|----------|
| engine-core | 64 | Game trait, adapter, registry, context |
| games-tictactoe | 27 | Game logic, encoding |
| games-connect4 | 21 | Game logic, encoding |
| mcts | 22 | Config, evaluator, tree, search |
| actor | 36 | Initialization, episodes, storage |
| web | 22 | Handlers, game session |
| trainer | ~20 | CLI, stats, smoke tests |

### Running Tests

```bash
# Rust tests
cd engine && cargo test     # 134 tests
cd actor && cargo test      # 36 tests
cd web && cargo test        # 22 tests

# Python tests
cd trainer && pytest

# All Rust formatting and linting
cargo fmt --check
cargo clippy -- -D warnings
```

### CI Pipeline

GitHub Actions workflow:
1. **rust-fmt**: Format check
2. **rust-clippy**: Lint with warnings as errors
3. **rust-test**: Full test suite
4. **python-lint**: Ruff + Black
5. **python-test**: Pytest
6. **frontend**: Svelte check + build
7. **docker**: Dockerfile validation

---

## Appendix: Key Design Patterns

### Type Erasure via Adapter

```
Typed Game<State, Action, Obs>
        ↓
GameAdapter<T: Game>
        ↓
Box<dyn ErasedGame>
        ↓
Registry HashMap<String, Factory>
```

### Arena Allocation (MCTS)

```rust
struct MctsTree {
    nodes: Vec<MctsNode>,  // Contiguous arena
    root: NodeId,          // NodeId(0)
}

struct NodeId(u32);  // Index into arena
```

Benefits: Cache locality, pointer-like references without lifetimes.

### Builder Pattern

```rust
let config = MctsConfig::for_training()
    .with_simulations(800)
    .with_temperature(1.0);

let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
    .with_board(3, 3)
    .with_actions(9);
```

### Atomic File Operations

Write-then-rename pattern:
```
1. Write to file.tmp
2. Atomic rename file.tmp → file
```

Ensures readers never see partial content.

### Dual Hot-Reload Strategy

```
inotify watcher (fast, event-based)
        ↓
    fallback
        ↓
Polling timer (reliable in Docker)
```

---

## Appendix: Quick Reference

### Common Commands

```bash
# Local training
python -m trainer loop --iterations 50 --episodes 500 --steps 1000

# Docker training
docker compose up alphazero

# Play in browser
docker compose up web frontend

# Evaluation
python -m trainer evaluate --model ./data/models/latest.onnx --games 100

# Clean local artifacts (models + stats)
rm -rf ./data/models/*.onnx ./data/stats.json ./data/loop_stats.json ./data/eval_stats.json ./data/best_model.json

# Clean PostgreSQL replay buffer volume (removes all compose volumes)
docker compose down -v
```

### File Locations

| File | Purpose |
|------|---------|
| `config.toml` | Central configuration |
| `data/models/latest.onnx` | Current model |
| `data/models/best.onnx` | Best model |
| `data/stats.json` | Training statistics |
| `data/loop_stats.json` | Orchestrator history |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CARTRIDGE_COMMON_ENV_ID` | Game to train |
| `CARTRIDGE_TRAINING_DEVICE` | cpu, cuda, mps |
| `CARTRIDGE_MCTS_NUM_SIMULATIONS` | MCTS simulations |
| `CARTRIDGE_STORAGE_MODEL_BACKEND` | filesystem, s3 |
| `CARTRIDGE_STORAGE_POSTGRES_URL` | PostgreSQL connection |
