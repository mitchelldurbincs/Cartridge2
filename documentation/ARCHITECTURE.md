# Cartridge2 Architecture Documentation

This is the reference for how Cartridge2 is built. Where a fact has an
authoritative home in the repo — config defaults, the SQL schema, test counts —
this document links to it rather than restating it, because every previous
restatement had drifted.

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Engine Component](#3-engine-component) — incl. [Generals 8x8](#36-generals-8x8)
4. [Actor Component](#4-actor-component)
5. [Trainer Component](#5-trainer-component)
6. [Web Component](#6-web-component)
7. [Storage Backends](#7-storage-backends) — incl. [game metadata single-sourcing](#73-game-metadata-single-sourcing)
8. [Configuration System](#8-configuration-system)
9. [Data Flow](#9-data-flow)
10. [Deployment](#10-deployment) — incl. [Observability](#observability)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Overview

Cartridge2 is a simplified AlphaZero training and visualization platform that enables training neural network game agents via self-play and lets users play against trained models through a web interface.

### Key Design Philosophy

- **Monolithic over Microservices**: local processes over shared storage, instead of gRPC between services. (Kubernetes manifests and Terraform modules do exist under `k8s/` and `terraform/` for cloud deployment — what is avoided is service-to-service RPC, not orchestration.)
- **Library-First**: Engine is a Rust library, not a service
- **Engine as source of truth**: game facts — board dimensions, action counts, observation layout — are declared once in the Rust game crates. The Python trainer reads them from a generated manifest, and the database row is a cross-check, not a second definition. See [§7.3](#73-game-metadata-single-sourcing).
- **Hot-Reloadable**: Models update without restarting services

### Target Games

| Game | Status | Board | Actions | Obs size | Network |
|------|--------|-------|---------|----------|---------|
| TicTacToe | Complete | 3x3 | 9 | 29 | MLP (hidden 128) |
| Connect 4 | Complete | 7x6 | 7 | 93 | ResNet 4x128 |
| Othello | Complete | 8x8 | 65 (64 cells + pass) | 195 | ResNet 6x256 |
| Generals 8x8 | Engine + trainer complete; not yet stronger than random; no web renderer | 8x8 | 257 (64 tiles x 4 dirs + wait) | 835 | ResNet 6x128, 9 planes |

See [§3.6](#36-generals-8x8) for the Generals ruleset and why it departs from
the real game.

### Technology Stack

| Layer | Technology |
|-------|------------|
| Game Engine | Rust (engine-core, engine-config, engine-games, games-*) |
| Search | Rust (mcts) |
| Self-Play | Rust (actor) |
| Training | Python (PyTorch, ONNX) + [`crucible`](https://github.com/mitchelldurbincs/crucible) for orchestration |
| Backend API | Rust (Axum) |
| Frontend | Svelte 5 + TypeScript |
| Storage | PostgreSQL + Filesystem/S3 |
| Observability | Prometheus, structured JSON logs, Weights & Biases |

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
├── engine-config/             # Centralized config.toml loading (shared by actor/web)
├── engine-core/               # Core abstractions (Game trait, registry, context)
│   └── src/
│       ├── lib.rs             # Public API exports
│       ├── typed.rs           # Game trait (compile-time type safety)
│       ├── erased.rs          # ErasedGame trait (runtime polymorphism)
│       ├── adapter.rs         # GameAdapter (typed → erased conversion)
│       ├── context.rs         # EngineContext high-level API
│       ├── registry.rs        # Static game registration
│       ├── metadata.rs        # GameMetadata for UI/config
│       ├── legal_mask.rs      # LegalMask (dynamic-width action mask)
│       ├── game_utils.rs      # Shared helpers for game implementations
│       └── board_game.rs      # TwoPlayerObs generic type
├── engine-games/              # Registration of all bundled games
│   ├── src/
│   │   ├── lib.rs             # register_all_games() + layout invariants
│   │   ├── manifest.rs        # Game-metadata manifest rendering
│   │   └── bin/
│   │       └── gen-game-manifest.rs   # `make game-manifest`
│   └── tests/
│       └── manifest_golden.rs # Fails when the committed manifest drifts
├── metrics-common/            # Prometheus registration/encoding shared by actor + web
├── games-tictactoe/           # TicTacToe implementation
├── games-connect4/            # Connect 4 implementation
├── games-othello/             # Othello implementation
├── games-generals/            # Generals 8x8 implementation
│   └── src/                   # action, board, mapgen, movement, obs, params, rules
├── mcts/                      # Monte Carlo Tree Search
│   ├── src/
│   │   ├── config.rs          # MctsConfig parameters
│   │   ├── evaluator.rs       # Evaluator trait + UniformEvaluator
│   │   ├── node.rs            # MctsNode (visit stats)
│   │   ├── tree.rs            # Arena-allocated tree
│   │   ├── search.rs          # Select/expand/backprop algorithm
│   │   ├── sampling.rs        # Dirichlet noise + action sampling
│   │   ├── types.rs           # SearchResult, SearchStats, errors
│   │   └── onnx.rs            # OnnxEvaluator (feature-gated)
│   ├── benches/               # Search microbenchmarks
│   └── examples/              # generals_{policy_probe,strength_probe,search_diag}
└── model-watcher/             # ONNX hot-reload utility (filesystem + S3)
```

### Core Abstractions

#### Game Trait (Typed)

The typed `Game` trait provides compile-time type safety:

```rust
pub trait Game: Send + Sync + Debug + 'static {
    type State;      // Game state (e.g., board configuration)
    type Action;     // Action type (e.g., position index)
    type Obs;        // Observation (neural network input)

    // Identity and self-description
    fn engine_id(&self) -> EngineId;
    fn capabilities(&self) -> Capabilities;   // incl. max_horizon, encodings
    fn metadata(&self) -> GameMetadata;       // board dims, obs layout, UI hints

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8])
        -> (Self::State, Self::Obs);
    fn step(&mut self, state: &mut Self::State, action: Self::Action,
        rng: &mut ChaCha20Rng) -> (Self::Obs, f32, bool, u64);

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError>;
    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError>;
    // ... plus encode/decode for Action, and encode_obs
}
```

`metadata()` is the linchpin of the engine-as-source-of-truth design: it is what
the actor, the web server, the database row and the trainer's generated manifest
all ultimately read. See [§7.3](#73-game-metadata-single-sourcing).

#### ErasedGame Trait (Runtime)

The erased trait enables runtime polymorphism with byte-only interface:

```rust
pub trait ErasedGame: Send + Sync + Debug + 'static {
    fn engine_id(&self) -> EngineId;
    fn capabilities(&self) -> Capabilities;
    fn metadata(&self) -> GameMetadata;

    fn reset(&mut self, seed: u64, hint: &[u8],
        out_state: &mut Vec<u8>, out_obs: &mut Vec<u8>) -> Result<(), ErasedGameError>;
    fn step(&mut self, state: &[u8], action: &[u8],
        out_state: &mut Vec<u8>, out_obs: &mut Vec<u8>)
        -> Result<(f32, bool, u64), ErasedGameError>;
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
    pub temperature: f32,          // Action selection only (default: 1.0)
    pub virtual_loss: f32,         // Applied to leaves pending evaluation (default: 1.0)
    pub eval_batch_size: usize,    // Leaves batched per NN call (default: 32)
}

// Presets
MctsConfig::for_training()    // With exploration noise
MctsConfig::for_evaluation()  // Greedy, no noise
MctsConfig::for_testing()     // Fast (50 sims, batch 8, greedy)
```

`eval_batch_size` is additionally capped at `num_simulations / 4` inside the
search, so evaluations interleave with selection. Without that cap, a search
whose simulations all fit in one batch would pick every leaf before the first
value came back, and visit counts would carry no value information at all.

#### Search Algorithm

Per simulation:
1. **Selection**: Traverse using UCB = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
2. **Expansion**: Add children with policy prior from neural network
3. **Evaluation**: Call evaluator.evaluate(obs) for policy/value
4. **Backpropagation**: Update visit counts and value sums

#### Evaluator Trait

```rust
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, obs: &[u8], legal_moves_mask: &LegalMask, num_actions: usize)
        -> Result<EvalResult, EvaluatorError>;

    fn evaluate_batch(&self, requests: &[(&[u8], &LegalMask)], num_actions: usize)
        -> Result<Vec<EvalResult>, EvaluatorError>;
}

pub struct EvalResult {
    pub policy: Vec<f32>,   // Probability distribution
    pub value: f32,         // -1 (loss) to +1 (win)
}
```

The mask is an `engine_core::LegalMask` (a dynamic-width bitset), not a `u64`.
The old `u64` capped action spaces at 64, which Othello (65) already exceeded
and Generals (257) exceeds by far; masks are read from the observation at
`legal_mask_offset` rather than from `info_bits`, which cannot hold them and
aliases the player/winner fields past 16 actions.

Implementations:
- `UniformEvaluator`: Equal probability for legal moves (testing)
- `OnnxEvaluator`: Neural network inference via ONNX Runtime

#### Search results and the training target

```rust
pub struct SearchResult {
    pub action: u32,        // The move to play
    pub policy: Vec<f32>,   // Training target: raw visit distribution (tau = 1)
    pub value: f32,
    pub simulations: u32,
    pub stats: SearchStats,
}
```

**`policy` is always the tau=1 visit distribution and is never sharpened by
`MctsConfig::temperature`.** Temperature decides which action gets *played*;
the stored target keeps the search's full relative-visit information.

This distinction matters more than it looks. Raising visit counts to `1/tau`
for a small tau destroys the soft-target signal AlphaZero policy learning
depends on: at the actor's late-game tau of 0.1 the counts are raised to the
10th power, so two moves visited 12 and 9 times become a 17.8:1 target and a
typical search collapses onto a near one-hot vector. In long games, where the
low-temperature phase covers almost the whole episode, that is the difference
between learning and not.

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

#### Othello

| Property | Value |
|----------|-------|
| Board | 8x8 grid |
| Actions | 65 (64 cells + 1 pass) |
| Observation | 195 f32s (128 board + 65 legal + 2 player) |
| Network | ResNet 6 blocks x 256 filters |
| Board Type | "grid" |

#### 3.6 Generals 8x8

| Property | Value |
|----------|-------|
| Board | 8x8 grid |
| Actions | 257 (64 tiles x 4 directions, + wait) |
| Observation | 835 f32s (576 planes + 257 legal + 2 player) |
| Obs channels | 9, **player-relative** |
| Network | ResNet 6 blocks x 128 filters |
| Board Type | "grid" (no dedicated renderer — see below) |
| Max horizon | 402 plies |

Ported from the Go engine in `GeneralsReinforcementLearning/internal/game/`:
combat, captures, production, general-capture tile transfer, and
seed-deterministic map generation. Action index is `(y*8 + x)*4 + dir`, with
directions ordered up/right/down/left.

**Deliberate departures from real Generals**, each made to keep self-play sound
or trainable:

- **Strictly alternating turns.** Real Generals resolves both players' moves in
  one tick. The obvious bridge — stash player 1's move and resolve it on player
  2's step — leaks the pending move into the searcher's true state, so each ply
  resolves immediately instead. Production and the round clock tick after player
  2's ply, which also keeps the actor's depth-parity outcome backfill sound.
- **No half-moves.** Every move sends `army - 1`, halving the action space.
- **Territory adjudication at the round cap.** At `MAX_TURNS` the game is
  decided on tiles, then total armies, drawing only on an exact tie. A pure draw
  cap collapsed self-play into 100% draws — zero value signal.
- **Parity-randomized ply cap.** The cap is `2 * MAX_TURNS` or one less,
  coin-flipped at reset and deliberately absent from the observation. With a
  fixed even cap player 2 always owns the pre-adjudication move, wins nearly
  every near-symmetric game, and the value head degenerates into a seat detector.
- **No fog of war.** The observation is full-information. Fog is not a flag that
  can be flipped: vanilla MCTS re-simulates from the true state and would be
  omniscient under it. The fog variant needs observation history — a recurrent
  policy or IS-MCTS — and gets its own env id, obs schema version, and algorithm.

The observation schema is versioned `generals_obs:v1` and is player-relative:
9 channels x 64 (own/enemy/neutral territory, own/enemy log-armies, cities,
mountains, generals +1/-1, turn progress). Because the planes are already
seat-relative, the network must **not** additionally receive the player
indicator — hence `player_relative_obs = true`.

**Current status.** The engine and trainer paths are complete and training runs
end to end, but no model has yet beaten random at local compute scale.
Diagnostics live in `engine/mcts/examples/`: `generals_policy_probe` (visit
distribution health), `generals_strength_probe` (MCTS+model vs random — the
honest strength measure; the trainer's built-in eval is argmax-only and
understates models), and `generals_search_diag` (branching factor vs search
budget, and how much of the visit distribution a temperature schedule discards).

**The web UI cannot show Generals yet.** The blocker is state decoding, not
rendering: `web/src/game.rs::parse_state` assumes a flat
`[board][current_player][winner]` layout, while Generals encodes a 12-byte
header plus 64 x 6-byte tiles (396 bytes). The length check passes, so a session
would render garbage rather than fail — hence the game is not exposed in the UI.

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
    ├── game_config.rs     # Type alias over engine GameMetadata
    ├── config.rs          # CLI configuration (defaults from engine-config)
    ├── health.rs          # Health check endpoint
    ├── metrics.rs         # Prometheus metrics
    ├── stats.rs           # Self-play statistics (actor_stats.json)
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
    model_watcher: Option<ModelWatcher>,  // None in --no-watch mode
    stats: ActorStats,
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

Temperature schedule (**disabled by default** — `temp_threshold` defaults to `0`):
- Before threshold: `temperature = 1.0` (exploration)
- After threshold: `temperature = 0.1` (exploitation)

This affects **action selection only**. The policy target stored in the
transition is the raw visit distribution regardless — see
[Search results and the training target](#search-results-and-the-training-target).

When enabling it, size the threshold against actual episode length: a value
tuned for a ~25-move game leaves a ~400-ply game playing near-greedily for 96%
of every episode, which flattens self-play diversity.

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

The actor does not define a config type of its own — it uses the engine's
metadata directly:

```rust
// actor/src/game_config.rs
pub type GameConfig = GameMetadata;

pub fn get_config(env_id: &str) -> Result<GameConfig> {
    Ok(EngineContext::new(env_id)?.metadata())
}
```

That alias is the pattern the rest of the system follows: consumers read game
facts from the engine rather than restating them.

### Episode Outcomes

An episode ends one of two ways:

```rust
pub(crate) enum EpisodeOutcome {
    Completed { steps: u32, total_reward: f32, stats: EpisodeStats },
    Abandoned { reason: AbandonReason, steps: u32, discarded: usize, timeout_secs: u64 },
}
```

`Abandoned` means the wall-clock budget ran out (`AbandonReason::Timeout`) or
the step guard tripped (`AbandonReason::MaxSteps`). **All of that episode's
transitions are discarded**: without a terminal state there is no game outcome
to backfill, and value targets *are* the game outcome. Storing them anyway
would push the trainer onto its `mcts_value` fallback and degrade training
quietly rather than loudly.

Because the loss is real, it is counted rather than swallowed: every
abandonment increments `episodes_abandoned` / `transitions_discarded` in
`actor_stats.json`, bumps `actor_episodes_abandoned_total{reason}` and
`actor_transitions_discarded_total`, and logs a warning carrying the running
abandonment rate.

The wall-clock budget scales with the game. `episode_timeout_secs` is treated
as a floor and raised to at least one second per move of the game's horizon, so
a timeout tuned for a short game does not silently truncate a long one — which
would bias the buffer, since the episodes it kills are the long ones.

---

## 5. Trainer Component

### Directory Structure

```
trainer/
├── pyproject.toml
├── tests/                 # Pytest suite
└── src/trainer/
    ├── __main__.py        # CLI (train, evaluate, loop, solver-eval)
    ├── trainer.py         # Training loop
    ├── network.py         # MLP architecture
    ├── resnet.py          # ResNet architecture
    ├── evaluator.py       # Model evaluation
    ├── solver_eval/       # Perfect-solver move scoring (Connect4)
    ├── wandb_logger.py    # W&B wrapper (shim over crucible)
    ├── config.py          # TrainerConfig
    ├── game_config.py     # Engine manifest + network overrides
    ├── game_metadata.json # GENERATED by `make game-manifest`
    ├── checkpoint.py      # ONNX + PyTorch save/load
    ├── checkpoint_runner.py
    ├── replay_setup.py    # Buffer setup + metadata cross-check
    ├── step_metrics.py
    ├── stats.py           # Statistics tracking
    ├── lr_scheduler.py    # Warmup + cosine annealing
    ├── atomic_io.py       # Shim over crucible
    ├── backoff.py         # Shim over crucible
    ├── logging_utils.py
    ├── structured_logging.py
    ├── central_config.py  # config.toml loading
    ├── metrics.py         # Prometheus metrics export
    ├── orchestrator/      # Synchronized AlphaZero loop
    │   ├── orchestrator.py # Main loop coordinator
    │   ├── cli.py         # `trainer loop` argument parsing
    │   ├── config.py      # LoopConfig
    │   ├── actor_runner.py # Actor process management
    │   ├── eval_runner.py # Evaluation runner
    │   └── stats_manager.py # Stats aggregation
    ├── policies/          # Random + ONNX policies (for evaluation)
    ├── games/             # Pure Python game logic (for evaluation)
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

# Perfect-solver move scoring (Connect4 only)
python -m trainer solver-eval --model ./data/models/latest.onnx --games 100

# Synchronized AlphaZero (recommended)
python -m trainer loop --iterations 50 --episodes 500 --steps 1000
```

> The trainer reads the replay-buffer connection string only from the
> `CARTRIDGE_STORAGE_POSTGRES_URL` environment variable (config.toml's
> `storage.postgres_url` is used by the Rust actor/web, not the trainer).

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

> **The orchestration core lives in a sibling repository.** The synchronized
> loop (`Orchestrator`), the `ActorRunner`/`EvalRunner` base classes, the stats
> manager, promotion and eval-reporting logic, `LoopConfig`, plus
> `wandb_logger`, `atomic_io` and `backoff` were extracted to
> [`crucible`](https://github.com/mitchelldurbincs/crucible). It is a declared
> dependency of `cartridge-trainer`, pinned to a commit; CI installs the same
> pin. Generation, training and evaluation backends are injected through the
> `typing.Protocol` seams in `crucible/protocols.py`.
>
> This repo keeps two things: **shims** under `trainer/src/trainer/` (and
> `trainer/src/trainer/orchestrator/`) that re-export from `crucible` so
> existing `trainer.*` imports keep working, and a **composition root** at
> `trainer/src/trainer/orchestrator/orchestrator.py` that binds this repo's
> concrete pieces — `storage.create_replay_buffer`, `Trainer`/`TrainerConfig`
> via `TrainSpec`, the shim runners, structured-logging tracing — into the core
> `Orchestrator`, keeping the `Orchestrator(config)` signature unchanged for
> callers.
>
> For local development against a sibling checkout, install it editable *first*
> (`pip install -e ../../crucible`); pip then keeps it instead of fetching the
> pinned URL.

```python
class LoopConfig:
    iterations: int = 100
    episodes_per_iteration: int = 500
    steps_per_iteration: int = 1000

    # MCTS simulation ramping
    mcts_start_sims: int = 50
    mcts_max_sims: int = 400
    mcts_sim_ramp_rate: int = 20

    # Evaluation gatekeeper
    eval_interval: int = 1
    eval_games: int = 50
    eval_win_threshold: float = 0.55

    # Weights & Biases (the local subclass exists to restore this default)
    wandb: WandbConfig
```

### Weights & Biases

One W&B run per `trainer loop`, logging `train/`, `eval/`, `solver/` and `loop/`
metrics against a shared global-training-step x-axis. Configured under
`[wandb]` in `config.toml`; disabled by default.

`wandb_logger` falls back to a null logger when W&B is unavailable, so a run
never fails because logging is down — set `required = true` to invert that and
fail loudly. `wandb login` (or `WANDB_API_KEY`) is needed when enabled;
`WANDB_MODE=offline` logs locally with no network, `WANDB_MODE=disabled` forces
it off, and `WANDB_PROJECT` / `WANDB_ENTITY` override the config.

### Perfect-Solver Evaluation (Connect 4)

`trainer solver-eval` scores model decisions against the `bitbully` perfect
solver, reporting value-optimal-move rate, blunder rate and exact-best rate,
broken down overall / by ply bucket / by seat.

It is not only a standalone command: the loop runs it automatically each
evaluation for Connect 4 (`solver_games`, `solver_seed`), and
`promotion_metric = "solver_optimal"` switches the gatekeeper from win-rate to
solver-optimal rate with a `promotion_margin` over the incumbent, falling back
to win-rate when solver eval is unavailable.

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
│   ├── main.rs            # Thin entry point
│   ├── startup.rs         # AppState, router, CORS, graceful shutdown
│   ├── game.rs            # GameSession management
│   ├── metrics.rs         # Prometheus metrics
│   ├── handlers/
│   │   ├── game.rs        # Game endpoints
│   │   ├── health.rs      # Health check
│   │   └── stats.rs       # Training + actor stats
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
        ├── LossOverTimePage.svelte # Full-screen charts
        └── lib/
            ├── api.ts     # API client
            ├── chart.ts   # Chart formatting utilities
            └── constants.ts # Polling intervals, etc.
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/games` | GET | List available games (only the configured game) |
| `/game-info/:id` | GET | Game metadata (403 for non-current games) |
| `/game/new` | POST | Start new game |
| `/game/state` | GET | Get current board |
| `/move` | POST | Make player move + bot response |
| `/stats` | GET | Training statistics |
| `/actor-stats` | GET | Actor self-play statistics |
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
# Connection string from the CARTRIDGE_STORAGE_POSTGRES_URL env var...
replay = create_replay_buffer()
# ...or passed explicitly
replay = create_replay_buffer(
    connection_string="postgresql://user:pass@host:5432/db"
)
```

Schema — the authoritative copy is [`sql/schema.sql`](../sql/schema.sql), which
the Rust actor embeds at compile time (`include_str!`) and the Python trainer
reads at runtime:

```sql
CREATE TABLE IF NOT EXISTS transitions (
    id TEXT PRIMARY KEY,
    env_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    state BYTEA NOT NULL,
    action BYTEA NOT NULL,
    next_state BYTEA NOT NULL,
    observation BYTEA NOT NULL,
    next_observation BYTEA NOT NULL,
    reward REAL NOT NULL,
    done BOOLEAN NOT NULL,
    timestamp BIGINT NOT NULL,
    policy_probs BYTEA,              -- f32[num_actions], tau=1 visit distribution
    mcts_value REAL DEFAULT 0.0,
    game_outcome REAL,               -- backfilled at episode end
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON transitions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transitions_episode   ON transitions(episode_id);
CREATE INDEX IF NOT EXISTS idx_transitions_env_id    ON transitions(env_id);

CREATE TABLE IF NOT EXISTS game_metadata (
    env_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    board_width INTEGER NOT NULL,
    board_height INTEGER NOT NULL,
    num_actions INTEGER NOT NULL,
    obs_size INTEGER NOT NULL,
    legal_mask_offset INTEGER NOT NULL,
    player_count INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Supports concurrent writers from multiple actors.

> **Note:** [`scripts/init-postgres.sql`](../scripts/init-postgres.sql) is a
> separate, larger script used for containerised first-boot. It creates the two
> tables above plus `training_stats` and `model_versions`, and issues `GRANT`s.
> The two files have diverged; `sql/schema.sql` is what the running code uses.

### 7.3 Game metadata single-sourcing

Game facts are declared once, in the Rust game crates, and flow outward:

```text
games-*/src/lib.rs  ──  metadata()  ──►  GameMetadata (engine-core)
                                            │
        ┌───────────────────────────────────┼──────────────────────────┐
        │                                   │                          │
   actor (in-process)              make game-manifest            web (in-process)
        │                                   │                          │
        │                     trainer/src/trainer/game_metadata.json    │
        │                                   │                          │
        │                          trainer GAME_CONFIGS                │
        ▼                                   ▼                          ▼
  game_metadata table  ◄── cross-checked by ──  replay_setup.py
```

- **The manifest is generated, not written.** `make game-manifest` renders every
  registered game's metadata to `trainer/src/trainer/game_metadata.json`, which
  ships inside the Python package (`[tool.setuptools.package-data]`) and is
  loaded with `importlib.resources`. A golden test in `engine-games` fails if
  the committed file drifts from the game crates, so `cargo test` catches
  staleness — CI cannot compare the two languages live, because the Python job
  has no Rust toolchain and no artifacts pass between jobs.
- **The trainer splits facts from choices.** `_ENGINE_FACT_FIELDS` come from the
  manifest; `_TRAINING_OVERRIDES` holds the network architecture, which has no
  engine counterpart. A manifest game with no override entry raises, so adding a
  game in Rust forces a deliberate architecture decision instead of silently
  defaulting a spatial game to an MLP.
- **The database row is a cross-check, not a source.** The actor upserts its
  metadata on startup, so the row reflects the actor binary that ran most
  recently. `replay_setup.py` compares it against the manifest and **raises** on
  an `obs_size` / `legal_mask_offset` / `num_actions` disagreement: that means
  the actor and trainer were built from different commits, and the buffered
  transitions do not match the network's input layout. Descriptive differences
  only warn.

Two invariants are asserted for every registered game, on both sides:

```text
legal_mask_offset == obs_channels * board_size
obs_size          == legal_mask_offset + num_actions + 2
```

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
3. **`config.toml`**: the local, gitignored-in-spirit overrides you edit
4. **`config.defaults.toml`**: the checked-in defaults, and the single source
   of truth for every key and its default value

### Where the schema lives

The full key-by-key reference is **not duplicated here** — a partial copy in a
document is exactly how the previous version of this section came to disagree
with the shipped files on ~25 keys. Read them at the source:

| What | Where |
|------|-------|
| Every key, with its default and a comment | [`config.defaults.toml`](../config.defaults.toml) |
| Schema reference | [`engine/engine-config/SCHEMA.md`](../engine/engine-config/SCHEMA.md) |
| Rust structs | `engine/engine-config/src/structs.rs` |
| Python mirror | `trainer/src/trainer/central_config.py` |

`config.defaults.toml` is loaded by both languages: Rust embeds it at compile
time (`include_str!`), Python reads it at runtime and deep-merges `config.toml`
over it.

Sections: `[common]`, `[training]`, `[evaluation]`, `[actor]`, `[web]`,
`[mcts]`, `[logging]`, `[storage]`, `[wandb]`.

> **Not every section reaches every component.** `[wandb]` and the solver-eval
> keys under `[evaluation]` (`solver_games`, `solver_seed`, `promotion_metric`,
> `promotion_margin`) have no counterpart in the Rust `CentralConfig` — they are
> read only by the Python trainer.

### Environment Variable Format

New format (preferred):
```bash
CARTRIDGE_COMMON_ENV_ID=connect4
CARTRIDGE_TRAINING_ITERATIONS=50
CARTRIDGE_MCTS_NUM_SIMULATIONS=800
```

Legacy format (Python trainer only) — 11 variables are mapped, including
`ALPHAZERO_ENV_ID`, `ALPHAZERO_ITERATIONS`, `ALPHAZERO_START_ITERATION`,
`ALPHAZERO_EPISODES`, `ALPHAZERO_STEPS`, `ALPHAZERO_BATCH_SIZE`,
`ALPHAZERO_LR`, `ALPHAZERO_DEVICE`, `ALPHAZERO_CHECKPOINT_INTERVAL`,
`ALPHAZERO_EVAL_INTERVAL`, and a bare `DATA_DIR`
(`trainer/src/trainer/central_config.py`).

> **The two languages do not honour the same set.** Python parses
> `CARTRIDGE_<SECTION>_<KEY>` generically, so any key can be overridden. Rust
> matches against an explicit list in `engine/engine-config/src/loader.rs`, so
> keys outside it — `logging.format`, the MCTS ramping keys, `num_actors`,
> `allowed_origins`, `health_port` — are honoured by the trainer but **ignored
> by the actor and web server**. Set those in `config.toml` rather than the
> environment.

### Search Paths

Rust (`engine/engine-config/src/loader.rs`):

1. `$CARTRIDGE_CONFIG` (if set)
2. `./config.toml`
3. `../config.toml`
4. `/app/config.toml` (Docker)

Python (`trainer/src/trainer/central_config.py`) differs slightly: `./config.toml`,
`/app/config.toml`, then a project-root fallback — no `../config.toml`.

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
  promotion_metric?
        │
   ┌────┴─────────────────────┐
   │                          │
"win_rate"            "solver_optimal"  (Connect 4 only;
   │                          │          falls back to win_rate
   ▼                          ▼          when unavailable)
win_rate >          solver-optimal rate >
win_threshold       best's rate + promotion_margin
   │                          │
   └────────────┬─────────────┘
                │
           Yes  │  No
                │   │
                ▼   │
        Copy to best.onnx
                │   │
                │   ▼
                │  Keep previous best
                ▼
        Update best_model.json
```

---

## 10. Deployment

### Docker Compose Services

`docker-compose.yml` defines: `postgres`, `minio`, `minio-setup` (a one-shot
bucket initialiser that both `alphazero` and `web` wait on via
`service_completed_successfully`), `alphazero`, `web`, `frontend`, and
`prometheus`.

| Service | Host port | Notes |
|---------|-----------|-------|
| postgres | 5432 | replay buffer |
| minio | 9000 / 9001 | S3-compatible model storage + console |
| web | 8080 | API |
| frontend | 80 | nginx; container listens on **8080** |
| prometheus | 9092 | scrapes trainer:9090, actor:9091, web:8080 |

#### Local Development

```bash
# Synchronized AlphaZero training
docker compose up alphazero

# Play in browser
docker compose up web frontend
# Open http://localhost

# Standalone evaluation. --entrypoint is required: the image's ENTRYPOINT is
# `python -m trainer loop`, so a bare command would be appended to it.
docker compose run --rm --entrypoint python alphazero \
  -m trainer evaluate --model /app/data/models/latest.onnx
```

#### Kubernetes Simulation

```bash
# Full K8s-style stack (PostgreSQL + MinIO model storage)
docker compose -f docker-compose.yml -f docker-compose.k8s.yml up

# Parallel self-play is configured via [training].num_actors in config.toml
# (or CARTRIDGE_TRAINING_NUM_ACTORS), not by scaling containers
```

### Dockerfiles

| Image | Dockerfile | Base | Purpose |
|-------|------------|------|---------|
| `alphazero` | `Dockerfile.alphazero` | Ubuntu 24.04 | Actor + Trainer (synchronized loop) |
| `web` | `web/Dockerfile` | Ubuntu 24.04 | API server |
| `frontend` | `web/frontend/Dockerfile` | nginx alpine | Svelte UI |

### Feature Flags

PostgreSQL support is built in. Optional cargo features:
- `s3` (actor, web): S3/MinIO model storage
- `coreml` (actor): CoreML execution provider for ONNX on Apple Silicon
- `onnx` (web, default): MCTS bot with ONNX inference

Build with features:
```bash
docker build -f Dockerfile.alphazero --build-arg CARGO_FEATURES="s3" .
```

### Health Checks

| Service | Check | Defined in |
|---------|-------|-----------|
| postgres | `pg_isready -U cartridge -d cartridge` | docker-compose.yml |
| minio | `mc ready local` | docker-compose.yml |
| web | `curl -f http://localhost:8080/health` | `HEALTHCHECK` in web/Dockerfile |
| frontend | `wget -q --spider http://localhost:8080/` | `HEALTHCHECK` in web/frontend/Dockerfile |
| actor | HTTP endpoint on `actor.health_port` (default 8081) | actor/src/health.rs |

### Observability

Three Prometheus scrape targets (see [`prometheus.yml`](../prometheus.yml)); the
Prometheus UI is published on host port 9092, while the trainer and actor metrics
ports are `expose`-only inside the compose network.

| Component | Port | Notable metrics |
|-----------|------|-----------------|
| trainer | 9090 | training step/loss counters (`trainer/src/trainer/metrics.py`) |
| actor | 9091 | see below |
| web | 8080 (`/metrics`) | `web_games_created_total`, `web_games_active`, `web_moves_played_total`, `web_games_completed_total`, `web_request_duration_seconds{endpoint,method}`, `web_bot_move_seconds`, `web_model_loaded`, `web_model_reloads_total` |

Actor metrics (`actor/src/metrics.rs`) cover episodes
(`actor_episodes_total`, `actor_player1_wins_total`, `actor_player2_wins_total`,
`actor_draws_total`, `actor_episodes_per_second`, `actor_episode_duration_seconds`,
`actor_episode_steps`), search
(`actor_mcts_searches_total`, `actor_mcts_inference_seconds`,
`actor_mcts_search_seconds`, `actor_mcts_simulations_per_search`), storage
(`actor_transitions_stored_total`, `actor_db_write_seconds`, `actor_db_pool_*`),
model reloads, `actor_memory_rss_bytes`, and `actor_info{game,actor_id}`.

Two are worth watching specifically:

| Metric | Why |
|--------|-----|
| `actor_episodes_abandoned_total{reason}` | Episodes that never reached a terminal state (`reason` is `timeout` or `max_steps`). Non-zero means self-play data is being dropped — and dropped with a bias, since it is the long episodes that run out of wall clock. |
| `actor_transitions_discarded_total` | How many transitions went with them. |

Logging is `tracing` on the Rust side and `structured_logging.py` on the Python
side; set `logging.format = "json"` for cloud log aggregation.

---

## 11. Testing Strategy

### Test Coverage by Component

Exact counts are deliberately not recorded here — they change on almost every
commit and were wrong in three separate documents before this was written. Run
`make test` for the current numbers.

| Component | Coverage |
|-----------|----------|
| engine-core | Game trait, adapter, registry, context, legal masks |
| engine-config | Config loading, env overrides |
| engine-games | Registration, observation-layout invariants, manifest golden test |
| games-* | Game logic, encode/decode round-trips |
| mcts | Config, evaluator, tree, search, policy-target semantics |
| model-watcher | Hot-reload behavior |
| actor | Initialization, episodes, abandonment accounting, storage |
| web | Handlers, game session |
| trainer | Trainer, orchestrator composition, storage, eval, manifest integrity |

### Running Tests

```bash
make test          # engine + actor + web + trainer
make lint          # fmt/clippy + ruff/black

# Or individually
cargo test --manifest-path engine/Cargo.toml
cargo test --manifest-path actor/Cargo.toml
cargo test --manifest-path web/Cargo.toml
cd trainer && python -m pytest tests/ -v --tb=short
```

The trainer suite needs the `crucible` package. It is a declared dependency, so
`pip install -e "trainer/.[dev]"` pulls it; install a sibling checkout editable
first if you are developing it alongside.

Two checks are worth knowing about because they guard cross-cutting invariants:

- **`engine-games` manifest golden test** — fails when
  `trainer/src/trainer/game_metadata.json` drifts from the game crates.
  Regenerate with `make game-manifest`.
- **Observation-layout invariants** — asserted for every registered game on both
  the Rust side (`engine-games`) and the Python side
  (`TestManifestIntegrity`), so a layout change has to get past both.

### CI Pipeline

GitHub Actions workflow:
1. **rust-fmt**: Format (auto-fix committed on PRs)
2. **rust-clippy**: Lint with warnings as errors
3. **rust-test**: Full test suite
4. **rust-build**: Release build
5. **rust-security-audit**: cargo audit (non-blocking)
6. **python-lint**: Ruff + Black (auto-fix committed on PRs)
7. **python-test**: Pytest
8. **python-security-audit**: pip-audit (non-blocking)
9. **frontend**: Svelte check + build
10. **docker-build**: Docker image build validation
11. **secrets-scan**: GitLeaks (non-blocking)

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
| `config.defaults.toml` | Checked-in defaults; source of truth for every key |
| `config.toml` | Local overrides |
| `sql/schema.sql` | Database schema (embedded by Rust, read by Python) |
| `trainer/src/trainer/game_metadata.json` | **Generated** game manifest (`make game-manifest`) |
| `data/models/latest.onnx` | Current model |
| `data/models/best.onnx` | Best model |
| `data/models/best_model.json` | Best-model pointer (`{step, timestamp}`) |
| `data/stats.json` | Training statistics |
| `data/eval_stats.json` | Evaluation history |
| `data/solver_stats.json` | Perfect-solver eval history (Connect 4) |
| `data/actor_stats.json` | Self-play stats (written by actor, read by web) |
| `data/loop_stats.json` | Orchestrator history |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CARTRIDGE_COMMON_ENV_ID` | Game to train |
| `CARTRIDGE_TRAINING_DEVICE` | cpu, cuda, mps |
| `CARTRIDGE_MCTS_NUM_SIMULATIONS` | MCTS simulations |
| `CARTRIDGE_STORAGE_MODEL_BACKEND` | filesystem, s3 |
| `CARTRIDGE_STORAGE_POSTGRES_URL` | PostgreSQL connection |
