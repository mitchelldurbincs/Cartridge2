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
7. [Storage Backends](#7-storage-backends) — incl. [environment and algorithm catalog single-sourcing](#73-environment-and-algorithm-catalog-single-sourcing)
8. [Configuration System](#8-configuration-system)
9. [Data Flow](#9-data-flow)
10. [Deployment](#10-deployment) — incl. [Observability](#observability)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Overview

Cartridge2 is an algorithm-oriented reinforcement-learning platform. An
environment supplies game mechanics and capabilities; an algorithm cartridge
supplies a compatible collector, learner, orchestration recipe, experience
schema, model contract, evaluation suite, and serving implementation. The web
application remains a way to inspect training and play against supported
models.

### Key Design Philosophy

- **Monolithic over Microservices**: local processes over shared storage, instead of gRPC between services. (Kubernetes manifests and Terraform modules do exist under `k8s/` and `terraform/` for cloud deployment — what is avoided is service-to-service RPC, not orchestration.)
- **Library-First**: Engine is a Rust library, not a service
- **Engine as source of truth**: game facts — board dimensions, action counts, observation layout — are declared once in the Rust game crates. The Python trainer reads them from a generated manifest, and the database row is a cross-check, not a second definition. See [§7.3](#73-environment-and-algorithm-catalog-single-sourcing).
- **Explicit algorithm composition**: registration means an environment exists,
  not that every algorithm can train it. Runtime composition resolves a
  canonical algorithm ID and validates its generated compatibility profile
  before doing work.
- **Hot-Reloadable**: Models update without restarting services

### Environment and algorithm contracts

The canonical algorithm catalog lives in `engine/algorithm-core`. Each
descriptor has a stable ID and version plus language-neutral identifiers for
its seven replaceable components:

| Component | `alphazero_board_v1` contract |
|-----------|---------------------------------|
| Collector | `alphazero_mcts_self_play_v1` |
| Learner | `alphazero_policy_value_v1` |
| Orchestration | `synchronized_alphazero_v1` |
| Experience schema | `alphazero_transition_v1` |
| Model contract | `onnx_policy_value_v1` |
| Evaluation suite | `two_player_seat_balanced_v1` |
| Serving | `alphazero_mcts_web_v1` |

Rust dispatches the collector and evaluator through this catalog. Python reads
the same descriptors from manifest schema version 5 and binds its concrete
implementations in `trainer/algorithms/registry.py`. `[algorithm].id`,
`--algorithm`, and `CARTRIDGE_ALGORITHM_ID` all select the same canonical ID.

Before replay connections, model watchers, or evaluation matches begin, each
entry point resolves the algorithm and calls its compatibility guard for the
selected environment. Unknown IDs and reports with machine-checkable issues are
startup errors.

The first cartridge, `alphazero_board_v1`, supports exactly two fixed players,
alternating turns, indexed discrete actions, observation-embedded legal masks,
fixed spatial `f32` observations, perfect information, deterministic planning
snapshots, and terminal-only zero-sum rewards. The environment contract exposes
these agents/action spaces, turn, information, planning-state, chance,
transition, reward, horizon, and wire-encoding semantics in machine-readable
form. The optional nested board profile supplies the AlphaZero observation and
presentation facts, so the compatibility result is exact.

The generic ABI represents single-agent and simultaneous decisions, fixed or
dynamic populations, explicit or environment-sampled chance, partial
observation, stochastic transitions, general per-agent rewards, truncation, and
multi-discrete or continuous actions. The first cartridge rejects those
profiles; using them requires a matching collector/learner/model/evaluation
cartridge (and a recurrent model contract where history is required).

### Target Games

| Game | Status | Board | Actions | Obs size | Network |
|------|--------|-------|---------|----------|---------|
| TicTacToe | Complete | 3x3 | 9 | 29 | MLP (hidden 128) |
| Connect 4 | Complete | 7x6 | 7 | 93 | ResNet 4x128 |
| Othello | Complete | 8x8 | 65 (64 cells + pass) | 195 | ResNet 6x256 |
| Generals 8x8 | Engine/trainer/web complete; not yet stronger than random | 8x8 | 257 (64 tiles x 4 dirs + wait) | 899 | ResNet 6x128, 10 planes |

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
┌──────────────────────── Synchronized trainer process ───────────────────────┐
│ Python orchestrator                                                        │
│   ├─ bounded Rust collectors ── exact ReplaySelection ──► PostgreSQL        │
│   ├─ PyTorch learner ◄───────── same sealed selection ───┘                  │
│   ├─ Rust evaluator                                                        │
│   └─ immutable checkpoint/evaluation/RunCommit ── CAS RunHead ─► FS or S3  │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │ ChampionOrLatest + projection
                                 ┌────────────▼────────────┐
                                 │ Axum web server :8080   │
                                 └────────────┬────────────┘
                                              │ HTTP
                                 ┌────────────▼────────────┐
                                 │ Svelte frontend :5173   │
                                 └─────────────────────────┘
```

### Component Interactions

The orchestrator is the only owner of synchronized iteration state. It derives
the parent from RunHead, creates the collection scope, starts finite collector
children, verifies their exact episode seal, invokes the learner/evaluator, and
publishes the RunCommit. Collectors load Latest once; they are not independent
services. Web independently follows the accepted RunHead using
ChampionOrLatest, while the frontend sees only the web API.

### Training Modes

#### Synchronized AlphaZero (Recommended)

Each iteration follows this pattern:
```
┌────────────────────────────────────────────────────────────────────┐
│ Iteration N                                                        │
├────────────────────────────────────────────────────────────────────┤
│ 1. Resolve the exact parent checkpoint (or root)                   │
│ 2. Allocate a fresh cryptographic replay collection scope          │
│ 3. Run bounded collectors pinned to that source checkpoint         │
│ 4. Seal exactly N complete episodes; fail closed on any mismatch   │
│ 5. Train for M steps using only that exact replay selection        │
│ 6. Evaluate and write immutable checkpoint/evaluation/RunCommit    │
│ 7. Compare-and-set the sole models/channels/current.json RunHead   │
└────────────────────────────────────────────────────────────────────┘
```

Every attempt gets a new scope. The loop does not clear the profile; rows from
older or failed attempts are retained but cannot match the current selection.
Collection uses system entropy (`system_entropy_v1`, with no recorded seed), so
self-play is intentionally not bit reproducible. Evaluation uses the recorded
`evaluation_seed` and exact seat recipe and is deterministic evidence.

#### Standalone Commands (Disjoint)

Direct `train` requires an explicit `--collection-scope-id` and exactly one of
`--source-checkpoint-id` or `--source-root`. It can continue only a standalone
lineage; it refuses a RunHead owned by a synchronized recipe. Conversely, the
synchronized loop requires its exact persisted `RunRecipe` and cannot absorb a
standalone lineage. `solver-eval` is diagnostic/log-only and cannot write
evaluation authority. There is no continuous actor/trainer mode in Phase 1.

---

## 3. Engine Component

### Directory Structure

```
engine/
├── Cargo.toml                 # Workspace root
├── algorithm-core/           # Algorithm descriptors, compatibility, dispatch keys
├── engine-config/             # Centralized config.toml loading (shared by actor/web)
├── engine-core/               # Generic Environment ABI, registry, context
│   └── src/
│       ├── lib.rs             # Public API exports
│       ├── typed.rs           # Environment, Timestep, semantics, agents/actions
│       ├── contract.rs        # Shared descriptor/timestep validation
│       ├── erased.rs          # Sealed bytes-only runtime boundary
│       ├── adapter.rs         # Private typed erasure + validation
│       ├── context.rs         # EngineContext high-level API
│       ├── registry.rs        # Immutable environment registration
│       ├── metadata.rs        # Generic metadata + optional board profile
│       ├── board_view.rs      # Optional Board/Custom presentation projection
│       ├── legal_mask.rs      # LegalMask (dynamic-width action mask)
│       ├── board_game_utils.rs # Narrow two-player board helpers
│       └── board_game.rs      # Narrow board family → generic ABI adapter
├── envs-counter/              # Direct non-board Environment reference/canary
├── engine-games/              # Registration of all bundled games
│   ├── src/
│   │   ├── lib.rs             # register_all_environments() + profile invariants
│   │   ├── manifest.rs        # Environment/algorithm manifest rendering
│   │   └── bin/
│   │       └── generate-environment-manifest.rs # `make environment-manifest`
│   └── tests/
│       └── environment_manifest_golden.rs # Rejects committed catalog drift
├── evaluator/                 # Algorithm-dispatched `cartridge-eval`
│   └── src/
│       ├── lib.rs             # Match loop, seat alternation, position dump
│       ├── player.rs          # Random or model (with optional MCTS) seats
│       ├── results.rs         # EvalSummary + PositionRecord wire formats
│       └── main.rs            # CLI
├── metrics-common/            # Prometheus registration/encoding utilities
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

#### Environment Trait (Typed)

The typed `Environment` trait is algorithm-neutral and provides compile-time
type safety:

```rust
pub trait Environment: Send + Sync + Debug + 'static {
    type State;
    type Action;      // May encode a joint action for simultaneous decisions
    type Observation;

    fn engine_id(&self) -> EngineId;
    fn capabilities(&self) -> Capabilities;
    fn metadata(&self) -> EnvironmentMetadata;

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8])
        -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError>;
    fn step(&mut self, state: &mut Self::State, action: Self::Action,
        rng: &mut ChaCha20Rng)
        -> Result<Timestep<Self::Observation>, EnvironmentError>;

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError>;
    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError>;
    // ... plus strict Action codecs and Observation encoding
}
```

`Timestep` carries an explicit transition roster, observations and
`{reward, terminated, truncated}` outcomes per stable `AgentId`, plus
`Decision::{Agents, Chance, None}`, transition provenance, and episode status.
Fixed environments emit their complete declared population. Dynamic
environments include newly/currently active agents and retain a departing agent
on the transition that terminates or truncates it; that agent leaves the next
decision immediately and is omitted from the following timestep. This local
transition roster preserves its final outcome and provenance, but the generic
runtime does not yet validate source/identity lifecycles across arbitrary
branchable snapshots. `Capabilities` declares fixed/dynamic agents
with their own action spaces, wire codecs, an optional horizon, and exact
environment semantics. The environment ID is also the runtime artifact
namespace and is restricted to lowercase ASCII letters, digits, `_`, and `-`.

`CompleteSnapshot` promises that state bytes contain every transition-relevant
input and can be branched or replayed without hidden mutable state.
Environment-sampled chance consumes the runtime RNG stream and is therefore
required to declare `ExternalState`; explicit chance can remain a complete
snapshot because the resolved outcome is supplied as an action.

`metadata()` is display-oriented. Its `board` member is optional; board
dimensions, AlphaZero observation layout, players, and renderer are not fields
of the generic ABI. The trainer's generated manifest preserves this separation.
See
[§7.3](#73-environment-and-algorithm-catalog-single-sourcing).

The standard action encodings are one little-endian `u32` for discrete, one
little-endian `u32` per declared multi-discrete dimension, and flattened
row-major little-endian `f32` values for continuous actions. Simultaneous joint
actions and explicit chance outcomes currently use an environment-defined
`Custom` codec. The matching algorithm cartridge must understand that codec;
the generic ABI does not yet define a shared per-decision action envelope.

#### Sealed Runtime Boundary

The implementation uses a byte-only erased trait for runtime polymorphism, but
that trait and its adapter are private to `engine-core`. The public entry point
is `EngineContext`, constructed either from a registered environment ID or a
typed `Environment`. This prevents callers from installing unchecked erased
implementations or bypassing contract validation.

```rust
let mut registered = EngineContext::new("counter")?;
let mut isolated = EngineContext::from_environment(CounterEnvironment)?;
```

#### Adapter Pattern

The generic adapter validates identity, codecs, agent/action descriptors,
per-agent outcomes, decisions, and episode consistency before exposing bytes:

```
Typed Environment (State, Action, Observation)
        ↓
EnvironmentAdapter<E: Environment>
    ├─ Private to engine-core
    ├─ Validates immutable descriptors and every Timestep
    ├─ Manages RNG (re-seeded on reset)
    └─ Handles encode/decode
        ↓
sealed ErasedEnvironment trait (bytes-only)
        ↓
EngineContext
```

The bundled games use a second, explicitly narrow layer, exported through
`engine_core::board_profile`:
`BoardGame → BoardGameEnvironment → EnvironmentAdapter`. The board adapters are
private; only that layer converts alternating two-seat scalar actor rewards
into per-agent zero-sum outcomes and produces board presentations.

#### Registry System

Typed registration with runtime lookup. The registry derives its key from the
environment's validated descriptor, eliminating a duplicate caller-supplied ID:

```rust
// Registration (called at startup)
use engine_core::board_profile::register_board_game;

pub fn register_tictactoe() {
    register_board_game::<TicTacToe>()
        .expect("tictactoe must only be registered once");
}

// Lookup (at runtime)
let context = EngineContext::new("tictactoe")?;
```

#### EngineContext API

High-level convenience wrapper:

```rust
let mut ctx = EngineContext::new("tictactoe")?;
let reset = ctx.reset(42, &[])?;           // seed=42
let step = ctx.step(&reset.state, &action)?;
println!("Episode: {:?}", step.timestep.episode);
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

The mask is an `engine_core::board_profile::LegalMask`, a dynamic-width bitset.
It is read from the authoritative observation at `legal_mask_offset`, so board
action spaces are not constrained by a packed integer side channel.

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
| Observation | 899 f32s (640 planes + 257 legal + 2 player) |
| Obs channels | 10, **player-relative** |
| Network | ResNet 6 blocks x 128 filters |
| Board Type | "generals" |
| Max horizon | 400 plies |

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
  2's ply, preserving the alternating-turn contract required by the installed
  AlphaZero cartridge.
- **No half-moves.** Every move sends `army - 1`, halving the action space.
- **Territory adjudication at the round cap.** At `MAX_TURNS` the game is
  decided on tiles, then total armies, drawing only on an exact tie. A pure draw
  cap collapsed self-play into 100% draws — zero value signal.
- **Parity-randomized ply cap.** The cap is `2 * MAX_TURNS` or one less,
  coin-flipped at reset. The exact remaining-ply countdown is a constant
  observation plane, so the randomized rule state remains fully Markov-visible.
  With a fixed even cap player 2 always owns the pre-adjudication move, wins
  nearly every near-symmetric game, and the value head degenerates into a seat
  detector.
- **No fog of war.** The observation is full-information. Fog is not a flag that
  can be flipped: vanilla MCTS re-simulates from the true state and would be
  omniscient under it. The fog variant needs observation history — a recurrent
  policy or IS-MCTS — and gets its own env id, obs schema version, and algorithm.

The observation schema is versioned `generals_obs:v2` and is player-relative:
10 channels x 64 (own/enemy/neutral territory, own/enemy log-armies, cities,
mountains, generals +1/-1, turn progress, exact normalized plies remaining).
Because the planes are already seat-relative, the network must **not**
additionally receive the player indicator — hence
`player_relative_obs = true`. This schema change is Generals environment
contract version 2; v1 models and replay are intentionally incompatible.

**Current status.** The engine and trainer paths are complete and training runs
end to end, but no model has yet beaten random at local compute scale.
Diagnostics live in `engine/mcts/examples/`: `generals_policy_probe` (visit
distribution health), `generals_strength_probe` (MCTS+model vs random), and
`generals_search_diag` (branching factor vs search budget, and how much of the
visit distribution a temperature schedule discards). Note that `cartridge-eval
--p1-sims N` now measures the same thing as the strength probe, through the
regular evaluation path.

**Generals in the web UI.** Rendering goes through `BoardView`, so the web
server no longer decodes state bytes and the game's 12-byte header plus 64 x
6-byte tiles is a non-issue. Its `board_type` is `"generals"`: a move is
`(tile * 4) + direction` rather than a placement, so the frontend selects a
source tile and then an adjacent target. Its 257 actions are also why the web
API's action indices are `u32` rather than `u8`.

### Model Watcher

Hot-reload system for strict, content-addressed ONNX checkpoints. Filesystem
mode watches the selected profile's `models/channels/current.json`; S3 mode
polls the equivalent object key and downloads validated RunCommit,
checkpoint-manifest, and blob objects to a profile cache. Actor collection uses
`ModelSelection::Latest` once and drops the watcher. Web serving uses
`ModelSelection::ChampionOrLatest`: it selects champion state from the latest
validated RunCommit, falling back to that commit's latest checkpoint before
the first promotion.

```
Trainer publishes checkpoint:
  1. Safely validate the staged ONNX and learner-state envelope
  2. Create/verify immutable blobs, manifest, evaluation, and RunCommit objects
  3. Validate exact parent checkpoint/RunCommit lineage and persisted recipe
  4. Compare-and-set channels/current.json as the sole RunHeadV2 authority

ModelWatcher detects:
  1. RunHead generation change (filesystem notification or polling)
  2. Verify canonical RunHead and complete immutable RunCommit/checkpoint chains
  3. Apply Latest or ChampionOrLatest selection to the latest accepted commit
  4. Verify profile, blob size/digest, ONNX identity, and tensor interface
  5. Load the selected candidate, acquire evaluator write lock, and swap
  6. Signal subscribers, even when an accepted generation retains its champion
```

Features:
- Filesystem notifications plus polling, or S3 channel polling
- Immutable model/evaluation/RunCommit objects and one CAS-protected RunHead
- Concurrent-safe via Arc<RwLock<>>
- Schema-v1 artifact identity validation before the evaluator swap

Every ONNX file must carry these exact custom metadata values:

| Key | Required value |
|-----|----------------|
| `cartridge.schema_version` | `1` |
| `cartridge.algorithm_id` | Selected algorithm ID, currently `alphazero_board_v1` |
| `cartridge.model_contract` | The selected descriptor's model contract, currently `onnx_policy_value_v1` |
| `cartridge.env_id` | Selected environment ID |
| `cartridge.env_contract_version` | Selected environment contract version |

The expected identity is derived from the resolved algorithm descriptor and
startup environment, never from the artifact itself. Filesystem and S3 loads,
the actor, web server, evaluator, and direct MCTS ONNX loaders all use the same
check. If no `current` channel exists, the actor/web evaluator remains empty and
play falls back to the random policy. A present malformed or mismatched pointer,
manifest, or blob fails initial loading. Hot reload constructs and validates a
new evaluator before acquiring the swap lock; a rejected update is logged and
the last valid evaluator remains active.

---

## 4. Actor Component

### Directory Structure

```
actor/
├── Cargo.toml
└── src/
    ├── main.rs            # Entry point, CLI parsing
    ├── algorithms.rs      # Algorithm ID -> collector dispatch
    ├── actor.rs           # AlphaZeroCollector implementation
    ├── mcts_policy.rs     # MCTS action selection
    ├── config.rs          # CLI configuration (defaults from engine-config)
    ├── resources.rs       # Process resource diagnostics
    ├── stats.rs           # Process-local self-play telemetry
    └── storage/
        ├── mod.rs         # ReplayStore trait
        └── postgres.rs    # PostgreSQL backend
```

### Collector dispatch

`main.rs` passes the selected ID to `build_collector`. The dispatcher resolves
it through `algorithm-core`; the concrete collector then builds an
`EngineContext` and requires a compatible report before it pins a model or
opens replay storage.

```rust
pub trait CollectorAlgorithm {
    async fn run(&self) -> Result<()>;
    fn shutdown(&self);
}

pub struct AlphaZeroCollector {
    config: Config,
    replay_selection: ReplaySelection,
    board_metadata: BoardGameMetadata,
    engine: Mutex<EngineContext>,
    mcts_policy: Mutex<MctsPolicy>,
    replay: Arc<dyn ReplayStore>,
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
    stats: ActorStats,
}
```

Each actor is a bounded one-shot worker. `--max-episodes` and
`--collection-scope-id` are required. It loads `ModelSelection::Latest` once,
requires the loaded checkpoint to equal `--source-checkpoint-id` (or requires
an absent RunHead at root), then drops the watcher before opening replay. The
evaluator cannot change during collection.

### Episode Execution Flow

```
AlphaZeroCollector.run_episode():
  1. Reset game with random seed
  2. Loop while !done:
     a. Lock policy
     b. Select action via MCTS (or random if no model)
     c. Unlock policy
     d. Lock engine
     e. Execute action, get next state
     f. Unlock engine
     g. Retain observation, acting agent, and MCTS policy in memory
  3. Require a terminal outcome for both agents
  4. Encode each pending item with its acting agent's terminal value
  5. Wrap opaque payloads in ReplayRecord envelopes and batch store
  6. Return episode statistics
```

### MCTS Policy

```rust
pub struct MctsPolicy {
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,  // Pinned for actor lifetime
    config: MctsConfig,
    base_temperature: f32,
    late_temperature: f32,
    temp_threshold: u32,  // Move number to switch temps
}
```

Temperature schedule (**disabled by default** — `temp_threshold` defaults to `0`):
- Before threshold: `temperature = 1.0` (exploration)
- At and after threshold: `temperature = 0.1` (exploitation)

A nonzero threshold must be strictly below the selected environment's declared
`max_horizon`; the trainer and actor both reject an unreachable late phase.

This affects **action selection only**. The policy target stored in the
payload is the raw visit distribution regardless — see
[Search results and the training target](#search-results-and-the-training-target).

When enabling it, size the threshold against actual episode length: a value
tuned for a ~25-move game leaves a ~400-ply game playing near-greedily for 96%
of every episode, which flattens self-play diversity.

Fallback behavior:
- If no model loaded: Random legal action with uniform policy

### Replay Record and AlphaZero Payload

The storage envelope is generic. A future cartridge owns a different
`experience_schema` and payload codec without changing the replay table:

```rust
pub struct ReplayRecord {
    pub id: String,
    pub env_id: String,
    pub env_contract_version: u32,
    pub algorithm_id: String,
    pub experience_schema: String,
    pub collection_scope_id: String,
    pub source_checkpoint_id: Option<String>,
    pub episode_id: String,
    pub step_number: u32,
    pub payload: Vec<u8>,
}
```

For `alphazero_transition_v1`, `payload` is exactly
`observation[obs_size] || policy[num_actions] || terminal_value[1]` as
little-endian `f32` values. The collector keeps the acting agent alongside each
pending position only until the matching terminal value can be encoded; it is
not a storage column.

### Environment metadata use

The collector reads generic capabilities and `EnvironmentMetadata` from its
`EngineContext`, then explicitly requires the optional `board` profile because
`alphazero_board_v1` needs its observation layout and legal mask. Those facts
parameterize the cartridge-owned codec and model contract; PostgreSQL neither
stores nor interprets them.

### Episode Outcomes

An episode ends one of two ways:

```rust
pub(crate) enum EpisodeOutcome {
    Completed { steps: u32, player_one_outcome: f32, stats: EpisodeStats },
    Abandoned { reason: AbandonReason, steps: u32, discarded: usize, timeout_secs: u64 },
}
```

`Abandoned` means the wall-clock budget ran out (`AbandonReason::Timeout`), the
step guard tripped (`AbandonReason::MaxSteps`), or the environment truncated
without terminal outcomes (`AbandonReason::EnvironmentTruncated`). **All of
that episode's pending records are discarded**: without a terminal state the
AlphaZero codec cannot construct its required value target.

Because the loss is real, it is counted rather than swallowed: every
abandonment updates process-local `ActorStats`, emits a structured warning, and
then fails the bounded worker immediately. Partial rows remain quarantined in
that failed scope; a retry receives a new scope. This prevents
retry-until-success length bias and gives every actor process a hard terminal
condition. Bounded actors expose no HTTP/Prometheus service and write no shared
stats projection; successful workers emit one final structured stats/RSS log.

`episode_timeout_secs` is the exact authenticated wall-clock bound. It is never
silently raised from environment metadata. Operators must choose it for the
selected game and search budget; a timeout that is too short preferentially
kills long episodes and therefore biases collection by making the whole scoped
attempt fail.

---

## 5. Trainer Component

### Directory Structure

```
trainer/
├── pyproject.toml
├── tests/                 # Pytest suite
└── src/trainer/
    ├── __main__.py        # CLI (train, evaluate, loop, solver-eval)
    ├── algorithms/        # Algorithm protocols, registry, implementations
    ├── environment_catalog.py # Strict manifest-v4 parser and catalog
    ├── runtime_profile.py # Canonical runtime namespace
    ├── trainer.py         # AlphaZeroLearner
    ├── network.py         # MLP architecture
    ├── resnet.py          # ResNet architecture
    ├── evaluator.py       # Drives `cartridge-eval`; parses its summary
    ├── players.py         # Who occupies a seat in an evaluation game
    ├── registry.py        # Immutable player registry schema v5
    ├── tournament.py      # Round-robin + Bradley-Terry Elo
    ├── tournament_cli.py  # register-players / tournament commands
    ├── solver_eval/       # Perfect-solver move scoring (Connect4)
    ├── config.py          # AlphaZeroLearnerConfig
    ├── environment_manifest.json # GENERATED manifest schema v5
    ├── checkpoint.py      # ONNX + PyTorch save/load
    ├── checkpoint_runner.py
    ├── replay_setup.py    # Exact ReplaySelection setup
    ├── step_metrics.py
    ├── stats.py           # Statistics tracking
    ├── lr_scheduler.py    # Warmup + cosine annealing
    ├── logging_utils.py
    ├── structured_logging.py
    ├── central_config.py  # config.toml loading
    ├── metrics.py         # Prometheus metrics export
    ├── orchestrator/      # Synchronized AlphaZero loop
    │   ├── orchestrator.py # Main loop coordinator
    │   ├── cli.py         # Cartridge `loop` command arguments
    │   ├── config.py      # LoopConfig
    │   ├── actor_runner.py # Actor process management
    │   ├── eval_runner.py # Evaluation runner
    │   └── eval_reporting.py # Evaluation/result adapter
    └── storage/
        ├── base.py        # Exact replay profile + selection contracts
        ├── factory.py     # Replay backend factory
        ├── postgres.py    # PostgreSQL replay implementation
        ├── publisher.py   # Strict filesystem/S3 checkpoint publication
        └── schema.sql     # Packaged replay schema
```

### CLI Commands

```bash
# Train from one exact root replay selection (scope must already contain data)
python -m trainer --algorithm alphazero_board_v1 train \
  --steps 1000 \
  --collection-scope-id 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --source-root

# Model evaluation
python -m trainer --algorithm alphazero_board_v1 evaluate --games 100

# Perfect-solver move scoring (Connect4 only)
python -m trainer --algorithm alphazero_board_v1 solver-eval --env-id connect4 --games 100

# Synchronized AlphaZero (recommended)
python -m trainer --algorithm alphazero_board_v1 loop --iterations 50 --episodes 500 --steps 1000
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

> **The generic coordination core lives in a sibling repository.** Crucible's
> base `Orchestrator` and actor-runner protocol, iteration value types,
> `wandb_logger`, `atomic_io`, and `backoff` live in
> [`crucible`](https://github.com/mitchelldurbincs/crucible). It is a declared
> dependency of `cartridge-trainer`, pinned to a commit; CI installs the same
> pin.
>
> The composition root at
> `trainer/src/trainer/orchestrator/orchestrator.py` resolves `algorithm_id`
> and validates the selected environment and persisted RunRecipe before it
> opens an attempt scope. Cartridge2 owns the exact replay fencing, prepared-run
> journal, immutable evaluation/RunCommit publication, recovery, and disposable
> projection rebuilds; algorithm factories supply collection, learning, and
> evaluation implementations.
>
> For local development against a sibling checkout, install it editable *first*
> (`pip install -e ../../crucible`); pip then keeps it instead of fetching the
> pinned URL.

`--iterations` is a global target, not “iterations to add.” Restart resolves
the selected RunCommit chain and executes only missing iterations. Recipe
fields are authenticated in that chain; changing them under an existing
RunHead fails closed instead of silently changing the experiment.

### Weights & Biases

One W&B run per cartridge `loop` invocation, logging `train/`, `eval/`,
`solver/` and `loop/`
metrics against a shared global-training-step x-axis. Configured under
`[wandb]` in `config.toml`; disabled by default.

`wandb_logger` falls back to a null logger when W&B is unavailable, so a run
never fails because logging is down — set `required = true` to invert that and
fail loudly. `wandb login` (or `WANDB_API_KEY`) is needed when enabled;
`WANDB_MODE=offline` logs locally with no network, `WANDB_MODE=disabled` forces
it off, and `WANDB_PROJECT` / `WANDB_ENTITY` override the config.

### Evaluation

Evaluation games are played by the Rust `cartridge-eval` binary, not by Python:
the trainer launches it as a subprocess, exactly as it launches the actor for
self-play, and reads back a JSON summary.

```
trainer.evaluator.evaluate()
  -> cartridge-eval --algorithm A --env-id X --games N --p1 <model|random> --p2 ...
       (plays through EngineContext; optional MCTS per seat)
  -> eval.json  --> EvalResults --> promotion gate
```

The binary is found via `CARTRIDGE_EVAL_BINARY`, then
`engine/target/{release,debug}/cartridge-eval` (`/app/cartridge-eval` in the
Docker image). `make build-eval` builds it.

**Why it lives in the engine.** Playing in Python required a second
implementation of every game's rules (`trainer/games/`) that nothing kept in
sync with the engine — the promotion gate could silently score a different game
from the one being trained, and games nobody had reimplemented (Othello) could
not be evaluated at all. The two implementations had in fact already diverged:
the Python Connect 4 mirror stored its board column-major while the engine
stores it row-major.

**Search during evaluation.** `[evaluation] simulations` sets the MCTS budget
per move. It defaults to `0`, meaning the policy head is played directly for a
cheap evaluation pass. Raising it makes evaluation measure the system as it
actually plays: a Connect 4
checkpoint that scores 15/20 vs random at `simulations = 0` scores 19/20 at 100.

### Player Registry and Tournaments

A *player* is anything that can occupy a seat: the random baseline, a
checkpoint, the same checkpoint given a search budget, later a PPO checkpoint.
The selected runtime profile's `players.json` makes them explicit rather than
identified by filename convention. Registry schema v5 records the algorithm,
environment/model contract, checkpoint manifest ID, immutable ONNX blob path,
manifest step, and gameplay-adapter settings. A model player ID contains the
full checkpoint ID plus a hash of the versioned simulations/temperature
adapter. `trainer/registry.py` resolves and revalidates every registered
manifest and blob before producing the player spec `cartridge-eval` consumes;
missing or corrupt artifacts fail the tournament.

```bash
trainer --algorithm alphazero_board_v1 register-players --env-id connect4
trainer --algorithm alphazero_board_v1 tournament --env-id connect4 --games 40
```

`trainer/tournament.py` plays every pairing once and fits Bradley-Terry ratings
on the Elo scale, anchored so `random` sits at 0. It uses the
minorization-maximization iteration rather than sequential Elo updates: the
latter depend on the order games happen to be played, so the same round-robin
would rate differently depending on scheduling.

**Why a round-robin rather than win rate vs. random.** Win rate against one
opponent depends entirely on who that opponent was, and saturates — every decent
Connect 4 checkpoint can look similar against random exactly where more
resolution is needed. Registering immutable checkpoint identities and rating
the whole field separates them without relying on mutable filenames.

**Play temperature is not optional here.** Two greedy models on a deterministic
opening replay the same game every time, so a 40-game match is one game counted
20 times per seat, and the ratings come out confident and meaningless — measured:
every model-vs-model pairing scored exactly 0-20, 10-10 or 20-0. Registered
players default to temperature 0.2, and a tournament warns when two or more of
its field are deterministic.

Nothing in either module knows how a player was trained. A PPO checkpoint enters
the same pool and gets a comparable rating.

### Perfect-Solver Evaluation (Connect 4)

`trainer --algorithm alphazero_board_v1 solver-eval` scores model decisions
against the `bitbully` perfect
solver, reporting value-optimal-move rate, blunder rate and exact-best rate,
broken down overall / by ply bucket / by seat.

The default model is the latest checkpoint selected through
`models/channels/current.json`.
`--all-checkpoints` discovers and verifies every immutable repository manifest
in step order. `--model PATH` is deliberately a one-off escape hatch and its
result has no checkpoint ID or repository step. The two options are mutually
exclusive.

The engine plays and Python judges: `cartridge-eval --dump-positions` writes
every decision it made as JSONL, and the scorer replays that through bitbully,
cross-checking its mirrored board against the engine's own position at every
query. (Previously it played its own Python games and cross-checked bitbully
against *those*, so nothing compared either to the engine.)

It is not only a standalone command: the loop runs it automatically each
evaluation for Connect 4 (`solver_games`, `evaluation_seed`), and
`promotion_metric = "solver_optimal"` switches the gatekeeper from win-rate to
solver-optimal rate with a `promotion_margin` over the incumbent, falling back
to win-rate when solver eval is unavailable.

Standalone `solver-eval` prints diagnostics only. Synchronized solver evidence
is authoritative solely when embedded in a validated immutable
`EvaluationArtifactV2` selected through `RunCommitV1`; projection files are
never reused as promotion evidence.

MCTS ramping formula:
```
sims = min(start_sims + (iter-1) × ramp_rate, max_sims)
```

### Checkpoint System

Each checkpoint is a content-addressed pair of immutable blobs plus one
canonical manifest:

```text
models/blobs/sha256/{onnx_digest}.onnx
models/blobs/sha256/{learner_digest}.pt
models/manifests/sha256/{checkpoint_id}.json
models/evaluations/manifests/sha256/{evaluation_id}.json
models/run-commits/sha256/{run_commit_id}.json
models/channels/current.json
```

`checkpoint_id` is the SHA-256 of the canonical manifest bytes. The manifest
binds the exact profile, step, parent checkpoint, learner-config digest, and
both blob descriptors. A RunCommit binds one checkpoint, an exact embedded stats
snapshot, champion/evaluation state, and its parent RunCommit. `current` is the
sole mutable RunHead for inference and learner continuity. Staging creates only
immutable objects; full RunCommit/checkpoint lineages are validated before
`current` advances with compare-and-set semantics.

Each `evaluation_id` is the digest of canonical promotion evidence binding the
candidate checkpoint, prior champion/evaluation lineage, deterministic
seat-and-seed recipe, requested counts, observed head-to-head/solver results,
and decision. Champion state exists only in the selected RunCommit and is valid
only when that evidence and the full lineage resolve. Solver promotion
uses fresh symmetric candidate and incumbent runs; historical projections are
never reused as decision evidence.

Artifact identity schema version 1 is enforced inside both blobs:

- ONNX stores `cartridge.schema_version=1`, `cartridge.algorithm_id`,
  `cartridge.model_contract`, `cartridge.env_id`, and
  `cartridge.env_contract_version` as custom metadata.
- Learner state stores top-level `schema_version=1`, the exact profile, step,
  config digest, and model/optimizer/scheduler state.

Resume and inference verify the RunHead, RunCommit lineage, manifest digest,
profile, blob size/digest, and embedded artifact contract. An absent RunHead
means the learner starts fresh; a present invalid object raises. Old mutable checkpoint
filenames and ONNX exports are intentionally not accepted and must be
retrained/re-exported through the current publisher. There is no automatic
converter or shape-based fallback.

### Statistics Tracking

Statistics authority is part of the immutable RunCommit:

```text
models/run-commits/sha256/{run_commit_id}.json  # embeds exact stats snapshot
models/channels/current.json                    # sole mutable RunHead
stats.json                                      # disposable web projection
```

`stats_id` is the SHA-256 of the canonical embedded snapshot. Learner resume
validates it together with the selected RunCommit and checkpoint lineage; it
never reads `stats.json`, `eval_stats.json`, `solver_stats.json`, or
`loop_stats.json` as authority. Those files are disposable projections rebuilt
from the selected chain. An absent RunHead starts empty, while present corrupt
or incomplete authority fails closed.

```python
@dataclass
class TrainerStats:
    step: int
    total_steps: int
    metrics: Dict[str, float] # Algorithm-owned names such as loss/td
    learning_rate: float
    replay_record_count: int
    history: List[Dict]             # Step + metrics (downsampled)
    evaluation_history: List[Dict]  # Episode counts + arbitrary metrics
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
| `/model` | GET | Model info |

### GameSession

```rust
pub struct GameSession {
    ctx: EngineContext,
    board: BoardGameMetadata,
    state: Vec<u8>,
    timestep: ErasedTimestep,
    view: BoardView,
    human_player: u8,
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    mcts_config: MctsConfig,
}
```

Bot AI:
1. Narrow the generic timestep to one active board agent and its observation
2. If model loaded: Run MCTS (200 sims, temp=0.5)
3. If no model: Random legal move
4. Execute the selected action and validate the next timestep/presentation

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
profile = ReplayProfile(
    env_id="connect4",
    env_contract_version=1,
    algorithm_id="alphazero_board_v1",
    experience_schema="alphazero_transition_v1",
)
selection = ReplaySelection(
    profile=profile,
    collection_scope_id="<64 lowercase hex characters>",
    source_checkpoint_id="<checkpoint_id or None at root>",
)

# Connection string comes from CARTRIDGE_STORAGE_POSTGRES_URL...
replay = create_replay_store(selection)
# ...or is passed explicitly.
replay = create_replay_store(
    selection,
    connection_string="postgresql://user:pass@host:5432/db"
)
```

`ReplaySelection` is an isolation boundary, not just query metadata. The store's
`count`, `sample`, `clear`, `cleanup`, `store`, and `store_batch` operations
always filter by the exact
`(env_id, env_contract_version, algorithm_id, experience_schema,
collection_scope_id, source_checkpoint_id)` tuple. Nullable source identity is
compared with `IS NOT DISTINCT FROM`, never as a wildcard. Writes are rejected
if a `ReplayRecord` does not match the bound selection.
Storage interprets none of the payload bytes; the algorithm cartridge named by
`(algorithm_id, experience_schema)` owns their codec.

Schema — the authoritative copy is [`sql/schema.sql`](../sql/schema.sql), which
the Rust actor embeds at compile time (`include_str!`) and the Python trainer
reads at runtime:

```sql
CREATE TABLE IF NOT EXISTS cartridge_schema_versions (
    component TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
);

INSERT INTO cartridge_schema_versions (component, schema_version)
VALUES ('replay', 3)
ON CONFLICT (component) DO NOTHING;

CREATE TABLE IF NOT EXISTS replay_records (
    id TEXT NOT NULL,
    env_id TEXT NOT NULL,
    env_contract_version BIGINT NOT NULL
        CHECK (env_contract_version BETWEEN 1 AND 4294967295),
    algorithm_id TEXT NOT NULL,
    experience_schema TEXT NOT NULL,
    collection_scope_id TEXT NOT NULL
        CHECK (collection_scope_id ~ '^[0-9a-f]{64}$'),
    source_checkpoint_id TEXT
        CHECK (
            source_checkpoint_id IS NULL
            OR source_checkpoint_id ~ '^[0-9a-f]{64}$'
        ),
    episode_id TEXT NOT NULL,
    step_number BIGINT NOT NULL
        CHECK (step_number BETWEEN 0 AND 4294967295),
    payload BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        env_id, env_contract_version, algorithm_id, experience_schema,
        collection_scope_id, id
    )
);

CREATE INDEX IF NOT EXISTS idx_replay_records_selection_created
    ON replay_records(
        env_id, env_contract_version, algorithm_id, experience_schema,
        collection_scope_id, source_checkpoint_id, created_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_replay_records_selection_episode
    ON replay_records(
        env_id, env_contract_version, algorithm_id, experience_schema,
        collection_scope_id, source_checkpoint_id, episode_id, step_number
    );
```

Selection/creation-time and selection/episode indexes support concurrent
writers, sampling, retention cleanup, and episode inspection without adding
algorithm-specific columns.

> **Clean schema cutover:** existing replay databases must be recreated from
> `sql/schema.sql`. Cartridge2 deliberately provides no migration from the old
> concrete transition columns to an opaque payload because there is no honest,
> generic codec that can be inferred for those rows.

The runtime requires the exact replay protocol marker
`cartridge_schema_versions(component='replay', schema_version=3)`. The actor
embeds `sql/schema.sql`, and the trainer's packaged copy is byte-identical.
Compose and K8s bootstrap scripts implement the same tables, constraints, keys,
and indexes, then add deployment-specific grants. The only tables in the replay
contract are `cartridge_schema_versions` and `replay_records`; there are no
board, environment-metadata, model, or training-stat tables.

`alphazero_board_v1` owns `alphazero_transition_v1`. Its exact language-neutral
payload is a concatenation of little-endian `f32` values:

```text
observation[obs_size] || policy[num_actions] || terminal_value[1]
```

The actor validates and encodes this layout only after an episode has a terminal
target. The learner validates record lineage, payload length, finite values,
policy bounds and sum, and value bounds before constructing training tensors.

### 7.3 Environment and algorithm catalog single-sourcing

Environment facts and algorithm contracts are declared in Rust and flow
outward through one generated catalog:

```text
Environment::metadata() ──────┐
Environment::capabilities() ──┼──► engine-games manifest generator
algorithm-core catalog ───────┘                  │
                                                ▼
                      trainer/src/trainer/environment_manifest.json
                                                │
                           environment_catalog.py + algorithms/

EnvironmentMetadata.board ──► AlphaZero actor/web
algorithm descriptor + environment contract ──► ReplayProfile
orchestrator attempt + source checkpoint ──────► ReplaySelection
AlphaZero board dimensions ──► alphazero_transition_v1 codec
                                         │
                                         ▼
                               ReplayRecord.payload
                                         │
                                         ▼
                                  replay_records
```

- **The manifest is generated, not written.** `make environment-manifest`
  renders schema version 5 to
  `trainer/src/trainer/environment_manifest.json`. It ships inside the Python
  package and is loaded with `importlib.resources`. The `engine-games` golden
  test fails when the committed catalog drifts.
- **The v4 shape is generic.** The top level contains `algorithms` and
  `environments`; each environment is exactly `{metadata, capabilities,
  algorithm_profiles}`. Generic metadata is `{id, display_name, description,
  board}` and `board` may be null. Capabilities carry identity/version,
  encodings, semantics (including chance and rewards emitted per agent), optional
  horizon, and fixed/dynamic agents with action spaces.
- **Algorithms are first-class catalog entries.** The top-level `algorithms`
  collection carries descriptors and requirements. Every environment carries
  an `algorithm_profiles` map whose values include `compatible`, structured
  issues, and `unverified_assumptions`.
- **Python separates environment facts from algorithm recipes.**
  `environment_catalog.py` parses the engine-owned catalog. An installed
  algorithm module, such as `algorithms/alphazero_board_v1.py`, owns its network
  recipe and concrete factories. Merely adding an environment does not force it
  into an AlphaZero configuration table.
- **Replay storage is algorithm-neutral.** Environment compatibility supplies
  the dimensions required by an installed algorithm codec, while the database
  stores only the profile-bound envelope and opaque payload. Changing either
  the environment contract or payload codec means a new profile namespace.

The AlphaZero compatibility guard asserts the nested board-layout invariants:

```text
board.observation.legal_actions_offset
    == board.observation.spatial_channels * board.width * board.height
board.observation.elements
    == board.observation.legal_actions_offset + board.action_count + 2
```

### Model Storage

#### Filesystem (Local)

```text
./data/profiles/{algorithm}/{env}/v{env_contract_version}/
├── models/
│   ├── blobs/sha256/{onnx_digest}.onnx
│   ├── blobs/sha256/{learner_digest}.pt
│   ├── manifests/sha256/{checkpoint_id}.json
│   ├── evaluations/manifests/sha256/{evaluation_id}.json
│   ├── run-commits/sha256/{run_commit_id}.json
│   ├── run-preparations/by-parent/{root|parent_run_commit_id}.json
│   └── channels/current.json # Sole RunHeadV2 authority
├── stats.json                # Web projection
└── eval_stats.json, players.json, ...
```

#### S3 (Kubernetes)

```text
s3://{bucket}/profiles/{algorithm}/{env}/v{env_contract_version}/models/
├── blobs/sha256/{onnx_digest}.onnx
├── blobs/sha256/{learner_digest}.pt
├── manifests/sha256/{checkpoint_id}.json
├── evaluations/manifests/sha256/{evaluation_id}.json
├── run-commits/sha256/{run_commit_id}.json
├── run-preparations/by-parent/{root|parent_run_commit_id}.json
└── channels/current.json
```

The checkpoint ID is the SHA-256 digest of the canonical manifest bytes. Each
manifest binds the runtime profile, step, parent checkpoint, learner-config
digest, and the size/digest of both immutable blobs. Publication validates the
complete ONNX graph, tensor interface, dtypes, shapes, and five identity fields,
materializes all immutable objects, and only then compare-and-sets the sole
RunHead. Filesystem and S3 use the same repository layout and
validation contract.

S3 uses the exact `channels/current.json` object ETag as its serialization
token. Initial creation is conditional on absence; later advancement is
conditional on the ETag that was read and validated. There is no lock object,
lease, timeout, or recovery command. Filesystem publication uses a local
advisory directory lock and an exact compare-and-set check.

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

`config.defaults.toml` is loaded by both languages. Rust embeds the repository
copy at compile time (`include_str!`). The Python wheel ships a byte-identical
package resource; a source checkout prefers the repository copy and fails if
the two copies diverge. Python then deep-merges `config.toml` over those
canonical defaults. Missing or incomplete defaults are fatal.

Sections: `[common]`, `[algorithm]`, `[training]`, `[evaluation]`, `[actor]`, `[web]`,
`[mcts]`, `[logging]`, `[storage]`, `[wandb]`.

Both languages parse the complete schema, including `[wandb]` and the
solver/promotion fields. Individual processes act only on the settings in
their responsibility, but every component still rejects malformed or unknown
configuration before selecting a runtime profile.

### Environment Variable Format

Canonical format:
```bash
CARTRIDGE_ALGORITHM_ID=alphazero_board_v1
CARTRIDGE_COMMON_ENV_ID=connect4
CARTRIDGE_TRAINING_ITERATIONS=50
CARTRIDGE_MCTS_START_SIMS=100
CARTRIDGE_MCTS_MAX_SIMS=800
```

Python derives `CARTRIDGE_<SECTION>_<KEY>` names from the typed schema; Rust
maps the same complete set explicitly in `engine/engine-config/src/loader.rs`.
Unknown names are errors, not ignored compatibility aliases. Lists such as
`allowed_origins` and W&B tags use comma-separated values.

### Search Paths

Rust (`engine/engine-config/src/loader.rs`):

1. `$CARTRIDGE_CONFIG` (if set)
2. `./config.toml`
3. `../config.toml`
4. `/app/config.toml` (Docker)

Python (`trainer/src/trainer/central_config.py`) uses `./config.toml`,
`/app/config.toml`, then a source-checkout project-root fallback. Installed
wheels fall back to their packaged canonical defaults resource.

---

## 9. Data Flow

### Self-Play Data Flow

```
Orchestrator             Bounded collectors       PostgreSQL          Learner
     │                           │                      │                 │
     │ 1. New scope + source    │                      │                 │
     │─────────────────────────►│                      │                 │
     │                           │ 2. Load Latest once  │                 │
     │                           │ 3. Complete episodes │                 │
     │                           │ 4. Store exact rows ─►│                 │
     │◄──────── finite exit ─────│                      │                 │
     │ 5. Require exact distinct-episode seal          │                 │
     │────────────────────────────────────────────────►│                 │
     │                           │                      │◄─ 6. Sample ────│
     │                           │                      │   exact scope   │
     │                           │                      │                 │ 7. Train
     │◄──────────────────────────────────────────────────────────────────│
     │ 8. Evaluate, publish immutable evidence, CAS RunHead              │
```

The orchestrator gives every collector and the learner the same complete
`ReplaySelection`. Each row persists the scope and source lineage; every store
operation applies that exact tuple, including null-exact source matching. The
algorithm codec is the only layer that interprets `payload`. A failed attempt
gets a new scope rather than clearing or reusing the failed one.

### Statistics Flow

```
Trainer                    Artifact repository       Projection       Web
   │                                │                    │             │
   │ 1. Immutable RunCommit ───────►│                    │             │
   │    (embeds exact stats)        │                    │             │
   │ 2. CAS sole RunHead ──────────►│                    │             │
   │ 3. Rebuild from selected chain ├───────────────────►│ stats.json  │
   │                                │                    │◄── GET ─────│
```

`stats.json`, `eval_stats.json`, `solver_stats.json`, and `loop_stats.json` are
disposable projections. Live `stats_interval` refreshes may temporarily be
newer than RunHead and are intentionally lost after a crash; startup rebuilds
from the authoritative selected chain.

### Checkpoint Publication Flow

```text
Export validated ONNX + learner state
                  │
                  ▼
Hash and materialize immutable blobs + checkpoint manifest
                  │
                  ▼
Run evaluation; write immutable EvaluationArtifactV2 when scheduled
                  │
                  ▼
Create canonical RunCommitV1 with checkpoint, stats, recipe, replay scope,
evaluation/champion state, and parent lineage
                  │
                  ▼
Validate complete immutable chains and prepared-run journal
                  │
                  ▼
Compare-and-set sole RunHeadV2 at models/channels/current.json
                  │
                  ├─ Latest ───────────► learner and next bounded collectors
                  └─ ChampionOrLatest ─► web inference
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
| prometheus | 9092 | scrapes trainer:9090 and web:8080 |

#### Local Development

```bash
# Synchronized AlphaZero training
docker compose up alphazero

# Play in browser
docker compose up web frontend
# Open http://localhost

# One-off command through the image's canonical trainer entrypoint.
docker compose run --rm alphazero \
  --algorithm alphazero_board_v1 evaluate --env-id connect4
```

Compose already uses PostgreSQL and private MinIO. Kubernetes uses the
Kustomize manifests under `k8s/`; one `job/trainer` owns the synchronized loop
and spawns its configured bounded collectors. There is deliberately no actor
Deployment or independently scaled collector pool.

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

### Observability

The trainer and web process expose Prometheus scrape targets (see
[`prometheus.yml`](../prometheus.yml)). Bounded actor children have no HTTP
service lifecycle; they emit structured final snapshots and are supervised by
the parent orchestrator.

| Component | Port | Notable metrics |
|-----------|------|-----------------|
| trainer | 9090 | training step/loss counters and `trainer_replay_record_count` (`trainer/src/trainer/metrics.py`) |
| web | 8080 (`/metrics`) | `web_games_created_total`, `web_games_active`, `web_moves_played_total`, `web_games_completed_total`, `web_request_duration_seconds{endpoint,method}`, `web_bot_move_seconds`, `web_model_loaded`, `web_model_reloads_total` |

Bounded actor children deliberately expose no Prometheus endpoint. Their
process-local `ActorStats`, abandonment reason, discarded-record count, and RSS
are emitted as structured logs; the parent treats a failed child as a failed
scope attempt instead of silently retrying within that scope.

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
| engine-core | Generic Environment/Timestep ABI, validation, registry, context, optional profiles |
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

- **`engine-games` environment-manifest golden test** — fails when
  `trainer/src/trainer/environment_manifest.json` drifts from the Rust
  environment or algorithm contracts. Regenerate with
  `make environment-manifest`.
- **AlphaZero board-profile invariants** — asserted by the Rust compatibility
  guard and the strict Python catalog consumer. Generic environments without a
  board profile are valid engine registrations but incompatible with this
  cartridge.

### CI Pipeline

GitHub Actions workflow:
1. **rust-fmt**: Check Rust formatting
2. **rust-clippy**: Lint with warnings as errors
3. **rust-test**: Full test suite
4. **rust-build**: Release build
5. **rust-security-audit**: cargo audit (non-blocking)
6. **python-lint**: Check Ruff + Black
7. **python-test**: Pytest
8. **python-security-audit**: pip-audit (non-blocking)
9. **frontend**: Svelte check + build
10. **docker-build**: Docker image build validation
11. **secrets-scan**: GitLeaks (non-blocking)

The workflow is check-only and has read-only repository permissions; no job
rewrites or commits to a branch.

---

## Appendix: Key Design Patterns

### Type Erasure via Adapter

```
Typed Environment<State, Action, Observation>
        ↓
private EnvironmentAdapter<E: Environment>
        ↓
private Box<dyn ErasedEnvironment>
        ↓
Registry HashMap<String, Factory>
        ↓
public EngineContext
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

let metadata = EnvironmentMetadata::new("tictactoe", "Tic-Tac-Toe")
    .with_board(
        BoardGameMetadata::new(3, 3, 9)
            .with_observation(29, 2, 18, false),
    );
```

### Atomic File Operations

Write-then-rename pattern:
```
1. Write to file.tmp
2. Atomic rename file.tmp → file
```

Ensures readers never see partial content.

### Web Hot-Reload Strategy

```
inotify watcher (fast, event-based)
        ↓
    fallback
        ↓
Polling timer (reliable in Docker)
```

This applies to the web watcher. Bounded collectors resolve Latest once, verify
the requested source checkpoint, drop the watcher, and never change evaluators
during an attempt.

---

## Appendix: Quick Reference

### Common Commands

```bash
# Local training
python -m trainer --algorithm alphazero_board_v1 loop \
  --iterations 50 --episodes 500 --steps 1000

# Docker training
docker compose up alphazero

# Play in browser
docker compose up web frontend

# Evaluation
python -m trainer --algorithm alphazero_board_v1 evaluate --games 100

# Clean one exact runtime profile (destructive; preserve anything needed first)
make clean ALGORITHM=alphazero_board_v1 ENV_ID=tictactoe

# Clean PostgreSQL replay buffer volume (removes all compose volumes)
docker compose down -v
```

### File Locations

| File | Purpose |
|------|---------|
| `config.defaults.toml` | Checked-in defaults; source of truth for every key |
| `config.toml` | Local overrides |
| `sql/schema.sql` | Database schema (embedded by Rust, read by Python) |
| `trainer/src/trainer/environment_manifest.json` | **Generated** environment/algorithm manifest (`make environment-manifest`) |
| `data/profiles/{algorithm}/{env}/v{contract}/models/blobs/sha256/{digest}.onnx` | Immutable inference blob |
| `data/profiles/{algorithm}/{env}/v{contract}/models/blobs/sha256/{digest}.pt` | Immutable learner-state blob |
| `data/profiles/{algorithm}/{env}/v{contract}/models/manifests/sha256/{checkpoint_id}.json` | Canonical immutable checkpoint manifest |
| `data/profiles/{algorithm}/{env}/v{contract}/models/run-commits/sha256/{run_commit_id}.json` | Immutable checkpoint, statistics, evaluation/champion, replay-scope, and recipe commit |
| `data/profiles/{algorithm}/{env}/v{contract}/models/channels/current.json` | Sole mutable `RunHeadV2` authority |
| `data/profiles/{algorithm}/{env}/v{contract}/stats.json` | Disposable web-facing statistics projection |
| `data/profiles/{algorithm}/{env}/v{contract}/eval_stats.json` | Disposable evaluation-history projection |
| `data/profiles/{algorithm}/{env}/v{contract}/solver_stats.json` | Disposable solver-history projection (Connect 4) |
| `data/profiles/{algorithm}/{env}/v{contract}/loop_stats.json` | Disposable orchestrator-history projection |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CARTRIDGE_COMMON_ENV_ID` | Game to train |
| `CARTRIDGE_TRAINING_DEVICE` | cpu, cuda, mps |
| `CARTRIDGE_MCTS_START_SIMS` | First-iteration MCTS simulations |
| `CARTRIDGE_MCTS_MAX_SIMS` | Maximum ramped MCTS simulations |
| `CARTRIDGE_MCTS_SIM_RAMP_RATE` | Simulations added per iteration |
| `CARTRIDGE_STORAGE_MODEL_BACKEND` | filesystem, s3 |
| `CARTRIDGE_STORAGE_POSTGRES_URL` | PostgreSQL connection |
