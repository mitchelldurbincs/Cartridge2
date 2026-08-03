# Cartridge2 - Claude Code Guide

## Project Overview

Cartridge2 is an algorithm-oriented reinforcement-learning platform. Game
environments and learning algorithms are registered independently, and startup
composition binds a collector, learner, orchestration recipe, experience
schema, model contract, evaluation suite, and serving implementation only when
their compatibility profile passes.

The first installed algorithm is `alphazero_board_v1`. It is the working
AlphaZero board-game stack, now isolated behind explicit Rust and Python
algorithm registries rather than embedded in the generic environment ABI.

**Target Games:** TicTacToe (complete), Connect 4 (complete), Othello (complete), Generals 8×8 (`generals_8x8` — engine/trainer/web complete; training does not yet beat random at local compute scale)

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
┌──────────────────────────────────────────────────────────────────────┐
│ PostgreSQL: exact-profile replay rows                                │
│ Runtime storage: data/profiles/{algorithm}/{env}/v{contract}/        │
│   models/ + profile-scoped telemetry (filesystem or S3 for models)   │
└──────────────────────────────────────────────────────────────────────┘
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

### Algorithm contract

`[algorithm].id` is a required architecture choice, with
`alphazero_board_v1` as the checked-in default. The canonical Rust descriptor
is in `engine/algorithm-core`; Python binds implementations in
`trainer/algorithms/registry.py`. Do not put algorithm-specific learner or
collector assumptions into the generic environment catalog.

`make environment-manifest` generates
`trainer/src/trainer/environment_manifest.json` schema version 4. Its top-level
`environments` entries are exactly `{metadata, capabilities,
algorithm_profiles}`. `metadata.board` is an optional presentation/board-game
profile; generic capabilities own contract version, codecs, agents and their
action spaces, optional horizon, and explicit turn/information/planning/chance/
transition/reward semantics. Actor, trainer, evaluator, and web startup must
resolve the algorithm ID and call `require_compatible()` before creating
storage clients, watching models, or launching work.

The current AlphaZero cartridge is intentionally narrow: two fixed alternating players,
discrete actions, perfect information, deterministic planning state,
fixed spatial `f32` observations with a legal mask and player indicator, and
terminal zero-sum rewards. Registration alone is not evidence that a new
environment satisfies this profile. The generic ABI itself also represents
single-agent/simultaneous decisions, dynamic agents, chance, stochastic or
partially observed environments, general per-agent rewards, truncation, and
multi-discrete/continuous actions for future matching cartridges.

Model artifacts use a content-addressed repository beneath each profile's
`models/` root: immutable `blobs/sha256/*` and
`manifests/sha256/{checkpoint_id}.json`, with only
`channels/current.json` as the authoritative mutable `RunHeadV2`. Learner
continuity and collection use its latest checkpoint; web inference uses
champion state from the latest selected RunCommit, falling back to latest before
the first promotion. The checkpoint ID is the SHA-256 of canonical manifest
bytes. ONNX custom metadata must contain
exact `cartridge.algorithm_id`, `cartridge.model_contract`, and
`cartridge.env_id` values, plus `cartridge.schema_version=1` and
`cartridge.env_contract_version`. Learner-state blobs embed the exact profile,
step, and config digest. Keep pointer, manifest, blob, identity, and tensor
validation at every load boundary. An absent head permits a fresh/random
start, but a present invalid object must fail initial loading. A rejected hot
reload must leave the current valid evaluator in place.

Trainer statistics are the exact canonical snapshot embedded in the immutable
`RunCommitV1` selected by the sole `RunHeadV2`; `stats_id` hashes those bytes.
The profile-root `stats.json` is only the disposable web projection and must
never be used as learner resume authority. Existing partial or corrupt
RunCommit/checkpoint authority fails closed.

Every mutable file and model object belongs under the canonical namespace
`profiles/{algorithm_id}/{env_id}/v{env_contract_version}`. Never add a shared
root-level `models/`, stats file, player registry, or implicit environment
fallback.

## Programming
**IMPORTANT** Make sure to run tests / linters for the code changes you make.

### Rust (engine, actor, web)
```bash
# Format
cargo fmt --all --check --manifest-path engine/Cargo.toml
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
python -m ruff check src/ tests/ smoke_test.py
python -m black --check src/ tests/ smoke_test.py

# Auto-fix lint issues
python -m ruff check --fix src/ tests/ smoke_test.py
python -m black src/ tests/ smoke_test.py

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

- `algorithm-core/` - Canonical algorithm IDs, language-neutral component descriptors, compatibility reports, and Rust dispatch keys
- `engine-core/` - Generic `Environment`/`Timestep` ABI, validated erased adapter,
  registry, `EngineContext`, optional board metadata/presentation, and `LegalMask`
- `engine-config/` - Centralized configuration loading from config.toml
- `engine-games/` - Registration of bundled environments, board-profile
  invariants, and the schema-v4 environment/algorithm manifest generator
  (`make environment-manifest`) with its golden drift test
- `evaluator/` - `cartridge-eval`: plays evaluation games through the engine and
  writes a JSON summary (plus, optionally, every position it saw). The trainer
  shells out to this instead of implementing games in Python; see
  `documentation/ARCHITECTURE.md` for why
- `games-tictactoe/` - TicTacToe implementation
- `games-connect4/` - Connect 4 implementation
- `games-othello/` - Othello implementation
- `games-generals/` - Generals 8×8 implementation; see the crate's
  lib.rs for the ruleset (full-info, alternating turns, territory
  adjudication, Markov-visible parity-randomized ply cap) and `generals_obs:v2` layout
- `mcts/` - Monte Carlo Tree Search implementation; legal masks are
  dynamic-width (`LegalMask`) and read from the authoritative observation.
  Three diagnostic examples: `generals_policy_probe` (visit-distribution
  health), `generals_strength_probe` (MCTS+model vs random — the honest
  strength measure; the trainer's built-in eval is argmax-only and understates
  models), and `generals_search_diag` (branching factor vs search budget)
- `model-watcher/` - RunHead validation, web hot reload, and actor one-shot load
- `metrics-common/` - Prometheus registration/encoding used by serving/runtime components

### Actor (Rust Binary) - `actor/`
**Status: COMPLETE**

Algorithm-dispatched experience collector using engine-core directly:

- Resolves `[algorithm].id` and validates its environment profile before model watching or replay storage starts
- `alphazero_board_v1` binds to `AlphaZeroCollector`
- Uses `EngineContext` for game simulation (no gRPC)
- PostgreSQL storage backend (local development or K8s)
- MCTS policy with ONNX neural network evaluation
- Resolves the selected profile's content-addressed `channels/current.json`
  exactly once (via model-watcher, from filesystem or S3), verifies it equals
  the requested source checkpoint, then drops the watcher before collection
- Stores MCTS visit distributions as policy targets (raw tau=1, never sharpened
  by the play temperature)
- Stores an opaque AlphaZero payload inside a replay-v3 envelope tagged with
  `env_id`, `env_contract_version`, `algorithm_id`, `experience_schema`,
  `collection_scope_id`, and nullable `source_checkpoint_id`
- Terminal outcome encoding for value targets
- Requires the optional board profile from `EnvironmentMetadata`; storage never
  persists board metadata or interprets the algorithm-owned payload
- Episodes that time out are discarded whole — there is no outcome to backfill —
  and included in final structured `ActorStats`/RSS logs; the bounded worker
  then fails so the parent can abandon that scope

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

Algorithm-dispatched Python learning and orchestration. The installed
`alphazero_board_v1` binding provides:

**CLI Commands:** The only syntax is
`python -m trainer --algorithm <id> <command> [options]`. The installed
AlphaZero cartridge owns `train`, `evaluate`, `loop`, `solver-eval`,
`register-players`, and `tournament`.

**Features:**
- Reads exact-selection-bound replay records and decodes
  `alphazero_transition_v1`
- Uses an exact `(env_id, env_contract_version, algorithm_id,
  experience_schema, collection_scope_id, source_checkpoint_id)` replay fence;
  count, sample, clear, and cleanup cannot cross that selection
- MCTS policy distributions as soft targets
- Game outcome propagation for value targets
- MLP network for TicTacToe, ResNet for spatial games (Connect4, Othello, Generals)
- Evaluation games are played by the `cartridge-eval` binary (subprocess, like
  the actor), so the trainer holds no game rules; `make build-eval` builds it
- Environment facts, algorithm descriptors, and compatibility reports read
  from the engine-generated manifest (`environment_manifest.json`); the AlphaZero
  cartridge owns its network recipes
- Exports ONNX models with atomic write-then-rename
- Rebuilds disposable `stats.json` and `eval_stats.json` projections from the
  selected immutable RunCommit chain
- Cosine annealing LR schedule with warmup
- Gradient clipping for stability
- Model evaluation against random baseline (enabled by default in loop)
- Orchestrator derives missing iterations from the validated RunHead chain
- MCTS simulation ramping (start low, increase over iterations)
- Structured JSON logging for cloud deployments
- Prometheus metrics export

### crucible Dependency (Python) - sibling repo
**Status: EXTRACTED (github.com/mitchelldurbincs/crucible; not on PyPI — CI installs it from GitHub pinned to a commit, local dev uses the editable sibling checkout)**

Crucible supplies the generic base `Orchestrator` and actor-runner protocol,
iteration value types, `wandb_logger`, `atomic_io`, and `backoff`. Cartridge2
owns exact replay fencing, the prepared-run journal, immutable
EvaluationArtifact/RunCommit publication, recovery, promotion validation, and
disposable stats/evaluation projections. Algorithm cartridges inject the
concrete collection, learning, and evaluation implementations.

**Dev setup** - install crucible editable before the trainer:
```bash
cd trainer
pip install -e ../../crucible   # sibling checkout, relative to trainer/
pip install -e ".[dev]"
```

**Composition root:** `trainer/src/trainer/orchestrator/orchestrator.py`
resolves the requested algorithm, environment compatibility, and authenticated
RunRecipe before opening an attempt scope, then injects that algorithm's
learner, collector runner, and evaluation runner into Crucible.

## Directory Structure

```
cartridge2/
├── actor/                  # Rust actor binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Entry point
│       ├── actor.rs        # AlphaZero collector implementation
│       ├── algorithms.rs   # Collector dispatch
│       ├── config.rs       # CLI configuration (uses engine-config)
│       ├── mcts_policy.rs  # MCTS policy implementation
│       ├── resources.rs    # Process resource diagnostics
│       ├── stats.rs        # Self-play statistics
│       └── storage/        # Storage backends (PostgreSQL)
├── engine/                 # Rust workspace
│   ├── Cargo.toml         # Workspace config
│   ├── algorithm-core/   # Algorithm catalog + compatibility checks
│   ├── engine-core/       # Generic Environment ABI + EngineContext
│   │   └── src/
│   │       ├── contract.rs # Shared descriptor/timestep validation
│   │       ├── adapter.rs  # Private typed-to-erased boundary
│   │       ├── board_game.rs # Narrow board family -> generic ABI
│   │       ├── board_view.rs # Optional presentation projection
│   │       ├── context.rs  # EngineContext high-level API
│   │       ├── erased.rs   # Sealed bytes-only runtime boundary
│   │       ├── board_game_utils.rs # Narrow two-player board helpers
│   │       ├── metadata.rs # Generic + optional board metadata
│   │       ├── registry.rs # Static game registration
│   │       └── typed.rs    # Environment, Timestep, capabilities
│   ├── engine-config/     # Centralized configuration (shared by actor/web)
│   │   ├── src/
│   │   │   ├── lib.rs      # Public API exports
│   │   │   ├── defaults.rs # Default configuration values
│   │   │   ├── structs.rs  # Config struct definitions
│   │   │   ├── loader.rs   # Loading logic + env overrides
│   │   │   └── tests.rs    # Unit tests
│   │   └── SCHEMA.md       # Configuration schema documentation
│   ├── envs-counter/      # Direct non-board Environment reference
│   ├── engine-games/      # Registration + manifest generator + golden test
│   ├── evaluator/         # cartridge-eval binary (evaluation games)
│   ├── metrics-common/    # Prometheus registration/encoding utilities
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
│   └── model-watcher/     # RunHead validation and model selection/loading
├── web/                    # Web server + frontend
│   ├── Cargo.toml         # Axum server
│   ├── src/
│   │   ├── main.rs        # Thin entry point
│   │   ├── startup.rs     # Router, AppState, CORS, shutdown
│   │   ├── game.rs        # Game session management
│   │   ├── metrics.rs     # Prometheus metrics
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
│   ├── tests/             # Pytest suite
│   └── src/trainer/
│       ├── __main__.py    # CLI entrypoint (train, evaluate, loop, solver-eval)
│       ├── algorithms/    # Installed Python algorithm implementations
│       ├── environment_catalog.py # Strict manifest-v4 catalog
│       ├── runtime_profile.py # Canonical artifact namespace
│       ├── trainer.py     # AlphaZeroLearner
│       ├── network.py     # Neural network (MLP, used for TicTacToe)
│       ├── resnet.py      # ResNet architecture (used for Connect4, Othello)
│       ├── evaluator.py   # Model evaluation
│       ├── solver_eval/   # Perfect-solver move-quality evaluation (Connect4)
│       ├── replay_setup.py # Exact ReplaySelection setup
│       ├── environment_manifest.json # GENERATED by `make environment-manifest`
│       ├── stats.py       # Training statistics
│       ├── config.py      # AlphaZeroLearnerConfig dataclass
│       ├── lr_scheduler.py # LR schedule (warmup + cosine annealing)
│       ├── checkpoint.py  # Checkpoint save/load utilities
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
│       │   └── eval_reporting.py # Evaluation/result adapter
│       ├── players.py     # Who occupies a seat in an evaluation game
│       ├── registry.py    # Immutable player registry schema v5
│       ├── tournament.py  # Round-robin + Bradley-Terry Elo rating
│       └── storage/       # Exact replay selection + artifact publication
├── Dockerfile.alphazero   # Combined actor+trainer image for Docker
├── docker-compose.yml     # Local services (postgres, minio, training, web)
├── k8s/                   # Kustomize manifests (S3 model backend)
├── Makefile               # Convenience targets (setup, test, lint, train)
├── config.toml            # Central configuration file
├── config.defaults.toml   # Default values (single source of truth)
├── .github/workflows/
│   └── ci.yml             # CI pipeline (Rust fmt/clippy/test, Python lint/test, frontend build)
├── documentation/
│   ├── ARCHITECTURE.md    # Comprehensive architecture reference
│   └── API.md             # REST API documentation with examples
├── data/                  # Runtime root (gitignored)
│   └── profiles/{algorithm}/{env}/v{contract}/ # Models + telemetry
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
2. **Environment variables** - `CARTRIDGE_<SECTION>_<KEY>`
3. **`config.toml`** - your local overrides
4. **`config.defaults.toml`** - checked-in defaults

### Gotchas worth knowing

- **Config is one strict cross-language schema.** Python derives
  `CARTRIDGE_<SECTION>_<KEY>` overrides from its typed schema; Rust explicitly
  maps every declared field in `engine/engine-config/src/loader.rs`. Unknown
  TOML fields and malformed environment values fail instead of being ignored.
  Lists use comma-separated environment values.
- **Some settings have one runtime consumer.** Rust still parses `[wandb]` and
  solver/promotion settings so the shared document stays valid, while Python
  orchestration is the component that acts on them.
- **Empty `allowed_origins` does not mean "allow all".** `configure_cors` in
  `web/src/startup.rs` is deny-by-default: empty falls back to a localhost
  allowlist.
- **The trainer reads the replay DSN only from `CARTRIDGE_STORAGE_POSTGRES_URL`.**
  `storage.postgres_url` in `config.toml` is used by the Rust actor and web
  server, but *not* by the Python trainer.
- **Replay schema v3 is a clean cutover.** Existing databases do not have the
  required `env_contract_version`, `algorithm_id`, `experience_schema`,
  `collection_scope_id`, and nullable `source_checkpoint_id` fence or composite
  primary key. Recreate the database from `sql/schema.sql`; there is no data
  migration or implicit default profile.
- **The Phase 1 model format is also a clean cutover.** Old PyTorch checkpoints
  and ONNX exports without schema-v1 identity are rejected, even when their
  tensor shapes happen to fit. Retrain/re-export with the current code; do not
  add inferred IDs, compatibility loaders, or defaults for missing metadata.

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

# Select the installed algorithm explicitly
CARTRIDGE_ALGORITHM_ID=alphazero_board_v1 docker compose up alphazero

# Run in background
docker compose up alphazero -d
docker compose logs -f alphazero  # Watch progress

# Run a one-off cartridge command
docker compose run --rm alphazero \
  --algorithm alphazero_board_v1 evaluate --env-id connect4

# Play against trained model (in another terminal)
docker compose up web frontend
# Open http://localhost in browser
```

**To customize training:** Edit `config.toml` before running, or use environment variable overrides. See the Configuration section above for all available settings.

## Commands

```bash
# Build engine
cargo build --release --manifest-path engine/Cargo.toml

# Build actor
cargo build --release --manifest-path actor/Cargo.toml

# Build web server
cargo build --release --manifest-path web/Cargo.toml

# Run all tests (engine + actor + web + trainer)
make test

# Or individually
cargo test --manifest-path engine/Cargo.toml
cargo test --manifest-path actor/Cargo.toml
cargo test --manifest-path web/Cargo.toml
python -m pytest trainer/tests/ -v --tb=short

# Build the evaluation binary the trainer shells out to for every eval
make build-eval

# Regenerate the environment/algorithm catalog after changing metadata(),
# capabilities(), or an algorithm descriptor (tests reject a stale copy)
make environment-manifest

# Format and lint
cargo fmt --all --manifest-path engine/Cargo.toml
cargo clippy --manifest-path engine/Cargo.toml
cargo fmt --manifest-path actor/Cargo.toml
cargo clippy --manifest-path actor/Cargo.toml
cargo fmt --manifest-path web/Cargo.toml
cargo clippy --manifest-path web/Cargo.toml

# Start web server
cargo run --manifest-path web/Cargo.toml

# Start frontend dev server
npm --prefix web/frontend run dev

# ======= RECOMMENDED: Synchronized AlphaZero Training =======
# Each attempt: fresh exact replay scope -> bounded collection -> seal -> train -> evaluate
# The source checkpoint and configured episode quota are verified before learning
# Evaluation runs after each iteration by default!

# Install trainer package from the repository root (required for local training)
pip install -e "trainer/.[dev]"

# PostgreSQL must be running, and the Python trainer reads the replay-buffer
# connection string ONLY from this env var — config.toml's storage.postgres_url
# is used by the Rust actor/web but NOT by the trainer:
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:cartridge@localhost:5432/cartridge

# Basic synchronized training (TicTacToe) with evaluation
python -m trainer --algorithm alphazero_board_v1 loop --iterations 50 --episodes 200 --steps 500

# Connect4 with more data per iteration
python -m trainer --algorithm alphazero_board_v1 loop --env-id connect4 --iterations 100 --episodes 500 --steps 1000

# With GPU (evaluation runs by default every iteration)
python -m trainer --algorithm alphazero_board_v1 loop --device cuda --iterations 100

# Disable evaluation for faster training
python -m trainer --algorithm alphazero_board_v1 loop --eval-interval 0 --iterations 50

# Resume is automatic: --iterations is the immutable global target, and the
# loop derives the next missing iteration from RunHead.
python -m trainer --algorithm alphazero_board_v1 loop --iterations 100

# ======= Cartridge Commands =======

# Train from an explicitly named, already-populated root selection. Descendant
# data uses --source-checkpoint-id instead of --source-root.
python -m trainer --algorithm alphazero_board_v1 train \
  --steps 1000 \
  --collection-scope-id 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --source-root

# Evaluate model against random play
python -m trainer --algorithm alphazero_board_v1 evaluate --games 100

# Score Connect4 model decisions against a perfect solver (bitbully)
# Metrics: value-optimal-move rate, blunder rate, exact-best rate
# (overall / by ply bucket / by seat); diagnostic/log-only, no artifact output
python -m trainer --algorithm alphazero_board_v1 solver-eval --env-id connect4 --games 100
python -m trainer --algorithm alphazero_board_v1 solver-eval --env-id connect4 --all-checkpoints --games 100

# Register verified manifests, then rate every immutable checkpoint against
# every other; Elo is anchored at random = 0.
# A far better progress signal than win-rate-vs-random, which saturates.
python -m trainer --algorithm alphazero_board_v1 register-players --env-id connect4
python -m trainer --algorithm alphazero_board_v1 tournament --env-id connect4 --games 40

# The loop runs solver eval automatically each evaluation (connect4) and can
# log everything to W&B; opt into solver-based gatekeeping with:
python -m trainer --algorithm alphazero_board_v1 loop --env-id connect4 --wandb-enabled true --promotion-metric solver_optimal
```

## Current Status

- [x] Generic Environment/Timestep ABI, validated adapter, registry, and optional profiles
- [x] EngineContext high-level API
- [x] TicTacToe game implementation
- [x] Connect 4 game implementation
- [x] Othello game implementation
- [x] Removed gRPC/proto dependencies (library-only)
- [x] Actor core (bounded episode runner, profile-fenced PostgreSQL replay)
- [x] MCTS integration in actor with ONNX evaluation
- [x] Web model hot-reload via file watching (model-watcher crate)
- [x] Auto-derived AlphaZero board configuration from optional engine metadata
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
- [x] Engine-generated schema-v4 environment/algorithm manifest
- [x] Engine-owned state projection (`BoardView`); Generals playable in the web UI
- [x] Evaluation moved into the engine (`cartridge-eval`); Python game mirrors deleted
- [x] Player registry + round-robin tournaments with Bradley-Terry Elo

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
| `/model` | GET | Get info about loaded model |

## Using the Engine

```rust
use engine_core::{AgentId, EngineContext};
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
println!(
    "P1 reward: {:?}, episode: {:?}",
    step.timestep.reward_for(AgentId(1)),
    step.timestep.episode,
);
```

## Environment ABI Pattern

General environments implement the typed, algorithm-neutral contract. The
private runtime boundary validates immutable descriptors and timestep
invariants while erasing state, action, and observation types for registry
dispatch. Public consumers use `EngineContext` and cannot bypass validation:

```rust
pub trait Environment {
    type State;
    type Action;
    type Observation;

    fn capabilities(&self) -> Capabilities;
    fn metadata(&self) -> EnvironmentMetadata;
    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8])
        -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError>;
    fn step(&mut self, state: &mut Self::State, action: Self::Action, rng: &mut ChaCha20Rng)
        -> Result<Timestep<Self::Observation>, EnvironmentError>;
    fn encode_state(state: &Self::State, buf: &mut Vec<u8>) -> Result<(), EncodeError>;
    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError>;
    // ... strict action and observation codecs
}
```

The bundled deterministic two-seat games implement the narrower
`engine_core::board_profile::BoardGame` profile. Its private adapter supplies
alternating decisions and per-agent zero-sum outcomes; do not copy those
assumptions into `Environment`.

## Adding a New Environment

1. Create an `engine/envs-{name}/` crate (`games-{name}` is reserved for game domains)
2. Implement `Environment` with typed state/action/observation, an explicit current-agent roster, and per-agent timesteps
3. Declare immutable contract version, codecs, agents/action spaces, semantics, and optional presentation metadata
4. Register with `register_environment::<YourEnvironment>()` (`board_profile::register_board_game::<YourBoardGame>()` only for the narrow board family)
5. Add descriptor validation, transition, and strict codec round-trip tests
6. Run `make environment-manifest` and inspect compatibility reports

Example registration:
```rust
use engine_core::register_environment;
use envs_counter::CounterEnvironment;

pub fn register_counter() {
    register_environment::<CounterEnvironment>()
        .expect("counter must only be registered once");
}
```

## Differences from Cartridge1

| Aspect | Cartridge1 | Cartridge2 |
|--------|------------|------------|
| Architecture | 7 microservices | Monolith + Python |
| Communication | gRPC everywhere | Filesystem + HTTP |
| Replay Buffer | Go service + Redis | PostgreSQL |
| Model Storage | Go service + MinIO | Profile-scoped filesystem or S3 artifacts |
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

// Run search. The cartridge validates the generic timestep and derives its
// dynamic-width LegalMask from the active agent's observation.
let mut rng = ChaCha20Rng::seed_from_u64(42);
let result = run_mcts(
    &mut ctx, &evaluator, config, reset.state, reset.timestep, &mut rng,
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
- **Channel advancement is not yet Elo-driven.** Round-robin ratings are
  analysis output; they do not automatically select the checkpoint published
  to the `current` inference channel.
- **Evaluation defaults to no search.** `[evaluation] simulations = 0` plays
  the policy head directly for a cheap evaluation pass. Raising it measures
  the system as it actually plays — a Connect 4 checkpoint went 15/20
  vs random at 0 sims and 19/20 at 100 — at proportionally more eval wall-time.
  Promotion is still gated on the searchless number until this is raised.

## Reference

- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - Python AlphaZero reference
- [`documentation/ARCHITECTURE.md`](documentation/ARCHITECTURE.md) - full architecture reference
