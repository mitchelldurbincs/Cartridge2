<p align="center">
  <img src="./logo.png" alt="Cartridge2 Logo" width="500">
</p>

# Cartridge2

An algorithm-oriented reinforcement-learning platform. Environments describe
game mechanics and capabilities; algorithm cartridges bind collection,
experience, learning, model, and evaluation implementations to compatible
environments.

Two cartridges are installed. `alphazero_board_v1` packages the project's
board-game policy/value system. `dqn_v1` is the first non-board vertical slice:
single-agent discrete collection, transition replay, Q-learning, greedy ONNX
inference, return evaluation, and bounded off-policy orchestration on `counter`.

**Games:** TicTacToe, Connect 4, Othello (complete), and Generals 8x8
(engine/trainer/web complete; training has not yet beaten random at local
compute scale)

**Why the name?**
I love history (Hannibal Barca is my goat) and it also just happens to be close to cartridge which is a goal of this project - being able to easily add new games.
You may have noticed that the C in the logo is the esteemed harbor of Carthage.

**Improvements over v1**
The original Cartridge used 7 microservices with gRPC, Go, and Kubernetes—great for production scale, but overkill for experimentation. Cartridge2 simplifies everything: Rust + Python, shared storage, and a single `docker compose up` to start training. Algorithm identity is now explicit, so future cartridges can provide different collectors and learners without pretending every environment is an AlphaZero board game.

## Algorithm cartridges

Select the algorithm independently from the environment:

```toml
[common]
env_id = "connect4"

[algorithm]
id = "alphazero_board_v1"
```

The same selection can be supplied with `--algorithm alphazero_board_v1` or
`CARTRIDGE_ALGORITHM_ID=alphazero_board_v1`. The actor, trainer, and evaluator
resolve the selected ID through their local algorithm registry and reject an
unknown or incompatible algorithm/environment pair at startup.

The engine-generated `environment_manifest.json` catalog uses schema version 5.
Its top-level `environments` entries contain exactly `metadata`, `capabilities`,
and `algorithm_profiles`. Generic metadata is display-only and its nested
`board` profile may be null. Capabilities carry the immutable contract version,
wire codecs, explicit environment semantics, optional horizon, and per-agent
action spaces. Each installed algorithm descriptor names seven contracts:
collector, learner, orchestration, experience schema, model contract,
evaluation suite, and serving suite. Python strictly consumes this catalog
rather than inferring algorithm support from board dimensions or network
settings.

`alphazero_board_v1` currently requires:

- exactly two fixed players with one active player at a time;
- alternating turns, perfect information, and deterministic planning snapshots;
- finite indexed discrete actions with an observation-embedded legal mask;
- fixed-size spatial `f32` observations with a two-element player indicator; and
- terminal-only, zero-sum rewards.

`dqn_v1` instead requires one fixed agent, discrete actions, fixed `f32`
observations, Markov information, general per-step rewards, a finite horizon,
and no explicit chance decision. It does not require a board, alternating
turns, a legal-mask observation channel, zero-sum rewards, or terminal-only
rewards.

The generic ABI represents fixed or dynamic agents, single-agent, sequential,
and simultaneous decisions, explicit or environment-sampled chance, perfect or
partial observations, deterministic or stochastic transitions, general
per-agent rewards, terminated versus truncated episodes, and discrete,
multi-discrete, or continuous action spaces. That representation does not make
an algorithm compatible: single-agent, chance, simultaneous, multi-agent,
partial-observation, recurrent, and continuous-control workloads need matching
algorithm cartridges and are rejected by `alphazero_board_v1`.

The standard action codecs cover sequential discrete, multi-discrete, and
continuous actions. Simultaneous joint actions and explicit chance outcomes are
currently environment-defined `Custom` codecs, so their algorithm cartridge
must understand that declared codec; a shared decision-action envelope belongs
to the next runtime phase.

### Model artifact identity

Model files are part of the cartridge contract, not interchangeable blobs.
Checkpoint publication is content-addressed: immutable ONNX and learner-state
objects live under `models/blobs/sha256/`, their canonical manifest lives under
`models/manifests/sha256/{checkpoint_id}.json`, and
`models/channels/current.json` is the sole authoritative mutable `RunHeadV2`.
Learner continuation and collection select its latest checkpoint; web serving
selects champion state from its latest RunCommit, falling back to latest before
the first promotion. The checkpoint ID is the SHA-256 of the manifest bytes.

The manifest binds the selected algorithm/model/environment contract, step,
parent checkpoint, learner-config digest, and exact size/SHA-256 of both blobs.
ONNX identity schema version 1 also requires the exact
`cartridge.schema_version`, `cartridge.algorithm_id`,
`cartridge.model_contract`, `cartridge.env_id`, and
`cartridge.env_contract_version` custom metadata. Publication safely validates
the staged learner envelope and materializes only immutable objects. A canonical
`RunCommitV1` then binds that checkpoint, its exact embedded statistics,
evaluation/champion state, and its parent RunCommit. The sole mutable
`RunHeadV2` advances with compare-and-set semantics only after the complete
RunCommit and checkpoint chains validate. Consumers verify those chains,
manifest identity, profile, blob digest/size, learner envelope, and model
interface before loading.

No `current` head means fresh training and random play. A present invalid object
is rejected, and a rejected web hot reload never replaces the last valid
evaluator.
Old mutable checkpoint filenames and ONNX files without schema-v1 identity are
not loaded; there is no legacy inference or implicit identity fallback.

Training statistics schema v3 is algorithm-neutral: current and historical
metrics are finite named maps, and evaluation history stores arbitrary metrics,
episode counts, and mean episode length. Statistics are authoritative only as the exact canonical snapshot
embedded in the selected `RunCommitV1`; `stats_id` hashes those embedded bytes.
The profile-root `stats.json` is an atomically refreshed web projection, not
learner continuity state. Existing malformed or incomplete authority fails
closed.

Promotion evidence is likewise immutable under
`models/evaluations/manifests/sha256/{evaluation_id}.json`. The selected
RunCommit carries the champion checkpoint and supporting evaluation together;
there is no second champion pointer. Evaluation recipes, results, and recursive
champion lineage are verified before selection or promotion.

## Architecture

```
+--------------------------------------------------------------------------+
|                      PostgreSQL + Runtime storage                         |
| PostgreSQL - replay rows fenced by exact profile, scope, and source       |
| ./data/profiles/{algorithm}/{env}/v{contract}/ - models and telemetry     |
+--------------------------------------------------------------------------+
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

PostgreSQL is used for replay while model authority is stored on the filesystem
or S3. Every transition carries its environment contract, algorithm,
experience schema, collection scope, and nullable source checkpoint. A learner
opens an exact `ReplaySelection`; count, distinct-episode seal, sample, clear,
cleanup, and write operations cannot cross another cartridge, attempt, or model
generation.
For cloud deployments, S3/MinIO can be used for model storage.

## Quick Start

### Option 1: Local Development (Recommended)

Local development offers better performance and faster iteration. PostgreSQL is required for the replay buffer.

**Terminal 0** - Start PostgreSQL:
```bash
docker compose up postgres  # Or use a local PostgreSQL installation
```

**Terminal 1** - Start the Rust backend:
```bash
cargo run --manifest-path web/Cargo.toml
# Server starts on http://localhost:8080
```

**Terminal 2** - Start the Svelte frontend:
```bash
npm --prefix web/frontend install
npm --prefix web/frontend run dev
# Dev server starts on http://localhost:5173
```

**Terminal 3** - Train a model:
```bash
pip install -e "trainer/.[dev]"
make build-actor build-eval
# Required: the Python trainer reads the replay-buffer connection string only
# from this env var (config.toml's storage.postgres_url is not used by it)
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:cartridge@localhost:5432/cartridge
python -m trainer --algorithm alphazero_board_v1 loop \
  --iterations 50 --episodes 200 --steps 500
```

Open http://localhost:5173 to play!

### Option 2: Docker

Docker is convenient for quick testing but may have performance overhead.

```bash
# Train a model using synchronized AlphaZero loop
docker compose up alphazero

# Train TicTacToe instead of the Compose default (Connect 4)
CARTRIDGE_COMMON_ENV_ID=tictactoe docker compose up alphazero

# Play against the trained model
docker compose up web frontend
# Open http://localhost in browser
```

## Project Structure

```
cartridge2/
|-- actor/                     # Algorithm-dispatched experience collector
|   |-- src/
|   |   |-- main.rs            # Entry point
|   |   |-- algorithms.rs      # Algorithm -> collector dispatch
|   |   |-- actor.rs           # AlphaZero collector
|   |   |-- config.rs          # CLI configuration
|   |   |-- mcts_policy.rs     # MCTS policy implementation
|   |   |-- resources.rs       # Process resource diagnostics
|   |   |-- stats.rs           # Self-play statistics
|   |   +-- storage/           # Storage backends (PostgreSQL)
|   +-- tests/
|
|-- engine/                    # Rust workspace
|   |-- algorithm-core/        # Algorithm catalog + compatibility reports
|   |-- engine-config/         # Centralized configuration loading
|   |-- engine-core/           # Generic Environment ABI + registry
|   |   +-- src/
|   |       |-- typed.rs       # Typed Environment contract
|   |       |-- contract.rs    # Descriptor/timestep contract validation
|   |       |-- adapter.rs     # Private typed-to-erased adapter
|   |       |-- erased.rs      # Sealed bytes-only runtime boundary
|   |       |-- context.rs     # EngineContext API
|   |       |-- metadata.rs    # Generic + optional board metadata
|   |       |-- board_game.rs  # Narrow AlphaZero board-game adapter
|   |       |-- board_view.rs  # Optional presentation projection
|   |       +-- registry.rs    # Immutable environment registry
|   |-- engine-games/          # Bundled environments + manifest generator
|   |-- envs-counter/          # Direct non-board Environment reference
|   |-- evaluator/             # Algorithm-dispatched evaluation binary
|   |-- games-generals/         # Generals 8x8 implementation
|   |-- metrics-common/         # Prometheus registration/encoding utilities
|   |-- games-tictactoe/       # TicTacToe implementation
|   |-- games-connect4/        # Connect 4 implementation
|   |-- games-othello/         # Othello implementation
|   |-- mcts/                  # Monte Carlo Tree Search
|   +-- model-watcher/         # RunHead validation, web reload, actor one-shot load
|
|-- web/                       # HTTP server + frontend
|   |-- src/
|   |   |-- main.rs            # Axum server setup
|   |   |-- game.rs            # Session management
|   |   |-- metrics.rs         # Prometheus metrics
|   |   |-- startup.rs         # Router, AppState, CORS
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
|       |-- algorithms/        # Installed Python algorithm bindings
|       |-- environment_catalog.py # Strict manifest-v4 contract catalog
|       |-- runtime_profile.py # Canonical artifact namespace
|       |-- trainer.py         # AlphaZero learner implementation
|       |-- network.py         # Neural network (MLP)
|       |-- resnet.py          # ResNet architecture
|       |-- evaluator.py       # Model evaluation
|       |-- solver_eval/       # Perfect-solver move scoring (Connect4)
|       |-- environment_manifest.json # GENERATED by `make environment-manifest`
|       |-- stats.py           # Training statistics
|       |-- config.py          # AlphaZeroLearnerConfig dataclass
|       |-- checkpoint.py      # Checkpoint utilities
|       |-- central_config.py  # Central config.toml loading
|       |-- orchestrator/      # Synchronized AlphaZero orchestrator
|       |-- players.py           # Who occupies a seat in an evaluation game
|       +-- storage/           # Exact replay selection + artifact publication
|
|-- data/                      # Runtime root (gitignored)
|   +-- profiles/{algorithm}/{env}/v{contract}/
|       |-- models/            # PyTorch + ONNX checkpoints
|       +-- *.json             # Profile-scoped telemetry and registries
|
|-- config.toml                # Central configuration
|-- docker-compose.yml         # PostgreSQL + MinIO local stack
|-- k8s/                      # Kustomize deployment manifests
+-- terraform/                # Cloud infrastructure modules
```

## Components

### Engine Core (`engine/engine-core/`)

Pure Rust environment runtime:

- **Environment trait** - Algorithm-neutral typed reset/step contract
- **Timesteps** - Explicit transition rosters, per-agent observations and
  outcomes, decisions/sources, and separate terminated/truncated status
- **Capabilities** - Versioned wire encodings, agent/action spaces, and declared
  turn, information, planning-state, chance, transition, and reward semantics
- **Type erasure** - Sealed, validated bytes-only runtime polymorphism; public
  consumers enter through `EngineContext`
- **Optional profiles** - Board metadata, presentation, and legal masks stay in
  the explicit `engine_core::board_profile` namespace
- **Registry** - Immutable process-local environment registration
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

Algorithm-dispatched experience collector. For `alphazero_board_v1` it:

- Runs game simulations using `EngineContext`
- MCTS with ONNX neural network evaluation
- Resolves `ModelSelection::Latest` from the profile RunHead exactly once
- Stores opaque, exact-selection-bound replay records in PostgreSQL
- MCTS visit distributions saved as policy targets
- Terminal outcomes encoded for every collected position

```bash
# Requires PostgreSQL running (use docker compose up postgres)
cargo run --manifest-path actor/Cargo.toml -- \
  --algorithm alphazero_board_v1 \
  --env-id tictactoe \
  --max-episodes 10000 \
  --collection-scope-id "$(openssl rand -hex 32)"
```

This low-level example is valid only for a root profile with no RunHead. For a
non-root collection, also pass the exact `--source-checkpoint-id`. Normal use is
the synchronized `loop`, which allocates scopes, partitions quotas across
bounded collectors, and verifies the final episode seal automatically.

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

The Python algorithm registry owns learner construction. The
`alphazero_board_v1` binding builds a PyTorch learner that:

- Reads opaque replay records and decodes the cartridge-owned payload
- AlphaZero-style loss (policy cross-entropy + value MSE)
- MCTS visit distributions as soft policy targets
- Game outcomes propagated as value targets
- Publishes content-addressed checkpoint blobs/manifests, then advances channels
- Cosine annealing LR schedule
- Gradient clipping for stability
- Checkpoint management
- Model evaluation against random baseline

#### Synchronized AlphaZero training loop

The `alphazero_board_v1` cartridge includes an orchestrated, synchronous workflow
that coordinates the actor, trainer, and post-iteration evaluation. This
pipeline allocates a fresh replay collection scope for every attempt, pins its
bounded collectors to the exact source checkpoint, seals the configured number
of completed episodes, trains only from that scope, and then evaluates the
resulting candidate. Rows from older or abandoned attempts are retained but
cannot match the learner's exact replay selection.

Run locally (the checked-in `config.toml` targets Connect 4):

```bash
# Using the subcommand interface
trainer --algorithm alphazero_board_v1 loop --iterations 5 --episodes 200 --steps 500
# Or: python -m trainer --algorithm alphazero_board_v1 loop --iterations 5 --episodes 200 --steps 500
```

Configuration can be supplied via flags or `CARTRIDGE_` environment variables.
For example, to train Connect4 with GPU acceleration and disable evaluation for
speed:

```bash
CARTRIDGE_COMMON_ENV_ID=connect4 \
CARTRIDGE_TRAINING_DEVICE=cuda \
CARTRIDGE_EVALUATION_INTERVAL=0 \
    trainer --algorithm alphazero_board_v1 loop \
      --iterations 20 --episodes 300 --steps 1000
```

Docker usage mirrors the same interface:

```bash
docker compose up alphazero
# Override parameters as needed
CARTRIDGE_COMMON_ENV_ID=tictactoe docker compose up alphazero
```

See [Deployment Modes](#deployment-modes) for local Compose and Kubernetes.

## Deployment Modes

### Local Mode (PostgreSQL + Filesystem)

Direct local processes use PostgreSQL for replay and the filesystem for model
storage. Docker Compose instead configures MinIO so every container shares the
same profile-scoped artifacts.

```bash
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:cartridge@localhost:5432/cartridge
python -m trainer --algorithm alphazero_board_v1 loop
```

### Cloud Mode (PostgreSQL + S3)

Uses PostgreSQL for replay buffer and S3/MinIO for model storage. Enables distributed deployments.

```bash
# Compose configures MinIO for model storage
docker compose up

# Parallel self-play: the alphazero service runs multiple actor processes
# internally — tune [training].num_actors in config.toml (or
# CARTRIDGE_TRAINING_NUM_ACTORS) instead of scaling containers

# Access MinIO web console at http://localhost:9001
```

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `CARTRIDGE_STORAGE_MODEL_BACKEND` | `filesystem` or `s3` | `filesystem` |
| `CARTRIDGE_STORAGE_POSTGRES_URL` | PostgreSQL connection string | `postgresql://cartridge:cartridge@localhost:5432/cartridge` |
| `CARTRIDGE_STORAGE_S3_BUCKET` | S3 bucket for models | - |
| `CARTRIDGE_STORAGE_S3_ENDPOINT` | S3-compatible endpoint (MinIO) | - |

Note: the default `postgres_url` applies to the Rust actor and web server (via
config.toml). The Python trainer has no fallback — it requires
`CARTRIDGE_STORAGE_POSTGRES_URL` to be set in its environment.

## Security Configuration

### CORS (Cross-Origin Resource Sharing)

The web server implements **deny-by-default** CORS behavior for security:

- **Development mode**: When `web.allowed_origins` is empty, only localhost origins are allowed
- **Production mode**: Set explicit allowed origins in `config.toml`:
  ```toml
  [web]
  allowed_origins = ["https://yourdomain.com", "https://www.yourdomain.com"]
  ```

### Credential Management

**MinIO/S3 Credentials:**
- Never commit credentials to version control
- Use `.env` file for local development (copy from `.env.example`)
- In production, use proper secret management (Kubernetes secrets, AWS Secrets Manager, etc.)
- Default MinIO credentials for local development: `minioadmin` / `changeme` (change for production!)

**PostgreSQL Credentials:**
- Default local development password is `cartridge` (configured in `docker-compose.yml`)
- Use strong passwords in production
- Configure via `CARTRIDGE_STORAGE_POSTGRES_URL` environment variable

### Container Security

The Cartridge2 application images run as a non-root `cartridge` user. The web
and frontend images define health checks; Compose and Kubernetes add
service-level checks for the remaining long-running components. Infrastructure
images such as PostgreSQL and MinIO retain their upstream users and hardening
requirements.

### Secrets Scanning

The CI pipeline includes automated secrets scanning with GitLeaks to prevent accidental credential commits.

## Adding a New Environment

1. Create a new crate in `engine/envs-{name}/` (reserve `games-{name}` for
   environments that are specifically games).
2. Implement the generic `Environment` trait. Publish an immutable
   `contract_version`, exact state/action/observation encodings, agent action
   spaces, environment semantics, and per-agent `Timestep` values. Its `env_id`
   is a runtime namespace segment containing only lowercase ASCII letters,
   digits, `_`, or `-`.
3. Add optional `EnvironmentMetadata` and `Presentation` only for human-facing
   consumers. A non-board environment does not need board dimensions, a legal
   mask, or player-seat metadata.
4. Register the environment:

```rust
use engine_core::register_environment;
use envs_counter::CounterEnvironment;

pub fn register_counter() {
    register_environment::<CounterEnvironment>()
        .expect("counter must only be registered once");
}
```

For the existing deterministic two-seat board family, implement the narrower
`engine_core::board_profile::BoardGame` trait and use
`engine_core::board_profile::register_board_game::<YourBoardGame>()`; its private adapter is the only place
that maps scalar previous-actor rewards and alternating turns into the generic
per-agent ABI. `engine/envs-counter` is the reference implementation for a
direct, non-board `Environment`.

5. Add reset/step, descriptor-validation, and strict codec round-trip tests.
6. Regenerate the manifest with `make environment-manifest`.
7. Inspect the generated algorithm compatibility reports. Registering an
   environment does not make it trainable by every algorithm; if no installed
   cartridge is compatible, add or extend an algorithm binding deliberately.

## Development

### Build

```bash
# Build all Rust components
cargo build --release --manifest-path engine/Cargo.toml
cargo build --release --manifest-path actor/Cargo.toml
cargo build --release --manifest-path web/Cargo.toml

# Install Python dependencies
pip install -e "trainer/.[dev]"
```

### Test

```bash
make test          # engine + actor + web + trainer

# Or individually
cargo test --manifest-path engine/Cargo.toml
cargo test --manifest-path actor/Cargo.toml
cargo test --manifest-path web/Cargo.toml
python -m pytest trainer/tests/
```

### Format & Lint

```bash
cargo fmt --all --manifest-path engine/Cargo.toml
cargo clippy --manifest-path engine/Cargo.toml
cargo fmt --manifest-path actor/Cargo.toml
cargo clippy --manifest-path actor/Cargo.toml
cargo fmt --manifest-path web/Cargo.toml
cargo clippy --manifest-path web/Cargo.toml
```

## Current Status

**Core:**
- [x] Generic Environment ABI, validator/adapter, registry, and optional profiles
- [x] Explicit algorithm catalog and startup dispatch
- [x] Manifest v4 generic environment contract and compatibility profiles
- [x] EngineContext high-level API
- [x] TicTacToe game implementation
- [x] Connect 4 game implementation
- [x] Othello game implementation
- [x] MCTS implementation with ONNX evaluation

**Training:**
- [x] Bounded actor (one-shot episode runner, MCTS + pinned ONNX generation)
- [x] Python trainer (PyTorch, ONNX export, cosine LR)
- [x] Synchronized AlphaZero training loop (orchestrator)
- [x] MCTS policy targets + game outcome propagation
- [x] Model evaluation against random baseline

**Storage Backends:**
- [x] PostgreSQL replay buffer (default)
- [x] Filesystem model storage (default)
- [x] S3/MinIO model storage (Compose and distributed deployments)

**Web:**
- [x] Web server (Axum, game API)
- [x] Web frontend (Svelte, play UI, stats)
- [x] Loss visualization chart

**Deployment:**
- [x] Docker Compose (PostgreSQL, optional MinIO for S3)
- [x] Parallel self-play actors (`[training].num_actors`)
- [x] Kubernetes manifests (Kustomize, `k8s/`) + Terraform modules

## Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Architecture | Monolith + Python | Simplicity for MVP, easy local development |
| Deployment | Docker Compose | Simple orchestration, PostgreSQL included |
| Language | Rust + Python | Type safety + ML ecosystem |
| Environment Interface | Typed ABI + validated erasure | Compile-time safety + runtime flexibility without board assumptions |
| Replay Storage | PostgreSQL | Concurrent access, scales with multiple actors |
| Model Storage | Filesystem / S3 | Filesystem for local, S3/MinIO for distributed |
| Model Format | ONNX | Framework-agnostic, production-ready |
| RNG | ChaCha20 | Deterministic when seeded; synchronized collection intentionally uses system entropy while evaluation records a fixed seed |

## License

MIT
