# Config.toml Schema

This document defines the canonical config.toml schema used by all Cartridge2 components.
The Python trainer (`trainer/src/trainer/central_config.py`) must stay aligned with this schema.

## Configuration Priority

Settings are loaded with the following priority (highest to lowest):

1. **CLI arguments** - Direct command-line flags
2. **Environment variables** - `CARTRIDGE_<SECTION>_<KEY>`
3. **config.toml** - Central configuration file
4. **`config.defaults.toml`** - Embedded by Rust and loaded by Python

## Sections

### [common]

Shared configuration across all components.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_dir` | string | `"./data"` | Runtime root; components append `profiles/{algorithm}/{env}/v{contract}` |
| `env_id` | string | `"tictactoe"` | Default game environment ID |
| `log_level` | string | `"info"` | Log level: trace, debug, info, warn, error |

### [algorithm]

Selects the algorithm cartridge used by collectors, learners, model adapters,
and evaluation. Registration and compatibility are separate: an installed
algorithm still rejects environments that do not satisfy its requirements.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | `"alphazero_board_v1"` | Canonical algorithm cartridge ID |

### [training]

Training loop configuration (used by trainer).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iterations` | u64 | `100` | Number of training iterations |
| `episodes_per_iteration` | u32 | `500` | Self-play episodes per iteration |
| `steps_per_iteration` | u64 | `1000` | Training steps per iteration |
| `batch_size` | u64 | `64` | Training batch size |
| `learning_rate` | f64 | `0.001` | Initial learning rate |
| `weight_decay` | f64 | `0.0001` | L2 regularization weight decay |
| `grad_clip_norm` | f64 | `1.0` | Gradient clipping norm |
| `device` | string | `"cpu"` | Device: auto, cpu, cuda, mps |
| `checkpoint_interval` | u64 | `100` | Steps between checkpoints |
| `num_actors` | u32 | `1` | Parallel actor processes for self-play |

### [evaluation]

Model evaluation configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interval` | u64 | `1` | Evaluate every N iterations (0 = disable) |
| `games` | u32 | `50` | Games per evaluation |
| `win_threshold` | f64 | `0.55` | Win rate required for champion promotion |
| `eval_vs_random` | bool | `true` | Also evaluate against random baseline |
| `simulations` | u32 | `0` | MCTS simulations per evaluation move; 0 uses policy only |
| `temperature` | f32 | `0.2` | Action-selection temperature for evaluation games |
| `solver_games` | u32 | `0` | Connect4 perfect-solver games; nonzero values are rejected for other environments |
| `evaluation_seed` | u64 | `42` | Stable seed for every evaluation game family |
| `promotion_metric` | string | `"win_rate"` | `win_rate` or `solver_optimal` |
| `promotion_margin` | f64 | `0.0` | Required solver-optimal improvement; canonical zero while `promotion_metric = "win_rate"` |

### [actor]

Self-play actor configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `actor_id` | string | `"actor-1"` | Unique identifier for this actor |
| `episode_timeout_secs` | u64 | `30` | Timeout per episode in seconds |
| `log_interval` | u32 | `50` | Episodes between log messages |

### [web]

Web server configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"0.0.0.0"` | Server bind address |
| `port` | u16 | `8080` | Server port |
| `allowed_origins` | string[] | `[]` | Explicit CORS origins; empty selects the localhost-only allowlist |

### [mcts]

Monte Carlo Tree Search configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `start_sims` | u32 | `50` | Simulations for first iteration (ramping start) |
| `max_sims` | u32 | `400` | Maximum simulations after ramping completes |
| `sim_ramp_rate` | u32 | `20` | Simulations added per iteration |
| `c_puct` | f32 | `1.4` | Exploration constant |
| `temperature` | f32 | `1.0` | Action selection temperature |
| `late_temperature` | f32 | `1.0` | Action selection temperature at/after `temp_threshold`; must equal base temperature when threshold is 0 |
| `temp_threshold` | u32 | `0` | Move number after which to reduce temperature (0 = disabled) |
| `dirichlet_alpha` | f32 | `0.3` | Dirichlet noise alpha; alpha and weight must both be 0 to disable |
| `dirichlet_weight` | f32 | `0.25` | Dirichlet noise weight; alpha and weight must both be 0 to disable |
| `eval_batch_size` | u32 | `32` | Batch size for ONNX evaluation during MCTS |
| `onnx_intra_threads` | u32 | `1` | ONNX intra-op parallelism threads |

### [logging]

Structured logging configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | string | `"text"` | Log format: "text" (human-readable) or "json" (structured for cloud) |
| `include_timestamps` | bool | `true` | Include timestamps in log output |
| `include_target` | bool | `true` | Include module target in log output |

### [storage]

Storage backend configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_backend` | string | `"filesystem"` | Backend: filesystem, s3 |
| `postgres_url` | string? | `"postgresql://cartridge:cartridge@localhost:5432/cartridge"` | PostgreSQL connection URL |
| `s3_bucket` | string? | `None` | S3 bucket name (for s3 backend) |
| `s3_endpoint` | string? | `None` | S3 endpoint URL (for MinIO) |
| `pool_max_size` | usize | `16` | Max PostgreSQL pool connections |
| `pool_connect_timeout` | u64 | `30` | Pool connection timeout (seconds) |
| `pool_idle_timeout` | u64? | `300` | Pool idle timeout (seconds) |
| `replay_retained_scopes` | u32 | `2` | Newest collection scopes kept per profile; older scopes are reaped after each commit |
| `learner_state_retained_checkpoints` | u32 | `3` | Newest checkpoints keeping their learner-state (.pt) blob; every ONNX is kept forever |

### [wandb]

Trainer-only Weights & Biases integration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable W&B logging |
| `required` | bool | `false` | Fail instead of using a no-op logger when W&B is unavailable |
| `project` | string | `"cartridge2"` | W&B project |
| `entity` | string | `""` | W&B entity; empty uses the logged-in default |
| `group` | string | `""` | Optional run group |
| `tags` | string[] | `[]` | Run tags |
| `init_timeout_seconds` | f64 | `30.0` | W&B initialization timeout |

## Environment Variable Overrides

Both implementations accept the environment-variable pattern:

```
CARTRIDGE_<SECTION>_<KEY>=value
```

Rust maps every declared field explicitly in
`engine/engine-config/src/loader.rs`; Python derives the mapping from its typed
schema. Unknown keys are rejected rather than treated as compatibility aliases.
List values such as `allowed_origins` and W&B tags are comma-separated.

### Examples

```bash
# Common
CARTRIDGE_COMMON_ENV_ID=connect4
CARTRIDGE_COMMON_DATA_DIR=/data
CARTRIDGE_COMMON_LOG_LEVEL=debug

# Algorithm
CARTRIDGE_ALGORITHM_ID=alphazero_board_v1

# Training
CARTRIDGE_TRAINING_ITERATIONS=50
CARTRIDGE_TRAINING_BATCH_SIZE=128
CARTRIDGE_TRAINING_LEARNING_RATE=0.0005
CARTRIDGE_TRAINING_DEVICE=cuda

# Evaluation
CARTRIDGE_EVALUATION_INTERVAL=5
CARTRIDGE_EVALUATION_GAMES=100
CARTRIDGE_EVALUATION_EVALUATION_SEED=42

# Actor
CARTRIDGE_ACTOR_ACTOR_ID=actor-2
CARTRIDGE_ACTOR_EPISODE_TIMEOUT_SECS=60

# Web
CARTRIDGE_WEB_HOST=127.0.0.1
CARTRIDGE_WEB_PORT=3000

# MCTS
CARTRIDGE_MCTS_START_SIMS=100
CARTRIDGE_MCTS_MAX_SIMS=1600
CARTRIDGE_MCTS_SIM_RAMP_RATE=20
CARTRIDGE_MCTS_C_PUCT=2.0

# Storage
CARTRIDGE_STORAGE_MODEL_BACKEND=s3
CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://user:pass@host/db
CARTRIDGE_STORAGE_S3_BUCKET=my-bucket
```

## Config File Search Paths

Configuration is searched in the following order:

1. Path specified by `CARTRIDGE_CONFIG` environment variable
2. `config.toml` (current directory)
3. `../config.toml` (parent directory)
4. `/app/config.toml` (Docker container)

If no config file is found, Rust uses its embedded `config.defaults.toml`.
Python uses the repository copy in a source checkout and a byte-identical
package resource from an installed wheel. A checkout with divergent copies, or
an installation without complete canonical defaults, fails closed.

## Python Trainer Alignment

The Python trainer must maintain equivalent dataclasses in `trainer/src/trainer/central_config.py`.
When adding new fields to the Rust schema:

1. Add the field to the appropriate struct in `engine/engine-config/src/structs.rs`
2. Add the default value in `engine/engine-config/src/defaults.rs`
3. Add the env override in `engine/engine-config/src/loader.rs`
4. Update the corresponding Python dataclass in `trainer/src/trainer/central_config.py`
5. Update this schema document

## Type Mappings

| Rust Type | Python Type | TOML Type |
|-----------|-------------|-----------|
| `String` | `str` | string |
| `i32` | `int` | integer |
| `u32` | `int` | integer |
| `u64` | `int` | integer |
| `usize` | `int` | integer |
| `u16` | `int` | integer |
| `f64` | `float` | float |
| `Vec<String>` | `list[str]` | string array |
| `Option<String>` | `Optional[str]` | string or absent |
| `Option<u64>` | `Optional[int]` | integer or absent |
