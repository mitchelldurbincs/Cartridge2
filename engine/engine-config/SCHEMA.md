# Config.toml Schema

This document defines the canonical config.toml schema used by all Cartridge2 components.
The Python trainer (`trainer/src/trainer/central_config.py`) must stay aligned with this schema.

## Configuration Priority

Settings are loaded with the following priority (highest to lowest):

1. **CLI arguments** - Direct command-line flags
2. **Environment variables** - `CARTRIDGE_<SECTION>_<KEY>`
3. **config.toml** - Central configuration file
4. **Built-in defaults** - Hardcoded fallbacks in engine-config

## Sections

### [common]

Shared configuration across all components.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_dir` | string | `"./data"` | Base data directory for models, stats, replay buffer |
| `env_id` | string | `"tictactoe"` | Default game environment ID |
| `log_level` | string | `"info"` | Log level: trace, debug, info, warn, error |

### [training]

Training loop configuration (used by trainer).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iterations` | i32 | `100` | Number of training iterations |
| `start_iteration` | i32 | `1` | Starting iteration (for resume) |
| `episodes_per_iteration` | i32 | `500` | Self-play episodes per iteration |
| `steps_per_iteration` | i32 | `1000` | Training steps per iteration |
| `batch_size` | i32 | `64` | Training batch size |
| `learning_rate` | f64 | `0.001` | Initial learning rate |
| `weight_decay` | f64 | `0.0001` | L2 regularization weight decay |
| `grad_clip_norm` | f64 | `1.0` | Gradient clipping norm |
| `device` | string | `"cpu"` | Device: auto, cpu, cuda, mps |
| `checkpoint_interval` | i32 | `100` | Steps between checkpoints |
| `max_checkpoints` | i32 | `10` | Maximum checkpoints to keep |
| `num_actors` | i32 | `6` | Parallel actor processes for self-play |

### [evaluation]

Model evaluation configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interval` | i32 | `1` | Evaluate every N iterations (0 = disable) |
| `games` | i32 | `50` | Games per evaluation |
| `win_threshold` | f64 | `0.55` | Win rate to become new best model |
| `eval_vs_random` | bool | `true` | Also evaluate against random baseline |

### [actor]

Self-play actor configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `actor_id` | string | `"actor-1"` | Unique identifier for this actor |
| `max_episodes` | i32 | `-1` | Maximum episodes (-1 = unlimited) |
| `episode_timeout_secs` | u64 | `30` | Timeout per episode in seconds |
| `flush_interval_secs` | u64 | `5` | Interval to flush replay buffer |
| `log_interval` | u32 | `50` | Episodes between log messages |

### [web]

Web server configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"0.0.0.0"` | Server bind address |
| `port` | u16 | `8080` | Server port |

### [mcts]

Monte Carlo Tree Search configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `start_sims` | u32 | `50` | Simulations for first iteration (ramping start) |
| `max_sims` | u32 | `250` | Maximum simulations after ramping completes |
| `sim_ramp_rate` | u32 | `10` | Simulations added per iteration |
| `num_simulations` | u32 | `800` | Legacy: MCTS simulations per move (used if ramping not configured) |
| `c_puct` | f64 | `1.4` | Exploration constant |
| `temperature` | f64 | `1.0` | Action selection temperature |
| `temp_threshold` | u32 | `15` | Move number after which to reduce temperature (0 = disabled) |
| `dirichlet_alpha` | f64 | `0.3` | Dirichlet noise alpha |
| `dirichlet_weight` | f64 | `0.25` | Dirichlet noise weight |
| `eval_batch_size` | usize | `32` | Batch size for ONNX evaluation during MCTS |
| `onnx_intra_threads` | usize | `1` | ONNX intra-op parallelism threads |

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

## Environment Variable Overrides

All configuration values can be overridden via environment variables using the pattern:

```
CARTRIDGE_<SECTION>_<KEY>=value
```

### Examples

```bash
# Common
CARTRIDGE_COMMON_ENV_ID=connect4
CARTRIDGE_COMMON_DATA_DIR=/data
CARTRIDGE_COMMON_LOG_LEVEL=debug

# Training
CARTRIDGE_TRAINING_ITERATIONS=50
CARTRIDGE_TRAINING_BATCH_SIZE=128
CARTRIDGE_TRAINING_LEARNING_RATE=0.0005
CARTRIDGE_TRAINING_DEVICE=cuda

# Evaluation
CARTRIDGE_EVALUATION_INTERVAL=5
CARTRIDGE_EVALUATION_GAMES=100

# Actor
CARTRIDGE_ACTOR_ACTOR_ID=actor-2
CARTRIDGE_ACTOR_MAX_EPISODES=1000

# Web
CARTRIDGE_WEB_HOST=127.0.0.1
CARTRIDGE_WEB_PORT=3000

# MCTS
CARTRIDGE_MCTS_NUM_SIMULATIONS=1600
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

If no config file is found, built-in defaults are used.

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
| `Option<String>` | `Optional[str]` | string or absent |
| `Option<u64>` | `Optional[int]` | integer or absent |
