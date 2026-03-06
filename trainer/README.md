# Trainer

Python training loop for Cartridge2. Implements AlphaZero-style learning from self-play data.

## Overview

The trainer:
1. Reads transitions from PostgreSQL replay buffer
2. Trains a PyTorch neural network (MLP or ResNet based on game)
3. Exports ONNX models for the Rust actor
4. Writes stats.json for the web frontend

## Quick Start

```bash
# Install
pip install -e .

# Start PostgreSQL (required)
docker compose up postgres -d

# Set connection string
export CARTRIDGE_STORAGE_POSTGRES_URL="postgresql://cartridge:cartridge@localhost:5432/cartridge"

# Run training (defaults assume running from trainer/ directory)
trainer train --steps 1000
# Or: python -m trainer train --steps 1000

# With custom settings
trainer train \
    --steps 5000 \
    --batch-size 128 \
    --lr 0.001
```

## Available Subcommands

| Command | Description |
|---------|-------------|
| `trainer train` | Train on replay buffer data |
| `trainer evaluate` | Evaluate model against random baseline |
| `trainer loop` | Run synchronized AlphaZero training (actor + trainer + eval) |

All commands support `--help` for detailed argument information.

## Neural Network Architecture

The trainer automatically selects the appropriate network architecture based on the game's configuration. Two architectures are available:

### MLP (Multi-Layer Perceptron)

Used for simpler games like TicTacToe. A fully-connected feedforward network.

```
Input (obs_size)
    -> FC(hidden) -> ReLU
    -> FC(hidden) -> ReLU
    -> FC(hidden/2) -> ReLU
    -> Policy head: FC(num_actions)
    -> Value head: FC(hidden/4) -> ReLU -> FC(1) -> Tanh
```

**When to use:** Small board games where spatial relationships are less critical.

### ResNet (Convolutional Residual Network)

Used for spatially-structured games like Connect4 and Othello. Follows the AlphaZero paper architecture with residual blocks and batch normalization.

```
Input: (batch, obs_size) [flat tensor]
    -> Reshape to (batch, channels, height, width)
    -> Initial Conv Block: Conv2d(3x3) -> BatchNorm -> ReLU
    -> Residual Tower: N residual blocks
    -> Policy Head: Conv2d(1x1) -> BN -> ReLU -> Flatten -> FC(num_actions)
    -> Value Head: Conv2d(1x1) -> BN -> ReLU -> Flatten -> FC(256) -> ReLU -> FC(1) -> Tanh
```

Each **Residual Block** contains:
```
Input
    -> Conv2d(3x3, padding=1) -> BatchNorm -> ReLU
    -> Conv2d(3x3, padding=1) -> BatchNorm
    -> + skip connection (identity)
    -> ReLU
```

**When to use:** Board games with spatial structure where convolutions can learn local patterns (adjacent pieces, lines, etc.).

### Automatic Network Selection

The network type is determined by the game's configuration in `game_config.py`:

| Game | Network | Res Blocks | Filters | Input Channels | Hidden Size |
|------|---------|------------|---------|----------------|-------------|
| TicTacToe | MLP | - | - | - | 128 |
| Connect4 | ResNet | 4 | 128 | 2 | 512 |
| Othello | ResNet | 6 | 256 | 2 | 512 |

The `create_network(env_id)` factory function handles this automatically:

```python
from trainer.network import create_network

# Automatically creates appropriate network based on game config
network = create_network("tictactoe")  # Returns PolicyValueNetwork (MLP)
network = create_network("connect4")   # Returns ConvPolicyValueNetwork (ResNet)
```

### ResNet Configuration Details

For games using the ResNet architecture, the following parameters are configurable in `game_config.py`:

| Parameter | Description | Connect4 | Othello |
|-----------|-------------|----------|---------|
| `network_type` | Architecture to use | `"resnet"` | `"resnet"` |
| `num_res_blocks` | Number of residual blocks in tower | 4 | 6 |
| `num_filters` | Filters per convolutional layer | 128 | 256 |
| `input_channels` | Board encoding channels | 2 | 2 |
| `board_height` | Board height for reshaping | 6 | 8 |
| `board_width` | Board width for reshaping | 7 | 8 |

**Input Channel Encoding:** For two-player games, the observation is encoded as 2 channels:
- Channel 0: Current player's pieces (1 where piece present, 0 elsewhere)
- Channel 1: Opponent's pieces (1 where piece present, 0 elsewhere)

This spatial encoding allows the convolutions to learn patterns like connected pieces, blocking moves, and positional strategy.

### Weight Initialization

The ResNet uses He (Kaiming) initialization for all layers:
- Conv2d: `kaiming_normal_` with `fan_out` mode
- BatchNorm2d: weight=1, bias=0
- Linear: `kaiming_normal_` with `fan_out` mode

## CLI Arguments (`trainer train`)

Note: The replay buffer connection is configured via `CARTRIDGE_STORAGE_POSTGRES_URL` environment variable.

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | `../data/models` | Directory for ONNX checkpoints |
| `--stats` | `../data/stats.json` | Stats file for web polling |
| `--steps` | 1000 | Total training steps |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--weight-decay` | 0.0001 | L2 regularization |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--checkpoint-interval` | 100 | Steps between saves |
| `--device` | cpu | Training device (cpu/cuda/mps) |
| `--env-id` | tictactoe | Game environment |

### LR Schedule

The trainer uses a two-phase learning rate schedule (implemented in `lr_scheduler.py`):

1. **Warmup Phase**: LR increases linearly from `warmup_start_lr` to `target_lr`
2. **Cosine Annealing Phase**: LR decreases following a cosine curve to `min_lr`

```
LR
^
|        target_lr
|       /----------.
|      /            '.
|     /               '..
|    /                   ''..._____ min_lr
|   /
+--+----------------------------->  step
   0   warmup    total_steps
       steps
```

```bash
# Disable cosine annealing (constant LR)
trainer train --no-lr-schedule

# Custom min LR ratio (final LR = initial_lr * ratio)
trainer train --lr-min-ratio 0.01

# LR warmup (gradually increase LR at start of training)
trainer train --lr-warmup-steps 100 --lr-warmup-start-ratio 0.1
```

For multi-iteration training (`trainer loop`), the LR schedule can span the entire training run:

```bash
# Continuous decay across 50 iterations of 500 steps each
trainer loop --iterations 50 --steps 500 --lr-total-steps 25000
```

**Checkpoint Behavior**: When resuming from a checkpoint, warmup is automatically disabled to prevent loss spikes. The scheduler state is restored to continue the cosine annealing from where it left off.

### Wait Settings

The trainer waits for the replay database to exist and contain data:

```bash
# Custom wait interval and timeout
trainer train --wait-interval 5.0 --max-wait 600
```

## Architecture

PostgreSQL is the only replay buffer backend (consistent for local and cloud).

```
+-------------------+     +------------------+     +------------------+
|  PostgreSQL       |---->|  PyTorch Model   |---->|  ONNX Export     |
|  (replay buffer)  |     |  (policy+value)  |     |  (model.onnx)    |
+-------------------+     +------------------+     +------------------+
        ^                        |
        |                        v
  Rust Actor(s)          +------------------+
  (self-play)            |   stats.json     |
                         |   (telemetry)    |
                         +------------------+
```

### Storage Configuration

| Backend | Use Case | Environment Variable |
|---------|----------|---------------------|
| PostgreSQL | Replay buffer (required) | `CARTRIDGE_STORAGE_POSTGRES_URL` |
| S3/MinIO | Cloud model storage | `CARTRIDGE_STORAGE_S3_BUCKET` |
| Filesystem | Local model storage (default) | `CARTRIDGE_MODEL_DIR` |

## Loss Function

AlphaZero-style combined loss:

```
L = L_policy + L_value

L_policy = -sum(pi * log(p))    # Cross-entropy with MCTS policy (soft targets)
L_value  = (z - v)^2            # MSE with game outcome
```

Where:
- `pi`: Target policy from MCTS visit count distribution
- `p`: Predicted policy (softmax of logits)
- `z`: Target value (game outcome: +1 win, -1 loss, 0 draw)
- `v`: Predicted value

## Model Export

Models are exported with atomic write-then-rename:

1. Train PyTorch model
2. Export to `temp_model.onnx`
3. Rename to `model.onnx`

This prevents the Rust actor from loading a partially-written file.

Both ONNX (for actor inference) and PyTorch checkpoints (for training continuity) are saved:
- `model_step_NNNNNN.onnx` - ONNX checkpoint for the actor
- `checkpoint.pt` - PyTorch state for resuming training
- `latest.onnx` - Symlink/copy to most recent ONNX model

## Stats Output

The trainer writes `stats.json` for the web frontend:

```json
{
  "step": 1000,
  "total_steps": 5000,
  "total_loss": 0.523,
  "policy_loss": 0.412,
  "value_loss": 0.111,
  "learning_rate": 0.0001,
  "samples_seen": 64000,
  "replay_buffer_size": 10000,
  "last_checkpoint": "./data/models/model_step_001000.onnx",
  "timestamp": 1699999999,
  "env_id": "connect4",
  "history": [
    {"step": 100, "total_loss": 1.2, "value_loss": 0.8, "policy_loss": 0.4, "learning_rate": 0.001},
    {"step": 200, "total_loss": 0.9, "value_loss": 0.5, "policy_loss": 0.4, "learning_rate": 0.001}
  ],
  "last_eval": {
    "step": 1000,
    "win_rate": 0.72,
    "draw_rate": 0.16,
    "loss_rate": 0.12,
    "games_played": 100,
    "avg_game_length": 6.8,
    "timestamp": 1699999999
  },
  "eval_history": [...]
}
```

Stats history is bounded (default 2000 training entries, 50 evaluation entries) to prevent unbounded growth.

## Module Structure

```
src/trainer/
├── __init__.py       # Package exports
├── __main__.py       # CLI entrypoint (train, evaluate, loop)
├── trainer.py        # Training loop, Trainer class
├── lr_scheduler.py   # LR scheduling (warmup + cosine annealing)
├── network.py        # MLP network + AlphaZeroLoss + create_network() factory
├── resnet.py         # ResNet architecture (ConvPolicyValueNetwork, ResidualBlock)
├── evaluator.py      # Model evaluation against baselines
├── game_config.py    # Game-specific configurations (dimensions, network type)
├── stats.py          # TrainerStats, EvalStats, load/write functions
├── config.py         # TrainerConfig dataclass
├── checkpoint.py     # ONNX/PyTorch checkpoint save/load utilities
├── backoff.py        # Wait-with-backoff utilities for data availability
├── orchestrator.py   # Synchronized AlphaZero training orchestrator
├── central_config.py # Central config.toml loading
└── storage/          # Storage backend implementations
    ├── __init__.py   # Package exports
    ├── base.py       # Abstract interfaces (ReplayBufferBase, ModelStore)
    ├── factory.py    # Storage factory for backend selection
    ├── postgres.py   # PostgreSQL replay backend
    ├── s3.py         # S3/MinIO model storage backend (cloud)
    └── filesystem.py # Local filesystem model storage
```

## Game Configuration

Games are configured in `game_config.py` with the following properties:

```python
@dataclass
class GameConfig:
    # Game identity
    env_id: str              # e.g., "connect4"
    display_name: str        # e.g., "Connect 4"

    # Board dimensions
    board_width: int         # e.g., 7
    board_height: int        # e.g., 6

    # Neural network dimensions
    num_actions: int         # e.g., 7 (columns in Connect4)
    obs_size: int            # Total observation vector size
    legal_mask_offset: int   # Where legal moves start in obs

    # Network architecture
    hidden_size: int = 128   # MLP hidden layer size
    network_type: str = "mlp"  # "mlp" or "resnet"

    # CNN-specific (when network_type="resnet")
    num_res_blocks: int = 4    # Residual blocks in tower
    num_filters: int = 128     # Filters per conv layer
    input_channels: int = 2    # Board encoding channels
```

### Adding a New Game

1. Add a `GameConfig` entry in `game_config.py`
2. Choose `network_type="mlp"` for simple games or `network_type="resnet"` for spatial games
3. For ResNet, set appropriate `num_res_blocks`, `num_filters`, and `input_channels`
4. Ensure `obs_size` matches the Rust engine's observation encoding

Example for a hypothetical 4x4 game:
```python
"my_game": GameConfig(
    env_id="my_game",
    display_name="My Game",
    board_width=4,
    board_height=4,
    num_actions=16,
    obs_size=50,  # 16*2 (board) + 16 (legal) + 2 (player)
    legal_mask_offset=32,
    network_type="resnet",
    num_res_blocks=3,
    num_filters=64,
    input_channels=2,
)
```

## Storage Backends

The trainer uses pluggable storage backends for model storage, enabling deployment from local development to distributed Kubernetes clusters.

### Replay Buffer Backend

PostgreSQL is the only replay buffer backend, providing consistent behavior between local development (via Docker) and cloud deployment.

```bash
# Connection via URL (required)
export CARTRIDGE_STORAGE_POSTGRES_URL="postgresql://cartridge:cartridge@localhost:5432/cartridge"

# Start PostgreSQL locally with Docker
docker compose up postgres -d
```

### Model Storage Backends

| Backend | Description | Configuration |
|---------|-------------|---------------|
| **Filesystem** (default) | Local directory storage | `--model-dir ./data/models` |
| **S3/MinIO** | Cloud object storage for distributed training | `CARTRIDGE_STORAGE_MODEL_BACKEND=s3` |

S3 connection is configured via environment variables:
```bash
CARTRIDGE_STORAGE_S3_ENDPOINT=http://minio:9000
CARTRIDGE_STORAGE_S3_BUCKET=cartridge-models
CARTRIDGE_STORAGE_MODEL_BACKEND=s3
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

### Backend Selection

Backends are auto-selected based on configuration, or can be explicitly set:

```python
from trainer.storage import create_replay_buffer, create_model_store

# Auto-detect from environment/config
replay = create_replay_buffer()
models = create_model_store()

# Explicit backend selection
replay = create_replay_buffer(backend="postgres", connection_string="...")
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Format
black .
```

## Dependencies

**Core:**
- `torch>=2.0.0` - Neural network training
- `numpy>=1.24.0` - Numerical operations
- `onnx>=1.14.0` - Model export
- `onnxruntime>=1.15.0` - Model inference for evaluation
- `psycopg2-binary>=2.9.0` - PostgreSQL replay buffer backend
- `onnxscript>=0.1.0` - ONNX scripting
- `prometheus-client>=0.20.0` - Metrics collection
- `python-json-logger>=2.0.0` - Structured logging
- `tomli>=2.0.0` - TOML config parsing (Python <3.11)

**Optional (S3/MinIO model storage):**
- `boto3` - S3/MinIO model storage (not in core dependencies)

## Evaluator

The evaluator measures how well a trained model plays against random opponents.

### Quick Start

```bash
# Basic evaluation (100 games)
trainer evaluate --model ../data/models/latest.onnx --games 100
# Or: python -m trainer evaluate --model ../data/models/latest.onnx --games 100

# More games for statistical confidence
trainer evaluate --model ../data/models/latest.onnx --games 500

# Verbose mode to see individual game moves
trainer evaluate --model ../data/models/latest.onnx --games 10 --verbose

# Compare different checkpoints
trainer evaluate --model ../data/models/model_step_000100.onnx --games 100
```

### CLI Arguments

Note: Game metadata is loaded from PostgreSQL (via `CARTRIDGE_STORAGE_POSTGRES_URL`) or falls back to hardcoded configs.

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `../data/models/latest.onnx` | Path to ONNX model file |
| `--env-id` | `tictactoe` | Game environment to evaluate |
| `--games` | 100 | Number of games to play |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy/argmax) |
| `--verbose` | false | Print individual game moves |
| `--log-level` | INFO | Logging level |

### Output Example

```
==================================================
Evaluation Results: ONNX(latest.onnx) vs Random
==================================================
Games played: 100

ONNX(latest.onnx):
  Wins: 72 (72.0%)
    As X (first): 45
    As O (second): 27

Random:
  Wins: 12 (12.0%)
    As X (first): 8
    As O (second): 4

Draws: 16 (16.0%)
Average game length: 6.8 moves
==================================================

✓ Model is significantly better than random play!
```

### Interpreting Results

| Win Rate | Interpretation |
|----------|----------------|
| >70% | Model is significantly better than random |
| 50-70% | Model is slightly better than random |
| 30-50% | Model is roughly equivalent to random |
| <30% | Model is worse than random |

For TicTacToe specifically:
- A well-trained model should achieve **70%+ win rate** vs random
- High draw rates (>50%) indicate strong defensive play
- First-player (X) advantage is expected - watch for parity between X and O performance

For Connect4:
- Stronger spatial patterns mean higher potential win rates
- ResNet architecture should capture line-building strategies effectively
- Expect **80%+ win rate** with sufficient training

### Evaluating Training Progress

Compare checkpoints to see learning progress:

```bash
# Early checkpoint
trainer evaluate --model ./data/models/model_step_000100.onnx --games 100

# Mid checkpoint
trainer evaluate --model ./data/models/model_step_000500.onnx --games 100

# Final model
trainer evaluate --model ./data/models/latest.onnx --games 100
```

## Integration with Actor

```bash
# Start PostgreSQL first
docker compose up postgres -d
export CARTRIDGE_STORAGE_POSTGRES_URL="postgresql://cartridge:cartridge@localhost:5432/cartridge"

# Terminal 1: Actor generates data
cd actor
cargo run -- --env-id tictactoe

# Terminal 2: Trainer consumes data
cd trainer
trainer train
```

The actor will hot-reload `model.onnx` when it changes. Both actor and trainer connect to the same PostgreSQL database.

## Synchronized AlphaZero Loop

For the recommended synchronized training workflow (where each iteration clears
the buffer, generates fresh episodes, trains, and evaluates):

```bash
# Basic loop (5 iterations)
trainer loop --iterations 5 --episodes 200 --steps 500

# Connect4 with GPU (uses ResNet automatically)
trainer loop --env-id connect4 --device cuda --iterations 20

# Disable evaluation for faster training
trainer loop --eval-interval 0 --iterations 50

# Resume from a specific iteration
trainer loop --iterations 100 --start-iteration 25

# Configure LR decay across all iterations
trainer loop --iterations 50 --steps 500 --lr-total-steps 25000
```

For the full loop argument reference, run `python -m trainer.orchestrator --help`.

## AlphaZero Training Tips

### For Connect4 with ResNet

```bash
# Recommended settings for Connect4
trainer loop \
    --env-id connect4 \
    --iterations 100 \
    --episodes 500 \
    --steps 1000 \
    --batch-size 128 \
    --device cuda \
    --lr 0.001 \
    --lr-total-steps 100000

# The ResNet (4 blocks, 128 filters) is automatically selected
# based on game_config.py settings
```

### Key Hyperparameters

| Parameter | TicTacToe | Connect4 | Notes |
|-----------|-----------|----------|-------|
| Episodes/iter | 200-500 | 500-1000 | More for larger games |
| Steps/iter | 500 | 1000 | Match replay buffer size |
| Batch size | 64 | 128 | Larger for ResNet stability |
| Learning rate | 0.001 | 0.001 | Standard starting point |
| MCTS simulations | 200 | 800 | More for deeper games |
