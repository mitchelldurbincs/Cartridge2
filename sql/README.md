# Database Schema

Cartridge2 uses PostgreSQL for the replay buffer (storing training transitions) and training coordination.

## Schema Files

| File | Purpose | Used by |
|------|---------|---------|
| `schema.sql` | Core schema for local development | Rust actor, Python trainer |
| `../scripts/init-postgres.sql` | Extended schema for Docker/K8s deployments | Docker Compose (mounted as init script) |

### Differences

Both files share the same core tables (`transitions`, `game_metadata`). The `init-postgres.sql` file adds:

- **`training_stats`** - Stores per-iteration training metrics for distributed coordination
- **`model_versions`** - Tracks model versions and S3 keys for distributed model storage
- **Permission grants** for K8s service accounts

For local development, the actor and trainer create tables automatically if they don't exist. For Docker/K8s, `init-postgres.sql` is mounted into the PostgreSQL container's `docker-entrypoint-initdb.d/` directory and runs on first startup.

## Tables

### transitions

Main replay buffer table. Stores game episode data used for training.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Unique transition ID |
| `env_id` | TEXT | Game identifier (e.g., "tictactoe", "connect4") |
| `episode_id` | TEXT | Groups transitions from a single game |
| `step_number` | INTEGER | Position within the episode |
| `state` | BYTEA | Encoded game state |
| `action` | BYTEA | Encoded action taken |
| `next_state` | BYTEA | Encoded resulting state |
| `observation` | BYTEA | Neural network input for current state |
| `next_observation` | BYTEA | Neural network input for next state |
| `reward` | REAL | Immediate reward |
| `done` | BOOLEAN | Whether the game ended |
| `timestamp` | BIGINT | Unix timestamp for ordering |
| `policy_probs` | BYTEA | MCTS visit distribution (policy target) |
| `mcts_value` | REAL | MCTS value estimate |
| `game_outcome` | REAL | Backfilled final game result (+1, -1, 0) |
| `created_at` | TIMESTAMP | Row creation time |

**Indexes:** `timestamp` (sampling), `episode_id` (outcome backfill), `env_id` (filtering)

### game_metadata

Self-describing game configuration. Written by the actor so the trainer knows game dimensions.

| Column | Type | Description |
|--------|------|-------------|
| `env_id` | TEXT PK | Game identifier |
| `display_name` | TEXT | Human-readable name |
| `board_width` | INTEGER | Board width |
| `board_height` | INTEGER | Board height |
| `num_actions` | INTEGER | Action space size |
| `obs_size` | INTEGER | Observation vector length |
| `legal_mask_offset` | INTEGER | Offset into observation for legal move mask |
| `player_count` | INTEGER | Number of players |

### training_stats (K8s only)

Per-iteration training metrics for distributed coordination.

### model_versions (K8s only)

Tracks ONNX model versions and S3 storage keys. Used when `model_backend = "s3"`.

## Setup

```bash
# Local: Docker Compose starts PostgreSQL automatically
docker compose up postgres

# Or use a local PostgreSQL installation:
createdb cartridge
psql cartridge -f sql/schema.sql
psql cartridge -c "CREATE USER cartridge WITH PASSWORD 'cartridge'; GRANT ALL ON DATABASE cartridge TO cartridge;"
```

## Connection

Default connection string: `postgresql://cartridge:cartridge@localhost:5432/cartridge`

Override via `config.toml` or environment variable:
```bash
CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://user:pass@host:5432/dbname
```
