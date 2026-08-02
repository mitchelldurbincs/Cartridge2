# Deployment Guide

Cartridge2 supports three deployment modes, from simplest to most scalable.

## 1. Local Development

Best for experimentation and fast iteration. All processes run on your machine.

**Requirements:** PostgreSQL, Rust toolchain, Python 3.10+, Node.js 20+

```bash
# Terminal 0: Start PostgreSQL
docker compose up postgres
# Or use a local PostgreSQL: createdb cartridge && psql cartridge -f sql/schema.sql

# Terminal 1: Start web backend
cd web && cargo run

# Terminal 2: Start frontend dev server
cd web/frontend && npm install && npm run dev

# Terminal 3: Train a model
# This pulls the pinned `crucible` orchestration core from GitHub. If you are
# also developing crucible, install your sibling checkout editable FIRST
# (pip install -e ../../crucible) and pip will keep it.
cd trainer && pip install -e .
# Required: the trainer reads the replay-buffer connection string only from
# this env var (config.toml's storage.postgres_url is not used by the trainer)
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:cartridge@localhost:5432/cartridge
python -m trainer loop --iterations 50 --episodes 200 --steps 500
```

Open http://localhost:5173 to play against the model.

### macOS (Apple Silicon)

```bash
# Install PostgreSQL
brew install postgresql@16 && brew services start postgresql@16
createdb cartridge
psql cartridge -c "CREATE USER cartridge WITH PASSWORD 'cartridge'; GRANT ALL ON DATABASE cartridge TO cartridge;"

# Build actor with CoreML acceleration
cd actor && cargo build --release --features coreml

# Train with MPS (Metal) for PyTorch
python -m trainer loop --device auto
```

## 2. Docker Compose (Default)

Single-command training with PostgreSQL, MinIO (S3), and Prometheus included.

### Train a Model

```bash
# Start synchronized AlphaZero training
docker compose up alphazero

# Train a different game
CARTRIDGE_COMMON_ENV_ID=connect4 docker compose up alphazero

# Watch training logs
docker compose logs -f alphazero

# Run standalone evaluation.
# --entrypoint is required: the image's ENTRYPOINT is `python -m trainer loop`,
# so a bare `run ... python -m trainer evaluate` gets appended to it.
docker compose run --rm --entrypoint python alphazero \
  -m trainer evaluate --model /app/data/models/latest.onnx
```

### Play Against Trained Model

```bash
docker compose up web frontend
# Open http://localhost in browser
```

### Monitor Training

```bash
docker compose up prometheus
# Prometheus UI at http://localhost:9092
```

Metrics scraped from:
- Trainer: `http://alphazero:9090/metrics`
- Actor: `http://alphazero:9091/metrics`
- Web server: `http://web:8080/metrics`

### Services

| Service | Port | Description |
|---------|------|-------------|
| `alphazero` | 9090, 9091 (internal) | Synchronized training (actor + trainer) |
| `web` | 8080 | Backend API server |
| `frontend` | 80 (host) -> 8080 (container) | Nginx serving Svelte app |
| `postgres` | 5432 | Replay buffer database |
| `minio` | 9000 (API), 9001 (console) | S3-compatible model storage |
| `minio-setup` | - | One-shot bucket initialiser. `alphazero` and `web` both wait on it via `service_completed_successfully`, so starting them without it blocks. |
| `prometheus` | 9092 | Metrics collection |

### MinIO Console

Access at http://localhost:9001 with credentials from your `.env` file (`MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`). Default values are `minioadmin` / `changeme` for local development only. Models are stored in the `cartridge-models` bucket.

**Security Note:** Always change default MinIO credentials in production by setting `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` environment variables.

### Environment Variable Overrides

All `config.toml` settings can be overridden:

```bash
CARTRIDGE_COMMON_ENV_ID=connect4 \
CARTRIDGE_TRAINING_ITERATIONS=100 \
CARTRIDGE_TRAINING_EPISODES_PER_ITERATION=500 \
CARTRIDGE_TRAINING_DEVICE=cuda \
CARTRIDGE_EVALUATION_INTERVAL=5 \
docker compose up alphazero
```

## 3. Kubernetes (K8s Simulation)

Test distributed deployments locally using Docker Compose with the K8s overlay. This uses PostgreSQL for replay and MinIO for model storage, mimicking a Kubernetes environment.

```bash
# Start with K8s-style backends
docker compose -f docker-compose.yml -f docker-compose.k8s.yml up alphazero

# Parallel self-play workers run inside the alphazero service — tune
# [training].num_actors in config.toml (or CARTRIDGE_TRAINING_NUM_ACTORS)

# Play against trained model
docker compose -f docker-compose.yml -f docker-compose.k8s.yml up web frontend
```

For actual Kubernetes deployment the Kustomize manifests are in `k8s/`:

```bash
kubectl apply -k k8s/overlays/dev
```

> **Web backend safety constraint:** Keep the `web` Deployment at one replica.
> Its game state is currently a single process-local `GameSession`, with no
> client/session identifier or shared state store. The checked-in manifest uses
> one replica and a `Recreate` rollout so requests cannot be routed between
> divergent backend processes. This still does not isolate users: all browsers
> share that one game and can reset or move one another's board. Treat the game
> API as trusted single-user functionality until per-session ownership exists.

See `k8s/README.md` for details, and `terraform/README.md` for GCP infrastructure provisioning.

## Storage Backends

### Replay Buffer

Always uses PostgreSQL. Configure via:
```bash
CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://user:pass@host:5432/cartridge
```

### Model Storage

| Backend | Setting | Use case |
|---------|---------|----------|
| `filesystem` | Default | Local development, single-machine training |
| `s3` | `CARTRIDGE_STORAGE_MODEL_BACKEND=s3` | Distributed training, K8s deployments |

S3 configuration:
```bash
CARTRIDGE_STORAGE_S3_BUCKET=cartridge-models
CARTRIDGE_STORAGE_S3_ENDPOINT=http://minio:9000  # For MinIO
```

## Configuration

All deployment modes read from `config.toml`. See `engine/engine-config/SCHEMA.md` for the full schema reference. Settings are loaded in this priority order:

1. CLI arguments (highest)
2. Environment variables (`CARTRIDGE_<SECTION>_<KEY>`)
3. `config.toml`
4. `config.defaults.toml` — checked-in defaults, and the source of truth for
   every key (lowest)

> Rust enumerates `CARTRIDGE_*` variables explicitly in
> `engine/engine-config/src/loader.rs`, with one override for every field in its
> `CentralConfig`. List overrides such as `CARTRIDGE_WEB_ALLOWED_ORIGINS` use a
> JSON array. Python additionally supports its Python-only fields.
>
> `[wandb]` and the solver-eval keys (`solver_games`, `solver_seed`,
> `promotion_metric`, `promotion_margin`) are read only by the Python trainer.

A non-empty explicit `CARTRIDGE_CONFIG` path must exist and parse successfully.
Known, non-empty environment overrides are parsed and range-checked at startup;
malformed values stop the process. Empty `${VALUE:-}` placeholders remain unset
and defer to the file/default value.
