# Deployment Guide

Cartridge2 supports three deployment modes, from simplest to most scalable.

## 1. Local Development

Best for experimentation and fast iteration. All processes run on your machine.

**Requirements:** PostgreSQL, Rust toolchain, Python 3.11+, Node.js 20+

```bash
# Terminal 0: Start PostgreSQL
docker compose up postgres
# Or use a local PostgreSQL: createdb cartridge && psql cartridge -f sql/schema.sql

# Terminal 1: Start web backend
cd web && cargo run

# Terminal 2: Start frontend dev server
cd web/frontend && npm install && npm run dev

# Terminal 3: Train a model
cd trainer && pip install -e .
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

# Run standalone evaluation
docker compose run --rm alphazero python -m trainer evaluate --model /app/data/models/latest.onnx
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
| `frontend` | 80 | Nginx serving Svelte app |
| `postgres` | 5432 | Replay buffer database |
| `minio` | 9000 (API), 9001 (console) | S3-compatible model storage |
| `prometheus` | 9092 | Metrics collection |

### MinIO Console

Access at http://localhost:9001 with credentials `aspect` / `password123`. Models are stored in the `cartridge-models` bucket.

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

# Scale actors horizontally (4 parallel self-play workers)
docker compose -f docker-compose.yml -f docker-compose.k8s.yml up --scale actor=4

# Play against trained model
docker compose -f docker-compose.yml -f docker-compose.k8s.yml up web frontend
```

For actual Kubernetes deployment, see `k8s/README.md`. For GCP infrastructure provisioning, see `terraform/README.md`.

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
4. Built-in defaults (lowest)
