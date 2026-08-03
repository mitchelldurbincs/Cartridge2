# Deployment Guide

Cartridge2 supports three deployment modes, from simplest to most scalable.

## 1. Local Development

Best for experimentation and fast iteration. All processes run on your machine.

**Requirements:** PostgreSQL, Rust toolchain, Python 3.10+, Node.js 20+

```bash
# Terminal 0: Start PostgreSQL
docker compose up postgres
# Or use a local PostgreSQL: createdb cartridge && psql cartridge -f sql/schema.sql

# Terminal 1: Start web backend from the repository root
cargo run --manifest-path web/Cargo.toml

# Terminal 2: Start frontend dev server from the repository root
npm --prefix web/frontend install
npm --prefix web/frontend run dev

# Terminal 3: Train a model
# This pulls the pinned `crucible` orchestration core from GitHub. If you are
# also developing crucible, install your sibling checkout editable FIRST
# (pip install -e ../crucible) and pip will keep it.
pip install -e "trainer/.[dev]"
make build-actor build-eval
# Required: the trainer reads the replay-buffer connection string only from
# this env var (config.toml's storage.postgres_url is not used by the trainer)
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:cartridge@localhost:5432/cartridge
python -m trainer --algorithm alphazero_board_v1 loop \
  --iterations 50 --episodes 200 --steps 500
```

Open http://localhost:5173 to play against the model.

### macOS (Apple Silicon)

```bash
# Install PostgreSQL
brew install postgresql@16 && brew services start postgresql@16
createdb cartridge
psql cartridge -c "CREATE USER cartridge WITH PASSWORD 'cartridge'; GRANT ALL ON DATABASE cartridge TO cartridge;"

# Build actor with CoreML acceleration
cargo build --release --features coreml --manifest-path actor/Cargo.toml
make build-eval

# Train with MPS (Metal) for PyTorch
python -m trainer --algorithm alphazero_board_v1 loop --device auto
```

## 2. Docker Compose (Default)

Single-command training with PostgreSQL, MinIO (S3), and Prometheus included.

### Train a Model

```bash
# Start the synchronized workflow selected by [algorithm].id
docker compose up alphazero

# Select a compatible algorithm/environment pair
CARTRIDGE_ALGORITHM_ID=alphazero_board_v1 \
CARTRIDGE_COMMON_ENV_ID=connect4 \
docker compose up alphazero

# Watch training logs
docker compose logs -f alphazero

# Run a one-off command through the image's canonical trainer entrypoint.
docker compose run --rm alphazero \
  --algorithm alphazero_board_v1 evaluate --env-id connect4
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
- Web server: `http://web:8080/metrics`

Bounded collector children expose no HTTP service. Their final `ActorStats`,
abandonment details, discarded-row counts, and RSS are structured logs owned by
the parent loop's attempt.

### Services

| Service | Port | Description |
|---------|------|-------------|
| `alphazero` | 9090 (internal) | Synchronized parent loop metrics |
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
CARTRIDGE_ALGORITHM_ID=alphazero_board_v1 \
CARTRIDGE_TRAINING_ITERATIONS=100 \
CARTRIDGE_TRAINING_EPISODES_PER_ITERATION=500 \
CARTRIDGE_TRAINING_DEVICE=cuda \
CARTRIDGE_EVALUATION_INTERVAL=5 \
docker compose up alphazero
```

## 3. Kubernetes

The Kustomize manifests use PostgreSQL replay, private S3/MinIO model storage,
and profile-scoped runtime data. Both the synchronized `alphazero` image (which
contains the parent loop and bounded collector binary) and the web image must
be built with the `s3` feature. Deploy a single-node local cluster with:

```bash
kubectl apply -k k8s/overlays/local
```

The `dev` and `prod` overlays are GKE deployments backed by Filestore RWX
storage. Production deliberately contains no default Secret objects; provision
the documented PostgreSQL and MinIO credential contracts first. See
`k8s/README.md` for the exact image, secret, storage, and lineage requirements,
and `terraform/README.md` for GCP infrastructure provisioning.

## Storage Backends

### Replay Buffer

Always uses PostgreSQL. Configure via:
```bash
CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://user:pass@host:5432/cartridge
```

Replay access is bound to one exact `ReplaySelection`: the environment and
algorithm profile, a fresh 64-hex `collection_scope_id`, and the source
checkpoint ID (or `null` for root collection). Every write, count, sample,
clear, and retention operation includes that complete fence, with nullable
source comparison using `IS NOT DISTINCT FROM`. The synchronized loop creates a
new scope for every attempt and seals the exact configured episode count before
learning. It never clears the broader profile; rows from prior or abandoned
attempts remain invisible.

> **Phase 1 requires a fresh replay database.** Pre-v3 databases must be
> recreated from `sql/schema.sql`; there is intentionally no migration from
> older layouts to the exact scoped replay contract. For Compose,
> provisioning a fresh PostgreSQL volume is sufficient. `docker compose down
> -v` also does this, but removes every Compose-managed volume, so preserve
> anything needed first.

### Model Storage

| Backend | Setting | Use case |
|---------|---------|----------|
| `filesystem` | Default | Local development, single-machine training |
| `s3` | `CARTRIDGE_STORAGE_MODEL_BACKEND=s3` | Distributed training, K8s deployments |

Models and training checkpoints use a content-addressed repository. ONNX blobs
must contain custom metadata for
`cartridge.schema_version=1`, `cartridge.algorithm_id`,
`cartridge.model_contract`, `cartridge.env_id`, and
`cartridge.env_contract_version`. Every value must exactly match the configured
runtime profile. Learner-state blobs carry the equivalent profile plus the
training step and learner-configuration digest.

Filesystem paths and S3 object keys use the same namespace:

```text
profiles/{algorithm_id}/{env_id}/v{env_contract_version}/
├── models/
│   ├── blobs/sha256/{onnx_digest}.onnx
│   ├── blobs/sha256/{learner_digest}.pt
│   ├── manifests/sha256/{checkpoint_id}.json
│   ├── evaluations/manifests/sha256/{evaluation_id}.json
│   ├── run-commits/sha256/{run_commit_id}.json
│   ├── run-preparations/by-parent/{root|parent_run_commit_id}.json
│   └── channels/current.json
└── profile-scoped telemetry and registries
```

The checkpoint ID is the SHA-256 digest of the canonical manifest bytes. The
publisher safely validates both staged artifacts, creates only immutable
objects, verifies complete checkpoint/RunCommit lineage and profile/recipe
identity, then compare-and-sets `current`. Actor collection and learner
continuation select Latest from that RunHead; web selects ChampionOrLatest from
the latest RunCommit. Every consumer rejects cross-profile, wrong-digest,
wrong-size, malformed, or incompatible artifacts.

This is a Phase 1 clean cutover. Remove and retrain/re-export older PyTorch and
ONNX artifacts that lack this identity; deployment processes do not infer it
from paths, tensor shapes, or configuration defaults. No RunHead means the
trainer starts fresh and root collection/web play can remain random. Present
invalid authority fails startup. An invalid web hot-reload update is logged and
does not replace the last valid in-memory evaluator.

S3 configuration:
```bash
CARTRIDGE_STORAGE_S3_BUCKET=cartridge-models
CARTRIDGE_STORAGE_S3_ENDPOINT=http://minio:9000  # For MinIO
```

S3 publication serializes the sole mutable RunHead with its exact object ETag.
The first head uses `If-None-Match: *`; later heads use `If-Match` against the
version that was fully validated before publication. A concurrent writer gets a
compare-and-set failure. There is no lock object, lease, timeout, or manual
unlock procedure.

### Training Statistics

Training statistics are authoritative only as the canonical snapshot embedded
in the selected immutable `RunCommitV1`; `stats_id` is the SHA-256 of those
bytes. Learner resume follows the sole `RunHeadV2`, validates the entire
RunCommit/checkpoint lineage, and fails closed on corrupt or incomplete state.
The sibling `stats.json` is an atomically rebuilt projection for the web
service and is never a second resume channel.

## Configuration

All deployment modes read from `config.toml`. See `engine/engine-config/SCHEMA.md` for the full schema reference. Settings are loaded in this priority order:

1. CLI arguments (highest)
2. Environment variables (`CARTRIDGE_<SECTION>_<KEY>`)
3. `config.toml`
4. `config.defaults.toml` — checked-in defaults, and the source of truth for
   every key (lowest)

The algorithm is selected independently from the environment:

```toml
[algorithm]
id = "alphazero_board_v1"

[common]
env_id = "connect4"
```

The actor, trainer, evaluator, and web host validate this pair against strict
manifest schema v4 before starting collection, learning, serving, or
evaluation. A registered environment may still be incompatible with the
selected cartridge.

Rust maps every declared `CARTRIDGE_*` override explicitly, while Python
derives the same names from its typed schema. Unknown TOML fields and malformed
environment values fail closed. Lists such as `allowed_origins` and W&B tags
use comma-separated environment values. `[wandb]` and solver/promotion fields
are accepted cross-language but acted on by Python orchestration.
