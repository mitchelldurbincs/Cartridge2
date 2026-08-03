# Cartridge2 Kubernetes deployment

These manifests run the Phase 1 `alphazero_board_v1` cartridge as one
synchronized training Job plus the serving stack. There is deliberately no
independent actor Deployment: one loop process creates a new replay collection
scope for every attempt and owns all bounded collector subprocesses that may
write to it.

## Runtime architecture

```text
trainer Job
  ├─ bounded collector subprocesses ── scoped replay rows ──> PostgreSQL
  ├─ learner <──────────────────────── exact same scope ─────┘
  ├─ evaluator
  └─ immutable checkpoint / RunCommit publication ─────────> S3/MinIO
                                                               │
web replicas ── load champion (latest before first promotion) ──┘
      │
      └──────────────────────────────────────────────────────> frontend
```

The Job executes the canonical `loop` command. `TOTAL_ITERATIONS` is an
immutable global target: a restarted Job resumes from RunHead and runs only the
missing iterations. Reaching the target completes the Job successfully.

Every model and run artifact is namespaced beneath the runtime profile:

```text
profiles/{algorithm_id}/{env_id}/v{env_contract_version}
```

The contract version comes from the bundled environment manifest. It is not a
deployment knob.

## Components

| Component | Responsibility | Scale |
|---|---|---:|
| Trainer Job | Owns collection scopes, collectors, learning, evaluation, and commits | One active Job |
| Web | Serves games from the promoted champion, or latest before a champion exists | Horizontal |
| Frontend | Svelte UI | Horizontal |
| PostgreSQL | Opaque, collection-scoped replay records | One |
| MinIO | Private S3-compatible immutable artifact store | One |
| `runtime-data` PVC | Profile-scoped projections and local caches | Shared |

Parallel self-play is controlled by `NUM_ACTORS`. Those collectors are child
processes in the training pod, so they share the exact iteration scope and
source checkpoint. Scaling independent actor pods would violate that fence and
is unsupported.

## Quick start

Prerequisites:

- a Kubernetes cluster with `kubectl` and Kustomize support;
- images accessible to the cluster;
- an RWX storage class for multi-node deployments;
- GKE 1.33 or newer for the checked-in 100 GiB Basic HDD Filestore classes.

Build the S3-enabled runtime images:

```bash
docker build --build-arg CARGO_FEATURES=s3 -f Dockerfile.alphazero -t cartridge-alphazero .
docker build --build-arg CARGO_FEATURES=s3 -f web/Dockerfile -t cartridge-web .
docker build -f web/frontend/Dockerfile -t cartridge-frontend web/frontend
```

For a local single-node cluster:

```bash
kubectl apply -k k8s/overlays/local
kubectl get pods,jobs -n cartridge -w
```

GKE development and production use Filestore-backed shared storage:

```bash
kubectl apply -k k8s/overlays/dev
kubectl apply -k k8s/overlays/prod
```

Publish the images first and update the production image mappings. To rerun a
completed Job after deliberately changing the recipe or target, delete that
exact Job and apply the overlay again:

```bash
kubectl delete job trainer -n cartridge
kubectl apply -k k8s/overlays/local
```

Deleting the Job does not delete RunHead, immutable model artifacts, replay
rows, PostgreSQL, or the PVC.

## Configuration

The `cartridge-config` ConfigMap supplies the synchronized run recipe and
backend selection. Changing a recipe field for an existing RunHead is rejected;
start a new profile data root for a genuinely different experiment.

| Key | Default | Meaning |
|---|---:|---|
| `ALGORITHM_ID` | `alphazero_board_v1` | Algorithm cartridge |
| `ENV_ID` | `connect4` | Registered compatible environment |
| `TOTAL_ITERATIONS` | `100` | Global run target |
| `EPISODES_PER_ITERATION` | `500` | Completed self-play games per scope |
| `STEPS_PER_ITERATION` | `400` | Learner updates per iteration |
| `NUM_ACTORS` | `4` | Bounded collector child processes |
| `BATCH_SIZE` | `128` | Learner batch size |
| `LEARNING_RATE` | `0.001` | Optimizer learning rate |
| `WEIGHT_DECAY` | `0.0001` | Optimizer weight decay |
| `GRAD_CLIP` | `1.0` | Gradient clipping norm |
| `DEVICE` | `cpu` | Learner device |
| `ACTOR_LOG_INTERVAL` | `50` | Completed games between collector progress logs |
| `ACTOR_EPISODE_TIMEOUT_SECONDS` | `180` | Exact per-game hard timeout |
| `ACTOR_EVAL_BATCH_SIZE` | `32` | Collector ONNX evaluation batch |
| `ACTOR_ONNX_INTRA_THREADS` | `1` | ONNX threads per collector |
| `MCTS_START_SIMS` | `50` | First-iteration simulations |
| `MCTS_MAX_SIMS` | `400` | Simulation ramp ceiling |
| `MCTS_SIM_RAMP_RATE` | `20` | Simulations added each iteration |
| `MCTS_C_PUCT` | `1.4` | Collector exploration constant |
| `MCTS_TEMPERATURE` | `1.0` | Early-game action temperature |
| `MCTS_LATE_TEMPERATURE` | `0.1` | Late-game action temperature |
| `MCTS_TEMP_THRESHOLD` | `15` | Late-game temperature threshold |
| `MCTS_DIRICHLET_ALPHA` | `0.3` | Collector root-noise concentration |
| `MCTS_DIRICHLET_WEIGHT` | `0.25` | Collector root-noise mixture weight |
| `EVALUATION_INTERVAL` | `1` | Evaluation cadence |
| `EVALUATION_GAMES` | `50` | Seat-balanced evaluation games |
| `EVALUATION_SIMULATIONS` | `0` | Evaluation MCTS budget; zero uses the policy head |
| `EVALUATION_TEMPERATURE` | `0.2` | Evaluation action temperature |
| `EVALUATION_WIN_THRESHOLD` | `0.55` | Promotion threshold |
| `EVALUATION_VS_RANDOM` | `true` | Include the random-baseline evaluation family |
| `SOLVER_GAMES` | `100` | Connect4 solver-scored games |
| `EVALUATION_SEED` | `42` | Reproducible evaluation seed |
| `PROMOTION_METRIC` | `win_rate` | Promotion criterion |
| `PROMOTION_MARGIN` | `0.0` | Solver-optimal margin; must stay zero under win-rate promotion |

Secret contract:

- `postgres-credentials`: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_URL`;
- `minio-credentials`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`.

The local and dev overlays include intentionally weak development secrets. The
production overlay creates no Secrets; provision both through your secret
manager before applying it.

## Replay and artifact authority

Replay schema v3 records the environment/algorithm profile,
`collection_scope_id`, and `source_checkpoint_id` alongside opaque cartridge
payload bytes. Every writer, count, sample, cleanup, and clear is bound to one
exact selection. Rows from abandoned attempts or unrelated producers remain
invisible to the current learner.

Model publication writes digest-addressed blobs, canonical manifests,
evaluations, and RunCommit objects before compare-and-setting the single
mutable `models/channels/current.json` RunHead. S3 compare-and-set uses object
ETags; there is no lock object to recover manually.

The PVC is mounted at `/data`. It contains profile-scoped projections and
caches; immutable artifact authority is in S3/MinIO and RunHead. The local
overlay changes the claim to `ReadWriteOnce`, which is appropriate only on a
single-node cluster. The GKE overlays use provider-specific RWX Filestore
classes.

## Operations

Follow the synchronized run:

```bash
kubectl logs -n cartridge job/trainer -f
kubectl get job trainer -n cartridge
```

Inspect replay scopes:

```bash
kubectl exec -n cartridge statefulset/postgres -- \
  psql -U cartridge -d cartridge -c \
  "SELECT collection_scope_id, source_checkpoint_id, COUNT(*) FROM replay_records GROUP BY 1,2 ORDER BY 1;"
```

Access the UI:

```bash
kubectl port-forward -n cartridge svc/frontend 8080:80
```

If the Job fails, its logs should identify whether collection, exact replay
selection, learning, evaluation, journal recovery, or RunHead compare-and-set
failed. Do not start an ad-hoc actor against the same profile as a workaround;
it will not have the Job's collection scope and its rows are intentionally
unusable.

## Layout

```text
k8s/
├── base/
│   ├── trainer/job.yaml
│   ├── web/
│   ├── postgres/
│   ├── minio/
│   ├── configmap.yaml
│   ├── runtime-data-pvc.yaml
│   └── kustomization.yaml
├── development-secrets/
└── overlays/{local,dev,prod}/
```
