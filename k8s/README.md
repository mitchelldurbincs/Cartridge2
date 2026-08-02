# Cartridge2 Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Cartridge2 with horizontal scaling of self-play actors.

## Architecture

```
                                  ┌─────────────────────────────────────┐
                                  │           Ingress Controller        │
                                  │         (nginx/traefik/alb)         │
                                  └──────────────┬──────────────────────┘
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                        │                        │                        │
                        ▼                        ▼                        ▼
              ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
              │    Frontend     │     │   Web Backend   │     │     Stats       │
              │   (Svelte UI)   │     │  (Axum Server)  │     │    /stats       │
              │    replicas: 2  │     │   replicas: 1   │     │                 │
              └─────────────────┘     └────────┬────────┘     └─────────────────┘
                                               │
                                               │ Model Loading
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Shared Models PVC (ReadWriteMany)                       │
│                              or S3/MinIO bucket                                 │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Actor Pod 1   │     │   Actor Pod N   │     │     Trainer     │
│  (Self-play)    │     │  (Self-play)    │     │  (PyTorch NN)   │
│                 │     │                 │     │   replicas: 1   │
│  Reads models   │     │  Reads models   │     │  Writes models  │
│  Writes to DB   │     │  Writes to DB   │     │  Reads from DB  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       PostgreSQL        │
                    │    (Replay Buffer)      │
                    │   StatefulSet: 1 pod    │
                    └─────────────────────────┘
```

## Components

| Component | Purpose | Scalable? |
|-----------|---------|-----------|
| **Actor** | Self-play episode generation using MCTS | ✅ Yes (HPA enabled) |
| **Trainer** | Neural network training | ❌ No (single replica) |
| **Web** | Game API backend | ❌ No (temporary single-replica constraint) |
| **Frontend** | Svelte UI | ✅ Yes |
| **PostgreSQL** | Replay buffer storage | ❌ Single instance |
| **MinIO** | S3-compatible model storage | ❌ Single instance |

### Web backend replica constraint

Do not scale the `web` Deployment above one replica. `AppState` currently owns
one process-local `GameSession`; the API has no client/session identifier and
no shared session store. Multiple backend pods would therefore hold divergent
boards, and the Service could route consecutive requests from one browser to
different games. The manifest also uses the `Recreate` deployment strategy so
old and new backend pods do not overlap during a rollout.

One replica prevents cross-pod state divergence, but it does **not** provide
per-user isolation: all browsers still share the same in-process game, so one
client can reset or move another client's board. Limit the game API to trusted,
single-user use until opaque session IDs (or an equivalent shared-state design)
are implemented. The stateless frontend may remain horizontally scaled.

## Quick Start

### Prerequisites

- Kubernetes cluster (minikube, kind, EKS, GKE, etc.)
- kubectl configured
- Docker images built and accessible

### Build Docker Images

```bash
# Build all images
docker build -f Dockerfile.alphazero -t cartridge-alphazero .
docker build -f web/Dockerfile -t cartridge-web .
docker build -f web/frontend/Dockerfile -t cartridge-frontend web/frontend

# For minikube, load images directly:
minikube image load cartridge-alphazero:latest
minikube image load cartridge-web:latest
minikube image load cartridge-frontend:latest

# For remote clusters, push to your registry:
docker tag cartridge-alphazero your-registry/cartridge-alphazero:latest
docker push your-registry/cartridge-alphazero:latest
# ... etc
```

### Deploy (Development)

```bash
# Create namespace and deploy all resources
kubectl apply -k k8s/overlays/dev

# Watch pods come up
kubectl get pods -n cartridge -w

# Check actor logs
kubectl logs -n cartridge -l app.kubernetes.io/name=actor -f

# Check trainer logs
kubectl logs -n cartridge -l app.kubernetes.io/name=trainer -f
```

### Deploy (Production)

```bash
# Update image references in k8s/overlays/prod/kustomization.yaml first!
kubectl apply -k k8s/overlays/prod
```

### Scale Actors

```bash
# Manual scaling
kubectl scale deployment actor -n cartridge --replicas=8

# Check HPA status
kubectl get hpa -n cartridge

# HPA will auto-scale based on CPU (70% target)
```

### Access the UI

```bash
# Port forward for local access
kubectl port-forward -n cartridge svc/frontend 8080:80

# Or use the ingress (requires ingress controller)
# Add to /etc/hosts: <ingress-ip> cartridge.local
# Then visit http://cartridge.local
```

## Configuration

### Environment Variables

All components read from the `cartridge-config` ConfigMap. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV_ID` | `connect4` | Game environment |
| `TRAINING_ITERATIONS` | `100` | Training iterations |
| `EPISODES_PER_ITERATION` | `500` | Episodes per iteration |
| `MCTS_MAX_SIMS` | `400` | MCTS simulations |
| `BATCH_SIZE` | `128` | Training batch size |

### Secrets

- `postgres-credentials`: PostgreSQL username/password
- `minio-credentials`: MinIO/S3 access keys

**Important**: For production, use a proper secrets management solution (Sealed Secrets, External Secrets, Vault, etc.)

## Storage

### Shared Model Storage

The current setup uses a ReadWriteMany PVC for model files. This requires a storage class that supports RWX access:

- **AWS**: EFS (Elastic File System)
- **GCP**: Filestore
- **Azure**: Azure Files
- **On-prem**: NFS

For development with minikube/kind, the dev overlay uses ReadWriteOnce which works but means only the trainer can write.

### Future: S3 Model Storage

The codebase has partial S3 support. When complete:
- Trainer uploads models to S3/MinIO
- Actors poll S3 for model updates (ETag-based change detection)
- No need for shared PVC

## Monitoring

### Metrics

Actors and trainer write stats to `stats.json`. The web backend exposes `/stats` endpoint.

For production monitoring:
1. Deploy Prometheus + Grafana
2. Add ServiceMonitor for metrics scraping
3. Create dashboards for:
   - Episodes generated per second
   - Training loss over time
   - Actor CPU utilization
   - Replay buffer size

### Logging

All components log to stdout. Use your cluster's logging solution:
- EKS: CloudWatch Logs
- GKE: Cloud Logging
- Self-hosted: Loki, Elasticsearch

## Troubleshooting

### Actors not generating episodes

```bash
# Check actor logs
kubectl logs -n cartridge -l app.kubernetes.io/name=actor

# Verify PostgreSQL connection
kubectl exec -n cartridge -it deploy/actor -- nc -zv postgres 5432

# Check if model exists
kubectl exec -n cartridge -it deploy/actor -- ls -la /data/models/
```

### Trainer not training

```bash
# Check trainer logs
kubectl logs -n cartridge -l app.kubernetes.io/name=trainer

# Verify data in replay buffer
kubectl exec -n cartridge -it statefulset/postgres -- \
  psql -U cartridge -d cartridge -c "SELECT COUNT(*) FROM transitions;"
```

### Model not updating

```bash
# Check if trainer is saving models
kubectl exec -n cartridge -it deploy/trainer -- ls -la /data/models/

# Check PVC is mounted correctly
kubectl describe pod -n cartridge -l app.kubernetes.io/name=actor
```

## Directory Structure

```
k8s/
├── base/                    # Base manifests
│   ├── kustomization.yaml   # Kustomize config
│   ├── namespace.yaml       # Namespace definition
│   ├── configmap.yaml       # Shared config + secrets
│   ├── models-pvc.yaml      # Shared PVC for models
│   ├── postgres/            # PostgreSQL StatefulSet
│   ├── minio/               # MinIO deployment
│   ├── actor/               # Actor deployment + HPA
│   ├── trainer/             # Trainer deployment
│   └── web/                 # Web + Frontend + Ingress
└── overlays/
    ├── dev/                 # Development overrides
    │   └── kustomization.yaml
    └── prod/                # Production overrides
        └── kustomization.yaml
```
