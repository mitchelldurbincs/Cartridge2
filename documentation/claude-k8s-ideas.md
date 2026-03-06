# Kubernetes Migration Plan for Cartridge2

This document outlines the roadmap for migrating Cartridge2 to Kubernetes, both for local development and eventual cloud deployment.

## Overview

Current state: Monolithic architecture using Docker Compose with PostgreSQL replay buffer and ONNX model artifacts.

Target state: Kubernetes-native deployment with proper storage abstractions, CI/CD, and cloud-ready infrastructure.

---

## Phase 1: CI/CD Foundation (GitHub Actions)

### 1.1 Basic CI Pipeline
- [ ] Create `.github/workflows/ci.yml`
  - Run `cargo fmt --check` and `cargo clippy` for all Rust crates
  - Run `cargo test` for engine, actor, web, mcts
  - Run Python linting (`ruff`, `black --check`) for trainer
  - Run `pytest` for trainer tests (if any)
  - Cache Cargo registry and target directories
  - Cache pip dependencies

### 1.2 Docker Image Building
- [ ] Create `.github/workflows/docker.yml`
  - Build images on push to main and tags
  - Multi-platform builds (linux/amd64, linux/arm64)
  - Push to GitHub Container Registry (ghcr.io)
  - Tag images with git SHA and semantic version
  - Use BuildKit cache for faster builds

### 1.3 Release Workflow
- [ ] Create `.github/workflows/release.yml`
  - Trigger on version tags (v*)
  - Build and push production images
  - Generate changelog from commits
  - Create GitHub release with artifacts

---

## Phase 2: Dockerfile Optimization

### 2.1 Image Size Reduction
- [ ] Review all Dockerfiles for optimization opportunities:
  - `Dockerfile.alphazero` - Already uses multi-stage, ~good
  - `actor/Dockerfile` - Check for unnecessary build deps in final image
  - `trainer/Dockerfile` - Ensure slim base, no dev dependencies
  - `web/Dockerfile` - Check for static linking opportunities
  - `web/frontend/Dockerfile` - Ensure nginx:alpine base

### 2.2 Build Optimization
- [ ] Add `.dockerignore` files to reduce build context
- [ ] Use `cargo-chef` for better Rust dependency caching
- [ ] Consider using `sccache` for distributed Rust compilation cache
- [ ] Pin base image versions for reproducibility

### 2.3 Security Hardening
- [ ] Run containers as non-root user
- [ ] Use distroless or scratch base images where possible
- [ ] Scan images with Trivy in CI
- [ ] Remove unnecessary capabilities

---

## Phase 3: Local Kubernetes Development

### 3.1 Choose Local K8s Tool
Options (pick one):
- [ ] **kind** (Kubernetes IN Docker) - Recommended for CI, lightweight
- [ ] **k3d** (k3s in Docker) - Fast, good for dev
- [ ] **minikube** - Feature-rich, multi-driver support

### 3.2 Local Development Setup
- [ ] Create `k8s/local/` directory structure
- [ ] Write setup script for local cluster creation
- [ ] Configure local registry for image pushes
- [ ] Document dev workflow (build → push to local registry → deploy)

### 3.3 Development Tools
- [ ] Add Tilt or Skaffold configuration for hot-reload development
- [ ] Create `skaffold.yaml` or `Tiltfile` for iterative development
- [ ] Configure port-forwarding scripts

---

## Phase 4: Kubernetes Manifests

### 4.1 Base Manifests (`k8s/base/`)
- [ ] **Namespace**: `cartridge` namespace definition
- [ ] **ConfigMap**: Centralized config (from config.toml)
- [ ] **Secrets**: Placeholder for future secrets management
- [ ] **Storage**:
  - PersistentVolumeClaim for PostgreSQL data
  - PersistentVolumeClaim for ONNX models
  - Consider: Should models use object storage (S3/GCS) instead?

### 4.2 Workloads
- [ ] **Deployment: web** - Stateless API server
  - Readiness/liveness probes on `/health`
  - Resource requests/limits
  - HPA for autoscaling (when needed)

- [ ] **Deployment: frontend** - Nginx serving static files
  - Simple deployment, very low resources
  - Can scale horizontally easily

- [ ] **Job: alphazero** - Training as a Kubernetes Job
  - One-shot training runs
  - Or convert to CronJob for scheduled training

- [ ] **Deployment: actor** (optional) - Continuous self-play
  - StatefulSet if actors need stable identity
  - Or Deployment if stateless

### 4.3 Services & Networking
- [ ] **Service: web** - ClusterIP for internal access
- [ ] **Service: frontend** - ClusterIP for internal access
- [ ] **Ingress**: Route external traffic to frontend
  - nginx-ingress or traefik
  - TLS termination
  - Path-based routing (`/api/*` → web, `/*` → frontend)

### 4.4 Kustomize Overlays
- [ ] `k8s/overlays/local/` - Local development settings
- [ ] `k8s/overlays/staging/` - Staging environment
- [ ] `k8s/overlays/production/` - Production settings

---

## Phase 5: Architecture Refactoring for K8s

### 5.1 Storage Strategy (CRITICAL)
Current: PostgreSQL replay buffer + ONNX files on shared filesystem

**Problem**: Shared filesystem doesn't scale well in K8s across pods.

**Options**:

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **A. Keep single PostgreSQL + PVC** | Simple, already implemented | Limited HA/failover | Good for local/staging |
| **B. Managed PostgreSQL** | HA, backups, scalable ops | Higher cost | Better for production |
| **C. Redis for replay buffer** | Fast, ephemeral is OK | Data loss on restart | Good for high-throughput |

**Recommendations**:
- [ ] Short-term: Use ReadWriteOnce PVC for training jobs (single actor/trainer)
- [ ] Medium-term: Add managed PostgreSQL option for replay buffer
- [ ] Model storage: Consider S3-compatible storage (MinIO locally, S3/GCS in cloud)

### 5.2 Model Storage
- [ ] Add S3/MinIO support for model storage
  - Actor downloads latest model from object storage
  - Trainer uploads new models to object storage
  - Web server caches model locally, polls for updates
- [ ] Create `model-storage` abstraction (file vs S3)

### 5.3 Configuration Management
- [ ] Migrate from config.toml to ConfigMap-friendly format
- [ ] Support configuration via environment variables (already partial)
- [ ] Consider using Kubernetes ConfigMap mounted as file

---

## Phase 6: Cloud Cost Optimization

### 6.1 Resource Right-Sizing
- [ ] Profile actual resource usage (CPU/memory) for each component
- [ ] Set appropriate requests (guaranteed) and limits (max)
- [ ] Start conservative, tune based on metrics

**Initial estimates**:
| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| web | 100m | 500m | 128Mi | 256Mi |
| frontend | 50m | 100m | 64Mi | 128Mi |
| actor | 500m | 2000m | 512Mi | 1Gi |
| trainer (CPU) | 1000m | 4000m | 1Gi | 4Gi |
| trainer (GPU) | 500m | 1000m | 2Gi | 8Gi |

### 6.2 Compute Cost Strategies
- [ ] Use **Spot/Preemptible instances** for training workloads
  - Training can be checkpointed and resumed
  - 60-90% cost savings
- [ ] Use **small instances** for web/frontend (t3.small or equivalent)
- [ ] Consider **ARM instances** (Graviton on AWS) - cheaper, good for Rust
- [ ] Schedule training during off-peak hours if cloud provider offers discounts

### 6.3 Training Efficiency
- [ ] Implement training checkpointing (already have `--start-iteration`)
- [ ] Add graceful shutdown handling for spot instance termination
- [ ] Batch multiple training iterations before writing models
- [ ] Profile and optimize hot paths in actor/trainer

### 6.4 Storage Cost Optimization
- [ ] Use appropriate storage classes (SSD only where needed)
- [ ] Implement model retention policy (keep last N models)
- [ ] Compress old models/data or move to cold storage
- [ ] Clean up replay buffer after training iterations

### 6.5 Auto-Scaling
- [ ] **Web/Frontend**: HPA based on CPU/requests
- [ ] **Actor**: Consider KEDA for scaling based on training needs
- [ ] Scale to zero when not in use (especially for training workloads)

---

## Phase 7: Observability & Operations

### 7.1 Logging
- [ ] Ensure all components log to stdout (K8s standard)
- [ ] Structured JSON logging for easier parsing
- [ ] Log aggregation (Loki, CloudWatch, etc.)

### 7.2 Metrics
- [ ] Add Prometheus metrics endpoint to web server (`/metrics`)
- [ ] Export training metrics (loss, iteration, games played)
- [ ] Grafana dashboards for monitoring

### 7.3 Health Checks
- [ ] Readiness probes: Is the service ready to receive traffic?
- [ ] Liveness probes: Is the service healthy?
- [ ] Startup probes: For slow-starting containers (trainer)

---

## Phase 8: Security

### 8.1 Network Policies
- [ ] Restrict pod-to-pod communication
- [ ] Only frontend → web allowed
- [ ] Training jobs isolated

### 8.2 RBAC
- [ ] Service accounts per workload
- [ ] Minimal permissions

### 8.3 Secrets Management
- [ ] External secrets operator or sealed secrets
- [ ] No secrets in ConfigMaps

---

## Implementation Priority

### Must Have (Before Cloud)
1. CI/CD pipeline (Phase 1.1, 1.2)
2. Dockerfile security (Phase 2.3 - non-root)
3. Basic K8s manifests (Phase 4.1, 4.2)
4. Local K8s setup (Phase 3.1, 3.2)

### Should Have (For Production)
5. Resource limits defined (Phase 6.1)
6. Health checks (Phase 7.3)
7. Kustomize overlays (Phase 4.4)
8. Spot instance support (Phase 6.2)

### Nice to Have (Optimization)
9. Managed PostgreSQL replay backend (Phase 5.1)
10. S3 model storage (Phase 5.2)
11. Prometheus metrics (Phase 7.2)
12. Auto-scaling (Phase 6.5)

---

## Quick Wins (Do First)

1. **Add .dockerignore files** - Faster builds immediately
2. **Run as non-root in containers** - Security best practice
3. **Create basic CI workflow** - Catch issues early
4. **Set up kind cluster locally** - Test K8s compatibility
5. **Add health endpoints** - Already have `/health` in web

---

## Questions to Decide

1. **Storage backend**: Self-host PostgreSQL or use managed PostgreSQL?
2. **Model storage**: Local PVC or S3-compatible object storage?
3. **Cloud provider**: AWS, GCP, or other?
4. **GPU training**: Cloud GPU instances or local only?
5. **Multi-game support**: One cluster per game or multi-tenant?

---

## References

- [kind Quick Start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [Skaffold](https://skaffold.dev/)
- [Kustomize](https://kustomize.io/)
- [AWS Spot Best Practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html)
- [Kubernetes Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
