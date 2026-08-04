# Cartridge2 Terraform Infrastructure

This directory provisions the Google Cloud foundation used by the checked-in
Kubernetes deployment: a VPC, GKE Autopilot cluster, and Artifact Registry.
PostgreSQL and MinIO run inside the cluster and are owned by `k8s/`, not by
Terraform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Google Cloud                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    VPC Network                             │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              GKE Autopilot Cluster                   │  │  │
│  │  │  ┌─────────────────────┐ ┌───────────────────────┐  │  │  │
│  │  │  │ Trainer Job         │ │    Web + Frontend     │  │  │  │
│  │  │  │ (loop + collectors) │ │                       │  │  │  │
│  │  │  └─────────────────────┘ └───────────────────────┘  │  │  │
│  │  │  ┌─────────────────────┐ ┌───────────────────────┐  │  │  │
│  │  │  │ PostgreSQL (Replay) │ │    MinIO (Models)     │  │  │  │
│  │  │  └─────────────────────┘ └───────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│                    ┌────────────────────────────┐                │
│                    │     Artifact Registry      │                │
│                    │     (Container Images)     │                │
│                    └────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. [Terraform](https://www.terraform.io/downloads) >= 1.5.0
2. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
3. Docker, `kubectl`, and the standalone `kustomize` CLI
4. A GCP project with billing enabled

Serving the checked-in Ingress also requires an nginx Ingress controller. The
Terraform modules provision the cluster foundation, not that in-cluster
controller.

## Quick Start

### 1. Authenticate with GCP

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  file.googleapis.com \
  compute.googleapis.com
```

### 3. Initialize and Apply (Dev)

```bash
cd terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID

terraform init
terraform plan
terraform apply
```

### 4. Build and Publish Workload Images

The repository output is the complete Artifact Registry prefix. From the
development Terraform directory used above:

```bash
CARTRIDGE_REGISTRY="$(terraform output -raw docker_registry)"
CARTRIDGE_REGISTRY_HOST="${CARTRIDGE_REGISTRY%%/*}"
# A unique tag guarantees that GKE rolls out and pulls each build.
CARTRIDGE_IMAGE_TAG="$(git -C ../../.. rev-parse --short=12 HEAD)-$(date -u +%Y%m%d%H%M%S)"

gcloud auth configure-docker "$CARTRIDGE_REGISTRY_HOST"
cd ../../..

docker build --build-arg CARGO_FEATURES=s3 -f Dockerfile.alphazero \
  -t "$CARTRIDGE_REGISTRY/cartridge-alphazero:$CARTRIDGE_IMAGE_TAG" .
docker build --build-arg CARGO_FEATURES=s3 -f web/Dockerfile \
  -t "$CARTRIDGE_REGISTRY/cartridge-web:$CARTRIDGE_IMAGE_TAG" .
docker build -f web/frontend/Dockerfile \
  -t "$CARTRIDGE_REGISTRY/cartridge-frontend:$CARTRIDGE_IMAGE_TAG" web/frontend

docker push "$CARTRIDGE_REGISTRY/cartridge-alphazero:$CARTRIDGE_IMAGE_TAG"
docker push "$CARTRIDGE_REGISTRY/cartridge-web:$CARTRIDGE_IMAGE_TAG"
docker push "$CARTRIDGE_REGISTRY/cartridge-frontend:$CARTRIDGE_IMAGE_TAG"
```

The publishing identity needs
[Artifact Registry Writer](https://docs.cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#permissions).
GKE normally has same-project pull access; if workloads report
`ImagePullBackOff`, grant the cluster runtime identity Artifact Registry Reader
before retrying. Do not use a long-lived service-account key as an image pull
secret.

### 5. Deploy Workloads

The development overlay is GKE-specific: it defines a
[Filestore CSI class](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/filestore-csi-driver)
bound to Terraform's fixed `cartridge-dev-vpc` network because trainer and web
pods mount the same runtime-data claim. Its 100 GiB Basic HDD request requires
GKE 1.33 or newer. From the repository root:

```bash
# Get cluster credentials using the exact provisioned name, region, and project.
$(terraform -chdir=terraform/environments/dev output -raw get_credentials_command)

# Pin the overlay to the images published above before creating the immutable
# trainer Job pod template.
pushd k8s/overlays/dev
kustomize edit set image \
  cartridge-alphazero="$CARTRIDGE_REGISTRY/cartridge-alphazero:$CARTRIDGE_IMAGE_TAG" \
  cartridge-web="$CARTRIDGE_REGISTRY/cartridge-web:$CARTRIDGE_IMAGE_TAG" \
  cartridge-frontend="$CARTRIDGE_REGISTRY/cartridge-frontend:$CARTRIDGE_IMAGE_TAG"
popd

kubectl apply -k k8s/overlays/dev
kubectl wait -n cartridge --for=condition=complete job/trainer --timeout=24h
kubectl rollout status -n cartridge deployment/web
kubectl rollout status -n cartridge deployment/frontend
```

For production, repeat the workflow from `terraform/environments/prod`, choose
an immutable release tag, and apply `k8s/overlays/prod`. The checked-in
production image names are placeholders and must be replaced with the exact
published Artifact Registry coordinates during deployment. Production also
contains no default database or object-store credentials: provision the
`postgres-credentials` and `minio-credentials` Secrets described in
`k8s/README.md` before applying the overlay.

## Directory Structure

```
terraform/
├── environments/
│   ├── dev/                    # Development environment
│   │   ├── main.tf             # Module composition
│   │   ├── variables.tf        # Variable definitions
│   │   ├── outputs.tf          # Output values
│   │   └── terraform.tfvars    # Environment-specific values
│   └── prod/                   # Production environment
│       └── ...
└── modules/
    ├── networking/             # VPC, subnets, Cloud NAT
    ├── gke/                    # GKE Autopilot cluster
    └── artifact-registry/      # Container image repository
```

## Modules

### networking

Creates VPC network with:
- Primary subnet for GKE nodes
- Secondary ranges for pods and services
- Cloud NAT for egress traffic
- Private Google Access enabled

### gke

Creates GKE Autopilot cluster with:
- Private cluster (optional)
- VPC-native networking
- Cloud Logging and Monitoring

### artifact-registry

Creates:
- Artifact Registry for container images

## Workload Storage

Terraform emits only cluster and container-registry outputs. Applying the
checked-in Kustomize overlay creates PostgreSQL and private MinIO services,
their persistent volumes, and the configuration consumed by the synchronized
trainer Job and web service. See [`k8s/README.md`](../k8s/README.md) for the
exact runtime contract.

Terraform intentionally does not provision a model-artifact bucket. Cartridge2
implements filesystem storage and AWS-SDK S3 storage; the checked-in Kubernetes
manifests provide a private MinIO service for the latter. To use an external
object store instead, configure an AWS-SDK-compatible S3 endpoint, bucket, and
credentials in the workload manifests. Google Cloud Storage is not wired as a
model backend: these modules create neither HMAC credentials nor an adapter for
AWS request signing.

## Cost Estimates

### Development

| Resource | Spec | Monthly Cost (est.) |
|----------|------|---------------------|
| GKE Autopilot | ~4 vCPU, 8GB | $50-100 |
| Cloud NAT | 1 gateway | $32 |
| **Total** | | **~$80-140** |

### Production

| Resource | Spec | Monthly Cost (est.) |
|----------|------|---------------------|
| GKE Autopilot | ~16 vCPU, 32GB | $200-400 |
| Cloud NAT | 1 gateway | $32 |
| Load Balancer | 1 forwarding rule | $18 |
| **Total** | | **~$250-450** |

These estimates exclude persistent-volume charges. In particular, the GKE
production overlay dynamically provisions a 100 GiB Basic HDD Filestore volume
for shared profile data. They also exclude the cost of an independently managed
S3 service.

The GKE and Artifact Registry resources are created in the same project. If an
organization policy disables the usual default runtime permissions, explicitly
grant the cluster runtime identity `roles/artifactregistry.reader`; the human or
automation identity that publishes images needs `roles/artifactregistry.writer`.

## Cleanup

```bash
# Destroy all resources
cd terraform/environments/dev
terraform destroy

# Or just specific resources
terraform destroy -target=module.gke
```

### Terraform State

For team collaboration, configure remote state:

```hcl
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "cartridge/dev"
  }
}
```

The remote-state bucket is external to these modules and is unrelated to model
artifact storage. Create and secure it separately before enabling the backend.
