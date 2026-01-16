# Secret Management Design

## Overview

Replace hardcoded Kubernetes secrets with proper secret management using:
- **GCP Secret Manager** for storing secrets
- **External Secrets Operator (ESO)** for syncing secrets to Kubernetes
- **Workload Identity** for Cloud SQL authentication (passwordless)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Terraform                                │
│  Creates secrets in GCP Secret Manager                          │
│  Configures Workload Identity for Cloud SQL                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │ creates
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GCP Secret Manager                             │
│  projects/cartridge-dev/secrets/minio-credentials               │
│  projects/cartridge-prod/secrets/minio-credentials              │
└─────────────────────┬───────────────────────────────────────────┘
                      │ syncs via
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              External Secrets Operator (ESO)                     │
│  Runs in GKE, watches ExternalSecret CRDs                       │
│  Uses Workload Identity to auth to Secret Manager               │
└─────────────────────┬───────────────────────────────────────────┘
                      │ creates
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Kubernetes Secrets                             │
│  minio-credentials (synced from Secret Manager)                 │
│  No postgres-credentials needed (Workload Identity)             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cloud SQL auth | Workload Identity + IAM | No passwords, no rotation needed |
| MinIO auth | External Secrets + Secret Manager | Learn full ESO pattern |
| Secret creation | Terraform | Full infrastructure-as-code flow |
| Environments | Both dev and prod | Learn proper environment separation |
| Terraform structure | New `secrets` module | Clean separation of concerns |

## Terraform Module Design

### New Module: `terraform/modules/secrets/`

```
terraform/modules/secrets/
├── main.tf          # Secret Manager resources
├── variables.tf     # Input variables
├── outputs.tf       # Outputs for other modules
└── iam.tf           # Workload Identity bindings for ESO
```

### Resources Created

1. **Enable Secret Manager API**
   - `google_project_service` for `secretmanager.googleapis.com`

2. **MinIO secret**
   - `google_secret_manager_secret` + `google_secret_manager_secret_version`
   - Stores: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`
   - One secret per environment (dev/prod)

3. **Service Account for ESO**
   - `google_service_account` named `external-secrets-sa`
   - Granted `roles/secretmanager.secretAccessor` on the MinIO secret

4. **Workload Identity binding**
   - Links K8s ServiceAccount `external-secrets` in namespace `external-secrets` to GCP service account

### Usage from Environment

```hcl
# terraform/environments/dev/main.tf
module "secrets" {
  source      = "../../modules/secrets"
  project_id  = var.project_id
  environment = "dev"

  minio_access_key = var.minio_access_key  # From tfvars (gitignored)
  minio_secret_key = var.minio_secret_key
}
```

Secret values live in `dev.tfvars` / `prod.tfvars` (gitignored).

## Kubernetes External Secrets Setup

### New Directory: `k8s/base/external-secrets/`

```
k8s/base/external-secrets/
├── namespace.yaml           # external-secrets namespace
├── cluster-secret-store.yaml # ClusterSecretStore pointing to GCP
└── external-secret.yaml     # ExternalSecret for MinIO creds
```

### ClusterSecretStore

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: gcp-secret-manager
spec:
  provider:
    gcpsm:
      projectID: cartridge-${environment}  # Patched per overlay
      auth:
        workloadIdentity:
          clusterLocation: us-central1
          clusterName: cartridge-cluster
          serviceAccountRef:
            name: external-secrets
            namespace: external-secrets
```

### ExternalSecret

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: minio-credentials
  namespace: cartridge
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: gcp-secret-manager
  target:
    name: minio-credentials  # K8s Secret name (same as before)
  data:
    - secretKey: AWS_ACCESS_KEY_ID
      remoteRef:
        key: minio-credentials
        property: access_key
    - secretKey: AWS_SECRET_ACCESS_KEY
      remoteRef:
        key: minio-credentials
        property: secret_key
    - secretKey: MINIO_ROOT_USER
      remoteRef:
        key: minio-credentials
        property: root_user
    - secretKey: MINIO_ROOT_PASSWORD
      remoteRef:
        key: minio-credentials
        property: root_password
```

Kustomize overlays patch `projectID` for dev vs prod.

## Cloud SQL Workload Identity Setup

### Terraform Changes (in `cloud-sql` module)

1. **Enable IAM authentication on Cloud SQL instance:**
```hcl
resource "google_sql_database_instance" "main" {
  database_flags {
    name  = "cloudsql.iam_authentication"
    value = "on"
  }
}
```

2. **IAM database user:**
```hcl
resource "google_sql_user" "app_user" {
  instance = google_sql_database_instance.main.name
  name     = google_service_account.app.email
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}
```

3. **Service account for app pods:**
```hcl
resource "google_service_account" "app" {
  account_id = "cartridge-app-${var.environment}"
}

resource "google_project_iam_member" "cloudsql_client" {
  role   = "roles/cloudsql.client"
  member = "serviceAccount:${google_service_account.app.email}"
}
```

4. **Workload Identity binding** for app pods (actor, trainer, web).

### Connection String

With Cloud SQL Auth Proxy sidecar:
```
host=/cloudsql/PROJECT:REGION:INSTANCE dbname=cartridge user=SA_EMAIL
```

## Changes to Existing Files

### Remove from `k8s/base/configmap.yaml`

- Delete `postgres-credentials` Secret (lines 51-65)
- Delete `minio-credentials` Secret (lines 66-81)
- Keep ConfigMap portion (non-sensitive config)

### Update Deployments

1. **Add Cloud SQL Auth Proxy sidecar** to actor, trainer, web:
```yaml
- name: cloud-sql-proxy
  image: gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.8.0
  args:
    - "--private-ip"
    - "--structured-logs"
    - "PROJECT:REGION:INSTANCE"
  securityContext:
    runAsNonRoot: true
```

2. **Update `POSTGRES_HOST`** in ConfigMap from `postgres` to `localhost`

3. **Remove `k8s/base/postgres/`** directory entirely

### Update `k8s/base/kustomization.yaml`

- Remove postgres resources
- Add external-secrets resources

## ESO Installation

External Secrets Operator installed via Helm (before app deployment):

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets \
  --create-namespace \
  --set installCRDs=true
```

## Implementation Order

### Phase 1: Terraform

1. Create `modules/secrets/` with Secret Manager resources + IAM
2. Update `modules/cloud-sql/` for IAM auth
3. Update `modules/iam/` with Workload Identity bindings
4. Apply to create GCP resources

### Phase 2: Kubernetes Manifests

1. Add `k8s/base/external-secrets/` directory
2. Update `configmap.yaml` (remove hardcoded secrets)
3. Add Cloud SQL proxy sidecar to deployments
4. Remove `k8s/base/postgres/` directory
5. Update kustomization files

### Phase 3: Deployment

1. Install ESO via Helm
2. Apply K8s manifests
3. ESO syncs secrets automatically

## Files to Gitignore

Add to `.gitignore`:
```
terraform/environments/**/terraform.tfvars
terraform/environments/**/*.auto.tfvars
```
