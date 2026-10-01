# Single-host GPU pilot

This is a bounded **pipeline smoke**, not a learning-quality experiment. It runs
AlphaZero Connect4 for two committed iterations: 8 completed episodes and 4
learner updates each, two local actors, 8 MCTS simulations, and 4 evaluation
games per evaluation. The global target is 8 optimizer steps. Use a separate
experiment root; never point it at a valuable run or increase its frozen target.

## Portability and Terraform

The same trainer CLI, recipe, local Rust actors/evaluator, PostgreSQL replay and
filesystem checkpoint protocol run on a workstation or a VM. `--device` selects
CPU, native Apple MPS, or CUDA. Explicit unavailable accelerators fail before
collection/storage startup; only the ordinary CLI's `auto` mode permits fallback.
The pilot requires an explicit device and defaults to CUDA. GPU acceleration is
for the learner; Rust self-play/evaluation currently use CPU ONNX on Linux.

Existing `terraform/` provisions networking, GKE Autopilot and Artifact Registry.
It does not provision this single-VM pilot. No Terraform resources are changed or
applied here. A later small Compute Engine/disk module can own infrastructure
once project, zone, accelerator, retention, network and spending limits are
approved; it should launch this same image/config rather than alter RL logic.
No new orchestration platform is needed for local child processes.

Environment boundaries are image/driver compatibility, device selection, database
address/secret delivery, persistent mount paths, and the process/VM supervisor.
The application storage contract is **not** interchangeable across cloud APIs:
filesystem publication needs hard links, fsync, replace and locks; the S3 adapter
needs conditional writes. Use ext4/XFS on retained block storage. GCS is a backup
destination, not a live FUSE mount or an assumed S3-compatible writer.

## Build and prepare (no cloud provisioning)

From this revision, on a Linux amd64 builder with enough free disk for CUDA:

```sh
docker build --platform linux/amd64 -f Dockerfile.alphazero \
  --build-arg TORCH_ACCELERATOR=cu126 -t cartridge-alphazero:gpu-pilot .
```

The base image is pinned by digest; `trainer/constraints-runtime.txt` pins Torch
2.9.1 and the ONNX/export stack. The `cu126` wheel supplies CUDA 12.6 userspace
libraries; the host supplies the NVIDIA driver. The default image remains CPU.
The image build checks its Torch flavor, `pip check`, installed-package imports,
and a synthetic CPU backward/export/Rust-ONNX fixture. CI builds both variants;
these checks do **not** execute CUDA kernels. OS packages and remaining Python
transitives are not a complete reproducible lock: record the resulting immutable
image digest, package inventory and source SHA for the first GPU run.

Prerequisites on the authorized host:

- Linux amd64, supported NVIDIA GPU/driver and NVIDIA Container Toolkit configured
  for Docker. Verify both host `nvidia-smi` and container passthrough. CUDA minor
  compatibility has feature restrictions; pass the actual preflight below.
- Docker Engine and Compose v2 with GPU device reservations. Do not run the GPU
  Compose file on Docker Desktop for macOS; native MPS is a separate path below.
- An already-mounted retained POSIX disk. Select an absolute `PILOT_ROOT` there,
  with pre-existing `data/` (writable by UID/GID 1000) and `postgres/` (managed by
  the PostgreSQL image). Bind mounts refuse to create missing directories.
  Verify the mount with `findmnt -T "$PILOT_ROOT"`; an ordinary boot-disk directory
  is not evidence of persistence. Data-disk auto-delete must be disabled.
- An existing operator-managed PostgreSQL password file, readable by the trainer
  UID through Compose secrets. Do not put its value in a URL, command line,
  checked-in `.env`, or logs. Compose file-backed secrets use host permissions;
  provision appropriate ownership/group access without public readability.

Set paths and an image reference (these values are examples, not provisioning):

```sh
export PILOT_ROOT=/mnt/cartridge/pilot-001
export PILOT_POSTGRES_PASSWORD_FILE=/secure/cartridge/postgres-password
export PILOT_IMAGE=cartridge-alphazero:gpu-pilot
# On GCP use the tested Artifact Registry image@sha256:... instead of a tag.
docker compose --env-file /dev/null -f docker-compose.pilot.yml config --quiet
```

The standalone Compose file runs only PostgreSQL and one trainer. It publishes
no ports, uses an internal network, requests exactly one GPU, disables automatic
restarts and W&B, and mounts the exact revision's `config.defaults.toml` read-only.
Do not merge it with `docker-compose.yml`. Use the same source revision for the
image, schema and mounted defaults. Only one trainer may own this experiment;
the pilot also holds a filesystem lock across preflight and training.

## First authorized GPU run

First run the bounded hardware fixture without starting PostgreSQL or training:

```sh
docker compose --env-file /dev/null -f docker-compose.pilot.yml run --rm --no-deps trainer \
  --timeout-seconds 120 --grace-seconds 10 --log-file /app/data/preflight.log -- \
  python -m trainer.pilot --device cuda --data-dir /app/data/connect4-smoke --preflight-only
```

It requires CUDA, executes a small Connect4 forward/backward pass with finite
loss/gradient checks, exports ONNX and plays two synthetic Rust evaluation games.
Confirm the log identifies CUDA and the intended GPU. No replay DB is used.

Then, within a separately authorized GPU/spending window:

```sh
docker compose --env-file /dev/null -f docker-compose.pilot.yml up \
  --abort-on-container-exit --exit-code-from trainer
tail -n 80 "$PILOT_ROOT/data/pilot-smoke.log"
```

A separate Python supervisor bounds the entire trainer process group to 900
seconds plus at most 10 seconds of TERM grace. It KILLs remaining descendants
even when the loop exits first. Exit 124 means timeout; 130/143 mean interrupted,
not success. Docker `init: true` reaps orphaned workers. Current Rust processes
inherit their parent's process group; detaching future workers would require a
supervisor change. Do not invoke `trainer.pilot` unbounded for an operational run.
This is a userspace deadline, not protection from a stuck kernel/host failure.

The data mount retains the full profile authority, PostgreSQL files and appended
trainer logs. Inspect both service exit statuses; Compose stops the sibling when
a service exits. After interrupted Compose/client execution, explicitly stop the
project's services with `docker compose --env-file /dev/null -f
docker-compose.pilot.yml stop --timeout 15`. Do not use `down -v` or delete data.
Database initialization/health waiting precedes the trainer deadline: configure
a separate host/VM maximum runtime to cover that phase and all failure cases.

On GCP, the proposed topology is one private **on-demand** GPU VM, local PostgreSQL,
and a retained data disk; no GKE/MinIO/public web service is required. Approve the
project, zone/capacity/quota, machine/GPU/disk compatibility, image access, narrow
attached identity, private access path and total spending window before creating
anything. Configure Compute Engine maximum runtime with **STOP**, no automatic
restart, and retained data. This repository does not provision or stop the VM.
Stopping containers does not stop GPU VM charges, and retained disks/backups can
continue to cost money after VM stop. Use startup scripts/systemd/cloud-init,
not the deprecated Compute Engine container startup agent.

## Restart and restore

Restart with the same image, defaults, data mount, password and recipe. The
RunHead/RunCommit graph is the authority, never `stats.json`. The completed smoke
is a no-op for learning on restart; it must not create a third iteration. The
hardware preflight still runs. A new experiment requires a new root.

Interruption before the prepared-run journal loses the active iteration. A fresh
collection scope starts from the last committed checkpoint/optimizer/scheduler
step, excluding abandoned replay. Prepared work recovers idempotently through
evaluation/commit/head publication without repeating games. This does not claim
mid-iteration continuation, deterministic RNG restoration or Spot readiness.

Before valuable or unattended training, quiesce the loop and back up the entire
artifact authority (including preparations, manifests, blobs, evaluations and run
commits), exact config/source/image metadata, and a PostgreSQL-consistent dump or
snapshot. Never copy live PostgreSQL files as a backup. Upload an immutable backup
ID and checksum manifest to private GCS via native tooling and attached identity;
mark complete only after verifying every object. Restore to an empty retained
POSIX disk, verify digests/lineage and replay schema, then resume the unchanged
recipe. Backup automation and a real restore drill remain acceptance work.

## Native workstation fixture

With the trainer, pinned Crucible, matching Rust binaries and an isolated local
PostgreSQL database already installed, the same smoke can use `--device cpu` or
`--device mps`. Use the same canonical defaults as Compose, not a custom config:

```sh
export CARTRIDGE_CONFIG="$PWD/config.defaults.toml"
export CARTRIDGE_STORAGE_MODEL_BACKEND=filesystem
# Supply CARTRIDGE_STORAGE_POSTGRES_URL through your existing local secret setup.
python -m trainer.supervise --timeout-seconds 900 --grace-seconds 10 -- \
  python -m trainer.pilot --device mps --data-dir /absolute/new/pilot-smoke
```

## Acceptance evidence still required on real hardware

- CUDA kernel, backward and Python-to-Rust ONNX fixture passes in the exact image.
- Two linked committed iterations, step 8, finite losses, exact episode seals and
  correct latest/champion evidence; record GPU/CPU memory, phase timing and disk.
- SIGTERM/forced interruption in collection, learning and evaluation; no surviving
  children; only committed/prepared work recovers; completed-target restart adds
  no updates. Repeat with real PostgreSQL/container and VM restarts.
- Retained-disk remount and full backup restore verification, plus observed VM
  stop at its deadline. Four-game promotion results do not establish strength.

Local tests use CPU tensors, temporary artifact trees, fake replay stores and
short real subprocess fixtures. They prove selection/restart logic and process
cleanup, not CUDA, real DB durability, GCP IAM/networking, GPU performance or costs.

Official references: [PyTorch versions](https://pytorch.org/get-started/previous-versions/),
[NVIDIA Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/),
[VM runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime),
[GPU maintenance](https://docs.cloud.google.com/compute/docs/gpus/gpu-host-maintenance),
[container agent migration](https://docs.cloud.google.com/compute/docs/containers/prepare-for-container-agent-shutdown),
[GCS FUSE limitations](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/overview),
[GCS write preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions),
[Terraform's infrastructure role](https://developer.hashicorp.com/terraform/intro).
