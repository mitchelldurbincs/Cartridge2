#!/usr/bin/env bash
#
# Overnight synchronized AlphaZero training run (Connect4 by default).
#
# Runs `python -m trainer loop`: each iteration clears the replay buffer,
# generates self-play episodes with the current model, trains on them, and
# (every eval-interval iterations) evaluates + scores moves against the
# bitbully perfect solver.
#
# The orchestrator auto-resumes from the last completed iteration, so if the
# run is interrupted you can just re-run this script to continue.
#
# Usage:
#   scripts/overnight-run.sh                 # foreground, logs to data/logs/
#   nohup scripts/overnight-run.sh &         # detached (survives logout)
#   tmux new -s train 'scripts/overnight-run.sh'   # or run inside tmux
#
# Override any tunable via env vars, e.g.:
#   DEVICE=cpu ITERATIONS=100 WANDB=true scripts/overnight-run.sh
#
set -euo pipefail

# --- Tunables (env overrides; defaults mirror config.toml) --------------------
ENV_ID="${ENV_ID:-connect4}"
ITERATIONS="${ITERATIONS:-400}"
EPISODES="${EPISODES:-500}"
STEPS="${STEPS:-400}"
DEVICE="${DEVICE:-cuda}"          # auto | cpu | cuda | mps
NUM_ACTORS="${NUM_ACTORS:-10}"
WANDB="${WANDB:-false}"           # true to log this run to Weights & Biases
POSTGRES_URL="${CARTRIDGE_STORAGE_POSTGRES_URL:-postgresql://cartridge:cartridge@localhost:5432/cartridge}"

# --- Locate repo root ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CARTRIDGE_STORAGE_POSTGRES_URL="${POSTGRES_URL}"

LOG_DIR="${REPO_ROOT}/data/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/overnight-${ENV_ID}-$(date +%Y%m%d-%H%M%S).log"

log() { printf '\033[1;34m[overnight]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[overnight] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Pre-flight checks --------------------------------------------------------
log "Pre-flight checks..."

# 1. PostgreSQL reachable (replay buffer). Parse host/port from the URL.
pg_hostport="${POSTGRES_URL#*@}"          # strip user:pass@
pg_hostport="${pg_hostport%%/*}"          # strip /dbname...
pg_host="${pg_hostport%%:*}"
pg_port="${pg_hostport##*:}"
[ "${pg_port}" = "${pg_host}" ] && pg_port=5432
if command -v pg_isready >/dev/null 2>&1; then
  pg_isready -h "${pg_host}" -p "${pg_port}" >/dev/null 2>&1 \
    || die "PostgreSQL not reachable at ${pg_host}:${pg_port}. Start it (e.g. 'docker compose up -d postgres')."
elif command -v python3 >/dev/null 2>&1; then
  python3 - "$pg_host" "$pg_port" <<'PY' || die "PostgreSQL not reachable. Start it (e.g. 'docker compose up -d postgres')."
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=5):
    pass
PY
else
  log "WARN: cannot verify PostgreSQL (no pg_isready/python3); continuing."
fi
log "PostgreSQL reachable at ${pg_host}:${pg_port}."

# 2. Trainer package importable.
python -c "import trainer" 2>/dev/null || {
  log "Installing trainer package (pip install -e trainer)..."
  pip install -e ./trainer
}

# 3. Actor binary built (release).
if [ ! -x "${REPO_ROOT}/actor/target/release/actor" ]; then
  log "Building actor (release)..."
  cargo build --release --manifest-path "${REPO_ROOT}/actor/Cargo.toml"
fi
log "Actor binary: ${REPO_ROOT}/actor/target/release/actor"

# 4. GPU sanity check when cuda requested.
if [ "${DEVICE}" = "cuda" ]; then
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    die "DEVICE=cuda but torch.cuda.is_available() is False. Fix CUDA/PyTorch, or re-run with DEVICE=auto (falls back to CPU)."
  fi
  log "CUDA available."
fi

# --- Launch -------------------------------------------------------------------
log "Starting overnight run:"
log "  game=${ENV_ID} iterations=${ITERATIONS} episodes=${EPISODES} steps=${STEPS}"
log "  device=${DEVICE} num_actors=${NUM_ACTORS} wandb=${WANDB}"
log "  log file: ${LOG_FILE}"
log "  (auto-resumes from last completed iteration if re-run)"

set -x
python -m trainer loop \
  --env-id "${ENV_ID}" \
  --iterations "${ITERATIONS}" \
  --episodes "${EPISODES}" \
  --steps "${STEPS}" \
  --device "${DEVICE}" \
  --num-actors "${NUM_ACTORS}" \
  --wandb-enabled "${WANDB}" \
  2>&1 | tee "${LOG_FILE}"
