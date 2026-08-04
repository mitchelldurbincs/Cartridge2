# Cartridge2 Makefile
#
# Quick start (macOS Apple Silicon):
#   make setup        # one-time: install deps, create DB, build everything
#   make train        # run AlphaZero training loop
#
# Quick start (Linux / Docker):
#   docker compose up alphazero
#
# Common targets:
#   make setup        - One-time setup (postgres, trainer, actor build)
#   make train        - Run synchronized AlphaZero training loop
#   make play         - Start web server + frontend to play against model
#   make test         - Run all tests
#   make lint         - Run all linters
#   make clean        - Remove training artifacts

# --- Configuration (override with env vars or `make VAR=value`) ---

PYTHON           ?= python3
CARGO            ?= cargo
NPM              ?= npm
VENV_DIR         ?= .venv

# Detect OS for platform-specific defaults
UNAME            := $(shell uname -s)
ARCH             := $(shell uname -m)

# CoreML feature flag: auto-enable on Apple Silicon
ifeq ($(UNAME)-$(ARCH),Darwin-arm64)
  CARGO_FEATURES ?= --features coreml
else
  CARGO_FEATURES ?=
endif

# Training defaults (override via config.toml or env vars)
ITERATIONS       ?= 50
EPISODES         ?= 500
STEPS            ?= 400
ALGORITHM        ?= alphazero_board_v1
ENV_ID           ?= tictactoe
POSTGRES_URL     ?= postgresql://cartridge:cartridge@localhost:5432/cartridge

.PHONY: help setup setup-db setup-trainer setup-actor setup-frontend \
        train play web frontend \
        test test-engine test-actor test-web test-trainer \
        lint lint-rust lint-python lint-frontend \
        build build-actor build-eval build-web environment-manifest \
        clean clean-profile clean-all \
        db-start db-stop db-reset

# --- Help ---

help:
	@echo "Cartridge2 - Algorithm-Cartridge RL Platform"
	@echo ""
	@echo "Setup (run once):"
	@echo "  make setup          - Full setup: DB + trainer + actor + frontend"
	@echo "  make setup-db       - Create PostgreSQL database"
	@echo "  make setup-trainer  - Install Python trainer package"
	@echo "  make setup-actor    - Build actor binary (release)"
	@echo "  make setup-frontend - Install frontend npm packages"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Run AlphaZero training loop"
	@echo "  make train ENV_ID=connect4 ITERATIONS=100 EPISODES=1000 STEPS=800"
	@echo ""
	@echo "Play:"
	@echo "  make play           - Start web server + frontend dev server"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Run all linters"
	@echo "  make build          - Build all Rust binaries (release)"
	@echo "  make environment-manifest - Regenerate the strict environment catalog"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove the selected immutable runtime profile"
	@echo "  make db-reset       - Delete every replay collection for the selected profile"

# --- One-time setup ---

setup: setup-db setup-trainer setup-actor setup-frontend
	@echo ""
	@echo "Setup complete! Run 'make train' to start training."

setup-db:
	@echo "--- Setting up PostgreSQL ---"
ifeq ($(UNAME),Darwin)
	@brew list postgresql@16 >/dev/null 2>&1 || brew install postgresql@16
	@brew services start postgresql@16 2>/dev/null || true
	@sleep 1
endif
	@createdb cartridge 2>/dev/null || true
	@psql cartridge -c "DO \$$\$$ BEGIN \
		IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cartridge') THEN \
			CREATE ROLE cartridge WITH LOGIN PASSWORD 'cartridge'; \
		END IF; \
	END \$$\$$;" 2>/dev/null || true
	@psql cartridge -c "GRANT ALL PRIVILEGES ON DATABASE cartridge TO cartridge;" 2>/dev/null || true
	@psql cartridge -c "GRANT ALL ON SCHEMA public TO cartridge;" 2>/dev/null || true
	@echo "PostgreSQL ready."

setup-trainer:
	@echo "--- Installing trainer ---"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Created virtual environment at $(VENV_DIR)"; \
	fi
	$(VENV_DIR)/bin/pip install -e "trainer/.[dev]"

setup-actor: build-actor

setup-frontend:
	@echo "--- Installing frontend ---"
	cd web/frontend && $(NPM) install

# --- Build ---

build: build-actor build-eval build-web

# Regenerate the environment/algorithm manifest the Python trainer reads. The
# engine owns generic capabilities and optional presentation profiles; the
# algorithm catalog owns compatibility. `cargo test` rejects catalog drift.
environment-manifest:
	@echo "--- Regenerating environment manifest ---"
	$(CARGO) run -q --manifest-path engine/Cargo.toml --bin generate-environment-manifest

build-actor:
	@echo "--- Building actor (release) $(CARGO_FEATURES) ---"
	cd actor && $(CARGO) build --release $(CARGO_FEATURES)

# The evaluation binary the trainer shells out to for every eval. Without it
# the cartridge's `loop` command cannot gate promotions; see
# trainer/src/trainer/evaluator.py.
build-eval:
	@echo "--- Building cartridge-eval (release) ---"
	$(CARGO) build --release --manifest-path engine/Cargo.toml -p evaluator

build-web:
	@echo "--- Building web server (release) ---"
	cd web && $(CARGO) build --release

# --- Training ---

data:
	@mkdir -p data

train: data
	$(VENV_DIR)/bin/python -m trainer --algorithm $(ALGORITHM) loop \
		--env-id $(ENV_ID) \
		--iterations $(ITERATIONS) \
		--episodes $(EPISODES) \
		--steps $(STEPS)

# --- Play ---

play:
	@echo "Starting web server and frontend..."
	@echo "Open http://localhost:5173 in your browser"
	@$(MAKE) -j2 web frontend

web:
	$(CARGO) run --manifest-path web/Cargo.toml

frontend:
	$(NPM) --prefix web/frontend run dev

# --- Testing ---

test: test-engine test-actor test-web test-trainer

test-engine:
	$(CARGO) test --manifest-path engine/Cargo.toml

test-actor:
	$(CARGO) test --manifest-path actor/Cargo.toml --all-features

test-web:
	$(CARGO) test --manifest-path web/Cargo.toml

test-trainer:
	$(VENV_DIR)/bin/python -m pytest trainer/tests/ -v --tb=short

# --- Linting ---

lint: lint-rust lint-python

lint-rust:
	$(CARGO) fmt --all --check --manifest-path engine/Cargo.toml
	$(CARGO) fmt --check --manifest-path actor/Cargo.toml
	$(CARGO) fmt --check --manifest-path web/Cargo.toml
	$(CARGO) clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings
	$(CARGO) clippy --manifest-path engine/Cargo.toml -p model-watcher --all-targets --all-features -- -D warnings
	$(CARGO) clippy --manifest-path actor/Cargo.toml --all-targets --all-features -- -D warnings
	$(CARGO) clippy --manifest-path web/Cargo.toml --all-targets --all-features -- -D warnings

lint-python:
	$(VENV_DIR)/bin/python -m ruff check trainer/src/ trainer/tests/ trainer/smoke_test.py
	$(VENV_DIR)/bin/python -m ruff format --check trainer/src/ trainer/tests/ trainer/smoke_test.py

lint-frontend:
	cd web/frontend && $(NPM) run check

# --- Database ---

db-start:
ifeq ($(UNAME),Darwin)
	brew services start postgresql@16
else
	@echo "Start PostgreSQL with: sudo systemctl start postgresql"
endif

db-stop:
ifeq ($(UNAME),Darwin)
	brew services stop postgresql@16
else
	@echo "Stop PostgreSQL with: sudo systemctl stop postgresql"
endif

db-reset:
	@echo "Deleting every replay collection for $(ALGORITHM)/$(ENV_ID)..."
	@identity="$$( $(VENV_DIR)/bin/python -c \
		'from trainer.environment_catalog import get_algorithm_descriptor; from trainer.runtime_profile import resolve_runtime_profile; import sys; algorithm = get_algorithm_descriptor(sys.argv[1]); profile = resolve_runtime_profile(sys.argv[1], sys.argv[2]); print(f"{profile.env_contract_version}:{algorithm.components.experience_schema}")' \
		"$(ALGORITHM)" "$(ENV_ID)" )"; \
	env_contract_version="$${identity%%:*}"; \
	experience_schema="$${identity#*:}"; \
	psql "$(POSTGRES_URL)" -v ON_ERROR_STOP=1 \
		-v algorithm_id="$(ALGORITHM)" \
		-v env_id="$(ENV_ID)" \
		-v env_contract_version="$${env_contract_version}" \
		-v experience_schema="$${experience_schema}" \
		-c "DELETE FROM replay_records WHERE algorithm_id = :'algorithm_id' AND env_id = :'env_id' AND env_contract_version = :'env_contract_version'::BIGINT AND experience_schema = :'experience_schema';"

# --- Cleanup ---

clean: clean-profile

clean-profile:
	@profile_path="$$( $(VENV_DIR)/bin/python -c \
		'from trainer.central_config import get_config; from trainer.runtime_profile import resolve_runtime_profile; import sys; print(resolve_runtime_profile(sys.argv[1], sys.argv[2]).data_dir(get_config().data_root))' \
		"$(ALGORITHM)" "$(ENV_ID)" )"; \
	echo "Removing $${profile_path}"; \
	rm -rf -- "$${profile_path}"

clean-all:
	rm -rf data/
