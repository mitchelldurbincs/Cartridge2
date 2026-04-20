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

.PHONY: help setup setup-db setup-trainer setup-actor setup-frontend \
        train play web frontend \
        test test-engine test-actor test-web test-trainer \
        lint lint-rust lint-python lint-frontend \
        build build-actor build-web \
        clean clean-data clean-models clean-all \
        db-start db-stop db-reset

# --- Help ---

help:
	@echo "Cartridge2 - AlphaZero Training Platform"
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
	@echo "  make train ITERATIONS=100 EPISODES=1000 STEPS=800"
	@echo ""
	@echo "Play:"
	@echo "  make play           - Start web server + frontend dev server"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Run all linters"
	@echo "  make build          - Build all Rust binaries (release)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove training data and models"
	@echo "  make db-reset       - Clear replay buffer in PostgreSQL"

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

build: build-actor build-web

build-actor:
	@echo "--- Building actor (release) $(CARGO_FEATURES) ---"
	cd actor && $(CARGO) build --release $(CARGO_FEATURES)

build-web:
	@echo "--- Building web server (release) ---"
	cd web && $(CARGO) build --release

# --- Training ---

data:
	@mkdir -p data data/models

train: data
	$(VENV_DIR)/bin/python -m trainer loop \
		--iterations $(ITERATIONS) \
		--episodes $(EPISODES) \
		--steps $(STEPS)

# --- Play ---

play:
	@echo "Starting web server and frontend..."
	@echo "Open http://localhost:5173 in your browser"
	@$(MAKE) -j2 web frontend

web:
	cd web && $(CARGO) run

frontend:
	cd web/frontend && $(NPM) run dev

# --- Testing ---

test: test-engine test-actor test-web test-trainer

test-engine:
	$(CARGO) test --manifest-path engine/Cargo.toml

test-actor:
	$(CARGO) test --manifest-path actor/Cargo.toml

test-web:
	$(CARGO) test --manifest-path web/Cargo.toml

test-trainer:
	$(VENV_DIR)/bin/python -m pytest trainer/tests/ -v --tb=short

# --- Linting ---

lint: lint-rust lint-python

lint-rust:
	$(CARGO) fmt --check --manifest-path engine/Cargo.toml
	$(CARGO) fmt --check --manifest-path actor/Cargo.toml
	$(CARGO) fmt --check --manifest-path web/Cargo.toml
	$(CARGO) clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings
	$(CARGO) clippy --manifest-path actor/Cargo.toml --all-targets -- -D warnings
	$(CARGO) clippy --manifest-path web/Cargo.toml --all-targets -- -D warnings

lint-python:
	$(VENV_DIR)/bin/python -m ruff check trainer/src/
	$(VENV_DIR)/bin/python -m black --check trainer/src/

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
	@echo "Clearing replay buffer..."
	psql postgresql://cartridge:cartridge@localhost:5432/cartridge \
		-c "DELETE FROM transitions;" 2>/dev/null || true
	@echo "Replay buffer cleared."

# --- Cleanup ---

clean: clean-data clean-models

clean-data:
	rm -f data/stats.json data/loop_stats.json data/eval_stats.json data/best_model.json

clean-models:
	rm -f data/models/*.onnx data/models/*.onnx.data data/models/*.pt

clean-all: clean-data clean-models
	rm -rf data/
