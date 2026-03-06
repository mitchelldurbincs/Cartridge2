# Helpers for Cartridge2 development
#
# PostgreSQL is required for the replay buffer.
# Set CARTRIDGE_STORAGE_POSTGRES_URL env var or use Docker Compose.

ENV_ID                ?= tictactoe
ACTOR_EPISODES        ?= 50
ACTOR_LOG_INTERVAL    ?= 10
TRAIN_STEPS           ?= 50
TRAIN_BATCH           ?= 32
TRAIN_DEVICE          ?= cpu

PYTHON                ?= python3
CARGO                 ?= cargo
NPM                   ?= npm
VENV                  := $(HOME)/venvs/cartridge2
VENV_PIP              := $(VENV)/bin/pip
VENV_PYTHON           := $(VENV)/bin/python

# AlphaZero training iterations (actor + trainer cycles)
ITERATIONS            ?= 5

.PHONY: data actor trainer web-backend frontend-dev full-loop train-loop trainer-install venv clean-data clean-models clean-all \
        actor-tictactoe actor-connect4 train-loop-tictactoe train-loop-connect4 postgres-up postgres-down

# Data directory setup
data:
	mkdir -p data data/models

# PostgreSQL management (convenience targets)
postgres-up:
	docker compose up postgres -d

postgres-down:
	docker compose down postgres

# Clean targets
clean-data:
	rm -f ./data/stats.json ./data/loop_stats.json ./data/eval_stats.json ./data/best_model.json
	# Note: PostgreSQL data persists in volume, use 'docker compose down -v' to clear

clean-models:
	rm -f ./data/models/*.onnx ./data/models/*.onnx.data ./data/models/*.pt

clean-all: clean-data clean-models
	rm -rf ./data/

# Virtual environment setup
$(VENV):
	$(PYTHON) -m venv $(VENV)

# Run self-play to populate replay buffer (requires PostgreSQL)
actor: data postgres-up
	cd actor && $(CARGO) run -- --env-id $(ENV_ID) --max-episodes $(ACTOR_EPISODES) --log-interval $(ACTOR_LOG_INTERVAL) --data-dir ../data

# Game-specific actor shortcuts
actor-tictactoe: data postgres-up
	$(MAKE) actor ENV_ID=tictactoe

actor-connect4: data postgres-up
	$(MAKE) actor ENV_ID=connect4

# Install trainer dependencies
trainer-install: $(VENV)
	$(VENV_PIP) install -e trainer/

# Train on replay buffer data (requires PostgreSQL)
trainer: data postgres-up
	$(VENV_PYTHON) -m trainer train --steps $(TRAIN_STEPS) --batch-size $(TRAIN_BATCH) --device $(TRAIN_DEVICE) --env-id $(ENV_ID)

# Start the Rust backend (Axum)
web-backend: data
	cd web && $(CARGO) run

# Start the Svelte frontend dev server
frontend-dev:
	cd web/frontend && $(NPM) install && $(NPM) run dev

# Convenience: run actor then trainer with small defaults
full-loop: actor trainer

# AlphaZero training loop: synchronized actor + trainer + evaluation
# Usage: make train-loop ITERATIONS=5 ACTOR_EPISODES=500 TRAIN_STEPS=1000
train-loop: data postgres-up
	@echo "Starting synchronized AlphaZero training loop..."
	$(VENV_PYTHON) -m trainer loop --iterations $(ITERATIONS) --episodes $(ACTOR_EPISODES) --steps $(TRAIN_STEPS) --device $(TRAIN_DEVICE) --env-id $(ENV_ID)

# Game-specific training loop shortcuts
train-loop-tictactoe: data postgres-up
	$(MAKE) train-loop ENV_ID=tictactoe

train-loop-connect4: data postgres-up
	$(MAKE) train-loop ENV_ID=connect4
