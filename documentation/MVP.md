### Cartridge 2
Be able to visualize training of AlphaZero AND be able to play against it 
I think I can reuse the engine-rust and the actor from Cartridge

Big Game Goals:
* Tic tac toe (COMPLETE)
* Connect 4 (COMPLETE)
* Othello

Here is your rewritten design document, streamlined for the **Monolithic MVP** approach (v2).

### Cartridge 2 (Monolithic MVP)

Be able to visualize training of AlphaZero AND be able to play against it (without the DevOps headache).

I will reuse the engine-rust code, but refactor it from a Service into a Library.

**Big Game Goals:**

- Tic tac toe (COMPLETE)
- Connect 4 (COMPLETE)
- Othello

**I want to:**

- Monitor training locally (e.g., Python writes a `stats.json` that the frontend polls).
- Play the bots via a Web Portal (hosted by the Rust process).
- Have full control over training hyperparameters (Python).
- **Run everything on one machine** first (simplify the loop), then worry about K8s later.

**Components**

- **Shared Resources** - PostgreSQL replay buffer + filesystem/S3 model storage (`./data/models/`).
- **Rust Services** - Engine library + Actor (self-play) + Web server (play API/UI backend).
- **Python Trainer** - Handles the Neural Network training loop.

**MVP:**

- TicTacToe implementation (COMPLETE)
- Connect 4 implementation (COMPLETE)
- Web Interface to play against the current "best" model (COMPLETE)
- No K8s yet—just two terminal windows (one for Rust, one for Python).

[https://github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

**Later:**

- Tournament of alphazero
- Porting the Python Trainer to a GPU Cloud Node (communicating via S3 instead of local disk).

---

## Actor (The "Player")

**Status: COMPLETE**

Rust binary importing the Engine crate and ort (ONNX Runtime).
A long-running process that:

1. Watches `./data/models/latest.onnx` for updates (hot-reload).
2. Runs MCTS self-play loops using the Engine and ONNX model.
3. Saves completed games to PostgreSQL replay buffer.
4. Stores MCTS visit distributions as policy targets.
5. Backfills game outcomes to all positions in each episode.
    

## Engine (The "Rules")

**Status: COMPLETE**

Rust library crate. Refactored from gRPC Service -> pure Rust library.

TLDR: A pure library. Apply action a to state s -> returns new_state. No network I/O. Used directly by the Actor for maximum speed.

**API:**
```rust
use engine_core::EngineContext;
use games_tictactoe::register_tictactoe;

register_tictactoe();
let mut ctx = EngineContext::new("tictactoe").unwrap();
let reset = ctx.reset(seed, &[]).unwrap();
let step = ctx.step(&reset.state, &action).unwrap();
```

## Replay

**Status: COMPLETE**

PostgreSQL Database - A shared concurrent replay buffer for actor and trainer.

- **Actor (Rust):** Connects to PostgreSQL and inserts transitions with:
  - Game state and observations
  - MCTS policy distributions (visit counts normalized)
  - Game outcomes backfilled after episode completion

- **Learner (Python):**
    - **Sampling:** Fetches randomized batches of transitions with policy targets and game outcomes.
    - **Window Management:** Periodically cleans up old transitions to bound buffer size.
## Weights

**Status: COMPLETE**

A single file: `./data/models/latest.onnx`
- **Learner** exports this after every checkpoint interval.
- **Actor** hot-reloads this when the timestamp changes.
- Versioned checkpoints kept in `./data/models/model_step_NNNNNN.onnx`.
    

## Learner (The "Trainer")

**Status: COMPLETE**

Python training loop that trains the network on data from PostgreSQL and publishes versioned models safely.

**Process:**
1. **Data Loading (PostgreSQL):** Runs SQL queries to fetch randomized batches of transitions with MCTS policy targets and game outcomes.
2. **Training:** Updates the PyTorch model parameters θ to minimize the combined AlphaZero loss:
   - Policy: Cross-entropy with soft MCTS visit distribution targets
   - Value: MSE with propagated game outcomes
3. **Checkpointing (Garbage Collection):** Saves a historical snapshot every N steps (e.g., `model_step_001000.onnx`).
    - _Logic:_ Keep the last 10 checkpoints, and delete older ones.
4. **Atomic Publishing:** Prevents the "Half-Written Model" crash in Rust using **write-then-rename**:
    - Step A: Save to `temp_model.onnx`.
    - Step B: Execute `os.replace('temp_model.onnx', 'latest.onnx')`.
    - _Result:_ The Rust Actor only ever sees a fully written, valid file at `latest.onnx`.
5. **Telemetry:** Writes `stats.json` (loss, learning rate, etc.) for the Web Portal to poll.
6. **Evaluation:** Includes evaluator tool to measure model performance against random baseline.

---

# WEB

## Backend

Rust (The Actor).

The Actor process will run a lightweight HTTP server (e.g., Axum or Actix) to handle move requests from the browser.

## Frontend

Svelte + Typescript.
Talks directly to localhost:8080 (The Rust Actor).

## Messages

**None** (Replaced by Filesystem + HTTP).

## Containerization

Docker (Optional for MVP, but good for environment consistency).

## Orchestration

Manual / Shell Script

./start_training.sh -> launches the Rust binary and the Python script in parallel.

## Local K8s

**None** (Skipped for MVP).

## GitHub Actions

- cargo fmt, black, go fmt, golangci-lint

---

# Later
- **Remote Storage:** Swap local filesystem for S3/MinIO when moving to the cloud.
- **NATS:** Introduce NATS only when you need multiple Actor replicas to coordinate.
- **Embed the Actor:** Technically, the Actor already embeds the Engine in this v2 architecture!
