# Engine

Rust workspace containing generic environment contracts, installed games,
algorithm descriptors, AlphaZero search, model loading, and evaluation.

## Crates

| Crate | Description |
|-------|-------------|
| `algorithm-core` | Canonical algorithm descriptors, compatibility checks, artifact identity, runtime profiles |
| `engine-core` | Generic Environment ABI, wire/semantic capabilities, validated type erasure, registry, EngineContext |
| `engine-config` | Strict cross-language configuration schema and loading |
| `engine-games` | Bundled registration and generated environment/algorithm manifest |
| `envs-counter` | Non-board, single-agent reference environment and generic-contract canary |
| `evaluator` | Algorithm-dispatched `cartridge-eval` binary |
| `games-tictactoe` | TicTacToe reference implementation |
| `games-connect4` | Connect 4 implementation |
| `games-othello` | Othello (Reversi) implementation |
| `games-generals` | Full-information Generals 8x8 implementation |
| `mcts` | Search component for `alphazero_board_v1` |
| `model-watcher` | Strict filesystem/S3 artifact hot reload |
| `metrics-common` | Shared Prometheus helpers |

## Quick Start

```bash
# Build all crates
cargo build --release --manifest-path engine/Cargo.toml

# Run all tests
cargo test --manifest-path engine/Cargo.toml

# Run with ONNX support
cargo build --release --manifest-path engine/Cargo.toml --features mcts/onnx
```

## Architecture

```
+------------------+     +-------------------+
|  games-tictactoe |---->|    engine-core    |
+------------------+     +-------------------+
                               ^
+------------------+           |
|  games-connect4  |-----------|
+------------------+           |
                               |
+------------------+           |
|  games-othello   |-----------|
+------------------+           |
                               |
+------------------+           |
|       mcts       |-----------+
+------------------+

+------------------+
|  model-watcher   |  (model hot-reload; uses mcts with the onnx feature)
+------------------+
```

The bundled board games depend on `engine-core` for the narrow `BoardGame`
profile and its adapter into the generic `Environment` ABI. MCTS uses
`EngineContext`, but its compatibility guard accepts only environments that
meet the `alphazero_board_v1` board-game contract.
`engine-games` bundles the game crates behind a single
`register_all_environments()` entry point.

## Workspace Dependencies

Shared dependencies are defined in the root `Cargo.toml`:

- `rand_chacha` / `rand` - Deterministic randomness
- `thiserror` / `anyhow` - Error handling
- `once_cell` - Lazy static initialization
- `serde` - Serialization
- `tracing` - Logging
- `criterion` / `proptest` - Testing and benchmarks

## Adding a New Environment

1. Create an `envs-{name}` crate (use `games-{name}` only when the domain is
   specifically a game)
2. Add to workspace members in `Cargo.toml`
3. Implement `Environment` directly, or implement `BoardGame` only when the
   environment genuinely belongs to the bundled deterministic two-seat board
   family
4. Register it with `register_environment::<YourEnvironment>()`, or
   `board_profile::register_board_game::<YourBoardGame>()` for the narrow board profile
5. Declare an immutable contract version, exact wire encodings, agents and
   action spaces, turn/information/planning/chance/transition/reward semantics,
   and optional presentation metadata
6. Write tests for game logic and strict encoding round trips
7. Regenerate the manifest with `make environment-manifest` from the repository root
   and inspect each algorithm compatibility report

See `envs-counter` for the generic contract and `games-tictactoe` for the
optional board profile.

## Testing

```bash
# All tests and doc-tests
cargo test --manifest-path engine/Cargo.toml

# Specific crate
cargo test --manifest-path engine/Cargo.toml -p algorithm-core
cargo test --manifest-path engine/Cargo.toml -p engine-core
cargo test --manifest-path engine/Cargo.toml -p engine-config
cargo test --manifest-path engine/Cargo.toml -p engine-games
cargo test --manifest-path engine/Cargo.toml -p games-tictactoe
cargo test --manifest-path engine/Cargo.toml -p games-connect4
cargo test --manifest-path engine/Cargo.toml -p games-othello
cargo test --manifest-path engine/Cargo.toml -p games-generals
cargo test --manifest-path engine/Cargo.toml -p mcts
cargo test --manifest-path engine/Cargo.toml -p model-watcher --all-features

# With output
cargo test --manifest-path engine/Cargo.toml -- --nocapture
```

## Benchmarks

```bash
# Run benchmarks
cargo bench --manifest-path engine/Cargo.toml -p games-tictactoe

# MCTS microbenchmarks (Criterion)
cargo bench --manifest-path engine/Cargo.toml -p mcts --bench mcts
```

Recent run (devcontainer, plotters backend) highlights:

- `mcts_search_simulations` (Uniform policy): `50` sims ~126 µs; `100` sims ~295 µs; `200` sims ~579 µs; `400` sims ~1.11 ms; `800` sims ~2.21 ms.
- `mcts_game_phases`: opening ~558 µs; midgame ~68 µs; near-terminal ~31 µs.
- `mcts_tree_ops`: allocate node ~14.9 µs; select child ~57 ns; backpropagate depth 5 ~56 ns; root policy ~107 ns; root policy (τ=0.5) ~222 ns.
- `mcts_configs`: training config ~593 µs; evaluation config ~567 µs; `c_puct` 0.5/1.25/2.5/4.0 around 559/572/546/554 µs respectively.
