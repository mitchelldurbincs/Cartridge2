# games-tictactoe

TicTacToe implementation for Cartridge2. Reference implementation of the
narrow `BoardGame` profile, adapted into the generic `Environment` ABI.

## Overview

A complete TicTacToe game with:
- 3x3 board, two players (X and O)
- Win detection (rows, columns, diagonals)
- Draw detection (full board)
- Neural network-friendly observation encoding

## Usage

```rust
use engine_core::{AgentId, EngineContext};
use games_tictactoe::register_tictactoe;

// Register the game
register_tictactoe();

// Create context
let mut ctx = EngineContext::new("tictactoe").unwrap();

// Start a new game
let reset = ctx.reset(42, &[]).unwrap();

// Make a move (center square = position 4)
let action = 4u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();

// Read the generic per-agent outcome and episode status.
println!("Episode: {:?}", step.timestep.episode);
println!("P1 reward: {:?}", step.timestep.reward_for(AgentId(1)));
```

## Game Specification

### State

- **Board**: 9 cells, each 0 (empty), 1 (X), or 2 (O)
- **Current player**: 1 (X) or 2 (O)
- **Winner**: 0 (none), 1 (X wins), 2 (O wins), 3 (draw)

### Actions

Integer 0-8 representing board position:

```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

### Observation

27 floats for neural network input:

| Indices | Description |
|---------|-------------|
| 0-8 | Current player's pieces (1.0 where present) |
| 9-17 | Opponent's pieces (1.0 where present) |
| 18-26 | Legal moves mask (1.0 where legal) |

### Rewards

- **+1.0**: Win
- **-1.0**: Loss
- **0.0**: Draw or game continues

### Board adapter info

The narrow `BoardGame` transition packs an internal `u64` whose lower nine bits
encode the legal move mask. The generic ABI exposes auxiliary data as opaque
`timestep.info` bytes; algorithms must use the observation's declared legal-mask
offset instead of depending on this layout.

- Bit N is set if position N is a legal move
- Example: `0b111111111` = all positions legal (empty board)

## Encoding

State is encoded as 11 bytes:
- Bytes 0-8: Board cells
- Byte 9: Current player
- Byte 10: Winner

Action is encoded as 4 bytes (little-endian u32).

## Testing

```bash
cargo test --manifest-path engine/Cargo.toml -p games-tictactoe
```

## Benchmarks

```bash
cargo bench --manifest-path engine/Cargo.toml -p games-tictactoe
```

Benchmarks include:
- Reset performance
- Step performance
- Full episode simulation
