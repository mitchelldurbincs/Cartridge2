# games-connect4

Connect 4 implementation for Cartridge2. A complete two-player strategy game
using the narrow `BoardGame` profile and generic environment adapter.

## Overview

A complete Connect 4 game with:
- 7x6 board (7 columns, 6 rows)
- Two players (Red and Yellow)
- Win detection (horizontal, vertical, diagonal)
- Draw detection (full board)
- Neural network-friendly observation encoding

## Usage

```rust
use engine_core::{AgentId, EngineContext};
use games_connect4::register_connect4;

// Register the game
register_connect4();

// Create context
let mut ctx = EngineContext::new("connect4").unwrap();

// Start a new game
let reset = ctx.reset(42, &[]).unwrap();

// Make a move (drop in center column = column 3)
let action = 3u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();

// Read the generic per-agent outcome and episode status.
println!("Episode: {:?}", step.timestep.episode);
println!("P1 reward: {:?}", step.timestep.reward_for(AgentId(1)));
```

## Game Specification

### Board Layout

The board is stored in row-major order, with row 0 at the bottom:

```
Row 5: [35][36][37][38][39][40][41]  <- Top
Row 4: [28][29][30][31][32][33][34]
Row 3: [21][22][23][24][25][26][27]
Row 2: [14][15][16][17][18][19][20]
Row 1: [ 7][ 8][ 9][10][11][12][13]
Row 0: [ 0][ 1][ 2][ 3][ 4][ 5][ 6]  <- Bottom
        Col 0  1  2  3  4  5  6
```

### State

- **Board**: 42 cells, each 0 (empty), 1 (Red), or 2 (Yellow)
- **Current player**: 1 (Red) or 2 (Yellow)
- **Winner**: 0 (none), 1 (Red wins), 2 (Yellow wins), 3 (draw)
- **Column heights**: Tracks how many pieces in each column

### Actions

Integer 0-6 representing which column to drop a piece into.

### Observation

93 floats for neural network input:

| Indices | Description |
|---------|-------------|
| 0-41 | Red's pieces (1.0 where present) |
| 42-83 | Yellow's pieces (1.0 where present) |
| 84-90 | Legal moves mask (1.0 where column not full) |
| 91-92 | Current player indicator [is_Red, is_Yellow] |

### Rewards

- **+1.0**: Win
- **-1.0**: Loss
- **0.0**: Draw or game continues

### Board adapter info

The narrow `BoardGame` transition packs an internal `u64` with this layout. The
generic ABI exposes auxiliary data as opaque `timestep.info` bytes; algorithms
must use the observation's declared legal-mask offset instead of depending on
these bits.

- Bits 0-6: Legal move mask (7 columns)
- Bits 16-19: Current player (1 = Red, 2 = Yellow)
- Bits 20-23: Winner (0 = none, 1 = Red, 2 = Yellow, 3 = draw)
- Bits 24-31: Moves played so far (0-42)

## Encoding

State is encoded as 44 bytes:
- Bytes 0-41: Board cells
- Byte 42: Current player
- Byte 43: Winner

Action is encoded as 4 bytes (little-endian u32).

## Testing

```bash
cargo test --manifest-path engine/Cargo.toml -p games-connect4
```

20 tests covering:
- Initial state validation
- Legal move detection
- Piece dropping mechanics
- Horizontal/vertical/diagonal win detection
- Draw game detection
- State/action/observation encoding roundtrips
- Invalid state handling
- Random game invariants
