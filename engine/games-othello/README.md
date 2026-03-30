# Othello (Reversi)

8×8 board game where players place pieces to flip opponent's pieces. The player with the most pieces when neither player can move wins.

## Features

- Full Othello rules with flipping in all 8 directions
- Pass action when no legal moves available
- Game ends after two consecutive passes
- Winner determined by piece count

## Usage

```rust
use engine_core::EngineContext;
use games_othello::register_othello;

register_othello();
let mut ctx = EngineContext::new("othello").unwrap();
let reset = ctx.reset(42, &[]);
```

## Game Specification

| Property | Value |
|----------|-------|
| Board | 8×8 (64 cells), row-major order |
| Players | 2 (Black=1, White=2) |
| Actions | 65 discrete (0-63 board positions, 64=pass) |
| Observation | 195 floats |
| State size | 67 bytes |

### Observation Layout (195 floats)

| Range | Description |
|-------|-------------|
| 0-63 | Black piece positions (one-hot) |
| 64-127 | White piece positions (one-hot) |
| 128-192 | Legal moves (65 actions, one-hot) |
| 193-194 | Current player ([is_black, is_white]) |

### State Encoding (67 bytes)

| Offset | Size | Description |
|--------|------|-------------|
| 0-63 | 64 bytes | Board cells (0=empty, 1=Black, 2=White) |
| 64 | 1 byte | Current player (1=Black, 2=White) |
| 65 | 1 byte | Winner (0=none, 1=Black, 2=White, 3=draw) |
| 66 | 1 byte | Consecutive pass count (0-2) |

### Action Encoding (4 bytes)

u32 little-endian. Values 0-63 are board positions, 64 is pass.

### Rules

- A move must flip at least one opponent piece in any of 8 directions
- Pass (action 64) is legal only when no board moves are available
- Game ends when both players pass consecutively (pass count reaches 2)
- Winner is the player with more pieces; equal counts = draw

### Info Bits (u64)

| Bits | Description |
|------|-------------|
| 0-63 | Legal move mask (board positions) |
| 16-19 | Current player |
| 20-23 | Winner |
| 24-31 | Consecutive passes |

## Tests

25 tests covering game logic, encoding round-trips, legal move generation, flipping mechanics, and pass/endgame scenarios.

```bash
cargo test -p games-othello
```
