# Phase 1: Port the Generals Engine to a Cartridge2 Game Crate

> **STATUS 2026-07-19: IMPLEMENTED, with revisions.** The `games-generals`
> crate exists, registers as `generals_8x8`, and passes 25 tests plus an
> MCTS integration test. The plan below was revised before implementation
> after review — the deltas that matter:
>
> 1. **Framework generalization was done FIRST, not deferred to Phase 2.**
>    `engine_core::LegalMask` (dynamic bitset) replaced every `u64` legal
>    mask; MCTS actions widened `u8 -> u32`; MCTS/actor/web read masks from
>    the obs at `legal_mask_offset` instead of `info_bits`. This also fixed
>    a live Othello bug: its 64-bit mask collided with the player/winner
>    fields in `info_bits`, and `legal_mask_bits()` overflowed at 65
>    actions. Search microbenchmarks regressed ~13% (heap mask vs Copy u64)
>    — irrelevant under ONNX inference; optimize LegalMask to an inline
>    small-mask enum only if profiling ever says so.
> 2. **The pending-action turn bridge was dropped.** The step() bridge below
>    (stash P1's move, resolve on P2's step) leaks the pending move into the
>    searcher's true state. The implemented ruleset is **strictly
>    alternating**: each ply resolves immediately; production and the
>    500-round draw clock tick after player 2's ply. This also keeps the
>    actor's depth-parity outcome backfill sound.
> 3. **Visibility was NOT ported** (the plan said "ported but dormant").
>    The fog variant needs observation history (recurrent policy / IS-MCTS),
>    not just state bitfields — it gets its own env id, obs schema version,
>    and algorithm when that phase starts. Nothing in v1 state encodes
>    visibility.
> 4. **Obs schema is versioned (`generals_obs:v1`) and player-relative**
>    (own/enemy channels relative to the player to act), not the Go
>    serializer's layout: 9 channels x 64 (own/enemy/neutral territory,
>    own/enemy log-armies, cities, mountains, generals +1/-1, turn
>    progress), then the 257 legal mask at offset 576, then the player
>    one-hot. Direction order is canonically up/right/down/left (the Go
>    rules calculator's order; the Go experience serializer disagrees with
>    its own rules package — do not copy encodings from there).
> 5. Trainer ResNet was already channel-generic; the real fix was
>    `replay_setup.py` dropping network config on the DB-metadata path.
>    A generals `GAME_CONFIGS` entry (input_channels=9) is still TODO,
>    along with the pure-Python evaluator game and a web renderer.

Goal: a pure-Rust `games-generals` crate implementing Cartridge2's `Game` trait,
ported from the Go engine in `GeneralsReinforcementLearning/internal/game/`.
At the end of Phase 1 the crate **builds, passes ported tests, and registers with
the engine** — but is not yet trainable, because MCTS/actor/trainer changes
(the 64-action mask ceiling, N-channel networks) are Phase 2.

Written 2026-07-03. Companion repos:
- Source: `../GeneralsReinforcementLearning` (Go 1.24)
- Destination: this repo, `engine/games-generals/`

---

## Scope decisions (locked for Phase 1)

| Decision | Choice | Rationale |
|---|---|---|
| Map size | **Fixed 8×8**, registered as `generals_8x8` | `GameMetadata.obs_size` is static; register more sizes later as separate env_ids (curriculum) |
| Players | 2 | Matches Cartridge2's two-player assumption |
| Fog of war | **Ported but disabled** — obs is full-information | Vanilla AlphaZero MCTS is unsound under imperfect info; keep the visibility bitfields in `State` so flipping fog on later is a config change, not a rewrite |
| Half-move (`MoveAll=false`) | **Dropped** | Halves action space (4 dirs, not 5); Generals-lite doesn't need it |
| Action space | `W*H*4 = 256` actions, index = `(y*W + x)*4 + dir` (dir: 0=up 1=right 2=down 3=left) | Same encoding as Go `internal/experience/serializer.go:201` and `rules/legal_moves.go` |
| Turn structure | **Sequential submission, atomic resolution** (see below) | Fits `step(state, action)` one-action-at-a-time without framework changes |
| Max turns | 500, then draw (`winner=3`) | Mirrors Go `max_turns` fix from 2026-07-02; bounds episode length for the replay buffer |
| Config | Plain consts in a `Params` struct — **no Viper equivalent** | Go's `constants.go` just reads config; Cartridge2 games are self-contained |

### Simultaneous moves → sequential submission

Generals resolves both players' moves in the same tick. Cartridge2's `step()`
takes one action. Bridge: the state holds a pending-action slot.

```text
step(P1 action) -> stash action in state.pending, flip current_player, done=false, reward=0
step(P2 action) -> resolve full turn:
                     apply P1 move, then P2 move (priority alternates by turn parity,
                     as in Go turn_processor.go), combat, captures, eliminations,
                     production, win check
```

MCTS will treat this as an alternating game — an approximation (P2 "sees" P1's
pending move in the true state). Acceptable for Generals-lite with fog off;
revisit in the fog phase. A `wait` no-op action (reserve index 256 → NUM_ACTIONS
= 257) so a player with no legal move never deadlocks the turn.

---

## Crate layout

```
engine/games-generals/
├── Cargo.toml            # deps: engine-core, rand_chacha (match games-othello)
└── src/
    ├── lib.rs            # Game trait impl, register_generals(), GameMetadata
    ├── params.rs         # Game balance consts (see table below)
    ├── board.rs          # Tile, Board, tile types, visibility bitfields
    ├── action.rs         # Move repr, encode/decode, validate, is_valid_move (hot path)
    ├── movement.rs       # ApplyMoveAction port: combat, CaptureDetails, ProcessCaptures
    ├── production.rs     # per-turn army growth
    ├── visibility.rs     # 3×3 visibility update (ported, dormant in Phase 1)
    ├── mapgen.rs         # procedural map gen from ChaCha20Rng (deterministic by seed)
    ├── rules.rs          # win conditions + legal action mask
    ├── obs.rs            # observation tensor encoding
    └── tests.rs          # ported Go tests + encode/decode round-trips
```

## Go → Rust port map

Non-test Go engine code is ~2,000 lines; expect **~1,200–1,500 lines of Rust**
(the event bus, zerolog, Viper, and state machine all drop out).

| Go source (`internal/game/`) | LOC | Rust target | Notes |
|---|---|---|---|
| `core/board.go` | 159 | `board.rs` | `Tile { owner: i8, army: u32, kind: TileKind, visible: u32, discovered: u32 }`; `Vec<Tile>` row-major |
| `core/coordinate.go` | 153 | (inline in `board.rs`) | Most of it is JSON serialization — drop; keep index↔(x,y) helpers |
| `core/action.go` | 139 | `action.rs` | Port `IsValidMove` (allocation-free) as the single validator; drop the legacy dual-field coords |
| `core/movement.go` | 118 | `movement.rs` | Combat + `CaptureDetails` + tile turnover on general capture — port 1:1, this is the heart of the game |
| `core/errors.go` | 79 | (thin enum) | Collapse to one `MoveError` enum; most callers just need bool legality |
| `engine.go` + `turn_processor.go` | 541 | `lib.rs` `step()` | Strip context/events/state-machine; keep turn ordering: moves → captures → eliminations → production → win check |
| `production_manager.go` | 114 | `production.rs` | ~30 lines in Rust once events/logging drop |
| `visibility.go` + `visibility_optimized.go` | 382 | `visibility.rs` | Port the *simple* full-recompute version only; incremental optimization deferred until profiling says so |
| `rules/legal_moves.go` | 66 | `rules.rs` | Returns `Vec<bool>` len 257 → written into obs at `legal_mask_offset` |
| `rules/win_conditions.go` | 63 | `rules.rs` | General captured / last player alive / max-turns draw |
| `mapgen/generator.go` | 260 | `mapgen.rs` | Seed from `reset()`'s `ChaCha20Rng` — free determinism win over Go |
| `stats.go`, `rendering.go`, `events/`, `states/`, `processor/` | ~900 | **not ported** | RL loop doesn't need them; Cartridge2's web UI replaces rendering |

### Game balance params (`params.rs`)

From Go config defaults (`constants.go` → `config/`):

```rust
pub struct Params {
    pub width: usize,            // 8
    pub height: usize,           // 8
    pub city_ratio: usize,       // 1 city per 20 tiles
    pub city_start_army: u32,    // 40
    pub min_general_spacing: u32,// 5 (Manhattan)
    pub general_production: u32, // 1 per turn
    pub city_production: u32,    // 1 per turn
    pub normal_production: u32,  // 1
    pub normal_grow_interval: u32, // every 25 turns
    pub max_turns: u32,          // 500
    pub fog_enabled: bool,       // false in Phase 1
}
```

---

## Game trait mapping

### `type State`

```rust
pub struct State {
    board: Vec<Tile>,          // W*H, row-major
    turn: u32,
    current_player: u8,        // 1 or 2 (whose submission we're waiting on)
    pending: Option<Move>,     // P1's move awaiting P2's (see turn bridging)
    players: [PlayerState; 2], // alive, general_idx, owned-tile count
    winner: u8,                // 0=ongoing, 1, 2, 3=draw
}
```

`encode_state`/`decode_state`: fixed-layout little-endian — per tile
`(owner: u8, army: u32, kind: u8, visible: u32, discovered: u32)` = 14 B × 64
tiles + header ≈ 910 B. Cheap enough; don't bit-pack until it shows up in a
profile.

### `type Action`

`u32`, range `0..=256` (256 = wait). `encode_action`/`decode_action` via
`u32::to_le_bytes` — identical to Othello (`games-othello/src/lib.rs`, uses
`decode_action_u32` from `game_utils`).

### `type Obs` — layout and sizes

Reuse the Go 9-channel design (`internal/experience/serializer.go:37`), flattened
f32, followed by the legal mask and player one-hot so `legal_mask_offset`
works exactly like existing games:

```text
[ channels: 9 × 64 = 576 f32 ]   visibility(all-1s in Phase 1), ownership,
                                  log-army, 4× tile-type one-hot, turn/max_turns,
                                  (channel 8 repurposed: owned-mask, since the
                                  action mask moves to the standard slot below)
[ legal mask: 257 f32 ]           legal_mask_offset = 576
[ player one-hot: 2 f32 ]
                                  obs_size = 576 + 257 + 2 = 835
```

### `GameMetadata`

```rust
GameMetadata {
    env_id: "generals_8x8", display_name: "Generals 8×8",
    board_width: 8, board_height: 8,
    num_actions: 257, obs_size: 835, legal_mask_offset: 576,
    player_count: 2, board_type: "grid",   // web UI needs a new "generals" type eventually
    ...
}
```

### `step()` return

`(obs, reward, done, info_bits)` — reward from `game_utils::calculate_reward`
(+1/−1/0) on terminal, else 0. **`info_bits: u64` cannot hold a 257-action
mask** — return 0 and rely on the in-obs mask. This is the documented Phase 2
framework touchpoint (see below), not a Phase 1 blocker: the crate's own tests
don't need MCTS.

---

## Test plan (port these Go suites)

- [ ] `core/movement_test.go` (504 lines) → combat outcomes, capture details, general-capture tile turnover. **Highest value — port first.**
- [ ] `core/board_test.go` → tile/board invariants, visibility bitfield ops
- [ ] `action_mask_test.go` → legal-mask correctness vs. brute-force validation
- [ ] `fog_of_war_test.go` → visibility module (dormant but must stay correct)
- [ ] `mapgen/generator_test.go` → spacing/city-ratio invariants + **same-seed ⇒ same-map determinism**
- [ ] New: encode/decode round-trips for State/Action/Obs (Cartridge2 house convention)
- [ ] New: full-game smoke test — two random-legal players from `reset(seed)`, game reaches a winner or the 500-turn draw, no panics, army/production invariants hold each turn

Verification: `cargo test -p games-generals`, `cargo clippy -- -D warnings`,
plus a differential spot-check — run 3–5 identical scripted move sequences in
the Go engine and the Rust port, diff board states turn by turn.

## Work checklist

- [ ] Scaffold `engine/games-generals` crate; add to workspace `Cargo.toml`
- [ ] `board.rs` + `action.rs` + `movement.rs` (+ ported tests) — the pure core, ~1–2 days
- [ ] `mapgen.rs` + `production.rs` + `rules.rs` (+ tests) — ~1 day
- [ ] `visibility.rs` port (simple version) + fog tests — ~0.5 day
- [ ] `lib.rs`: State/step/reset, sequential-submission turn bridge, obs encoding, metadata — ~1–2 days
- [ ] Register in `engine-games/src/lib.rs` (behind the crate registration pattern)
- [ ] Differential test vs. Go engine
- [ ] `cargo fmt` / `clippy` / full engine test suite green

**Estimate: 4–6 working days.**

## Explicitly deferred to Phase 2 (framework generalization)

Recorded here so Phase 1 doesn't silently absorb them:

1. **MCTS mask width** — `mcts/src/search.rs:28` et al. use `u64` legal masks; needs `Vec<u64>`/bitvec for 257 actions.
2. **Actor** — reads legal mask from `info_bits`/obs assuming ≤64 actions; switch to obs-slice mask for large action spaces.
3. **Trainer ResNet input** — `trainer/src/trainer/resnet.py` assumes 2-channel `(2,H,W)` input; needs channel count from `game_config.py` (generals = 9 channels + non-spatial tail).
4. **Trainer `GAME_CONFIGS` entry + pure-Python generals** for the evaluator.
5. **Web UI** — new `board_type: "generals"` renderer in `GenericBoard.svelte` (armies, cities, mountains, fog shading).
6. **Fog of war + IS-MCTS / PPO** — the research phase; everything above is plumbing.
