//! Legal-move calculation, production, and win conditions, ported from the
//! Go engine's `rules/legal_moves.go`, `production_manager.go`, and
//! `rules/win_conditions.go`.

use crate::action::{encode_move, valid_move_target};
use crate::board::{Tile, TileKind};
use crate::params::{
    BOARD_SIZE, CITY_PRODUCTION, GENERAL_PRODUCTION, NORMAL_GROW_INTERVAL, NORMAL_PRODUCTION,
    NUM_ACTIONS, WAIT_ACTION,
};

/// Write the legal-move plane for `player` into `out` (length
/// [`NUM_ACTIONS`], 1.0 = legal). Wait is always legal for a live player.
pub fn fill_legal_moves(tiles: &[Tile], player: u8, alive: bool, out: &mut [f32]) {
    debug_assert_eq!(out.len(), NUM_ACTIONS);
    out.fill(0.0);
    if !alive {
        return;
    }
    out[WAIT_ACTION as usize] = 1.0;
    for from in 0..BOARD_SIZE {
        if tiles[from].owner != player || tiles[from].army <= 1 {
            continue;
        }
        for dir in 0..4 {
            if valid_move_target(tiles, player, from, dir).is_some() {
                out[encode_move(from, dir) as usize] = 1.0;
            }
        }
    }
}

/// Whether a single action is legal for `player` (used by tests and the
/// step guard; mirrors [`fill_legal_moves`]).
pub fn is_action_legal(tiles: &[Tile], player: u8, action: u32) -> bool {
    match crate::action::decode_move(action) {
        Some(crate::action::Move::Wait) => true,
        Some(crate::action::Move::Step { from, dir }) => {
            valid_move_target(tiles, player, from, dir).is_some()
        }
        None => false,
    }
}

/// Apply end-of-round production for both players (Go
/// `ProcessTurnProduction`): generals and owned cities always produce;
/// normal owned tiles grow every [`NORMAL_GROW_INTERVAL`] rounds.
pub fn apply_production(tiles: &mut [Tile], round: u32) {
    let grow_normal = round.is_multiple_of(NORMAL_GROW_INTERVAL);
    for tile in tiles.iter_mut() {
        if tile.is_neutral() {
            continue;
        }
        match tile.kind {
            TileKind::General => tile.army += GENERAL_PRODUCTION,
            TileKind::City => tile.army += CITY_PRODUCTION,
            TileKind::Normal => {
                if grow_normal {
                    tile.army += NORMAL_PRODUCTION;
                }
            }
            TileKind::Mountain => {}
        }
    }
}

/// Winner determination for two players (Go `CheckGameOver` semantics):
/// returns 0 while both live, the surviving player's ID when one falls, or
/// 3 (draw) if somehow neither is alive.
pub fn check_winner(alive: [bool; 2]) -> u8 {
    match (alive[0], alive[1]) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

/// Adjudicate a game that reached the round cap: the player controlling
/// more tiles wins; total armies break ties; a truly even position is a
/// draw (3). Zero-sum by construction, unlike a both-players draw penalty,
/// which the MCTS backprop negation and the actor's parity outcome
/// backfill would corrupt.
pub fn adjudicate_at_cap(tiles: &[Tile]) -> u8 {
    let mut tile_count = [0u32; 2];
    let mut army_count = [0u64; 2];
    for tile in tiles {
        if tile.owner == 1 || tile.owner == 2 {
            let p = tile.owner as usize - 1;
            tile_count[p] += 1;
            army_count[p] += tile.army as u64;
        }
    }
    match tile_count[0].cmp(&tile_count[1]) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => 2,
        std::cmp::Ordering::Equal => match army_count[0].cmp(&army_count[1]) {
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Less => 2,
            std::cmp::Ordering::Equal => 3,
        },
    }
}
