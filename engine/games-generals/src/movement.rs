//! Army movement and combat resolution, ported 1:1 from the Go engine's
//! `core/movement.go` (all-army moves only) plus the elimination/tile
//! turnover logic from `engine.go`.

use crate::board::{Tile, TileKind};

/// Details of a tile changing ownership, mirroring Go's `CaptureDetails`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capture {
    pub tile_idx: usize,
    pub kind: TileKind,
    pub capturing_player: u8,
    pub previous_owner: u8,
    pub previous_army: u32,
}

/// Apply a validated move of all-but-one army from `from` to `to`.
///
/// Combat rules (Go `ApplyMoveAction`):
/// - Moving onto an own tile consolidates armies.
/// - Attacking with more armies than the defender captures the tile; the
///   difference remains.
/// - Attacking with fewer or equal armies fails; the defender keeps the
///   tile and loses the attacking armies.
///
/// Returns capture details if the tile changed ownership.
pub fn apply_move(tiles: &mut [Tile], player: u8, from: usize, to: usize) -> Option<Capture> {
    let armies_to_move = tiles[from].army - 1;
    tiles[from].army = 1;

    let defender = tiles[to];
    if defender.owner == player {
        tiles[to].army += armies_to_move;
        return None;
    }

    if armies_to_move > defender.army {
        tiles[to].owner = player;
        tiles[to].army = armies_to_move - defender.army;
        Some(Capture {
            tile_idx: to,
            kind: defender.kind,
            capturing_player: player,
            previous_owner: defender.owner,
            previous_army: defender.army,
        })
    } else {
        tiles[to].army -= armies_to_move;
        None
    }
}

/// Transfer every tile of `eliminated` to `new_owner` (armies unchanged,
/// tile kinds unchanged — the captured general stays a general-kind tile),
/// mirroring Go `handleEliminationsAndTileTurnover`.
pub fn transfer_tiles(tiles: &mut [Tile], eliminated: u8, new_owner: u8) {
    for tile in tiles.iter_mut() {
        if tile.owner == eliminated {
            tile.owner = new_owner;
        }
    }
}
