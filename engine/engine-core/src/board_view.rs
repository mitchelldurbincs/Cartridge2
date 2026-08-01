//! Display projection of a game state.
//!
//! Consumers outside the engine — the web server, the evaluation harness'
//! position dump — need to know what is on the board. Before this existed
//! they read the raw bytes of `encode_state` and assumed a layout
//! (`[board][current_player][winner]`), which is a private serialization
//! format: it silently excluded Generals (12-byte header + 6-byte tiles) and
//! would have broken any future game that stores more than one byte per cell.
//!
//! [`BoardView`] is the engine's answer to "what does this state look like",
//! so no caller has to decode state bytes itself. It is a *projection*, not a
//! state: it carries what a renderer or a scorer needs, not enough to resume
//! play.

use serde::{Deserialize, Serialize};

/// Terrain or structure occupying a cell.
///
/// The union of what the bundled games need, not an open extension point —
/// add a variant when a game needs one. Games with a uniform board (TicTacToe,
/// Connect 4, Othello) report [`CellKind::Normal`] everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellKind {
    /// Plain, playable cell.
    #[default]
    Normal,
    /// A player's home tile; capturing it wins (Generals).
    General,
    /// A capturable strongpoint (Generals).
    City,
    /// Impassable terrain (Generals).
    Mountain,
}

/// One cell as a renderer sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CellView {
    /// 0 = empty or neutral, 1/2 = owning player.
    pub owner: u8,
    /// Terrain. [`CellKind::Normal`] for games without terrain.
    pub kind: CellKind,
    /// Game-defined quantity displayed on the cell (Generals' army count).
    /// 0 for games that have no per-cell quantity.
    pub value: u32,
}

/// A game state projected for display.
///
/// `cells` is row-major and always `board_width * board_height` long, so a
/// caller can index it with the same `y * width + x` it uses for actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardView {
    /// Row-major cells, `board_width * board_height` of them.
    pub cells: Vec<CellView>,
    /// Player to act: 1 or 2.
    pub current_player: u8,
    /// 0 = ongoing, 1/2 = winner, 3 = draw.
    pub winner: u8,
}

impl BoardView {
    /// Build a view for a game whose board is one owner byte per cell.
    ///
    /// Covers every game with a uniform board; Generals builds its cells
    /// directly because it also carries terrain and armies.
    pub fn from_owners(owners: &[u8], current_player: u8, winner: u8) -> Self {
        Self {
            cells: owners
                .iter()
                .map(|&owner| CellView {
                    owner,
                    ..Default::default()
                })
                .collect(),
            current_player,
            winner,
        }
    }

    /// The owner byte of every cell, row-major.
    ///
    /// The inverse of [`from_owners`](Self::from_owners), for consumers that
    /// want the flat board a uniform game would have encoded.
    pub fn owners(&self) -> Vec<u8> {
        self.cells.iter().map(|cell| cell.owner).collect()
    }

    /// Whether the game has ended.
    pub fn game_over(&self) -> bool {
        self.winner != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_owners_round_trips_through_owners() {
        let board = [0u8, 1, 2, 1, 0, 0, 2, 2, 1];
        let view = BoardView::from_owners(&board, 2, 0);

        assert_eq!(view.owners(), board);
        assert_eq!(view.current_player, 2);
        assert!(!view.game_over());
    }

    #[test]
    fn from_owners_leaves_terrain_and_value_empty() {
        let view = BoardView::from_owners(&[1, 2], 1, 0);

        for cell in &view.cells {
            assert_eq!(cell.kind, CellKind::Normal);
            assert_eq!(cell.value, 0);
        }
    }

    #[test]
    fn game_over_follows_the_winner_byte() {
        assert!(!BoardView::from_owners(&[0], 1, 0).game_over());
        assert!(BoardView::from_owners(&[0], 1, 1).game_over());
        assert!(BoardView::from_owners(&[0], 1, 3).game_over());
    }

    #[test]
    fn cell_kind_serializes_as_snake_case() {
        // The frontend switches on these strings.
        let json = serde_json::to_string(&CellKind::Mountain).unwrap();
        assert_eq!(json, "\"mountain\"");
        let json = serde_json::to_string(&CellKind::Normal).unwrap();
        assert_eq!(json, "\"normal\"");
    }
}
