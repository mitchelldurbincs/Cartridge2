//! Environment metadata and optional presentation profiles.
//!
//! The base environment contract owns tensor shapes and action availability.
//! This module contains only human-facing metadata and optional presentation
//! profiles such as a board renderer.

use serde::{Deserialize, Serialize};

/// Display-oriented metadata shared by every environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentMetadata {
    /// Stable environment identifier. Must equal `Capabilities::id.env_id`.
    pub id: String,
    pub display_name: String,
    pub description: String,
    /// Present only when the environment exposes the board-game profile.
    pub board: Option<BoardGameMetadata>,
}

impl EnvironmentMetadata {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            description: String::new(),
            board: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_board(mut self, board: BoardGameMetadata) -> Self {
        self.board = Some(board);
        self
    }

    pub fn require_board(&self) -> Result<&BoardGameMetadata, MetadataError> {
        self.board
            .as_ref()
            .ok_or_else(|| MetadataError::MissingBoard {
                env_id: self.id.clone(),
            })
    }
}

/// Board renderer understood by the bundled web presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoardRenderer {
    Grid,
    DropColumn,
    Generals,
}

/// Human-facing information for one board-game seat.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardPlayerMetadata {
    pub name: String,
    pub symbol: String,
}

impl BoardPlayerMetadata {
    pub fn new(name: impl Into<String>, symbol: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            symbol: symbol.into(),
        }
    }
}

/// Metadata required only by board presentation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardGameMetadata {
    pub width: usize,
    pub height: usize,
    pub players: Vec<BoardPlayerMetadata>,
    pub renderer: BoardRenderer,
}

impl BoardGameMetadata {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            players: vec![
                BoardPlayerMetadata::new("Player 1", "1"),
                BoardPlayerMetadata::new("Player 2", "2"),
            ],
            renderer: BoardRenderer::Grid,
        }
    }

    pub fn with_players(mut self, players: Vec<BoardPlayerMetadata>) -> Self {
        self.players = players;
        self
    }

    pub fn with_renderer(mut self, renderer: BoardRenderer) -> Self {
        self.renderer = renderer;
        self
    }

    pub fn board_size(&self) -> Result<usize, MetadataError> {
        self.width
            .checked_mul(self.height)
            .ok_or(MetadataError::BoardDimensionsOverflow {
                width: self.width,
                height: self.height,
            })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MetadataError {
    #[error("environment '{env_id}' does not expose the board-game profile")]
    MissingBoard { env_id: String },
    #[error("board dimensions {width}x{height} overflow the platform size")]
    BoardDimensionsOverflow { width: usize, height: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> BoardGameMetadata {
        BoardGameMetadata::new(3, 3).with_players(vec![
            BoardPlayerMetadata::new("X", "X"),
            BoardPlayerMetadata::new("O", "O"),
        ])
    }

    #[test]
    fn generic_metadata_does_not_fabricate_a_board() {
        let metadata = EnvironmentMetadata::new("counter", "Counter");
        assert!(metadata.board.is_none());
        assert!(matches!(
            metadata.require_board(),
            Err(MetadataError::MissingBoard { .. })
        ));
    }

    #[test]
    fn metadata_round_trips_without_flattening_board_fields() {
        let metadata = EnvironmentMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_description("Get three in a row")
            .with_board(board());
        let value = serde_json::to_value(&metadata).unwrap();
        assert_eq!(value["id"], "tictactoe");
        assert_eq!(value["board"]["width"], 3);
        assert!(value.get("board_width").is_none());
        assert_eq!(
            serde_json::from_value::<EnvironmentMetadata>(value).unwrap(),
            metadata
        );
    }

    #[test]
    fn board_size_reports_dimension_overflow() {
        let board = BoardGameMetadata::new(usize::MAX, 2);
        assert_eq!(
            board.board_size(),
            Err(MetadataError::BoardDimensionsOverflow {
                width: usize::MAX,
                height: 2,
            })
        );
    }
}
