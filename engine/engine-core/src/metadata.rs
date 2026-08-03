//! Environment metadata and optional presentation profiles.
//!
//! The base environment contract deliberately contains no board, player-count,
//! tensor-shape, or legal-action assumptions.  Those facts live in the
//! optional [`BoardGameMetadata`] profile used by the board-game cartridge.

use serde::{Deserialize, Serialize};

use crate::legal_mask::{LegalMask, LegalMaskError};

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

/// Tensor layout exposed by the AlphaZero-compatible board observation.
///
/// This is nested under the optional board profile rather than the generic
/// environment metadata. Other cartridges are free to use unrelated codecs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardObservationMetadata {
    pub elements: usize,
    pub spatial_channels: usize,
    pub legal_actions_offset: usize,
    pub player_relative: bool,
}

/// Metadata required by board presentation and the board-game cartridge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardGameMetadata {
    pub width: usize,
    pub height: usize,
    pub action_count: usize,
    pub observation: BoardObservationMetadata,
    pub players: Vec<BoardPlayerMetadata>,
    pub renderer: BoardRenderer,
}

impl BoardGameMetadata {
    pub fn new(width: usize, height: usize, action_count: usize) -> Self {
        Self {
            width,
            height,
            action_count,
            observation: BoardObservationMetadata {
                elements: 0,
                spatial_channels: 0,
                legal_actions_offset: 0,
                player_relative: false,
            },
            players: vec![
                BoardPlayerMetadata::new("Player 1", "1"),
                BoardPlayerMetadata::new("Player 2", "2"),
            ],
            renderer: BoardRenderer::Grid,
        }
    }

    pub fn with_observation(
        mut self,
        elements: usize,
        spatial_channels: usize,
        legal_actions_offset: usize,
        player_relative: bool,
    ) -> Self {
        self.observation = BoardObservationMetadata {
            elements,
            spatial_channels,
            legal_actions_offset,
            player_relative,
        };
        self
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

    pub fn legal_mask_from_obs(&self, obs: &[u8]) -> Result<LegalMask, LegalMaskError> {
        LegalMask::from_obs(
            obs,
            self.observation.legal_actions_offset,
            self.action_count,
        )
    }

    pub fn extract_legal_moves(&self, obs: &[u8]) -> Result<Vec<usize>, LegalMaskError> {
        Ok(self.legal_mask_from_obs(obs)?.iter_ones().collect())
    }

    pub fn is_action_legal(&self, obs: &[u8], action: usize) -> Result<bool, LegalMaskError> {
        if action >= self.action_count {
            return Ok(false);
        }
        Ok(self.legal_mask_from_obs(obs)?.is_legal(action))
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
        BoardGameMetadata::new(3, 3, 9)
            .with_observation(29, 2, 18, false)
            .with_players(vec![
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
    fn board_profile_extracts_dynamic_legal_actions() {
        let board = board();
        let mut obs = vec![0u8; 29 * 4];
        for action in [1, 3, 5] {
            let offset = (18 + action) * 4;
            obs[offset..offset + 4].copy_from_slice(&1.0f32.to_le_bytes());
        }
        assert_eq!(board.extract_legal_moves(&obs).unwrap(), vec![1, 3, 5]);
        assert!(board.is_action_legal(&obs, 3).unwrap());
        assert!(!board.is_action_legal(&obs, 8).unwrap());
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
        let board = BoardGameMetadata::new(usize::MAX, 2, 1);
        assert_eq!(
            board.board_size(),
            Err(MetadataError::BoardDimensionsOverflow {
                width: usize::MAX,
                height: 2,
            })
        );
    }
}
