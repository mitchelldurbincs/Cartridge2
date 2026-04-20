//! Game metadata for UI and configuration
//!
//! This module provides display-oriented metadata about games that can be
//! used by frontends, actors, and trainers to configure themselves dynamically.

use serde::{Deserialize, Serialize};

/// Metadata about a game for UI display and configuration
///
/// This struct contains all the information needed to:
/// - Display the game in a UI (board dimensions, player symbols)
/// - Configure actors and trainers (obs_size, num_actions)
/// - Parse observations correctly (legal_mask_offset)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameMetadata {
    /// Environment identifier (e.g., "tictactoe", "connect4")
    pub env_id: String,

    /// Human-readable display name (e.g., "Tic-Tac-Toe", "Connect 4")
    pub display_name: String,

    /// Board width in cells
    pub board_width: usize,

    /// Board height in cells
    pub board_height: usize,

    /// Number of possible actions
    pub num_actions: usize,

    /// Size of observation vector (number of f32 values)
    pub obs_size: usize,

    /// Offset in observation where legal moves mask starts
    /// (index of first legal move indicator in the obs array)
    pub legal_mask_offset: usize,

    /// Number of players (typically 2)
    pub player_count: usize,

    /// Display names for each player (e.g., ["X", "O"] or ["Black", "White"])
    pub player_names: Vec<String>,

    /// Single-character symbols for each player (e.g., ['X', 'O'] or ['B', 'W'])
    pub player_symbols: Vec<char>,

    /// Brief description of the game rules for UI tooltips
    pub description: String,

    /// Board rendering type for the frontend
    /// - "grid": Simple grid where clicks place pieces directly (TicTacToe, Othello)
    /// - "drop_column": Column-based where pieces drop to bottom (Connect 4)
    pub board_type: String,
}

impl GameMetadata {
    /// Create a new GameMetadata with required fields
    pub fn new(env_id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            env_id: env_id.into(),
            display_name: display_name.into(),
            board_width: 0,
            board_height: 0,
            num_actions: 0,
            obs_size: 0,
            legal_mask_offset: 0,
            player_count: 2,
            player_names: vec!["Player 1".to_string(), "Player 2".to_string()],
            player_symbols: vec!['1', '2'],
            description: String::new(),
            board_type: "grid".to_string(),
        }
    }

    /// Builder method for board dimensions
    pub fn with_board(mut self, width: usize, height: usize) -> Self {
        self.board_width = width;
        self.board_height = height;
        self
    }

    /// Builder method for action count
    pub fn with_actions(mut self, num_actions: usize) -> Self {
        self.num_actions = num_actions;
        self
    }

    /// Builder method for observation size and legal mask offset
    pub fn with_observation(mut self, obs_size: usize, legal_mask_offset: usize) -> Self {
        self.obs_size = obs_size;
        self.legal_mask_offset = legal_mask_offset;
        self
    }

    /// Builder method for player information
    pub fn with_players(mut self, count: usize, names: Vec<String>, symbols: Vec<char>) -> Self {
        self.player_count = count;
        self.player_names = names;
        self.player_symbols = symbols;
        self
    }

    /// Builder method for description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Builder method for board type
    /// - "grid": Simple grid where clicks place pieces directly (TicTacToe, Othello)
    /// - "drop_column": Column-based where pieces drop to bottom (Connect 4)
    pub fn with_board_type(mut self, board_type: impl Into<String>) -> Self {
        self.board_type = board_type.into();
        self
    }

    /// Get the total number of board cells
    pub fn board_size(&self) -> usize {
        self.board_width * self.board_height
    }

    /// Create a bitmask for extracting legal moves from info bits
    /// based on num_actions
    pub fn legal_mask_bits(&self) -> u64 {
        (1u64 << self.num_actions) - 1
    }

    /// Extract a legal moves mask from the observation bytes.
    ///
    /// The observation contains f32 values where indices starting at `legal_mask_offset`
    /// are the legal moves (1.0 = legal, 0.0 = illegal). This function extracts
    /// those values and packs them into a u64 bitmask.
    ///
    /// # Arguments
    /// * `obs` - The observation byte buffer (f32 values in little-endian format)
    ///
    /// # Returns
    /// A u64 bitmask where bit i is set if action i is legal.
    /// Returns a full mask (all actions legal) if the observation is too short.
    pub fn extract_legal_mask(&self, obs: &[u8]) -> u64 {
        let legal_start_byte = self.legal_mask_offset * 4;
        let legal_end_byte = legal_start_byte + self.num_actions * 4;

        if obs.len() < legal_end_byte {
            // Return fallback mask if observation is too short
            return self.legal_mask_bits();
        }

        let mut mask = 0u64;
        for i in 0..self.num_actions {
            let byte_offset = legal_start_byte + i * 4;
            let value = f32::from_le_bytes([
                obs[byte_offset],
                obs[byte_offset + 1],
                obs[byte_offset + 2],
                obs[byte_offset + 3],
            ]);
            if value > 0.5 {
                mask |= 1u64 << i;
            }
        }
        mask
    }

    /// Extract legal moves as a vector of action indices.
    ///
    /// Convenience method that returns legal action indices instead of a bitmask.
    ///
    /// # Arguments
    /// * `obs` - The observation byte buffer (f32 values in little-endian format)
    ///
    /// # Returns
    /// A vector of action indices that are legal.
    pub fn extract_legal_moves(&self, obs: &[u8]) -> Vec<usize> {
        // For games with >64 actions (like Othello with 65), we can't use u64 bitmask
        // without overflow. Read directly from observation array instead.
        (0..self.num_actions)
            .filter(|&i| self.is_action_legal(obs, i))
            .collect()
    }

    /// Check if a specific action is legal based on observation bytes.
    ///
    /// # Arguments
    /// * `obs` - The observation byte buffer
    /// * `action` - The action index to check
    ///
    /// # Returns
    /// `true` if the action is legal, `false` otherwise.
    pub fn is_action_legal(&self, obs: &[u8], action: usize) -> bool {
        if action >= self.num_actions {
            return false;
        }

        let byte_offset = (self.legal_mask_offset + action) * 4;
        if byte_offset + 4 > obs.len() {
            return false;
        }

        let value = f32::from_le_bytes([
            obs[byte_offset],
            obs[byte_offset + 1],
            obs[byte_offset + 2],
            obs[byte_offset + 3],
        ]);
        value > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builder() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_board(3, 3)
            .with_actions(9)
            .with_observation(29, 18)
            .with_players(2, vec!["X".to_string(), "O".to_string()], vec!['X', 'O'])
            .with_description("Get three in a row to win!");

        assert_eq!(meta.env_id, "tictactoe");
        assert_eq!(meta.display_name, "Tic-Tac-Toe");
        assert_eq!(meta.board_width, 3);
        assert_eq!(meta.board_height, 3);
        assert_eq!(meta.num_actions, 9);
        assert_eq!(meta.obs_size, 29);
        assert_eq!(meta.legal_mask_offset, 18);
        assert_eq!(meta.player_count, 2);
        assert_eq!(meta.player_names, vec!["X", "O"]);
        assert_eq!(meta.player_symbols, vec!['X', 'O']);
        assert_eq!(meta.description, "Get three in a row to win!");
    }

    #[test]
    fn test_board_size() {
        let meta = GameMetadata::new("test", "Test").with_board(7, 6);
        assert_eq!(meta.board_size(), 42);
    }

    #[test]
    fn test_legal_mask_bits() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe").with_actions(9);
        assert_eq!(meta.legal_mask_bits(), 0x1FF);

        let meta = GameMetadata::new("connect4", "Connect 4").with_actions(7);
        assert_eq!(meta.legal_mask_bits(), 0x7F);
    }

    #[test]
    fn test_serialization() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_board(3, 3)
            .with_actions(9);

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: GameMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(meta, parsed);
    }

    #[test]
    fn test_extract_legal_mask() {
        // TicTacToe: 9 actions, legal_mask at offset 18
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_actions(9)
            .with_observation(29, 18);

        // Create observation with legal moves at positions 0, 2, 4, 6, 8 (odd positions illegal)
        let mut obs = vec![0u8; 29 * 4];
        for i in 0..9 {
            let byte_offset = (18 + i) * 4;
            let value: f32 = if i % 2 == 0 { 1.0 } else { 0.0 };
            let bytes = value.to_le_bytes();
            obs[byte_offset..byte_offset + 4].copy_from_slice(&bytes);
        }

        let mask = meta.extract_legal_mask(&obs);
        // Expected: bits 0, 2, 4, 6, 8 set = 0b101010101 = 0x155
        assert_eq!(mask, 0x155);
    }

    #[test]
    fn test_extract_legal_mask_all_legal() {
        let meta = GameMetadata::new("connect4", "Connect 4")
            .with_actions(7)
            .with_observation(93, 84);

        // Create observation with all moves legal
        let mut obs = vec![0u8; 93 * 4];
        for i in 0..7 {
            let byte_offset = (84 + i) * 4;
            let bytes = 1.0f32.to_le_bytes();
            obs[byte_offset..byte_offset + 4].copy_from_slice(&bytes);
        }

        let mask = meta.extract_legal_mask(&obs);
        assert_eq!(mask, 0x7F); // All 7 bits set
    }

    #[test]
    fn test_extract_legal_mask_short_obs() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_actions(9)
            .with_observation(29, 18);

        // Observation too short - should return fallback (all legal)
        let obs = vec![0u8; 10];
        let mask = meta.extract_legal_mask(&obs);
        assert_eq!(mask, 0x1FF); // Fallback: all 9 bits set
    }

    #[test]
    fn test_extract_legal_moves() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_actions(9)
            .with_observation(29, 18);

        // Create observation with legal moves at positions 1, 3, 5
        let mut obs = vec![0u8; 29 * 4];
        for i in [1, 3, 5] {
            let byte_offset = (18 + i) * 4;
            let bytes = 1.0f32.to_le_bytes();
            obs[byte_offset..byte_offset + 4].copy_from_slice(&bytes);
        }

        let moves = meta.extract_legal_moves(&obs);
        assert_eq!(moves, vec![1, 3, 5]);
    }

    #[test]
    fn test_is_action_legal() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_actions(9)
            .with_observation(29, 18);

        // Create observation with only position 4 (center) legal
        let mut obs = vec![0u8; 29 * 4];
        let byte_offset = (18 + 4) * 4;
        let bytes = 1.0f32.to_le_bytes();
        obs[byte_offset..byte_offset + 4].copy_from_slice(&bytes);

        assert!(meta.is_action_legal(&obs, 4));
        assert!(!meta.is_action_legal(&obs, 0));
        assert!(!meta.is_action_legal(&obs, 8));
        assert!(!meta.is_action_legal(&obs, 100)); // Out of range
    }

    #[test]
    fn test_is_action_legal_short_obs() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_actions(9)
            .with_observation(29, 18);

        let obs = vec![0u8; 10]; // Too short
        assert!(!meta.is_action_legal(&obs, 0));
    }
}
