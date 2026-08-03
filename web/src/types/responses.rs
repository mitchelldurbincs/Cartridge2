//! Response types for the web API.

use anyhow::{anyhow, Result};
use engine_core::board_profile::{BoardRenderer, CellView};
use engine_core::EnvironmentMetadata;
use serde::{Deserialize, Serialize};

/// Health check response.
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// List of available games.
#[derive(Serialize, Deserialize)]
pub struct GamesListResponse {
    pub games: Vec<String>,
}

/// Game metadata response.
#[derive(Debug, Serialize, Deserialize)]
pub struct GameInfoResponse {
    pub env_id: String,
    pub display_name: String,
    pub board_width: usize,
    pub board_height: usize,
    pub num_actions: usize,
    pub obs_size: usize,
    pub legal_mask_offset: usize,
    pub player_count: usize,
    pub player_names: Vec<String>,
    pub player_symbols: Vec<String>,
    pub description: String,
    pub board_type: String,
}

impl TryFrom<EnvironmentMetadata> for GameInfoResponse {
    type Error = anyhow::Error;

    fn try_from(meta: EnvironmentMetadata) -> Result<Self> {
        let board = meta.board.ok_or_else(|| {
            anyhow!(
                "environment '{}' has no board presentation profile",
                meta.id
            )
        })?;
        if board.players.len() != 2 {
            return Err(anyhow!(
                "AlphaZero web serving requires exactly two board players, got {}",
                board.players.len()
            ));
        }

        let board_type = match board.renderer {
            BoardRenderer::Grid => "grid",
            BoardRenderer::DropColumn => "drop_column",
            BoardRenderer::Generals => "generals",
        };

        Ok(Self {
            env_id: meta.id,
            display_name: meta.display_name,
            board_width: board.width,
            board_height: board.height,
            num_actions: board.action_count,
            obs_size: board.observation.elements,
            legal_mask_offset: board.observation.legal_actions_offset,
            player_count: board.players.len(),
            player_names: board
                .players
                .iter()
                .map(|player| player.name.clone())
                .collect(),
            player_symbols: board
                .players
                .iter()
                .map(|player| player.symbol.clone())
                .collect(),
            description: meta.description,
            board_type: board_type.to_string(),
        })
    }
}

/// Current game state.
#[derive(Serialize, Deserialize)]
pub struct GameStateResponse {
    /// Board cells, row-major, straight from the engine's `BoardView`. Each
    /// carries owner (0=empty, 1=player, 2=bot), terrain, and any per-cell
    /// quantity — the flat games leave the latter two at their defaults.
    pub cells: Vec<CellView>,
    /// Current player: 1=X, 2=O
    pub current_player: u8,
    /// Which player the human is: 1 or 2 (depends on who went first)
    pub human_player: u8,
    /// Winner: 0=ongoing, 1=X wins, 2=O wins, 3=draw
    pub winner: u8,
    /// Is the game over?
    pub game_over: bool,
    /// Legal moves (action indices; Generals has 257 of them, so not a u8)
    pub legal_moves: Vec<u32>,
    /// Status message
    pub message: String,
}

/// Response after making a move.
#[derive(Serialize, Deserialize)]
pub struct MoveResponse {
    /// Updated game state
    #[serde(flatten)]
    pub state: GameStateResponse,
    /// Bot's move position (if bot moved)
    pub bot_move: Option<u32>,
}

/// Training history entry for loss visualization.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct HistoryEntry {
    pub step: u64,
    pub total_loss: f64,
    pub value_loss: f64,
    pub policy_loss: f64,
    pub learning_rate: f64,
    pub grad_norm: Option<f64>,
}

/// Evaluation stats from a single evaluation run.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct EvalStats {
    pub step: u64,
    pub win_rate: f64,
    pub draw_rate: f64,
    pub loss_rate: f64,
    pub games_played: u64,
    pub avg_game_length: f64,
    pub timestamp: f64,
}

/// Training stats read from Python trainer and sent to frontend.
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct TrainingStats {
    pub step: u64,
    pub total_steps: u64,
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub samples_seen: u64,
    pub replay_record_count: u64,
    pub last_checkpoint: String,
    pub learning_rate: f64,
    pub timestamp: f64,
    pub env_id: String,
    pub last_eval: Option<EvalStats>,
    pub eval_history: Vec<EvalStats>,
    pub history: Vec<HistoryEntry>,
}

/// Model information response.
#[derive(Serialize)]
pub struct ModelInfoResponse {
    /// Whether a model is currently loaded
    pub loaded: bool,
    /// Content identity of the checkpoint manifest.
    pub checkpoint_id: Option<String>,
    /// SHA-256 digest of the immutable ONNX blob.
    pub model_sha256: Option<String>,
    /// Path to the loaded model file
    pub path: Option<String>,
    /// When the model was loaded into memory (Unix timestamp)
    pub loaded_at: Option<u64>,
    /// Training step declared by the verified checkpoint manifest.
    pub training_step: Option<u64>,
    /// Human-readable status message
    pub status: String,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
#[path = "responses_tests.rs"]
mod tests;
