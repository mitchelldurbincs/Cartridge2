//! Response types for the web API.

use engine_core::GameMetadata;
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
#[derive(Serialize, Deserialize)]
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
    pub player_symbols: Vec<char>,
    pub description: String,
    pub board_type: String,
}

impl From<GameMetadata> for GameInfoResponse {
    fn from(meta: GameMetadata) -> Self {
        Self {
            env_id: meta.env_id,
            display_name: meta.display_name,
            board_width: meta.board_width,
            board_height: meta.board_height,
            num_actions: meta.num_actions,
            obs_size: meta.obs_size,
            legal_mask_offset: meta.legal_mask_offset,
            player_count: meta.player_count,
            player_names: meta.player_names,
            player_symbols: meta.player_symbols,
            description: meta.description,
            board_type: meta.board_type,
        }
    }
}

/// Current game state.
#[derive(Serialize, Deserialize)]
pub struct GameStateResponse {
    /// Board cells: 0=empty, 1=X (player), 2=O (bot)
    pub board: Vec<u8>,
    /// Current player: 1=X, 2=O
    pub current_player: u8,
    /// Which player the human is: 1 or 2 (depends on who went first)
    pub human_player: u8,
    /// Winner: 0=ongoing, 1=X wins, 2=O wins, 3=draw
    pub winner: u8,
    /// Is the game over?
    pub game_over: bool,
    /// Legal moves (positions)
    pub legal_moves: Vec<u8>,
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
    pub bot_move: Option<u8>,
}

/// Training history entry for loss visualization.
#[derive(Deserialize, Serialize, Clone, Default)]
pub struct HistoryEntry {
    #[serde(default)]
    pub step: u32,
    #[serde(default)]
    pub total_loss: f64,
    #[serde(default)]
    pub value_loss: f64,
    #[serde(default)]
    pub policy_loss: f64,
    #[serde(default)]
    pub learning_rate: f64,
}

/// Evaluation stats from a single evaluation run.
#[derive(Deserialize, Serialize, Clone, Default)]
pub struct EvalStats {
    #[serde(default)]
    pub step: u32,
    #[serde(default)]
    pub win_rate: f64,
    #[serde(default)]
    pub draw_rate: f64,
    #[serde(default)]
    pub loss_rate: f64,
    #[serde(default)]
    pub games_played: u32,
    #[serde(default)]
    pub avg_game_length: f64,
    #[serde(default)]
    pub timestamp: f64,
}

/// Training stats read from Python trainer and sent to frontend.
#[derive(Serialize, Deserialize, Default)]
pub struct TrainingStats {
    #[serde(default)]
    pub step: u32,
    #[serde(default)]
    pub total_steps: u32,
    #[serde(default)]
    pub total_loss: f64,
    #[serde(default)]
    pub policy_loss: f64,
    #[serde(default)]
    pub value_loss: f64,
    #[serde(default)]
    pub replay_buffer_size: u64,
    #[serde(default)]
    pub learning_rate: f64,
    #[serde(default)]
    pub timestamp: f64,
    #[serde(default)]
    pub env_id: String,
    #[serde(default)]
    pub last_eval: Option<EvalStats>,
    #[serde(default)]
    pub eval_history: Vec<EvalStats>,
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
}

/// Model information response.
#[derive(Serialize)]
pub struct ModelInfoResponse {
    /// Whether a model is currently loaded
    pub loaded: bool,
    /// Path to the loaded model file
    pub path: Option<String>,
    /// When the model file was last modified (Unix timestamp)
    pub file_modified: Option<u64>,
    /// When the model was loaded into memory (Unix timestamp)
    pub loaded_at: Option<u64>,
    /// Training step from filename (if parseable)
    pub training_step: Option<u32>,
    /// Human-readable status message
    pub status: String,
}

/// Actor self-play statistics (from actor_stats.json).
#[derive(Serialize, Deserialize, Default)]
pub struct ActorStats {
    /// Environment being used for self-play
    #[serde(default)]
    pub env_id: String,
    /// Number of episodes completed
    #[serde(default)]
    pub episodes_completed: u32,
    /// Total game steps across all episodes
    #[serde(default)]
    pub total_steps: u64,
    /// Episodes that ended in player 1 win
    #[serde(default)]
    pub player1_wins: u32,
    /// Episodes that ended in player 2 win
    #[serde(default)]
    pub player2_wins: u32,
    /// Episodes that ended in draw
    #[serde(default)]
    pub draws: u32,
    /// Average episode length
    #[serde(default)]
    pub avg_episode_length: f64,
    /// Episodes completed per second
    #[serde(default)]
    pub episodes_per_second: f64,
    /// Total runtime in seconds
    #[serde(default)]
    pub runtime_seconds: f64,
    /// Average MCTS inference time in microseconds
    #[serde(default)]
    pub mcts_avg_inference_us: f64,
    /// When these stats were last updated (Unix timestamp)
    #[serde(default)]
    pub timestamp: u64,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
#[path = "responses_tests.rs"]
mod tests;
