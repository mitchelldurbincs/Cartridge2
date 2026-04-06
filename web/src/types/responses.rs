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
mod tests {
    use super::*;

    // ========================================
    // HealthResponse Tests
    // ========================================

    #[test]
    fn test_health_response_creation() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "1.0.0".to_string(),
        };

        assert_eq!(response.status, "ok");
        assert_eq!(response.version, "1.0.0");
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "1.0.0".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("status"));
        assert!(json.contains("ok"));
        assert!(json.contains("version"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_health_response_deserialization() {
        let json = r#"{"status": "error", "version": "2.0.0"}"#;
        let response: HealthResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.status, "error");
        assert_eq!(response.version, "2.0.0");
    }

    // ========================================
    // GamesListResponse Tests
    // ========================================

    #[test]
    fn test_games_list_response_serialization() {
        let response = GamesListResponse {
            games: vec!["tictactoe".to_string(), "connect4".to_string()],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("games"));
        assert!(json.contains("tictactoe"));
        assert!(json.contains("connect4"));
    }

    #[test]
    fn test_games_list_response_deserialization() {
        let json = r#"{"games": ["chess", "checkers"]}"#;
        let response: GamesListResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.games.len(), 2);
        assert_eq!(response.games[0], "chess");
        assert_eq!(response.games[1], "checkers");
    }

    // ========================================
    // GameInfoResponse Tests
    // ========================================

    #[test]
    fn test_game_info_response_from_game_metadata() {
        let metadata = GameMetadata::new("test_game", "Test Game Display")
            .with_board(5, 5)
            .with_actions(25)
            .with_observation(50, 25)
            .with_players(2, vec!["Player A".to_string(), "Player B".to_string()], vec!['A', 'B']);

        let response: GameInfoResponse = metadata.into();

        assert_eq!(response.env_id, "test_game");
        assert_eq!(response.display_name, "Test Game Display");
        assert_eq!(response.board_width, 5);
        assert_eq!(response.board_height, 5);
        assert_eq!(response.num_actions, 25);
        assert_eq!(response.obs_size, 50);
        assert_eq!(response.legal_mask_offset, 25);
        assert_eq!(response.player_count, 2);
        assert_eq!(response.player_names, vec!["Player A", "Player B"]);
        assert_eq!(response.player_symbols, vec!['A', 'B']);
    }

    #[test]
    fn test_game_info_response_serialization() {
        let response = GameInfoResponse {
            env_id: "tictactoe".to_string(),
            display_name: "Tic-Tac-Toe".to_string(),
            board_width: 3,
            board_height: 3,
            num_actions: 9,
            obs_size: 29,
            legal_mask_offset: 18,
            player_count: 2,
            player_names: vec!["X".to_string(), "O".to_string()],
            player_symbols: vec!['X', 'O'],
            description: "Classic game".to_string(),
            board_type: "grid".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("env_id"));
        assert!(json.contains("tictactoe"));
        assert!(json.contains("board_width"));
        assert!(json.contains("3"));
    }

    // ========================================
    // GameStateResponse Tests
    // ========================================

    #[test]
    fn test_game_state_response_serialization() {
        let response = GameStateResponse {
            board: vec![0, 1, 2, 0, 1, 0, 0, 2, 1],
            current_player: 1,
            human_player: 1,
            winner: 0,
            game_over: false,
            legal_moves: vec![0, 2, 5, 6],
            message: "Your turn".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("board"));
        assert!(json.contains("current_player"));
        assert!(json.contains("legal_moves"));
        assert!(json.contains("message"));
    }

    #[test]
    fn test_game_state_response_deserialization() {
        let json = r#"{
            "board": [0, 0, 0, 1, 2, 0, 0, 0, 0],
            "current_player": 2,
            "human_player": 1,
            "winner": 0,
            "game_over": false,
            "legal_moves": [0, 1, 2, 5, 6, 7, 8],
            "message": "Bot's turn"
        }"#;

        let response: GameStateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.board.len(), 9);
        assert_eq!(response.current_player, 2);
        assert_eq!(response.human_player, 1);
        assert!(!response.game_over);
    }

    // ========================================
    // MoveResponse Tests
    // ========================================

    #[test]
    fn test_move_response_serialization_flattened() {
        let state = GameStateResponse {
            board: vec![1, 0, 0, 0, 2, 0, 0, 0, 0],
            current_player: 1,
            human_player: 1,
            winner: 0,
            game_over: false,
            legal_moves: vec![1, 2, 3, 5, 6, 7, 8],
            message: "Your turn".to_string(),
        };

        let response = MoveResponse {
            state,
            bot_move: Some(4),
        };

        let json = serde_json::to_string(&response).unwrap();
        // Fields from GameStateResponse should be at top level due to flatten
        assert!(json.contains("board"));
        assert!(json.contains("bot_move"));
        assert!(json.contains("4"));
    }

    #[test]
    fn test_move_response_deserialization() {
        // Note: flatten works both ways - when deserializing, fields are distributed
        let json = r#"{
            "board": [1, 2, 0, 0, 0, 0, 0, 0, 0],
            "current_player": 1,
            "human_player": 1,
            "winner": 0,
            "game_over": false,
            "legal_moves": [2, 3, 4, 5, 6, 7, 8],
            "message": "Your turn",
            "bot_move": 1
        }"#;

        let response: MoveResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.bot_move, Some(1));
        assert_eq!(response.state.board[0], 1);
    }

    // ========================================
    // HistoryEntry Tests
    // ========================================

    #[test]
    fn test_history_entry_serialization() {
        let entry = HistoryEntry {
            step: 100,
            total_loss: 0.5,
            value_loss: 0.2,
            policy_loss: 0.3,
            learning_rate: 0.001,
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("step"));
        assert!(json.contains("100"));
        assert!(json.contains("total_loss"));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn test_history_entry_deserialization_with_defaults() {
        // Test that missing fields use defaults
        let json = r#"{"step": 50}"#;
        let entry: HistoryEntry = serde_json::from_str(json).unwrap();

        assert_eq!(entry.step, 50);
        assert_eq!(entry.total_loss, 0.0); // default
        assert_eq!(entry.value_loss, 0.0); // default
        assert_eq!(entry.policy_loss, 0.0); // default
        assert_eq!(entry.learning_rate, 0.0); // default
    }

    // ========================================
    // EvalStats Tests
    // ========================================

    #[test]
    fn test_eval_stats_serialization() {
        let eval = EvalStats {
            step: 200,
            win_rate: 0.6,
            draw_rate: 0.2,
            loss_rate: 0.2,
            games_played: 100,
            avg_game_length: 15.5,
            timestamp: 1234567890.0,
        };

        let json = serde_json::to_string(&eval).unwrap();
        assert!(json.contains("win_rate"));
        assert!(json.contains("0.6"));
        assert!(json.contains("games_played"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_eval_stats_deserialization_with_defaults() {
        let json = r#"{"step": 150, "win_rate": 0.55}"#;
        let eval: EvalStats = serde_json::from_str(json).unwrap();

        assert_eq!(eval.step, 150);
        assert!((eval.win_rate - 0.55).abs() < f64::EPSILON);
        assert_eq!(eval.draw_rate, 0.0); // default
        assert_eq!(eval.games_played, 0); // default
    }

    // ========================================
    // TrainingStats Tests
    // ========================================

    #[test]
    fn test_training_stats_serialization() {
        let stats = TrainingStats {
            step: 500,
            total_steps: 10000,
            total_loss: 0.4,
            policy_loss: 0.25,
            value_loss: 0.15,
            replay_buffer_size: 50000,
            learning_rate: 0.0005,
            timestamp: 1234567890.0,
            env_id: "tictactoe".to_string(),
            last_eval: None,
            eval_history: vec![],
            history: vec![],
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("step"));
        assert!(json.contains("500"));
        assert!(json.contains("tictactoe"));
        assert!(json.contains("replay_buffer_size"));
    }

    #[test]
    fn test_training_stats_with_nested_eval() {
        let eval = EvalStats {
            step: 500,
            win_rate: 0.7,
            draw_rate: 0.2,
            loss_rate: 0.1,
            games_played: 50,
            avg_game_length: 12.0,
            timestamp: 1234567890.0,
        };

        let stats = TrainingStats {
            step: 500,
            total_steps: 1000,
            total_loss: 0.3,
            policy_loss: 0.2,
            value_loss: 0.1,
            replay_buffer_size: 10000,
            learning_rate: 0.001,
            timestamp: 1234567890.0,
            env_id: "connect4".to_string(),
            last_eval: Some(eval.clone()),
            eval_history: vec![eval.clone()],
            history: vec![HistoryEntry {
                step: 500,
                total_loss: 0.3,
                value_loss: 0.1,
                policy_loss: 0.2,
                learning_rate: 0.001,
            }],
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("last_eval"));
        assert!(json.contains("eval_history"));
        assert!(json.contains("win_rate"));
    }

    // ========================================
    // ModelInfoResponse Tests
    // ========================================

    #[test]
    fn test_model_info_response_serialization() {
        let info = ModelInfoResponse {
            loaded: true,
            path: Some("/models/latest.onnx".to_string()),
            file_modified: Some(1234567890),
            loaded_at: Some(1234567891),
            training_step: Some(1000),
            status: "Model loaded (step 1000)".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("loaded"));
        assert!(json.contains("true"));
        assert!(json.contains("path"));
        assert!(json.contains("/models/latest.onnx"));
        assert!(json.contains("training_step"));
    }

    #[test]
    fn test_model_info_response_not_loaded() {
        let info = ModelInfoResponse {
            loaded: false,
            path: None,
            file_modified: None,
            loaded_at: None,
            training_step: None,
            status: "No model loaded".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("loaded"));
        assert!(json.contains("false"));
        assert!(json.contains("null")); // None serializes to null
    }

    // ========================================
    // ActorStats Tests
    // ========================================

    #[test]
    fn test_actor_stats_serialization() {
        let stats = ActorStats {
            env_id: "tictactoe".to_string(),
            episodes_completed: 1000,
            total_steps: 15000,
            player1_wins: 450,
            player2_wins: 400,
            draws: 150,
            avg_episode_length: 15.0,
            episodes_per_second: 5.5,
            runtime_seconds: 180.0,
            mcts_avg_inference_us: 450.0,
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("env_id"));
        assert!(json.contains("tictactoe"));
        assert!(json.contains("episodes_completed"));
        assert!(json.contains("1000"));
        assert!(json.contains("player1_wins"));
        assert!(json.contains("450"));
    }

    #[test]
    fn test_actor_stats_deserialization_with_defaults() {
        let json = r#"{"env_id": "connect4", "episodes_completed": 500}"#;
        let stats: ActorStats = serde_json::from_str(json).unwrap();

        assert_eq!(stats.env_id, "connect4");
        assert_eq!(stats.episodes_completed, 500);
        assert_eq!(stats.total_steps, 0); // default
        assert_eq!(stats.player1_wins, 0); // default
        assert_eq!(stats.draws, 0); // default
    }
}
