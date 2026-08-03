//! Unit tests for response types.

use super::*;
use engine_core::board_profile::{BoardGameMetadata, BoardPlayerMetadata, BoardView};

fn test_board_metadata() -> EnvironmentMetadata {
    EnvironmentMetadata::new("test_game", "Test Game Display").with_board(
        BoardGameMetadata::new(5, 5, 25)
            .with_observation(52, 1, 25, false)
            .with_players(vec![
                BoardPlayerMetadata::new("Player A", "A"),
                BoardPlayerMetadata::new("Player B", "B"),
            ]),
    )
}

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
fn test_game_info_response_from_environment_metadata() {
    let metadata = test_board_metadata();

    let response = GameInfoResponse::try_from(metadata).unwrap();

    assert_eq!(response.env_id, "test_game");
    assert_eq!(response.display_name, "Test Game Display");
    assert_eq!(response.board_width, 5);
    assert_eq!(response.board_height, 5);
    assert_eq!(response.num_actions, 25);
    assert_eq!(response.obs_size, 52);
    assert_eq!(response.legal_mask_offset, 25);
    assert_eq!(response.player_count, 2);
    assert_eq!(response.player_names, vec!["Player A", "Player B"]);
    assert_eq!(response.player_symbols, vec!["A", "B"]);
}

#[test]
fn test_game_info_rejects_environment_without_board_profile() {
    let metadata = EnvironmentMetadata::new("counter", "Counter");
    let error = GameInfoResponse::try_from(metadata)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no board presentation profile"));
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
        player_symbols: vec!["X".to_string(), "O".to_string()],
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
        cells: BoardView::from_owners(&[0, 1, 2, 0, 1, 0, 0, 2, 1], 1, 0).cells,
        current_player: 1,
        human_player: 1,
        winner: 0,
        game_over: false,
        legal_moves: vec![0, 2, 5, 6],
        message: "Your turn".to_string(),
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("cells"));
    assert!(json.contains("current_player"));
    assert!(json.contains("legal_moves"));
    assert!(json.contains("message"));
}

#[test]
fn test_game_state_response_deserialization() {
    let json = r#"{
        "cells": [{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":1,"kind":"normal","value":0},{"owner":2,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0}],
        "current_player": 2,
        "human_player": 1,
        "winner": 0,
        "game_over": false,
        "legal_moves": [0, 1, 2, 5, 6, 7, 8],
        "message": "Bot's turn"
    }"#;

    let response: GameStateResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.cells.len(), 9);
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
        cells: BoardView::from_owners(&[1, 0, 0, 0, 2, 0, 0, 0, 0], 1, 0).cells,
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
    assert!(json.contains("cells"));
    assert!(json.contains("bot_move"));
    assert!(json.contains("4"));
}

#[test]
fn test_move_response_deserialization() {
    // Note: flatten works both ways - when deserializing, fields are distributed
    let json = r#"{
        "cells": [{"owner":1,"kind":"normal","value":0},{"owner":2,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0},{"owner":0,"kind":"normal","value":0}],
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
    assert_eq!(response.state.cells[0].owner, 1);
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
        grad_norm: Some(1.5),
    };

    let json = serde_json::to_string(&entry).unwrap();
    assert!(json.contains("step"));
    assert!(json.contains("100"));
    assert!(json.contains("total_loss"));
    assert!(json.contains("0.5"));
}

#[test]
fn test_history_entry_rejects_partial_legacy_shape() {
    let json = r#"{"step": 50}"#;
    let error = serde_json::from_str::<HistoryEntry>(json).unwrap_err();

    assert!(error.to_string().contains("total_loss"));
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
fn test_eval_stats_rejects_partial_legacy_shape() {
    let json = r#"{"step": 150, "win_rate": 0.55}"#;
    let error = serde_json::from_str::<EvalStats>(json).unwrap_err();

    assert!(error.to_string().contains("draw_rate"));
}

#[test]
fn test_eval_stats_accepts_u64_game_counts_from_the_authoritative_stats_schema() {
    let games_played = u64::from(u32::MAX) + 1;
    let stats: EvalStats = serde_json::from_value(serde_json::json!({
        "step": 1,
        "win_rate": 0.5,
        "draw_rate": 0.25,
        "loss_rate": 0.25,
        "games_played": games_played,
        "avg_game_length": 8.0,
        "timestamp": 1.0
    }))
    .unwrap();

    assert_eq!(stats.games_played, games_played);
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
        samples_seen: 32000,
        replay_record_count: 50000,
        last_checkpoint: "checkpoint-id".to_string(),
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
    assert!(json.contains("replay_record_count"));
}

#[test]
fn test_training_stats_accepts_u64_steps_from_the_authoritative_stats_schema() {
    let step = u64::from(u32::MAX) + 1;
    let json = serde_json::json!({
        "step": step,
        "total_steps": step,
        "total_loss": 0.4,
        "value_loss": 0.15,
        "policy_loss": 0.25,
        "learning_rate": 0.0005,
        "samples_seen": step,
        "replay_record_count": 50_000,
        "last_checkpoint": "checkpoint-id",
        "timestamp": 1_234_567_890.0,
        "history": [{
            "step": step,
            "total_loss": 0.4,
            "value_loss": 0.15,
            "policy_loss": 0.25,
            "learning_rate": 0.0005,
            "grad_norm": null
        }],
        "env_id": "tictactoe",
        "last_eval": null,
        "eval_history": []
    });

    let stats: TrainingStats = serde_json::from_value(json).unwrap();
    assert_eq!(stats.step, step);
    assert_eq!(stats.total_steps, step);
    assert_eq!(stats.history[0].step, step);
}

#[test]
fn test_training_stats_rejects_partial_legacy_shape() {
    let error = serde_json::from_str::<TrainingStats>(r#"{"step": 50}"#).unwrap_err();

    assert!(error.to_string().contains("total_steps"));
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
        samples_seen: 32000,
        replay_record_count: 10000,
        last_checkpoint: "checkpoint-id".to_string(),
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
            grad_norm: None,
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
        checkpoint_id: Some("a".repeat(64)),
        model_sha256: Some("b".repeat(64)),
        path: Some("/models/blobs/sha256/model.onnx".to_string()),
        loaded_at: Some(1234567891),
        training_step: Some(1000),
        status: "Model loaded (step 1000)".to_string(),
    };

    let json = serde_json::to_string(&info).unwrap();
    assert!(json.contains("loaded"));
    assert!(json.contains("true"));
    assert!(json.contains("path"));
    assert!(json.contains("/models/blobs/sha256/model.onnx"));
    assert!(json.contains("checkpoint_id"));
    assert!(json.contains("model_sha256"));
    assert!(json.contains("training_step"));
}

#[test]
fn test_model_info_response_not_loaded() {
    let info = ModelInfoResponse {
        loaded: false,
        checkpoint_id: None,
        model_sha256: None,
        path: None,
        loaded_at: None,
        training_step: None,
        status: "No model loaded".to_string(),
    };

    let json = serde_json::to_string(&info).unwrap();
    assert!(json.contains("loaded"));
    assert!(json.contains("false"));
    assert!(json.contains("null")); // None serializes to null
}
