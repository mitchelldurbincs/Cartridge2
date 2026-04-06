//! Stats and model info handlers.

use axum::{extract::State, Json};
use std::sync::Arc;

use crate::types::{ActorStats, ModelInfoResponse, TrainingStats};
use crate::AppState;

/// Get training stats from stats.json.
pub async fn get_stats(State(state): State<Arc<AppState>>) -> Json<TrainingStats> {
    let stats_path = format!("{}/stats.json", state.data_dir);

    match tokio::fs::read_to_string(&stats_path).await {
        Ok(content) => match serde_json::from_str::<TrainingStats>(&content) {
            Ok(stats) => Json(stats),
            Err(e) => {
                tracing::warn!("Failed to parse stats.json: {}", e);
                Json(TrainingStats::default())
            }
        },
        Err(_) => {
            // Return empty stats if file doesn't exist
            Json(TrainingStats::default())
        }
    }
}

/// Get info about the currently loaded model.
pub async fn get_model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    // model_info uses std::sync::RwLock (shared with model_watcher crate).
    // This is safe because the lock is held briefly and not across await points.
    let info = state
        .model_info
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default();

    let status = if info.loaded {
        match info.training_step {
            Some(step) => format!("Model loaded (step {})", step),
            None => "Model loaded".to_string(),
        }
    } else {
        "No model loaded - bot plays randomly".to_string()
    };

    Json(ModelInfoResponse {
        loaded: info.loaded,
        path: info.path,
        file_modified: info.file_modified,
        loaded_at: info.loaded_at,
        training_step: info.training_step,
        status,
    })
}

/// Get actor self-play stats from actor_stats.json.
pub async fn get_actor_stats(State(state): State<Arc<AppState>>) -> Json<ActorStats> {
    let stats_path = format!("{}/actor_stats.json", state.data_dir);

    match tokio::fs::read_to_string(&stats_path).await {
        Ok(content) => match serde_json::from_str::<ActorStats>(&content) {
            Ok(stats) => Json(stats),
            Err(e) => {
                tracing::warn!("Failed to parse actor_stats.json: {}", e);
                Json(ActorStats::default())
            }
        },
        Err(_) => {
            // Return empty stats if file doesn't exist
            Json(ActorStats::default())
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorStats, EvalStats, HistoryEntry, ModelInfoResponse, TrainingStats};

    #[test]
    fn test_model_info_response_default() {
        let info = ModelInfoResponse {
            loaded: false,
            path: None,
            file_modified: None,
            loaded_at: None,
            training_step: None,
            status: "No model loaded".to_string(),
        };

        assert!(!info.loaded);
        assert!(info.path.is_none());
        assert_eq!(info.status, "No model loaded");
    }

    #[test]
    fn test_model_info_response_loaded() {
        let info = ModelInfoResponse {
            loaded: true,
            path: Some("/models/latest.onnx".to_string()),
            file_modified: Some(1234567890),
            loaded_at: Some(1234567891),
            training_step: Some(100),
            status: "Model loaded (step 100)".to_string(),
        };

        assert!(info.loaded);
        assert_eq!(info.path, Some("/models/latest.onnx".to_string()));
        assert_eq!(info.training_step, Some(100));
    }

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats::default();
        
        assert_eq!(stats.step, 0);
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.total_loss, 0.0);
        assert_eq!(stats.replay_buffer_size, 0);
        assert!(stats.last_eval.is_none());
        assert!(stats.eval_history.is_empty());
        assert!(stats.history.is_empty());
    }

    #[test]
    fn test_training_stats_with_eval() {
        let eval = EvalStats {
            step: 100,
            win_rate: 0.6,
            draw_rate: 0.3,
            loss_rate: 0.1,
            games_played: 50,
            avg_game_length: 15.5,
            timestamp: 1234567890.0,
        };

        let stats = TrainingStats {
            step: 100,
            total_steps: 1000,
            total_loss: 0.5,
            policy_loss: 0.3,
            value_loss: 0.2,
            replay_buffer_size: 10000,
            learning_rate: 0.001,
            timestamp: 1234567890.0,
            env_id: "tictactoe".to_string(),
            last_eval: Some(eval.clone()),
            eval_history: vec![eval],
            history: vec![HistoryEntry {
                step: 100,
                total_loss: 0.5,
                value_loss: 0.2,
                policy_loss: 0.3,
                learning_rate: 0.001,
            }],
        };

        assert_eq!(stats.step, 100);
        assert_eq!(stats.total_steps, 1000);
        assert!((stats.total_loss - 0.5).abs() < f64::EPSILON);
        assert!(stats.last_eval.is_some());
        
        let last_eval = stats.last_eval.unwrap();
        assert!((last_eval.win_rate - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_actor_stats_default() {
        let stats = ActorStats::default();
        
        assert_eq!(stats.env_id, "");
        assert_eq!(stats.episodes_completed, 0);
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.player1_wins, 0);
        assert_eq!(stats.player2_wins, 0);
        assert_eq!(stats.draws, 0);
        assert_eq!(stats.avg_episode_length, 0.0);
    }

    #[test]
    fn test_actor_stats_with_data() {
        let stats = ActorStats {
            env_id: "tictactoe".to_string(),
            episodes_completed: 100,
            total_steps: 1500,
            player1_wins: 45,
            player2_wins: 40,
            draws: 15,
            avg_episode_length: 15.0,
            episodes_per_second: 2.5,
            runtime_seconds: 40.0,
            mcts_avg_inference_us: 500.0,
            timestamp: 1234567890,
        };

        assert_eq!(stats.env_id, "tictactoe");
        assert_eq!(stats.episodes_completed, 100);
        assert_eq!(stats.total_steps, 1500);
        assert!((stats.episodes_per_second - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_history_entry_default() {
        let entry = HistoryEntry::default();
        
        assert_eq!(entry.step, 0);
        assert_eq!(entry.total_loss, 0.0);
        assert_eq!(entry.value_loss, 0.0);
        assert_eq!(entry.policy_loss, 0.0);
        assert_eq!(entry.learning_rate, 0.0);
    }

    #[test]
    fn test_eval_stats_default() {
        let eval = EvalStats::default();
        
        assert_eq!(eval.step, 0);
        assert_eq!(eval.win_rate, 0.0);
        assert_eq!(eval.draw_rate, 0.0);
        assert_eq!(eval.loss_rate, 0.0);
        assert_eq!(eval.games_played, 0);
        assert_eq!(eval.avg_game_length, 0.0);
    }

    #[test]
    fn test_eval_stats_rates_sum_to_one() {
        // Test that rates can be set and sum to 1.0
        let eval = EvalStats {
            step: 100,
            win_rate: 0.5,
            draw_rate: 0.3,
            loss_rate: 0.2,
            games_played: 100,
            avg_game_length: 20.0,
            timestamp: 1234567890.0,
        };

        let sum = eval.win_rate + eval.draw_rate + eval.loss_rate;
        assert!((sum - 1.0).abs() < f64::EPSILON);
    }
}
