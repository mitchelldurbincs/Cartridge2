//! Stats and model info handlers.

use axum::{extract::State, http::StatusCode, Json};
use serde::de::DeserializeOwned;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

use crate::types::{ModelInfoResponse, TrainingStats};
use crate::AppState;

type StatsResult<T> = Result<Json<T>, (StatusCode, String)>;

fn stats_error(path: &Path, detail: impl std::fmt::Display) -> (StatusCode, String) {
    tracing::error!(
        path = %path.display(),
        error = %detail,
        "Training stats projection is invalid"
    );
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!(
            "Training stats projection '{}' is invalid: {detail}",
            path.display()
        ),
    )
}

/// Read one strict JSON projection. Absence means the run has not emitted a
/// projection yet; a present object must be a readable regular file whose JSON
/// exactly matches the response schema.
async fn read_stats_file<T: DeserializeOwned + Default>(
    data_dir: &str,
    filename: &str,
) -> StatsResult<T> {
    let stats_path = Path::new(data_dir).join(filename);
    let metadata = match tokio::fs::metadata(&stats_path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Json(T::default())),
        Err(error) => return Err(stats_error(&stats_path, error)),
    };
    if !metadata.is_file() {
        return Err(stats_error(&stats_path, "path is not a regular file"));
    }

    let content = match tokio::fs::read_to_string(&stats_path).await {
        Ok(content) => content,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Json(T::default())),
        Err(error) => return Err(stats_error(&stats_path, error)),
    };
    serde_json::from_str::<T>(&content)
        .map(Json)
        .map_err(|error| stats_error(&stats_path, error))
}

/// Get training stats from stats.json.
pub async fn get_stats(State(state): State<Arc<AppState>>) -> StatsResult<TrainingStats> {
    read_stats_file(&state.data_dir, "stats.json").await
}

/// Get info about the currently loaded model.
pub async fn get_model_info(State(state): State<Arc<AppState>>) -> StatsResult<ModelInfoResponse> {
    // model_info uses std::sync::RwLock (shared with model_watcher crate).
    // This is safe because the lock is held briefly and not across await points.
    let info = state
        .model_info
        .read()
        .map_err(|error| {
            tracing::error!(error = %error, "Model information lock is poisoned");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Model information is unavailable".to_string(),
            )
        })?
        .clone();

    let status = if info.loaded {
        match info.training_step {
            Some(step) => format!("Model loaded (step {})", step),
            None => "Model loaded".to_string(),
        }
    } else {
        "No model loaded - bot plays randomly".to_string()
    };

    Ok(Json(ModelInfoResponse {
        loaded: info.loaded,
        checkpoint_id: info.checkpoint_id,
        model_sha256: info.model_sha256,
        path: info.path,
        loaded_at: info.loaded_at,
        training_step: info.training_step,
        status,
    }))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::types::{EvalStats, HistoryEntry, ModelInfoResponse, TrainingStats};

    #[test]
    fn test_model_info_response_default() {
        let info = ModelInfoResponse {
            loaded: false,
            checkpoint_id: None,
            model_sha256: None,
            path: None,
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
            checkpoint_id: Some("a".repeat(64)),
            model_sha256: Some("b".repeat(64)),
            path: Some("/models/blobs/sha256/model.onnx".to_string()),
            loaded_at: Some(1234567891),
            training_step: Some(100),
            status: "Model loaded (step 100)".to_string(),
        };

        assert!(info.loaded);
        assert_eq!(
            info.path,
            Some("/models/blobs/sha256/model.onnx".to_string())
        );
        assert_eq!(info.training_step, Some(100));
    }

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats::default();

        assert_eq!(stats.step, 0);
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.total_loss, 0.0);
        assert_eq!(stats.samples_seen, 0);
        assert_eq!(stats.replay_record_count, 0);
        assert!(stats.last_checkpoint.is_empty());
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
            samples_seen: 6400,
            replay_record_count: 10000,
            last_checkpoint: "checkpoint-id".to_string(),
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
                grad_norm: None,
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
