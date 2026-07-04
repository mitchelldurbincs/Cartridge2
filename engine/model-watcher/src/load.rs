//! Model metadata types and filesystem load helpers.
//!
//! This module holds the [`ModelInfo`] type plus the metadata extraction and
//! static model-loading helpers used by [`ModelWatcher`](crate::ModelWatcher).
//! They are factored out here so the watcher implementation in `lib.rs` stays
//! focused on the watching/polling logic.

use anyhow::{anyhow, Result};
use mcts::OnnxEvaluator;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

/// Information about the currently loaded model.
///
/// Always available. Use [`ModelWatcher::model_info`](crate::ModelWatcher::model_info)
/// to get a shared reference that is updated on each model reload.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Whether a model is currently loaded
    pub loaded: bool,
    /// Path to the loaded model file
    pub path: Option<String>,
    /// When the model file was last modified (Unix timestamp)
    pub file_modified: Option<u64>,
    /// When the model was loaded into memory (Unix timestamp)
    pub loaded_at: Option<u64>,
    /// Training step from filename (if parseable, e.g., "model_step_000100.onnx")
    pub training_step: Option<u32>,
}

/// Extract metadata from a model file path.
///
/// Returns the file modification time (Unix seconds) and the training step
/// parsed from the filename (e.g., "model_step_000100.onnx"), if available.
pub(crate) fn extract_metadata(path: &Path) -> (Option<u64>, Option<u32>) {
    // Get file modification time
    let file_modified = path
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs());

    // Try to parse training step from filename (e.g., "model_step_000100.onnx")
    let training_step = path.file_name().and_then(|n| n.to_str()).and_then(|name| {
        name.split('_')
            .filter_map(|part| part.trim_end_matches(".onnx").parse::<u32>().ok())
            .next_back()
    });

    (file_modified, training_step)
}

/// Load a model and update shared state.
///
/// This is a free function so it can be called from spawned tasks that
/// don't hold a reference to a `ModelWatcher`.
pub(crate) fn load_model_static(
    path: &Path,
    obs_size: usize,
    intra_threads: usize,
    evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
    last_mtime: &Arc<RwLock<Option<SystemTime>>>,
    model_info: Option<&Arc<RwLock<ModelInfo>>>,
) -> Result<()> {
    // Get file modification time for polling comparison
    let file_mtime = path.metadata().and_then(|m| m.modified()).ok();

    let new_evaluator = OnnxEvaluator::load(path, obs_size, intra_threads)
        .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

    {
        let mut guard = evaluator
            .write()
            .map_err(|e| anyhow!("Failed to acquire evaluator write lock: {}", e))?;
        *guard = Some(new_evaluator);
    }

    // Update last modification time (for polling)
    {
        let mut mtime_guard = last_mtime
            .write()
            .map_err(|e| anyhow!("Failed to acquire last_mtime write lock: {}", e))?;
        *mtime_guard = file_mtime;
    }

    // Update model info if metadata tracking is enabled
    if let Some(info_lock) = model_info {
        let (file_modified, training_step) = extract_metadata(path);
        let mut info = info_lock
            .write()
            .map_err(|e| anyhow!("Failed to acquire model_info write lock: {}", e))?;
        *info = ModelInfo {
            loaded: true,
            path: Some(path.to_string_lossy().to_string()),
            file_modified,
            loaded_at: Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            ),
            training_step,
        };
        info!(
            "Model loaded successfully from {:?} (modified: {:?}, step: {:?})",
            path, file_modified, training_step
        );
    } else {
        info!("Model loaded successfully from {:?}", path);
    }

    Ok(())
}
