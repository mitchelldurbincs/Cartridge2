//! Model file watcher for hot-reloading ONNX models
//!
//! This crate provides a shared implementation for watching and hot-reloading
//! ONNX model files used by both the actor and web server components.
//!
//! # Features
//!
//! - `metadata`: Enable additional metadata tracking (load time, file modification time,
//!   training step extraction). Useful for web server display.
//! - `s3`: Enable S3/MinIO backend for Kubernetes deployments where models are
//!   stored in S3-compatible storage.
//!
//! # Polling Fallback
//!
//! In addition to inotify-based file watching, this crate includes a polling fallback
//! that periodically checks for file changes. This is essential for Docker environments
//! where inotify events don't reliably propagate across container boundaries with
//! bind-mounted volumes.
//!
//! The poll interval defaults to 5 seconds and can be configured with
//! [`ModelWatcher::with_poll_interval`].
//!
//! # Example (Filesystem)
//!
//! ```ignore
//! use model_watcher::ModelWatcher;
//! use std::sync::{Arc, RwLock};
//!
//! let evaluator = Arc::new(RwLock::new(None));
//! let watcher = ModelWatcher::new("./data/models", "latest.onnx", 29, 1, evaluator);
//!
//! // Try to load existing model
//! watcher.try_load_existing()?;
//!
//! // Start watching for changes (uses both inotify and polling)
//! let mut rx = watcher.start_watching().await?;
//! while let Some(()) = rx.recv().await {
//!     println!("Model reloaded!");
//! }
//! ```
//!
//! # Example (S3 - requires `s3` feature)
//!
//! ```ignore
//! use model_watcher::s3::{S3ModelWatcher, S3Config};
//! use std::sync::{Arc, RwLock};
//!
//! let evaluator = Arc::new(RwLock::new(None));
//! let config = S3Config {
//!     bucket: "my-bucket".to_string(),
//!     key: "models/latest.onnx".to_string(),
//!     endpoint_url: Some("http://minio:9000".to_string()),
//!     region: Some("us-east-1".to_string()),
//!     cache_dir: "/tmp/model-cache".into(),
//! };
//!
//! let watcher = S3ModelWatcher::new(config, 29, evaluator).await?;
//! watcher.try_load_existing().await?;
//!
//! let mut rx = watcher.start_watching().await?;
//! while let Some(()) = rx.recv().await {
//!     println!("Model reloaded from S3!");
//! }
//! ```

use anyhow::{anyhow, Result};
use mcts::OnnxEvaluator;
use notify::{recommended_watcher, Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// S3/MinIO backend for model watching (requires `s3` feature).
#[cfg(feature = "s3")]
pub mod s3;

/// Information about the currently loaded model.
///
/// Always available. Use [`ModelWatcher::model_info`] to get a shared reference
/// that is updated on each model reload.
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

/// Default polling interval for checking model file changes.
///
/// This is used as a fallback when inotify events don't propagate
/// (common in Docker with bind mounts).
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Watches for new ONNX model files and hot-reloads them.
///
/// This is the shared implementation used by both actor and web server components.
/// It uses both inotify (for fast detection when it works) and polling (as a
/// fallback for Docker/cross-container scenarios).
pub struct ModelWatcher {
    /// Path to the models directory
    model_dir: PathBuf,
    /// Expected model filename
    model_filename: String,
    /// Observation size for the model
    obs_size: usize,
    /// Number of intra-op threads for ONNX inference (0 = auto-detect)
    intra_threads: usize,
    /// Shared evaluator to update
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Last known modification time of the loaded model file.
    /// Used by polling to detect changes.
    last_mtime: Arc<RwLock<Option<SystemTime>>>,
    /// Polling interval for checking file changes
    poll_interval: Duration,
    /// Current model info (None = metadata tracking disabled)
    model_info: Option<Arc<RwLock<ModelInfo>>>,
}

impl ModelWatcher {
    /// Create a new model watcher without metadata tracking.
    ///
    /// # Arguments
    /// * `model_dir` - Directory to watch for model files
    /// * `model_filename` - Name of the model file to watch (e.g., "latest.onnx")
    /// * `obs_size` - Observation size expected by the model
    /// * `intra_threads` - Number of intra-op threads for ONNX (0 = auto-detect)
    /// * `evaluator` - Shared evaluator reference to update on reload
    pub fn new(
        model_dir: impl AsRef<Path>,
        model_filename: impl Into<String>,
        obs_size: usize,
        intra_threads: usize,
        evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            model_filename: model_filename.into(),
            obs_size,
            intra_threads,
            evaluator,
            last_mtime: Arc::new(RwLock::new(None)),
            poll_interval: DEFAULT_POLL_INTERVAL,
            model_info: None,
        }
    }

    /// Enable metadata tracking and return `self` for chaining.
    ///
    /// When enabled, the watcher tracks model info (path, modification time,
    /// training step) that can be queried via [`model_info`](Self::model_info).
    pub fn with_metadata(mut self) -> Self {
        self.model_info = Some(Arc::new(RwLock::new(ModelInfo::default())));
        self
    }

    /// Set a custom polling interval.
    ///
    /// The default is 5 seconds. Shorter intervals provide faster detection
    /// but use more CPU. Longer intervals are more efficient but slower to
    /// detect changes.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Get the full path to the model file.
    pub fn model_path(&self) -> PathBuf {
        self.model_dir.join(&self.model_filename)
    }

    /// Get the current model info (if metadata tracking is enabled).
    pub fn model_info(&self) -> Arc<RwLock<ModelInfo>> {
        self.model_info
            .clone()
            .unwrap_or_else(|| Arc::new(RwLock::new(ModelInfo::default())))
    }

    /// Try to load the model if it exists.
    ///
    /// Returns `Ok(true)` if a model was loaded, `Ok(false)` if no model exists.
    pub fn try_load_existing(&self) -> Result<bool> {
        let path = self.model_path();
        if path.exists() {
            info!("Found existing model at {:?}", path);
            self.load_model(&path)?;
            Ok(true)
        } else {
            debug!("No existing model at {:?}", path);
            Ok(false)
        }
    }

    /// Extract metadata from a model file path.
    fn extract_metadata(path: &Path) -> (Option<u64>, Option<u32>) {
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

    /// Load a model from the given path.
    fn load_model(&self, path: &Path) -> Result<()> {
        Self::load_model_static(
            path,
            self.obs_size,
            self.intra_threads,
            &self.evaluator,
            &self.last_mtime,
            self.model_info.as_ref(),
        )
    }

    /// Start watching for model changes.
    ///
    /// This spawns two background tasks:
    /// 1. An inotify-based watcher for fast detection when events propagate
    /// 2. A polling fallback that checks file modification time periodically
    ///
    /// The polling fallback is essential for Docker environments where inotify
    /// events don't reliably propagate across container boundaries with bind mounts.
    ///
    /// Returns a channel that receives `()` when a new model is loaded.
    pub async fn start_watching(&self) -> Result<mpsc::Receiver<()>> {
        let (tx, rx) = mpsc::channel(16);
        let model_dir = self.model_dir.clone();
        let model_filename = self.model_filename.clone();
        let obs_size = self.obs_size;
        let intra_threads = self.intra_threads;
        let evaluator = Arc::clone(&self.evaluator);
        let last_mtime = Arc::clone(&self.last_mtime);
        let poll_interval = self.poll_interval;
        let model_info = self.model_info.clone();

        // Create channel for file system events
        let (fs_tx, mut fs_rx) = mpsc::channel(100);

        // Set up the file watcher
        let mut watcher = recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    // Non-blocking send - drop events if channel is full
                    let _ = fs_tx.blocking_send(event);
                }
                Err(e) => {
                    warn!("File watcher error: {}", e);
                }
            }
        })
        .map_err(|e| anyhow!("Failed to create file watcher: {}", e))?;

        // Create directory if it doesn't exist
        if !model_dir.exists() {
            std::fs::create_dir_all(&model_dir)
                .map_err(|e| anyhow!("Failed to create model directory: {}", e))?;
        }

        // Start watching the directory
        watcher
            .watch(&model_dir, RecursiveMode::NonRecursive)
            .map_err(|e| anyhow!("Failed to watch directory: {}", e))?;

        info!("Started watching {:?} for model updates", model_dir);

        // Clone for the inotify task
        let inotify_tx = tx.clone();
        let inotify_model_dir = model_dir.clone();
        let inotify_model_filename = model_filename.clone();
        let inotify_evaluator = Arc::clone(&evaluator);
        let inotify_last_mtime = Arc::clone(&last_mtime);
        let inotify_model_info = model_info.clone();

        // Spawn task to handle inotify file events
        tokio::spawn(async move {
            // Keep watcher alive in this task
            let _watcher = watcher;

            // Debounce timer to avoid rapid reloads
            let debounce_duration = Duration::from_millis(500);
            // Initialize the timer far enough in the past so the first
            // filesystem event is never dropped by the debounce guard.
            let mut last_reload = std::time::Instant::now() - debounce_duration;

            while let Some(event) = fs_rx.recv().await {
                // Only care about create/modify events
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {}
                    _ => continue,
                }

                // Check if this affects our model file
                let model_path = inotify_model_dir.join(&inotify_model_filename);
                let is_our_file = event
                    .paths
                    .iter()
                    .any(|p| p.file_name() == model_path.file_name());

                if !is_our_file {
                    continue;
                }

                // Debounce rapid events
                if last_reload.elapsed() < debounce_duration {
                    debug!("Debouncing model reload event (inotify)");
                    continue;
                }

                // Small delay to ensure file is fully written
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Verify file exists and is readable
                if !model_path.exists() {
                    debug!("Model file doesn't exist yet");
                    continue;
                }

                // Try to load the model
                info!("Model file changed (inotify), reloading {:?}", model_path);

                match Self::load_model_static(
                    &model_path,
                    obs_size,
                    intra_threads,
                    &inotify_evaluator,
                    &inotify_last_mtime,
                    inotify_model_info.as_ref(),
                ) {
                    Ok(()) => {
                        last_reload = std::time::Instant::now();
                        let _ = inotify_tx.send(()).await;
                    }
                    Err(e) => {
                        error!("Failed to reload model (inotify): {}", e);
                    }
                }
            }
        });

        // Spawn polling task as fallback for Docker/cross-container scenarios
        let poll_tx = tx;
        let poll_model_dir = model_dir;
        let poll_model_filename = model_filename;
        let poll_evaluator = evaluator;
        let poll_last_mtime = last_mtime;
        let poll_model_info = model_info;

        tokio::spawn(async move {
            info!(
                "Started polling fallback for model updates (interval: {:?})",
                poll_interval
            );

            let mut interval = tokio::time::interval(poll_interval);
            // Don't fire immediately - let inotify have first chance
            interval.tick().await;

            loop {
                interval.tick().await;

                let model_path = poll_model_dir.join(&poll_model_filename);

                // Check if file exists
                if !model_path.exists() {
                    debug!("Polling: model file doesn't exist yet");
                    continue;
                }

                // Get current file modification time
                let current_mtime = match model_path.metadata().and_then(|m| m.modified()) {
                    Ok(mtime) => mtime,
                    Err(e) => {
                        debug!("Polling: failed to get file mtime: {}", e);
                        continue;
                    }
                };

                // Check if file has changed since last load
                let needs_reload = {
                    let last = poll_last_mtime.read().ok();
                    match last.as_deref() {
                        Some(Some(last_time)) => current_mtime > *last_time,
                        Some(None) => true, // No model loaded yet
                        None => {
                            warn!("Polling: failed to read last_mtime lock");
                            continue;
                        }
                    }
                };

                if !needs_reload {
                    debug!("Polling: model file unchanged");
                    continue;
                }

                // Small delay to ensure file is fully written
                tokio::time::sleep(Duration::from_millis(100)).await;

                info!("Model file changed (polling), reloading {:?}", model_path);

                match Self::load_model_static(
                    &model_path,
                    obs_size,
                    intra_threads,
                    &poll_evaluator,
                    &poll_last_mtime,
                    poll_model_info.as_ref(),
                ) {
                    Ok(()) => {
                        let _ = poll_tx.send(()).await;
                    }
                    Err(e) => {
                        error!("Failed to reload model (polling): {}", e);
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Load a model and update shared state.
    ///
    /// This is a static method so it can be called from spawned tasks that
    /// don't hold a reference to `self`.
    fn load_model_static(
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
            let (file_modified, training_step) = Self::extract_metadata(path);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_model_watcher_creation() {
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator);

        assert_eq!(
            watcher.model_path(),
            PathBuf::from("/tmp/models/latest.onnx")
        );
    }

    #[test]
    fn test_try_load_nonexistent() {
        let temp_dir = tempdir().unwrap();
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new(
            temp_dir.path(),
            "nonexistent.onnx",
            29,
            1,
            evaluator.clone(),
        );

        let result = watcher.try_load_existing();
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should return false - no file

        // Evaluator should still be None
        let guard = evaluator.read().unwrap();
        assert!(guard.is_none());
    }

    #[test]
    fn test_model_info_default() {
        let info = ModelInfo::default();
        assert!(!info.loaded);
        assert!(info.path.is_none());
        assert!(info.file_modified.is_none());
        assert!(info.loaded_at.is_none());
        assert!(info.training_step.is_none());
    }

    #[test]
    fn test_extract_metadata_training_step() {
        // Test with step in filename
        let path = Path::new("/tmp/models/model_step_000100.onnx");
        let (_, training_step) = ModelWatcher::extract_metadata(path);
        assert_eq!(training_step, Some(100));

        // Test without step
        let path = Path::new("/tmp/models/latest.onnx");
        let (_, training_step) = ModelWatcher::extract_metadata(path);
        assert!(training_step.is_none());
    }

    #[test]
    fn test_with_metadata_creates_model_info() {
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator);
        assert!(watcher.model_info.is_none());

        let evaluator2 = Arc::new(RwLock::new(None));
        let watcher2 =
            ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator2).with_metadata();
        assert!(watcher2.model_info.is_some());
    }
}
