//! S3/MinIO backend for model watching.
//!
//! This module provides S3-based model watching for Kubernetes deployments
//! where models are stored in S3-compatible storage (AWS S3, MinIO, etc.).
//!
//! Unlike the filesystem watcher, S3 watching is polling-only since S3
//! doesn't support filesystem events.

use anyhow::{anyhow, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use mcts::OnnxEvaluator;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

#[cfg(feature = "metadata")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "metadata")]
use crate::ModelInfo;

mod download;

#[cfg(test)]
mod tests;

pub use download::S3Config;

/// Default polling interval for S3 model checks.
const DEFAULT_S3_POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Default ONNX intra-op thread setting for S3-backed model loading.
///
/// `0` delegates to the evaluator's auto-detection logic, which matches the
/// behavior of the filesystem watcher when no explicit thread count is wired in.
const DEFAULT_ONNX_INTRA_THREADS: usize = 0;

/// S3-backed model watcher for Kubernetes deployments.
///
/// Polls an S3 bucket for model updates and hot-reloads them.
pub struct S3ModelWatcher {
    /// S3 client
    client: Client,
    /// S3 bucket name
    bucket: String,
    /// S3 object key for the model (e.g., "models/latest.onnx")
    key: String,
    /// Observation size for the model
    obs_size: usize,
    /// Shared evaluator to update
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Last known ETag of the model
    last_etag: Arc<RwLock<Option<String>>>,
    /// Polling interval
    poll_interval: Duration,
    /// Local cache directory for downloaded models
    cache_dir: PathBuf,
    /// Current model info (only with metadata feature)
    #[cfg(feature = "metadata")]
    model_info: Arc<RwLock<ModelInfo>>,
}

impl S3ModelWatcher {
    /// Create a new S3 model watcher.
    ///
    /// # Arguments
    /// * `config` - S3 configuration
    /// * `obs_size` - Observation size expected by the model
    /// * `evaluator` - Shared evaluator reference to update on reload
    pub async fn new(
        config: S3Config,
        obs_size: usize,
        evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Result<Self> {
        let mut sdk_config_loader = aws_config::defaults(BehaviorVersion::latest()).region(
            aws_sdk_s3::config::Region::new(
                config
                    .region
                    .clone()
                    .unwrap_or_else(|| "us-east-1".to_string()),
            ),
        );

        // Use custom endpoint for MinIO/LocalStack
        if let Some(endpoint) = &config.endpoint_url {
            sdk_config_loader = sdk_config_loader.endpoint_url(endpoint);
        }

        let sdk_config = sdk_config_loader.load().await;
        let mut s3_config_builder = aws_sdk_s3::config::Builder::from(&sdk_config);

        // Force path-style access for MinIO compatibility
        if config.endpoint_url.is_some() {
            s3_config_builder = s3_config_builder.force_path_style(true);
        }

        let client = Client::from_conf(s3_config_builder.build());

        // Ensure cache directory exists
        tokio::fs::create_dir_all(&config.cache_dir).await?;

        Ok(Self {
            client,
            bucket: config.bucket,
            key: config.key,
            obs_size,
            evaluator,
            last_etag: Arc::new(RwLock::new(None)),
            poll_interval: DEFAULT_S3_POLL_INTERVAL,
            cache_dir: config.cache_dir,
            #[cfg(feature = "metadata")]
            model_info: Arc::new(RwLock::new(ModelInfo::default())),
        })
    }

    /// Set a custom polling interval.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Get the current model info.
    #[cfg(feature = "metadata")]
    pub fn model_info(&self) -> Arc<RwLock<ModelInfo>> {
        Arc::clone(&self.model_info)
    }

    /// Try to load the model if it exists in S3.
    ///
    /// Returns `Ok(true)` if a model was loaded, `Ok(false)` if no model exists.
    pub async fn try_load_existing(&self) -> Result<bool> {
        match self.check_and_download().await {
            Ok(Some(path)) => {
                self.load_model(&path)?;
                Ok(true)
            }
            Ok(None) => {
                debug!("No model found in S3 at s3://{}/{}", self.bucket, self.key);
                Ok(false)
            }
            Err(e) => {
                warn!("Failed to check S3 for model: {}", e);
                Ok(false)
            }
        }
    }

    /// Load a model from the given local path.
    fn load_model(&self, path: &PathBuf) -> Result<()> {
        info!("Loading model from {:?}", path);

        #[cfg(feature = "metadata")]
        let training_step = Self::extract_training_step(path);

        let new_evaluator = OnnxEvaluator::load(path, self.obs_size, DEFAULT_ONNX_INTRA_THREADS)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        {
            let mut guard = self
                .evaluator
                .write()
                .map_err(|e| anyhow!("Failed to acquire evaluator write lock: {}", e))?;
            *guard = Some(new_evaluator);
        }

        #[cfg(feature = "metadata")]
        {
            let mut info = self
                .model_info
                .write()
                .map_err(|e| anyhow!("Failed to acquire model_info write lock: {}", e))?;
            *info = ModelInfo {
                loaded: true,
                path: Some(format!("s3://{}/{}", self.bucket, self.key)),
                file_modified: None, // S3 doesn't provide this easily
                loaded_at: Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                ),
                training_step,
            };
        }

        info!("Model loaded successfully from S3");
        Ok(())
    }

    /// Start watching for model changes in S3.
    ///
    /// Returns a channel that receives `()` when a new model is loaded.
    pub async fn start_watching(&self) -> Result<mpsc::Receiver<()>> {
        let (tx, rx) = mpsc::channel(16);

        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let key = self.key.clone();
        let obs_size = self.obs_size;
        let evaluator = Arc::clone(&self.evaluator);
        let last_etag = Arc::clone(&self.last_etag);
        let poll_interval = self.poll_interval;
        let cache_dir = self.cache_dir.clone();
        #[cfg(feature = "metadata")]
        let model_info = Arc::clone(&self.model_info);

        tokio::spawn(async move {
            info!(
                "Started S3 model watcher for s3://{}/{} (interval: {:?})",
                bucket, key, poll_interval
            );

            let mut interval = tokio::time::interval(poll_interval);

            loop {
                interval.tick().await;

                match Self::check_and_download_static(
                    &client, &bucket, &key, &last_etag, &cache_dir,
                )
                .await
                {
                    Ok(Some(path)) => {
                        #[cfg(feature = "metadata")]
                        let result = Self::load_model_static(
                            &path,
                            obs_size,
                            &evaluator,
                            &bucket,
                            &key,
                            &model_info,
                        );

                        #[cfg(not(feature = "metadata"))]
                        let result = Self::load_model_static(&path, obs_size, &evaluator);

                        match result {
                            Ok(()) => {
                                let _ = tx.send(()).await;
                            }
                            Err(e) => {
                                error!("Failed to load model from S3: {}", e);
                            }
                        }
                    }
                    Ok(None) => {
                        debug!("S3 model unchanged");
                    }
                    Err(e) => {
                        warn!("Failed to check S3 for model updates: {}", e);
                    }
                }
            }
        });

        Ok(rx)
    }
}
