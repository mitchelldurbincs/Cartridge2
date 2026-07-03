//! S3 configuration plus download/ETag and model-loading helpers.
//!
//! This submodule holds [`S3Config`] and the download/ETag comparison logic
//! for [`S3ModelWatcher`](super::S3ModelWatcher), factored out of the parent
//! module so the watcher's polling/orchestration logic stays focused.

use anyhow::{anyhow, Result};
use aws_sdk_s3::Client;
use mcts::OnnxEvaluator;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tracing::info;

use super::{S3ModelWatcher, DEFAULT_ONNX_INTRA_THREADS};

#[cfg(feature = "metadata")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "metadata")]
use crate::ModelInfo;

/// Configuration for S3 model watcher.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// S3 object key (path within bucket)
    pub key: String,
    /// Optional custom endpoint URL (for MinIO, LocalStack, etc.)
    pub endpoint_url: Option<String>,
    /// Optional region (defaults to us-east-1)
    pub region: Option<String>,
    /// Local directory to cache downloaded models
    pub cache_dir: PathBuf,
}

impl S3ModelWatcher {
    /// Check S3 for model updates and download if changed.
    ///
    /// Returns the local path if a new model was downloaded.
    pub(super) async fn check_and_download(&self) -> Result<Option<PathBuf>> {
        // HEAD request to check if object exists and get ETag
        let head_result = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&self.key)
            .send()
            .await;

        let head = match head_result {
            Ok(h) => h,
            Err(e) => {
                // Check if it's a "not found" error
                if e.to_string().contains("NotFound") || e.to_string().contains("NoSuchKey") {
                    return Ok(None);
                }
                return Err(anyhow!("Failed to HEAD S3 object: {}", e));
            }
        };

        let current_etag = head.e_tag().map(|s| s.to_string());

        // Check if ETag has changed
        let needs_download = {
            let last = self
                .last_etag
                .read()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            match (&*last, &current_etag) {
                (Some(last_etag), Some(curr_etag)) => last_etag != curr_etag,
                (None, Some(_)) => true,
                _ => false,
            }
        };

        if !needs_download {
            return Ok(None);
        }

        info!(
            "Downloading model from s3://{}/{} (ETag: {:?})",
            self.bucket, self.key, current_etag
        );

        // Download the object
        let get_result = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&self.key)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to GET S3 object: {}", e))?;

        let body = get_result
            .body
            .collect()
            .await
            .map_err(|e| anyhow!("Failed to read S3 object body: {}", e))?;

        // Write to local cache
        let local_path = self.cache_dir.join("latest.onnx");
        let temp_path = self.cache_dir.join("latest.onnx.tmp");

        tokio::fs::write(&temp_path, body.into_bytes())
            .await
            .map_err(|e| anyhow!("Failed to write model to cache: {}", e))?;

        // Atomic rename
        tokio::fs::rename(&temp_path, &local_path)
            .await
            .map_err(|e| anyhow!("Failed to rename temp file: {}", e))?;

        // Update last ETag
        {
            let mut last = self
                .last_etag
                .write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            *last = current_etag;
        }

        Ok(Some(local_path))
    }

    /// Extract training step from model key if present.
    #[cfg(feature = "metadata")]
    pub(super) fn extract_training_step(path: &Path) -> Option<u32> {
        path.file_name().and_then(|n| n.to_str()).and_then(|name| {
            name.split('_')
                .filter_map(|part| part.trim_end_matches(".onnx").parse::<u32>().ok())
                .next_back()
        })
    }

    /// Static method to check and download from S3.
    pub(super) async fn check_and_download_static(
        client: &Client,
        bucket: &str,
        key: &str,
        last_etag: &Arc<RwLock<Option<String>>>,
        cache_dir: &Path,
    ) -> Result<Option<PathBuf>> {
        let head_result = client.head_object().bucket(bucket).key(key).send().await;

        let head = match head_result {
            Ok(h) => h,
            Err(e) => {
                if e.to_string().contains("NotFound") || e.to_string().contains("NoSuchKey") {
                    return Ok(None);
                }
                return Err(anyhow!("Failed to HEAD S3 object: {}", e));
            }
        };

        let current_etag = head.e_tag().map(|s| s.to_string());

        let needs_download = {
            let last = last_etag.read().map_err(|e| anyhow!("Lock error: {}", e))?;
            match (&*last, &current_etag) {
                (Some(last_e), Some(curr_e)) => last_e != curr_e,
                (None, Some(_)) => true,
                _ => false,
            }
        };

        if !needs_download {
            return Ok(None);
        }

        info!(
            "Downloading model from s3://{}/{} (ETag: {:?})",
            bucket, key, current_etag
        );

        let get_result = client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to GET S3 object: {}", e))?;

        let body = get_result
            .body
            .collect()
            .await
            .map_err(|e| anyhow!("Failed to read S3 object body: {}", e))?;

        let local_path = cache_dir.join("latest.onnx");
        let temp_path = cache_dir.join("latest.onnx.tmp");

        tokio::fs::write(&temp_path, body.into_bytes())
            .await
            .map_err(|e| anyhow!("Failed to write model to cache: {}", e))?;

        tokio::fs::rename(&temp_path, &local_path)
            .await
            .map_err(|e| anyhow!("Failed to rename temp file: {}", e))?;

        {
            let mut last = last_etag
                .write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            *last = current_etag;
        }

        Ok(Some(local_path))
    }

    /// Static method to load model (for use in spawned task).
    #[cfg(feature = "metadata")]
    pub(super) fn load_model_static(
        path: &PathBuf,
        obs_size: usize,
        evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
        bucket: &str,
        key: &str,
        model_info: &Arc<RwLock<ModelInfo>>,
    ) -> Result<()> {
        let training_step = Self::extract_training_step(path);

        let new_evaluator = OnnxEvaluator::load(path, obs_size, DEFAULT_ONNX_INTRA_THREADS)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        {
            let mut guard = evaluator
                .write()
                .map_err(|e| anyhow!("Failed to acquire evaluator write lock: {}", e))?;
            *guard = Some(new_evaluator);
        }

        {
            let mut info = model_info
                .write()
                .map_err(|e| anyhow!("Failed to acquire model_info write lock: {}", e))?;
            *info = ModelInfo {
                loaded: true,
                path: Some(format!("s3://{}/{}", bucket, key)),
                file_modified: None,
                loaded_at: Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                ),
                training_step,
            };
        }

        info!("Model reloaded from S3 (step: {:?})", training_step);
        Ok(())
    }

    /// Static method to load model (for use in spawned task) - without metadata.
    #[cfg(not(feature = "metadata"))]
    pub(super) fn load_model_static(
        path: &PathBuf,
        obs_size: usize,
        evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Result<()> {
        let new_evaluator = OnnxEvaluator::load(path, obs_size, DEFAULT_ONNX_INTRA_THREADS)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        {
            let mut guard = evaluator
                .write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
            *guard = Some(new_evaluator);
        }

        info!("Model reloaded from S3");
        Ok(())
    }
}
