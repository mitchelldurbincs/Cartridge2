//! ONNX contract and information about the loaded immutable checkpoint.

use algorithm_core::ModelArtifactContract;
use anyhow::{anyhow, Result};
use mcts::OnnxEvaluator;
use std::path::Path;

/// Validated environment/model shape needed to load and independently verify
/// one inference checkpoint.
#[derive(Clone)]
pub struct ModelLoadSpec {
    pub(crate) obs_size: usize,
    pub(crate) num_actions: usize,
    pub(crate) intra_threads: usize,
    pub(crate) environment_max_horizon: u32,
    pub(crate) identity: ModelArtifactContract,
}

impl ModelLoadSpec {
    pub fn new(
        obs_size: usize,
        num_actions: usize,
        intra_threads: usize,
        environment_max_horizon: u32,
        identity: ModelArtifactContract,
    ) -> Result<Self> {
        if obs_size == 0 || num_actions == 0 || intra_threads == 0 {
            return Err(anyhow!(
                "model observation/action dimensions and ONNX threads must be positive"
            ));
        }
        if environment_max_horizon == 0 {
            return Err(anyhow!("runtime environment max_horizon must be positive"));
        }
        Ok(Self {
            obs_size,
            num_actions,
            intra_threads,
            environment_max_horizon,
            identity,
        })
    }

    pub(crate) fn load(&self, path: &Path) -> Result<OnnxEvaluator> {
        OnnxEvaluator::load_from_file(
            path,
            self.obs_size,
            self.num_actions,
            self.intra_threads,
            &self.identity,
        )
        .map_err(|error| anyhow!("failed to load ONNX model: {error}"))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub loaded: bool,
    /// Content identity of the complete checkpoint manifest.
    pub checkpoint_id: Option<String>,
    /// Digest of the immutable ONNX blob.
    pub model_sha256: Option<String>,
    pub path: Option<String>,
    pub loaded_at: Option<u64>,
    pub training_step: Option<u64>,
}
