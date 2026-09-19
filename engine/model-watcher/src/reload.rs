//! Publication of a validated candidate, shared by filesystem and S3 watchers.
//!
//! Repository reads, artifact validation, selection, ONNX loading, and the final
//! authority reread stay with the caller. This module owns only the in-memory
//! transition and its compare-and-set/reuse guards.

use anyhow::{anyhow, Result};
use mcts::SharedOnnxEvaluator;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

use crate::artifact::CheckpointManifestV1;
use crate::ModelInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadOutcome {
    Absent,
    Unchanged,
    Advanced,
    Loaded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AcceptedHead {
    pub model_checkpoint_id: String,
    pub run_commit_id: String,
}

#[derive(Clone)]
pub(crate) struct ReloadState {
    evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    accepted_head: Arc<RwLock<Option<AcceptedHead>>>,
    model_info: Arc<RwLock<ModelInfo>>,
}

impl ReloadState {
    pub fn new(evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>) -> Self {
        Self {
            evaluator,
            accepted_head: Arc::new(RwLock::new(None)),
            model_info: Arc::new(RwLock::new(ModelInfo::default())),
        }
    }

    pub fn model_info(&self) -> Arc<RwLock<ModelInfo>> {
        Arc::clone(&self.model_info)
    }

    pub fn accepted_head(&self) -> Result<Option<AcceptedHead>> {
        self.accepted_head
            .read()
            .map(|guard| guard.clone())
            .map_err(|error| anyhow!("failed to read accepted run head: {error}"))
    }

    /// Publish only after the caller has validated and reread its authority.
    /// `None` reuses the evaluator only for the exact already-loaded checkpoint.
    pub fn commit_candidate(
        &self,
        candidate: AcceptedHead,
        manifest: &CheckpointManifestV1,
        model_location: String,
        new_evaluator: Option<SharedOnnxEvaluator>,
        expected_accepted_head: Option<AcceptedHead>,
    ) -> Result<LoadOutcome> {
        let mut accepted_guard = self
            .accepted_head
            .write()
            .map_err(|error| anyhow!("failed to lock accepted run head: {error}"))?;
        if accepted_guard
            .as_ref()
            .is_some_and(|accepted| accepted.run_commit_id == candidate.run_commit_id)
        {
            return Ok(LoadOutcome::Unchanged);
        }
        if *accepted_guard != expected_accepted_head {
            debug!(
                candidate_checkpoint = %candidate.model_checkpoint_id,
                candidate_run_commit = %candidate.run_commit_id,
                "Discarding candidate because another load advanced the accepted RunHead"
            );
            return Ok(LoadOutcome::Unchanged);
        }

        // Keep one lock order for both transports. Acquire every lock and check
        // reuse before changing any state, so errors preserve the last good load.
        let mut evaluator_guard = self
            .evaluator
            .write()
            .map_err(|error| anyhow!("failed to lock model evaluator: {error}"))?;
        let mut info_guard = self
            .model_info
            .write()
            .map_err(|error| anyhow!("failed to lock model information: {error}"))?;
        if new_evaluator.is_none()
            && (accepted_guard.as_ref().is_none_or(|accepted| {
                accepted.model_checkpoint_id != candidate.model_checkpoint_id
            }) || evaluator_guard.is_none())
        {
            return Err(anyhow!(
                "cannot reuse an evaluator that does not match the selected checkpoint"
            ));
        }
        let model_changed = new_evaluator.is_some();
        if let Some(new_evaluator) = new_evaluator {
            *evaluator_guard = Some(new_evaluator);
        }
        *accepted_guard = Some(candidate.clone());
        if model_changed {
            *info_guard = ModelInfo {
                loaded: true,
                checkpoint_id: Some(candidate.model_checkpoint_id.clone()),
                model_sha256: Some(manifest.onnx.sha256.clone()),
                path: Some(model_location.clone()),
                loaded_at: Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|duration| duration.as_secs())
                        .unwrap_or(0),
                ),
                training_step: Some(manifest.step),
            };
        }

        info!(
            checkpoint_id = %candidate.model_checkpoint_id,
            run_commit_id = %candidate.run_commit_id,
            model_sha256 = %manifest.onnx.sha256,
            step = manifest.step,
            path = %model_location,
            model_changed,
            "Content-addressed RunHead accepted"
        );
        Ok(if model_changed {
            LoadOutcome::Loaded
        } else {
            LoadOutcome::Advanced
        })
    }
}

#[cfg(test)]
mod tests;
