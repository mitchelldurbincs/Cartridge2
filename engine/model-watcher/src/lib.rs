//! Content-addressed ONNX checkpoint watching.
//!
//! Runtimes watch channels/current.json, not a mutable model file. A run-head
//! update is accepted only after its canonical identity, immutable manifest,
//! exact runtime profile, blob size, blob SHA-256, and ONNX contract all pass.

use anyhow::{anyhow, Result};
use mcts::OnnxEvaluator;
use notify::{recommended_watcher, Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

mod artifact;
mod load;

use artifact::{read_filesystem_head, resolve_filesystem_head, run_head_path, ResolvedCheckpoint};

#[cfg(feature = "s3")]
pub mod s3;

#[cfg(test)]
mod tests;

pub use load::{ModelInfo, ModelLoadSpec};

const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadOutcome {
    Absent,
    Unchanged,
    Advanced,
    Loaded,
}

/// Policy for selecting an inference checkpoint from the authoritative RunHead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSelection {
    /// Load the checkpoint selected directly by the latest RunCommit.
    Latest,
    /// Load the latest RunCommit's champion, falling back to latest before one exists.
    ChampionOrLatest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AcceptedHead {
    model_checkpoint_id: String,
    run_commit_id: String,
}

pub struct ModelWatcher {
    model_root: PathBuf,
    model_spec: ModelLoadSpec,
    selection: ModelSelection,
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    accepted_head: Arc<RwLock<Option<AcceptedHead>>>,
    poll_interval: Duration,
    model_info: Arc<RwLock<ModelInfo>>,
}

impl ModelWatcher {
    pub fn new(
        model_root: impl AsRef<Path>,
        model_spec: ModelLoadSpec,
        selection: ModelSelection,
        evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Self {
        Self {
            model_root: model_root.as_ref().to_path_buf(),
            model_spec,
            selection,
            evaluator,
            accepted_head: Arc::new(RwLock::new(None)),
            poll_interval: DEFAULT_POLL_INTERVAL,
            model_info: Arc::new(RwLock::new(ModelInfo::default())),
        }
    }

    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    pub fn run_head_path(&self) -> PathBuf {
        run_head_path(&self.model_root)
    }

    pub fn model_info(&self) -> Arc<RwLock<ModelInfo>> {
        Arc::clone(&self.model_info)
    }

    pub fn try_load_existing(&self) -> Result<bool> {
        Ok(!matches!(self.load_current()?, LoadOutcome::Absent))
    }

    fn load_current(&self) -> Result<LoadOutcome> {
        Self::load_current_static(
            &self.model_root,
            &self.model_spec,
            self.selection,
            &self.evaluator,
            &self.accepted_head,
            &self.model_info,
        )
    }

    fn current_accepted_head(
        accepted_head: &Arc<RwLock<Option<AcceptedHead>>>,
    ) -> Result<Option<AcceptedHead>> {
        accepted_head
            .read()
            .map(|guard| guard.clone())
            .map_err(|error| anyhow!("failed to read accepted run head: {error}"))
    }

    fn commit_candidate(
        candidate: ResolvedCheckpoint,
        new_evaluator: Option<OnnxEvaluator>,
        expected_accepted_head: Option<AcceptedHead>,
        evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
        accepted_head: &Arc<RwLock<Option<AcceptedHead>>>,
        model_info: &Arc<RwLock<ModelInfo>>,
    ) -> Result<LoadOutcome> {
        let mut accepted_guard = accepted_head
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
                candidate_checkpoint = %candidate.checkpoint_id,
                candidate_run_commit = %candidate.run_commit_id,
                "Discarding candidate because another load advanced the accepted RunHead"
            );
            return Ok(LoadOutcome::Unchanged);
        }
        let mut evaluator_guard = evaluator
            .write()
            .map_err(|error| anyhow!("failed to lock model evaluator: {error}"))?;
        let mut info_guard = model_info
            .write()
            .map_err(|error| anyhow!("failed to lock model information: {error}"))?;

        if new_evaluator.is_none()
            && (accepted_guard
                .as_ref()
                .is_none_or(|accepted| accepted.model_checkpoint_id != candidate.checkpoint_id)
                || evaluator_guard.is_none())
        {
            return Err(anyhow!(
                "cannot reuse an evaluator that does not match the selected checkpoint"
            ));
        }
        let model_changed = new_evaluator.is_some();
        let loaded_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        if let Some(new_evaluator) = new_evaluator {
            *evaluator_guard = Some(new_evaluator);
        }
        *accepted_guard = Some(AcceptedHead {
            model_checkpoint_id: candidate.checkpoint_id.clone(),
            run_commit_id: candidate.run_commit_id.clone(),
        });
        if model_changed {
            *info_guard = ModelInfo {
                loaded: true,
                checkpoint_id: Some(candidate.checkpoint_id.clone()),
                model_sha256: Some(candidate.manifest.onnx.sha256.clone()),
                path: Some(candidate.model_path.to_string_lossy().into_owned()),
                loaded_at: Some(loaded_at),
                training_step: Some(candidate.manifest.step),
            };
        }

        info!(
            checkpoint_id = %candidate.checkpoint_id,
            run_commit_id = %candidate.run_commit_id,
            model_sha256 = %candidate.manifest.onnx.sha256,
            step = candidate.manifest.step,
            path = %candidate.model_path.display(),
            model_changed,
            "Content-addressed RunHead accepted"
        );
        Ok(if model_changed {
            LoadOutcome::Loaded
        } else {
            LoadOutcome::Advanced
        })
    }

    fn load_current_static(
        model_root: &Path,
        model_spec: &ModelLoadSpec,
        selection: ModelSelection,
        evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
        accepted_head: &Arc<RwLock<Option<AcceptedHead>>>,
        model_info: &Arc<RwLock<ModelInfo>>,
    ) -> Result<LoadOutcome> {
        let Some(head) = read_filesystem_head(model_root)? else {
            return Ok(LoadOutcome::Absent);
        };
        let accepted = Self::current_accepted_head(accepted_head)?;
        if accepted
            .as_ref()
            .is_some_and(|accepted| accepted.run_commit_id == head.run_commit_id)
        {
            return Ok(LoadOutcome::Unchanged);
        }

        let candidate = resolve_filesystem_head(
            model_root,
            head.clone(),
            &model_spec.identity,
            model_spec.environment_max_horizon,
            selection,
        )?;
        let new_evaluator = if accepted
            .as_ref()
            .is_some_and(|accepted| accepted.model_checkpoint_id == candidate.checkpoint_id)
        {
            None
        } else {
            Some(model_spec.load(&candidate.model_path)?)
        };

        let latest = read_filesystem_head(model_root)?
            .ok_or_else(|| anyhow!("current run head disappeared during model validation"))?;
        if latest != head {
            debug!(
                candidate = %candidate.checkpoint_id,
                current = %latest.checkpoint_id,
                candidate_run_commit = %candidate.run_commit_id,
                current_run_commit = %latest.run_commit_id,
                "Discarding stale run-head candidate"
            );
            return Ok(LoadOutcome::Unchanged);
        }

        Self::commit_candidate(
            candidate,
            new_evaluator,
            accepted,
            evaluator,
            accepted_head,
            model_info,
        )
    }

    pub async fn start_watching(&self) -> Result<mpsc::Receiver<()>> {
        let (updates_tx, updates_rx) = mpsc::channel(16);
        let channels_dir = self.model_root.join("channels");
        std::fs::create_dir_all(&channels_dir).map_err(|error| {
            anyhow!(
                "failed to create model channel directory {:?}: {error}",
                channels_dir
            )
        })?;

        let (events_tx, mut events_rx) = mpsc::channel(100);
        let mut watcher =
            recommended_watcher(move |result: Result<Event, notify::Error>| match result {
                Ok(event) => {
                    let _ = events_tx.blocking_send(event);
                }
                Err(error) => warn!("Model channel watcher error: {error}"),
            })
            .map_err(|error| anyhow!("failed to create model channel watcher: {error}"))?;
        watcher
            .watch(&channels_dir, RecursiveMode::NonRecursive)
            .map_err(|error| anyhow!("failed to watch model channel directory: {error}"))?;

        let event_root = self.model_root.clone();
        let event_spec = self.model_spec.clone();
        let event_selection = self.selection;
        let event_evaluator = Arc::clone(&self.evaluator);
        let event_accepted = Arc::clone(&self.accepted_head);
        let event_info = Arc::clone(&self.model_info);
        let event_updates = updates_tx.clone();
        tokio::spawn(async move {
            let _watcher = watcher;
            let expected_path = run_head_path(&event_root);
            while let Some(event) = events_rx.recv().await {
                if !matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_))
                    || !event.paths.iter().any(|path| path == &expected_path)
                {
                    continue;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                match Self::load_current_static(
                    &event_root,
                    &event_spec,
                    event_selection,
                    &event_evaluator,
                    &event_accepted,
                    &event_info,
                ) {
                    Ok(LoadOutcome::Loaded) => {
                        let _ = event_updates.send(()).await;
                    }
                    Ok(LoadOutcome::Absent | LoadOutcome::Unchanged | LoadOutcome::Advanced) => {}
                    Err(error) => error!("Rejected model channel update: {error}"),
                }
            }
        });

        let poll_root = self.model_root.clone();
        let poll_spec = self.model_spec.clone();
        let poll_selection = self.selection;
        let poll_evaluator = Arc::clone(&self.evaluator);
        let poll_accepted = Arc::clone(&self.accepted_head);
        let poll_info = Arc::clone(&self.model_info);
        let poll_interval = self.poll_interval;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(poll_interval);
            interval.tick().await;
            loop {
                interval.tick().await;
                match Self::load_current_static(
                    &poll_root,
                    &poll_spec,
                    poll_selection,
                    &poll_evaluator,
                    &poll_accepted,
                    &poll_info,
                ) {
                    Ok(LoadOutcome::Loaded) => {
                        let _ = updates_tx.send(()).await;
                    }
                    Ok(LoadOutcome::Absent | LoadOutcome::Unchanged | LoadOutcome::Advanced) => {}
                    Err(error) => error!("Rejected polled model channel update: {error}"),
                }
            }
        });

        info!(
            run_head = %self.run_head_path().display(),
            poll_interval = ?self.poll_interval,
            "Started content-addressed model watcher"
        );
        Ok(updates_rx)
    }
}
