//! S3/MinIO watcher for content-addressed checkpoints selected by RunHeadV2.

use anyhow::{anyhow, bail, Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::{error::ProvideErrorMetadata, operation::get_object::GetObjectError, Client};
use mcts::SharedOnnxEvaluator;
use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::artifact::{
    parse_canonical_json, parse_run_commit, select_inference_checkpoint, sha256_hex,
    validate_manifest, validate_run_head, validate_selected_head, validate_spliced_chain,
    verify_blob_bytes, ChainCache, CheckpointManifestV1, ResolvedRunCommit, RunHeadV2,
    RUN_HEAD_CHANNEL,
};
use crate::{AcceptedHead, ModelInfo, ModelLoadSpec, ModelSelection};

#[cfg(test)]
mod tests;

const DEFAULT_S3_POLL_INTERVAL: Duration = Duration::from_secs(10);
static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct S3Config {
    pub bucket: String,
    /// Runtime profile prefix that contains blobs/, manifests/, and channels/.
    pub prefix: String,
    pub endpoint_url: Option<String>,
    pub region: Option<String>,
    pub cache_dir: PathBuf,
}

enum RemoteUpdate {
    Absent,
    Unchanged,
    Advanced,
    Loaded,
}

struct S3LoadContext<'a> {
    client: &'a Client,
    bucket: &'a str,
    prefix: &'a str,
    cache_dir: &'a Path,
    model_spec: &'a ModelLoadSpec,
    selection: ModelSelection,
    evaluator: &'a Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    accepted_head: &'a Arc<RwLock<Option<AcceptedHead>>>,
    model_info: &'a Arc<RwLock<ModelInfo>>,
    chain_cache: &'a Arc<RwLock<ChainCache>>,
}

pub struct S3ModelWatcher {
    client: Client,
    bucket: String,
    prefix: String,
    model_spec: ModelLoadSpec,
    selection: ModelSelection,
    evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    accepted_head: Arc<RwLock<Option<AcceptedHead>>>,
    poll_interval: Duration,
    cache_dir: PathBuf,
    model_info: Arc<RwLock<ModelInfo>>,
    chain_cache: Arc<RwLock<ChainCache>>,
}

impl S3ModelWatcher {
    pub async fn new(
        config: S3Config,
        model_spec: ModelLoadSpec,
        selection: ModelSelection,
        evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    ) -> Result<Self> {
        if config.bucket.trim().is_empty() {
            bail!("S3 model watcher bucket cannot be empty");
        }
        if config.prefix.trim().is_empty()
            || config.prefix.starts_with('/')
            || config.prefix.ends_with('/')
        {
            bail!("S3 model watcher prefix must be a non-empty relative prefix without a trailing slash");
        }

        let mut loader = aws_config::defaults(BehaviorVersion::latest()).region(
            aws_sdk_s3::config::Region::new(
                config
                    .region
                    .clone()
                    .unwrap_or_else(|| "us-east-1".to_string()),
            ),
        );
        if let Some(endpoint) = &config.endpoint_url {
            loader = loader.endpoint_url(endpoint);
        }
        let sdk_config = loader.load().await;
        let mut builder = aws_sdk_s3::config::Builder::from(&sdk_config);
        if config.endpoint_url.is_some() {
            builder = builder.force_path_style(true);
        }

        Ok(Self {
            client: Client::from_conf(builder.build()),
            bucket: config.bucket,
            prefix: config.prefix,
            model_spec,
            selection,
            evaluator,
            accepted_head: Arc::new(RwLock::new(None)),
            poll_interval: DEFAULT_S3_POLL_INTERVAL,
            cache_dir: config.cache_dir,
            model_info: Arc::new(RwLock::new(ModelInfo::default())),
            chain_cache: Arc::new(RwLock::new(ChainCache::default())),
        })
    }

    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    pub fn model_info(&self) -> Arc<RwLock<ModelInfo>> {
        Arc::clone(&self.model_info)
    }

    fn run_head_key(prefix: &str) -> String {
        format!("{prefix}/channels/{RUN_HEAD_CHANNEL}.json")
    }

    fn manifest_key(prefix: &str, checkpoint_id: &str) -> String {
        format!("{prefix}/manifests/sha256/{checkpoint_id}.json")
    }

    fn run_commit_key(prefix: &str, run_commit_id: &str) -> String {
        format!("{prefix}/run-commits/sha256/{run_commit_id}.json")
    }

    fn model_key(prefix: &str, digest: &str) -> String {
        format!("{prefix}/blobs/sha256/{digest}.onnx")
    }

    fn is_absent(error: &GetObjectError) -> bool {
        error.is_no_such_key() || matches!(error.code(), Some("NoSuchKey" | "NotFound" | "404"))
    }

    async fn get_optional(client: &Client, bucket: &str, key: &str) -> Result<Option<Vec<u8>>> {
        let response = match client.get_object().bucket(bucket).key(key).send().await {
            Ok(response) => response,
            Err(error) if error.as_service_error().is_some_and(Self::is_absent) => return Ok(None),
            Err(error) => return Err(anyhow!("failed to GET s3://{bucket}/{key}: {error}")),
        };
        let body = response
            .body
            .collect()
            .await
            .with_context(|| format!("failed to read s3://{bucket}/{key}"))?;
        Ok(Some(body.into_bytes().to_vec()))
    }

    async fn get_required(client: &Client, bucket: &str, key: &str) -> Result<Vec<u8>> {
        Self::get_optional(client, bucket, key)
            .await?
            .ok_or_else(|| anyhow!("required immutable object s3://{bucket}/{key} does not exist"))
    }

    async fn cache_model(cache_dir: &Path, digest: &str, bytes: &[u8]) -> Result<PathBuf> {
        let directory = cache_dir.join("blobs").join("sha256");
        tokio::fs::create_dir_all(&directory)
            .await
            .with_context(|| format!("failed to create model cache {directory:?}"))?;
        let path = directory.join(format!("{digest}.onnx"));
        if let Ok(existing) = tokio::fs::read(&path).await {
            if sha256_hex(&existing) != digest {
                bail!("immutable model cache blob {path:?} has the wrong digest");
            }
            return Ok(path);
        }

        let counter = CACHE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary = directory.join(format!(".{digest}.{}.{}.tmp", std::process::id(), counter));
        tokio::fs::write(&temporary, bytes)
            .await
            .with_context(|| format!("failed to write model cache candidate {temporary:?}"))?;
        match tokio::fs::hard_link(&temporary, &path).await {
            Ok(()) => {
                tokio::fs::remove_file(&temporary).await.with_context(|| {
                    format!("failed to remove linked model cache candidate {temporary:?}")
                })?;
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                let existing = tokio::fs::read(&path)
                    .await
                    .with_context(|| format!("failed to verify raced model cache {path:?}"))?;
                let _ = tokio::fs::remove_file(&temporary).await;
                if sha256_hex(&existing) != digest {
                    bail!("raced immutable model cache blob {path:?} has the wrong digest");
                }
            }
            Err(error) => {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error)
                    .with_context(|| format!("failed to publish model cache blob {path:?}"));
            }
        }
        Ok(path)
    }

    async fn resolve_run_commit_chain(
        client: &Client,
        bucket: &str,
        prefix: &str,
        head: &RunHeadV2,
        model_spec: &ModelLoadSpec,
        chain_cache: &Arc<RwLock<ChainCache>>,
    ) -> Result<Arc<Vec<ResolvedRunCommit>>> {
        let snapshot = chain_cache
            .read()
            .map_err(|error| anyhow!("failed to read S3 chain cache: {error}"))?
            .snapshot();
        if let Some((cached_head, chain)) = &snapshot {
            if *cached_head == head.run_commit_id {
                validate_selected_head(chain, head)?;
                return Ok(Arc::clone(chain));
            }
        }

        // Walk toward the root, stopping at the last validated head. Each
        // commit beyond the cached prefix costs two GETs (commit + manifest);
        // the prefix itself costs none.
        let mut reversed_suffix = Vec::new();
        let mut seen = HashSet::new();
        let mut cached_prefix: Option<Arc<Vec<ResolvedRunCommit>>> = None;
        let mut current_id = Some(head.run_commit_id.clone());
        while let Some(run_commit_id) = current_id {
            if let Some((cached_head, chain)) = &snapshot {
                if *cached_head == run_commit_id {
                    cached_prefix = Some(Arc::clone(chain));
                    break;
                }
            }
            if !seen.insert(run_commit_id.clone()) {
                bail!("S3 RunCommit lineage contains a cycle");
            }
            let run_commit_key = Self::run_commit_key(prefix, &run_commit_id);
            let run_commit_bytes = Self::get_required(client, bucket, &run_commit_key).await?;
            let commit = parse_run_commit(
                "S3 RunCommit",
                &run_commit_id,
                &run_commit_bytes,
                &model_spec.identity,
                model_spec.environment_max_horizon,
            )?;
            let manifest_key = Self::manifest_key(prefix, &commit.checkpoint_id);
            let manifest_bytes = Self::get_required(client, bucket, &manifest_key).await?;
            let actual_checkpoint_id = sha256_hex(&manifest_bytes);
            if actual_checkpoint_id != commit.checkpoint_id {
                bail!(
                    "S3 checkpoint manifest digest is {actual_checkpoint_id}, RunCommit requires {}",
                    commit.checkpoint_id
                );
            }
            let manifest: CheckpointManifestV1 =
                parse_canonical_json("S3 checkpoint manifest", &manifest_bytes)?;
            validate_manifest(&manifest, &model_spec.identity)?;
            current_id = commit.parent_run_commit_id.clone();
            reversed_suffix.push(ResolvedRunCommit {
                run_commit_id,
                commit,
                manifest,
            });
        }
        let prefix_entries: &[ResolvedRunCommit] =
            cached_prefix.as_deref().map_or(&[], Vec::as_slice);
        if prefix_entries
            .iter()
            .any(|entry| seen.contains(&entry.run_commit_id))
        {
            bail!("S3 RunCommit lineage contains a cycle");
        }
        let mut chain = Vec::with_capacity(prefix_entries.len() + reversed_suffix.len());
        chain.extend_from_slice(prefix_entries);
        chain.extend(reversed_suffix.into_iter().rev());
        validate_spliced_chain(prefix_entries.len(), &chain, head)?;
        let chain = Arc::new(chain);
        chain_cache
            .write()
            .map_err(|error| anyhow!("failed to update S3 chain cache: {error}"))?
            .store(head.run_commit_id.clone(), Arc::clone(&chain));
        Ok(chain)
    }

    async fn load_current(context: S3LoadContext<'_>) -> Result<RemoteUpdate> {
        let S3LoadContext {
            client,
            bucket,
            prefix,
            cache_dir,
            model_spec,
            selection,
            evaluator,
            accepted_head,
            model_info,
            chain_cache,
        } = context;
        let run_head_key = Self::run_head_key(prefix);
        let Some(run_head_bytes) = Self::get_optional(client, bucket, &run_head_key).await? else {
            return Ok(RemoteUpdate::Absent);
        };
        let run_head: RunHeadV2 = parse_canonical_json("S3 current run head", &run_head_bytes)?;
        validate_run_head(&run_head)?;
        let accepted = accepted_head
            .read()
            .map_err(|error| anyhow!("failed to read accepted run head: {error}"))?
            .clone();
        if accepted
            .as_ref()
            .is_some_and(|accepted| accepted.run_commit_id == run_head.run_commit_id)
        {
            return Ok(RemoteUpdate::Unchanged);
        }

        let chain = Self::resolve_run_commit_chain(
            client,
            bucket,
            prefix,
            &run_head,
            model_spec,
            chain_cache,
        )
        .await?;
        let selected = select_inference_checkpoint(&chain, selection)?;
        let checkpoint_id = selected.commit.checkpoint_id.clone();
        let manifest = selected.manifest.clone();

        let model_key = Self::model_key(prefix, &manifest.onnx.sha256);
        let model_bytes = Self::get_required(client, bucket, &model_key).await?;
        verify_blob_bytes("S3 ONNX", &model_bytes, &manifest.onnx)?;
        let new_evaluator = if accepted
            .as_ref()
            .is_some_and(|accepted| accepted.model_checkpoint_id == checkpoint_id)
        {
            None
        } else {
            let model_path =
                Self::cache_model(cache_dir, &manifest.onnx.sha256, &model_bytes).await?;
            Some(model_spec.load(&model_path)?)
        };

        let latest_bytes = Self::get_required(client, bucket, &run_head_key).await?;
        let latest: RunHeadV2 = parse_canonical_json("S3 current run head", &latest_bytes)?;
        validate_run_head(&latest)?;
        if latest != run_head {
            debug!(
                candidate = %run_head.checkpoint_id,
                current = %latest.checkpoint_id,
                "Discarding stale S3 checkpoint candidate"
            );
            return Ok(RemoteUpdate::Unchanged);
        }

        let mut accepted_guard = accepted_head
            .write()
            .map_err(|error| anyhow!("failed to lock accepted run head: {error}"))?;
        if accepted_guard
            .as_ref()
            .is_some_and(|value| value.run_commit_id == run_head.run_commit_id)
        {
            return Ok(RemoteUpdate::Unchanged);
        }
        if *accepted_guard != accepted {
            debug!(
                candidate_checkpoint = %run_head.checkpoint_id,
                candidate_run_commit = %run_head.run_commit_id,
                "Discarding S3 candidate after an accepted-head race"
            );
            return Ok(RemoteUpdate::Unchanged);
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
                .is_none_or(|accepted| accepted.model_checkpoint_id != checkpoint_id)
                || evaluator_guard.is_none())
        {
            return Err(anyhow!(
                "cannot reuse an evaluator that does not match the selected S3 checkpoint"
            ));
        }
        let model_changed = new_evaluator.is_some();
        if let Some(new_evaluator) = new_evaluator {
            *evaluator_guard = Some(new_evaluator);
        }
        *accepted_guard = Some(AcceptedHead {
            model_checkpoint_id: checkpoint_id.clone(),
            run_commit_id: run_head.run_commit_id.clone(),
        });
        if model_changed {
            *info_guard = ModelInfo {
                loaded: true,
                checkpoint_id: Some(checkpoint_id.clone()),
                model_sha256: Some(manifest.onnx.sha256.clone()),
                path: Some(format!("s3://{bucket}/{model_key}")),
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
            checkpoint_id = %checkpoint_id,
            run_commit_id = %run_head.run_commit_id,
            model_sha256 = %manifest.onnx.sha256,
            step = manifest.step,
            model_changed,
            "Content-addressed S3 RunHead accepted"
        );
        Ok(if model_changed {
            RemoteUpdate::Loaded
        } else {
            RemoteUpdate::Advanced
        })
    }

    pub async fn try_load_existing(&self) -> Result<bool> {
        Ok(!matches!(
            Self::load_current(S3LoadContext {
                client: &self.client,
                bucket: &self.bucket,
                prefix: &self.prefix,
                cache_dir: &self.cache_dir,
                model_spec: &self.model_spec,
                selection: self.selection,
                evaluator: &self.evaluator,
                accepted_head: &self.accepted_head,
                model_info: &self.model_info,
                chain_cache: &self.chain_cache,
            })
            .await?,
            RemoteUpdate::Absent
        ))
    }

    pub async fn start_watching(&self) -> Result<mpsc::Receiver<()>> {
        let (updates_tx, updates_rx) = mpsc::channel(16);
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let prefix = self.prefix.clone();
        let cache_dir = self.cache_dir.clone();
        let model_spec = self.model_spec.clone();
        let selection = self.selection;
        let evaluator = Arc::clone(&self.evaluator);
        let accepted = Arc::clone(&self.accepted_head);
        let model_info = Arc::clone(&self.model_info);
        let chain_cache = Arc::clone(&self.chain_cache);
        let poll_interval = self.poll_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(poll_interval);
            loop {
                interval.tick().await;
                match Self::load_current(S3LoadContext {
                    client: &client,
                    bucket: &bucket,
                    prefix: &prefix,
                    cache_dir: &cache_dir,
                    model_spec: &model_spec,
                    selection,
                    evaluator: &evaluator,
                    accepted_head: &accepted,
                    model_info: &model_info,
                    chain_cache: &chain_cache,
                })
                .await
                {
                    Ok(RemoteUpdate::Loaded) => {
                        let _ = updates_tx.send(()).await;
                    }
                    Ok(RemoteUpdate::Absent | RemoteUpdate::Unchanged | RemoteUpdate::Advanced) => {
                    }
                    Err(error) => error!("Rejected S3 model channel update: {error}"),
                }
            }
        });

        info!(
            bucket = %self.bucket,
            run_head = %Self::run_head_key(&self.prefix),
            poll_interval = ?self.poll_interval,
            "Started content-addressed S3 model watcher"
        );
        Ok(updates_rx)
    }
}
