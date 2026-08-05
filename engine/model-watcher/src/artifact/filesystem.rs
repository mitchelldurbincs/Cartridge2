use algorithm_core::ModelArtifactContract;
use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::checkpoint::validate_manifest;
use super::codec::{parse_canonical_json, sha256_hex, validate_digest, validate_run_head};
use super::lineage::{validate_selected_head, validate_spliced_chain, ChainCache};
use super::run_commit::parse_run_commit;
use super::types::*;

pub(crate) fn run_head_path(model_root: &Path) -> PathBuf {
    model_root
        .join("channels")
        .join(format!("{RUN_HEAD_CHANNEL}.json"))
}

pub(crate) fn manifest_path(model_root: &Path, checkpoint_id: &str) -> PathBuf {
    model_root
        .join("manifests")
        .join("sha256")
        .join(format!("{checkpoint_id}.json"))
}

pub(crate) fn run_commit_path(model_root: &Path, run_commit_id: &str) -> PathBuf {
    model_root
        .join("run-commits")
        .join("sha256")
        .join(format!("{run_commit_id}.json"))
}

pub(crate) fn onnx_blob_path(model_root: &Path, digest: &str) -> PathBuf {
    model_root
        .join("blobs")
        .join("sha256")
        .join(format!("{digest}.onnx"))
}

pub(crate) fn verify_blob_bytes(
    label: &str,
    bytes: &[u8],
    reference: &BlobReference,
) -> Result<()> {
    let actual_size = u64::try_from(bytes.len()).context("blob size exceeds u64")?;
    if actual_size != reference.size_bytes {
        bail!(
            "{label} blob size is {actual_size}, expected {}",
            reference.size_bytes
        );
    }
    let actual_digest = sha256_hex(bytes);
    if actual_digest != reference.sha256 {
        bail!(
            "{label} blob digest is {actual_digest}, expected {}",
            reference.sha256
        );
    }
    Ok(())
}

pub(crate) fn read_filesystem_head(model_root: &Path) -> Result<Option<RunHeadV2>> {
    let path = run_head_path(model_root);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(anyhow!("failed to read model run head {:?}: {error}", path)),
    };
    let head: RunHeadV2 = parse_canonical_json("current run head", &bytes)?;
    validate_run_head(&head)?;
    Ok(Some(head))
}

fn read_filesystem_manifest(
    model_root: &Path,
    checkpoint_id: &str,
    expected_contract: &ModelArtifactContract,
) -> Result<CheckpointManifestV1> {
    validate_digest("checkpoint_id", checkpoint_id)?;
    let path = manifest_path(model_root, checkpoint_id);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read checkpoint manifest {path:?}"))?;
    let actual_id = sha256_hex(&bytes);
    if actual_id != checkpoint_id {
        bail!("checkpoint manifest digest is {actual_id}, expected {checkpoint_id}");
    }
    let manifest: CheckpointManifestV1 = parse_canonical_json("checkpoint manifest", &bytes)?;
    validate_manifest(&manifest, expected_contract)?;
    Ok(manifest)
}

pub(crate) fn resolve_filesystem_run_commit_chain(
    model_root: &Path,
    head: &RunHeadV2,
    expected_contract: &ModelArtifactContract,
    environment_max_horizon: u32,
    cache: Option<&RwLock<ChainCache>>,
) -> Result<Arc<Vec<ResolvedRunCommit>>> {
    let snapshot = match cache {
        Some(lock) => lock
            .read()
            .map_err(|error| anyhow!("failed to read chain cache: {error}"))?
            .snapshot(),
        None => None,
    };
    if let Some((cached_head, chain)) = &snapshot {
        if *cached_head == head.run_commit_id {
            // Same head as the last validated chain: only re-check the head
            // binding, which costs no reads.
            validate_selected_head(chain, head)?;
            return Ok(Arc::clone(chain));
        }
    }

    // Walk toward the root, stopping at the last validated head: only the
    // suffix beyond that prefix is read, parsed, hashed, and validated.
    let mut reversed_suffix = Vec::new();
    let mut seen = HashSet::new();
    let mut prefix: Option<Arc<Vec<ResolvedRunCommit>>> = None;
    let mut current_id = Some(head.run_commit_id.clone());
    while let Some(run_commit_id) = current_id {
        if let Some((cached_head, chain)) = &snapshot {
            if *cached_head == run_commit_id {
                prefix = Some(Arc::clone(chain));
                break;
            }
        }
        if !seen.insert(run_commit_id.clone()) {
            bail!("RunCommit lineage contains a cycle");
        }
        let path = run_commit_path(model_root, &run_commit_id);
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read immutable RunCommit {path:?}"))?;
        let commit = parse_run_commit(
            "RunCommit",
            &run_commit_id,
            &bytes,
            expected_contract,
            environment_max_horizon,
        )?;
        let manifest =
            read_filesystem_manifest(model_root, &commit.checkpoint_id, expected_contract)?;
        current_id = commit.parent_run_commit_id.clone();
        reversed_suffix.push(ResolvedRunCommit {
            run_commit_id,
            commit,
            manifest,
        });
    }
    let prefix_entries: &[ResolvedRunCommit] = prefix.as_deref().map_or(&[], Vec::as_slice);
    if prefix_entries
        .iter()
        .any(|entry| seen.contains(&entry.run_commit_id))
    {
        bail!("RunCommit lineage contains a cycle");
    }
    let mut chain = Vec::with_capacity(prefix_entries.len() + reversed_suffix.len());
    chain.extend_from_slice(prefix_entries);
    chain.extend(reversed_suffix.into_iter().rev());
    validate_spliced_chain(prefix_entries.len(), &chain, head)?;
    let chain = Arc::new(chain);
    if let Some(lock) = cache {
        lock.write()
            .map_err(|error| anyhow!("failed to update chain cache: {error}"))?
            .store(head.run_commit_id.clone(), Arc::clone(&chain));
    }
    Ok(chain)
}
