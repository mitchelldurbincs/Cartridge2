use algorithm_core::ModelArtifactContract;
use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::sync::RwLock;

use super::filesystem::{onnx_blob_path, resolve_filesystem_run_commit_chain, verify_blob_bytes};
use super::lineage::ChainCache;
use super::types::{ResolvedCheckpoint, ResolvedRunCommit, RunHeadV2};
use crate::ModelSelection;

pub(crate) fn select_inference_checkpoint(
    chain: &[ResolvedRunCommit],
    selection: ModelSelection,
) -> Result<&ResolvedRunCommit> {
    let latest = chain
        .last()
        .ok_or_else(|| anyhow!("RunCommit lineage is empty"))?;
    let checkpoint_id = match selection {
        ModelSelection::Latest => latest.commit.checkpoint_id.as_str(),
        ModelSelection::ChampionOrLatest => latest
            .commit
            .champion
            .as_ref()
            .map_or(latest.commit.checkpoint_id.as_str(), |champion| {
                champion.checkpoint_id.as_str()
            }),
    };
    chain
        .iter()
        .find(|entry| entry.commit.checkpoint_id == checkpoint_id)
        .ok_or_else(|| anyhow!("selected inference checkpoint is absent from RunCommit lineage"))
}

pub(crate) fn resolve_filesystem_head(
    model_root: &Path,
    head: RunHeadV2,
    expected_contract: &ModelArtifactContract,
    environment_max_horizon: u32,
    selection: ModelSelection,
    chain_cache: Option<&RwLock<ChainCache>>,
) -> Result<ResolvedCheckpoint> {
    let chain = resolve_filesystem_run_commit_chain(
        model_root,
        &head,
        expected_contract,
        environment_max_horizon,
        chain_cache,
    )?;
    let selected = select_inference_checkpoint(&chain, selection)?;
    let checkpoint_id = selected.commit.checkpoint_id.clone();
    let manifest = selected.manifest.clone();

    let model_path = onnx_blob_path(model_root, &manifest.onnx.sha256);
    let model_bytes = std::fs::read(&model_path)
        .with_context(|| format!("failed to read immutable ONNX blob {model_path:?}"))?;
    verify_blob_bytes("ONNX", &model_bytes, &manifest.onnx)?;

    Ok(ResolvedCheckpoint {
        checkpoint_id,
        run_commit_id: head.run_commit_id,
        manifest,
        model_path,
    })
}
