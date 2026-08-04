//! Strict content-addressed checkpoint protocol shared by filesystem and S3.

mod checkpoint;
mod codec;
mod filesystem;
mod lineage;
mod recipe;
mod run_commit;
mod selection;
mod types;
mod validation;

pub(crate) use checkpoint::validate_manifest;
pub(crate) use codec::{parse_canonical_json, sha256_hex, validate_run_head};
pub(crate) use filesystem::{read_filesystem_head, run_head_path, verify_blob_bytes};
pub(crate) use lineage::validate_run_commit_chain;
pub(crate) use run_commit::parse_run_commit;
pub(crate) use selection::{resolve_filesystem_head, select_inference_checkpoint};
pub(crate) use types::{
    CheckpointManifestV1, ResolvedCheckpoint, ResolvedRunCommit, RunHeadV2, RUN_HEAD_CHANNEL,
};

#[cfg(test)]
use codec::validate_digest;
#[cfg(test)]
pub(crate) use filesystem::{manifest_path, onnx_blob_path, run_commit_path};
#[cfg(test)]
pub(crate) use types::{ArtifactProfile, BlobReference};
#[cfg(test)]
use validation::validate_utc_timestamp;

#[cfg(test)]
mod tests;
