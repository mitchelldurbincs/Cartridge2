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

pub(crate) use filesystem::{read_filesystem_head, run_head_path};
pub(crate) use selection::resolve_filesystem_head;
pub(crate) use types::ResolvedCheckpoint;

#[cfg(any(test, feature = "s3"))]
pub(crate) use checkpoint::validate_manifest;
#[cfg(any(test, feature = "s3"))]
pub(crate) use codec::{parse_canonical_json, sha256_hex, validate_run_head};
#[cfg(any(test, feature = "s3"))]
pub(crate) use filesystem::verify_blob_bytes;
#[cfg(feature = "s3")]
pub(crate) use lineage::validate_run_commit_chain;
#[cfg(feature = "s3")]
pub(crate) use run_commit::parse_run_commit;
#[cfg(feature = "s3")]
pub(crate) use selection::select_inference_checkpoint;
#[cfg(any(test, feature = "s3"))]
pub(crate) use types::{CheckpointManifestV1, RunHeadV2};
#[cfg(feature = "s3")]
pub(crate) use types::{ResolvedRunCommit, RUN_HEAD_CHANNEL};

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
