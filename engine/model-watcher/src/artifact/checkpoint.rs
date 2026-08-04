use algorithm_core::ModelArtifactContract;
use anyhow::{bail, Result};

use super::codec::validate_digest;
use super::types::{ArtifactProfile, CheckpointManifestV1, CHECKPOINT_MANIFEST_SCHEMA_VERSION};

pub(crate) fn validate_manifest(
    manifest: &CheckpointManifestV1,
    expected_contract: &ModelArtifactContract,
) -> Result<()> {
    if manifest.schema_version != CHECKPOINT_MANIFEST_SCHEMA_VERSION {
        bail!(
            "unsupported checkpoint manifest schema {}, expected {}",
            manifest.schema_version,
            CHECKPOINT_MANIFEST_SCHEMA_VERSION
        );
    }
    let expected_profile = ArtifactProfile::from(expected_contract);
    if manifest.profile != expected_profile {
        bail!(
            "checkpoint profile {:?} does not match runtime profile {:?}",
            manifest.profile,
            expected_profile
        );
    }
    validate_digest("config_sha256", &manifest.config_sha256)?;
    if let Some(parent) = &manifest.parent_checkpoint_id {
        validate_digest("parent_checkpoint_id", parent)?;
    }
    for (label, blob) in [
        ("onnx", &manifest.onnx),
        ("learner_state", &manifest.learner_state),
    ] {
        validate_digest(&format!("{label}.sha256"), &blob.sha256)?;
        if blob.size_bytes == 0 {
            bail!("{label}.size_bytes must be positive");
        }
    }
    Ok(())
}
