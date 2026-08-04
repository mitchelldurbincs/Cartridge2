use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::{AlgorithmDescriptor, AlgorithmError};

pub const PROFILE_NAMESPACE_DIR: &str = "profiles";

/// Version of the required model-artifact identity metadata schema.
pub const MODEL_ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// ONNX custom-metadata keys used to identify a model artifact.
pub const MODEL_METADATA_SCHEMA_VERSION: &str = "cartridge.schema_version";
pub const MODEL_METADATA_ALGORITHM_ID: &str = "cartridge.algorithm_id";
pub const MODEL_METADATA_CONTRACT: &str = "cartridge.model_contract";
pub const MODEL_METADATA_ENV_ID: &str = "cartridge.env_id";
pub const MODEL_METADATA_ENV_CONTRACT_VERSION: &str = "cartridge.env_contract_version";

impl AlgorithmDescriptor {
    /// Build the exact compatibility contract every model consumer must
    /// require for an artifact intended for `env_id`.
    pub fn model_artifact_contract(
        &self,
        env_id: impl Into<String>,
        env_contract_version: u32,
    ) -> ModelArtifactContract {
        assert!(
            env_contract_version > 0,
            "environment contract version must be greater than zero"
        );
        ModelArtifactContract {
            schema_version: self.model_artifact_schema_version,
            algorithm_id: self.id.to_string(),
            model_contract: self.components.model_contract.to_string(),
            env_id: env_id.into(),
            env_contract_version,
        }
    }
}

/// Exact compatibility contract required from a serialized model artifact.
///
/// The fields are deliberately non-optional: model loading is never allowed
/// to infer an algorithm, model contract, or environment from tensor shapes or
/// from the process that happened to open the file. The content identity of a
/// specific set of weights is the checkpoint-manifest SHA-256, not this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelArtifactContract {
    pub schema_version: u32,
    pub algorithm_id: String,
    pub model_contract: String,
    pub env_id: String,
    pub env_contract_version: u32,
}

impl ModelArtifactContract {
    /// Required ONNX custom metadata in stable key order.
    pub fn required_metadata(&self) -> [(&'static str, String); 5] {
        [
            (
                MODEL_METADATA_SCHEMA_VERSION,
                self.schema_version.to_string(),
            ),
            (MODEL_METADATA_ALGORITHM_ID, self.algorithm_id.clone()),
            (MODEL_METADATA_CONTRACT, self.model_contract.clone()),
            (MODEL_METADATA_ENV_ID, self.env_id.clone()),
            (
                MODEL_METADATA_ENV_CONTRACT_VERSION,
                self.env_contract_version.to_string(),
            ),
        ]
    }
}

/// Canonical namespace for every mutable/runtime artifact belonging to one
/// algorithm/environment contract revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeProfile {
    pub algorithm_id: String,
    pub env_id: String,
    pub env_contract_version: u32,
}

impl RuntimeProfile {
    pub fn new(
        algorithm_id: impl Into<String>,
        env_id: impl Into<String>,
        env_contract_version: u32,
    ) -> Result<Self, AlgorithmError> {
        let profile = Self {
            algorithm_id: algorithm_id.into(),
            env_id: env_id.into(),
            env_contract_version,
        };
        validate_profile_segment("algorithm_id", &profile.algorithm_id)?;
        validate_profile_segment("env_id", &profile.env_id)?;
        if profile.env_contract_version == 0 {
            return Err(AlgorithmError::InvalidRuntimeProfile {
                field: "env_contract_version",
                value: "0".to_string(),
            });
        }
        Ok(profile)
    }

    /// Language-neutral slash-separated namespace used by filesystems and
    /// object stores alike.
    pub fn storage_prefix(&self) -> String {
        format!(
            "{PROFILE_NAMESPACE_DIR}/{}/{}/v{}",
            self.algorithm_id, self.env_id, self.env_contract_version
        )
    }

    pub fn data_dir(&self, data_root: impl AsRef<Path>) -> PathBuf {
        data_root.as_ref().join(self.storage_prefix())
    }

    pub fn model_dir(&self, data_root: impl AsRef<Path>) -> PathBuf {
        self.data_dir(data_root).join("models")
    }

    pub fn model_prefix(&self) -> String {
        format!("{}/models", self.storage_prefix())
    }
}

fn validate_profile_segment(field: &'static str, value: &str) -> Result<(), AlgorithmError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
        })
    {
        return Err(AlgorithmError::InvalidRuntimeProfile {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}
