//! Algorithm identity, component contracts, and environment compatibility.
//!
//! An environment being registered does not imply that every learner can use
//! it. This crate is the small shared catalog that lets the actor, trainer
//! manifest, and evaluator agree on which algorithm is being requested and
//! why an environment is (or is not) compatible with it.

mod catalog;
mod compatibility;
mod error;
mod profile;

pub use catalog::{
    algorithm_descriptors, available_algorithm_ids, compatibility_reports,
    compatible_algorithm_ids, resolve_algorithm, AlgorithmComponents, AlgorithmDescriptor,
    BuiltinAlgorithm, ALPHAZERO_BOARD_V1_ID, DQN_V1_ID,
};
pub use compatibility::{CompatibilityIssue, CompatibilityReport};
pub use error::AlgorithmError;
pub use profile::{
    ModelArtifactContract, RuntimeProfile, MODEL_ARTIFACT_SCHEMA_VERSION,
    MODEL_METADATA_ALGORITHM_ID, MODEL_METADATA_CONTRACT, MODEL_METADATA_ENV_CONTRACT_VERSION,
    MODEL_METADATA_ENV_ID, MODEL_METADATA_SCHEMA_VERSION, PROFILE_NAMESPACE_DIR,
};

#[cfg(test)]
mod tests;
