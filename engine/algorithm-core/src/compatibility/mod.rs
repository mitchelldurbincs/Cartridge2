mod alphazero;
mod dqn;

use engine_core::typed::{ActionEncoding, Capabilities, WIRE_ENCODING_SCHEMA_VERSION};
use engine_core::EnvironmentMetadata;
use serde::Serialize;

use crate::AlgorithmError;

pub(crate) use alphazero::alphazero_compatibility;
#[cfg(test)]
pub(crate) use alphazero::ALPHAZERO_UNVERIFIED_ASSUMPTIONS;
pub(crate) use dqn::dqn_compatibility;
#[cfg(test)]
pub(crate) use dqn::DQN_UNVERIFIED_ASSUMPTIONS;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompatibilityIssue {
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompatibilityReport {
    pub algorithm_id: &'static str,
    pub env_id: String,
    pub compatible: bool,
    pub issues: Vec<CompatibilityIssue>,
    pub unverified_assumptions: &'static [&'static str],
}

impl CompatibilityReport {
    pub fn require_compatible(&self) -> Result<(), AlgorithmError> {
        if self.compatible {
            return Ok(());
        }
        let reasons = self
            .issues
            .iter()
            .map(|issue| format!("{}: {}", issue.code, issue.message))
            .collect::<Vec<_>>()
            .join("; ");
        Err(AlgorithmError::Incompatible {
            algorithm_id: self.algorithm_id.to_string(),
            env_id: self.env_id.clone(),
            reasons,
        })
    }
}

fn add_issue(issues: &mut Vec<CompatibilityIssue>, code: &'static str, message: String) {
    issues.push(CompatibilityIssue { code, message });
}

fn add_identity_and_wire_issues(
    capabilities: &Capabilities,
    metadata: &EnvironmentMetadata,
    issues: &mut Vec<CompatibilityIssue>,
) {
    if capabilities.id.env_id != metadata.id {
        add_issue(
            issues,
            "identity.env_id_mismatch",
            format!(
                "capabilities use '{}' but metadata uses '{}'",
                capabilities.id.env_id, metadata.id
            ),
        );
    }
    if capabilities.contract_version == 0 {
        add_issue(
            issues,
            "identity.contract_version",
            "requires a non-zero immutable environment contract version".to_string(),
        );
    }
    if capabilities.encoding.schema_version != WIRE_ENCODING_SCHEMA_VERSION {
        add_issue(
            issues,
            "encoding.schema_version",
            format!(
                "requires wire encoding schema version {}, got {}",
                WIRE_ENCODING_SCHEMA_VERSION, capabilities.encoding.schema_version
            ),
        );
    }
    if capabilities.encoding.action != ActionEncoding::DiscreteU32LittleEndian {
        add_issue(
            issues,
            "action.encoding",
            format!(
                "requires DiscreteU32LittleEndian actions, got {:?}",
                capabilities.encoding.action
            ),
        );
    }
}

fn add_horizon_issue(capabilities: &Capabilities, issues: &mut Vec<CompatibilityIssue>) {
    if capabilities.max_horizon.is_none() || capabilities.max_horizon == Some(0) {
        add_issue(
            issues,
            "episode.max_horizon",
            "requires a finite non-zero maximum horizon".to_string(),
        );
    }
}
