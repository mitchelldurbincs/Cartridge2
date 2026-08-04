//! DQN-owned collector configuration and replay payload codec.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const COLLECTOR_CONFIG_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DqnCollectorConfig {
    pub schema_version: u32,
    pub epsilon: f32,
    pub seed: u64,
    pub onnx_intra_threads: u32,
}

impl DqnCollectorConfig {
    pub fn parse(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|error| anyhow::anyhow!("invalid DQN collector config: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != COLLECTOR_CONFIG_SCHEMA_VERSION {
            bail!(
                "DQN collector config schema_version must be exactly {}",
                COLLECTOR_CONFIG_SCHEMA_VERSION
            );
        }
        if !self.epsilon.is_finite() || !(0.0..=1.0).contains(&self.epsilon) {
            bail!("DQN collector epsilon must be finite and in [0, 1]");
        }
        if self.onnx_intra_threads == 0 {
            bail!("DQN collector onnx_intra_threads must be positive");
        }
        Ok(())
    }
}

/// Encode one fixed-shape ``dqn_transition_v1`` record.
///
/// Layout: observation f32s, action u32, reward f32, next-observation f32s,
/// terminated u8, truncated u8, then one next-availability u8 per action.
pub struct DqnTransition<'a> {
    pub observation: &'a [u8],
    pub action: u32,
    pub reward: f32,
    pub next_observation: &'a [u8],
    pub terminated: bool,
    pub truncated: bool,
    pub next_availability: &'a [bool],
}

pub fn encode_transition(
    transition: DqnTransition<'_>,
    obs_size: usize,
    num_actions: usize,
) -> Result<Vec<u8>> {
    let DqnTransition {
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
        next_availability,
    } = transition;
    let observation_bytes = obs_size
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("DQN observation byte count overflow"))?;
    if observation.len() != observation_bytes || next_observation.len() != observation_bytes {
        bail!(
            "DQN observations must each contain {observation_bytes} bytes, got {} and {}",
            observation.len(),
            next_observation.len()
        );
    }
    if action as usize >= num_actions {
        bail!("DQN action {action} is outside [0, {num_actions})");
    }
    if !reward.is_finite() {
        bail!("DQN reward must be finite");
    }
    if terminated && truncated {
        bail!("DQN transition cannot be both terminated and truncated");
    }
    if next_availability.len() != num_actions {
        bail!(
            "DQN next availability has {} actions, expected {num_actions}",
            next_availability.len()
        );
    }
    if (terminated || truncated) && next_availability.iter().any(|available| *available) {
        bail!("completed DQN transitions must have empty next-action availability");
    }
    if !terminated && !truncated && !next_availability.iter().any(|available| *available) {
        bail!("running DQN transitions require at least one available next action");
    }

    let capacity = observation_bytes
        .checked_mul(2)
        .and_then(|value| value.checked_add(10))
        .and_then(|value| value.checked_add(num_actions))
        .ok_or_else(|| anyhow::anyhow!("DQN transition byte count overflow"))?;
    let mut payload = Vec::with_capacity(capacity);
    payload.extend_from_slice(observation);
    payload.extend_from_slice(&action.to_le_bytes());
    payload.extend_from_slice(&reward.to_le_bytes());
    payload.extend_from_slice(next_observation);
    payload.push(u8::from(terminated));
    payload.push(u8::from(truncated));
    payload.extend(
        next_availability
            .iter()
            .map(|available| u8::from(*available)),
    );
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_is_strict_and_versioned() {
        let json = r#"{"schema_version":1,"epsilon":0.25,"seed":7,"onnx_intra_threads":1}"#;
        assert_eq!(DqnCollectorConfig::parse(json).unwrap().epsilon, 0.25);
        assert!(DqnCollectorConfig::parse(
            r#"{"schema_version":1,"epsilon":0.25,"seed":7,"onnx_intra_threads":1,"mcts":5}"#
        )
        .is_err());
        assert!(DqnCollectorConfig::parse(
            r#"{"schema_version":2,"epsilon":0.25,"seed":7,"onnx_intra_threads":1}"#
        )
        .is_err());
    }

    #[test]
    fn transition_codec_preserves_immediate_reward_and_completion_kinds() {
        let observation = [0.0f32, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let next = [0.5f32, 0.875]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let payload = encode_transition(
            DqnTransition {
                observation: &observation,
                action: 1,
                reward: -0.01,
                next_observation: &next,
                terminated: false,
                truncated: false,
                next_availability: &[true, true],
            },
            2,
            2,
        )
        .unwrap();
        assert_eq!(payload.len(), 28);
        assert_eq!(u32::from_le_bytes(payload[8..12].try_into().unwrap()), 1);
        assert_eq!(
            f32::from_le_bytes(payload[12..16].try_into().unwrap()),
            -0.01
        );
        assert_eq!(&payload[26..], &[1, 1]);
    }
}
