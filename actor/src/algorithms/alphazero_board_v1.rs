//! Experience codec owned by the `alphazero_board_v1` cartridge.
//!
//! `alphazero_transition_v1` is a compact concatenation of little-endian f32
//! values with lengths supplied by the environment's compatible board profile:
//!
//! 1. observation: `obs_size` values
//! 2. policy target: `num_actions` values
//! 3. terminal value target: one value

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const COLLECTOR_CONFIG_SCHEMA_VERSION: u32 = 1;

/// Exact collector configuration owned by the AlphaZero cartridge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AlphaZeroCollectorConfig {
    pub schema_version: u32,
    pub num_simulations: u32,
    pub c_puct: f32,
    pub temperature: f32,
    pub late_temperature: f32,
    pub temp_threshold: u32,
    pub dirichlet_alpha: f32,
    pub dirichlet_weight: f32,
    pub eval_batch_size: u32,
    pub onnx_intra_threads: u32,
}

impl AlphaZeroCollectorConfig {
    pub fn parse(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|error| anyhow::anyhow!("invalid AlphaZero collector config: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != COLLECTOR_CONFIG_SCHEMA_VERSION {
            bail!(
                "AlphaZero collector config schema_version must be exactly {}",
                COLLECTOR_CONFIG_SCHEMA_VERSION
            );
        }
        for (name, value) in [
            ("num_simulations", self.num_simulations),
            ("eval_batch_size", self.eval_batch_size),
            ("onnx_intra_threads", self.onnx_intra_threads),
        ] {
            if value == 0 {
                bail!("{name} must be greater than 0");
            }
        }
        for (name, value) in [
            ("c_puct", self.c_puct),
            ("temperature", self.temperature),
            ("late_temperature", self.late_temperature),
            ("dirichlet_alpha", self.dirichlet_alpha),
            ("dirichlet_weight", self.dirichlet_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be finite and nonnegative");
            }
        }
        if self.dirichlet_weight > 1.0 {
            bail!("dirichlet_weight must be in [0, 1]");
        }
        if (self.dirichlet_alpha == 0.0) != (self.dirichlet_weight == 0.0) {
            bail!("dirichlet_alpha and dirichlet_weight must both be zero to disable noise");
        }
        if self.temp_threshold == 0 {
            if self.late_temperature != self.temperature {
                bail!("late_temperature must equal temperature when temp_threshold is zero");
            }
        } else if self.late_temperature == self.temperature {
            bail!("late_temperature must differ from temperature when the schedule is enabled");
        }
        Ok(())
    }
}

fn decode_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().expect("exact f32 chunk"))
}

pub(crate) fn encode_experience(
    observation: &[u8],
    obs_size: usize,
    policy_target: &[f32],
    num_actions: usize,
    value_target: f32,
) -> Result<Vec<u8>> {
    let expected_observation_bytes = obs_size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("AlphaZero observation size overflow"))?;
    if observation.len() != expected_observation_bytes {
        bail!(
            "AlphaZero observation is {} bytes, expected {} ({} little-endian f32 values)",
            observation.len(),
            expected_observation_bytes,
            obs_size
        );
    }
    if let Some((index, value)) = observation
        .chunks_exact(4)
        .map(decode_f32)
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!("AlphaZero observation contains non-finite value at index {index}: {value}");
    }

    if policy_target.len() != num_actions {
        bail!(
            "AlphaZero policy target has {} actions, expected {num_actions}",
            policy_target.len()
        );
    }
    if let Some((index, value)) = policy_target
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        bail!("AlphaZero policy target has invalid probability at index {index}: {value}");
    }
    let policy_sum: f32 = policy_target.iter().sum();
    if (policy_sum - 1.0).abs() > 1e-3 {
        bail!("AlphaZero policy target sums to {policy_sum}, expected 1");
    }

    if !value_target.is_finite() || !(-1.0..=1.0).contains(&value_target) {
        bail!("AlphaZero value target must be finite and in [-1, 1], got {value_target}");
    }

    let policy_bytes = num_actions
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("AlphaZero policy size overflow"))?;
    let mut payload = Vec::with_capacity(expected_observation_bytes + policy_bytes + 4);
    payload.extend_from_slice(observation);
    for probability in policy_target {
        payload.extend_from_slice(&probability.to_le_bytes());
    }
    payload.extend_from_slice(&value_target.to_le_bytes());
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector_config() -> AlphaZeroCollectorConfig {
        AlphaZeroCollectorConfig {
            schema_version: COLLECTOR_CONFIG_SCHEMA_VERSION,
            num_simulations: 100,
            c_puct: 1.4,
            temperature: 1.0,
            late_temperature: 1.0,
            temp_threshold: 0,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            eval_batch_size: 32,
            onnx_intra_threads: 1,
        }
    }

    #[test]
    fn collector_config_is_strict_and_versioned() {
        let json = serde_json::to_string(&collector_config()).unwrap();
        assert_eq!(
            AlphaZeroCollectorConfig::parse(&json).unwrap(),
            collector_config()
        );
        assert!(AlphaZeroCollectorConfig::parse(
            r#"{"schema_version":1,"num_simulations":1,"c_puct":1.0,"temperature":1.0,"late_temperature":1.0,"temp_threshold":0,"dirichlet_alpha":0.0,"dirichlet_weight":0.0,"eval_batch_size":1,"onnx_intra_threads":1,"legacy":true}"#
        )
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
    }

    #[test]
    fn collector_config_rejects_invalid_search_domains() {
        let mut config = collector_config();
        config.num_simulations = 0;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("num_simulations"));

        let mut config = collector_config();
        config.dirichlet_alpha = 0.0;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("both be zero"));

        let mut config = collector_config();
        config.temp_threshold = 1;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must differ"));
    }

    fn observation(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn codec_has_exact_language_neutral_layout() {
        let payload =
            encode_experience(&observation(&[1.0, -2.0]), 2, &[0.25, 0.75], 2, -1.0).unwrap();
        let values = payload.chunks_exact(4).map(decode_f32).collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, -2.0, 0.25, 0.75, -1.0]);
    }

    #[test]
    fn codec_rejects_shape_probability_and_value_corruption() {
        assert!(encode_experience(&observation(&[1.0]), 2, &[1.0], 1, 0.0).is_err());
        assert!(encode_experience(&observation(&[1.0]), 1, &[0.2, 0.2], 2, 0.0).is_err());
        assert!(encode_experience(&observation(&[1.0]), 1, &[1.0], 1, f32::NAN).is_err());
        assert!(encode_experience(&observation(&[f32::INFINITY]), 1, &[1.0], 1, 0.0).is_err());
    }
}
