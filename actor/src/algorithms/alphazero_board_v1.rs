//! Experience codec owned by the `alphazero_board_v1` cartridge.
//!
//! `alphazero_transition_v1` is a compact concatenation of little-endian f32
//! values with lengths supplied by the environment's compatible board profile:
//!
//! 1. observation: `obs_size` values
//! 2. policy target: `num_actions` values
//! 3. terminal value target: one value

use anyhow::{bail, Result};

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
