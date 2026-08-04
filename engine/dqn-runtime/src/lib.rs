//! Runtime inference adapter for the ``onnx_q_values_v1`` DQN model contract.

use algorithm_core::{BuiltinAlgorithm, ModelArtifactContract};
use anyhow::{anyhow, bail, Result};
use engine_core::ActionAvailability;
use ort::{
    session::Session,
    value::{TensorElementType, Value, ValueType},
};
use std::path::Path;
use std::sync::Mutex;

const MODEL_CONTRACT: &str = "onnx_q_values_v1";

/// Resolve the action set accepted by the DQN discrete-action contract.
pub fn available_actions(
    availability: &ActionAvailability,
    action_count: usize,
) -> Result<Vec<u32>> {
    let actions: Vec<u32> = match availability {
        ActionAvailability::All => (0..action_count).map(|action| action as u32).collect(),
        ActionAvailability::DiscreteMask { mask } if mask.num_actions() == action_count => {
            mask.iter_ones().map(|action| action as u32).collect()
        }
        ActionAvailability::DiscreteMask { mask } => bail!(
            "DQN legal mask has {} actions, expected {action_count}",
            mask.num_actions()
        ),
        ActionAvailability::Custom { contract, .. } => {
            bail!("DQN does not support custom action availability '{contract}'")
        }
    };
    if actions.is_empty() {
        bail!("DQN decision has no available action");
    }
    Ok(actions)
}

/// Encode action availability for replay targets. Completed transitions use a
/// separate all-false vector and therefore never call this function.
pub fn availability_bits(
    availability: &ActionAvailability,
    action_count: usize,
) -> Result<Vec<bool>> {
    match availability {
        ActionAvailability::All => Ok(vec![true; action_count]),
        ActionAvailability::DiscreteMask { mask } if mask.num_actions() == action_count => Ok((0
            ..action_count)
            .map(|action| mask.is_legal(action))
            .collect()),
        ActionAvailability::DiscreteMask { mask } => bail!(
            "DQN legal mask has {} actions, expected {action_count}",
            mask.num_actions()
        ),
        ActionAvailability::Custom { contract, .. } => {
            bail!("DQN does not support custom action availability '{contract}'")
        }
    }
}

/// Select the highest-valued available action with deterministic tie-breaking.
pub fn greedy_action(q_values: &[f32], actions: &[u32]) -> Result<u32> {
    if q_values.is_empty() || q_values.iter().any(|value| !value.is_finite()) {
        bail!("DQN q_values must be nonempty and finite");
    }
    if actions.is_empty() {
        bail!("DQN greedy selection requires an available action");
    }
    if actions
        .iter()
        .any(|action| usize::try_from(*action).map_or(true, |index| index >= q_values.len()))
    {
        bail!("DQN available action is outside the q_values vector");
    }
    Ok(*actions
        .iter()
        .max_by(|left, right| {
            q_values[**left as usize]
                .total_cmp(&q_values[**right as usize])
                .then_with(|| right.cmp(left))
        })
        .expect("validated nonempty actions"))
}

/// A validated Q-value model. Loading fails closed on artifact identity and
/// the exact ONNX tensor interface before any inference is attempted.
pub struct DqnQPolicy {
    session: Mutex<Session>,
    obs_size: usize,
    action_count: usize,
}

impl DqnQPolicy {
    pub fn load(
        path: &Path,
        obs_size: usize,
        action_count: usize,
        intra_threads: usize,
        expected: &ModelArtifactContract,
    ) -> Result<Self> {
        let descriptor = BuiltinAlgorithm::DqnV1.descriptor();
        if expected.algorithm_id != descriptor.id
            || expected.schema_version != descriptor.model_artifact_schema_version
            || expected.model_contract != MODEL_CONTRACT
        {
            bail!(
                "DQN policy requires the '{}'/'{MODEL_CONTRACT}' artifact contract",
                descriptor.id
            );
        }
        if obs_size == 0 || action_count == 0 || intra_threads == 0 {
            bail!("DQN model dimensions and ONNX threads must be positive");
        }
        let mut builder = Session::builder()
            .map_err(|error| anyhow!("failed to create DQN ONNX session: {error}"))?
            .with_intra_threads(intra_threads)
            .map_err(|error| anyhow!("failed to configure DQN ONNX threads: {error}"))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|error| anyhow!("failed to load DQN ONNX model: {error}"))?;
        let metadata = session
            .metadata()
            .map_err(|error| anyhow!("failed to read DQN ONNX metadata: {error}"))?;
        let mut problems = Vec::new();
        for (key, expected_value) in expected.required_metadata() {
            match metadata.custom(key) {
                Some(actual) if actual == expected_value => {}
                Some(actual) => problems.push(format!(
                    "'{key}' expected '{expected_value}', found '{actual}'"
                )),
                None => problems.push(format!("missing required key '{key}'")),
            }
        }
        drop(metadata);
        if !problems.is_empty() {
            bail!(
                "DQN model artifact identity validation failed: {}",
                problems.join("; ")
            );
        }
        let inputs = session.inputs();
        let outputs = session.outputs();
        if inputs.len() != 1 || inputs[0].name() != "observation" {
            bail!("onnx_q_values_v1 requires exactly input 'observation'");
        }
        if outputs.len() != 1 || outputs[0].name() != "q_values" {
            bail!("onnx_q_values_v1 requires exactly output 'q_values'");
        }
        validate_signature("observation", inputs[0].dtype(), obs_size)?;
        validate_signature("q_values", outputs[0].dtype(), action_count)?;
        Ok(Self {
            session: Mutex::new(session),
            obs_size,
            action_count,
        })
    }

    pub fn q_values(&self, observation: &[u8]) -> Result<Vec<f32>> {
        let expected_bytes = self
            .obs_size
            .checked_mul(4)
            .ok_or_else(|| anyhow!("DQN observation size overflow"))?;
        if observation.len() != expected_bytes {
            bail!(
                "DQN observation has {} bytes, expected {expected_bytes}",
                observation.len()
            );
        }
        let values = observation
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            bail!("DQN observation contains non-finite values");
        }
        let input = Value::from_array(([1usize, self.obs_size], values))
            .map_err(|error| anyhow!("failed to create DQN ONNX input: {error}"))?;
        let mut session = self
            .session
            .lock()
            .map_err(|error| anyhow!("failed to lock DQN ONNX session: {error}"))?;
        let outputs = session
            .run(ort::inputs!["observation" => input])
            .map_err(|error| anyhow!("DQN ONNX inference failed: {error}"))?;
        let output = outputs
            .get("q_values")
            .ok_or_else(|| anyhow!("DQN model did not return q_values"))?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow!("failed to extract DQN q_values: {error}"))?;
        let expected_actions = i64::try_from(self.action_count)?;
        if shape.as_ref() != [1, expected_actions] || values.len() != self.action_count {
            bail!("DQN q_values runtime shape is {shape:?}, expected [1, {expected_actions}]");
        }
        if values.iter().any(|value| !value.is_finite()) {
            bail!("DQN model returned non-finite q_values");
        }
        Ok(values.to_vec())
    }

    /// Apply the cartridge's ``dqn_greedy_v1`` serving policy.
    pub fn select_greedy(
        &self,
        observation: &[u8],
        availability: &ActionAvailability,
    ) -> Result<u32> {
        let actions = available_actions(availability, self.action_count)?;
        greedy_action(&self.q_values(observation)?, &actions)
    }
}

fn validate_signature(name: &str, value_type: &ValueType, width: usize) -> Result<()> {
    let width = i64::try_from(width)?;
    let ValueType::Tensor { ty, shape, .. } = value_type else {
        bail!("DQN tensor '{name}' is not a tensor");
    };
    if *ty != TensorElementType::Float32 || shape.as_ref() != [-1, width] {
        bail!("DQN tensor '{name}' must be float32 [dynamic_batch, {width}], got {value_type}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_serving_respects_availability_and_breaks_ties_by_action_id() {
        assert_eq!(greedy_action(&[10.0, 2.0, 4.0], &[1, 2]).unwrap(), 2);
        assert_eq!(greedy_action(&[1.0, 1.0], &[0, 1]).unwrap(), 0);
        assert!(greedy_action(&[1.0], &[1]).is_err());
        assert!(greedy_action(&[f32::NAN], &[0]).is_err());
    }
}
