//! ONNX Runtime evaluator for neural network inference.
//!
//! This module provides an evaluator that uses ONNX models exported from
//! the Python trainer. The ONNX model takes observations as input and
//! outputs policy logits and value estimates.
//!
//! # Model Format
//!
//! The ONNX model is expected to have:
//! - Input: "observation" - shape (batch_size, obs_size) float32
//! - Output: "policy_logits" - shape (batch_size, action_size) float32
//! - Output: "value" - shape (batch_size, 1) float32
//!
//! For TicTacToe: obs_size=29, action_size=9

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use algorithm_core::ModelArtifactContract;
use engine_core::board_profile::LegalMask;
use ort::{
    session::builder::SessionBuilder,
    session::Session,
    value::{TensorElementType, Value, ValueType},
};
use tracing::debug;

use crate::evaluator::{EvalResult, Evaluator, EvaluatorError};

const ONNX_POLICY_VALUE_V1_CONTRACT: &str = "onnx_policy_value_v1";

/// ONNX Runtime evaluator that loads and runs neural network models.
///
/// Uses a Mutex internally because `Session::run` requires `&mut self`,
/// but the `Evaluator` trait uses `&self` for thread-safe sharing.
pub struct OnnxEvaluator {
    session: Mutex<Session>,
    obs_size: usize,
    num_actions: usize,
    model_contract: ModelArtifactContract,
    /// Number of inferences performed (for diagnostics)
    inference_count: AtomicU64,
    /// Total inference time in microseconds (for diagnostics)
    total_inference_time_us: AtomicU64,
    /// Total time spent preparing inputs (obs conversion) in microseconds
    total_prep_time_us: AtomicU64,
    /// Total time spent in post-processing (softmax etc) in microseconds
    total_post_time_us: AtomicU64,
}

/// Diagnostic stats from the ONNX evaluator.
#[derive(Debug, Clone, Default)]
pub struct OnnxStats {
    /// Number of inference calls made
    pub inference_count: u64,
    /// Total inference time in microseconds
    pub total_inference_us: u64,
    /// Total input preparation time in microseconds
    pub total_prep_us: u64,
    /// Total post-processing time in microseconds
    pub total_post_us: u64,
}

impl std::fmt::Debug for OnnxEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEvaluator")
            .field("obs_size", &self.obs_size)
            .field("num_actions", &self.num_actions)
            .field("model_contract", &self.model_contract)
            .finish_non_exhaustive()
    }
}

impl OnnxEvaluator {
    /// Register CoreML execution provider if the `coreml` feature is enabled.
    /// Registration failures are surfaced instead of silently changing providers.
    fn register_coreml_ep(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, EvaluatorError> {
        #[cfg(feature = "coreml")]
        {
            let coreml_ep = ort::ep::CoreML::default().build();
            debug!("Attempting to register CoreML execution provider");
            builder.with_execution_providers([coreml_ep]).map_err(|e| {
                EvaluatorError::ModelError(format!(
                    "Failed to register CoreML execution provider: {}",
                    e
                ))
            })
        }
        #[cfg(not(feature = "coreml"))]
        {
            Ok(builder)
        }
    }

    fn session_builder(intra_threads: usize) -> Result<SessionBuilder, EvaluatorError> {
        if intra_threads == 0 {
            return Err(EvaluatorError::ModelError(
                "ONNX intra_threads must be greater than 0".to_string(),
            ));
        }
        let builder = Session::builder()
            .map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to create session builder: {e}"))
            })?
            .with_intra_threads(intra_threads)
            .map_err(|e| EvaluatorError::ModelError(format!("Failed to set intra threads: {e}")))?;
        Self::register_coreml_ep(builder)
    }

    fn validate_metadata_with(
        expected: &ModelArtifactContract,
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<(), EvaluatorError> {
        let mut problems = Vec::new();
        for (key, expected_value) in expected.required_metadata() {
            match lookup(key) {
                Some(actual) if actual == expected_value => {}
                Some(actual) => problems.push(format!(
                    "'{key}' expected '{expected_value}', found '{actual}'"
                )),
                None => problems.push(format!(
                    "missing required key '{key}' (expected '{expected_value}')"
                )),
            }
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(EvaluatorError::ModelError(format!(
                "model artifact identity validation failed: {}",
                problems.join("; ")
            )))
        }
    }

    fn validate_tensor_signature(
        kind: &str,
        name: &str,
        value_type: &ValueType,
        expected_width: usize,
    ) -> Result<(), EvaluatorError> {
        let expected_width = i64::try_from(expected_width).map_err(|_| {
            EvaluatorError::ModelError(format!(
                "{kind} '{name}' width does not fit in an ONNX dimension"
            ))
        })?;

        let ValueType::Tensor { ty, shape, .. } = value_type else {
            return Err(EvaluatorError::ModelError(format!(
                "{kind} '{name}' must be a tensor, found {value_type}"
            )));
        };
        if *ty != TensorElementType::Float32 {
            return Err(EvaluatorError::ModelError(format!(
                "{kind} '{name}' must use f32 elements, found {ty}"
            )));
        }
        if shape.len() != 2 || shape[0] != -1 || shape[1] != expected_width {
            return Err(EvaluatorError::ModelError(format!(
                "{kind} '{name}' must have shape [dynamic_batch, {expected_width}], found {shape}"
            )));
        }
        Ok(())
    }

    fn validate_interface_with<'a>(
        inputs: &[(&'a str, &'a ValueType)],
        outputs: &[(&'a str, &'a ValueType)],
        obs_size: usize,
        num_actions: usize,
    ) -> Result<(), EvaluatorError> {
        if inputs.len() != 1 {
            return Err(EvaluatorError::ModelError(format!(
                "onnx_policy_value_v1 requires exactly 1 input, found {}",
                inputs.len()
            )));
        }
        if outputs.len() != 2 {
            return Err(EvaluatorError::ModelError(format!(
                "onnx_policy_value_v1 requires exactly 2 outputs, found {}",
                outputs.len()
            )));
        }

        let observation = inputs
            .iter()
            .find(|(name, _)| *name == "observation")
            .ok_or_else(|| {
                EvaluatorError::ModelError(
                    "onnx_policy_value_v1 is missing input 'observation'".to_string(),
                )
            })?;
        Self::validate_tensor_signature("input", observation.0, observation.1, obs_size)?;

        let policy = outputs
            .iter()
            .find(|(name, _)| *name == "policy_logits")
            .ok_or_else(|| {
                EvaluatorError::ModelError(
                    "onnx_policy_value_v1 is missing output 'policy_logits'".to_string(),
                )
            })?;
        Self::validate_tensor_signature("output", policy.0, policy.1, num_actions)?;

        let value = outputs
            .iter()
            .find(|(name, _)| *name == "value")
            .ok_or_else(|| {
                EvaluatorError::ModelError(
                    "onnx_policy_value_v1 is missing output 'value'".to_string(),
                )
            })?;
        Self::validate_tensor_signature("output", value.0, value.1, 1)?;
        Ok(())
    }

    fn validate_model_interface(
        session: &Session,
        obs_size: usize,
        num_actions: usize,
    ) -> Result<(), EvaluatorError> {
        let inputs = session
            .inputs()
            .iter()
            .map(|input| (input.name(), input.dtype()))
            .collect::<Vec<_>>();
        let outputs = session
            .outputs()
            .iter()
            .map(|output| (output.name(), output.dtype()))
            .collect::<Vec<_>>();
        Self::validate_interface_with(&inputs, &outputs, obs_size, num_actions)
    }

    fn validate_runtime_tensor(
        name: &str,
        shape: &[i64],
        element_count: usize,
        expected_shape: &[usize],
    ) -> Result<(), EvaluatorError> {
        let expected_dims = expected_shape
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                EvaluatorError::ModelError(format!(
                    "expected shape for output '{name}' does not fit in ONNX dimensions"
                ))
            })?;
        let expected_elements = expected_shape
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| {
                EvaluatorError::ModelError(format!(
                    "expected element count for output '{name}' overflowed"
                ))
            })?;

        if shape != expected_dims.as_slice() || element_count != expected_elements {
            return Err(EvaluatorError::ModelError(format!(
                "output '{name}' expected shape {expected_dims:?} with {expected_elements} elements, found shape {shape:?} with {element_count} elements"
            )));
        }
        Ok(())
    }

    fn from_session(
        session: Session,
        obs_size: usize,
        num_actions: usize,
        expected_contract: &ModelArtifactContract,
    ) -> Result<Self, EvaluatorError> {
        let metadata = session.metadata().map_err(|e| {
            EvaluatorError::ModelError(format!("Failed to read ONNX model metadata: {e}"))
        })?;
        Self::validate_metadata_with(expected_contract, |key| metadata.custom(key))?;
        drop(metadata);
        if expected_contract.model_contract != ONNX_POLICY_VALUE_V1_CONTRACT {
            return Err(EvaluatorError::ModelError(format!(
                "OnnxEvaluator only supports model contract '{ONNX_POLICY_VALUE_V1_CONTRACT}', found '{}'",
                expected_contract.model_contract
            )));
        }
        Self::validate_model_interface(&session, obs_size, num_actions)?;

        Ok(Self {
            session: Mutex::new(session),
            obs_size,
            num_actions,
            model_contract: expected_contract.clone(),
            inference_count: AtomicU64::new(0),
            total_inference_time_us: AtomicU64::new(0),
            total_prep_time_us: AtomicU64::new(0),
            total_post_time_us: AtomicU64::new(0),
        })
    }

    /// Load an ONNX model file and require its embedded artifact identity.
    pub fn load_from_file<P: AsRef<Path>>(
        model_path: P,
        obs_size: usize,
        num_actions: usize,
        intra_threads: usize,
        expected_contract: &ModelArtifactContract,
    ) -> Result<Self, EvaluatorError> {
        let mut builder = Self::session_builder(intra_threads)?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| EvaluatorError::ModelError(format!("Failed to load model: {e}")))?;
        Self::from_session(session, obs_size, num_actions, expected_contract)
    }

    /// Load ONNX bytes and require their embedded artifact identity.
    pub fn load_from_bytes(
        model_bytes: &[u8],
        obs_size: usize,
        num_actions: usize,
        intra_threads: usize,
        expected_contract: &ModelArtifactContract,
    ) -> Result<Self, EvaluatorError> {
        let mut builder = Self::session_builder(intra_threads)?;
        let session = builder
            .commit_from_memory(model_bytes)
            .map_err(|e| EvaluatorError::ModelError(format!("Failed to load model: {e}")))?;
        Self::from_session(session, obs_size, num_actions, expected_contract)
    }

    /// Identity validated before this evaluator was constructed.
    pub fn model_contract(&self) -> &ModelArtifactContract {
        &self.model_contract
    }

    /// Get diagnostic stats from this evaluator.
    pub fn get_stats(&self) -> OnnxStats {
        OnnxStats {
            inference_count: self.inference_count.load(Ordering::Relaxed),
            total_inference_us: self.total_inference_time_us.load(Ordering::Relaxed),
            total_prep_us: self.total_prep_time_us.load(Ordering::Relaxed),
            total_post_us: self.total_post_time_us.load(Ordering::Relaxed),
        }
    }

    /// Log a summary of the evaluator's performance stats.
    pub fn log_stats(&self) {
        let stats = self.get_stats();
        if stats.inference_count == 0 {
            return;
        }

        let avg_inference_us = stats.total_inference_us / stats.inference_count;
        let avg_prep_us = stats.total_prep_us / stats.inference_count;
        let avg_post_us = stats.total_post_us / stats.inference_count;
        let total_us = stats.total_inference_us + stats.total_prep_us + stats.total_post_us;
        let inference_pct = (stats.total_inference_us as f64 / total_us as f64) * 100.0;

        debug!(
            inference_count = stats.inference_count,
            avg_inference_us = avg_inference_us,
            avg_prep_us = avg_prep_us,
            avg_post_us = avg_post_us,
            inference_pct = format!("{:.1}%", inference_pct),
            "ONNX evaluator stats"
        );
    }

    /// Convert observation bytes to f32 vector.
    /// The observation is stored as f32 values in little-endian byte order.
    fn obs_bytes_to_f32(&self, obs: &[u8]) -> Result<Vec<f32>, EvaluatorError> {
        if obs.len() != self.obs_size * 4 {
            return Err(EvaluatorError::InvalidState(format!(
                "Expected {} bytes for observation, got {}",
                self.obs_size * 4,
                obs.len()
            )));
        }

        let mut result = Vec::with_capacity(self.obs_size);
        for chunk in obs.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
        }
        Ok(result)
    }

    /// Apply softmax with masking for illegal moves, rejecting non-finite model output.
    fn masked_softmax(
        logits: &[f32],
        legal_mask: &LegalMask,
        num_actions: usize,
    ) -> Result<Vec<f32>, EvaluatorError> {
        if logits.len() != num_actions {
            return Err(EvaluatorError::ModelError(format!(
                "policy logits contain {} values, expected {num_actions}",
                logits.len()
            )));
        }
        if let Some((index, value)) = logits
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(EvaluatorError::EvaluationFailed(format!(
                "policy logit at action {index} is not finite: {value}"
            )));
        }

        let mut max_logit = f32::NEG_INFINITY;
        for (i, &logit) in logits.iter().enumerate().take(num_actions) {
            if legal_mask.is_legal(i) && logit > max_logit {
                max_logit = logit;
            }
        }

        // Handle case where no legal moves
        if max_logit == f32::NEG_INFINITY {
            return Ok(vec![0.0; num_actions]);
        }

        let mut exp_sum = 0.0;
        let mut exp_values = vec![0.0; num_actions];

        for (i, &logit) in logits.iter().enumerate().take(num_actions) {
            if legal_mask.is_legal(i) {
                let exp_val = (logit - max_logit).exp();
                exp_values[i] = exp_val;
                exp_sum += exp_val;
            }
        }

        if !exp_sum.is_finite() || exp_sum <= 0.0 {
            return Err(EvaluatorError::EvaluationFailed(format!(
                "masked policy normalization is invalid: {exp_sum}"
            )));
        }
        for value in &mut exp_values {
            *value /= exp_sum;
        }

        Ok(exp_values)
    }

    fn validate_value(value: f32, batch_index: usize) -> Result<f32, EvaluatorError> {
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(EvaluatorError::EvaluationFailed(format!(
                "value output at batch index {batch_index} must be finite and in [-1, 1], got {value}"
            )));
        }
        Ok(value)
    }
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(
        &self,
        obs: &[u8],
        legal_moves_mask: &LegalMask,
        num_actions: usize,
    ) -> Result<EvalResult, EvaluatorError> {
        if num_actions != self.num_actions {
            return Err(EvaluatorError::InvalidState(format!(
                "Evaluator was loaded for {} actions, evaluation requested {num_actions}",
                self.num_actions
            )));
        }
        if legal_moves_mask.num_actions() != self.num_actions {
            return Err(EvaluatorError::InvalidState(format!(
                "Legal mask has width {}, expected {}",
                legal_moves_mask.num_actions(),
                self.num_actions
            )));
        }

        // Track prep time: converting bytes and creating tensor
        let prep_start = Instant::now();

        // Convert observation bytes to f32 vector
        let obs_f32 = self.obs_bytes_to_f32(obs)?;

        // Create input tensor with shape (1, obs_size)
        // Use tuple (shape, data) format for compatibility across ort versions
        let input_value = Value::from_array(([1usize, self.obs_size], obs_f32)).map_err(|e| {
            EvaluatorError::ModelError(format!("Failed to create input tensor: {}", e))
        })?;

        let prep_time_us = prep_start.elapsed().as_micros() as u64;

        // Run inference - extract all data inside the lock scope
        let inference_start = Instant::now();
        let (policy_logits, value) = {
            let mut session = self.session.lock().map_err(|e| {
                EvaluatorError::EvaluationFailed(format!("Failed to acquire session lock: {}", e))
            })?;
            let outputs = session
                .run(ort::inputs!["observation" => input_value])
                .map_err(|e| {
                    EvaluatorError::EvaluationFailed(format!("Inference failed: {}", e))
                })?;

            // Extract policy logits - output is shape (1, action_size)
            let policy_output = outputs.get("policy_logits").ok_or_else(|| {
                EvaluatorError::ModelError("Missing policy_logits output".to_string())
            })?;

            let (policy_shape, policy_data) =
                policy_output.try_extract_tensor::<f32>().map_err(|e| {
                    EvaluatorError::ModelError(format!("Failed to extract policy tensor: {}", e))
                })?;
            Self::validate_runtime_tensor(
                "policy_logits",
                policy_shape,
                policy_data.len(),
                &[1, self.num_actions],
            )?;

            let policy_logits: Vec<f32> = policy_data.to_vec();

            // Extract value - output is shape (1, 1)
            let value_output = outputs
                .get("value")
                .ok_or_else(|| EvaluatorError::ModelError("Missing value output".to_string()))?;

            let (value_shape, value_data) =
                value_output.try_extract_tensor::<f32>().map_err(|e| {
                    EvaluatorError::ModelError(format!("Failed to extract value tensor: {}", e))
                })?;
            Self::validate_runtime_tensor("value", value_shape, value_data.len(), &[1, 1])?;

            let value = value_data[0];
            (policy_logits, value)
        };

        // Track inference timing for diagnostics
        let inference_time_us = inference_start.elapsed().as_micros() as u64;

        // Track post-processing time: softmax
        let post_start = Instant::now();
        let policy = Self::masked_softmax(&policy_logits, legal_moves_mask, self.num_actions)?;
        let value = Self::validate_value(value, 0)?;
        let post_time_us = post_start.elapsed().as_micros() as u64;

        // Update all timing stats
        self.total_prep_time_us
            .fetch_add(prep_time_us, Ordering::Relaxed);
        self.total_inference_time_us
            .fetch_add(inference_time_us, Ordering::Relaxed);
        self.total_post_time_us
            .fetch_add(post_time_us, Ordering::Relaxed);
        let count = self.inference_count.fetch_add(1, Ordering::Relaxed) + 1;

        // Log stats periodically (every 10,000 inferences)
        if count.is_multiple_of(10_000) {
            self.log_stats();
        }

        Ok(EvalResult { policy, value })
    }

    // The tensor-prep/session-lock/extract plumbing mirrors evaluate() above;
    // kept separate because the outputs are batch- vs single-shaped.
    fn evaluate_batch(
        &self,
        observations: &[&[u8]],
        legal_moves_masks: &[&LegalMask],
        num_actions: usize,
    ) -> Result<Vec<EvalResult>, EvaluatorError> {
        if num_actions != self.num_actions {
            return Err(EvaluatorError::InvalidState(format!(
                "Evaluator was loaded for {} actions, batch evaluation requested {num_actions}",
                self.num_actions
            )));
        }
        if observations.len() != legal_moves_masks.len() {
            return Err(EvaluatorError::InvalidState(format!(
                "Batch has {} observations but {} legal masks",
                observations.len(),
                legal_moves_masks.len()
            )));
        }
        if let Some((index, mask)) = legal_moves_masks
            .iter()
            .enumerate()
            .find(|(_, mask)| mask.num_actions() != self.num_actions)
        {
            return Err(EvaluatorError::InvalidState(format!(
                "Legal mask at batch index {index} has width {}, expected {}",
                mask.num_actions(),
                self.num_actions
            )));
        }
        if observations.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = observations.len();

        // Track prep time: converting bytes and creating tensor
        let prep_start = Instant::now();

        // Convert all observations to f32 and flatten
        let mut flat_obs = Vec::with_capacity(batch_size * self.obs_size);
        for obs in observations {
            let obs_f32 = self.obs_bytes_to_f32(obs)?;
            flat_obs.extend(obs_f32);
        }

        // Create input tensor with shape (batch_size, obs_size)
        // Use tuple (shape, data) format for compatibility across ort versions
        let input_value =
            Value::from_array(([batch_size, self.obs_size], flat_obs)).map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to create batch input tensor: {}", e))
            })?;

        let prep_time_us = prep_start.elapsed().as_micros() as u64;

        // Run inference - extract all data inside the lock scope
        let inference_start = Instant::now();
        let (policy_flat, values) = {
            let mut session = self.session.lock().map_err(|e| {
                EvaluatorError::EvaluationFailed(format!("Failed to acquire session lock: {}", e))
            })?;
            let outputs = session
                .run(ort::inputs!["observation" => input_value])
                .map_err(|e| {
                    EvaluatorError::EvaluationFailed(format!("Batch inference failed: {}", e))
                })?;

            // Extract policy logits - output is shape (batch_size, action_size)
            let policy_output = outputs.get("policy_logits").ok_or_else(|| {
                EvaluatorError::ModelError("Missing policy_logits output".to_string())
            })?;

            let (policy_shape, policy_data) =
                policy_output.try_extract_tensor::<f32>().map_err(|e| {
                    EvaluatorError::ModelError(format!("Failed to extract policy tensor: {}", e))
                })?;
            Self::validate_runtime_tensor(
                "policy_logits",
                policy_shape,
                policy_data.len(),
                &[batch_size, self.num_actions],
            )?;

            // Extract value - output is shape (batch_size, 1)
            let value_output = outputs
                .get("value")
                .ok_or_else(|| EvaluatorError::ModelError("Missing value output".to_string()))?;

            let (value_shape, value_data) =
                value_output.try_extract_tensor::<f32>().map_err(|e| {
                    EvaluatorError::ModelError(format!("Failed to extract value tensor: {}", e))
                })?;
            Self::validate_runtime_tensor(
                "value",
                value_shape,
                value_data.len(),
                &[batch_size, 1],
            )?;

            let policy_flat: Vec<f32> = policy_data.to_vec();
            let values: Vec<f32> = value_data.to_vec();
            (policy_flat, values)
        };

        // Track inference timing for diagnostics
        let inference_time_us = inference_start.elapsed().as_micros() as u64;

        // Track post-processing time: softmax for each item
        let post_start = Instant::now();

        // Build results for each batch item
        let mut results = Vec::with_capacity(batch_size);

        for (i, legal_mask) in legal_moves_masks.iter().enumerate() {
            let logits_start = i * self.num_actions;
            let logits_end = logits_start + self.num_actions;
            let logits = &policy_flat[logits_start..logits_end];

            let policy = Self::masked_softmax(logits, legal_mask, self.num_actions)?;
            let value = Self::validate_value(values[i], i)?;

            results.push(EvalResult { policy, value });
        }

        let post_time_us = post_start.elapsed().as_micros() as u64;

        // Update all timing stats (per-sample accounting for batch)
        let batch_size_u64 = batch_size as u64;
        self.total_prep_time_us
            .fetch_add(prep_time_us * batch_size_u64, Ordering::Relaxed);
        self.total_inference_time_us
            .fetch_add(inference_time_us * batch_size_u64, Ordering::Relaxed);
        self.total_post_time_us
            .fetch_add(post_time_us * batch_size_u64, Ordering::Relaxed);
        let count = self
            .inference_count
            .fetch_add(batch_size_u64, Ordering::Relaxed)
            + batch_size_u64;

        // Log stats periodically (every 10,000 inferences)
        if count.is_multiple_of(10_000) {
            self.log_stats();
        }

        Ok(results)
    }
}

/// A thread-safe wrapper around OnnxEvaluator.
/// This can be shared across threads for parallel MCTS.
pub struct SharedOnnxEvaluator {
    inner: Arc<OnnxEvaluator>,
}

impl SharedOnnxEvaluator {
    /// Create a new shared evaluator from an OnnxEvaluator.
    pub fn new(evaluator: OnnxEvaluator) -> Self {
        Self {
            inner: Arc::new(evaluator),
        }
    }

    /// Load a shared ONNX model file and require its artifact identity.
    pub fn load_from_file<P: AsRef<Path>>(
        model_path: P,
        obs_size: usize,
        num_actions: usize,
        intra_threads: usize,
        expected_contract: &ModelArtifactContract,
    ) -> Result<Self, EvaluatorError> {
        let evaluator = OnnxEvaluator::load_from_file(
            model_path,
            obs_size,
            num_actions,
            intra_threads,
            expected_contract,
        )?;
        Ok(Self::new(evaluator))
    }

    /// Load shared ONNX bytes and require their artifact identity.
    pub fn load_from_bytes(
        model_bytes: &[u8],
        obs_size: usize,
        num_actions: usize,
        intra_threads: usize,
        expected_contract: &ModelArtifactContract,
    ) -> Result<Self, EvaluatorError> {
        let evaluator = OnnxEvaluator::load_from_bytes(
            model_bytes,
            obs_size,
            num_actions,
            intra_threads,
            expected_contract,
        )?;
        Ok(Self::new(evaluator))
    }
}

impl Clone for SharedOnnxEvaluator {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Evaluator for SharedOnnxEvaluator {
    fn evaluate(
        &self,
        obs: &[u8],
        legal_moves_mask: &LegalMask,
        num_actions: usize,
    ) -> Result<EvalResult, EvaluatorError> {
        self.inner.evaluate(obs, legal_moves_mask, num_actions)
    }

    fn evaluate_batch(
        &self,
        observations: &[&[u8]],
        legal_moves_masks: &[&LegalMask],
        num_actions: usize,
    ) -> Result<Vec<EvalResult>, EvaluatorError> {
        self.inner
            .evaluate_batch(observations, legal_moves_masks, num_actions)
    }
}

#[cfg(test)]
#[path = "onnx_tests.rs"]
mod tests;
