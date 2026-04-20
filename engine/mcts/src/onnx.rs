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

use ort::{session::Session, value::Value};
use tracing::debug;

use crate::evaluator::{EvalResult, Evaluator, EvaluatorError};

/// ONNX Runtime evaluator that loads and runs neural network models.
///
/// Uses a Mutex internally because `Session::run` requires `&mut self`,
/// but the `Evaluator` trait uses `&self` for thread-safe sharing.
pub struct OnnxEvaluator {
    session: Mutex<Session>,
    obs_size: usize,
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
            .finish_non_exhaustive()
    }
}

impl OnnxEvaluator {
    /// Resolve the intra_threads setting:
    /// - 0 = auto-detect (use all available CPU cores)
    /// - n > 0 = use exactly n threads
    fn resolve_intra_threads(intra_threads: usize) -> usize {
        if intra_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            intra_threads
        }
    }

    /// Register CoreML execution provider if the `coreml` feature is enabled.
    /// Falls back silently to CPU if CoreML is unavailable (e.g. on non-Apple platforms).
    fn register_coreml_ep(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, EvaluatorError> {
        #[cfg(feature = "coreml")]
        {
            let coreml_ep = ort::ep::CoreML::default().build();
            debug!("Attempting to register CoreML execution provider");
            return builder.with_execution_providers([coreml_ep]).map_err(|e| {
                EvaluatorError::ModelError(format!(
                    "Failed to register CoreML execution provider: {}",
                    e
                ))
            });
        }
        #[cfg(not(feature = "coreml"))]
        {
            Ok(builder)
        }
    }

    /// Load an ONNX model from the given path.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `obs_size` - Size of the observation vector (e.g., 29 for TicTacToe)
    /// * `intra_threads` - Number of intra-op threads (0 = auto-detect, uses all CPU cores)
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        obs_size: usize,
        intra_threads: usize,
    ) -> Result<Self, EvaluatorError> {
        let threads = Self::resolve_intra_threads(intra_threads);
        let builder = Session::builder()
            .map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to create session builder: {}", e))
            })?
            .with_intra_threads(threads)
            .map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to set intra threads: {}", e))
            })?;

        let mut builder = Self::register_coreml_ep(builder)?;

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| EvaluatorError::ModelError(format!("Failed to load model: {}", e)))?;

        Ok(Self {
            session: Mutex::new(session),
            obs_size,
            inference_count: AtomicU64::new(0),
            total_inference_time_us: AtomicU64::new(0),
            total_prep_time_us: AtomicU64::new(0),
            total_post_time_us: AtomicU64::new(0),
        })
    }

    /// Load an ONNX model from memory.
    ///
    /// # Arguments
    /// * `model_data` - Raw ONNX model bytes
    /// * `obs_size` - Size of the observation vector
    /// * `intra_threads` - Number of intra-op threads (0 = auto-detect, uses all CPU cores)
    pub fn load_from_memory(
        model_data: &[u8],
        obs_size: usize,
        intra_threads: usize,
    ) -> Result<Self, EvaluatorError> {
        let threads = Self::resolve_intra_threads(intra_threads);
        let builder = Session::builder()
            .map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to create session builder: {}", e))
            })?
            .with_intra_threads(threads)
            .map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to set intra threads: {}", e))
            })?;

        let mut builder = Self::register_coreml_ep(builder)?;

        let session = builder.commit_from_memory(model_data).map_err(|e| {
            EvaluatorError::ModelError(format!("Failed to load model from memory: {}", e))
        })?;

        Ok(Self {
            session: Mutex::new(session),
            obs_size,
            inference_count: AtomicU64::new(0),
            total_inference_time_us: AtomicU64::new(0),
            total_prep_time_us: AtomicU64::new(0),
            total_post_time_us: AtomicU64::new(0),
        })
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

    /// Apply softmax with masking for illegal moves.
    fn masked_softmax(logits: &[f32], legal_mask: u64, num_actions: usize) -> Vec<f32> {
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &logit) in logits.iter().enumerate().take(num_actions) {
            if (legal_mask >> i) & 1 == 1 && logit > max_logit {
                max_logit = logit;
            }
        }

        // Handle case where no legal moves
        if max_logit == f32::NEG_INFINITY {
            return vec![0.0; num_actions];
        }

        let mut exp_sum = 0.0;
        let mut exp_values = vec![0.0; num_actions];

        for (i, &logit) in logits.iter().enumerate().take(num_actions) {
            if (legal_mask >> i) & 1 == 1 {
                let exp_val = (logit - max_logit).exp();
                exp_values[i] = exp_val;
                exp_sum += exp_val;
            }
        }

        if exp_sum > 0.0 {
            for v in &mut exp_values {
                *v /= exp_sum;
            }
        }

        exp_values
    }
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(
        &self,
        obs: &[u8],
        legal_moves_mask: u64,
        num_actions: usize,
    ) -> Result<EvalResult, EvaluatorError> {
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

            let (_shape, policy_data) = policy_output.try_extract_tensor::<f32>().map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to extract policy tensor: {}", e))
            })?;

            let policy_logits: Vec<f32> = policy_data.to_vec();

            // Extract value - output is shape (1, 1)
            let value_output = outputs
                .get("value")
                .ok_or_else(|| EvaluatorError::ModelError("Missing value output".to_string()))?;

            let (_shape, value_data) = value_output.try_extract_tensor::<f32>().map_err(|e| {
                EvaluatorError::ModelError(format!("Failed to extract value tensor: {}", e))
            })?;

            let value = value_data.first().cloned().unwrap_or(0.0);
            (policy_logits, value)
        };

        // Track inference timing for diagnostics
        let inference_time_us = inference_start.elapsed().as_micros() as u64;

        // Track post-processing time: softmax
        let post_start = Instant::now();
        let policy = Self::masked_softmax(&policy_logits, legal_moves_mask, num_actions);
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

    fn evaluate_batch(
        &self,
        observations: &[&[u8]],
        legal_moves_masks: &[u64],
        num_actions: usize,
    ) -> Result<Vec<EvalResult>, EvaluatorError> {
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
        let (policy_flat, values, action_size) = {
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

            let action_size = if policy_shape.len() > 1 {
                policy_shape[1] as usize
            } else {
                num_actions
            };

            // Extract value - output is shape (batch_size, 1)
            let value_output = outputs
                .get("value")
                .ok_or_else(|| EvaluatorError::ModelError("Missing value output".to_string()))?;

            let (_value_shape, value_data) =
                value_output.try_extract_tensor::<f32>().map_err(|e| {
                    EvaluatorError::ModelError(format!("Failed to extract value tensor: {}", e))
                })?;

            let policy_flat: Vec<f32> = policy_data.to_vec();
            let values: Vec<f32> = value_data.to_vec();
            (policy_flat, values, action_size)
        };

        // Track inference timing for diagnostics
        let inference_time_us = inference_start.elapsed().as_micros() as u64;

        // Track post-processing time: softmax for each item
        let post_start = Instant::now();

        // Build results for each batch item
        let mut results = Vec::with_capacity(batch_size);

        for (i, &legal_mask) in legal_moves_masks.iter().enumerate().take(batch_size) {
            let logits_start = i * action_size;
            let logits_end = logits_start + action_size;
            let logits = &policy_flat[logits_start..logits_end];

            let policy = Self::masked_softmax(logits, legal_mask, num_actions);
            let value = values.get(i).cloned().unwrap_or(0.0);

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

    /// Load a shared ONNX model from the given path.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `obs_size` - Size of the observation vector
    /// * `intra_threads` - Number of intra-op threads (0 = auto-detect)
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        obs_size: usize,
        intra_threads: usize,
    ) -> Result<Self, EvaluatorError> {
        let evaluator = OnnxEvaluator::load(model_path, obs_size, intra_threads)?;
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
        legal_moves_mask: u64,
        num_actions: usize,
    ) -> Result<EvalResult, EvaluatorError> {
        self.inner.evaluate(obs, legal_moves_mask, num_actions)
    }

    fn evaluate_batch(
        &self,
        observations: &[&[u8]],
        legal_moves_masks: &[u64],
        num_actions: usize,
    ) -> Result<Vec<EvalResult>, EvaluatorError> {
        self.inner
            .evaluate_batch(observations, legal_moves_masks, num_actions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_softmax_all_legal() {
        let logits = vec![1.0, 2.0, 3.0];
        let mask = 0b111;
        let policy = OnnxEvaluator::masked_softmax(&logits, mask, 3);

        // Should sum to 1.0
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit should have higher probability
        assert!(policy[2] > policy[1]);
        assert!(policy[1] > policy[0]);
    }

    #[test]
    fn test_masked_softmax_with_illegal() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mask = 0b0101; // Only actions 0 and 2 are legal
        let policy = OnnxEvaluator::masked_softmax(&logits, mask, 4);

        // Sum should be 1.0
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Illegal moves should be 0
        assert!(policy[1].abs() < 1e-6);
        assert!(policy[3].abs() < 1e-6);

        // Legal move 2 (logit=3.0) should be higher than legal move 0 (logit=1.0)
        assert!(policy[2] > policy[0]);
    }

    #[test]
    fn test_masked_softmax_no_legal() {
        let logits = vec![1.0, 2.0, 3.0];
        let mask = 0b0; // No legal moves
        let policy = OnnxEvaluator::masked_softmax(&logits, mask, 3);

        // All should be 0
        for p in &policy {
            assert!(p.abs() < 1e-6);
        }
    }
}
