//! Tests for the ONNX Runtime evaluator.

use super::*;
use algorithm_core::{resolve_algorithm, ALPHAZERO_BOARD_V1_ID};
use ort::value::{Shape, SymbolicDimensions};
use std::collections::HashMap;

// Minimal observation -> (policy_logits, value) ONNX graph with the complete
// `contract_test` artifact identity embedded as custom metadata.
const CONTRACT_TEST_MODEL: &[u8] = &[
    8, 8, 58, 202, 1, 10, 38, 10, 11, 111, 98, 115, 101, 114, 118, 97, 116, 105, 111, 110, 18, 13,
    112, 111, 108, 105, 99, 121, 95, 108, 111, 103, 105, 116, 115, 34, 8, 73, 100, 101, 110, 116,
    105, 116, 121, 10, 62, 10, 11, 111, 98, 115, 101, 114, 118, 97, 116, 105, 111, 110, 18, 5, 118,
    97, 108, 117, 101, 34, 10, 82, 101, 100, 117, 99, 101, 77, 101, 97, 110, 42, 11, 10, 4, 97,
    120, 101, 115, 64, 1, 160, 1, 7, 42, 15, 10, 8, 107, 101, 101, 112, 100, 105, 109, 115, 24, 1,
    160, 1, 2, 18, 13, 99, 111, 110, 116, 114, 97, 99, 116, 95, 116, 101, 115, 116, 90, 27, 10, 11,
    111, 98, 115, 101, 114, 118, 97, 116, 105, 111, 110, 18, 12, 10, 10, 8, 1, 18, 6, 10, 0, 10, 2,
    8, 3, 98, 29, 10, 13, 112, 111, 108, 105, 99, 121, 95, 108, 111, 103, 105, 116, 115, 18, 12,
    10, 10, 8, 1, 18, 6, 10, 0, 10, 2, 8, 3, 98, 21, 10, 5, 118, 97, 108, 117, 101, 18, 12, 10, 10,
    8, 1, 18, 6, 10, 0, 10, 2, 8, 1, 66, 4, 10, 0, 16, 13, 114, 29, 10, 24, 99, 97, 114, 116, 114,
    105, 100, 103, 101, 46, 115, 99, 104, 101, 109, 97, 95, 118, 101, 114, 115, 105, 111, 110, 18,
    1, 49, 114, 44, 10, 22, 99, 97, 114, 116, 114, 105, 100, 103, 101, 46, 97, 108, 103, 111, 114,
    105, 116, 104, 109, 95, 105, 100, 18, 18, 97, 108, 112, 104, 97, 122, 101, 114, 111, 95, 98,
    111, 97, 114, 100, 95, 118, 49, 114, 48, 10, 24, 99, 97, 114, 116, 114, 105, 100, 103, 101, 46,
    109, 111, 100, 101, 108, 95, 99, 111, 110, 116, 114, 97, 99, 116, 18, 20, 111, 110, 110, 120,
    95, 112, 111, 108, 105, 99, 121, 95, 118, 97, 108, 117, 101, 95, 118, 49, 114, 33, 10, 16, 99,
    97, 114, 116, 114, 105, 100, 103, 101, 46, 101, 110, 118, 95, 105, 100, 18, 13, 99, 111, 110,
    116, 114, 97, 99, 116, 95, 116, 101, 115, 116, 114, 35, 10, 30, 99, 97, 114, 116, 114, 105,
    100, 103, 101, 46, 101, 110, 118, 95, 99, 111, 110, 116, 114, 97, 99, 116, 95, 118, 101, 114,
    115, 105, 111, 110, 18, 1, 49,
];

fn identity() -> ModelArtifactContract {
    resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor()
        .model_artifact_contract("connect4", 1)
}

fn contract_test_identity() -> ModelArtifactContract {
    resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor()
        .model_artifact_contract("contract_test", 1)
}

#[test]
fn byte_loader_accepts_only_the_required_artifact_identity() {
    let expected = contract_test_identity();
    let evaluator =
        OnnxEvaluator::load_from_bytes(CONTRACT_TEST_MODEL, 3, 3, 1, &expected).unwrap();
    assert_eq!(evaluator.model_contract(), &expected);

    let wrong_env = resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor()
        .model_artifact_contract("different_env", 1);
    let error = OnnxEvaluator::load_from_bytes(CONTRACT_TEST_MODEL, 3, 3, 1, &wrong_env)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cartridge.env_id"));
    assert!(error.contains("different_env"));
    assert!(error.contains("contract_test"));
}

#[test]
fn loaders_reject_zero_intra_threads() {
    let expected = contract_test_identity();
    let error = OnnxEvaluator::load_from_bytes(CONTRACT_TEST_MODEL, 3, 3, 0, &expected)
        .unwrap_err()
        .to_string();
    assert!(error.contains("intra_threads must be greater than 0"));
}

#[test]
fn file_loader_validates_before_accepting_the_session() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.onnx");
    std::fs::write(&path, CONTRACT_TEST_MODEL).unwrap();

    let expected = contract_test_identity();
    let evaluator = OnnxEvaluator::load_from_file(&path, 3, 3, 1, &expected).unwrap();
    assert_eq!(evaluator.model_contract(), &expected);

    let mut wrong_contract = expected;
    wrong_contract.model_contract = "different_contract".into();
    let error = OnnxEvaluator::load_from_file(&path, 3, 3, 1, &wrong_contract)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cartridge.model_contract"));
}

#[test]
fn artifact_identity_requires_every_metadata_key() {
    let expected = identity();
    let metadata: HashMap<String, String> = expected
        .required_metadata()
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();

    OnnxEvaluator::validate_metadata_with(&expected, |key| metadata.get(key).cloned()).unwrap();

    let mut missing = metadata;
    missing.remove("cartridge.env_id");
    let error = OnnxEvaluator::validate_metadata_with(&expected, |key| missing.get(key).cloned())
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing required key 'cartridge.env_id'"));
}

#[test]
fn artifact_identity_reports_all_mismatches() {
    let expected = identity();
    let mut metadata: HashMap<String, String> = expected
        .required_metadata()
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    metadata.insert("cartridge.schema_version".into(), "2".into());
    metadata.insert("cartridge.algorithm_id".into(), "ppo_v1".into());
    metadata.insert("cartridge.model_contract".into(), "actor_critic_v1".into());
    metadata.insert("cartridge.env_id".into(), "othello".into());

    let error = OnnxEvaluator::validate_metadata_with(&expected, |key| metadata.get(key).cloned())
        .unwrap_err()
        .to_string();
    for key in [
        "cartridge.schema_version",
        "cartridge.algorithm_id",
        "cartridge.model_contract",
        "cartridge.env_id",
    ] {
        assert!(error.contains(key), "missing {key} from {error}");
    }
}

fn tensor_type(ty: TensorElementType, shape: [i64; 2]) -> ValueType {
    ValueType::Tensor {
        ty,
        shape: Shape::new(shape),
        dimension_symbols: SymbolicDimensions::empty(2),
    }
}

#[test]
fn loader_rejects_contract_dimension_mismatches() {
    let expected = contract_test_identity();

    let observation_error = OnnxEvaluator::load_from_bytes(CONTRACT_TEST_MODEL, 4, 3, 1, &expected)
        .unwrap_err()
        .to_string();
    assert!(observation_error.contains("input 'observation'"));
    assert!(observation_error.contains("dynamic_batch, 4"));

    let policy_error = OnnxEvaluator::load_from_bytes(CONTRACT_TEST_MODEL, 3, 4, 1, &expected)
        .unwrap_err()
        .to_string();
    assert!(policy_error.contains("output 'policy_logits'"));
    assert!(policy_error.contains("dynamic_batch, 4"));
}

#[test]
fn interface_requires_f32_dynamic_batch_and_exact_io_set() {
    let observation = tensor_type(TensorElementType::Float32, [-1, 3]);
    let policy = tensor_type(TensorElementType::Float32, [-1, 3]);
    let value = tensor_type(TensorElementType::Float32, [-1, 1]);

    OnnxEvaluator::validate_interface_with(
        &[("observation", &observation)],
        &[("policy_logits", &policy), ("value", &value)],
        3,
        3,
    )
    .unwrap();

    let error = OnnxEvaluator::validate_interface_with(
        &[("wrong_name", &observation)],
        &[("policy_logits", &policy), ("value", &value)],
        3,
        3,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("missing input 'observation'"));

    let fixed_observation = tensor_type(TensorElementType::Float32, [1, 3]);
    let error = OnnxEvaluator::validate_interface_with(
        &[("observation", &fixed_observation)],
        &[("policy_logits", &policy), ("value", &value)],
        3,
        3,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("dynamic_batch"));

    let integer_policy = tensor_type(TensorElementType::Int64, [-1, 3]);
    let error = OnnxEvaluator::validate_interface_with(
        &[("observation", &observation)],
        &[("policy_logits", &integer_policy), ("value", &value)],
        3,
        3,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("must use f32"));

    let extra = tensor_type(TensorElementType::Float32, [-1, 1]);
    let error = OnnxEvaluator::validate_interface_with(
        &[("observation", &observation)],
        &[
            ("policy_logits", &policy),
            ("value", &value),
            ("extra", &extra),
        ],
        3,
        3,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("exactly 2 outputs"));
}

#[test]
fn runtime_outputs_require_exact_shape_and_element_count() {
    OnnxEvaluator::validate_runtime_tensor("policy_logits", &[2, 3], 6, &[2, 3]).unwrap();

    let shape_error = OnnxEvaluator::validate_runtime_tensor("policy_logits", &[1, 6], 6, &[2, 3])
        .unwrap_err()
        .to_string();
    assert!(shape_error.contains("expected shape [2, 3]"));
    assert!(shape_error.contains("found shape [1, 6]"));

    let count_error = OnnxEvaluator::validate_runtime_tensor("value", &[2, 1], 1, &[2, 1])
        .unwrap_err()
        .to_string();
    assert!(count_error.contains("with 2 elements"));
    assert!(count_error.contains("with 1 elements"));
}

#[test]
fn evaluation_rejects_action_mask_and_batch_cardinality_mismatches() {
    let expected = contract_test_identity();
    let evaluator =
        OnnxEvaluator::load_from_bytes(CONTRACT_TEST_MODEL, 3, 3, 1, &expected).unwrap();
    let observation = [0.0f32; 3]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let mask = LegalMask::all_legal(3);

    let result = evaluator.evaluate(&observation, &mask, 3).unwrap();
    assert_eq!(result.policy.len(), 3);
    assert_eq!(result.value, 0.0);

    let action_error = evaluator
        .evaluate(&observation, &mask, 4)
        .unwrap_err()
        .to_string();
    assert!(action_error.contains("loaded for 3 actions"));

    let mask_error = evaluator
        .evaluate(&observation, &LegalMask::all_legal(4), 3)
        .unwrap_err()
        .to_string();
    assert!(mask_error.contains("mask has width 4"));

    let batch_error = evaluator
        .evaluate_batch(&[&observation, &observation], &[&mask], 3)
        .unwrap_err()
        .to_string();
    assert!(batch_error.contains("2 observations but 1 legal masks"));
}

#[test]
fn test_masked_softmax_all_legal() {
    let logits = vec![1.0, 2.0, 3.0];
    let mask = LegalMask::from_u64(0b111, 3);
    let policy = OnnxEvaluator::masked_softmax(&logits, &mask, 3).unwrap();

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
    let mask = LegalMask::from_u64(0b0101, 4); // Only actions 0 and 2 are legal
    let policy = OnnxEvaluator::masked_softmax(&logits, &mask, 4).unwrap();

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
    let mask = LegalMask::new(3); // No legal moves
    let policy = OnnxEvaluator::masked_softmax(&logits, &mask, 3).unwrap();

    // All should be 0
    for p in &policy {
        assert!(p.abs() < 1e-6);
    }
}

#[test]
fn non_finite_policy_and_out_of_contract_values_are_rejected() {
    let mask = LegalMask::all_legal(3);
    let error = OnnxEvaluator::masked_softmax(&[0.0, f32::NAN, 1.0], &mask, 3)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not finite"));

    for value in [f32::NAN, f32::INFINITY, -1.01, 1.01] {
        assert!(OnnxEvaluator::validate_value(value, 2).is_err());
    }
    assert_eq!(OnnxEvaluator::validate_value(-1.0, 0).unwrap(), -1.0);
    assert_eq!(OnnxEvaluator::validate_value(1.0, 0).unwrap(), 1.0);
}
