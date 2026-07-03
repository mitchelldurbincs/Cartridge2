//! Tests for the ONNX Runtime evaluator.

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
