use super::*;
use algorithm_core::{resolve_algorithm, ModelArtifactContract, ALPHAZERO_BOARD_V1_ID};

fn identity() -> ModelArtifactContract {
    resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor()
        .model_artifact_contract("tictactoe", 1)
}

fn manifest() -> CheckpointManifestV1 {
    CheckpointManifestV1 {
        schema_version: 1,
        profile: ArtifactProfile::from(&identity()),
        step: 42,
        parent_checkpoint_id: None,
        config_sha256: "a".repeat(64),
        onnx: BlobReference {
            sha256: "b".repeat(64),
            size_bytes: 10,
        },
        learner_state: BlobReference {
            sha256: "c".repeat(64),
            size_bytes: 20,
        },
    }
}

#[test]
fn digest_requires_lowercase_sha256() {
    assert!(validate_digest("value", &"a".repeat(64)).is_ok());
    assert!(validate_digest("value", &"A".repeat(64)).is_err());
    assert!(validate_digest("value", "abc").is_err());
}

#[test]
fn canonical_json_rejects_whitespace_and_unknown_fields() {
    let valid = br#"{"checkpoint_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","run_commit_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","schema_version":2}"#;
    let head: RunHeadV2 = parse_canonical_json("run head", valid).unwrap();
    validate_run_head(&head).unwrap();

    assert!(parse_canonical_json::<RunHeadV2>("run head", b"{ }").is_err());
    let unknown = br#"{"checkpoint_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","extra":1,"run_commit_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","schema_version":2}"#;
    assert!(parse_canonical_json::<RunHeadV2>("run head", unknown).is_err());
}

#[test]
fn canonical_json_unicode_is_raw_utf8_cross_language_golden() {
    // The Python publisher (trainer.storage.publisher.canonical_json_bytes)
    // asserts these exact golden bytes; both sides must agree
    // byte-for-byte or content-addressed IDs diverge across the boundary.
    let golden = "{\"name\":\"café\",\"piece\":\"♟\",\"z\":1}".as_bytes();
    assert_eq!(
        sha256_hex(golden),
        "2e4360a215d64b8654fc51e28743d8761b5816bef04a30ceefa3314a1f151189"
    );
    let value: serde_json::Value = parse_canonical_json("unicode value", golden).unwrap();
    assert_eq!(value["name"], "café");

    // Python's default ASCII-escaped spelling of the same object is NOT
    // canonical here and must be rejected.
    let escaped = br#"{"name":"caf\u00e9","piece":"\u265f","z":1}"#;
    let error = parse_canonical_json::<serde_json::Value>("unicode value", escaped)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not canonical JSON"), "{error}");
}

#[test]
fn manifest_is_bound_to_exact_runtime_profile() {
    validate_manifest(&manifest(), &identity()).unwrap();
    let other = resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor()
        .model_artifact_contract("connect4", 1);
    assert!(validate_manifest(&manifest(), &other).is_err());
}

#[test]
fn blob_verification_checks_size_and_digest() {
    let bytes = b"model";
    let reference = BlobReference {
        sha256: sha256_hex(bytes),
        size_bytes: bytes.len() as u64,
    };
    verify_blob_bytes("test", bytes, &reference).unwrap();
    assert!(verify_blob_bytes("test", b"other", &reference).is_err());
}

#[test]
fn orchestration_timestamps_require_real_utc_calendar_values() {
    assert!(validate_utc_timestamp("timestamp", "2024-02-29T23:59:59.000001Z").is_ok());
    assert!(validate_utc_timestamp("timestamp", "2025-02-29T23:59:59.000001Z").is_err());
    assert!(validate_utc_timestamp("timestamp", "2024-12-31T24:00:00.000000Z").is_err());
}
