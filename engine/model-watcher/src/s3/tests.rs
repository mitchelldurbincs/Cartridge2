use super::*;

#[test]
fn test_s3_config() {
    let config = S3Config {
        bucket: "test-bucket".to_string(),
        key: "models/latest.onnx".to_string(),
        endpoint_url: Some("http://localhost:9000".to_string()),
        region: Some("us-east-1".to_string()),
        cache_dir: PathBuf::from("/tmp/models"),
    };

    assert_eq!(config.bucket, "test-bucket");
    assert_eq!(config.key, "models/latest.onnx");
}

#[cfg(feature = "metadata")]
#[test]
fn test_extract_training_step() {
    let path = PathBuf::from("/tmp/model_step_000100.onnx");
    let step = S3ModelWatcher::extract_training_step(&path);
    assert_eq!(step, Some(100));

    let path = PathBuf::from("/tmp/latest.onnx");
    let step = S3ModelWatcher::extract_training_step(&path);
    assert!(step.is_none());
}
