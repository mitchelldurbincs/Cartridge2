use super::*;

#[test]
fn object_keys_follow_the_content_addressed_protocol() {
    assert_eq!(
        S3ModelWatcher::run_head_key("profiles/example/models"),
        "profiles/example/models/channels/current.json"
    );
    assert_eq!(
        S3ModelWatcher::manifest_key("profiles/example/models", "a"),
        "profiles/example/models/manifests/sha256/a.json"
    );
    assert_eq!(
        S3ModelWatcher::run_commit_key("profiles/example/models", "c"),
        "profiles/example/models/run-commits/sha256/c.json"
    );
    assert_eq!(
        S3ModelWatcher::model_key("profiles/example/models", "b"),
        "profiles/example/models/blobs/sha256/b.onnx"
    );
}

#[tokio::test]
async fn concurrent_cache_publication_is_create_or_verify() {
    let directory = tempfile::tempdir().unwrap();
    let bytes = b"immutable model";
    let digest = sha256_hex(bytes);
    let first = S3ModelWatcher::cache_model(directory.path(), &digest, bytes);
    let second = S3ModelWatcher::cache_model(directory.path(), &digest, bytes);
    let (first, second) = tokio::join!(first, second);

    let first = first.unwrap();
    assert_eq!(second.unwrap(), first);
    assert_eq!(tokio::fs::read(&first).await.unwrap(), bytes);
    let entries = std::fs::read_dir(first.parent().unwrap())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(entries.len(), 1);
}

#[test]
fn structured_absence_detection_is_narrow() {
    let missing = GetObjectError::NoSuchKey(aws_sdk_s3::types::error::NoSuchKey::builder().build());
    assert!(S3ModelWatcher::is_absent(&missing));
}
