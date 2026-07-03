use super::*;
use crate::load::extract_metadata;
use tempfile::tempdir;

#[test]
fn test_model_watcher_creation() {
    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator);

    assert_eq!(
        watcher.model_path(),
        PathBuf::from("/tmp/models/latest.onnx")
    );
}

#[test]
fn test_try_load_nonexistent() {
    let temp_dir = tempdir().unwrap();
    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new(
        temp_dir.path(),
        "nonexistent.onnx",
        29,
        1,
        evaluator.clone(),
    );

    let result = watcher.try_load_existing();
    assert!(result.is_ok());
    assert!(!result.unwrap()); // Should return false - no file

    // Evaluator should still be None
    let guard = evaluator.read().unwrap();
    assert!(guard.is_none());
}

#[test]
fn test_model_info_default() {
    let info = ModelInfo::default();
    assert!(!info.loaded);
    assert!(info.path.is_none());
    assert!(info.file_modified.is_none());
    assert!(info.loaded_at.is_none());
    assert!(info.training_step.is_none());
}

#[test]
fn test_extract_metadata_training_step() {
    // Test with step in filename
    let path = Path::new("/tmp/models/model_step_000100.onnx");
    let (_, training_step) = extract_metadata(path);
    assert_eq!(training_step, Some(100));

    // Test without step
    let path = Path::new("/tmp/models/latest.onnx");
    let (_, training_step) = extract_metadata(path);
    assert!(training_step.is_none());
}

#[test]
fn test_with_metadata_creates_model_info() {
    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator);
    assert!(watcher.model_info.is_none());

    let evaluator2 = Arc::new(RwLock::new(None));
    let watcher2 =
        ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator2).with_metadata();
    assert!(watcher2.model_info.is_some());
}
