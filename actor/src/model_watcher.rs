//! Model file watcher for hot-reloading ONNX models
//!
//! This module re-exports the shared `model_watcher` crate.
//! See that crate for full documentation.

pub use model_watcher::ModelWatcher;

// ============================================================================
// Unit Tests - Verify re-exports work correctly
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, RwLock};

    /// Test that ModelWatcher can be instantiated through the re-export
    #[test]
    fn test_model_watcher_reexport_basic() {
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator);

        // Verify the watcher was created
        assert_eq!(
            watcher.model_path().to_string_lossy(),
            "/tmp/models/latest.onnx"
        );
    }

    /// Test that ModelWatcher with_metadata works through re-export
    #[test]
    fn test_model_watcher_reexport_with_metadata() {
        let evaluator = Arc::new(RwLock::new(None));
        let watcher =
            ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator).with_metadata();

        let info = watcher.model_info();
        let guard = info.read().unwrap();
        assert!(!guard.loaded);
    }

    /// Test that ModelWatcher with_poll_interval works through re-export
    #[test]
    fn test_model_watcher_reexport_with_poll_interval() {
        use std::time::Duration;

        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator)
            .with_poll_interval(Duration::from_secs(10));

        // Just verify it compiles and runs - the poll interval is internal state
        assert_eq!(
            watcher.model_path().to_string_lossy(),
            "/tmp/models/latest.onnx"
        );
    }

    /// Test that try_load_existing returns Ok for non-existent model
    #[test]
    fn test_model_watcher_reexport_try_load_nonexistent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new(temp_dir.path(), "nonexistent.onnx", 29, 1, evaluator);

        let result = watcher.try_load_existing();
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    /// Test creating multiple watchers with different configurations
    #[test]
    fn test_model_watcher_reexport_multiple_configs() {
        let evaluator1 = Arc::new(RwLock::new(None));
        let evaluator2 = Arc::new(RwLock::new(None));

        let watcher1 = ModelWatcher::new("/tmp/models1", "model1.onnx", 29, 1, evaluator1);
        let watcher2 = ModelWatcher::new("/tmp/models2", "model2.onnx", 116, 4, evaluator2);

        assert_eq!(
            watcher1.model_path().to_string_lossy(),
            "/tmp/models1/model1.onnx"
        );
        assert_eq!(
            watcher2.model_path().to_string_lossy(),
            "/tmp/models2/model2.onnx"
        );
    }
}
