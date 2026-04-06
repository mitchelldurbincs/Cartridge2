//! Model file watcher for hot-reloading ONNX models in the web server
//!
//! This module re-exports the shared `model_watcher` crate with the
//! `metadata` feature enabled for tracking model info.
//! See that crate for full documentation.

pub use model_watcher::{ModelInfo, ModelWatcher};

// ============================================================================
// Unit Tests - Verify re-exports work correctly for web server
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, RwLock};

    /// Test that ModelInfo can be created through the web re-export
    #[test]
    fn test_model_info_default_web() {
        let info = ModelInfo::default();
        assert!(!info.loaded);
        assert!(info.path.is_none());
        assert!(info.file_modified.is_none());
        assert!(info.loaded_at.is_none());
        assert!(info.training_step.is_none());
    }

    /// Test that ModelWatcher can be instantiated through the web re-export
    #[test]
    fn test_model_watcher_reexport_web_basic() {
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator);

        assert_eq!(
            watcher.model_path().to_string_lossy(),
            "/tmp/models/latest.onnx"
        );
    }

    /// Test that with_metadata method works through web re-export
    #[test]
    fn test_model_watcher_reexport_with_metadata_web() {
        let evaluator = Arc::new(RwLock::new(None));
        let watcher =
            ModelWatcher::new("/tmp/models", "latest.onnx", 29, 1, evaluator).with_metadata();

        let info = watcher.model_info();
        let guard = info.read().unwrap();
        assert!(!guard.loaded);
    }

    /// Test creating multiple watchers with different observation sizes (common web use case)
    #[test]
    fn test_model_watcher_web_different_obs_sizes() {
        // TicTacToe uses obs_size 29
        let evaluator1 = Arc::new(RwLock::new(None));
        let watcher1 = ModelWatcher::new("/tmp/models1", "tictactoe.onnx", 29, 1, evaluator1);

        // Connect4 uses obs_size 116
        let evaluator2 = Arc::new(RwLock::new(None));
        let watcher2 = ModelWatcher::new("/tmp/models2", "connect4.onnx", 116, 1, evaluator2);

        assert_eq!(
            watcher1.model_path().to_string_lossy(),
            "/tmp/models1/tictactoe.onnx"
        );
        assert_eq!(
            watcher2.model_path().to_string_lossy(),
            "/tmp/models2/connect4.onnx"
        );
    }

    /// Test try_load_existing returns Ok for non-existent model
    #[test]
    fn test_model_watcher_reexport_try_load_nonexistent_web() {
        let temp_dir = tempfile::tempdir().unwrap();
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new(temp_dir.path(), "nonexistent.onnx", 29, 1, evaluator);

        let result = watcher.try_load_existing();
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    /// Test ModelInfo fields can be updated
    #[test]
    fn test_model_info_manual_update() {
        let info = Arc::new(RwLock::new(ModelInfo::default()));

        {
            let mut guard = info.write().unwrap();
            guard.loaded = true;
            guard.path = Some("/models/test.onnx".to_string());
            guard.file_modified = Some(1234567890);
            guard.loaded_at = Some(1234567891);
            guard.training_step = Some(500);
        }

        let guard = info.read().unwrap();
        assert!(guard.loaded);
        assert_eq!(guard.path, Some("/models/test.onnx".to_string()));
        assert_eq!(guard.file_modified, Some(1234567890));
        assert_eq!(guard.loaded_at, Some(1234567891));
        assert_eq!(guard.training_step, Some(500));
    }
}
