//! Tests for the configuration module.

use super::*;

#[test]
fn test_default_config() {
    let config = CentralConfig::default();
    assert_eq!(config.common.env_id, "tictactoe");
    assert_eq!(config.common.data_dir, "./data");
    assert_eq!(config.common.log_level, "info");
    assert_eq!(config.actor.actor_id, "actor-1");
    assert_eq!(config.actor.max_episodes, -1);
    assert_eq!(config.web.host, "0.0.0.0");
    assert_eq!(config.web.port, 8080);
    assert_eq!(config.mcts.num_simulations, 800);
}

#[test]
fn test_training_defaults() {
    let config = CentralConfig::default();
    assert_eq!(config.training.iterations, 100);
    assert_eq!(config.training.start_iteration, 1);
    assert_eq!(config.training.episodes_per_iteration, 500);
    assert_eq!(config.training.steps_per_iteration, 1000);
    assert_eq!(config.training.batch_size, 64);
    assert!((config.training.learning_rate - 0.001).abs() < f64::EPSILON);
    assert!((config.training.weight_decay - 0.0001).abs() < f64::EPSILON);
    assert!((config.training.grad_clip_norm - 1.0).abs() < f64::EPSILON);
    assert_eq!(config.training.device, "cpu");
    assert_eq!(config.training.checkpoint_interval, 100);
    assert_eq!(config.training.max_checkpoints, 10);
}

#[test]
fn test_evaluation_defaults() {
    let config = CentralConfig::default();
    assert_eq!(config.evaluation.interval, 1);
    assert_eq!(config.evaluation.games, 50);
}

#[test]
fn test_mcts_defaults() {
    let config = CentralConfig::default();
    assert_eq!(config.mcts.num_simulations, 800);
    assert!((config.mcts.c_puct - 1.4).abs() < f64::EPSILON);
    assert!((config.mcts.temperature - 1.0).abs() < f64::EPSILON);
    assert!((config.mcts.dirichlet_alpha - 0.3).abs() < f64::EPSILON);
    assert!((config.mcts.dirichlet_weight - 0.25).abs() < f64::EPSILON);
    assert_eq!(config.mcts.eval_batch_size, 32);
    assert_eq!(config.mcts.onnx_intra_threads, 1);
}

#[test]
fn test_storage_config_defaults() {
    let config = CentralConfig::default();
    assert_eq!(config.storage.model_backend, "filesystem");
    assert_eq!(
        config.storage.postgres_url,
        Some("postgresql://cartridge:cartridge@localhost:5432/cartridge".to_string())
    );
    assert!(config.storage.s3_bucket.is_none());
    assert!(config.storage.s3_endpoint.is_none());
    assert_eq!(config.storage.pool_max_size, 16);
    assert_eq!(config.storage.pool_connect_timeout, 30);
    assert_eq!(config.storage.pool_idle_timeout, Some(300));
}

#[test]
fn test_cartridge_env_overrides() {
    std::env::set_var("CARTRIDGE_COMMON_ENV_ID", "connect4");
    std::env::set_var("CARTRIDGE_ACTOR_MAX_EPISODES", "7");
    std::env::set_var("CARTRIDGE_TRAINING_WEIGHT_DECAY", "0.5");

    let config = load_config();
    assert_eq!(config.common.env_id, "connect4");
    assert_eq!(config.actor.max_episodes, 7);
    assert!((config.training.weight_decay - 0.5).abs() < f64::EPSILON);

    std::env::remove_var("CARTRIDGE_COMMON_ENV_ID");
    std::env::remove_var("CARTRIDGE_ACTOR_MAX_EPISODES");
    std::env::remove_var("CARTRIDGE_TRAINING_WEIGHT_DECAY");
}

#[test]
fn test_parse_config_toml() {
    let toml_content = r#"
[common]
env_id = "connect4"
data_dir = "/custom/data"

[actor]
actor_id = "my-actor"
max_episodes = 100

[training]
iterations = 50
batch_size = 128
"#;
    let config: CentralConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.common.env_id, "connect4");
    assert_eq!(config.common.data_dir, "/custom/data");
    assert_eq!(config.actor.actor_id, "my-actor");
    assert_eq!(config.actor.max_episodes, 100);
    assert_eq!(config.training.iterations, 50);
    assert_eq!(config.training.batch_size, 128);
}

#[test]
fn test_partial_config() {
    let toml_content = r#"
[common]
env_id = "connect4"
"#;
    let config: CentralConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.common.env_id, "connect4");
    assert_eq!(config.common.data_dir, "./data"); // Default
    assert_eq!(config.actor.actor_id, "actor-1"); // Default
    assert_eq!(config.web.port, 8080); // Default
}

#[test]
fn test_storage_config_from_toml() {
    let toml_content = r#"
[storage]
model_backend = "s3"
postgres_url = "postgresql://user:pass@localhost:5432/cartridge"
s3_bucket = "my-bucket"
s3_endpoint = "http://minio:9000"
pool_max_size = 32
"#;
    let config: CentralConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.storage.model_backend, "s3");
    assert_eq!(
        config.storage.postgres_url,
        Some("postgresql://user:pass@localhost:5432/cartridge".to_string())
    );
    assert_eq!(config.storage.s3_bucket, Some("my-bucket".to_string()));
    assert_eq!(
        config.storage.s3_endpoint,
        Some("http://minio:9000".to_string())
    );
    assert_eq!(config.storage.pool_max_size, 32);
}

#[test]
fn test_storage_env_overrides() {
    std::env::set_var("CARTRIDGE_STORAGE_MODEL_BACKEND", "s3");
    std::env::set_var(
        "CARTRIDGE_STORAGE_POSTGRES_URL",
        "postgresql://test@localhost/db",
    );

    let config = load_config();
    assert_eq!(config.storage.model_backend, "s3");
    assert_eq!(
        config.storage.postgres_url,
        Some("postgresql://test@localhost/db".to_string())
    );

    std::env::remove_var("CARTRIDGE_STORAGE_MODEL_BACKEND");
    std::env::remove_var("CARTRIDGE_STORAGE_POSTGRES_URL");
}

#[test]
fn test_web_config() {
    let toml_content = r#"
[web]
host = "127.0.0.1"
port = 3000
"#;
    let config: CentralConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.web.host, "127.0.0.1");
    assert_eq!(config.web.port, 3000);
}

#[test]
fn test_mcts_config_from_toml() {
    let toml_content = r#"
[mcts]
num_simulations = 1600
c_puct = 2.0
temperature = 0.5
dirichlet_alpha = 0.5
dirichlet_weight = 0.3
eval_batch_size = 64
onnx_intra_threads = 4
"#;
    let config: CentralConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.mcts.num_simulations, 1600);
    assert!((config.mcts.c_puct - 2.0).abs() < f64::EPSILON);
    assert!((config.mcts.temperature - 0.5).abs() < f64::EPSILON);
    assert!((config.mcts.dirichlet_alpha - 0.5).abs() < f64::EPSILON);
    assert!((config.mcts.dirichlet_weight - 0.3).abs() < f64::EPSILON);
    assert_eq!(config.mcts.eval_batch_size, 64);
    assert_eq!(config.mcts.onnx_intra_threads, 4);
}

#[test]
fn test_config_clone() {
    let config = CentralConfig::default();
    let cloned = config.clone();
    assert_eq!(config.common.env_id, cloned.common.env_id);
    assert_eq!(config.actor.actor_id, cloned.actor.actor_id);
}

// ============================================================================
// Config file loading
//
// A config.toml that exists but cannot be parsed must be an error: silently
// substituting defaults reverts env_id (among everything else) to the built-in
// value, so self-play fills the replay buffer with a different game than the
// trainer expects -- visible only hours later, in the model. None of this path
// had any coverage before.
// ============================================================================

use std::io::Write;

/// Write `contents` to a `config.toml` inside a fresh temp dir.
fn config_file(contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("config.toml");
    let mut file = std::fs::File::create(&path).expect("create config");
    file.write_all(contents.as_bytes()).expect("write config");
    (dir, path)
}

#[test]
fn load_from_path_parses_a_valid_file() {
    let (_dir, path) = config_file(
        r#"
[common]
env_id = "connect4"
data_dir = "/tmp/cartridge"
"#,
    );

    let config = load_from_path(&path).expect("valid config should load");

    assert_eq!(config.common.env_id, "connect4");
    assert_eq!(config.common.data_dir, "/tmp/cartridge");
}

#[test]
fn load_from_path_fills_unspecified_sections_from_defaults() {
    // Partial files are legitimate: only the keys present should override.
    let (_dir, path) = config_file("[common]\nenv_id = \"othello\"\n");

    let config = load_from_path(&path).expect("partial config should load");

    assert_eq!(config.common.env_id, "othello");
    assert_eq!(config.web.port, CentralConfig::default().web.port);
}

#[test]
fn load_from_path_rejects_malformed_toml() {
    let (_dir, path) = config_file("[common\nenv_id = \"connect4\"\n");

    let err = load_from_path(&path).expect_err("malformed TOML must not load");

    assert!(matches!(err, ConfigError::Parse { .. }));
    assert_eq!(err.path(), path);
}

#[test]
fn load_from_path_rejects_a_schema_violation() {
    // Right TOML, wrong types. Previously this discarded the whole file.
    let (_dir, path) = config_file("[web]\nport = \"not-a-number\"\n");

    let err = load_from_path(&path).expect_err("bad value must not load");

    assert!(matches!(err, ConfigError::Parse { .. }));
}

#[test]
fn load_from_path_reports_an_unreadable_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("does-not-exist.toml");

    let err = load_from_path(&missing).expect_err("missing file must be an error here");

    assert!(matches!(err, ConfigError::Read { .. }));
}

#[test]
fn config_error_message_names_the_file_and_the_cause() {
    // An operator reading this line should not have to guess which file.
    let (_dir, path) = config_file("[common\n");

    let err = load_from_path(&path).unwrap_err();
    let message = err.to_string();

    assert!(message.contains(&path.display().to_string()), "{message}");
    assert!(message.contains("parse"), "{message}");
}

#[test]
fn load_config_still_falls_back_so_the_clap_lazy_cannot_panic() {
    // load_config() is infallible by design: the actor's Lazy<CentralConfig>
    // is forced from clap default_value_t and has nowhere to return an error.
    // try_load_config() runs first, in main(), to catch a broken file.
    let (_dir, path) = config_file("[common\n");

    assert!(load_from_path(&path).is_err());
    let _ = load_config(); // must not panic regardless of what is on disk
}
