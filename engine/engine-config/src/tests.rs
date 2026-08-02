//! Tests for the configuration module.

use super::*;
use std::io::Write;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

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
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_COMMON_ENV_ID", "connect4");
    std::env::set_var("CARTRIDGE_ACTOR_MAX_EPISODES", "7");
    std::env::set_var("CARTRIDGE_TRAINING_WEIGHT_DECAY", "0.5");

    let config = load_config().unwrap();
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
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_STORAGE_MODEL_BACKEND", "s3");
    std::env::set_var(
        "CARTRIDGE_STORAGE_POSTGRES_URL",
        "postgresql://test@localhost/db",
    );

    let config = load_config().unwrap();
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

#[test]
fn test_explicit_missing_config_is_an_error() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let missing = temp_dir.path().join("missing.toml");
    std::env::set_var("CARTRIDGE_CONFIG", &missing);

    let error = load_config().unwrap_err();

    std::env::remove_var("CARTRIDGE_CONFIG");
    assert!(matches!(error, ConfigError::MissingExplicitPath { .. }));
    assert!(error.to_string().contains("missing.toml"));
}

#[test]
fn test_empty_explicit_config_placeholder_uses_normal_search() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_CONFIG", "");

    let config = load_config().unwrap();

    std::env::remove_var("CARTRIDGE_CONFIG");
    assert!(!config.common.env_id.is_empty());
}

#[test]
fn test_malformed_config_file_is_an_error() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "[training\niterations = 10").unwrap();

    let error = load_from_path(file.path()).unwrap_err();

    assert!(matches!(error, ConfigError::Parse { .. }));
}

#[test]
fn test_missing_config_file_is_a_read_error() {
    let temp_dir = tempfile::tempdir().unwrap();
    let error = load_from_path(&temp_dir.path().join("missing.toml")).unwrap_err();

    assert!(matches!(error, ConfigError::Read { .. }));
}

#[test]
fn test_invalid_non_empty_numeric_environment_override_is_an_error() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_MCTS_EVAL_BATCH_SIZE", "many");

    let error = apply_env_overrides(CentralConfig::default()).unwrap_err();

    std::env::remove_var("CARTRIDGE_MCTS_EVAL_BATCH_SIZE");
    assert!(matches!(error, ConfigError::InvalidEnvironment { .. }));
    assert!(error.to_string().contains("CARTRIDGE_MCTS_EVAL_BATCH_SIZE"));
}

#[test]
fn test_empty_environment_placeholder_uses_existing_value() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_MCTS_EVAL_BATCH_SIZE", "");

    let config = apply_env_overrides(CentralConfig::default()).unwrap();

    std::env::remove_var("CARTRIDGE_MCTS_EVAL_BATCH_SIZE");
    assert_eq!(config.mcts.eval_batch_size, 32);
}

#[test]
fn test_all_declared_rust_fields_have_working_environment_overrides() {
    let _guard = ENV_LOCK.lock().unwrap();
    let overrides = [
        ("CARTRIDGE_TRAINING_NUM_ACTORS", "3"),
        ("CARTRIDGE_EVALUATION_WIN_THRESHOLD", "0.75"),
        ("CARTRIDGE_EVALUATION_EVAL_VS_RANDOM", "no"),
        ("CARTRIDGE_ACTOR_HEALTH_PORT", "9091"),
        (
            "CARTRIDGE_WEB_ALLOWED_ORIGINS",
            r#"["https://example.com"]"#,
        ),
        ("CARTRIDGE_MCTS_TEMP_THRESHOLD", "12"),
        ("CARTRIDGE_MCTS_START_SIMS", "60"),
        ("CARTRIDGE_MCTS_MAX_SIMS", "500"),
        ("CARTRIDGE_MCTS_SIM_RAMP_RATE", "25"),
        ("CARTRIDGE_LOGGING_FORMAT", "json"),
        ("CARTRIDGE_LOGGING_INCLUDE_TIMESTAMPS", "0"),
        ("CARTRIDGE_LOGGING_INCLUDE_TARGET", "off"),
    ];
    for (key, value) in overrides {
        std::env::set_var(key, value);
    }

    let config = apply_env_overrides(CentralConfig::default()).unwrap();

    for (key, _) in overrides {
        std::env::remove_var(key);
    }
    assert_eq!(config.training.num_actors, 3);
    assert!((config.evaluation.win_threshold - 0.75).abs() < f64::EPSILON);
    assert!(!config.evaluation.eval_vs_random);
    assert_eq!(config.actor.health_port, 9091);
    assert_eq!(
        config.web.allowed_origins,
        vec!["https://example.com".to_string()]
    );
    assert_eq!(config.mcts.temp_threshold, 12);
    assert_eq!(config.mcts.start_sims, 60);
    assert_eq!(config.mcts.max_sims, 500);
    assert_eq!(config.mcts.sim_ramp_rate, 25);
    assert_eq!(config.logging.format, "json");
    assert!(!config.logging.include_timestamps);
    assert!(!config.logging.include_target);
}

#[test]
fn test_invalid_boolean_environment_override_is_an_error() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_LOGGING_INCLUDE_TARGET", "sometimes");

    let error = apply_env_overrides(CentralConfig::default()).unwrap_err();

    std::env::remove_var("CARTRIDGE_LOGGING_INCLUDE_TARGET");
    assert!(matches!(error, ConfigError::InvalidEnvironment { .. }));
    assert!(error
        .to_string()
        .contains("CARTRIDGE_LOGGING_INCLUDE_TARGET"));
}

#[test]
fn test_invalid_environment_string_value_is_validated() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_LOGGING_FORMAT", "xml");

    let error = apply_env_overrides(CentralConfig::default()).unwrap_err();

    std::env::remove_var("CARTRIDGE_LOGGING_FORMAT");
    assert!(matches!(error, ConfigError::InvalidValue { .. }));
    assert!(error.to_string().contains("logging.format"));
}

#[test]
fn test_documented_zero_sentinels_remain_valid() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut config = CentralConfig::default();
    config.training.grad_clip_norm = 0.0;
    config.evaluation.interval = 0;
    config.actor.log_interval = 0;
    config.mcts.dirichlet_alpha = 0.0;
    config.mcts.onnx_intra_threads = 0;

    assert!(apply_env_overrides(config).is_ok());
}

#[test]
fn test_invalid_ranges_are_rejected_after_overrides() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut config = CentralConfig::default();
    config.storage.pool_max_size = 0;

    let error = apply_env_overrides(config).unwrap_err();

    assert!(matches!(error, ConfigError::InvalidValue { .. }));
    assert!(error.to_string().contains("storage.pool_max_size"));
}
