//! Tests for the configuration module.

use super::*;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_default_config() {
    let config = CentralConfig::default();
    assert_eq!(config.common.env_id, "tictactoe");
    assert_eq!(config.common.data_dir, "./data");
    assert_eq!(config.common.log_level, "info");
    assert_eq!(config.algorithm.id, "alphazero_board_v1");
    assert_eq!(config.actor.actor_id, "actor-1");
    assert_eq!(config.web.host, "0.0.0.0");
    assert_eq!(config.web.port, 8080);
    assert!((config.mcts.late_temperature - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_training_defaults() {
    let config = CentralConfig::default();
    assert_eq!(config.training.iterations, 100);
    assert_eq!(config.training.episodes_per_iteration, 500);
    assert_eq!(config.training.steps_per_iteration, 1000);
    assert_eq!(config.training.batch_size, 64);
    assert!((config.training.learning_rate - 0.001).abs() < f64::EPSILON);
    assert!((config.training.weight_decay - 0.0001).abs() < f64::EPSILON);
    assert!((config.training.grad_clip_norm - 1.0).abs() < f64::EPSILON);
    assert_eq!(config.training.device, "cpu");
    assert_eq!(config.training.checkpoint_interval, 100);
}

#[test]
fn test_evaluation_defaults() {
    let config = CentralConfig::default();
    assert_eq!(config.evaluation.interval, 1);
    assert_eq!(config.evaluation.games, 50);
    assert_eq!(config.evaluation.simulations, 0);
    assert!((config.evaluation.temperature - 0.2).abs() < f32::EPSILON);
    assert_eq!(config.evaluation.solver_games, 0);
    assert_eq!(config.evaluation.evaluation_seed, 42);
}

#[test]
fn test_mcts_defaults() {
    let config = CentralConfig::default();
    assert!((config.mcts.c_puct - 1.4).abs() < f32::EPSILON);
    assert!((config.mcts.temperature - 1.0).abs() < f32::EPSILON);
    assert!((config.mcts.dirichlet_alpha - 0.3).abs() < f32::EPSILON);
    assert!((config.mcts.dirichlet_weight - 0.25).abs() < f32::EPSILON);
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
    std::env::set_var("CARTRIDGE_ALGORITHM_ID", "test_algorithm");
    std::env::set_var("CARTRIDGE_TRAINING_WEIGHT_DECAY", "0.5");
    std::env::set_var("CARTRIDGE_EVALUATION_EVALUATION_SEED", "17");
    std::env::set_var("CARTRIDGE_EVALUATION_TEMPERATURE", "0.35");

    let config = load_config().unwrap();
    assert_eq!(config.common.env_id, "connect4");
    assert_eq!(config.algorithm.id, "test_algorithm");
    assert!((config.training.weight_decay - 0.5).abs() < f64::EPSILON);
    assert_eq!(config.evaluation.evaluation_seed, 17);
    assert!((config.evaluation.temperature - 0.35).abs() < f32::EPSILON);

    std::env::remove_var("CARTRIDGE_COMMON_ENV_ID");
    std::env::remove_var("CARTRIDGE_ALGORITHM_ID");
    std::env::remove_var("CARTRIDGE_TRAINING_WEIGHT_DECAY");
    std::env::remove_var("CARTRIDGE_EVALUATION_EVALUATION_SEED");
    std::env::remove_var("CARTRIDGE_EVALUATION_TEMPERATURE");
}

#[test]
fn test_parse_config_toml() {
    let toml_content = r#"
[common]
env_id = "connect4"
data_dir = "/custom/data"

[algorithm]
id = "alphazero_board_v1"

[actor]
actor_id = "my-actor"

[training]
iterations = 50
batch_size = 128
"#;
    let config: CentralConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.common.env_id, "connect4");
    assert_eq!(config.common.data_dir, "/custom/data");
    assert_eq!(config.algorithm.id, "alphazero_board_v1");
    assert_eq!(config.actor.actor_id, "my-actor");
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
    assert_eq!(config.algorithm.id, "alphazero_board_v1");
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
    std::env::set_var("CARTRIDGE_STORAGE_S3_BUCKET", "test-models");
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
    std::env::remove_var("CARTRIDGE_STORAGE_S3_BUCKET");
    std::env::remove_var("CARTRIDGE_STORAGE_POSTGRES_URL");
}

#[test]
fn test_unknown_config_key_is_rejected() {
    let error = toml::from_str::<CentralConfig>(
        r#"
[algorithm]
idd = "typo"
"#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("unknown field"));
    assert!(error.contains("idd"));
}

#[test]
fn test_removed_actor_service_keys_are_rejected() {
    for field in ["max_episodes", "flush_interval_secs", "health_port"] {
        let source = format!("[actor]\n{field} = 1\n");
        let error = toml::from_str::<CentralConfig>(&source)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field"), "{error}");
        assert!(error.contains(field), "{error}");
    }
}

#[test]
fn test_removed_mcts_num_simulations_config_is_rejected() {
    let error = toml::from_str::<CentralConfig>("[mcts]\nnum_simulations = 800\n")
        .unwrap_err()
        .to_string();
    assert!(error.contains("unknown field"), "{error}");
    assert!(error.contains("num_simulations"), "{error}");
}

#[test]
fn test_unknown_known_section_environment_keys_are_rejected() {
    let _guard = ENV_LOCK.lock().unwrap();
    for key in [
        "CARTRIDGE_MCTS_NUM_SIMULATIONS",
        "CARTRIDGE_ACTOR_MAX_EPISODES",
        "CARTRIDGE_ACTOR_HEALTH_PORT",
        "CARTRIDGE_EVALUATION_SEED",
    ] {
        std::env::set_var(key, "1");
        let error = apply_env_overrides(CentralConfig::default())
            .unwrap_err()
            .to_string();
        std::env::remove_var(key);
        assert!(error.contains(key), "{error}");
    }
}

#[test]
fn test_operational_cartridge_environment_keys_are_allowed() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_TRACE_ID", "trace-1");
    std::env::set_var("CARTRIDGE_EVAL_BINARY", "/tmp/evaluator");
    let result = apply_env_overrides(CentralConfig::default());
    std::env::remove_var("CARTRIDGE_TRACE_ID");
    std::env::remove_var("CARTRIDGE_EVAL_BINARY");
    assert!(result.is_ok());
}

#[test]
fn test_malformed_explicit_file_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, "[algorithm\nid = broken").unwrap();

    let error = load_from_path(&path).unwrap_err().to_string();
    assert!(error.contains("failed to parse configuration"));
    assert!(error.contains("config.toml"));
}

#[test]
fn test_invalid_typed_environment_override_is_rejected() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CARTRIDGE_WEB_PORT", "not-a-port");
    let error = apply_env_overrides(CentralConfig::default())
        .unwrap_err()
        .to_string();
    std::env::remove_var("CARTRIDGE_WEB_PORT");

    assert!(error.contains("CARTRIDGE_WEB_PORT"));
    assert!(error.contains("not-a-port"));
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
c_puct = 2.0
temperature = 0.5
late_temperature = 0.05
dirichlet_alpha = 0.5
dirichlet_weight = 0.3
eval_batch_size = 64
onnx_intra_threads = 4
"#;
    let config: CentralConfig = toml::from_str(toml_content).unwrap();
    assert!((config.mcts.c_puct - 2.0).abs() < f32::EPSILON);
    assert!((config.mcts.late_temperature - 0.05).abs() < f32::EPSILON);
    assert!((config.mcts.temperature - 0.5).abs() < f32::EPSILON);
    assert!((config.mcts.dirichlet_alpha - 0.5).abs() < f32::EPSILON);
    assert!((config.mcts.dirichlet_weight - 0.3).abs() < f32::EPSILON);
    assert_eq!(config.mcts.eval_batch_size, 64);
    assert_eq!(config.mcts.onnx_intra_threads, 4);
}

#[test]
fn shared_count_widths_match_the_authenticated_wire_contract() {
    let config: CentralConfig = toml::from_str(
        r#"
[training]
iterations = 4294967296
episodes_per_iteration = 2147483648
steps_per_iteration = 4294967296
batch_size = 4294967296
checkpoint_interval = 4294967296
num_actors = 1

[evaluation]
interval = 4294967296
games = 2147483648
solver_games = 0

[mcts]
eval_batch_size = 2147483648
onnx_intra_threads = 2147483648
"#,
    )
    .unwrap();
    assert_eq!(config.training.iterations, 1_u64 << 32);
    assert_eq!(config.training.episodes_per_iteration, 1_u32 << 31);
    assert_eq!(config.evaluation.games, 1_u32 << 31);
    assert_eq!(config.mcts.eval_batch_size, 1_u32 << 31);

    for source in [
        "[training]\nepisodes_per_iteration = 4294967296\n",
        "[training]\nnum_actors = 4294967296\n",
        "[evaluation]\ngames = 4294967296\n",
        "[evaluation]\nsolver_games = 4294967296\n",
        "[mcts]\neval_batch_size = 4294967296\n",
        "[mcts]\nonnx_intra_threads = 4294967296\n",
    ] {
        assert!(toml::from_str::<CentralConfig>(source).is_err(), "{source}");
    }
}

#[test]
fn test_invalid_search_numbers_are_rejected() {
    let _guard = ENV_LOCK.lock().unwrap();
    for mutate in [
        |config: &mut CentralConfig| config.evaluation.temperature = f32::NAN,
        |config: &mut CentralConfig| config.mcts.c_puct = f32::INFINITY,
        |config: &mut CentralConfig| config.mcts.dirichlet_weight = 1.01,
        |config: &mut CentralConfig| config.mcts.dirichlet_alpha = 0.0,
        |config: &mut CentralConfig| config.mcts.temp_threshold = 1,
        |config: &mut CentralConfig| config.mcts.start_sims = 0,
        |config: &mut CentralConfig| config.mcts.eval_batch_size = 0,
    ] {
        let mut config = CentralConfig::default();
        mutate(&mut config);
        assert!(apply_env_overrides(config).is_err());
    }
}

#[test]
fn test_noncanonical_mcts_schedules_are_rejected() {
    let _guard = ENV_LOCK.lock().unwrap();
    for mutate in [
        |config: &mut CentralConfig| {
            config.mcts.max_sims = config.mcts.start_sims;
        },
        |config: &mut CentralConfig| {
            config.mcts.max_sims = config.mcts.start_sims + 10;
            config.mcts.sim_ramp_rate = 11;
        },
        |config: &mut CentralConfig| {
            config.training.iterations = 2;
            config.mcts.max_sims = config.mcts.start_sims + 10;
            config.mcts.sim_ramp_rate = 5;
        },
    ] {
        let mut config = CentralConfig::default();
        mutate(&mut config);
        assert!(apply_env_overrides(config).is_err());
    }
}

#[test]
fn test_ineffective_solver_settings_are_rejected() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut wrong_environment = CentralConfig::default();
    wrong_environment.evaluation.solver_games = 1;
    let error = apply_env_overrides(wrong_environment)
        .unwrap_err()
        .to_string();
    assert!(error.contains("solver_games"), "{error}");

    let mut missing_games = CentralConfig::default();
    missing_games.common.env_id = "connect4".to_string();
    missing_games.evaluation.promotion_metric = "solver_optimal".to_string();
    let error = apply_env_overrides(missing_games).unwrap_err().to_string();
    assert!(error.contains("solver_optimal"), "{error}");
}

#[test]
fn test_inactive_promotion_parameters_are_rejected() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut win_rate = CentralConfig::default();
    win_rate.evaluation.promotion_margin = 0.1;
    let error = apply_env_overrides(win_rate).unwrap_err().to_string();
    assert!(error.contains("promotion_margin"), "{error}");

    let mut solver = CentralConfig::default();
    solver.common.env_id = "connect4".to_string();
    solver.evaluation.promotion_metric = "solver_optimal".to_string();
    solver.evaluation.solver_games = 1;
    let error = apply_env_overrides(solver).unwrap_err().to_string();
    assert!(error.contains("win_threshold"), "{error}");
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
    let _guard = ENV_LOCK.lock().unwrap();
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
    let _guard = ENV_LOCK.lock().unwrap();
    // Partial files are legitimate: only the keys present should override.
    let (_dir, path) = config_file("[common]\nenv_id = \"othello\"\n");

    let config = load_from_path(&path).expect("partial config should load");

    assert_eq!(config.common.env_id, "othello");
    assert_eq!(config.web.port, CentralConfig::default().web.port);
}

#[test]
fn load_from_path_rejects_malformed_toml() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (_dir, path) = config_file("[common\nenv_id = \"connect4\"\n");

    let err = load_from_path(&path).expect_err("malformed TOML must not load");

    assert!(matches!(err, ConfigError::Parse { .. }));
    assert_eq!(err.path(), Some(path.as_path()));
}

#[test]
fn load_from_path_rejects_a_schema_violation() {
    let _guard = ENV_LOCK.lock().unwrap();
    // Right TOML, wrong types. Previously this discarded the whole file.
    let (_dir, path) = config_file("[web]\nport = \"not-a-number\"\n");

    let err = load_from_path(&path).expect_err("bad value must not load");

    assert!(matches!(err, ConfigError::Parse { .. }));
}

#[test]
fn load_from_path_reports_an_unreadable_file() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("does-not-exist.toml");

    let err = load_from_path(&missing).expect_err("missing file must be an error here");

    assert!(matches!(err, ConfigError::Read { .. }));
}

#[test]
fn config_error_message_names_the_file_and_the_cause() {
    let _guard = ENV_LOCK.lock().unwrap();
    // An operator reading this line should not have to guess which file.
    let (_dir, path) = config_file("[common\n");

    let err = load_from_path(&path).unwrap_err();
    let message = err.to_string();

    assert!(message.contains(&path.display().to_string()), "{message}");
    assert!(message.contains("parse"), "{message}");
}

#[test]
fn load_config_rejects_a_malformed_explicit_file() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (_dir, path) = config_file("[common\n");
    std::env::set_var("CARTRIDGE_CONFIG", &path);

    let error = load_config().unwrap_err();
    std::env::remove_var("CARTRIDGE_CONFIG");

    assert!(matches!(error, ConfigError::Parse { .. }));
    assert_eq!(error.path(), Some(path.as_path()));
}
