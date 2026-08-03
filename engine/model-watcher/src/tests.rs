use super::*;
use crate::artifact::{
    manifest_path, onnx_blob_path, read_filesystem_head, resolve_filesystem_head, run_commit_path,
    sha256_hex, ArtifactProfile, BlobReference, CheckpointManifestV1, RunHeadV2,
};
use algorithm_core::{resolve_algorithm, ModelArtifactContract, ALPHAZERO_BOARD_V1_ID};
use serde::Serialize;
use tempfile::tempdir;

pub(crate) const CONTRACT_TEST_MODEL: &[u8] = &[
    8, 8, 58, 202, 1, 10, 38, 10, 11, 111, 98, 115, 101, 114, 118, 97, 116, 105, 111, 110, 18, 13,
    112, 111, 108, 105, 99, 121, 95, 108, 111, 103, 105, 116, 115, 34, 8, 73, 100, 101, 110, 116,
    105, 116, 121, 10, 62, 10, 11, 111, 98, 115, 101, 114, 118, 97, 116, 105, 111, 110, 18, 5, 118,
    97, 108, 117, 101, 34, 10, 82, 101, 100, 117, 99, 101, 77, 101, 97, 110, 42, 11, 10, 4, 97,
    120, 101, 115, 64, 1, 160, 1, 7, 42, 15, 10, 8, 107, 101, 101, 112, 100, 105, 109, 115, 24, 1,
    160, 1, 2, 18, 13, 99, 111, 110, 116, 114, 97, 99, 116, 95, 116, 101, 115, 116, 90, 27, 10, 11,
    111, 98, 115, 101, 114, 118, 97, 116, 105, 111, 110, 18, 12, 10, 10, 8, 1, 18, 6, 10, 0, 10, 2,
    8, 3, 98, 29, 10, 13, 112, 111, 108, 105, 99, 121, 95, 108, 111, 103, 105, 116, 115, 18, 12,
    10, 10, 8, 1, 18, 6, 10, 0, 10, 2, 8, 3, 98, 21, 10, 5, 118, 97, 108, 117, 101, 18, 12, 10, 10,
    8, 1, 18, 6, 10, 0, 10, 2, 8, 1, 66, 4, 10, 0, 16, 13, 114, 29, 10, 24, 99, 97, 114, 116, 114,
    105, 100, 103, 101, 46, 115, 99, 104, 101, 109, 97, 95, 118, 101, 114, 115, 105, 111, 110, 18,
    1, 49, 114, 44, 10, 22, 99, 97, 114, 116, 114, 105, 100, 103, 101, 46, 97, 108, 103, 111, 114,
    105, 116, 104, 109, 95, 105, 100, 18, 18, 97, 108, 112, 104, 97, 122, 101, 114, 111, 95, 98,
    111, 97, 114, 100, 95, 118, 49, 114, 48, 10, 24, 99, 97, 114, 116, 114, 105, 100, 103, 101, 46,
    109, 111, 100, 101, 108, 95, 99, 111, 110, 116, 114, 97, 99, 116, 18, 20, 111, 110, 110, 120,
    95, 112, 111, 108, 105, 99, 121, 95, 118, 97, 108, 117, 101, 95, 118, 49, 114, 33, 10, 16, 99,
    97, 114, 116, 114, 105, 100, 103, 101, 46, 101, 110, 118, 95, 105, 100, 18, 13, 99, 111, 110,
    116, 114, 97, 99, 116, 95, 116, 101, 115, 116, 114, 35, 10, 30, 99, 97, 114, 116, 114, 105,
    100, 103, 101, 46, 101, 110, 118, 95, 99, 111, 110, 116, 114, 97, 99, 116, 95, 118, 101, 114,
    115, 105, 111, 110, 18, 1, 49,
];

pub(crate) fn identity(env_id: &str) -> ModelArtifactContract {
    resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor()
        .model_artifact_contract(env_id, 1)
}

fn model_spec(identity: ModelArtifactContract) -> ModelLoadSpec {
    ModelLoadSpec::new(3, 3, 1, 9, identity).unwrap()
}

#[test]
fn model_load_spec_rejects_zero_dimensions_threads_and_horizon() {
    let identity = identity("contract_test");

    assert!(ModelLoadSpec::new(0, 3, 1, 9, identity.clone()).is_err());
    assert!(ModelLoadSpec::new(3, 0, 1, 9, identity.clone()).is_err());
    assert!(ModelLoadSpec::new(3, 3, 0, 9, identity.clone()).is_err());
    assert!(ModelLoadSpec::new(3, 3, 1, 0, identity).is_err());
}

fn canonical<T: Serialize>(value: &T) -> Vec<u8> {
    let value = serde_json::to_value(value).unwrap();
    serde_json::to_vec(&value).unwrap()
}

fn config_sha256() -> String {
    sha256_hex(&canonical(&serde_json::json!({"batch_size": 1})))
}

fn synchronized_recipe() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "learner_config_sha256": config_sha256(),
        "learner_recipe": {"batch_size": 1},
        "total_iterations": 10,
        "episodes_per_iteration": 3,
        "training_steps_per_iteration": 1,
        "num_actors": 1,
        "collector_episode_timeout_seconds": 60,
        "collector_eval_batch_size": 8,
        "collector_onnx_intra_threads": 1,
        "mcts_start_simulations": 5,
        "mcts_max_simulations": 10,
        "mcts_simulation_ramp": 2,
        "collector_c_puct": f64::from(1.4_f32),
        "collector_temperature": f64::from(1.0_f32),
        "collector_late_temperature": f64::from(1.0_f32),
        "temperature_move_threshold": 0,
        "collector_dirichlet_alpha": f64::from(0.3_f32),
        "collector_dirichlet_weight": f64::from(0.25_f32),
        "collector_seed_strategy": "system_entropy_v1",
        "evaluation_interval": 2,
        "evaluation_games": 2,
        "evaluation_simulations": 0,
        "evaluation_temperature": f64::from(0.2_f32),
        "evaluation_win_threshold": 0.55,
        "evaluation_vs_random": true,
        "solver_games": 0,
        "evaluation_seed": 42,
        "replay_policy": "scoped_fresh_iteration_v1",
        "promotion_metric": "win_rate",
        "promotion_margin": 0.0,
    })
}

fn orchestration(iteration: u64, source_checkpoint_id: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "iteration": iteration,
        "collection_scope_id": format!("{iteration:064x}"),
        "source_checkpoint_id": source_checkpoint_id,
        "episodes_generated": 3,
        "transitions_generated": 3,
        "training_steps": 1,
        "collector_simulations": 5 + u32::try_from(iteration - 1).unwrap() * 2,
        "collector_seed": null,
        "evaluation_seed": null,
        "actor_time_seconds": 0.0,
        "trainer_time_seconds": 0.0,
        "eval_time_seconds": 0.0,
        "total_time_seconds": 0.0,
        "eval_win_rate": null,
        "eval_draw_rate": null,
        "timestamp": format!("2026-01-01T00:00:{:02}.000000Z", iteration),
        "evaluation_id": null,
    })
}

fn publish_test_run_commit(
    root: &Path,
    identity: &ModelArtifactContract,
    checkpoint_id: &str,
    step: u64,
    parent_run_commit_id: Option<String>,
) -> String {
    let profile = ArtifactProfile::from(identity);
    let stats_snapshot = serde_json::json!({
        "schema_version": 2,
        "profile": profile,
        "config_sha256": config_sha256(),
        "checkpoint_id": checkpoint_id,
        "step": step,
        "stats": {
            "step": step,
            "total_steps": step,
            "total_loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "learning_rate": 0.0,
            "samples_seen": 0,
            "replay_record_count": 0,
            "last_checkpoint": checkpoint_id,
            "timestamp": 0.0,
            "history": [],
            "env_id": identity.env_id,
            "last_eval": null,
            "eval_history": [],
        },
    });
    let stats_bytes = canonical(&stats_snapshot);
    let commit = serde_json::json!({
        "schema_version": 1,
        "profile": ArtifactProfile::from(identity),
        "config_sha256": config_sha256(),
        "parent_run_commit_id": parent_run_commit_id,
        "checkpoint_id": checkpoint_id,
        "run_recipe_id": null,
        "run_recipe": null,
        "stats_id": sha256_hex(&stats_bytes),
        "stats_snapshot": stats_snapshot,
        "champion": null,
        "evaluation_head_id": null,
        "orchestration": null,
    });
    let commit_bytes = canonical(&commit);
    let run_commit_id = sha256_hex(&commit_bytes);
    let path = run_commit_path(root, &run_commit_id);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, commit_bytes).unwrap();
    run_commit_id
}

fn select_test_head(root: &Path, checkpoint_id: &str, run_commit_id: &str) {
    let head = RunHeadV2 {
        schema_version: 2,
        checkpoint_id: checkpoint_id.to_string(),
        run_commit_id: run_commit_id.to_string(),
    };
    let path = crate::artifact::run_head_path(root);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, canonical(&head)).unwrap();
}

struct EvaluationSelection<'a> {
    evaluation_id: &'a str,
    champion_checkpoint_id: &'a str,
    champion_evaluation_id: &'a str,
}

fn replace_with_evaluated_synchronized_commit(
    root: &Path,
    source_run_commit_id: &str,
    recipe: &serde_json::Value,
    iteration: u64,
    source_checkpoint_id: Option<&str>,
    selection: EvaluationSelection<'_>,
) -> String {
    let original = std::fs::read(run_commit_path(root, source_run_commit_id)).unwrap();
    let mut commit: serde_json::Value = serde_json::from_slice(&original).unwrap();
    commit["run_recipe_id"] = serde_json::Value::String(sha256_hex(&canonical(recipe)));
    commit["run_recipe"] = recipe.clone();
    let mut facts = orchestration(iteration, source_checkpoint_id);
    facts["evaluation_seed"] = serde_json::json!(42);
    facts["evaluation_id"] = serde_json::Value::String(selection.evaluation_id.to_string());
    commit["orchestration"] = facts;
    commit["evaluation_head_id"] = serde_json::Value::String(selection.evaluation_id.to_string());
    commit["champion"] = serde_json::json!({
        "checkpoint_id": selection.champion_checkpoint_id,
        "evaluation_id": selection.champion_evaluation_id,
    });
    let bytes = canonical(&commit);
    let run_commit_id = sha256_hex(&bytes);
    std::fs::write(run_commit_path(root, &run_commit_id), bytes).unwrap();
    run_commit_id
}

fn replace_with_synchronized_commit(
    root: &Path,
    source_run_commit_id: &str,
    recipe: &serde_json::Value,
    iteration: u64,
    source_checkpoint_id: Option<&str>,
    collection_scope_id: Option<&str>,
) -> String {
    let original = std::fs::read(run_commit_path(root, source_run_commit_id)).unwrap();
    let mut commit: serde_json::Value = serde_json::from_slice(&original).unwrap();
    commit["run_recipe_id"] = serde_json::Value::String(sha256_hex(&canonical(recipe)));
    commit["run_recipe"] = recipe.clone();
    let mut facts = orchestration(iteration, source_checkpoint_id);
    if let Some(collection_scope_id) = collection_scope_id {
        facts["collection_scope_id"] = serde_json::Value::String(collection_scope_id.to_string());
    }
    commit["orchestration"] = facts;
    let bytes = canonical(&commit);
    let run_commit_id = sha256_hex(&bytes);
    std::fs::write(run_commit_path(root, &run_commit_id), bytes).unwrap();
    run_commit_id
}

fn publish_test_checkpoint(
    root: &Path,
    model: &[u8],
    identity: &ModelArtifactContract,
    step: u64,
) -> String {
    publish_test_checkpoint_with_parent(root, model, identity, step, None, None).0
}

fn publish_test_checkpoint_with_parent(
    root: &Path,
    model: &[u8],
    identity: &ModelArtifactContract,
    step: u64,
    parent_checkpoint_id: Option<String>,
    parent_run_commit_id: Option<String>,
) -> (String, String) {
    let model_digest = sha256_hex(model);
    let model_path = onnx_blob_path(root, &model_digest);
    std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
    std::fs::write(&model_path, model).unwrap();

    let manifest = CheckpointManifestV1 {
        schema_version: 1,
        profile: ArtifactProfile::from(identity),
        step,
        parent_checkpoint_id,
        config_sha256: config_sha256(),
        onnx: BlobReference {
            sha256: model_digest,
            size_bytes: model.len() as u64,
        },
        learner_state: BlobReference {
            sha256: "b".repeat(64),
            size_bytes: 1,
        },
    };
    let manifest_bytes = canonical(&manifest);
    let checkpoint_id = sha256_hex(&manifest_bytes);
    let path = manifest_path(root, &checkpoint_id);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, manifest_bytes).unwrap();

    let run_commit_id =
        publish_test_run_commit(root, identity, &checkpoint_id, step, parent_run_commit_id);
    select_test_head(root, &checkpoint_id, &run_commit_id);
    (checkpoint_id, run_commit_id)
}

#[test]
fn watcher_starts_empty_without_a_channel() {
    let root = tempdir().unwrap();
    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(identity("contract_test")),
        ModelSelection::Latest,
        Arc::clone(&evaluator),
    );

    assert_eq!(
        watcher.run_head_path(),
        root.path().join("channels/current.json")
    );
    assert!(!watcher.try_load_existing().unwrap());
    assert!(evaluator.read().unwrap().is_none());
}

#[test]
fn watcher_loads_manifest_identity_and_immutable_blob() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 42);
    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::clone(&evaluator),
    );

    assert!(watcher.try_load_existing().unwrap());
    assert!(evaluator.read().unwrap().is_some());
    let info = watcher.model_info();
    let info = info.read().unwrap();
    assert_eq!(info.checkpoint_id.as_deref(), Some(checkpoint_id.as_str()));
    assert_eq!(info.training_step, Some(42));
    assert_eq!(
        info.model_sha256.as_deref(),
        Some(sha256_hex(CONTRACT_TEST_MODEL).as_str())
    );
}

#[test]
fn champion_selection_falls_back_to_latest_before_a_champion_exists() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::ChampionOrLatest,
        Arc::new(RwLock::new(None)),
    );

    assert!(watcher.try_load_existing().unwrap());
    assert_eq!(
        watcher
            .model_info()
            .read()
            .unwrap()
            .checkpoint_id
            .as_deref(),
        Some(checkpoint_id.as_str())
    );
}

#[test]
fn champion_selection_tracks_rejection_and_promotion_without_losing_head_generation() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let root_checkpoint = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let root_standalone = read_filesystem_head(root.path()).unwrap().unwrap();
    let mut recipe = synchronized_recipe();
    recipe["evaluation_interval"] = serde_json::json!(1);
    let first_evaluation = "a".repeat(64);
    let root_commit = replace_with_evaluated_synchronized_commit(
        root.path(),
        &root_standalone.run_commit_id,
        &recipe,
        1,
        None,
        EvaluationSelection {
            evaluation_id: &first_evaluation,
            champion_checkpoint_id: &root_checkpoint,
            champion_evaluation_id: &first_evaluation,
        },
    );
    select_test_head(root.path(), &root_checkpoint, &root_commit);

    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected.clone()),
        ModelSelection::ChampionOrLatest,
        Arc::clone(&evaluator),
    );
    assert!(watcher.try_load_existing().unwrap());
    let champion_evaluator = evaluator
        .read()
        .unwrap()
        .as_ref()
        .map(|value| value as *const OnnxEvaluator as usize)
        .unwrap();

    let (rejected_checkpoint, rejected_standalone) = publish_test_checkpoint_with_parent(
        root.path(),
        CONTRACT_TEST_MODEL,
        &expected,
        2,
        Some(root_checkpoint.clone()),
        Some(root_commit),
    );
    let rejected_evaluation = "b".repeat(64);
    let rejected_commit = replace_with_evaluated_synchronized_commit(
        root.path(),
        &rejected_standalone,
        &recipe,
        2,
        Some(&root_checkpoint),
        EvaluationSelection {
            evaluation_id: &rejected_evaluation,
            champion_checkpoint_id: &root_checkpoint,
            champion_evaluation_id: &first_evaluation,
        },
    );
    select_test_head(root.path(), &rejected_checkpoint, &rejected_commit);

    assert!(watcher.try_load_existing().unwrap());
    assert_eq!(
        evaluator
            .read()
            .unwrap()
            .as_ref()
            .map(|value| value as *const OnnxEvaluator as usize),
        Some(champion_evaluator)
    );
    assert_eq!(
        watcher.accepted_head.read().unwrap().as_ref(),
        Some(&AcceptedHead {
            model_checkpoint_id: root_checkpoint.clone(),
            run_commit_id: rejected_commit.clone(),
        })
    );
    assert_eq!(
        watcher
            .model_info()
            .read()
            .unwrap()
            .checkpoint_id
            .as_deref(),
        Some(root_checkpoint.as_str())
    );

    let (promoted_checkpoint, promoted_standalone) = publish_test_checkpoint_with_parent(
        root.path(),
        CONTRACT_TEST_MODEL,
        &expected,
        3,
        Some(rejected_checkpoint.clone()),
        Some(rejected_commit),
    );
    let promoted_evaluation = "c".repeat(64);
    let promoted_commit = replace_with_evaluated_synchronized_commit(
        root.path(),
        &promoted_standalone,
        &recipe,
        3,
        Some(&rejected_checkpoint),
        EvaluationSelection {
            evaluation_id: &promoted_evaluation,
            champion_checkpoint_id: &promoted_checkpoint,
            champion_evaluation_id: &promoted_evaluation,
        },
    );
    select_test_head(root.path(), &promoted_checkpoint, &promoted_commit);

    assert!(watcher.try_load_existing().unwrap());
    assert_eq!(
        watcher.accepted_head.read().unwrap().as_ref(),
        Some(&AcceptedHead {
            model_checkpoint_id: promoted_checkpoint.clone(),
            run_commit_id: promoted_commit,
        })
    );
    assert_eq!(
        watcher
            .model_info()
            .read()
            .unwrap()
            .checkpoint_id
            .as_deref(),
        Some(promoted_checkpoint.as_str())
    );
}

#[test]
fn watcher_accepts_exact_recipe_bound_orchestration() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let original_head = read_filesystem_head(root.path()).unwrap().unwrap();
    let original =
        std::fs::read(run_commit_path(root.path(), &original_head.run_commit_id)).unwrap();
    let mut commit: serde_json::Value = serde_json::from_slice(&original).unwrap();
    let recipe = synchronized_recipe();
    commit["run_recipe_id"] = serde_json::Value::String(sha256_hex(&canonical(&recipe)));
    commit["run_recipe"] = recipe;
    commit["orchestration"] = orchestration(1, None);
    let bytes = canonical(&commit);
    let run_commit_id = sha256_hex(&bytes);
    std::fs::write(run_commit_path(root.path(), &run_commit_id), bytes).unwrap();
    select_test_head(root.path(), &checkpoint_id, &run_commit_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    assert!(watcher.try_load_existing().unwrap());
}

fn rejected_root_recipe_error(recipe: &serde_json::Value) -> String {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let standalone = read_filesystem_head(root.path()).unwrap().unwrap();
    let run_commit_id = replace_with_synchronized_commit(
        root.path(),
        &standalone.run_commit_id,
        recipe,
        1,
        None,
        None,
    );
    select_test_head(root.path(), &checkpoint_id, &run_commit_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );
    format!("{:#}", watcher.try_load_existing().unwrap_err())
}

#[test]
fn watcher_rejects_noncanonical_or_half_disabled_collector_search_values() {
    let mut noncanonical = synchronized_recipe();
    noncanonical["collector_temperature"] = serde_json::json!(0.2_f64);
    let error = rejected_root_recipe_error(&noncanonical);
    assert!(
        error.contains("exact canonical value of a finite f32"),
        "{error}"
    );

    let mut inactive_alpha_only = synchronized_recipe();
    inactive_alpha_only["collector_dirichlet_alpha"] = serde_json::json!(0.0);
    let error = rejected_root_recipe_error(&inactive_alpha_only);
    assert!(error.contains("must both be zero"), "{error}");

    let mut inactive_late_temperature = synchronized_recipe();
    inactive_late_temperature["collector_late_temperature"] = serde_json::json!(f64::from(0.1_f32));
    let error = rejected_root_recipe_error(&inactive_late_temperature);
    assert!(error.contains("must equal base temperature"), "{error}");

    let mut ineffective_enabled_temperature = synchronized_recipe();
    ineffective_enabled_temperature["temperature_move_threshold"] = serde_json::json!(1);
    let error = rejected_root_recipe_error(&ineffective_enabled_temperature);
    assert!(error.contains("must differ from base"), "{error}");

    let mut unreachable_temperature = synchronized_recipe();
    unreachable_temperature["temperature_move_threshold"] = serde_json::json!(9);
    unreachable_temperature["collector_late_temperature"] = serde_json::json!(f64::from(0.1_f32));
    let error = rejected_root_recipe_error(&unreachable_temperature);
    assert!(error.contains("unreachable for its environment"), "{error}");

    let mut inactive_promotion_margin = synchronized_recipe();
    inactive_promotion_margin["promotion_margin"] = serde_json::json!(0.1);
    let error = rejected_root_recipe_error(&inactive_promotion_margin);
    assert!(error.contains("canonical zero"), "{error}");

    let mut unreachable_cap = synchronized_recipe();
    unreachable_cap["total_iterations"] = serde_json::json!(3);
    let error = rejected_root_recipe_error(&unreachable_cap);
    assert!(error.contains("does not reach its cap"), "{error}");

    let mut noncanonical_ramp = synchronized_recipe();
    noncanonical_ramp["mcts_simulation_ramp"] = serde_json::json!(0);
    let error = rejected_root_recipe_error(&noncanonical_ramp);
    assert!(error.contains("noncanonical ramp"), "{error}");

    let mut ineffective_solver = synchronized_recipe();
    ineffective_solver["solver_games"] = serde_json::json!(1);
    let error = rejected_root_recipe_error(&ineffective_solver);
    assert!(error.contains("connect4 profile"), "{error}");
}

#[test]
fn watcher_rejects_collector_search_recipe_changes_within_a_run() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let root_checkpoint = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let root_standalone = read_filesystem_head(root.path()).unwrap().unwrap();
    let recipe = synchronized_recipe();
    let root_commit = replace_with_synchronized_commit(
        root.path(),
        &root_standalone.run_commit_id,
        &recipe,
        1,
        None,
        None,
    );
    select_test_head(root.path(), &root_checkpoint, &root_commit);

    let (child_checkpoint, child_standalone) = publish_test_checkpoint_with_parent(
        root.path(),
        CONTRACT_TEST_MODEL,
        &expected,
        2,
        Some(root_checkpoint.clone()),
        Some(root_commit),
    );
    let mut changed = recipe;
    changed["collector_c_puct"] = serde_json::json!(f64::from(1.5_f32));
    let child_commit = replace_with_synchronized_commit(
        root.path(),
        &child_standalone,
        &changed,
        2,
        Some(&root_checkpoint),
        None,
    );
    select_test_head(root.path(), &child_checkpoint, &child_commit);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(
        error.contains("preserve their mode and exact recipe"),
        "{error}"
    );
}

#[test]
fn watcher_rejects_same_checkpoint_run_commit_children() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let parent = read_filesystem_head(root.path()).unwrap().unwrap();
    let child_run_commit_id = publish_test_run_commit(
        root.path(),
        &expected,
        &checkpoint_id,
        1,
        Some(parent.run_commit_id),
    );
    select_test_head(root.path(), &checkpoint_id, &child_run_commit_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("must select a new checkpoint"));
}

#[test]
fn watcher_rejects_adopting_synchronized_mode_after_a_standalone_root() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let root_checkpoint = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let root_head = read_filesystem_head(root.path()).unwrap().unwrap();
    let (child_checkpoint, child_commit_id) = publish_test_checkpoint_with_parent(
        root.path(),
        CONTRACT_TEST_MODEL,
        &expected,
        2,
        Some(root_checkpoint.clone()),
        Some(root_head.run_commit_id),
    );
    let original = std::fs::read(run_commit_path(root.path(), &child_commit_id)).unwrap();
    let mut commit: serde_json::Value = serde_json::from_slice(&original).unwrap();
    let recipe = synchronized_recipe();
    commit["run_recipe_id"] = serde_json::Value::String(sha256_hex(&canonical(&recipe)));
    commit["run_recipe"] = recipe;
    commit["orchestration"] = orchestration(1, Some(&root_checkpoint));
    let bytes = canonical(&commit);
    let adopted_commit_id = sha256_hex(&bytes);
    std::fs::write(run_commit_path(root.path(), &adopted_commit_id), bytes).unwrap();
    select_test_head(root.path(), &child_checkpoint, &adopted_commit_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("standalone and synchronized run modes cannot be mixed"));
}

#[test]
fn watcher_rejects_root_collection_with_a_source_checkpoint() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let standalone = read_filesystem_head(root.path()).unwrap().unwrap();
    let mut recipe = synchronized_recipe();
    recipe["evaluation_interval"] = serde_json::json!(0);
    let invalid_source = "d".repeat(64);
    let run_commit_id = replace_with_synchronized_commit(
        root.path(),
        &standalone.run_commit_id,
        &recipe,
        1,
        Some(&invalid_source),
        None,
    );
    select_test_head(root.path(), &checkpoint_id, &run_commit_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("source_checkpoint_id must equal its parent checkpoint"));
}

#[test]
fn watcher_rejects_reused_collection_scope_within_a_run() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let root_checkpoint = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let root_standalone = read_filesystem_head(root.path()).unwrap().unwrap();
    let mut recipe = synchronized_recipe();
    recipe["evaluation_interval"] = serde_json::json!(0);
    let repeated_scope = format!("{:064x}", 1);
    let root_commit = replace_with_synchronized_commit(
        root.path(),
        &root_standalone.run_commit_id,
        &recipe,
        1,
        None,
        Some(&repeated_scope),
    );
    select_test_head(root.path(), &root_checkpoint, &root_commit);
    let (child_checkpoint, child_standalone) = publish_test_checkpoint_with_parent(
        root.path(),
        CONTRACT_TEST_MODEL,
        &expected,
        2,
        Some(root_checkpoint.clone()),
        Some(root_commit),
    );
    let child_commit = replace_with_synchronized_commit(
        root.path(),
        &child_standalone,
        &recipe,
        2,
        Some(&root_checkpoint),
        Some(&repeated_scope),
    );
    select_test_head(root.path(), &child_checkpoint, &child_commit);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("collection_scope_id must be unique"));
}

#[test]
fn corrupt_or_cross_profile_updates_preserve_last_valid_model() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let accepted = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::clone(&evaluator),
    );
    watcher.try_load_existing().unwrap();

    let rejected = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &identity("other"), 2);
    assert_ne!(accepted, rejected);
    let error = watcher.try_load_existing().unwrap_err().to_string();
    assert!(error.contains("does not match runtime profile"));
    assert!(evaluator.read().unwrap().is_some());
    assert_eq!(
        watcher
            .model_info()
            .read()
            .unwrap()
            .checkpoint_id
            .as_deref(),
        Some(accepted.as_str())
    );
}

#[test]
fn blob_digest_corruption_is_rejected_before_onnx_loading() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let digest = sha256_hex(CONTRACT_TEST_MODEL);
    std::fs::write(onnx_blob_path(root.path(), &digest), b"corrupt").unwrap();
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = watcher.try_load_existing().unwrap_err().to_string();
    assert!(error.contains("blob size") || error.contains("blob digest"));
}

#[test]
fn malformed_run_head_is_present_but_fatal() {
    let root = tempdir().unwrap();
    let head_path = crate::artifact::run_head_path(root.path());
    std::fs::create_dir_all(head_path.parent().unwrap()).unwrap();
    std::fs::write(head_path, b"{ }").unwrap();
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(identity("contract_test")),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );
    assert!(watcher.try_load_existing().is_err());
}

#[test]
fn unchanged_checkpoint_is_not_reloaded() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 7);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );
    assert!(watcher.try_load_existing().unwrap());
    assert!(watcher.try_load_existing().unwrap());
    assert_eq!(
        watcher
            .model_info()
            .read()
            .unwrap()
            .checkpoint_id
            .as_deref(),
        Some(checkpoint_id.as_str())
    );
}

#[test]
fn accepted_head_compare_and_set_prevents_overlapping_load_regression() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 7);
    let evaluator = Arc::new(RwLock::new(None));
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected.clone()),
        ModelSelection::Latest,
        Arc::clone(&evaluator),
    );
    watcher.try_load_existing().unwrap();
    let head_one = read_filesystem_head(root.path()).unwrap().unwrap();
    let stale_expected = Some(AcceptedHead {
        model_checkpoint_id: checkpoint_id.clone(),
        run_commit_id: head_one.run_commit_id.clone(),
    });

    let (checkpoint_two, run_commit_two) = publish_test_checkpoint_with_parent(
        root.path(),
        CONTRACT_TEST_MODEL,
        &expected,
        8,
        Some(checkpoint_id.clone()),
        Some(head_one.run_commit_id),
    );
    watcher.try_load_existing().unwrap();

    let (checkpoint_three, run_commit_three) = publish_test_checkpoint_with_parent(
        root.path(),
        CONTRACT_TEST_MODEL,
        &expected,
        9,
        Some(checkpoint_two),
        Some(run_commit_two.clone()),
    );
    let candidate_three = resolve_filesystem_head(
        root.path(),
        RunHeadV2 {
            schema_version: 2,
            checkpoint_id: checkpoint_three,
            run_commit_id: run_commit_three,
        },
        &expected,
        9,
        ModelSelection::Latest,
    )
    .unwrap();
    let new_evaluator = watcher
        .model_spec
        .load(&candidate_three.model_path)
        .unwrap();

    assert_eq!(
        ModelWatcher::commit_candidate(
            candidate_three,
            Some(new_evaluator),
            stale_expected,
            &evaluator,
            &watcher.accepted_head,
            &watcher.model_info,
        )
        .unwrap(),
        LoadOutcome::Unchanged
    );
    assert_eq!(
        watcher
            .accepted_head
            .read()
            .unwrap()
            .as_ref()
            .map(|head| head.run_commit_id.as_str()),
        Some(run_commit_two.as_str())
    );
}

#[test]
fn missing_run_commit_is_rejected_before_model_loading() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    select_test_head(root.path(), &checkpoint_id, &"d".repeat(64));
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("failed to read immutable RunCommit"));
}

#[test]
fn duplicate_run_commit_fields_are_rejected() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let first_head = read_filesystem_head(root.path()).unwrap().unwrap();
    let original = std::fs::read(run_commit_path(root.path(), &first_head.run_commit_id)).unwrap();
    let mut duplicate = br#"{"schema_version":1,"#.to_vec();
    duplicate.extend_from_slice(&original[1..]);
    let duplicate_id = sha256_hex(&duplicate);
    let path = run_commit_path(root.path(), &duplicate_id);
    std::fs::write(path, duplicate).unwrap();
    select_test_head(root.path(), &checkpoint_id, &duplicate_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("duplicate field `schema_version`"));
}

#[test]
fn omitted_required_nullable_run_commit_field_is_rejected() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let first_head = read_filesystem_head(root.path()).unwrap().unwrap();
    let original = std::fs::read(run_commit_path(root.path(), &first_head.run_commit_id)).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
    value.as_object_mut().unwrap().remove("champion");
    let incomplete = canonical(&value);
    let incomplete_id = sha256_hex(&incomplete);
    std::fs::write(run_commit_path(root.path(), &incomplete_id), incomplete).unwrap();
    select_test_head(root.path(), &checkpoint_id, &incomplete_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );

    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("RunCommit fields must be exact"));
}

#[test]
fn run_commit_rejects_noncanonical_whitespace_and_wrong_stats_id() {
    let root = tempdir().unwrap();
    let expected = identity("contract_test");
    let checkpoint_id = publish_test_checkpoint(root.path(), CONTRACT_TEST_MODEL, &expected, 1);
    let first_head = read_filesystem_head(root.path()).unwrap().unwrap();
    let original = std::fs::read(run_commit_path(root.path(), &first_head.run_commit_id)).unwrap();

    let mut spaced = vec![b'{', b' '];
    spaced.extend_from_slice(&original[1..]);
    let spaced_id = sha256_hex(&spaced);
    std::fs::write(run_commit_path(root.path(), &spaced_id), spaced).unwrap();
    select_test_head(root.path(), &checkpoint_id, &spaced_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected.clone()),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );
    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("non-canonical JSON whitespace"));

    let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
    value["stats_id"] = serde_json::Value::String("d".repeat(64));
    let wrong_stats = canonical(&value);
    let wrong_stats_id = sha256_hex(&wrong_stats);
    std::fs::write(run_commit_path(root.path(), &wrong_stats_id), wrong_stats).unwrap();
    select_test_head(root.path(), &checkpoint_id, &wrong_stats_id);
    let watcher = ModelWatcher::new(
        root.path(),
        model_spec(expected),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );
    let error = format!("{:#}", watcher.try_load_existing().unwrap_err());
    assert!(error.contains("embedded stats snapshot SHA-256"));
}

#[test]
fn path_errors_are_not_treated_as_absence() {
    let root = tempdir().unwrap();
    let blocker = root.path().join("not-a-directory");
    std::fs::write(&blocker, b"blocker").unwrap();
    let watcher = ModelWatcher::new(
        blocker,
        model_spec(identity("contract_test")),
        ModelSelection::Latest,
        Arc::new(RwLock::new(None)),
    );
    assert!(watcher.try_load_existing().is_err());
}
