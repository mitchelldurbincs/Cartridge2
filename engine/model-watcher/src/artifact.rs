//! Strict content-addressed checkpoint protocol shared by filesystem and S3.

use algorithm_core::ModelArtifactContract;
use anyhow::{anyhow, bail, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::ModelSelection;

pub(crate) const CHECKPOINT_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub(crate) const RUN_HEAD_SCHEMA_VERSION: u32 = 2;
pub(crate) const RUN_COMMIT_SCHEMA_VERSION: u32 = 1;
pub(crate) const STATS_SNAPSHOT_SCHEMA_VERSION: u32 = 2;
pub(crate) const RUN_HEAD_CHANNEL: &str = "current";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactProfile {
    pub algorithm_id: String,
    pub env_id: String,
    pub env_contract_version: u32,
    pub model_artifact_schema_version: u32,
    pub model_contract: String,
}

impl From<&ModelArtifactContract> for ArtifactProfile {
    fn from(identity: &ModelArtifactContract) -> Self {
        Self {
            algorithm_id: identity.algorithm_id.clone(),
            env_id: identity.env_id.clone(),
            env_contract_version: identity.env_contract_version,
            model_artifact_schema_version: identity.schema_version,
            model_contract: identity.model_contract.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BlobReference {
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CheckpointManifestV1 {
    pub schema_version: u32,
    pub profile: ArtifactProfile,
    pub step: u64,
    pub parent_checkpoint_id: Option<String>,
    pub config_sha256: String,
    pub onnx: BlobReference,
    pub learner_state: BlobReference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunHeadV2 {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub run_commit_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChampionReferenceV1 {
    pub checkpoint_id: String,
    pub evaluation_id: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OrchestrationCommitV1 {
    pub iteration: u64,
    pub collection_scope_id: String,
    pub source_checkpoint_id: Option<String>,
    pub episodes_generated: u64,
    pub transitions_generated: u64,
    pub training_steps: u64,
    pub collector_simulations: u32,
    pub collector_seed: Option<u64>,
    pub evaluation_seed: Option<u64>,
    pub actor_time_seconds: f64,
    pub trainer_time_seconds: f64,
    pub eval_time_seconds: f64,
    pub total_time_seconds: f64,
    pub eval_win_rate: Option<f64>,
    pub eval_draw_rate: Option<f64>,
    pub timestamp: String,
    pub evaluation_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunRecipeV1 {
    pub schema_version: u32,
    pub learner_config_sha256: String,
    pub learner_recipe: Box<RawValue>,
    pub total_iterations: u64,
    pub episodes_per_iteration: u32,
    pub training_steps_per_iteration: u64,
    pub num_actors: u32,
    pub collector_episode_timeout_seconds: u64,
    pub collector_eval_batch_size: u32,
    pub collector_onnx_intra_threads: u32,
    pub mcts_start_simulations: u32,
    pub mcts_max_simulations: u32,
    pub mcts_simulation_ramp: u32,
    pub collector_c_puct: f64,
    pub collector_temperature: f64,
    pub collector_late_temperature: f64,
    pub temperature_move_threshold: u32,
    pub collector_dirichlet_alpha: f64,
    pub collector_dirichlet_weight: f64,
    pub collector_seed_strategy: String,
    pub evaluation_interval: u64,
    pub evaluation_games: u32,
    #[serde(rename = "evaluation_simulations")]
    pub _evaluation_simulations: u32,
    pub evaluation_temperature: f64,
    pub evaluation_win_threshold: f64,
    #[serde(rename = "evaluation_vs_random")]
    pub _evaluation_vs_random: bool,
    pub solver_games: u32,
    pub evaluation_seed: u64,
    pub replay_policy: String,
    pub promotion_metric: String,
    pub promotion_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HistoryEntryV1 {
    pub step: u64,
    pub total_loss: f64,
    pub value_loss: f64,
    pub policy_loss: f64,
    pub learning_rate: f64,
    pub grad_norm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluationStatsV1 {
    pub step: u64,
    pub win_rate: f64,
    pub draw_rate: f64,
    pub loss_rate: f64,
    pub games_played: u64,
    pub avg_game_length: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainingStatsV2 {
    pub step: u64,
    pub total_steps: u64,
    pub total_loss: f64,
    pub value_loss: f64,
    pub policy_loss: f64,
    pub learning_rate: f64,
    pub samples_seen: u64,
    pub replay_record_count: u64,
    pub last_checkpoint: String,
    pub timestamp: f64,
    pub history: Vec<HistoryEntryV1>,
    pub env_id: String,
    pub last_eval: Option<EvaluationStatsV1>,
    pub eval_history: Vec<EvaluationStatsV1>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StatsSnapshotV2 {
    pub schema_version: u32,
    pub profile: ArtifactProfile,
    pub config_sha256: String,
    pub checkpoint_id: String,
    pub step: u64,
    pub stats: TrainingStatsV2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunCommitWireV1 {
    schema_version: u32,
    profile: ArtifactProfile,
    config_sha256: String,
    parent_run_commit_id: Option<String>,
    checkpoint_id: String,
    run_recipe_id: Option<String>,
    run_recipe: Option<Box<RawValue>>,
    stats_id: String,
    stats_snapshot: Box<RawValue>,
    champion: Option<ChampionReferenceV1>,
    evaluation_head_id: Option<String>,
    orchestration: Option<OrchestrationCommitV1>,
}

#[derive(Debug, Clone)]
pub(crate) struct RunCommitV1 {
    pub profile: ArtifactProfile,
    pub config_sha256: String,
    pub parent_run_commit_id: Option<String>,
    pub checkpoint_id: String,
    pub run_recipe_id: Option<String>,
    pub run_recipe: Option<RunRecipeV1>,
    pub stats_snapshot: StatsSnapshotV2,
    pub champion: Option<ChampionReferenceV1>,
    pub evaluation_head_id: Option<String>,
    pub orchestration: Option<OrchestrationCommitV1>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedRunCommit {
    pub run_commit_id: String,
    pub commit: RunCommitV1,
    pub manifest: CheckpointManifestV1,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedCheckpoint {
    pub checkpoint_id: String,
    pub run_commit_id: String,
    pub manifest: CheckpointManifestV1,
    pub model_path: PathBuf,
}

const RUN_COMMIT_FIELDS: &[&str] = &[
    "schema_version",
    "profile",
    "config_sha256",
    "parent_run_commit_id",
    "checkpoint_id",
    "run_recipe_id",
    "run_recipe",
    "stats_id",
    "stats_snapshot",
    "champion",
    "evaluation_head_id",
    "orchestration",
];
const STATS_SNAPSHOT_FIELDS: &[&str] = &[
    "schema_version",
    "profile",
    "config_sha256",
    "checkpoint_id",
    "step",
    "stats",
];
const TRAINING_STATS_FIELDS: &[&str] = &[
    "step",
    "total_steps",
    "total_loss",
    "value_loss",
    "policy_loss",
    "learning_rate",
    "samples_seen",
    "replay_record_count",
    "last_checkpoint",
    "timestamp",
    "history",
    "env_id",
    "last_eval",
    "eval_history",
];
const HISTORY_FIELDS: &[&str] = &[
    "step",
    "total_loss",
    "value_loss",
    "policy_loss",
    "learning_rate",
    "grad_norm",
];
const EVALUATION_STATS_FIELDS: &[&str] = &[
    "step",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "games_played",
    "avg_game_length",
    "timestamp",
];
const CHAMPION_FIELDS: &[&str] = &["checkpoint_id", "evaluation_id"];
const ORCHESTRATION_FIELDS: &[&str] = &[
    "iteration",
    "collection_scope_id",
    "source_checkpoint_id",
    "episodes_generated",
    "transitions_generated",
    "training_steps",
    "collector_simulations",
    "collector_seed",
    "evaluation_seed",
    "actor_time_seconds",
    "trainer_time_seconds",
    "eval_time_seconds",
    "total_time_seconds",
    "eval_win_rate",
    "eval_draw_rate",
    "timestamp",
    "evaluation_id",
];
const RUN_RECIPE_FIELDS: &[&str] = &[
    "schema_version",
    "learner_config_sha256",
    "learner_recipe",
    "total_iterations",
    "episodes_per_iteration",
    "training_steps_per_iteration",
    "num_actors",
    "collector_episode_timeout_seconds",
    "collector_eval_batch_size",
    "collector_onnx_intra_threads",
    "mcts_start_simulations",
    "mcts_max_simulations",
    "mcts_simulation_ramp",
    "collector_c_puct",
    "collector_temperature",
    "collector_late_temperature",
    "temperature_move_threshold",
    "collector_dirichlet_alpha",
    "collector_dirichlet_weight",
    "collector_seed_strategy",
    "evaluation_interval",
    "evaluation_games",
    "evaluation_simulations",
    "evaluation_temperature",
    "evaluation_win_threshold",
    "evaluation_vs_random",
    "solver_games",
    "evaluation_seed",
    "replay_policy",
    "promotion_metric",
    "promotion_margin",
];

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn validate_digest(label: &str, digest: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be a lowercase 64-character SHA-256 digest, got {digest:?}");
    }
    Ok(())
}

pub(crate) fn parse_canonical_json<T: DeserializeOwned>(label: &str, bytes: &[u8]) -> Result<T> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .with_context(|| format!("failed to parse {label} as JSON"))?;
    let canonical =
        serde_json::to_vec(&value).with_context(|| format!("failed to canonicalize {label}"))?;
    if canonical != bytes {
        bail!("{label} is not canonical JSON");
    }
    serde_json::from_value(value).with_context(|| format!("invalid {label} contract"))
}

pub(crate) fn validate_run_head(head: &RunHeadV2) -> Result<()> {
    if head.schema_version != RUN_HEAD_SCHEMA_VERSION {
        bail!(
            "unsupported run-head schema {}, expected {}",
            head.schema_version,
            RUN_HEAD_SCHEMA_VERSION
        );
    }
    validate_digest("run-head checkpoint_id", &head.checkpoint_id)?;
    validate_digest("run-head run_commit_id", &head.run_commit_id)
}

fn validate_exact_fields(label: &str, value: &serde_json::Value, fields: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("{label} must be a JSON object"))?;
    let missing: Vec<_> = fields
        .iter()
        .filter(|field| !object.contains_key(**field))
        .copied()
        .collect();
    let extra: Vec<_> = object
        .keys()
        .filter(|field| !fields.contains(&field.as_str()))
        .cloned()
        .collect();
    if !missing.is_empty() || !extra.is_empty() || object.len() != fields.len() {
        bail!("{label} fields must be exact (missing={missing:?}, extra={extra:?})");
    }
    Ok(())
}

fn validate_stats_json_shape(value: &serde_json::Value) -> Result<()> {
    validate_exact_fields("stats snapshot", value, STATS_SNAPSHOT_FIELDS)?;
    let stats = &value["stats"];
    validate_exact_fields("training stats", stats, TRAINING_STATS_FIELDS)?;
    let history = stats["history"]
        .as_array()
        .ok_or_else(|| anyhow!("stats.history must be an array"))?;
    for (index, entry) in history.iter().enumerate() {
        validate_exact_fields(&format!("stats.history[{index}]"), entry, HISTORY_FIELDS)?;
    }
    if !stats["last_eval"].is_null() {
        validate_exact_fields(
            "stats.last_eval",
            &stats["last_eval"],
            EVALUATION_STATS_FIELDS,
        )?;
    }
    let evaluations = stats["eval_history"]
        .as_array()
        .ok_or_else(|| anyhow!("stats.eval_history must be an array"))?;
    for (index, evaluation) in evaluations.iter().enumerate() {
        validate_exact_fields(
            &format!("stats.eval_history[{index}]"),
            evaluation,
            EVALUATION_STATS_FIELDS,
        )?;
    }
    Ok(())
}

fn validate_run_commit_json_shape(
    bytes: &[u8],
    stats_raw: &RawValue,
    recipe_raw: Option<&RawValue>,
) -> Result<()> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("failed to inspect RunCommit JSON fields")?;
    validate_exact_fields("RunCommit", &value, RUN_COMMIT_FIELDS)?;
    if !value["champion"].is_null() {
        validate_exact_fields("RunCommit champion", &value["champion"], CHAMPION_FIELDS)?;
    }
    if !value["orchestration"].is_null() {
        validate_exact_fields(
            "RunCommit orchestration",
            &value["orchestration"],
            ORCHESTRATION_FIELDS,
        )?;
    }
    let stats_value: serde_json::Value =
        serde_json::from_str(stats_raw.get()).context("failed to inspect stats snapshot fields")?;
    validate_stats_json_shape(&stats_value)?;
    if let Some(recipe_raw) = recipe_raw {
        let recipe_value: serde_json::Value = serde_json::from_str(recipe_raw.get())
            .context("failed to inspect run recipe fields")?;
        validate_exact_fields("run recipe", &recipe_value, RUN_RECIPE_FIELDS)?;
    }
    Ok(())
}

fn validate_finite(label: &str, value: f64, nonnegative: bool) -> Result<()> {
    if !value.is_finite() || (nonnegative && value < 0.0) {
        let qualifier = if nonnegative {
            "finite and nonnegative"
        } else {
            "finite"
        };
        bail!("{label} must be {qualifier}");
    }
    Ok(())
}

fn validate_rate(label: &str, value: f64) -> Result<()> {
    validate_finite(label, value, true)?;
    if value > 1.0 {
        bail!("{label} must be between zero and one");
    }
    Ok(())
}

fn validate_utc_timestamp(label: &str, value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    let separators = [
        (4, b'-'),
        (7, b'-'),
        (10, b'T'),
        (13, b':'),
        (16, b':'),
        (19, b'.'),
        (26, b'Z'),
    ];
    if bytes.len() != 27
        || separators
            .iter()
            .any(|(index, expected)| bytes[*index] != *expected)
        || bytes.iter().enumerate().any(|(index, byte)| {
            !separators.iter().any(|(separator, _)| *separator == index) && !byte.is_ascii_digit()
        })
    {
        bail!("{label} must be UTC YYYY-MM-DDTHH:MM:SS.ffffffZ");
    }
    let parse = |range: std::ops::Range<usize>| -> Result<u32> {
        value[range]
            .parse::<u32>()
            .with_context(|| format!("{label} contains an invalid timestamp component"))
    };
    let year = parse(0..4)?;
    let month = parse(5..7)?;
    let day = parse(8..10)?;
    let hour = parse(11..13)?;
    let minute = parse(14..16)?;
    let second = parse(17..19)?;
    let leap_year =
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > days_in_month || hour >= 24 || minute >= 60 || second >= 60 {
        bail!("{label} is not a valid UTC calendar timestamp");
    }
    Ok(())
}

fn validate_evaluation_stats(label: &str, stats: &EvaluationStatsV1) -> Result<()> {
    for (field, value) in [
        ("win_rate", stats.win_rate),
        ("draw_rate", stats.draw_rate),
        ("loss_rate", stats.loss_rate),
    ] {
        validate_rate(&format!("{label}.{field}"), value)?;
    }
    validate_finite(
        &format!("{label}.avg_game_length"),
        stats.avg_game_length,
        true,
    )?;
    validate_finite(&format!("{label}.timestamp"), stats.timestamp, true)?;
    let rate_sum = stats.win_rate + stats.draw_rate + stats.loss_rate;
    if stats.games_played == 0 {
        if rate_sum != 0.0 || stats.avg_game_length != 0.0 {
            bail!("{label} with zero games must contain zero rates and length");
        }
    } else if stats.avg_game_length <= 0.0 || (rate_sum - 1.0).abs() > 1e-12 {
        bail!("{label} rates must sum to one and average length must be positive");
    }
    Ok(())
}

fn validate_training_stats(snapshot: &StatsSnapshotV2) -> Result<()> {
    let stats = &snapshot.stats;
    if stats.step != snapshot.step {
        bail!("stats.step does not match stats snapshot step");
    }
    if stats.total_steps < stats.step {
        bail!("stats.total_steps cannot be less than stats.step");
    }
    if stats.last_checkpoint != snapshot.checkpoint_id {
        bail!("stats.last_checkpoint does not match stats snapshot checkpoint_id");
    }
    if stats.env_id != snapshot.profile.env_id {
        bail!("stats.env_id does not match stats snapshot profile");
    }
    for (field, value) in [
        ("total_loss", stats.total_loss),
        ("value_loss", stats.value_loss),
        ("policy_loss", stats.policy_loss),
        ("learning_rate", stats.learning_rate),
        ("timestamp", stats.timestamp),
    ] {
        validate_finite(&format!("stats.{field}"), value, true)?;
    }
    let mut previous_history_step = None;
    for (index, entry) in stats.history.iter().enumerate() {
        if entry.step > stats.step
            || previous_history_step.is_some_and(|previous| entry.step <= previous)
        {
            bail!("stats.history steps must strictly increase within stats.step");
        }
        previous_history_step = Some(entry.step);
        for (field, value) in [
            ("total_loss", entry.total_loss),
            ("value_loss", entry.value_loss),
            ("policy_loss", entry.policy_loss),
            ("learning_rate", entry.learning_rate),
        ] {
            validate_finite(&format!("stats.history[{index}].{field}"), value, true)?;
        }
        if let Some(grad_norm) = entry.grad_norm {
            validate_finite(
                &format!("stats.history[{index}].grad_norm"),
                grad_norm,
                true,
            )?;
        }
    }
    let mut previous_eval_step = None;
    let mut previous_eval_timestamp = None;
    for (index, evaluation) in stats.eval_history.iter().enumerate() {
        validate_evaluation_stats(&format!("stats.eval_history[{index}]"), evaluation)?;
        if evaluation.step > stats.step
            || previous_eval_step.is_some_and(|previous| evaluation.step <= previous)
            || previous_eval_timestamp.is_some_and(|previous| evaluation.timestamp < previous)
        {
            bail!("stats.eval_history chronology is invalid");
        }
        previous_eval_step = Some(evaluation.step);
        previous_eval_timestamp = Some(evaluation.timestamp);
    }
    match (&stats.last_eval, stats.eval_history.last()) {
        (None, None) => {}
        (Some(last), Some(expected)) if last == expected => {
            validate_evaluation_stats("stats.last_eval", last)?;
        }
        _ => bail!("stats.last_eval must equal the final eval_history record"),
    }
    Ok(())
}

fn validate_orchestration(orchestration: &OrchestrationCommitV1) -> Result<()> {
    if orchestration.iteration == 0 {
        bail!("orchestration.iteration must be positive");
    }
    validate_digest(
        "orchestration.collection_scope_id",
        &orchestration.collection_scope_id,
    )?;
    if let Some(source_checkpoint_id) = &orchestration.source_checkpoint_id {
        validate_digest("orchestration.source_checkpoint_id", source_checkpoint_id)?;
    }
    let phase_total = orchestration.actor_time_seconds
        + orchestration.trainer_time_seconds
        + orchestration.eval_time_seconds;
    for (field, value) in [
        ("actor_time_seconds", orchestration.actor_time_seconds),
        ("trainer_time_seconds", orchestration.trainer_time_seconds),
        ("eval_time_seconds", orchestration.eval_time_seconds),
        ("total_time_seconds", orchestration.total_time_seconds),
    ] {
        validate_finite(&format!("orchestration.{field}"), value, true)?;
    }
    if orchestration.total_time_seconds + 1e-12 < phase_total {
        bail!("orchestration.total_time_seconds is shorter than its phases");
    }
    match (orchestration.eval_win_rate, orchestration.eval_draw_rate) {
        (None, None) => {}
        (Some(win), Some(draw)) => {
            validate_rate("orchestration.eval_win_rate", win)?;
            validate_rate("orchestration.eval_draw_rate", draw)?;
            if win + draw > 1.0 {
                bail!("orchestration evaluation rates exceed one");
            }
        }
        _ => bail!("orchestration evaluation rates are incomplete"),
    }
    validate_utc_timestamp("orchestration.timestamp", &orchestration.timestamp)?;
    match &orchestration.evaluation_id {
        None => {
            if orchestration.eval_time_seconds != 0.0 || orchestration.eval_win_rate.is_some() {
                bail!("non-evaluation orchestration contains evaluation metrics");
            }
        }
        Some(evaluation_id) => validate_digest("orchestration.evaluation_id", evaluation_id)?,
    }
    Ok(())
}

fn validate_run_recipe(
    recipe: &RunRecipeV1,
    config_sha256: &str,
    environment_max_horizon: u32,
) -> Result<()> {
    if recipe.schema_version != 1 {
        bail!(
            "unsupported run recipe schema {}, expected 1",
            recipe.schema_version
        );
    }
    validate_digest(
        "run_recipe.learner_config_sha256",
        &recipe.learner_config_sha256,
    )?;
    let learner_recipe: serde_json::Value = serde_json::from_str(recipe.learner_recipe.get())
        .context("invalid algorithm-owned learner recipe")?;
    if learner_recipe
        .as_object()
        .is_none_or(|object| object.is_empty())
    {
        bail!("run_recipe.learner_recipe must be a nonempty JSON object");
    }
    let actual_learner_config = sha256_hex(recipe.learner_recipe.get().as_bytes());
    if recipe.learner_config_sha256 != actual_learner_config
        || recipe.learner_config_sha256 != config_sha256
    {
        bail!("run recipe learner config digest does not match its object/RunCommit");
    }
    if recipe.total_iterations == 0
        || recipe.training_steps_per_iteration == 0
        || recipe.num_actors == 0
        || recipe.collector_episode_timeout_seconds == 0
        || recipe.collector_eval_batch_size == 0
        || recipe.collector_onnx_intra_threads == 0
        || recipe.mcts_start_simulations == 0
        || recipe.mcts_max_simulations == 0
        || recipe.evaluation_games == 0
    {
        bail!("run recipe positive-count fields must be positive");
    }
    if recipe.mcts_start_simulations > recipe.mcts_max_simulations {
        bail!("run recipe MCTS start simulations exceed its maximum");
    }
    if recipe.mcts_start_simulations == recipe.mcts_max_simulations {
        if recipe.mcts_simulation_ramp != 0 {
            bail!("run recipe constant MCTS schedule requires a zero ramp");
        }
    } else {
        let delta = recipe.mcts_max_simulations - recipe.mcts_start_simulations;
        if recipe.mcts_simulation_ramp == 0 || recipe.mcts_simulation_ramp > delta {
            bail!("run recipe ramped MCTS schedule has a noncanonical ramp");
        }
        let steps_to_cap = u64::from(delta.div_ceil(recipe.mcts_simulation_ramp));
        if recipe.total_iterations - 1 < steps_to_cap {
            bail!("run recipe MCTS schedule does not reach its cap within the run");
        }
    }
    if recipe.num_actors > recipe.episodes_per_iteration {
        bail!("run recipe num_actors exceeds episodes_per_iteration");
    }
    if recipe.collector_seed_strategy != "system_entropy_v1" {
        bail!("unsupported run recipe collector_seed_strategy");
    }
    if recipe.replay_policy != "scoped_fresh_iteration_v1" {
        bail!("unsupported run recipe replay_policy");
    }
    if !matches!(
        recipe.promotion_metric.as_str(),
        "win_rate" | "solver_optimal"
    ) {
        bail!("unsupported run recipe promotion_metric");
    }
    if recipe.promotion_metric == "solver_optimal" && recipe.solver_games == 0 {
        bail!("solver_optimal run recipe requires solver games");
    }
    u64::from(recipe.evaluation_games.max(recipe.solver_games))
        .checked_sub(1)
        .and_then(|last_index| recipe.evaluation_seed.checked_add(last_index))
        .ok_or_else(|| anyhow!("run recipe evaluation seed schedule overflows u64"))?;
    for (label, value) in [
        ("run_recipe.collector_c_puct", recipe.collector_c_puct),
        (
            "run_recipe.collector_temperature",
            recipe.collector_temperature,
        ),
        (
            "run_recipe.collector_late_temperature",
            recipe.collector_late_temperature,
        ),
        (
            "run_recipe.collector_dirichlet_alpha",
            recipe.collector_dirichlet_alpha,
        ),
        (
            "run_recipe.collector_dirichlet_weight",
            recipe.collector_dirichlet_weight,
        ),
        (
            "run_recipe.evaluation_temperature",
            recipe.evaluation_temperature,
        ),
    ] {
        validate_finite(label, value, true)?;
        if value > f64::from(f32::MAX)
            || f64::from(value as f32) != value
            || (value == 0.0 && value.is_sign_negative())
        {
            bail!("{label} must be the exact canonical value of a finite f32");
        }
    }
    if recipe.collector_dirichlet_weight > 1.0 {
        bail!("run_recipe.collector_dirichlet_weight must be a rate in [0, 1]");
    }
    if (recipe.collector_dirichlet_alpha == 0.0) != (recipe.collector_dirichlet_weight == 0.0) {
        bail!("run recipe collector Dirichlet alpha and weight must both be zero to disable noise");
    }
    if recipe.temperature_move_threshold == 0 {
        if recipe.collector_late_temperature != recipe.collector_temperature {
            bail!(
                "run recipe collector late temperature must equal base temperature when its move threshold is zero"
            );
        }
    } else if recipe.collector_late_temperature == recipe.collector_temperature {
        bail!(
            "run recipe collector late temperature must differ from base temperature when its schedule is enabled"
        );
    }
    if environment_max_horizon == 0 {
        bail!("runtime environment max_horizon must be positive");
    }
    if recipe.temperature_move_threshold != 0
        && recipe.temperature_move_threshold >= environment_max_horizon
    {
        bail!("run recipe temperature threshold is unreachable for its environment");
    }
    validate_rate(
        "run_recipe.evaluation_win_threshold",
        recipe.evaluation_win_threshold,
    )?;
    validate_rate("run_recipe.promotion_margin", recipe.promotion_margin)?;
    if (recipe.promotion_metric == "win_rate" && recipe.promotion_margin != 0.0)
        || (recipe.promotion_metric == "solver_optimal" && recipe.evaluation_win_threshold != 0.0)
    {
        bail!("run recipe inactive promotion parameter must be canonical zero");
    }
    Ok(())
}

fn expected_collector_simulations(recipe: &RunRecipeV1, iteration: u64) -> Result<u32> {
    let ramp_count = iteration
        .checked_sub(1)
        .ok_or_else(|| anyhow!("orchestration iteration must be positive"))?;
    let ramp_rate = u64::from(recipe.mcts_simulation_ramp);
    if ramp_rate == 0 {
        return Ok(recipe.mcts_start_simulations);
    }
    let start = u64::from(recipe.mcts_start_simulations);
    let maximum = u64::from(recipe.mcts_max_simulations);
    let remaining = maximum - start;
    let steps_to_cap = remaining.div_ceil(ramp_rate);
    if ramp_count >= steps_to_cap {
        return Ok(recipe.mcts_max_simulations);
    }
    let simulations = start + ramp_count * ramp_rate;
    u32::try_from(simulations).context("collector simulation count exceeds u32")
}

pub(crate) fn parse_run_commit(
    label: &str,
    run_commit_id: &str,
    bytes: &[u8],
    expected_contract: &ModelArtifactContract,
    environment_max_horizon: u32,
) -> Result<RunCommitV1> {
    validate_digest("run_commit_id", run_commit_id)?;
    let actual_id = sha256_hex(bytes);
    if actual_id != run_commit_id {
        bail!("{label} SHA-256 is {actual_id}, expected {run_commit_id}");
    }
    let mut in_string = false;
    let mut escaped = false;
    for byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
        } else if *byte == b'"' {
            in_string = true;
        } else if byte.is_ascii_whitespace() {
            bail!("{label} contains non-canonical JSON whitespace");
        }
    }
    // Do not serde-roundtrip these bytes: Python and serde_json intentionally
    // differ for some finite float spellings. RawValue preserves the exact
    // embedded stats bytes so stats_id remains a cross-language guarantee.
    // The Python publisher is the canonical-encoding authority; this consumer
    // independently enforces exact SHA identity, no insignificant whitespace,
    // duplicate/unknown-field rejection, and every serving-relevant binding.
    let wire: RunCommitWireV1 =
        serde_json::from_slice(bytes).with_context(|| format!("invalid {label} contract"))?;
    validate_run_commit_json_shape(
        bytes,
        wire.stats_snapshot.as_ref(),
        wire.run_recipe.as_deref(),
    )?;
    if wire.schema_version != RUN_COMMIT_SCHEMA_VERSION {
        bail!(
            "unsupported run-commit schema {}, expected {}",
            wire.schema_version,
            RUN_COMMIT_SCHEMA_VERSION
        );
    }
    let expected_profile = ArtifactProfile::from(expected_contract);
    if wire.profile != expected_profile {
        bail!(
            "run-commit profile {:?} does not match runtime profile {:?}",
            wire.profile,
            expected_profile
        );
    }
    validate_digest("run_commit.config_sha256", &wire.config_sha256)?;
    validate_digest("run_commit.checkpoint_id", &wire.checkpoint_id)?;
    validate_digest("run_commit.stats_id", &wire.stats_id)?;
    if let Some(parent) = &wire.parent_run_commit_id {
        validate_digest("run_commit.parent_run_commit_id", parent)?;
    }
    let run_recipe = match (&wire.run_recipe_id, &wire.run_recipe) {
        (None, None) => None,
        (Some(run_recipe_id), Some(recipe_raw)) => {
            validate_digest("run_commit.run_recipe_id", run_recipe_id)?;
            let actual_recipe_id = sha256_hex(recipe_raw.get().as_bytes());
            if actual_recipe_id != *run_recipe_id {
                bail!(
                    "embedded run recipe SHA-256 is {actual_recipe_id}, expected {run_recipe_id}"
                );
            }
            let recipe: RunRecipeV1 = serde_json::from_str(recipe_raw.get())
                .context("invalid embedded run recipe contract")?;
            validate_run_recipe(&recipe, &wire.config_sha256, environment_max_horizon)?;
            if wire.profile.env_id != "connect4"
                && (recipe.solver_games > 0 || recipe.promotion_metric == "solver_optimal")
            {
                bail!("run recipe solver evaluation settings require the connect4 profile");
            }
            Some(recipe)
        }
        _ => bail!("run_recipe and run_recipe_id must both be null or both be present"),
    };
    let actual_stats_id = sha256_hex(wire.stats_snapshot.get().as_bytes());
    if actual_stats_id != wire.stats_id {
        bail!(
            "embedded stats snapshot SHA-256 is {actual_stats_id}, expected {}",
            wire.stats_id
        );
    }
    let stats_snapshot: StatsSnapshotV2 = serde_json::from_str(wire.stats_snapshot.get())
        .context("invalid embedded stats snapshot contract")?;
    if stats_snapshot.schema_version != STATS_SNAPSHOT_SCHEMA_VERSION {
        bail!(
            "unsupported stats snapshot schema {}, expected {}",
            stats_snapshot.schema_version,
            STATS_SNAPSHOT_SCHEMA_VERSION
        );
    }
    if stats_snapshot.profile != wire.profile
        || stats_snapshot.config_sha256 != wire.config_sha256
        || stats_snapshot.checkpoint_id != wire.checkpoint_id
    {
        bail!("embedded stats snapshot binding does not match RunCommit");
    }
    validate_digest(
        "stats_snapshot.config_sha256",
        &stats_snapshot.config_sha256,
    )?;
    validate_digest(
        "stats_snapshot.checkpoint_id",
        &stats_snapshot.checkpoint_id,
    )?;
    validate_training_stats(&stats_snapshot)?;
    if let Some(champion) = &wire.champion {
        validate_digest("champion.checkpoint_id", &champion.checkpoint_id)?;
        validate_digest("champion.evaluation_id", &champion.evaluation_id)?;
        if wire.evaluation_head_id.is_none() {
            bail!("a RunCommit champion requires an evaluation head");
        }
    }
    if let Some(evaluation_head_id) = &wire.evaluation_head_id {
        validate_digest("run_commit.evaluation_head_id", evaluation_head_id)?;
    }
    if let Some(orchestration) = &wire.orchestration {
        validate_orchestration(orchestration)?;
    }
    Ok(RunCommitV1 {
        profile: wire.profile,
        config_sha256: wire.config_sha256,
        parent_run_commit_id: wire.parent_run_commit_id,
        checkpoint_id: wire.checkpoint_id,
        run_recipe_id: wire.run_recipe_id,
        run_recipe,
        stats_snapshot,
        champion: wire.champion,
        evaluation_head_id: wire.evaluation_head_id,
        orchestration: wire.orchestration,
    })
}

pub(crate) fn validate_manifest(
    manifest: &CheckpointManifestV1,
    expected_contract: &ModelArtifactContract,
) -> Result<()> {
    if manifest.schema_version != CHECKPOINT_MANIFEST_SCHEMA_VERSION {
        bail!(
            "unsupported checkpoint manifest schema {}, expected {}",
            manifest.schema_version,
            CHECKPOINT_MANIFEST_SCHEMA_VERSION
        );
    }
    let expected_profile = ArtifactProfile::from(expected_contract);
    if manifest.profile != expected_profile {
        bail!(
            "checkpoint profile {:?} does not match runtime profile {:?}",
            manifest.profile,
            expected_profile
        );
    }
    validate_digest("config_sha256", &manifest.config_sha256)?;
    if let Some(parent) = &manifest.parent_checkpoint_id {
        validate_digest("parent_checkpoint_id", parent)?;
    }
    for (label, blob) in [
        ("onnx", &manifest.onnx),
        ("learner_state", &manifest.learner_state),
    ] {
        validate_digest(&format!("{label}.sha256"), &blob.sha256)?;
        if blob.size_bytes == 0 {
            bail!("{label}.size_bytes must be positive");
        }
    }
    Ok(())
}

pub(crate) fn validate_run_commit_chain(
    chain: &[ResolvedRunCommit],
    head: &RunHeadV2,
) -> Result<()> {
    let selected = chain
        .last()
        .ok_or_else(|| anyhow!("RunCommit lineage is empty"))?;
    if selected.run_commit_id != head.run_commit_id
        || selected.commit.checkpoint_id != head.checkpoint_id
        || selected.manifest.step != selected.commit.stats_snapshot.step
    {
        bail!("run head does not match its selected RunCommit/checkpoint binding");
    }

    let mut parent: Option<&ResolvedRunCommit> = None;
    let mut latest_orchestration: Option<&OrchestrationCommitV1> = None;
    let mut lineage_checkpoints = HashSet::new();
    let mut collection_scopes = HashSet::new();
    for entry in chain {
        let commit = &entry.commit;
        let manifest = &entry.manifest;
        if manifest.profile != commit.profile
            || manifest.config_sha256 != commit.config_sha256
            || manifest.step != commit.stats_snapshot.step
        {
            bail!("RunCommit checkpoint does not match its profile/config/stats binding");
        }
        lineage_checkpoints.insert(commit.checkpoint_id.as_str());
        if commit
            .champion
            .as_ref()
            .is_some_and(|champion| !lineage_checkpoints.contains(champion.checkpoint_id.as_str()))
        {
            bail!("RunCommit champion must select a checkpoint in its lineage");
        }

        let inherited_champion = parent.and_then(|value| value.commit.champion.as_ref());
        let inherited_evaluation =
            parent.and_then(|value| value.commit.evaluation_head_id.as_ref());
        let parent_step = parent.map_or(0, |value| value.manifest.step);
        match parent {
            None => {
                if commit.parent_run_commit_id.is_some() || manifest.parent_checkpoint_id.is_some()
                {
                    bail!("the first RunCommit must start both lineages");
                }
            }
            Some(parent_entry) => {
                if commit.parent_run_commit_id.as_deref()
                    != Some(parent_entry.run_commit_id.as_str())
                {
                    bail!("RunCommit parent identity is inconsistent");
                }
                if commit.profile != parent_entry.commit.profile
                    || commit.config_sha256 != parent_entry.commit.config_sha256
                {
                    bail!("RunCommit profile/config changed within a run");
                }
                if commit.checkpoint_id == parent_entry.commit.checkpoint_id {
                    bail!("RunCommit children must select a new checkpoint");
                } else if manifest.parent_checkpoint_id.as_deref()
                    != Some(parent_entry.commit.checkpoint_id.as_str())
                    || manifest.step <= parent_entry.manifest.step
                {
                    bail!("RunCommit checkpoint must be a strictly newer direct child");
                }
            }
        }

        match parent {
            None => {
                if commit.run_recipe.is_some() != commit.orchestration.is_some() {
                    bail!(
                        "a root RunCommit must be either standalone or recipe-owned orchestration"
                    );
                }
            }
            Some(parent_entry) if parent_entry.commit.run_recipe.is_none() => {
                if commit.run_recipe.is_some() || commit.orchestration.is_some() {
                    bail!("standalone and synchronized run modes cannot be mixed");
                }
            }
            Some(parent_entry) => {
                if commit.run_recipe_id != parent_entry.commit.run_recipe_id
                    || commit.run_recipe.is_none()
                    || commit.orchestration.is_none()
                {
                    bail!("recipe-owned RunCommits must preserve their mode and exact recipe");
                }
            }
        }

        match &commit.orchestration {
            None => {
                if commit.champion.as_ref() != inherited_champion
                    || commit.evaluation_head_id.as_ref() != inherited_evaluation
                {
                    bail!("standalone RunCommit changed evaluation state");
                }
            }
            Some(orchestration) => {
                let recipe = commit
                    .run_recipe
                    .as_ref()
                    .ok_or_else(|| anyhow!("orchestration RunCommit requires a run recipe"))?;
                if orchestration.iteration > recipe.total_iterations
                    || orchestration.episodes_generated != u64::from(recipe.episodes_per_iteration)
                    || orchestration.training_steps != recipe.training_steps_per_iteration
                    || orchestration.collector_simulations
                        != expected_collector_simulations(recipe, orchestration.iteration)?
                    || orchestration.collector_seed.is_some()
                {
                    bail!("orchestration does not match its immutable run recipe");
                }
                let expected_source = parent.map(|entry| entry.commit.checkpoint_id.as_str());
                if orchestration.source_checkpoint_id.as_deref() != expected_source {
                    bail!("orchestration source_checkpoint_id must equal its parent checkpoint");
                }
                if !collection_scopes.insert(orchestration.collection_scope_id.as_str()) {
                    bail!("orchestration collection_scope_id must be unique within the run");
                }
                if orchestration.training_steps == 0
                    || manifest.step.checked_sub(parent_step) != Some(orchestration.training_steps)
                {
                    bail!("orchestration training_steps does not match checkpoint progress");
                }
                let expected_iteration = match latest_orchestration {
                    Some(previous) => previous
                        .iteration
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("orchestration iteration overflows u64"))?,
                    None => 1,
                };
                if orchestration.iteration != expected_iteration
                    || latest_orchestration
                        .is_some_and(|previous| orchestration.timestamp <= previous.timestamp)
                {
                    bail!("orchestration chronology must be contiguous from one");
                }
                let evaluation_scheduled = recipe.evaluation_interval != 0
                    && orchestration.iteration % recipe.evaluation_interval == 0;
                if evaluation_scheduled != orchestration.evaluation_id.is_some() {
                    bail!("orchestration evaluation presence disagrees with its run recipe");
                }
                if let Some(evaluation_id) = &orchestration.evaluation_id {
                    if orchestration.evaluation_seed != Some(recipe.evaluation_seed) {
                        bail!("evaluation orchestration does not match its run recipe");
                    }
                    // Promotion/evaluation evidence is fully validated by the
                    // publisher before its CAS. The model-loading boundary does
                    // not download historical evaluation artifacts; it proves
                    // the selected evaluation head and checkpoint/run lineage.
                    if commit.evaluation_head_id.as_deref() != Some(evaluation_id.as_str()) {
                        bail!("evaluated RunCommit does not select its evaluation as head");
                    }
                } else {
                    if orchestration.evaluation_seed.is_some() {
                        bail!("non-evaluation orchestration cannot carry evaluation_seed");
                    }
                    if commit.champion.as_ref() != inherited_champion
                        || commit.evaluation_head_id.as_ref() != inherited_evaluation
                    {
                        bail!("non-evaluation orchestration changed evaluation state");
                    }
                }
                latest_orchestration = Some(orchestration);
            }
        }
        parent = Some(entry);
    }
    Ok(())
}

pub(crate) fn run_head_path(model_root: &Path) -> PathBuf {
    model_root
        .join("channels")
        .join(format!("{RUN_HEAD_CHANNEL}.json"))
}

pub(crate) fn manifest_path(model_root: &Path, checkpoint_id: &str) -> PathBuf {
    model_root
        .join("manifests")
        .join("sha256")
        .join(format!("{checkpoint_id}.json"))
}

pub(crate) fn run_commit_path(model_root: &Path, run_commit_id: &str) -> PathBuf {
    model_root
        .join("run-commits")
        .join("sha256")
        .join(format!("{run_commit_id}.json"))
}

pub(crate) fn onnx_blob_path(model_root: &Path, digest: &str) -> PathBuf {
    model_root
        .join("blobs")
        .join("sha256")
        .join(format!("{digest}.onnx"))
}

pub(crate) fn verify_blob_bytes(
    label: &str,
    bytes: &[u8],
    reference: &BlobReference,
) -> Result<()> {
    let actual_size = u64::try_from(bytes.len()).context("blob size exceeds u64")?;
    if actual_size != reference.size_bytes {
        bail!(
            "{label} blob size is {actual_size}, expected {}",
            reference.size_bytes
        );
    }
    let actual_digest = sha256_hex(bytes);
    if actual_digest != reference.sha256 {
        bail!(
            "{label} blob digest is {actual_digest}, expected {}",
            reference.sha256
        );
    }
    Ok(())
}

pub(crate) fn read_filesystem_head(model_root: &Path) -> Result<Option<RunHeadV2>> {
    let path = run_head_path(model_root);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(anyhow!("failed to read model run head {:?}: {error}", path)),
    };
    let head: RunHeadV2 = parse_canonical_json("current run head", &bytes)?;
    validate_run_head(&head)?;
    Ok(Some(head))
}

fn read_filesystem_manifest(
    model_root: &Path,
    checkpoint_id: &str,
    expected_contract: &ModelArtifactContract,
) -> Result<CheckpointManifestV1> {
    validate_digest("checkpoint_id", checkpoint_id)?;
    let path = manifest_path(model_root, checkpoint_id);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read checkpoint manifest {path:?}"))?;
    let actual_id = sha256_hex(&bytes);
    if actual_id != checkpoint_id {
        bail!("checkpoint manifest digest is {actual_id}, expected {checkpoint_id}");
    }
    let manifest: CheckpointManifestV1 = parse_canonical_json("checkpoint manifest", &bytes)?;
    validate_manifest(&manifest, expected_contract)?;
    Ok(manifest)
}

pub(crate) fn resolve_filesystem_run_commit_chain(
    model_root: &Path,
    head: &RunHeadV2,
    expected_contract: &ModelArtifactContract,
    environment_max_horizon: u32,
) -> Result<Vec<ResolvedRunCommit>> {
    let mut reversed = Vec::new();
    let mut seen = HashSet::new();
    let mut current_id = Some(head.run_commit_id.clone());
    while let Some(run_commit_id) = current_id {
        if !seen.insert(run_commit_id.clone()) {
            bail!("RunCommit lineage contains a cycle");
        }
        let path = run_commit_path(model_root, &run_commit_id);
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read immutable RunCommit {path:?}"))?;
        let commit = parse_run_commit(
            "RunCommit",
            &run_commit_id,
            &bytes,
            expected_contract,
            environment_max_horizon,
        )?;
        let manifest =
            read_filesystem_manifest(model_root, &commit.checkpoint_id, expected_contract)?;
        current_id = commit.parent_run_commit_id.clone();
        reversed.push(ResolvedRunCommit {
            run_commit_id,
            commit,
            manifest,
        });
    }
    reversed.reverse();
    validate_run_commit_chain(&reversed, head)?;
    Ok(reversed)
}

pub(crate) fn select_inference_checkpoint(
    chain: &[ResolvedRunCommit],
    selection: ModelSelection,
) -> Result<&ResolvedRunCommit> {
    let latest = chain
        .last()
        .ok_or_else(|| anyhow!("RunCommit lineage is empty"))?;
    let checkpoint_id = match selection {
        ModelSelection::Latest => latest.commit.checkpoint_id.as_str(),
        ModelSelection::ChampionOrLatest => latest
            .commit
            .champion
            .as_ref()
            .map_or(latest.commit.checkpoint_id.as_str(), |champion| {
                champion.checkpoint_id.as_str()
            }),
    };
    chain
        .iter()
        .find(|entry| entry.commit.checkpoint_id == checkpoint_id)
        .ok_or_else(|| anyhow!("selected inference checkpoint is absent from RunCommit lineage"))
}

pub(crate) fn resolve_filesystem_head(
    model_root: &Path,
    head: RunHeadV2,
    expected_contract: &ModelArtifactContract,
    environment_max_horizon: u32,
    selection: ModelSelection,
) -> Result<ResolvedCheckpoint> {
    let chain = resolve_filesystem_run_commit_chain(
        model_root,
        &head,
        expected_contract,
        environment_max_horizon,
    )?;
    let selected = select_inference_checkpoint(&chain, selection)?;
    let checkpoint_id = selected.commit.checkpoint_id.clone();
    let manifest = selected.manifest.clone();

    let model_path = onnx_blob_path(model_root, &manifest.onnx.sha256);
    let model_bytes = std::fs::read(&model_path)
        .with_context(|| format!("failed to read immutable ONNX blob {model_path:?}"))?;
    verify_blob_bytes("ONNX", &model_bytes, &manifest.onnx)?;

    Ok(ResolvedCheckpoint {
        checkpoint_id,
        run_commit_id: head.run_commit_id,
        manifest,
        model_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use algorithm_core::{resolve_algorithm, ALPHAZERO_BOARD_V1_ID};

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
}
