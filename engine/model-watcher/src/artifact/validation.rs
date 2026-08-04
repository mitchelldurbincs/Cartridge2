use anyhow::{anyhow, bail, Context, Result};
use serde_json::value::RawValue;
use std::collections::BTreeMap;

use super::codec::validate_digest;
use super::types::*;

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
    "metrics",
    "learning_rate",
    "samples_seen",
    "replay_record_count",
    "last_checkpoint",
    "timestamp",
    "history",
    "env_id",
    "last_evaluation",
    "evaluation_history",
];
const HISTORY_FIELDS: &[&str] = &["step", "metrics", "learning_rate", "grad_norm"];
const EVALUATION_STATS_FIELDS: &[&str] = &[
    "step",
    "metrics",
    "episodes",
    "mean_episode_length",
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
    if !stats["last_evaluation"].is_null() {
        validate_exact_fields(
            "stats.last_evaluation",
            &stats["last_evaluation"],
            EVALUATION_STATS_FIELDS,
        )?;
    }
    let evaluations = stats["evaluation_history"]
        .as_array()
        .ok_or_else(|| anyhow!("stats.evaluation_history must be an array"))?;
    for (index, evaluation) in evaluations.iter().enumerate() {
        validate_exact_fields(
            &format!("stats.evaluation_history[{index}]"),
            evaluation,
            EVALUATION_STATS_FIELDS,
        )?;
    }
    Ok(())
}

pub(super) fn validate_run_commit_json_shape(
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

pub(super) fn validate_finite(label: &str, value: f64, nonnegative: bool) -> Result<()> {
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

pub(super) fn validate_rate(label: &str, value: f64) -> Result<()> {
    validate_finite(label, value, true)?;
    if value > 1.0 {
        bail!("{label} must be between zero and one");
    }
    Ok(())
}

pub(super) fn validate_utc_timestamp(label: &str, value: &str) -> Result<()> {
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

fn validate_metrics(label: &str, metrics: &BTreeMap<String, f64>) -> Result<()> {
    for (name, value) in metrics {
        if name.is_empty() || name.trim() != name {
            bail!("{label} names must be nonempty trimmed strings");
        }
        validate_finite(&format!("{label}.{name}"), *value, false)?;
    }
    Ok(())
}

fn validate_evaluation_stats(label: &str, stats: &EvaluationStatsV1) -> Result<()> {
    validate_metrics(&format!("{label}.metrics"), &stats.metrics)?;
    validate_finite(
        &format!("{label}.mean_episode_length"),
        stats.mean_episode_length,
        true,
    )?;
    validate_finite(&format!("{label}.timestamp"), stats.timestamp, true)?;
    if stats.episodes == 0 {
        if !stats.metrics.is_empty() || stats.mean_episode_length != 0.0 {
            bail!("{label} with zero episodes must contain no metrics and zero length");
        }
    } else if stats.mean_episode_length <= 0.0 {
        bail!("{label} mean episode length must be positive");
    }
    Ok(())
}

pub(super) fn validate_training_stats(snapshot: &StatsSnapshotV3) -> Result<()> {
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
    validate_metrics("stats.metrics", &stats.metrics)?;
    for (field, value) in [
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
        validate_metrics(&format!("stats.history[{index}].metrics"), &entry.metrics)?;
        validate_finite(
            &format!("stats.history[{index}].learning_rate"),
            entry.learning_rate,
            true,
        )?;
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
    for (index, evaluation) in stats.evaluation_history.iter().enumerate() {
        validate_evaluation_stats(&format!("stats.evaluation_history[{index}]"), evaluation)?;
        if evaluation.step > stats.step
            || previous_eval_step.is_some_and(|previous| evaluation.step <= previous)
            || previous_eval_timestamp.is_some_and(|previous| evaluation.timestamp < previous)
        {
            bail!("stats.evaluation_history chronology is invalid");
        }
        previous_eval_step = Some(evaluation.step);
        previous_eval_timestamp = Some(evaluation.timestamp);
    }
    match (&stats.last_evaluation, stats.evaluation_history.last()) {
        (None, None) => {}
        (Some(last), Some(expected)) if last == expected => {
            validate_evaluation_stats("stats.last_evaluation", last)?;
        }
        _ => bail!("stats.last_evaluation must equal the final evaluation_history record"),
    }
    Ok(())
}

pub(super) fn validate_orchestration(orchestration: &OrchestrationCommitV1) -> Result<()> {
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
