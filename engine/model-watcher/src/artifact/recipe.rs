use anyhow::{anyhow, bail, Context, Result};

use super::codec::{sha256_hex, validate_digest};
use super::types::RunRecipeV1;
use super::validation::{validate_finite, validate_rate};

pub(super) fn validate_run_recipe(
    recipe: &RunRecipeV1,
    config_sha256: &str,
    environment_max_horizon: u32,
) -> Result<()> {
    validate_recipe_identity(recipe, config_sha256)?;
    validate_recipe_schedule(recipe)?;
    validate_canonical_floats(recipe)?;
    validate_temperature_schedule(recipe, environment_max_horizon)?;
    validate_promotion(recipe)
}

fn validate_recipe_identity(recipe: &RunRecipeV1, config_sha256: &str) -> Result<()> {
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
    if recipe.num_actors > recipe.episodes_per_iteration {
        bail!("run recipe num_actors exceeds episodes_per_iteration");
    }
    if recipe.collector_seed_strategy != "system_entropy_v1" {
        bail!("unsupported run recipe collector_seed_strategy");
    }
    if recipe.replay_policy != "scoped_fresh_iteration_v1" {
        bail!("unsupported run recipe replay_policy");
    }
    Ok(())
}

fn validate_recipe_schedule(recipe: &RunRecipeV1) -> Result<()> {
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
    u64::from(recipe.evaluation_games.max(recipe.solver_games))
        .checked_sub(1)
        .and_then(|last_index| recipe.evaluation_seed.checked_add(last_index))
        .ok_or_else(|| anyhow!("run recipe evaluation seed schedule overflows u64"))?;
    Ok(())
}

fn validate_canonical_floats(recipe: &RunRecipeV1) -> Result<()> {
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
    Ok(())
}

fn validate_temperature_schedule(recipe: &RunRecipeV1, max_horizon: u32) -> Result<()> {
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
    if max_horizon == 0 {
        bail!("runtime environment max_horizon must be positive");
    }
    if recipe.temperature_move_threshold != 0 && recipe.temperature_move_threshold >= max_horizon {
        bail!("run recipe temperature threshold is unreachable for its environment");
    }
    Ok(())
}

fn validate_promotion(recipe: &RunRecipeV1) -> Result<()> {
    if !matches!(
        recipe.promotion_metric.as_str(),
        "win_rate" | "solver_optimal"
    ) {
        bail!("unsupported run recipe promotion_metric");
    }
    if recipe.promotion_metric == "solver_optimal" && recipe.solver_games == 0 {
        bail!("solver_optimal run recipe requires solver games");
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

pub(super) fn expected_collector_simulations(recipe: &RunRecipeV1, iteration: u64) -> Result<u32> {
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
