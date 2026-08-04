use algorithm_core::ModelArtifactContract;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::value::RawValue;

use super::codec::{sha256_hex, validate_digest};
use super::recipe::validate_run_recipe;
use super::types::*;
use super::validation::{
    validate_orchestration, validate_run_commit_json_shape, validate_training_stats,
};

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

pub(crate) fn parse_run_commit(
    label: &str,
    run_commit_id: &str,
    bytes: &[u8],
    expected_contract: &ModelArtifactContract,
    environment_max_horizon: u32,
) -> Result<RunCommitV1> {
    validate_run_commit_bytes(label, run_commit_id, bytes)?;
    let wire: RunCommitWireV1 =
        serde_json::from_slice(bytes).with_context(|| format!("invalid {label} contract"))?;
    validate_run_commit_json_shape(
        bytes,
        wire.stats_snapshot.as_ref(),
        wire.run_recipe.as_deref(),
    )?;
    validate_wire_identity(&wire, expected_contract)?;
    let run_recipe = parse_recipe(&wire, environment_max_horizon)?;
    let stats_snapshot = parse_stats_snapshot(&wire)?;
    validate_evaluation_references(&wire)?;
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

fn validate_run_commit_bytes(label: &str, run_commit_id: &str, bytes: &[u8]) -> Result<()> {
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
    Ok(())
}

fn validate_wire_identity(
    wire: &RunCommitWireV1,
    expected_contract: &ModelArtifactContract,
) -> Result<()> {
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
    Ok(())
}

fn parse_recipe(
    wire: &RunCommitWireV1,
    environment_max_horizon: u32,
) -> Result<Option<RunRecipeV1>> {
    match (&wire.run_recipe_id, &wire.run_recipe) {
        (None, None) => Ok(None),
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
            Ok(Some(recipe))
        }
        _ => bail!("run_recipe and run_recipe_id must both be null or both be present"),
    }
}

fn parse_stats_snapshot(wire: &RunCommitWireV1) -> Result<StatsSnapshotV3> {
    let actual_stats_id = sha256_hex(wire.stats_snapshot.get().as_bytes());
    if actual_stats_id != wire.stats_id {
        bail!(
            "embedded stats snapshot SHA-256 is {actual_stats_id}, expected {}",
            wire.stats_id
        );
    }
    let stats: StatsSnapshotV3 = serde_json::from_str(wire.stats_snapshot.get())
        .context("invalid embedded stats snapshot contract")?;
    if stats.schema_version != STATS_SNAPSHOT_SCHEMA_VERSION {
        bail!(
            "unsupported stats snapshot schema {}, expected {}",
            stats.schema_version,
            STATS_SNAPSHOT_SCHEMA_VERSION
        );
    }
    if stats.profile != wire.profile
        || stats.config_sha256 != wire.config_sha256
        || stats.checkpoint_id != wire.checkpoint_id
    {
        bail!("embedded stats snapshot binding does not match RunCommit");
    }
    validate_digest("stats_snapshot.config_sha256", &stats.config_sha256)?;
    validate_digest("stats_snapshot.checkpoint_id", &stats.checkpoint_id)?;
    validate_training_stats(&stats)?;
    Ok(stats)
}

fn validate_evaluation_references(wire: &RunCommitWireV1) -> Result<()> {
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
    Ok(())
}
