use anyhow::{anyhow, bail, Result};
use std::collections::HashSet;

use super::recipe::expected_collector_simulations;
use super::types::{OrchestrationCommitV1, ResolvedRunCommit, RunCommitV1, RunHeadV2};

pub(crate) fn validate_run_commit_chain(
    chain: &[ResolvedRunCommit],
    head: &RunHeadV2,
) -> Result<()> {
    validate_selected_head(chain, head)?;
    let mut parent: Option<&ResolvedRunCommit> = None;
    let mut latest_orchestration: Option<&OrchestrationCommitV1> = None;
    let mut lineage_checkpoints = HashSet::new();
    let mut collection_scopes = HashSet::new();
    for entry in chain {
        validate_checkpoint_edge(entry, parent, &mut lineage_checkpoints)?;
        validate_run_mode(&entry.commit, parent)?;
        validate_orchestration_edge(
            entry,
            parent,
            &mut latest_orchestration,
            &mut collection_scopes,
        )?;
        parent = Some(entry);
    }
    Ok(())
}

fn validate_selected_head(chain: &[ResolvedRunCommit], head: &RunHeadV2) -> Result<()> {
    let selected = chain
        .last()
        .ok_or_else(|| anyhow!("RunCommit lineage is empty"))?;
    if selected.run_commit_id != head.run_commit_id
        || selected.commit.checkpoint_id != head.checkpoint_id
        || selected.manifest.step != selected.commit.stats_snapshot.step
    {
        bail!("run head does not match its selected RunCommit/checkpoint binding");
    }
    Ok(())
}

fn validate_checkpoint_edge<'a>(
    entry: &'a ResolvedRunCommit,
    parent: Option<&ResolvedRunCommit>,
    lineage_checkpoints: &mut HashSet<&'a str>,
) -> Result<()> {
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
    match parent {
        None if commit.parent_run_commit_id.is_some()
            || manifest.parent_checkpoint_id.is_some() =>
        {
            bail!("the first RunCommit must start both lineages");
        }
        Some(parent_entry) => {
            if commit.parent_run_commit_id.as_deref() != Some(parent_entry.run_commit_id.as_str()) {
                bail!("RunCommit parent identity is inconsistent");
            }
            if commit.profile != parent_entry.commit.profile
                || commit.config_sha256 != parent_entry.commit.config_sha256
            {
                bail!("RunCommit profile/config changed within a run");
            }
            if commit.checkpoint_id == parent_entry.commit.checkpoint_id {
                bail!("RunCommit children must select a new checkpoint");
            }
            if manifest.parent_checkpoint_id.as_deref()
                != Some(parent_entry.commit.checkpoint_id.as_str())
                || manifest.step <= parent_entry.manifest.step
            {
                bail!("RunCommit checkpoint must be a strictly newer direct child");
            }
        }
        None => {}
    }
    Ok(())
}

fn validate_run_mode(commit: &RunCommitV1, parent: Option<&ResolvedRunCommit>) -> Result<()> {
    match parent {
        None if commit.run_recipe.is_some() != commit.orchestration.is_some() => {
            bail!("a root RunCommit must be either standalone or recipe-owned orchestration");
        }
        Some(parent_entry) if parent_entry.commit.run_recipe.is_none() => {
            if commit.run_recipe.is_some() || commit.orchestration.is_some() {
                bail!("standalone and synchronized run modes cannot be mixed");
            }
        }
        Some(parent_entry)
            if commit.run_recipe_id != parent_entry.commit.run_recipe_id
                || commit.run_recipe.is_none()
                || commit.orchestration.is_none() =>
        {
            bail!("recipe-owned RunCommits must preserve their mode and exact recipe");
        }
        _ => {}
    }
    Ok(())
}

fn validate_orchestration_edge<'a>(
    entry: &'a ResolvedRunCommit,
    parent: Option<&ResolvedRunCommit>,
    latest: &mut Option<&'a OrchestrationCommitV1>,
    collection_scopes: &mut HashSet<&'a str>,
) -> Result<()> {
    let commit = &entry.commit;
    let inherited_champion = parent.and_then(|value| value.commit.champion.as_ref());
    let inherited_evaluation = parent.and_then(|value| value.commit.evaluation_head_id.as_ref());
    let Some(orchestration) = &commit.orchestration else {
        if commit.champion.as_ref() != inherited_champion
            || commit.evaluation_head_id.as_ref() != inherited_evaluation
        {
            bail!("standalone RunCommit changed evaluation state");
        }
        return Ok(());
    };
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
    let expected_source = parent.map(|value| value.commit.checkpoint_id.as_str());
    if orchestration.source_checkpoint_id.as_deref() != expected_source {
        bail!("orchestration source_checkpoint_id must equal its parent checkpoint");
    }
    if !collection_scopes.insert(orchestration.collection_scope_id.as_str()) {
        bail!("orchestration collection_scope_id must be unique within the run");
    }
    let parent_step = parent.map_or(0, |value| value.manifest.step);
    if orchestration.training_steps == 0
        || entry.manifest.step.checked_sub(parent_step) != Some(orchestration.training_steps)
    {
        bail!("orchestration training_steps does not match checkpoint progress");
    }
    validate_orchestration_chronology(orchestration, *latest)?;
    validate_orchestration_evaluation(
        commit,
        orchestration,
        recipe,
        inherited_champion,
        inherited_evaluation,
    )?;
    *latest = Some(orchestration);
    Ok(())
}

fn validate_orchestration_chronology(
    current: &OrchestrationCommitV1,
    previous: Option<&OrchestrationCommitV1>,
) -> Result<()> {
    let expected_iteration = match previous {
        Some(value) => value
            .iteration
            .checked_add(1)
            .ok_or_else(|| anyhow!("orchestration iteration overflows u64"))?,
        None => 1,
    };
    if current.iteration != expected_iteration
        || previous.is_some_and(|value| current.timestamp <= value.timestamp)
    {
        bail!("orchestration chronology must be contiguous from one");
    }
    Ok(())
}

fn validate_orchestration_evaluation(
    commit: &RunCommitV1,
    orchestration: &OrchestrationCommitV1,
    recipe: &super::types::RunRecipeV1,
    inherited_champion: Option<&super::types::ChampionReferenceV1>,
    inherited_evaluation: Option<&String>,
) -> Result<()> {
    let scheduled = recipe.evaluation_interval != 0
        && orchestration.iteration % recipe.evaluation_interval == 0;
    if scheduled != orchestration.evaluation_id.is_some() {
        bail!("orchestration evaluation presence disagrees with its run recipe");
    }
    if let Some(evaluation_id) = &orchestration.evaluation_id {
        if orchestration.evaluation_seed != Some(recipe.evaluation_seed) {
            bail!("evaluation orchestration does not match its run recipe");
        }
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
    Ok(())
}
