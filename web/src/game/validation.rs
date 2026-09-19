//! AlphaZero board-serving validation at the engine/session boundary.
//!
//! These checks narrow the generic engine contract to the two-seat board
//! cartridge. Keep them here rather than imposing board rules on engine-core.

use anyhow::{anyhow, Result};
use engine_core::board_profile::{BoardGameMetadata, BoardView};
use engine_core::{
    ActionAvailability, AgentId, Decision, EngineContext, EpisodeStatus, ErasedTimestep, LegalMask,
    Presentation, TransitionSource,
};

/// Validate the complete transition before reading its presentation. Callers
/// commit state, revision, and history only after this boundary succeeds.
/// Validation order is intentional: it also determines the reported error.
pub(super) fn validate_position(
    ctx: &EngineContext,
    state: &[u8],
    timestep: &ErasedTimestep,
    board: &BoardGameMetadata,
    expected: ExpectedTransition,
) -> Result<BoardView> {
    validate_timestep(timestep, expected)?;
    let view = require_board_presentation(ctx.presentation(state)?)?;
    validate_board_view(view, timestep, board)
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ExpectedTransition {
    Reset,
    Agent(AgentId),
}

/// Narrow one generic timestep to the position shape this serving cartridge
/// understands. Chance and simultaneous decisions are rejected here rather
/// than being assigned a synthetic board player.
pub(super) fn active_observation(
    timestep: &ErasedTimestep,
) -> Result<(AgentId, &[u8], &LegalMask)> {
    if timestep.episode != EpisodeStatus::Running {
        return Err(anyhow!(
            "AlphaZero board action selection requires a running episode, got {:?}",
            timestep.episode
        ));
    }

    let decision = timestep.decision.sole_agent().ok_or_else(|| {
        anyhow!(
            "AlphaZero board serving requires exactly one active decision agent, got {:?}",
            timestep.decision
        )
    })?;
    let active_agent = decision.agent_id;
    if !matches!(active_agent, AgentId(1) | AgentId(2)) {
        return Err(anyhow!(
            "AlphaZero board serving only supports seats 1 and 2, got {}",
            active_agent.0
        ));
    }

    let observation = timestep.sole_observation()?;
    if observation.agent_id != active_agent {
        return Err(anyhow!(
            "sole observation belongs to agent {}, but active decision belongs to agent {}",
            observation.agent_id.0,
            active_agent.0
        ));
    }
    let ActionAvailability::DiscreteMask { mask } = &decision.availability else {
        anyhow::bail!(
            "AlphaZero board serving requires a discrete legal mask for agent {}",
            active_agent.0
        );
    };
    Ok((active_agent, observation.data.as_slice(), mask))
}

fn validate_two_player_outcomes(timestep: &ErasedTimestep) -> Result<()> {
    if timestep.outcomes.len() != 2 {
        return Err(anyhow!(
            "AlphaZero board timestep requires exactly two per-agent outcomes, got {}",
            timestep.outcomes.len()
        ));
    }

    for agent_id in [AgentId(1), AgentId(2)] {
        let matches = timestep
            .outcomes
            .iter()
            .filter(|outcome| outcome.agent_id == agent_id)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(anyhow!(
                "AlphaZero board timestep requires exactly one outcome for agent {}, got {}",
                agent_id.0,
                matches.len()
            ));
        }
        let outcome = matches[0];
        if !outcome.reward.is_finite() {
            return Err(anyhow!(
                "AlphaZero board timestep has a non-finite reward for agent {}",
                agent_id.0
            ));
        }
        let flags_match = match timestep.episode {
            EpisodeStatus::Running => !outcome.terminated && !outcome.truncated,
            EpisodeStatus::Terminated => outcome.terminated && !outcome.truncated,
            EpisodeStatus::Truncated => !outcome.terminated && outcome.truncated,
        };
        if !flags_match {
            return Err(anyhow!(
                "outcome flags for agent {} disagree with episode status {:?}",
                agent_id.0,
                timestep.episode
            ));
        }
    }

    let seat_one = timestep
        .reward_for(AgentId(1))
        .expect("validated outcome for seat 1");
    let seat_two = timestep
        .reward_for(AgentId(2))
        .expect("validated outcome for seat 2");
    match timestep.episode {
        EpisodeStatus::Running | EpisodeStatus::Truncated if seat_one != 0.0 || seat_two != 0.0 => {
            Err(anyhow!(
                "AlphaZero terminal-only reward contract emitted ({seat_one}, {seat_two}) for {:?}",
                timestep.episode
            ))
        }
        EpisodeStatus::Terminated if (seat_one + seat_two).abs() > 1e-6 => Err(anyhow!(
            "AlphaZero terminal rewards must be zero-sum, got ({seat_one}, {seat_two})"
        )),
        _ => Ok(()),
    }
}

fn validate_timestep(timestep: &ErasedTimestep, expected: ExpectedTransition) -> Result<()> {
    match (expected, &timestep.source) {
        (ExpectedTransition::Reset, TransitionSource::Reset) => {}
        (ExpectedTransition::Agent(expected_actor), TransitionSource::Agents { agent_ids })
            if agent_ids.as_slice() == [expected_actor] => {}
        (ExpectedTransition::Reset, source) => {
            return Err(anyhow!(
                "reset timestep has invalid transition source {source:?}"
            ))
        }
        (ExpectedTransition::Agent(expected_actor), source) => {
            return Err(anyhow!(
                "step by agent {} has invalid transition source {source:?}",
                expected_actor.0
            ))
        }
    }

    validate_two_player_outcomes(timestep)?;
    match timestep.episode {
        EpisodeStatus::Running => {
            let (next_agent, _, _) = active_observation(timestep)?;
            if let ExpectedTransition::Agent(actor) = expected {
                if next_agent == actor {
                    return Err(anyhow!(
                        "alternating-turn board transition kept agent {} active",
                        actor.0
                    ));
                }
            }
        }
        EpisodeStatus::Terminated | EpisodeStatus::Truncated => {
            if timestep.decision != Decision::None {
                return Err(anyhow!(
                    "completed board timestep must have no next decision, got {:?}",
                    timestep.decision
                ));
            }
            let observation = timestep.sole_observation()?;
            if !matches!(observation.agent_id, AgentId(1) | AgentId(2)) {
                return Err(anyhow!(
                    "completed board observation belongs to unsupported agent {}",
                    observation.agent_id.0
                ));
            }
        }
    }
    Ok(())
}

fn validate_board_view(
    view: BoardView,
    timestep: &ErasedTimestep,
    board: &BoardGameMetadata,
) -> Result<BoardView> {
    let expected_cells = board.board_size()?;
    if view.cells.len() != expected_cells {
        return Err(anyhow!(
            "board presentation has {} cells, metadata declares {}x{} ({expected_cells} cells)",
            view.cells.len(),
            board.width,
            board.height
        ));
    }
    if !matches!(view.current_player, 1 | 2) {
        return Err(anyhow!(
            "board presentation current player must be seat 1 or 2, got {}",
            view.current_player
        ));
    }
    if view.cells.iter().any(|cell| cell.owner > 2) {
        return Err(anyhow!(
            "board presentation contains an owner outside seats 1 and 2"
        ));
    }
    let observation = timestep.sole_observation()?;
    if observation.agent_id.0 != u32::from(view.current_player) {
        return Err(anyhow!(
            "board presentation current player {} disagrees with sole observation agent {}",
            view.current_player,
            observation.agent_id.0
        ));
    }

    match timestep.episode {
        EpisodeStatus::Running => {
            if view.winner != 0 {
                return Err(anyhow!(
                    "running episode has terminal board winner {}",
                    view.winner
                ));
            }
            let (active_agent, _, _) = active_observation(timestep)?;
            if u32::from(view.current_player) != active_agent.0 {
                return Err(anyhow!(
                    "board presentation current player {} disagrees with active agent {}",
                    view.current_player,
                    active_agent.0
                ));
            }
        }
        EpisodeStatus::Terminated => {
            if !matches!(view.winner, 1..=3) {
                return Err(anyhow!(
                    "terminated episode requires winner 1, 2, or draw marker 3, got {}",
                    view.winner
                ));
            }
            let seat_one = timestep
                .reward_for(AgentId(1))
                .ok_or_else(|| anyhow!("terminal board timestep is missing seat 1 reward"))?;
            let seat_two = timestep
                .reward_for(AgentId(2))
                .ok_or_else(|| anyhow!("terminal board timestep is missing seat 2 reward"))?;
            let rewards_match_winner = match view.winner {
                1 => seat_one > 0.0 && seat_two < 0.0,
                2 => seat_one < 0.0 && seat_two > 0.0,
                3 => seat_one == 0.0 && seat_two == 0.0,
                _ => unreachable!("winner range validated above"),
            };
            if !rewards_match_winner {
                return Err(anyhow!(
                    "board winner {} disagrees with per-agent rewards ({seat_one}, {seat_two})",
                    view.winner
                ));
            }
        }
        EpisodeStatus::Truncated => {
            if view.winner != 0 {
                return Err(anyhow!(
                    "truncated episode must not fabricate a winner, got {}",
                    view.winner
                ));
            }
        }
    }

    Ok(view)
}

fn require_board_presentation(presentation: Option<Presentation>) -> Result<BoardView> {
    match presentation {
        Some(Presentation::Board { view }) => Ok(view),
        Some(Presentation::Custom { contract, .. }) => Err(anyhow!(
            "AlphaZero web serving requires a board presentation, got custom contract '{contract}'"
        )),
        None => Err(anyhow!(
            "AlphaZero web serving requires the environment to expose a board presentation"
        )),
    }
}

#[cfg(test)]
mod tests;
