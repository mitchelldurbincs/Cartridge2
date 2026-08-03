//! Bundled environment registration for the Cartridge engine.
//!
//! This crate provides a single initialization point for registering all
//! available environments with the engine-core registry.
//!
//! # Usage
//!
//! ```rust
//! use engine_games::register_all_environments;
//!
//! // Call once at startup - safe to call multiple times
//! register_all_environments();
//! ```

use std::sync::Once;

pub mod manifest;

static INIT: Once = Once::new();

/// Register all bundled environments with the engine-core registry.
///
/// This function uses `std::sync::Once` to ensure registration only
/// happens once, even if called multiple times. Safe to call from
/// multiple threads.
///
/// Currently registers:
/// - Counter (`"counter"`, generic single-agent reference)
/// - TicTacToe (`"tictactoe"`)
/// - Connect 4 (`"connect4"`)
/// - Othello (`"othello"`)
/// - Generals (`"generals_8x8"`)
pub fn register_all_environments() {
    INIT.call_once(|| {
        envs_counter::register_counter();
        games_tictactoe::register_tictactoe();
        games_connect4::register_connect4();
        games_othello::register_othello();
        games_generals::register_generals();
    });
}

// Re-export individual registration functions for advanced use cases
pub use envs_counter::register_counter;
pub use games_connect4::register_connect4;
pub use games_generals::register_generals;
pub use games_othello::register_othello;
pub use games_tictactoe::register_tictactoe;

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{
        is_registered, list_registered_environments, Decision, EpisodeStatus, Presentation,
        TransitionSource,
    };

    #[test]
    fn test_register_all_environments() {
        register_all_environments();

        assert!(is_registered("counter"));
        assert!(is_registered("tictactoe"));
        assert!(is_registered("connect4"));
        assert!(is_registered("othello"));
        assert!(is_registered("generals_8x8"));
    }

    #[test]
    fn test_register_all_environments_idempotent() {
        register_all_environments();
        register_all_environments();
        register_all_environments();

        let environments = list_registered_environments();
        let tictactoe_count = environments.iter().filter(|id| *id == "tictactoe").count();
        let connect4_count = environments.iter().filter(|id| *id == "connect4").count();
        let othello_count = environments.iter().filter(|id| *id == "othello").count();

        assert_eq!(tictactoe_count, 1);
        assert_eq!(connect4_count, 1);
        assert_eq!(othello_count, 1);
    }

    #[test]
    fn test_bundled_games_conform_to_the_alphazero_spatial_profile() {
        register_all_environments();
        let algorithm = algorithm_core::resolve_algorithm(algorithm_core::ALPHAZERO_BOARD_V1_ID)
            .expect("built-in AlphaZero profile");
        let mut checked = 0;

        for env_id in list_registered_environments() {
            let context = engine_core::EngineContext::new(&env_id).unwrap_or_else(|error| {
                panic!("{env_id} registered but not constructible: {error}")
            });
            if !algorithm.compatibility(&context).compatible {
                continue;
            }
            checked += 1;
            let metadata = context.metadata();
            let board = metadata.require_board().unwrap();
            let board_size = board.board_size().unwrap();

            assert!(
                board.observation.spatial_channels > 0,
                "{env_id}: obs_channels must be declared (got 0)"
            );
            assert!(board_size > 0, "{env_id}: board_size must be non-zero");
            assert_eq!(
                board.observation.legal_actions_offset,
                board.observation.spatial_channels * board_size,
                "{env_id}: legal mask must start immediately after the board planes \
                 (obs_channels={} * board_size={})",
                board.observation.spatial_channels,
                board_size,
            );
            assert_eq!(
                board.observation.elements,
                board.observation.legal_actions_offset + board.action_count + 2,
                "{env_id}: obs_size must be planes + legal mask + 2-element player one-hot",
            );
        }

        assert!(
            checked > 0,
            "at least one bundled game must exercise the AlphaZero profile"
        );
    }

    #[test]
    fn test_every_board_environment_projects_a_well_formed_view() {
        register_all_environments();
        let mut checked = 0;

        for env_id in list_registered_environments() {
            let mut ctx = engine_core::EngineContext::new(&env_id).unwrap_or_else(|error| {
                panic!("{env_id} registered but not constructible: {error}")
            });
            let metadata = ctx.metadata();
            let Some(board) = metadata.board else {
                continue;
            };
            checked += 1;
            let reset = ctx.reset(7, &[]).expect("reset");

            let view = match ctx.presentation(&reset.state).unwrap() {
                Some(Presentation::Board { view }) => view,
                other => panic!("{env_id}: expected board presentation, got {other:?}"),
            };

            assert_eq!(
                view.cells.len(),
                board.board_size().unwrap(),
                "{env_id}: view must have one cell per board position",
            );
            assert!(
                (1..=board.players.len() as u8).contains(&view.current_player),
                "{env_id}: current_player {} out of range",
                view.current_player,
            );
            assert_eq!(view.winner, 0, "{env_id}: a fresh game cannot be decided");
            for (i, cell) in view.cells.iter().enumerate() {
                assert!(
                    cell.owner as usize <= board.players.len(),
                    "{env_id}: cell {i} owner {} is not a player or neutral",
                    cell.owner,
                );
            }

            // A constant stub would satisfy everything above. Playing a legal
            // move must move the view.
            let active_agent = match &reset.timestep.decision {
                Decision::Agents { agent_ids } if agent_ids.len() == 1 => agent_ids[0],
                decision => panic!("{env_id}: expected one actor, got {decision:?}"),
            };
            let observation = reset
                .timestep
                .observation_for(active_agent)
                .expect("observation for actor");
            let action = board
                .extract_legal_moves(observation)
                .unwrap_or_else(|error| panic!("{env_id}: invalid fresh observation: {error}"))
                .first()
                .copied()
                .unwrap_or_else(|| panic!("{env_id}: fresh game has no legal move"));
            let step = ctx
                .step(&reset.state, &(action as u32).to_le_bytes())
                .expect("step");
            let after = match ctx.presentation(&step.state).unwrap() {
                Some(Presentation::Board { view }) => view,
                other => panic!("{env_id}: expected board presentation, got {other:?}"),
            };

            assert_ne!(
                view, after,
                "{env_id}: view did not change after a legal move — is it a stub?",
            );
        }

        assert!(
            checked > 0,
            "at least one board environment must be checked"
        );
    }

    #[test]
    fn alphazero_board_trajectories_honor_declared_behavior() {
        register_all_environments();
        let algorithm = algorithm_core::resolve_algorithm(algorithm_core::ALPHAZERO_BOARD_V1_ID)
            .expect("built-in AlphaZero profile");

        for env_id in list_registered_environments() {
            let mut first = engine_core::EngineContext::new(&env_id).unwrap();
            if !algorithm.compatibility(&first).compatible {
                continue;
            }
            let mut second = engine_core::EngineContext::new(&env_id).unwrap();
            let board = first.metadata().require_board().unwrap().clone();
            let left_reset = first.reset(19, &[]).unwrap();
            let right_reset = second.reset(19, &[]).unwrap();
            assert_eq!(
                left_reset, right_reset,
                "{env_id}: reset is not deterministic"
            );
            let mut left_state = left_reset.state;
            let mut left_timestep = left_reset.timestep;
            let mut right_state = right_reset.state;
            let max_horizon = first.capabilities().max_horizon.unwrap();
            let mut previous_actor = None;

            for _ in 0..=max_horizon {
                if left_timestep.episode.is_done() {
                    assert_eq!(left_timestep.episode, EpisodeStatus::Terminated);
                    assert!(left_timestep
                        .outcomes
                        .iter()
                        .all(|outcome| outcome.terminated));
                    let reward_sum: f32 = left_timestep
                        .outcomes
                        .iter()
                        .map(|outcome| outcome.reward)
                        .sum();
                    assert!(
                        reward_sum.abs() < 1e-6,
                        "{env_id}: terminal rewards not zero-sum"
                    );
                    break;
                }

                assert!(
                    left_timestep
                        .outcomes
                        .iter()
                        .all(|outcome| outcome.reward == 0.0),
                    "{env_id}: non-terminal reward is not zero"
                );
                let actor = match &left_timestep.decision {
                    Decision::Agents { agent_ids } if agent_ids.len() == 1 => agent_ids[0],
                    decision => panic!("{env_id}: unexpected decision {decision:?}"),
                };
                if let Some(previous) = previous_actor {
                    assert_ne!(actor, previous, "{env_id}: seats did not alternate");
                }
                let observation = left_timestep.observation_for(actor).unwrap();
                let action = board.extract_legal_moves(observation).unwrap()[0] as u32;
                let next_left = first.step(&left_state, &action.to_le_bytes()).unwrap();
                let next_right = second.step(&right_state, &action.to_le_bytes()).unwrap();
                assert_eq!(
                    next_left, next_right,
                    "{env_id}: transition is not deterministic"
                );
                assert_eq!(
                    next_left.timestep.source,
                    TransitionSource::Agents {
                        agent_ids: vec![actor]
                    },
                    "{env_id}: transition source does not identify the actor"
                );
                previous_actor = Some(actor);
                left_state = next_left.state;
                left_timestep = next_left.timestep;
                right_state = next_right.state;
            }

            assert!(
                left_timestep.episode.is_done(),
                "{env_id}: exceeded max_horizon"
            );
        }
    }
}
