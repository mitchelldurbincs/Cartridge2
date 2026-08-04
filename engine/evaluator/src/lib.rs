//! Evaluation games played through the engine.
//!
//! This is the trainer's scoreboard. It lives here, next to the rules, because
//! the alternative — a second implementation of every game in Python so the
//! trainer could play them itself — is a duplicate source of truth for game
//! rules that nothing keeps in sync, and it silently limited evaluation to
//! whichever games someone had reimplemented.
//!
//! Playing here also means evaluation can use MCTS. The Python evaluator could
//! only play the raw policy argmax, which understates a model: search is how
//! an AlphaZero system actually plays.

pub mod player;
pub mod results;

use algorithm_core::{resolve_algorithm, BuiltinAlgorithm};
use anyhow::{anyhow, Result};
use engine_core::board_profile::{BoardGameMetadata, BoardView};
use engine_core::{
    AgentId, AgentOutcome, Decision, EngineContext, EpisodeStatus, ErasedTimestep, Presentation,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use player::AlphaZeroPosition;

pub use player::{DqnPlayer, Player};
pub use results::{DqnEvalSummary, EvalSummary, PositionRecord};

/// Validate and canonicalize the play temperature stored by a model player.
///
/// Evaluation evidence records this as an exact f32. Signed zero has the same
/// play semantics as positive zero, so collapse it before it reaches either
/// direct-policy sampling or MCTS.
pub fn canonical_evaluation_temperature(temperature: f32) -> Result<f32> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(anyhow!(
            "evaluation temperature must be a finite nonnegative f32"
        ));
    }
    Ok(if temperature == 0.0 { 0.0 } else { temperature })
}

/// Require an explicit, nonempty game schedule whose per-game seeds fit u64.
pub fn validate_evaluation_schedule(games: u32, seed: u64) -> Result<()> {
    if games == 0 {
        return Err(anyhow!("evaluation games must be greater than zero"));
    }
    seed.checked_add(u64::from(games - 1))
        .ok_or_else(|| anyhow!("evaluation seed plus game index exceeds u64"))?;
    Ok(())
}

/// ONNX thread selection is explicit in the clean evaluator contract.
pub fn validate_onnx_intra_threads(intra_threads: usize) -> Result<()> {
    if intra_threads == 0 {
        return Err(anyhow!(
            "evaluation ONNX intra-op threads must be greater than zero"
        ));
    }
    Ok(())
}

/// Hard stop on game length.
///
/// Every bundled game terminates on its own (Generals adjudicates at a ply cap,
/// Othello ends on two passes), so reaching this means a rules bug. Erroring is
/// the point: an evaluation that hangs stalls the whole training loop with no
/// diagnostic, which is far worse than a failed iteration.
const MAX_PLIES: u32 = 10_000;

fn resolve_compatible_algorithm(algorithm_id: &str, env_id: &str) -> Result<BuiltinAlgorithm> {
    let algorithm = resolve_algorithm(algorithm_id)?;
    engine_games::register_all_environments();
    let context = EngineContext::new(env_id)
        .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?;
    algorithm.compatibility(&context).require_compatible()?;
    Ok(algorithm)
}

/// Validate an algorithm/environment pair before loading model players or
/// creating evaluation output files.
pub fn preflight_evaluation(algorithm_id: &str, env_id: &str) -> Result<BuiltinAlgorithm> {
    resolve_compatible_algorithm(algorithm_id, env_id)
}

/// Dispatch to the evaluation suite owned by `algorithm_id` after validating
/// that the environment satisfies its machine-checkable contract.
pub fn run_evaluation(
    algorithm_id: &str,
    env_id: &str,
    player1: &mut Player,
    player2: &mut Player,
    games: u32,
    seed: u64,
    positions: Option<&mut Vec<PositionRecord>>,
) -> Result<EvalSummary> {
    validate_evaluation_schedule(games, seed)?;
    let algorithm = resolve_compatible_algorithm(algorithm_id, env_id)?;

    match algorithm {
        BuiltinAlgorithm::AlphaZeroBoardV1 => {
            run_alphazero_evaluation(env_id, player1, player2, games, seed, positions)
        }
        BuiltinAlgorithm::DqnV1 => Err(anyhow!(
            "DQN uses the single-agent return suite, not the two-player evaluation API"
        )),
    }
}

/// Evaluate a DQN policy by episode return through the environment contract.
pub fn run_dqn_evaluation(
    algorithm_id: &str,
    env_id: &str,
    player: &mut DqnPlayer,
    episodes: u32,
    seed: u64,
) -> Result<DqnEvalSummary> {
    validate_evaluation_schedule(episodes, seed)?;
    let algorithm = resolve_compatible_algorithm(algorithm_id, env_id)?;
    if algorithm != BuiltinAlgorithm::DqnV1 {
        return Err(anyhow!(
            "Algorithm '{}' does not own the single-agent return suite",
            algorithm.descriptor().id
        ));
    }
    let mut ctx = EngineContext::new(env_id)
        .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?;
    let capabilities = ctx.capabilities();
    let max_horizon = capabilities
        .max_horizon
        .filter(|horizon| *horizon > 0)
        .ok_or_else(|| anyhow!("DQN evaluation requires a finite non-zero horizon"))?;
    let agents = capabilities
        .agents
        .fixed_agents()
        .ok_or_else(|| anyhow!("DQN evaluation requires one fixed agent"))?;
    let [agent] = agents else {
        return Err(anyhow!(
            "DQN evaluation requires one fixed agent, got {}",
            agents.len()
        ));
    };
    let action_count = match agent.action_space {
        engine_core::ActionSpace::Discrete { size } => usize::try_from(size)?,
        ref other => return Err(anyhow!("DQN requires discrete actions, got {other:?}")),
    };
    player.require_environment_profile(env_id, capabilities.contract_version)?;

    let mut total_return = 0.0f64;
    let mut min_return = f64::INFINITY;
    let mut max_return = f64::NEG_INFINITY;
    let mut total_steps = 0u64;
    let mut terminated_episodes = 0u32;
    let mut truncated_episodes = 0u32;

    for episode in 0..episodes {
        let episode_seed = seed + u64::from(episode);
        let mut rng = ChaCha20Rng::seed_from_u64(episode_seed);
        let reset = ctx.reset(episode_seed, &[])?;
        if reset.timestep.episode != EpisodeStatus::Running
            || reset.timestep.source != engine_core::TransitionSource::Reset
        {
            return Err(anyhow!(
                "DQN environment reset must produce a running reset timestep"
            ));
        }
        let mut state = reset.state;
        let mut timestep = reset.timestep;
        let mut episode_return = 0.0f64;

        for step in 0..max_horizon {
            let action = player.select_action(&timestep, action_count, &mut rng)?;
            let transition = ctx.step(&state, &action.to_le_bytes())?;
            let outcome = transition
                .timestep
                .outcomes
                .iter()
                .find(|outcome| outcome.agent_id == agent.id)
                .ok_or_else(|| anyhow!("DQN transition omitted its fixed-agent outcome"))?;
            if !outcome.reward.is_finite() {
                return Err(anyhow!("DQN evaluation observed a non-finite reward"));
            }
            episode_return += f64::from(outcome.reward);
            total_steps += 1;
            match transition.timestep.episode {
                EpisodeStatus::Running => {
                    if outcome.terminated || outcome.truncated {
                        return Err(anyhow!("running DQN timestep contains a completed outcome"));
                    }
                    state = transition.state;
                    timestep = transition.timestep;
                }
                EpisodeStatus::Terminated => {
                    if !outcome.terminated || outcome.truncated {
                        return Err(anyhow!(
                            "DQN termination flags disagree with episode status"
                        ));
                    }
                    terminated_episodes += 1;
                    break;
                }
                EpisodeStatus::Truncated => {
                    if outcome.terminated || !outcome.truncated {
                        return Err(anyhow!("DQN truncation flags disagree with episode status"));
                    }
                    truncated_episodes += 1;
                    break;
                }
            }
            if step + 1 == max_horizon {
                return Err(anyhow!(
                    "DQN environment remained running beyond its declared horizon"
                ));
            }
        }
        total_return += episode_return;
        min_return = min_return.min(episode_return);
        max_return = max_return.max(episode_return);
    }

    Ok(DqnEvalSummary {
        env_id: env_id.to_string(),
        player_name: player.name(),
        episodes_played: episodes,
        terminated_episodes,
        truncated_episodes,
        mean_return: total_return / f64::from(episodes),
        min_return,
        max_return,
        avg_episode_length: total_steps as f64 / f64::from(episodes),
    })
}

/// Play `games` matches between two players and summarize the result.
///
/// Seats alternate deterministically: player 1 takes seat 1 on even game
/// indices and seat 2 on odd game indices, controlling for first-player
/// advantage while preserving stable per-game seeds.
///
/// Each game is reset and played with `seed + game index`, so a run is
/// reproducible and two models face identical conditions.
fn run_alphazero_evaluation(
    env_id: &str,
    player1: &mut Player,
    player2: &mut Player,
    games: u32,
    seed: u64,
    mut positions: Option<&mut Vec<PositionRecord>>,
) -> Result<EvalSummary> {
    let mut ctx = EngineContext::new(env_id)
        .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?;
    let metadata = ctx.metadata();
    let board = metadata.require_board()?;
    let capabilities = ctx.capabilities();
    let env_contract_version = capabilities.contract_version;
    let action_count = match capabilities.action_space(AgentId(1)) {
        Some(engine_core::ActionSpace::Discrete { size }) => *size as usize,
        other => {
            return Err(anyhow!(
                "AlphaZero requires discrete actions, got {other:?}"
            ))
        }
    };
    player1.require_environment_profile(env_id, env_contract_version)?;
    player2.require_environment_profile(env_id, env_contract_version)?;

    let mut summary = EvalSummary {
        env_id: env_id.to_string(),
        player1_name: player1.name(),
        player2_name: player2.name(),
        ..Default::default()
    };

    let mut total_plies: u64 = 0;

    for game in 0..games {
        let player1_first = game % 2 == 0;
        let game_seed = seed.wrapping_add(u64::from(game));
        let mut rng = ChaCha20Rng::seed_from_u64(game_seed);

        let reset = ctx.reset(game_seed, &[])?;
        let mut state = reset.state;
        let mut timestep = reset.timestep;
        let mut ply = 0u32;

        let winner = loop {
            let view = require_board_presentation(&ctx, &state, board)?;
            match timestep.episode {
                EpisodeStatus::Running => {
                    if view.game_over() {
                        return Err(anyhow!(
                            "Environment '{env_id}' reports a running episode with a terminal board presentation"
                        ));
                    }
                    if ply >= MAX_PLIES {
                        return Err(anyhow!(
                            "Game {game} of '{env_id}' did not terminate within {MAX_PLIES} plies"
                        ));
                    }

                    let position = AlphaZeroPosition::new(&state, &timestep, action_count)?;
                    validate_running_outcomes(&timestep)?;
                    let seat = require_board_seat(position.agent_id)?;
                    if view.current_player != seat {
                        return Err(anyhow!(
                            "Board presentation says seat {} acts, timestep says agent {}",
                            view.current_player,
                            position.agent_id.0
                        ));
                    }

                    // Player 1 alternates seats by game index while remaining
                    // logical player 1 for result attribution.
                    let is_player1 = (position.agent_id == AgentId(1)) == player1_first;
                    let acting = if is_player1 {
                        &mut *player1
                    } else {
                        &mut *player2
                    };
                    let action = acting.select_action(&position, &mut rng)?;

                    if let Some(positions) = positions.as_deref_mut() {
                        positions.push(PositionRecord {
                            game,
                            ply,
                            player: seat,
                            by: if is_player1 { "p1" } else { "p2" }.to_string(),
                            action,
                            board: view.owners(),
                            legal: position.legal_mask.iter_ones().map(|i| i as u32).collect(),
                        });
                    }

                    let step = ctx.step(&state, &action.to_le_bytes())?;
                    state = step.state;
                    timestep = step.timestep;
                    ply += 1;
                }
                EpisodeStatus::Terminated => {
                    break terminal_winner(&timestep, &view)?;
                }
                EpisodeStatus::Truncated => {
                    return Err(anyhow!(
                        "Game {game} of '{env_id}' was truncated after {ply} plies; the AlphaZero evaluation suite requires a terminal winner or draw"
                    ));
                }
            }
        };

        total_plies += u64::from(ply);
        summary.record(winner_from_player1(winner, player1_first), player1_first);
    }

    summary.avg_game_length = if games == 0 {
        0.0
    } else {
        total_plies as f64 / f64::from(games)
    };

    Ok(summary)
}

fn require_board_presentation(
    ctx: &EngineContext,
    state: &[u8],
    board: &BoardGameMetadata,
) -> Result<BoardView> {
    let view = match ctx.presentation(state)? {
        Some(Presentation::Board { view }) => view,
        Some(Presentation::Custom { contract, .. }) => {
            return Err(anyhow!(
                "AlphaZero board evaluation requires a board presentation, got custom contract '{contract}'"
            ))
        }
        None => {
            return Err(anyhow!(
                "AlphaZero board evaluation requires the environment to expose a board presentation"
            ))
        }
    };
    let expected_cells = board.board_size()?;
    if view.cells.len() != expected_cells {
        return Err(anyhow!(
            "Board presentation has {} cells, metadata declares {}x{} ({expected_cells} cells)",
            view.cells.len(),
            board.width,
            board.height
        ));
    }
    Ok(view)
}

fn require_board_seat(agent_id: AgentId) -> Result<u8> {
    match agent_id {
        AgentId(1) => Ok(1),
        AgentId(2) => Ok(2),
        other => Err(anyhow!(
            "AlphaZero seat-balanced evaluation requires agent 1 or 2, got {}",
            other.0
        )),
    }
}

fn outcome_for(timestep: &ErasedTimestep, agent_id: AgentId) -> Result<&AgentOutcome> {
    let mut matching = timestep
        .outcomes
        .iter()
        .filter(|outcome| outcome.agent_id == agent_id);
    let outcome = matching
        .next()
        .ok_or_else(|| anyhow!("Timestep is missing outcome for agent {}", agent_id.0))?;
    if matching.next().is_some() {
        return Err(anyhow!(
            "Timestep contains duplicate outcomes for agent {}",
            agent_id.0
        ));
    }
    Ok(outcome)
}

fn validate_running_outcomes(timestep: &ErasedTimestep) -> Result<()> {
    if timestep.outcomes.len() != 2 {
        return Err(anyhow!(
            "AlphaZero requires one outcome for each of two seats, got {}",
            timestep.outcomes.len()
        ));
    }
    for agent_id in [AgentId(1), AgentId(2)] {
        let outcome = outcome_for(timestep, agent_id)?;
        if outcome.reward != 0.0 || outcome.terminated || outcome.truncated {
            return Err(anyhow!(
                "Running AlphaZero timestep has invalid outcome for agent {}: reward={}, terminated={}, truncated={}",
                agent_id.0,
                outcome.reward,
                outcome.terminated,
                outcome.truncated
            ));
        }
    }
    Ok(())
}

/// Determine the absolute winning seat from per-agent terminal outcomes and
/// require the board presentation to agree. The presentation remains a display
/// projection; rewards are the authoritative environment result.
fn terminal_winner(timestep: &ErasedTimestep, view: &BoardView) -> Result<u8> {
    if timestep.episode != EpisodeStatus::Terminated {
        return Err(anyhow!(
            "Expected a terminated timestep, got {:?}",
            timestep.episode
        ));
    }
    if timestep.decision != Decision::None {
        return Err(anyhow!(
            "Terminated AlphaZero timestep must have no next decision, got {:?}",
            timestep.decision
        ));
    }
    if timestep.outcomes.len() != 2 {
        return Err(anyhow!(
            "Terminated AlphaZero timestep requires two agent outcomes, got {}",
            timestep.outcomes.len()
        ));
    }

    let seat1 = outcome_for(timestep, AgentId(1))?;
    let seat2 = outcome_for(timestep, AgentId(2))?;
    for outcome in [seat1, seat2] {
        if !outcome.reward.is_finite()
            || !(-1.0..=1.0).contains(&outcome.reward)
            || !outcome.terminated
            || outcome.truncated
        {
            return Err(anyhow!(
                "Invalid terminal outcome for agent {}: reward={}, terminated={}, truncated={}",
                outcome.agent_id.0,
                outcome.reward,
                outcome.terminated,
                outcome.truncated
            ));
        }
    }
    if (seat1.reward + seat2.reward).abs() > f32::EPSILON {
        return Err(anyhow!(
            "AlphaZero terminal rewards must be zero-sum, got {} and {}",
            seat1.reward,
            seat2.reward
        ));
    }

    let winner = if (seat1.reward - seat2.reward).abs() <= f32::EPSILON {
        3
    } else if seat1.reward > seat2.reward {
        1
    } else {
        2
    };
    if !view.game_over() {
        return Err(anyhow!(
            "Terminated timestep has a non-terminal board presentation"
        ));
    }
    if view.winner != winner {
        return Err(anyhow!(
            "Terminal outcomes identify winner {winner}, board presentation reports {}",
            view.winner
        ));
    }
    Ok(winner)
}

/// Translate the engine's absolute winner byte (0=ongoing, 1/2=seat, 3=draw)
/// into player 1's perspective: `Some(1)` for a player-1 win, `Some(2)` for a
/// player-2 win, `None` for a draw.
fn winner_from_player1(winner: u8, player1_first: bool) -> Option<u8> {
    let player1_seat = if player1_first { 1 } else { 2 };
    match winner {
        1 | 2 if winner == player1_seat => Some(1),
        1 | 2 => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terminal_timestep(seat1_reward: f32, seat2_reward: f32) -> ErasedTimestep {
        ErasedTimestep {
            agents: vec![AgentId(1), AgentId(2)],
            observations: Vec::new(),
            outcomes: vec![
                AgentOutcome {
                    agent_id: AgentId(1),
                    reward: seat1_reward,
                    terminated: true,
                    truncated: false,
                },
                AgentOutcome {
                    agent_id: AgentId(2),
                    reward: seat2_reward,
                    terminated: true,
                    truncated: false,
                },
            ],
            decision: Decision::None,
            episode: EpisodeStatus::Terminated,
            source: engine_core::TransitionSource::Agents {
                agent_ids: vec![AgentId(2)],
            },
            info: Vec::new(),
        }
    }

    fn eval(env_id: &str, games: u32) -> EvalSummary {
        engine_games::register_all_environments();
        let mut p1 = Player::Random;
        let mut p2 = Player::Random;
        run_evaluation(
            algorithm_core::ALPHAZERO_BOARD_V1_ID,
            env_id,
            &mut p1,
            &mut p2,
            games,
            42,
            None,
        )
        .expect("random-vs-random evaluation")
    }

    fn dqn_eval(episodes: u32, seed: u64) -> DqnEvalSummary {
        engine_games::register_all_environments();
        let mut player = DqnPlayer::Random;
        run_dqn_evaluation(
            algorithm_core::DQN_V1_ID,
            "counter",
            &mut player,
            episodes,
            seed,
        )
        .expect("random counter evaluation")
    }

    #[test]
    fn winner_translation_follows_player_ones_seat() {
        assert_eq!(winner_from_player1(1, true), Some(1));
        assert_eq!(winner_from_player1(2, true), Some(2));
        assert_eq!(winner_from_player1(1, false), Some(2));
        assert_eq!(winner_from_player1(2, false), Some(1));
        assert_eq!(winner_from_player1(3, true), None);
        assert_eq!(winner_from_player1(0, true), None);
    }

    #[test]
    fn terminal_result_comes_from_per_agent_outcomes() {
        let seat1_win = BoardView::from_owners(&[1], 2, 1);
        let draw = BoardView::from_owners(&[0], 1, 3);

        assert_eq!(
            terminal_winner(&terminal_timestep(1.0, -1.0), &seat1_win).unwrap(),
            1
        );
        assert_eq!(
            terminal_winner(&terminal_timestep(0.0, 0.0), &draw).unwrap(),
            3
        );
    }

    #[test]
    fn terminal_result_rejects_presentation_disagreement() {
        let wrong_view = BoardView::from_owners(&[2], 1, 2);
        let error = terminal_winner(&terminal_timestep(1.0, -1.0), &wrong_view)
            .unwrap_err()
            .to_string();

        assert!(error.contains("outcomes identify winner 1"));
        assert!(error.contains("reports 2"));
    }

    #[test]
    fn every_game_is_played_to_a_conclusion_and_counted_once() {
        for env_id in ["tictactoe", "connect4", "othello", "generals_8x8"] {
            let summary = eval(env_id, 4);
            assert_eq!(summary.games_played, 4, "{env_id}");
            assert_eq!(
                summary.player1_wins + summary.player2_wins + summary.draws,
                4,
                "{env_id}: outcomes must partition the games"
            );
            assert!(
                summary.avg_game_length > 0.0,
                "{env_id}: games cannot be zero plies"
            );
        }
    }

    #[test]
    fn runs_are_reproducible_for_a_given_seed() {
        assert_eq!(eval("connect4", 6), eval("connect4", 6));
    }

    #[test]
    fn dqn_return_suite_runs_without_board_or_second_player_assumptions() {
        let summary = dqn_eval(12, 9);
        assert_eq!(summary.env_id, "counter");
        assert_eq!(summary.player_name, "Random");
        assert_eq!(summary.episodes_played, 12);
        assert_eq!(
            summary.terminated_episodes + summary.truncated_episodes,
            summary.episodes_played
        );
        assert!(summary.mean_return.is_finite());
        assert!(summary.min_return <= summary.mean_return);
        assert!(summary.mean_return <= summary.max_return);
        assert!(summary.avg_episode_length > 0.0);
        assert_eq!(summary, dqn_eval(12, 9));
    }

    #[test]
    fn wins_remain_attributed_to_logical_players_across_alternating_seats() {
        // TicTacToe is decisive enough from random play that both seat
        // assignments produce wins for logical player 1.
        let summary = eval("tictactoe", 20);
        assert_eq!(
            summary.player1_wins,
            summary.player1_wins_as_first + summary.player1_wins_as_second
        );
        assert!(
            summary.player1_wins_as_first > 0 && summary.player1_wins_as_second > 0,
            "player 1 should win from both seats over 20 random games: {summary:?}"
        );
    }

    #[test]
    fn dumped_positions_track_the_game_that_was_played() {
        engine_games::register_all_environments();
        let mut p1 = Player::Random;
        let mut p2 = Player::Random;
        let mut positions = Vec::new();
        let summary = run_evaluation(
            algorithm_core::ALPHAZERO_BOARD_V1_ID,
            "connect4",
            &mut p1,
            &mut p2,
            3,
            7,
            Some(&mut positions),
        )
        .expect("evaluation");

        assert!(!positions.is_empty());
        assert_eq!(
            positions.len() as f64,
            summary.avg_game_length * f64::from(summary.games_played),
            "one record per move played"
        );

        for record in &positions {
            assert_eq!(record.board.len(), 42, "connect4 board is 7x6");
            assert!(
                record.legal.contains(&record.action),
                "a dumped action must have been legal"
            );
            assert!(record.player == 1 || record.player == 2);
            assert!(record.by == "p1" || record.by == "p2");
            let player1_first = record.game % 2 == 0;
            let expected_player = if (record.player == 1) == player1_first {
                "p1"
            } else {
                "p2"
            };
            assert_eq!(record.by, expected_player);
        }

        // Plies restart per game and the first position of each is empty.
        let openers: Vec<&PositionRecord> = positions.iter().filter(|r| r.ply == 0).collect();
        assert_eq!(openers.len(), 3, "one opening position per game");
        assert_eq!(openers[0].game, 0);
        assert_eq!(openers[0].player, 1);
        assert_eq!(openers[0].by, "p1");
        assert_eq!(openers[1].game, 1);
        assert_eq!(openers[1].player, 1);
        assert_eq!(openers[1].by, "p2");
        assert_eq!(openers[2].game, 2);
        assert_eq!(openers[2].player, 1);
        assert_eq!(openers[2].by, "p1");
        for opener in openers {
            assert!(opener.board.iter().all(|&owner| owner == 0));
        }
    }

    #[test]
    fn empty_or_overflowing_evaluation_schedules_are_rejected() {
        engine_games::register_all_environments();
        let mut p1 = Player::Random;
        let mut p2 = Player::Random;
        let empty = run_evaluation(
            algorithm_core::ALPHAZERO_BOARD_V1_ID,
            "tictactoe",
            &mut p1,
            &mut p2,
            0,
            42,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(empty.contains("greater than zero"));

        let overflow = validate_evaluation_schedule(2, u64::MAX)
            .unwrap_err()
            .to_string();
        assert!(overflow.contains("exceeds u64"));
    }

    #[test]
    fn evaluator_numeric_settings_fail_closed_and_canonicalize_zero() {
        assert_eq!(canonical_evaluation_temperature(-0.0).unwrap().to_bits(), 0);
        assert_eq!(canonical_evaluation_temperature(0.25).unwrap(), 0.25);
        for invalid in [f32::NAN, f32::INFINITY, -0.25] {
            assert!(canonical_evaluation_temperature(invalid).is_err());
        }
        assert!(validate_onnx_intra_threads(0).is_err());
        assert!(validate_onnx_intra_threads(1).is_ok());
    }

    #[test]
    fn algorithm_dispatch_rejects_unknown_ids_before_playing() {
        engine_games::register_all_environments();
        let mut p1 = Player::Random;
        let mut p2 = Player::Random;
        let error = run_evaluation("ppo", "tictactoe", &mut p1, &mut p2, 1, 42, None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("ppo"));
        assert!(error.contains(algorithm_core::ALPHAZERO_BOARD_V1_ID));
    }

    #[test]
    fn preflight_rejects_unknown_environments() {
        let error =
            preflight_evaluation(algorithm_core::ALPHAZERO_BOARD_V1_ID, "missing_environment")
                .unwrap_err()
                .to_string();

        assert!(error.contains("missing_environment"));
        assert!(error.contains("not registered"));
    }
}
