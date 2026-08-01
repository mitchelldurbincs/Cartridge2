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

use anyhow::{anyhow, Result};
use engine_core::EngineContext;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub use player::Player;
pub use results::{EvalSummary, PositionRecord};

/// Hard stop on game length.
///
/// Every bundled game terminates on its own (Generals adjudicates at a ply cap,
/// Othello ends on two passes), so reaching this means a rules bug. Erroring is
/// the point: an evaluation that hangs stalls the whole training loop with no
/// diagnostic, which is far worse than a failed iteration.
const MAX_PLIES: u32 = 10_000;

/// Play `games` matches between two players and summarize the result.
///
/// Players alternate seats: player 1 takes the first seat for the first half of
/// the games and the second seat for the rest — the same split the Python
/// evaluator used, so numbers stay comparable across the migration.
///
/// Each game is reset and played with `seed + game index`, so a run is
/// reproducible and two models face identical conditions.
pub fn run_evaluation(
    env_id: &str,
    player1: &mut Player,
    player2: &mut Player,
    games: u32,
    seed: u64,
    mut positions: Option<&mut Vec<PositionRecord>>,
) -> Result<EvalSummary> {
    let mut ctx =
        EngineContext::new(env_id).ok_or_else(|| anyhow!("Game '{env_id}' not registered"))?;
    let metadata = ctx.metadata();

    let mut summary = EvalSummary {
        env_id: env_id.to_string(),
        player1_name: player1.name(),
        player2_name: player2.name(),
        ..Default::default()
    };

    let games_as_first = games / 2;
    let mut total_plies: u64 = 0;

    for game in 0..games {
        let player1_first = game < games_as_first;
        let game_seed = seed.wrapping_add(u64::from(game));
        let mut rng = ChaCha20Rng::seed_from_u64(game_seed);

        let reset = ctx.reset(game_seed, &[])?;
        let mut state = reset.state;
        let mut obs = reset.obs;
        let mut ply = 0u32;

        let winner = loop {
            let view = ctx.view(&state)?;
            if view.game_over() {
                break view.winner;
            }
            if ply >= MAX_PLIES {
                return Err(anyhow!(
                    "Game {game} of '{env_id}' did not terminate within {MAX_PLIES} plies"
                ));
            }

            // Player 1 acts on seat 1 when it went first, seat 2 otherwise.
            let seat = view.current_player;
            let is_player1 = (seat == 1) == player1_first;
            let mask = metadata.legal_mask_from_obs(&obs);
            let acting = if is_player1 {
                &mut *player1
            } else {
                &mut *player2
            };
            let action =
                acting.select_action(&state, &obs, &mask, metadata.num_actions, &mut rng)?;

            if let Some(positions) = positions.as_deref_mut() {
                positions.push(PositionRecord {
                    game,
                    ply,
                    player: seat,
                    by: if is_player1 { "p1" } else { "p2" }.to_string(),
                    action,
                    board: view.owners(),
                    legal: mask.iter_ones().map(|i| i as u32).collect(),
                });
            }

            let step = ctx.step(&state, &action.to_le_bytes())?;
            state = step.state;
            obs = step.obs;
            ply += 1;
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

    fn eval(env_id: &str, games: u32) -> EvalSummary {
        engine_games::register_all_games();
        let mut p1 = Player::Random;
        let mut p2 = Player::Random;
        run_evaluation(env_id, &mut p1, &mut p2, games, 42, None)
            .expect("random-vs-random evaluation")
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
    fn seats_alternate_at_the_halfway_point() {
        // TicTacToe is decisive enough from random play that both halves
        // produce wins; the point is that they are attributed to both seats.
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
        engine_games::register_all_games();
        let mut p1 = Player::Random;
        let mut p2 = Player::Random;
        let mut positions = Vec::new();
        let summary = run_evaluation("connect4", &mut p1, &mut p2, 2, 7, Some(&mut positions))
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
        }

        // Plies restart per game and the first position of each is empty.
        let openers: Vec<&PositionRecord> = positions.iter().filter(|r| r.ply == 0).collect();
        assert_eq!(openers.len(), 2, "one opening position per game");
        for opener in openers {
            assert!(opener.board.iter().all(|&owner| owner == 0));
        }
    }

    #[test]
    fn no_games_yields_an_empty_but_valid_summary() {
        let summary = eval("tictactoe", 0);
        assert_eq!(summary.games_played, 0);
        assert_eq!(summary.avg_game_length, 0.0);
        assert_eq!(summary.player1_win_rate(), 0.0);
    }
}
