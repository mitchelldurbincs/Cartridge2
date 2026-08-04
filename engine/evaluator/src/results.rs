//! What an evaluation run reports.

use serde::{Deserialize, Serialize};

/// Aggregate result of an evaluation run.
///
/// Field names match the Python `EvalResults` dataclass exactly: the trainer
/// deserializes this JSON straight into it, and the orchestrator's eval record
/// and W&B metrics are built from those names.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EvalSummary {
    pub env_id: String,
    pub player1_name: String,
    pub player2_name: String,
    pub games_played: u32,
    pub player1_wins: u32,
    pub player2_wins: u32,
    pub draws: u32,
    pub player1_wins_as_first: u32,
    pub player1_wins_as_second: u32,
    pub player2_wins_as_first: u32,
    pub player2_wins_as_second: u32,
    pub avg_game_length: f64,
}

/// Aggregate result for the DQN single-agent return suite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DqnEvalSummary {
    pub env_id: String,
    pub player_name: String,
    pub episodes_played: u32,
    pub terminated_episodes: u32,
    pub truncated_episodes: u32,
    pub mean_return: f64,
    pub min_return: f64,
    pub max_return: f64,
    pub avg_episode_length: f64,
}

impl EvalSummary {
    /// Record one finished game.
    ///
    /// `winner` is from player 1's perspective: `Some(1)`, `Some(2)`, or
    /// `None` for a draw. `player1_first` is which seat player 1 held.
    pub fn record(&mut self, winner: Option<u8>, player1_first: bool) {
        self.games_played += 1;
        match winner {
            Some(1) => {
                self.player1_wins += 1;
                if player1_first {
                    self.player1_wins_as_first += 1;
                } else {
                    self.player1_wins_as_second += 1;
                }
            }
            Some(_) => {
                self.player2_wins += 1;
                // These fields name the seat player 2 actually held.
                if player1_first {
                    self.player2_wins_as_second += 1;
                } else {
                    self.player2_wins_as_first += 1;
                }
            }
            None => self.draws += 1,
        }
    }

    pub fn player1_win_rate(&self) -> f64 {
        self.rate(self.player1_wins)
    }

    pub fn draw_rate(&self) -> f64 {
        self.rate(self.draws)
    }

    fn rate(&self, count: u32) -> f64 {
        if self.games_played == 0 {
            0.0
        } else {
            f64::from(count) / f64::from(self.games_played)
        }
    }
}

/// One decision made during an evaluation game.
///
/// Written as JSONL when `--dump-positions` is set, so a Python-side perfect
/// solver can score the moves that were actually played by the engine. Before
/// this existed, solver eval replayed games against a hand-maintained Python
/// copy of the rules and cross-checked the solver against *that*, so a drift
/// between the two rulesets would have scored a game nobody was training on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PositionRecord {
    /// Index of the game within the run.
    pub game: u32,
    /// 0-based move number within the game.
    pub ply: u32,
    /// Seat to move: 1 or 2.
    pub player: u8,
    /// Which configured player moved: "p1" or "p2".
    pub by: String,
    /// The action chosen.
    pub action: u32,
    /// Board owners, row-major, before the action — the engine's own view,
    /// which is what a mirrored solver board must agree with.
    pub board: Vec<u8>,
    /// Legal actions in this position.
    pub legal: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_splits_each_players_wins_by_their_actual_seat() {
        let mut s = EvalSummary::default();
        s.record(Some(1), true);
        s.record(Some(1), false);
        s.record(Some(2), true);
        s.record(Some(2), false);
        s.record(None, false);

        assert_eq!(s.games_played, 5);
        assert_eq!(s.player1_wins, 2);
        assert_eq!(s.player1_wins_as_first, 1);
        assert_eq!(s.player1_wins_as_second, 1);
        assert_eq!(s.player2_wins, 2);
        assert_eq!(s.player2_wins_as_first, 1);
        assert_eq!(s.player2_wins_as_second, 1);
        assert_eq!(s.draws, 1);
    }

    #[test]
    fn rates_are_zero_on_an_empty_run_rather_than_nan() {
        let s = EvalSummary::default();
        assert_eq!(s.player1_win_rate(), 0.0);
        assert_eq!(s.draw_rate(), 0.0);
    }

    #[test]
    fn rates_divide_by_games_played() {
        let mut s = EvalSummary::default();
        s.record(Some(1), true);
        s.record(Some(2), true);
        s.record(None, true);
        s.record(None, true);

        assert!((s.player1_win_rate() - 0.25).abs() < 1e-9);
        assert!((s.draw_rate() - 0.5).abs() < 1e-9);
    }
}
