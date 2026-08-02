//! Game registration for Cartridge engine
//!
//! This crate provides a single initialization point for registering all
//! available games with the engine-core registry.
//!
//! # Usage
//!
//! ```rust
//! use engine_games::register_all_games;
//!
//! // Call once at startup - safe to call multiple times
//! register_all_games();
//! ```

use std::sync::Once;

pub mod manifest;

static INIT: Once = Once::new();

/// Register all available games with the engine-core registry.
///
/// This function uses `std::sync::Once` to ensure registration only
/// happens once, even if called multiple times. Safe to call from
/// multiple threads.
///
/// Currently registers:
/// - TicTacToe (`"tictactoe"`)
/// - Connect 4 (`"connect4"`)
/// - Othello (`"othello"`)
/// - Generals (`"generals_8x8"`)
pub fn register_all_games() {
    INIT.call_once(|| {
        games_tictactoe::register_tictactoe();
        games_connect4::register_connect4();
        games_othello::register_othello();
        games_generals::register_generals();
    });
}

// Re-export individual registration functions for advanced use cases
pub use games_connect4::register_connect4;
pub use games_generals::register_generals;
pub use games_othello::register_othello;
pub use games_tictactoe::register_tictactoe;

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{is_registered, list_registered_games};

    #[test]
    fn test_register_all_games() {
        register_all_games();

        assert!(is_registered("tictactoe"));
        assert!(is_registered("connect4"));
        assert!(is_registered("othello"));
        assert!(is_registered("generals_8x8"));
    }

    #[test]
    fn test_register_all_games_idempotent() {
        register_all_games();
        register_all_games();
        register_all_games();

        let games = list_registered_games();
        let tictactoe_count = games.iter().filter(|g| *g == "tictactoe").count();
        let connect4_count = games.iter().filter(|g| *g == "connect4").count();
        let othello_count = games.iter().filter(|g| *g == "othello").count();

        assert_eq!(tictactoe_count, 1);
        assert_eq!(connect4_count, 1);
        assert_eq!(othello_count, 1);
    }

    /// Every game currently bundled conforms to the **AlphaZero spatial
    /// observation profile**:
    /// `[obs_channels * board_size][legal mask: num_actions][player one-hot: 2]`.
    ///
    /// Both halves matter downstream: the trainer reshapes the leading slice
    /// using `obs_channels`, and reads the player indicator at
    /// `legal_mask_offset + num_actions`. A game that declares an encoding
    /// inconsistent with its own obs_size would mistrain silently, so assert
    /// it here for every registered game rather than per-crate.
    ///
    /// **This is a profile the current trainer requires, not a property of
    /// `Game`.** It holds for every game in the registry today because every
    /// game today is a two-player perfect-information board game trained by
    /// AlphaZero. An environment that is not — a vector/scalar observation, a
    /// recurrent or fog-of-war encoding, a simultaneous-turn adapter, anything
    /// driven by PPO — will legitimately violate it, and the answer then is to
    /// scope this assertion to the games claiming the profile rather than to
    /// force the new environment into a board-shaped layout. Deliberately not
    /// building that opt-out yet: there is no second algorithm to design it
    /// against, and guessing the seam before one exists is how it ends up
    /// fitting neither.
    #[test]
    fn test_bundled_games_conform_to_the_alphazero_spatial_profile() {
        register_all_games();

        for env_id in list_registered_games() {
            let meta = engine_core::create_game(&env_id)
                .unwrap_or_else(|| panic!("{env_id} registered but not constructible"))
                .metadata();
            let board_size = meta.board_size();

            assert!(
                meta.obs_channels > 0,
                "{env_id}: obs_channels must be declared (got 0)"
            );
            assert!(board_size > 0, "{env_id}: board_size must be non-zero");
            assert_eq!(
                meta.legal_mask_offset,
                meta.obs_channels * board_size,
                "{env_id}: legal mask must start immediately after the board planes \
                 (obs_channels={} * board_size={})",
                meta.obs_channels,
                board_size,
            );
            assert_eq!(
                meta.obs_size,
                meta.legal_mask_offset + meta.num_actions + 2,
                "{env_id}: obs_size must be planes + legal mask + 2-element player one-hot",
            );
        }
    }

    /// Every bundled game must declare its turn order.
    ///
    /// The actor derives each transition's value-target sign from step-index
    /// parity, which is only valid when the acting player alternates on every
    /// recorded step. `alternating_turns` defaults to `false` precisely so a
    /// game that forgets trips the actor rather than inheriting a claim; this
    /// test makes "forgot to declare it" a build failure instead of a runtime
    /// one for the games we ship.
    ///
    /// A future non-alternating game is legitimate — it just cannot use the
    /// parity backfill, so it should land together with a per-transition
    /// acting-player record rather than by flipping this assertion.
    #[test]
    fn test_bundled_games_declare_alternating_turns() {
        register_all_games();

        for env_id in list_registered_games() {
            let meta = engine_core::create_game(&env_id)
                .unwrap_or_else(|| panic!("{env_id} registered but not constructible"))
                .metadata();

            assert!(
                meta.alternating_turns,
                "{env_id}: must declare .with_alternating_turns(...). Every game \
                 bundled today alternates; if a new one does not, it needs a \
                 value-target backfill that does not rely on step parity."
            );
        }
    }

    /// A terminal step's info bits must report the true winning seat.
    ///
    /// This is the invariant that makes `info_bits::outcome_from_info` safe.
    /// The legal-move mask shares the same `u64` starting at bit 0, so a game
    /// with more than 16 actions can overlap the winner field at bits 20-23
    /// *during play* — Othello (65 actions) documents exactly that on its own
    /// `compute_info_bits`.
    ///
    /// The requirement at a terminal step is precisely: **whatever mask the
    /// game packs must not reach the winner bits.** Games satisfy that in two
    /// different ways, and it is worth being exact about which, because the
    /// looser phrasing "terminal positions have no legal moves" is not true:
    ///
    /// - Othello zeroes its mask once `is_done()`, so nothing collides.
    /// - Generals always packs a zero mask (257 actions could never fit), so
    ///   its terminal observations may well still contain legal moves.
    ///
    /// Either way the decode is exact, and that is what this asserts.
    ///
    /// The seat is checked against an **independently derived** expectation —
    /// ply parity for who moved last, plus the sign of the terminal reward —
    /// rather than against the same bits being tested. Without that, a decoder
    /// hardcoded to return `Player1Win` for every decisive game would pass.
    #[test]
    fn test_terminal_info_bits_report_the_true_winner() {
        use engine_core::{EngineContext, GameOutcome};
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        register_all_games();

        for env_id in list_registered_games() {
            let mut ctx = EngineContext::new(&env_id).expect("registered game");
            let meta = ctx.metadata();

            // Enough seeds to reach terminals of every shape: wins for either
            // seat, and draws. Asserted below, not assumed -- a seed set that
            // only ever produced one seat's win would leave the interesting
            // half of the decode untested.
            let mut reached_terminal = 0;
            let mut saw: Vec<GameOutcome> = Vec::new();

            for seed in 0..24u64 {
                let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
                let reset = ctx.reset(seed, &[]).expect("reset");
                let mut state = reset.state;
                let mut obs = reset.obs;

                // Every bundled game declares alternating turns (asserted by
                // the test above), so the seat that moves at ply N is
                // determined by parity: player 1 on even plies.
                for ply in 0..meta.num_actions * 64 {
                    let legal = meta.extract_legal_moves(&obs);
                    let Some(&action) = legal.choose(&mut rng) else {
                        break;
                    };

                    let step = ctx
                        .step(&state, &(action as u32).to_le_bytes())
                        .expect("step");

                    if step.done {
                        let decoded =
                            engine_core::game_utils::info_bits::outcome_from_info(step.info);

                        // Derive the expected outcome WITHOUT touching the info
                        // bits: who moved last (ply parity) plus whether the
                        // reward says that mover won, lost, or drew. The reward
                        // is relative to the mover, so it identifies the seat
                        // only in combination with the parity.
                        let mover_was_player1 = ply % 2 == 0;
                        let expected = if step.reward == 0.0 {
                            GameOutcome::Draw
                        } else {
                            let mover_won = step.reward > 0.0;
                            if mover_won == mover_was_player1 {
                                GameOutcome::Player1Win
                            } else {
                                GameOutcome::Player2Win
                            }
                        };

                        assert_eq!(
                            decoded,
                            Some(expected),
                            "{env_id} seed {seed}: info bits decoded {decoded:?} \
                             but ply {ply} (mover = player {}) with terminal \
                             reward {} means {expected:?}. info=0x{:x} -- if the \
                             mask is reaching bits 20-23 this is where it shows.",
                            if mover_was_player1 { 1 } else { 2 },
                            step.reward,
                            step.info,
                        );

                        if !saw.contains(&expected) {
                            saw.push(expected);
                        }
                        reached_terminal += 1;
                        break;
                    }

                    state = step.state;
                    obs = step.obs;
                }
            }

            assert!(
                reached_terminal > 0,
                "{env_id}: no random playout reached a terminal state, so the \
                 invariant was never exercised"
            );

            // Both seats must actually have been observed winning, or the
            // seat-specific half of the decode is untested for this game.
            assert!(
                saw.contains(&GameOutcome::Player1Win) && saw.contains(&GameOutcome::Player2Win),
                "{env_id}: {reached_terminal} terminals reached but only saw \
                 {saw:?}; both seats must win at least once or a decoder \
                 hardcoded to one seat would pass this test"
            );
        }
    }
}
