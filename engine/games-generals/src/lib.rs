//! Generals (8x8, full-information, turn-based) for the Cartridge engine.
//!
//! A port of the Go engine in `GeneralsReinforcementLearning/internal/game/`
//! reduced to a ruleset that is sound for AlphaZero-style training:
//!
//! - **Full information**: no fog of war. A fog variant is a separate,
//!   future env (vanilla MCTS re-simulates from the true state and would be
//!   omniscient under fog; that phase needs IS-MCTS or a recurrent
//!   on-policy method instead).
//! - **Strictly alternating turns**: player 1 moves and the move resolves
//!   immediately, then player 2. One full round = both plies; production
//!   and the draw clock tick at the end of each round. This deliberately
//!   replaces real Generals' simultaneous ticks — a pending-action bridge
//!   would leak the first mover's choice into the searcher's true state.
//! - **No half-moves**: every move sends `army - 1`. This halves the action
//!   space (4 directions per tile + wait = 257 actions).
//!
//! Illegal actions submitted to `step()` are rejected without mutating the
//! state. The observation's legal mask is the exact accepted action set.
//!
//! Board mechanics preserved from the Go engine: combat (larger army wins,
//! difference remains), city/general production of 1 per round, normal-tile
//! growth every 25 rounds, and general capture eliminating the player and
//! transferring all their tiles (armies and tile kinds unchanged).
//!
//! Departure from real Generals: at the round cap ([`params::MAX_TURNS`])
//! the game is adjudicated by territory (tiles, then armies) instead of
//! drawn. A pure draw cap collapsed self-play into 100% thousand-ply draws
//! (zero value signal); territory adjudication keeps the game zero-sum
//! while making almost every game decisive.

use engine_core::board_profile::{
    calculate_reward, decode_action_u32, opponent, register_board_game, BoardGame,
    BoardGameMetadata, BoardPlayerMetadata, BoardRenderer, BoardTransition, BoardView, CellKind,
    CellView,
};
use engine_core::typed::{
    ActionSpace, AgentId, AgentModel, Capabilities, DecodeError, EncodeError, Encoding, EngineId,
    EnvironmentSemantics, TensorSpec,
};
use engine_core::{EnvironmentError, EnvironmentMetadata, LegalMask};
use rand_chacha::ChaCha20Rng;

pub mod action;
pub mod board;
pub mod mapgen;
pub mod movement;
pub mod obs;
pub mod params;
pub mod rules;

use action::{decode_move, valid_move_target, Move};
use board::{Tile, TileKind};
use movement::{apply_move, transfer_tiles};
use obs::GeneralsObs;
use params::{BOARD_SIZE, MAX_TURNS, NUM_ACTIONS};
use rules::{adjudicate_at_cap, apply_production, check_winner};

/// Immutable environment contract revision for wire formats and semantics.
pub const ENV_CONTRACT_VERSION: u32 = 3;

/// Sentinel for an eliminated player's general index.
const NO_GENERAL: u8 = u8::MAX;

/// Register Generals with the global game registry as `generals_8x8`.
pub fn register_generals() {
    register_board_game::<Generals>()
        .expect("generals_8x8 environment must only be registered once");
}

/// Complete game state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Row-major tiles, length [`BOARD_SIZE`].
    pub tiles: Vec<Tile>,
    /// Completed rounds (both players moved).
    pub round: u32,
    /// Player to act: 1 or 2.
    pub current_player: u8,
    /// Which players are alive (index 0 = player 1).
    pub alive: [bool; 2],
    /// General tile index per player; [`NO_GENERAL`] once eliminated.
    pub generals: [u8; 2],
    /// 0 = ongoing, 1/2 = winner, 3 = draw.
    pub winner: u8,
    /// Ply count at which the game is adjudicated: `2 * MAX_TURNS` or one
    /// less, coin-flipped at reset so each seat gets the final move in
    /// half of all games. The authoritative observation exposes the exact
    /// remaining-ply countdown so this sampled rule state is Markov-visible.
    pub cap_plies: u16,
}

impl State {
    pub fn is_done(&self) -> bool {
        self.winner != 0
    }
}

/// Generals game implementation.
#[derive(Debug, Default)]
pub struct Generals;

impl Generals {
    pub fn new() -> Self {
        Self
    }

    fn observation(state: &State) -> GeneralsObs {
        GeneralsObs::from_state(state)
    }
}

impl BoardGame for Generals {
    type State = State;
    type Action = u32;
    type Observation = GeneralsObs;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "generals_8x8".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: ENV_CONTRACT_VERSION,
            encoding: Encoding::discrete_u32_le(
                "generals_state:v1",
                TensorSpec::f32_fixed([
                    ("channel", obs::NUM_CHANNELS as u32),
                    ("row", params::HEIGHT as u32),
                    ("column", params::WIDTH as u32),
                ]),
            ),
            semantics:
                EnvironmentSemantics::deterministic_alternating_perfect_information_terminal_zero_sum(),
            // The sampled cap is either this exact bound or one ply earlier.
            max_horizon: Some(MAX_TURNS * 2),
            agents: AgentModel::fixed_homogeneous_masked(
                [AgentId(1), AgentId(2)],
                ActionSpace::discrete(NUM_ACTIONS as u32),
            ),
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("generals_8x8", "Generals 8×8")
            .with_description(
                "Capture the enemy general! Full-information turn-based Generals: \
                 move armies, take cities, and grow your territory.",
            )
            .with_board(
                BoardGameMetadata::new(params::WIDTH, params::HEIGHT)
                    .with_players(vec![
                        BoardPlayerMetadata::new("Red", "R"),
                        BoardPlayerMetadata::new("Blue", "B"),
                    ])
                    .with_renderer(BoardRenderer::Generals),
            )
    }

    fn reset(
        &mut self,
        rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Self::Observation), EnvironmentError> {
        use rand::Rng;
        let map = mapgen::generate_map(rng);
        let state = State {
            tiles: map.tiles,
            round: 0,
            current_player: 1,
            alive: [true, true],
            generals: [map.generals[0] as u8, map.generals[1] as u8],
            winner: 0,
            cap_plies: (MAX_TURNS * 2 - rng.gen_range(0..=1)) as u16,
        };
        let obs = Self::observation(&state);
        Ok((state, obs))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<BoardTransition<Self::Observation>, EnvironmentError> {
        let previous_player = state.current_player;

        if state.is_done()
            || !state.alive[previous_player as usize - 1]
            || !rules::is_action_legal(&state.tiles, previous_player, action)
        {
            return Err(EnvironmentError::InvalidAction(format!(
                "action {action} is not legal for player {previous_player} in the current Generals state"
            )));
        }

        if !state.is_done() {
            // Resolve the already-validated ply.
            if let Some(Move::Step { from, dir }) = decode_move(action) {
                if let Some(to) = valid_move_target(&state.tiles, previous_player, from, dir) {
                    if let Some(capture) = apply_move(&mut state.tiles, previous_player, from, to) {
                        if capture.kind == TileKind::General && capture.previous_owner != 0 {
                            let loser = capture.previous_owner;
                            transfer_tiles(&mut state.tiles, loser, previous_player);
                            state.alive[loser as usize - 1] = false;
                            state.generals[loser as usize - 1] = NO_GENERAL;
                        }
                    }
                }
            }

            state.winner = check_winner(state.alive);

            // End of round after player 2's ply: production.
            if state.winner == 0 && previous_player == 2 {
                state.round += 1;
                apply_production(&mut state.tiles, state.round);
            }

            // Territory adjudication at the (parity-randomized) ply cap.
            if state.winner == 0 {
                let plies_played = state.round * 2 + if previous_player == 1 { 1 } else { 0 };
                if plies_played >= state.cap_plies as u32 {
                    state.winner = adjudicate_at_cap(&state.tiles);
                }
            }

            if state.winner == 0 {
                state.current_player = opponent(previous_player);
            }
        }

        let obs = Self::observation(state);
        let reward = calculate_reward(state.winner, previous_player);
        let done = state.is_done();
        Ok(BoardTransition {
            observation: obs,
            actor_reward: reward,
            terminated: done,
        })
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        // Header: round u32 | current_player | winner | alive x2 | generals x2
        //         | cap_plies u16
        out.extend_from_slice(&state.round.to_le_bytes());
        out.push(state.current_player);
        out.push(state.winner);
        out.push(state.alive[0] as u8);
        out.push(state.alive[1] as u8);
        out.push(state.generals[0]);
        out.push(state.generals[1]);
        out.extend_from_slice(&state.cap_plies.to_le_bytes());
        // Tiles: owner | kind | army u32
        for tile in &state.tiles {
            out.push(tile.owner);
            out.push(tile.kind as u8);
            out.extend_from_slice(&tile.army.to_le_bytes());
        }
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        const HEADER: usize = 4 + 2 + 2 + 2 + 2;
        const TILE: usize = 6;
        let expected = HEADER + BOARD_SIZE * TILE;
        if buf.len() != expected {
            return Err(DecodeError::InvalidLength {
                expected,
                actual: buf.len(),
            });
        }

        let round = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let current_player = buf[4];
        let winner = buf[5];
        let alive = [buf[6] != 0, buf[7] != 0];
        let generals = [buf[8], buf[9]];
        let cap_plies = u16::from_le_bytes(buf[10..12].try_into().unwrap());

        if current_player != 1 && current_player != 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid current_player: {}",
                current_player
            )));
        }
        if winner > 3 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid winner: {}",
                winner
            )));
        }
        if cap_plies == 0 || cap_plies as u32 > MAX_TURNS * 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid cap_plies: {}",
                cap_plies
            )));
        }

        let mut tiles = Vec::with_capacity(BOARD_SIZE);
        for i in 0..BOARD_SIZE {
            let at = HEADER + i * TILE;
            let owner = buf[at];
            if owner > 2 {
                return Err(DecodeError::CorruptedData(format!(
                    "Invalid tile owner: {}",
                    owner
                )));
            }
            let kind = TileKind::from_u8(buf[at + 1]).ok_or_else(|| {
                DecodeError::CorruptedData(format!("Invalid tile kind: {}", buf[at + 1]))
            })?;
            let army = u32::from_le_bytes(buf[at + 2..at + 6].try_into().unwrap());
            tiles.push(Tile { owner, army, kind });
        }

        Ok(State {
            tiles,
            round,
            current_player,
            alive,
            generals,
            winner,
            cap_plies,
        })
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        if *action >= NUM_ACTIONS as u32 {
            return Err(EncodeError::InvalidData(format!(
                "Invalid action: {}. Must be 0-{}",
                action,
                NUM_ACTIONS - 1
            )));
        }
        out.extend_from_slice(&action.to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        let action = decode_action_u32(buf)?;
        if action >= NUM_ACTIONS as u32 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid action: {}. Must be 0-{}",
                action,
                NUM_ACTIONS - 1
            )));
        }
        Ok(action)
    }

    fn encode_observation(obs: &Self::Observation, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }

    fn legal_actions(state: &Self::State) -> Result<LegalMask, EnvironmentError> {
        let mut mask = LegalMask::new(NUM_ACTIONS);
        if state.is_done() || !state.alive[state.current_player as usize - 1] {
            return Ok(mask);
        }
        for action in 0..NUM_ACTIONS as u32 {
            if rules::is_action_legal(&state.tiles, state.current_player, action) {
                mask.set(action as usize);
            }
        }
        Ok(mask)
    }

    fn view(state: &Self::State) -> BoardView {
        BoardView {
            cells: state
                .tiles
                .iter()
                .map(|tile| CellView {
                    owner: tile.owner,
                    kind: match tile.kind {
                        TileKind::Normal => CellKind::Normal,
                        TileKind::General => CellKind::General,
                        TileKind::City => CellKind::City,
                        TileKind::Mountain => CellKind::Mountain,
                    },
                    value: tile.army,
                })
                .collect(),
            current_player: state.current_player,
            winner: state.winner,
        }
    }
}

#[cfg(test)]
mod tests;
