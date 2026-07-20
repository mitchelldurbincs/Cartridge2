//! Tests for the Generals port: combat/capture semantics ported from the Go
//! suite (`core/movement_test.go`), action-mask correctness vs brute force
//! (`action_mask_test`), mapgen invariants (`mapgen/generator_test.go`),
//! encode/decode round-trips, and a full-game random smoke test.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use super::*;
use crate::action::{encode_move, move_target};
use crate::board::{idx, new_board};
use crate::params::{CITY_RATIO, CITY_START_ARMY, GENERAL_START_ARMY, HEIGHT, WAIT_ACTION, WIDTH};

fn rng(seed: u64) -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(seed)
}

/// A hand-built state: two generals in opposite corners, no terrain.
fn flat_state() -> State {
    let mut tiles = new_board();
    let g1 = idx(0, 0);
    let g2 = idx(WIDTH - 1, HEIGHT - 1);
    tiles[g1] = Tile {
        owner: 1,
        army: GENERAL_START_ARMY,
        kind: TileKind::General,
    };
    tiles[g2] = Tile {
        owner: 2,
        army: GENERAL_START_ARMY,
        kind: TileKind::General,
    };
    State {
        tiles,
        round: 0,
        current_player: 1,
        alive: [true, true],
        generals: [g1 as u8, g2 as u8],
        winner: 0,
        cap_plies: (MAX_TURNS * 2) as u16,
    }
}

// ==========================================================================
// Movement / combat (ported from Go core/movement_test.go)
// ==========================================================================

#[test]
fn test_move_to_own_tile_consolidates() {
    let mut tiles = new_board();
    tiles[0] = Tile {
        owner: 1,
        army: 5,
        kind: TileKind::Normal,
    };
    tiles[1] = Tile {
        owner: 1,
        army: 3,
        kind: TileKind::Normal,
    };
    let capture = apply_move(&mut tiles, 1, 0, 1);
    assert!(capture.is_none());
    assert_eq!(tiles[0].army, 1);
    assert_eq!(tiles[1].army, 7);
    assert_eq!(tiles[1].owner, 1);
}

#[test]
fn test_successful_capture_of_neutral_tile() {
    let mut tiles = new_board();
    tiles[0] = Tile {
        owner: 1,
        army: 5,
        kind: TileKind::Normal,
    };
    // tiles[1] neutral, 0 armies
    let capture = apply_move(&mut tiles, 1, 0, 1).expect("should capture");
    assert_eq!(tiles[1].owner, 1);
    assert_eq!(tiles[1].army, 4); // 4 attackers vs 0 defenders
    assert_eq!(capture.previous_owner, 0);
    assert_eq!(capture.previous_army, 0);
    assert_eq!(capture.capturing_player, 1);
}

#[test]
fn test_successful_capture_of_enemy_tile() {
    let mut tiles = new_board();
    tiles[0] = Tile {
        owner: 1,
        army: 10,
        kind: TileKind::Normal,
    };
    tiles[1] = Tile {
        owner: 2,
        army: 4,
        kind: TileKind::Normal,
    };
    let capture = apply_move(&mut tiles, 1, 0, 1).expect("should capture");
    assert_eq!(tiles[1].owner, 1);
    assert_eq!(tiles[1].army, 5); // 9 attackers - 4 defenders
    assert_eq!(capture.previous_owner, 2);
    assert_eq!(capture.previous_army, 4);
}

#[test]
fn test_failed_attack_defender_keeps_tile() {
    let mut tiles = new_board();
    tiles[0] = Tile {
        owner: 1,
        army: 4,
        kind: TileKind::Normal,
    };
    tiles[1] = Tile {
        owner: 2,
        army: 8,
        kind: TileKind::Normal,
    };
    let capture = apply_move(&mut tiles, 1, 0, 1);
    assert!(capture.is_none());
    assert_eq!(tiles[1].owner, 2);
    assert_eq!(tiles[1].army, 5); // 8 defenders - 3 attackers
    assert_eq!(tiles[0].army, 1);
}

#[test]
fn test_equal_armies_attack_fails() {
    // Go semantics: attacker needs strictly MORE armies than the defender
    let mut tiles = new_board();
    tiles[0] = Tile {
        owner: 1,
        army: 5,
        kind: TileKind::Normal,
    };
    tiles[1] = Tile {
        owner: 2,
        army: 4,
        kind: TileKind::Normal,
    };
    let capture = apply_move(&mut tiles, 1, 0, 1);
    assert!(capture.is_none());
    assert_eq!(tiles[1].owner, 2);
    assert_eq!(tiles[1].army, 0); // 4 - 4: wiped out but holds the tile
}

#[test]
fn test_general_capture_transfers_all_tiles() {
    let mut state = flat_state();
    let g2 = state.generals[1] as usize;
    // Give player 2 extra territory and put a huge player-1 army next to g2
    let p2_extra = idx(0, HEIGHT - 1);
    state.tiles[p2_extra] = Tile {
        owner: 2,
        army: 7,
        kind: TileKind::Normal,
    };
    let attacker_idx = idx(WIDTH - 2, HEIGHT - 1);
    state.tiles[attacker_idx] = Tile {
        owner: 1,
        army: 50,
        kind: TileKind::Normal,
    };

    let mut game = Generals::new();
    let action = encode_move(attacker_idx, 1); // right, onto g2
    let (_obs, reward, done, _info) = game.step(&mut state, action, &mut rng(0));

    assert!(done);
    assert_eq!(state.winner, 1);
    assert!((reward - 1.0).abs() < f32::EPSILON); // from player 1's perspective
    assert!(!state.alive[1]);
    // All of player 2's tiles (including the extra one) belong to player 1
    assert_eq!(state.tiles[p2_extra].owner, 1);
    assert_eq!(state.tiles[g2].owner, 1);
    // The captured general tile keeps its kind (Go behavior)
    assert_eq!(state.tiles[g2].kind, TileKind::General);
}

// ==========================================================================
// Legal mask vs brute force (ported from Go action_mask_test.go)
// ==========================================================================

#[test]
fn test_legal_mask_matches_brute_force() {
    let mut game = Generals::new();
    let mut r = rng(7);
    let (state, obs) = game.reset(&mut r, &[]);

    for action in 0..NUM_ACTIONS as u32 {
        let expected = rules::is_action_legal(&state.tiles, state.current_player, action);
        let in_mask = obs.legal_moves[action as usize] > 0.5;
        assert_eq!(
            in_mask, expected,
            "action {} mask disagreement (mask={}, brute={})",
            action, in_mask, expected
        );
    }
}

#[test]
fn test_wait_always_legal_and_moves_require_army() {
    let state = flat_state();
    let mut mask = [0.0f32; NUM_ACTIONS];
    rules::fill_legal_moves(&state.tiles, 1, true, &mut mask);

    assert!(mask[WAIT_ACTION as usize] > 0.5);
    // General has 2 armies: its in-bounds moves are legal
    let g1 = state.generals[0] as usize;
    assert!(mask[encode_move(g1, 1) as usize] > 0.5); // right
    assert!(mask[encode_move(g1, 2) as usize] > 0.5); // down
                                                      // Off-board directions from the corner are illegal
    assert!(mask[encode_move(g1, 0) as usize] < 0.5); // up
    assert!(mask[encode_move(g1, 3) as usize] < 0.5); // left
                                                      // A tile the player doesn't own offers no moves
    assert!(mask[encode_move(idx(3, 3), 0) as usize] < 0.5);
}

#[test]
fn test_mountain_blocks_moves() {
    let mut state = flat_state();
    let g1 = state.generals[0] as usize;
    let right = move_target(g1, 1).unwrap();
    state.tiles[right].kind = TileKind::Mountain;

    let mut mask = [0.0f32; NUM_ACTIONS];
    rules::fill_legal_moves(&state.tiles, 1, true, &mut mask);
    assert!(mask[encode_move(g1, 1) as usize] < 0.5); // blocked by mountain
    assert!(mask[encode_move(g1, 2) as usize] > 0.5); // down still fine
}

// ==========================================================================
// Production
// ==========================================================================

#[test]
fn test_production_generals_cities_and_growth() {
    let mut tiles = new_board();
    tiles[0] = Tile {
        owner: 1,
        army: 2,
        kind: TileKind::General,
    };
    tiles[1] = Tile {
        owner: 1,
        army: 40,
        kind: TileKind::City,
    };
    tiles[2] = Tile {
        owner: 1,
        army: 3,
        kind: TileKind::Normal,
    };
    tiles[3] = Tile {
        owner: 0,
        army: 40,
        kind: TileKind::City, // neutral city: no production
    };

    apply_production(&mut tiles, 1); // non-growth round
    assert_eq!(tiles[0].army, 3);
    assert_eq!(tiles[1].army, 41);
    assert_eq!(tiles[2].army, 3); // normal tile unchanged
    assert_eq!(tiles[3].army, 40); // neutral city unchanged

    apply_production(&mut tiles, params::NORMAL_GROW_INTERVAL); // growth round
    assert_eq!(tiles[2].army, 4);
}

// ==========================================================================
// Map generation (ported from Go mapgen/generator_test.go)
// ==========================================================================

#[test]
fn test_mapgen_same_seed_same_map() {
    let a = mapgen::generate_map(&mut rng(1234));
    let b = mapgen::generate_map(&mut rng(1234));
    assert_eq!(a.tiles, b.tiles);
    assert_eq!(a.generals, b.generals);
}

#[test]
fn test_mapgen_different_seeds_differ() {
    let a = mapgen::generate_map(&mut rng(1));
    let b = mapgen::generate_map(&mut rng(2));
    assert_ne!(
        (a.tiles, a.generals),
        (b.tiles, b.generals),
        "different seeds should produce different maps"
    );
}

#[test]
fn test_mapgen_invariants() {
    for seed in 0..50 {
        let map = mapgen::generate_map(&mut rng(seed));

        // Exactly two generals, owned by players 1 and 2, correct armies
        for (player, &g) in map.generals.iter().enumerate() {
            let tile = &map.tiles[g];
            assert_eq!(tile.kind, TileKind::General, "seed {}", seed);
            assert_eq!(tile.owner, player as u8 + 1, "seed {}", seed);
            assert_eq!(tile.army, GENERAL_START_ARMY, "seed {}", seed);
        }

        // City count and army (mountain veins may reduce free space but
        // 8x8 with 1 vein always fits 3 cities)
        let cities: Vec<_> = map
            .tiles
            .iter()
            .filter(|t| t.kind == TileKind::City)
            .collect();
        assert_eq!(cities.len(), BOARD_SIZE / CITY_RATIO, "seed {}", seed);
        for city in cities {
            assert_eq!(city.army, CITY_START_ARMY, "seed {}", seed);
            assert!(city.is_neutral(), "seed {}", seed);
        }

        // Mountains carry no armies and no owner
        for tile in map.tiles.iter().filter(|t| t.is_mountain()) {
            assert_eq!(tile.army, 0, "seed {}", seed);
            assert!(tile.is_neutral(), "seed {}", seed);
        }
    }
}

#[test]
fn test_mapgen_general_spacing() {
    // Spacing is best-effort (fallback relaxes it), but on an 8x8 board with
    // a single short vein the spaced placement should essentially always
    // succeed. Verify across many seeds.
    let spacing = params::MIN_GENERAL_SPACING.min(WIDTH / 2 + HEIGHT / 2);
    for seed in 0..50 {
        let map = mapgen::generate_map(&mut rng(seed));
        let d = board::manhattan(map.generals[0], map.generals[1]);
        assert!(
            d >= spacing,
            "seed {}: generals {} apart, want >= {}",
            seed,
            d,
            spacing
        );
    }
}

// ==========================================================================
// Encode/decode round-trips (Cartridge2 house convention)
// ==========================================================================

#[test]
fn test_state_encode_decode_roundtrip() {
    let mut game = Generals::new();
    let mut r = rng(99);
    let (mut state, _obs) = game.reset(&mut r, &[]);

    // Mutate into a mid-game-looking state
    state.round = 42;
    state.current_player = 2;
    state.tiles[10] = Tile {
        owner: 1,
        army: 123,
        kind: TileKind::Normal,
    };

    let mut buf = Vec::new();
    Generals::encode_state(&state, &mut buf).unwrap();
    let decoded = Generals::decode_state(&buf).unwrap();
    assert_eq!(state, decoded);
}

#[test]
fn test_state_decode_rejects_garbage() {
    assert!(Generals::decode_state(&[]).is_err());
    assert!(Generals::decode_state(&[0u8; 10]).is_err());

    // Corrupt current_player
    let mut game = Generals::new();
    let (state, _) = game.reset(&mut rng(1), &[]);
    let mut buf = Vec::new();
    Generals::encode_state(&state, &mut buf).unwrap();
    buf[4] = 9;
    assert!(Generals::decode_state(&buf).is_err());
}

#[test]
fn test_action_encode_decode_roundtrip() {
    for action in [0u32, 1, 100, 255, WAIT_ACTION] {
        let mut buf = Vec::new();
        Generals::encode_action(&action, &mut buf).unwrap();
        assert_eq!(Generals::decode_action(&buf).unwrap(), action);
    }
    assert!(Generals::encode_action(&(NUM_ACTIONS as u32), &mut Vec::new()).is_err());
    assert!(Generals::decode_action(&(NUM_ACTIONS as u32).to_le_bytes()).is_err());
}

#[test]
fn test_obs_shape_and_metadata_agree() {
    let game = Generals::new();
    let meta = game.metadata();
    assert_eq!(meta.num_actions, NUM_ACTIONS);
    assert_eq!(meta.obs_size, OBS_SIZE);
    assert_eq!(meta.legal_mask_offset, LEGAL_MASK_OFFSET);

    let mut g = Generals::new();
    let (_state, obs) = g.reset(&mut rng(5), &[]);
    let mut buf = Vec::new();
    Generals::encode_obs(&obs, &mut buf).unwrap();
    assert_eq!(buf.len(), OBS_SIZE * 4);

    // The advertised legal_mask_offset must point at the legal plane
    let mask = meta.legal_mask_from_obs(&buf);
    assert!(mask.is_legal(WAIT_ACTION as usize));
    for a in mask.iter_ones() {
        assert!(obs.legal_moves[a] > 0.5);
    }
}

#[test]
fn test_obs_is_player_relative() {
    let state = flat_state();
    let obs_p1 = GeneralsObs::from_tiles(&state.tiles, 1, state.alive, 0);
    let obs_p2 = GeneralsObs::from_tiles(&state.tiles, 2, state.alive, 0);

    let g1 = state.generals[0] as usize;
    let g2 = state.generals[1] as usize;
    // Player 1's view: own territory at g1, enemy at g2
    assert!(obs_p1.channels[g1] > 0.5); // ch0 own
    assert!(obs_p1.channels[BOARD_SIZE + g2] > 0.5); // ch1 enemy
    assert!(obs_p1.channels[7 * BOARD_SIZE + g1] > 0.5); // ch7 own general +1
    assert!(obs_p1.channels[7 * BOARD_SIZE + g2] < -0.5); // ch7 enemy general -1
                                                          // Player 2's view is mirrored
    assert!(obs_p2.channels[g2] > 0.5);
    assert!(obs_p2.channels[BOARD_SIZE + g1] > 0.5);
    assert!(obs_p2.channels[7 * BOARD_SIZE + g2] > 0.5);
    assert!(obs_p2.channels[7 * BOARD_SIZE + g1] < -0.5);
}

// ==========================================================================
// Step semantics
// ==========================================================================

#[test]
fn test_alternating_turns_and_round_clock() {
    let mut game = Generals::new();
    let mut r = rng(3);
    let (mut state, _) = game.reset(&mut r, &[]);

    assert_eq!(state.current_player, 1);
    game.step(&mut state, WAIT_ACTION, &mut r);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.round, 0); // round not over yet
    game.step(&mut state, WAIT_ACTION, &mut r);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.round, 1); // both moved: round ticked, production ran
}

#[test]
fn test_production_runs_at_round_end() {
    let mut game = Generals::new();
    let mut r = rng(3);
    let (mut state, _) = game.reset(&mut r, &[]);
    let g1 = state.generals[0] as usize;
    let before = state.tiles[g1].army;

    game.step(&mut state, WAIT_ACTION, &mut r); // P1
    assert_eq!(state.tiles[g1].army, before); // not yet
    game.step(&mut state, WAIT_ACTION, &mut r); // P2 -> round end
    assert_eq!(state.tiles[g1].army, before + params::GENERAL_PRODUCTION);
}

#[test]
fn test_illegal_action_degrades_to_wait() {
    let mut game = Generals::new();
    let mut r = rng(3);
    let (mut state, _) = game.reset(&mut r, &[]);
    let snapshot = state.tiles.clone();

    // Move from an unowned tile: must change nothing but the turn
    let unowned = (0..BOARD_SIZE)
        .find(|&i| state.tiles[i].is_neutral())
        .unwrap();
    let (_obs, reward, done, _info) = game.step(&mut state, encode_move(unowned, 1), &mut r);
    assert_eq!(state.tiles, snapshot);
    assert_eq!(state.current_player, 2);
    assert!(!done);
    assert!(reward.abs() < f32::EPSILON);
}

#[test]
fn test_max_turns_symmetric_position_is_draw() {
    // flat_state is perfectly symmetric (one general each, equal armies),
    // so cap adjudication yields a true draw.
    let mut game = Generals::new();
    let mut r = rng(3);
    let mut state = flat_state();
    state.round = MAX_TURNS - 1;

    game.step(&mut state, WAIT_ACTION, &mut r); // P1
    let (_obs, reward, done, _info) = game.step(&mut state, WAIT_ACTION, &mut r); // P2
    assert!(done);
    assert_eq!(state.winner, 3);
    assert!(reward.abs() < f32::EPSILON); // draw reward is 0
}

#[test]
fn test_max_turns_adjudicates_by_territory() {
    let mut game = Generals::new();
    let mut r = rng(3);
    let mut state = flat_state();
    // Player 2 holds an extra tile
    state.tiles[idx(3, 3)] = Tile {
        owner: 2,
        army: 1,
        kind: TileKind::Normal,
    };
    state.round = MAX_TURNS - 1;

    game.step(&mut state, WAIT_ACTION, &mut r); // P1
    let (_obs, reward, done, _info) = game.step(&mut state, WAIT_ACTION, &mut r); // P2
    assert!(done);
    assert_eq!(state.winner, 2);
    assert!((reward - 1.0).abs() < f32::EPSILON); // P2 moved last and wins
}

#[test]
fn test_odd_cap_gives_player1_the_last_move() {
    // With cap_plies = 2*MAX_TURNS - 1, the game adjudicates after
    // player 1's final ply — player 2 never gets to answer.
    let mut game = Generals::new();
    let mut r = rng(3);
    let mut state = flat_state();
    state.cap_plies = (MAX_TURNS * 2 - 1) as u16;
    state.round = MAX_TURNS - 1;
    // Give player 1 a pending extra tile so adjudication is decisive
    state.tiles[idx(3, 3)] = Tile {
        owner: 1,
        army: 1,
        kind: TileKind::Normal,
    };

    let (_obs, reward, done, _info) = game.step(&mut state, WAIT_ACTION, &mut r); // P1's last ply
    assert!(done, "game must end after P1's ply at an odd cap");
    assert_eq!(state.winner, 1);
    assert!((reward - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_reset_samples_both_cap_parities() {
    let mut game = Generals::new();
    let mut seen = [false, false];
    for seed in 0..40 {
        let (state, _) = game.reset(&mut rng(seed), &[]);
        let cap = state.cap_plies as u32;
        assert!(cap == MAX_TURNS * 2 || cap == MAX_TURNS * 2 - 1);
        seen[(MAX_TURNS * 2 - cap) as usize] = true;
    }
    assert!(
        seen[0] && seen[1],
        "40 seeds should produce both cap parities"
    );
}

#[test]
fn test_adjudication_army_tiebreak() {
    use crate::rules::adjudicate_at_cap;
    let mut state = flat_state();
    // Equal tile counts; give player 1 more armies
    state.tiles[state.generals[0] as usize].army = 10;
    assert_eq!(adjudicate_at_cap(&state.tiles), 1);
    // Perfectly even position is a draw
    state.tiles[state.generals[0] as usize].army = GENERAL_START_ARMY;
    assert_eq!(adjudicate_at_cap(&state.tiles), 3);
}

// ==========================================================================
// Full-game smoke test: random-legal self-play from seeds
// ==========================================================================

#[test]
fn test_random_playout_terminates_cleanly() {
    for seed in 0..10 {
        let mut game = Generals::new();
        let mut r = rng(seed);
        let (mut state, mut obs) = game.reset(&mut r, &[]);
        let mut plies = 0u32;

        loop {
            // Pick a uniformly random legal action from the obs mask
            let legal: Vec<u32> = (0..NUM_ACTIONS as u32)
                .filter(|&a| obs.legal_moves[a as usize] > 0.5)
                .collect();
            assert!(!legal.is_empty(), "seed {}: no legal actions", seed);
            let action = legal[r.gen_range(0..legal.len())];

            let (new_obs, _reward, done, _info) = game.step(&mut state, action, &mut r);
            obs = new_obs;
            plies += 1;
            assert!(
                plies <= MAX_TURNS * 2 + 2,
                "seed {}: game exceeded the horizon",
                seed
            );

            // Army conservation sanity: no tile ever has armies on a mountain
            debug_assert!(state.tiles.iter().all(|t| !t.is_mountain() || t.army == 0));

            if done {
                assert!(state.winner >= 1 && state.winner <= 3, "seed {}", seed);
                break;
            }
        }
    }
}

// ==========================================================================
// Registration / EngineContext integration
// ==========================================================================

#[test]
fn test_register_and_play_via_context() {
    use engine_core::EngineContext;
    register_generals();

    let mut ctx = EngineContext::new("generals_8x8").expect("registered");
    let reset = ctx.reset(42, &[]).unwrap();
    assert_eq!(reset.obs.len(), OBS_SIZE * 4);

    // Step a wait action through the byte interface
    let action = WAIT_ACTION.to_le_bytes().to_vec();
    let step = ctx.step(&reset.state, &action).unwrap();
    assert!(!step.done);
    assert_eq!(step.obs.len(), OBS_SIZE * 4);

    // Metadata round-trip through the erased layer
    let meta = ctx.metadata();
    assert_eq!(meta.env_id, "generals_8x8");
    assert_eq!(meta.num_actions, NUM_ACTIONS);
}
