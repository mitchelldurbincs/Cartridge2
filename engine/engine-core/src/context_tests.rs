//! Tests for the EngineContext API

use super::*;
use crate::adapter::GameAdapter;
use crate::registry::{clear_registry, register_game};
use crate::test_utils::REGISTRY_TEST_MUTEX;
use crate::typed::{DecodeError, EncodeError, Encoding, Game};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[derive(Debug, Default)]
struct SimpleGame;

impl Game for SimpleGame {
    type State = u32;
    type Action = u32; // Changed from u8 to match sample_random_action's u32 output
    type Obs = f32;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "simple".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "u32".into(),
                action: "u32".into(),
                obs: "f32".into(),
                schema_version: 1,
            },
            max_horizon: 10,
            action_space: ActionSpace::Discrete(4),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("simple", "Simple Game")
            .with_board(2, 2)
            .with_actions(4)
            .with_observation(1, 0)
    }

    fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        (0, 0.0)
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        *state += action;
        let done = *state >= 10;
        (*state as f32, 1.0, done, *state as u64)
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&state.to_le_bytes());
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        if buf.len() != 4 {
            return Err(DecodeError::InvalidLength {
                expected: 4,
                actual: buf.len(),
            });
        }
        Ok(u32::from_le_bytes(buf.try_into().unwrap()))
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&action.to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 4 {
            return Err(DecodeError::InvalidLength {
                expected: 4,
                actual: buf.len(),
            });
        }
        Ok(u32::from_le_bytes(buf.try_into().unwrap()))
    }

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&obs.to_le_bytes());
        Ok(())
    }
}

fn setup_registry() {
    clear_registry();
    register_game("simple".to_string(), || {
        Box::new(GameAdapter::new(SimpleGame))
    });
}

#[test]
fn test_context_creation() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let ctx = EngineContext::new("simple");
    assert!(ctx.is_some());

    let ctx = ctx.unwrap();
    assert_eq!(ctx.engine_id().env_id, "simple");
}

#[test]
fn test_context_nonexistent_game() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let ctx = EngineContext::new("nonexistent");
    assert!(ctx.is_none());
}

#[test]
fn test_context_reset() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();
    let result = ctx.reset(42, &[]).unwrap();

    assert_eq!(result.state.len(), 4);
    assert_eq!(result.obs.len(), 4);

    let state = u32::from_le_bytes(result.state.try_into().unwrap());
    assert_eq!(state, 0);
}

#[test]
fn test_context_step() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();
    let reset_result = ctx.reset(42, &[]).unwrap();

    let action = 3u32.to_le_bytes().to_vec();
    let step_result = ctx.step(&reset_result.state, &action).unwrap();

    assert_eq!(step_result.reward, 1.0);
    assert!(!step_result.done);

    let new_state = u32::from_le_bytes(step_result.state.try_into().unwrap());
    assert_eq!(new_state, 3);
}

#[test]
fn test_context_sample_random_action() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let ctx = EngineContext::new("simple").unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let action = ctx.sample_random_action(&mut rng);
    assert!(!action.is_empty());
}

#[test]
fn test_context_full_episode() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let ctx_opt = EngineContext::new("simple");
    assert!(ctx_opt.is_some(), "simple game should be registered");
    let mut ctx = ctx_opt.unwrap();

    let reset_result = ctx.reset(42, &[]).unwrap();
    let mut state = reset_result.state;
    let mut done = false;
    let mut steps = 0;

    while !done && steps < 20 {
        let action = 2u32.to_le_bytes().to_vec(); // Always add 2
        let step_result = ctx.step(&state, &action).unwrap();
        state = step_result.state;
        done = step_result.done;
        steps += 1;
    }

    assert!(done);
    assert!(steps <= 10); // Should terminate when state >= 10
}

// =========================================================================
// Fuzz-style tests for EngineContext
// =========================================================================

#[test]
fn test_context_random_seeds_deterministic() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    // Same seed should produce same results
    for seed in [0u64, 42, 12345, u64::MAX] {
        let mut ctx1 = EngineContext::new("simple").unwrap();
        let mut ctx2 = EngineContext::new("simple").unwrap();

        let result1 = ctx1.reset(seed, &[]).unwrap();
        let result2 = ctx2.reset(seed, &[]).unwrap();

        assert_eq!(
            result1.state, result2.state,
            "Same seed should produce same state"
        );
        assert_eq!(
            result1.obs, result2.obs,
            "Same seed should produce same obs"
        );
    }
}

#[test]
fn test_context_many_random_episodes() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();

    for episode_seed in 0..20u64 {
        let mut rng = ChaCha20Rng::seed_from_u64(episode_seed);

        let reset_result = ctx.reset(episode_seed, &[]).unwrap();
        let mut state = reset_result.state;
        let mut done = false;
        let mut steps = 0;
        const MAX_STEPS: usize = 100;

        while !done && steps < MAX_STEPS {
            // Sample random action
            let action = ctx.sample_random_action(&mut rng);

            let step_result = ctx.step(&state, &action);
            assert!(
                step_result.is_ok(),
                "Step should not error (episode={}, step={})",
                episode_seed,
                steps
            );

            let step_result = step_result.unwrap();
            state = step_result.state;
            done = step_result.done;
            steps += 1;
        }

        // Episode should terminate (SimpleGame terminates when state >= 10)
        assert!(
            done || steps == MAX_STEPS,
            "Episode {} should terminate",
            episode_seed
        );
    }
}

#[test]
fn test_context_invalid_action_bytes() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();
    let reset_result = ctx.reset(42, &[]).unwrap();

    // SimpleGame expects 4 bytes for action (u32), test with wrong sizes
    let wrong_sizes = [vec![], vec![1], vec![1, 2], vec![1, 2, 3, 4, 5]];

    for bad_action in &wrong_sizes {
        let result = ctx.step(&reset_result.state, bad_action);
        assert!(
            result.is_err(),
            "Should reject action of size {}",
            bad_action.len()
        );
    }
}

#[test]
fn test_context_invalid_state_bytes() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();

    // SimpleGame expects 4 bytes for state (u32)
    let wrong_state_sizes = [vec![], vec![1], vec![1, 2], vec![1, 2, 3, 4, 5]];
    let valid_action = 1u32.to_le_bytes().to_vec();

    for bad_state in &wrong_state_sizes {
        let result = ctx.step(bad_state, &valid_action);
        assert!(
            result.is_err(),
            "Should reject state of size {}",
            bad_state.len()
        );
    }
}

#[test]
fn test_context_step_roundtrip_encoding() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();

    // Reset and verify state decodes correctly
    let reset = ctx.reset(123, &[]).unwrap();
    let initial_state = u32::from_le_bytes(reset.state.clone().try_into().unwrap());
    assert_eq!(initial_state, 0);

    // Step and verify output state decodes correctly
    let action = 5u32.to_le_bytes().to_vec();
    let step = ctx.step(&reset.state, &action).unwrap();
    let new_state = u32::from_le_bytes(step.state.clone().try_into().unwrap());
    assert_eq!(new_state, 5);

    // Step again to verify chaining works
    let action2 = 3u32.to_le_bytes().to_vec();
    let step2 = ctx.step(&step.state, &action2).unwrap();
    let final_state = u32::from_le_bytes(step2.state.try_into().unwrap());
    assert_eq!(final_state, 8);
}

// =========================================================================
// Zero-copy API tests
// =========================================================================

#[test]
fn test_reset_into_basic() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();
    let mut state = Vec::new();
    let mut obs = Vec::new();

    ctx.reset_into(42, &[], &mut state, &mut obs).unwrap();

    assert_eq!(state.len(), 4);
    assert_eq!(obs.len(), 4);

    let state_val = u32::from_le_bytes(state.try_into().unwrap());
    assert_eq!(state_val, 0);
}

#[test]
fn test_step_into_basic() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();
    let mut state = Vec::new();
    let mut obs = Vec::new();

    ctx.reset_into(42, &[], &mut state, &mut obs).unwrap();

    let mut next_state = Vec::new();
    let mut next_obs = Vec::new();
    let action = 3u32.to_le_bytes().to_vec();

    let (reward, done, info) = ctx
        .step_into(&state, &action, &mut next_state, &mut next_obs)
        .unwrap();

    assert_eq!(reward, 1.0);
    assert!(!done);
    assert_eq!(info, 3);

    let new_state_val = u32::from_le_bytes(next_state.try_into().unwrap());
    assert_eq!(new_state_val, 3);
}

#[test]
fn test_zero_copy_full_episode() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();

    // Allocate buffers once with capacity
    let mut state = Vec::with_capacity(16);
    let mut obs = Vec::with_capacity(16);
    let mut next_state = Vec::with_capacity(16);
    let mut next_obs = Vec::with_capacity(16);

    ctx.reset_into(42, &[], &mut state, &mut obs).unwrap();

    let mut done = false;
    let mut steps = 0;

    while !done && steps < 20 {
        let action = 2u32.to_le_bytes().to_vec();

        let (_, d, _) = ctx
            .step_into(&state, &action, &mut next_state, &mut next_obs)
            .unwrap();

        // Swap buffers - this reuses memory, no allocations
        std::mem::swap(&mut state, &mut next_state);
        std::mem::swap(&mut obs, &mut next_obs);

        done = d;
        steps += 1;
    }

    assert!(done);
    assert!(steps <= 10);
}

#[test]
fn test_zero_copy_buffer_reuse() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx = EngineContext::new("simple").unwrap();

    // Pre-allocate with specific capacity
    let mut state = Vec::with_capacity(64);
    let mut obs = Vec::with_capacity(64);

    // First reset
    ctx.reset_into(1, &[], &mut state, &mut obs).unwrap();
    let cap_after_first = (state.capacity(), obs.capacity());

    // Second reset - should reuse same allocation
    ctx.reset_into(2, &[], &mut state, &mut obs).unwrap();
    let cap_after_second = (state.capacity(), obs.capacity());

    // Capacities should be unchanged (no reallocation)
    assert_eq!(cap_after_first, cap_after_second);
}

#[test]
fn test_zero_copy_deterministic() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    // Same seed should produce same results with both APIs
    for seed in [0u64, 42, 12345] {
        let mut ctx1 = EngineContext::new("simple").unwrap();
        let mut ctx2 = EngineContext::new("simple").unwrap();

        // Original API
        let result1 = ctx1.reset(seed, &[]).unwrap();

        // Zero-copy API
        let mut state2 = Vec::new();
        let mut obs2 = Vec::new();
        ctx2.reset_into(seed, &[], &mut state2, &mut obs2).unwrap();

        assert_eq!(
            result1.state, state2,
            "States should match for seed {}",
            seed
        );
        assert_eq!(result1.obs, obs2, "Obs should match for seed {}", seed);
    }
}

#[test]
fn test_zero_copy_matches_original_api() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    setup_registry();

    let mut ctx1 = EngineContext::new("simple").unwrap();
    let mut ctx2 = EngineContext::new("simple").unwrap();

    // Reset with both APIs
    let reset1 = ctx1.reset(42, &[]).unwrap();

    let mut state2 = Vec::new();
    let mut obs2 = Vec::new();
    ctx2.reset_into(42, &[], &mut state2, &mut obs2).unwrap();

    assert_eq!(reset1.state, state2);
    assert_eq!(reset1.obs, obs2);

    // Step with both APIs
    let action = 5u32.to_le_bytes().to_vec();

    let step1 = ctx1.step(&reset1.state, &action).unwrap();

    let mut next_state2 = Vec::new();
    let mut next_obs2 = Vec::new();
    let (reward2, done2, info2) = ctx2
        .step_into(&state2, &action, &mut next_state2, &mut next_obs2)
        .unwrap();

    assert_eq!(step1.state, next_state2);
    assert_eq!(step1.obs, next_obs2);
    assert_eq!(step1.reward, reward2);
    assert_eq!(step1.done, done2);
    assert_eq!(step1.info, info2);
}
