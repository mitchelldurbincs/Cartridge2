//! Tests for the GameAdapter type

use super::*;
use crate::typed::{ActionSpace, DecodeError, EncodeError, Encoding};
use rand::RngCore;

// Test game implementation
#[derive(Debug, PartialEq)]
struct TestGame {
    id: String,
    reset_count: u32,
    step_count: u32,
}

impl TestGame {
    fn new(id: String) -> Self {
        Self {
            id,
            reset_count: 0,
            step_count: 0,
        }
    }
}

impl Game for TestGame {
    type State = u32;
    type Action = u8;
    type Obs = Vec<f32>;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: self.id.clone(),
            build_id: "0.1.0".to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "u32:v1".to_string(),
                action: "u8:v1".to_string(),
                obs: "f32_vec:v1".to_string(),
                schema_version: 1,
            },
            max_horizon: 100,
            action_space: ActionSpace::Discrete(4),
            preferred_batch: 32,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("test", "Test Game")
            .with_board(2, 2)
            .with_actions(4)
            .with_observation(2, 0)
    }

    fn reset(&mut self, rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        self.reset_count += 1;
        self.step_count = 0;

        // Use RNG to ensure it's properly seeded
        use rand::Rng;
        let random_val = rng.gen::<u32>() % 100;

        (random_val, vec![random_val as f32])
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        self.step_count += 1;
        *state += action as u32;

        let obs = vec![*state as f32, self.step_count as f32];
        let reward = action as f32;
        let done = *state >= 20 || self.step_count >= 10;

        let info = ((*state as u64) << 32) | self.step_count as u64;

        (obs, reward, done, info)
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
        out.push(*action);
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 1 {
            return Err(DecodeError::InvalidLength {
                expected: 1,
                actual: buf.len(),
            });
        }
        Ok(buf[0])
    }

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        // Encode length first, then values
        let len = obs.len() as u32;
        out.extend_from_slice(&len.to_le_bytes());
        for &value in obs {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    fn view(_state: &Self::State) -> crate::BoardView {
        // Test double: no board to project.
        crate::BoardView::from_owners(&[], 1, 0)
    }
}

#[test]
fn test_adapter_basic_functionality() {
    let game = TestGame::new("test".to_string());
    let adapter = GameAdapter::new(game);

    // Test engine_id passthrough
    let id = adapter.engine_id();
    assert_eq!(id.env_id, "test");

    // Test capabilities passthrough
    let caps = adapter.capabilities();
    assert_eq!(caps.id.env_id, "test");
    assert_eq!(caps.max_horizon, 100);
}

#[test]
fn test_adapter_reset() {
    let game = TestGame::new("test".to_string());
    let mut adapter = GameAdapter::new(game);

    let mut state_buf = Vec::new();
    let mut obs_buf = Vec::new();

    adapter
        .reset(42, &[], &mut state_buf, &mut obs_buf)
        .unwrap();

    // State should be encoded as 4 bytes (u32)
    assert_eq!(state_buf.len(), 4);
    let state_value = u32::from_le_bytes(state_buf.try_into().unwrap());

    // Obs should be encoded as length + values
    assert!(obs_buf.len() >= 4); // At least length header
    let obs_len = u32::from_le_bytes(obs_buf[0..4].try_into().unwrap());
    assert_eq!(obs_len, 1); // One f32 value
    assert_eq!(obs_buf.len(), 4 + 4); // Length + one f32

    // Verify the observation value matches the state
    let obs_value = f32::from_le_bytes(obs_buf[4..8].try_into().unwrap());
    assert_eq!(obs_value, state_value as f32);
}

#[test]
fn test_adapter_step() {
    let game = TestGame::new("test".to_string());
    let mut adapter = GameAdapter::new(game);

    // Reset first
    let mut state_buf = Vec::new();
    let mut obs_buf = Vec::new();
    adapter
        .reset(42, &[], &mut state_buf, &mut obs_buf)
        .unwrap();

    // Prepare action
    let action_bytes = vec![3u8];

    // Take a step
    let mut new_state_buf = Vec::new();
    let mut new_obs_buf = Vec::new();
    let (reward, _done, info) = adapter
        .step(
            &state_buf,
            &action_bytes,
            &mut new_state_buf,
            &mut new_obs_buf,
        )
        .unwrap();

    // Verify reward
    assert_eq!(reward, 3.0);
    assert!(info > 0);

    // Decode new state
    let new_state = u32::from_le_bytes(new_state_buf.try_into().unwrap());
    let old_state = u32::from_le_bytes(state_buf.try_into().unwrap());
    assert_eq!(new_state, old_state + 3);

    // Verify obs structure
    assert!(new_obs_buf.len() >= 4);
    let obs_len = u32::from_le_bytes(new_obs_buf[0..4].try_into().unwrap());
    assert_eq!(obs_len, 2); // Two f32 values (state and step_count)
}

#[test]
fn test_adapter_deterministic_reset() {
    let game1 = TestGame::new("test".to_string());
    let mut adapter1 = GameAdapter::new(game1);

    let game2 = TestGame::new("test".to_string());
    let mut adapter2 = GameAdapter::new(game2);

    // Reset with same seed
    let mut state1 = Vec::new();
    let mut obs1 = Vec::new();
    adapter1.reset(12345, &[], &mut state1, &mut obs1).unwrap();

    let mut state2 = Vec::new();
    let mut obs2 = Vec::new();
    adapter2.reset(12345, &[], &mut state2, &mut obs2).unwrap();

    // Results should be identical
    assert_eq!(state1, state2);
    assert_eq!(obs1, obs2);
}

#[test]
fn test_adapter_different_seeds() {
    let game1 = TestGame::new("test".to_string());
    let mut adapter1 = GameAdapter::new(game1);

    let game2 = TestGame::new("test".to_string());
    let mut adapter2 = GameAdapter::new(game2);

    // Reset with different seeds
    let mut state1 = Vec::new();
    let mut obs1 = Vec::new();
    adapter1.reset(12345, &[], &mut state1, &mut obs1).unwrap();

    let mut state2 = Vec::new();
    let mut obs2 = Vec::new();
    adapter2.reset(54321, &[], &mut state2, &mut obs2).unwrap();

    // Results should be different (with very high probability)
    // Note: There's a tiny chance they could be the same due to randomness
    assert!(state1 != state2 || obs1 != obs2);
}

#[test]
fn test_adapter_inner_access() {
    let game = TestGame::new("test".to_string());
    let mut adapter = GameAdapter::new(game);

    // Test mutable access
    adapter.game_mut().id = "modified".to_string();
    assert_eq!(adapter.game().id, "modified");

    // Test into_inner
    let inner_game = adapter.into_inner();
    assert_eq!(inner_game.id, "modified");
}

#[test]
fn test_adapter_invalid_action_decoding() {
    let game = TestGame::new("test".to_string());
    let mut adapter = GameAdapter::new(game);

    // Reset first
    let mut state_buf = Vec::new();
    let mut obs_buf = Vec::new();
    adapter
        .reset(42, &[], &mut state_buf, &mut obs_buf)
        .unwrap();

    // Try step with invalid action (wrong length)
    let invalid_action = vec![1, 2, 3]; // Should be 1 byte
    let mut new_state_buf = Vec::new();
    let mut new_obs_buf = Vec::new();

    let result = adapter.step(
        &state_buf,
        &invalid_action,
        &mut new_state_buf,
        &mut new_obs_buf,
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        ErasedGameError::Decoding(_) => {
            // Test passes - we got the expected error type
        }
        _ => panic!("Expected Decoding error"),
    }
}

#[test]
fn test_adapter_invalid_state_decoding() {
    let game = TestGame::new("test".to_string());
    let mut adapter = GameAdapter::new(game);

    // Try step with invalid state (wrong length)
    let invalid_state = vec![1, 2, 3]; // Should be 4 bytes for u32
    let action = vec![1u8];
    let mut new_state_buf = Vec::new();
    let mut new_obs_buf = Vec::new();

    let result = adapter.step(
        &invalid_state,
        &action,
        &mut new_state_buf,
        &mut new_obs_buf,
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        ErasedGameError::Decoding(_) => {
            // Test passes - we got the expected error type
        }
        _ => panic!("Expected Decoding error"),
    }
}

#[derive(Debug)]
struct CounterGame;

impl Game for CounterGame {
    type State = u64;
    type Action = u8;
    type Obs = [u8; 8];

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "counter".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "u64:v1".into(),
                action: "u8:v1".into(),
                obs: "bytes:v1".into(),
                schema_version: 1,
            },
            max_horizon: 10,
            action_space: ActionSpace::Discrete(8),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("counter", "Counter Game")
            .with_board(1, 1)
            .with_actions(8)
            .with_observation(8, 0)
    }

    fn reset(&mut self, rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        let value = rng.next_u64();
        (value, value.to_le_bytes())
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        *state = state.wrapping_add(action as u64);
        let _ = rng.next_u64();
        ((*state).to_le_bytes(), *state as f32, false, *state)
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&state.to_le_bytes());
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        if buf.len() != 8 {
            return Err(DecodeError::InvalidLength {
                expected: 8,
                actual: buf.len(),
            });
        }
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(buf);
        Ok(u64::from_le_bytes(bytes))
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.push(*action);
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 1 {
            return Err(DecodeError::InvalidLength {
                expected: 1,
                actual: buf.len(),
            });
        }
        Ok(buf[0])
    }

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(obs);
        Ok(())
    }

    fn view(_state: &Self::State) -> crate::BoardView {
        // Test double: no board to project.
        crate::BoardView::from_owners(&[], 1, 0)
    }
}

#[test]
fn test_adapter_reset_reseeds_rng_after_step() {
    let mut adapter = GameAdapter::new(CounterGame);

    let mut initial_state = Vec::new();
    let mut initial_obs = Vec::new();
    adapter
        .reset(9, &[], &mut initial_state, &mut initial_obs)
        .unwrap();

    // Advance the RNG state through a step
    let mut action_bytes = Vec::new();
    CounterGame::encode_action(&3, &mut action_bytes).unwrap();
    let mut stepped_state = Vec::new();
    let mut stepped_obs = Vec::new();
    adapter
        .step(
            &initial_state,
            &action_bytes,
            &mut stepped_state,
            &mut stepped_obs,
        )
        .unwrap();

    // Reset with the same seed should reproduce the original bytes
    let mut second_state = Vec::new();
    let mut second_obs = Vec::new();
    adapter
        .reset(9, &[], &mut second_state, &mut second_obs)
        .unwrap();

    assert_eq!(initial_state, second_state);
    assert_eq!(initial_obs, second_obs);
}

#[test]
fn test_step_clears_prepopulated_output_buffers() {
    let mut adapter = GameAdapter::new(CounterGame);

    let mut state_buf = Vec::new();
    let mut obs_buf = Vec::new();
    adapter
        .reset(123, &[], &mut state_buf, &mut obs_buf)
        .unwrap();

    let mut action_bytes = Vec::new();
    CounterGame::encode_action(&1, &mut action_bytes).unwrap();

    let mut new_state_buf = vec![0xFF; 32];
    let mut new_obs_buf = vec![0xAA; 32];
    adapter
        .step(
            &state_buf,
            &action_bytes,
            &mut new_state_buf,
            &mut new_obs_buf,
        )
        .unwrap();

    assert_eq!(new_state_buf.len(), 8);
    assert_eq!(new_obs_buf.len(), 8);
}

#[derive(Debug)]
struct ResetEncodingFailsGame;

impl Game for ResetEncodingFailsGame {
    type State = ();
    type Action = ();
    type Obs = ();

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "reset_fail".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "void".into(),
                action: "void".into(),
                obs: "void".into(),
                schema_version: 1,
            },
            max_horizon: 1,
            action_space: ActionSpace::Discrete(1),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("reset_fail", "Reset Fail Game")
            .with_board(1, 1)
            .with_actions(1)
            .with_observation(0, 0)
    }

    fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        ((), ())
    }

    fn step(
        &mut self,
        _state: &mut Self::State,
        _action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        ((), 0.0, false, 0)
    }

    fn encode_state(_state: &Self::State, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
        Err(EncodeError::InvalidData("state fail".into()))
    }

    fn decode_state(_buf: &[u8]) -> Result<Self::State, DecodeError> {
        Ok(())
    }

    fn encode_action(_action: &Self::Action, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
        Ok(())
    }

    fn decode_action(_buf: &[u8]) -> Result<Self::Action, DecodeError> {
        Ok(())
    }

    fn encode_obs(_obs: &Self::Obs, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
        Ok(())
    }

    fn view(_state: &Self::State) -> crate::BoardView {
        // Test double: no board to project.
        crate::BoardView::from_owners(&[], 1, 0)
    }
}

#[test]
fn test_reset_propagates_encoding_errors() {
    let mut adapter = GameAdapter::new(ResetEncodingFailsGame);
    let mut state_buf = Vec::new();
    let mut obs_buf = Vec::new();

    let err = adapter
        .reset(0, &[], &mut state_buf, &mut obs_buf)
        .unwrap_err();
    match err {
        ErasedGameError::Encoding(msg) => assert!(msg.contains("state fail")),
        other => panic!("expected encoding error, got {other:?}"),
    }
}

#[derive(Debug)]
struct StepEncodingFailsGame;

impl Game for StepEncodingFailsGame {
    type State = u8;
    type Action = ();
    type Obs = bool;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "step_fail".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "u8:v1".into(),
                action: "unit".into(),
                obs: "bool".into(),
                schema_version: 1,
            },
            max_horizon: 1,
            action_space: ActionSpace::Discrete(1),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("step_fail", "Step Fail Game")
            .with_board(1, 1)
            .with_actions(1)
            .with_observation(1, 0)
    }

    fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        (0, false)
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        _action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        *state = state.wrapping_add(1);
        (true, 0.0, false, 0)
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.push(*state);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        if buf.len() != 1 {
            return Err(DecodeError::InvalidLength {
                expected: 1,
                actual: buf.len(),
            });
        }
        Ok(buf[0])
    }

    fn encode_action(_action: &Self::Action, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if !buf.is_empty() {
            return Err(DecodeError::InvalidLength {
                expected: 0,
                actual: buf.len(),
            });
        }
        Ok(())
    }

    fn encode_obs(obs: &Self::Obs, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
        if *obs {
            Err(EncodeError::InvalidData("obs fail".into()))
        } else {
            Ok(())
        }
    }

    fn view(_state: &Self::State) -> crate::BoardView {
        // Test double: no board to project.
        crate::BoardView::from_owners(&[], 1, 0)
    }
}

#[test]
fn test_step_propagates_encoding_errors() {
    let mut adapter = GameAdapter::new(StepEncodingFailsGame);

    let mut state_buf = Vec::new();
    let mut obs_buf = Vec::new();
    adapter.reset(0, &[], &mut state_buf, &mut obs_buf).unwrap();

    let mut new_state_buf = Vec::new();
    let mut new_obs_buf = Vec::new();
    let err = adapter
        .step(&state_buf, &[], &mut new_state_buf, &mut new_obs_buf)
        .unwrap_err();

    match err {
        ErasedGameError::Encoding(msg) => assert!(msg.contains("obs fail")),
        other => panic!("expected encoding error, got {other:?}"),
    }
}
