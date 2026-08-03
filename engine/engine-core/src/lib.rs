//! Core traits and types for the Cartridge game engine
//!
//! This crate provides the fundamental abstractions for game simulation:
//! - `Environment`: generic typed environment contract
//! - a sealed, validated bytes-only runtime boundary
//! - `BoardGame`: explicitly narrow adapter contract for bundled board games
//! - `EngineContext`: high-level API for running environments

mod adapter;
mod board_game;
mod board_game_utils;
mod board_view;
pub mod context;
mod contract;
mod erased;
mod legal_mask;
mod metadata;
mod registry;
pub mod typed;

// Re-export main types for convenience
pub use board_view::Presentation;
pub use context::{EngineContext, EngineContextError, ResetResult, StepResult};
pub use erased::{EncodedObservation, ErasedEnvironmentError, ErasedTimestep};
pub use metadata::EnvironmentMetadata;
pub use registry::{
    is_registered, list_registered_environments, register_environment, RegistryError,
};
pub use typed::{
    ActionEncoding, ActionSpace, AgentId, AgentModel, AgentObservation, AgentOutcome, AgentSpec,
    Capabilities, ChanceModel, Decision, DecodeError, EncodeError, Encoding, EngineId, Environment,
    EnvironmentError, EnvironmentSemantics, EpisodeStatus, InformationModel, ObservationEncoding,
    PlanningStateModel, RewardModel, SequentialTurnOrder, Timestep, TimestepAccessError,
    TransitionDynamics, TransitionSource, TurnModel, WIRE_ENCODING_SCHEMA_VERSION,
};

/// Explicitly narrow types and helpers for the bundled two-seat board profile.
pub mod board_profile {
    pub use crate::board_game::{BoardGame, BoardTransition, TwoPlayerObs, TwoPlayerObsError};
    pub use crate::board_game_utils::{
        calculate_reward, decode_action_u32, encode_f32_slices, opponent, validate_board_cells,
        validate_player_and_winner,
    };
    pub use crate::board_view::{BoardView, CellKind, CellView};
    pub use crate::legal_mask::{LegalMask, LegalMaskError};
    pub use crate::metadata::{
        BoardGameMetadata, BoardObservationMetadata, BoardPlayerMetadata, BoardRenderer,
        MetadataError,
    };
    pub use crate::registry::register_board_game;
}

/// Test utilities (internal use only)
#[cfg(test)]
pub(crate) mod test_utils {
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    /// Global mutex to serialize all registry-dependent tests
    pub static REGISTRY_TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
}
