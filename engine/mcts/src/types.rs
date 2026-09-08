//! Plain data types used by the MCTS search implementation.
//!
//! These are extracted from `search.rs` and re-exported there (and from the
//! crate root) so the public API is unchanged.

use engine_core::board_profile::LegalMask;
use thiserror::Error;

use crate::evaluator::EvaluatorError;
use crate::node::NodeId;
use crate::MctsConfigError;

/// A leaf node waiting for neural network evaluation.
pub(crate) struct PendingLeaf {
    pub(crate) node_id: NodeId,
    pub(crate) state: Vec<u8>,
    pub(crate) obs: Vec<u8>,
    pub(crate) legal_mask: LegalMask,
}

/// Result of selecting a leaf node during MCTS simulation.
pub(crate) enum LeafResult {
    /// Terminal node - has known value, no NN evaluation needed.
    Terminal { node_id: NodeId, value: f32 },
    /// Needs neural network evaluation before expansion.
    NeedsEvaluation {
        node_id: NodeId,
        state: Vec<u8>,
        obs: Vec<u8>,
        legal_mask: LegalMask,
    },
    /// Already expanded (edge case - shouldn't normally happen).
    AlreadyExpanded,
}

/// Errors that can occur during MCTS search.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("Invalid MCTS configuration: {0}")]
    InvalidConfig(#[from] MctsConfigError),

    #[error("Engine error: {0}")]
    EngineError(String),

    #[error("Evaluator error: {0}")]
    EvaluatorError(#[from] EvaluatorError),

    #[error("No legal moves available")]
    NoLegalMoves,

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Invalid AlphaZero timestep: {0}")]
    InvalidTimestep(String),

    #[error("Environment is incompatible with AlphaZero MCTS: {0}")]
    IncompatibleEnvironment(String),

    #[error("Legal mask width mismatch: expected {expected} actions, got {actual}")]
    LegalMaskWidthMismatch { expected: usize, actual: usize },

    #[error("MCTS only supports discrete action spaces")]
    UnsupportedActionSpace,
}

/// Result of an MCTS search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Best action to take
    pub action: u32,

    /// Training target: the root visit distribution at temperature 1
    /// (raw visit counts, normalized).
    ///
    /// Deliberately NOT sharpened by [`MctsConfig::temperature`]. That
    /// temperature governs which action is *played*; the stored target must
    /// keep the search's full relative-visit information. Raising counts to
    /// `1/tau` for a small tau destroys it — the actor's late-game tau of
    /// 0.1 means `counts^10`, so two moves with 12 and 9 visits become a
    /// 17.8:1 target and a typical 250-simulation search collapses onto a
    /// near one-hot vector. That throws away the soft-target signal
    /// AlphaZero policy learning depends on, and is worst in long games
    /// where the low-temperature phase covers almost the whole episode.
    pub policy: Vec<f32>,

    /// Value estimate at root
    pub value: f32,

    /// Number of simulations performed
    pub simulations: u32,

    /// Performance statistics for this search
    pub stats: SearchStats,
}

/// Performance statistics for an MCTS search.
/// All times are in microseconds for precision.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Total wall-clock time for the search
    pub total_time_us: u64,
    /// Time spent in tree selection (traversing to leaves)
    pub selection_time_us: u64,
    /// Time spent in neural network inference
    pub inference_time_us: u64,
    /// Time spent expanding nodes (running game simulations)
    pub expansion_time_us: u64,
    /// Time spent in backpropagation
    pub backprop_time_us: u64,
    /// Number of batched NN calls made
    pub num_batches: u32,
    /// Total observations evaluated (at most `simulations`; terminal hits
    /// and already-expanded leaves complete without an NN call)
    pub total_evals: u32,
    /// Number of game step() calls during expansion
    pub game_steps: u32,
    /// Number of terminal nodes hit (no NN needed)
    pub terminal_hits: u32,
}

/// Optional root-only diagnostics. All action values use the root actor's
/// perspective. These describe one search, not calibrated win probabilities.
#[derive(Debug, Clone)]
pub struct RootDiagnostics {
    pub network_value: f32,
    pub actions: Vec<ActionDiagnostics>,
    pub completed_simulations: u32,
    pub root_visits: u32,
    /// Includes the initial root evaluation omitted by SearchStats.total_evals.
    pub neural_evaluations: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub struct ActionDiagnostics {
    pub action: u32,
    pub network_prior: f32,
    pub search_prior: Option<f32>,
    pub visit_share: f32,
    pub selection_probability: f32,
    pub visits: u32,
    /// None means unvisited, not an estimated draw/neutral value.
    pub q_value: Option<f32>,
    pub expanded: bool,
}
