//! Typed, algorithm-neutral environment contract.
//!
//! The contract models per-agent observations and outcomes, simultaneous or
//! chance decisions, and termination separately from truncation. Narrow game
//! families (such as the bundled two-player board games) adapt into this ABI;
//! they do not define it.

use crate::board_view::Presentation;
use crate::legal_mask::LegalMask;
use crate::metadata::EnvironmentMetadata;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineId {
    pub env_id: String,
    pub build_id: String,
}

pub const WIRE_ENCODING_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionEncoding {
    /// Exactly one unsigned 32-bit action index in little-endian byte order.
    DiscreteU32LittleEndian,
    /// One unsigned 32-bit index per declared dimension, in declaration order.
    MultiDiscreteU32LittleEndian,
    /// One IEEE-754 `f32` per flattened action element, in row-major order.
    ContinuousF32LittleEndian,
    /// Environment-defined bytes identified by a stable, non-empty codec ID.
    Custom { id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservationEncoding {
    Tensor { spec: TensorSpec },
    Custom { id: String },
}

/// Scalar representation used by a tensor observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    F32LittleEndian,
    U8,
    I64LittleEndian,
    Bool,
}

impl TensorDType {
    pub const fn element_size(self) -> usize {
        match self {
            Self::F32LittleEndian => std::mem::size_of::<f32>(),
            Self::U8 | Self::Bool => 1,
            Self::I64LittleEndian => std::mem::size_of::<i64>(),
        }
    }
}

/// One named tensor dimension. `size=None` declares a dynamic dimension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDimension {
    pub name: String,
    pub size: Option<u32>,
}

impl TensorDimension {
    pub fn fixed(name: impl Into<String>, size: u32) -> Self {
        Self {
            name: name.into(),
            size: Some(size),
        }
    }

    pub fn dynamic(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size: None,
        }
    }
}

/// Shape and scalar encoding for one agent observation tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub dtype: TensorDType,
    pub dimensions: Vec<TensorDimension>,
}

impl TensorSpec {
    pub fn f32_fixed<const N: usize>(dimensions: [(&str, u32); N]) -> Self {
        Self {
            dtype: TensorDType::F32LittleEndian,
            dimensions: dimensions
                .into_iter()
                .map(|(name, size)| TensorDimension::fixed(name, size))
                .collect(),
        }
    }

    /// Number of scalar elements when every dimension is fixed.
    pub fn fixed_elements(&self) -> Option<usize> {
        self.dimensions
            .iter()
            .try_fold(1usize, |elements, dimension| {
                elements.checked_mul(usize::try_from(dimension.size?).ok()?)
            })
    }

    /// Exact encoded byte count when every dimension is fixed.
    pub fn fixed_bytes(&self) -> Option<usize> {
        self.fixed_elements()?
            .checked_mul(self.dtype.element_size())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    pub state: String,
    pub action: ActionEncoding,
    /// Codec for one [`AgentObservation`], not for the timestep envelope.
    pub observation: ObservationEncoding,
    pub schema_version: u32,
}

impl Encoding {
    pub fn discrete_u32_le(state: impl Into<String>, observation: TensorSpec) -> Self {
        Self {
            state: state.into(),
            action: ActionEncoding::DiscreteU32LittleEndian,
            observation: ObservationEncoding::Tensor { spec: observation },
            schema_version: WIRE_ENCODING_SCHEMA_VERSION,
        }
    }

    pub fn multi_discrete_u32_le(state: impl Into<String>, observation: TensorSpec) -> Self {
        Self {
            action: ActionEncoding::MultiDiscreteU32LittleEndian,
            ..Self::discrete_u32_le(state, observation)
        }
    }

    pub fn continuous_f32_le(state: impl Into<String>, observation: TensorSpec) -> Self {
        Self {
            action: ActionEncoding::ContinuousF32LittleEndian,
            ..Self::discrete_u32_le(state, observation)
        }
    }

    pub fn custom(
        state: impl Into<String>,
        action: impl Into<String>,
        observation: impl Into<String>,
    ) -> Self {
        Self {
            state: state.into(),
            action: ActionEncoding::Custom { id: action.into() },
            observation: ObservationEncoding::Custom {
                id: observation.into(),
            },
            schema_version: WIRE_ENCODING_SCHEMA_VERSION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequentialTurnOrder {
    Alternating,
    EnvironmentDefined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnModel {
    SingleAgent,
    Sequential { order: SequentialTurnOrder },
    Simultaneous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InformationModel {
    PerfectInformationMarkov,
    PartiallyObserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningStateModel {
    /// Encoded state contains everything needed to reproduce future
    /// transitions. A step must not depend on hidden mutable state.
    CompleteSnapshot,
    /// Transition-relevant state exists outside the encoded state bytes, so
    /// callers cannot branch or replay solely from a saved snapshot.
    ExternalState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionDynamics {
    Deterministic,
    Stochastic,
}

/// How stochastic chance is exposed at the decision boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChanceModel {
    None,
    /// Chance appears as [`Decision::Chance`] and is resolved explicitly.
    Explicit,
    /// The environment samples chance internally during reset or step.
    EnvironmentSampled,
}

/// Reward semantics. Rewards themselves are always emitted per agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardModel {
    /// Rewards are zero before global termination and sum to zero on the
    /// terminal timestep. The runtime validates both invariants.
    TerminalZeroSum,
    /// Finite per-agent rewards with no additional structural restriction.
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentSemantics {
    pub turn_model: TurnModel,
    pub information_model: InformationModel,
    pub planning_state_model: PlanningStateModel,
    pub transition_dynamics: TransitionDynamics,
    pub chance_model: ChanceModel,
    pub reward_model: RewardModel,
}

impl EnvironmentSemantics {
    pub const fn deterministic_alternating_perfect_information_terminal_zero_sum() -> Self {
        Self {
            turn_model: TurnModel::Sequential {
                order: SequentialTurnOrder::Alternating,
            },
            information_model: InformationModel::PerfectInformationMarkov,
            planning_state_model: PlanningStateModel::CompleteSnapshot,
            transition_dynamics: TransitionDynamics::Deterministic,
            chance_model: ChanceModel::None,
            reward_model: RewardModel::TerminalZeroSum,
        }
    }

    pub const fn deterministic_single_agent_general_reward() -> Self {
        Self {
            turn_model: TurnModel::SingleAgent,
            information_model: InformationModel::PerfectInformationMarkov,
            planning_state_model: PlanningStateModel::CompleteSnapshot,
            transition_dynamics: TransitionDynamics::Deterministic,
            chance_model: ChanceModel::None,
            reward_model: RewardModel::General,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionSpace {
    Discrete {
        size: u32,
    },
    MultiDiscrete {
        dimensions: Vec<u32>,
    },
    Continuous {
        low: Vec<f32>,
        high: Vec<f32>,
        shape: Vec<u32>,
    },
}

impl ActionSpace {
    pub const fn discrete(size: u32) -> Self {
        Self::Discrete { size }
    }
}

/// Stable ID assigned by an environment to one participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentId(pub u32);

impl From<u8> for AgentId {
    fn from(value: u8) -> Self {
        Self(u32::from(value))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub id: AgentId,
    pub action_space: ActionSpace,
    pub action_availability: ActionAvailabilityContract,
}

/// Availability representation an environment promises for one agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionAvailabilityContract {
    All,
    DiscreteMask,
    Custom { id: String },
}

/// Agent population and action spaces for an environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentModel {
    Fixed {
        agents: Vec<AgentSpec>,
    },
    /// Dynamic agents share one action-space contract.
    Dynamic {
        action_space: ActionSpace,
        action_availability: ActionAvailabilityContract,
    },
}

impl AgentModel {
    pub fn fixed_homogeneous(
        ids: impl IntoIterator<Item = AgentId>,
        action_space: ActionSpace,
    ) -> Self {
        Self::Fixed {
            agents: ids
                .into_iter()
                .map(|id| AgentSpec {
                    id,
                    action_space: action_space.clone(),
                    action_availability: ActionAvailabilityContract::All,
                })
                .collect(),
        }
    }

    pub fn fixed_homogeneous_masked(
        ids: impl IntoIterator<Item = AgentId>,
        action_space: ActionSpace,
    ) -> Self {
        Self::Fixed {
            agents: ids
                .into_iter()
                .map(|id| AgentSpec {
                    id,
                    action_space: action_space.clone(),
                    action_availability: ActionAvailabilityContract::DiscreteMask,
                })
                .collect(),
        }
    }

    pub fn action_space(&self, agent_id: AgentId) -> Option<&ActionSpace> {
        match self {
            Self::Fixed { agents } => agents
                .iter()
                .find(|agent| agent.id == agent_id)
                .map(|agent| &agent.action_space),
            Self::Dynamic { action_space, .. } => Some(action_space),
        }
    }

    pub fn action_availability(&self, agent_id: AgentId) -> Option<&ActionAvailabilityContract> {
        match self {
            Self::Fixed { agents } => agents
                .iter()
                .find(|agent| agent.id == agent_id)
                .map(|agent| &agent.action_availability),
            Self::Dynamic {
                action_availability,
                ..
            } => Some(action_availability),
        }
    }

    pub fn fixed_agents(&self) -> Option<&[AgentSpec]> {
        match self {
            Self::Fixed { agents } => Some(agents),
            Self::Dynamic { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub id: EngineId,
    /// Immutable environment-contract revision under `id.env_id`.
    pub contract_version: u32,
    pub encoding: Encoding,
    pub semantics: EnvironmentSemantics,
    /// `None` represents a continuing environment with no declared horizon.
    pub max_horizon: Option<u32>,
    pub agents: AgentModel,
    pub preferred_batch: u32,
}

impl Capabilities {
    pub fn action_space(&self, agent_id: AgentId) -> Option<&ActionSpace> {
        self.agents.action_space(agent_id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentObservation<O> {
    pub agent_id: AgentId,
    pub observation: O,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentOutcome {
    pub agent_id: AgentId,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeStatus {
    Running,
    Terminated,
    Truncated,
}

impl EpisodeStatus {
    pub const fn is_done(self) -> bool {
        !matches!(self, Self::Running)
    }
}

/// Concrete availability accompanying one requested agent action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionAvailability {
    All,
    DiscreteMask { mask: LegalMask },
    Custom { contract: String, data: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentDecision {
    pub agent_id: AgentId,
    pub availability: ActionAvailability,
}

/// Who must supply the next action. `Agents` may contain more than one entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Decision {
    Agents { decisions: Vec<AgentDecision> },
    Chance,
    None,
}

impl Decision {
    pub fn agents(agent_ids: impl IntoIterator<Item = AgentId>) -> Self {
        Self::Agents {
            decisions: agent_ids
                .into_iter()
                .map(|agent_id| AgentDecision {
                    agent_id,
                    availability: ActionAvailability::All,
                })
                .collect(),
        }
    }

    pub fn single(agent_id: AgentId, availability: ActionAvailability) -> Self {
        Self::Agents {
            decisions: vec![AgentDecision {
                agent_id,
                availability,
            }],
        }
    }

    pub fn sole_agent(&self) -> Option<&AgentDecision> {
        match self {
            Self::Agents { decisions } if decisions.len() == 1 => decisions.first(),
            _ => None,
        }
    }
}

/// Provenance of the transition that produced a timestep.
///
/// In a dynamic environment, a departing source agent remains in the
/// resulting [`Timestep::agents`] transition roster so its final outcome and
/// provenance stay locally valid. It must not appear in the next decision and
/// is omitted from the following timestep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TransitionSource {
    Reset,
    Agents { agent_ids: Vec<AgentId> },
    Chance,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Timestep<O> {
    /// Agent population represented by this transition envelope.
    ///
    /// Fixed environments emit their declared population. Dynamic
    /// environments emit newly/currently active agents plus any departing
    /// agent on the transition that terminates or truncates it. A departing
    /// agent is excluded from [`Self::decision`] immediately, then omitted from
    /// the following timestep. This is deliberately a transition roster, not
    /// only the post-step live roster.
    pub agents: Vec<AgentId>,
    pub observations: Vec<AgentObservation<O>>,
    pub outcomes: Vec<AgentOutcome>,
    pub decision: Decision,
    pub episode: EpisodeStatus,
    pub source: TransitionSource,
    pub info: Vec<u8>,
}

impl<O> Timestep<O> {
    pub fn reward_for(&self, agent_id: AgentId) -> Option<f32> {
        self.outcomes
            .iter()
            .find(|outcome| outcome.agent_id == agent_id)
            .map(|outcome| outcome.reward)
    }

    pub fn observation_for(&self, agent_id: AgentId) -> Option<&O> {
        self.observations
            .iter()
            .find(|observation| observation.agent_id == agent_id)
            .map(|observation| &observation.observation)
    }

    pub fn sole_observation(&self) -> Result<&AgentObservation<O>, TimestepAccessError> {
        match self.observations.as_slice() {
            [observation] => Ok(observation),
            observations => Err(TimestepAccessError::ExpectedOneObservation {
                actual: observations.len(),
            }),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TimestepAccessError {
    #[error("expected exactly one observation, got {actual}")]
    ExpectedOneObservation { actual: usize },
}

/// Main typed interface implemented by general environments.
pub trait Environment: Send + Sync + std::fmt::Debug + 'static {
    type State: Send + Sync + 'static;
    /// May be a joint action when [`TurnModel::Simultaneous`] is declared.
    type Action: Send + Sync + 'static;
    type Observation: Send + Sync + 'static;

    fn engine_id(&self) -> EngineId;
    fn capabilities(&self) -> Capabilities;
    fn metadata(&self) -> EnvironmentMetadata;
    /// Optional labels/targets for finite discrete actions. Other action spaces
    /// and environments without presentation support return None.
    fn describe_discrete_action(
        &self,
        _agent: AgentId,
        _action: u32,
    ) -> Option<crate::ActionPresentation> {
        None
    }

    fn reset(
        &mut self,
        rng: &mut ChaCha20Rng,
        hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError>;

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError>;

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError>;
    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError>;
    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError>;
    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError>;
    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError>;

    fn presentation(_state: &Self::State) -> Option<Presentation> {
        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("invalid reset hint: {0}")]
    InvalidHint(String),
    #[error("invalid environment state: {0}")]
    InvalidState(String),
    #[error("invalid environment action: {0}")]
    InvalidAction(String),
    #[error("environment transition failed: {0}")]
    Transition(String),
}

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("failed to encode data: {0}")]
    SerializationError(String),
    #[error("buffer too small, needed {needed} bytes but got {available}")]
    BufferTooSmall { needed: usize, available: usize },
    #[error("invalid data: {0}")]
    InvalidData(String),
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("failed to decode data: {0}")]
    DeserializationError(String),
    #[error("invalid buffer length: expected {expected} but got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("corrupted data: {0}")]
    CorruptedData(String),
    #[error("unsupported version: {version}")]
    UnsupportedVersion { version: u32 },
}

#[cfg(test)]
#[path = "typed_tests.rs"]
mod tests;
