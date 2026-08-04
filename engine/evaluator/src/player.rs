//! The two things a seat can be: a random baseline, or a model.

use algorithm_core::{BuiltinAlgorithm, ModelArtifactContract};
use anyhow::{anyhow, Result};
use dqn_runtime::{available_actions, DqnQPolicy};
use engine_core::board_profile::LegalMask;
use engine_core::{ActionAvailability, ActionSpace, EngineContext, EpisodeStatus, ErasedTimestep};
use mcts::{run_mcts, Evaluator, MctsConfig, OnnxEvaluator};
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use std::path::Path;

use crate::{canonical_evaluation_temperature, validate_onnx_intra_threads};

/// How a model picks moves.
///
/// `simulations == 0` means "no search": sample straight from the policy head,
/// which is what the trainer's Python evaluator did and therefore what keeps
/// existing eval numbers comparable. Anything above 0 runs MCTS, which is the
/// honest measure of the system's strength — the policy head alone understates
/// it (a 50-sim search went 1-19 vs random where 1-sim went 10-10 on generals,
/// before the search fixes).
pub struct ModelPlayer {
    evaluator: OnnxEvaluator,
    temperature: f32,
    simulations: u32,
    /// MCTS needs a context of its own for rollouts, separate from the one
    /// stepping the real game.
    sim_ctx: EngineContext,
    model_contract: ModelArtifactContract,
    label: String,
}

/// Fully validated AlphaZero decision input.
///
/// Keeping the complete timestep here is intentional: MCTS owns extraction of
/// its root observation and legal mask from the environment transition. The
/// direct-policy and random paths use the same validated observation rather
/// than accepting the removed scalar/observation result fields as arguments.
pub(crate) struct AlphaZeroPosition<'a> {
    pub state: &'a [u8],
    pub timestep: &'a ErasedTimestep,
    pub agent_id: engine_core::AgentId,
    pub observation: &'a [u8],
    pub legal_mask: LegalMask,
    pub action_count: usize,
}

impl<'a> AlphaZeroPosition<'a> {
    pub fn new(state: &'a [u8], timestep: &'a ErasedTimestep, action_count: usize) -> Result<Self> {
        if timestep.episode != EpisodeStatus::Running {
            return Err(anyhow!(
                "AlphaZero can only select an action from a running timestep, got {:?}",
                timestep.episode
            ));
        }

        let decision = timestep.decision.sole_agent().ok_or_else(|| {
            anyhow!(
                "AlphaZero requires exactly one active decision agent, got {:?}",
                timestep.decision
            )
        })?;
        let agent_id = decision.agent_id;
        let encoded = timestep.sole_observation()?;
        if encoded.agent_id != agent_id {
            return Err(anyhow!(
                "AlphaZero observation belongs to agent {}, but active agent is {}",
                encoded.agent_id.0,
                agent_id.0
            ));
        }

        let ActionAvailability::DiscreteMask { mask } = &decision.availability else {
            return Err(anyhow!(
                "AlphaZero requires a discrete legal mask for agent {}",
                agent_id.0
            ));
        };
        let legal_mask = mask.clone();
        if legal_mask.num_actions() != action_count {
            return Err(anyhow!(
                "AlphaZero legal mask has {} actions, action space declares {}",
                legal_mask.num_actions(),
                action_count
            ));
        }
        if legal_mask.count_ones() == 0 {
            return Err(anyhow!(
                "AlphaZero running timestep for agent {} has no legal actions",
                agent_id.0
            ));
        }

        Ok(Self {
            state,
            timestep,
            agent_id,
            observation: &encoded.data,
            legal_mask,
            action_count,
        })
    }
}

/// A seat in an evaluation match.
pub enum Player {
    Random,
    Model(Box<ModelPlayer>),
}

pub struct DqnModelPlayer {
    policy: DqnQPolicy,
    model_contract: ModelArtifactContract,
    label: String,
}

/// A policy accepted by the single-agent DQN return evaluation suite.
pub enum DqnPlayer {
    Random,
    Model(Box<DqnModelPlayer>),
}

impl DqnPlayer {
    pub fn model(
        model_contract: &ModelArtifactContract,
        model_path: &str,
        intra_threads: usize,
    ) -> Result<Self> {
        validate_onnx_intra_threads(intra_threads)?;
        let descriptor = BuiltinAlgorithm::DqnV1.descriptor();
        if model_contract.schema_version != descriptor.model_artifact_schema_version
            || model_contract.algorithm_id != descriptor.id
            || model_contract.model_contract != descriptor.components.model_contract
        {
            return Err(anyhow!(
                "Model identity is not the '{}' DQN artifact contract",
                descriptor.id
            ));
        }
        let env_id = &model_contract.env_id;
        let ctx = EngineContext::new(env_id)
            .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?;
        let capabilities = ctx.capabilities();
        if capabilities.contract_version != model_contract.env_contract_version {
            return Err(anyhow!(
                "Model targets environment '{}' contract v{}, but the registered environment is v{}",
                env_id,
                model_contract.env_contract_version,
                capabilities.contract_version
            ));
        }
        let obs_size = match &capabilities.encoding.observation {
            engine_core::ObservationEncoding::Tensor { spec } => spec
                .fixed_elements()
                .ok_or_else(|| anyhow!("DQN requires a fixed observation tensor"))?,
            other => return Err(anyhow!("DQN requires a tensor observation, got {other:?}")),
        };
        let agents = capabilities
            .agents
            .fixed_agents()
            .ok_or_else(|| anyhow!("DQN requires one fixed agent"))?;
        let [agent] = agents else {
            return Err(anyhow!(
                "DQN requires one fixed agent, got {}",
                agents.len()
            ));
        };
        let action_count = match agent.action_space {
            ActionSpace::Discrete { size } => usize::try_from(size)?,
            ref other => return Err(anyhow!("DQN requires discrete actions, got {other:?}")),
        };
        let policy = DqnQPolicy::load(
            Path::new(model_path),
            obs_size,
            action_count,
            intra_threads,
            model_contract,
        )?;
        Ok(Self::Model(Box::new(DqnModelPlayer {
            policy,
            model_contract: model_contract.clone(),
            label: format!(
                "ONNX({})",
                Path::new(model_path)
                    .file_name()
                    .unwrap_or_else(|| model_path.as_ref())
                    .to_string_lossy()
            ),
        })))
    }

    pub fn name(&self) -> String {
        match self {
            Self::Random => "Random".to_string(),
            Self::Model(model) => model.label.clone(),
        }
    }

    pub(crate) fn require_environment_profile(
        &self,
        env_id: &str,
        env_contract_version: u32,
    ) -> Result<()> {
        let Self::Model(model) = self else {
            return Ok(());
        };
        if model.model_contract.env_id != env_id
            || model.model_contract.env_contract_version != env_contract_version
        {
            return Err(anyhow!(
                "Player '{}' targets environment '{}' contract v{}, not '{}' contract v{}",
                model.label,
                model.model_contract.env_id,
                model.model_contract.env_contract_version,
                env_id,
                env_contract_version
            ));
        }
        Ok(())
    }

    pub(crate) fn select_action(
        &mut self,
        timestep: &ErasedTimestep,
        action_count: usize,
        rng: &mut ChaCha20Rng,
    ) -> Result<u32> {
        if timestep.episode != EpisodeStatus::Running {
            return Err(anyhow!("DQN can only act on a running timestep"));
        }
        let decision = timestep
            .decision
            .sole_agent()
            .ok_or_else(|| anyhow!("DQN requires exactly one active agent"))?;
        let observation = timestep.sole_observation()?;
        if observation.agent_id != decision.agent_id {
            return Err(anyhow!("DQN decision and observation agents disagree"));
        }
        let actions = available_actions(&decision.availability, action_count)?;
        match self {
            Self::Random => Ok(actions[rng.gen_range(0..actions.len())]),
            Self::Model(model) => model
                .policy
                .select_greedy(&observation.data, &decision.availability),
        }
    }
}

impl Player {
    /// Load a model player only after its artifact identity is validated.
    pub fn model(
        model_contract: &ModelArtifactContract,
        model_path: &str,
        temperature: f32,
        simulations: u32,
        intra_threads: usize,
    ) -> Result<Self> {
        let temperature = canonical_evaluation_temperature(temperature)?;
        validate_onnx_intra_threads(intra_threads)?;
        let descriptor = BuiltinAlgorithm::AlphaZeroBoardV1.descriptor();
        if model_contract.schema_version != descriptor.model_artifact_schema_version
            || model_contract.algorithm_id != descriptor.id
            || model_contract.model_contract != descriptor.components.model_contract
        {
            return Err(anyhow!(
                "Model identity is not the '{}' AlphaZero artifact contract",
                descriptor.id
            ));
        }

        let env_id = &model_contract.env_id;
        let ctx = EngineContext::new(env_id)
            .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?;
        let capabilities = ctx.capabilities();
        if capabilities.contract_version != model_contract.env_contract_version {
            return Err(anyhow!(
                "Model targets environment '{}' contract v{}, but the registered environment is v{}",
                env_id,
                model_contract.env_contract_version,
                capabilities.contract_version
            ));
        }
        let obs_size = match &capabilities.encoding.observation {
            engine_core::ObservationEncoding::Tensor { spec } => spec
                .fixed_elements()
                .ok_or_else(|| anyhow!("AlphaZero requires a fixed observation tensor"))?,
            other => {
                return Err(anyhow!(
                    "AlphaZero requires a tensor observation, got {other:?}"
                ))
            }
        };
        let action_count = match ctx.action_space(engine_core::AgentId(1)) {
            Some(ActionSpace::Discrete { size }) => size as usize,
            other => {
                return Err(anyhow!(
                    "AlphaZero requires discrete actions, got {other:?}"
                ))
            }
        };
        let evaluator = OnnxEvaluator::load_from_file(
            model_path,
            obs_size,
            action_count,
            intra_threads,
            model_contract,
        )
        .map_err(|e| anyhow!("Failed to load model '{model_path}': {e}"))?;

        Ok(Player::Model(Box::new(ModelPlayer {
            evaluator,
            temperature,
            simulations,
            sim_ctx: ctx,
            model_contract: model_contract.clone(),
            // Matches Python's ModelPlayer.name. Eval records and W&B runs key
            // off these strings, so the two must not drift.
            label: format!(
                "ONNX({})",
                Path::new(model_path)
                    .file_name()
                    .unwrap_or_else(|| model_path.as_ref())
                    .to_string_lossy()
            ),
        })))
    }

    /// Canonical policy name reported consistently by Rust and Python evaluators.
    pub fn name(&self) -> String {
        match self {
            Player::Random => "Random".to_string(),
            Player::Model(m) => m.label.clone(),
        }
    }

    /// Reject a model loaded for a different immutable environment profile.
    pub(crate) fn require_environment_profile(
        &self,
        env_id: &str,
        env_contract_version: u32,
    ) -> Result<()> {
        let Player::Model(model) = self else {
            return Ok(());
        };
        if model.model_contract.env_id != env_id
            || model.model_contract.env_contract_version != env_contract_version
        {
            return Err(anyhow!(
                "Player '{}' targets environment '{}' contract v{}, not '{}' contract v{}",
                model.label,
                model.model_contract.env_id,
                model.model_contract.env_contract_version,
                env_id,
                env_contract_version
            ));
        }
        Ok(())
    }

    /// Choose an action for the current position.
    pub(crate) fn select_action(
        &mut self,
        position: &AlphaZeroPosition<'_>,
        rng: &mut ChaCha20Rng,
    ) -> Result<u32> {
        let legal: Vec<u32> = position.legal_mask.iter_ones().map(|i| i as u32).collect();

        match self {
            Player::Random => Ok(legal[rng.gen_range(0..legal.len())]),
            Player::Model(m) if m.simulations == 0 => {
                let result = m
                    .evaluator
                    .evaluate(
                        position.observation,
                        &position.legal_mask,
                        position.action_count,
                    )
                    .map_err(|e| anyhow!("Model evaluation failed: {e}"))?;
                Ok(sample_policy(&result.policy, &legal, m.temperature, rng))
            }
            Player::Model(m) => {
                let config = MctsConfig::for_evaluation()
                    .with_simulations(m.simulations)
                    // Evaluations interleave in waves rather than all leaves
                    // being selected before any result returns; a batch larger
                    // than a quarter of the budget makes visit counts carry no
                    // value information at all.
                    .with_eval_batch_size((m.simulations as usize / 4).max(1))
                    .with_temperature(m.temperature);
                let result = run_mcts(
                    &mut m.sim_ctx,
                    &m.evaluator,
                    config,
                    position.state.to_vec(),
                    position.timestep.clone(),
                    rng,
                )?;
                Ok(result.action)
            }
        }
    }
}

/// Sample an action from `policy`, restricted to `legal`.
///
/// Temperature 0 is greedy. Above 0 the legal probabilities are raised to
/// `1/temperature` and renormalized — the standard AlphaZero play-temperature,
/// and the reason head-to-head evals do not replay one identical game.
fn sample_policy(policy: &[f32], legal: &[u32], temperature: f32, rng: &mut ChaCha20Rng) -> u32 {
    if temperature <= 0.0 {
        return *legal
            .iter()
            .max_by(|&&a, &&b| {
                policy[a as usize]
                    .partial_cmp(&policy[b as usize])
                    .expect("policy has no NaN")
            })
            .expect("legal is non-empty");
    }

    let inv = 1.0 / temperature;
    let weights: Vec<f32> = legal
        .iter()
        .map(|&a| policy[a as usize].max(0.0).powf(inv))
        .collect();
    let total: f32 = weights.iter().sum();

    // A uniformly zero policy over the legal moves (an untrained or degenerate
    // head) would otherwise sample from nothing. NaN fails both comparisons, so
    // check finiteness explicitly rather than relying on `!(total > 0.0)`.
    if total <= 0.0 || !total.is_finite() {
        return legal[rng.gen_range(0..legal.len())];
    }

    let mut point = rng.gen_range(0.0..total);
    for (i, w) in weights.iter().enumerate() {
        point -= w;
        if point <= 0.0 {
            return legal[i];
        }
    }
    *legal.last().expect("legal is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(1)
    }

    #[test]
    fn temperature_zero_picks_the_best_legal_action() {
        let policy = vec![0.1, 0.7, 0.2];
        // Action 1 is the argmax overall but illegal here.
        assert_eq!(sample_policy(&policy, &[0, 2], 0.0, &mut rng()), 2);
        assert_eq!(sample_policy(&policy, &[0, 1, 2], 0.0, &mut rng()), 1);
    }

    #[test]
    fn sampling_only_ever_returns_legal_actions() {
        let policy = vec![0.9, 0.05, 0.05];
        let mut r = rng();
        for _ in 0..200 {
            let action = sample_policy(&policy, &[1, 2], 1.0, &mut r);
            assert!(action == 1 || action == 2, "picked illegal action {action}");
        }
    }

    #[test]
    fn an_all_zero_policy_falls_back_to_uniform_instead_of_dividing_by_zero() {
        let policy = vec![0.0, 0.0, 0.0];
        let mut r = rng();
        for _ in 0..50 {
            let action = sample_policy(&policy, &[0, 2], 1.0, &mut r);
            assert!(action == 0 || action == 2);
        }
    }

    #[test]
    fn low_temperature_concentrates_on_the_favourite() {
        let policy = vec![0.6, 0.4];
        let mut r = rng();
        let favourite = (0..400)
            .filter(|_| sample_policy(&policy, &[0, 1], 0.1, &mut r) == 0)
            .count();
        assert!(
            favourite > 380,
            "temperature 0.1 should be near-greedy, got {favourite}/400"
        );
    }
}
