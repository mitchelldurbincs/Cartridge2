//! MCTS search implementation.
//!
//! Implements the core MCTS algorithm:
//! 1. Selection: Traverse tree using UCB to find a leaf
//! 2. Expansion: Add children to the leaf using policy prior
//! 3. Evaluation: Get value estimate from evaluator
//! 4. Backpropagation: Update statistics along the path

use std::time::Instant;

use algorithm_core::BuiltinAlgorithm;
use engine_core::board_profile::LegalMask;
use engine_core::{
    ActionAvailability, ActionSpace, AgentId, EngineContext, EpisodeStatus, ErasedTimestep,
    TransitionSource,
};
use rand_chacha::ChaCha20Rng;
use tracing::debug;

use crate::config::MctsConfig;
use crate::evaluator::Evaluator;
use crate::node::NodeId;
use crate::sampling::{dirichlet_noise, sample_action};
use crate::tree::MctsTree;
use crate::types::{ActionDiagnostics, LeafResult, PendingLeaf, RootDiagnostics};

// Re-export the plain data types so the public API is unchanged.
pub use crate::types::{SearchError, SearchResult, SearchStats};

/// MCTS search state.
pub struct MctsSearch<'a, E: Evaluator> {
    tree: MctsTree,
    ctx: &'a mut EngineContext,
    evaluator: &'a E,
    config: MctsConfig,
    num_actions: usize,
    root_state: Vec<u8>,
    root_obs: Vec<u8>,
    step_state_buf: Vec<u8>,
    step_timestep_buf: ErasedTimestep,
}

impl<'a, E: Evaluator> MctsSearch<'a, E> {
    /// Create a new MCTS search from the given game state.
    pub fn new(
        ctx: &'a mut EngineContext,
        evaluator: &'a E,
        config: MctsConfig,
        state: Vec<u8>,
        timestep: ErasedTimestep,
    ) -> Result<Self, SearchError> {
        config.validate()?;
        BuiltinAlgorithm::AlphaZeroBoardV1
            .compatibility(ctx)
            .require_compatible()
            .map_err(|error| SearchError::IncompatibleEnvironment(error.to_string()))?;
        let active = timestep.decision.sole_agent().ok_or_else(|| {
            SearchError::InvalidTimestep(format!(
                "AlphaZero requires one active agent, got {:?}",
                timestep.decision
            ))
        })?;
        let active_agent = active.agent_id;
        let obs = timestep
            .observation_for(active_agent)
            .ok_or_else(|| {
                SearchError::InvalidTimestep(format!(
                    "missing observation for active agent {}",
                    active_agent.0
                ))
            })?
            .to_vec();
        let num_actions = match ctx.action_space(active_agent) {
            Some(ActionSpace::Discrete { size }) => size as usize,
            _ => return Err(SearchError::UnsupportedActionSpace),
        };
        let legal_moves_mask = decision_legal_mask(active, num_actions)?.clone();
        if legal_moves_mask.num_actions() != num_actions {
            return Err(SearchError::LegalMaskWidthMismatch {
                expected: num_actions,
                actual: legal_moves_mask.num_actions(),
            });
        }

        let tree = MctsTree::new(legal_moves_mask);
        let state_capacity = state.len().max(64);
        Ok(Self {
            tree,
            ctx,
            evaluator,
            config,
            num_actions,
            root_state: state,
            root_obs: obs,
            step_state_buf: Vec::with_capacity(state_capacity),
            step_timestep_buf: ErasedTimestep::default(),
        })
    }

    /// Run the MCTS search for the configured number of simulations.
    ///
    /// Uses batched neural network evaluation for efficiency. The effective
    /// batch is capped at one quarter of the simulation budget so selection
    /// interleaves with evaluated values instead of queuing the entire search
    /// from priors and virtual loss alone.
    pub fn run(&mut self, rng: &mut ChaCha20Rng) -> Result<SearchResult, SearchError> {
        self.run_internal(rng, false).map(|(result, _)| result)
    }

    /// Inspect a fresh search without additional inference or RNG draws.
    /// Existing collectors can keep using run(), which skips diagnostics.
    pub fn run_with_diagnostics(
        &mut self,
        rng: &mut ChaCha20Rng,
    ) -> Result<(SearchResult, RootDiagnostics), SearchError> {
        if self.tree.get(self.tree.root()).is_expanded() {
            return Err(SearchError::InvalidState(
                "root diagnostics require a fresh search".into(),
            ));
        }
        let (result, diagnostics) = self.run_internal(rng, true)?;
        Ok((
            result,
            diagnostics.expect("fresh analyzed search retains root evaluation"),
        ))
    }

    fn run_internal(
        &mut self,
        rng: &mut ChaCha20Rng,
        inspect: bool,
    ) -> Result<(SearchResult, Option<RootDiagnostics>), SearchError> {
        let search_start = Instant::now();
        let mut stats = SearchStats::default();
        let mut root_evaluation = None;

        // First, expand the root if needed (single NN call, special case).
        // Backpropagate the root evaluation so the root starts with one
        // visit: with zero parent visits the UCB exploration term is
        // c*prior*sqrt(0) = 0 for every child, making the first selection a
        // pure tie broken by child order instead of by prior.
        if !self.tree.get(self.tree.root()).is_expanded() {
            let root_id = self.tree.root();
            let legal_mask = self.tree.get(root_id).legal_moves_mask.clone();
            let root_state = self.root_state.clone();
            let root_obs = self.root_obs.clone();
            let eval = self
                .evaluator
                .evaluate(&root_obs, &legal_mask, self.num_actions)?;
            self.expand_node_with_eval_counted(root_id, &root_state, &legal_mask, &eval)?;
            self.tree.backpropagate(root_id, eval.value);
            if inspect {
                root_evaluation = Some(eval);
            }
        }

        // Add Dirichlet noise to root if configured
        if self.config.dirichlet_alpha > 0.0 {
            self.add_dirichlet_noise(rng);
        }

        // Batched simulation loop. Cap the batch to a quarter of the
        // simulation budget so evaluations interleave with selection: if
        // every simulation were collected before the first evaluation
        // returned, visit counts would be driven purely by priors and
        // virtual loss, and the search would never use its value estimates.
        let target_simulations = self.config.num_simulations;
        let batch_size = effective_eval_batch_size(&self.config);
        let mut pending: Vec<PendingLeaf> = Vec::with_capacity(batch_size);
        let mut completed_simulations: u32 = 0;

        while completed_simulations < target_simulations {
            // Selection phase: collect leaves until batch is full or we've queued enough
            let selection_start = Instant::now();
            let remaining = target_simulations - completed_simulations - pending.len() as u32;
            while pending.len() < batch_size && remaining > 0 {
                match self.select_leaf()? {
                    LeafResult::Terminal { node_id, value } => {
                        // Terminal nodes don't need NN - backprop immediately
                        let backprop_start = Instant::now();
                        self.tree.backpropagate(node_id, value);
                        stats.backprop_time_us += backprop_start.elapsed().as_micros() as u64;
                        completed_simulations += 1;
                        stats.terminal_hits += 1;
                    }
                    LeafResult::NeedsEvaluation {
                        node_id,
                        state,
                        obs,
                        legal_mask,
                    } => {
                        pending.push(PendingLeaf {
                            node_id,
                            state,
                            obs,
                            legal_mask,
                        });
                    }
                    LeafResult::AlreadyExpanded => {
                        // Rare edge case - count as completed but no backprop
                        completed_simulations += 1;
                    }
                }

                // Recalculate remaining after each selection
                let total_in_flight = completed_simulations + pending.len() as u32;
                if total_in_flight >= target_simulations {
                    break;
                }
            }
            stats.selection_time_us += selection_start.elapsed().as_micros() as u64;

            // Evaluation phase: batch evaluate all pending leaves
            if !pending.is_empty() {
                let batch_count = pending.len() as u32;
                self.evaluate_and_expand_batch_with_stats(&pending, &mut stats)?;
                completed_simulations += batch_count;
                stats.total_evals += batch_count;
                stats.num_batches += 1;
                pending.clear();
            }
        }

        // Record total time
        stats.total_time_us = search_start.elapsed().as_micros() as u64;

        // Log stats at debug level
        debug!(
            total_ms = stats.total_time_us as f64 / 1000.0,
            selection_ms = stats.selection_time_us as f64 / 1000.0,
            inference_ms = stats.inference_time_us as f64 / 1000.0,
            expansion_ms = stats.expansion_time_us as f64 / 1000.0,
            backprop_ms = stats.backprop_time_us as f64 / 1000.0,
            num_batches = stats.num_batches,
            total_evals = stats.total_evals,
            game_steps = stats.game_steps,
            terminal_hits = stats.terminal_hits,
            avg_batch_size = stats
                .total_evals
                .checked_div(stats.num_batches)
                .unwrap_or(0),
            "MCTS search stats"
        );

        // Extract result. The returned policy is the training target and is
        // always the raw visit distribution (tau = 1); the configured
        // temperature only decides which action we play from it. See
        // `SearchResult::policy`.
        let root = self.tree.get(self.tree.root());
        let policy = self.tree.root_policy(self.num_actions, 1.0);

        let action = if self.config.temperature < 1e-6 {
            // Greedy
            self.tree
                .best_action()
                .map(|(a, _)| a)
                .ok_or(SearchError::NoLegalMoves)?
        } else if (self.config.temperature - 1.0).abs() < 1e-6 {
            // Play temperature matches the target; sample from it directly.
            sample_action(&policy, rng)?
        } else {
            // Sample from the temperature-scaled distribution, leaving the
            // stored target untouched.
            let play_policy = self
                .tree
                .root_policy(self.num_actions, self.config.temperature);
            sample_action(&play_policy, rng)?
        };

        // Reconstruct only for diagnostics, using the exact same engine helper
        // and tie-breaking as selection. This consumes no randomness.
        let diagnostics = root_evaluation.map(|eval| {
            let selection_temperature = if (self.config.temperature - 1.0).abs() < 1e-6 {
                1.0
            } else {
                self.config.temperature
            };
            let selection = self
                .tree
                .root_policy(self.num_actions, selection_temperature);
            let actions = root
                .legal_moves_mask
                .iter_ones()
                .map(|action| {
                    let child = root
                        .children
                        .iter()
                        .find(|(id, _)| *id == action as u32)
                        .map(|(_, id)| self.tree.get(*id));
                    ActionDiagnostics {
                        action: action as u32,
                        network_prior: eval.policy[action],
                        search_prior: child.map(|node| node.prior),
                        visit_share: policy[action],
                        selection_probability: selection[action],
                        visits: child.map_or(0, |node| node.visit_count),
                        q_value: child
                            .filter(|node| node.visit_count > 0)
                            .map(|node| -node.mean_value()),
                        expanded: child.is_some(),
                    }
                })
                .collect();
            RootDiagnostics {
                network_value: eval.value,
                actions,
                completed_simulations,
                root_visits: root.visit_count,
                neural_evaluations: stats.total_evals + 1,
                temperature: self.config.temperature,
            }
        });

        Ok((
            SearchResult {
                action,
                policy,
                value: root.mean_value(),
                simulations: root.visit_count,
                stats,
            },
            diagnostics,
        ))
    }

    /// Select a leaf node by traversing the tree using UCB.
    fn select(&self) -> (NodeId, Vec<NodeId>) {
        let mut path = vec![self.tree.root()];
        let mut current = self.tree.root();

        loop {
            let node = self.tree.get(current);

            // Stop at terminal or unexpanded nodes
            if node.is_terminal || !node.is_expanded() {
                break;
            }

            // Select best child
            match self.tree.select_child(current, self.config.c_puct) {
                Some(child_id) => {
                    path.push(child_id);
                    current = child_id;
                }
                None => break, // No children (shouldn't happen if expanded)
            }
        }

        (current, path)
    }

    /// Select a leaf node and return its evaluation requirements.
    /// This is used in batched evaluation mode to collect leaves before evaluating.
    fn select_leaf(&mut self) -> Result<LeafResult, SearchError> {
        let (leaf_id, path) = self.select();

        // Check terminal/expanded status first (no mask cloning for early returns)
        let (is_terminal, terminal_value, is_expanded) = {
            let leaf = self.tree.get(leaf_id);
            (leaf.is_terminal, leaf.terminal_value, leaf.is_expanded())
        };

        if is_terminal {
            return Ok(LeafResult::Terminal {
                node_id: leaf_id,
                value: terminal_value,
            });
        }

        if is_expanded {
            return Ok(LeafResult::AlreadyExpanded);
        }

        let legal_mask = self.tree.get(leaf_id).legal_moves_mask.clone();

        // Reconstruct state/obs for this path on demand.
        let (state, obs) = self.reconstruct_position(&path)?;

        // Apply virtual loss to discourage selecting this node again before
        // it's been evaluated (prevents duplicate selection in a batch).
        // Nodes store values from their own (opponent-of-parent) perspective
        // and UCB negates them, so making the node LESS attractive to its
        // parent means RAISING its value sum.
        let leaf_mut = self.tree.get_mut(leaf_id);
        leaf_mut.visit_count += 1;
        leaf_mut.value_sum += self.config.virtual_loss;

        Ok(LeafResult::NeedsEvaluation {
            node_id: leaf_id,
            state,
            obs,
            legal_mask,
        })
    }

    /// Reconstruct state/observation at a path by replaying actions from root.
    fn reconstruct_position(&mut self, path: &[NodeId]) -> Result<(Vec<u8>, Vec<u8>), SearchError> {
        let mut state = self.root_state.clone();
        let mut obs = self.root_obs.clone();

        for &node_id in path.iter().skip(1) {
            let action_bytes = self.tree.get(node_id).action.to_le_bytes();
            self.ctx
                .step_into(
                    &state,
                    &action_bytes,
                    &mut self.step_state_buf,
                    &mut self.step_timestep_buf,
                )
                .map_err(|e| SearchError::EngineError(e.to_string()))?;

            std::mem::swap(&mut state, &mut self.step_state_buf);
            obs = decision_observation(&self.step_timestep_buf)?.to_vec();
        }

        Ok((state, obs))
    }

    /// Evaluate and expand a batch of pending leaves with stats tracking.
    /// Makes a single batched NN call for all leaves, then expands and backpropagates each.
    fn evaluate_and_expand_batch_with_stats(
        &mut self,
        pending: &[PendingLeaf],
        stats: &mut SearchStats,
    ) -> Result<(), SearchError> {
        if pending.is_empty() {
            return Ok(());
        }

        // Prepare batch inputs
        let observations: Vec<&[u8]> = pending.iter().map(|p| p.obs.as_slice()).collect();
        let legal_masks: Vec<&LegalMask> = pending.iter().map(|p| &p.legal_mask).collect();

        // Single batched NN call - track inference time
        let inference_start = Instant::now();
        let results =
            self.evaluator
                .evaluate_batch(&observations, &legal_masks, self.num_actions)?;
        stats.inference_time_us += inference_start.elapsed().as_micros() as u64;

        // Expand each node with its result and backpropagate
        for (leaf, eval) in pending.iter().zip(results.iter()) {
            // Remove virtual loss before applying real values
            let node = self.tree.get_mut(leaf.node_id);
            node.visit_count -= 1;
            node.value_sum -= self.config.virtual_loss;

            // Expand the node with children - track expansion time and game
            // steps. A leaf can appear more than once in a batch (it stays
            // unexpanded while pending); only the first occurrence expands,
            // or children would be duplicated.
            let expansion_start = Instant::now();
            let steps = if !self.tree.get(leaf.node_id).is_expanded() {
                self.expand_node_with_eval_counted(
                    leaf.node_id,
                    &leaf.state,
                    &leaf.legal_mask,
                    eval,
                )?
            } else {
                0
            };
            stats.expansion_time_us += expansion_start.elapsed().as_micros() as u64;
            stats.game_steps += steps;

            // Backpropagate the value - track backprop time
            let backprop_start = Instant::now();
            self.tree.backpropagate(leaf.node_id, eval.value);
            stats.backprop_time_us += backprop_start.elapsed().as_micros() as u64;
        }

        Ok(())
    }

    /// Expand a node using a pre-computed evaluation result, counting game steps.
    fn expand_node_with_eval_counted(
        &mut self,
        node_id: NodeId,
        parent_state: &[u8],
        legal_mask: &LegalMask,
        eval: &crate::evaluator::EvalResult,
    ) -> Result<u32, SearchError> {
        let mut step_count = 0;

        // Add children for each legal action
        for action in legal_mask.iter_ones() {
            let prior = eval.policy[action];
            if prior < 1e-8 {
                continue; // Skip zero-prior actions
            }

            // Simulate the action using zero-copy buffers.
            let action_bytes = (action as u32).to_le_bytes();
            self.ctx
                .step_into(
                    parent_state,
                    &action_bytes,
                    &mut self.step_state_buf,
                    &mut self.step_timestep_buf,
                )
                .map_err(|e| SearchError::EngineError(e.to_string()))?;
            step_count += 1;

            let actor = transition_actor(&self.step_timestep_buf)?;
            let reward = self.step_timestep_buf.reward_for(actor).ok_or_else(|| {
                SearchError::InvalidTimestep(format!(
                    "missing outcome for acting agent {}",
                    actor.0
                ))
            })?;
            let done = self.step_timestep_buf.episode != EpisodeStatus::Running;
            // Terminal nodes have no next decision. Running nodes carry their
            // exact availability in the decision envelope.
            let child_legal_mask = if done {
                LegalMask::new(self.num_actions)
            } else {
                let next = self
                    .step_timestep_buf
                    .decision
                    .sole_agent()
                    .ok_or_else(|| {
                        SearchError::InvalidTimestep(format!(
                            "expected one next actor, got {:?}",
                            self.step_timestep_buf.decision
                        ))
                    })?;
                decision_legal_mask(next, self.num_actions)?.clone()
            };

            // Terminal value (negated for opponent's perspective)
            let terminal_value = if done { -reward } else { 0.0 };

            self.tree.add_child(
                node_id,
                action as u32,
                prior,
                child_legal_mask,
                done,
                terminal_value,
            );
        }

        Ok(step_count)
    }

    /// Add Dirichlet noise to root node priors for exploration.
    fn add_dirichlet_noise(&mut self, rng: &mut ChaCha20Rng) {
        let root_id = self.tree.root();
        let root = self.tree.get(root_id);
        let num_children = root.children.len();

        if num_children == 0 {
            return;
        }

        // Generate Dirichlet noise using Gamma distribution
        let noise = dirichlet_noise(num_children, self.config.dirichlet_alpha, rng);

        // Mix noise with existing priors
        let eps = self.config.dirichlet_epsilon;
        let children: Vec<_> = root.children.iter().map(|(_, id)| *id).collect();

        for (i, child_id) in children.into_iter().enumerate() {
            let child = self.tree.get_mut(child_id);
            child.prior = (1.0 - eps) * child.prior + eps * noise[i];
        }
    }

    /// Get the search tree (for inspection/debugging).
    pub fn tree(&self) -> &MctsTree {
        &self.tree
    }
}

fn effective_eval_batch_size(config: &MctsConfig) -> usize {
    config
        .eval_batch_size
        .max(1)
        .min(((config.num_simulations as usize) / 4).max(1))
}

fn decision_observation(timestep: &ErasedTimestep) -> Result<&[u8], SearchError> {
    let agent_id = timestep
        .decision
        .sole_agent()
        .ok_or_else(|| {
            SearchError::InvalidTimestep(format!(
                "expected one next actor, got {:?}",
                timestep.decision
            ))
        })?
        .agent_id;
    timestep.observation_for(agent_id).ok_or_else(|| {
        SearchError::InvalidTimestep(format!("missing observation for next agent {}", agent_id.0))
    })
}

fn decision_legal_mask(
    decision: &engine_core::AgentDecision,
    num_actions: usize,
) -> Result<&LegalMask, SearchError> {
    let ActionAvailability::DiscreteMask { mask } = &decision.availability else {
        return Err(SearchError::InvalidTimestep(format!(
            "AlphaZero requires a discrete legal mask for agent {}",
            decision.agent_id.0
        )));
    };
    if mask.num_actions() != num_actions {
        return Err(SearchError::LegalMaskWidthMismatch {
            expected: num_actions,
            actual: mask.num_actions(),
        });
    }
    Ok(mask)
}

fn transition_actor(timestep: &ErasedTimestep) -> Result<AgentId, SearchError> {
    match &timestep.source {
        TransitionSource::Agents { agent_ids } if agent_ids.len() == 1 => Ok(agent_ids[0]),
        source => Err(SearchError::InvalidTimestep(format!(
            "expected one acting agent, got {source:?}"
        ))),
    }
}

/// Convenience function to run a single MCTS search.
pub fn run_mcts<E: Evaluator>(
    ctx: &mut EngineContext,
    evaluator: &E,
    config: MctsConfig,
    state: Vec<u8>,
    timestep: ErasedTimestep,
    rng: &mut ChaCha20Rng,
) -> Result<SearchResult, SearchError> {
    let mut search = MctsSearch::new(ctx, evaluator, config, state, timestep)?;
    search.run(rng)
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
