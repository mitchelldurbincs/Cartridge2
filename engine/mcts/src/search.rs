//! MCTS search implementation.
//!
//! Implements the core MCTS algorithm:
//! 1. Selection: Traverse tree using UCB to find a leaf
//! 2. Expansion: Add children to the leaf using policy prior
//! 3. Evaluation: Get value estimate from evaluator
//! 4. Backpropagation: Update statistics along the path

use std::time::Instant;

use engine_core::game_utils::info_bits;
use engine_core::{ActionSpace, EngineContext};
use rand_chacha::ChaCha20Rng;
use tracing::debug;

use crate::config::MctsConfig;
use crate::evaluator::Evaluator;
use crate::node::NodeId;
use crate::sampling::{dirichlet_noise, sample_action};
use crate::tree::MctsTree;
use crate::types::{LeafResult, PendingLeaf};

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
    step_obs_buf: Vec<u8>,
}

impl<'a, E: Evaluator> MctsSearch<'a, E> {
    /// Create a new MCTS search from the given game state.
    pub fn new(
        ctx: &'a mut EngineContext,
        evaluator: &'a E,
        config: MctsConfig,
        state: Vec<u8>,
        obs: Vec<u8>,
        legal_moves_mask: u64,
    ) -> Result<Self, SearchError> {
        let num_actions = match ctx.action_space() {
            ActionSpace::Discrete(n) => n as usize,
            _ => return Err(SearchError::UnsupportedActionSpace),
        };

        let tree = MctsTree::new(legal_moves_mask);
        let state_capacity = state.len().max(64);
        let obs_capacity = obs.len().max(64);

        Ok(Self {
            tree,
            ctx,
            evaluator,
            config,
            num_actions,
            root_state: state,
            root_obs: obs,
            step_state_buf: Vec::with_capacity(state_capacity),
            step_obs_buf: Vec::with_capacity(obs_capacity),
        })
    }

    /// Run the MCTS search for the configured number of simulations.
    ///
    /// Uses batched neural network evaluation for efficiency. Leaves are collected
    /// until `eval_batch_size` is reached, then evaluated together in a single NN call.
    pub fn run(&mut self, rng: &mut ChaCha20Rng) -> Result<SearchResult, SearchError> {
        let search_start = Instant::now();
        let mut stats = SearchStats::default();

        // First, expand the root if needed (single NN call, special case)
        if !self.tree.get(self.tree.root()).is_expanded() {
            let root_id = self.tree.root();
            let legal_mask = self.tree.get(root_id).legal_moves_mask;
            let root_state = self.root_state.clone();
            let root_obs = self.root_obs.clone();
            self.expand_node(root_id, &root_state, &root_obs, legal_mask)?;
        }

        // Add Dirichlet noise to root if configured
        if self.config.dirichlet_alpha > 0.0 {
            self.add_dirichlet_noise(rng);
        }

        // Batched simulation loop
        let batch_size = self.config.eval_batch_size.max(1);
        let mut pending: Vec<PendingLeaf> = Vec::with_capacity(batch_size);
        let mut completed_simulations: u32 = 0;
        let target_simulations = self.config.num_simulations;

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

        // Extract result
        let root = self.tree.get(self.tree.root());
        let policy = self
            .tree
            .root_policy(self.num_actions, self.config.temperature);

        let action = if self.config.temperature < 1e-6 {
            // Greedy
            self.tree
                .best_action()
                .map(|(a, _)| a)
                .ok_or(SearchError::NoLegalMoves)?
        } else {
            // Sample from policy
            sample_action(&policy, rng)?
        };

        Ok(SearchResult {
            action,
            policy,
            value: root.mean_value(),
            simulations: root.visit_count,
            stats,
        })
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

        // Check terminal/expanded status first (no cloning needed for early returns)
        let (is_terminal, terminal_value, is_expanded, legal_mask) = {
            let leaf = self.tree.get(leaf_id);
            (
                leaf.is_terminal,
                leaf.terminal_value,
                leaf.is_expanded(),
                leaf.legal_moves_mask,
            )
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

        // Reconstruct state/obs for this path on demand.
        let (state, obs) = self.reconstruct_position(&path)?;

        // Apply virtual loss to discourage selecting this node again
        // before it's been evaluated (prevents duplicate selection in batch)
        let leaf_mut = self.tree.get_mut(leaf_id);
        leaf_mut.visit_count += 1;
        leaf_mut.value_sum -= self.config.virtual_loss;

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
            let action_bytes = (self.tree.get(node_id).action as u32).to_le_bytes();
            self.ctx
                .step_into(
                    &state,
                    &action_bytes,
                    &mut self.step_state_buf,
                    &mut self.step_obs_buf,
                )
                .map_err(|e| SearchError::EngineError(e.to_string()))?;

            std::mem::swap(&mut state, &mut self.step_state_buf);
            std::mem::swap(&mut obs, &mut self.step_obs_buf);
        }

        Ok((state, obs))
    }

    /// Expand a node by adding all legal children.
    /// Returns the value estimate from the evaluator (to be used for backpropagation).
    fn expand_node(
        &mut self,
        node_id: NodeId,
        parent_state: &[u8],
        parent_obs: &[u8],
        legal_mask: u64,
    ) -> Result<f32, SearchError> {
        let node = self.tree.get(node_id);

        // Don't expand terminal nodes
        if node.is_terminal {
            return Ok(node.terminal_value);
        }

        // Get policy AND value from evaluator (single NN call)
        // We use the policy for child priors and return the value for backpropagation
        let eval = self
            .evaluator
            .evaluate(parent_obs, legal_mask, self.num_actions)?;
        self.expand_node_with_eval_counted(node_id, parent_state, legal_mask, &eval)?;

        // Return the value estimate from this single evaluation
        Ok(eval.value)
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
        let legal_masks: Vec<u64> = pending.iter().map(|p| p.legal_mask).collect();

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
            node.value_sum += self.config.virtual_loss;

            // Expand the node with children - track expansion time and game steps
            let expansion_start = Instant::now();
            let steps = self.expand_node_with_eval_counted(
                leaf.node_id,
                &leaf.state,
                leaf.legal_mask,
                eval,
            )?;
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
        legal_mask: u64,
        eval: &crate::evaluator::EvalResult,
    ) -> Result<u32, SearchError> {
        let mut step_count = 0;

        // Add children for each legal action
        for action in 0..self.num_actions {
            if (legal_mask >> action) & 1 == 0 {
                continue; // Illegal action
            }

            let prior = eval.policy[action];
            if prior < 1e-8 {
                continue; // Skip zero-prior actions
            }

            // Simulate the action using zero-copy buffers.
            let action_bytes = (action as u32).to_le_bytes();
            let (reward, done, info) = self
                .ctx
                .step_into(
                    parent_state,
                    &action_bytes,
                    &mut self.step_state_buf,
                    &mut self.step_obs_buf,
                )
                .map_err(|e| SearchError::EngineError(e.to_string()))?;
            step_count += 1;

            // Extract legal moves from info bits
            let child_legal_mask = info_bits::extract_legal_mask(info, self.num_actions as u32);

            // Terminal value (negated for opponent's perspective)
            let terminal_value = if done { -reward } else { 0.0 };

            self.tree.add_child(
                node_id,
                action as u8,
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

/// Convenience function to run a single MCTS search.
pub fn run_mcts<E: Evaluator>(
    ctx: &mut EngineContext,
    evaluator: &E,
    config: MctsConfig,
    state: Vec<u8>,
    obs: Vec<u8>,
    legal_moves_mask: u64,
    rng: &mut ChaCha20Rng,
) -> Result<SearchResult, SearchError> {
    let mut search = MctsSearch::new(ctx, evaluator, config, state, obs, legal_moves_mask)?;
    search.run(rng)
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
