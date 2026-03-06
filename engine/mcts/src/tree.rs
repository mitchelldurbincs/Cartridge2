//! MCTS tree structure with arena allocation.
//!
//! The tree uses arena allocation for efficient node storage and
//! cache-friendly traversal. Nodes are stored in a contiguous Vec
//! and referenced by NodeId indices.

use crate::node::{MctsNode, NodeId};

/// MCTS tree with arena-based node storage.
#[derive(Debug)]
pub struct MctsTree {
    /// Arena storing all nodes
    nodes: Vec<MctsNode>,

    /// Root node index (always 0 after initialization)
    root: NodeId,
}

impl MctsTree {
    /// Create a new tree with the given root legal move mask.
    pub fn new(legal_moves_mask: u64) -> Self {
        let root_node = MctsNode::new_root(legal_moves_mask);
        Self {
            nodes: vec![root_node],
            root: NodeId(0),
        }
    }

    /// Get the root node ID.
    #[inline]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Get a reference to a node by ID.
    #[inline]
    pub fn get(&self, id: NodeId) -> &MctsNode {
        &self.nodes[id.0 as usize]
    }

    /// Get a mutable reference to a node by ID.
    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> &mut MctsNode {
        &mut self.nodes[id.0 as usize]
    }

    /// Allocate a new node and return its ID.
    pub fn allocate(&mut self, node: MctsNode) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Get the total number of nodes in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if tree is empty (should never be true after construction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the arena slice for read access.
    #[inline]
    pub fn arena(&self) -> &[MctsNode] {
        &self.nodes
    }

    /// Select the best child of a node using UCB.
    /// Returns the NodeId of the best child.
    pub fn select_child(&self, node_id: NodeId, c_puct: f32) -> Option<NodeId> {
        let node = self.get(node_id);
        // Pre-compute sqrt once instead of per-child comparison
        let parent_visits_sqrt = (node.visit_count as f32).sqrt();

        node.children
            .iter()
            .max_by(|(_, id_a), (_, id_b)| {
                let score_a = self.get(*id_a).ucb_score(parent_visits_sqrt, c_puct);
                let score_b = self.get(*id_b).ucb_score(parent_visits_sqrt, c_puct);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, id)| *id)
    }

    /// Add a child to a parent node.
    /// Returns the new child's NodeId.
    #[allow(clippy::too_many_arguments)]
    pub fn add_child(
        &mut self,
        parent_id: NodeId,
        action: u8,
        prior: f32,
        legal_moves_mask: u64,
        is_terminal: bool,
        terminal_value: f32,
    ) -> NodeId {
        let child = MctsNode::new_child(
            parent_id,
            action,
            prior,
            legal_moves_mask,
            is_terminal,
            terminal_value,
        );
        let child_id = self.allocate(child);

        // Add to parent's children
        self.get_mut(parent_id).children.push((action, child_id));

        child_id
    }

    /// Backpropagate a value from a leaf to the root.
    /// Value is negated at each level (opponent's perspective).
    pub fn backpropagate(&mut self, leaf_id: NodeId, value: f32) {
        let mut current_id = leaf_id;
        let mut current_value = value;

        while current_id.is_some() {
            let node = self.get_mut(current_id);
            node.visit_count += 1;
            node.value_sum += current_value;

            // Remove virtual loss if any
            if node.virtual_loss > 0.0 {
                node.virtual_loss = 0.0;
            }

            // Negate for opponent's perspective
            current_value = -current_value;

            current_id = node.parent;
        }
    }

    /// Apply virtual loss to nodes along a path (for parallel MCTS).
    pub fn apply_virtual_loss(&mut self, path: &[NodeId], loss: f32) {
        for &node_id in path {
            self.get_mut(node_id).virtual_loss += loss;
        }
    }

    /// Remove virtual loss from nodes along a path.
    pub fn remove_virtual_loss(&mut self, path: &[NodeId]) {
        for &node_id in path {
            self.get_mut(node_id).virtual_loss = 0.0;
        }
    }

    /// Get the best action from root based on visit counts.
    /// Returns (action, visit_count) or None if root has no children.
    pub fn best_action(&self) -> Option<(u8, u32)> {
        let root = self.get(self.root);
        root.children
            .iter()
            .map(|(action, id)| (*action, self.get(*id).visit_count))
            .max_by_key(|(_, visits)| *visits)
    }

    /// Get the policy (visit distribution) from the root.
    /// Returns a vector of (action, probability) pairs.
    pub fn root_policy(&self, num_actions: usize, temperature: f32) -> Vec<f32> {
        let root = self.get(self.root);
        let mut policy = vec![0.0; num_actions];

        if root.children.is_empty() {
            return policy;
        }

        if temperature < 1e-6 {
            // Greedy: all mass on best action
            if let Some((action, _)) = self.best_action() {
                policy[action as usize] = 1.0;
            }
        } else {
            // Temperature-scaled visit counts
            let visits: Vec<f32> = root
                .children
                .iter()
                .map(|(_, id)| {
                    let v = self.get(*id).visit_count as f32;
                    if temperature == 1.0 {
                        v
                    } else {
                        v.powf(1.0 / temperature)
                    }
                })
                .collect();

            let total: f32 = visits.iter().sum();
            if total > 0.0 {
                for ((action, _), &v) in root.children.iter().zip(visits.iter()) {
                    policy[*action as usize] = v / total;
                }
            }
        }

        policy
    }

    /// Get statistics about the tree for debugging.
    pub fn stats(&self) -> TreeStats {
        let root = self.get(self.root);
        TreeStats {
            total_nodes: self.nodes.len(),
            root_visits: root.visit_count,
            root_value: root.mean_value(),
            max_depth: self.compute_max_depth(self.root, 0),
        }
    }

    fn compute_max_depth(&self, node_id: NodeId, current_depth: u32) -> u32 {
        let node = self.get(node_id);
        if node.children.is_empty() {
            return current_depth;
        }

        node.children
            .iter()
            .map(|(_, id)| self.compute_max_depth(*id, current_depth + 1))
            .max()
            .unwrap_or(current_depth)
    }
}

/// Statistics about an MCTS tree.
#[derive(Debug, Clone)]
pub struct TreeStats {
    pub total_nodes: usize,
    pub root_visits: u32,
    pub root_value: f32,
    pub max_depth: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree() {
        let tree = MctsTree::new(0b111);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root(), NodeId(0));

        let root = tree.get(tree.root());
        assert!(root.parent.is_none());
    }

    #[test]
    fn test_add_child() {
        let mut tree = MctsTree::new(0b111);

        let child_id = tree.add_child(
            tree.root(),
            1,   // action
            0.5, // prior
            0b110,
            false,
            0.0,
        );

        assert_eq!(tree.len(), 2);
        assert_eq!(child_id, NodeId(1));

        let root = tree.get(tree.root());
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0], (1, NodeId(1)));

        let child = tree.get(child_id);
        assert_eq!(child.parent, tree.root());
        assert_eq!(child.action, 1);
        assert!((child.prior - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_backpropagate() {
        let mut tree = MctsTree::new(0b111);

        // Create a chain: root -> child -> grandchild
        let child_id = tree.add_child(tree.root(), 0, 0.5, 0b11, false, 0.0);
        let grandchild_id = tree.add_child(child_id, 1, 0.5, 0b1, false, 0.0);

        // Backpropagate value 1.0 from grandchild
        tree.backpropagate(grandchild_id, 1.0);

        // Check visits
        assert_eq!(tree.get(grandchild_id).visit_count, 1);
        assert_eq!(tree.get(child_id).visit_count, 1);
        assert_eq!(tree.get(tree.root()).visit_count, 1);

        // Check values (negated at each level)
        assert!((tree.get(grandchild_id).value_sum - 1.0).abs() < 1e-6);
        assert!((tree.get(child_id).value_sum - (-1.0)).abs() < 1e-6);
        assert!((tree.get(tree.root()).value_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_select_child() {
        let mut tree = MctsTree::new(0b111);

        // Add two children with different priors
        tree.add_child(tree.root(), 0, 0.3, 0, false, 0.0);
        tree.add_child(tree.root(), 1, 0.7, 0, false, 0.0);

        // Initially, higher prior should win (UCB dominated by prior when unvisited)
        let best = tree.select_child(tree.root(), 1.0).unwrap();
        assert_eq!(best, NodeId(2)); // Second child has higher prior
    }

    #[test]
    fn test_root_policy() {
        let mut tree = MctsTree::new(0b111);

        // Add children and simulate visits
        let c1 = tree.add_child(tree.root(), 0, 0.5, 0, false, 0.0);
        let c2 = tree.add_child(tree.root(), 1, 0.5, 0, false, 0.0);

        tree.get_mut(c1).visit_count = 30;
        tree.get_mut(c2).visit_count = 70;

        // Temperature 1.0: proportional to visits
        let policy = tree.root_policy(9, 1.0);
        assert!((policy[0] - 0.3).abs() < 1e-6);
        assert!((policy[1] - 0.7).abs() < 1e-6);
        for p in policy.iter().take(9).skip(2) {
            assert!(p.abs() < 1e-6);
        }

        // Temperature 0.0: greedy
        let greedy = tree.root_policy(9, 0.0);
        assert!(greedy[0].abs() < 1e-6);
        assert!((greedy[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tree_stats() {
        let mut tree = MctsTree::new(0b111);
        tree.add_child(tree.root(), 0, 0.5, 0b11, false, 0.0);

        let stats = tree.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.max_depth, 1);
    }
}
