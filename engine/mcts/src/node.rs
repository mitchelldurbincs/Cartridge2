//! MCTS tree node representation.
//!
//! Each node represents a game state reached by taking an action from the parent.
//! Nodes store visit statistics used for UCB selection and policy improvement.

/// Index into the node arena. Using a newtype for type safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const NONE: NodeId = NodeId(u32::MAX);

    pub fn is_none(self) -> bool {
        self == Self::NONE
    }

    pub fn is_some(self) -> bool {
        !self.is_none()
    }
}

/// A node in the MCTS tree.
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// Parent node index (NONE for root)
    pub parent: NodeId,

    /// Action that led to this node from parent (0-indexed)
    pub action: u8,

    /// Number of times this node has been visited
    pub visit_count: u32,

    /// Sum of values backpropagated through this node.
    /// Q(s,a) = value_sum / visit_count
    pub value_sum: f32,

    /// Prior probability from the policy network.
    /// P(s,a) - probability of selecting action `a` from parent state.
    pub prior: f32,

    /// Whether this is a terminal state (game over)
    pub is_terminal: bool,

    /// Terminal reward (only valid if is_terminal)
    pub terminal_value: f32,

    /// Legal moves mask at this state (packed into bits)
    pub legal_moves_mask: u64,

    /// Children: Vec of (action, NodeId) pairs.
    /// Empty until node is expanded.
    pub children: Vec<(u8, NodeId)>,

    /// Virtual loss for parallel MCTS (subtracted during selection, added back after eval)
    pub virtual_loss: f32,
}

impl MctsNode {
    /// Create a new root node.
    pub fn new_root(legal_moves_mask: u64) -> Self {
        Self {
            parent: NodeId::NONE,
            action: 0,
            visit_count: 0,
            value_sum: 0.0,
            prior: 1.0, // Root has prior 1.0
            is_terminal: false,
            terminal_value: 0.0,
            legal_moves_mask,
            children: Vec::new(),
            virtual_loss: 0.0,
        }
    }

    /// Create a new child node.
    #[allow(clippy::too_many_arguments)]
    pub fn new_child(
        parent: NodeId,
        action: u8,
        prior: f32,
        legal_moves_mask: u64,
        is_terminal: bool,
        terminal_value: f32,
    ) -> Self {
        Self {
            parent,
            action,
            visit_count: 0,
            value_sum: 0.0,
            prior,
            is_terminal,
            terminal_value,
            legal_moves_mask,
            children: Vec::new(),
            virtual_loss: 0.0,
        }
    }

    /// Calculate mean value Q(s,a) = value_sum / visit_count.
    /// Returns 0.0 if never visited.
    #[inline]
    pub fn mean_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }

    /// Calculate UCB score for child selection.
    /// UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
    ///
    /// Higher scores are better (more promising to explore).
    ///
    /// IMPORTANT: The value stored in each node is from that node's perspective
    /// (the player whose turn it is at that node). When the parent selects among
    /// children, it needs to negate the Q value because the child represents the
    /// opponent's position. A child with Q=-1 (bad for opponent) is actually
    /// Q=+1 from the parent's perspective (good for us).
    ///
    /// Note: Takes pre-computed sqrt(parent_visits) to avoid redundant sqrt calls
    /// when comparing multiple children.
    #[inline]
    pub fn ucb_score(&self, parent_visits_sqrt: f32, c_puct: f32) -> f32 {
        // Negate Q because the child stores value from opponent's perspective
        let q = -self.mean_value() - self.virtual_loss;
        let u = c_puct * self.prior * parent_visits_sqrt / (1.0 + self.visit_count as f32);
        q + u
    }

    /// Calculate UCB score (convenience method that computes sqrt internally).
    /// Prefer `ucb_score` with pre-computed sqrt when comparing multiple children.
    #[inline]
    pub fn ucb_score_with_parent_visits(&self, parent_visits: u32, c_puct: f32) -> f32 {
        self.ucb_score((parent_visits as f32).sqrt(), c_puct)
    }

    /// Check if this node has been expanded (has children).
    #[inline]
    pub fn is_expanded(&self) -> bool {
        !self.children.is_empty()
    }

    /// Check if this is a leaf node (not expanded or terminal).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.is_terminal || !self.is_expanded()
    }

    /// Get the most visited child action.
    /// Returns None if no children or all unvisited.
    pub fn best_child_by_visits<'a>(&self, arena: &'a [MctsNode]) -> Option<(u8, &'a MctsNode)> {
        self.children
            .iter()
            .map(|(action, id)| (*action, &arena[id.0 as usize]))
            .max_by_key(|(_, node)| node.visit_count)
    }

    /// Get visit count distribution over children (for training targets).
    /// Returns a vector of (action, visit_fraction) pairs.
    pub fn visit_distribution(&self, arena: &[MctsNode]) -> Vec<(u8, f32)> {
        let total_visits: u32 = self
            .children
            .iter()
            .map(|(_, id)| arena[id.0 as usize].visit_count)
            .sum();

        if total_visits == 0 {
            return Vec::new();
        }

        self.children
            .iter()
            .map(|(action, id)| {
                let node = &arena[id.0 as usize];
                (*action, node.visit_count as f32 / total_visits as f32)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_none() {
        assert!(NodeId::NONE.is_none());
        assert!(!NodeId::NONE.is_some());
        assert!(!NodeId(0).is_none());
        assert!(NodeId(0).is_some());
    }

    #[test]
    fn test_new_root() {
        let node = MctsNode::new_root(0b111);

        assert!(node.parent.is_none());
        assert_eq!(node.visit_count, 0);
        assert!((node.prior - 1.0).abs() < 1e-6);
        assert!(!node.is_terminal);
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_mean_value() {
        let mut node = MctsNode::new_root(0);

        // Unvisited
        assert!((node.mean_value()).abs() < 1e-6);

        // After visits
        node.visit_count = 4;
        node.value_sum = 2.0;
        assert!((node.mean_value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ucb_score() {
        let mut node = MctsNode::new_root(0);
        node.prior = 0.5;
        node.visit_count = 10;
        node.value_sum = 5.0; // Q from child's perspective = 0.5

        let c_puct = 1.0;
        let parent_visits = 100;
        let parent_visits_sqrt = (parent_visits as f32).sqrt();

        // UCB = -Q + c_puct * P * sqrt(N_parent) / (1 + N)
        // Note: Q is negated because child stores value from opponent's perspective
        // UCB = -0.5 + 1.0 * 0.5 * 10 / 11 = -0.5 + 0.4545... ≈ -0.0455
        let ucb = node.ucb_score(parent_visits_sqrt, c_puct);
        assert!((ucb - (-0.0455)).abs() < 0.01);

        // Test convenience method gives same result
        let ucb2 = node.ucb_score_with_parent_visits(parent_visits, c_puct);
        assert!((ucb - ucb2).abs() < 1e-6);
    }

    #[test]
    fn test_is_leaf() {
        let mut node = MctsNode::new_root(0);

        // Initially a leaf (no children)
        assert!(node.is_leaf());

        // Add a child
        node.children.push((0, NodeId(1)));
        assert!(!node.is_leaf());

        // Terminal nodes are always leaves
        let mut terminal = MctsNode::new_root(0);
        terminal.is_terminal = true;
        terminal.children.push((0, NodeId(1)));
        assert!(terminal.is_leaf());
    }
}
