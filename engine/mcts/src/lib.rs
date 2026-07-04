//! Monte Carlo Tree Search (MCTS) implementation for AlphaZero-style game playing.
//!
//! This crate provides a game-agnostic MCTS implementation that works with any
//! game implementing the `engine-core` Game trait.
//!
//! # Overview
//!
//! MCTS is a search algorithm that builds a search tree by running simulations.
//! Each simulation consists of four phases:
//!
//! 1. **Selection**: Traverse the tree using UCB (Upper Confidence Bound) to
//!    balance exploration and exploitation
//! 2. **Expansion**: When reaching a leaf, expand it by adding children for
//!    each legal action
//! 3. **Evaluation**: Use a policy/value network (or uniform prior for testing)
//!    to estimate the value of the new state
//! 4. **Backpropagation**: Update visit counts and value estimates along the
//!    path from leaf to root
//!
//! # Usage
//!
//! ```rust,ignore
//! use mcts::{MctsConfig, UniformEvaluator, run_mcts};
//! use engine_core::EngineContext;
//! use rand_chacha::ChaCha20Rng;
//! use rand::SeedableRng;
//!
//! // Register your game
//! games_tictactoe::register_tictactoe();
//!
//! // Create game context
//! let mut ctx = EngineContext::new("tictactoe").unwrap();
//! let reset = ctx.reset(42, &[]).unwrap();
//!
//! // Set up MCTS
//! let evaluator = UniformEvaluator::new();
//! let config = MctsConfig::for_testing();
//! let legal_mask = 0b111111111u64; // All 9 positions legal at the start
//!
//! // Run search
//! let mut rng = ChaCha20Rng::seed_from_u64(42);
//! let result = run_mcts(
//!     &mut ctx,
//!     &evaluator,
//!     config,
//!     reset.state,
//!     reset.obs,
//!     legal_mask,
//!     &mut rng,
//! ).unwrap();
//!
//! println!("Best action: {}", result.action);
//! println!("Policy: {:?}", result.policy);
//! println!("Value: {}", result.value);
//! ```
//!
//! # Configuration
//!
//! The [`MctsConfig`] struct controls search behavior:
//!
//! - `num_simulations`: Number of simulations per search (default: 800)
//! - `c_puct`: Exploration constant for UCB (default: 1.25)
//! - `dirichlet_alpha`: Noise parameter for exploration at root (default: 0.3)
//! - `temperature`: Temperature for action selection (1.0 = proportional, 0.0 = greedy)
//!
//! # Evaluators
//!
//! The search requires an [`Evaluator`] to estimate policy and value:
//!
//! - [`UniformEvaluator`]: Returns uniform policy over legal moves (for testing)
//! - Custom evaluators can wrap ONNX models for neural network inference
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         MctsSearch                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │  MctsTree   │  │EngineContext│  │     Evaluator       │  │
//! │  │  (arena)    │  │ (game sim)  │  │ (policy/value)      │  │
//! │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
//! │         │                │                    │             │
//! │         ▼                ▼                    ▼             │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              select → expand → evaluate →            │   │
//! │  │                     backpropagate                    │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod config;
pub mod evaluator;
pub mod node;
pub mod search;
pub mod tree;

mod sampling;
mod types;

#[cfg(feature = "onnx")]
pub mod onnx;

// Re-export main types
pub use config::MctsConfig;
pub use evaluator::{EvalResult, Evaluator, EvaluatorError, UniformEvaluator};
pub use node::{MctsNode, NodeId};
pub use search::{run_mcts, MctsSearch, SearchError, SearchResult, SearchStats};
pub use tree::{MctsTree, TreeStats};

#[cfg(feature = "onnx")]
pub use onnx::{OnnxEvaluator, OnnxStats, SharedOnnxEvaluator};
