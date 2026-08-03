# mcts

Search implementation owned by the `alphazero_board_v1` cartridge. It does
not accept every registered environment: construction requires the exact
two-player, alternating-turn, deterministic, perfect-information,
terminal-zero-sum capability contract, indexed discrete actions, and the
AlphaZero observation/legal-mask layout.

## Overview

MCTS builds a search tree by running simulations. Each simulation has four phases:

1. **Selection** - Traverse tree using PUCT (Polynomial UCT) formula
2. **Expansion** - Add child nodes for each legal action at leaf
3. **Evaluation** - Get policy/value estimate from evaluator
4. **Backpropagation** - Update visit counts and values up to root

## Usage

```rust
use mcts::{MctsConfig, UniformEvaluator, run_mcts};
use engine_core::EngineContext;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

// Register game
games_tictactoe::register_tictactoe();

// Create game context
let mut ctx = EngineContext::new("tictactoe").unwrap();
let reset = ctx.reset(42, &[]).unwrap();

// Set up MCTS
let evaluator = UniformEvaluator::new();
let config = MctsConfig::for_training()
    .with_simulations(800)
    .with_temperature(1.0);

let mut rng = ChaCha20Rng::seed_from_u64(42);
let result = run_mcts(
    &mut ctx,
    &evaluator,
    config,
    reset.state,
    reset.timestep,
    &mut rng,
).unwrap();

println!("Best action: {}", result.action);
println!("Policy: {:?}", result.policy);
println!("Value: {}", result.value);
```

## Configuration

```rust
let config = MctsConfig {
    num_simulations: 800,    // Simulations per search
    c_puct: 1.25,            // Exploration constant
    dirichlet_alpha: 0.3,    // Root noise for exploration
    dirichlet_epsilon: 0.25, // Weight of noise vs prior
    temperature: 1.0,        // Action selection temperature
    virtual_loss: 1.0,       // Discourage duplicate pending leaves
    eval_batch_size: 32,     // Batched leaf evaluation
};

// Presets
let training = MctsConfig::for_training();  // Exploratory
let testing = MctsConfig::for_testing();    // Fewer sims
let playing = MctsConfig::for_evaluation(); // No noise, greedy selection
```

## Evaluators

The `Evaluator` trait provides policy priors and value estimates:

```rust
pub trait Evaluator: Send + Sync {
    fn evaluate(
        &self,
        obs: &[u8],
        legal_mask: &LegalMask,
        num_actions: usize,
    ) -> Result<EvalResult, EvaluatorError>;
}

pub struct EvalResult {
    pub policy: Vec<f32>,  // Prior probabilities
    pub value: f32,        // State value estimate [-1, 1]
}
```

### Built-in Evaluators

- **UniformEvaluator** - Returns uniform policy over legal moves (for testing)

### ONNX Evaluator (optional)

Enable the `onnx` feature for neural network inference:

```bash
cargo build --manifest-path engine/Cargo.toml -p mcts --features onnx
```

```rust
use mcts::OnnxEvaluator;

let contract = algorithm.model_artifact_contract(env_id, env_contract_version);
let evaluator = OnnxEvaluator::load_from_file(
    "model.onnx",
    obs_size,
    num_actions,
    1,
    &identity,
)?;
```

ONNX loading requires the exact model identity, tensor names, dtypes, dynamic
batch dimension, and policy/value shapes. A filename or matching dimensions do
not authorize an artifact.

## Module Structure

```
src/
|-- lib.rs        # Public exports
|-- config.rs     # MctsConfig
|-- evaluator.rs  # Evaluator trait, UniformEvaluator
|-- node.rs       # MctsNode (visit_count, value_sum, prior, children)
|-- tree.rs       # MctsTree with arena allocation
|-- search.rs     # Selection, expansion, backprop, run_mcts
+-- onnx.rs       # OnnxEvaluator (feature-gated)
```

## Search Result

```rust
pub struct SearchResult {
    pub action: u32,        // Best action index
    pub policy: Vec<f32>,   // Visit count distribution
    pub value: f32,         // Root value estimate
    pub simulations: u32,   // Total simulations run
    pub stats: SearchStats, // Search timing and counters
}
```

## Testing

```bash
# All tests
cargo test --manifest-path engine/Cargo.toml -p mcts

# With ONNX support
cargo test --manifest-path engine/Cargo.toml -p mcts --features onnx
```

## Dependencies

- `engine-core` - Generic environment ABI; MCTS accepts only the guarded
  AlphaZero board profile
- `rand` / `rand_chacha` - Deterministic randomness
- `rand_distr` - Dirichlet distribution for root noise
- `ort` / `ndarray` - ONNX Runtime (optional)
