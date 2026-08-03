# Actor

Rust experience-collection host for Cartridge2. The binary resolves an
algorithm cartridge, validates its environment compatibility profile, builds
that cartridge's collector, and writes experience to PostgreSQL.

Installed collectors are `alphazero_board_v1` (`AlphaZeroCollector`) and
`dqn_v1` (`DqnCollector`). The DQN collector emits immediate-reward transition
records and uses the shared `dqn_greedy_v1` Q-policy adapter after root random
collection.

## Quick start

From the repository root:

```bash
# PostgreSQL is required.
docker compose up postgres -d

# Collect 100 compatible TicTacToe episodes.
cargo run --manifest-path actor/Cargo.toml -- \
  --algorithm alphazero_board_v1 \
  --env-id tictactoe \
  --max-episodes 100 \
  --collection-scope-id aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
```

The omitted `--source-checkpoint-id` marks this as root collection and is
accepted only while the profile has no RunHead. Descendant collection must pass
the exact 64-hex checkpoint selected by the current RunHead.

Multiple one-shot collectors may write the same orchestrator-owned selection.
They must receive the same collection scope and source checkpoint, but distinct
actor IDs and finite episode quotas:

```bash
cargo run --manifest-path actor/Cargo.toml -- --algorithm alphazero_board_v1 --actor-id actor-1 --max-episodes 50 --collection-scope-id aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &
cargo run --manifest-path actor/Cargo.toml -- --algorithm alphazero_board_v1 --actor-id actor-2 --max-episodes 50 --collection-scope-id aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &
```

## Startup contract

Startup proceeds in this order:

1. Parse and validate configuration.
2. Resolve `--algorithm` through `algorithm-core`.
3. Register the engine environments and require a compatible
   algorithm/environment profile.
4. Derive the exact model and experience identities from the algorithm
   descriptor.
5. Load `ModelSelection::Latest` once and require it to equal the declared
   source checkpoint (or require no RunHead at root).
6. Only after model identity succeeds, connect to the exact replay selection.

For `alphazero_board_v1`, compatibility currently requires two fixed
alternating players, discrete actions, perfect-information deterministic
planning state, fixed spatial `f32` observations, an observation-embedded legal
mask and player indicator, and terminal zero-sum outcomes. The generated
schema-v5 manifest verifies these requirements against per-agent action spaces,
explicit environment semantics, wire codecs, and the optional board profile.
`dqn_v1` instead validates the single-agent discrete profile and never requests
board metadata.

Unknown IDs and incompatible pairs fail before model or database side effects.

## Components

- `src/main.rs`: one-shot process host, tracing, and graceful shutdown.
- `src/algorithms.rs`: algorithm ID to `CollectorAlgorithm` dispatch.
- `src/actor.rs`: `AlphaZeroCollector` episode generation and terminal-target encoding.
- `src/mcts_policy.rs`: AlphaZero MCTS action selection and policy targets.
- `src/dqn_actor.rs`: DQN epsilon-greedy collection and transition encoding.
- `src/storage/postgres.rs`: pooled PostgreSQL persistence.
- `src/stats.rs`: final structured collection, abandonment, and RSS telemetry.

The AlphaZero collector requires `EnvironmentMetadata.board` and derives its
dimensions and observation layout from that optional profile. Replay storage
does not know about boards, observations, actions, rewards, policies, or value
targets; those details remain owned by the selected algorithm cartridge.
The DQN payload is observation, action, immediate reward, next observation,
termination/truncation flags, and next-action availability.

## Model artifacts

The collector resolves the immutable environment contract version and loads
the selected runtime profile exactly once:

```text
filesystem: {data_root}/profiles/{algorithm_id}/{env_id}/v{env_contract_version}/models/channels/current.json
S3:         profiles/{algorithm_id}/{env_id}/v{env_contract_version}/models/channels/current.json
```

The sole mutable channel is `RunHeadV2`. The actor resolves and validates the
complete immutable RunCommit/checkpoint lineage, applies
`ModelSelection::Latest` to the accepted RunCommit, then follows its canonical
manifest to the digest-addressed ONNX blob. It verifies RunHead and RunCommit
schemas/lineage, canonical manifest identity, profile, blob size/digest, and the
ONNX tensor and metadata contract.

A valid ONNX blob must carry exact custom metadata:

| Key | Expected value |
|-----|----------------|
| `cartridge.schema_version` | `1` |
| `cartridge.algorithm_id` | Selected algorithm ID |
| `cartridge.model_contract` | Selected descriptor's model contract |
| `cartridge.env_id` | Selected environment ID |
| `cartridge.env_contract_version` | Selected environment contract version |

The actor is a bounded one-shot worker and never subscribes to model updates.
For root collection, the `current` channel must be absent and MCTS uses a
uniform evaluator. For descendant collection, `current` must resolve and its
latest checkpoint must exactly equal `--source-checkpoint-id`. A missing,
different, or invalid source fails before replay storage is opened. This pins
every record in the selection to one model generation. Older mutable artifacts
are not discovered or migrated.

## Replay contract

Each record carries an immutable storage envelope and an opaque,
algorithm-owned payload:

```rust
pub struct ReplayRecord {
    pub id: String,
    pub env_id: String,
    pub env_contract_version: u32,
    pub algorithm_id: String,
    pub experience_schema: String,
    pub collection_scope_id: String,
    pub source_checkpoint_id: Option<String>,
    pub episode_id: String,
    pub step_number: u32,
    pub payload: Vec<u8>,
}
```

The authoritative schema is [`../sql/schema.sql`](../sql/schema.sql). Its
`replay_records` primary key is
`(env_id, env_contract_version, algorithm_id, experience_schema,
collection_scope_id, id)`. Every write, record count, distinct-episode count,
and clear is bound to the full `ReplaySelection`; nullable source identity uses
SQL `IS NOT DISTINCT FROM`, never wildcard semantics. Phase 1 is a clean schema
cutover: recreate old replay databases rather than assigning implicit lineage,
scope, source, or payload codecs to existing rows. The database contains only
the replay-v3 marker and the generic `replay_records` table.

`alphazero_board_v1` owns the `alphazero_transition_v1` codec. Its payload is
the exact concatenation of little-endian `f32` values
`observation[obs_size] || policy[num_actions] || terminal_value[1]`. The policy
is the raw tau=1 MCTS visit distribution and the terminal value is from the
perspective of that position. An episode that times out is discarded in full
because it has no valid terminal target.

## Configuration

CLI arguments have highest priority, followed by supported `CARTRIDGE_*`
environment variables, central configuration, and checked-in defaults. Run
`cargo run --manifest-path actor/Cargo.toml -- --help` for the complete CLI
surface.

| CLI | Environment | Purpose |
|-----|-------------|---------|
| `--algorithm` | `CARTRIDGE_ALGORITHM_ID` | Cartridge ID |
| `--env-id` | `CARTRIDGE_COMMON_ENV_ID` | Environment ID |
| `--actor-id` | `CARTRIDGE_ACTOR_ACTOR_ID` | Collector instance ID |
| `--max-episodes` | none | Required positive episode quota for this process |
| `--collection-scope-id` | none | Required orchestrator-owned 64-hex attempt identity |
| `--source-checkpoint-id` | none | Required 64-hex model generation after the root iteration |
| `--data-dir` | `CARTRIDGE_COMMON_DATA_DIR` | Runtime root; the actor appends the canonical profile namespace |
| `--postgres-url` | `CARTRIDGE_STORAGE_POSTGRES_URL` | Replay database |
| `--num-simulations` | none | Required exact MCTS budget computed by the synchronized loop |
| `--c-puct` | `CARTRIDGE_MCTS_C_PUCT` | UCB exploration constant |
| `--temperature` | `CARTRIDGE_MCTS_TEMPERATURE` | Early-move action-selection temperature |
| `--late-temperature` | `CARTRIDGE_MCTS_LATE_TEMPERATURE` | Action-selection temperature after the threshold |
| `--temp-threshold` | `CARTRIDGE_MCTS_TEMP_THRESHOLD` | Move at which late temperature begins; 0 disables it, otherwise it must be below the environment horizon |
| `--dirichlet-alpha` | `CARTRIDGE_MCTS_DIRICHLET_ALPHA` | Root-noise concentration; set with weight to 0 to disable |
| `--dirichlet-weight` | `CARTRIDGE_MCTS_DIRICHLET_WEIGHT` | Root-noise mixture weight; set with alpha to 0 to disable |
| `--eval-batch-size` | `CARTRIDGE_MCTS_EVAL_BATCH_SIZE` | Batched leaf evaluation size |
| `--onnx-intra-threads` | `CARTRIDGE_MCTS_ONNX_INTRA_THREADS` | ONNX intra-op threads |

Central settings live in [`../config.defaults.toml`](../config.defaults.toml)
and [`../config.toml`](../config.toml).

## Synchronized training

Use `python -m trainer --algorithm alphazero_board_v1 loop`. The orchestrator
creates a fresh collection scope for each attempt, passes the authoritative
source checkpoint to every bounded actor, and seals the exact distinct episode
count before training. Direct long-running actor/trainer deployments are not a
supported topology.

## Validation

```bash
cargo fmt --manifest-path actor/Cargo.toml -- --check
cargo clippy --manifest-path actor/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path actor/Cargo.toml
```
