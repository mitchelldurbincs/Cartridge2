# engine-core

Algorithm-neutral environment runtime for Cartridge2. This crate defines the
typed environment ABI, validates and erases it for runtime dispatch, and keeps
board-game conventions in an optional adapter/profile rather than in the core
contract.

## Contract layers

- `Environment`: typed state, action, and observation implementation.
- `Timestep`: the transition roster, per-agent observations and outcomes,
  the next decision, episode status, transition source, and opaque environment
  info.
- `Capabilities`: immutable contract version, exact wire codecs, optional
  horizon, agent/action-space model, batching hint, and declared semantics.
- `EnvironmentMetadata`: display name/description plus an optional nested
  `BoardGameMetadata` profile.
- `EngineContext`: the public reset/step/presentation boundary. It seals the
  bytes-only erased implementation layer and validates immutable descriptors
  and every produced timestep.
- `BoardGameEnvironment`: explicitly narrow adapter for the bundled
  deterministic, alternating, two-seat board games.

The generic ABI represents fixed or dynamic agents, single-agent, sequential,
or simultaneous decisions, explicit or environment-sampled chance, perfect or
partial observations, deterministic or stochastic transitions, general
per-agent rewards, termination versus truncation, and discrete,
multi-discrete, or continuous action spaces. Representation is not algorithm
support: each algorithm cartridge must publish and enforce its own compatibility
requirements. `alphazero_board_v1`, for example, rejects every profile outside
its narrow board-game contract.

## Typed environment

```rust
pub trait Environment: Send + Sync + std::fmt::Debug + 'static {
    type State: Send + Sync + 'static;
    type Action: Send + Sync + 'static;
    type Observation: Send + Sync + 'static;

    fn engine_id(&self) -> EngineId;
    fn capabilities(&self) -> Capabilities;
    fn metadata(&self) -> EnvironmentMetadata;

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

    // Strict state/action codecs and observation encoding are also required.
}
```

Each timestep explicitly names its transition `agents`. Fixed environments
must emit their complete declared population. For dynamic environments this is
not only the post-step live roster: it contains newly/currently active agents
and retains a departing agent on the transition that terminates or truncates
it, so the envelope can carry that agent's final outcome and transition
provenance. A departing agent is excluded from the next `Decision` immediately
and omitted from the following timestep. Each roster member has exactly one
`AgentOutcome` containing its own `reward`, `terminated`, and `truncated`
flags. `Decision::Agents` may name one agent or, for a simultaneous environment,
several; `Decision::Chance` is valid only when explicit chance is declared;
completed episodes use `Decision::None`.

Validation of dynamic lifecycles is intentionally local to one transition: it
does not prove that `TransitionSource` equals the preceding timestep's
`Decision`, forbid later reuse of a departed ID, or otherwise track an episode
across arbitrary branchable state snapshots. A standard decision-action
envelope or a caller-supplied prior-timestep handle is required before those
cross-transition invariants can become part of the generic runtime ABI.

Capabilities do not expose one global action space. Fixed environments attach
an `ActionSpace` to every `AgentSpec`; dynamic populations declare one shared
space:

```rust
let agents = AgentModel::fixed_homogeneous(
    [AgentId(1), AgentId(2)],
    ActionSpace::Discrete { size: 9 },
);
```

Changing a codec, semantics, observation meaning, agent model, or other
compatibility-relevant contract under an existing environment ID requires an
`env_contract_version` bump.

`CompleteSnapshot` means the encoded state contains every input needed to
reproduce future transitions; `step` may not depend on hidden mutable state.
Environment-sampled chance uses the runtime RNG stream and therefore must
declare `ExternalState`. Explicit chance may remain snapshot-compatible because
the resolved outcome arrives as the action.

The standard action codecs are exact: discrete is one little-endian `u32`;
multi-discrete is one little-endian `u32` per declared dimension; continuous is
one little-endian IEEE-754 `f32` per flattened row-major element. Joint actions
for simultaneous decisions and explicit chance outcomes use an
environment-defined `Custom` codec until a standard decision-action envelope is
introduced. The environment's strict decoder remains authoritative for action
lengths and bounds.

## Metadata and presentation

Every environment publishes only generic display metadata by default:

```rust
let metadata = EnvironmentMetadata::new("counter", "Counter")
    .with_description("Increment until the target is reached");
assert!(metadata.board.is_none());
```

An AlphaZero-compatible board environment may attach a nested profile:

```rust
let metadata = EnvironmentMetadata::new("tictactoe", "Tic-Tac-Toe")
    .with_board(
        BoardGameMetadata::new(3, 3, 9)
            .with_observation(29, 2, 18, false),
    );
```

Board-only types are intentionally namespaced under
`engine_core::board_profile`; generic environments do not import them.

`Environment::presentation` is optional and independent of planning state. It
may return a `Presentation::Board` projection or a versioned custom payload;
algorithms must not depend on it.

## Registration and runtime use

General environments implement `Default` and use generic registration. The
registry key is derived from the environment's validated descriptor, so there
is no second caller-supplied ID that can drift. Environment IDs are runtime
namespace segments containing only lowercase ASCII letters, digits, `_`, or
`-`:

```rust
use engine_core::register_environment;

pub fn register_counter() {
    register_environment::<CounterEnvironment>()
        .expect("counter must only be registered once");
}
```

The narrow bundled board family instead uses
`board_profile::register_board_game::<TicTacToe>()`. Its private adapter converts the board
profile into the generic contract before the same validation and type-erasure
boundary. Duplicate IDs are errors; production registration has no replacement
or removal path. `envs-counter` is the non-board reference implementation.

```rust
use engine_core::{AgentId, EngineContext};

let mut context = EngineContext::new("tictactoe").expect("registered");
let reset = context.reset(42, &[])?;
let action = 4u32.to_le_bytes();
let step = context.step(&reset.state, &action)?;

println!("decision: {:?}", step.timestep.decision);
println!("P1 reward: {:?}", step.timestep.reward_for(AgentId(1)));
println!("episode: {:?}", step.timestep.episode);
```

The sealed erased boundary retains the roster and all per-agent
observations/outcomes; it never collapses them into one scalar reward or one
`done` flag. Consumers cannot construct an unchecked erased environment or
bypass `EngineContext` validation.

## Module map

```text
src/
├── typed.rs       # Environment, Timestep, agents/actions, semantics, codecs
├── contract.rs    # Descriptor/timestep invariants shared by runtime paths
├── adapter.rs     # Private validated typed-to-erased conversion
├── erased.rs      # Private runtime trait + public erased data/error types
├── context.rs     # EngineContext
├── registry.rs    # Immutable process-local registration
├── metadata.rs    # Generic metadata + optional board profile
├── board_game.rs  # Private board adapter; exported through board_profile
├── board_game_utils.rs # Board-only helpers exported through board_profile
├── board_view.rs  # Board-only presentation exported through board_profile
└── legal_mask.rs  # Board-only masks exported through board_profile
```

## Testing

From the repository root:

```bash
cargo test --manifest-path engine/Cargo.toml -p engine-core
```

Tests cover strict codec round trips, descriptor/timestep contract violations,
per-agent and simultaneous/chance envelopes, registry behavior, context reuse,
optional metadata/presentation, and the narrow board adapter.
