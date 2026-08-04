# Trainer

Python algorithm host for Cartridge2 learning, evaluation, and synchronized
orchestration. Environment facts and algorithm implementations are separate:

- `environment_catalog.py` strictly parses engine-generated manifest schema v5.
- `algorithms/registry.py` resolves an installed algorithm ID.
- `algorithms/alphazero_board_v1.py` owns the current AlphaZero learner,
  network recipes, collector runner, and evaluation runner.
- `algorithms/dqn_v1.py` owns single-agent transition decoding, Q-learning,
  greedy return evaluation, and bounded off-policy orchestration.

There is no generic promise that every registered game can use AlphaZero. Each
entry point validates the requested algorithm/environment profile before it
opens replay storage, loads a model, or launches work.

## Quick start

From the repository root:

```bash
pip install -e "trainer/.[dev]"

# PostgreSQL is the replay backend.
docker compose up postgres -d
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:cartridge@localhost:5432/cartridge

# Train the installed cartridge from compatible replay data.
python -m trainer --algorithm alphazero_board_v1 train \
  --env-id tictactoe \
  --steps 1000 \
  --collection-scope-id 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --source-root
```

Direct `train` consumes an already populated exact selection. Its source must
equal the current RunHead checkpoint, or be `--source-root` only when no RunHead
exists. It is permanently disjoint from recipe-owned synchronized lineages.

The CLI requires a subcommand. Available commands are:

| Command | Purpose |
|---------|---------|
| `python -m trainer --algorithm ID train` | Train on one explicitly named replay selection and source |
| `python -m trainer --algorithm ID evaluate` | Evaluate a model through the Rust engine |
| `python -m trainer --algorithm ID loop` | Run synchronized collection, learning, and evaluation |
| `python -m trainer --algorithm ID solver-eval` | Score Connect 4 moves with the perfect solver |
| `python -m trainer --algorithm ID register-players` | Register checkpoint players |
| `python -m trainer --algorithm ID tournament` | Run a profile-scoped round robin and rate players |

The command set is cartridge-owned. `dqn_v1` exposes `collect`, `train`,
`evaluate`, and `loop`; it does not expose board tournaments or the Connect 4
solver.

```bash
python -m trainer --algorithm dqn_v1 loop \
  --env-id counter \
  --iterations 10 \
  --episodes-per-iteration 100 \
  --steps-per-iteration 500
```

Each DQN iteration allocates a fresh replay scope bound to the current RunHead,
collects epsilon-greedy transitions, trains a direct-child Q checkpoint, and
evaluates the committed greedy policy by episode return.

Pass `--algorithm alphazero_board_v1` explicitly in scripts. The checked-in
`[algorithm].id` supplies the interactive default.

## The installed AlphaZero cartridge

`alphazero_board_v1` provides `AlphaZeroLearner` and
`AlphaZeroLearnerConfig`. Its current compatibility profile is deliberately
narrow: two fixed alternating players, discrete actions, perfect
information, deterministic planning snapshots, fixed spatial `f32`
observations with a legal-action mask and player indicator, and terminal
zero-sum rewards.

The learner optimizes the AlphaZero policy/value objective:

```text
L = -sum(pi * log(p)) + (z - v)^2
```

`pi` is the raw tau=1 MCTS visit distribution and `z` is the terminal
outcome from that position's perspective. Sampled rows must contain both
targets at the exact action/observation widths; the learner does not invent a
one-hot policy or substitute the search value for a missing outcome.

### Network recipes

Generic engine facts come from the `capabilities` object in
`environment_manifest.json`. The AlphaZero binding additionally requires the
optional nested `metadata.board` profile and reads its dimensions, action count,
observation width, channels, and legal-mask offset. Network tuning belongs to
the algorithm module:

| Environment | Recipe |
|-------------|--------|
| `tictactoe` | MLP, hidden size 128 |
| `connect4` | ResNet, 4 blocks, 128 filters |
| `othello` | ResNet, 6 blocks, 256 filters |
| `generals_8x8` | ResNet, 6 blocks, 128 filters |

A structurally compatible environment without a tuning override receives the
cartridge's safe default MLP recipe. Registration and semantic compatibility
still remain separate checks.

## Replay isolation

The algorithm descriptor supplies the experience contract. The learner binds
an algorithm-neutral PostgreSQL store to one exact selection:

```python
from trainer.storage import ReplayProfile, ReplaySelection, create_replay_store

profile = ReplayProfile(
    env_id="connect4",
    env_contract_version=1,
    algorithm_id="alphazero_board_v1",
    experience_schema="alphazero_transition_v1",
)
selection = ReplaySelection(
    profile=profile,
    collection_scope_id="0123456789abcdef" * 4,
    source_checkpoint_id=None,  # root collection
)
replay = create_replay_store(selection)
```

Count, sample, clear, and retention cleanup always filter by
`(env_id, env_contract_version, algorithm_id, experience_schema,
collection_scope_id, source_checkpoint_id)`, using null-exact source matching.
Writes that do not match the complete selection are rejected. This prevents a
changed environment contract, another cartridge, a prior attempt, or a
different model generation from sharing, training on, or deleting the wrong
experience. PostgreSQL stores only `ReplayRecord` envelope fields and opaque
`payload` bytes; it has no board, action, reward, policy, or value columns.

Learner setup waits for one usable record in the exact selection. Every
positive `ReplayStore.sample(n)` returns exactly `n` records, drawing with
replacement when the selection is smaller than the requested minibatch;
sampling an empty selection is an immediate error. Batch size therefore does
not impose a minimum replay-selection capacity.

For `alphazero_transition_v1`, the cartridge decodes the payload as
`observation[obs_size] || policy[num_actions] || terminal_value[1]`, all
little-endian `f32`. Another algorithm installs its own codec without changing
the replay table. The exact selection's count is exposed as
`TrainerStats.replay_record_count` and the Prometheus gauge
`trainer_replay_record_count`.

Replay v3 is a clean schema cutover. The database must contain only
`cartridge_schema_versions` with marker `('replay', 3)` and `replay_records`
with the exact v3 contract. Older replay databases must be recreated from
[`../sql/schema.sql`](../sql/schema.sql); no migration, compatibility view, or
implicit scope/source identity is provided.

## Model and checkpoint identity

Every checkpoint is content-addressed beneath the resolved profile's `models/`
directory or S3 prefix:

```text
blobs/sha256/{onnx_sha256}.onnx
blobs/sha256/{learner_state_sha256}.pt
manifests/sha256/{checkpoint_id}.json
evaluations/manifests/sha256/{evaluation_id}.json
run-commits/sha256/{run_commit_id}.json
run-preparations/by-parent/{root|parent_run_commit_id}.json
channels/current.json
```

`checkpoint_id` is the SHA-256 of the canonical manifest bytes. The manifest
binds the exact algorithm/environment/model profile, step, parent checkpoint,
learner-config digest, and `{sha256, size_bytes}` descriptor for both blobs.
Publication safely validates the learner envelope before it creates repository
objects. Checkpoint staging does not make a model visible. A canonical
`RunCommitV1` binds one checkpoint to its exact learner-stats snapshot and any
orchestration/evaluation state. The sole mutable `RunHeadV2` selects both the
checkpoint and run commit with compare-and-set semantics. Standalone training
commits each staged child immediately; synchronized training stages only one
final direct child and lets the parent loop commit it after evaluation. All
consumers resolve the same RunHead: learner/collector use Latest, while web uses
ChampionOrLatest from the selected RunCommit.

The ONNX blob uses identity schema version 1 and must carry exact
`cartridge.schema_version`, `cartridge.algorithm_id`,
`cartridge.model_contract`, `cartridge.env_id`, and
`cartridge.env_contract_version` custom metadata. The learner-state blob embeds
the same profile plus its step and configuration digest. Resume verifies the
run head, run-commit digest and schema, checkpoint manifest hash, profile,
config hash, both blob sizes/digests, bound stats snapshot, and the embedded
learner state and complete parent lineage before loading optimizer or scheduler
state. Rust
inference consumers additionally validate the ONNX graph and tensor contract
before swapping evaluators.

All mutable files belong to one canonical runtime namespace:

```text
{data_root}/profiles/{algorithm_id}/{env_id}/v{env_contract_version}/
```

The command's selected environment determines that path. Omitting a path flag
uses the namespace automatically; explicit paths are still validated against
the selected artifact contract.

An absent `current` run head means training starts fresh and inference has no
model to load. Any present head, run commit, manifest, snapshot, or blob that is
missing, non-canonical, corrupt, or mismatched raises instead of silently
resetting. A rejected web hot reload does not replace the last valid evaluator.

Old mutable checkpoint filenames and ONNX files without schema-v1 identity are
not accepted. Retrain/re-export them with the current code; there is no legacy
loader, shape-based inference, or automatic converter.

Filesystem publication uses a local advisory lock. S3 publication needs no
coordination object: the exact ETag of `models/channels/current.json` is the
serialization token. Creating the first head uses `If-None-Match: *`; advancing
an existing head uses `If-Match`. A competing writer therefore fails closed
without an operator-managed lock or recovery command.

## Evaluation and champion identity

Promotion evidence is content-addressed alongside checkpoints:

```text
evaluations/manifests/sha256/{evaluation_id}.json
run-commits/sha256/{run_commit_id}.json
channels/current.json
```

The immutable evaluation artifact binds the candidate checkpoint, prior
champion and its supporting evaluation, exact deterministic seat/seed recipe,
requested game counts, head-to-head and solver evidence, decision, and
timestamps. Its canonical JSON digest is `evaluation_id`. Champion and
evaluation state advance together inside a validated RunCommit selected by the
single RunHead; there is no independent champion channel. Promotion requires
the candidate, prior lineage, evidence, and recomputed decision to agree.
Solver-based promotion always compares fresh, symmetric candidate and champion
runs using the same game count and solver version.

## Training-statistics identity

`stats_id` is the SHA-256 of canonical stats-snapshot-v2 bytes. The snapshot
binds the full checkpoint profile, learner-config digest, checkpoint ID, and
step to an exact nested `TrainerStats` object. Training and evaluation histories
have exact schemas, ordered steps, finite normalized numeric values, and
cross-record invariants. Signed and integer zero spellings are normalized before
hashing.

The immutable `RunCommitV1` embeds both `stats_id` and the exact parsed snapshot;
there is no stats channel or second resume head. Resume follows `RunHeadV2`,
verifies the run commit and snapshot together, and restores cumulative steps,
samples, replay count, history, and evaluation state. `stats.json` is only an
atomically rebuilt web projection of the selected run head and is never read as
authority. A missing half, corrupt digest, mismatched profile/config/checkpoint,
or noncanonical snapshot fails closed.

Between checkpoints, `stats_interval` refreshes that projection from live
in-process counters without staging a model or advancing `RunHeadV2`. Those
ephemeral updates can be newer than the selected run and are intentionally lost
after a crash; startup always rebuilds the projection from the authoritative
run commit. Checkpoint, evaluation, and final-training boundaries publish the
checkpoint-bound snapshots used for resume.

## Synchronized workflow

The recommended loop keeps model generations and experience aligned:

```bash
python -m trainer --algorithm alphazero_board_v1 loop \
  --env-id connect4 \
  --iterations 50 \
  --episodes 500 \
  --steps 1000
```

Each iteration attempt allocates a new cryptographic replay collection scope
bound to the exact source checkpoint (or `null` for the root). One or more
bounded algorithm-provided collectors write only to that selection. The loop
requires the distinct completed-episode count to equal `--episodes` before it
allows `AlphaZeroLearner` to sample, export a candidate, evaluate it, and commit
the iteration. Old and abandoned scopes remain invisible; the loop never
clears or vacuums a profile to manufacture freshness. MCTS simulations can
ramp over iterations with `--mcts-start-sims`, `--mcts-max-sims`, and
`--mcts-sim-ramp-rate`. The committed run recipe also authenticates the exact
collector `c_puct`, early/late temperatures and threshold, Dirichlet alpha and
weight, evaluation batch size, and ONNX thread count. Evaluation simulations
and temperature are authenticated separately from collector search settings.

The orchestration engine lives in the pinned `crucible` dependency. Cartridge2
keeps a composition root in `orchestrator/orchestrator.py` and concrete
adapters in `orchestrator/actor_runner.py`, `eval_runner.py`, and
`eval_reporting.py`. These adapters bind the resolved algorithm to Crucible's
protocols; they are not compatibility aliases for the old trainer API.

For development against the sibling checkout, install it before this package:

```bash
pip install -e ../crucible
pip install -e "trainer/.[dev]"
```

## Evaluation and tournaments

Python does not implement game rules. Evaluation shells out to the Rust
`cartridge-eval` binary and passes the same algorithm ID and environment:

```bash
make build-eval

python -m trainer --algorithm alphazero_board_v1 evaluate \
  --env-id connect4 \
  --games 100 \
  --simulations 100
```

Evaluation requires a positive u32 game count, nonnegative u32 search budget,
and a nonnegative u64 seed whose complete per-game schedule cannot overflow.
Model temperature is canonicalized to a finite nonnegative f32 before launch.

Connect 4 can also be scored against the perfect solver:

```bash
python -m trainer --algorithm alphazero_board_v1 solver-eval \
  --env-id connect4 \
  --games 100
```

Without `--model`, solver evaluation resolves `models/channels/current.json`.
`--all-checkpoints` evaluates every verified manifest in repository step order;
an explicit `--model` remains a one-off file with no checkpoint ID or repository
step. `--model` and `--all-checkpoints` are mutually exclusive. This standalone
command prints diagnostics only and has no artifact-output
flag. Synchronized solver evidence becomes authoritative only through an
immutable `EvaluationArtifactV2` selected by RunCommit.

Player registry schema v5 records the full environment/model contract,
checkpoint manifest ID, immutable ONNX blob path, manifest step, and gameplay
adapter settings. Canonical player IDs contain the full checkpoint ID plus a
hash of `{schema_version, simulations, temperature}`. Registration discovers
verified repository manifests rather than filenames, and tournaments fail if a
registered manifest or blob is missing or corrupt:

```bash
python -m trainer --algorithm alphazero_board_v1 register-players \
  --env-id connect4

python -m trainer --algorithm alphazero_board_v1 tournament \
  --env-id connect4 \
  --games 40
```

## Configuration

Configuration priority is CLI, `CARTRIDGE_*` environment variables,
`config.toml`, then `config.defaults.toml`. The Python trainer requires the
replay DSN in `CARTRIDGE_STORAGE_POSTGRES_URL`; it does not use
`storage.postgres_url` as a fallback. A source checkout verifies that the root
defaults and the wheel-owned package resource are byte-identical; an installed
wheel loads that package resource. Missing, incomplete, or divergent canonical
defaults fail closed.

Useful training flags include:

- `--batch-size`, `--lr`, `--weight-decay`, and `--grad-clip`;
- `--checkpoint-interval` and `--device`;
- `--lr-warmup-steps`, `--lr-min-ratio`, and `--lr-total-steps`;
- `--wait-interval` and `--max-wait`; and
- `--replay-window` and `--replay-cleanup-interval`.

Use `python -m trainer --algorithm <id> <command> --help` for the complete
surface and
[`../config.defaults.toml`](../config.defaults.toml) for checked-in defaults.

## Module map

```text
src/trainer/
├── __main__.py                 # Required-subcommand CLI
├── environment_catalog.py     # Strict manifest-v4 parser
├── environment_manifest.json # Generated Rust-owned catalog
├── runtime_profile.py         # Canonical runtime namespace
├── algorithms/
│   ├── base.py                # Language-local cartridge protocol
│   ├── registry.py            # Installed algorithm dispatch
│   └── alphazero_board_v1.py  # AlphaZero recipe and factories
├── config.py                  # AlphaZeroLearnerConfig
├── trainer.py                 # AlphaZeroLearner
├── network.py                 # MLP, loss, network factory
├── resnet.py                  # Spatial policy/value network
├── checkpoint.py              # Strict PyTorch/ONNX artifacts
├── replay_setup.py            # Manifest/database cross-check
├── storage/                   # Exact replay selection + artifact publisher
├── evaluator.py               # Rust evaluator subprocess adapter
├── solver_eval/               # Connect 4 perfect-solver scoring
├── registry.py                # Immutable player registry schema v5
├── tournament.py              # Round robin + Bradley-Terry Elo
└── orchestrator/              # Cartridge-to-Crucible composition adapters
```

## Adding support

For a new environment, implement and register it in Rust, regenerate
`environment_manifest.json` with `make environment-manifest`, and inspect every algorithm
compatibility report. Add a network tuning override only when the installed
algorithm needs one; do not copy environment facts into Python.

For a new algorithm, add its canonical descriptor in `algorithm-core`, its
compatibility rules and manifest entry, then install concrete Rust/Python
dispatch bindings for the collector, learner, experience schema, model
contract, orchestration recipe, and evaluation suite.

## Validation

```bash
python -m ruff check trainer/src trainer/tests trainer/smoke_test.py
python -m ruff format --check trainer/src trainer/tests trainer/smoke_test.py
python -m pytest trainer/tests -v --tb=short
```
