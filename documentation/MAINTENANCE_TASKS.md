# Periodic Maintenance Tasks

Companion to `FILE_LENGTH_ANALYSIS.md`. Five one-off/recurring hygiene tasks, each sized as
one PR, with paste-ready prompts. Baseline observations (2026-07-03): CI already runs
`cargo audit`/`pip-audit`/`npm audit`; zero TODO/FIXME comments; 11 lint suppressions total;
146 `unwrap()/expect(` occurrences in `actor/src`, 122 in `web/src` (many in inline tests);
no mypy/pyright; no Dependabot/Renovate; nothing keeps the Python game re-implementations or
the three copies of config defaults in sync.

## Sequencing vs. the file-length PRs

| Task | Files touched | When to run |
|---|---|---|
| A. Dependabot | `.github/dependabot.yml` (new) | Any time — conflicts with nothing |
| B. mypy adoption | `trainer/`, `.github/workflows/ci.yml` | **After** file-length PRs 1–4 merge |
| C. unwrap audit | `actor/src/`, `web/src/`, their `Cargo.toml`s | **After** file-length PRs 9–10 merge |
| D. Game consistency tests | `engine/games-tictactoe/`, `engine/games-connect4/`, `trainer/tests/`, `testdata/` (new) | Any time — all new files |
| E. Config-drift tests | `engine/engine-config/`, `trainer/tests/` | Any time |

A, D, and E can run in parallel with the file-length refactors. None of these touch
`CLAUDE.md`; fold doc updates (new lint commands, testdata/ dir) into the final docs-sync PR.

## Prompts

### A — Automated dependency updates (Dependabot)

> Add `.github/dependabot.yml` to this repo so dependencies get update PRs automatically. CI already audits for vulnerabilities but nothing bumps stale deps. Cover: `cargo` for the three workspaces (directories `/engine`, `/actor`, `/web`), `pip` for `/trainer` (PEP 621 pyproject.toml — verify Dependabot supports this layout; if it cannot parse it, say so in the PR description rather than silently skipping it), `npm` for `/web/frontend`, and `github-actions` for `/`. Use a monthly schedule, and group minor+patch updates into a single PR per ecosystem so this doesn't create noise (majors stay separate). Keep open-PR limits low (e.g. 3 per ecosystem). This PR adds only the one YAML file — no code changes, no CLAUDE.md changes. Validate the YAML (e.g. yamllint or a schema check) since Dependabot config only executes on the default branch.

### B — Adopt mypy in the trainer (run after file-length PRs 1–4)

> Add mypy type checking to the Python trainer. Goals: (1) add `mypy` to the dev extras in `trainer/pyproject.toml` with a `[tool.mypy]` config; (2) get `python -m mypy src/` passing; (3) add it as a step in the Python CI job in `.github/workflows/ci.yml` next to ruff/black. Adopt gradually: global config with `check_untyped_defs = true` but `disallow_untyped_defs = false`, plus per-module `ignore_missing_imports` overrides for untyped third-party libs (torch/onnx/onnxruntime/wandb/psycopg/bitbully as needed); then enable strict(er) settings per-module for the small pure modules first (`stats.py`, `config.py`, `lr_scheduler.py`, `backoff.py`, `game_config.py`). Fix findings by adding annotations and narrowing — do NOT refactor logic; if mypy reveals a genuine bug, fix it in a clearly-labeled separate commit. Do not update CLAUDE.md. Verify with: `cd trainer && python -m ruff check src/ && python -m black --check src/ && python -m mypy src/ && python -m pytest tests/ -v --tb=short`.

### C — unwrap/expect audit in the services (run after file-length PRs 9–10)

> Audit panic points in the two long-running Rust services. `actor/src` has ~146 `.unwrap()`/`.expect(` occurrences and `web/src` ~122, many in test modules (fine to keep) but the rest are potential panics in processes meant to run unattended. For production code paths only: (1) startup/one-time initialization may keep `.expect("descriptive message")` or switch to `anyhow::Context` — a crash at boot with a clear message is acceptable; (2) steady-state paths (the actor episode loop, web request handlers, model reload, storage flush) must not panic — propagate errors with context or log-and-continue, matching how each callsite's module already handles errors (see the existing `lock_engine`/`lock_mcts_policy` helpers in `actor/src/actor.rs` for the poisoned-lock pattern). Convert bare `.unwrap()` to one of those; do not redesign error types or change behavior beyond panic-to-handled-error. Then ratchet: add `[lints.clippy] unwrap_used = "warn"` to `actor/Cargo.toml` and `web/Cargo.toml` (CI runs clippy with `-D warnings`, so this becomes enforced) and put `#![allow(clippy::unwrap_used)]` at the top of test-module files. Only touch `actor/` and `web/` (splitting into one PR per crate is fine). Do not update CLAUDE.md. Verify with both crates' fmt/clippy/test commands from CLAUDE.md.

### D — Cross-language game consistency tests

> The trainer deliberately re-implements TicTacToe and Connect4 in pure Python (`trainer/src/trainer/games/`) mirroring the Rust engines (`engine/games-tictactoe/`, `engine/games-connect4/`), and nothing currently forces them to agree. Add golden-trace consistency tests: (1) a deterministic generator on the Rust side (an `#[ignore]`d test or a `cargo run --example generate_traces` per game crate) that plays ~50 seeded games per game using uniformly-random legal moves (seeded ChaCha20Rng, fixed seeds) and writes JSON traces to a new repo-root `testdata/game-traces/` directory — recording, per step: the action taken, the set of legal actions before the move, whether the game ended, and the final winner; (2) a Rust test in each game crate that replays the committed traces through the Rust implementation and asserts they still match (resolve paths via `CARGO_MANIFEST_DIR`); (3) a Python test in `trainer/tests/` that replays the same traces through the Python implementations and asserts legal-move sets, terminal flags, and winners match (translating action/player encodings between the two sides in the test as needed). Compare rules only — legality, transitions, outcomes — not observation encodings. Commit the generated traces and document the regeneration command in `testdata/game-traces/README.md`. New files only (plus test-module wiring in the two game crates); no changes to game logic, trainer source, or CLAUDE.md. Verify with the engine cargo fmt/clippy/test commands and `cd trainer && python -m pytest tests/ -v --tb=short` (plus ruff/black on the new test).

### E — Config-defaults drift tests

> Config defaults for this project live in three places that must agree: `config.defaults.toml` (documented as the single source of truth), the Rust defaults in `engine/engine-config/src/defaults.rs`, and the Python defaults in `trainer/src/trainer/central_config.py`. Add drift tests: (1) a Rust test in `engine-config` that parses the repo-root `config.defaults.toml` (path via `CARGO_MANIFEST_DIR/../..`) and asserts every value matches the built-in default config structs; (2) a Python test in `trainer/tests/` that does the same for the settings the trainer consumes from `central_config.py`. Minimal production changes are allowed if needed to expose defaults for comparison (e.g. a function returning the default mapping), but no behavior changes. Stretch goal if cheap: a test asserting every key in `config.defaults.toml` is mentioned in `engine/engine-config/SCHEMA.md`, so the schema doc can't silently rot. Only touch `engine/engine-config/` and `trainer/tests/`; do not update CLAUDE.md. Verify with the engine cargo fmt/clippy/test commands and `cd trainer && python -m pytest tests/ -v --tb=short` (plus ruff/black on the new test).

## Code quality / readability prompts

Unlike PRs A–E and the file-length batch, these are **cross-cutting sweeps over the same
files**, so they conflict with the refactor PRs and with each other. Run them after the
file-length batch merges, one at a time (merge each before starting the next). Any order
works; F and J give the most readability per hour. All are behavior-preserving except K.

### F — Function-size and complexity ratchet (after the file-length batch)

> Files in this repo are being kept under 500 lines; apply the same discipline to functions and enforce it with lints. Rust: add `too_many_lines = "warn"` under `[lints.clippy]` in the engine workspace and the actor and web crates, with a `clippy.toml` in each workspace root setting `too-many-lines-threshold = 100` (CI runs clippy with `-D warnings`, so this becomes enforced). Known offenders to split by extracting well-named helper functions: `Actor::run` (~180 lines in `actor/src/actor.rs` or `episode.rs` post-refactor), `MctsSearch::run` (~125 lines), `ModelWatcher::start_watching` (~200 lines in `engine/model-watcher`), and whatever else the lint flags. Python: in `trainer/pyproject.toml`, extend ruff with `C901` (`[tool.ruff.lint.mccabe] max-complexity = 12`) and the `PLR091x` refactor rules (too many branches/statements/args); fix findings the same way — known offender: `Trainer._record_step_metrics` (~95 lines, or its post-refactor home). Pure extraction: no behavior changes, helpers named for what they do, no pass-through wrappers that just hide length. For a function that is genuinely one cohesive unit, a targeted `#[allow]`/`# noqa` with a one-line justification beats a bad split — but expect these to be rare. Do not update CLAUDE.md. Verify with every component's fmt/lint/test commands from CLAUDE.md.

### G — Public API and module documentation ratchet

> Make the codebase navigable without reading implementations. Rust: add `missing_docs = "warn"` under `[lints.rust]` for the engine workspace's library crates (engine-core, engine-config, mcts, model-watcher, engine-games, games-*) — CI's `-D warnings` enforces it — then write doc comments for every public item and a `//!` module overview for each module explaining its purpose and how it fits (see `engine/model-watcher/src/lib.rs` for the house style). Python: extend ruff in `trainer/pyproject.toml` with the pydocstyle `D1xx` rules for `src/` (undocumented-public-* only; skip the style-nitpick D2xx/D4xx rules, exclude tests), then add the missing module/class/function docstrings. Quality bar: docs must say why the thing exists, when to use it, and any non-obvious constraints — a docstring that paraphrases the name is worse than none, so spend the words on intent and invariants. No code changes beyond documentation. Do not update CLAUDE.md. Verify with the engine cargo fmt/clippy/test commands and `cd trainer && python -m ruff check src/ && python -m black --check src/ && python -m pytest tests/ -v --tb=short`.

### H — Duplication sweep

> Find and consolidate copy-pasted logic across the codebase. Detection: run a clone detector (e.g. `npx jscpd --min-tokens 70` over `engine/*/src actor/src web/src trainer/src web/frontend/src`, excluding tests/target/node_modules) and manually review the top clusters. Known suspects to inspect regardless of tool output: `actor/src/metrics.rs` vs `web/src/metrics.rs`; `actor/src/model_watcher.rs` (88 lines) vs `web/src/model_watcher.rs` (107 lines) — both wrap the shared `engine/model-watcher` crate, so any duplicated wrapper logic should move into the crate; the trainer storage backends (`storage/postgres.py`, `s3.py`, `base.py`) for shared serialization/retry logic that belongs in the base; repeated request-handling plumbing in `web/src/handlers/`. Consolidation rule: only merge code with genuinely identical semantics into its shared home (engine crate for cross-binary Rust, base module for Python); for near-duplicates that differ for a real reason, leave them but add a one-line comment cross-referencing the sibling. Behavior-preserving; public APIs stable. List what was consolidated and what was deliberately left in the PR description. Do not update CLAUDE.md. Verify with every affected component's fmt/lint/test commands from CLAUDE.md.

### I — Simplification lint ratchet

> Tighten the linters toward simpler, more idiomatic code — mechanical fixes only. Python: in `trainer/pyproject.toml`, extend ruff's `select` with `SIM` (flake8-simplify), `B` (bugbear), `RET` (flake8-return), `ARG` (unused arguments), and `UP` (pyupgrade); fix all findings, using a per-line `# noqa: <rule>` with a short reason only where the autofix would genuinely hurt readability. Rust: run `cargo clippy --all-targets -- -W clippy::pedantic` on each workspace (engine, actor, web), review the output, and cherry-pick the 5–10 pedantic lints that fire with real readability value for this codebase (e.g. `uninlined_format_args`, `map_unwrap_or`, `semicolon_if_nothing_returned`, `explicit_iter_loop` — judge from actual output, do NOT adopt pedantic wholesale) into each `[lints.clippy]` table at `warn`, then fix them. Every change must be behavior-preserving and locally reviewable; if a lint fix would require restructuring, skip it and note it in the PR description instead. Do not update CLAUDE.md. Verify with every component's fmt/lint/test commands from CLAUDE.md.

### J — Test suite readability (trainer-first; after file-length PR 4)

> Make the trainer test suite readable without scrolling: the big test files (test_storage, test_stats, test_checkpoint, test_solver_eval and their post-split successors) repeat setup boilerplate heavily. (1) Lift duplicated setup — temp data dirs, fake TrainerConfig/game configs, minimal networks, stats-file builders, fake replay buffers — into shared pytest fixtures in `trainer/tests/conftest.py` (create it if absent) with docstrings saying what each fixture provides. (2) Normalize test names to `test_<unit>_<behavior>[_<condition>]` so failures read as specifications. (3) Where a test asserts several unrelated behaviors, split it; where several tests assert the same thing with different values, parametrize with `pytest.mark.parametrize`. No production code changes; the suite must stay green with the same or better coverage — if any tests are merged via parametrization, say so in the PR description. Run `pytest --durations=10` before and after and include the slowest-tests list in the PR description as a baseline for future cleanup. Do not update CLAUDE.md. Verify with: `cd trainer && python -m ruff check . && python -m black --check . && python -m pytest tests/ -v --tb=short`.

### K — Legacy/compat path removal (behavior-changing; run last, after the docs-sync PR)

> Audit and remove vestigial compatibility paths so there is one way to do everything. Known candidates: the legacy `ALPHAZERO_*` environment-variable overrides in the Python trainer (superseded by `CARTRIDGE_*`, which all components support) and the legacy `mcts.num_simulations` fallback (superseded by `start_sims`/`max_sims` ramping — `config.defaults.toml` itself labels it legacy). Also grep the repo for "legacy", "deprecated", and "backward" to catch others. For each: remove the code path, its tests, and every doc reference (CLAUDE.md, SCHEMA.md, config.defaults.toml comments — this task runs after the docs-sync PR, so editing CLAUDE.md here is fine), OR, if it must stay, add a comment at the definition site saying why and until when. NOTE: this intentionally drops documented behavior (e.g. ALPHAZERO_* support) — only run it if you're comfortable with that. Verify with the full trainer and engine test suites, and grep docs afterward for stale references to the removed names.

## Recurring cadence (no prompt needed)

- **Monthly:** review/merge Dependabot PRs; prune merged `claude/*` branches.
- **Quarterly:** docs sync (CLAUDE.md tree, test counts, API.md/SCHEMA.md); dead-code sweep
  (`cargo machete`, `vulture`, `knip`); slowest/flaky test review (`pytest --durations=10`);
  re-check the ~11 lint suppressions; consider one stricter clippy/ruff rule family.
- **After significant MCTS work:** `cargo bench -p mcts -- --save-baseline main` and compare
  against the previous baseline.
