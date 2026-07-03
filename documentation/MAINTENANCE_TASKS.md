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

## Recurring cadence (no prompt needed)

- **Monthly:** review/merge Dependabot PRs; prune merged `claude/*` branches.
- **Quarterly:** docs sync (CLAUDE.md tree, test counts, API.md/SCHEMA.md); dead-code sweep
  (`cargo machete`, `vulture`, `knip`); slowest/flaky test review (`pytest --durations=10`);
  re-check the ~11 lint suppressions; consider one stricter clippy/ruff rule family.
- **After significant MCTS work:** `cargo bench -p mcts -- --save-baseline main` and compare
  against the previous baseline.
