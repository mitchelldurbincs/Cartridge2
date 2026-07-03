# File Length Analysis

Analysis date: 2026-07-03 (repo at `819328c`). Target: every source file under **500 lines**.

## Summary

- 124 source files (`.rs`, `.py`, `.svelte`, `.ts`, excluding `node_modules`/`target`/generated).
- **26 files exceed 500 lines**; 43 exceed 400.
- Two distinct root causes:
  1. **Inline Rust test modules.** Seven files are fine once their `#[cfg(test)] mod tests` block moves to a sibling file (a pattern this repo already uses: `games-tictactoe/src/tests.rs`, `engine-core/src/adapter_tests.rs`). E.g. `web/src/main.rs` is 935 lines, but only ~380 are production code.
  2. **Genuinely oversized modules.** The trainer (`solver_eval.py`, `trainer.py`, `eval_runner.py`), `mcts/search.rs`, `actor/actor.rs`, `games-othello/lib.rs`, model-watcher, and the three big Svelte components need real decomposition.

## Files over 500 lines

### Production code (prod/test split from inline `mod tests` position)

| File | Total | Prod | Inline tests | Fix |
|---|---:|---:|---:|---|
| `web/src/main.rs` | 935 | ~380 | ~555 | Move tests out; extract startup helpers |
| `engine/mcts/src/search.rs` | 846 | ~586 | ~260 | Move tests out **and** extract types/helpers |
| `engine/engine-core/src/context.rs` | 796 | ~290 | ~506 | Move tests out |
| `trainer/src/trainer/solver_eval.py` | 735 | 735 | — | Split into package (4 modules) |
| `actor/src/actor.rs` | 730 | ~643 | ~87 | Move tests out **and** extract episode module |
| `web/frontend/src/Stats.svelte` | 683 | 683 | — | Extract formatters + subcomponent |
| `engine/engine-core/src/typed.rs` | 659 | ~202 | ~457 | Move tests out |
| `trainer/src/trainer/trainer.py` | 650 | 650 | — | Extract replay/metrics/checkpoint helpers |
| `web/src/types/responses.rs` | 632 | ~206 | ~426 | Move tests out |
| `engine/games-othello/src/lib.rs` | 631 | 631 | — | Split into `state.rs` / `obs.rs` / `lib.rs` |
| `web/frontend/src/GenericBoard.svelte` | 623 | 623 | — | Split into GridBoard / DropBoard |
| `actor/src/mcts_policy.rs` | 603 | ~269 | ~334 | Move tests out |
| `web/frontend/src/LossOverTimePage.svelte` | 588 | 588 | — | Extract subcomponents/helpers |
| `engine/model-watcher/src/lib.rs` | 575 | ~504 | ~71 | Move tests out + extract types |
| `trainer/src/trainer/orchestrator/eval_runner.py` | 548 | 548 | — | Extract promotion + reporting modules |
| `engine/model-watcher/src/s3.rs` | 539 | ~510 | ~29 | Move tests out + extract download logic |
| `engine/engine-core/src/registry.rs` | 539 | ~211 | ~328 | Move tests out |
| `engine/mcts/src/onnx.rs` | 531 | ~483 | ~48 | Move tests out |

### Dedicated test/bench files (lower priority)

| File | Lines | Natural split |
|---|---:|---|
| `engine/engine-core/src/adapter_tests.rs` | 678 | By feature area |
| `trainer/tests/test_storage.py` | 675 | 6 test classes |
| `trainer/tests/test_stats.py` | 669 | 6 test classes |
| `engine/games-tictactoe/src/tests.rs` | 630 | Game logic vs encoding |
| `engine/games-othello/src/tests.rs` | 628 | Game logic vs encoding |
| `trainer/tests/test_checkpoint.py` | 553 | 4 test classes |
| `trainer/tests/test_solver_eval.py` | 544 | 12 test classes |
| `engine/mcts/benches/mcts.rs` | 541 | By benchmark group |

### Watch list (400–500 lines, no action needed yet)

`actor/src/stats.rs` (488), `trainer/tests/test_eval_runner.py` (484), `trainer/tests/test_orchestrator_wandb.py` (472), `trainer/tests/test_central_config.py` (471), `actor/src/storage/postgres.rs` (469), `web/src/handlers/game.rs` (460), `trainer/src/trainer/orchestrator/orchestrator.py` (458), `engine/games-connect4/src/tests.rs` (458), `engine/engine-config/src/structs.rs` (444), `web/src/game.rs` (443), `trainer/src/trainer/central_config.py` (438), `trainer/tests/test_lr_scheduler.py` (437), `trainer/src/trainer/evaluator.py` (432).

## PR plan

Each PR owns one component directory, so **no two PRs touch the same files** — they can be developed and merged in any order without conflicts. Two rules make that hold:

1. **Pure code motion, stable APIs.** Every split preserves public APIs and import paths (via `pub use` / `__init__.py` re-exports), so no PR forces changes in another component.
2. **Nobody touches `CLAUDE.md`.** Its directory tree would otherwise be edited by every PR — a guaranteed conflict magnet. PR 11 syncs docs once at the end.

| PR | Scope (files touched) | Priority |
|---|---|---|
| 1 | `trainer/src/trainer/solver_eval.py` → package | High (trainer) |
| 2 | `trainer/src/trainer/trainer.py` + new sibling modules | High (trainer) |
| 3 | `trainer/src/trainer/orchestrator/` | High (trainer) |
| 4 | `trainer/tests/` splits | Optional; merge after 1–3 |
| 5 | `engine/engine-core/src/` | Medium |
| 6 | `engine/mcts/src/` | Medium |
| 7 | `engine/model-watcher/src/` | Medium |
| 8 | `engine/games-othello/src/` | Medium |
| 9 | `actor/src/` | Medium |
| 10 | `web/src/` | Medium |
| 11 | `web/frontend/src/` | Medium |
| 12 | `CLAUDE.md` doc sync | Last, after all merge |

## Prompts

Each prompt is self-contained — paste one per session/PR.

### PR 1 — trainer: split `solver_eval.py` into a package

> Refactor `trainer/src/trainer/solver_eval.py` (735 lines) into a `trainer/src/trainer/solver_eval/` package so no file exceeds 500 lines (aim for <300 each). This is a pure code-motion refactor: no behavior changes. Suggested layout: `judgment.py` (classify_score, ply_bucket, MoveJudgment, judge_move), `results.py` (BucketStats, SolverEvalResults, infer_step_from_filename), `scorer.py` (SolverScorer, solver_evaluate), `cli.py` (discover_checkpoints, append_solver_stats, format_progression_table, add_solver_eval_arguments, run_solver_evaluation, main). Add an `__init__.py` that re-exports every name currently importable from `trainer.solver_eval`, so existing imports (`__main__.py`, orchestrator, tests) keep working unchanged — do not edit files outside the new package. Do not update CLAUDE.md. Verify with: `cd trainer && python -m ruff check src/ && python -m black --check src/ && python -m pytest tests/ -v --tb=short`.

### PR 2 — trainer: slim down `trainer.py`

> Refactor `trainer/src/trainer/trainer.py` (650 lines) to under 400 lines by extracting cohesive helpers into new modules — pure code motion, no behavior changes. Suggested extractions from the `Trainer` class: (a) replay-buffer concerns (`_create_replay_buffer`, `_setup_replay`, `_wait_with_backoff`, `_handle_replay_cleanup`) into `trainer/src/trainer/replay_setup.py`; (b) checkpoint/eval orchestration (`_discover_existing_checkpoints`, `_handle_checkpoint_and_eval`, `_evaluate_checkpoint`, `_save_checkpoint`) into `trainer/src/trainer/checkpoint_runner.py`, building on the existing `checkpoint.py` utilities; (c) the ~95-line `_record_step_metrics` plus `_write_stats` into `trainer/src/trainer/step_metrics.py`. Keep `Trainer`'s public interface (`Trainer(config)`, `.train()`) identical and keep `trainer.py` owning `__init__` and the core `train`/`_train_step` loop. Only touch `trainer.py`, the new modules, and `tests/test_trainer.py` if imports require it. Do not update CLAUDE.md. Verify with: `cd trainer && python -m ruff check src/ && python -m black --check src/ && python -m pytest tests/ -v --tb=short`.

### PR 3 — trainer: slim down `orchestrator/eval_runner.py`

> Refactor `trainer/src/trainer/orchestrator/eval_runner.py` (548 lines) to under 400 lines by extracting two modules — pure code motion, no behavior changes. (a) `trainer/src/trainer/orchestrator/promotion.py`: `should_promote` plus the best-model bookkeeping (`_load_best_model_info`, `_save_best_model_info`, `promote_to_best`, `_effective_promotion_metric`, `_ensure_best_solver_rate`). (b) `trainer/src/trainer/orchestrator/eval_reporting.py`: eval-record building and logging (`_build_eval_record`, `_append_solver_history`, `_log_eval_to_wandb`). Keep `EvalRunner`'s public interface and re-export `should_promote` from `eval_runner.py` so existing imports and tests keep working. Only touch files under `trainer/src/trainer/orchestrator/` plus `tests/test_eval_runner.py` if imports require it. Do not update CLAUDE.md. Verify with: `cd trainer && python -m ruff check src/ && python -m black --check src/ && python -m pytest tests/ -v --tb=short`.

### PR 4 — trainer: split oversized test files (optional)

> Split these trainer test files so each is under 500 lines, along their existing test-class boundaries: `tests/test_storage.py` (675), `tests/test_stats.py` (669), `tests/test_checkpoint.py` (553), `tests/test_solver_eval.py` (544). Pure code motion: move whole test classes into new sibling files (e.g. `test_storage_postgres.py` / `test_storage_filesystem.py`, `test_stats_io.py`, …), duplicating only imports/fixtures or lifting shared fixtures into `conftest.py`. No production code changes, no CLAUDE.md changes. The full suite must pass with the same test count: `cd trainer && python -m pytest tests/ -v --tb=short` (also `python -m ruff check` and `python -m black --check` on the tests).

### PR 5 — engine-core: move inline test modules to sibling files

> In `engine/engine-core/src/`, three files exceed 500 lines only because of inline `#[cfg(test)] mod tests` blocks: `context.rs` (796, ~506 test lines), `typed.rs` (659, ~457), `registry.rs` (539, ~328). Move each inline test module into a sibling file (`context_tests.rs`, `typed_tests.rs`, `registry_tests.rs`) using the pattern already used by `adapter.rs` (`#[cfg(test)] #[path = "adapter_tests.rs"] mod adapter_tests;`). Pure code motion — no production code or test logic changes, no public API changes, test count stays 70. Only touch `engine/engine-core/`. Do not update CLAUDE.md. Verify with: `cargo fmt --check --manifest-path engine/Cargo.toml && cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings && cargo test --manifest-path engine/Cargo.toml`.

### PR 6 — mcts: split `search.rs`, externalize tests

> In `engine/mcts/src/`: (a) `search.rs` is 846 lines (~586 production + ~260 inline tests). Extract the plain data types (`PendingLeaf`, `LeafResult`, `SearchError`, `SearchResult`, `SearchStats`) into a new `types.rs` and the free helpers (`sample_action`, `dirichlet_noise`) into `sampling.rs`, re-exporting from `lib.rs`/`search.rs` so the public API (`run_mcts`, `MctsSearch`, `SearchResult`, …) is unchanged; move the inline test module to a sibling `search_tests.rs` via `#[path]` (see `engine-core/src/adapter.rs` for the pattern). (b) `onnx.rs` (531): move its inline tests out the same way. Pure code motion, no algorithm changes. Only touch `engine/mcts/`. Do not update CLAUDE.md. Verify with: `cargo fmt --check --manifest-path engine/Cargo.toml && cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings && cargo test --manifest-path engine/Cargo.toml`. Since this touches MCTS, sanity-check `cargo bench --manifest-path engine/Cargo.toml -p mcts` still compiles/runs (pure code motion should not move the numbers).

### PR 7 — model-watcher: split lib.rs and s3.rs

> In `engine/model-watcher/src/`, get both files under ~450 production lines. `lib.rs` (575): move the inline test module to a sibling `tests.rs`, and extract `ModelInfo` plus the metadata/step-extraction helpers (`extract_metadata`, the static load helpers) into a new `types.rs` or `load.rs`, re-exporting from `lib.rs` so `model_watcher::ModelInfo`/`ModelWatcher` paths are unchanged. `s3.rs` (539): move its inline tests out, and extract `S3Config` plus the download/ETag logic (`check_and_download`, `check_and_download_static`, `extract_training_step`, the `load_model_static` variants) into a submodule (e.g. `s3/download.rs` or `s3_download.rs`), keeping `model_watcher::s3::S3ModelWatcher` and `S3Config` paths stable. Pure code motion, no behavior changes; the actor and web crates must compile unchanged. Only touch `engine/model-watcher/`. Do not update CLAUDE.md. Verify with the engine fmt/clippy/test commands, plus `cargo clippy`/`cargo test` for `actor/` and `web/` to confirm downstream compiles.

### PR 8 — games-othello: split lib.rs into modules

> Split `engine/games-othello/src/lib.rs` (631 lines, tests already external in `tests.rs`) into: `state.rs` (the `State` struct with move generation/flipping logic, `DIRECTIONS`, board constants), `obs.rs` (`OthelloObs`, `observation_from_state`, `OBS_SIZE`), and a slim `lib.rs` keeping the `Othello` type, its `Game` impl, `register_othello()`, and `pub use` re-exports so every currently-public path (`games_othello::State`, `COLS`, `PASS_ACTION`, …) is unchanged — `tests.rs` and downstream crates must compile without edits. Pure code motion, no game-logic changes, all 25 tests still pass. Only touch `engine/games-othello/`. Do not update CLAUDE.md. Verify with: `cargo fmt --check --manifest-path engine/Cargo.toml && cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings && cargo test --manifest-path engine/Cargo.toml`.

### PR 9 — actor: split actor.rs, externalize mcts_policy tests

> In `actor/src/`: (a) `actor.rs` (730 lines) — move `EpisodeContext`, `EpisodeStats`, and the episode-scoped `impl Actor` methods (`check_episode_limits`, `finalize_episode`, `run_episode`) into a new `episode.rs` (a second `impl Actor` block there is fine), and move the inline `#[cfg(test)] mod tests` to a sibling file via `#[path]`; `actor.rs` should land near ~350 lines keeping `Actor::new/run/shutdown`. (b) `mcts_policy.rs` (603 lines, ~334 of them inline tests) — move the test module to a sibling file the same way. Pure code motion, no behavior changes, `Actor`'s public API unchanged, all 86 tests pass. Only touch `actor/src/`. Do not update CLAUDE.md. Verify with: `cargo fmt --check --manifest-path actor/Cargo.toml && cargo clippy --manifest-path actor/Cargo.toml --all-targets -- -D warnings && cargo test --manifest-path actor/Cargo.toml`.

### PR 10 — web backend: split main.rs, externalize responses tests

> In `web/src/`: (a) `main.rs` (935 lines, ~555 of them inline tests) — move the `#[cfg(test)] mod tests` into a sibling file via `#[path]`, and extract server plumbing (`configure_cors`, `init_tracing`, `shutdown_signal`, and app/state construction like `create_app_with_cors`/`create_app`/`create_test_state` if it helps) into a new module (e.g. `startup.rs` or `app.rs`), leaving `main.rs` as a thin entrypoint well under 500 lines. Keep `AppState`, `ModelInfo`, and the router construction functions importable from their current paths (re-export from `main.rs` if needed by tests). (b) `types/responses.rs` (632 lines, ~426 inline tests) — move its test module to a sibling file. Pure code motion, no route or behavior changes. Only touch `web/src/`. Do not update CLAUDE.md. Verify with: `cargo fmt --check --manifest-path web/Cargo.toml && cargo clippy --manifest-path web/Cargo.toml --all-targets -- -D warnings && cargo test --manifest-path web/Cargo.toml`.

### PR 11 — frontend: split the three big Svelte components

> In `web/frontend/src/`, get these under 500 lines each by extracting subcomponents and pure helpers — no visual or behavioral changes: (a) `Stats.svelte` (683 = 163 script / 227 markup / 293 style): move the pure formatters (`formatNumber`, `formatPercent`, `formatTimestamp`, `formatTimeAgo`, `formatSpeed`, `formatEta`, `formatGradNorm`, color helpers) into `lib/format.ts`, and consider extracting the eval/best-model panel into a subcomponent with its own styles. (b) `GenericBoard.svelte` (623 = 256 script / 104 markup / 263 style): split into `GridBoard.svelte` (tictactoe/othello grid rendering) and `DropBoard.svelte` (all the `DROP_*` constants, drop animation logic, column click handling for Connect 4), with `GenericBoard.svelte` as a thin dispatcher on `board_type` — each component takes its styles with it. (c) `LossOverTimePage.svelte` (588): extract the range selector and/or tooltip into subcomponents and move pure chart math into the existing `lib/chart.ts`. Keep all props/events used by `App.svelte` unchanged. Only touch `web/frontend/src/`. Do not update CLAUDE.md. Verify with: `cd web/frontend && npm run check && npm run build`.

### PR 12 — docs sync (last, after all others merge)

> Update `CLAUDE.md` (and `documentation/ARCHITECTURE.md` if it references file layouts) to reflect the recent file-split refactors: the directory-structure tree, any per-crate test counts that changed, and the `trainer/src/trainer/` module list (new `solver_eval/` package, `replay_setup.py`, `checkpoint_runner.py`, `step_metrics.py`, orchestrator `promotion.py`/`eval_reporting.py`, model-watcher/mcts/actor/web module changes). Check the actual tree with `git ls-files` rather than trusting the old doc. Docs-only change: no code edits.

## Shared guardrails baked into every prompt

- Pure code motion — no behavior, API, or dependency changes; re-export moved names from their old paths.
- One component directory per PR, so PRs never overlap on files and can merge in any order (PR 4 after 1–3; PR 12 last).
- `CLAUDE.md` untouched until the final docs PR (it's the one file every PR would otherwise fight over).
- Run that component's formatter, linter, and full test suite (commands from `CLAUDE.md`) before pushing.
