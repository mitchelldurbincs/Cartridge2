# Cartridge2 — Engineering Review

**Date:** 2026-08-01
**Scope:** whole repository at `a1267ff` (engine, actor, web, trainer, frontend, infra, CI, docs)
**Method:** full source read of the Rust workspaces and Python package, plus targeted
reproduction of suspected defects. `cargo test --manifest-path engine/Cargo.toml` was run
and passes. No code was changed.

---

## 1. Executive summary

**Overall health: good, with a small number of serious silent-failure defects.**

This is an unusually well-tended hobby-scale ML platform. The engine layer is genuinely
well designed: the `Game` → `GameAdapter` → `ErasedGame` erasure boundary is clean, the
dynamic-width `LegalMask` fixed a real 64-action ceiling properly rather than by
workaround, and the engine-generated `game_metadata.json` manifest makes the Rust engine
the single source of truth for observation layout. Several modules
(`trainer/src/trainer/game_config.py`, `trainer/src/trainer/replay_setup.py`,
`actor/src/actor.rs`'s timeout/liveness coupling) carry comments that explain *why* a
decision was made and what invariant it protects — the rarest and most valuable form of
documentation. Those should not be touched.

**The most serious weaknesses are not architectural — they are silent failures.** The
recurring pattern is: a component encounters a problem, logs a warning or nothing at all,
substitutes a plausible-looking default, and carries on. Four instances of this pattern
cause real damage:

- The Python trainer's `_ensure_schema()` skips *every* statement in `sql/schema.sql`
  while logging "schema validated/created" (**confirmed by reproduction**).
- A malformed `config.toml` causes the Rust actor and web server to silently discard the
  entire file and run on built-in defaults — including a different game.
- The actor's win/loss/draw attribution is structurally incapable of ever recording a
  player-2 win, so the one metric that would reveal seat bias always reads 100% P1.
- The actor reports itself live and ready forever if it has never completed a single
  episode, so a permanently broken pod is never restarted.

**The main long-term maintenance risk** is the manually-mirrored game implementations:
`trainer/src/trainer/games/{tictactoe,connect4,generals}.py` reimplement the engine's
rules *and* observation encoding in Python, with no test comparing them against the Rust
engine. Every evaluation, every promotion decision, and the whole training loop's
gatekeeper depend on those mirrors agreeing byte-for-byte with an engine they are not
checked against. This is the one place where a future change lands as "the model stopped
improving" rather than as a test failure.

**Verdict: targeted cleanup, not restructuring.** The module boundaries are mostly right.
Fix the silent failures, close the two cross-language duplication gaps, pin the Rust
dependency graph, and this repository is in good shape for multi-year maintenance. There
is no case here for a redesign, and no case for adding layers, interfaces, or services.

---

## 2. Architecture map

```
                     config.defaults.toml  ──compile-time include──►  engine-config (Rust)
                              │                                              │
                              └──runtime read──►  central_config.py (Python)  │
                                                                              │
  ┌──────────────────────────────────────────────────────────────────────────┴────────┐
  │                                                                                   │
engine/ (library only, no I/O)                                                        │
  engine-core     Game trait, GameAdapter (typed→erased), registry, EngineContext,     │
                  GameMetadata, LegalMask, game_utils                                  │
  games-*         tictactoe, connect4, othello, generals_8x8                           │
  engine-games    registration + manifest generator + golden drift test ───► game_metadata.json
  mcts            search/tree/node/sampling + OnnxEvaluator (feature-gated)            │
  model-watcher   inotify + polling hot-reload of latest.onnx                          │
  metrics-common  Prometheus registration/encoding                                     │
                                                                                       │
actor/ (binary)                          web/ (binary)                                 │
  actor.rs   episode loop, MCTS,           startup.rs  router/CORS/AppState  ◄──────────┘
             transition build, backfill    game.rs     GameSession (parse_state)
  mcts_policy MCTS + temperature schedule  handlers/   game, stats, health
  storage/    PostgreSQL (deadpool)        types/      request/response DTOs
  stats.rs    actor_stats.json                    ▲
  health.rs   /health /ready /metrics            │ reads data/stats.json, data/actor_stats.json
                    │                            │
                    ▼                            │
             PostgreSQL transitions ◄────────────┴──── trainer/ (Python)
                    ▲                                    trainer.py     train loop
                    │                                    storage/       PG + S3 + FS
                    └──── data/models/latest.onnx ◄────── checkpoint_runner (ONNX export)
                                                          evaluator.py  vs random baseline
                                                          games/*.py    PYTHON RULE MIRRORS
                                                          orchestrator/ composition root over `crucible`
```

### Where state and decisions are owned

| State / decision | Owner today | Correct? |
|---|---|---|
| Game rules, observation layout | `engine/games-*` (Rust) | Yes — and the manifest propagates it |
| ...*except* for evaluation | `trainer/src/trainer/games/*.py` (duplicate) | **No** — second implementation |
| Legal-move extraction | `GameMetadata` / `LegalMask` from observation bytes | Yes |
| Replay buffer schema | `sql/schema.sql`, applied by two divergent splitters | **No** — one is broken |
| Config defaults | `config.defaults.toml`, consumed two different ways | Mostly; drift risk is real |
| Which game the trainer trains | `game_config.py` (manifest) | Yes |
| Which game the *evaluator* evaluates | DB metadata, falling back to manifest | **No** — contradicts the above |
| Actor liveness | `HealthState`, but nothing ever marks unhealthy | **No** — gap |
| Current web game session | Single global `Mutex<GameSession>` | Adequate single-user; wrong for >1 client |
| Model file → evaluator | `model-watcher`, via two racing tasks | Works, but duplicates |

### Boundaries that are unclear

1. **Who owns "the game's facts"** — the engine manifest, or the `game_metadata` DB row?
   `game_config.py` says the manifest; `evaluator.py` prefers the DB. Both are live.
2. **Who owns the replay schema** — `sql/schema.sql` is the artifact, but two independent
   parsers apply it and only one is correct.
3. **Who decides an actor is unhealthy** — nobody. `set_unhealthy()` is `#[allow(dead_code)]`.

---

## 3. Prioritized findings

### CRITICAL

---

#### C1. `_ensure_schema()` silently skips every statement in `sql/schema.sql`

- **Severity:** Critical
- **Confidence:** Confirmed (reproduced)
- **Location:** `trainer/src/trainer/storage/postgres.py:113-127` (splitting at lines 119-124);
  correct sibling implementation at `actor/src/storage/postgres.rs:32-42`
- **Current behavior:** The method splits `schema.sql` on `;`, then skips any chunk whose
  stripped text `startswith("--")`. Because every statement in `sql/schema.sql` is preceded
  by a comment line, the comment becomes the *first line of the chunk* and the whole
  statement is dropped. Reproduction:

  ```
  chunk 0: skipped=True   '-- Cartridge2 PostgreSQL Schema'   → CREATE TABLE transitions
  chunk 1: skipped=True   '-- Indices for efficient querying' → CREATE INDEX idx_transitions_timestamp
  chunk 2: skipped=False  'CREATE INDEX idx_transitions_episode ...'
  chunk 3: skipped=False  'CREATE INDEX idx_transitions_env_id ...'
  chunk 4: skipped=True   '-- Game metadata table ...'        → CREATE TABLE game_metadata
  ```

  Both `CREATE TABLE`s and one index are dropped; the method then logs
  `"PostgreSQL schema validated/created"`.
- **Why it is a problem:** Against a fresh database the two surviving `CREATE INDEX`
  statements fail with `UndefinedTable`, and the error message points at an index rather
  than at the missing table. Against an existing database it "works" only because the Rust
  actor created the schema first — the Python path has never actually been exercised. The
  same logic exists correctly 40 lines away in Rust (`split_sql_statements` strips comment
  *lines* before splitting); this is a duplicated concept where one copy is wrong.
- **Realistic consequence:** `python -m trainer train` against a new database, or in any
  deployment where the trainer starts before the actor, fails with a misleading error. If
  a future schema change adds a column, the trainer will report success while having
  applied nothing.
- **Recommended change:** Delete the Python splitter and port the Rust one verbatim: filter
  out lines whose `lstrip()` starts with `--`, join, split on `;`, drop empties. Add a
  regression test asserting the splitter yields exactly the 5 statements in `schema.sql`
  and that none of them starts with `--`.
- **Why better:** Removes a duplicated-and-divergent implementation and turns a silent
  no-op into a real operation. It is a five-line change.
- **Scope:** small · **Risk:** very low · **Validation:** unit test on the splitter output;
  integration test that runs `_ensure_schema()` against an empty Postgres and then
  `count()`.

---

#### C2. Malformed `config.toml` silently discards the entire configuration

- **Severity:** Critical
- **Confidence:** Confirmed (code reading)
- **Location:** `engine/engine-config/src/loader.rs:54-68`
- **Current behavior:**
  ```rust
  Err(e) => {
      warn!("Failed to parse {}: {}, using defaults", path.display(), e);
      apply_env_overrides(CentralConfig::default())
  }
  ```
  A TOML syntax error, a misspelled key that fails to deserialize, or an unreadable file
  causes the actor and web server to abandon the operator's entire config and run on
  built-in defaults.
- **Why it is a problem:** `config.toml` currently sets `env_id = "connect4"`; the built-in
  default is `"tictactoe"`. A single stray character therefore starts a self-play actor
  generating *tictactoe* episodes into a buffer the trainer expects to hold connect4 —
  behind one `warn!` line in a log that is otherwise full of `info!`. Python's loader
  (`central_config.py:417-422`) lets `tomllib` raise, so the two languages disagree about
  whether a broken config is fatal. Note also that the `parse` arm of the `env_override!`
  macro (`loader.rs:79-85`) maps a parse failure to `VarError::NotPresent`, so
  `CARTRIDGE_TRAINING_BATCH_SIZE=sixty` is silently ignored too.
- **Realistic consequence:** Hours of compute produce a model for the wrong game, or with
  the wrong learning rate. `check_metadata_agrees` catches the *layout* half of this (the
  actor writes tictactoe metadata, the trainer expects connect4) — but only after the actor
  has already filled the buffer, and not at all if both happen to be misconfigured the
  same way.
- **Recommended change:** Make `load_config()` return `Result<CentralConfig, ConfigError>`
  and let `main()` in both binaries exit non-zero on a parse/read failure of a file that
  was *found*. Keep the "no config file at all → defaults" path as-is (that is a legitimate
  first-run case). Separately, make a failed `parse` in `env_override!` log a warning
  naming the variable rather than silently no-op.
- **Why better:** A misconfiguration should stop the process at second zero, not surface as
  a bad model an hour later. "File absent" and "file present but broken" are different
  situations and should have different outcomes.
- **Scope:** small · **Risk:** low — this changes startup behavior for currently-broken
  configs only · **Validation:** unit tests for `load_from_path` on valid/invalid/missing
  files; a smoke test asserting the actor exits non-zero on a corrupt `config.toml`.

---

#### C3. Actor outcome attribution can never record a player-2 win

- **Severity:** Critical (silent corruption of the primary training diagnostic)
- **Confidence:** Confirmed
- **Location:** `actor/src/actor.rs:699` and `:497,:508`; `actor/src/stats.rs:96-109`;
  `actor/src/metrics.rs:231-240`; consumed at `web/frontend/src/Stats.svelte:371-385`
- **Current behavior:** `engine_core::game_utils::calculate_reward(winner, previous_player)`
  (`game_utils.rs:114-134`) returns the reward **from the perspective of the player who
  just moved**. The winning move is always made by the winner, so the terminal step's reward
  is `+1.0` for *either* player's victory. `actor.rs` accumulates
  `total_reward += step_result.reward` and passes that sum to:
  ```rust
  self.stats.record_episode(steps, total_reward);   // actor.rs:508
  metrics::record_outcome(total_reward);            // actor.rs:497
  ```
  Both interpret positive as "player 1 won" (`stats.rs:102-108`,
  `metrics.rs:231-240`, whose comment explicitly claims `+1 = player 1 wins`).
- **Why it is a problem:** `player2_wins` and `actor_player2_wins_total` are structurally
  pinned at zero; `player1_wins` counts every decisive game. The frontend renders these as
  a P1/Draw/P2 outcome bar, which will always show 100% of decisive games as P1. This is
  not a cosmetic bug: the codebase itself identifies seat imbalance as a training-collapse
  mode — `metadata.rs:56-59` warns that "a systematically advantaged seat lets the value
  head collapse into a seat detector", and `game_config.py:62-67` records that exact failure
  being observed in generals. The single metric that would surface it is broken.
- **Realistic consequence:** A first-player-advantage blowup, or a bug that makes one seat
  always win, is invisible in both the dashboard and Prometheus. An operator staring at a
  100%-P1 bar has no way to tell whether that is the bug or the truth.
- **Recommended change:** Stop deriving the outcome from an accumulated reward. Decode the
  winner from the terminal state (the games already carry a `winner` byte, which
  `web/src/game.rs::parse_state` reads) or have `run_episode` return an explicit
  `Outcome { Player1Win, Player2Win, Draw }` computed from the final mover's identity plus
  the sign of the terminal reward. Pass that enum to `ActorStats` and `metrics`, replacing
  the `f32` parameter.
- **Why better:** Replaces a lossy float that three call sites reinterpret with a type that
  cannot be misread, and removes the implicit "reward sign encodes seat" assumption that is
  false in this engine.
- **Scope:** small–medium (touches `actor.rs`, `stats.rs`, `metrics.rs`, and their tests) ·
  **Risk:** low · **Validation:** an actor-level test playing a scripted tictactoe game to a
  player-2 win and asserting `snapshot.player2_wins == 1`. Note the existing tests
  (`stats.rs:317-346`) assert the *current* wrong mapping and must be rewritten.

---

### HIGH

---

#### H1. A permanently broken actor reports healthy and ready forever

- **Severity:** High · **Confidence:** Confirmed
- **Location:** `actor/src/health.rs:98-105` and `:76-80`; `actor/src/actor.rs:546-550`
- **Current behavior:** `is_making_progress()` returns `true` when `last_episode_time == 0`
  ("no episodes completed yet, but that's ok during startup"). `set_unhealthy()` is marked
  `#[allow(dead_code)]` and is never called anywhere. The main loop's error arm logs
  `error!("Episode {} failed: {}", ...)` and continues.
- **Why it is a problem:** If every episode fails after startup — the database becomes
  unreachable, the model file is corrupt, the game panics on a specific state — the actor
  spins forever, `record_episode_complete()` is never reached, `last_episode_time` stays
  `0`, and `/health` returns 200 indefinitely. Kubernetes will never restart it and
  `/ready` (which only checks `ready && healthy`) keeps it in the Service.
- **Realistic consequence:** A silent, permanently-idle actor fleet. Episodes-per-second
  drops to zero and no probe fires; the only signal is the absence of new transitions, which
  nothing alerts on.
- **Recommended change:** Two small changes. (a) Record a *start* timestamp in `HealthState`
  and treat "no episode completed within the liveness window since start" as not making
  progress, rather than special-casing `0` as healthy. (b) Track consecutive episode
  failures in the main loop and call `set_unhealthy()` past a threshold, so the existing
  flag stops being dead code.
- **Why better:** Gives the probe the one thing it currently lacks — the ability to
  distinguish "starting up" from "has never worked". The existing window sizing
  (`effective_episode_timeout_secs` × 2, `health.rs:51-61`) is already correct and would be
  reused unchanged.
- **Scope:** small · **Risk:** low, but validate the window against the slowest game
  (generals, horizon 402) so a genuinely slow first episode is not killed ·
  **Validation:** extend the existing `test_progress_*` tests with a "never completed an
  episode, window elapsed" case.

---

#### H2. No `Cargo.lock` for the deployed binaries; every security scan is `continue-on-error`

- **Severity:** High · **Confidence:** Confirmed
- **Location:** `.gitignore:7` (`Cargo.lock`); `.github/workflows/ci.yml` — `cargo audit`,
  `pip-audit`, `npm audit`, and gitleaks all carry `continue-on-error: true`
- **Current behavior:** `.gitignore` excludes `Cargo.lock` wholesale. `actor/Cargo.lock` and
  `web/Cargo.lock` do not exist on disk. Every CI security job is configured so it can never
  fail the build.
- **Why it is a problem:** `actor` and `web` are **binaries**, not libraries — the Cargo
  convention is to commit their lockfiles precisely so the artifact you build in CI is the
  artifact you tested. Without them, each CI run and each developer resolves fresh
  semver-compatible versions of the entire transitive graph, including `ort` (the ONNX
  runtime binding, which is the component most likely to change inference behavior between
  patch releases), `deadpool-postgres`, `axum`, and `tokio`. Combined with advisories that
  cannot fail the build, there is no pinning and no gate.
- **Realistic consequence:** "It built and passed yesterday, it fails today, nothing
  changed" — with no lockfile to bisect. A behavioral change in `ort` would show up as
  degraded model play rather than a build error. A published advisory in any dependency is
  reported and ignored.
- **Recommended change:** Remove `Cargo.lock` from `.gitignore`, commit
  `engine/Cargo.lock`, `actor/Cargo.lock`, `web/Cargo.lock`, and add `--locked` to CI build
  and test commands. Separately, decide deliberately which audits gate: at minimum make
  `cargo audit` and gitleaks fail the build, and if `pip-audit`/`npm audit` are too noisy,
  scope them (`--ignore-vuln`) rather than blanket-suppressing.
- **Why better:** Reproducible builds are the precondition for debugging anything else on
  this list. Scanners that cannot fail provide the appearance of supply-chain hygiene
  without the substance.
- **Scope:** small · **Risk:** low; expect a one-time flurry of advisories to triage ·
  **Validation:** CI green with `--locked`; deliberate `cargo update` PRs thereafter.

---

#### H3. The web server has exactly one game, shared by every client

- **Severity:** High · **Confidence:** Confirmed
- **Location:** `web/src/startup.rs:49-62` (`AppState.session: Mutex<GameSession>`);
  `web/src/handlers/game.rs:88-131, 134-189`
- **Current behavior:** `AppState` holds a single `GameSession`. `POST /game/new` replaces
  it; `GET /game/state` and `POST /move` operate on it. There is no session identity in any
  request or response.
- **Why it is a problem:** Two browsers on the same server share one board. The second
  player's "New Game" resets the first player's position mid-game; their moves are rejected
  with "Not your turn" from a state they never saw. This is correct-by-construction for a
  single local user, which is the documented use case — but it is invisible in the API
  shape, so the failure mode is silent cross-talk rather than an error, and the `/games`
  and `/game-info/:id` handlers already contain guard logic (`game.rs:53-74, 101-112`)
  written as though multi-tenancy mattered.
- **Realistic consequence:** The first time this is demoed to two people at once, or
  deployed behind the k8s manifests that already exist in `k8s/base/web/`, it looks
  haunted. There is no log line that would explain it.
- **Recommended change:** The smallest honest fix is a session id: have `POST /game/new`
  return an opaque `session_id`, require it on `/move` and `/game/state`, and store
  sessions in a `Mutex<HashMap<SessionId, GameSession>>` with a last-touched timestamp and
  a cap (say 1 000 sessions / 30-minute idle eviction) so it cannot grow without bound.
  If multi-user support is explicitly out of scope, the alternative is equally acceptable
  and much cheaper: document the single-session constraint in `documentation/API.md` and
  have the handlers return `409 Conflict` when a request arrives for a session other than
  the active one — but do not leave it undocumented and silent.
- **Why better:** Either choice makes the constraint explicit at the API boundary instead
  of leaving it as an emergent property of `AppState`'s shape.
- **Scope:** medium (session map) or small (documented + 409) · **Risk:** low; the
  session map needs the eviction cap or it becomes an unbounded collection ·
  **Validation:** a two-concurrent-client integration test in `web/src/main_tests.rs`.

---

#### H4. Every training step runs a full `COUNT(*)` over `transitions`

- **Severity:** High · **Confidence:** Confirmed
- **Location:** `trainer/src/trainer/storage/postgres.py:196-266`, specifically the
  `self.count(env_id)` call at `:207-209`; interacts with `trainer.py:134-136`
- **Current behavior:** `sample()` computes its `TABLESAMPLE` percentage as
  `min(100.0, (batch_size * 10.0) / max(1, self.count(env_id)))`. `count()` issues
  `SELECT COUNT(*) FROM transitions WHERE env_id = %s` — a sequential scan in PostgreSQL —
  and does so from inside `sample()`'s own `_connection()` block, checking out a *second*
  pooled connection for the duration.
- **Why it is a problem:** Three separate costs. (1) `trainer.py:134-136` introduces
  `_buffer_size_cache` with the comment "avoid expensive count() calls every step" — and
  `sample()` then performs exactly that count on every step anyway, defeating the cache
  entirely. (2) At 500 episodes × ~25 moves per iteration the table holds ~12 500 rows and
  the scan is cheap; at the sliding-window sizes the config contemplates it is not, and the
  cost grows linearly with buffer size on the hottest path in the system. (3) The nested
  checkout means `sample()` requires ≥2 free connections; with `pool_size=1` psycopg2's
  `ThreadedConnectionPool` raises `PoolError: connection pool exhausted` rather than
  blocking.
- **Realistic consequence:** Training throughput degrades as the buffer grows, in a way
  that looks like "PyTorch got slower" rather than "the sampler is scanning the table".
  Note also that when the sample percentage is small `TABLESAMPLE` returns fewer than
  `batch_size` rows and the code falls through to a second full `ORDER BY RANDOM()` query
  (`:238-264`) — so the common path is *three* queries per training step, one of which sorts
  the whole table.
- **Recommended change:** Pass the already-known buffer size into `sample()` (the trainer
  has it in `_buffer_size_cache`) instead of recomputing it, or cache it inside the buffer
  with a refresh interval. Then reuse the caller's connection rather than nesting. For the
  sampling itself, `ORDER BY RANDOM()` on a large table is the real cost — a random-offset
  or `id`-range approach would be a worthwhile follow-up, but the `COUNT(*)` removal is the
  cheap 80%.
- **Why better:** Restores the caching the trainer already tried to do, removes the nested
  pool checkout, and cuts one full scan per training step.
- **Scope:** small · **Risk:** low — a slightly stale count only perturbs the sample
  percentage · **Validation:** benchmark `sample()` at 10k / 100k / 1M rows before and
  after; assert `pool_size=1` no longer raises.

---

#### H5. Game rules and observation encoding are reimplemented in Python with no conformance test

- **Severity:** High · **Confidence:** Confirmed
- **Location:** `trainer/src/trainer/games/tictactoe.py:109-133`,
  `connect4.py`, `generals.py` (433 lines, docstring: "Mirrors engine/games-generals rules
  exactly"); consumed by `trainer/src/trainer/policies/onnx.py:62`
  (`state.to_observation(config)`) and `evaluator.py:157-202`
- **Current behavior:** Each Python mirror hand-writes the observation layout, e.g.
  tictactoe places player-1 pieces in plane 0 and player-2 in plane 1, writes the legal
  mask at `config.legal_mask_offset`, and the player one-hot at
  `legal_mask_offset + num_actions`. The engine manifest supplies only the *dimensions*
  (`obs_size`, `legal_mask_offset`, `obs_channels`, `player_relative_obs`); the *semantics*
  are duplicated. No test compares a Rust-produced observation to a Python-produced one for
  the same position.
- **Why it is a problem:** The manifest work already established that the engine owns these
  facts — `game_config.py:5-8` says so explicitly and a golden test enforces manifest
  freshness. But that mechanism stops at the numbers. A change to plane ordering, to the
  meaning of `player_relative_obs` (generals sets it `true` with 9 channels), or to where
  the pass action sits, updates the Rust side and the manifest dimensions while leaving the
  Python encoder wrong — and the manifest golden test still passes, because the *dimensions*
  did not change.
- **Realistic consequence:** `python -m trainer evaluate` feeds the ONNX model observations
  it was never trained on. Win rate collapses toward random. Because evaluation drives
  promotion (`--promotion-metric win_rate`), a correct model is rejected and a worse one is
  kept — and the symptom is "training regressed", which nobody would trace to a Python file
  in `games/`.
- **Recommended change:** Add a cross-language conformance test. The cheapest form: a small
  Rust binary (sibling to the existing manifest generator in `engine-games/src/bin/`) that
  emits, for each registered game, a fixed sequence of seeded positions with their exact
  observation bytes as a JSON golden file; a pytest then replays the same action sequence
  through the Python mirror and asserts byte equality. This reuses the pattern the manifest
  generator already established and costs one new binary plus one test.
- **Why better:** Converts a silent training-quality regression into a red test, using the
  same generator-plus-golden mechanism the repository already trusts for metadata.
- **Scope:** medium · **Risk:** low; expect the first run to surface genuine existing
  discrepancies · **Validation:** the test itself.

---

#### H6. Two components disagree about who owns game facts

- **Severity:** High · **Confidence:** Confirmed
- **Location:** `trainer/src/trainer/evaluator.py:33-58` vs
  `trainer/src/trainer/game_config.py:1-17`
- **Current behavior:** `game_config.py`'s module docstring states the engine manifest is
  the single source of truth and that this module "must never restate them". Meanwhile
  `get_game_metadata_or_config()` connects to PostgreSQL, prefers the `game_metadata` row,
  and only falls back to the manifest when the database is unreachable — swallowing any
  failure with `except Exception as e: logger.warning(...)`. It returns
  `GameConfig | GameMetadata`, two types with different fields: `GameMetadata` has no
  `obs_channels`, no `player_relative_obs`, no `network_type`.
- **Why it is a problem:** The evaluator's observation encoding therefore depends on
  *database reachability*. With the DB up it uses the mutable row written by whichever actor
  started most recently; with the DB down it uses the manifest. Those can differ, and
  `replay_setup.check_metadata_agrees` — the function whose entire job is detecting exactly
  this disagreement — is not consulted on this path. The union return type pushes the
  ambiguity onto every downstream caller (`OnnxPolicy.select_action`, `play_game`,
  `state.to_observation`), none of which can access the fields only one branch provides.
- **Realistic consequence:** Evaluation silently uses a different observation layout than
  training, depending on transient infrastructure state — the hardest class of bug to
  reproduce, because it is not deterministic in the code.
- **Recommended change:** Delete `get_game_metadata_or_config()`. Have the evaluator call
  `game_config.get_config(env_id)` unconditionally, matching the trainer. If cross-checking
  against the DB row has value, call `check_metadata_agrees` and *raise* on disagreement —
  which is what it already does for training. The return type becomes plain `GameConfig`.
- **Why better:** Restores the single ownership the manifest was built to establish,
  removes a union type from four call sites, and eliminates a `except Exception` that hides
  connection failures.
- **Scope:** small · **Risk:** low · **Validation:** existing `test_evaluator.py` plus a
  test asserting evaluation is identical with the DB unreachable.

---

#### H7. Value-target backfill assumes strict turn alternation, unasserted and untested

- **Severity:** High · **Confidence:** Confirmed (the assumption; the consequence is latent)
- **Location:** `actor/src/actor.rs:587-621`, specifically `:598-602`
- **Current behavior:**
  ```rust
  let steps_from_end = total_steps.saturating_sub(1).saturating_sub(t.step_number);
  let sign = if steps_from_end % 2 == 0 { 1.0 } else { -1.0 };
  t.game_outcome = Some(final_reward * sign);
  ```
  Every transition's value target is derived from step-index parity.
- **Why it is a problem:** This is correct only if the acting player alternates on *every
  recorded step*, for every game, forever. That invariant lives nowhere except this
  comment — no assertion, no test, and no field in `GameMetadata` declaring it. It holds
  today (Othello's pass is an explicit action, so it consumes a step), but it is exactly the
  kind of thing a new game breaks: the real Generals ruleset allows multiple moves per turn,
  and the crate lib.rs notes the 8×8 variant was deliberately restricted to "alternating
  turns" — a restriction this code silently depends on.
- **Realistic consequence:** A game that ever records two consecutive moves by the same
  player gets **half its value targets sign-inverted**. Training does not error; the value
  head simply never converges. Diagnosing that from "the model doesn't learn" would take
  days.
- **Recommended change:** Make the invariant explicit and checkable. Add
  `alternating_turns: bool` to `GameMetadata` (it is engine-owned, flows through the
  manifest for free, and the DB row already carries similar fields), have `finalize_episode`
  assert it, and have each game declare it. Longer term the robust fix is to record the
  acting player per transition and derive the sign from it — but the assertion is the cheap
  step that converts a silent trainer bug into a loud actor bug.
- **Why better:** Turns an invisible cross-component assumption into a declared, tested
  property of each game, checked at the point that depends on it.
- **Scope:** small (assertion + metadata field) or medium (per-transition player) ·
  **Risk:** low · **Validation:** a test asserting the sign pattern for a scripted 5-move
  and 6-move episode, and that a synthetic non-alternating game trips the assertion.

---

#### H8. CI rewrites contributors' branches instead of enforcing formatting

- **Severity:** High · **Confidence:** Confirmed
- **Location:** `.github/workflows/ci.yml` — `rust-fmt` and `python-lint` jobs
- **Current behavior:** `rust-fmt` runs `cargo fmt` (not `--check`) across all three
  workspaces and then `stefanzweifel/git-auto-commit-action@v5` pushes a `style:` commit
  back to the PR branch. `python-lint` does the same with `ruff check --fix` and `black`.
  The workflow declares `permissions: contents: write` and triggers on `pull_request`, and
  the checkout uses `ref: ${{ github.head_ref }}`.
- **Why it is a problem:** Three distinct issues. (1) `CLAUDE.md` documents
  `cargo fmt --check` as the contract, but `--check` never runs anywhere — formatting is
  mutated, never verified. (2) On a fork PR the `pull_request` trigger issues a read-only
  token, so the push fails and the job goes red for reasons unrelated to the contribution;
  `ref: github.head_ref` also resolves against the base repository, which is not where the
  fork's commits are. External contribution is effectively blocked. (3) Auto-commits land
  *after* review, invalidating approvals and re-triggering CI.
- **Realistic consequence:** Any outside contributor's first PR fails on a formatting job
  they cannot fix. Maintainers develop the habit of ignoring that job, which is exactly the
  job that should be trustworthy.
- **Recommended change:** Replace the format jobs with verification: `cargo fmt --check`,
  `ruff check`, `black --check`. Fixing formatting is a two-second local command and belongs
  to the author. If auto-fixing is genuinely wanted, move it to a manually-dispatched
  workflow or a bot comment, not the PR gate.
- **Why better:** Makes CI a gate rather than a mutator, unblocks forks, stops post-approval
  commits, and aligns CI with the documented commands.
- **Scope:** small · **Risk:** low; the first run will red-flag any current drift ·
  **Validation:** run the `--check` variants locally first.

---

### MEDIUM

---

#### M1. `GameMetadata::extract_legal_mask` / `legal_mask_bits` overflow for real games and disagree with their sibling on short input

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `engine/engine-core/src/metadata.rs:166-205` vs `:241-258`
- **Current behavior:** `legal_mask_bits()` computes `(1u64 << self.num_actions) - 1`. For
  othello (65 actions) and generals (257) this shift is out of range: a debug-build panic,
  and in release a masked shift producing garbage. `extract_legal_mask()` calls it as the
  fallback for a too-short observation, meaning short input yields **all actions legal**.
  Forty lines away, `is_action_legal()` returns **false** for the same out-of-bounds case,
  and `extract_legal_moves()` (built on it) returns an empty list.
- **Why it is a problem:** Two functions in the same struct answer "the observation is
  truncated" with opposite defaults — fail-open vs fail-closed. Both are `pub`. The doc
  comment on `legal_mask_bits` acknowledges the overflow and tells callers to use something
  else, which is documentation standing in for a design decision. Production callers have
  correctly migrated to `legal_mask_from_obs`; these two are now dead weight that still
  compiles, is still public, is still tested (`metadata.rs:294-300, 315-363`), and is still
  reachable by the next person who greps for "legal mask".
- **Realistic consequence:** A new game or a new call site picks the u64 API, works on
  tictactoe and connect4, and panics or silently permits illegal moves on othello/generals.
- **Recommended change:** Delete both methods and their tests. `LegalMask` covers every use.
  If a u64 view is genuinely needed somewhere, expose it on `LegalMask` with an explicit
  `Option<u64>` return for >64 actions.
- **Scope:** small · **Risk:** low (`grep` confirms no production callers) ·
  **Validation:** compilation; the `LegalMask` tests already cover the replacement.

---

#### M2. Model hot-reload races itself between the inotify and polling paths

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `engine/model-watcher/src/lib.rs:263-326` (inotify task) and `:336-405`
  (polling task)
- **Current behavior:** Two independent tokio tasks both call `load::load_model_static` on
  the same shared `evaluator` and `last_mtime`. The inotify task debounces against a
  task-local `last_reload` (`:271, :292`); the polling task debounces against the shared
  `last_mtime`. Neither coordinates with the other, and both sleep 100 ms between deciding
  to reload and reloading.
- **Why it is a problem:** A single model write can trigger both paths within the same
  window: two full ONNX `Session` constructions, two writes to the evaluator, and two `()`
  events on the channel. In the actor that means `metrics::MODEL_RELOADS.inc()` fires twice
  per model (`actor.rs:428-433`), so "how many model generations has this actor consumed"
  — the metric you would check to diagnose a stalled loop — over-reports by up to 2×. The
  duplicate `Session` construction also doubles peak memory briefly.
- **Recommended change:** Give the two paths a shared `Mutex<()>` reload guard, or collapse
  them: have inotify events simply wake the polling task rather than performing their own
  load. The mtime check the polling path already performs is the correct idempotency key —
  make it the only entry point.
- **Scope:** small · **Risk:** low · **Validation:** a test that writes the model file once
  and asserts exactly one channel event within a 2-second window.

---

#### M3. Unreadable stats files are reported to the UI as zeroed stats

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `web/src/handlers/stats.rs:12-28`
- **Current behavior:** `read_stats_file` returns `T::default()` both when the file is
  absent (silently) and when `serde_json` fails to parse it (with a `warn!`).
- **Why it is a problem:** "Training has not started" and "the stats file is corrupt or
  written by an incompatible version" render identically: step 0, loss 0.0, buffer size 0.
  An operator watching the dashboard during an incident sees a plausible number and
  concludes training reset, when in fact the writer and reader disagree about the schema.
- **Recommended change:** Keep `default()` for `NotFound`. Return `503` with a short body
  for a parse failure, and let the frontend distinguish "no data yet" from "stats
  unavailable". The frontend's `getStats()` (`api.ts:135-139`) already throws on non-OK.
- **Scope:** small · **Risk:** low · **Validation:** handler test with a truncated JSON file.

---

#### M4. Three environment-variable conventions, and config keys the Rust side silently ignores

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `engine/engine-config/src/loader.rs:105-302` (`CARTRIDGE_*`, explicit list);
  `actor/src/config.rs:26-113` (`ACTOR_*`, a second explicit list);
  `trainer/src/trainer/central_config.py:267-315` (`CARTRIDGE_*` generic + legacy
  `ALPHAZERO_*` + bare `DATA_DIR`)
- **Current behavior:** Three parallel override mechanisms. Python parses `CARTRIDGE_*`
  generically, so any key can be overridden; Rust matches a hand-maintained list, so keys
  outside it are ignored. `actor/src/config.rs` adds a third namespace, and
  `default_temp_threshold()` (`:90-94`) hardcodes `.unwrap_or(0)` instead of reading
  `CENTRAL_CONFIG.mcts.temp_threshold` — so setting `mcts.temp_threshold` in `config.toml`
  has no effect on the actor at all, despite the key existing in `config.defaults.toml`,
  in the Rust struct, and in `SCHEMA.md`.
- **Why it is a problem:** `CLAUDE.md:355-370` already documents this as a known gotcha with
  a list of ignored keys — which is the tell that the design has a hole and the response was
  documentation. A key that exists in the schema, parses, and does nothing is worse than a
  key that does not exist. `temp_threshold` in particular controls the exploration schedule;
  an operator tuning it in `config.toml` gets no error and no effect.
- **Recommended change:** Two steps, both small. (a) Fix `default_temp_threshold` to read
  the central config, closing the concrete hole. (b) Add a test that walks the
  `CentralConfig` struct fields and asserts every one has a corresponding `env_override!`
  entry, so the list cannot silently fall behind the schema. Leave the `ACTOR_*` aliases
  alone or deprecate them in a separate pass — they are a compatibility surface, not a bug.
- **Scope:** small · **Risk:** low · **Validation:** the coverage test; a manual check that
  `CARTRIDGE_MCTS_TEMP_THRESHOLD=15` changes actor behavior.

---

#### M5. MCTS replays the full path from root on every simulation

- **Severity:** Medium · **Confidence:** Confirmed (the behavior; the impact is
  workload-dependent)
- **Location:** `engine/mcts/src/search.rs:295-315` (`reconstruct_position`), called from
  `select_leaf` at `:275`
- **Current behavior:** The tree stores no game state. Each selection walks to a leaf and
  then re-executes every action from the root via `ctx.step_into`, costing `O(depth)` engine
  steps per simulation. Expansion (`:401-450`) additionally steps once per legal action.
- **Why it is a problem:** Total engine steps per search are roughly
  `simulations × average_depth + expansions × branching_factor`. `CLAUDE.md` records
  generals' mean branching factor as 37 (p90 71) against a 50–250 simulation budget — so
  expansion alone dominates, and the quadratic replay term compounds it as the tree deepens.
  The repository already ships `generals_search_diag` to measure this and lists "generals
  training does not beat random" as a known gap; this is a plausible contributor.
- **Why it is nonetheless defensible today:** not storing states keeps the arena small and
  the code short, and for tictactoe/connect4 (depth ≤ 42) the replay cost is negligible.
  This is the right trade at those sizes.
- **Recommended change:** Do not restructure speculatively. First run
  `cargo bench -p mcts` and `generals_search_diag` to quantify the split between replay and
  expansion. If replay dominates, cache the state bytes on expanded nodes behind a
  configurable node budget (state is already `Vec<u8>` and small for every current game).
  If expansion dominates, the fix is a progressive-widening cap on children, not caching.
- **Scope:** medium · **Risk:** medium — MCTS is the correctness-critical hot path ·
  **Validation:** the existing `mcts/benches/mcts.rs` before/after, plus
  `generals_strength_probe` to confirm playing strength is unchanged.

---

#### M6. `cleanup()` ignores `env_id` and uses an anti-join that degrades

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `trainer/src/trainer/storage/postgres.py:300-320`; called from
  `replay_setup.py:199-210`
- **Current behavior:**
  ```sql
  DELETE FROM transitions
  WHERE id NOT IN (SELECT id FROM transitions ORDER BY created_at DESC LIMIT %s)
  ```
  No `env_id` filter, and a `NOT IN` against a subquery of `window_size` rows.
- **Why it is a problem:** Two issues. (1) The sliding window is global, so in a database
  holding two games' transitions, trimming to a 100k window for connect4 deletes othello's
  data as collateral — while every other method on the class takes an `env_id` parameter,
  so the asymmetry is easy to miss. (2) `NOT IN (subquery)` cannot use an index and
  materializes the keep-set for every row; at buffer sizes where cleanup matters, it is the
  slowest query in the system, run on the training thread.
- **Recommended change:** Add `env_id` to the signature (matching every sibling method) and
  rewrite as a keyed delete: `DELETE FROM transitions WHERE env_id = %s AND created_at <
  (SELECT created_at FROM transitions WHERE env_id = %s ORDER BY created_at DESC OFFSET %s
  LIMIT 1)`, backed by an index on `(env_id, created_at)` — which `sql/schema.sql` does not
  currently have.
- **Scope:** small · **Risk:** low — verify the boundary case where fewer than
  `window_size` rows exist (the subquery returns NULL; guard it) · **Validation:** unit test
  with two `env_id`s asserting only the target game is trimmed.

---

#### M7. The two fields that exist to expose data loss are never displayed

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `actor/src/stats.rs:55-61` (fields and their rationale);
  `web/frontend/src/lib/api.ts:147-159` (TS interface omits them);
  `web/frontend/src/Stats.svelte` (no reference)
- **Current behavior:** `episodes_abandoned` and `transitions_discarded` are computed by the
  actor, serialized into `actor_stats.json`, deserialized by `web/src/types`, and served on
  `/actor-stats` — then dropped, because the frontend's `ActorStats` interface does not
  declare them and no component reads them.
- **Why it is a problem:** Their doc comment states the purpose exactly: "a non-zero value
  here means self-play data is being lost — and lost with a bias, since the episodes that
  run out of wall clock are the long ones." The whole point is operator visibility, and the
  last hop is missing. The `actor_episodes_abandoned_total` Prometheus counter exists, but
  the dashboard the project actually ships does not show it.
- **Recommended change:** Add both fields to the TS interface and render them in
  `Stats.svelte` next to the outcome bar, highlighted when non-zero.
- **Scope:** small · **Risk:** none · **Validation:** visual; `npm run check` catches the
  interface change.

---

#### M8. `GAMES_ACTIVE` leaks on every abandoned browser game

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `web/src/handlers/game.rs:94-96` and `:168-183`
- **Current behavior:** `new_game` increments `GAMES_ACTIVE` unconditionally. It is
  decremented only when a game reaches a terminal state inside `make_move`. Starting a new
  game while one is in progress — the common case, since there is only one session —
  increments again without decrementing the abandoned one.
- **Realistic consequence:** `web_games_active` climbs monotonically and never returns to
  zero. Any alert or dashboard built on it is meaningless. Given H3 (one global session),
  the gauge's true value is always 0 or 1.
- **Recommended change:** Decrement in `new_game` when the outgoing session was unfinished,
  or derive the gauge from the session map once H3 is addressed.
- **Scope:** small · **Risk:** none · **Validation:** handler test issuing two `new_game`
  calls and asserting the gauge.

---

#### M9. PostgreSQL connections are hardcoded to `NoTls`

- **Severity:** Medium · **Confidence:** Confirmed
- **Location:** `actor/src/storage/postgres.rs:13, 173`
- **Current behavior:** `cfg.create_pool(Some(Runtime::Tokio1), NoTls)`. No configuration
  path can enable TLS; any `sslmode` in the connection string is silently ignored because
  the pool config is rebuilt field-by-field from the parsed URL (`:139-160`) and `sslmode`
  is not among the copied fields.
- **Why it is a problem:** Acceptable for `docker compose` on a private network. Not
  acceptable for the Cloud SQL deployment the `terraform/modules/cloud-sql` module provisions
  — credentials and every transition (including full observation tensors) cross the network
  in clear text. The failure mode is bad: it works, so nobody notices.
- **Recommended change:** Add a `storage.postgres_tls` setting and wire
  `tokio-postgres-rustls` (or the Cloud SQL Auth Proxy, if that is the intended deployment —
  in which case document it in `documentation/DEPLOYMENT.md`, because right now nothing
  states the assumption).
- **Scope:** small–medium · **Risk:** low · **Validation:** connect to a TLS-required
  instance.

---

#### M10. Documentation contradicts the implementation in three places

- **Severity:** Medium · **Confidence:** Confirmed
- **Locations:**
  - `engine/engine-config/SCHEMA.md:90` gives `temp_threshold` default `15`;
    `config.defaults.toml:58` and `defaults.rs` test `:297` both say `0`.
  - `SCHEMA.md:8-13` lists the priority chain without `config.defaults.toml`, though it is
    the compile-time source of every Rust default.
  - `CLAUDE.md:349-351` says `config.toml` is "layered over the checked-in defaults in
    `config.defaults.toml`. Both are read by every component" — true for Python
    (`central_config.py:406-422` deep-merges the two files at runtime), but the Rust side
    embeds the defaults at *compile time* (`defaults.rs:10`) and reads only `config.toml` at
    runtime. Same result, materially different mechanism: editing `config.defaults.toml`
    without rebuilding changes Python behavior and not Rust behavior.
- **Why it is a problem:** These are precisely the documents someone consults when a config
  key does not take effect — the situation M4 makes likely.
- **Recommended change:** Correct the `temp_threshold` row, document the compile-time
  embedding explicitly, and add a `defaults.rs` test asserting each `SCHEMA.md` default
  matches the parsed TOML value (the table is small enough to check mechanically).
- **Scope:** small · **Risk:** none.

---

### LOW

- **L1. Low-value tests inflate the suite.** `web/src/handlers/game.rs:198-457` — ~260 lines
  constructing response structs and asserting the fields just assigned. The file's own
  comment (`:195-197`) admits they "overlap with the suites in types/requests.rs and
  types/responses.rs". They exercise no handler logic — `new_game`, `make_move`, and the
  game-switch rejection at `:101-112` have no direct tests here. Replace with handler-level
  tests via the existing `create_test_state` harness.
- **L2. `evaluator.py:311` hardcodes `choices=["tictactoe", "connect4", "generals_8x8"]`.**
  Othello is registered in the engine and present in the manifest but rejected by the CLI.
  Derive the list from `game_config.list_games()`.
- **L3. Broad exception swallowing.** `evaluator.py:55-56` (`except Exception` around DB
  access, logged as a warning), `trainer.py:107-108` (scheduler-state restore failure
  warned and ignored — training silently resumes with a reset LR schedule).
- **L4. `evaluate_batch` trusts the model's declared shape.** `mcts/src/onnx.rs:362-366,
  392-395` derives `action_size` from the tensor shape and slices `policy_flat` without
  bounds-checking against `batch_size * action_size`; a malformed model panics rather than
  returning `EvaluatorError`. An explicit length check would keep the error typed.
- **L5. `EngineContext::new` returns `Option`,** discarding why creation failed; callers
  reconstruct a message (`web/src/game.rs:91-92`, `actor/src/actor.rs:238-239`,
  `game_config.rs:38-44`) three different ways.
- **L6. `getGameInfo` interpolates `envId` into a URL unencoded** (`api.ts:175`). Harmless
  today (ids come from `/games`), free to fix with `encodeURIComponent`.
- **L7. `main.rs:76-85` falls back to tictactoe's `obs_size`** when the configured game is
  unregistered, then `expect`s tictactoe exists. Starting with an obs size that belongs to a
  different game guarantees every inference fails; failing at startup would be clearer.
- **L8. Nine `#[allow(dead_code)]` annotations** across the Rust tree, including
  `HealthState::set_unhealthy` (see H1). Each is either a missing caller or a deletion.

---

## 4. Large-file and responsibility analysis

| File or component | ~Size | Responsibilities currently present | Cohesive? | Recommended action |
|---|---:|---|---|---|
| `actor/src/actor.rs` | 943 (≈750 prod) | episode loop, timeout policy, MCTS invocation, transition construction, outcome backfill, Prometheus recording, stats, progress bar, shutdown | **Mostly** — one job, done end-to-end | **Keep.** Extract only `finalize_episode`'s backfill into a pure, testable function (see H7); the rest is one coherent narrative |
| `engine/games-othello/src/lib.rs` | 634 | rules, encoding, metadata for one game | Yes | Keep |
| `actor/src/mcts_policy.rs` | 607 (≈270 prod) | MCTS config, temperature schedule, random fallback | Yes | Keep |
| `engine/mcts/src/search.rs` | 497 | selection, expansion, batching, Dirichlet, result extraction | Yes — one algorithm | Keep |
| `engine/mcts/src/onnx.rs` | 488 | session lifecycle, tensor marshalling, masked softmax, timing counters | Borderline: inference + self-instrumentation | Keep; the diagnostics earned their place given the perf work |
| `actor/src/storage/postgres.rs` | 469 | SQL splitting, batch-insert SQL generation, pool config, `ReplayStore` impl | Yes | Keep; **extract `split_sql_statements` to a shared place** so Python cannot re-diverge (C1) |
| `web/src/handlers/game.rs` | 457 (**197 prod / 260 test**) | 5 handlers + struct-construction tests | Prod yes; tests no | Delete/replace the test module (L1) |
| `engine/engine-config/src/structs.rs` | 444 | 30 `d_*` fns + 9 structs + 9 `Default` impls | Repetitive but single-purpose | Keep. The triplication (serde default fn, field attr, `Default` impl) is boilerplate a macro could remove — **not worth it**; it is greppable and obvious |
| `web/src/game.rs` | 414 | session state, **flat state parsing**, legal moves, bot move + MCTS, response formatting | **No** | `parse_state` (`:137-153`) hardcodes `[board][player][winner]` and is the sole reason generals is unplayable. Move state decoding behind the engine (a `decode_display_state` on `ErasedGame`) so the web server stops knowing byte layouts |
| `trainer/.../storage/base.py` | 423 | `ReplayBufferBase` + `ModelStore` ABCs + `Transition`/`GameMetadata`/`ModelInfo` + checkpoint-naming helpers + tensor conversion | **No** — two unrelated abstractions plus filename conventions | Split into `replay.py` (buffer ABC + `Transition` + `sample_batch_tensors`) and `models.py` (`ModelStore` + `ModelInfo` + checkpoint naming). They change for different reasons: replay evolves with the training data schema, model storage with the deployment target |
| `trainer/.../games/generals.py` | 433 | full Python reimplementation of the Rust game | Internally yes | Keep the file; add the conformance test (H5). Do not attempt to unify the languages |
| `trainer/.../central_config.py` | 438 | 9 dataclasses + file discovery + merge + env overrides + type coercion + cache | Borderline | Keep. Splitting a config loader across files makes it harder to read, not easier |
| `web/src/main_tests.rs` | 555 | integration tests | Yes | Keep — this is the file carrying real handler coverage |

**Proposed split for `storage/base.py`** (the only structural split worth doing):

```
trainer/src/trainer/storage/
  replay.py    ReplayBufferBase, Transition, GameMetadata, sample_batch_tensors
  models.py    ModelStore, ModelInfo, checkpoint_filename/parse_checkpoint_step,
               encode/decode_best_model_metadata
  __init__.py  re-export both so existing imports keep working
```

Boundary: `replay.py` owns "training data in and out"; `models.py` owns "artifacts on
disk/S3". They share nothing today except the file they live in.

### Large files that are cohesive and should stay

`actor/src/actor.rs`, `engine/mcts/src/search.rs`, `engine/games-othello/src/lib.rs`, and
`engine/engine-config/src/structs.rs` are all large and all fine. Each has one reason to
change. Splitting them would scatter a single narrative across files and add navigation
cost for no isolation benefit.

---

## 5. Ownership analysis

**1. Game observation semantics**
- *Lacks a clear owner in practice.* The engine owns dimensions (via the manifest); the
  Python mirrors independently own encoding.
- *Interacting components:* `engine/games-*`, `engine-games` manifest generator,
  `trainer/games/*.py`, `policies/onnx.py`, `evaluator.py`, `web/src/game.rs::parse_state`.
- *Should own it:* the engine, completely.
- *Others should know:* dimensions and a `to_observation` result. They should **not** know
  plane ordering, mask position, or player-indicator placement.
- *How information should cross:* generated golden fixtures (H5) for Python; a
  `decode_display_state`-style engine method for the web server, replacing `parse_state`.

**2. Replay-buffer schema**
- *Lacks a single applier.* `sql/schema.sql` is the artifact; two independent splitters
  apply it, one broken (C1).
- *Interacting components:* `actor/src/storage/postgres.rs`,
  `trainer/storage/postgres.py`, `scripts/init-postgres.sql`, k8s init.
- *Should own it:* one splitter, shared. Simplest correct answer: make the Rust actor the
  only component that runs DDL, and have the Python trainer *verify* the schema exists
  rather than create it — which matches reality, since the actor always starts first in
  every documented workflow.
- *How information should cross:* the trainer raises a clear "schema not initialized; start
  the actor or run `make setup-db`" error instead of silently doing nothing.

**3. "Is this actor healthy?"**
- *No owner.* `HealthState` holds the flag; nothing sets it.
- *Interacting components:* `actor/src/actor.rs` (the only place that observes failures),
  `actor/src/health.rs` (the only place that reports them), k8s probes.
- *Should own it:* the main loop in `actor.rs`, which is the only component that knows an
  episode failed. `health.rs` should own only *reporting*.
- *How information should cross:* the loop calls `set_unhealthy()` past a consecutive-failure
  threshold; `health.rs` keeps its current pure-reporting role.

**4. Configuration defaults**
- *Split ownership with divergent mechanisms.* `config.defaults.toml` is the artifact; Rust
  embeds it at compile time, Python merges it at runtime, and the Rust env-override list is
  hand-maintained and incomplete.
- *Should own it:* `config.defaults.toml` remains the artifact — that part is right. What is
  missing is enforcement.
- *How information should cross:* a Rust test asserting every `CentralConfig` field has an
  env override (M4), and a test asserting `SCHEMA.md`'s documented defaults match the parsed
  TOML (M10).

**5. Game outcome (who won)**
- *Duplicated and derived incorrectly.* The engine knows the winner (it is a byte in the
  state); the actor re-derives it from a reward sign that does not carry seat information
  (C3), and the trainer re-derives value targets from step parity (H7).
- *Should own it:* the engine, reported explicitly at episode end.
- *How information should cross:* `StepResult` gains a terminal-outcome field, or the actor
  decodes the winner from the final state. `ActorStats` and `metrics` take an enum, not an
  `f32`.

---

## 6. Coding standards report

### Correctness-related inconsistencies (fix these)

1. **Missing-value handling has no house style.** Same struct, opposite defaults for
   truncated input: `extract_legal_mask` → all legal, `is_action_legal` → not legal (M1).
   Config parse failure → Rust substitutes defaults, Python raises (C2). Stats parse failure
   → zeroed struct (M3). DB metadata failure → silent fallback to a different source (H6).
   **Rule worth adopting repository-wide:** a malformed input is an error; only a *missing*
   input may take a default, and the default must be logged at `warn` with the reason.
2. **Duplicated logic across the language boundary, unpinned by any test.** SQL splitting
   (C1), game rules and observation encoding (H5), checkpoint filename conventions
   (`storage/base.py:23-34` vs `trainer/checkpoint.py`), metrics naming
   (`actor/src/metrics.rs`, `web/src/metrics.rs`, `trainer/metrics.py`). **Rule:** any
   concept implemented in both Rust and Python needs either a shared generated artifact or a
   conformance test — the manifest already demonstrates the pattern.
3. **Outcome/reward sign conventions are asserted in comments, not types** (C3, H7). The
   `f32` reward is doing three jobs: step reward, episode outcome, and seat attribution.
   **Rule:** where a value's meaning depends on perspective, encode the perspective in the
   type.

### Maintainability-related inconsistencies

4. **Three env-var namespaces** (`CARTRIDGE_*`, `ACTOR_*`, `ALPHAZERO_*`) with three
   different resolution strategies (M4).
5. **Two error-handling idioms in the Rust binaries:** `anyhow` with string context in the
   actor, `(StatusCode, String)` tuples in web handlers. Both are reasonable; the
   inconsistency is that `internal_error` (`handlers/game.rs:21-26`) formats the raw
   internal error into the HTTP body, so engine and lock-poisoning messages reach the
   browser. Log the detail, return a generic message.
6. **Logging style is split** between structured fields
   (`info!(component = "web", event = "...", ...)`) and format strings
   (`info!("Loaded game config for {}: ...", ...)`), sometimes in the same file
   (`actor.rs:232` vs `:361-367`). Structured is the better default given the JSON logging
   support in `engine_config::init_tracing`; worth converging on it in new code rather than
   through a sweep.
7. **Test-module placement varies:** inline `#[cfg(test)] mod tests`, `#[path = "x_tests.rs"]`,
   and `tests/` directories all appear. Harmless, but it makes "where are the tests for X"
   a lookup rather than a rule.

### Cosmetic

8. `ActorStats` names three different types across the codebase (`actor/src/stats.rs`,
   `web/src/types/responses.rs`, `api.ts`) with drifting field sets (M7).
9. Section-banner comments (`// ====== Unit Tests ======`) are used inconsistently.
10. `snapshot()` / `to_response()` / `summary()` all mean "serialize for output" in
    different modules.

---

## 7. What should not be changed

- **The `Game` → `GameAdapter` → `ErasedGame` erasure.** It gives static typing inside each
  game and dynamic dispatch at the registry, with byte-slice encoding as the boundary. It
  is the right shape for a plugin-style game set and it has stayed simple across four games
  with wildly different action spaces (9 → 257). Do not add a trait object layer or a
  plugin system on top.

- **`trainer/src/trainer/game_config.py`.** The explicit `_ENGINE_FACT_FIELDS` whitelist
  with its comment explaining why an intersection would be wrong (`:119-133`), and the
  hard failure when a manifest game has no `_TRAINING_OVERRIDES` entry with its explanation
  of the silent regression that would otherwise occur (`:176-185`) — this is the best code
  in the repository. It separates two things that change for different reasons and says so.

- **`replay_setup.check_metadata_agrees`.** The docstring (`:43-76`) explicitly enumerates
  what the check *cannot* catch and why closing those gaps needs a per-transition schema id
  that does not exist yet. Honest documentation of a partial guarantee is worth more than a
  more complete-looking check that quietly over-promises.

- **`effective_episode_timeout_secs` and its coupling to the liveness window**
  (`actor.rs:105-127`, `health.rs:48-61`). Deriving both the episode budget and the health
  window from one function, with a regression test
  (`actor.rs:808-834`) naming the exact restart-loop bug it prevents, is exactly right.

- **`LegalMask` as a dynamic-width type.** The commit that replaced `u64` masks solved the
  root cause rather than special-casing othello. Keep the direction; finish it by deleting
  the u64 remnants (M1).

- **The tau=1 policy-target invariant** (`search.rs:190-213`, restated in `CLAUDE.md` and
  `mcts_policy.rs:19-23`). Separating "the training target" from "the action we play" is
  subtle, easy to get wrong, and correctly documented in three places. Leave it.

- **Extraction of orchestration to `crucible`.** The composition root
  (`orchestrator/orchestrator.py`) documents every factory binding and why a wrapper class
  was chosen over `functools.partial`. The shim modules keep existing imports working. This
  is a well-executed extraction; do not second-guess it.

- **Large cohesive files** (`actor.rs`, `search.rs`, `structs.rs`, game `lib.rs` files) —
  see §4. Do not split on line count.

---

## 8. Remediation roadmap

### Phase 1 — Immediate correctness and safety

**Goal:** eliminate every silent failure identified as Critical/High. All of these are small,
independent, and high-confidence.

| # | Change | Finding | Depends on |
|---|---|---|---|
| 1 | Port the Rust SQL splitter to Python; add a splitter unit test | C1 | — |
| 2 | Make a parse failure of a *found* config file fatal in Rust | C2 | — |
| 3 | Replace the `f32` outcome with an explicit `Outcome` enum in actor stats + metrics | C3 | — |
| 4 | Liveness: track start time; call `set_unhealthy()` after N consecutive episode failures | H1 | — |
| 5 | Commit `Cargo.lock` ×3; add `--locked` to CI; make `cargo audit` + gitleaks gating | H2 | — |
| 6 | Assert `alternating_turns` in `finalize_episode` (metadata field + per-game declaration) | H7 | — |

**Changed together:** #3 and #6 both touch `actor.rs`'s episode-completion path and its
tests — do them in one PR. Note that #3 invalidates existing assertions in `stats.rs`
tests, which encode the current wrong behavior.
**Kept separate:** #1 (Python) and #2 (Rust) touch unrelated config/DB code — separate PRs
keep bisecting clean.
**Risks:** #2 will surface any latent config typos as hard failures — run it against the
current `config.toml` first. #5 will surface a backlog of advisories; triage, do not
suppress.
**Validation:** `make test`; a fresh-database `python -m trainer train` smoke run for #1; a
scripted player-2 win for #3.

### Phase 2 — Clarify ownership and boundaries

**Goal:** one owner per fact, enforced by a test rather than by documentation.

| # | Change | Finding | Depends on |
|---|---|---|---|
| 7 | Cross-language observation conformance test (Rust golden generator + pytest) | H5 | — |
| 8 | Delete `get_game_metadata_or_config`; evaluator uses the manifest unconditionally | H6 | 7 (so a regression is caught) |
| 9 | Trainer verifies the schema instead of creating it; the actor owns DDL | C1 follow-up | 1 |
| 10 | Fix `default_temp_threshold`; add the env-override coverage test | M4 | — |
| 11 | Move flat-state decoding out of `web/src/game.rs::parse_state` into the engine | §4 | 7 |

**Order:** 7 first — it is the safety net for 8 and 11.
**Changed together:** 8 and 11 both remove a component's private knowledge of observation
layout; sequencing them behind 7 means any mistake shows up as a test failure.
**Kept separate:** 9 and 10 are unrelated to the observation work.
**Risks:** 7's first run will likely find real discrepancies — budget time to fix them, and
treat each as its own finding. 11 changes an engine trait, so all four games must be
updated together.

### Phase 3 — Structural cleanup

**Goal:** remove duplication and dead API surface now that ownership is settled.

| # | Change | Finding |
|---|---|---|
| 12 | Delete `legal_mask_bits` / `extract_legal_mask` and their tests | M1 |
| 13 | Split `storage/base.py` into `replay.py` + `models.py`, re-exported | §4 |
| 14 | Replace `handlers/game.rs`'s struct-construction tests with handler tests | L1 |
| 15 | Add `env_id` to `cleanup()`; rewrite the anti-join; index `(env_id, created_at)` | M6 |
| 16 | Remove the `COUNT(*)` from `sample()`; stop nesting pool checkouts | H4 |
| 17 | Serialize model-watcher reloads behind one guard | M2 |
| 18 | Fix documentation drift; add the SCHEMA.md-vs-TOML consistency test | M10 |

**Order:** 16 before 15 (both touch `postgres.py`; the sampler is hotter).
**Kept separate:** 13 is a pure Python move; 12 and 17 are pure Rust. No coupling.
**Risks:** 15 changes a DELETE — test the "fewer rows than the window" boundary explicitly,
where the `OFFSET` subquery returns NULL.

### Phase 4 — Testability and operational improvements

**Goal:** close the gap between what CI runs and what production does.

| # | Change | Finding |
|---|---|---|
| 19 | Replace CI auto-commit with `--check` verification | H8 |
| 20 | Add a PostgreSQL service to CI; un-`#[ignore]` the four DB tests | §7 gap |
| 21 | Resolve the web session model: session map with eviction, or documented + 409 | H3 |
| 22 | Surface `episodes_abandoned` / `transitions_discarded` in the UI | M7 |
| 23 | `503` (not zeroed stats) on a stats parse failure | M3 |
| 24 | Configurable Postgres TLS, or document the Auth Proxy assumption | M9 |
| 25 | Fix the `GAMES_ACTIVE` leak | M8 |
| 26 | Quantify the MCTS replay-vs-expansion split before changing anything | M5 |

**Order:** 19 and 20 first — they change what CI can catch, which makes everything after
safer.
**Changed together:** 21, 22, 23, 25 all touch the web/frontend pair; batching them means
one round of `npm run check`.
**Kept separate:** 26 is a measurement task, not a change. Do not begin MCTS restructuring
until the benchmark says which term dominates.
**Risks:** 20 will lengthen CI and may expose genuine storage-layer bugs currently hidden by
the `#[ignore]`s — that is the point. 21's session map must be bounded or it becomes a new
unbounded collection.

---

## 9. Top ten recommended actions

Ordered by impact relative to effort.

1. **Fix `_ensure_schema()`'s statement splitter** (C1) — five lines; removes a
   confirmed silent no-op and a divergent duplicate of correct Rust code.
2. **Make a malformed `config.toml` fatal in Rust** (C2) — small; prevents training the
   wrong game for hours behind a single `warn!`.
3. **Replace the `f32` outcome with an explicit enum** (C3) — small; restores the seat-bias
   diagnostic the codebase itself identifies as a training-collapse detector.
4. **Commit `Cargo.lock` for all three workspaces and make security audits gate** (H2) —
   trivial; the precondition for reproducing and bisecting anything else.
5. **Fix actor liveness so a never-succeeding actor fails its probe** (H1) — small; turns a
   silently idle pod into a restarting one.
6. **Add the cross-language observation conformance test** (H5) — medium; closes the largest
   remaining silent-drift risk and unblocks §8 Phase 2.
7. **Assert the alternating-turns invariant in `finalize_episode`** (H7) — small; converts a
   future half-inverted-value-targets disaster into a startup assertion.
8. **Replace CI's auto-commit formatting with `--check`, and add a Postgres service** (H8,
   §7) — small; makes CI a gate, unblocks fork contributions, and starts running the four
   storage tests that have never executed.
9. **Delete `get_game_metadata_or_config` so evaluation stops depending on DB reachability**
   (H6) — small; one owner for game facts, and removes a union return type from four call
   sites.
10. **Remove the per-step `COUNT(*)` from `sample()`** (H4) — small; restores the caching
    the trainer already attempted and removes a full scan from the hottest path.

Items 1–5 and 7 are each under an hour and independently mergeable. Item 6 is the one that
warrants real time, and it is the one that protects the training loop's correctness for the
next several years.
