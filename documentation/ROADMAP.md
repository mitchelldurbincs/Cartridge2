# Cartridge2 — Audit & Roadmap

**Date:** 2026-08-02
**Scope:** whole repository at `d89220a`, audited against the project's stated goal:
a state-of-the-art RL platform with good metrics, playable bots, support for
many games and algorithms, self-play tournaments, and infrastructure that runs
locally or in the cloud with ease — all well-written, easy to understand, and
beautiful.

This document complements [`ENGINEERING_REVIEW.md`](ENGINEERING_REVIEW.md)
(2026-08-01, defect-focused). Where that review asks "is the code correct?",
this one asks "how far is the platform from its goal, and in what order should
the gap be closed?" Findings already tracked there are referenced, not
restated.

---

## 1. Where we are — scorecard against the vision

| Vision pillar | Reality today | Grade |
|---|---|---|
| Well-written, easy to understand | Genuinely strong. Zero `unsafe`, zero TODO/FIXME anywhere, ~700 tests, clippy `-D warnings`, and an unusual culture of comments that explain *why* and name the failure mode they prevent. Docs are honest about their own gaps. | **A−** |
| Good metrics | Instrumentation is broad but the consumption side is broken: no Grafana/alerting anywhere, the Prometheus actor target has never returned data, five registered metrics are never emitted, the web `/stats` layer silently drops fields the frontend renders, and solver-eval results are never surfaced. | **C** |
| Play against your bots | Works for 3 of 4 games, single-player-demo grade: one global session per server, sync MCTS under the session mutex, no model/checkpoint picker, `u8` action cap, Generals blocked by `parse_state`. | **C+** |
| Any game | Really "any 2-player alternating board game": adding one is a well-paved ~400-line path with a golden manifest test, but MCTS backprop hardcodes 2-player zero-sum, the obs profile requires board-shaped games, and the web UI knows two board shapes. | **B−** |
| Any algorithm | AlphaZero only. The `Evaluator` trait is a real seam; nothing else is. Trainer hard-binds Adam + AlphaZeroLoss; the replay sampling contract is AlphaZero-shaped. | **D** |
| Tournaments / self-play eval | Absent. No Elo, no opponent pool, no model-vs-model beyond candidate-vs-current-best. The promotion gate (50 games, 0.55, draws-count-as-losses) is statistically underpowered and can structurally freeze. | **D** |
| Local infra | Docker Compose story is mature; Dockerfiles and CI are thoughtful. (`make train` is broken on a fresh clone — missing `CARTRIDGE_STORAGE_POSTGRES_URL`.) | **B+** |
| Cloud infra | Aspirational fiction. K8s has provably never run (trainer pod passes a nonexistent flag; actor watches the wrong model dir). Terraform targets GCP while the manifests assume AWS. The S3/MinIO model backend is dead code in both languages. Nothing in CI validates any of it. | **F** |

**Overall:** a very well-crafted single-algorithm, two-player, four-game local
platform — with a metrics/evaluation layer that cannot yet be trusted, and a
cloud layer that is documentation for a system that does not exist.

## 2. New findings from this audit

Highest-impact first. (Items already in `ENGINEERING_REVIEW.md` carry their ID.)

### Critical

1. **Connect4's Python mirror feeds the network a transposed board.**
   The Rust engine indexes the board row-major (`row * COLS + col`,
   `engine/games-connect4/src/lib.rs:126-128`); the Python mirror is
   column-major (`board[col * HEIGHT + row]`,
   `trainer/src/trainer/games/connect4.py:19,49`) and copies that flat array
   straight into the observation with no remapping (`connect4.py:143-147`).
   Every Python-side Connect4 evaluation — `trainer evaluate`, the in-loop
   eval, the **promotion gatekeeper**, and **all solver-eval metrics** —
   scores a model fed observations it was never trained on. This is exactly
   the failure ENGINEERING_REVIEW's H5 predicted. The Python test suite is
   green because `tests/test_games.py:16-18` asserts against a helper that
   bakes in the same wrong convention. TicTacToe and Generals mirrors were
   cross-checked and are faithful.

### High

2. **The web `/stats` endpoint silently drops fields the frontend renders.**
   `handlers/stats.rs` deserializes `stats.json` into Rust structs and
   re-serializes; any field missing from `web/src/types/responses.rs` is
   dropped. `grad_norm`, `became_new_best`, `current_iteration`,
   `opponent_iteration` are all written by the trainer and rendered by
   `Stats.svelte` — and never arrive. The grad-norm tile always shows `-`,
   the "New Best" badge never fires. Nothing tests the wire format.
3. **The promotion gate is statistically underpowered and draw-hostile.**
   50 games at a 0.55 threshold is ~0.7σ; `vs_best_win_rate` counts draws as
   losses (`evaluator.py:87-91`), so in drawish games promotion can become
   structurally unreachable and the loop silently stops improving. First eval
   promotes unconditionally and records `win_rate = 1.0` into history.
4. **MCTS replays the full action path from the root for every simulation**
   (`mcts/src/search.rs:295-315`) — no state caching on nodes, no tree reuse
   across moves, no parallel search. For Generals (~180 legal moves, deep
   trees) this is plausibly *the* reason training doesn't beat random at
   local compute scale. The actor additionally runs one episode at a time on
   one thread; scaling is by process count only.
5. **K8s has never worked** — four independent proofs: trainer pod passes
   `--stats-path` (real flag is `--stats`) and `--steps=0` means zero steps,
   not infinite; the actor watches `<PVC>/latest.onnx` while the trainer
   writes `<PVC>/models/latest.onnx`; the prod overlay's
   `configMapGenerator behavior: merge` targets a non-generated ConfigMap;
   Terraform's Workload Identity binds to ServiceAccounts no manifest
   creates. Terraform provisions GCP; the manifests assume AWS (`efs-sc`,
   `m5.xlarge`).
6. **The S3/MinIO model backend is dead code in all three languages.**
   `S3ModelWatcher` is never constructed (Rust), `create_model_store` is
   never called (Python), `storage.model_backend` is never branched on, and
   `boto3` isn't even a declared dependency. MinIO is a mandatory startup
   dependency in Compose for a service nothing writes to.
7. **The actor Prometheus scrape target has never returned data** —
   `prometheus.yml` scrapes `:9091`, but actor metrics are served on the
   health port (8081, and 8081-8090 under `num_actors`). No Grafana, no
   alert rules, no Prometheus in K8s at all.
8. **`schema.sql` resolution breaks in any installed deployment** —
   `postgres.py:22` walks five parents from `__file__`, which resolves to
   `<prefix>/lib/sql/schema.sql` in a wheel install; `Dockerfile.alphazero`
   copies it to `/usr/local/sql/`. Off by one directory; raises
   `FileNotFoundError`.

### Medium (selected)

- `legal_mask_bits` shifts `1u64 << num_actions` — panics/wraps for Othello
  (65) and Generals (257); reachable as the fallback of `extract_legal_mask`
  (`engine-core/src/metadata.rs:195,217`). Both are dead in production —
  delete rather than fix.
- Web API caps actions at 255 (`MoveRequest.position: u8`), and a >255 bot
  action silently degrades to a random move (`web/src/game.rs:256-269`).
- Metrics defects: `web_request_duration_seconds` never observed (no
  middleware), `web_model_loaded`/`web_model_reloads_total` never touched,
  `web_games_active` leaks monotonically, episode histograms bucket at
  100 steps / 10 s so every Generals episode lands in `+Inf`,
  `EPISODES_PER_SECOND` is `1/last_duration`, not a rate.
- Replay buffer: `sample()` runs `COUNT(*)` per training step and checks out
  a second pooled connection inside an open one; `ORDER BY RANDOM()`
  fallback dominates the post-clear regime; `cleanup()` ignores `env_id` and
  anti-joins on unindexed `created_at`.
- `trainer.py:277-279` labels the final checkpoint `start_step + total_steps`
  even when training stopped early — mislabeled artifacts.
- `make train` omits `CARTRIDGE_STORAGE_POSTGRES_URL` and fails on a fresh
  clone; `make lint` skips the frontend; benches never run in CI; frontend
  has zero tests; `run_episode` has no CI-runnable test (all gated on live
  Postgres).
- Othello exists in the manifest but not the Python mirrors, so it trains
  but cannot be evaluated at all.

## 3. The way forward

Ordering principle: **restore trust in the numbers first** (a metrics platform
whose metrics lie is worse than none), **then throughput** (every future
feature — tournaments, more games, harder games — is downstream of self-play
speed), **then the product features**, **then generality**, and cloud last —
after a single node is worth scaling.

### Phase 0 — Truth & trust (~1-2 weeks)

*Goal: every number the platform shows is real.*

- Fix the Connect4 mirror transposition, then build the **cross-language
  golden observation test** (ENGINEERING_REVIEW H5): a Rust generator in
  `engine-games/src/bin/` emits seeded action sequences with exact
  observation bytes per game; a pytest replays them through the Python
  mirrors and asserts byte equality. The manifest golden test proves this
  mechanism works. This permanently closes the mirror-drift class of bug.
- Fix `/stats`: either proxy the trainer's JSON verbatim or complete the
  Rust structs; add a wire-format test (trainer-written fixture →
  web handler → assert fields survive). Surface `solver_stats.json` while
  in there.
- Fix the promotion gate: draws count as half, raise eval games (or use a
  sequential test), stop recording the unconditional first promotion as
  `win_rate 1.0`.
- Fix the metrics plumbing: actor scrape port, request-latency middleware,
  wire `web_model_*`, fix `games_active` and the episode histogram buckets,
  make episodes/sec a real rate.
- **Ship one Grafana dashboard + a handful of alert rules** into Compose
  (training progress, self-play throughput, abandonment rate, buffer size,
  eval win-rate/Elo panel placeholder). Cheap: the metric names are already
  good.
- Quick sweeps: `make train` env var, delete `legal_mask_bits` /
  `extract_legal_mask` / `LegalMask::from_u64` dead code, fix the final
  checkpoint step label, orphaned `smoke_test.py`, dead frontend code.

### Phase 1 — Throughput (~2-4 weeks)

*Goal: order-of-magnitude more self-play per box; Generals becomes trainable.*

- **Cache state bytes on MCTS nodes** (kill the per-simulation path replay) —
  the single biggest lever; then tree reuse across moves.
- Parallelize self-play: N concurrent episodes per actor process feeding a
  shared batched evaluator (this is also what makes a GPU worth attaching).
  Consider root-parallel or leaf-parallel search only after episode-level
  parallelism saturates.
- Add Othello and Generals benches (the two large action spaces); **run
  benches in CI** with a stored baseline so perf regressions gate like
  correctness regressions.
- Replay-buffer hygiene: cache the row count, index `created_at`, scope
  `cleanup` by `env_id`, drop `next_observation` (redundant with the next
  row) — ~40% write-volume cut for Generals.
- Exit criterion: sims/sec and episodes/sec on the dashboard, and a Generals
  run that beats random. (Consider Gumbel root selection here or in Phase 4 —
  it is designed exactly for the 50-250-sim budgets Generals is stuck with.)

### Phase 2 — Evaluation as a product (~2-3 weeks)

*Goal: the "tournaments through self-play" pillar; strength you can trust.*

- Keep a **historical opponent pool** (every promoted model + every Nth
  checkpoint), not just `best.onnx`.
- **Elo over the pool**: after each iteration the candidate plays a gauntlet
  against pool members; ratings update and persist. The aspirational
  `model_versions` table in `scripts/init-postgres.sql` is the natural home —
  wire it for real (env_id, step, git SHA, config hash, rating, games).
- `trainer tournament` CLI: round-robin over selected checkpoints/pools,
  results to Postgres + W&B. This also exposes rock-paper-scissors cycling
  that candidate-vs-best cannot see.
- Promotion switches from raw win-rate to rating-based, with the solver
  anchor retained for Connect4.
- Surface it: Elo-over-time chart and solver-metrics panel in the web UI.

### Phase 3 — The web arena (~2-3 weeks)

*Goal: play any checkpoint; watch bots fight; Generals on screen.*

- Session refactor: `DashMap<SessionId, GameSession>` + TTL (the enabler for
  everything below); MCTS onto `spawn_blocking`; `u32` actions end-to-end.
- Model registry API: `GET /models` (scan checkpoints + registry table),
  `model` field on `NewGameRequest`, per-model evaluator cache with LRU.
- **Bot-vs-bot spectating**: no-human sessions, server steps the game, SSE
  (or a `POST /game/:id/step` the client drives). Combined with Phase 2 this
  gives "watch iteration 80 vs iteration 40".
- **Engine-side board decoding**: add a display-view method to `ErasedGame`
  so the web server stops guessing byte layouts — deletes `parse_state`,
  unblocks Generals, and makes "any game renders" true by construction. A
  tile-based renderer (armies, cities, mountains) becomes a frontend-only
  task.

### Phase 4 — Second algorithm (~3-5 weeks)

*Goal: "any algorithm" becomes true by demonstration, not by trait.*

- Do **not** design an abstract Algorithm interface first. Build the second
  concrete algorithm, then extract the seam from two real implementations.
- Recommended sequence: **Gumbel AlphaZero** first (small, slots into the
  existing MCTS, directly improves low-budget search = Generals), then
  something off-policy (DQN-family) — the transition schema already stores
  `reward`/`done`/`next_observation`, so the data path exists; the work is a
  second `Trainer` and a widened sampling contract
  (`ReplayBufferBase.sample_batch_tensors` is the AlphaZero-shaped
  bottleneck).
- Crucible already injects `trainer_factory`/runners via Protocols — the
  process-level seam is there; use it.

### Phase 5 — Cloud, honestly (~2-4 weeks, or 1 day)

*Goal: either the cloud path works, or it doesn't exist. Nothing in between.*

- **Decision point.** Option A (recommended for now): delete `k8s/` and
  `terraform/`, and make the cloud story "one big GPU VM + Docker Compose" —
  which is the truthful current capability and the right scale until Phase 1
  saturates a single node. Option B: make K8s real — fix the four breakages,
  pick **one** cloud (GCP, since Terraform is the stronger half), wire the
  model store for real (GCS/S3) or stay PVC, create the ServiceAccounts,
  and add `kustomize build` + `kubeconform` + `terraform validate` to CI so
  it can never silently rot again.
- Either way: delete the dead S3 backend or wire it — the current state
  (configured, documented, never executed) is the worst of both.
- Versioning basics whenever the first artifact leaves the laptop: git tags,
  a release workflow that pushes images, model artifacts stamped with git
  SHA + config hash.

### Cross-cutting, throughout

- **Contract tests at every language boundary** — the three worst bugs found
  (Connect4 transposition, `/stats` field drops, K8s flag names) are all
  cross-boundary drift that no single-language test could catch. Golden
  files and wire-format tests are the antidote; the manifest golden test is
  the house pattern — apply it everywhere.
- CI: run `run_episode` against a Postgres service container; add a vitest
  smoke test for the frontend; make `make lint` match CI.
- Keep the documentation-honesty culture (it is the repo's superpower); this
  document should gain `Status:` lines the way ENGINEERING_REVIEW.md did.

## 4. What not to do

- **No microservices, no gRPC** — the monolith-over-shared-storage bet is
  correct and is what makes the codebase understandable.
- **No speculative abstraction** for algorithms or non-board games before the
  second concrete case exists (`engine-games/src/lib.rs:88-98` already states
  this principle — hold to it).
- **No K8s investment before Phase 1** — scaling an actor that uses one core
  per box multiplies waste, not throughput.
- **No fog-of-war Generals** until an IS-MCTS/recurrent path exists; the
  engine docs are right that it is a different algorithm, not a flag.
- **Don't touch** the erasure boundary, the manifest pipeline, the actor's
  timeout/liveness coupling, or the tau=1 policy-target invariant — they are
  the best-engineered parts of the system.
