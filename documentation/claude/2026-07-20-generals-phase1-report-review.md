# Review: `GENERALS_PORT_PHASE1.md` — weaknesses

**Reviewed document:** `documentation/GENERALS_PORT_PHASE1.md` (STATUS header dated
2026-07-19, "IMPLEMENTED, with revisions")
**Method:** every checkable claim in the report was verified against the tree at
`f416849` (current `main`). Citations below are file:line in that tree.

## Summary verdict

The report is two documents fused together: a 2026-07-03 plan whose body was never
updated, and a 2026-07-19 STATUS header that corrects *some* of the body's
obsolete decisions. The STATUS header is the stronger half — its five revision
bullets are mostly accurate — but it contains one outright factual error about the
implemented ruleset (the "500-round draw clock"), several claims that were already
stale on the day it says it was written (TODOs that are done, a test count that is
low), and it omits the two most consequential design changes actually shipped
(territory adjudication and the parity-randomized ply cap). Below the header, the
plan body contradicts both the STATUS and the code on the turn bridge, fog, obs
layout, state encoding, and terminal rule, with no markers saying which parts are
superseded. Finally, the report's own verification story is unresolved: every
checklist box is unchecked, and the differential test against the Go engine — the
one check that would validate a *port* — appears never to have been done, dropped
silently.

---

## 1. Factual errors and stale claims in the STATUS header

The STATUS header is the part a reader trusts as current. It has four problems.

### 1.1 The terminal rule is misdescribed — "500-round draw clock" is wrong twice

STATUS bullet 2 says the implemented ruleset ticks "production and the **500-round
draw clock** … after player 2's ply."

- The cap is **200** rounds, not 500: `engine/games-generals/src/params.rs:45`
  (`MAX_TURNS: u32 = 200`).
- It is **not a draw clock**. At the cap the game is *adjudicated by territory*:
  more tiles wins, total armies break ties, and only a perfectly even position is
  a draw (`engine/games-generals/src/rules.rs:84` `adjudicate_at_cap`, applied at
  `engine/games-generals/src/lib.rs:213-218`).

This is not a nitpick: the crate docs (`lib.rs:27-31`) and `params.rs:41-44`
record that the original 500-round pure-draw cap **collapsed self-play into 100%
draws (zero value signal)** and was deliberately replaced. The report's "revised
after review — the deltas that matter" list omits the single largest ruleset
delta actually shipped, and states the abandoned design as the implemented one.

### 1.2 The parity-randomized ply cap is absent

`State.cap_plies` is coin-flipped at reset between `2*MAX_TURNS` and one less, so
each seat gets the final pre-adjudication move in half of all games, and the cap
is deliberately *excluded from the observation* because a fixed even cap made the
value net degenerate into a seat detector (`lib.rs:80-86`, `lib.rs:176`). This is
a subtle, load-bearing training-soundness decision — exactly the kind of thing a
status report exists to record (CLAUDE.md even name-checks "parity-randomized ply
cap"). The report never mentions it.

### 1.3 TODOs that were already done

STATUS bullet 5: "A generals `GAME_CONFIGS` entry (input_channels=9) is still
TODO, along with the pure-Python evaluator game and a web renderer."

Two of the three are done in the same tree that contains the report:

- `GAME_CONFIGS["generals_8x8"]` exists with `input_channels=9`,
  `network_type="resnet"`, 6 blocks / 128 filters, `player_relative_obs=True`
  (`trainer/src/trainer/game_config.py:143-159`).
- The pure-Python evaluator game exists — `trainer/src/trainer/games/generals.py`
  — and is wired into the evaluation path (`trainer/src/trainer/games/__init__.py:97`;
  `generals_8x8` is an accepted `--env-id` choice in
  `trainer/src/trainer/evaluator.py:311`).

Only the web renderer is genuinely still TODO (no generals support anywhere in
`web/frontend/src`). A reader planning next steps from this report would
re-implement two finished components.

### 1.4 Stale test count

STATUS preamble: "passes 25 tests plus an MCTS integration test." Actual:
**29** `#[test]` functions in `engine/games-generals/src/tests.rs`, plus the MCTS
regression test `test_mcts_generals_257_actions`
(`engine/mcts/src/search_tests.rs:220`). CLAUDE.md carries the correct 29. Minor,
but it signals the header was not re-checked against the tree it describes.

### 1.5 Overstated: "replaced every u64 legal mask"

STATUS bullet 1 says `LegalMask` "replaced every u64 legal mask." The new
dynamic-width mask exists and is what MCTS/actor/web use
(`engine/engine-core/src/legal_mask.rs:14`; `engine/mcts/src/search.rs:426`;
`actor/src/mcts_policy.rs:7`; `web/src/game.rs:208`), but legacy u64 mask helpers
remain live in the tree:

- `GameMetadata::legal_mask_bits` — still documented as overflowing at 65 actions
  (`engine/engine-core/src/metadata.rs:126-134`) — and the u64-returning
  `GameMetadata::extract_legal_mask` beneath it;
- `info_bits::extract_legal_mask` (`engine/engine-core/src/game_utils.rs:186`);
- `actor/src/game_config.rs` tests exercising both.

(Open PR #140 deletes some of these.) "Every" is the kind of absolute a review
report should not use while the old paths still compile and are called by tests.

---

## 2. The superseded plan body was never reconciled

Everything below the `---` under the STATUS header is the 2026-07-03 plan, left
verbatim. The STATUS bullets correct five points, but the body is not annotated,
so the corrections and the corrected text sit in the same document with equal
typographic weight. Concretely, the body still asserts:

| Body claim | Reality |
|---|---|
| Turn bridge: `step(P1) -> stash in state.pending`, `State { pending: Option<Move>, … }` | Dropped (per STATUS bullet 2). `State` has no pending field — strictly alternating plies (`lib.rs:67-87`) |
| Scope table: fog "**Ported but disabled**"; crate layout lists `visibility.rs`; Tile has `visible: u32, discovered: u32`; `Params.fog_enabled`; test plan ports `fog_of_war_test.go` | None exist. No `visibility.rs` in `src/`; `Tile { owner, army, kind }` only (`board.rs:40-44`); no fog flag; no fog tests. STATUS bullet 3 says so — but the body's "locked" scope table still says the opposite |
| Obs channels: "visibility(all-1s), ownership, log-army, 4× tile-type one-hot, turn/max_turns, ch8 owned-mask" | Actual `generals_obs:v1`: own/enemy/neutral territory, own/enemy log-armies, cities, mountains, generals ±1, turn progress (`obs.rs:1-24`). Two mutually incompatible layouts in one document |
| Scope table: "Max turns 500, then draw (`winner=3`)" | 200 rounds, territory adjudication, draw only on exact tie (§1.1) |
| State encoding: "per tile `(owner u8, army u32, kind u8, visible u32, discovered u32)` = 14 B × 64 + header ≈ 910 B" | Actual: 12 B header + 6 B/tile × 64 = **396 B**, no visibility fields (`lib.rs:233-251`) |
| `step()` info_bits: "return 0 and rely on the in-obs mask" | Actual: packs player/winner/round into info_bits, omitting only the mask (`lib.rs:108-118`) |
| `Params` struct with `fog_enabled` | Plain consts, no struct, no fog flag; `GENERAL_START_ARMY = 2` (`params.rs:30`) appears nowhere in the report |

None of these carry a "superseded — see STATUS" marker. A reader who skims past
the header — or lands mid-document via search — gets a confidently wrong picture
of the state format, the obs schema, and the game rules. The one line most likely
to mislead future work is the scope table's claim that fog is "ported but
disabled," i.e. "flipping fog on later is a config change, not a rewrite": the
STATUS explicitly concludes the opposite (fog needs a new env id, obs schema, and
algorithm).

---

## 3. Verification gaps — the report doesn't establish what it claims

### 3.1 The differential test against the Go engine silently vanished

The plan lists it twice — in the test plan ("run 3–5 identical scripted move
sequences in the Go engine and the Rust port, diff board states turn by turn")
and in the work checklist. There is no trace of any such test in
`games-generals` (`tests.rs` has no Go/differential/scripted-sequence fixture),
and the STATUS neither claims it was done nor records that it was dropped. For a
document whose title is "Port the Generals Engine," this is the weakest point:
the one verification step that would validate port fidelity is unaccounted for.
(It also can't be waved off as moot — the report itself says combat/capture
mechanics were "port 1:1, this is the heart of the game.")

### 3.2 Every checklist box is still unchecked

Both the test plan and the work checklist sit at `- [ ]` under an "IMPLEMENTED"
banner. The reader cannot distinguish "done but not ticked" from "not done"
(§3.1 shows at least one item is genuinely the latter — which means the ticks
can't be assumed).

### 3.3 The ~13% benchmark regression is unverifiable

STATUS bullet 1 cites a "~13%" search microbenchmark regression. The benchmark
harness exists (`engine/mcts/benches/mcts.rs`) but no baseline numbers, bench
output, or measurement conditions are recorded anywhere in the repo. As written
it's an unreproducible number — fine as a hallway remark, weak as the recorded
justification for accepting a performance regression.

### 3.4 No honest outcome statement

The report ends at "implemented, passes tests." CLAUDE.md states the
operative caveat: generals **training does not yet beat random at local compute
scale**. A status report on Phase 1 of a training port should carry that outcome
(or point at `generals_strength_probe`, the crate's own honest strength measure)
— "implemented" without "and here is where training actually stands" overstates
completion.

---

## 4. Smaller omissions

- **Illegal actions auto-convert to `wait`** (`lib.rs:18-20`) — a rules-level
  behavior difference from strict rejection; affects anyone writing a client or
  replaying transitions. Not in the report.
- **`max_horizon = 2*MAX_TURNS + 2`** (`lib.rs:143`) and the reset-time coin flip
  belong with the cap discussion (§1.2).
- The report never states which Go commit/version was ported ("Go 1.24" is the
  toolchain, not a revision) — with the source repo evolving (the report itself
  references a "2026-07-02 max_turns fix"), the port baseline is unpinned.

---

## 5. What the report gets right (verified)

For balance — these claims all check out:

- `generals_8x8` registration, 257 actions (256 moves + wait at index 256),
  direction order up/right/down/left, index `(y*W+x)*4 + dir`
  (`action.rs:8-9`, `params.rs:16-21`).
- STATUS bullet 4's obs layout matches the code exactly: 9×64 spatial channels,
  mask at offset 576, player one-hot, `obs_size = 835`, player-relative,
  versioned `generals_obs:v1` (`obs.rs:1-39`).
- STATUS bullet 2's *reasoning* for dropping the pending-action bridge (leaks the
  first mover's choice into the searcher's true state) matches the crate docs
  (`lib.rs:10-14`).
- STATUS bullet 3's fog conclusion (needs new env/schema/algorithm) matches
  `obs.rs:22-24`.
- `LegalMask` is a dynamic bitset, MCTS actions are `u32`, and MCTS/actor/web all
  read masks from the obs at `legal_mask_offset` (§1.5 citations); the described
  Othello `info_bits` collision is corroborated by `legal_mask.rs:1-10`.
- The `replay_setup.py` fix (network config on the DB-metadata path) is in place
  (`trainer/src/trainer/replay_setup.py:68-83`), and the trainer ResNet is
  channel-generic via `input_channels`.
- The two MCTS diagnostic examples exist
  (`engine/mcts/examples/generals_{policy,strength}_probe.rs`).

---

## 6. Recommendations

1. **Fix the STATUS header's ruleset sentence**: 200-round cap, territory
   adjudication (tiles → armies → draw), and add the parity-randomized `cap_plies`
   decision with its rationale. These are the two design facts future phases will
   need most.
2. **Mark the body as historical** — either strike superseded sections
   (`~~…~~` or `<details>`), or add one-line "superseded by STATUS #N" notes at
   the turn-bridge, fog, obs-layout, encoding, and scope-table locations.
3. **Resolve the differential-test question explicitly**: either do it (even 3
   scripted games) or record why it was dropped and what replaced it as the
   port-fidelity check.
4. **Tick or prune the checklists**, and refresh test counts / TODO list
   (GAME_CONFIGS and the Python evaluator game are done; web renderer is not).
5. **Add the outcome line**: training vs. random status and a pointer to
   `generals_strength_probe` as the honest measure.
6. Pin the ported Go revision (commit hash) in the companion-repos section.
