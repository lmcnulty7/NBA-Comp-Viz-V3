# P8 pair gate: the scorer's matching with the gate applied before the assignment (design, version 1, 2026-11-27)

Report only, registered before any number. A measured sensitivity for Lucien's option (d) ("work on the truths and
scorer first"): the frozen scorer stays the reported rule, and nothing here changes a reported row. No frozen file is
edited. hou_cle_2016 is never touched (no file of it read or hashed); gsw_sac_2015 stays apart; the allow-list
{gsw_cha_2016, gsw_bkn_2015} and the P8 guards run on every name.

## 1. Question

The scorer pairs box feet with SportVU players through sportvu.truth._match (sportvu/truth.py lines 59 to 63): Hungarian
on the full distance matrix, then the 5 ft gate (sportvu.bench.MATCH_FT) applied to the pairs it chose. A pair beyond
the gate can take a box away from a player it is within the gate of, so the gate after the assignment can drop pairs
that a gated assignment would keep. D214's diagnostic found that this changes about 1 percent of team rows. How much
would every D202 row move on gsw_cha_2016 and gsw_bkn_2015 if the gate were applied before the assignment?

## 2. What can move, by construction

tools/p8_scorecard.score_window uses the pairs for the matched players' position errors (position p50 and p90, the
near-field bias) and for the team rows. The missed row (nearest foot within 2.5 ft), the ghost rows (nearest player
within 3 ft), net error and the one-to-one rows (tools.p8_resetfix.unmatched_1to1, already gated, in score_pair) do
not use _match. Registered prediction and check: those rows change by exactly 0; if any does, the run stops as a code
fault.

## 3. The capped variant (fixed now)

The same Hungarian on the cost matrix C = min(D, MATCH_FT): every pair beyond the gate costs the gate, so how far beyond
the gate a pair lies no longer steers the assignment; the pairs kept are those with D at most MATCH_FT, as now. (It does
not maximise the number of within-gate pairs: two within-gate pairs near the gate can still lose to one close pair and
one beyond it; see section 7.) It is
computed in tools/ by wrapping: tools/p8_pairgate.py replaces the sportvu.truth._match attribute with the capped
function for the duration of the capped pass only (score_window imports _match from sportvu.truth at call time), and
restores it after; no frozen file is edited. A test checks that the wrapper restores the frozen function, that the
capped function equals the frozen one whenever no pair lies beyond the gate, and that it keeps a within-gate pair the
frozen one drops on a constructed case.

## 4. Gates and rows (fixed now)

- **G1, reproduction.** The uncapped path (the frozen _match) through tools/p8_resetfix.score_pair, D202's own scoring
  call (tools/p8_rebaseline.py score_set), reproduces D202's record (work_ref/rebaseline/score__<game>__current.json,
  run 20261127T171848Z) on every GATE_FIELDS field of every window of both games; else stop. The record's per-window
  hashes (sidecar, time map, labels, identity) and the scorecard pin are checked first, as D202's run did.
- **Rows.** Every D202 row (tools/p8_rebaseline.change_rows: missed, detection, placement, ghosts per box and per player,
  net error, one-to-one net error, coverage, position p50 and p90, near-field bias, team), capped minus uncapped, per
  game and pooled (cha plus bkn), with paired window-bootstrap intervals (2,000 resamples, seed 0, as change_rows
  does).
- **Counts beside them.** Pairs per game under each rule, pairs gained and lost, team rows gained and lost, and the
  share of frames where the two assignments differ.

## 5. Reading (fixed now)

A row is "scorer-sensitive" when the absolute value of its pooled change exceeds the half-width of its D202 pooled
interval (reports/research/p8_rebaseline.json, pooled_cha_bkn, current truth); else "insensitive". Per game the same
against D202's per-game interval. The table states each row's change, its interval, D202's half-width and the label.
Nothing is adopted: a scorer-sensitive row is reported for Lucien's option (d); the reported rule does not change.

## 6. Cost and writes

One lock job (frozen CV python, "p8 pairgate"): the two scoring passes over the 156 windows of the two games from
stored files (D202 took about 11 lock minutes for its scoring), so about 10 to 20 lock minutes, cap 30. 0 Colab units.
Writes: reports/research/p8_pairgate.json only (checked with find_private); manifests of the reference root's labels,
label_build, truth_accept, cache and builds/r23 watched before and after. hou_cle_2016 never touched.

## 7. Limits

The capped rule is one of several gated assignments (a 1e6 cost beyond the gate, as tools.p8_one2one.assign_1to1 uses,
maximises the count of within-gate pairs instead); only the capped rule is registered. The truth set and its limits
are D202's (truth frames selected with r23's boxes).

## 8. Version 1.1 (2026-11-27, local; the design review of 60f143d5, GO WITH CHANGES; before any code)

Where this section differs from sections 1 to 7, it rules. Dates in this item are local time.

**8.1 What is nearly decided already, stated first.** Team "insensitive" is nearly certain: D214 found at most 477 of
49,674 team rows change under a gate-first rule (0.0096; P8_teambound.md's capped diagnostic, 250 on cha plus 227 on
bkn), against a D202 pooled half-width of 0.0155 for team. The open
rows are position p90 and the near-field bias. The reading of section 5 is a practical-size rule, not a test: a row is
scorer-sensitive when the absolute pooled change exceeds D202's half-width, (hi - lo) / 2 of its pooled interval.

**8.2 The "about 1 percent" came from a different rule.** D214's diagnostic used the 1e6 rule (pairs beyond the gate
cost 1e6, tools/p8_teambound.py), not the cap registered here. Report only: the frames and pairs where the capped and
1e6 assignments differ are counted.

**8.3 Watch (replaces section 6's).** As P8_teambound.md 9.12 (5): per pool window, builds/r23/<window>_*,
label_build/<window>*, cache/scorecard/<window>.json and the current truth's labels file; per game, both truth_accept
records. No hou_cle_2016 file is hashed.

**8.4 The cache.** cache_gaps (tools.p8_placecause) runs on every window before scoring, and any gap stops the run, since
a missing entry would make frame_refs write the cache.

**8.5 G1 (replaces section 4's).** G1 calls tools.p8_resetfix.score_pair directly, as score_set's loop does
(tools/p8_rebaseline.py lines 248 to 275), never D202's command (which moves score_run.json). D202's per-window hashes
(sidecar, time map, labels, identity where recorded) and its record's SHA-256 and run id are checked first.

**8.6 The swap took effect.**
- Calls are counted: in the uncapped pass the frozen _match runs under a counting wrapper that calls it unchanged, and
  the capped function is called 0 times; in the capped pass the capped function is called exactly as many times as the
  frozen one was in the uncapped pass (the same scored frames with boxes).
- At least one pair must differ between the two rules over the run, else the run stops ("the swap had no effect").
- The original is restored in a finally block, and a test checks it.

**8.7 Rows, two ways (completes section 4).** Each rule's own D202 values (its own matched players and its own team n,
tools.p8_rebaseline.unit_rows on that rule's windows) and their difference, labelled "own rule values"; beside them the
paired change on players matched under both (tools.p8_rebaseline.change_rows), labelled "paired, common players". The
reading of 8.1 uses the paired change; both are reported. The rows that must not move (section 2) are checked equal
window by window.

**8.8 Cost.** Re-estimated at about 3 to 5 lock minutes for the two passes (cap 15). Writes: reports/research/
p8_pairgate.json only.

**8.9 Code note (2026-11-27, local).** The test of section 3's "constructed case" is restated: capping does not keep
more within-gate pairs than the frozen rule in general (it lowers the cost of pairs beyond the gate, so it can keep
fewer as well as more). What it changes is which pairs: distances beyond the gate no longer steer the assignment. The
test builds a case where the frozen rule keeps one within-gate pair and the capped rule another, and the pair counts
gained and lost are reported, not assumed to rise.

**8.10 Before the run (2026-11-27, local; the code check of 3a642ba5, GO).** Team rows gained and lost are not counted
row by row: each rule's team n (counts.team_n_uncapped and counts.team_n_capped per scope) stands in for them, and their
difference is the net change. The tool reads only the pool keys of reports/research/p8_rebaseline.json
(pooled_cha_bkn and per_game gsw_cha_2016__current and gsw_bkn_2015__current); no hou key is read.


## 9. Result (2026-11-27, local; run at d2a06798; reports/research/p8_pairgate.json)

**9.1 The run.** The fast suite under the lock first ("p8 pairgate fast tests", 357 s from the lock's "done in" line in the chain
log, passed), then one lock job ("p8 pairgate", 502 s from the lock's "done in" line; the record's own seconds are
489.7). hou_cle_2016: no file read or hashed.
- G1: the uncapped path reproduces D202's GATE_FIELDS on every one of the 156 windows. The D202 records hashed as pinned,
  every window's inputs hashed as D202 recorded them, and cache_gaps found nothing; the watched files were unchanged.
- The swap: the frozen _match ran 5,893 times in the uncapped pass, the capped function 5,893 times in the capped
  pass and 0 times in the uncapped one.
- The rows that do not use _match (missed, detection, placement, both ghost rows, net error, one-to-one net error,
  coverage) changed by exactly 0 on every window, as section 2 predicted.

**9.2 Pairs.** The frozen rule kept 51,244 pairs and the capped rule 51,522: 953 gained and 675 lost, on 745 of
the 5,893 scored frames with boxes. Team n: 49,674 uncapped against 49,948 capped (gsw_cha_2016 24,708
and 24,862; gsw_bkn_2015 24,966 and 25,086). Report only (8.2): the capped and the 1e6 assignments differ on 181 frames and in 588
pairs.

**9.3 Rows, as registered (8.1, 8.7).** Own-rule values (each rule's own matched players and team n) and their
difference; beside them the paired change on players matched under both rules with its window-bootstrap interval, the
D202 half-width and the label. The eight rows of section 2 are 0 in every scope and are not repeated here.

| scope | row | own, uncapped | own, capped | own difference | paired change [interval] | D202 half-width | reading |
|---|---|---|---|---|---|---|---|
| gsw_cha_2016 | position p50 (ft) | 1.5664 | 1.5594 | -0.0070 | -0.0023 [-0.0047, -0.0001] | 0.0827 | insensitive |
| gsw_cha_2016 | position p90 (ft) | 3.3038 | 3.2663 | -0.0376 | -0.0148 [-0.0303, -0.0061] | 0.1659 | insensitive |
| gsw_cha_2016 | near-field bias (ft) | 1.2171 | 1.2121 | -0.0050 | -0.0051 [-0.0112, +0.0004] | 0.0961 | insensitive |
| gsw_cha_2016 | team | 0.8101 | 0.8067 | -0.0034 | -0.0034 [-0.0057, -0.0012] | 0.0242 | insensitive |
| gsw_bkn_2015 | position p50 (ft) | 1.5931 | 1.5869 | -0.0062 | -0.0035 [-0.0061, -0.0013] | 0.0641 | insensitive |
| gsw_bkn_2015 | position p90 (ft) | 3.4304 | 3.4186 | -0.0118 | -0.0105 [-0.0231, -0.0032] | 0.1489 | insensitive |
| gsw_bkn_2015 | near-field bias (ft) | 1.5954 | 1.5895 | -0.0058 | -0.0059 [-0.0146, -0.0011] | 0.1064 | insensitive |
| gsw_bkn_2015 | team | 0.8572 | 0.8553 | -0.0019 | -0.0019 [-0.0036, -0.0004] | 0.0146 | insensitive |
| pooled | position p50 (ft) | 1.5803 | 1.5727 | -0.0076 | -0.0026 [-0.0044, -0.0012] | 0.0547 | insensitive |
| pooled | position p90 (ft) | 3.3705 | 3.3419 | -0.0286 | -0.0135 [-0.0237, -0.0073] | 0.1167 | insensitive |
| pooled | near-field bias (ft) | 1.4107 | 1.4043 | -0.0063 | -0.0067 [-0.0095, -0.0010] | 0.0919 | insensitive |
| pooled | team | 0.8338 | 0.8311 | -0.0027 | -0.0027 [-0.0041, -0.0013] | 0.0155 | insensitive |

Note: for team, the "paired" column is the own-rule difference (change_rows' team row uses each rule's own matched
players); only the three position rows are paired on players matched under both rules.

**Every row reads insensitive** in every scope: no change comes within a third of its D202 half-width. As 8.1
expected, team was nearly certain to; position p90 and the near-field bias, the open rows, also read insensitive.

**9.4 Post hoc, not registered.**
- Under the capped rule every matched-position row moves slightly toward smaller errors, and team moves slightly down
  (pooled -0.0027, interval [-0.0041, -0.0013]) while team n rises by 274: the pairs the capped rule adds or reassigns
  are, on these numbers, a little less often team-correct than the rest. Not tested.
- change_rows' team row uses each rule's own matched players (its label says so), so for team the "paired" column
  equals the own-rule difference; only the three position rows are on players matched under both.
- The own-rule p90 difference (-0.029 ft pooled) is about twice the paired one (-0.013 ft), since each rule's own set
  includes the pairs only it keeps.
- For Lucien's option (d): under the capped rule, applying the gate before the assignment moves no D202 row by more
  than about a sixth of its half-width on cha and bkn (the 1e6 rule was not scored). The frozen rule stays the reported
  one.
