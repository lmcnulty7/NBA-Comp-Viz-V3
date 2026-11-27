# P8: Level 2 rows of r23 on the fixed tracker, on the decided truth set (registered 2026-11-27, before any number)

Status: design only, report only. No number in this document has been computed. Nothing trains, nothing is adopted,
no rule changes, and no row here is a gate. Code, after review: tools/p8_rebaseline.py with tests; record
reports/research/p8_rebaseline.json.

## 1. Question

What are the Level 2 rows of the reference build, r23 on the fixed tracker, on the truth set decided in D199? Beside
them, report only: the same build on each game's previous truth, so the effect of the truth change on the numbers is
visible.

## 2. Inputs (all exist; nothing is rebuilt)

**Build.** r23 on the reference platform with the tracker reset fixed (D193): the reference root's builds/r23, the
check C composite (the reset-fix arm's build on each planned reset window, the unfixed r23_nofix build elsewhere,
byte-equal to it there by each game's equivalence check, resetfix/equiv__<game>.json). Before scoring, the tool asserts
for every window that the composite's frames, trajectories and identity hash as the planned source (the arm folder on
reset windows, r23_nofix elsewhere), and, for gsw_cha_2016 and gsw_bkn_2015, that work_ms's copy of builds/r23 is
byte-identical to the reference root's.

**Truths** (each game's status from its acceptance record, asserted "use" for the current truth):

| game | current truth (D199) | root of its labels | previous truth, report only |
|---|---|---|---|
| gsw_cha_2016 | multipool v1 | work_ms | Mac MPS camera truth (P8 workspace), the truth in use before D199; its fixed CPU truth was "unused" and is not reported |
| gsw_bkn_2015 | multipool v1 | work_ms | fixed CPU truth (reference root), the reset-fix arm's |
| hou_cle_2016 | fixed CPU truth | reference root | none reported (its multipool fit is deferred; its Mac MPS truth predates the fixed tracker and is left out) |

gsw_sac_2015 (apart on check C) and gsw_ind_2016 (no truth) are not scored. Every labels set used is pinned by SHA-256
(every labels/<game>/*.jsonl, <game>_result.json and <game>_offsets.json) before any score, and a score refuses changed
files.

**Clips, time maps, caches.** Scoring reads the clip and time map of each window from the root whose labels it uses
(work_ms links to the reference root's label_build; the workspace has its own). The tool asserts, per window, that the
clip is byte-identical to the reference root's and that the time map has the same content (frames and spans; run
fields such as ocr_seconds ignored, the lesson of P8_sac_checkc.md). The scorecard caches (gate v2 wide score, referee
detections, the cut signal) depend only on the clip: every score uses one cache folder, the reference root's
cache/scorecard; a window missing there is copied from the workspace's cache when the clips are byte-identical, and
any entry still missing (referee detections on truth frames not cached before) is computed by the score step and
timed. No file is written in work_ms or in the P8 workspace.

## 3. Scoring (pinned, unchanged)

Per window and truth: tools.p8_scorecard.score_window (D88, the scorecard pin unchanged), truth mode camera,
foot_stab, then tools.p8_e1_score.window_stats and the reset-fix scorer's extra counts on the same scored frames
(tools.p8_resetfix.score_pair: far misses, one-to-one missed, duplicates, one-to-one unmatched players and boxes),
with the build as both of score_pair's sides so each window is scored once. No rule, threshold or code path of the
scorer changes.

## 4. Rows

Per game, pooled over the three games, and for each with a window-cluster bootstrap (2,000 resamples, seed 0, the
resample indices of tools.p8_e1_score.boot):
- missed (no stabilised foot within 2.5 ft), split (D184) into detection (no foot within 5 ft) and placement (2.5 to
  5 ft; missed minus detection);
- ghosts per box (the ghost row, non-referee ghosts over non-referee boxes) and per in-frame player;
- net error E = (missed + non-referee ghosts) / in-frame players, and one-to-one net error (unmatched in-frame players
  plus unmatched non-referee boxes, one-to-one within 2.5 ft, over in-frame players; P8_resetfix.md section 21);
- position p50 and p90 and near-field bias (tools.p8_e1_score.rows on the matched players);
- team (majority accuracy, as rows);
- coverage, as context;
- counts: windows, scored frames, in-frame players, boxes.

Ratio rows are bootstrapped from window sums; position, bias and team by calling tools.p8_e1_score.rows on each
resample. Intervals describe this build on this truth; they are not differences and gate nothing.

## 5. The truth change, report only

For gsw_cha_2016 and gsw_bkn_2015, and pooled over the two: each row on the previous truth beside the current one,
and the change (current minus previous) with a paired window-cluster bootstrap over the windows both truths score
(each resample draws windows, and each row is computed on each truth from those windows). The two truths accept
different frames, so the change mixes the truth's frame selection and its geometry; it is reported as the effect of
the truth change on these numbers, not as a property of either truth. Counts of windows and frames scored by each
truth, and by both, are reported beside it.

## 6. Window sets

Per game and truth: every label window with a composite build and truth rows (gate-empty windows out), the set the
score file names; the frames are the pinned scorer's universe for that truth (live, wide and accepted camera truth).
Pooled: the union of the three games' windows on their current truths. The truth-change comparison uses each game's
windows scored by both truths, and names how many fall out on either side.

## 7. hou_cle_2016

hou_cle_2016 is a development game here, in the decided truth set (D199), and no longer a held-back test for any closed
arm: the reset-fix arm is closed (D198; its hou stage was never run, P8_resetfix.md section 25 built hou only for truth
acceptance and pinned or scored nothing), and no open arm reserves hou. This re-baseline is its own registration, not
part of that arm. Every game list runs through tools.p8_common.guard and tools.research_testset.refuse_cv_test, as in
every P8 step (hou_cle_2016 has passed both throughout P8); a refusal stops the run.

## 8. Cost

Scoring only, no build: five game and truth pairs (cha and bkn on two truths, hou on one), each window scored once.
The reset-fix arm scored gsw_bkn_2015 (83 windows, two arms where they differ plus the bracket) in 146 s; here about
60 to 120 s per pair, so about 5 to 10 minutes, plus cache entries for truth frames not cached before (estimate 0 to
30 minutes, measured per window and recorded). One lock, "p8 rebaseline score", frozen CV python, for all five pairs
in sequence; cap 1.5 times the upper estimate (60 minutes); at the cap it stops, reports nothing, and is not retried
without a new go. The report step (system python, no lock) reads only the score records.

## 9. Guards

Report only: no gate, no adoption, and no registered number is replaced (r23's registered rows and every earlier
result stand as recorded; D198). Nothing held out is read; the D85 test set is refused. Writes: the score records under
the reference root's rebaseline/ folder, the cache entries in its cache/scorecard, and reports/research/
p8_rebaseline.json; nothing under work_ms, the P8 workspace, truth_v2/ or truth_accept/.
