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

## 10. Version 2 (2026-11-27; the review of 97258274, GO WITH CHANGES, no blockers; before any number)

Where this section differs from sections 1 to 9, it rules.

**10.1 Name and claim limits (finding 4).** The rows are the **development standard rows (D193 (2))**: r23 on the
fixed tracker, the build every later development arm builds both sides on. They are not the reported baseline, which
stays r23's registered rows (D198); the record carries this label. Limits, stated with every row:
(a) shared input: each truth's frames were selected by fits on the same r23 label build's boxes (detector and
tracker shared with this build), so the rows are optimistic in absolute terms and biased toward r23;
(b) multipool v1 is a better fit by the acceptance measures, not a truth shown to be better (D199).

**10.2 Writes (finding 1).** tools.p8_frozen.install_paths is called only on the reference root. Each truth is scored
with {**P, "labels": <that truth's root>/labels}, the cache and label_build left on the reference root's layout;
install_paths is never called on work_ms or the P8 workspace (work_ms/cache is empty and stays so). Before the first
score and after the last, the tool takes a SHA-256 manifest and a listing of labels/, label_build/, cache/ and
truth_accept/ in work_ms and in the P8 workspace, and of labels/, label_build/ and truth_accept/ in the reference root;
any difference stops the run and reports nothing. The reference root's cache/scorecard is the only place it adds
entries (counted, 10.6). Writes: the reference root's rebaseline/ (score records, the bracket's filtered labels),
cache entries there, and reports/research/p8_rebaseline.json.

**10.3 Pooling (finding 3).** Pooled rows cover gsw_cha_2016 and gsw_bkn_2015 only (one truth version, multipool v1);
hou_cle_2016 is reported apart, per the P8 convention (D153). The pooled interval is window-level only (no game
interval with two games); windows enter the bootstrap in sorted name order (finding 9).

**10.4 The truth change (finding 2; replaces section 5).** Per game only; the pooled truth-change row is dropped, since
the two changes differ in kind. What changed: gsw_cha_2016, from the Mac MPS truth to multipool v1 on CPU label builds
with the fixed tracker, so the platform, the tracker bug's thinning of post-reset truth frames (P8_resetfix.md part
a) and the multipool fit all change at once; gsw_bkn_2015, from the fixed CPU truth to multipool v1 on the same label
builds, so only the fit changes. The section 5 change is computed on the windows both truths score; its position, bias
and team columns cover the players matched under both truths, which differs from section 4's rows (players matched
under one truth). Report-only split, the bracket method of P8_resetfix.md section 14.6: both truths' rows on only the
frames both truths accepted (filtered labels written under the reference root's rebaseline/labels_common/), so the
change on common frames shows the geometry part apart from the frame-selection part; counts of frames accepted by
each truth only and by both are reported.

**10.5 Reproduction gate (finding 5).** Before any other score: gsw_bkn_2015 on its previous truth (the fixed CPU
truth) with the composite must reproduce, per window, the reset-fix arm's score record (reference root
resetfix/score__gsw_bkn_2015.json, windows[*].new) on missed, in_frame, nr_ghosts, nr_boxes, matched, far,
missed_1to1 and unmatched_1to1, exactly; its truth pin must equal resetfix/truth_pin__gsw_bkn_2015.json's files.
Otherwise nothing is reported. Also asserted: reports/research/p8_scorecard_pin.json as tools/p8_scorecard.py checks
it; `git status -- tools sportvu` clean; HEAD recorded.

**10.6 Caches (finding 6).** Before scoring, per window and truth, the missing wide, referee and corr entries are
counted (against the frames the score will need); every window with any is timed, and the counts and seconds are
recorded. (The reset-fix arm's record said "cache seconds 0" while 1,217 referee entries in 65 gsw_bkn_2015 windows were
computed inside its 146 s; corrected in P8_resetfix.md section 24.)

**10.7 What becomes seen about hou_cle_2016 (finding 7).** Seen after this run: the development standard rows of r23
on the fixed tracker on hou_cle_2016's fixed CPU truth (section 4's rows with intervals). Limits: (i) no difference
between fixed and unfixed builds on hou_cle_2016 is computed; (ii) no breakdown beyond section 4's rows (no thirds,
zones or per-window lists in the report); (iii) any later arm that uses hou_cle_2016 cites this list as seen
(a forking path to name).

**10.8 Asserts (finding 8).** SHA-256 of every file score_window reads per window (sidecar, clip, time map, labels,
cache file) recorded before scoring; equal window lists, mirror and moments across the truths' roots; guard and
refuse_cv_test on every game, stem and SportVU id; the score step refuses to run without V3_HEAVY_HELD; find_private on
the record before it is written; the 60-minute cap checked between windows.

**10.9 Equivalence wording (finding 10; corrects section 2).** On no-reset windows the composite equals the unfixed
build by the code argument (the fix acts only inside PlayerTracker.reset, so a window with no reset makes the same
calls) and by two tested equivalence windows per game (resetfix/equiv__<game>.json); not every no-reset window was
compared byte for byte.

Section 8's cost stands, with the reproduction gate (one more pair) and the bracket split (two more scores per game
for cha and bkn) added: about 10 to 20 minutes of scoring plus the cache entries; the cap stays 60 minutes.

**10.10 Tool (2026-11-27, written, not run).** tools/p8_rebaseline.py (score inside the lock "p8 rebaseline score",
report in system python) and tests/test_p8_rebaseline.py. Order: gsw_bkn_2015 first, its previous truth first, so the
reproduction gate runs before any other number is kept; then gsw_cha_2016, then hou_cle_2016. A window whose cache
file is missing in the reference root gets the workspace's file when the clips are byte-identical (counted); every
entry still missing is counted per window (wide, referees, corr absent) and the window's scoring time recorded. Tests:
the ratio rows (placement as missed minus detection), intervals that contain their values, the truth change against
tools.p8_e1_score.diffs, the gate, the manifests and pins, the content test of time maps, the order.

**10.11 Re-check fixes (2026-11-27, PASS, GO WITH CHANGES; before the run).** The common-frames record stores the
frames accepted by the previous truth only, the current truth only and both, per game; the pooled bootstrap sorts all
pooled stems; every field of the scorecard pin is compared; the cache timing is named window_seconds; each truth
root's label stems must equal the scored stems, and the identity file joins the per-window hashes; every score file
carries the run's id, report refuses a file from another run, and an earlier run record is moved aside under
rebaseline/earlier/, never deleted; the gate also compares team, cov_acc, cov_live and duplicates.
