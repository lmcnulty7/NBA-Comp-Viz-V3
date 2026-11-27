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

Correction (2026-11-27, from the ml-review of 6e739eb1): in this section's change, only position p50, p90 and
near-field bias are on players matched under both truths; team is computed from each truth's own matched players on
the windows both score (tools.p8_e1_score.rows counts each side's team rows).

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

## 11. Result (2026-11-27; run 20261127T171848Z; record reports/research/p8_rebaseline.json)

**Development standard rows (D193 (2)): r23 on the fixed tracker; not the reported baseline (D198).** Claim limits:
(a) shared input, each truth's frames were selected by fits on the same r23 label build's boxes, so the rows are
optimistic in absolute terms and biased toward r23; (b) multipool v1 is a better fit by the acceptance measures, not a
truth shown to be better (D199). Every interval is a window-cluster bootstrap (2,000 resamples, seed 0) of the value
on that truth; none is a difference unless named so, and nothing gates.

**Run.** Score step 654.6 s (the run record's seconds) inside the lock "p8 rebaseline score"; report in system python.
Every pre-run check held, the manifests before and after were equal, and the reproduction gate passed: gsw_bkn_2015 on
its previous truth reproduces the reset-fix arm's record on all 83 windows and all twelve gate fields, with the arm's
truth pin. Caches: 73 gsw_cha_2016 and 93 hou_cle_2016 cache files copied from the workspace (same clips), none for
gsw_bkn_2015 (present); entries computed in the score step, all referee detections (no wide or cut entry was missing):

| game and truth | windows with entries computed | referee entries | window seconds |
|---|---|---|---|
| gsw_bkn_2015, previous | 0 | 0 | 0 |
| gsw_bkn_2015, current | 49 | 195 | 50.6 |
| gsw_cha_2016, previous | 0 | 0 | 0 |
| gsw_cha_2016, current | 54 | 1,677 | 102.8 |
| hou_cle_2016, current | 78 | 3,858 | 233.6 |

**Rows on the decided truths** (value [window interval]; missed = detection + placement; scored-frame counts are
not in the record and are left out, the windows and in-frame players stand for the sample size):

| row | gsw_cha_2016 (multipool v1) | gsw_bkn_2015 (multipool v1) | pooled cha and bkn | hou_cle_2016 (fixed CPU), apart |
|---|---|---|---|---|
| windows; in-frame players | 73; 28,427 | 83; 29,370 | 156; 57,797 | 93; 71,960 |
| missed | 0.2085 [0.1913, 0.2280] | 0.2200 [0.2053, 0.2365] | 0.2143 [0.2026, 0.2263] | 0.1580 [0.1455, 0.1706] |
| detection (no foot within 5 ft) | 0.0523 [0.0471, 0.0588] | 0.0652 [0.0569, 0.0751] | 0.0589 [0.0541, 0.0651] | 0.0439 [0.0395, 0.0485] |
| placement (2.5 to 5 ft) | 0.1562 [0.1416, 0.1711] | 0.1548 [0.1437, 0.1673] | 0.1555 [0.1460, 0.1648] | 0.1140 [0.1045, 0.1235] |
| ghosts per box | 0.1245 [0.1106, 0.1408] | 0.1177 [0.1056, 0.1316] | 0.1211 [0.1112, 0.1322] | 0.0709 [0.0614, 0.0803] |
| ghosts per in-frame player | 0.1182 [0.1047, 0.1344] | 0.1076 [0.0964, 0.1207] | 0.1128 [0.1033, 0.1233] | 0.0659 [0.0570, 0.0748] |
| net error E | 0.3267 [0.2968, 0.3606] | 0.3276 [0.3038, 0.3546] | 0.3272 [0.3078, 0.3493] | 0.2238 [0.2031, 0.2449] |
| one-to-one net error | 0.4063 [0.3725, 0.4450] | 0.4136 [0.3858, 0.4459] | 0.4100 [0.3875, 0.4346] | 0.2936 [0.2698, 0.3175] |
| position p50 (ft) | 1.566 [1.484, 1.649] | 1.593 [1.536, 1.664] | 1.580 [1.527, 1.636] | 1.556 [1.500, 1.620] |
| position p90 (ft) | 3.304 [3.150, 3.482] | 3.430 [3.293, 3.591] | 3.370 [3.260, 3.493] | 3.400 [3.280, 3.525] |
| near-field bias (ft) | 1.217 [1.134, 1.326] | 1.595 [1.484, 1.697] | 1.411 [1.319, 1.502] | 1.172 [1.076, 1.292] |
| team | 0.8101 [0.7855, 0.8339] | 0.8572 [0.8423, 0.8716] | 0.8338 [0.8180, 0.8490] | 0.7558 [0.7206, 0.7884] |
| coverage | 0.9930 [0.9818, 0.9993] | 0.9981 [0.9958, 0.9997] | 0.9957 [0.9902, 0.9990] | 0.9997 [0.9993, 1.0000] |

**The same build on the previous truths, report only.** gsw_cha_2016 on the Mac MPS truth (73 windows, 13,284 in-frame players): missed 0.2915 [0.2728, 0.3150], detection 0.0696 [0.0598, 0.0817], placement 0.2219 [0.2109, 0.2364], ghosts per box 0.1884 [0.1713, 0.2087], ghosts per in-frame player 0.1785 [0.1625, 0.1978], net error E 0.4700 [0.4366, 0.5113], one-to-one net error 0.5710 [0.5385, 0.6110], position p50 1.522 [1.424, 1.635], position p90 3.108 [2.942, 3.368], near-field bias 1.152 [1.049, 1.261], team 0.8116 [0.7804, 0.8385], coverage 0.9930 [0.9818, 0.9993]. gsw_bkn_2015 on the fixed CPU truth (83 windows, 28,554 in-frame players; the reset-fix arm's own rows, as the gate requires): missed 0.2526 [0.2391, 0.2687], detection 0.0748 [0.0654, 0.0862], placement 0.1778 [0.1676, 0.1889], ghosts per box 0.1382 [0.1247, 0.1542], ghosts per in-frame player 0.1262 [0.1135, 0.1410], net error E 0.3788 [0.3554, 0.4082], one-to-one net error 0.4736 [0.4481, 0.5050], position p50 1.583 [1.526, 1.653], position p90 3.406 [3.262, 3.560], near-field bias 1.533 [1.416, 1.635], team 0.8576 [0.8421, 0.8717], coverage 0.9981 [0.9958, 0.9997].

**The truth change, per game (current minus previous, paired window bootstrap over the windows both truths score, all
of each game's windows):**

| row | gsw_cha_2016: platform, the bug's thinning and the fit changed | gsw_bkn_2015: the fit only |
|---|---|---|
| missed | -0.0830 [-0.1087, -0.0577] | -0.0326 [-0.0398, -0.0256] |
| detection (no foot within 5 ft) | -0.0173 [-0.0298, -0.0067] | -0.0096 [-0.0129, -0.0063] |
| placement (2.5 to 5 ft) | -0.0658 [-0.0850, -0.0473] | -0.0230 [-0.0301, -0.0160] |
| ghosts per box | -0.0639 [-0.0876, -0.0405] | -0.0205 [-0.0271, -0.0141] |
| ghosts per in-frame player | -0.0602 [-0.0830, -0.0377] | -0.0186 [-0.0249, -0.0127] |
| net error E | -0.1433 [-0.1904, -0.0970] | -0.0512 [-0.0639, -0.0390] |
| one-to-one net error | -0.1646 [-0.2137, -0.1171] | -0.0600 [-0.0741, -0.0461] |
| position p50 (ft), players matched under both truths | +0.011 [-0.108, +0.137] | +0.000 [-0.001, +0.002] |
| position p90 (ft), players matched under both truths | -0.014 [-0.219, +0.218] | -0.002 [-0.008, +0.004] |
| near-field bias (ft), players matched under both truths | +0.008 [-0.083, +0.121] | +0.003 [+0.000, +0.010] |
| team, each truth's own matched players on the windows both score | -0.0014 [-0.0197, +0.0210] | -0.0003 [-0.0037, +0.0036] |
| coverage | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

**Common-frames split** (both truths' rows on only the frames both accepted; the geometry part): gsw_cha_2016 912 frames in both, 454 previous only, 2,012 current only: missed -0.1108 [-0.1383, -0.0801], E -0.1914 [-0.2377, -0.1404], ghosts per box -0.0847 [-0.1061, -0.0614], team +0.00122 [-0.00426, +0.00613]. gsw_bkn_2015 2,543 frames in both, 360 previous only, 441 current only: missed -0.0318 [-0.0390, -0.0239], E -0.0499 [-0.0617, -0.0380], ghosts per box -0.0196 [-0.0251, -0.0141], team +0.00037 [-0.00195, +0.00269].
The position p50, p90 and near-field bias columns of the change already sit on players matched under both truths, so
they coincide with the split's; team does not (it counts each truth's own matched players), so the split's team change
is given above.

**Post hoc, not registered.** Most of each game's truth change in missed is placement (gsw_cha_2016 -0.066 of -0.083,
gsw_bkn_2015 -0.023 of -0.033), and the change persists on the frames both truths accept (gsw_cha_2016's is larger
there), so it comes mainly from the truth's geometry, not from which frames it accepts. Given limit (a), the geometry
part may be the fit moving truth positions toward r23's own boxes, and for gsw_cha_2016 it also mixes in the platform
change; it does not show which truth is nearer the players' real positions. The same build reads lower missed, ghosts
and net error under the multipool truths than under the previous ones by these amounts; given limit (b), that is a
change in the reference, not an improvement of the build.

Review notes (2026-11-27, the ml-review of 6e739eb1, FAIL on one label): the truth change's team column was labelled as
players matched under both truths; it is each truth's own matched players on the windows both score (for example
gsw_cha_2016: 11,298 team rows on the previous truth against 24,708 on the current), corrected in the record (report
rerun, every number unchanged) and here, with the split's team change given apart; values rounded once from the record;
ghosts per player and coverage added to the change table and the previous-truth rows, coverage with its intervals; the
limit (a) sentence added to the post hoc reading.
