# P8 placement route, step 1: is r23's foot offset systematic or scattered? (registered before any number)

Written 2026-11-26 after E1b stopped at its transfer gate (docs/research/P8_E1B.md sections 2b and 2c). Report only:
no model runs, nothing is built or trained, no sidecar is changed, no Colab unit. Tool: tools/p8_placement.py (rules in
its docstring, the same as below); record: reports/research/p8_placement.json. This section was committed before the
tool produced any number.

## Why

On the E1 corpus frames, 57 percent of r23's raw misses are players with a box whose foot lands 2.5 to 5 ft from
SportVU and no model places it closer (P8_E1B.md section 2). Detector labels cannot reach them, and raw placement gains
keep only about 0.3 of their size at build level, because the build's foot stabilisation acts on the same error. If the
build-level offset is systematic, a correction of the stabilised foot could lower the missed row at almost no cost.
If it is scattered, the misses at the 2.5 ft gate are set by truth, timing or foot noise, and the Level 2 missed target
must be read against that floor.

## Registered definitions

Frames and games:

- Development pool: gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015. Every measure, fit and threshold uses these only.
- hou_cle_2016 is a check: measured and reported apart; the corrections fitted on the whole pool are applied to it;
  nothing is fitted on it or chosen with it.
- Sidecars: the reference-platform r23 builds (detector on CPU, D175): builds/e1m6_r23_detcpu (gsw_bkn_2015) and
  builds/m9_r23_detcpu__<game>. Windows: the scored windows of their registered score files (e1/m6__e1__gsw_bkn_2015.json,
  e1/m9__r23_detcpu__<game>.json).
- Universe: the pinned scorer's (tools.p8_scorecard.score_window, truth mode camera): camera-truth rows on live wide
  frames (cached wide score at or above the cached threshold) that the sidecar processed; in-frame players with the 10
  px margin. Check: per game, the in-frame and missed counts must equal the registered r23 counts summed over the
  score file's windows, or the tool stops.
- Guards: tools.p8_common.guard and tools.research_testset.refuse_cv_test on every game, SportVU id and window. No
  held-out game, no D85 test game, no hand labelling.
- Inputs are stored P8-workspace files only (sidecars, score files, truth rows, scorecard cache, time maps, SportVU
  moments, mirror files). Nothing under DATA_ROOT, no model, no video decode, so no heavy-job lock.

Population:

- Each in-frame player gets its nearest foot_stab (court ft through the camera truth; nearest, not one to one, as the
  scorer's missed rule) at distance d: hit (d at most 2.5 ft), placement miss (2.5 to 5 ft), far miss (no foot within
  5 ft).
- Matched comparison: hits on the same frames, with their medians in every camera third by expected height cell, and
  the hits' median reweighted to the placement misses' cell mix.

Measures:

- Offset: nearest foot minus the SportVU position. Depth: along the court-plane unit vector from the camera's ground
  point (camera_C of the truth row) to the player, positive when the foot lands beyond the player. Lateral: across it
  (the depth vector turned 90 degrees counter-clockwise).
- Medians, interquartile range, share beyond, share depth-dominant; by camera third (near, middle, far) and by
  expected height of a 6.5 ft player (under 150, 150 to 200, 200 to 250, 250 px and over).
- Timing signature: for hits and placement misses moving at 3 ft/s or more (SportVU velocity by central difference
  over plus or minus 0.2 s), the offset's projection on the direction of motion. Its least-squares slope on speed
  through the origin is a lag in seconds.
- Lag scan: SportVU positions shifted by k moments, k from -12 to +12 (0.04 s a step, plus or minus 0.48 s), feet
  unchanged; the missed rate of the players present at every lag.
- Intervals: window bootstrap, 2,000 resamples, seed 0, windows resampled whole within a game.

Corrections, each fitted leave one game out on the other two pool games (fit population: hits and placement misses)
and applied to every sidecar foot of the held-out game, the missed rule then recomputed with all boxes:

- C1: depth shift by camera third (the median depth offset of the foot's own third, removed along its own depth
  direction; the third comes from the foot, so the correction needs no truth to apply).
- C2: depth shift by the box's own pixel height (under 150, 150 to 200, 200 to 250, 250 and over).
- C3: depth and lateral shift by third.
- L: a lag, the k that minimises the two training games' pooled lag-scan missed rate (ties to the smaller |k|),
  applied to the held-out game's SportVU positions.

## Reading rule (fixed now)

- SYSTEMATIC, a correctable foot offset, if some correction C1, C2 or C3 has, on every one of the three held-out games:
  (a) net missed change at most -0.010 over all in-frame players of the game; (b) at least 10 percent of the game's
  placement misses moved within 2.5 ft; and (c) the placement misses' median depth offset has the same sign on all
  three games, with each game's window interval excluding zero (with 3 games this is the game-cluster condition).
- TIMING if the lag L is non-zero in every fold and lowers the held-out game's missed rate by at least 0.010 on every
  held-out game. Then the corrections are fitted and tested again with each fold's lag applied to SportVU, and
  SYSTEMATIC is read from that second pass.
- SCATTERED if neither holds. A correction that passes on two games of three is scattered by this rule.

Why these numbers:

- -0.010 net on every game is the missed bar of the adoption rule this project would apply in step 2 (as P8_E1B.md
  section 5: pooled at most -0.010 with every game below zero). A correction that cannot reach it on each game cannot
  pass that rule, so step 2 would be wasted.
- 10 percent of placement misses moved in is the floor that ties the gain to the population named. Placement misses
  are roughly 0.12 to 0.2 of in-frame players, so 10 percent of them is about 0.012 to 0.02 of in-frame players
  gross, enough to leave -0.010 net after some hits are pushed out; a correction that only reshuffles hits fails it.
- (c) asks for direction, not only size: a shift helps only if the offset leans the same way on every game.
- The lag bar (-0.010 on every game) is the same size, for the same reason.

Multiplicity: four forms are tested (C1, C2, C3, L), which raises the chance that one passes by luck; the every-game
condition and the leave-one-game-out fit are the guard. The forms were fixed before any number.

## Next step for each reading

- SYSTEMATIC: step 2. The passing correction (if several, the one with the largest pooled net change) is fitted again
  leave one game out and applied to foot_stab in the build's own court frame (the build's H, not the truth's, so it
  is deployable), written as corrected sidecar copies in a scratch folder, and rescored with the pinned scorer as M5
  rescored bottom centres (about 2 lock minutes a game, under the heavy-job lock, no units). Its rule is registered
  then: missed at most -0.010 pooled with every game below zero, and position p50 and p90, near-field bias, ghosts
  and team non-inferior to r23 within E1b's margins; hou_cle_2016 as the new-arena check.
- TIMING: not a pipeline fix. The lag sits in the truth alignment (the frozen offset rule stays, D166 (1)), so the
  misses it explains are a truth artefact. Proposal to the lead: a registered truth-sync check, and the Level 2 missed
  target read against the lag-corrected floor; step 2 only if the second pass reads SYSTEMATIC.
- SCATTERED: step 3. The Level 2 missed target is restated against the truth's own floor at the 2.5 ft gate, and the
  detector is revisited only for the presence share (about 0.016 of in-frame players at raw level).

## Caveats fixed now

- The camera truth was fitted from r23's builds with player feet among its inputs (D98, D99), and each window's
  clock offset was solved at that fit. Both can absorb part of a systematic foot offset or a timing lag, so this
  measurement can understate them.
- A correction fitted with the truth camera's depth direction is not yet deployable; step 2 moves it to the build's
  own frame.
- The 10 px margin, the nearest-foot rule and the 2.5 ft gate are the scorer's, so "moved within 2.5 ft" can also come
  from a foot moving closer to a different player; the net change over all players counts that both ways.

## Results (computed 2026-11-26, after the definitions above were committed in 31f72a2e)

Run on stored P8-workspace files only (no lock, no model, no DATA_ROOT; about 45 seconds). Two code fixes were
committed before any result was read, neither changing a rule: the count check first stopped on gsw_cha_2016 because
the tool read the registered MPS side of the M9 score file instead of its CPU side (fixed in its own commit), and a
JSON output fix. The count check then passed on every game: in-frame and missed players equal the registered
reference-platform r23 counts exactly. Record: reports/research/p8_placement.json.

Population (build level, nearest foot_stab through the camera truth):

| Game | In-frame | Missed (rate) | Placement misses (share of misses) | Far misses, no foot within 5 ft (rate of in-frame) |
| --- | --- | --- | --- | --- |
| gsw_bkn_2015 | 20,457 | 5,166 (0.2525) | 3,295 (64%) | 1,871 (0.091) |
| gsw_cha_2016 | 13,284 | 3,922 (0.2952) | 2,924 (75%) | 998 (0.075) |
| gsw_sac_2015 | 21,144 | 4,355 (0.2060) | 3,251 (75%) | 1,104 (0.052) |
| hou_cle_2016 (check) | 43,030 | 8,439 (0.1961) | 5,712 (68%) | 2,727 (0.063) |

Offsets (depth: positive when the foot lands beyond the player, away from the camera; window intervals):

| Game | Placement: median depth [interval]; IQR | Placement: median lateral; median abs lateral, abs depth | Hits: median depth [interval]; reweighted to the placement mix |
| --- | --- | --- | --- |
| gsw_bkn_2015 | -1.88 [-2.08, -1.54]; -2.93 to +2.30 | +0.05; 1.40, 2.73 | -0.43 [-0.53, -0.33]; -0.47 |
| gsw_cha_2016 | -1.10 [-1.60, -0.59]; -2.88 to +2.49 | +0.26; 1.49, 2.73 | -0.19 [-0.25, -0.12]; -0.27 |
| gsw_sac_2015 | -1.92 [-2.29, -1.39]; -2.98 to +2.24 | +0.06; 1.28, 2.72 | -0.40 [-0.48, -0.32]; -0.43 |
| hou_cle_2016 | -2.40 [-2.51, -2.10]; -3.13 to +2.58 | -0.01; 0.71, 2.91 | -0.20 [-0.24, -0.15]; -0.30 |

By camera third, placement misses' median depth (near, middle, far): gsw_bkn_2015 +2.43, -1.84, -2.62; gsw_cha_2016
+2.64, -0.72, -2.69; gsw_sac_2015 +0.75, -2.23, -2.47; hou_cle_2016 +2.80, -2.10, -2.94. Hits by third: -0.07, -0.48,
-0.77; +0.35, -0.24, -0.81; -0.17, -0.44, -0.65; +0.33, -0.17, -0.92. By expected height the same picture: players
200 to 250 px tall (near the camera) land beyond (+1.7 to +3.1 ft), players under 150 px land short (-1.0 to -2.8 ft).
The depth component dominates in 69 to 82 percent of placement misses.

Timing: no signature. The slope of the offset's projection on the direction of motion is 0.005 s [-0.015, 0.030],
0.010 s [-0.011, 0.034], 0.001 s [-0.028, 0.031] and -0.004 s [-0.014, 0.004] (bkn, cha, sac, hou_cle), and placement
misses lie ahead of the motion 48 to 53 percent of the time. The lag scan's lowest missed rate is at k = 0 on every
game, pooled and per fold; one step (0.04 s) either way raises missed by 0.0004 to 0.0054.

Leave-one-game-out corrections (net change in missed over all in-frame players of the held-out game [window
interval]; share of its placement misses moved within 2.5 ft; moved in, pushed out):

| Correction | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 |
| --- | --- | --- | --- |
| C1, depth by third | -0.0123 [-0.0170, -0.0073]; 20.0%; 658, 406 | -0.0146 [-0.0219, -0.0073]; 17.1%; 501, 307 | +0.0041 [-0.0010, 0.0090]; 17.8%; 578, 664 |
| C2, depth by box height | -0.0089 [-0.0127, -0.0050]; 15.7%; 518, 337 | -0.0032 [-0.0089, 0.0026]; 14.8%; 432, 389 | -0.0004 [-0.0048, 0.0035]; 14.8%; 481, 472 |
| C3, depth and lateral by third | -0.0123 [-0.0174, -0.0070]; 20.1%; 661, 409 | -0.0168 [-0.0240, -0.0094]; 17.8%; 519, 296 | +0.0046 [-0.0004, 0.0097]; 18.0%; 584, 681 |
| L, lag | 0 chosen, no change | 0 chosen, no change | 0 chosen, no change |

Fitted shifts (median depth of hits and placement misses of the two training games): far -0.88 to -1.08 ft, middle
-0.40 to -0.51 ft, near -0.07 to +0.25 ft, depending on the fold.

Check game, report only (corrections fitted on the whole pool): hou_cle_2016 C1 -0.0206 [-0.0252, -0.0164], 25.6
percent moved in; C2 -0.0078; C3 -0.0205.

## Reading by the registered rule

SCATTERED. Condition (c) holds: the placement misses' median depth offset is negative on all three pool games and
each interval excludes zero. Conditions (a) and (b) do not hold for any correction on every game: C1 and C3 pass on
gsw_bkn_2015 and gsw_cha_2016 and fail on gsw_sac_2015 (net +0.004 and +0.005, pushing out more hits than they move
in); C2 fails on two games. TIMING does not hold: the lag chosen in every fold is 0. By the rule fixed in advance, a
correction that passes on two games of three is read as scattered.

What the numbers say beyond the rule (post hoc, tested by nothing here):

- The error is along the camera ray, not across it, and it has a structure: near-camera players' feet land beyond
  them and far players' feet land short, on every game, for hits as well as misses. That is the shape of a depth
  error that grows with distance (foot point against SportVU's body position, or the camera truth's pose), not
  random scatter.
- But its size varies by game, mostly in the near third (hits' near median -0.17 to +0.35 ft), so a shift fitted on
  two games over-corrects the third. On gsw_sac_2015, the game with the fewest misses, that pushes out more hits than
  it brings in.
- Placement misses are not the tail of the hits' shift: in the same third and height cells, hits' median depth is
  -0.27 to -0.47 ft against -1.1 to -1.9 ft for placement misses, with an interquartile range of about plus or minus
  2.5 ft. A single shift cannot move most of them.
- hou_cle_2016 gains -0.021 from the pooled C1. It is a check game and chose nothing, but it was seen; any new
  correction arm designed after this is a forking path and must say so.

## Next step (registered: SCATTERED leads to step 3)

Step 3, proposed for the lead:

1. Restate the Level 2 missed row as two rows, each with its own target. The detection row is the far miss (no foot
   within 5 ft, sportvu.bench.MATCH_FT): 0.052 to 0.091 of in-frame players at build level on the pool, 0.063 on
   hou_cle_2016. That is already near the first draft's 5 percent target, and it is the row a detector change can
   move. The placement row is the miss between 2.5 and 5 ft, 0.15 to 0.22 of in-frame players on the pool (0.13 on
   hou_cle_2016). It is set by a depth
   error along the camera ray whose size varies by game, so at the 2.5 ft gate it is a floor that the truth and
   geometry set as much as the pipeline does.
2. The detector question is revisited only against the detection row (presence: about 0.016 of in-frame players at
   raw level, P8_E1B.md section 2). E1b's labels do not reach it by enough (section 2b there).
3. A per-game depth correction is not proposed as step 2: by this rule it did not pass. If the lead wants to try
   one, it needs a new registration that names this result. One example is a correction that estimates the near-third
   shift within each game without truth, from the build's own geometry. The registration must name its forking
   paths: the third-dependent structure and hou_cle_2016's gain were seen here.

Cost of step 3: no compute; a D line and a rewrite of the Level 2 rows in TRACKING_LEVELS.md. Splitting the scorecard's
missed row into the two rows would be a reporting change to tools/p8_scorecard.py's breakdowns (report only, the pinned
rows unchanged), about an hour of code and tests, and minutes of rescoring per game under the lock.
