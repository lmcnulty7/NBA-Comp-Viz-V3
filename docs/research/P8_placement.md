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

## Results

Not computed yet.
