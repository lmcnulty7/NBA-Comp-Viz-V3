# P8 placement route: the short-foot bias under the build's H (registered before any number)

Written 2026-11-26 after placement step 1b read MIXED (docs/research/P8_placement.md). Part 2 of the lead's decision:
part 1, the gsw_cha_2016 camera-truth rebuild, is queued separately. Report only: no model runs, nothing is built,
trained or changed, no Colab unit. Tool: tools/p8_footbias.py (rules in its docstring, the same as here); record:
reports/research/p8_footbias.json. This document and the tool were committed before the tool produced any number.

## Question

Step 1b found that in the middle and far thirds the foot lands short of the player under both geometries. That is
the build's H and the truth camera alike. A correction that uses the build's H needs no truth to apply, so it could
ship. Is the bias, measured under the build's H, consistent enough across games to correct? And would a correction
fitted leave one game out lower the registered missed row on every pool game without raising ghosts?

## Registered definitions

Games and inputs:

- Pool: gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015, for every measure, fit and rule. hou_cle_2016 is report only: the
  corrections fitted on the whole pool are applied to it, and nothing is fitted on it or chosen with it.
- Inputs: the stored P8-workspace files of step 1. These are the reference-platform r23 sidecars and their registered
  score files, the camera-truth rows, the scorecard cache (wide scores, threshold, referee detections), the time maps,
  the SportVU moments and the mirror. Nothing under DATA_ROOT is read, no model runs, and no video is decoded, so no
  lock is needed.
- Guards: tools.p8_common.guard and tools.research_testset.refuse_cv_test run on every game, SportVU id and window. No
  held-out game, no D85 test game, no hand labelling.

Scoring, made explicit:

- The registered scorer (tools.p8_scorecard.score_window, truth mode camera) maps every foot_stab pixel to the court
  through the truth camera's H, as float32 pixels through a float64 H. Missed means no such foot lies within 2.5 ft
  of an in-frame player (10 px margin). A ghost is a box whose foot lies more than 3.0 ft from all 10 SportVU
  players. Referee boxes are those with IoU of at least 0.5 to a cached referee detection of the frame, and the ghost
  row is non-referee ghosts over non-referee boxes.
- The tool reproduces this exactly. Per game, the in-frame, missed, non-referee ghost and non-referee box counts must
  equal the registered score file's sums for the CPU build, or the tool stops.
- A correction in the build's court frame changes the court point q to q'. q' goes back to the image through the
  inverse of the build's H, p' = B^-1(q'), and p' is scored through the truth camera by the same path. A correction
  in pixel space gives p' directly, which is scored the same way. No correction changes the truth.

The measurement under the build's H:

- On frames whose sidecar row is accepted with an H, every box foot pixel p goes to q = B(p). The build's camera comes
  from sportvu.camera.pose_from_h(B), with ground point G_B.
- For each in-frame player, the nearest q is taken, re-matched under the build's H. If it lies within 5 ft, the
  player enters the offset population with offset o = q - SportVU. Depth is o along the unit vector from G_B to q,
  positive when the foot lands beyond the player, away from the camera. Lateral is o across it.
- The offset population holds hits and misses alike: every in-frame player with a foot within 5 ft under the build's
  H.
- Cells: the box's own pixel height (under 150, 150 to 200, 200 to 250, 250 and over) by the third of q, using
  tools.p8_scorecard.third_of with the near side of the build's H. Report only: the distance from G_B to q, in four
  bins at the quartiles of the pool's offset population, and medians by third and by height alone.
- Reported per cell: the median depth and lateral offset per game and pooled, the game spread (largest minus smallest
  game median) and the counts.

Consistency (fixed now):

- A cell is consistent when:
  - it holds at least 200 offset rows on every pool game;
  - the three game medians of depth share the sign of the pooled median;
  - the game spread is at most half the absolute pooled median.
- The bias is consistent when the consistent cells hold at least 60 percent of the pool's offset population.
- Why: if the game spread is at most half the bias, a shift fitted on two games misses the third game's bias by at
  most about half of it, so it still removes about half. The 60 percent asks that this hold for most players, not
  for a niche.

Corrections, each fitted leave one game out on the other two pool games' offset populations and applied to every box
of the held-out game:

- K1, depth shift by cell. The fit games' pooled median depth of the cell is removed along the unit vector from G_B to
  q. A cell with fewer than 50 fit rows gets no shift. The result is scored through the truth camera as above.
  Frames without an accepted H or a camera pose keep their feet.
- K2, vertical pixel offset by box height. For each fit row, the target pixel is the SportVU position sent into the
  image through the inverse of the build's H. The fit games' median of (target y minus foot y) over box height is
  taken per height bin. The correction moves the foot to p'_y = p_y + ratio times box height, with x unchanged, and
  p' is scored through the truth camera.

Worth registering a correction arm (fixed now), all of:

1. The bias is consistent, as above.
2. Some correction, K1 or K2, has net change in missed of at most -0.010 on every pool game held out. Missed is
   counted over all in-frame players and scored through the truth camera.
3. That same correction has a non-referee ghost rate change of at most 0 on every pool game (point estimates).

Why the bars: -0.010 on every game is the missed bar of the adoption rule an arm would face. "Does not raise ghosts"
is taken literally, as no rise on the point estimate. Window bootstrap intervals (2,000 resamples, seed 0, windows
resampled whole) are reported beside the point estimates.

If these hold, the next step is a correction arm with its own registration and review. It is scored as above on the
pool and on hou_cle_2016, with the other rows (position, near-field bias, team) as non-inferiority rows. If they do
not hold, the short-foot bias is recorded as not correctable by a fixed shift under the build's H, and the placement
row stays the measured floor of the step 3 row split.

## Forking paths, named now

- Step 1's numbers have been seen. A depth shift by third, fitted through the truth camera, lowered missed by -0.0123
  and -0.0146 on gsw_bkn_2015 and gsw_cha_2016, and raised it by +0.0041 on gsw_sac_2015.
- Step 1b's numbers have been seen: the middle and far short-foot bias under both geometries, and the near-third
  disagreement on gsw_cha_2016 and hou_cle_2016.
- hou_cle_2016's -0.021 from step 1's pooled correction has been seen. Here hou_cle_2016 is report only again.
- The two correction forms (K1, K2) were chosen after those results. K1 is step 1's form moved into the build's
  frame with a finer cell; K2 is the pixel-space form the lead asked for.

## Results

Not computed yet.
