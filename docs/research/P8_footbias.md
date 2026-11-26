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

The measurement under the build's H. Here B is the r23 court model's per-frame H, which the CPU and MPS r23 builds
share; it is not the H the camera truth started from (see the correction below).

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

## Results (computed 2026-11-26, after the definitions above were committed in c7452134)

Stored files only, no lock, about 13 seconds. The count check passed on every game: in-frame, missed, non-referee
ghost and non-referee box counts equal the registered score files (so the baseline missed rates are the registered
0.2525, 0.2952, 0.2060 and 0.1961, and the ghost rates 0.1255, 0.1876, 0.1144). Every scored frame had an accepted
build H and a camera pose. Offset population (in-frame players with a foot within 5 ft under the build's H): 18,618,
12,534, 20,046 and 39,934 (bkn, cha, sac, hou_cle_2016). Record: reports/research/p8_footbias.json.

The bias under the build's H (median depth, ft; negative: the foot lands short of the player, toward the camera):

| Game | Near, middle, far third | Box height under 150, 150 to 200, 200 to 250 px | Camera distance quartiles (under 98.7, to 107.2, to 116.3, over 116.3 ft) |
| --- | --- | --- | --- |
| gsw_bkn_2015 | -0.29, -0.86, -1.34 | -1.19, -0.77, -0.51 | -0.25, -0.73, -1.12, -1.38 |
| gsw_cha_2016 | -0.48, -0.94, -0.98 | -0.98, -0.84, -0.41 | -0.32, -0.84, -1.02, -0.97 |
| gsw_sac_2015 | -0.41, -0.84, -1.04 | -0.77, -0.78, -0.61 | -0.32, -0.74, -0.87, -1.02 |
| hou_cle_2016 (check) | -0.71, -0.96, -0.84 | -0.77, -0.90, -0.78 | -0.74, -0.98, -1.01, -0.76 |

Cells (box height by third) with at least 200 rows on every pool game, game medians (bkn, cha, sac) and spread:

| Cell | Game medians | Pooled | Spread | Consistent |
| --- | --- | --- | --- | --- |
| 150 to 200, far | -1.30, -1.00, -1.15 | -1.16 | 0.30 | yes |
| 150 to 200, middle | -0.83, -0.94, -0.86 | -0.87 | 0.11 | yes |
| 150 to 200, near | -0.31, -0.55, -0.42 | -0.41 | 0.24 | no (spread above half) |
| 200 to 250, middle | -0.95, -0.61, -0.93 | -0.87 | 0.34 | yes |
| 200 to 250, near | -0.24, -0.28, -0.45 | -0.34 | 0.21 | no |
| under 150, far | -1.40, -0.96, -0.91 | -1.10 | 0.49 | yes |
| under 150, middle | -0.94, -1.06, -0.67 | -0.88 | 0.39 | yes |
| under 150, near | -0.34, -0.33, -0.12 | -0.21 | 0.22 | no |

Lateral medians are within plus or minus 0.3 ft in every populated cell. The consistent cells hold 69.5 percent of
the pool's offset population, so the bias is consistent by the registered rule (60 percent).

Leave-one-game-out corrections, scored through the truth camera (net change in missed over all in-frame players
[window interval]; change in the non-referee ghost rate [window interval]):

| Correction | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 |
| --- | --- | --- | --- |
| K1, depth by cell (build's frame) | -0.0045 [-0.0110, +0.0021]; +0.0009 [-0.0024, +0.0045] | -0.0076 [-0.0165, +0.0014]; -0.0032 [-0.0109, +0.0035] | +0.0084 [-0.0002, +0.0165]; +0.0083 [+0.0031, +0.0134] |
| K2, foot pixel by box height | -0.0029 [-0.0090, +0.0034]; +0.0021 [-0.0014, +0.0060] | -0.0030 [-0.0115, +0.0051]; +0.0003 [-0.0063, +0.0066] | +0.0081 [-0.0006, +0.0161]; +0.0079 [+0.0033, +0.0122] |

Fitted shifts: K1 removes 0.15 to 0.61 ft in the near third and 0.66 to 1.23 ft in the middle and far thirds; K2 moves
the foot pixel up by 1.7 to 4.5 percent of the box height (the fitted ratios are negative).

Check game, report only (fitted on the whole pool): hou_cle_2016 K1 -0.0087 [-0.0141, -0.0037], ghosts -0.0048; K2
-0.0060, ghosts -0.0033.

## Reading by the registered rule

Not worth registering a correction arm. Condition 1 holds: the bias is consistent (69.5 percent of the offset
population in consistent cells). Conditions 2 and 3 fail for both corrections: no game reaches -0.010, gsw_sac_2015
gets worse on missed (+0.008) and on ghosts (+0.008, interval above zero), and gsw_bkn_2015's ghosts rise for both.

Why a consistent bias does not lower the missed row (post hoc, consistent with step 1b, not tested here): the bias is
measured in the r23 court's frame, but missed is scored through the truth camera. The truth camera was initialised
from, and fitted to, the V3 label build (same detector and foot pixels, V3 court model; sportvu/camera.py line 169,
tools/p8_frozen.py line 495), so it already absorbs part of the same short-foot bias (step 1b: hits' median depth
-0.2 to -0.9 ft under the truth against -0.8 to -1.4 ft under the r23 court's H). Removing the build-frame bias therefore over-corrects in the truth frame, by about
the depth of -d_H (step 1b: +0.3 to +0.8 ft for hits' median). On gsw_sac_2015 it pushes found players out. The
size of -d_H alone does not explain which game loses: gsw_cha_2016 has the largest -d_H and still gains a little, so
the per-player spread of -d_H, not only its median, likely decides it (not measured).

So the short-foot bias is recorded as not correctable for the missed row by a fixed shift under the build's H, and
the placement row stays the measured floor of step 3's row split.

## What this points to (proposals, each needing its own registration)

- The scorer's position rows (position p50 and p90, near-field bias) map the foot through the build's H, not the
  truth camera. A bias that is consistent in the build's frame is exactly what those rows see. A K1-style depth
  correction could therefore be a position-row arm, with missed and ghosts as non-inferiority rows (missed would need
  a margin, given the +0.008 on gsw_sac_2015 here). Before any registration: a stored-files measurement, registered
  first, of what K1 fitted leave one game out does to the position rows as the scorer computes them. It takes
  minutes, needs no lock and costs no units. Its numbers were not computed here, so that this proposal stays
  untested.
- Forking paths for such an arm: this document's numbers, steps 1 and 1b, and hou_cle_2016's gains (-0.021 in step
  1, -0.009 here) are all seen.

## Correction (2026-11-26, from the ml-review of step 1b)

Step 1b and the first draft of this document called B "the selecting build's H", the one the camera truth started
from. That was wrong. The camera truth was initialised from, and fitted to, the V3 label build's H
(sportvu/camera.py line 169, pose_from_h(fr["H_v3"]); tools/p8_frozen.py line 495 reads P["label_build"]): the V3
court model. B here is the r23 court model's H, shared by the CPU and MPS r23 builds; on gsw_cha_2016 s00_w00 and
s00_w01 the label build's H matches builds/r23's H on no frame. So d_H = B(p) - T(p) is the sum of a court-model
part (r23 court against V3 court) and a fit part (V3 start against the fitted truth). The measurements and the
reading above are unchanged; only the description of B is.

## Decomposition of d_H (registered before any of its numbers)

Stored files only (the label build sidecars, label_build/<window>_frames.json, read beside step 1's inputs). Report
only: it changes no rule and no reading. Tool: python3 -m tools.p8_footbias decompose; record:
reports/research/p8_footbias_decompose.json.

- Player-frames: step 1b's population on frames where the r23 sidecar row is accepted with an H, the label build row
  has an H, and the build's camera has a pose. p is the foot_stab pixel of the player's nearest box under the truth
  camera; the player enters when that box lies within 5 ft (hits and placement misses).
- Three maps of the same pixel: B(p) (r23 court), V(p) (V3 label build), T(p) (truth). d_H = B(p) - T(p) = c + f, with
  the court-model part c = B(p) - V(p) and the fit part f = V(p) - T(p).
- Convention check first, because the stored V3 H may use another court convention: V(p) is taken in each of four
  conventions (as stored; x mirrored to 94 - x; y mirrored to 50 - y; both), and the convention with the smallest
  median |V(p) - T(p)| over all pool player-frames is used. If that median is 3 ft or more, no convention is trusted and
  only the medians of each convention are reported, not the decomposition. The chosen convention and every
  convention's median are recorded.
- Reported per game (pool, and hou_cle_2016 apart) and camera third, for hits and placement misses: the median depth
  of d_H, c and f (depth along the unit vector from the build camera's ground point to the SportVU position, positive
  away from the camera), the median |d_H|, |c| and |f|, and the share of |d_H|'s squared size carried by c and by f
  (sum of c.d_H over the sum of |d_H|^2, and the same for f; the two shares add to 1). Frames left out are counted.

### Decomposition result (computed 2026-11-26, after its registration was committed)

Stored files only, about 14 seconds, no lock. Every scored player-frame had a label-build H, an r23-court H and a
camera pose (none left out); 18,586, 12,286, 20,040 and 40,303 player-frames (bkn, cha, sac, hou_cle_2016).

Convention check, median |V(p) - T(p)| over the pool's player-frames: as stored 6.92 ft; x mirrored 56.74 ft; y
mirrored 19.27 ft; both mirrored 60.97 ft. The best (as stored) is above the 3 ft bar, so by the registered rule no
convention is trusted and the decomposition into court and fit parts is not reported. Record:
reports/research/p8_footbias_decompose.json.

What this leaves open: the stored orientation is clearly the closest of the four, so the V3 H is not mirrored against
the truth. But a 6.9 ft median gap at the feet could be either a real disagreement of the V3 court model with the
fitted truth (camera.py's docstring puts V3's H 20 to 40 px off, which is several feet at the far side), or a
convention the check did not list (an origin or scale difference). Telling them apart needs its own registered
check, for example an affine fit of V(p) to T(p) over the pool, with its residual. It was not run.
