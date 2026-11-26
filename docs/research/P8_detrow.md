# P8 detection row: causes of the far misses (registered before any number)

Written 2026-11-26 after the output-position line closed (three pool looks, STOP; hou_cle_2016 unspent for that
arm). This is report only: no model runs, nothing is built, trained or changed, and no unit is spent. Tool:
tools/p8_detrow.py, which carries the same rules in its docstring. Record: reports/research/p8_detrow.json. This
document and the tool were committed before the tool produced any number.

## Why

P8_placement.md step 3 split the missed row in two. The placement part (a foot 2.5 to 5 ft off) is set by geometry and
truth. The detection part, players with no foot within 5 ft under the truth camera, is the part a detector or the
tracking could move. It is 0.052 to 0.091 of in-frame players on the pool and 0.063 on hou_cle_2016. Its causes decide
which fix is worth designing.

## Registered definitions

Data:

- Games: the pool (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015). hou_cle_2016 is reported apart as a check and chooses
  nothing.
- Guards run on every game, SportVU id and window. No held-out game, no D85 test game, no hand labelling.
- Inputs are stored P8-workspace files only, so no lock is needed: the reference-platform r23 sidecars through the
  scorer's universe (tools.p8_footbias.load_frames, with its count check against the registered score files), M4's raw
  CPU r23 boxes on the corpus JPEGs, and the E1 corpus teacher preds.
- hou_cle_2016 has no corpus frames, so its raw and teacher classes are not measured.

Population:

- In-frame players (the scorer's 10 px margin through the truth camera) whose nearest foot_stab, sent through the
  truth camera as the scorer maps it, lies more than 5 ft away, or who have no box at all.
- Check: per game, the count must equal step 1's far misses (reports/research/p8_placement.json), or the tool stops.

Expected box of a player:

- From the truth camera's pinhole model (tools.p8_e1_postmortem.foot_and_height).
- Bottom centre at the projected foot, the height of a 6.5 ft player, width 0.4 of the height.
- A frame without a camera pose has no expected box.

Cause classes, mutually exclusive. A player gets the first class that holds:

1. Frame edge: less than 90 percent of the expected box's area lies inside the image.
2. Occluded or overlapping: the expected box's largest IoU with another in-image player's expected box is at least 0.3.
3. Small: expected height under 150 px, or the player is in the far camera third.
4. Raw had a box: on a corpus frame, some M4 r23 raw box at the operating point (0.40) has its bottom-centre foot,
   through the truth camera, within 5 ft. The build then lost it, through tracking, gating or stabilisation.
5. Teacher only: on a corpus frame, some YOLOX-X box at 0.1471 has its foot within 5 ft, and no r23 raw box at 0.40
   does.
6. None: on a corpus frame, none of the above.
7. Not measured: past classes 1 to 3 on a frame without corpus raw boxes (every hou_cle_2016 frame and any pool frame
   outside the corpus).

Why these thresholds:

- 90 percent inside is an edge cut that removes at least a tenth of the body, such as the head or the feet.
- IoU 0.3 means about half of a player's box lies behind another player at equal sizes.
- 150 px is the small bin used since the E1 post-mortem (P8_E1B.md section 2).
- 5 ft is the far-miss gate itself.

Measures:

- Per class, per game and pooled: the count, the share of far misses, and the rate among in-frame players.
- Window intervals: 2,000 resamples, seed 0, windows resampled whole within each game.
- Report only:
  - each attribute's prevalence among far misses without the ordering (edge, occluded, small, raw had a box, teacher
    reaches, r23 raw between 0.10 and 0.40 within 5 ft);
  - the part of classes 5 and 6 that an r23 raw box between 0.10 and 0.40 reaches, which a threshold change could
    reach.

## Reading rule (fixed now)

A class matters when its rate among in-frame players is at least 0.010 on every pool game. That is the adoption bar
for a missed gain: a class smaller than it cannot deliver the bar even if fixed completely.

What each outcome leads to:

- Class 5 (teacher only) matters: a detector change could move it. The next design is a presence-only label recipe,
  with E1b's transfer evidence (presence gains keep 0.36 to 0.62 at build level, P8_E1B.md section 2b) built into its
  upper bound.
- Class 4 (raw had a box) matters: only tracking or gating could move it. The next step is a stored-files measurement
  of where those boxes are lost (the tracker's activation and gating, the gate v2 frame drop, or foot stabilisation).
- Classes 1 to 3 matter: hard geometry that no measured source reaches. They are reported as the floor of the
  detection row, and no fix is proposed from this measurement.
- Class 6 (none) matters: a cause no measured source reaches, such as truth or timing error, or players no model
  boxes. The next step would be a truth-quality check on those player-frames.
- Several classes can matter at once. All are reported.

## Results (computed 2026-11-26, after the definitions above were committed in 08d330f3)

Stored files only, no lock, about 12 seconds. Both checks passed: load_frames' count check against the registered
score files, and the far-miss counts equal step 1's on every game. Every scored pool frame is a corpus frame, so no
pool far miss is "not measured". Record: reports/research/p8_detrow.json.

Far misses: 1,871 of 20,457 in-frame players (0.091) on gsw_bkn_2015; 998 of 13,284 (0.075) on gsw_cha_2016; 1,104 of
21,144 (0.052) on gsw_sac_2015; 2,727 of 43,030 (0.063) on hou_cle_2016.

Ordered classes (rate among in-frame players, window interval; share of far misses):

| Class | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 | Pooled share | hou_cle_2016 (check) |
| --- | --- | --- | --- | --- | --- |
| 1 edge | 0.0005; 0.6% | 0.0005; 0.6% | 0.0010; 1.9% | 1.0% | 0.0018; 2.8% |
| 2 occluded | 0.0255 [0.0178, 0.0345]; 27.8% | 0.0207 [0.0161, 0.0272]; 27.6% | 0.0157 [0.0124, 0.0201]; 30.0% | 28.4% | 0.0151; 23.8% |
| 3 small | 0.0311 [0.0198, 0.0446]; 34.0% | 0.0277 [0.0226, 0.0338]; 36.9% | 0.0181 [0.0132, 0.0244]; 34.6% | 34.9% | 0.0232; 36.6% |
| 4 raw had a box | 0.0171 [0.0037, 0.0420]; 18.7% | 0.0072 [0.0034, 0.0136]; 9.5% | 0.0072 [0.0034, 0.0133]; 13.9% | 15.0% | not measured |
| 5 teacher only | 0.0092 [0.0066, 0.0127]; 10.0% | 0.0062 [0.0042, 0.0085]; 8.2% | 0.0049 [0.0021, 0.0100]; 9.3% | 9.4% | not measured |
| 6 none | 0.0081 [0.0059, 0.0110]; 8.8% | 0.0130 [0.0098, 0.0169]; 17.2% | 0.0054 [0.0034, 0.0080]; 10.3% | 11.4% | not measured |
| 7 not measured | 0 | 0 | 0 | 0 | 0.0233; 36.8% |

Report only, each attribute's prevalence among far misses without the ordering (bkn, cha, sac; pooled):

| Attribute | Prevalence |
| --- | --- |
| edge | 0.6, 0.6, 1.9 percent; 1.0 percent |
| occluded | 27.8, 27.6, 30.0 percent; 28.4 percent |
| small | 53.2, 56.2, 52.3 percent; 53.7 percent |
| r23 raw had a box at 0.40 within 5 ft | 42.0, 26.2, 39.0 percent; 37.2 percent |
| teacher reaches within 5 ft | 78.4, 56.3, 70.6 percent; 70.7 percent |
| r23 raw box at 0.10 to 0.40 within 5 ft | 4.4, 5.4, 6.8 percent; 5.3 percent |

Threshold-reachable parts of classes 5 and 6 (an r23 raw box between 0.10 and 0.40 within 5 ft): 29 and 2 players
pooled.

## Reading by the registered rule

Two classes matter: 2 occluded (0.016 to 0.025 of in-frame players) and 3 small (0.018 to 0.031), on every pool game.
Class 4 (raw had a box) reaches 0.010 on gsw_bkn_2015 only (0.017; 0.007 on the other two), so it does not matter
by the rule. Classes 5 (teacher only) and 6
(none) stay under 0.010 on at least one game. By the rule, the detection row is mostly hard geometry: close to two
thirds of the far misses (63 percent pooled) are overlapping or small or far-third players. No fix is proposed from
this measurement, and those classes are reported as the floor of the row. Neither a detector change (class 5) nor
tracking or gating (class 4) reaches the bar on every game in the ordered breakdown.

What the numbers say beyond the rule (post hoc, not tested; shares and the hou_cle_2016 column above are point
estimates unless an interval is shown, the share intervals are in the record):

- The ordering hides a large build-side loss. Without the ordering, r23's own raw detector at its operating point
  had a box within 5 ft for 37 percent of far misses (point estimates, not bootstrapped: rate 0.038, 0.020 and 0.020
  of in-frame players). These are upper bounds on any missed gain: only a recovered foot within 2.5 ft on the player
  moves the missed row (a recovered box 2.5 to 5 ft away becomes a placement miss), and that share is not measured. Most of these players are also small or overlapping, so the ordered rule files them under geometry. For a
  far miss every kept build foot_stab lies beyond 5 ft, so the build either lost the box or placed its foot_stab
  outside 5 ft while the raw bottom centre is inside; the second case needs a shift of only 5 ft minus the raw
  distance, which this record does not measure. The raw run is on the corpus JPEGs and the build on decoded video, so decode differences are mixed in ; the
  decode share is not measured here (M6's 6.9 percent is the overall raw-only share on gsw_bkn_2015, build losses and
  decode together, per box, not per far miss).
- The teacher reaches 71 percent of far misses within 5 ft, but the 5 ft gate is loose in crowds, and 9.4 percent are
  teacher-only in the ordered breakdown. A detector change aimed at presence has a small registered target, in line
  with P8_E1B.md (presence 0.016 of in-frame players at raw level).
- Lowering r23's threshold reaches almost nothing: 5.3 percent of far misses have a box between 0.10 and 0.40, and 31
  players are in classes 5 and 6.

Proposal for the lead (it needs its own registration and names this result as seen): a stored-files measurement of
where the build loses r23's raw boxes for far-miss players. On corpus frames, it would match M4's raw boxes to sidecar
boxes, splitting tracker activation (no track started), track loss, gate v2 drops, and foot stabilisation moving the
foot beyond 5 ft. That measurement costs minutes and no units. Whether to register it, given that the ordered rule did
not select class 4, is the lead's decision.
