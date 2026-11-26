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

## Results

Not computed yet.
