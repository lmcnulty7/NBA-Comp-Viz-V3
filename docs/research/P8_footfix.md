# P8: a foot-model correction on the decided truths (design, version 1, registered before any number)

Version 1, 2026-11-27, written under the ml-training-rigor protocol before any compute: it fits a correction, so it is
treated as training. Design only; nothing has been fitted or scored for it. Stored files only, on the Mac, no Colab
unit. No held-out game, no D85 test game and no test arena is read (CLAUDE.md rule 4). hou_cle_2016 stays unspent
until stage 2, a separate step that runs only on a pool pass and the lead's go. It goes to the training-design-reviewer
before any code runs; after that the design changes only by a dated version.

## 1. Question and mechanism

**Question.** On the decided truths (D199: multipool v1 on gsw_cha_2016 and gsw_bkn_2015), does a fixed correction of
the box's stabilised foot pixel (foot_stab), fitted on one pool game and applied to the other, lower the missed row of
D202 by at least 0.010 on each game, without making ghosts, position p90, near-field bias, near-third p90 or team worse
than fixed margins?

**Target row.** D205's P-ii: an in-frame player with no box within 2.5 ft after the scorer's one-to-one assignment, and
a box no one claims 2.5 to 5 ft away (median 3.20 and 3.21 ft). 0.1141 [0.1060, 0.1231] of in-frame players pooled,
0.734 of the placement misses.

**Mechanism.** foot_stab is a stabilised box-bottom point, not the ground contact point. The gap between the two (box
bottom below the feet or at the trailing foot of a stride, a cut box, a player in the air) is an image offset that
plausibly scales with the box's size. Through a homography, a fixed vertical pixel offset becomes a court error along
the camera ray that grows with distance: the D184 and D186 pattern (short of the player, growing with depth). Moving
the foot pixel by the right amount for its box size and depth would bring part of P-ii's boxes inside 2.5 ft.

**What changed since D186 to D188.** Those arms fitted toward SportVU sent through the build's own court H (the r23
court model's per-frame H, here B) and were scored on the old truths. The missed row maps foot_stab through the truth
camera's H (here T), not B. The decided truths moved most of the missed change into placement (D202: missed -0.083 on
gsw_cha_2016 and -0.033 on gsw_bkn_2015 from the truth change alone, mostly placement). And the multipool truth camera
was fitted to these same foot_stab pixels (P8_multistart_fit.md; D205's limits), so the part of the foot bias that a
single camera can absorb is already absorbed by T. What is left for the missed row is a residual by depth and box size.
D204's C1 description shows its shape on placement misses: along the ray, +1.15 and +1.30 ft (beyond the player) in the
near third, -2.12 and -1.90 ft in the middle third, -1.91 and -2.46 ft in the far third (gsw_cha_2016, gsw_bkn_2015).
This design fits toward the decided truth's projection of the player, which is what the missed row scores.

**The risk named now.** Because T was fitted to these feet, the residual is partly T's own fitting error. A correction
fitted toward T can copy that error, lowering missed without placing feet nearer the real positions. The position rows
are the independent check: the scorer maps the same foot_stab through B for p50, p90 and near-field bias, and B comes from
the court model and the frame's lines, with no SportVU. A correction that helps missed through T and hurts the position rows through B is
read as copying T's error. In the near third the two frames disagree in sign on placement misses (T: beyond; B: short,
D186), so near-field bias and near-third p90 are the rows most at risk (section 9 forecast).

## 2. Known before this item (stated as known, not retested)

- D184: placement offset along the camera ray, median depth -1.1 to -1.9 ft, depth falling from near to far; no timing
  signature; no leave-one-game-out correction passed on every game (depth by third: -0.0123, -0.0146, +0.0041 on
  gsw_sac_2015). SCATTERED by its rule.
- D186: under B the foot lands short by 0.3 to 1.4 ft, growing with depth, consistent across games. Leave one game out,
  scored through the (old) truth camera: K1 (depth shift by box height and third, in B's frame) -0.0045, -0.0076,
  +0.0084; K2 (vertical pixel offset by box height, target SportVU through B's inverse) -0.0029, -0.0030, +0.0081;
  ghosts +0.008 on gsw_sac_2015 with the interval above zero. No arm for missed.
- D187: K1 lowers position p50 by 0.25 to 0.28 ft and near-field bias by 0.94 to 1.09 ft on every pool game, but fails
  non-inferiority on detection (gsw_sac_2015 missed upper end +0.0165, ghosts +0.0134, against +0.010 margins).
- D188: shifting only the output court positions (K1, K0, K1m, K1t) closed with STOP after three pool looks: every form
  raised near-third p90 beyond +0.05 ft on every game. Lead decisions carried here: near-third tail growth counts as harm;
  no fourth form is tuned on the pool; no margin is widened after a look.
- D202's limits: truth frames are selected by the same r23 boxes, so the rows are optimistic and lean toward r23;
  multipool is a better fit, not a truth shown to be better.
- D204: placement misses are carried by C3a's shared-box clause (one box nearest to two players), attributable
  fraction 0.232 pooled; box height or foot move (C4) 0.126. A foot shift cannot give two players one box.
- E1b ledger lesson (2026-11-26): placement gains keep about 0.3 at build level.

What is new here, and the only reason to look again: the target frame (the decided truth's projection, not B), the
decided truths themselves, and the one-to-one pairing of fit rows (section 4), which keeps a neighbour's box out of the
fit. If none of these matters, stage 0 says so in minutes.

## 3. Upper bound (from D205's rows)

If every P-ii player's unclaimed box moved inside 2.5 ft and no other box moved out, missed would fall by P-ii:

| | missed (D202) | P-ii (D205) | missed at the ceiling |
|---|---|---|---|
| gsw_cha_2016 | 0.2085 [0.1913, 0.2280] | 0.1178 [0.1045, 0.1310] | 0.0907 |
| gsw_bkn_2015 | 0.2200 [0.2053, 0.2365] | 0.1105 [0.1009, 0.1214] | 0.1095 |
| pooled | 0.2143 [0.2026, 0.2263] | 0.1141 [0.1060, 0.1231] | 0.1002 |

This is a ceiling, not a forecast, for four reasons:
1. A fixed correction moves every box. Hits near the 2.5 ft edge can move out, so the net is smaller than the P-ii rows
   moved in.
2. P-ii's box is only "unclaimed": it may be a ghost (D205 post hoc (b)). Moving a ghost onto a player lowers the
   nearest-foot missed row without finding anyone. Guard: the one-to-one missed row and the ghost row (section 6).
3. P-i (0.0405) and the detection classes are outside the target: their nearest box is a neighbour's, and a foot shift
   reaches them only by moving a neighbour's box, a gain the one-to-one row would not confirm.
4. A median shift by cell removes only the systematic part of each offset; P-ii's median distance is 3.2 ft, so half
   of the P-ii rows need more than 0.7 ft removed in the right direction.

The ceiling (0.11) is ten times the bar (0.010), so the bound does not stop the item. The reachable part for these
forms is measured first, in sample, by stage 0's falsifier.

## 4. The candidate corrections (two, fixed now)

Both use only quantities the pipeline has at run time (the box, its foot pixel, B), so a passing form could be applied
without a truth. Both are fitted on the fit game's rows only, by medians, with D186's cells and its minimum of 50 fit
rows per cell (a cell with fewer gets no correction). Nothing is tuned: no bin edge, minimum or form is chosen on these
games' scores.

**Fit rows (shared by both forms).** On the fit game's scored frames (the pinned scorer's universe: live, wide,
accepted camera truth; in-frame players at the 10 px margin), pair in-frame players with non-referee boxes by D205's
two-stage one-to-one assignment under T (tools.p8_one2one.two_stage with assign_1to1: stage A at 2.5 ft, stage B on
the leftovers at 5 ft). Each pair (box k, player j) is one fit row: hits (stage A) and P-ii (stage B) alike, as D186's
population held hits and misses. Referee boxes are those of the scorer's rule (IoU at least 0.5 to a cached referee
detection). The fit target of player j is t_j = T^-1(SportVU_j), the pixel where the decided truth images the player.

**F1, depth shift by cell toward the truth's projection, applied in B's frame.** For a fit row on a frame with an
accepted B and a camera pose from sportvu.camera.pose_from_h(B) with ground point G_B: q = B(p), u = unit vector from
G_B to q (tools.p8_placement.depth_basis), depth d = (q - B(t_j)) . u. Cell = the box's pixel height (under 150, 150 to
200, 200 to 250, 250 and over) by the third of q (tools.p8_scorecard.third_of with B's near side): 12 cells. s_cell = the
median d of the cell's fit rows. Applied to every box of a frame with an accepted B and pose: q' = q - s_cell u, p' =
B^-1(q'). Other frames keep their feet. This is D186's K1 with the target moved from SportVU under B to the truth's
image of the player.

**F2, vertical pixel offset scaled by box height.** For each fit row: r = (t_j,y - p_y) / h, with h the box's pixel
height. r_bin = the median r per height bin (the four bins above). Applied to every box: p'_y = p_y + r_bin h, p'_x =
p_x. This is D186's K2 with the target moved to the truth's image of the player.

**Folds.** Leave one game out over the two pool games: gsw_cha_2016 is scored with tables fitted on gsw_bkn_2015, and
the reverse. gsw_sac_2015 is not used (apart on check C) and gsw_ind_2016 has no truth (D199). For stage 2 only, the
chosen form is refitted on both pool games pooled.

**Choice between forms** (only if both pass stage 1): the one with the lower pooled missed change; if the two are
within 0.003 (below the noise floor), F2, which has 4 parameters to F1's 12. Fixed now, so the choice is not a fork.

**Operating point.** No threshold is trained. The fixed constants (bins, 50 rows, median, the 2.5 and 5 ft stages) are
D186's and D205's, set before these numbers.

## 5. Decision metric and scoring

**Scorer.** The pinned tools.p8_scorecard.score_window (truth mode camera, foot_stab) through tools.p8_resetfix.
score_pair, as D202 ran it: side "r23" is D202's composite builds/r23 in the reference root; side "new" is a corrected
copy of it. score_pair already asserts that both sides have the same in-frame players. Rows by tools.p8_e1_score.
window_stats and tools.p8_e1_score.rows, plus score_pair's extra counts (one-to-one missed, unmatched, duplicates).
Truths: work_ms labels for both pool games, pinned by D202's SHA-256 labels pins; a changed file stops the run.

**The corrected build.** A folder holding, per window, frames.json rewritten with each box's foot_stab replaced by p'
and every other field equal; every other file of the window byte-copied. The tool asserts, per window, that the box
count, order, tids, bboxes and H rows are unchanged, that only foot_stab differs, and that the non-frames files hash
equal to builds/r23's. No truth, frame selection, mirror, time map or cache changes, so the in-frame players are the
same by construction.

**Rows, each as the change (corrected minus baseline), paired on the same windows,** with a window-cluster bootstrap
(2,000 resamples, seed 0, tools.p8_e1_score.boot's indices; each resample recomputes both sides' rows on the drawn
windows): missed (D202's row; with placement and detection beside it), ghosts per box (non-referee), net error,
one-to-one missed, one-to-one net error, position p50 and p90, near-field bias (tools.p8_e1_score._bias: the larger of
the absolute median toward-camera errors of the near and far thirds), near-third p90 (as tools.p8_outpos.near_p90_boot,
matched in both arms, SportVU third), team, and counts (windows, in-frame players, boxes, matched players in each arm and
in both). Per game and pooled (each game under its own fold's table). The rule reads per-game rows only.

## 6. Adoption rule (stage 1, the pilot gate), written now

A form passes on a held-out pool game when all of these hold on that game:

1. Missed: change at most -0.010, and the upper end of its interval below 0.
2. Ghosts per box: upper end of the change at most +0.010.
3. Near-field bias: upper end of the change at most +0.10 ft.
4. Position p90: upper end of the change at most +0.10 ft.
5. Near-third p90: upper end of the change at most +0.05 ft (D188's margin, unchanged).
6. Team: lower end of the change at least -0.010.
7. Denominator checks (point estimates): one-to-one missed change below 0, and net error change below 0. In-frame
   players and boxes equal in both arms (by construction; a difference stops the run).

**Stage 1 PASS:** some form passes on both gsw_cha_2016 and gsw_bkn_2015. Then the choice of section 4, and a request
to the lead for stage 2. **FAIL:** no form passes on both games. **No "inconclusive" route to more pool looks:** a FAIL
closes the arm on these truths.

Why these bars. Missed -0.010 on every game is the bar every correction arm has faced since D186 (a tenth of the
ceiling). Ghosts +0.010 is D187's margin. The 0.10 ft position margins are a little above the paired half-widths
D187 measured (p90 0.072 to 0.078 ft; near-field bias 0.022 to 0.090 ft), so a form with no effect on them passes with
reasonable probability, while a form that moves them by a tenth of a foot fails. Team's paired noise has never been
measured; -0.010 matches the rate rows' margin and is fixed now, not after a look. Row 7 is the denominator guard of
section 3 (2): a gain made by moving ghosts onto players shows as missed falling while one-to-one missed does not.

## 7. Noise floor

The rule reads paired changes on identical windows and players, so the floor is the paired interval, not D202's
per-game intervals (missed about plus or minus 0.016 to 0.019, which describe level, not change). The closest measured
paired intervals are D186 and D187's leave-one-game-out corrections on the same scorer:

| change (half-width of the window interval) | gsw_bkn_2015 | gsw_cha_2016 |
|---|---|---|
| missed (K1, D186) | 0.0065 | 0.0090 |
| ghosts per box (K1) | 0.0035 | 0.0072 |
| position p90 (K1, D187) | 0.078 ft | 0.072 ft |
| near-field bias (K1, D187) | 0.022 ft | 0.090 ft |
| position p50 (K1, D187) | 0.037 ft | 0.044 ft |

Those were on the old truths with fewer gsw_cha_2016 frames (13,284 in-frame players against 28,427 now), so the new
intervals should be no wider. The smallest missed change whose upper end clears 0 is therefore about 0.007 to 0.009:
the -0.010 point bar is the binding condition, sitting just above the floor. Two games from one arena and one
broadcaster are a leave-one-game-out, not a leave-one-arena-out; hou_cle_2016 (another arena, another truth fit) is the
only arena transfer test (section 8).

**Expected effect, with reasoning.** -0.005 to -0.020 per game. The low end is D186's K1 and K2 on the old truths. The
high end is a guess, not a computation: it assumes the new target frame removes a further part of P-ii's 2.5 to 3.0 ft
edge, whose size no record holds. The range straddles the floor, which is why stage 0's falsifier runs first and can
stop the arm within minutes of lock time.

## 8. The ladder, its gates and the kill rules

**Pre-flight (no lock, before stage 0).**
- Code tests on synthetic frames: a planted depth shift per cell and a planted pixel ratio per bin are recovered by the
  fits and removed by the corrections; the identity table leaves every foot byte-equal; a fit called for one game never
  reads a row of the other (asserted on the rows' game ids); hou_cle_2016 is refused by stages 0 and 1 in code; the
  corrected-build check rejects a planted change to any field other than foot_stab.
- Guards on every game, SportVU id and window: tools.p8_common.guard and tools.research_testset.refuse_cv_test.
- Scorer pin test passes; tools/p8_scorecard.py and tools/p8_resetfix.py unchanged (their hashes recorded).

**Stage 0 (one lock, "p8 footfix s0", frozen CV python), smoke, reproduction, fit audit and the falsifier.**
- 0a Smoke: one window per pool game through the whole path (identity and F2), scored. Gate: identity equals D202's
  record on that window for every field of tools.p8_rebaseline.GATE_FIELDS; F2's corrected build passes the copy check.
- 0b Reproduction: the identity correction on every window of both pool games. Gate: all twelve GATE_FIELDS equal
  D202's records (score__<game>__current.json) on every window; else stop.
- 0c Fit-row audit (report only, gates nothing): per game and cell, the fit rows, the share from stage B (possible
  ghosts), the median depth (F1) and ratio (F2) with the two games' spread; and the frame-conflict table, per cell, the
  median depth toward T's image of the player beside D186's measure (SportVU under B). Also P-ii's distance
  distribution in 0.5 ft steps from 2.5 to 5 ft.
- 0d Falsifier, in sample: each form fitted on a game and scored on the same game. **Gate G0: some form has missed
  change at most -0.020 on both games and meets rules 2 to 7 of section 6 by point estimate.** Why -0.020: a correction
  fitted on another game keeps about half of an in-sample gain when the games' spread is at most half the bias (D186's
  reasoning), so an in-sample gain below twice the bar cannot clear the bar leave one game out. **G0 fails: KILL.** No
  leave-one-game-out run; the arm closes on these truths. G0 can only stop the arm; it chooses nothing, and stage 1 runs
  both forms whatever their in-sample order.

**Stage 1, the pilot (one lock, "p8 footfix s1"), leave one game out on gsw_cha_2016 and gsw_bkn_2015.** Both forms,
both folds, the rule of section 6. There is no larger run to scale to: the pool is these two games (gsw_sac_2015
apart, gsw_ind_2016 without a truth), so a FAIL ends the arm. Never rescued by a third form, a new cell, a new margin or a
refit on these games (D188).

**Stage 2, hou_cle_2016, the held-back test (a separate step).** Runs only after a stage 1 PASS, its ml-review, and
the lead's go. Before hou_cle_2016 is read: the chosen form's table, refitted on both pool games pooled, is written to
reports/research/p8_footfix_table.json, its SHA-256 recorded, and committed. Then, in one lock ("p8 footfix hou"):
identity first, which must equal D202's hou_cle_2016 records on every window for every GATE_FIELDS field; then the
committed table, scored once. The tool refuses a second hou score (a record marker, as tools.p8_outpos.check_once).
Truth: hou_cle_2016's fixed CPU truth (reference root), pinned. The rule is section 6 on hou_cle_2016 alone. Its truth
was fitted to the same detector's feet but by a different fit, in another arena: this is the arm's only arena and
truth transfer test. **PASS:** reported, and a separate D line is needed before anything enters the pipeline or the
site. **FAIL:** the arm closes and hou_cle_2016 is spent for it.

**Retries.** A step that fails for a mechanical reason is retried at most twice, then stopped and reported. A failed
gate is not a mechanical failure and is not retried.

## 9. What would change the plan (written now)

| result | reading | next step |
|---|---|---|
| G0 fails | even fitted in sample, a fixed foot shift does not move missed by 0.020 on both games on the decided truths | close; the foot-model line stops on these truths; the queue moves to recall (P-i, D-i) or to the truth's near-third geometry |
| G0 passes, stage 1 fails rule 1 | the in-sample gain does not travel between games: each game's residual is its own (likely its truth fit's) | close; post hoc reading only |
| stage 1 passes rule 1, fails rule 3 or 5 | the forecast below: the gain copies T's near-third residual against B | close; record that the scorer's two frames disagree in the near third; the next question is the truth's near-third geometry, not the foot |
| stage 1 fails rule 2 or 7 | the gain is boxes moved onto players without finding them | close |
| stage 1 passes | a correction travels between the two games | request stage 2 |
| stage 2 passes | the correction travels to another arena and truth fit | a D line for any further use; nothing changes in the pipeline or site here |
| stage 2 fails | the gain is local to the GSW pair or to the multipool fits | close; hou_cle_2016 spent |

**Forecast (written before any number; gates nothing).** In the near third, T places placement misses beyond the
player (D204 C1) while B places feet short (D186). Both forms, fitted toward T, will move near-third feet toward the
camera, which under B adds to the short bias there. Expected: near-third p90 and near-field bias rise on at least one
game, and rule 3 or 5 is the likeliest failure. If the forecast holds, it is a measured fact about the scorer's two
frames, recorded as such.

## 10. Forking paths, named now

- D184's hou_cle_2016 gain of -0.021 from a pooled shift, and D186's report-only hou_cle_2016 K1 change (-0.0087, on its
  old truth), have been seen. D202's hou_cle_2016 rows have been seen. No number of any correction on hou_cle_2016's
  current truth has been computed.
- D204's C1 depth by third and D205's P-ii rates on both pool games have been seen; they shaped the target and the
  forecast. The two forms were set by the lead's request after D186 to D188, as K1 and K2 with a new target.
- Stage 0's in-sample numbers will be seen before stage 1; they can only stop the arm.

## 11. Cost and writes

Stored files only, on the Mac, 0 units. No build, no detector, no video decode beyond the scorer's own clip reads.
Measured rates: D202 scored five game and truth pairs in 654.6 s including cache work (about 130 s per pair at most);
D205's collect took 55 to 59 s per pool game and 113 s on hou_cle_2016.

| stage | work | estimate | cap |
|---|---|---|---|
| 0 | smoke, identity on both games, fit rows, two forms in sample on both games | about 15 minutes | 40 minutes |
| 1 | two forms, two folds | about 10 minutes | 30 minutes |
| 2 | identity and one form on hou_cle_2016 | about 5 minutes | 20 minutes |

Each stage is one chain in the background under one lock (`python3 -m export.heavy --name "p8 footfix s0" -- ...`),
frozen CV python for scoring, system python for the report; at a cap it stops and reports nothing; reported the moment
the chain ends. Writes: the corrected builds, fit tables and score records under the reference root's footfix/ folder
(under 1 GB measured and recorded; builds/r23 is 306 MB for all 331 windows and every file type), and the records
reports/research/p8_footfix_s0.json, p8_footfix_s1.json, and for stage 2 p8_footfix_table.json and p8_footfix_hou.json.
Nothing under work_ms, the P8 workspace, truth_v2/, truth_accept/ or builds/r23 is written; SHA-256 manifests of those
folders before and after each stage stop the run on any difference. One ledger line per stage in
docs/research/TRAINING_PROTOCOL.md.

## 12. Limits

- The build's and the truth's errors are not separable and not independent (D202, D205): T was fitted to these feet.
  A pass here shows agreement with the decided truths, not feet shown nearer the real positions; the position rows
  through B and stage 2 on another truth are the partial checks.
- Two pool games, one arena, one broadcaster. The window intervals describe these windows, not games in general.
- Frames the truth rejected are not seen; one-to-one assignment for the fit rows is a convention (D205).
- Report and research only: no registered row is replaced, and nothing enters the pipeline or the site without a
  further D line.

## 13. Version 1.1 (2026-11-27, after the design review of 4abb21a9; before any code or number)

The independent review (training-design-reviewer) read version 1 as GO WITH CHANGES, with three blockers, three serious
findings and six minors. Sections 1 to 12 stand as registered except where this section replaces them; where the two
differ, this section rules.

### 13.1 Circularity (blocker 1)

The truth camera T removes the hits' mean offset by construction: fit_game pools foot-to-SportVU pairs, each frame's
camera is then refitted to that frame's inlier feet (sportvu.camera.fit_frame on the pairs of sportvu.truth.window_truth,
then refine_lines), and frames are kept only at a median foot residual of 2.0 ft or less (sportvu.autolabel
MAX_FEET_RESID_FT). All three games share the procedure, so leave one game out does not break the circle. Registered:

(a) **Reading of a stage 1 pass.** A form that passes section 6 reads **"feet nearer"** only if position p50 under B
changes by 0 or less (point estimate) on both pool games and (b) below does not read "copying T". Otherwise it reads
**"agreement with T only"**: stage 2 is not requested and hou_cle_2016 stays unspent.

(b) **Split by truth-fit inliers.** For every scored frame, the inliers of that frame's truth fit are recomputed as the
truth step computed them: sportvu.truth.window_truth on the window's label-build sidecar and time map (the
work_ref/label_build files the truths were fitted from; work_ms links to them), as tools.p8_multistart.build_inputs and
tools.p8_frozen's fit step call it. Gate: every frame the truth accepted (every labels row) has a recomputed row with
status ok and the same q and moment; else the run stops. A player is an **inlier player** on a frame when its SportVU id
is in an inlier pair of that frame's recomputed row; the label-build box of that pair is reported beside the scored box
(the share of scored frames whose label-build boxes equal builds/r23's in bbox and foot_stab is reported). The missed
change and the position p50 change are reported separately for inlier and non-inlier players, per game, with window
intervals. **"Copying T"** reads when, on either game, for missed or for p50, the inlier players' change is below 0 and
the non-inlier players' change is 0 or above (point estimates).

(c) **F2 is a predicted-null control, not a candidate.** T was fitted to these feet, so a vertical offset proportional to
box height is largely absorbed by T's per-frame pose. Predicted now: every populated bin of every F2 table (in sample and
per fold) has |r_bin| at most 0.01 (about 2 px on a 200 px box; D186's ratios toward SportVU under B were -0.017 to
-0.036). F2 is scored in stages 0 and 1 and reported beside F1, but G0, stage 1's pass and stage 2 read F1 only.
Section 4's choice between forms is withdrawn. If the prediction fails, that is reported as T not absorbing the vertical
offset; it changes no gate.

### 13.2 Guard readings and power (blocker 2)

The guard bars of section 6 sit near the paired noise half-widths. A guard failure counts as **harm** only when the
point estimate is beyond the bar. A failure by the interval's end alone reads **"not shown non-inferior,
inconclusive"**. Both close the arm: there is no rescue, no further pool look and no bar is widened (D188).

Computed power (section 7 is extended by this). For a form whose true effect on a guard row is zero, the chance that the
row passes is about Phi(1.96 (bar / u - 1)), with u the distance from the point to the upper end of the closest measured
paired interval (D186 and D187's K1; D188's K1m for near-third p90). Team has no measured paired interval and is left out.

| guard (bar) | gsw_bkn_2015: u, pass chance | gsw_cha_2016: u, pass chance |
|---|---|---|
| ghosts per box (+0.010) | 0.0036, 1.00 | 0.0067, 0.84 |
| near-field bias (+0.10 ft) | 0.028 ft, 1.00 | 0.158 ft, 0.24 |
| position p90 (+0.10 ft) | 0.078 ft, 0.71 | 0.072 ft, 0.78 |
| near-third p90 (+0.05 ft) | 0.052 ft, 0.46 | 0.038 ft, 0.73 |
| all four, if independent | 0.33 | 0.11 |

Both games together: about 0.04. The primary row passes with chance 0.50 at a true change of -0.010 on either game, 0.93
(bkn) and 0.86 (cha) at -0.015, and 0.98 or more at -0.020. So a form with a real -0.015 effect and no harm to any
guard passes the whole rule with a chance of a few percent, if the guards are independent (they are not: p90 and
near-third p90 share players, which raises it). The proxies come from older truths with fewer gsw_cha_2016 frames, so
the gsw_cha_2016 figures are likely pessimistic. Read plainly: the registered rule rarely passes a harmless form, and
stage 1's main information is likely the harm or inconclusive reading per row, not a pass.

### 13.3 Rule 7 (blocker 3)

Moving a P-ii box onto its player also lowers one-to-one missed, and it removes that box from the ghost row (its foot
comes within 3 ft of a player). So neither one-to-one missed nor ghosts catches a ghost box moved onto a player; this is
a limit of the design, listed in section 12. Rule 7 becomes: **one-to-one missed change at most -0.0075 on each game,
or at most 0.75 times the missed change** (a one-to-one fall of at least three quarters of the missed fall). In-frame
players and boxes stay equal by construction (a difference stops the run). Net error is reported, not ruled.

### 13.4 Serious findings 4 to 6

4. **G0 (self-decided by the lead):** F1, fitted and scored in sample, has missed change at most -0.010 on both games and
   meets rules 2 to 7 by point estimate. This replaces the -0.020 bar of section 8: it keeps a cheap kill without the
   false-negative risk of -0.020. G0 fails: KILL, as before.
5. **hou_cle_2016.**
   - The once-marker is written before any hou_cle_2016 file is read, at work_ref/footfix/hou_once.json. That folder is
     outside the folders the manifests watch. If the marker exists, the tool refuses. Any retry after the marker needs
     the lead's decision.
   - Stage 2 needs, besides a stage 1 pass of F1, the reading "feet nearer" of 13.1 (a). It also needs F1's ml-review
     and the lead's go.
   - Section 10 adds D185's reading as seen: near-third misses on gsw_cha_2016 and hou_cle_2016 were mostly
     disagreement between the r23 court's H and the truth camera (old truths).
   - A stage 2 FAIL cannot separate the correction from hou_cle_2016's single-pool fit (its truth is the fixed CPU fit,
     not multipool). It is recorded as a FAIL of the arm with that limit stated.
   - Team on hou_cle_2016: only the change rule 6 needs is reported, with no team breakdown (D203 keeps
     hou_cle_2016's team rows for a team arm).
6. **The reviewer's bound.** The likely in-sample effect is at the low end of section 7's range, and a G0 kill is likely
   (about 70 percent, the reviewer's estimate). 0c's same-row conflict table (13.6) is the real premise of the arm. If T
   and B disagree in sign on the same rows in the cells that carry P-ii, no single foot shift can serve both frames.

### 13.5 Minors 7 to 12

7. Fit-row eligibility: only in-frame players (the scorer's 10 px margin) give fit rows. The assignment follows
   assign_1to1's convention: non-referee boxes against all ten players. This is D205's main split:
   two_stage(feet, sv, range(10), non-referee boxes, _assign_all). A pair whose player is out of frame claims its box
   but gives no row.
8. Byte-equal identity: a box whose cell has no correction (no table entry, a zero entry, or F1 on a frame without an
   accepted B or pose) keeps its stored foot_stab values exactly, with no image to court round trip. The identity table
   therefore writes a sidecar whose frames equal builds/r23's (checked by hash in the smoke).
9. score_pair: the corrected arm is scored with its own folder as both sides and the corrected sidecar passed as both
   sides' sidecar, so its extra counts (one-to-one missed, far, duplicates, unmatched) come from the corrected feet. The
   baseline is the identity arm, which must equal D202's records. Asserted per arm: if missed differs from the baseline
   on a game, missed_1to1 differs too; else the run stops, because the sidecar handed to the extra counts would be
   stale.
10. F1 has 11 populated cells on D186's tables (no box of 250 px or more in the far third), not 12.
11. Disk: about 0.9 GB for the corrected builds (stage 0: two forms on both games in sample, plus the identity copy;
    stage 1: two forms on both games; stage 2: identity and one form on hou_cle_2016).
12. Section 12 adds: the scored frames are themselves selected on agreement between T and the feet (median foot
    residual at most 2.0 ft), so frames where the feet disagree most with T are not scored.

### 13.6 Stage 0's audit, extended (0c, in the same lock, before G0 is computed)

- The F1 and F2 tables fitted on each game's inlier rows only and on its non-inlier rows only, beside the all-rows table.
- F2's ratios against the predicted 0 of 13.1 (c), per bin, in sample and for each fold table.
- The same-row comparison: for every F1 fit row, the depth toward T's image of the player (F1's fit quantity) and the
  depth toward SportVU under B (D186's quantity), on the same row. Per cell and game: both medians, their difference,
  and the share of rows where the two have opposite signs. Cells holding P-ii rows are flagged with their P-ii counts.
- These are report only and gate nothing. They are written to the stage 0 record before the in-sample scoring starts.

### 13.7 Summary of the rule as it now stands

- **Stage 0:** smoke, then identity reproduction of D202 on every window, then 0c, then G0 on F1 in sample (13.4 (4)).
  G0 fails: the arm closes.
- **Stage 1:** F1 leave one game out, with section 6 rules 1 to 6 and rule 7 as in 13.3.
  - A guard failure reads harm or inconclusive (13.2). Either way the arm closes.
  - A pass reads "feet nearer" or "agreement with T only" (13.1).
  - Only "feet nearer" requests stage 2.
- **Stage 2:** as 13.4 (5).

## 14. Version 1.2 (2026-11-27, after the code check of 3f3c6ee2, GO WITH CHANGES; before `score s0`)

No code bug blocked the run. These are registered before any pool number; where they differ from sections 1 to 13,
this section rules.

14.1 **Copying T, second clause.** "Copying T" (13.1 (b)) also reads when, on either game, the non-inlier players'
missed change is below 0 while their position p50 change under B is above 0 (point estimates). The audit adds a cross
table of fit rows by stage (A or B) and inlier status, per game.

14.2 **Rule 7, stated plainly.** Under rule 1 (missed change at most -0.010), 0.75 times the missed change is at most
-0.0075. So the fraction clause never binds, and rule 7 reduces to "one-to-one missed change at most -0.0075 on each
game". It is kept as registered.

14.3 **The reading record** names, per game:
- a rule 1 failure;
- a rule 7 failure;
- the harm rows;
- the inconclusive rows (an interval end beyond the bar, or a row that could not be computed).

These are reported beside the verdict, its reading and the stage 2 flag.

14.4 **Cost.** Each score step hashes the watched folders before and after it: about 17 GB, most of it the
label_build clips of the workspace and the reference root. That time counts inside the cap. The estimates become about
18 minutes for stage 0, 13 for stage 1 and 8 for stage 2. The caps (40, 30, 20 minutes) are unchanged.

14.5 **Versions.** Every run.json records the cv2, numpy and Python versions.

14.6 **Wording of 13.6, corrected.** The audit is written to the stage 0 folder (audit.json) before the in-sample
scoring starts. Its SHA-256 lands in run.json at the end of the run, and `report s0` checks it. Its content enters
reports/research/p8_footfix_s0.json only through the report step. 13.6 said it is "written to the stage 0 record"
before the in-sample scoring; the record meant is audit.json, not p8_footfix_s0.json.

14.7 **A test** checks that cmd_score_hou writes the marker before its first hou_cle_2016 read, refuses a second
start, and refuses an empty go.

14.8 **The audit readings, registered before the run** (report s0, per game, from 0c's tables):

- **A, absorbed.** The all-rows F1 cell medians are within 0.3 ft of zero (either sign) in cells that together hold
  at least 80 percent of the game's F1 fit rows. A cell with fewer than 50 rows has no entry and counts as within,
  since its shift is 0. If A holds on both games, the reading is "a median shift toward T is too small to reach the
  bar". This closes the toward-T line whatever G0 says: `report s0` says so, and `score s1` refuses.
- **B, frame conflict.** A cell conflicts when the medians of d1 and dB have opposite signs, or when more than half its
  rows have opposite signs. If the conflicting cells hold at least half of the game's P-ii fit rows (rows with a B
  pose), the reading is "no single foot shift serves both frames".
- **C, T's residual dominates.** A cell qualifies when it is populated in both the inlier-only and the
  non-inlier-only F1 tables and the two medians have opposite signs. If the qualifying cells hold more than half of the
  game's F1 fit rows, C holds.
- **D, F2 null.** Every populated bin of the game's in-sample F2 table has |r| at most 0.01.
- **E, edge ceiling.** P-ii fit rows (stage B) whose box lies 2.5 to 3.0 ft away under T, over the game's in-frame
  players. Under 0.02 puts the -0.010 bar out of reach on that game for shifts of 0.5 ft or less.
- **F, box match.** If the share of scored frames whose label-build boxes equal builds/r23's (bbox and foot_stab) is
  under 0.9, that game's inlier split is descriptive only: it is reported, but the copying-T reading does not read it.

Readings B to E change no gate. They are reported with G0.
