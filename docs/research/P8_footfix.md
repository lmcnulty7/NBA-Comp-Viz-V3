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
