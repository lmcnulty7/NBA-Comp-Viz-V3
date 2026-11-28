# P8 C1 reach: can a teacher box the players the detector stage loses? (design, version 1, 2026-11-27)

The pre-training label audit for any detector arm (docs/research/TRAINING_PROTOCOL.md, ml-training-rigor section 2:
what a label source teaches, measured against independent truth before any training). Report only: nothing is fitted,
tuned or trained. No download in stage 1. hou_cle_2016 stays unspent and gsw_sac_2015 apart; the allow-list
{gsw_cha_2016, gsw_bkn_2015} and the P8 guards (tools.p8_common.guard, refuse_cv_test) run on every name. The held-out
list (sportvu/splits.json) is never read or scored: heldout_game 12.16.2015 PHX at GSW, heldout_arena the Cleveland
arena, heldout_era the 2013 GSW at NYK game (video 6sAFbDBBAZc). Any outside label source a later stage brings in is
screened against the same list, and against the D85 test homes (MIA, TOR, LAC, LAL), before it is used (D150 (2)).

## 1. Question and mechanism

D208 and its supplement put the detector stage, C1, at 0.693 of the no-own-box misses on all 120 windows (post hoc,
P8_recallcause_supp.md 10.6): 3,384 misses, of which C1a_other (the player's image position inside only another
player's box) 1,491, C1c (no candidate at all near the player) 1,515, C1b (a referee box near the player) 230 and
C1a_own 148. A detector arm can only lower these if its labels box such players separately. Question: on the frames
and players where r23 loses them, how many C1 misses does a teacher box, per subclass, and what would that be worth at
build level after the measured raw-to-build transfer?

Stage 1 measures the teacher already held, YOLOX-X trained on SportsMOT (MixSort; P8_teachers.md, qualified at 0.1471;
its labels "predict the bounding box of the athlete in the case of occlusion", P8_labelsource.md). Stage 2,
Iter-Deformable-DETR trained on CrowdHuman, runs only after stage 1's rule (section 7), a docs/DATA_SOURCES.md row and
a verbatim re-read of its licence sentence and CrowdHuman table (P8_labelsource.md section 7); it gets its own dated
amendment before any download.

## 2. What E1b already measured, and what is new

E1b (P8_E1B.md sections 2 and 2b) measured the held teacher's reach on all of r23's raw misses on the 5,592 E1 corpus
frames of three pool games: 31.2 percent at 0.1471, four fifths placement (r23 had a box within 5 ft), with the
transfer-adjusted perfect-imitation build ceiling -0.0199 pooled and -0.0079 on gsw_cha_2016 (gate failed, D183). It
did not split misses by cause, it used corpus frames (read the a3run way, since shown not to be the build's frames on
seek windows, P8_recallcause.md 9.7), and its reach rule was the nearest foot, which a box shared with a neighbour can
satisfy for two players. New here: the misses are D208's C1 misses on the build's own frames (build_read), split by
subclass; reach is one to one; and the cost rows sit beside it.

## 3. Upper bound (from stored records, at design time)

If a student reached every C1 miss, E1b's measured transfer fractions per game (P8_E1B.md 2b: gsw_bkn_2015 presence
0.616, placement 0.378; gsw_cha_2016 0.364 and 0.240), with D-i misses counted as presence (no box within 5 ft) and
P-i as placement (a box 2.5 to 5 ft away), give a build-level fall in missed of 0.0153 of in-frame players on
gsw_cha_2016 (774 D-i and 642 P-i C1 misses over 28,427 in-frame players, D202's record), 0.0348 on gsw_bkn_2015 (1,170
and 798 over 29,370) and 0.0252 pooled. C1 alone cannot reach E1b's 0.020-and-0.010 adoption bar on gsw_cha_2016 even at
perfect reach. So stage 1 asks whether a teacher reaches enough of C1 to close E1b's gap on that game when added to what
E1b's ceiling already counts (section 7), not whether C1 alone makes an arm.

## 4. Misses, frames and the sample

- **C1 misses.** Recomputed with tools/p8_recallsupp.py classify_window from the stored candidates (recallcause/ for
  the supplement's 46 certified windows, recallsupp/ for its 74 re-read ones), on D205's rows recomputed with the row
  gate (tools.p8_recallcause.d205_rows). Each C1 miss: game, window, frame, SportVU pid, unit (P-i or D-i), subclass.
  **Gate R0:** the counts equal the supplement's units_120_post_hoc C1 subclass counts in every unit, per game and
  pooled (P8_recallcause_supp.md 10.5); else stop.
- **Sample (fixed now).** In each of the 120 windows, the frames holding at least one C1 miss, sorted by frame index;
  if more than 8, the 8 at ranks round(linspace(0, n - 1, 8)); every C1 miss on a sampled frame is in the sample. At most
  960 frames. Shares are computed on the sample; counts are scaled to the full C1 set per window (window count over
  sampled count) before the build-level rows.
- **Frames.** Read with the build's read path (tools/p8_recallsupp.py build_read, the supplement's copy of posefoot's,
  with stage A's environment: HF_HUB_OFFLINE=1, PYTORCH_ENABLE_MPS_FALLBACK=1, the gate device from each window's build
  record). A window whose read validation fails is left out and counted. **Gate R1:** on the first sampled frame of
  every window, r23's capture (tools.p8_recallcause.Raw) equals the stored R_post exactly, so the decoded frame is the
  one the C1 classes were read on; a window that fails is left out and counted, and if more than 5 windows fail the run
  stops.

## 5. The teacher (stage 1)

tools.p8_teachers.load_runner("yolox_x_sportsmot") on CPU, unchanged: test size 800x1440, its NMS 0.7, the 0.05 floor,
boxes at or above 0.1471 used for reach (P8_teachers.md's qualified threshold), the 0.05 to 0.1471 band reported apart.
The teacher boxes players only (SportsMOT labels no referees), so no referee rule applies to it. **Gate R2 (pre-flight):**
the weights hash as recorded (yolox_x_sports_train.pth.tar, SHA-256 58547880...0ed1c, the teachers workspace's
preds/yolox_x_sportsmot.json), and on the first 20 of that record's 50 frames (CPU, 0.839 s a frame) the run's
predictions equal the stored ones at their rounding; else stop (platform or code drift).

## 6. Statistics (fixed now)

For each sampled frame: SportVU positions of all ten players (D205's row), H_truth, in-frame by sportvu.eval's margin.
Box feet are bottom centres through H_truth (as D208).
- **Reach.** A C1 miss j is reached by a box set when, in a one-to-one assignment (Hungarian on foot distance, pairs
  within sportvu.eval.MISS_FT, 2.5 ft) of the set's feet to all ten players, j is assigned a box. So a box shared with a
  neighbour counts once.
- **Floor.** The same rule on r23's own class-0 boxes at conf 0.25 after its NMS (the stored R_post plus R_low): what
  lowering r23's threshold alone gives. **Beyond the floor:** misses the teacher reaches and the floor does not.
- **Rows,** per subclass (C1a_other, C1c, C1b, C1a_own), per unit (P-i, D-i) and per game and pooled: reach share of the
  teacher at 0.1471, of the floor, and beyond the floor, each with a window-cluster bootstrap (2,000 resamples, seed 0,
  tools.p8_resetfix.resample_picks over sorted windows).
- **Cost rows, beside reach,** per sampled frame and pooled: teacher boxes with no player of all ten within
  sportvu.eval.GHOST_FT (3.0 ft), the ghost proxy; duplicates (in-frame players with two or more teacher feet within
  2.5 ft, tools.p8_resetfix.frame_extra's count); the same two for r23 at 0.40 and at the floor for comparison.
- **Transfer-adjusted C1 ceiling** of a box set S, per game: C_S = -(R_D x f_pres + R_P x f_plc) / N, where R_D and R_P
  are S's beyond-floor reached C1 misses of D-i and P-i (scaled to the full set), f_pres and f_plc E1b's per-game
  fractions (section 3) and N the game's in-frame players (D202). Pooled: the sums. No loss term (a student's lost
  finds are not measured here), so C_S is optimistic, as E1b's C_opt was.
- **Headroom** H, per game: C of the C1a_other and C1c misses that neither the floor nor the held teacher reaches, as
  if all were reached.

## 7. Decision rule (registered now)

E1b's ceiling for the held teacher's labels was -0.0199 pooled and -0.0079 on gsw_cha_2016 against the bar of -0.020
and -0.010 on every game; the gap on gsw_cha_2016 is 0.0021. E1b's reading was that a realistic student keeps about half
of a ceiling.
- **Stage 1 never adopts anything.** The held teacher's labels were already gated by E1b (D183); stage 1 sets the bar
  a new source must beat and checks that the measurement works.
- **Stage 1 to stage 2:** stage 2 (Iter-Deformable-DETR) is designed only if H is at most -0.0042 on gsw_cha_2016 (twice
  the gap, since a student keeps about half) and below 0 pooled. Else the CrowdHuman-detector stage is not run: even a
  crowd detector that boxed every remaining overlap and no-candidate miss could not close gsw_cha_2016's gap at a
  realistic transfer.
- **After stage 2 (registered now, applied then):** a training-arm design is justified only if stage 2's increment over
  the held teacher, the transfer-adjusted C1 ceiling of the misses Iter-Deformable-DETR reaches and neither the floor
  nor the held teacher does, is at most -0.0042 on gsw_cha_2016 with its window interval below -0.0021, and below 0
  pooled; and its ghost proxy per sampled frame is at most twice the held teacher's. The arm's own design then needs E1b
  section 2b's full bound for the new labels, on all misses, before any unit.
- **Kill:** if stage 1's H misses the stage 2 bar, or stage 2's increment is above -0.0021 on gsw_cha_2016 (perfect
  imitation of its C1 gains would not close the gap), the detector line stays paused under D183 for these label
  sources, and the next candidates (SHOT and E-BARD labels, P8_labelsource.md section 5) would need their own audit.
- The thresholds come from E1b's registered bar and gap; no number of this item has been seen.

## 8. Limits

- Raw reach on build frames is not a build effect; the transfer fractions are E1's, measured on corpus frames and a
  different student, and E1b found the transfer model uncertain by about 0.0035 per game in either direction (2b).
- The C1 classes are the supplement's post hoc 120-window classes; the 46 certified windows use stored candidates
  checked equal to the build read by G0 and G1 of the supplement.
- Truth bias (D202): truth frames were selected with r23's boxes, so misses on frames r23 handles badly are under-read.
- The sample caps big windows at 8 frames; the scaled counts assume a window's sampled frames stand for its others.
- The teacher boxes players only; a teacher box on a referee near a C1b miss is not separable here.

## 9. Cost and writes

One lock job ("p8 c1reach stage1", frozen CV python, CPU detector and teacher): D205's rows on 120 windows (about the
supplement's 3 minutes), the build read of 120 windows (stage A's rate, about 8 s a window, 16 minutes), R1's 120 r23
frames (about 1 minute), the teacher on at most 960 frames (P8_teachers.md: 348.5 CPU s for 400 frames, so about 14
minutes) and R2's 20 frames: about 35 minutes, cap 60. 0 Colab units. Writes: reports/research/p8_c1reach_stage1.json
(checked with find_private) and nothing else; the teacher's per-frame boxes stay in the reference root's c1reach/
folder, not committed. Manifests watched before and after (labels, label_build, truth_accept, cache, builds/r23,
recallcause/, recallsupp/, the teachers workspace); any change stops the run.

## 10. Version 1.1 (2026-11-27; the design review of 658e3a98, GO WITH CHANGES; rebased on v4-research b4262d7d; before any code)

Where this section differs from sections 1 to 9, it rules.

**10.1 Section 3, corrected.** As written, section 3 is wrong: perfect C1 reach would lower missed by 0.0153 on
gsw_cha_2016, which passes E1b's -0.010 bar, since that bound has no loss term. The quantity that matters is the reach
beyond what the held teacher's labels already give, under a union label set (the held teacher's labels plus the new
source's extra boxes). Section 3's numbers were computed at design time from stored records (the supplement's tables
and D202's in-frame counts); they are bounds, not results. "Pooled" in this design is gsw_cha_2016 plus gsw_bkn_2015,
not E1b's pooled with gsw_sac_2015.

**10.2 Stage 1 recast.** Stage 1's rule cannot fail: the headroom H of the C1a_other and C1c misses on gsw_cha_2016 is at
most -0.0142 if every one were reached (709 D-i and 607 P-i misses at f_pres 0.364 and f_plc 0.240 over 28,427 in-frame
players), 3.4 times the bar. So stage 1 is the held-teacher reference set and a measurement check, not a gate: it
measures the held teacher's reach and cost and validates the measurement (R0, R1, R2 and the positive control, 10.9). It
decides nothing by itself; stage 2 is designed after it as section 7 says, with the rule of 10.6.

**10.3 The environments (blocker).** The teacher cannot import in the frozen CV python (no loguru), and R2's record was
made in the teachers venv, kept outside the repository (Python 3.13.3, torch 2.15.0). Nothing is installed into the frozen python. One
lock job runs two processes in sequence:
- **Step A, the frozen CV python:** the guards; D205's rows with the row gate; R0; the sample; build_read on the 120
  windows with r23's capture on every sampled frame (R1, 10.8); every sampled frame written losslessly (PNG) to the
  reference root's c1reach/frames/, with its SHA-256; one inputs file (per sampled frame: the ten SportVU positions,
  in-frame flags, H_truth, the C1 misses with unit, subclass and pid, r23's stored R_post and conf 0.25 boxes, and the
  frame's SHA-256). The frames stay for stage 2.
- **Step B, the venv:** checks every frame's SHA-256 against the inputs file; R2 (section 5) on the box-truth frames of
  the teachers workspace, which are exempt from the allow-list (they include gsw_ind_2016) and admitted only after
  tools.p8_teachers' record check passes on them; then tools.p8_teachers.load_runner("yolox_x_sportsmot", ..., "cpu")
  on every sampled frame with torch.set_num_threads(4). Never cmd_run, which would overwrite the R2 record. Writes the
  teacher's boxes to c1reach/.
- **Step C, the system python, no lock:** the statistics of section 6 and 10.4 to 10.9 and the record.

**10.4 The bar and the noise floor (replaces section 7's "twice the gap").** From the review: gsw_cha_2016's window
half-width on a build-level missed row is about 0.006 (P8_E1R.md), so a C1 increment must be about -0.0041 or better to
stand out; the bar stays -0.0042, now justified by that noise floor rather than by doubling the gap. The -0.0021 line
(the gap itself) stays as the kill line.

**10.5 The transfer's uncertainty.**
- The intervals include the uncertainty of f: in each bootstrap replicate, our windows are resampled and E1's corpus
  windows are resampled within each game, and f_pres and f_plc are recomputed from E1's per-window counts, as E1b 2b
  did (tools/p8_e1b_bound.py's transfer counts, from stored files).
- A sensitivity row adds gsw_cha_2016's +0.0048 transfer-model residual (P8_E1B.md 2b) to its C.
- Registered: if the stage-2 increment's interval necessarily spans the bar given f's spread (about +-0.0045 on
  gsw_cha_2016 from f alone), the reading is "not resolvable with these truths" and the detector line stays paused.

**10.6 Stage 2's outcomes (replaces section 7's rule after stage 2; every zone has an action).**
- Increment at most -0.0042 on gsw_cha_2016, its interval below -0.0021, below 0 pooled, and the ghost proxy at most twice
  the held teacher's: a training-arm design is justified (a union label set; it needs E1b 2b's full bound on all misses,
  with a loss term if any design replaces the held teacher's labels instead of adding to them).
- The same, but the interval crosses -0.0021: "not resolvable with these truths" (10.5) when f's spread alone forces
  it; otherwise "not shown". Either way no arm; the detector line stays paused.
- Increment between -0.0042 and -0.0021: short of the noise-floor bar; no arm; paused.
- Increment above -0.0021: the CrowdHuman-detector line is closed for these sources (kill, as section 7).
- The increment passes but the ghost proxy fails: no arm on its raw boxes; a filtered-label design may be registered
  as a new item; paused meanwhile.

**10.7 The P-i mapping and its extremes.** P-i misses count as placement (f_plc) and D-i as presence (f_pres), as
registered; beside it, the two extremes (every C1 miss at f_plc; every one at f_pres) are reported.

**10.8 A court filter, and R1 everywhere.** Before the ghost proxy and the reach, both teachers' boxes are kept only
when their foot, through H_truth, lies inside the court lines plus 3 ft on every side (SportVU court coordinates, 94 by
50 ft). The teacher may box referees (its labels exclude them, but the model is not prevented from boxing them); the
cost rows report its boxes within 2.5 ft of no player separately from those near one. R1 runs on every sampled frame,
not only the first: r23's capture must equal the stored R_post on each, a frame that fails is left out and counted, and
more than 5 percent failing stops the run.

**10.9 The assignment and a positive control.** Reach uses tools.p8_one2one.assign_1to1 (Hungarian on foot distance
over all ten players, pairs beyond 2.5 ft cost 1e6, no referee boxes for the teacher): a single stage at 2.5 ft, unlike
O.two_stage, which adds a second stage at 5 ft for the leftovers. Positive control: on the same frames, the teacher's
one-to-one reach of r23's own hits (in-frame players assigned an r23 box at 0.40 by assign_1to1). Gate: at least 0.90;
else the measurement is suspect and stage 1 reports nothing (E1b found the teacher reaching 97 percent of r23's finds by
the nearest rule).

**10.10 Wording.** hou_cle_2016: no data read. The held teacher carries its own residual risk under D150 (2): it was
trained on SportsMOT, whose games cannot be identified, so a held-out game in its training set cannot be ruled out;
disclosed, as for every use of it.

**10.11 A smoke before the job.** Both environments on one seek window, gsw_cha_2016_s07_w00 (about 2 lock minutes): step
A's build read, the frames dumped with their hashes, R1 on every sampled frame; then step B's hash check, R2, the teacher
and the positive control. The 35-minute job runs only after the smoke passes and the lead's go. Cost of the frames: about
960 PNG frames of 1280 by 720, about 1.5 GB in the reference root.

**10.12 The code check of 826d5d06 (2026-11-27; GO WITH CHANGES; after the smoke, before the full run).** Where this
differs from 10.1 to 10.11, it rules.
- **The union-baseline sensitivity (replaces 10.5's residual row).** E1b's C on gsw_cha_2016 for the held teacher's
  labels is -0.0079; with the transfer model's +0.0048 residual it is -0.0031, so the gap stage 2 would face is 0.0069
  instead of 0.0021. The record reports that pair (union_baseline), and stage 2's reading (10.6) is also shown under the
  larger gap.
- **Order.** D205's rows, the classes and R0 run for every window before any frame is read.
- **Caps in code.** Step A stops past 45 minutes (checked between windows), step B past 20 minutes (checked between
  frames).
- **Watches.** Step B hashes the teachers workspace's preds/yolox_x_sportsmot.json and the weights before and after;
  any change stops it. Step A checks and records the library versions against D208's pinned set.
- **The PNG round trip.** Each written frame is read back and must equal the decoded array exactly.
- **Bootstrap.** The share intervals are stratified by scope: a game's windows within the game; pooled, within each
  game.
- **Denominators.** Windows left out by the read validation, frames failing R1, and the C1 misses on them are
  reported, with the share of C1 misses in windows that have sampled frames.
- **Corrections.** R2's box-truth frames come from the games 201206070BOS and 201602270OKC (the teachers workspace's
  frames.json), not gsw_ind_2016 as 10.3 said. tools/p8_c1reach.py e1_per_window loads the E1 corpus prediction chunks
  whole (they include gsw_sac_2015's) but uses only gsw_cha_2016 and gsw_bkn_2015 frames.
- **The smoke** (gsw_cha_2016_s07_w00, at 826d5d06; reports/research/p8_c1reach_stage1_smoke.json, its report step
  rerun with this version's code): R1 0 of 8 failed, every hash equal, R2 20 of 20 equal, positive control 58 of 59
  (0.983). Its reach numbers are one window and are not read.
- The full run: steps A and B chained under one lock command, then step C.


## 11. Stage 1 result (2026-11-28; run at e75c0152; reports/research/p8_c1reach_stage1.json)

**11.1 The run.** One lock job ("p8 c1reach stage1", 1,600 s from the lock's "done in" line, including a short wait for
another job): step A 906 s and step B 664 s (the seconds in the reference root's c1reach/inputs.json and
c1reach/teacher.json, which also hold R1's and R2's results); step C on the system python, no lock. hou_cle_2016: no
data read.
- R0 passed: the C1 recount equals the supplement's tables in every unit, per game and pooled.
- R1: 0 of 775 sampled frames failed; 0 windows undetermined; every one of the 3,384 C1 misses lies in a window
  with sampled frames (1,012 of them on the sampled frames; windows with fewer than 8 C1 frames contribute all of
  theirs, so 775 frames, not 960).
- R2: 20 of 20 box-truth frames reproduce the stored teacher predictions exactly; every frame and image hash matched;
  the teachers workspace was unchanged.
- Positive control: the held teacher reaches 4,831 of r23's 5,057 in-frame hits on the same frames, 0.955
  (gate 0.90): passed.

**11.2 Reach on the sampled C1 misses (one to one at 2.5 ft; window bootstrap, stratified by game).** Shares of the
sampled misses: the held teacher at 0.1471 (court-filtered), the floor (r23 at 0.25), beyond the floor (the teacher
reaches, the floor does not) with its interval, and the teacher with its 0.05 to 0.1471 band.

| unit | subclass | sampled misses | teacher | floor | beyond the floor [interval] | teacher with band |
|---|---|---|---|---|---|---|
| P-i | C1a_other | 137 | 0.299 | 0.022 | 0.277 [0.202, 0.356] | 0.299 |
| P-i | C1c | 268 | 0.280 | 0.015 | 0.265 [0.204, 0.330] | 0.287 |
| P-i | C1b | 11 | 0.545 | 0.091 | 0.455 [0.143, 0.800] | 0.545 |
| P-i | C1a_own | 26 | 0.192 | 0.000 | 0.192 [0.048, 0.364] | 0.192 |
| P-i | C1 | 442 | 0.287 | 0.018 | 0.269 [0.219, 0.320] | 0.292 |
| D-i | C1a_other | 263 | 0.540 | 0.061 | 0.490 [0.426, 0.556] | 0.544 |
| D-i | C1c | 236 | 0.203 | 0.038 | 0.169 [0.114, 0.231] | 0.203 |
| D-i | C1b | 45 | 0.711 | 0.089 | 0.622 [0.467, 0.761] | 0.711 |
| D-i | C1a_own | 26 | 0.154 | 0.000 | 0.154 [0.038, 0.296] | 0.154 |
| D-i | C1 | 570 | 0.396 | 0.051 | 0.353 [0.300, 0.405] | 0.398 |
| both | C1a_other | 400 | 0.458 | 0.048 | 0.417 [0.367, 0.468] | 0.460 |
| both | C1c | 504 | 0.244 | 0.026 | 0.220 [0.176, 0.268] | 0.248 |
| both | C1b | 56 | 0.679 | 0.089 | 0.589 [0.458, 0.705] | 0.679 |
| both | C1a_own | 52 | 0.173 | 0.000 | 0.173 [0.080, 0.269] | 0.173 |
| both | C1 | 1012 | 0.349 | 0.037 | 0.316 [0.276, 0.357] | 0.352 |

By game, all C1:

| game | sampled misses | teacher | floor | beyond the floor [interval] |
|---|---|---|---|---|
| gsw_cha_2016 | 457 | 0.274 | 0.031 | 0.247 [0.189, 0.310] |
| gsw_bkn_2015 | 555 | 0.411 | 0.041 | 0.373 [0.320, 0.420] |

**11.3 Transfer-adjusted C1 ceilings (beyond the floor, scaled to the full C1 counts; P-i at f_plc and D-i at f_pres,
with the two extremes; joint bootstrap of our windows and E1's, and the f-only interval).** E1's per-window counts
reproduced E1b's record on both games. Pooled is gsw_cha_2016 plus gsw_bkn_2015.

| scope | C | C, f only | C, all at f_plc | C, all at f_pres | H (C1a_other and C1c left) | H, f only |
|---|---|---|---|---|---|---|
| gsw_cha_2016 | -0.0053 [-0.0082, -0.0029] | [-0.0069, -0.0035] | -0.0039 [-0.0056, -0.0023] | -0.0059 [-0.0098, -0.0029] | -0.0086 [-0.0118, -0.0059] | [-0.0108, -0.0064] |
| gsw_bkn_2015 | -0.0157 [-0.0223, -0.0102] | [-0.0179, -0.0132] | -0.0115 [-0.0159, -0.0076] | -0.0187 [-0.0270, -0.0119] | -0.0157 [-0.0202, -0.0117] | [-0.0179, -0.0133] |
| pooled | -0.0106 [-0.0142, -0.0074] | [-0.0120, -0.0090] | -0.0077 [-0.0101, -0.0056] | -0.0124 [-0.0169, -0.0084] | -0.0123 [-0.0149, -0.0097] | [-0.0138, -0.0106] |

Union baseline (10.12): E1b's C on gsw_cha_2016 for the held teacher's labels -0.0079, with the transfer model's residual
-0.0031; the gap to the -0.010 bar 0.0021, or 0.0069 with the residual.

**11.4 Cost rows (all 775 sampled frames).**

| box set | boxes | ghost proxy (no player within 3 ft) | near no player (beyond 2.5 ft) | duplicates |
|---|---|---|---|---|
| held teacher (0.1471, court-filtered) | 7,628 | 1,402 | 1,898 | 540 |
| r23 at 0.40 | 7,053 | 1,441 | 1,879 | 397 |
| r23 floor (0.25) | 7,258 | 1,577 | 2,027 | 413 |

**11.5 Reading.**
- As registered (10.2), stage 1 decides nothing; it is the held-teacher reference set, and its measurement checks
  passed.
- The held teacher reaches 0.349 of the sampled C1 misses, 0.316 beyond what lowering r23's threshold to 0.25 gives
  (0.037). By subclass it reaches C1a_other 0.458 (D-i 0.540, P-i 0.299), 0.417 beyond the floor, and C1c 0.244.
- H, the C1a_other and C1c misses neither the floor nor the held teacher reaches, is -0.0086 on gsw_cha_2016 (interval
  [-0.0118, -0.0059]), beyond the -0.0042 bar of 10.4 even at the interval's upper end; under the larger gap of
  0.0069 its point is past that gap, but its interval's upper end (-0.0059) does not clear it. H is the ceiling of a
  source that reaches every remaining C1a_other and C1c miss: a crowd detector would need about 49 percent of exactly
  the misses the in-domain teacher fails on to make -0.0042 (0.0042 / 0.0086), and about 80 percent to make 0.0069. So a
  stage 2 design is worth writing, with that as its prior; whether a crowd detector reaches those misses is stage 2's
  question.
- The held teacher's C on gsw_cha_2016 (-0.0053) is part of what E1b's whole-label ceiling already counted, on other frames
  and by the nearest rule; it is not added to E1b's -0.0079.
- Post hoc, not tested: the teacher reaches C1b misses (a referee box near the player) at 0.679; its labels exclude
  referees, but some of those reaches may be a box on the referee assigned to the player (10.8's caveat). It draws more
  boxes and duplicates than r23 at 0.40 (7,628 against 7,053; 540 against 397) at about the same ghost proxy. The
  teacher's counts are after the court filter and r23's are not, so this comparison slightly favours the teacher.
- Limits as in section 8: raw reach, E1's transfer, post hoc classes, truth bias, the sample.

## 12. Stage 2 design draft (2026-11-28; not registered; held for Lucien's licence ruling; nothing downloaded)

A draft for the lead, to be registered by a dated amendment before any stage 2 download or number. It takes one of two
crowd detectors, chosen by that amendment:
- **S2-I, Iter-Deformable-DETR R-50 (CrowdHuman)**, if Lucien accepts the MIT grant that rests on its README sentence
  alone (no LICENSE file; P8_labelsource.md section 7; D160 puts a weights-licence question like this to him);
- **S2-C, CrowdDet RCNN EMD with RM (CrowdHuman)** otherwise: an MIT LICENSE file is present, and its weights, with no
  stated licence and trained on non-commercial data, follow the held teacher's MixSort precedent under D150 (1).

**12.1 Before any download.** A docs/DATA_SOURCES.md row with the chosen repository's licence text and CrowdHuman's
terms re-read verbatim (P8_labelsource.md section 2); the weights' URL and, after download, their SHA-256; a separate
environment outside the repository, with nothing installed into the frozen python or the teachers venv. A feasibility
probe on 5 of the stored frames, timed, comes first: Iter-Deformable-DETR's deformable attention is a CUDA op upstream,
so on this Mac it needs the pure PyTorch path, which may be slow; CrowdDet is a plain Faster R-CNN FPN. If the probe
projects more than 60 lock minutes for the 775 frames, the amendment says so and the lead decides.

**12.2 The operating point (fixed before any number).** P8_teachers.md's qualification (recall at r23's precision on
the 50 box-truth frames) failed the COCO person detectors (recall 0.035 and 0.060), which box spectators and benches
as well as players. A CrowdHuman detector boxes every person too, so that rule is likely to fail it for the same
reason, and choosing a threshold on the C1 frames is not allowed. Proposed instead: a fixed score of 0.5, with
0.3 and 0.7 reported as sensitivities, and the court filter of 10.8 before everything. The ghost proxy and the near-no-
player count carry the cost of non-player boxes.

**12.3 What runs.** The 775 stored PNG frames (hash-checked against stage 1's inputs), the same C1 misses, the same
one-to-one reach, floor, cost rows and transfer; the positive control on r23's hits at 0.90 again. The increment is the
C1 misses the crowd detector reaches that neither the floor nor the held teacher reaches; its ceiling is C of that
increment, with the joint and f-only intervals, the extremes, and the union-baseline pair (gaps 0.0021 and 0.0069).
The reading follows 10.6 with every zone's action, shown under both gaps.

**12.4 Cost.** No lock time for the frames' decode (they are stored). The crowd detector on 775 frames at the probe's
measured rate, under the lock; the report on the system python. hou_cle_2016: no data read.

## 13. Stage 2 registered: S2-C, CrowdDet (2026-11-28; the lead's ruling; before any download or number)

Where this section differs from section 12, it rules.

**13.0 The prior, first.** Stage 1's H on gsw_cha_2016 (-0.0086) is the ceiling of a source that reaches every
C1a_other and C1c miss that neither r23's floor nor the in-domain held teacher reaches. To make the -0.0042 bar, CrowdDet
would need to reach about 49 percent of exactly those misses; to make 0.0069, the gap under the transfer model's
residual, about 80 percent. They are the misses an in-domain player detector already fails on, and CrowdDet is an
out-of-domain (web images) person detector; that is the honest prior for this stage.

**13.1 The choice.** Stage 2 is S2-C, CrowdDet. Options weighed by the lead: waiting for Lucien's ruling on
Iter-Deformable-DETR's README-only grant would leave the machine idle until morning; CrowdDet has an MIT LICENSE file,
and its weights fall under the MixSort precedent of D150 (1), so it is within current rules. Iter-Deformable-DETR stays a
possible later S2-I if Lucien accepts its grant; it would then be reported beside CrowdDet as a second source, never
as a rescue of a failed S2-C.

**13.2 The source, read verbatim on 2026-11-28** (raw files fetched with curl, not a summarising fetch):
- Code: github.com/xg-chu/CrowdDet (Purkialo/CrowdDet now resolves there), default branch master; LICENSE "MIT
  License", "Copyright (c) 2020 Megvii Technology", the standard MIT grant over "this software and associated
  documentation files" (raw.githubusercontent.com/xg-chu/CrowdDet/master/LICENSE; identical bytes from the Purkialo raw
  URL and the GitHub API, spdx_id MIT). The code is cloned at the commit current at download, whose SHA is recorded.
- Weights: "RCNN EMD with RM (trained with PyTorch)", AP 0.9030, MR 0.4128, JI 0.8263, rcnn_emd_refine.pth
  (README.md's Google Drive link). Chosen over the MegEngine-converted row (0.9097, 0.4102, 0.8271) because it was
  trained by the code that will run it. The README states no licence for the weights and does not name their training
  data in words; its table is CrowdHuman's. SHA-256 recorded at download.
- Training data's terms (www.crowdhuman.org/download.html): "You will use the data only for non-commercial research and
  educational purposes." and "You will NOT distribute the above images." Use here is non-commercial research, and the
  weights are never redistributed or committed.
- The docs/DATA_SOURCES.md row is written in this commit, marked not downloaded.
- Held-out screen: CrowdHuman is web images whose games cannot be identified; a residual risk that a held-out game's
  image is among them is disclosed (D150 (2)).

**13.3 The environment.** A separate venv outside iCloud and outside the repository, with the CrowdDet code and weights
in the P8 data folder; nothing is installed into the frozen python or the teachers venv. Its Python, torch, torchvision
and OpenCV versions are recorded. CrowdDet's test path may assume CUDA; the runner maps it to the CPU through the
loader's map_location and the model's device only. Any further change to CrowdDet's code is listed in the record; a
change to its model or post-processing logic stops stage 2 for a dated amendment.

**13.4 The 5-frame probe (under the lock, before the full stage 2).** The first 5 frames of stage 1's inputs, hash-checked;
CrowdDet's own inference and post-processing (its test configuration's thresholds and set NMS, unchanged) on the CPU
with 4 threads; seconds per frame, boxes per frame, and that every frame yields boxes. Gate: the projection for the
775 frames is at most 60 lock minutes, else the lead decides. The probe's boxes are not scored.

**13.5 What runs (after the probe and the lead's go).** Stage 1's 775 stored frames and inputs (hash-checked); CrowdDet's
boxes with score at least 0.5, court-filtered (10.8); sensitivities at 0.3 and 0.7. The same statistics as stage 1:
one-to-one reach (assign_1to1), the floor, the cost rows, the positive control on r23's hits (gate 0.90). The increment
is the C1 misses CrowdDet reaches that neither the floor nor the held teacher reaches (stage 1's stored teacher boxes);
its ceiling C with the joint and f-only intervals, the extremes, and the union-baseline pair. The reading is 10.6's,
every zone with its action, shown under the gap of 0.0021 and under 0.0069. The ghost condition compares CrowdDet's
ghost proxy per sampled frame with the held teacher's on the same frames. Stage 2 runs only after stage 1's ml-review
passes, or after its fixes, if any, leave stage 1's inputs and teacher boxes unchanged.

**13.6 Cost and writes.** The probe about 2 lock minutes; the full pass at the probe's rate, cap 60; the report on the
system python. Records: reports/research/p8_c1reach_stage2_probe.json and p8_c1reach_stage2.json; CrowdDet's boxes in
the reference root's c1reach/. hou_cle_2016: no data read.

**13.7 Code note (2026-11-28; tools/p8_c1reach_s2.py written; nothing downloaded or run).**
- The clone is pinned to xg-chu/CrowdDet at e3620a6bb300256fb8b460948138839ca5dc7c5f (master on 2026-11-28, read
  through the GitHub API), and the runner refuses a clone at another commit or with local changes. The weights' SHA-256
  is a constant set by a dated amendment after the download; until then the runner refuses to start.
- The runner calls CrowdDet's own tools/inference.py get_data (reading each stored PNG) and post_process unchanged,
  with the rcnn_emd_refine test configuration as shipped: set NMS at 0.5, pred_cls_threshold 0.01, and
  visulize_threshold 0.3, so CrowdDet itself returns nothing under 0.3 and the 0.3 sensitivity is its own output floor.
  Its training boxes are CrowdHuman's full boxes (gt_boxes_name 'fbox' in that configuration). Its network uses
  torchvision's roi_align and nms only, with no compiled extension, and the shipped loader already maps to the CPU.
- The venv steps import nothing from the repository but the runner itself: the repository's top-level config.py would
  otherwise shadow CrowdDet's config module. The report step, on the system python, runs every guard.
- Under the larger gap (0.0069), the zones of 10.6 move with it: the kill line at -0.0069 and the bar 0.0021 below it,
  at -0.0090, the same distance as under the registered gap.

**13.8 The S2-C design review (2026-11-28; on 3777ec61, GO WITH CHANGES; before any download or number).** Where this
differs from 13.1 to 13.7, it rules. Its first item (the 49 and 80 percent prior) is 13.0.
- **The action follows the 0.0021 gap**; the 0.0069 gap is a sensitivity (kill at -0.0069, bar at -0.0090, the lead's
  confirmation). An increment that passes under 0.0021 but not under 0.0069 reads "passes, not robust to the transfer
  residual", and any training-arm design must carry that limit.
- **The headline increment is on C1a_other plus C1c**, the population of H and of the prior; C1b and C1a_own are
  reported apart. A registered sensitivity adds a referee mask: CrowdDet boxes at IoU 0.5 or more with r23's class-1
  (referee) boxes at 0.40 are dropped. Those boxes come from the stored candidates' "other": step A's inputs now carry
  them ("r23_ref"); for the stage 1 inputs already made, the report reads them from the same stored candidate files step
  A read and checks, frame by frame, that their R_post equals step A's.
- **The union merge rule.** A CrowdDet box joins the union only if no held-teacher box at 0.1471 has IoU 0.5 or more with
  it. The increment is the C1 misses the union reaches (one to one) and neither the floor nor the held teacher reaches.
  The ghost bar is on the union: at most twice the held teacher's ghost proxy per sampled frame (stage 1: 1,402 over
  775 frames, 1.81); the union's duplicates are reported.
- **The reading uses score 0.5 only**, applied after set NMS as upstream does; 0.3 and 0.7 are reported and cannot rescue
  a failed 0.5.
- **The port.** torch.load with weights_only=True first; if it fails, the run stops and the lead is told, with no
  fallback to weights_only=False on the pickle. load_state_dict is strict. The record holds the weights path, the
  checkpoint's keys and the use of its "state_dict" key, and the load method. The download records the file's SHA-256
  and size, both registered as constants by a dated amendment before the probe.
- **The probe's checks, besides timing:** (i) CrowdDet's one-to-one reach of r23's hits on its 5 frames, at least 0.90;
  (ii) CrowdDet's boxes per frame before and after the court filter, and its in-court ghost proxy per frame; (iii) the
  versions and the load method; (iv) the probe's boxes stored with a hash, and the full run's first 5 frames must
  reproduce them exactly, else it stops.
- **docs/DATA_SOURCES.md:** the weights' training data is "inferred" from the README's table, and D160 (Lucien treated a
  repository's MIT licence as covering its released weights) is cited beside the MixSort precedent.

**13.9 The code check of 656f88cd (2026-11-28; GO WITH CHANGES; before any download).** The venv, the clone and the
download wait for Lucien's yes; nothing of CrowdDet's is fetched until then.
- After its imports, the loader requires CrowdDet's config, network and nms_utils modules to come from the pinned
  clone's src folder, and records their paths.
- The probe passes only if every one of its frames yields boxes (13.4), besides the positive control and the timing.
- The record reports each population's loss to the scaling beside its increment: misses in windows whose sampled
  frames hold none of that population. From stage 1's inputs that is 61 of the 378 C1b and C1a_own misses, and 0 of the
  3,006 C1a_other and C1c misses.
- H is recomputed in the stage-2 record with the headline population's own scaling (H_per_population), since stage 1
  scaled all C1 subclasses together.
- Union losses are reported: C1 misses the held teacher reached that the union no longer reaches (a one-to-one
  assignment can move when boxes are added).
- Wording on 13.8's referee boxes: the frame-by-frame R_post check proves the stored candidate file is unchanged since
  step A read it; the frame's identity rests on R1.
