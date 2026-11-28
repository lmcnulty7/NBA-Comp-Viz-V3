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
