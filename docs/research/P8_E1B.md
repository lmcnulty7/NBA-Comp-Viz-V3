# P8 E1b: r23 fine-tuned on teacher boxes (design, before any compute)

Version 1, 2026-11-26. Design only: no Colab, no training, no model build. Written under docs/research/TRAINING_PROTOCOL.md
and the user-scope ml-training-rigor standard, after E1 (D171), its post-mortem (P8_E1.md, M1 to M9) and E1R
(P8_E1R.md on branch r-p8-e1r). Nothing below runs until training-design-reviewer has reviewed this version and the
lead has written its D line. The one measurement made for this design (section 2) reads stored files only: no model
ran and nothing under DATA_ROOT was read.

Status after review (2026-11-26): training-design-reviewer returned NO-GO on version 1 (moderate confidence). Its
first finding: the raw to build transfer of E1's gains, measured on the same frames, is lower than section 2 assumed
(placement about 0.32, presence about 0.54, raw losses carried at about 0.42), so the build-level ceiling of section 2
is too high. The lead's decision: test that first, before any other fix. Section 2b registers the minimum effect that
matters and a transfer-adjusted gate, written and committed before its numbers were computed. The reviewer's other
findings (pilot fold, checkpoint epochs, mask placement, duplicate guard, team margin, oracle gate, learning rate,
selection metric, classes and referees, checklist) are not fixed in this version; they matter only if section 2b's
gate passes.

## 1. Question and mechanism

Question: can a detector fine-tuned from r23 on boxes from the qualified outside teacher lower the missed-players row
on the reference platform (D175: detector and tracker on CPU) by an amount that matters, without giving back the
other rows?

What E1 and E1R leave open. E1 seed 0 (yolov8m COCO start, agreement pseudo-labels with mean boxes) passes the
registered pool on the reference platform, but its missed gain is small: -0.0067 pooled [-0.0120, -0.0016], -0.0019
on the two blind games, +0.0069 on gsw_cha_2016 alone, and -0.0107 to +0.0014 by fold on hou_cle_2016, where Stage 2
fails. It gains steadily on ghosts, position, near-field bias and team. Its labels had a ceiling near the noise
floor: 1,102 of 55,147 in-image players (0.020) were new positives, and the mean-box rule replaced 4,081 teacher boxes
whose feet were closer to SportVU.

Mechanism of E1b, three changes against E1:

1. Labels: every YOLOX-X SportsMOT box at its qualified threshold (0.1471, P8_teachers.md), in the teacher's own
   geometry. No agreement rule, no mean box, no SportVU filter. The 4,081 foot-placement corrections of E1's source 2
   are teacher boxes, so they are kept as drawn. This is distillation from the one teacher that qualified on the
   human box truth (+0.0419 recall at our precision); r23's own boxes are never labels (LABEL_SCHEMA.md).
2. Initialisation: r23 (models/player_detector.pt, sha256 4c886d5f...), keeping its 5-class head so every weight
   transfers. E1 lost 1,523 of r23's raw finds on the corpus frames (section 2); starting from r23 aims to keep them.
3. Players that no label box reaches are not taught as background: they are masked out of the training images
   (section 3.3).

Checkpoint and threshold are chosen against SportVU on training-game validation frames, not on pseudo-label mAP
(sections 8 and 9).

## 2. Upper bound (measured on 2026-11-26)

Tool: tools/p8_e1b_bound.py (rules in its docstring, committed in 2bfe0f61 before it ran); record:
reports/research/p8_e1b_bound.json. Frames: the 5,592 E1 corpus frames of the 3 pool games (development training
games). Boxes: r23 at 0.40 (corpus preds), r23 down to 0.10 and E1 seed 0 at 0.40 (M4's CPU runs on the corpus JPEGs),
the teacher down to its 0.05 floor (corpus preds). Rules: the scorer's (in-frame with the 10 px margin, a player is
reached when the nearest bottom-centre box foot lies within 2.5 ft), so r23's row is M3's "ours" row: the tool
stopped unless its per-game in-frame, r23-missed and teacher-missed counts equal postmortem_m3.json, and they do.
These are raw boxes with bottom-centre feet, not the tracked build rows (r23's CPU build misses 0.2449 on the pool
windows; raw here 0.2556).

Every r23 miss gets the first reach class that holds: the teacher at 0.1471 reaches it; the teacher between 0.05 and
0.1471 does; r23 between 0.10 and 0.40 does; some box of either model has its foot 2.5 to 5 ft away ("placement");
no box of either model within 5 ft ("none").

| Game | In-frame | r23 missed | Teacher | Teacher low | r23 low | Placement | None |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsw_bkn_2015 | 20,477 | 4,845 (0.2366) | 1,845 (38.1%) | 20 | 9 | 2,534 (52.3%) | 437 (9.0%) |
| gsw_cha_2016 | 13,414 | 4,747 (0.3539) | 911 (19.2%) | 6 | 13 | 2,970 (62.6%) | 847 (17.8%) |
| gsw_sac_2015 | 21,144 | 4,473 (0.2116) | 1,632 (36.5%) | 3 | 11 | 2,488 (55.6%) | 339 (7.6%) |
| Pooled | 55,035 | 14,065 (0.2556) | 4,388 (31.2%) | 29 (0.2%) | 33 (0.2%) | 7,992 (56.8%) | 1,623 (11.5%) |

Pooled shares with window intervals (2,000 resamples, seed 0, 175 corpus windows): teacher 0.312 [0.277, 0.347],
placement 0.568 [0.547, 0.589], none 0.115 [0.087, 0.152].

Ceilings, as missed rates on the same frames (a player counts as missed only when no source reaches it):

| | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 | Pooled |
| --- | --- | --- | --- | --- |
| r23 alone | 0.2366 | 0.3539 | 0.2116 | 0.2556 |
| r23 or the teacher at 0.1471 | 0.1465 | 0.2860 | 0.1344 | 0.1758 |
| plus the teacher's 0.05 to 0.1471 band | 0.1455 | 0.2855 | 0.1342 | 0.1753 |
| plus r23's 0.10 to 0.40 band | 0.1451 | 0.2846 | 0.1337 | 0.1747 |

The teacher-label ceiling is -0.0797 [-0.0840, -0.0756] pooled. It assumes a student that keeps every r23 find and
learns every teacher box: an upper bound, not a forecast.

Presence against placement (report only, added after the first run). Of the 4,388 teacher-reachable misses, 3,509
(80%) already have an r23 box within 5 ft: r23 boxes the player and the teacher's foot lands closer. Only 879 (0.016
of in-frame players) are players r23 has no box near at all. Of the 7,992 "placement" misses no model reaches, 7,162
(90%) have an r23 box within 5 ft.

By cause (pooled; miss rate, and the teacher-reachable share of those misses):

| Cause | Bin: in-frame, r23 miss rate, teacher-reachable share |
| --- | --- |
| Camera third | near 15,226, 0.251, 33%; middle 24,383, 0.202, 28%; far 15,426, 0.344, 33% |
| Expected height (6.5 ft player) | under 150 px 10,930, 0.370, 27%; 150 to 200 px 37,489, 0.221, 32%; 200 to 250 px 6,444, 0.265, 35%; 250 px and over 172, 0.250 |
| Image edge (foot within 40 px) | edge 705, 0.387, 22%; not 54,330, 0.254, 31% |
| Overlap (largest IoU of expected boxes) | none 27,687, 0.240, 22%; partial 19,404, 0.251, 35%; heavy 7,944, 0.320, 47% |

What teacher-box labels do to r23's finds: of 40,970 players r23 reaches, the teacher at 0.1471 reaches 39,827
(kept), has a box 2.5 to 5 ft away for 1,089 (displaced) and has no box within 5 ft for 54 (dropped). Unlike E1's
agreement labels, teacher-box labels leave almost none of r23's finds unlabelled.

E1's raw gain, decomposed (report only, M4 boxes at 0.40 on each fold's left-out game): E1 reaches 2,382 of r23's
misses, 2,080 of them (87%) placement (r23 had a box within 5 ft) and 302 presence, and loses 1,523 of r23's finds;
the net 859 is M4's -0.0156.

Reading:

- Share of misses any label recipe from these models can reach: 31.2% with teacher boxes at the qualified threshold
  (0.080 of in-frame players). Truth-confirmed low-confidence teacher boxes and r23's own low band add 0.4% (0.001):
  not worth a SportVU-filtered label source, so E1b has none. 68% of misses (placement 56.8%, none 11.5%) are out of
  reach of any box these models draw: a box is near but its foot is 2.5 to 5 ft off, or nothing is boxed.
- The reachable part is mostly placement, not presence: 0.064 of in-frame players against 0.016. E1's raw gain was
  87% placement too, and placement gains shrink in the build because r23's foot stabilisation already corrects part
  of them (M5: foot_stab lowers r23's missed rate by 0.011 more than E1's). This is the leading explanation (post hoc)
  for E1's raw gain of -0.0156 becoming -0.0067 on the reference platform.
- Even at the label ceiling about 0.17 of in-frame players stay missed on raw boxes. A detector recipe alone cannot
  reach Level 2's 5% target; the placement problem (foot point, near-field geometry, or truth and timing error, not
  separated here) is the larger lever and needs its own measurement (section 11).
- The bound is above the needed effect (section 6): with presence transferring fully and placement at E1's raw to
  build ratio (0.43), the ceiling is about 0.016 + 0.064 x 0.43 = 0.044 at build level against 0.020 needed. So the
  bound permits E1b, by a factor of about two, not more.

## 2b. Minimum effect that matters, and the transfer-adjusted gate (registered before computing)

Minimum effect that matters. The registered adoption rule (section 5) needs build-level missed at most -0.010 pooled,
with the window and the game intervals wholly below zero. With 3 games the game interval is about the range of the
three game effects, so every game must improve at build level, not only the pool. A trained student realistically
reaches about half of what its labels offer after transfer (E1, the only measured case, kept less than its labels'
nominal headroom at build level). So the labels' transfer-adjusted build ceiling, for a student that imitated them
perfectly, must be at least twice the bar: at most -0.020 pooled, and at most -0.010 on every game so that half of it
still lies below zero on each game.

Transfer-adjusted ceiling (tool: tools/p8_e1b_bound.py transfer; rules in its docstring, committed with this section;
record: reports/research/p8_e1b_transfer.json):

- Frames: per pool game, the corpus frames that both reference-platform builds processed (r23: builds/e1m6_r23_detcpu
  and builds/m9_r23_detcpu__<game>; E1 seed 0 of the fold named after the game: builds/e1m6_e1_gsw_bkn_2015_detcpu and
  builds/e1r_e1_<game>_detcpu). In-frame players with the scorer's 10 px margin.
- Raw reach: nearest bottom-centre foot within 2.5 ft; r23 and E1 at 0.40 from M4's CPU run on the corpus JPEGs, the
  teacher at 0.1471 from the corpus preds. Build reach: nearest foot_stab of the sidecar boxes within 2.5 ft.
- A raw gain is presence when r23 (0.10 floor) has no box foot within 5 ft of the player, else placement.
- Transfer fractions, measured per game from E1: f_pres and f_plc, the shares of E1's presence and placement raw
  gains on which, at build level, E1 reaches the player and r23 does not; f_loss, the share of E1's raw losses on
  which, at build level, E1 misses the player and r23 reaches it.
- L1's net per game: P presence gains and Q placement gains (r23 raw misses the teacher reaches), and L losses (r23
  raw finds the teacher does not reach, the displaced and dropped players of section 2).
- Perfect-imitation build ceiling per game: C = -(P f_pres + Q f_plc - L f_loss) / N. Pooled: the sum of the three
  numerators over the sum of N, each game with its own fractions.
- Gate, on the point estimates: C pooled at most -0.020 AND C at most -0.010 on every game. If any game is above
  -0.010, E1b with this label set stops before any unit is spent.
- Report only, beside the gate: the optimistic student-free variant C_opt = -(P f_pres + Q f_plc) / N (keeps every
  r23 find, so no loss term); half of C; E1's own raw and build differences on the same frames beside what the
  transfer model predicts for E1 (a check of the model); window bootstrap intervals (2,000, seed 0, windows resampled
  within each game, fractions recomputed in each resample).
- Caveats fixed now: the fractions come from one arm (E1, seed 0) whose gains were mostly placement; applying them to
  L1 assumes L1's gains of each kind carry into the build like E1's did. The transfer of a presence gain is estimated
  from few events on some games.

## 3. Recipe (fixed)

### 3.1 Data and folds

- Corpus frames: the E1 corpus images (every accepted Mac camera-truth frame of gsw_bkn_2015, gsw_cha_2016 and
  gsw_sac_2015), relabelled; written to a new folder e1b_corpus/ in the P8 workspace, never the repository.
- Folds: leave one game out (D156), named after the left-out game. A fold's left-out game never enters its training,
  validation, checkpoint choice or threshold choice; the training code asserts it (as tools/p8_e1_colab.py does).
- Validation windows: the last 10 percent of each training game's windows in time order, whole windows
  (tools.p8_e1_colab.split_windows), excluded from training. They serve the truth-aligned validation of section 8.
- hou_cle_2016, gsw_ind_2016, every held-out game of sportvu/splits.json and the D85 test set are never read by any
  E1b training or selection code. tools.p8_common.guard and tools.research_testset.refuse_cv_test run on every game,
  SportVU id and video id list.

### 3.2 Labels (L1)

- Every YOLOX-X SportsMOT box with confidence at or above 0.1471 in the corpus preds, class 0 (player), YOLO format,
  the box as the teacher drew it. Nothing else: no r23 box, no agreement rule, no mean box, no SportVU-selected box.
- Licence carried from the teacher: SportsMOT weights are CC BY-NC 4.0, as for E1 (P8_teachers.md).

### 3.3 Players no label reaches: masked, with how Ultralytics 8.4.48 allows it

Ultralytics 8.4.48 has no ignore region. In v8DetectionLoss.get_assigned_targets_and_loss (ultralytics/utils/loss.py)
the classification BCE runs over every anchor and every class with target 0 wherever the TaskAlignedAssigner assigns no
ground truth, so anything unlabelled is background. There is no ignore class and no per-anchor weight from the label
file. A loss-level ignore needs a custom criterion and a way to carry ignore boxes through augmentation: as an extra
class id it fails the dataset check (data/utils.py verify_image_label asserts class ids below nc), and as a sixth class
it changes r23's head shape so the class branch would not transfer (intersect_dicts keeps only matching shapes).

E1b therefore masks in the image, which needs no framework change:

- Masked player: a SportVU player in the image (tools.p8_teachers.targets_in_frame, no margin) that no label box
  reaches one to one within 5 ft (the 2.5 ft assignment, then the unclaimed label boxes within 5 ft, as
  tools.p8_e1_postmortem.own_label_within). A player whose label box has its foot 2.5 to 5 ft off is labelled, not
  masked.
- Region: the player's expected box (pinhole camera of the camera truth, 6.5 ft nominal height, width 0.4 of the
  height, bottom centre at the projected foot), widened by 10 percent on each side, clipped to the image. Its pixels
  outside every label box are painted (114, 114, 114), the value Ultralytics fills mosaic and affine borders with.
  A frame with no camera pose is dropped from training and counted.
- Expected size, from section 2: about 3 to 5 percent of in-frame players (the 1,623 "none" misses plus the 54
  dropped finds, plus players no teacher box reaches within 5 ft though r23's does). The written count is a
  pre-flight output with a kill rule (section 7).
- Expected value: small, and mainly protective. The masked players are mostly ones no model boxes, and E1's net change
  on such players was -50 (P8_E1.md, implication 3). It is kept because it is cheap and because it stops the student
  from being taught that the hardest visible players are background.

### 3.4 Model, classes and optimiser (the framework's defaults read, then fixed)

- Start: r23's checkpoint, all 5 classes (player, referee, ball, rim, number) kept in data.yaml so the whole head
  transfers. Labels use class 0 only. The other classes receive only negatives and drift; the pipeline tracks class 0
  only (detect/tracker.py, classes=[config.PERSON_CLASS]), so they are unused. Pre-flight proves the transfer (section
  7, P3).
- What "auto" resolves to in 8.4.48 (engine/trainer.py build_optimizer): iterations = ceil(N / max(batch, nbs=64)) x
  epochs; MuSGD at lr 0.01 above 10,000 iterations, else AdamW at lr 0.002 x 5 / (4 + nc). E1 (nc 1, 3,230 training
  images, 100 epochs, 5,100 iterations) therefore ran AdamW at 0.002 (its results.csv shows 0.00196 after warmup). For
  E1b (nc 5, about 3,100 to 3,800 images, 30 epochs) auto would give AdamW at 0.00111.
- Fixed explicitly: optimizer AdamW, lr0 0.001, lrf 0.1 (linear decay to 0.0001), momentum 0.9, weight decay 0.0005,
  epochs 30, warmup_epochs 1, warmup_bias_lr 0.0, imgsz 640 (r23's), batch 16, mosaic 1.0 with close_mosaic 10, other
  augmentation at defaults, patience off (selection is section 8), save_period 5, seed fixed per run, deterministic.
  warmup_bias_lr is set to 0 because 8.4.48's default (0.1) ramps the bias group down from lr 0.1 during warmup, a
  large AdamW step for a converged head.
- Not chosen: imgsz 1280 (closer to the teacher's 1440 test width, but about four times the CPU build cost, and only
  20 percent of in-frame players are under 150 px tall in the 720p frame); freezing the backbone (no evidence either
  way; one fewer fork).

## 4. The cheapest falsifier: the label oracle and its gate

Before any upload, the written L1 label files are scored as if they were a detector's boxes, with M3's rules
(tools.p8_e1_postmortem.oracle_counts: in-frame with the 10 px margin, missed at 2.5 ft nearest foot, ghosts at 3.0
ft), against r23's corpus boxes at 0.40 on the same 5,592 frames, window bootstrap (2,000, seed 0).

Gate, all must hold, else E1b stops:

1. Pooled missed, labels minus r23: at most -0.045, with the window interval's upper end at most -0.035.
2. Every game: at most -0.030.
3. Pooled ghosts, labels minus r23: at most 0 (point).
4. Boxes per in-frame player: at most 1.10 times r23's.
5. Mechanics: the written files reproduce the corpus teacher row of M3 exactly (in-frame, missed, ghosts and boxes
   per game), and the image count, label count and masked count agree with the manifest.

Why -0.045: the needed build effect is 0.020 (section 6), and E1 kept about 0.45 of its raw gain at build level
(-0.0156 to -0.0067). A label set with less headroom than about 0.020 / 0.45 cannot be expected to deliver it, since a
student rarely beats its labels.

Disclosure: L1 is exactly M3's teacher row, already measured: missed 0.1966 against 0.2556, -0.0590
[-0.0642, -0.0538]; by game -0.0716, -0.0444, -0.0559; ghosts 0.1397 against 0.1765. The gate was set knowing these
values, so for L1 it is a check that the written files are the intended labels (item 5), not a blind test. The blind
falsifiers are the pilot gates (section 7).

## 5. Decision metric and adoption rule

Decision metric: the pinned scorecard (tools/p8_scorecard.py, D88) through E1R's scorer (score_window, truth camera,
foot_stab, tools.p8_e1_score.window_stats), on the label windows with a registered r23 sidecar (83, 73 and 82 on the
pool games; 93 on hou_cle_2016), paired against r23's reference-platform builds (builds/e1m6_r23_detcpu and
builds/m9_r23_detcpu__<game>), with E1R's per-window checks (device and weights records, equal processed frame sets,
identical court H, equal in-frame counts, equal gate-empty sets). Window and game bootstrap, 2,000 resamples, seed 0.

Adoption rule for a seed's pool (E1b minus r23), all must hold:

1. Missed (the targeted row): pooled at most -0.010, window interval wholly below zero, game interval wholly below
   zero (with 3 games: every game's difference below zero).
2. Untargeted rows, non-inferiority against r23 on the window interval's bad-side end: ghosts at most +0.010;
   position p50 at most +0.05 ft; p90 at most +0.10 ft; near-field bias at most +0.10 ft; team at least -0.03.
   Coverage holds its floor (registered).
3. Denominator guard: E1b's non-referee boxes per in-frame player at most 1.05 times r23's, and its duplicate rate
   (in-frame players with two or more box feet within 2.5 ft, on the scored frames) at most r23's plus 0.01. A missed
   gain made by extra boxes is not a find.

The margins are about the size of the detector-device change alone on r23 (M9: ghosts +0.003 to +0.010, p50 and p90
-0.002 to -0.039 ft, bias within 0.01 ft, team -0.011 to +0.038), that is, changes the project already treats as
platform-sized. E1 improved these rows, but E1b is compared with r23, not with E1, and the skill's point applies: an
every-row-must-improve rule has little power for a change that targets one row. This replaces D5's "every row improves
on the point estimate" for E1b only, which needs the lead's D line. The registered D5 verdict (tools.p8_e1_score.report,
unchanged) is computed and reported beside it, as E1R did.

Adoption overall: D85 (3) asks 3 seeds, each seed's pool passing. Adoption here means E1b passed its registered rule;
replacing the production detector is a separate decision.

Stage 2 rule (hou_cle_2016, the only non-GSW arena, seed 0 only, each fold's model): mean of the 3 folds' missed
differences at most -0.005; no fold's missed window interval wholly above zero; the untargeted margins of item 2 on
each fold. hou_cle_2016's window intervals are wide (E1R: plus or minus 0.013 to 0.015 a fold), so this stage checks
direction and harm, not size.

## 6. Expected effect, noise floor and minimum detectable effect

Noise, from E1R on the reference platform:

- Window sampling: pooled missed window interval half-width about 0.0052 (SE about 0.0027); per game 0.0055
  (gsw_bkn_2015), 0.0057 (gsw_cha_2016), 0.0109 (gsw_sac_2015); hou_cle_2016 about 0.013 to 0.015 a fold.
- Training variation: the 3 E1 folds scored on hou_cle_2016 (models trained on different game pairs, same windows)
  give -0.0012, -0.0107, +0.0014, SD about 0.0064. No second seed exists, so this is the closest estimate of
  run-to-run variation; it mixes training data with seed.
- Game spread: E1's per-game differences -0.0147, +0.0069, -0.0074, SD about 0.011.

Minimum detectable effect: pooled over 3 fold models, SE about sqrt(0.0027^2 + 0.0064^2 / 3) = 0.0046, so about 0.013
for the pooled window interval to exclude zero at 80 percent power. The game interval needs every game below zero:
with a game SD of 0.011 that holds about 90 percent of the time at a mean effect of 0.020 and about 76 percent at
0.015. The effect that matters and the effect the rule can detect are both about 0.020 at build level; the adoption
bar of -0.010 pooled sits below it only as a floor.

Expected effect, with its reasoning: the label oracle offers -0.059 raw. If the student reaches about half of it (E1
reached about all of its honest headroom, but a 640 px yolov8m distilling a 1440 px YOLOX-X is a harder transfer),
the raw gain is about -0.030, of which about a quarter is presence (section 2). With presence kept in the build and
placement kept at about 0.43, the build gain is about 0.008 + 0.022 x 0.43, about -0.017. Plausible range -0.010 to
-0.025: centred near the needed effect, not clearly above it. The chance that the pool rule passes is moderate (a
judgement, about one in two). That is why the pilot is cheap and its gate is one E1 fails (section 7).

## 7. Ladder, gates and kill criteria

Pre-flight on the Mac, no GPU (code reviewed and tested first; any check that reads DATA_ROOT or runs a model goes
under the heavy-job lock, one job at a time):

- P1. Write e1b_corpus (labels L1, masked images, manifest, meta with the teacher and r23 sha256, thresholds, code
  commit, the masking counts). Kill: more than 10 percent of in-image players masked, or any count mismatch.
- P2. Label oracle and its gate (section 4).
- P3. Initialisation check: the model the trainer builds from r23 with E1b's data.yaml, saved before any optimiser
  step, run on 20 corpus frames on CPU at the 0.10 floor, reproduces r23's raw class 0 boxes (coordinates within 0.01
  px, confidences within 1e-4). Kill on any difference: the head did not transfer.
- P4. Truth-validation packs per fold (frames of the validation windows: image size, camera truth, the 10 mirrored
  SportVU positions) and r23's reference rates on them (missed and ghosts at 0.40 from the corpus preds).
- P5. Baseline through E1b's build path: r23 on gsw_cha_2016_s00_w00 with the threshold patch set to 0.40 is
  byte-identical to builds/m9_r23_detcpu__gsw_cha_2016 (as E1R's smoke gate 1), and E1R's gate 4 regression
  (p8_e1_m6.json reproduced) passes on E1b's scorer.
- P6. Mechanics: a rendered count check (each label file's boxes equal the manifest; masked pixel share per frame
  recorded); guards on every list.

Smoke (Colab, about 0.3 T4-hours): fold gsw_sac_2015, 64 training images, 3 epochs; then 32 images for 30 epochs.
Gates: the loss falls over the 3 epochs; the 32-image run reaches box and class loss under half their first-epoch
values; the truth-validation callback writes its file; the checkpoint downloads with a matching sha256; on the Mac it
builds one window (gsw_sac_2015_s00_w00) through E1b's path and that window passes every scorer check. Any failure
stops the ladder until fixed.

Pilot (the cheapest run that shows direction): fold gsw_sac_2015, seed 0, the full E1b schedule (30 epochs; it is a
short fine-tune, so the pilot model is also seed 0's model for that fold). Chosen in advance because E1, the recipe
being replaced, fails both pilot gates on this fold (raw -0.0162, build -0.0074) while the game is neither E1's worst
nor the most inspected (gsw_bkn_2015 carried M6 to M8).

- In-run kill (Colab callback, section 8): at epoch 10, truth-validation missed at the iso-ghost threshold worse
  than r23's on the same frames by more than 0.010. Stop the run; no more units on that fold.
- Selection (Mac CPU, section 9): if no saved checkpoint has an eligible threshold, stop.
- Raw gate (Mac CPU, about 15 minutes): E1b at its threshold against r23 at 0.40 on gsw_sac_2015's 2,151 corpus
  frames, M4's rules. Pass: missed at most -0.020 with the window interval wholly below zero; ghosts at most +0.005;
  boxes per in-frame player at most 1.05 times r23's. Fail: stop, no build.
- Build gate (Mac CPU, about 2.4 lock hours, 82 windows): missed at most -0.010 with the window interval wholly below
  zero; the untargeted margins and the denominator guard of section 5 on this game's window intervals.

Pass both: train folds gsw_bkn_2015 and gsw_cha_2016 (seed 0), select, raw-check (the same raw gate, recorded and
used as a stop), build, and score the pool with the section 5 rule. Then Stage 2 on hou_cle_2016 with all three folds,
only if the pool passes. Seeds 1 and 2 (pool only, D85 (3)) only if seed 0 passes the pool and Stage 2. Seeds are
never added to rescue a failed gate, and extra seeds are not run once the rule can no longer pass (D171).

Kill criteria at any stage: a guard or provenance check fails (weights sha256, run record, fold, seed, device record);
a window build fails twice; any per-window scorer check fails; the left-out game appears in a fold's training or
validation lists; Colab spend for E1b passes 25 units without a decision.

## 8. Truth-aligned validation during training, and the checkpoint rule

- Frames: the fold's validation windows (training games only). On Colab a callback (on_fit_epoch_end) runs at epochs
  1, 5, 10, 15, 20, 25 and 30: the EMA model predicts class 0 at a 0.10 floor (NMS 0.45, imgsz 640) on those frames,
  and missed and ghosts are computed with M3's rules from the packed camera truth and SportVU positions, at every
  threshold from 0.25 to 0.70 by 0.05, beside r23's rates. Written to truth_val.csv in the run folder. Epoch 1 is there
  to catch a warmup collapse. This is a monitor and the in-run kill; it does not choose anything.
- Checkpoints: save_period 5 keeps epochs 5 to 30 (six files); all are downloaded with their sha256.
- Choice (Mac CPU, the reference platform, under the lock, about 15 minutes a fold): each saved checkpoint runs on the
  same validation frames; its score is the missed rate at its iso-ghost threshold (section 9); the lowest score wins,
  ties to the earlier epoch. Ultralytics' best.pt is not used: 8.4.48 picks it by mAP50-95 alone (utils/metrics.py
  fitness weights [0, 0, 0, 1]) on the pseudo-label validation split, which measures agreement with the labels. Its
  epoch is recorded only.
- Learning curves read after each fold, and written in the result: the train and pseudo-label validation losses, and
  the truth-validation missed rate by epoch. Flags: truth-validation missed rising from epoch 5 (forgetting r23's
  finds), losses flat from the start (rate or data problem), a NaN.

## 9. Operating point

A fine-tuned model's scores are not on r23's scale, so 0.40 is not carried over. On the fold's validation frames, on
the Mac CPU, for each saved checkpoint: the threshold is the lowest value in {0.25, 0.30, ..., 0.70} whose ghost rate
is at most r23's ghost rate at 0.40 on the same frames (iso-ghost). The floor 0.25 keeps BoT-SORT's first stage as it
is (track_high_thresh and new_track_thresh are 0.25 in 8.4.48), so every passed box can start a track as now. The
chosen checkpoint's threshold goes into every build of that fold through a tools-side patch of config.PLAYER_CONF in
the build child (no frozen file changes), recorded in each window's detector record. Nothing about it reads the
left-out game or hou_cle_2016.

## 10. Budget

Colab at about 1.15 units a T4-hour; training time from E1's measured epochs (about 140 to 170 s an epoch for 3,200
to 3,800 images), plus about 0.3 T4-hours a run for setup, transfers and callbacks. Mac lock hours at about 0.44 s a
processed frame on the CPU detector (M6, M9, E1R).

| Stage | Colab | Mac lock hours |
| --- | --- | --- |
| Pre-flight P1 to P6 | 0 | about 0.7 |
| Smoke | about 0.4 units | about 0.3 |
| Pilot, fold gsw_sac_2015 (1.6 T4-hours) | about 1.9 units | selection and raw gate 0.5; build 2.4 |
| Seed 0, folds gsw_bkn_2015 and gsw_cha_2016 (3.2 T4-hours) | about 3.7 units | selection and raw 1.0; builds 4.8 |
| Stage 2, hou_cle_2016, 3 folds | 0 | about 8.7 |
| Seeds 1 and 2, pool only (9.7 T4-hours) | about 11.2 units | about 15.8 |
| Total | about 17 units; 22 with a 25 percent reserve for lost runtimes | about 34 |
| To the pilot's verdict | about 2.3 units | about 3.9 |

The cap left is about 40 units, so the whole ladder fits with the reserve. A faster build route (the fixed MPS path,
queue item 6) is not assumed: it failed its parity rule (D174 (5)) and registered builds stay on CPU (D175).

## 11. What would change the plan

- P2 fails on the written files: a label-writing bug; fix the tool, no training. P3 fails: fix the loader.
- In-run kill, or no eligible checkpoint: the fine-tune forgets r23 or raises ghosts at every threshold. Stop. A new
  version (for example a lower rate) needs its own review; the run is not repeated larger.
- Raw gate fails with a raw gain between -0.010 and -0.020: E1b reproduces E1-sized gains. Close E1b and move to the
  placement problem, which holds 57 percent of r23's misses: first a cheap measurement on the same corpus frames of
  the direction and size of the foot offset for placement misses (systematic, so learnable or correctable, against
  scattered, so truth or timing noise), then the near-camera bias correction (Level 2 step 5).
- Raw gate passes and the build gate fails: the gain does not survive tracking and foot stabilisation. Diagnose with
  M5's read-only bottom-centre rescore of the pilot build (minutes); if foot_stab absorbs the gain, E1b's value is in
  placement that the pipeline already partly corrects, and the placement route above is the next bet.
- Pilot passes and the pool fails on missed: report per game; with 3 games one game can sink it, which is the
  claim being tested. No seeds 1 and 2.
- Pool passes and Stage 2 fails: "the pass does not carry to the new arena"; seeds 1 and 2 wait for the lead.
- An untargeted row breaks its margin while missed passes: not adopted; the result says which row and why.

## 12. What was already seen (forking paths)

- E1b's recipe was chosen after E1's result, the M1 to M9 post-mortem and E1R's Stage 1 and 2 results, and after the
  upper bound of section 2. The label oracle gate was set knowing L1's oracle values (section 4).
- The pilot fold was chosen knowing E1's per-game results (that is its stated reason).
- The pool games are the label-source games; in each fold the left-out game's labels are never read by that fold, but
  its corpus frames shaped section 2. hou_cle_2016 has no corpus frames and is the only game no E1b choice read; r23's
  and E1's rows on it are known.
- The camera truth was fitted from r23's MPS builds (D98, D99); both arms share it.
- The validation windows (training games) choose the checkpoint and the threshold: about 290 to 420 frames a fold, a
  small set, so some selection optimism is expected there and none can reach the scored games.

## 13. Code to write and review before any compute

- tools/p8_e1b_labels.py: P1, P2 (gate in code), P4, P6; tests on synthetic frames (masking geometry, label lines,
  oracle counts equal to tools.p8_e1_postmortem's).
- tools/p8_e1b_colab.py: the fold run from r23 with the fixed recipe, the callback, the in-run kill, the run record
  (recipe, resolved optimiser, versions, splits, data hashes, the r23 sha256), the left-out game assertion; P3's saved
  pre-step model.
- tools/p8_e1b.py: selection and raw gate (Mac CPU), build (E1R's path with E1b's weights and threshold patch, folder
  names e1b_<fold>_seed<S>_detcpu[__hou_cle_2016]), score with E1R's checks, report with the section 5 rule, the
  registered D5 verdict beside it, and the stage order enforced in code (no build before a passed raw gate; no other
  fold before a passed pilot; no hou_cle_2016 before a passed pool; no seed 1 before Stage 2).
- Reused unchanged: tools/p8_e1_pack.py, tools/p8_colab_xfer.py (hash checks), the Colab runbook (P8_E1_colab.md
  section 5: poll at most every 5 minutes, chain the upload into the runtime job, D179).
- No Drive mount path typed in the new code (paths come in as arguments, as on API runtimes). On v4-research today
  tests/test_export_contract.py::test_owned_files_hold_no_dash_and_no_private_string already fails on
  tools/p8_e1_colab.py for its Drive constants; E1b must not add to it.

## 14. Not measured, or open

- Whether placement misses are truth or timing error or a learnable foot offset: not separated (section 11 proposes
  the measurement). The bound counts them as unreachable by labels from these models.
- How much of a raw gain survives the build for E1b's boxes: estimated from one arm (E1); the pilot measures it.
- Seed-to-seed variation: no second seed of anything exists; the fold spread on hou_cle_2016 stands in.
- The masking count is estimated (3 to 5 percent), measured at P1.
- Whether YOLOX-X SportsMOT boxes referees: not measured; the iso-ghost threshold and the ghost margin bound its
  effect.
- The section 2 rates are on camera-truth frames, which favour frames where the pipeline works; the miss share on all
  live frames is likely higher.

## 15. Ledger line, to fill when E1b ends

| Date | Run | Stage | Compute | Outcome against its gate | Lesson |
| --- | --- | --- | --- | --- | --- |
| 2026-11-26 | E1b design | design and upper bound | none (stored files, seconds) | teacher-label ceiling -0.080 raw, 31 percent of misses, mostly placement | the reachable part of misses is mostly foot placement, which the build partly corrects already |
