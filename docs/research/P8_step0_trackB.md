# P8 step 0, Track B: labels without synced footage (draft for docs/research/P8_step0.md)

Drafted 2026-11-21 under PLAN.md D128 to D137, for the lead to fold into docs/research/P8_step0.md. Plan and measure
only: no training, no model change. Every number cites a repository file; a number no file records is marked "to
check". Files cited most: docs/research/P8_teachers.md (teacher qualification and the source 2 count),
docs/research/P8_baseline.md (scorecard v2 on training games), docs/research/P8_E2.md (events),
docs/research/P8_FEASIBILITY.md (targets, bases b1 to b5, the first ranking), docs/DATA_SOURCES.md (licenses).

## 1. Sources and licenses

One line each; docs/DATA_SOURCES.md holds the full rows and the dates the terms were read.

- SportVU game logs (github.com/linouk23/NBA-Player-Movements): positions and ids for training games; the license of
  the repository and the NBA's rights are to check (DATA_SOURCES.md, first row).
- basketball-reference play-by-play: scoring truth only; its terms forbid training AI models, so it is never a
  training label without Lucien's decision (PLAN.md D84; DATA_SOURCES.md).
- YouTube and Bilibili uploads (Track A): research and education, no video redistributed (DATA_SOURCES.md; CLAIMS.md
  caveat 4).
- SportsMOT: CC BY-NC 4.0, non-commercial, no redistribution without permission (DATA_SOURCES.md).
- MixSort YOLOX-X trained on SportsMOT (the qualified teacher): code MIT; weights carry no stated license and are
  treated as CC BY-NC 4.0, non-commercial, never redistributed (DATA_SOURCES.md).
- RF-DETR-L: Apache-2.0 (failed qualification) (DATA_SOURCES.md).
- YOLO11x: AGPL-3.0 (failed qualification) (DATA_SOURCES.md).
- DeepSportRadar: CC BY-NC-SA per FIX_PLAN.md D5, **not confirmed at the source**: neither its site nor its GitHub
  organisation stated it on 2026-11-21, and the Kaggle page is unread (DATA_SOURCES.md). Not downloaded; to check
  before any use.
- Existing human labels (the 50 box-truth files, the 2026-09 audit of 300 labels): used to qualify teachers and check
  the scorecard only; no new human labels (D132).

## 2. Teacher qualification

From docs/research/P8_teachers.md (rules fixed in tools/p8_teachers.py before any teacher ran; results in the P8
teacher workspace, results.json).

- Truth: the 50 box-truth files, 453 player boxes from 7 clips of 2 games; the guard dropped none. The files hold
  players only (referees unboxed), so our detector's player boxes are the comparator; that choice was fixed before
  any teacher ran, with the reason in the tool.
- Rule: greedy matching at IoU 0.5 (as colab_autolabel.py --qualify); each teacher's threshold swept from 0.05; a
  teacher qualifies when its recall at our detector's precision beats our recall and the 2,000-replicate frame
  bootstrap interval of the difference lies wholly above zero. Every model on CPU.

| Model | Recall at our precision (threshold) | Teacher minus ours [interval] | Qualifies |
| --- | --- | --- | --- |
| Our detector (player boxes, conf 0.40) | precision 0.9661 [0.9487, 0.9823], recall 0.9426 [0.9152, 0.9663] | | comparator |
| YOLOX-X SportsMOT | 0.9845 (0.1471) | +0.0419 [+0.0229, +0.0644] | yes |
| RF-DETR-L | 0.0353 (0.8588) | -0.9073 [-0.9326, -0.6882] | no |
| YOLO11x | 0.0596 (0.8456) | -0.8830 [-0.9202, -0.5245] | no |

- YOLOX-X boxes away from every box of ours: 13, of which 12 are true players (precision 0.9231 [0.7273, 1.0]).
- Open: whether the two box-truth games are among SportsMOT's training videos is not checked (P8_teachers.md). The
  evidence is 50 frames from 2012 and 2016 broadcasts, not the SportVU season.
- The 2026-07-18 Grounding DINO run failed the same gate (DEVLOG.md 2026-07-18; LABEL_SCHEMA.md rule 1).

## 3. The D131 pseudo-label rule, as three sources

PLAN.md D131: pseudo-labels come from (1) agreement boxes of our detector and an outside teacher, (2) outside-teacher
boxes at SportVU-projected feet on synced training games, and (3) boxes two independent outside teachers agree on
in unsynced footage; disagreements are dropped. Only (2) and (3) can surface our detector's misses (LABEL_SCHEMA.md).

| Source | Status | Measured | Still needed |
| --- | --- | --- | --- |
| 1. Agreement, ours and YOLOX-X | measured on the same frames as source 2 | gsw_bkn_2015, 400 frames: 3,626 agreement boxes, 9.065 a frame [8.9675, 9.1575]; 2,989 have a foot within 2.5 ft of a SportVU player, precision against SportVU 0.8243 [0.8080, 0.8398]. gsw_ind_2016 apart (D98): 3,615 boxes, 9.0375 a frame [8.9525, 9.12]; 2,192 within 2.5 ft, 0.6064 [0.5934, 0.6195] | a teacher pass over every accepted frame; precision on the human box truth not counted (to check) |
| 2. YOLOX-X at SportVU feet, synced training games | measured on one game | gsw_bkn_2015, 400 of 2,075 camera-truth frames: our detector misses 1,050 of 3,953 in-frame players (0.2656 [0.2497, 0.2818]); YOLOX-X recovers 400 of them (0.3810 [0.3437, 0.4175]), 1.0 a frame. gsw_ind_2016 apart (D98): 273 recovered, 0.6825 a frame (P8_teachers.md) | camera truth on more training games (gsw_cha_2016 and gsw_sac_2015 have none in the Mac workspace, P8_baseline.md); a teacher pass over every accepted frame |
| 3. Two teachers agree, unsynced footage | blocked | none | a second qualified teacher (none yet: RF-DETR-L and YOLO11x fail, Grounding DINO failed in 2026-07); unsynced footage from Track A's search |

Source 1 rule (tools/p8_teachers.py misses-agree, output agree_results.json in the P8 teacher workspace): on the 400
sampled camera-truth frames per game of source 2, our player boxes (conf 0.40) and YOLOX-X's (0.1471) are matched one
to one, greedily in our confidence order, at IoU >= 0.5; the label is the mean box of the pair. Its foot (bottom
centre, through the camera truth) counts as confirmed when a Hungarian match puts it within 2.5 ft of one of the 10
SportVU players. Intervals: 2,000 frame bootstrap replicates, seed 20261121. As for source 2, a foot beyond 2.5 ft
is not necessarily a false box (foot point, occlusion, camera truth error); on gsw_ind_2016 the weak camera truth
(D98) likely explains most of the lower share.

Reading of source 2 (P8_teachers.md): a miss there is a box foot more than 2.5 ft from SportVU, not only a missing
box (ours draws 3,738 boxes for 3,953 in-frame players on gsw_bkn_2015), so part of what YOLOX-X recovers is foot
placement, not detection. Camera-truth frames favour scenes where our detector works, so the miss share on all live
frames is likely higher.

Volume, as a projection and not a count: 1.0 recovered box a frame over gsw_bkn_2015's 2,075 Mac camera-truth frames
is about 2,075 boxes; over the 13,975 Colab R1.3 camera-truth frames of the 4 training games (ROADMAP_LOG.md, run
r13_20261004_1846) it would be about 14,000, if the gsw_bkn_2015 rate held on Colab truth (to check: the rate was
measured on one game with Mac truth).

## 4. Scoring protocol on training games

- Scorer: scorecard v2, tools/p8_scorecard.py, adopted and pinned (D88); it changes only with a PLAN.md line, and
  tests/test_p8_scorecard.py fails on any change to it or its constants.
- Truth: camera truth (the per-game camera model with per-frame pan, tilt and zoom, accepted by sportvu/autolabel.py's
  rules). Universe: live wide frames with accepted truth; the truth share is printed beside every row (0.118 on
  gsw_bkn_2015, P8_baseline.md).
- Games: the R1.3 training games pooled, game bootstrap through `combine`; window-cluster bootstrap (1,000 resamples,
  seed 0) per game. Today only gsw_bkn_2015 has Mac camera truth, so the pooled set is one game and there is no game
  bootstrap (P8_baseline.md). gsw_cha_2016 and gsw_sac_2015 wait for their Mac relabel (queued by the lead).
- D98: gsw_ind_2016 is kept out of pooled and gate rows and reported apart, flagged, because the Mac camera fit there
  is off (pooled reprojection 18.35 px against 6.53 on Colab). D99 and D100: the cause is the MPS detector and a
  fragile clock offset solve; the Colab one-cell build and any change of the offset rule are Lucien's decisions.
- In sample: r23's court model was trained on the R1.3 games, so position, bias and coverage rows there are not a
  held-out measure (the scorer's caveat).
- Not measurable on r23 (scorecard v2 reports it so; P8_baseline.md): ball in the image, ball carrier, identity v2,
  identity reported, shots, rebounds, shooter and rebounder position (the build writes none of their inputs), and the
  swap test (outside the scorer). The team row has no interval: the pinned scorer reports none (P8_baseline.md).
- The final score of P8 is on the D85 broadcast test set, once (D129); nothing in this protocol touches it.

## 5. Targets, fixed now

The Stage 1 targets are Lucien's (ROADMAP.md, confirmed 2026-10-03, per P8_FEASIBILITY.md 6.1), as the frozen scorer
states them (sportvu/bench.py TARGETS). The new rows are P8_FEASIBILITY.md 6.2, with constants pinned in
tools/p8_scorecard.py CONSTANTS (D88). D85 (2): done means the 6.2 rows meet their targets and every Stage 1 row
improves under the adoption rule (D5) on the test set. None is changed here. The baseline column is gsw_bkn_2015, r23,
camera truth (P8_baseline.md; E2 rows from P8_E2.md, build e2).

| Row | Target | Source | Baseline, gsw_bkn_2015 |
| --- | --- | --- | --- |
| Position error | p50 <= 2.0 ft, p90 <= 5.0 ft | bench.py TARGETS | p50 1.56, p90 3.36 ft (in sample) |
| Near-field bias | within +-0.5 ft in the near and the far third | bench.py TARGETS | near 0.03, far 1.57 ft |
| Coverage | >= 50% of live wide seconds | bench.py TARGETS | 0.998 |
| Missed players | <= 5% | bench.py TARGETS | 0.26 [0.246, 0.278] |
| Ghost boxes | <= 5%, referees excluded by class | bench.py TARGETS | 0.118 [0.103, 0.134] |
| Team labels | >= 95% | bench.py TARGETS | 0.868 (no interval) |
| Court line error | reported; goal <= 3 px on 95% of accepted wide frames | bench.py TARGETS | not measurable on camera truth (scorer) |
| Generalisation | held-out arena within 1.5x of the held-out game | bench.py TARGETS | not scored on training games |
| Ball in the image | F1 >= 0.75, tolerance 2x the median camera-truth reprojection of SportVU feet onto box feet, fixed by the Phase A baseline before any ball model | 6.2; CONSTANTS ball_f1_target, ball_tol_rule | not measurable on r23 |
| Ball carrier | >= 85% of live frames with a SportVU holder (one player within 3 ft, ball below 6 ft) | 6.2; CONSTANTS carrier_target, holder_max_ft, holder_ball_max_z_ft | not measurable on r23 |
| Identity | >= 80% of matched player-frames name the SportVU player | 6.2; CONSTANTS identity_target | not measurable on r23 |
| Identity, reported | jersey tracklet accuracy on tracklets with >= 8 crops; goal <= 5 switches a possession | 6.2; CONSTANTS jersey_min_crops, switches_goal_per_possession | 13.73 switches a possession (frozen identity row) |
| Shots | >= 95% of field-goal attempts within 1.0 s; <= 5% false events per true event | 6.2; CONSTANTS shots_found_target, shots_false_max, event_tol_s | E2 M0: 0.616 [0.455, 0.766] found, 0.42 false per true (P8_E2.md) |
| Rebounds | >= 90% within 1.0 s | 6.2; CONSTANTS rebounds_found_target | E2 M0: 0.607 [0.441, 0.784] (P8_E2.md) |
| Shooter position | p50 <= 3.0 ft, p90 <= 6.0 ft | 6.2; CONSTANTS shooter_p50_ft, shooter_p90_ft | not measurable (needs E4 or E7) |
| Rebounder position | p50 <= 4.0 ft | 6.2; CONSTANTS rebounder_p50_ft | not measurable (needs E4 or E7) |
| Swap test | per lane, set by the lane's owner before the run | 6.2; D3 | outside the scorer |

Open on the shots and rebounds rows (P8_E2.md 2.6): whether the truth is the logged play-by-play time or the
physical release (about 2 s earlier) is the lead's and Lucien's call; the row cannot be read until it is fixed.

## 6. GPU and Mac hours per experiment

Bases (P8_FEASIBILITY.md 5.1), plus the two new measurements of Track B:

- b1: detector fine-tune, YOLOv8m, 654 images, about 0.5 h on a T4 (DEVLOG.md 2026-06-26).
- b2: camera-truth labelling, 4 games, 9,787 s of video, 2.7 h on an A100, 13,975 accepted labels; this Mac 2.15 s a
  video second (ROADMAP_LOG.md, lines on the R1.3 run).
- b3: court model training arm, about 52 min on an A100 (ROADMAP_LOG.md side study and pass 14).
- b4: scoring a court model on three games, about 20 min by court replay, about 1 h by full rebuild (ROADMAP_LOG.md).
- b5: teacher proposals (Grounding DINO, YOLOv8x-pose and the detector) on 6,197 frames, 197 min on a T4 (DEVLOG.md
  2026-07-18).
- b6 (new): YOLOX-X teacher pass on this Mac's CPU, 348.5 s for 400 frames, 0.871 s a frame (P8_teachers.md). A GPU time for YOLOX-X is not measured: to check.
- b7 (new): scorecard v2 per game on the Mac, inside the lock: a run of under a minute a game once builds exist (time
  printed by the lock wrapper, not recorded in a file: to check).

Colab units: T4 1.19, L4 1.71 and A100 5.40 units an hour, as given by the lead on 2026-11-21 (no repository file
records them: to check against the Colab pricing page). Units below are hours times those rates; L4 hours are not
given because no basis relates L4 speed to T4 or A100 (to check).

| Work | Basis and arithmetic | GPU (estimate) | Colab units | Mac under the lock (estimate) |
| --- | --- | --- | --- | --- |
| Teacher pass, YOLOX-X, the 13,975 R1.3 camera-truth frames | b6: 13,975 x 0.871 s | not measured; b5's 197 T4-min for 6,197 frames of three models is a ceiling (to check) | at most 3.28 T4-h x 1.19 = 3.9 for 6,197 frames (b5) | 3.4 h CPU |
| Teacher pass, a 6,000 to 12,000 frame corpus (P8_FEASIBILITY.md 5.2) | b6 | as above | as above | 1.5 to 2.9 h CPU |
| E1 training, one seed, 13,975 images (sources 1 and 2 on Colab truth) | b1 scaled by images at the same epochs: 0.5 x 13,975 / 654 | 10.7 T4-h | 12.7 | not advised (b1: local MPS about 22 h for 654 images) |
| E1 training, 3 seeds | 3 x the line above | 32.0 T4-h | 38.1 | none |
| E1 training, one seed, 2,075 images (Mac camera truth of gsw_bkn_2015 only) | 0.5 x 2,075 / 654 | 1.6 T4-h | 1.9 | none |
| E1 scoring, per game per seed | b4 (rebuild) | none | none | about 0.3 h a game (P8_FEASIBILITY.md 5.2) |
| Phase A camera truth, a new game | b2: 2.7 A100-h / 4 games | 0.68 A100-h | 3.6 | about 1.5 h a game at 2.15 s a video second (P8_FEASIBILITY.md 5.2) |
| E5 court model, one seed | b3 | 0.87 A100-h (P8_FEASIBILITY.md 5.2 gives 1 to 2 A100-h with labels) | 4.7 to 10.8 | about 20 min a model to score (b4) |
| E3 team embeddings | no GPU (docs/research/tracking.md) | none | none | 4 to 8 h |
| E2 events | no GPU; time not measured (P8_FEASIBILITY.md 5.2 estimate) | none | none | under 1 h a game (to check) |

Totals for the next step, E1 with 3 seeds on the 4 R1.3 games' Colab truth: about 32 T4-h (38 units) of training
plus a teacher pass of about 3.4 Mac CPU hours, before scoring. These scale b1 linearly in images, which is an
assumption (to check with a one-epoch pilot). Every Colab session is Lucien's to run (D137).

## 7. E1 to E7, re-ranked from Track B's evidence

P8_FEASIBILITY.md section 8 ranked E2 first and E1 second, with E1's gain to be set by Phase A's missed rate ("If
scorecard v2 puts missed players under 10 percent, E1's gain drops to 3"). Track B's evidence:

| Rank | Experiment | Evidence now | Gain | Effort |
| --- | --- | --- | --- | --- |
| 1 | E1 detector on teacher boxes (D131 sources 1 and 2), referee as its own class | missed 0.26 and ghosts 0.118 against 5% targets (P8_baseline.md), so the gain stays 5; a qualified teacher exists (+0.0419 recall at our precision, P8_teachers.md); source 2 yields about 1 miss-targeted box a frame. Limit: YOLOX-X itself misses 0.19 on the same frames, and part of the miss row is foot placement, so E1 alone is unlikely to reach 5% (P8_teachers.md) | 5 | 3 |
| 2 | E3 team labels from tracklet embeddings | team 0.868 against 95% (P8_baseline.md, no interval); no GPU; pre-registered (docs/research/tracking.md) | 3 | 2 |
| 3 | E2 events from the play-by-play and the clock | measured and not met: shots 0.616 found with 0.42 false per true, rebounds 0.607 (P8_E2.md); held until Lucien decides the offset rule (D99, D100) and the truth meaning of the row; no more work moves it before then | 4 | 2 (held) |
| 4 | E4 jersey identity | 13.73 switches a possession against a goal of 5 (P8_baseline.md); GPU unknown until a pilot | 4 | 3 |
| 5 | E5 court model on new-arena labels, 3 seeds | **slot for Track A**: its yield of identity-confirmed broadcasts in new arenas (the go bar is 8 broadcasts from 4 new arenas, P8_step0.md) decides whether new-arena labels exist; with them E5 can move above E4, without them it stays here | 3 | 3 |
| 6 | E6 learned fill of missed player-frames | depends on E1 and E3 outputs | 4 | 4 |
| 7 | E7 ball detector in the image | ball rows not measurable on r23; would also settle E2's release-time question | 4 | 5 |

Go or no-go for Track B (P8_step0.md: at least one teacher that beats our detector on the human box truth, and a
baseline of the scorecard rows measurable on training games): the first part is met (YOLOX-X); the baseline exists
for one pooled game (gsw_bkn_2015) and one apart (gsw_ind_2016), with gsw_cha_2016 and gsw_sac_2015 queued. Whether
one pooled game is enough is the lead's call.
