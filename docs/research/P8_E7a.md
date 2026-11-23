# P8 E7a: a first measurement of the ball row with WASB's basketball model (PLAN.md D155 (2))

Status: stopped at step 1 on 2026-11-22 (license); resumed under PLAN.md D160 on 2026-11-23. The registration below
was committed before any run (no frame read, no truth computed, no model run on a game frame before it).

## License finding

Read at the source (github.com/nttcom/WASB-SBDT, default branch, last commit 923462ca of 2023-11-23):

- The basketball WASB weights are `wasb_basketball_best.pth.tar` on Google Drive (id
  1nfECuSyJvPUmz3njZCdFERSQQbERt8FU), linked from MODEL_ZOO.md (row "WASB (Ours)", column "Basketball") and fetched
  by `src/setup_scripts/setup_weights.sh` line 54. An HTTP HEAD request gives 6,102,633 bytes.
- MODEL_ZOO.md says only: "We provide model weights of SOTA methods (including WASB) for each dataset." It names no
  license or terms for them. GET_STARTED.md ("Model Preparation") says only "Download pretrained models listed in
  MODEL_ZOO.md". The README has an MIT badge for the repository.
- LICENSE.md is the MIT License, "Copyright (c) 2023 NTT Communications Corporation". Its grant covers "this
  software and associated documentation files (the "Software")". The weights are not in the repository; they are
  hosted on Google Drive, and neither LICENSE.md nor any other file in the repository says whether the grant extends
  to them.
- The weights were trained on the WASB basketball labels over NUST-NBA181 video (WASB's basketball dataset config
  lists 2018-19 game ids), which its authors release "upon request and only for academic research"
  (docs/research/P8_datasets.md).

Reading: no license explicitly covers the weights. As the task instructs for that case, the finding is recorded
(docs/DATA_SOURCES.md, row "WASB basketball model weights") and E7a stops before any download. Whether the
repository's MIT grant is taken to cover weights its authors distribute for it, or the authors are asked, is a
decision for the lead or Lucien. Steps 2 to 4 (the registration of frames, truth, in-frame and occlusion rule,
metric, tolerance and intervals, then the runs) have not started.

## Resumed under D160 (Lucien, 2026-11-23)

D160 treats the repository's MIT license as covering its released weights; docs/DATA_SOURCES.md says so. Report-only:
no training, no tuning, and nothing from WASB's label zip is used (its license is still unstated).

Weights: `wasb_basketball_best.pth.tar`, 6,102,633 bytes, sha256
8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948, downloaded 2026-11-23 to the P8 data workspace
(models/wasb), loaded with `torch.load(weights_only=True)`. Code: the repository at commit 923462ca, cloned beside it.

## Registration (fixed before any run; tools/p8_e7a.py holds the same constants)

1. **Frames.** Games gsw_bkn_2015, gsw_cha_2016 and gsw_sac_2015 (the 3 R1.3 games with Mac camera truth), and
   hou_cle_2016 reported apart (the new-arena check; it enters no pooled number). Candidates are every accepted
   camera-truth frame (labels/<game>/*.jsonl in the P8 workspace). Sampled by time: within each window clip, in frame
   order, a frame is kept when it is at least 120 clip frames (2.0 s at the clips' 59.94 fps) after the last kept
   one. Chosen by time alone, never by a model or by the truth.
2. **Truth.** The SportVU ball of the frame's moment (the row's quarter and moment, in the scorecard's Moments
   order), mirrored as the players are (the game's mirror), projected through the row's camera truth (camera_C,
   camera_rvec, camera_f) with tools.p8_scorecard.project3d at 1280 x 720.
3. **Usable truth frames, in-image and height rule.** A sampled frame is a truth frame when the SportVU ball is
   tracked (finite) and its height lies in [-1, 30] ft; others are counted and left out. The ball is in the image
   when its projection lies in front of the camera and inside 0 <= u < 1280, 0 <= v < 720. No frame is removed for
   occlusion: there is no occlusion truth, so occlusion is reported, not ruled out, through two splits fixed now:
   height bands (low below 4 ft, mid 4 to 10 ft, high from 10 ft, so a ball high in the air is reported apart), and
   held or loose (held: the scorecard's holder rule, exactly one player within 3 ft of the ball on the court plane
   and the ball below 6 ft).
4. **Counting (the scorecard's section 6.2 ball rule, tools/p8_scorecard.py).** Over truth frames: a true positive
   when the ball is in the image and the prediction lies within the tolerance; otherwise a false positive when there
   is a prediction (including any prediction while the ball is out of the image) and a false negative when the ball
   is in the image. Precision, recall and F1 from the counts.
5. **Tolerance: 32.07 px at 720p.** The scorecard's registered rule is "2 x the median camera-truth reprojection
   error of SportVU feet onto the box feet (px), over the development games"; its fixing command refuses today
   (4 development games have no r23 camera scorecard), so the same formula is applied to the 3 games here: the r23
   camera scorecard observations give medians of 16.087 (gsw_bkn_2015, 17,167), 18.65 (gsw_cha_2016, 10,961) and
   14.667 px (gsw_sac_2015, 18,120), pooled 16.033 px over 46,248, so 2 x 16.033 = 32.07 px. This does not fix
   reports/research/p8_constants.json. Why not WASB's 4 px: WASB scores hand-clicked ball centres; our truth is a
   projected SportVU ball whose own error (camera truth, SportVU sampling at 25 Hz against 59.94 fps frames) is about
   the size of the feet reprojection error, so a 4 px tolerance would score the truth's error, not the model's.
   Reported beside it, not used for any reading: 2.67 px (WASB's 4 px at 1080p scaled to 720p), 10 px and 20 px.
6. **Intervals.** Pooled over the 3 games with a game bootstrap (games resampled with replacement, 2,000 resamples,
   seed 7, percentile 95%); per game a frame bootstrap (sampled frames resampled, the same settings).
   hou_cle_2016 gets its own frame bootstrap.
7. **Model, as released.** HRNet from configs model/wasb.yaml (3 frames in, 3 out, 512 x 288), the frame warped by
   the repository's affine (centre, scale max(h, w)), ImageNet normalisation, sigmoid heatmaps, connected-component
   blobs above 0.5 weighted by the heatmap (detector/tracknetv2.yaml postprocessor), step 3, and the online tracker
   (tracker/online.yaml, max displacement 300 px). Departures, fixed now: CPU instead of CUDA (the repository's
   detector refuses CPU, so its pieces are called directly), and each sampled frame is predicted from a segment of 4
   consecutive triplets (12 frames, 0.2 s) ending at it, the tracker refreshed per segment, instead of a whole
   annotated clip.
8. **Runs.** One game at a time under the heavy-job lock; tools.research_testset.refuse_cv_test and
   tools.p8_common.guard on every game, SportVU id and video id before any frame is read; clips decoded from frame 0
   without seeking (as tools/p8_e1data.py). Output in the P8 data workspace (e7a/), the summary in
   reports/research/p8_e7a.json.
9. **Reading.** Report-only, a first measurement of the ball row. Nothing is adopted; the scorecard's F1 target
   (0.75) is quoted for context only. E2's release-time question is noted, not settled.

## Changes after registration

1. 2026-11-23, before any prediction was written: the first gsw_bkn_2015 run stopped after 614 s of lock on a sampled
   frame 0 of a clip, which no triplet can end at (an empty segment). Rule added: a sampled frame 0 or 1 of a clip
   takes the clip's first triplet (frames 0 to 2) alone. No output of that run exists, so nothing was seen.

## Results (2026-11-23; reports/research/p8_e7a.json)

Report-only. Every sampled frame of the 3 games had a tracked ball in the height range except 2 of gsw_sac_2015
(ball not tracked); the truth ball was in the image on 806 of 809 truth frames (0.9963), and on 466 of 467 for
hou_cle_2016. Tolerance 32.07 px.

| Game | Truth frames | TP | FP | FN | Precision | Recall | F1 (frame bootstrap 95%) |
|---|---|---|---|---|---|---|---|
| gsw_bkn_2015 | 296 | 39 | 106 | 257 | 0.2690 | 0.1318 | 0.1769 [0.1287, 0.2277] |
| gsw_cha_2016 | 235 | 34 | 111 | 201 | 0.2345 | 0.1447 | 0.1789 [0.1290, 0.2341] |
| gsw_sac_2015 | 278 | 40 | 115 | 235 | 0.2581 | 0.1455 | 0.1860 [0.1375, 0.2393] |
| pooled 3 (game bootstrap) | 809 | 113 | 332 | 693 | 0.2539 [0.2345, 0.2690] | 0.1402 [0.1318, 0.1455] | 0.1807 [0.1769, 0.1860] |
| hou_cle_2016 (apart) | 467 | 86 | 132 | 380 | 0.3945 | 0.1845 | 0.2515 [0.2077, 0.2957] |

Reported beside it (pooled 3): F1 0.0064 at 2.67 px, 0.0352 at 10 px, 0.0863 at 20 px. Recall by height band
(pooled TP over TP plus FN): low 78 of 482, mid 33 of 273, high 2 of 51 (per game in the JSON). Held against
loose: recall 91 of 578 held, 22 of 228 loose (per game in the JSON). The game-bootstrap interval of 3 games is
narrow because the 3 games agree; it carries only 3 resamplable units.

Reading, against the scorecard's 0.75 target quoted for context: far below it. Two things point at the truth more
than at the model, and neither changes the registered reading. (1) F1 rises steadily with the tolerance (0.006,
0.035, 0.086, 0.181 from 2.67 to 32 px): where WASB predicts, its point lies a median 52.6 px (gsw_bkn_2015) and
41.4 px (hou_cle_2016) from the projected SportVU ball, with a median vertical offset of -12 to -15 px (the
prediction above the truth). (2) Three gsw_bkn_2015 frames were drawn with both points to see which is on the ball
(a look for this report, no label made, nothing scored from it): in the two with the ball in flight, WASB's point
sits on the visible ball and the projected SportVU ball lies 27 and 90 px away. A ball truth this far off makes the
ball row unmeasurable at any tolerance that would also reject wrong detections; the projected SportVU ball needs its
own error measured (time offset and height) before a ball model can be scored against it.

Lock time: 3,236 s (0.90 h): plan 6 s, the stopped first gsw_bkn_2015 run 614 s, then 616, 505, 589 and 906 s for
gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 and hou_cle_2016. Model time on CPU about 0.7 s a 3-frame triplet.
