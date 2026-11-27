# P8: a per-box foot from pose keypoints (design, version 1, registered before any number)

Version 1, 2026-11-27. Written under the ml-training-rigor protocol as a falsifier first. Report only: nothing is
fitted, tuned or trained. A pretrained pose model is run as is, and no truth, build or score changes. No Colab unit is
used, no held-out game, no D85 test game and no test arena is read (CLAUDE.md rule 4), and there is no hand labelling.
hou_cle_2016 is not read at all and stays unspent. No download and no compute happen before this design is reviewed.

## 1. Question and mechanism

**Question.** On the boxes that matter, is a foot point taken from pose keypoints (the ankles) nearer the player than
the build's stabilised foot (foot_stab)? Is it nearer under the truth camera T, which the missed row reads, and not
farther under the build's court H, B, which is the independent check? And does it stay as good on hits?

**The boxes that matter.** D207 (P8_piibox.md section 13) found that the box 2.5 to 5 ft from a P-ii player is mostly
the player's own track ("own", all four kinds: 0.541 of P-ii boxes, 0.062 of in-frame players), and that box geometry
barely marks it. D206 found that no population shift of the foot reaches the missed bar. So if these own boxes are to
come within 2.5 ft, it must be by a per-box signal, a foot point read from each box's own pixels. Ankle keypoints are
such a signal: they locate the feet inside the box whatever the box bottom does (a stride, a raised foot, a box edge on
the shadow or the floor).

**Mechanism and its known risk.** The ankle sits above the floor. Read through a ground-plane homography, an ankle
pixel lands beyond the player along the camera ray. So the ankle is lifted to a fixed ankle height before it is
compared (section 4). The truth camera T was fitted to foot_stab pixels (D199, D206 section 13.1). Under T, foot_stab
is therefore favoured by construction, because T absorbed its average offset, while the pose foot is not favoured. B
was fitted to court lines only, so B is the fair frame. The registered rule reads both.

## 2. Known before this item (stated as known)

- D205: P-ii is 0.1141 of in-frame players pooled, its box a median 3.2 ft away.
- D207's class counts of P-ii boxes:

  | class | gsw_cha_2016 | gsw_bkn_2015 | pooled |
  |---|---|---|---|
  | own (all four) | 1,831 | 1,737 | 3,568 |
  | switch | 1,122 | 1,043 | 2,165 |
  | another's (both) | 354 | 430 | 784 |
  | ghost (all three) | 43 | 36 | 79 |

  About 0.566 of the own boxes and 0.577 of the switch boxes are scorer ghosts on their frame. Box geometry barely
  separates P-ii boxes from hits' boxes.
- D206: two median foot shifts reach about -0.004 in sample; per-box scatter is one reading, not separated.
- D184, D186 to D188: the foot lands short along the camera ray under B, and position-only corrections raise the
  near-third tail.
- Decode and alignment (D190, P8_a3align.md, P8_a3run.md):
  - The build's own decode of a window clip can differ from a sequential decode of it. 14 of 175 pool windows showed
    the corpus decode one processed frame behind the build's.
  - On three control windows, a replay started by sequential reads from frame 0 matched the corpus decode (0.996) but
    not the build's sidecars.
  - Truth rows are inferred, not checked, to attach to the build-side decode.
  - So a frame index alone does not prove that the decoded image is the image the sidecar boxes came from.

## 3. Upper bound

A perfect per-box foot is defined as each box's foot landing within 2.5 ft of its own player, with every hit staying a
hit. On D205's and D207's rows, it could move at most:

| | gsw_cha_2016 | gsw_bkn_2015 | pooled |
|---|---|---|---|
| missed, own (all four) P-ii boxes only | 1,831 / 28,427 = 0.064 | 1,737 / 29,370 = 0.059 | 0.062 |
| missed, own plus switch (if a switch box is the player's own on that frame) | 0.104 | 0.095 | 0.099 |
| ghosts per box, own scorer ghosts (0.566 of own boxes) | 0.039 of 26,992 boxes | 0.036 of 26,868 | 0.037 (0.121 to about 0.084) |
| ghosts per box, own plus switch scorer ghosts | 0.063 | 0.058 | 0.061 |

Another's and ghost boxes (0.015 of in-frame players) are out of reach of a foot point. These are ceilings, not
forecasts. A pose foot is not perfect, and it moves hits too, which this item measures (section 5).

## 4. The pose model, the foot point and the decode

**Model.** usyd-community/vitpose-base-simple, through transformers' VitPoseForPoseEstimation. transformers 4.49.0 is
already installed in the frozen CV python, so no new package is needed.
- It is a top-down ViTPose-B (85.9M parameters) trained on COCO keypoints (17 points, ankles 15 and 16).
- It runs on CPU with VitPoseImageProcessor on the full decoded frame and the stored boxes (COCO x, y, w, h), at its
  default 256 by 192 input and its default box scale.
- Licence, from the Hugging Face model card read 2026-11-27 (metadata only, no weights fetched): apache-2.0.
  docs/DATA_SOURCES.md gets the row in this commit. The training data's terms (COCO) and the upstream repository's
  licence are marked "to check" there.
- Before the first download, the model revision (commit hash) is pinned in the tool. After it, the safetensors file's
  SHA-256 is recorded. The weights stay in the Hugging Face cache outside the repository and are never redistributed.
- Fallback, only if the licence check fails: torchvision's KeypointRCNN_ResNet50_FPN COCO_V1 (BSD-3-Clause code,
  already installed). Its use would be a dated amendment, not a silent switch.

**The pose foot.**
- From the 17 keypoints of each box, the ankles are used when their score is at least 0.3. That is a fixed common
  default, not tuned here.
- Both ankles confident: their pixel midpoint. One ankle confident: that ankle. Neither: no pose foot. The box is then
  counted in the coverage and left out of the paired distances.
- **Ankle lift.** The pose foot is the ground point beneath the ankle. The ankle pixel's ray is cut by the plane at
  ANKLE_FT = 0.25 ft above the floor, and that point is dropped to the floor.
  - Under T, this uses the label row's own camera (camera_C, camera_rvec, camera_f), the camera that made H_truth.
    D207 checked that it reproduces H_truth.
  - Under B, it uses sportvu.camera.pose_from_h(B). A frame without an accepted B or pose has no B-frame distance.
  - The unlifted ankle (plane at 0) is reported beside it as a sensitivity and is not ruled.

**The foot_stab comparison.** foot_stab is mapped as the scorer maps it: float32 pixels through T for the T-frame
distance, and through B for the B-frame distance.

**Distances.** For box b of player j, the reference is SportVU_j in the truth's court frame (mirror applied):
- d_stab,T = |T(foot_stab) - SportVU_j| and d_pose,T = |ground_T(ankle) - SportVU_j|;
- d_stab,B and d_pose,B the same under B;
- the paired change, delta = d_pose - d_stab, per frame (T and B).

**Decode, in order (the D190 lessons).**
1. Each window clip (label_build/<window>.mp4) is read once, sequentially from frame 0, with videoseq.SeqReader. Every
   stride frame is read in increasing order, with no seek inside the window, as P8_a3run.md amendments 1 and 2.
2. Only frames that hold a selected box are kept in memory, one frame at a time.
3. **Per-window alignment check, before any pose number of the window is kept.** r23's detector (the main checkout's
   models/player_detector.pt, SHA-256 checked as r23's; CPU, confidence 0.40, NMS 0.45 as P8_a3run.md) runs on four
   processed frames of the window. These are the selected frames at the window's 0, 1/3, 2/3 and last positions. For
   each frame f, it runs on the decoded frames f - stride, f and f + stride, and the detections are matched to the
   sidecar's boxes at f (Hungarian on IoU, pairs at IoU 0.3 or more).
   - The statistic per offset is the median IoU of the matched pairs.
   - A window is **aligned** when offset 0 has the highest median IoU on at least 3 of the 4 frames, and its median
     IoU over the four frames is at least 0.80.
   - A window that is not aligned is left out of every pose number and counted. It is not re-indexed, which would be a
     fork.
   - If more than 15 percent of a game's windows are not aligned, the run stops and reports, since the decode is then
     not under control.

## 5. The boxes measured, and the count gate

- **Games:** gsw_cha_2016 and gsw_bkn_2015 only. hou_cle_2016 is not opened.
- **Selection, by D207's definitions recomputed** (tools.p8_piibox: the same two-stage assignment, tracks and
  classes):
  - every P-ii box, all classes (6,596). The rule reads the own group, and the other classes are reported;
  - the control, hit boxes on the same frames: on each frame that holds a selected P-ii box, up to 3 hit boxes chosen
    with numpy's default_rng(0), in frame order, from that frame's stage A hit boxes, all classes.
- **Count gate, before any pose is run:** the recomputed classes equal D207's record per window (P-ii and hit counts
  per class from work_ref/piibox/<game>.json). D207's own gates already tie these to D202 and D205. The run stops on
  any difference.
- **No hand labelling,** and nothing is viewed.

## 6. The registered falsifier (fixed now)

All rows are per game, with window-cluster bootstrap intervals (2,000 resamples, seed 0, windows resampled whole) over
aligned windows and boxes with a pose foot. Coverage (the share of selected boxes with a pose foot) is reported beside
every row.

- **F1, rescue on own P-ii boxes.** The share of own (all four) P-ii boxes whose pose foot lies within 2.5 ft of the
  player under T: R_own. The median of delta_T over the same boxes: m_own,T.
- **F2, the same boxes under B.** m_own,B.
- **F3, the cost on hits.** The share of control hit boxes whose pose foot lies beyond 2.5 ft under T: L_hit. The
  median of delta_T: m_hit,T.
- **F4, the net estimate.** An estimate of the change in missed if pose feet replaced foot_stab on the selected kinds
  of box:

  est = (-R_all x N_P-ii + L_hit x N_hits) x coverage / in-frame players.

  R_all is the rescue share over all P-ii boxes, N_P-ii and N_hits are D207's counts, and the hit share is assumed to
  hold for all hits. It ignores players rescued by another player's box and shared boxes, so it is an estimate, not
  the scorer's row.

**The pose line continues to a registered next stage only if, on both games:**
1. R_own is at least 0.20 (a fifth of own P-ii boxes come within 2.5 ft);
2. m_own,T is at most -0.5 ft, and the upper end of its interval is below 0 (nearer by a margin, under T);
3. m_own,B is at most 0 (point estimate): not farther under B;
4. m_hit,T is at most +0.10 ft, and L_hit is at most 0.05 (not worse on hits);
5. est is at most -0.010 (the bar every missed arm has faced).

Otherwise **the pose line stops**. The failing rows are named. The next step is a dated D line, not a rescue: no
other model, threshold, ankle height or crop scale is tried on these games.

Why these bars:
- -0.010 is the standing missed bar.
- R_own at least 0.20 asks the signal to reach a meaningful part of the ceiling (0.2 x 0.062 is about 0.012).
- -0.5 ft is a sixth of P-ii's 3.2 ft median distance.
- +0.10 ft and 0.05 keep hits, which are most boxes, from paying for the gain.
- Rule 3 is the D206 lesson: under T alone a gain could be agreement with T's fit, while B is fitted with no SportVU
  and no feet.

**Readings (descriptive, fixed now).**
- **Pass:** a per-box foot that is nearer on the boxes that matter in both frames and does not cost hits. Next, a
  registered stage would score pose feet through the pinned scorer on whole windows (all boxes of the scored frames,
  the D202 rows paired, under the lock), with hou_cle_2016 kept for a later single test.
- **Fails rule 1 or 2:** ankles do not locate these feet better than foot_stab. The box is placed wrongly as a whole,
  or the player's foot is hidden.
- **Passes 1 and 2 and fails 3:** agreement with T only.
- **Fails 4:** the gain on P-ii boxes is paid for on hits.
- **Passes 1 to 4 and fails 5:** too few boxes move to reach the bar.

**Report only, beside the rule:**
- every row for each D207 class separately, and for switch;
- the unlifted ankle;
- the share of P-ii boxes whose pose foot is within 3.0 ft of some player (the ghost-row view);
- coverage by box height bin and third;
- the median offset vector from foot_stab to the pose foot along and across the ray. This is described only. Fitting
  it would be a D206-style population shift, which is not done.

## 7. Noise floor

There is no earlier paired measure of a pose foot. The intervals of rules 1 to 4 are the window-cluster bootstrap's on
about 1,700 to 1,800 own boxes and about 70 to 80 windows per game. For a share near 0.2 on about 1,750 boxes
clustered in 75 windows, the half-width is expected near 0.03 to 0.05: D207's class shares on similar counts had
half-widths of 0.03 to 0.06. That resolves R_own against the 0.20 bar only when the true share is outside about 0.15
to 0.25. A point estimate inside that band is reported as near the bar, and the rule reads the point.

## 8. Ladder, cost and guards

**Stage 0, licence and pin (no lock).** The licence row is in docs/DATA_SOURCES.md. The revision hash is pinned, then
the one download follows (about 0.35 GB of safetensors), and its SHA-256 is recorded. This happens only after the
review's go.

**Stage 1, smoke (one lock, "p8 posefoot smoke").** Two windows, one per game:
- decode, the alignment check, the count gate, pose on every selected box, the rows;
- measured rates: decode seconds per window, detector seconds per frame, pose seconds per crop;
- a mechanical check: the pose foot of a box lies inside or just below its box (within 0.2 box heights), for 0.95 of
  boxes with a pose foot. Else stop, since the crop mapping is wrong;
- an overlay is not drawn or viewed (no hand labelling).

**Stage 2, the falsifier (one lock, "p8 posefoot falsifier"),** both games, then the report in system python.

**Cost (estimates; the smoke replaces them with measured rates before stage 2 runs):**
- crops: about 6,600 P-ii boxes plus at most 3 hit boxes per P-ii frame, so about 9,000 to 12,000 hit boxes;
- pose: about 0.1 to 0.2 CPU seconds per crop, about 27 to 63 minutes;
- detector: 4 frames x 3 offsets x 156 windows, about 1,900 frames at about 0.3 to 0.5 s, about 10 to 16 minutes;
- decode: about 1 to 3 minutes per game;
- manifests: about 2 minutes.

So stage 2 is about 45 to 85 lock minutes, cap 120. If the smoke's rates project above the cap, stage 2 does not run
and the cap is not raised; the lead decides. About 0.35 GB of weights and a few MB of records are written, and 0
units are used.

**Guards.**
- tools.p8_common.guard and tools.research_testset.refuse_cv_test on every game, SportVU id, video id and window.
- The tool refuses hou_cle_2016 in code.
- tools/ and sportvu/ are clean at the run's head, and the scorer pin is checked.
- SHA-256 manifests of work_ref's and work_ms's labels, builds and cache before and after each step.
- Writes go only to work_ref/posefoot/ and reports/research/p8_posefoot.json.
- Memory: one decoded frame at a time, the pose model and the detector loaded once, each step under the lock.
- Retries: a mechanical failure is retried at most twice, then stopped and reported. A failed rule is not retried.
- One background chain per stage, reported the moment it ends.

## 9. Forking paths, named now

- D207's result (own 0.541, geometry not separating) and D206's lesson motivated this item. The bars of section 6
  were set after those numbers but before any pose number.
- The model was chosen as the installed top-down COCO model with a clear licence, not by any accuracy on these games.
  Its threshold, ankle height and crop scale are fixed defaults.
- Hits are sampled on the P-ii frames, not at random over all frames. Their crowding and depth may differ from all
  hits', so L_hit is read with that limit.

## 10. Limits

- Under T, foot_stab is favoured by construction (section 1). Under B, B's own court-model error adds to both feet
  alike.
- The pose model was trained on COCO photos, not broadcast basketball at 720p. Small far boxes and occluded feet may
  give no ankle, and coverage is reported for that reason.
- Alignment is checked on four frames per window, not every frame.
- The net estimate (F4) is not the scorer's row. Only the next stage would give the paired D202 rows.
- Two games, one arena, one broadcaster.
