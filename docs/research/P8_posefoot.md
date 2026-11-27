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

## 11. Version 1.1 (2026-11-27, after the design review of 2f19de21, GO WITH CHANGES; before any download, code or number)

The independent review found three blockers and six should-fix items. Its cheapest step changes the order
(self-decided by the lead): a foot_raw stage runs first, before any pose download. Sections 1 to 10 stand except where
this section replaces them; where they differ, this section rules.

### 11.1 Correction to section 1

Section 1 said ankle keypoints "locate the feet inside the box whatever the box bottom does". That is the hypothesis,
not a fact. The sentence now reads: ankle keypoints may locate the feet where the box bottom does not (a stride, a
raised foot, a box edge on a shadow or the floor), and this item tests whether they do.

### 11.2 The new order of stages

| stage | what | inputs | lock |
|---|---|---|---|
| 0a | foot_raw against foot_stab on the same boxes (11.3) | stored sidecars, truths, D207's selection | about 2 minutes a game |
| A | decode alignment of all 156 pool windows (11.6) | clips, r23's detector; no download | 11.9 |
| 0 | licence closed at the source, revision pinned, the one download (11.8) | after A and its review | none |
| 1 | pose smoke on two aligned windows | | 11.9 |
| 2 | the pose falsifier (11.4, 11.5) | aligned windows only | 11.9 |

### 11.3 Stage 0a: foot_raw, registered now

**What foot_raw is.** Each sidecar box stores foot_raw, the tracked box's bottom centre on that frame
(detect/types.py Track.foot_point). foot_stab is detect/footpoint.py's per-track EMA of it, with alpha 0.6 and an
occlusion correction when the box is short. At stride 6 (59.94 fps) the EMA lags by (1 - alpha) / alpha = 0.67 steps,
about 0.067 s, which is about 1 ft at 15 ft/s.

**Selection and gate.**
- The boxes are D207's, recomputed with tools.p8_piibox's functions (the same two-stage assignment, tracks, classes
  and select_items): every P-ii box (all classes), and every classified hit box (stage A boxes of hits, all classes).
- Per window, the recomputed class counts for each kind (P-ii, hit) must equal D207's record
  (work_ref/piibox/<game>.json). D202's six count fields must equal its record. Else the run stops.
- hou_cle_2016 is refused in code before any file is read.

**Distances.** On each box:
- under T: d = |T(p) - SportVU_j|, with p mapped as the scorer maps foot_stab (float32 pixels through H_truth);
- under B: the same through the sidecar's accepted H (frames without one have no B distance);
- for p = foot_stab, p = foot_raw, and p = the null (11.5);
- the scorer-ghost flag under T for each foot (more than 3.0 ft from all ten players).

**Rules for foot_raw**, the same five as for the pose hybrid (11.4), with foot_raw as the candidate and coverage 1:
1. R_raw - R_null at least 0.10 on own (all four) P-ii boxes. R is the share within 2.5 ft under T.
2. The median of d_raw,T - d_stab,T on own P-ii boxes at most -0.5 ft, with the upper end of its window interval
   below 0.
3. The same median under B at most 0 (point estimate).
4. On hits: the median of d_raw,T - d_stab,T at most +0.10 ft, and the share beyond 2.5 ft at most 0.05.
5. est at most -0.010. est is defined in 11.4 (F4), with its window interval and the ghost change beside it.

**Readings of stage 0a, registered now.**
- (a) **The regression-to-the-mean baseline.** P-ii boxes are selected because foot_stab lies 2.5 to 5 ft off, so any
  other foot, even a noisy one, lands nearer on average. R_raw, R_null and the medians of rules 1 and 2 are that
  baseline. The pose hybrid's rules 1 and 2 are read beside it.
- (b) **The EMA lag.** R_raw on own P-ii boxes is reported by SportVU player speed (under 5, 5 to 10, 10 ft/s and
  over; S.Moments.speed). If the lag is a cause, R_raw rises with speed. This is reported, not ruled.
- (c) If foot_raw itself clears rules 1, 2, 3 and 5 on own P-ii boxes on both games, that is a finding about the
  stabiliser. It is reported and not adopted, and it needs its own D line.
- (d) If foot_raw fails rule 4 on either game, the pose arm is read on the hybrid only (11.4).

Stage 0a stops nothing about the pose line.

### 11.4 The two pose arms (blocker 1)

From the same pose outputs:
- **Pose-raw:** the lifted ankle ground point itself. It replaces the box bottom on that frame, so it is compared with
  **foot_raw**. Its rule 4 is stated against foot_raw (median d_pose - d_raw on hits at most +0.10 ft, share lost at
  most 0.05), not against foot_stab. It is report only.
- **Hybrid:** foot_stab plus the pose ground point's pixel offset from foot_raw on that frame,
  p_hyb = foot_stab + (p_pose - foot_raw). It keeps the stabiliser's smoothing and adds the per-frame pose correction.
  **Rules 1 to 5 are read on the hybrid against foot_stab, and only the hybrid's failure stops the line.**

F4, the net estimate, made explicit:

est = (-n_rescued,P-ii / c_P-ii + s_hit x n_lost,hit / c_hit) / in-frame players, where:
- n_rescued,P-ii counts P-ii boxes (all classes) with a candidate foot that comes within 2.5 ft under T;
- n_lost,hit counts measured hit boxes with a candidate foot that leaves 2.5 ft;
- c_P-ii and c_hit are the two coverages, kept separate (the share of each kind with a candidate foot; 1 for foot_raw);
- s_hit = N_hits / measured hits. N_hits is D205's hit class count of in-frame players (all hits by the nearest-foot
  rule, including D207's unclassified ones), so the measured hits' loss share is applied to every hit.

The bootstrap interval of est resamples windows (the numerator and the in-frame players per window). Beside it, the
ghost change per box is (sum over measured P-ii boxes of ghost_new - ghost_stab, plus s_hit times the same sum over
measured hits) / non-referee boxes. It ignores players rescued through another player's box, shared boxes and boxes of
other kinds, so it is an estimate, not the scorer's row.

### 11.5 The same-run null (blocker 2)

For each box b, its null foot is b's foot_stab plus another box k's offset from the same frame, scaled by box height:
- k is the non-referee box on the same frame, other than b and with a candidate foot, whose height is nearest b's
  (ties: lower index);
- the offset is k's candidate offset (p_pose - foot_raw for the hybrid; foot_raw - foot_stab for stage 0a), times
  h_b / h_k;
- with no such k, there is no null for b, and this is counted.

R_null is the share of own P-ii boxes whose null foot lies within 2.5 ft under T. **Rule 1 becomes R_own - R_null at
least 0.10** (it was R_own at least 0.20). The null takes in regression to the mean and the size of a generic
per-frame offset. Only a per-box signal beats it.

### 11.6 Stage A, decode alignment (blocker 3)

- **All 156 pool windows (73 and 83), detector only, no download.** Each window is read in tools/p8_a3run.py's order:
  the pre-gate pass, then frame 0 and every stride frame to the first processed frame, then the processed frames with
  gaps filled, with no seek inside the window.
- **The form** is P8_recallcause.md 8.10 B2's (branch r-p8-recallcause, not merged; its functions are copied with
  credit, not imported):
  - r23's detector (models/player_detector.pt, SHA-256 checked; CPU, confidence 0.40, NMS 0.45) predicts at f only;
  - the prediction is compared with the sidecar's boxes at f - stride, f and f + stride;
  - the statistic is the share of each sidecar frame's boxes matched one to one at IoU 0.5 (Hungarian on IoU);
  - unmatched sidecar boxes and unmatched detections are counted per frame.
- **Rule:** a window is aligned when the mean share at f is higher than at both neighbours and at least 0.9.
- **Edge cases, fixed now:**
  - Alignment frames are the window's first five processed frames that have both neighbours in the sidecar and at
    least 4 boxes at f.
  - A window with fewer than three such frames is "undetermined", left out and counted.
  - A tie between f and a neighbour reads not aligned.
  - A frame the reader cannot decode makes the window undetermined.
  - Neighbour frames with no boxes cannot occur, since a neighbour needs a sidecar row, and an empty row is excluded.
- **Agreement with P8_a3align.md:** on the windows both cover, a cross table of this verdict against a3align's
  "misaligned". a3align compared the corpus decode with the build's, so the two questions differ, and the table is
  reported, not ruled. A window a3align marked misaligned is expected to fail here, because this read order matched
  the corpus decode in P8_a3run.md.
- **Per-game stop bar:** a3align found 8 of 62 bkn windows (0.13) and 4 of 53 cha windows (0.08) misaligned. Stage A
  is expected to exclude a share near those. The run stops and reports, and no pose stage follows, if more than 0.25 of
  a game's windows are not aligned (undetermined included).
- **Writes:** work_ref/posefoot/alignment/<game>.json and the alignment block of the record. The pose stages read
  aligned windows only.

### 11.7 Should-fix items

4. **Lift.**
   - A unit test checks the sign: for a camera above the floor looking down, the lifted ankle's ground point lies
     nearer the camera than the unlifted one.
   - 0.35 ft is reported beside 0.25 ft and the unlifted point. The rule reads 0.25 ft.
   - Section 1's sentence is corrected in 11.1.
5. **Colour.** The decoder returns BGR. The frame is converted to RGB before the image processor, and a test checks the
   channel order on a synthetic frame.
6. **Hits and est.**
   - The pose control samples up to 2 hit boxes per scored frame, from all scored frames (not only P-ii frames), with
     numpy default_rng(0) in frame order. Stage 0a uses every classified hit.
   - est keeps the two coverages apart and names N_hits (11.4).
   - est has a window bootstrap interval, and the ghost change is reported beside it.
7. **Ankles.** Rules 1 to 5 read boxes where both ankles are confident. Boxes with one confident ankle are reported
   apart, with the same rows.
8. **Licence and pinning, before any download.**
   - The upstream repository's licence file (github.com/ViTAE-Transformer/ViTPose, LICENSE) and the COCO dataset's
     terms of use (cocodataset.org) are read at the source and recorded in docs/DATA_SOURCES.md. Any "to check" left
     there blocks the download.
   - The Hugging Face revision is pinned as a commit hash in the tool (from_pretrained(revision=...)).
   - After the one download, every load runs with HF_HUB_OFFLINE=1.
9. **Decode cost, corrected.** Section 8's "about 1 to 3 minutes per game" was wrong. a3run's read order decodes the
   whole clip (the pre-gate pass plus every stride frame), about 5 to 20 s a window by the earlier P8 runs' order of
   magnitude, so about 13 to 52 minutes for 156 windows, per pass. Stage A makes one pass, and stage 2 another. Stage
   1's smoke measures the rate, and stage 2's estimate becomes about 60 to 130 lock minutes. Its cap of 120 minutes
   holds: if the smoke projects above it, stage 2 does not run, and the lead decides.
10. **hou_cle_2016** is refused by every stage before any file is read, and a test checks it.

### 11.8 Licence and download (stage 0)

Unchanged from section 8, except that it now follows stage A and needs 11.7 (8) closed first.

### 11.9 Locks and caps

- Stage 0a: one lock per game ("p8 posefoot 0a <game>"), cap 15 minutes each.
- Stage A: one lock per game ("p8 posefoot align <game>"), cap 60 minutes each.
- Stages 1 and 2: as section 8, with 11.7 (9)'s estimate.
- Each stage runs as one background chain and is reported the moment it ends.

### 11.10 Note on the stage 0a tool (2026-11-27, before any number)

The tool's D207 gate is stricter than 11.3 states. Per window, it checks the ordered list of (kind, class, box height)
rows against D207's record, not only the class counts.

## 12. Version 1.2 (2026-11-27, after the code check of 323b90ce: GO for stage 0a with fixes, NO-GO on stage A's rule; before any number)

12.1 **Rule 1 on the same boxes (A1).** R and R_null are computed on the same own P-ii boxes, those that have a null
foot. R over all own boxes is reported beside it.

12.2 **Speed (A2).** The count of own P-ii rows with no SportVU speed is reported. The speed split sees player motion
only. Camera pan moves every box on the frame alike, so its lag is common to the frame and not in this split.

12.3 **Wording.** foot_raw is the tracked box's bottom centre. The ultralytics tracker's boxes are its Kalman state,
not the raw detection, so foot_raw is not unfiltered. Only foot_stab's EMA and occlusion correction are absent from
it. Where 11.3 calls it "the per-frame box bottom, no EMA", read "the tracked box's bottom, without the foot
stabiliser".

12.4 **Nulls.**
- A null partner whose box overlaps b at IoU 0.5 or more is excluded, and the exclusions are counted.
- Beside the same-frame null, a cross-frame null is reported, not ruled: the nearest-height non-referee box on the
  window's first scored frame at least 1 s after f (else the last at least 1 s before), its offset scaled the same way
  and added to b's foot_stab.
- A test checks that the module's only writes are its records (atomic_write at the two record paths).

12.5 **est (11.4).** The coverage division is dropped before stage 2. est's numerator counts rescued P-ii boxes and
lost hits among the boxes with a candidate foot, without dividing by coverage. Coverage is reported beside it. For
stage 0a, coverage is 1, so nothing changes there.

12.6 **Stage A's alignment rule is not run as written in 11.6.** It is re-registered in its own dated section before
any stage A code, and dry-run read-only on reports/research/p8_a3align.json's stored offsets before any lock.
