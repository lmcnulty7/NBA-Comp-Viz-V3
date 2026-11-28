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

## 13. Result of stage 0a (2026-11-27, from bda0e954; record reports/research/p8_posefoot_0a.json)

**Run.** One lock per game, from work_ref/logs/posefoot.log's heavy-job lines: 57 s for gsw_cha_2016 and 57 s for
gsw_bkn_2015 (the collects' own times were 51.0 and 51.4 s). The report took 7 s.

**Gates.** D202's six fields and D207's ordered (kind, class, height) rows were equal on every window (73 and 83), and
the manifests were unchanged.

**Counters** (gsw_cha_2016, gsw_bkn_2015):
- no same-frame null partner: 0, 0;
- partners excluded at IoU 0.5: 63, 71;
- no cross-frame partner: 107, 139;
- no SportVU speed: 3, 1, all on hit rows, none on own P-ii rows.

**The five rules for foot_raw (11.3, with 12.1):** own P-ii boxes number 1,831 and 1,737, and hits 21,871 and 22,005.

| rule | bar | gsw_cha_2016 | gsw_bkn_2015 | outcome |
|---|---|---|---|---|
| 1. R - R_null (same boxes) | at least 0.10 | 0.1496 [0.1234, 0.1763] - 0.1333 [0.1110, 0.1577] = +0.016 | 0.1387 [0.1102, 0.1695] - 0.1048 [0.0871, 0.1239] = +0.034 | fails both |
| 2. median d_raw - d_stab, T | at most -0.5 ft, upper end below 0 | +0.048 ft [-0.005, +0.141] | +0.010 ft [-0.033, +0.059] | fails both |
| 3. median, B | at most 0 | +0.077 ft [+0.005, +0.157] | +0.025 ft [-0.015, +0.083] | fails both |
| 4. hits: median d_raw - d_stab, T; share lost | at most +0.10 ft; at most 0.05 | +0.084 ft [+0.061, +0.106]; 0.0597 [0.0508, 0.0687] | +0.041 ft [+0.022, +0.063]; 0.0486 [0.0411, 0.0572] | fails gsw_cha_2016 (share lost), passes gsw_bkn_2015 |
| 5. est | at most -0.010 | +0.0290 [+0.0213, +0.0369] | +0.0211 [+0.0140, +0.0289] | fails both |

- Beside est, the ghost change per box is +0.0169 [+0.0098, +0.0233] and +0.0094 [+0.0039, +0.0155].
- s_hit is 1.029 and 1.041, from D205's hit class counts of 22,501 and 22,909.
- The rescue share over all P-ii boxes is 0.155 and 0.152.
- Pooled: R 0.144 [0.126, 0.164] against R_null 0.119 [0.103, 0.135]; median d_raw - d_stab +0.025 ft under T and
  +0.049 ft under B; est +0.025 [+0.020, +0.030].

**Readings (11.3), as registered.**
- (a) **Regression-to-the-mean baseline.** About 0.10 to 0.15 of own P-ii boxes come within 2.5 ft under any
  per-frame perturbation of the foot:
  - foot_raw: 0.150 and 0.139;
  - the same-frame null: 0.133 and 0.105;
  - the cross-frame null (12.4, report only): 0.118 and 0.136, against R 0.149 and 0.140 on its boxes.

  The pose hybrid's rule 1 must beat its own same-run null by 0.10 on top of this.
- (b) **The EMA lag.** foot_raw is the bottom of the tracker's Kalman-state box, so the tracker's own lag is in both
  feet. This split tests only foot_stab's EMA and occlusion step. R_raw on own P-ii boxes by SportVU speed (under 5,
  5 to 10, 10 ft/s and over; n in brackets):
  - gsw_cha_2016: 0.167 (980), 0.128 (632), 0.132 (219);
  - gsw_bkn_2015: 0.130 (1,015), 0.148 (546), 0.159 (176);
  - the intervals overlap.

  R_raw does not rise with speed on gsw_cha_2016, and the rise on gsw_bkn_2015 is inside its intervals. So the
  stabiliser's lag is not visible as a cause of P-ii, as far as player motion shows it (12.2: pan lag is common to the
  frame and not in this split). "Not visible" has low power: the 10 ft/s and over bins hold about 200 boxes each, and
  their intervals are wide (0.079 to 0.188 and 0.086 to 0.235).
- (c) **No finding about the stabiliser.** foot_raw does not clear rules 1, 2, 3 and 5.
  - Its median distance is slightly larger than foot_stab's under both T and B. That holds for the point estimates;
    only gsw_cha_2016 under B has an interval that excludes 0.
  - est, an estimate and not the scorer's row, is positive (+0.025 pooled).
- (d) **The pose arm reads on the hybrid only.** foot_raw fails rule 4 on gsw_cha_2016 (share lost 0.0597).

**A registered risk for the pose hybrid (stated after stage 0a; it changes no bar).**
- The chance floor (about 0.10 to 0.15 of own P-ii boxes within 2.5 ft) and foot_raw's hit loss on gsw_cha_2016
  (0.0597) both came from a perturbation of foot_raw's size: |foot_raw - foot_stab| over every box of the two games'
  builds/r23 sidecars has a median of 7.03 px (quartiles 3.39 and 13.10 px; 308,128 boxes in 156 sidecars). The figure
  is computed by report0a's offset_sizes and recorded in reports/research/p8_posefoot_0a.json
  (foot_raw_minus_foot_stab_px).
- An ankle error of 5 to 10 px on top of any real signal would move the hybrid's feet by a similar amount, and so puts
  its rule 4 at the same risk.
- The rules stay as registered in version 1.1 (self-decided by the lead): loosening a bar after seeing stage 0a would be
  a forking path.

Stage 0a stops nothing about the pose line (11.3). Its next step, stage A, waits for the re-registered alignment rule
(12.6).

## 14. Stage A's alignment rule, re-registered (2026-11-27, after the code check of 323b90ce; before any stage A code or lock)

This replaces 11.6's rule. 11.6's read order, its scope (all 156 windows, detector only, no download), the a3align
cross table and its writes stand.

**Frames.**
- Eligible frames: the window's processed frames that have both neighbours (f - stride and f + stride) in the sidecar
  and at least 4 boxes at f.
- The sidecar's own box motion at f is the median, over the tracks present at f - stride and at f + stride (matched by
  tid), of the box centre's displacement in pixels between those two frames.
- The eligible frames, in time order, are cut into 15 equal segments. In each segment, the frame with the largest
  motion is taken (ties: the earlier frame).
- A window with fewer than 10 eligible frames is undetermined, left out and counted.

**Statistic.**
- r23's detector predicts at f only (11.6).
- For each of the sidecar's frames f - stride, f and f + stride, the sidecar's boxes are matched to the predictions
  one to one, by Hungarian on IoU.
- The mean IoU is taken over the sidecar's boxes, with unmatched sidecar boxes and pairs below IoU 0.1 counting as 0.
  This is M_-1, M_0 or M_+1, pooled over the window's chosen frames.
- The level at f, L_0, is a3align's: among the sidecar's boxes whose matched IoU at f is at least 0.3, the median of
  that IoU.
- Unmatched sidecar boxes and unmatched detections are counted per frame.

**Rule (fixed now).**
- A window is **misaligned** when a neighbour beats f by the margin (M_-1 or M_+1 at least M_0 + 0.05), or when the
  level at f is too low (L_0 under 0.75). Otherwise it is **aligned**. Ties read aligned.
- 0.05 and 0.75 are a3align's thresholds, kept for the reasons it gives: players move between processed frames, and a
  matched tracked box sits near 0.9 IoU.
- A frame the reader cannot decode makes the window undetermined.
- Per-game stop bar, unchanged from 11.6: if more than 0.25 of a game's windows are misaligned or undetermined, no pose
  stage follows.

**Dry run, read-only, before any lock** (reports/research/p8_a3align.json's stored per-window offsets; 2 s, no
lock). The new statistic was not stored by a3align, so the dry run applies the rule's form to a3align's stored share
S_k at offsets -1, 0 and +1 only, and to its stored median same-frame IoU:

| margin | gsw_bkn_2015 (62) | gsw_cha_2016 (53) | gsw_sac_2015 (60, not in this item) | agreement with a3align's verdicts |
|---|---|---|---|---|
| 0.05 (registered) | 8 (0.129) | 4 (0.075) | 2 (0.033) | identical on every window |
| 0.02 (sensitivity) | 10 (0.161) | 7 (0.132) | 3 (0.050) | adds 2, 3 and 1 windows |
| 0.10 (sensitivity) | 7 (0.113) | 3 (0.057) | 2 (0.033) | misses 1, 1 and 0 |

At the registered margin, the ±1 rule reproduces a3align's 13 and 8 percent exactly. Its offsets of ±2 added no
window. The level rule alone flags 7, 2 and 1 windows.

Limits of the dry run:
- It checks the thresholds on a3align's statistic (the share of corpus raw boxes at IoU 0.5), not on stage A's mean
  IoU.
- a3align compared the corpus decode with the sidecar on 62 and 53 windows of the old truths. Stage A compares a
  build-order decode on the 83 and 73 windows of the decided truths.

Stage A's own result may therefore differ. The 0.25 stop bar leaves room for about twice a3align's bkn share.

### 14.1 Note on the order (2026-11-27)

The margins in section 14's dry-run table were computed read-only before section 14 was committed, and section 14 was
committed with its dry run in one commit (127beebe). The registered margin, 0.05, is a3align's own threshold, chosen
for that reason and not from the table. The other margins are labelled sensitivities. No stage A number exists yet.

### 13.1 Note (2026-11-27, ml-review of stage 0a, PASS)

Readings (b) and (c) above were reworded per the review. The n per speed bin and the offset-size block were added to
the stage 0a record by report0a, regenerated from the stored stage 0a files. No row was rescored: every other field of
the record is identical to 26a7c761's.

### 14.2 Version 1.3 (2026-11-27, after the stage A code check of 062e5e76, GO WITH CHANGES; before any lock)

The code check ran stored a3run detections through 062e5e76's functions, with no lock. It found two problems:
- The 14 rule passed windows the decode does not reproduce: M_0 0.60 and 0.75, L_0 0.79 and 0.83, on
  gsw_bkn_2015_s07_w00 and gsw_cha_2016_s07_w00.
- The cause: stage A filled the build's long jumps sequentially, where the build seeks. build_trajectories.py jumps to
  each pre-gate interval's start (lines 217 to 224). videoseq.SeqReader seeks when a jump is backwards or longer than
  300 frames (videoseq.py lines 28 to 35). The build also opens the clip with cv2.CAP_FFMPEG first (line 172).

Self-decided by the lead: fix the decode, not only the detection. Where this differs from section 14, it rules.

**The read path (replaces 11.6's read order).** Stage A reproduces the build's own reads (tools/p8_posefoot.py
build_read, with pregate_intervals and main_reads copied from build_trajectories.py with credit):
1. Open with cv2.CAP_FFMPEG, else the default backend. cap.getBackendName() is recorded.
2. Run the pre-gate pass with the build's gate, at its pg_stride and pg_pad, and merge the intervals. The gate is gate
   v2, as sportvu.autolabel.gate_args gives the label builds: the main checkout's trained_head_v2 at thresholds.json
   "v2". It runs on the device the window's build record resolved for its gate (mps on these builds).
3. Run the tracking loop's reads through one videoseq.SeqReader: jump to each interval's start, step by the stride,
   and stop at the build's max_frames ((f_end - f_start) // stride + 2). A frame counts towards max_frames when it has
   a sidecar row, which stands in for the gate's per-frame decision.
4. Record every seek (pre-gate and main) per window.
5. **Read validation:** every sidecar frame must be among the replayed reads. If not, the window is undetermined
   ("the build's read path is not reproduced").

Limit: the gate is rerun now on the same device type. A borderline gate score that differs from the build's would
move an interval edge. The validation catches a lost sidecar frame, but not an extra read in a dead gap. The detector
check below is the final arbiter.

**The rule (replaces section 14's thresholds):**
- A window is **misaligned** when M_0 is under 0.82, or L_0 is under 0.85, or a neighbour beats f by 0.05
  (M_-1 or M_+1 at least M_0 + 0.05). Otherwise it is aligned, and ties read aligned.
- An eligible frame now also needs boxes at both neighbours.
- **Forking path, named:** 0.82 and 0.85 were calibrated on 5 windows labelled good or bad by a3run's reproduction
  check (P8_a3run.md), not by any stage A output. The 13 and 8 percent dry run of section 14 does not apply to these
  thresholds.

**What a stop means (registered now).** If more than 0.25 of a game's windows are not aligned (misaligned or
undetermined), the pose line pauses for a decode fix. It does not run on the aligned subset, and no threshold is
moved.

**A 6-window smoke under the lock, before the full pass** (align GAME --smoke, then report_align --smoke):
- gsw_bkn_2015_s04_w00, gsw_cha_2016_s05_w00 and gsw_cha_2016_s05_w03: known good (a3run reproduced them);
- gsw_bkn_2015_s07_w00 and gsw_cha_2016_s07_w00: known bad under the old read;
- gsw_bkn_2015_s00_w07: the first window by name whose first processed frame lies between 1 and 300, with no gap over
  300 frames (first frame 54, 114 frames).
- **Gate:** the three good windows read aligned. With the build's read path, the two formerly bad windows should now
  read aligned. If they still read misaligned, the new rule must flag them, which it does by construction.
- M_0, L_0, the read validation, the backend and the seeks of all six are reported before the full pass. If the gate
  fails, the full pass does not run, and the lead decides.

**Cost.** The pre-gate pass adds one CLIP embedding per pg_stride frame (about 1 in 30), on the build's gate device.
The decode is one pass per window, as before.

### 14.3 Stage A smoke result (2026-11-27, from 7584a109; record reports/research/p8_posefoot_align_smoke.json)

**Run.** One lock per game: 60 s for gsw_cha_2016 and 59 s for gsw_bkn_2015. The report took 1 s. The manifests were
unchanged. The backend was AVFOUNDATION in every window: this cv2 4.13.0 has no FFMPEG backend, so the build's
CAP_FFMPEG open falls back the same way in this runtime. The gate ran on mps, as each window's build record resolved.

| window | role | M_-1 | M_0 | M_+1 | L_0 | verdict | sidecar frames read | extra reads | main seeks |
|---|---|---|---|---|---|---|---|---|---|
| gsw_cha_2016_s05_w00 | good | 0.493 | 0.885 | 0.515 | 0.904 | aligned | 1,053 of 1,053 | 54 | [0] |
| gsw_cha_2016_s05_w03 | good | 0.604 | 0.886 | 0.646 | 0.904 | aligned | 206 of 206 | 25 | [0] |
| gsw_cha_2016_s07_w00 | bad under the old read | 0.590 | 0.887 | 0.605 | 0.895 | aligned | 968 of 968 | 27 | [150, 4440] |
| gsw_bkn_2015_s00_w07 | first frame 54 | 0.637 | 0.903 | 0.742 | 0.915 | aligned | 114 of 114 | 22 | [0] |
| gsw_bkn_2015_s04_w00 | good | 0.574 | 0.894 | 0.589 | 0.908 | aligned | 1,102 of 1,102 | 24 | [0] |
| gsw_bkn_2015_s07_w00 | bad under the old read | 0.576 | 0.890 | 0.605 | 0.906 | aligned | 910 of 910 | 65 | [240] |

**Gate: passes.** The three good windows read aligned, and the two formerly bad windows now read aligned too.

- Extra reads are frames the build's gate rejected inside its intervals. The build reads them too, but this cannot be
  confirmed per frame without the per-frame gate.
- The build records no seeks. The replay's seeks agree with what the sidecar shows: each sidecar gap over 300 frames
  lies either inside a replayed interval or at a replayed seek.
- Post hoc, not tested: the two formerly bad windows are the ones whose first interval starts after frame 0 (at 150
  and 240), which the old sequential read did not seek to.

The full pass waits for the lead's read of this smoke and a code check of the read-path code.

### 14.4 Before the full pass (2026-11-27, the code check of the read path: GO)

- The run sets HF_HUB_OFFLINE=1 (no hub fetch) and PYTORCH_ENABLE_MPS_FALLBACK=1 (as sportvu.rebuild sets it for the
  build). The tool refuses to run without both.
- Each game record holds the gate's provenance: the head's and its coefficient file's SHA-256, the "v2" threshold value
  and the CLIP backbone id.
- report_align lists the windows that are not aligned, by cause, so the stop share reads by cause. Five windows are
  undetermined before any decode because they have fewer than 10 eligible frames: gsw_cha_2016_s01_w05, s03_w02,
  s06_w02 and s07_w06, and gsw_bkn_2015_s00_w05. These count toward the 0.25 stop share.
- **The smoke mainly confirmed the read path.** Five of its six windows were the windows the 0.82 and 0.85 thresholds
  were calibrated on, so the smoke is not a test of those thresholds. The full pass is.

### 14.5 Stage A full pass result (2026-11-27, from 105d8f92; record reports/research/p8_posefoot_align.json)

**Run.** One lock per game, from work_ref/logs/posefoot.log's heavy-job lines: 579 s for gsw_cha_2016 and 657 s for
gsw_bkn_2015 (the collects' own 572.3 and 651.0 s). The report took 1 s. The manifests were unchanged, HF_HUB_OFFLINE
and the MPS fallback were set, and the gate was trained_head_v2 (coefficients SHA-256 b95a30fd...1fcc2c) at the v2
threshold 0.33 with openai/clip-vit-base-patch32 on mps.

| | gsw_cha_2016 | gsw_bkn_2015 |
|---|---|---|
| windows | 73 | 83 |
| aligned | 69 | 82 |
| misaligned | 0 | 0 |
| undetermined, fewer than 10 eligible frames | 4 (s01_w05, s03_w02, s06_w02, s07_w06) | 1 (s00_w05) |
| share not aligned (stop above 0.25) | 0.055 | 0.012 |
| read validation (every sidecar frame read) | 69 of 69 | 82 of 82 |
| lowest M_0; median M_0 | 0.866; 0.892 | 0.841; 0.892 |
| lowest L_0 | 0.873 | 0.853 |
| largest neighbour minus M_0 | -0.093 | -0.056 |

**Reading by the registered rule: no stop.** The pose stages may use the 151 aligned windows. The five undetermined
windows are left out by their cause.

The 0.82 and 0.85 thresholds were not near binding: the lowest M_0 is 0.841 and the lowest L_0 0.853, both on
gsw_bkn_2015, and no neighbour came within 0.05 of f.

**Post hoc, not tested by a registered rule.** All 12 windows that P8_a3align.md called misaligned (4 of gsw_cha_2016,
8 of gsw_bkn_2015) read aligned here, under the build's own read path. So a3align's misalignment was a property of the
corpus's sequential read, not of the build. This fits P8_a3run.md's controls, where a sequential read matched the corpus
and not the sidecars. It bears on the analyses a3align listed (the E1 labels, the E1b bound, M1 to M4, detrow), which
scored the corpus JPEGs. Those may carry the offset; the build-side rows do not.

## 15. The pose stages, as coded (2026-11-27, before any pose number; tools/p8_posefoot_pose.py)

This section states how the code implements 11.4, 11.5 and 11.7 for stages 1 and 2. Where the earlier sections leave
a choice open, this fixes it before any pose number exists.

- **Start condition.** The tool refuses to start unless reports/research/p8_posefoot_align.json exists, reads stop
  false and is the full pass (not the smoke). It reads only that record's aligned windows (151). Every frame comes
  through tools.p8_posefoot.build_read, the build's own read path, with its read validation. A window that stage A
  aligned but whose read is now not reproduced stops the run.
- **Boxes.** D207's selection is recomputed and gated per window as in stage 0a. The boxes are every P-ii box, plus up
  to 2 hit boxes per scored frame:
  - the hits are drawn with numpy default_rng(0) over the game's windows and frames in order;
  - the draw runs for every window, aligned or not, so the sample does not depend on alignment.
- **Model.** The pinned ViTPose-B (MODEL_REV), loaded with HF_HUB_OFFLINE=1, runs on CPU. Each frame is converted
  BGR to RGB and the boxes are given in COCO format. The image processor's defaults are used, and nothing is tuned.
- **The ankle and the lift.** The ankle point is the midpoint of both ankles at score 0.3 or more, or the one
  confident ankle. It is lifted to 0.25 ft and dropped to the floor, through two cameras:
  - under T, the label row's own camera (camera_C, camera_rvec, camera_f);
  - under B, sportvu.camera.pose_from_h(B).

  At 0 ft the lift equals the ground mapping, which is tested. The lifted point lies nearer the camera, which is the
  sign test. 0.35 ft and 0 ft are reported beside it, for the pose-raw and hybrid distances under T.
- **The arms.**
  - Pose-raw: the lifted ground point itself.
  - Hybrid: foot_stab + (the lifted ground point's pixel - foot_raw), the pixel taken through T for T-frame distances
    and through B for B-frame ones.
- **The null (11.5, made exact).**
  - Pose runs on the two nearest-height non-referee partners of each selected box (IoU under 0.5 with it).
  - The null partner is the first of those two with a both-ankle pose. Its lifted ground pixel minus its foot_raw,
    scaled by h_b / h_k, is added to b's foot_stab.
  - With no such partner, there is no null, and this is counted.
  - "With a candidate foot" in 11.5 is read as "with a both-ankle pose", since the rules read both-ankle boxes
    (11.7 (7)).
- **The rules** read both-ankle boxes (11.7 (7)), on the hybrid, through the same unit_rows and rule_set as stage 0a,
  with rule 1 on the same boxes as the null (12.1) and est without a coverage division (12.5).
  - One-ankle boxes get the same rows apart.
  - Pose-raw's rule 4 is reported against foot_raw (11.4): the median of d_poseraw - d_raw on hits, and the share
    lost.
  - Coverage (2, 1 and 0 ankles) is reported per game and kind.
- **Stage 1, the smoke:**
  - Windows: gsw_cha_2016_s00_w00 and gsw_bkn_2015_s00_w00, the first window of each game by name, both aligned in
    stage A.
  - Gate (mechanical): the ankle point of at least 0.95 of the boxes with an ankle lies inside its box or within 0.2
    box heights below it.
  - The smoke measures the seconds per crop with the decode included. From that it projects stage 2's lock time
    against the 120-minute cap. If the projection is above the cap, stage 2 does not run and the lead decides.
  - The smoke's rules are computed for the mechanics only and are not read.
- **Writes:** work_ref/posefoot/pose[_smoke]/<game>.json and reports/research/p8_posefoot_pose[_smoke].json. A test
  checks that these are the only writes.

### 15.1 Amendment (2026-11-27, after the pose code check of 41766bda, GO WITH CHANGES; before any pose number)

Where this differs from section 15, it rules.

1. **B distances (the blocker).** pose_from_h(B) does not reproduce B: at lift 0, its floor point differs from B's
   mapping by a median of 0.5 to 1.8 ft and a p90 of 2.9 to 5.7 ft on 4 windows (the code check's measurement). So a
   camera now gives only the lift shift:
   - The ankle pixel's ray is cut at ANKLE_FT, and the floor point beneath it is projected back through the same
     camera to a pixel, p_lift. At lift 0, p_lift is the ankle pixel.
   - p_lift is mapped through B, or through T, like any foot:
     - poseraw = |H(p_lift) - SportVU|;
     - hybrid = |H(foot_stab + (p_lift - foot_raw)) - SportVU|;
     - the null is built the same way.
   - T uses the label row's camera, and B uses pose_from_h(B), for the lift only.
   - A test on a real stored B (builds/r23, gsw_cha_2016_s00_w00, frame 30) checks that at lift 0, with the ankle at
     foot_raw, the hybrid equals foot_stab and maps where foot_stab maps.
2. **est** is computed from every selected row, with a missing hybrid foot reading as no change: its distance and
   ghost flag are foot_stab's. So a failed lift is not counted as a ghost removed. Rules 1 to 4 stay on both-ankle
   boxes. This replaces 12.5's "among the boxes with a candidate foot" for the pose stages.
3. **Cap:** 60 minutes per game, checked before each window.
4. **Cost projection** (the smoke's report), per game, from three measured parts:
   - stage A's measured decode and gate seconds over the aligned windows (work_ref/posefoot/alignment/<game>.json);
   - the selection seconds measured over every window;
   - the exact crop count over all aligned windows, counted in the smoke with no pose run, times the smoke's pose
     seconds per crop.
5. **A window whose read is not reproduced** is skipped and counted, not a stop of the game. More than 3 such windows
   in a game stop the run.
6. **Cost reduction (self-decided by the lead, the reviewer's choice):** pose runs on the selected boxes only. The
   null partner is now the nearest-height **sampled hit box** on the same frame: other than b, IoU under 0.5 with b,
   both ankles. Hits stay at 2 per frame. This replaces section 15's two-partner rule and roughly halves the crops.
   - If the projection still exceeds 120 minutes in total, the registered second step is 1 hit per frame.
   - If that is still over, the item stops and the lead is told.

### 15.2 Stage 1 smoke result (2026-11-28, from 53c44192; record reports/research/p8_posefoot_pose_smoke.json)

**Run.** The fast suite ran first under the lock, and the smoke started after it. Lock times: 118 s for
gsw_cha_2016_s00_w00 and 107 s for gsw_bkn_2015_s00_w00, from the heavy-job lines. The report took 2 s.

**Mechanical gate: passes.**
- The ankle point lies inside its box, or within 0.2 box heights below it, for 1.000 of 233 boxes (gsw_cha_2016) and
  0.994 of 175 (gsw_bkn_2015). The bar is 0.95.
- Both-ankle coverage is 0.974 (P-ii) and 0.967 (hits) on gsw_cha_2016, and 0.968 and 0.947 on gsw_bkn_2015. No ankle
  was found on 0.000 to 0.009 of boxes.
- No read failed to reproduce.
- No null partner: 4 and 6 boxes.

**Cost projection (15.1 (4)): under the cap, so stage 2 runs with 2 hits per frame and no fallback.**
- Pose took 0.208 s per crop (85.5 s for 411 crops).
- Crops over the aligned windows: 9,172 (gsw_cha_2016) and 9,210 (gsw_bkn_2015).
- Projected totals: 42.0 and 43.4 minutes. Each adds stage A's measured decode and gate seconds (564.7 and 643.0) and
  the selection seconds (46.0 and 46.9) to the pose seconds.
- 85.4 minutes in all, against 120.

**Disclosure.** The smoke record computes the rule booleans for its two windows, for the mechanics. The lead saw
them while reading the record. Nothing was changed on that basis, and every bar stays as registered. The teammate did
not read them.
