# P8: is the own-box P-ii row partly a truth-side error? (design, version 1, registered before any number)

Version 1, 2026-11-28. Report only, stored files only: nothing is fitted, tuned or trained, and no truth, build or score
changes. No video is decoded, no model runs and no Colab unit is used. No held-out game, no D85 test game and no test
arena is read (CLAUDE.md rule 4). hou_cle_2016 is not read and stays unspent. No lock is taken until this design is
reviewed.

## 1. Question

D205's P-ii row (0.114 of in-frame players pooled on gsw_cha_2016 and gsw_bkn_2015) is a player with no box within
2.5 ft after one-to-one assignment, and an unclaimed box 2.5 to 5 ft away. D207 found that box to be the player's own
track in 0.541 of cases (own, all four kinds: 0.062 of in-frame players). The per-box foot routes are now closed: D206,
posefoot stage 0a (foot_raw) and stage 2 (pose ankles).

This item asks whether part of the own-box P-ii row is an error on the truth side rather than the build's:
- **a time-sync error.** The truth's time map attaches the wrong SportVU moment to the frame. Every moving player's
  SportVU position then sits behind or ahead of the video along the player's velocity, by about speed times the time
  error;
- **a camera error.** The truth camera T maps the right foot pixel to the wrong court point. The error then grows with
  depth, or with T's own per-frame foot residual.

Each has a signature in stored data. This item measures those signatures, beside hits, and bounds how much of the row
each could explain.

## 2. Known before this item

- D205 and D207's selection and counts (P8_one2one.md section 11, P8_piibox.md section 13).
- P8_truth_arenas.md acceptance check C: each window's residual clock offset is solved at the truth fit, and its
  stability is bounded (median at most 0.54 s). The label rows store the window's offset_s (-1.34 to +1.50 s on
  gsw_cha_2016) and each frame's feet_resid_ft: the median residual of the truth fit's feet, at most 2.0 ft by the
  acceptance rule, 0.23 to 2.00 ft on gsw_cha_2016.
- D184 and D186: the foot lands short along the camera ray under B, growing with depth.
- D206 (P8_footfix.md section 15): the depth toward T's image and toward SportVU under B have the same sign in every
  cell.
- Posefoot stage 0a (P8_posefoot.md section 13, branch r-p8-posefoot, under ml-review): the rescue share of foot_raw
  does not rise with player speed (low power at 10 ft/s and over). That split used foot_raw against foot_stab. This
  item uses SportVU against the box, the question a time error would answer.
- The truth frames are themselves selected on agreement between T and the feet (feet residual at most 2.0 ft), so
  frames with a large camera error are partly removed already.

## 3. Universe, selection and gate

- **Games:** gsw_cha_2016 and gsw_bkn_2015, on D202's windows and the pinned scorer's scored frames.
- **Selection:** D207's, recomputed with tools.p8_piibox's functions (D205's two-stage assignment, tracks, classes,
  select_items).
  - P-ii players with their stage B box, read in D207's own group: own (all four kinds), and the other classes apart.
  - Hit players with their stage A box (the classified hits), as the control.
- **Gate, per window, before any number is kept:**
  - D202's six count fields equal its record;
  - D207's ordered (kind, class, box height) rows equal its record (work_ref/piibox/<game>.json), as posefoot stage
    0a did;
  - sidecar, time map, labels and identity hashes equal D202's record.

  Any difference stops the run.
- **Posefoot 0a's per-box T and B distances** are not on v4-research yet, since that branch is under ml-review. The
  same quantities are recomputed here, as the scorer maps foot_stab: d_T = |T(foot_stab) - SportVU| and d_B through the
  sidecar's accepted H.

## 4. Quantities per player-frame (P-ii and hit rows)

- **Offset:** o = T(foot_stab of the row's box) - SportVU_j, in the truth's court frame, with the mirror applied as the
  scorer applies it.
- **Velocity:** v_j from the SportVU moments over plus or minus 0.2 s (tools.p8_scorecard.Moments.speed's window and
  formula, as a vector, mirror applied). Speed is |v_j|.
- **Along and across velocity:** o_along = o . v_hat (positive: the box lies ahead of the SportVU position in the
  direction of motion); o_across = the perpendicular component. A player with speed under 1 ft/s has no direction and
  is counted apart.
- **Depth and lateral:** o_depth = o . u, with u the unit vector from the label row camera's ground point (camera_C x,
  y) to SportVU_j (positive: beyond the player, away from the camera); o_lateral = the perpendicular component.
- **Image region:** the scorer's third of SportVU_j under T (tools.p8_scorecard.third_of, the truth's near side). Also
  a frame-edge flag: the player's pixel T^-1(SportVU_j) within 60 px of the image border, inside the 10 px in-frame
  margin.
- **Truth residual:** the frame's feet_resid_ft from its label row. Bins: under 1.0, 1.0 to 1.5, and 1.5 to 2.0 ft.
- **B agreement (report only):** d_B of the row's box against 2.5 ft.

## 5. Measures, each for own P-ii rows beside hit rows, with window-cluster intervals

All are per game and pooled, with 2,000 resamples (seed 0) and windows resampled whole within each game
(tools.p8_piibox's picks).

- **(1) Timing.**
  - The own P-ii rate per in-frame player, and the hit rate, by speed bin (under 5, 5 to 10, 10 ft/s and over).
  - By speed bin, the median of o_along and of |o_along| / (|o_along| + |o_across|), for own P-ii rows and hits.
  - The implied time error per window: the median of o_along / speed over the window's moving rows (P-ii and hits,
    speed at least 5 ft/s). Its distribution over windows, and the own P-ii rate in windows where it is at least
    0.1 s in size against the others.
- **(2) Camera.**
  - The own P-ii rate and the hit rate by third, by the frame-edge flag and by the residual bin.
  - The median |o| and the median o_depth of own P-ii rows by the same splits.
- **(3) Direction.**
  - The share of rows with |o_depth| > |o_lateral|, for own P-ii rows and for hits.
  - The medians of o_depth, o_lateral, o_along and o_across.
- **B agreement (report only).** The share of own P-ii rows whose box lies within 2.5 ft of SportVU under B, against
  the share of hits beyond 2.5 ft under B. These are the rows where the build's own court model and the truth camera
  disagree on this box.

## 6. Registered reading table (fixed now)

Each pattern is read per game. A pattern **holds** when all of its conditions hold on both games, by point estimate,
with the interval of the stated ratio or difference excluding the neutral value on both games.

| pattern | conditions | reading |
|---|---|---|
| **timing** | (a) the own P-ii rate at 10 ft/s and over is at least 1.5 times its rate under 5 ft/s, and that ratio is at least 1.3 times the same ratio for hits; (b) at 10 ft/s and over, the median along-velocity share of own P-ii rows is at least 0.6, and at least 0.1 above the hits'; (c) the median o_along of own P-ii rows at 10 ft/s and over has the same sign on both games and is at least 1.0 ft in size | a time-map error carries part of P-ii: the next step would be the truth's time map (a truth-side arm), not the build |
| **camera** | (a) the own P-ii rate in the residual bin 1.5 to 2.0 ft is at least 1.5 times its rate under 1.0 ft, and that ratio is at least 1.3 times the hits' ratio; or the same with the far third against the near third; (b) the depth share (\|o_depth\| > \|o_lateral\|) of own P-ii rows is at least 0.6, and at least 0.1 above the hits' | a camera error in the truth carries part of P-ii: the next step would be the truth's camera fit, not the build |
| **both** | timing and camera both hold | both are read, and the bounds of section 7 say which is larger |
| **neither** | neither holds | the own-box P-ii row is not shown to be truth-side by these signatures. It stays a build-side per-box error as far as this view can tell, with D206 and posefoot closed |

Why these bars:
- 1.5 times is a clear rise across the bins.
- 1.3 times the hits' ratio removes what speed or depth does to every box (crowding, size, blur).
- 0.6 is a clear majority direction.
- 1.0 ft is a third of P-ii's 3.2 ft median distance.

The timing pattern needs the sign to agree across games because a time-map error has one sign within a window. A
pooled sign over many windows of mixed signs would read as no timing pattern by condition (c). Section 5's per-window
implied time error is reported for that reason.

## 7. Bounds (each a ceiling, not a forecast)

- **Timing ceiling.** Own P-ii rows on moving players (speed at least 5 ft/s), less what the slow bin's rate predicts
  for them: (n_moving_own_P-ii - rate_own_P-ii,<5 x n_moving_in-frame) / n_in-frame. A stationary player cannot carry
  a time error, so the excess over the slow-bin rate is the most a timing error could explain.
  - Report only beside it: shift each window's SportVU by its implied time error times each player's velocity, and
    count the own P-ii rows whose box comes within 2.5 ft. This is in sample, so it is a ceiling.
- **Camera ceiling.** Own P-ii rows less what the lowest-residual bin's rate predicts, over the in-frame players:
  (n_own_P-ii - rate_own_P-ii,<1.0 x n_in-frame) / n_in-frame. The same is reported against the near third's rate.
  - Report only beside it: the B-agreement share of section 5 times the own P-ii rate. This is the most that
    replacing T by B could recover, and B has its own errors.
- **Neither.** The own P-ii rate (0.062) less the larger of the two ceilings.

Each ceiling is read beside D205's 0.114 and D207's own-box 0.062.

## 8. What it can and cannot tell

- Speed is confounded with crowding, pose and blur, and depth with box size and D184's short foot. The hits' ratios
  remove part of that, not all.
- The truth frames were selected on low foot residual, so a camera error large enough to fail acceptance is not seen.
- A time error smaller than about 0.1 s moves a 10 ft/s player about 1 ft. At the 2.5 ft gate that is only partly
  visible.
- Build and truth errors are not separable and not independent (D202, D205). A pattern here points to a side, it does
  not prove it.

## 9. Cost and guards

- One collect per game under the lock ("p8 piiside <game>", frozen CV python, V3_P8_ROOT set to work_ref), about 1
  minute each, as posefoot stage 0a (57 s each). The report runs in system python. About 3 lock minutes in all, cap 15
  minutes per game.
- Guards on every game, SportVU id, video id and window: tools.p8_common.guard and tools.research_testset.refuse_cv_test.
  The tool refuses hou_cle_2016 before any file is read, and a test checks it.
- SHA-256 manifests of work_ref's and work_ms's labels, builds and cache before and after each step.
- Writes go only to work_ref/piiside/<game>.json and reports/research/p8_piiside.json. A test checks that these are
  the only writes.
- One chain, reported the moment it ends.

## 10. Version 1.1 (2026-11-28, after the design review of a9b6b41b, NO-GO as version 1; before any number)

The review confirmed a circularity in version 1. The truth camera T is refit per frame to foot_stab, and each window's
offset_s is solved to minimise the feet residual. So under T, a window's time error is near zero by construction, and
"neither" was near certain whatever the truth. Version 1.1 replaces sections 4 to 7. Sections 1 to 3, 8 and 9 stand,
with the changes below.

### 10.1 The frame and the foot

- **The primary frame is B:** the sidecar's accepted per-frame H from the r23 court model. It is fitted to the
  frame's painted lines, with no SportVU and no feet.
- **The primary foot is foot_raw:** the tracked box's bottom centre, without the foot stabiliser's EMA.
- Under B and foot_raw, neither the truth's camera nor its time solve enters the offset. T enters only through the
  selection of the outcome rows (D207's classes), and the timing estimate below uses rows T does not select.
- **Acceptance pre-selects the frames where T fits the feet** (feet residual at most 2.0 ft). So the camera question
  is answered only through B, never through T's own residual.

### 10.2 Timing (replaces section 5 (1))

- **Estimation rows (non-outcome rows).** On each scored frame, in-frame players are matched one to one to non-referee
  box feet B(foot_raw) within 5 ft (the Hungarian assignment of tools.p8_one2one.assign_1to1, at 5 ft, on B's
  feet). T has no part in this selection.
- **Moving rows:** SportVU speed at least 5 ft/s, with velocity v from the moments over plus or minus 0.2 s
  (section 4's definition, mirror applied).
- **Per-window time offset:** tau_B = the median over the window's moving estimation rows of
  ((B(foot_raw) - SportVU) . v_hat) / speed, in seconds. Positive means the box lies ahead of the SportVU position in
  the direction of motion. A window with fewer than 30 moving estimation rows has no tau_B and is counted.
- **The null, registered now.** With no window-level timing error, every window shares one constant lag tau_0 (the
  tracker's, plus any constant along-velocity bias of B). The across-window spread then comes only from B's
  measurement noise. The null is simulated per game, 2,000 times (numpy default_rng(0)):
  - tau_0 is the across-window median of tau_B;
  - each window's moving rows are given along-velocity offsets tau_0 x speed + e. The e are resampled from the game's
    pooled along-velocity offsets of slow estimation rows (speed 1 to 5 ft/s, direction defined), centred on their
    median;
  - each simulated window gets the same row count and speeds as the real one, and its simulated tau is computed the
    same way;
  - the statistic is the across-window interquartile range of tau, and the null is its distribution over the
    simulations.
- **Timing rule (fixed now).**
  - **Answered, no window timing error:** the observed interquartile range of tau_B is at or below the null's 97.5th
    percentile on both games.
  - **Otherwise,** the enrichment test runs. A window is "off" when |tau_B - tau_0| is at least 0.1 s. Timing
    contributes when the own P-ii rate in off windows is at least 1.5 times its rate in the other windows, with the
    window-cluster interval of the ratio excluding 1, on both games.
  - Otherwise timing is not shown.
- **Report only:**
  - tau_B against each window's offset_s, with their rank correlation per game;
  - check C's quantities as a predictor (per window where stored, else the game-level values of the acceptance
    record: n, median and share over 0.2 s);
  - the speed-bin rates of version 1's (1), descriptive only.

### 10.3 The build's own lag (the method check)

- foot_stab's EMA (alpha 0.6 at stride 6) lags foot_raw by about 0.067 s. That gives a one-signed along-velocity
  offset of about 0.067 s times speed.
- tau_B is computed twice per window, from foot_raw and from foot_stab on the same rows. The registered check: the
  across-window median of tau_B(foot_stab) - tau_B(foot_raw) lies within 0.03 s of -0.067 s, the sign being the lag
  behind the motion.
  - If it does, the method resolves a lag of this size.
  - If it does not, the timing result is reported as "not resolvable at this noise". It is not read as "answered".
- Under the null, a constant lag (the stabiliser's, the tracker's, or a constant truth offset) cannot make a timing
  pattern. A truth timing pattern must show window-to-window spread beyond the null: different sizes or sign changes
  across windows.

### 10.4 Camera (replaces section 5 (2))

- **Disagreement:** d_TB = |T(foot_raw) - B(foot_raw)| at each own P-ii box and each hit box on the same frames.
- **B's own error beside it:** d_B = |B(foot_raw) - SportVU_j| at the same boxes.
- Both are reported by third (the scorer's third of SportVU_j under T) and pooled.
- **Camera rule (fixed now).** A camera pattern holds when both hold on both games:
  - (a) the median d_TB at own P-ii boxes is at least 1.5 times the median d_TB at hit boxes on the same frames, with
    the window-cluster interval of the ratio excluding 1;
  - (b) at own P-ii boxes, the share with d_B at most 2.5 ft is at least 0.5. Where T and B disagree, B places the
    foot near the player.

  When it holds, T's camera is the likelier side at those boxes; that is the reading. It is not a proof, since B has
  its own error.

### 10.5 Dropped, and decided

- **The residual branch** (T's feet residual) is circular, because acceptance and the per-frame refit both minimise
  it. It is reported descriptively only, with no rule.
- **The depth-share test** is dropped. The depth shares were seen during the review: 0.764 and 0.785 for own P-ii rows
  against 0.727 and 0.725 for hits. That condition is decided, not a test. It is quoted here and in the result as
  seen.
- **The 1.3 times the hits factor** is replaced by build-side controls, reported beside the speed-bin rates and not
  ruled:
  - foot_raw against foot_stab by speed bin: the median along-velocity difference, expected near 0.067 s times
    speed;
  - B offsets by speed bin on the estimation rows: the median along-velocity and across-velocity components.

### 10.6 Reading table (replaces section 6)

| pattern | condition | reading |
|---|---|---|
| timing | 10.2's enrichment holds on both games, with 10.3's method check passing | a window-level time-map error carries part of P-ii; the next step is the truth's time map |
| timing answered | 10.2's spread is inside the null on both games, with 10.3 passing | no window-level timing error beyond B's noise; timing is ruled out at this resolution |
| timing not resolvable | 10.3's method check fails | no timing reading |
| camera | 10.4's rule holds | T's camera is the likelier side at own P-ii boxes; the next step is the truth's camera fit |
| neither | neither timing nor camera holds | not shown truth-side by these tests |

### 10.7 Bounds (replaces section 7; each is a bound, not a forecast)

- **Timing, an upper bound:** the own P-ii rows in off windows, less what the other windows' own P-ii rate predicts
  for them, over in-frame players.
- **Camera, an upper bound:** the own P-ii rows whose B foot lies within 2.5 ft of the player (d_B at most 2.5 ft),
  over in-frame players. These are the most a T-side camera error could explain.
- **Neither, a lower bound on the build-side share:** the own P-ii rate (0.062) less the two upper bounds.
- All are read beside D205's 0.114 and D207's own-box 0.062.

### 10.8 Cost and gates

Unchanged from section 9 (about 3 lock minutes, stored files only), with two additions:
- The selection gate of section 3 still runs, with D207's ordered rows and D202's fields.
- The null simulation runs in the report step (system python, no lock).

## 11. Version 1.2 (2026-11-28, after the second review of 432b7fed, GO WITH CHANGES; before any code or number)

Where this differs from section 10, it rules. It adds no lock time.

### 11.1 The null's noise (replaces 10.2's null)

- For each moving estimation row, the noise is drawn as a resampled offset vector (B(foot_raw) - SportVU) from the
  same window's slow estimation rows (speed 1 to 5 ft/s), projected onto the moving row's own v_hat.
  - With fewer than 10 slow rows in the window, the draw falls back to the same game's slow rows in the same third
    (the scorer's third of SportVU under T).
  - Drawing the vector, not a centred scalar, keeps B's per-window bias and the depth-dependent short foot (D184, D186)
    inside the null.
- The null's tau_0 stays the across-window median of tau_B. The statistic stays the across-window interquartile
  range.
- **Report only:**
  - tau_B on TRACK-state frames only, as a sensitivity;
  - a lateral-axis estimator: the median over rows of (o . w) / (v . w), where w is the unit vector across the camera
    ray (o is the offset, v the velocity), on rows with |v . w| at least 5 ft/s.

### 11.2 Method check and power (replaces 10.3)

- **Synthetic shift.** SportVU is re-read at plus and minus 3 moments (plus and minus 0.12 s at 25 Hz) on the same rows,
  and tau_B is recomputed. The across-window median of tau_B must move by that shift, within 0.02 s, in both directions
  on both games. If it does not, timing reads "not resolvable".
- **Power.** A shift of plus or minus 0.1 s (sign at random) is injected into a quarter of each game's windows (chosen
  at random), on top of the null's simulated rows, 2,000 times (seed 1). "Answered" (the spread inside the null) may be
  read only if, in at least 80 percent of those simulations, the interquartile range exceeds the null's 97.5th
  percentile. Otherwise timing reads "not resolvable at this noise".
- **The EMA difference is report only:** the across-window median of tau_B(foot_stab) - tau_B(foot_raw), beside its
  prediction from the stored foot_stab - foot_raw pixel offsets. The prediction is the same quantity computed with
  those pixel offsets mapped through B in place of SportVU.

### 11.3 Camera rule (replaces 10.4's rule)

- **(a), within third.** The ratio of median d_TB at own P-ii boxes to median d_TB at hit boxes on the same frames is
  computed within each third, then combined as the median of the per-third ratios, weighted by own P-ii rows. Its
  window-cluster interval must exclude 1, with the point at least 1.5.
- **(b), against a build-side null.** For each own P-ii row, a null T offset is its own T offset plus the T-minus-B
  disagreement vector of a same-frame hit box drawn at random (seed 0, 2,000 draws). The share of own P-ii rows with
  d_B at most 2.5 ft must exceed the null's 97.5th percentile of the same share.
  - Hits' own d_B share is reported beside it as the T-side reference.
- The camera pattern holds when (a) and (b) both hold on both games.

### 11.4 Off windows and acceptance (adds to 10.2)

- At least 5 off windows per game are needed for the enrichment test. With fewer, timing reads "not resolvable" on that
  game.
- Windows the truth's acceptance rejected cannot contribute: they have no truth rows. The offset solve is known to be
  fragile across builds. D99: small build differences move a window's offset by 0.3 to 0.8 s. D166: the median offset
  difference between builds that differ only in the court model fell from 0.52 to 0.367 s under truth v2, not to its
  0.26 s bar. A window whose time error is large enough to fail acceptance is therefore not seen here.

### 11.5 Is B independent of the truth on these games? It is not (item 6, checked against the training record)

The r23 court model was **trained on labels derived from H_truth** on both pool games.
- ROADMAP_LOG.md pass 13: sportvu/grid_dataset.py built r23's training set from "every 2nd kept R1.3 label", with
  "labels written by generate_labels' own functions through H_truth".
- R1.3's labels are the camera-model truth of the four training games, gsw_bkn_2015, gsw_cha_2016, gsw_ind_2016 and
  gsw_sac_2015 (tools.p8_frozen.R13_TAGS; P8_FEASIBILITY.md section 2.3's table).
- That truth placed one camera per game from pooled feet-to-SportVU pairs, then fitted pan, tilt and zoom per frame to
  feet and painted lines.
- tools.p8_scorecard already marks r23's rows on these games "IN SAMPLE".

So B on gsw_cha_2016 and gsw_bkn_2015 comes from a court model trained to reproduce an earlier SportVU-fitted truth's
court projection on these same games. B is not a frame free of SportVU here.

- B's per-frame H still comes from the frame's own painted lines through the tracker. But its keypoints were learned
  from truth-projected labels of these frames' games, possibly of these very windows (R1.3's kept frames).
- The independence 10.1 claims for B does not hold on these games. The R1.3 truth is the old frozen camera fit, not
  multipool v1. It shares the feet-to-SportVU construction, and its per-window offsets came from the same fragile solve
  (11.4).
- What this does to each test:
  - **Timing:** a per-window time error in R1.3's truth would have moved that truth's per-frame camera, and so the
    court labels r23 learned. Pooled training over many windows averages such errors, but r23 may have learned them in
    sample. So tau_B could understate a truth-side timing error that B shares.
  - **Camera:** T against B compares two descendants of SportVU-fitted truths on these games. Agreement between them
    is weaker evidence of a correct camera than it would be from a model that never saw these games.
- **This is a finding the lead must rule on before any code**, not a detail to note and pass over. The options:
  - (i) run as registered, with this limit in every reading;
  - (ii) wait for a court model not trained on these games (none exists in the repository: the public court model of
    ROADMAP R2.2 is "far behind V3", and V3's own grid model was trained on Roboflow nbacourt frames only, per FIX_PLAN.md's
    provenance note);
  - (iii) close the item as not measurable with an independent frame.
- A SportVU-free H does exist for every frame: the V3 label build's H (label_build/<window>_frames.json; V3's grid
  model, trained on Roboflow nbacourt frames only). But it sits a median 6.9 ft from the truth at the feet (D186's
  decomposition), far too coarse for a 2.5 ft question. The truth T was also initialised from it (sportvu/camera.py,
  pose_from_h of H_v3). So it is not a usable independent frame either.

### 11.6 Ruling on 11.5 and the asymmetric reading (2026-11-28, self-decided by the lead; before any code)

**Ruling: option (i).** The item runs as registered.
- Option (ii) waits on a court model no one plans to train.
- Option (iii) gives up a measure of about 3 lock minutes whose positive reading would still be informative.

**Why the reading is one-sided.**
- B learned court geometry from R1.3's H_truth. So an R1.3 timing error reaches B only second-hand, through R1.3's
  camera fit.
- tau_B pairs B's geometry with SportVU timed by the current multipool truth's time map, and the two truths' window
  offsets are known to differ (D99, D166).
- Errors that B and T share bias every reading toward "neither".

**Registered asymmetric reading:**
- A **timing** or **camera** finding is read as found, conservatively: a shared error could only have hidden it.
- **"Timing answered", "neither" and the build-side lower bound** are read as "not shown; B is not independent of
  SportVU on these games (r23 trained in sample on R1.3 labels)". They are never read as evidence that the row is
  build-side.

The limit is carried in every reading line of the result and in the bounds.

**The facts, with their sources:**
- ROADMAP_LOG.md pass 13: r23's training set is "every 2nd kept R1.3 label", with "labels written by generate_labels'
  own functions through H_truth" (sportvu/grid_dataset.py).
- tools.p8_frozen.R13_TAGS is ("gsw_bkn_2015", "gsw_cha_2016", "gsw_ind_2016", "gsw_sac_2015").
- tools.p8_scorecard flags r23's rows on those games "IN SAMPLE: the r23 court model was trained on this game's R1.3
  labels".

## 12. Version 1.3 (2026-11-28, after the code was written and before any run; self-decided by the lead)

Where this differs from sections 10 and 11, it rules.

- **The timing statistic is now the share of off windows.** It is the share of windows with |tau_B - tau_0| at least
  0.1 s, tau_0 being the across-window median. It is compared, per game, with the null's distribution of that same
  share: the share in each simulation, against its own median, over 2,000 simulations of 11.1's null.
  - **Answered** needs the observed share at or below the null's 97.5th percentile on both games, with the power check
    passing.
  - Otherwise the enrichment test runs, unchanged (10.2 and 11.4: at least 5 off windows, the own P-ii rate ratio of
    at least 1.5 with its interval excluding 1, on both games).
- **The power check stays.** It still injects plus or minus 0.1 s into a quarter of windows, 2,000 simulations, seed
  1. It now uses this statistic: "answered" may be read only if at least 80 percent of the injected simulations exceed
  the null's 97.5th percentile of the share.
- **The across-window IQR stays as report only.** It was replaced because it is robust to a quarter of windows: the
  middle half of the windows sets it, so errors confined to a quarter of them barely move it. A test shows this. With
  zero noise, a quarter-injection leaves the IQR at 0 in most simulations, while the share of off windows catches it in
  every simulation.
- The readings of 11.6 are unchanged.

### 12.1 Amendment (2026-11-28, after the code check of 1752d07d, GO WITH CHANGES; before the run)

Where this differs from sections 10 to 12, it rules.

1. **tau_B's estimation rows leave out the players who are P-ii on that frame** (D205's class on the frame). The rule
   reads this version. The version with them is reported beside it (tau, tau_0 and the off share).
2. **"Timing found" reads:** "a window-level along-velocity error; the truth's time map is the likelier side; build
   lag not excluded". The off share of the TRACK-only tau and of the lateral-axis estimator are reported beside it.
3. **The EMA prediction is dropped.** Per row, tau(foot_stab) - tau(foot_raw) is B's pixel offset stab - raw along the
   velocity, by construction, so the "prediction" equals the difference. The difference stays, described as a units
   check only.
4. **11.2's roles are restated.**
   - The plus and minus 3 moment shift confirms the sign and the units of tau. Resolvability comes from the power
     check.
   - A shifted row is accepted only when the game clock moved 0.12 s across those 3 moments, within 0.02 s. Rows across
     a moment gap or a stopped clock are left out and counted.
5. **11.3 (b)'s wording is corrected.** The null B offset of an own P-ii row is its own T offset **minus** a same-frame
   hit's T-minus-B disagreement vector (11.3 said "plus"; B(foot) - SV = (T(foot) - SV) - (T(foot) - B(foot))). The
   null's hit is drawn from the same third when the frame has one, else from any hit on the frame.
6. **10.1, added.** The in-frame test and the third both use T: the in-frame test is T^-1(SportVU) inside the 10 px
   margin, and the third is the scorer's under T. Neither tracks a time error, since both place SportVU's own position,
   so neither biases tau_B toward or away from one.

## 13. Result (2026-11-28, from dd3a5e81; record reports/research/p8_piiside.json)

**Run.** The fast suite ran first under the lock: 2,669 passed, 7 skipped, 65 deselected. Then one collect per game
under the lock: 65 s for gsw_cha_2016 and 61 s for gsw_bkn_2015, from the heavy-job lines of work_ref/logs/piiside.log.
The report took 31 s.

- D202's and D207's gates passed on every window, and the manifests were unchanged.
- Every scored frame had an accepted B.
- Velocity was missing on 3 and 1 rows.
- Shifted rows left out (the clock did not move 0.12 s, or a moment was missing): 233 and 433.
- hou_cle_2016 was not read and stays unspent.

**Readings, as registered (11.6): timing, camera and overall read "not shown",** each with the limit: "B is not
independent of SportVU on these games (r23 trained in sample on R1.3 labels written through H_truth); never read as
evidence that the row is build-side". The record's reading strings repeat "not shown" once, because the limit text
begins with it; the meaning is unchanged.

**Timing** (gsw_cha_2016 / gsw_bkn_2015; on rows without the frame's P-ii players, 12.1 (1)):

| quantity | gsw_cha_2016 | gsw_bkn_2015 |
|---|---|---|
| windows with a tau_B (at least 30 moving rows) | 36 of 73 | 41 of 83 |
| tau_0 | 0.118 s | 0.108 s |
| synthetic shift, +3 and -3 moments (expected -0.12 and +0.12 s) | -0.120, +0.121 s: ok | -0.121, +0.121 s: ok |
| share of off windows (at least 0.1 s from tau_0) | 0.583 | 0.707 |
| the null's 97.5th percentile of that share | 0.028 | 0.049 |
| power (quarter-injection above the null's 97.5th) | 0.980 | 0.949 |
| off windows | 21 | 29 |
| enrichment, own P-ii rate in off windows over the rest (bar 1.5, interval excluding 1) | 1.31 [1.0009, 1.75]: the lower end just above 1 | 1.13 [0.90, 1.42] |

- The window-level spread lies far beyond the null on both games, with power above 0.8, so timing is **not
  answered**.
- The enrichment test then fails its bar on both games, so timing is **not shown** (with the limit).
- Beside it:
  - the TRACK-only off share is 0.583 / 0.707, and the lateral estimator's is 0.600 / 0.484;
  - with the P-ii players included, the off share is 0.605 / 0.761, with tau_0 0.148 / 0.105 s;
  - the IQR (report only) is 0.231 / 0.283 s, against its null's 0.036 / 0.049 s;
  - tau_B against the window's offset_s: Spearman 0.32 / 0.42;
  - the EMA difference (a units check) is -0.047 / -0.044 s.

**Camera:**
- **(a), within third:** the ratio of own P-ii to same-frame hit d_TB is 1.41 [1.25, 1.69] and 1.35 [1.23, 1.46],
  against the 1.5 bar. It fails on both games.
- **(b):** own P-ii boxes within 2.5 ft under B are 0.463 and 0.428, against the build-side null's 97.5th percentile of
  0.261 and 0.258. It passes on both games. Hits' own d_B share, the T-side reference, is 0.820 and 0.810.
- The pattern needs (a) and (b), so camera is **not shown** (with the limit).
- By third, the median d_TB at own P-ii boxes against hits is:
  - near: 1.22 against 0.87 ft, and 1.31 against 0.97 ft;
  - middle: 1.20 against 0.93, and 1.21 against 0.92;
  - far: 2.10 against 1.05, and 1.98 against 1.44.

**Bounds, each a bound and not a forecast:**

| | gsw_cha_2016 | gsw_bkn_2015 |
|---|---|---|
| own P-ii rate | 0.064 | 0.059 |
| timing, upper bound | 0.008 | 0.005 |
| camera, upper bound | 0.030 | 0.025 |
| neither, lower bound on the build-side share | 0.027 | 0.029 |

The lower bound is read with the limit: not shown, B is not independent of SportVU on these games, and it is never
evidence that the row is build-side.

**Post hoc, not ruled. The near-misses are not passes, and nothing is re-ruled.**
- (i) Most of the windows with a tau_B (36 of 73 and 41 of 83) carry one more than 0.1 s from the game's median. Either the truth's per-window time map or B
  (in sample, through the tracker's Kalman state, and with camera pan) varies by window at that scale. The truth's
  offset solve is known to be fragile: D99 found offsets moving 0.3 to 0.8 s with small build differences, and D166
  found the median offset difference between builds fell only from 0.52 to 0.367 s under truth v2. But P-ii is only
  weakly enriched in those windows (1.31 and 1.13).
- (ii) The camera upper bound is 0.030 and 0.025. Above the build-side null's 97.5th percentile, the excess is about
  0.013 and 0.010 of in-frame players. Neither is a plausible size: B's in-sample R1.3 lineage could hide a camera error
  or create a disagreement. Not shown.
