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
