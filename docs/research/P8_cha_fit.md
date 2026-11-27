# P8: why gsw_cha_2016's camera fit broke with the reset-fixed label builds (design, before any number)

Written 2026-11-27, after the truth phase of P8_truth_arenas.md section 16 version 4.3, at the lead's request. Report
only: nothing is retuned, no truth changes, no bar changes, and section 5's verdict for gsw_cha_2016 (A and B fail, so
the truth is "unused" and the MPS truth stays in use) stands whatever this finds. gsw_cha_2016 is a development game
(D85 (3)); no held-out game and no D85 test game is read. This document was committed before any number below the
"Already known" list was computed.

## 1. Question

With the reset fix (PLAN.md D192), the label builds of gsw_cha_2016 kept more tracked boxes after resets, and the
frozen game-level camera fit (sportvu/camera.py fit_game, run by tools/p8_frozen.py step_fit) moved: camera position C
from (46.9, 127.8, -31.4) to (47.1, 104.3, -34.6) ft, median reprojection after refinement 7.06 to 15.18 px, accepted
frames 2,459 to 971, with line_matches_low 985 to 2,317 and feet_resid_high 467 to 1,320. Which change in the inputs
moved the fit, and is the new C a different minimum of the same problem or the right answer to a changed problem?

Already known (from the two fits' result files, read for the truth report): inits 4,123 (unfixed) and 4,943 (fixed);
pooled frames 400 in both; reprojection before refinement 106.98 and 109.15 px; per-window frozen offsets identical
(median absolute difference 0.00 s over 73 windows). hou_cle_2016, gsw_bkn_2015 and gsw_sac_2015 fitted well with the
fix (A 5.54, 7.41 and 6.65 px).

## 2. How the fit works (the parts that matter here)

- Inits: every frame with at least 6 inlier foot-to-SportVU pairs (sportvu.truth.window_truth, frames_from_rows) and a
  camera decomposition of its V3 court H (pose_from_h). C0 is the median of the inits' decomposed positions.
- Pool: 400 inits drawn by numpy default_rng(0).choice over the init list, then a pooled bundle adjustment from C0
  (shared C, per-frame rotation and focal length; soft-L1 on foot reprojection; at most 200 evaluations).
- Because the pool is a random draw over the init list, any change in the number or order of inits changes which 400
  frames are pooled, not only the added frames.
- Per frame: C fixed, rotation and focal length refit to the feet, then to the painted-line ridges; the acceptance
  rules (line matches, zoom bound, feet residual, line support) then reject or keep the frame.

## 3. Candidate causes

- H1, pool resampling. The fixed build adds about 820 inits, so the seed-0 draw picks a different 400. The pooled fit is
  sensitive to which frames it pools; the new C is one draw's outcome.
- H2, a local minimum (as D98's gsw_ind_2016 MPS fit, 18.35 px). The pooled problem has more than one basin and this
  pool from this C0 fell into a worse one; the old C would score better on the same pool.
- H3, the new frames pull C. Frames that newly qualify (mostly after a window's first reset, where the unfixed tracker
  kept about 0.3 of the boxes) carry pairs whose geometry disagrees with the rest (for example foot pixels of
  newly confirmed tracks, identity swaps after a reset, or frames of another camera after a cut), and C moves to fit
  them.
- H4, the court side changed. The V3 court H of the same frame differs between the two label builds (the court model
  runs on MPS, whose repeatability is not guaranteed), so C0 and the inits' decompositions moved.
- H5, the pairs of the old frames changed. Frames that were inits before have different inlier pairs now (the fix
  changes the boxes after a reset in those frames too), not only new frames entering.

## 4. Measurements

Stage 1, stored files only (no fit, no model, no video decode, no lock; system python):

- S1 Arena consistency. C and C0 of every recorded camera fit of the GSW home games (gsw_bkn_2015, gsw_cha_2016,
  gsw_sac_2015 on MPS, CPU unfixed where it exists, CPU fixed; gsw_ind_2016's MPS fit for reference) and of
  hou_cle_2016. A broadcast main camera at one arena should sit in nearly the same place across games.
- S2 Court identity. For every label window and frame present in both gsw_cha_2016 label builds (superseded_nofix and
  the reference root's label_build): whether the sidecar's per-frame H is identical, and the distribution of the
  difference in decomposed camera position (pose_from_h, a closed-form decomposition, not a fit) where not.
- S3 Where the change sits. Per window and over the game, relative to each window's first reset (the reset logs of the
  fixed builds, and tools.p8_trackreset.reset_like on the unfixed sidecars): tracked boxes per frame, and frames with
  at least 6 boxes (an upper bound on frames that can become inits), before and after the first reset, in both builds.
- S4 What the truth kept. For the accepted rows of both truths (labels/gsw_cha_2016/*.jsonl): count, feet_resid_ft and
  line_matches, before and after the window's first reset, and the windows that lost or gained most accepted frames.
- S5 C0 against C. Both fits' C0 (stored in their result files) beside C: whether the start moved (points to H4 or H3
  through the init population) or only the end point (points to H1 or H2).

Stage 2, needs the lead's approval before it runs (it reruns frozen fitting code: sportvu.truth.window_truth and
sportvu.camera.fit_game on the stored label sidecars; no model, no video decode beyond reading clip sizes; under the
heavy-job lock, about 5 to 10 minutes, after the arm's builds):

- F1 Seeds: fit_game on the fixed inits with seeds 1 to 10 (the frozen fit uses seed 0). The spread of C and of
  reprojection tells H1.
- F2 Same pool, two starts: the fixed seed-0 pool fitted from the unfixed C and from the fixed C0; compare final cost
  and C. A lower cost near the old C on the same pool is H2.
- F3 Old inits only: fit_game on the fixed builds restricted to the frames that were inits in the unfixed build (seed
  0), and on the fixed inits minus frames after each window's first reset. C back near the old one says H3; still far
  says H5 (or H1 and H2).
- F4 Same frames, both builds: for frames that are inits in both builds, the change in inlier pair count and foot
  pixels; a large change on old frames is H5.

## 5. What each pattern would mean

- S1 shows the fixed cha C as the outlier among GSW fits, S5 shows C0 near the old start, and F1 or F2 land near the
  old C with equal or lower cost: the pooled fit is fragile (H1 or H2), and the bad truth is a fit failure, not a
  change in the data. A remedy (multi-start, a larger pool, or a fixed arena C) would be a new registration under D98;
  it is not proposed here.
- S2 shows the court H differing between the builds: H4; the court side is not repeatable on MPS for this game, and
  that would matter for every truth rebuild, not only the fix.
- S3 and S4 show the change concentrated after first resets and F3 returns the old C without those frames: H3; the
  frames the fix newly exposes carry pairs that mislead the fit. That would bear on the reset-fix arm too, since the
  same frames enter its scored universe.
- F4 shows large changes on frames that were inits before: H5.
- Several can hold together; the report states each measurement and which causes it supports or rules out.

## 6. Outputs and cost

A tool, tools/p8_cha_fit.py (stage 1 now, stage 2 only after approval), with tests on synthetic inputs; record
reports/research/p8_cha_fit.json; a dated result section here. Stage 1 runs in seconds to minutes on stored JSON. No
Colab units. The accept phase of section 16 is not affected and still waits for the lead's word.
