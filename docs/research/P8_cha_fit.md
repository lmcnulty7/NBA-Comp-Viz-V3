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

## 7. Stage 1 results (computed 2026-11-27, after this design was committed in 9e7b34b1)

Stored files only, about 2 seconds; record reports/research/p8_cha_fit.json.

S1 and S5, every recorded fit (C and C0 in ft, court frame; reprojection after refinement):

| Fit | C | C0 | Distance C to C0 | Reprojection | Inits | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| gsw_bkn_2015 MPS | (46.9, 125.2, -29.5) | (41.5, 108.1, -35.1) | 18.8 | 6.52 | 3,234 | 2,075 |
| gsw_bkn_2015 CPU, fix | (48.1, 122.8, -29.7) | (46.8, 105.1, -34.8) | 18.5 | 7.41 | 4,728 | 2,903 |
| gsw_cha_2016 MPS | (42.6, 118.7, -31.1) | (40.0, 103.9, -35.0) | 15.5 | 7.65 | 2,586 | 1,366 |
| gsw_cha_2016 CPU, no fix | (46.9, 127.8, -31.4) | (47.2, 103.6, -34.9) | 24.5 | 7.06 | 4,123 | 2,459 |
| gsw_cha_2016 CPU, fix | (47.1, 104.3, -34.6) | (47.2, 103.5, -34.9) | 0.9 | 15.18 | 4,943 | 971 |
| gsw_sac_2015 MPS | (47.7, 134.1, -30.1) | (36.8, 106.8, -34.7) | 29.8 | 5.47 | 3,277 | 2,151 |
| gsw_sac_2015 CPU, fix | (45.4, 126.5, -29.8) | (41.2, 106.5, -34.5) | 21.0 | 6.65 | 4,721 | 2,949 |
| hou_cle_2016 MPS | (43.5, 131.6, -38.4) | (41.6, 127.3, -43.7) | 7.1 | 6.40 | 6,863 | 4,427 |
| hou_cle_2016 CPU, fix | (44.7, 135.5, -37.5) | (44.6, 126.8, -43.7) | 10.7 | 5.54 | 10,804 | 7,447 |
| gsw_ind_2016 MPS (apart, D98) | (46.6, 106.0, -34.7) | (46.7, 106.0, -35.3) | 0.6 | 18.35 | 3,900 | 970 |

(The C to C0 distances are computed from the rounded values shown.) Among the GSW home fits, the fixed gsw_cha_2016 C
is 22.0 ft from the median C of the other GSW fits; the others are 1.5 to 9.0 ft from theirs. The two broken fits on
record, gsw_cha_2016 CPU with the fix and gsw_ind_2016 MPS, are the only ones whose C stayed within about 1 ft of its
start C0; every good fit moved 15 to 30 ft from C0 (7 to 11 ft at hou_cle_2016, whose C0 already sits near its C).
S5: between gsw_cha_2016's two CPU fits, C0 moved 0.09 ft and C moved 23.65 ft.

S2, court identity: of 18,083 frames in both label builds, 17,595 have an identical court H, 488 have no H in either
build, and none differs. The fix did not touch the court side.

S3, boxes: before each window's first reset the two builds have identical boxes (15,016 frames, median 9 boxes, 13,142
frames with at least 6). After a first reset (3,067 frames in the 12 windows with one) the unfixed build has a median
of 3 boxes and 527 frames with at least 6; the fixed build a median of 9 and 2,792.

S4, accepted rows: before the first reset 2,416 (no fix) against 856 (fix); after it 43 against 115. Feet residual
medians 1.20 and 1.25 ft (no fix) against 1.47 and 1.56 ft (fix); line matches medians 53 and 40 against 37 and 37. The
largest losses are in windows with no reset at all (gsw_cha_2016_s07_w00 265 to 115, s08_w00 205 to 76, s03_w01 114
to 33); the gains are in reset windows (s05_w00 0 to 57, s01_w00 4 to 15, s06_w00 0 to 12).

Reading against section 5, stage 1 only:

- H4 (court side changed) is ruled out: every court H is identical, and C0 is unchanged.
- The loss is not confined to the frames the fix newly exposes: accepted frames fall most in windows without any
  reset, whose inputs are identical in both builds, and their residuals rise. That is the signature of a wrong shared
  camera position applied to every frame, not of bad post-reset frames being rejected.
- The fixed fit ended where it started (C within 0.9 ft of C0), as did gsw_ind_2016's broken MPS fit (0.6 ft); every
  good fit moved 7 to 30 ft. This points to the pooled bundle adjustment not leaving its start on these pools (H1 or
  H2), with the new post-reset inits (S3) changing the pool through the seed-0 draw. Stage 1 cannot tell H1 from H2, or
  whether the new frames' pairs (H3, H5) are what held the fit at its start; stage 2's F1 to F4 can.
- Nothing is retuned; gsw_cha_2016's section 5 verdict (A and B fail, unused) stands.
