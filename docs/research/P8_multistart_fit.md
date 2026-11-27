# P8: a multi-pool camera fit for camera truth (design, version 2, before any number)

Version 2, 2026-11-27, after the independent design review of version 1 (commit 8a5f8981; verdict GO WITH CHANGES,
three blockers). Version 2 replaces version 1 entirely; section 11 maps each review finding to its change. Design only:
nothing has run for this item and every figure quoted comes from committed records (P8_cha_fit.md and its records,
P8_truth_arenas.md sections 15 to 17). No held-out game, no D85 test game and no test arena is read (CLAUDE.md rule 4).
The frozen code (sportvu/camera.py, sportvu/truth.py, sportvu/autolabel.py, tools.p8_frozen.step_fit) is not edited;
everything new goes in tools/ and calls it.

## 1. Question and upper bound

The frozen game-level camera fit (sportvu.camera.fit_game: one seeded pool of up to 400 inits, one pooled bundle
adjustment from C0, capped at 200 evaluations) is fragile. On gsw_cha_2016's reset-fixed build, 6 of 10 other seeds
gave a run that was not good (P8_cha_fit.md section 9, F1: 3 stayed within 4 ft of C0, 3 left it but ended above 10
px); every recorded seed-0 run stopped at the 200-evaluation cap (F0, F2); both broken fits on record (gsw_cha_2016
fixed, gsw_ind_2016 MPS, D98) ended within about 1 ft of their start (stage 1).

Question: does a multi-pool fit (several pools, all from the one start C0, each run beyond 200 evaluations, one run kept
by a rule fixed now that reads only the fit's own objective) give camera truths that are stable when the whole
procedure is repeated with other pools?

Upper bound, stated now: one game. Check C does not depend on the camera fit (it compares the frozen offsets of the V3
label build and the r23 composite), so gsw_sac_2015 stays "apart" whatever this does; gsw_bkn_2015 already reads
"use" and can only keep or lose it; hou_cle_2016's check A already passes and its status waits on check C. Only
gsw_cha_2016 can change status, from "unused" to "use". The design is therefore built around one cheap falsifier on
gsw_cha_2016 first (stage 0), and the other games run only if gsw_cha_2016's result can change (section 8).

## 2. Definitions (fixed now)

- Inputs per game: built exactly as tools.p8_frozen.step_fit builds them (the same windows in the same order,
  sportvu.truth.window_truth, sportvu.camera.frames_from_rows) from the label build named in section 7. Inits and C0
  exactly as fit_game (frames with at least 6 inlier pairs whose V3 court H decomposes into a camera; C0 the median of
  their positions). C0 does not depend on the seed; a seed changes only which 400 inits form the pool.
- Per-frame cost at a position C (finding 6). For init k with pairs (px_k, ft_k), fit_frame(C, frame k) (frozen: C
  fixed, rotation and log focal length refit, initialised from the frame's V3 court H, soft-L1 with f_scale 6 px, focal
  length bounded to F_BOUNDS_PX, at most 200 evaluations) returns (rvec_k, f_k). With r = project(C, rvec_k, f_k, w, h,
  ft_k) - px_k (all 2n components, px), cost_k(C) = 36 * sum_j (sqrt(1 + r_j^2 / 36) - 1), which is scipy's soft-L1
  cost with f_scale 6 at that solution.
- Common objective: J(C) = sum of cost_k(C) over every init of the game. It is the frozen fit's own objective
  (reprojection of matched SportVU positions onto the build's feet, soft-L1), evaluated on all inits instead of one
  pool, so runs of different pools can be compared.
- Pooled median (finding 2): for a pooled run, the median of the absolute residual components at its final parameter
  vector, computed as fit_game computes reproj_px_median_after (np.median(np.abs(sol.fun))). This is the number check A
  reads from a truth's result file. The all-inits median of |r| from J's per-frame solutions is reported beside it and
  is not ruled.

## 3. Stage 0: the falsifier (first, before any multi-pool run)

Can the selection rule prefer a good position over the broken one on gsw_cha_2016 at all? If J does not, no multi-pool
fit under this rule can rescue gsw_cha_2016, and with the one-game upper bound nothing else is worth running.

- Under the heavy-job lock ("p8 multistart stage0 gsw_cha_2016"), rebuild gsw_cha_2016's fixed inputs; check that they
  give 4,943 inits and that fit_game(frames, seed=0) reproduces the recorded fixed C within 0.01 ft (otherwise stop).
- Evaluate J at five recorded positions (ft, court frame; from reports/research/p8_cha_fit_stage2.json):
  - B, the broken fixed fit: (47.0815, 104.3231, -34.5585);
  - G1, C_old, the unfixed CPU fit: (46.9099, 127.7549, -31.3734);
  - G2, fixed seed 1: (46.4770, 130.3553, -29.9345); G3, fixed seed 7: (47.3303, 130.5270, -33.9374);
    G4, fixed seed 10: (47.8121, 137.3554, -30.9693).
- Reading rule, fixed now; it reads J only (never checks A, B or C, accepted frames, line support, the scorer or any
  score): with J_B = J(B) and J_G = the lowest of J(G1) to J(G4):
  - J_B <= J_G: STOP. The rule would select the broken basin over every good position on record; no multi-pool run
    under this rule can rescue gsw_cha_2016. Report and end this item.
  - J_G < J_B but by less than 1 percent of J_B: STOP as inconclusive; report and leave the next step to the lead.
  - J_G <= 0.99 J_B: continue to stage 1.
- Per-frame optimality check (finding 6), on the seed-0 pool at B: for each of its 400 frames, cost at fit_frame's
  solution against the cost at the pooled solution's own (rvec, f) for that frame (from the tools-side copy's final
  parameter vector at 200 evaluations, which reproduces fit_game). If more than 5 percent of the frames have a
  fit_frame cost more than 1 percent above the pooled pose's cost, J's per-frame minimisation is not reliable: STOP and
  report, whatever the J reading.
- Chunked-continuation smoke (finding 5), report only: on the seed-0 pool from C0, ten chunks of 200 evaluations,
  each continuing from the previous chunk's end, against the recorded single 2,000-evaluation run (P8_cha_fit.md F2: C
  (47.0457, 104.4133, -34.5452), cost 1,062,115); the C distance and cost difference are reported.
- Timing: inputs, fit_game, each J evaluation, each chunk, in seconds. Estimate 900 s; cap 1,800 s.
- Record: reports/research/p8_multistart_stage0.json; outputs under work_ref/diag/multistart/ only.

## 4. Stages 1 and 2: the multi-pool fit

1. Runs: one run per seed in a seed set; each is fit_game's pooled problem for that pool from C0, advanced in chunks of
   200 evaluations, each continuing from the previous chunk's end (same residual, soft-L1, f_scale 6 px, x_scale
   "jac", sparsity), to 4,000 evaluations. The first chunk of every run must reproduce fit_game(frames, seed) within
   0.01 ft, checked by calling fit_game itself (kill rule 1). The continuation is a tools-side copy of fit_game's
   least-squares call because fit_game cannot run past 200 evaluations or report its status; the copy reproduced
   fit_game exactly on both gsw_cha_2016 builds (P8_cha_fit.md F0).
2. Convergence is recorded and reported, not gated (finding 1): a run counts as converged when a chunk ends with scipy
   status 1 to 4, or two consecutive chunk ends differ by less than 0.05 ft in C and less than 0.1 percent in pooled
   cost; otherwise "not converged at 4,000". Every run competes in the selection whatever its convergence.
3. Selection: the run with the lowest J at its final C; a tie (relative difference under 1e-9) goes to the lower seed.
   Seed 0's run starts as the frozen fit (its first chunk is the frozen call) and continues; the frozen fit's own C is
   not itself a candidate unless seed 0's run ends there.
4. The selected run is written as a truth by calling tools.p8_frozen.step_fit unchanged, with sportvu.camera.fit_game
   replaced in the running process by a function that returns the selected run's C, C0, pooled frames, inits and pooled
   median (section 2), with the selection record (finding 9 lists its asserts).

What the rule reads: J and, for the record only, convergence. It never reads checks A, B or C or their inputs (the
pooled median, accepted frames, mapped clock, offset stability), the per-frame acceptance rules, line matches or line
support, the scorer, any score, or the earlier truths. The pooled median of each run is recorded and never compared.

## 5. Seeds and why (finding 4)

- This is a multi-pool fit from one start: every run starts at C0, and the seed changes only the pool.
- Seed sets: truth set {0..7}; stability sets {16..23} and {24..31}. Disclosed: F1 already ran seeds 1 to 10 on
  gsw_cha_2016's fixed build, so the truth set contains seeds known to give good runs there (1, 4, 7) and seeds known to
  give runs that were not good (2, 3, 5, 6); {0..7} is kept because it is the consecutive default that includes the
  frozen pool, not because of those outcomes. The stability sets avoid every seen seed.
- From F1's not-good rate on the fixed build (6 of 10), 8 independent pools would all be not good with probability
  0.6^8, about 0.017, and fewer than 2 good with probability about 0.106; across three sets, at least one set with no
  good run about 0.05. Pools of one game share inits, so they are not fully independent. These figures say how often
  the procedure may still miss on gsw_cha_2016; they are not a guarantee.

## 6. Stability (finding 3)

Repeat sections 4.1 to 4.3 with each stability set on the same inputs, giving three selected positions per game.

- Measure in truth space: for each pair of selections and every init (the same frames for all three), the frame's
  pair foot pixels mapped to court feet through each selection's per-frame H (from fit_frame at that selection's C,
  sportvu.camera.h_px2ft); the distance between the two mappings of each foot; the median over all feet of all inits.
- Bar: at most 0.25 ft for every pair of selections, one tenth of the 2.5 ft gate; the platform change of section 15
  moved gsw_cha_2016's truth a median 1.10 ft at the same pixels.
- Reported beside it, not ruled: the pairwise C distances and pooled medians.
- A stable result is not evidence of a correct fit: three selections in the same basin near C0 (where both broken fits
  sat) would be stable. Correctness is judged only by section 5 of P8_truth_arenas.md on the written truth.

## 7. Games

- gsw_cha_2016: stages 0 and 1 (multi-pool, stability, truth step).
- gsw_bkn_2015: stage 1 (multi-pool, stability, truth step): the pilot's check that the procedure is stable on a game
  whose frozen fit is good, and the truth needed if gsw_cha_2016 switches (section 9).
- hou_cle_2016: stage 2 (multi-pool, stability, truth step; its acceptance stays pending until the arm's hou
  equivalence exists).
- gsw_sac_2015: stage 2, multi-pool and stability only, report; no truth step (it stays apart whatever the fit, by
  check C).
- gsw_ind_2016: stage 2, report only, on its MPS label build in the P8 workspace (work/label_build, read only, unfixed,
  MPS); never a truth; it stays apart under D98.
- All are development games; the guard and refuse_cv_test run on every game and window. No held-out game (sportvu/
  splits.json) and no D85 test game, video or arena.

## 8. Ladder, estimates, caps and kill rules (findings 1, 7, 10)

| Stage | What | Estimate | Cap |
| --- | --- | --- | --- |
| 0 | the falsifier, optimality check and chunk smoke on gsw_cha_2016 (section 3) | 900 s | 1,800 s |
| 1 | gsw_cha_2016 and gsw_bkn_2015: three seed sets each, stability, truth step in work_ms | from stage 0's timing | 1.5 times its estimate |
| 2 | hou_cle_2016 (with truth step), gsw_sac_2015 and gsw_ind_2016 (report) | from stage 0's timing | 1.5 times its estimate |
| 3 | acceptance in work_ms, moved reports, records | 600 s | 900 s |

Stages 1 and 2 get a single estimate and cap in seconds each, computed from stage 0's measured timing (per chunk, per
J evaluation, per input rebuild; truth step 178 s from the truth phase) as 24 runs times 20 chunks plus 24 J
evaluations per game, and written as a dated amendment here after stage 0 and before stage 1 runs. 0 Colab units. One
heavy job at a time, each under its own lock name ("p8 multistart <stage> <game> <step>"), resumable.

Kill and stop rules, fixed now:

1. Reproduction: any first chunk that does not reproduce fit_game(frames, seed) within 0.01 ft stops that game.
2. Stage 0's reading (section 3): STOP or inconclusive ends the item before stage 1.
3. Pilot stability on the good fit: gsw_bkn_2015 fails the stability bar: stop before stage 2.
4. Upper bound (finding 7): stage 2 runs only if gsw_cha_2016's selected C is more than 5 ft (the "leaves its start"
   distance of P8_cha_fit.md section 8) from the frozen fixed C. Otherwise no game's status can change, and the item
   ends with the stage 1 report.
5. Budget: a stage's lock time passes its cap: stop and re-estimate.
6. Report only (was version 1's kill rule 2): how many runs per set converge within 4,000 evaluations.

Not a kill rule: gsw_cha_2016 stable but still failing section 5. That answers the question for that game. No seed
set, cap, tolerance, objective, bar or start changes after any number of this item is seen; a change is a new dated
version reviewed before it runs.

## 9. Downstream (finding 8)

- A new truth version, "multipool v1", written in a new P8 root, work_ms (finding 9), for the games with a truth step.
- Acceptance: tools/p8_truth_accept.py in work_ms with section 5's bars unchanged. Check C needs builds/r23 there: a
  copy of work_ref's composite and its r23_checkc records. Each game is also compared with its current fixed truth by
  tools/p8_truth_moved.py and section 0's bars (report only).
- Switch rule: a game switches to multipool v1 if and only if it meets the stability bar and reads "use". All games are
  decided at once, after stage 3 and before any arm is scored on multipool v1. Fallback version per game if it does
  not switch, named now: gsw_cha_2016 its MPS truth (D191, in use); gsw_bkn_2015 its reset-fixed CPU truth ("use",
  D197); hou_cle_2016 its reset-fixed CPU truth (acceptance pending); gsw_sac_2015 its reset-fixed CPU truth, apart.
  The lead confirms the decisions; all games of one comparison use one truth version.
- Earlier results (E1R, placement steps 1 and 1b, footbias, footpos, outpos, detrow, buildloss, trackreset, a3align,
  the reset-fix arm's pinned scores) stand as recorded; nothing is rescored here. A rescore on multipool v1 needs its
  own registration per arm; the reset-fix arm keeps its pinned truths unless its own design is amended.

## 10. Guards (finding 9)

- work_ms: its real path must differ from work's and work_ref's, and lie outside DATA_ROOT, the repository and the
  main checkout (tools.p8_common.check_outside). It holds read-only links to work_ref's label builds, clips, time maps,
  sync maps, moments and mirror records, copies (not links) of work_ref's label done markers and of builds/r23 with its
  r23_checkc records, and writes only labels/, truth_v2/dev/, truth_accept/, truth_moved/, meta/done/ (fit markers)
  and multistart/. Stage 0 writes only under work_ref/diag/multistart/.
- Hashing: before and after every stage, a SHA-256 manifest of work_ref's labels/, builds/r23/, resetfix/truth_pin__*
  and label_build sidecars; any difference stops the run. Every labels file written in work_ms is hashed into the stage
  record.
- In-replacement asserts: at entry sportvu.camera.fit_game is the original function; the replacement is called exactly
  once per step_fit, with frames whose (window, frame) list hashes to the list the selection used; it returns the
  selected C exactly; the original is restored in a finally block and checked after.

## 11. Changes from version 1 (review findings)

1. Convergence no longer gates selection; every run competes; version 1's kill rule 2 is report only (section 8.6).
2. The written truth's reproj_px_median_after is pinned as the selected run's pooled median computed as fit_game does;
   the all-inits median is reported, not ruled; version 1's wording about "median reprojection after refinement" is
   corrected (sections 2 and 4).
3. Stability is stated in truth space (median foot distance between selections' per-frame mappings, 0.25 ft) with the
   caveat that stability near C0 is not correctness (section 6).
4. Uses F1's not-good rate (6 of 10) with corrected arithmetic; "multi-pool from one start"; seen seeds disclosed;
   stability sets moved to unseen seeds; the frozen-run wording fixed (sections 4 and 5).
5. A chunked-continuation smoke, report only (section 3).
6. The per-frame cost formula is pinned and a per-frame optimality check added (sections 2 and 3).
7. The one-game upper bound is stated; stage 2 runs only if gsw_cha_2016's selected C moves more than 5 ft; no truth
   step for gsw_sac_2015 (sections 1, 7 and 8).
8. Switch if and only if stable and "use"; all games decided at once before any arm scores on v1; fallbacks named
   (section 9).
9. work_ms guards, hashing and in-replacement asserts (section 10).
10. A single estimate and cap in seconds per stage; stages 1 and 2 fixed from stage 0's timing before they run
    (section 8).
11. Registration: the constants here (8 seeds a set, the seed sets, 200-evaluation chunks, the 4,000 cap, the
    convergence tolerances, the 1 percent and 5 percent stage 0 rules, the 0.25 ft and 5 ft distances) are a new
    registration made before any number of this item. D98 forbids changing a bar, constant or fit setting of an existing
    registration after its numbers are seen; nothing registered is changed here (fit_game, step_fit, section 5's bars
    and every recorded truth stay as they are), and the new procedure's constants are fixed before its own numbers. The
    gsw_cha_2016 numbers already seen (P8_cha_fit.md F1 and F2) are disclosed above and are not used to pick a seed,
    start or subset.

## 12. What is not done

No compute, no numbers, no truth written, nothing rescored, no D line. The frozen fit_game and step_fit and their
recorded outputs stay as they are.
