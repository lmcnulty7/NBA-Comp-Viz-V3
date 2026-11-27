# P8: a multi-start camera fit for camera truth (design, version 1, before any compute)

Written 2026-11-27 for PLAN.md D197's next item. Design only: nothing here has run and there is no number in it that
was computed for it; every figure quoted comes from records already committed (P8_cha_fit.md, P8_truth_arenas.md
sections 15 to 17). No held-out game, no D85 test game and no test arena is read (CLAUDE.md rule 4). The frozen code
(sportvu/camera.py, sportvu/truth.py, sportvu/autolabel.py) is not edited; everything new goes in tools/ and calls it.

## 1. Question

The frozen game-level camera fit (sportvu.camera.fit_game: one seeded pool of up to 400 frames, a pooled bundle
adjustment from C0, capped at 200 evaluations) is fragile: on gsw_cha_2016 it stayed at its start for 3 of 10 seeds on
the reset-fixed build and 4 of 10 on the unfixed one (P8_cha_fit.md section 9, F1), every recorded run stopped at the
200-evaluation cap without reaching scipy's convergence test (F0, F2), and the two broken fits on record
(gsw_cha_2016 fixed, 15.18 px; gsw_ind_2016 MPS, 18.35 px, D98) both ended within about 1 ft of their start (stage 1).
A game can lose its truth to the draw of one pool.

Question: does running the fit from several pools, continuing each run until it converges, and keeping one run by a
rule fixed now that reads only the fit's own objective and convergence, give camera truths that are stable when the
whole procedure is repeated with other pools? And, secondarily, what does it do to the four reset-fixed games' truths
and to gsw_ind_2016's (report only for the latter)?

## 2. The procedure (fixed now)

Inputs per game, built exactly as tools.p8_frozen.step_fit builds them (the same windows in the same order,
sportvu.truth.window_truth, sportvu.camera.frames_from_rows), from the game's label build named in section 5.

1. Inits and C0: as fit_game (every frame with at least 6 inlier pairs whose V3 court H decomposes into a camera; C0 the
   median of their positions). C0 does not depend on the seed; a seed only changes which 400 inits are pooled.
2. Runs: one run per seed s in a seed set (section 3). A run is the pooled least-squares problem of fit_game for pool s
   started from C0 and the per-frame poses of the pool's inits, advanced in chunks of 200 evaluations, each chunk
   continuing from the previous chunk's end (same residual, loss soft-L1, f_scale 6 px, x_scale "jac", sparsity).
   The first chunk is exactly fit_game's call: before anything else, for every game and seed, the first chunk's C must
   equal fit_game(frames, seed=s)'s C within 0.01 ft (calling fit_game itself), or the game stops (kill rule 1).
   fit_game cannot run past 200 evaluations or report its status, so the chunked continuation is a tools-side copy of
   its least-squares call (the copy already reproduced fit_game exactly on gsw_cha_2016's two builds, P8_cha_fit.md F0).
3. Convergence (section 4): a run converges when a chunk ends with scipy status 1 to 4 (a tolerance met), or when two
   consecutive chunk ends differ by less than 0.05 ft in C and by less than 0.1 percent in pooled cost. A run that
   reaches the cap without converging is "not converged" and cannot be selected.
4. Common objective. Pooled costs of different seeds are sums over different frames and cannot be compared. Each
   converged run's C is therefore scored on the same frames: for every init of the game, sportvu.camera.fit_frame with
   C fixed (the frozen per-frame refit of rotation and focal length to that frame's pairs, initialised from its V3 H),
   and the frame's soft-L1 cost at its solution with fit_frame's own loss and scale (6 px). The common objective of a
   run is the sum over all inits. This is the quantity the frozen fit minimises, on all inits instead of one pool.
5. Selection rule: among converged runs, the lowest common objective; a tie (equal to 1e-9 relative) goes to the lower
   seed. If no run converges, the game has no multi-start fit (kill rule 2 for the pilot; "no fit" for a game).
6. The selected C replaces fit_game's output in the rest of the frozen fit step: per-frame fit_frame, refine_lines,
   the acceptance rules and the label writing of tools.p8_frozen.step_fit, unchanged. The tool calls step_fit with
   sportvu.camera.fit_game replaced in the running process by a function that returns the selected run (same keys as
   fit_game, plus the selection record); no file is edited.

What the rule reads: the pooled objective and the common objective, which are the frozen fit's own objective (the
reprojection of matched SportVU positions onto the build's feet, soft-L1), and convergence. What it never reads: checks
A, B or C of P8_truth_arenas.md section 5 or any of their inputs (median reprojection after refinement, accepted
frames, mapped clock, offset stability), the per-frame acceptance rules, line matches or line support, the scorer, any
score, any truth-against-SportVU agreement beyond the fit's objective, and the earlier truths. The median reprojection
of each run is recorded but never compared. So the rule cannot be tuned toward passing section 5.

A known risk, stated now: on gsw_cha_2016's fixed seed-0 pool the run near C0 had the lower pooled cost after 200
evaluations but the worse median reprojection (P8_cha_fit.md F2, post hoc item (a)). If the common objective also
prefers that position, the rule selects it and gsw_cha_2016 stays unused; that is a result, not a reason to change the
rule.

## 3. Number of starts

Seeds per multi-start: 8. From F1, 7 of 20 runs (3 of 10 fixed, 4 of 10 unfixed) stayed at their start. If runs were
independent with that rate (0.35), all 8 staying stuck would have probability 0.35^8, about 0.0002, and fewer than 2
good runs about 0.004; with 6 seeds the first would be about 0.002. Runs on the same game share inits, so they are not
fully independent; 8 is chosen with that margin. Seed sets: {0..7} for the truth (seed 0 is the frozen fit's own pool,
so the frozen run is always one of the candidates), and {8..15} and {16..23} for the stability repeats (section 5).

## 4. Evaluation cap and convergence criterion

- Cap: 4,000 evaluations a run (20 chunks of 200). Records: every recorded seed-0 run stopped at 200 without scipy's
  test being met, and the report-only gsw_cha_2016 run from C0 still had not met it at 2,000 (its C moved 0.098 ft and
  its cost fell 3.5 percent between 200 and 2,000 evaluations, P8_cha_fit.md F2). Twice that is allowed.
- The chunk rule (C within 0.05 ft and cost within 0.1 percent between consecutive chunk ends) is set against that
  record: the stuck run's average change per chunk over 200 to 2,000 was about 0.011 ft and about 0.39 percent in cost,
  so the cost part would not have declared it converged, while a run that has settled (changes far below 1 ft of C,
  the scale on which the good and broken fits differ) stops early. 0.05 ft of C moves a truth foot by far less than the
  2.5 ft gate (a 10 ft change of C moved gsw_cha_2016's truth about 1.1 ft at foot pixels, section 15).
- A run still not converged at 4,000 is excluded from selection, recorded with its last C and cost.

## 5. Stability measure and bar

Stability is measured by repeating the whole selection (sections 2 to 4) with the two other seed sets, {8..15} and
{16..23}, on the same inputs, giving three selected positions per game. The repeats score their converged runs on the
same common objective, and each repeat's selected C gets its median per-frame reprojection over all inits (from the
common objective's fit_frame solutions, so no video is decoded; recorded, never used to select).

- Bar, per game: the three selected C lie within 2.0 ft of each other (largest pairwise distance), and their median
  per-frame reprojections lie within 0.5 px of each other.
- Why: across the six good fits on record at the GSW arena (three games, MPS and CPU), C differs by up to 16.3 ft
  between games and platforms, and the two broken fits sit 13.8 to 30.1 ft from the good ones (P8_cha_fit.md stage 1,
  S1, computed from its rounded table). 2.0 ft is far inside both, so a repeat that falls into the other basin cannot
  pass, and 2.0 ft of C moves truth feet by roughly 0.2 ft (by the 10 ft to 1.1 ft record above). 0.5 px is half the 1.0 px reprojection bar of section 0's "moved" test.
- A game meets the stability bar or it does not; a game that does not gets no multi-start truth (section 7).

## 6. Games

Development games only, all reset-fixed reference-platform builds in work_ref (D85 (3); the guard and refuse_cv_test
run on every game and window):

- gsw_cha_2016, hou_cle_2016, gsw_bkn_2015, gsw_sac_2015: their reset-fixed CPU label builds (work_ref/label_build,
  read only).
- gsw_ind_2016: apart under D98. Its only label build is the MPS build in the P8 workspace (work/label_build, read
  only, unfixed, MPS detector). It runs the same procedure as a report only: its multi-start result is never a truth and
  never re-accepted here; whether it leaves D98's "apart" would be a separate decision on a rebuilt reference-platform
  truth.
- No held-out game (sportvu/splits.json heldout) and no D85 test game, video or arena.

## 7. What it changes downstream

- A new truth version, "multistart v1", for each of the four fixed games that meets the stability bar: the frozen fit
  step with the selected C (section 2 step 6) on the same label builds, written to a new P8 root, work_ms, which holds
  read-only links to work_ref's label builds, clips, time maps, sync maps, moments and mirror records and a copy of
  work_ref's check C composite builds/r23 (and its r23_checkc records), so nothing in work_ref is overwritten and the
  reset-fix arm's pinned truths stay byte for byte. Records under reports/research/p8_multistart_<game>.json.
- Re-acceptance: tools/p8_truth_accept.py in work_ms with section 5's bars unchanged. Check C does not depend on the
  camera fit (it compares the label build's and the r23 composite's frozen offsets), so it is rerun but expected
  unchanged; A and B are new. Also reported against each game's current fixed truth with tools/p8_truth_moved.py and
  section 0's bars (report only).
- Which version a game uses is the lead's decision after the results. Rule proposed now: a game may switch to
  multistart v1 only if it meets the stability bar and reads "use" under section 5; all games of one comparison use the
  same truth version (no mixing within a comparison).
- Arms and results that used earlier truths (E1R, placement steps 1 and 1b, footbias, footpos, outpos, detrow,
  buildloss, trackreset, a3align, and the reset-fix arm's pinned scores): their numbers stand as recorded and nothing
  is rescored by this item. A rescore on multistart v1 needs its own registration per arm; the reset-fix arm keeps its
  pinned truths unless its own design is amended.

## 8. Ladder, cost and kill rules

Cost, from P8_cha_fit.md stage 2 (419 s of lock for: rebuilding two builds' inputs, 25 runs of 200 evaluations, one
of 2,000 and two subset fits), with the split between input rebuilding and fitting not recorded: a 200-evaluation
chunk takes about 7 to 10 s on gsw_cha_2016 (if input rebuilding took 45 to 90 s a build), and rebuilding one build's
inputs 45 to 90 s. Per game, worst case (every run to the 4,000 cap): inputs about 90 s, three seed sets of 8 runs of
20 chunks (480 chunks, about 3,400 to 4,800 s), the common objective for up to 24 runs (fit_frame on every init, about
3 to 6 minutes a run on 4,000 to 11,000 inits, not yet measured), and the truth step once (the frozen fit step's
per-frame part, 178 s on gsw_cha_2016 in the truth phase). Expected well under the worst case if runs converge early.

| Stage | What | Lock hours | Gate |
| --- | --- | --- | --- |
| 0 | tools/p8_multistart.py (inputs, chunked runs, convergence, common objective, selection, stability, truth step in work_ms) with tests on synthetic frames; design review | 0 | tests pass; find_private clean; guards refuse spent and test games |
| 1 pilot | gsw_cha_2016 (the failed fit) and gsw_bkn_2015 (a good one), all three seed sets, truth step for each | about 1.0 to 3.5 (measured) | kill rules 1 to 4 |
| 2 | hou_cle_2016, gsw_sac_2015, then gsw_ind_2016 (report only) | about 1.5 to 5.0 (re-estimated from stage 1's timing before it starts) | kill rules 1 and 4 per game |
| 3 | re-acceptance in work_ms, moved reports, records | under 0.3 | none (results) |

0 Colab units. One heavy job at a time, each step under its own lock name ("p8 multistart <game> <step>"), resumable.

Kill rules, fixed now:

1. Reproduction: for any game, a seed's first chunk does not reproduce fit_game(frames, seed) within 0.01 ft: that game
   stops and the copy is diagnosed before anything else runs.
2. Pilot convergence: on either pilot game, fewer than 2 of the 8 runs of the truth seed set converge within the cap:
   the procedure stops before stage 2 (the cap or the convergence rule is wrong for this problem; a new version of this
   design would be needed, not a silent change).
3. Pilot stability on the good fit: gsw_bkn_2015 fails the stability bar: the procedure itself is unstable on a game
   whose frozen fit is good; stop before stage 2.
4. Budget: a stage's lock time passes 1.5 times its estimate: stop and re-estimate before going on.

Not a kill rule: gsw_cha_2016 still failing section 5 with a stable selection. That is the answer to the question for
that game and is reported as such. No seed set, cap, tolerance, start or objective is changed after any pilot number
is seen; a change is a new dated version of this design, reviewed before it runs.

## 9. What is not done

No compute, no numbers, no truth written, nothing rescored. The frozen fit_game and step_fit are not edited and their
recorded outputs stay as recorded. No D line is written here.
