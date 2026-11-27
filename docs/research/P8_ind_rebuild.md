# P8: gsw_ind_2016's camera truth rebuilt on the reference platform (design, version 1, before any compute)

## 0. Deferred (2026-11-27, after the design review; the lead's decision)

**The prior the review found.** gsw_ind_2016 already fails check C on its MPS build: P8_truth_v2.md (court-model pairs
table, line 94) records the frozen rule's median absolute offset difference 0.60 s and share over 0.2 s 0.7978 on 89
windows, against section 5's bars of 0.54 s and 0.7945. Check C does not depend on the camera fit, so neither the reset
fix's frozen fit nor the multi-pool fit can move it except through the rebuilt label and r23 builds. The three games
with both an MPS record (P8_truth_v2.md) and a CPU reset-fix record (P8_truth_arenas.md section 17) moved as follows:

| Game | MPS: median, share | CPU with the fix: median, share | Change in median, share |
| --- | --- | --- | --- |
| gsw_bkn_2015 | 0.44 s, 0.7273 | 0.48 s, 0.6883 | +0.04 s, -0.0390 |
| gsw_cha_2016 | 0.54 s, 0.7812 | 0.50 s, 0.7188 | -0.04 s, -0.0624 |
| gsw_sac_2015 | 0.52 s, 0.7945 | 0.64 s, 0.8219 | +0.12 s, +0.0274 |

gsw_ind_2016 would need its median to fall by at least 0.06 s (0.60 to 0.54) and its share by at least 0.0033; no game
moved its median down by that much. The review put the chance of passing check C at about one in four at best, and in
section 4's order check C is seen only after steps 2 to 5, about 6.8 of the 7.5 lock hours.

**Decision (the lead's, self-decided; options weighed: run now with a futility gate, about 2.3 lock hours before it
could stop, against deferring): deferred.** e1b-design's check C diagnosis for gsw_sac_2015 (P8_sac_checkc) is about to
run on stored files and may show why check C fails on GSW games. If its reading points to a cause a rebuild would
change, this design is revived as a new dated version; otherwise it stays deferred. D98's "apart" stands.

**Changes a revived version must make (the review's other findings), before any run:**

1. Code changes, all written and reviewed before anything runs:
   - gsw_ind_2016 in tools/p8_truth_cpu.sh's game table with the revived caps;
   - gsw_ind_2016 in the multi-pool tool's allowed list (tools/p8_multistart.py stage 1 functions), touching nothing
     else;
   - step 4's checks (no gsw_ind_2016 file in builds/r23 before; every window reset_fix true, cpu, reference devices,
     no clip fix after; complete coverage) in code, not only in the doc;
   - the D98 guards: tools/p8_truth_fixprep.py, tools/p8_truth_v2.py, tools/p8_truth_moved.py and the compose step
     refuse gsw_ind_2016 today (D98, "apart"); a revived version lifts each refusal only for this registration, by a
     named, tested exception;
   - the reference-root manifest: builds/r23/gsw_ind_2016_* is excluded from the before-and-after manifest while step 4
     writes it, then added, and a complete-file record (every label window with a sidecar has its r23 build) is written
     before check C runs.
2. A narrower upper bound: one game whose status can change, and at best a third usable GSW game (gsw_sac_2015 is
   apart), not a new arena. It would take the usable development games from 3 (gsw_cha_2016, gsw_bkn_2015,
   hou_cle_2016) to 4, so a game-level bootstrap (D85 (3), games as clusters) would go from 3 to 4 clusters; the revived
   design must say what that buys for the comparisons that would use it.
3. Switch wording: the rule of section 6 is D199's unchanged; the revived text says "the lead's whole-set decision" and
   does not imply that a passing truth lifts D98 by itself.
4. Build order with a futility rule: build and check section by section, and stop once check C can no longer pass
   (for example, after enough sections that even the best remaining windows cannot bring the median to 0.54 s), so the
   lock hours spent before check C is known are bounded; the rule and its arithmetic are fixed in the revived version
   before any section runs.
5. Every estimate and cap is restated per section for the futility rule.

Sections 1 to 8 below are version 1 as reviewed; they are not run.

Written 2026-11-27 for PLAN.md D199's queue item 1. Design only: nothing here has run; every figure quoted comes from
committed records (P8_truth_arenas.md sections 5, 15 to 17; P8_multistart_fit.md; P8_cha_fit.md; the P8 workspace's
recorded MPS build and fit of gsw_ind_2016, read only). No held-out game, no D85 test game and no test arena is read
(CLAUDE.md rule 4). Frozen code is not edited; the steps call tools.p8_frozen, tools.p8_truth_cpu.sh,
tools.p8_multistart and the acceptance tools, which are already reviewed.

## 1. Question, and a correction to the item's wording

Can gsw_ind_2016 get a camera truth that section 5 of P8_truth_arenas.md accepts ("use"), built on the reference
platform (detector on CPU, D175) with the tracker reset fix (D192) and the multi-pool camera fit (D199)?

Correction, stated before anything else: gsw_ind_2016 is a Warriors home game (home GSW, SportVU 01.22.2016.IND.at.GSW,
data/harvest/games.json), so it would not add a fourth arena. Development camera truth now covers two non-sealed
arenas, GSW (gsw_cha_2016 and gsw_bkn_2015 on multipool v1; gsw_sac_2015 apart) and HOU (hou_cle_2016); a usable
gsw_ind_2016 would be a fourth in-use game and a third usable GSW game, not a new arena. What it adds: one more
development game with truth for pooled GSW rows, and the game that D98 set apart back in the pool if it passes.

## 2. Why D98 can be revisited

D98 kept gsw_ind_2016 apart because its Mac (MPS) truth fit was off: pooled reprojection 18.35 px against 6.53 on Colab,
camera height 106 ft against 123, "until the cause is known". Since then:

- P8_cha_fit.md found that the frozen pooled fit (sportvu.camera.fit_game) can stay at its start for some pools on any
  build, and that both broken fits on record (gsw_cha_2016 CPU fixed, gsw_ind_2016 MPS) ended within about 1 ft of their
  start.
- P8_multistart_fit.md (D199) registered a reading for gsw_ind_2016 before its numbers and ran it on the MPS build
  (report only): D98's recorded fit was reproduced exactly, and the multi-pool selection was 33.17 ft from the frozen C,
  its common objective J 0.564 of the frozen fit's, and stable (0.033 to 0.066 ft): "same fragility". So the most likely
  cause D98 waited for is known: the fit, not the data.
- The registered limitation (P8_multistart_fit.md section 17) holds here too: that reading was on an MPS build that
  carries the tracker bug, so it does not show that a CPU rebuild will pass. A CPU rebuild is the separate registration
  that limitation calls for, and this is it. D98's "apart" stands until this design's result is decided.

## 3. Guards

- gsw_ind_2016 is a development game: it is in sportvu/splits.json "train" (one of the four R1.3 games; checked for this
  design: it is in "train" and in none of heldout_game, heldout_arena, heldout_era or excluded), and it is not a D85 test
  game, video or arena (D85 (1): the Miami, Toronto and Staples Center games; this is a Golden State home game).
- tools.p8_common.guard and tools.research_testset.refuse_cv_test run on the tag, the SportVU id, the video id
  (UQHXGIH12Ds) and every window, in every step (each tool already does).
- The P8 workspace's MPS build and labels of gsw_ind_2016 (work/label_build, work/labels) are read only and never
  written; nothing here changes D98's records.
- The reference root's other games' files are protected by the SHA-256 manifest of tools.p8_multistart (labels,
  builds/r23, truth pins, label sidecars) before and after each step, extended to every gsw_ind_2016 file in builds/r23
  after it is built.

## 4. Steps

All under the heavy-job lock, one job at a time, each step under its own lock name, resumable, at most two retries per
step, in the reference root (work_ref) unless named; the detector on CPU, court and gate auto, no clip fix,
P8_TRACKER_RESET_FIX=1, TRACKER_REID unset, the r23 detector checked at start (tools/p8_truth_cpu.sh's start guards).

1. Link: tools.p8_frozen link (gsw_ind_2016 is an R1.3 game): read-only links to the main checkout's sections, SportVU
   moments and sync maps, the inputs of its earlier truths. Mirror from the R1.2 record (status ok, identity, vote share
   0.92), as for the other GSW games. Layout abc_finals16_720 (frozen).
2. Label builds with the fix, one job a section (tools/p8_truth_cpu.sh truth phase; its supersede step finds nothing
   to move for a game with no earlier reference-root builds), then the label-check (device records and reset logs,
   tools/p8_truth_fixprep.py). The driver's game table gets gsw_ind_2016 with this design's caps (a reviewed code
   change); its report-only comparisons after the fit run against the workspace's MPS truth (p8_truth_moved; the d_H
   report has no r23 sidecar for this game and is skipped).
3. Frozen-fit truth: tools.p8_frozen fit in work_ref (the truth phase's fit step): the reset-fixed CPU truth with the
   frozen fit, the game's fallback version (section 6).
4. Check C's r23 side: the frozen rebuild step (tools.p8_frozen rebuild --build r23, the r23 court checked against its
   SHA-256), with the same flags as the label builds, one job a section, writing builds/r23/gsw_ind_2016_* in work_ref.
   Why not the reset-fix arm's reset-window builds: those compose with an unfixed CPU r23 baseline (M9 and M6 builds),
   and gsw_ind_2016 has none, so a baseline would be a full build anyway; one full fixed build costs the same and needs
   no composite. This amends section 16 version 4.1's "compose is the only writer of builds/r23" for this game only:
   builds/r23 must hold no gsw_ind_2016 file before the step, and afterwards every gsw_ind_2016 window's device record
   must show reset_fix true, the detector requested and resolved cpu, court and gate auto, no clip fix; every label
   window with a sidecar must have an r23 build ("complete"), or check C does not run.
5. Acceptance of the frozen-fit truth (work_ref): tools.p8_truth_v2 measure-dev and tools.p8_truth_accept, section 5's
   bars unchanged.
6. Multi-pool fit (only if step 5's check C passes; section 5): tools.p8_multistart's stage 1 procedure for this game
   (three seed sets of 8 runs, 20 chunks of 200 evaluations, kill rule 1 after chunk 1, J at every final C, lowest J per
   set, stability at unfitted feet with the 0.25 ft bar), unchanged; the game is added to the tool's allowed list by a
   reviewed code change that touches nothing else.
7. Multi-pool truth in work_ms (only if stable) and its acceptance there (copies of the game's builds/r23 files), as
   for gsw_cha_2016 and gsw_bkn_2015.

## 5. Kill rules

1. A guard fails: nothing runs.
2. Label builds: more than 10 percent of the game's windows fail to build (gate-empty windows are not failures), or the
   label-check finds any window without the fix or the reference devices: stop and diagnose before any fit.
3. A step's lock time passes its cap (section 7): stop and re-estimate.
4. Check C fails on the frozen-fit truth (step 5): stop before the multi-pool steps. Check C does not depend on the fit,
   so no fit can change it; gsw_ind_2016 is reported apart on check C, and the multi-pool steps (about 0.7 h) are saved.
5. Multi-pool: a first chunk that does not reproduce fit_game stops the game (kill rule 1 of P8_multistart_fit.md); an
   unstable game gets no multi-pool truth (its fallback stands).
6. Not kill rules: the frozen fit failing A or B (that is the case the multi-pool fit exists for), or the multi-pool
   truth failing A or B (a result).

## 6. What decides the version, and the switch rule unchanged

The switch rule of P8_multistart_fit.md section 9 (D199) applies unchanged: the game uses multipool v1 if and only if
it is stable and its multi-pool truth reads "use"; otherwise its reset-fixed CPU truth with the frozen fit, if that
reads "use"; otherwise it has no truth in use ("apart" or "unused" by section 5). The lead decides it with the rest of
the set; each comparison uses each game's decided version; nothing is rescored by this item, and any rescore that adds
gsw_ind_2016 is its own registration. If gsw_ind_2016 reads "use" in either version, D98's "apart" is lifted by the
lead's decision, not by this document.

What this cannot show: a truth that passes section 5 is consistent (registration, coverage, offset stability); it is
not shown to be closer to the players' true positions than another passing truth (P8_multistart_fit.md section 18).

## 7. Cost (lock seconds), estimates and caps

From the reference-platform truth runs of section 17 and the multi-pool records; gsw_ind_2016's label windows are 92
windows and 25,058 processed frames on its MPS build (the CPU build processes the same windows and frames).

| Step | Basis | Estimate | Cap (1.5 times) |
| --- | --- | --- | --- |
| 1 link | no model | 0 | 0 |
| 2 label builds and label-check | CPU label rate with the fix, 0.47 to 0.58 s a processed frame (hou_cle_2016 to gsw_sac_2015, section 17); MPS label time 5,918 s times the CPU/MPS ratio of the other games (1.95 to 2.43) | 13,300 s | 19,950 s |
| 3 frozen-fit truth | fit steps 178 to 285 s (section 17), MPS fit 195 s | 250 s | 375 s |
| 4 r23 rebuild with the fix | 0.437 to 0.446 s a processed frame (M6, M9 CPU r23 builds) | 11,000 s | 16,500 s |
| 5 acceptance (frozen fit) | measure-dev 152 to 181 s | 200 s | 300 s |
| 6 multi-pool | measured 1,425 to 1,661 s a game (P8_multistart_fit.md), scaled for more inits on a CPU build | 2,000 s | 3,000 s |
| 7 multi-pool truth and acceptance | 212 to 218 s and 156 to 168 s | 400 s | 600 s |
| All | | 27,150 s (7.5 h) | |

If kill rule 4 stops at step 5, steps 6 and 7 do not run (about 0.7 h saved). 0 Colab units.

## 8. What is not done

No compute, no numbers, no truth written, nothing rescored, no D line. The workspace's MPS build and D98's records stay
as they are. The code changes (gsw_ind_2016 in the truth driver's game table with these caps, in the multi-pool tool's
allowed list, and the checks of step 4) are written and reviewed before anything runs.
