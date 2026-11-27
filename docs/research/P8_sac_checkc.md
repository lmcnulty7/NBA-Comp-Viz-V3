# P8: why gsw_sac_2015's fixed CPU truth fails check C (registered 2026-11-27, before any number)

Status: design only, report only. No number in this document has been computed yet. Nothing trains; no rule of
P8_truth_arenas.md or P8_resetfix.md changes; gsw_sac_2015 stays "apart" whatever this finds (D199). Code, when
approved: tools/p8_sac_checkc.py with tests; record reports/research/p8_sac_checkc.json.

## 1. Question

Check C (P8_truth_arenas.md section 5) measures how stable the frozen per-window clock offset is between the V3 label
build and the r23 rebuild of the same window (tools.p8_truth_v2 measure-dev: median absolute difference and share of
windows over 0.2 s, on windows with at least 20 usable frames). gsw_sac_2015's camera truth rebuilt on the CPU detector
with the reset fix fails it (median 0.64 s against the bar 0.54; share over 0.2 s 0.822 against 0.7945), while its
earlier Mac MPS truth passed. The share bar itself came from gsw_sac_2015's MPS record (the worst in-use pooled game of
P8_truth_v2.md), so the MPS truth sat at the bar. What moved the offsets: a few windows or all of them, the reset-fix
windows or the others, the V3 side or the r23 side, the clock, or the frames the fixed tracker now admits?

## 2. Stored inputs (all exist; nothing is rebuilt)

| record | what it holds |
|---|---|
| reference root truth_v2/dev/gsw_sac_2015.json (fixed) | per window: frozen offset of the fixed V3 label build and of check C's r23 composite (builds/r23: the reset-fix arm on the planned reset windows, the unfixed baseline elsewhere), v2 offsets with their frame counts, the stored frozen offset of the fit, candidates; the game summary |
| P8 workspace truth_v2/gsw_sac_2015.json (MPS) | the same fields for the MPS V3 label build against the workspace's r23 rebuild (the registered truth v2 measure) |
| reference root resetfix/plan__gsw_sac_2015.json, equiv__gsw_sac_2015.json | the 10 reset windows; the arm's reset log frames; the equivalence result |
| label_build/<window>_frames.json and _timemap.json in both roots; reference root builds/r23, builds/r23_nofix, builds/resetfix_r23_detcpu__gsw_sac_2015; workspace builds/r23 | the sidecars and time maps both records were computed from |
| labels/gsw_sac_2015_offsets.json in both roots | the fits' stored offsets |
| reference root truth_v2/dev/gsw_bkn_2015.json and the workspace's gsw_bkn_2015 record | the same comparison on a game whose fixed truth passes C (control) |

## 3. Candidate causes and the measurements that tell them apart

The game's fixed truth was built with the detector on CPU and the reset fix, so two changes are confounded against
the MPS record: the platform (CPU against MPS) and the fix. They separate by window: in a window with no reset, the
fixed and unfixed CPU builds are identical (this arm's equivalence windows, and by construction), so a change there is
the platform's; only the reset windows can carry the fix's effect.

**M1. Where the failure sits (stage 1, stored records).** Per window, d = |frozen V3 - frozen r23| in each record, on
the windows check C measures (both offsets present). Report the paired table by window: d(MPS), d(fixed), and the
windows that cross 0.2 s in either direction; the median and share for the reset windows (plan) and the other windows
separately, in each record; how many windows carry the change in share (the count of windows that cross upward minus
those that cross downward). Same for gsw_bkn_2015 as a control.

**M2. Which side moved (stage 1).** Per window, the change of each side between the records: V3 side, frozen V3
(fixed) - frozen V3 (MPS); r23 side, frozen r23 (fixed) - frozen r23 (MPS). Their medians and shares over 0.2 s, split
as in M1. Also each fixed V3 offset against the fit's stored offset (stored_frozen_v3, and the summary's
frozen_recompute_mismatch list), which shows whether measure-dev reproduced the fit's offsets.

**M3. Clock and time maps (stage 1).** Per window, are the two roots' time maps identical (SHA-256), and if not, the
largest clock difference at the same frame; the two fits' stored offsets per window (labels/<game>_offsets.json),
their differences, and how many windows hit the grid edge (plus or minus 1.5 s) in either record.

**M4. Frames now admitted (stage 1 counts; stage 2 only if needed).** The frozen rule uses frames with a clock, an H
and at least 4 boxes. Stage 1: per window and side, the number of such frames in each sidecar (fixed V3, MPS V3,
composite r23, r23_nofix, workspace r23), counted from the stored sidecars, and, for reset windows, how many lie after
the arm's first logged reset. Stage 2 (only if stage 1 leaves the cause open): recompute the frozen offset with the
frozen rule (sportvu.truth.solve_window_offset through tools.p8_truth_v2.rows_pc_of, unchanged) on stored sidecars for
(a) the fixed V3 build and the composite r23 restricted to frames before each window's first logged reset, and (b) the
fixed V3 build against builds/r23_nofix (the composite with the fix taken out of the r23 side). The recompute first
reproduces the fixed record's offsets on the full sidecars, else nothing more is reported.

## 4. Readings (fixed now)

- **Few windows:** if the upward crossings minus downward crossings are 3 or fewer and the median moves through a few
  large d, the failure rests on a handful of windows near the bar (the MPS record sat at it); it is reported as
  fragile, with those windows named.
- **Platform, not the fix:** if the windows that cross upward are mostly (at least two thirds) no-reset windows, the
  cause is the CPU platform (as gsw_cha_2016's MOVED finding, P8_truth_arenas.md section 15), not the reset fix.
- **The fix:** if they are mostly reset windows, and M4 shows extra admitted frames after a reset there, the fix's
  newly confirmed boxes (frames after camera cuts) move the offsets; stage 2 (a) tests it: offsets on frames before the
  first reset returning to within 0.2 s of the MPS pair supports it.
- **V3 side or r23 side:** M2's side with the larger median change, and stage 2 (b) for the r23 side.
- **Clock:** if any time map differs, or the fits' stored offsets differ by more than 0.2 s in the windows that cross,
  the cause is the clock or the sync, not the detector; this outranks the readings above for those windows.
- **Not separated:** if none of these patterns holds (for example, broad small shifts on both sides in all windows),
  the study says so and names the largest contributor.
Each reading is a description; post hoc explanations are labelled post hoc.

## 5. Cost and guards

Stage 1 reads stored JSON only (records, sidecars, time maps, plans): system python, no model, no lock, seconds. Stage
2 runs the frozen offset solver on stored sidecars (frozen CV python, no model, CPU numpy over at most 82 windows and
four sidecar variants); it reads DATA_ROOT-free workspace files but imports frozen code, so it runs under the
heavy-job lock as measure-dev did, after its own go. Guards: tools.research_testset.refuse_cv_test and
tools.p8_common.guard on the game list; gsw_sac_2015 and gsw_bkn_2015 only; nothing held out is read.
