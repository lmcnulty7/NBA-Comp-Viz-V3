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

## 6. Version 2 (2026-11-27; the review of cd0f989f, NO-GO with 4 blockers; still before any number)

This version replaces sections 3, 4 and 5 where it differs; sections 1 and 2 stand, with the corrections below.
Nothing has been computed. Facts quoted from the reviewer are marked as such; the study re-derives every one in
stage 1 and stops if any differs.

### 6.1 Window classes (replaces section 3's separation)

Each gsw_sac_2015 window is classed by whether its builds reset, read from the identity files (camera_cut_resets
above 0) of all four builds: MPS V3 (workspace label_build), MPS r23 (workspace builds/r23), CPU V3 (reference root
label_build, built with the fix) and CPU r23 (reference root builds/r23_nofix; the plan's reset windows):
- **both**: at least one MPS build and at least one CPU build reset;
- **MPS only**: an MPS build resets, no CPU build does;
- **CPU only**: a CPU build resets, no MPS build does;
- **none**: no build resets.
Reviewer's count, to be re-derived: the MPS builds reset in 12 windows (s01_w00, s02_w00, s04_w02, s04_w08, s05_w04,
s07_w00, s07_w02, s07_w08, s08_w02, s08_w04, s09_w01, s09_w05), the CPU builds in the plan's 10; s02_w00, s04_w08 and
s05_w04 reset on MPS only, s06_w02 on CPU only. A "platform only" effect is claimed only for the none class: in MPS-only
windows the MPS builds carry the bug while the CPU builds do not reset, and in both and CPU-only windows the fix acts,
so platform and fix are confounded there. The MPS builds' reset frames are not stored (no reset log; only counts), so
nothing is cut at an MPS reset.

### 6.2 The share change is known (replaces "how many windows carry the change")

From the records (reviewer's figures, re-derived in stage 1): the share moves from 58 to 60 of 73 windows (0.7945 to
0.8219), a net change of +2 windows; the median from 0.52 to 0.64 s. Stage 1 reports the gross crossings (windows going
above 0.2 s and windows going below) by class, and judges the size of the change against the controls (6.6).

### 6.3 Decomposition and readings (replaces section 4's Platform and Fix rules)

Swap hybrids, diagnostic only (never a check C value): **(i)** the fixed record's d in the none class and the MPS
record's d in every other class; **(ii)** the reverse. For each statistic (the share over 0.2 s and the median of d,
each judged separately), a hybrid "moves" when it covers at least half of the change from the MPS value to the fixed
value, in the same direction. Readings, in this fixed priority, for each statistic:

| order | reading | condition |
|---|---|---|
| 1 | Clock | any window time map, the SportVU moments file or the game's mirror differs between the roots (6.4) |
| 2 | Platform | (i) moves and (ii) does not |
| 3 | Reset-affected windows (labelled Fix) | (ii) moves and (i) does not; the class mixes the fix (both, CPU only) and the MPS builds' bug (MPS only), and the report says so |
| 4 | Joint | both move |
| 5 | Neither | neither moves |

The few-windows note stands as a description: the gross crossings and the windows carrying them are named.

### 6.4 Clock (replaces M3's reading)

Clock rests only on identity, checked by SHA-256: each window's time map in the two roots, the SportVU moments file
the two fits read, and the game's mirror (direction result). The fits' stored offsets are not part of it (they move
with the V3 side, which the reviewer finds moved in 0.41 of windows); M3 keeps the stored offsets and grid-edge hits as
descriptions only (edge hits are counted, never a reading).

### 6.5 Side (replaces M2's reading)

Cross pairs, valid only when every time map is identical (6.4): **V3 swap**, d of (fixed V3, MPS r23) per window;
**r23 swap**, d of (MPS V3, fixed r23). A side is named only if its swap alone moves both the median and the share at
least halfway from the MPS value toward the fixed value; otherwise the reading is "both". Note: the r23 side changes
platform as well (the workspace's r23 rebuild is MPS, the reference root's is CPU), so an r23-side reading includes the
platform; it is not the fix alone.

### 6.6 Controls

- gsw_bkn_2015: the MPS record and the fixed record (its fixed truth passes C).
- gsw_cha_2016: three records, the MPS record (workspace truth_v2/gsw_cha_2016.json), the CPU unfixed record
  (reference root superseded_nofix/truth_v2/dev/gsw_cha_2016.json) and the CPU fixed record (reference root
  truth_v2/dev/gsw_cha_2016.json), each on its own windows and on the windows common to all three.
- For each control: the change in median and share between its records. Reading, report only: sac's change from MPS
  to fixed is "within the controls' range of change" when both its median change and its share change lie inside the
  range spanned by the controls' changes (bkn MPS to fixed, cha MPS to unfixed, cha MPS to fixed, cha unfixed to
  fixed); otherwise "outside", with the statistic named.

### 6.7 Evidence for "identical without a reset"

Cited exactly: reference root resetfix/equiv__gsw_sac_2015.json, both equivalence windows (s01_w01, s10_w05)
identical to builds/r23_nofix (frames as canonical JSON without "source", trajectories and identity byte for byte),
and the prefix check passing on the 10 reset windows; the same file for gsw_cha_2016 and gsw_bkn_2015. New in stage 1,
for the V3 side: every gsw_cha_2016 window whose CPU V3 label build has no reset is compared with its unfixed CPU label
build (reference root superseded_nofix/label_build) by SHA-256 of frames, trajectories and identity; the count of
identical and differing windows is reported, and a differing window is named.

### 6.8 Stage 1 assertions and cost

Before any reading, stage 1 asserts and stops on a mismatch: both sac records hold the same 82 windows; the fixed
record's check C set is 73 windows; stability recomputed from each record's window rows equals its stored summary
(fixed: n 73, median 0.64, share 0.8219; MPS: median 0.52, share 0.7945, n re-derived and reported); the controls'
summaries likewise. Stage 1 reads one file at a time and counts frames with a clock through the frozen
sportvu.truth.frame_clock (as tools.p8_truth_v2.rows_pc_of does). System python, no model, no lock.

### 6.9 Stage 2 (replaces section 3's M4 stage 2 and section 5's stage 2)

Runs only if stage 1's reading for the share or the median is Fix, Joint or Neither, and after its own go.
- **Reproduction gate first, exact:** with the same code path (tools.p8_truth_v2.rows_pc_of and solve_frozen,
  unchanged) and the same interpreter as measure-dev (the frozen CV python), recompute the frozen offset of every
  window and side of the sac fixed and MPS records on the full stored sidecars; every value must equal the record on
  the 0.04 s grid, and the summaries must reproduce (fixed n 73, 0.64, 0.8219; MPS 0.52, 0.7945). Otherwise nothing is
  reported.
- **(a) Before the reset.** For the CPU reset windows, both sides (fixed V3 and composite r23) are cut at the same
  frame, the arm's first logged reset (resetfix log of the CPU r23 build), keeping frames before it; the MPS builds
  are not cut (their reset frames are not stored). A cut window with fewer than MIN_FRAMES (20) usable frames on either
  side gets no offset and is reported as dropped (the reviewer expects at most 6 of 10 to remain). A remaining window
  "returns" when |d(cut) - d(MPS)| is at most 0.2 s. Reading: the fix moves the offsets through post-reset frames if
  at least k = 3 remaining windows return and they are more than half of the remaining windows; otherwise not shown.
- **(b) r23 side without the fix.** d of (fixed V3, builds/r23_nofix) per window. Byte check first: for every
  no-reset window, builds/r23_nofix and the composite builds/r23 hold byte-identical frames, trajectories and identity
  (else (b) is not reported). Reading: if replacing the composite by r23_nofix moves both the median and the share at
  least halfway back toward the MPS values, the fix on the r23 side contributes; otherwise it does not.
- Never run: cmd_measure_dev, or any write under truth_v2/ or truth_accept/ in either root; the tool writes only its
  own record. Lock name "sac checkc stage 2", estimate 159 s (measure-dev's recorded time for this game, per the
  reviewer), cap 1.5 times (240 s).

### 6.10 Guards and corrections

- Games: gsw_sac_2015 and the controls gsw_bkn_2015 and gsw_cha_2016 only, through
  tools.research_testset.refuse_cv_test and tools.p8_common.guard. Nothing held out is read.
- No window is excluded from check C, nothing is adopted, and any change of gsw_sac_2015's status ("apart", D199) is a
  new registration.
- Correction to section 4's Platform reading: P8_truth_arenas.md section 15 found that gsw_cha_2016's truth MOVED on
  accepted frames, while its fits' stored per-window offsets did not move (median 0.08 s); it is not evidence that the
  CPU platform moves check C offsets, and this study no longer cites it as such.
