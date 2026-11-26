# P8: which court output is closest to camera truth? Design before any number

Written 2026-11-26 (UTC), before any number, for queue item 3 after D179. Version 2, the same day and still before
any run, applies the ml-review of d86ac0e2 and 82190894 (section 8); where the versions differ, version 2 holds. Report only; JSON only (no model, no video
decode, no lock). The tool is `tools/p8_court_truth.py`; it is not run until this design has been reviewed. The games
are gsw_bkn_2015 and gsw_ind_2016, R1.3 training games (development data; gsw_ind_2016 is reported apart, D98). No
held-out game is read and no held-out number is computed (CLAUDE.md rule 4).

## 1. The question

D179 (docs/research/P8_cuda_check.md section 13) found the court model's homography device-sensitive by several pixels
on every device pair: Mac CPU against Mac MPS 3.8 and 4.7 px, CUDA against Mac CPU 8.8 and 8.0 px, CUDA against Mac
MPS 9.4, 10.6 and (r23 court, gsw_bkn_2015_s04_w00) 10.1 px. Which of these court outputs is closest to camera truth?
D175 keeps the court model on MPS for reference builds; nothing yet says that output is the most accurate one.

## 2. What is measured

For each window, each build and each camera-truth frame of the window's frame set (section 3):

- **Primary: the point gap.** The median pixel distance between the images of three court points (47, 25), (20, 10)
  and (70, 40) ft under the build's homography and under H_truth (both map image to court; each is inverted to map the
  points into the image). This is the measure of the CUDA check (tools.p8_platform_summary.h_gap_px, per frame), so a
  build-to-truth gap reads on the same scale as the build-to-build gaps above.
- **Secondary: the frozen court line error.** `sportvu.bench.line_error_px(H_build, H_truth, 1280, 720)`, the
  scorecard's court line measure (the template drawn through each homography, symmetric median px, on-frame samples
  only). Reported beside; it does not decide.

Per build: frames measured (both homographies present), median and p90 of the point gap, median line error. Per pair
of builds A and B: the median of the per-frame paired difference (gap of A minus gap of B) over frames where both
have a homography, with a frame bootstrap (2,000 resamples, seed 0, 2.5 and 97.5 percentiles). Beside it, a block
bootstrap over runs of 10 consecutive truth frames, because neighbouring frames are correlated and a frame bootstrap
gives intervals that are too narrow; the frame bootstrap decides, as asked, and the block interval is reported.

Reading of a pair, fixed now: A is **closer to this truth** than B when the median paired difference is at most
-1.0 px and its frame-bootstrap interval lies below zero; B in the mirror case; otherwise **no difference resolved**.
1.0 px is about a quarter of the smallest build-to-build court gap (3.8 px, Mac CPU against MPS), so a smaller
difference would not separate devices that are themselves 4 to 10 px apart. Every reading means closer to this camera
truth, not more accurate (section 4).

Beside each pair (report only, version 2): the number of bootstrap blocks, the median build-to-build point gap on the
same frames, and the paired difference of the frozen line error (on-frame template samples only) with the same two
bootstraps. Beside each window: the median paint edge_px of its primary frames (how far the painted key sits from
H_truth on those frames) and the count of truth frames by how many of the three court points land on the frame (0 to
3). The frame interval decides each pair's reading. An outcome that would lead to a D line also needs the block
interval to exclude zero on the same side and the line-error difference to have the same sign; where they disagree,
the outcome is reported as not ready for a D line.

## 3. Windows, builds and frames (fixed now)

**gsw_bkn_2015_s04_w00 first.** Camera truth: the P8 workspace's labels/gsw_bkn_2015/gsw_bkn_2015_s04_w00.jsonl (140
frames; the gsw_bkn_2015 fit, reprojection 6.52 px), on the same clip every build read. Builds: Mac MPS (the
registered r23 build, builds/r23), mac_detcpu (Mac detector-CPU, M6, builds/e1m6_r23_detcpu) and cuda_a (the CUDA
check's result).
Both Mac builds ran the court model on MPS and their court homographies are identical (gap 0.0 px, D179), so this
window compares two court outputs, MPS and CUDA; the detector-CPU build is kept as a check that its court output equals
MPS's.

Frames: the **106 frames the GSW gold-key paint check keeps** (labels/gsw_bkn_2015_paint.jsonl, keep true), primary;
all 140 reported beside as a sensitivity row. Why, fixed in advance: the paint check tests H_truth against the painted
key edges in the image, independently of every build's court output, and the 34 frames it rejects (33 paint_edge_high,
1 paint_edge_unmatched) are frames where the truth itself is shown off at the key. Measuring builds against truth known
to be off there would mix truth error into the ranking.

**gsw_ind_2016_s02_w03 and s02_w04, only after a frame-alignment check.** Camera truth: the Colab R1.3 labels in the
main checkout, data/sportvu/labels/gsw_ind_2016/<window>.jsonl (72 and 62 frames; the A100 fit, reprojection 6.53 px),
with the same paint rule (data/sportvu/labels/gsw_ind_2016_paint.jsonl keeps 70 and 36). The Mac's own gsw_ind_2016
truth (the broken fit, D98) is not used. Builds: mac_mps, mac_cpu, mac_detcpu (s02_w03 only;
tools.p8_truth_v2.DEVICE_BUILDS) and cuda_a. The detector-CPU build is named mac_detcpu on every window. Here the court ran on three devices (Mac CPU, MPS, CUDA), so these two
windows carry the Mac CPU against MPS question.

The R1.3 clips were cut on Colab, so frame indices could differ from the Mac clips the builds read. Alignment check,
fixed now and run first: each R1.3 truth row carries the boxes of the A100 build at that frame. For offsets of -1, 0
and +1 build steps (6 frames at this window's stride), the share of truth-row boxes that have a box with IoU at least
0.9 in the Mac CPU build at frame + offset. The window is **aligned** when the share at offset 0 is at least 0.9 and
higher than at both neighbours; otherwise it is skipped and recorded. (Seen while writing this design, one row only:
frame 24 of s02_w03 carries the box [929, 351, 1024, 552], exactly the Mac CPU build's box at frame 24. The check
decides, not this.)

## 4. The selection and initialisation caveat

Camera truth is fit to SportVU positions and the painted lines, but it is not independent of one build's court
output. It is initialised from that build's homography: the camera position C0 is the median of poses taken from the
build's H (sportvu/camera.py, fit_game, lines 169 to 174), each frame's fit starts from the pose of the build's H
(fit_frame, lines 206 to 208), and the line refinement searches near that projection. And the frames that reach it
were chosen through the same build's projected feet: for gsw_bkn_2015, the V3 label build on MPS (the
court model there is V3, not the r23 court of the builds compared); for gsw_ind_2016, the A100 build (court on CUDA).
Selection and initialisation can both favour the court output of that build. The reading in section 5 treats a pattern
that follows the selecting platform as consistent with a lean, not as accuracy.

The minimum honest claim this measurement can support, written now: "On these frames, build X's court homography is
closer to camera truth than build Y's by D px (interval [a, b]); the truth was picked and initialised through one
build's homography, so this is closeness to that truth, not accuracy." Accepted frames only; three windows on two games is a
description, not a platform ranking.

## 5. What each outcome would change (version 2)

- **Court device for reference builds (gsw_ind_2016 windows, mac_cpu against mac_mps).** If the CPU court is closer to
  this truth on both aligned windows, D-line ready (section 2): propose a confirmation measurement, a D line for a test,
  not a device change. That test needs windows from at least two games whose truth was not picked or initialised
  through a CUDA or MPS build, or a refit of truth from each build's homography showing that H_truth moves by under
  1.0 px. If MPS is closer on both: D175 stands as is. Otherwise (mixed, no difference resolved, or not D-line ready):
  no change.
- **CUDA's court.** CUDA counts as closer to this truth only if it is closer, D-line ready, on s04_w00 and on both
  aligned gsw_ind_2016 windows (against mac_mps there, and against mac_cpu on the gsw_ind_2016 windows). Closer on the
  gsw_ind_2016 windows only is consistent with a lean (their truth was picked and initialised through a CUDA build),
  and no change follows. Farther confirms D179 (1).
- **A lean toward the MPS court in existing truth.** On s04_w00 (truth picked through the V3 court on MPS; the builds
  compared use the r23 court, so this half tests the V3 court on MPS through r23 builds) mac_mps closer than cuda_a,
  while on both aligned gsw_ind_2016 windows (truth picked through a CUDA build) cuda_a closer than mac_mps: that
  pattern is consistent with a lean toward the selecting build's court, which would add weight to the CPU truth
  rebuild of P8_truth_arenas.md section 0. Any other pattern does not show a lean and does not rule it out. Both
  gsw_ind_2016 windows must be aligned for this reading.
- Nothing is adopted or rescored here; every change above is a separate decision.

## 6. Output and guards

Output: <P8 root>/court_truth/result.json, holding the alignment checks, per-build and per-pair rows of both frame
sets, both intervals and the readings, with no local path. Guards: refuse_cv_test and tools.p8_common.guard on both
games and every window before any file is read. It reads JSON only, so no heavy-job lock is needed.

## 7. Command (after review; not run yet)

    <frozen CV python> -m tools.p8_court_truth run --cuda-result <P8 data folder>/cuda_check/result

The frozen CV Python is used because `sportvu.bench.line_error_px` imports the frozen court code and OpenCV. Every input
named in section 3 was checked to exist (file existence only, no number read) before this design was sent to review.

## 8. Version 2 (2026-11-26, after the ml-review of d86ac0e2 and 82190894; before any run)

1. Section 4: camera truth is initialised from the selecting build's homography (camera.py 169 to 174 and 206 to 208,
   and the line refinement searches near it), not only selected through it. The "CPU closer on both" outcome now
   proposes a confirmation test, not a device change (section 5), and the minimum honest claim is written.
2. Every reading means "closer to this truth", not "more accurate". Beside each window: the median paint edge_px of the
   primary frames; beside each pair: the median build-to-build gap on the same frames (report only).
3. The frame interval decides each pair's reading; an outcome that would lead to a D line also needs the block
   interval to exclude zero.
4. CUDA counts as closer only if closer on s04_w00 and on both aligned gsw_ind_2016 windows; closer on the gsw_ind_2016
   windows only is consistent with a lean, and no change follows.
5. The bias reading is "consistent with a lean" or "does not show a lean, does not rule it out"; the s04_w00 half tests
   the V3 court on MPS through r23 builds; it needs both gsw_ind_2016 windows aligned.
6. Each pair row records its number of bootstrap blocks. s02_w04's primary set is 36 frames, so its block bootstrap
   has 4 blocks of 10 frames (the last shorter) and is coarse; it is reported as such.
7. Each window records, per its frame sets, how many truth frames have 0, 1, 2 or 3 of the three court points on the
   frame; each pair adds the paired line-error difference (on-frame samples only) with the same bootstraps; an outcome
   that would lead to a D line is not ready when the point gap and the line error disagree in sign.
8. The frame size is read from each window's clip header (no frame decoded) and must be 1280x720, or the tool stops.
   All three clips were checked: 1280x720.
9. The detector-CPU build is named mac_detcpu on every window.

## 9. Result (2026-11-26, run by the lead at 9afeb905; report only)

Record: reports/research/p8_court_truth.json (a copy of the P8 workspace's court_truth/result.json). Frozen CV Python,
JSON and clip headers only. Every reading below means **closer to this camera truth, not more accurate** (section 4):
the truth was picked and initialised through one build's homography.

**Alignment.** Both gsw_ind_2016 windows are aligned: the share of R1.3 truth boxes matched in the Mac CPU build is
0.971 at offset 0 against 0.212 and 0.224 one build step either side (s02_w03), and 0.953 against 0.121 and 0.137
(s02_w04). The detector-CPU build's court homography equals the MPS build's on every frame of every window (gap 0.0
px), as expected, so mac_detcpu adds no court output of its own.

**Per window and build, primary frames** (point gap median and p90, frozen line error median, px):

| Window (primary frames; truth frames with 0/1/2/3 court points on the frame; paint edge_px median) | cuda_a | mac_mps | mac_cpu |
| --- | --- | --- | --- |
| gsw_bkn_2015_s04_w00 (106; 8/95/3/0; 6.85) | 38.20 (99.55), line 6.68 | 38.13 (92.09), line 6.73 | not built |
| gsw_ind_2016_s02_w03 (70; 0/70/0/0; 6.26) | 259.55 (318.89), line 18.71 | 265.79 (324.43), line 19.15 | 266.34 (326.78), line 19.88 |
| gsw_ind_2016_s02_w04 (36; 0/34/2/0; 7.54) | 223.53 (305.99), line 19.14 | 221.75 (315.55), line 20.06 | 222.03 (319.53), line 20.95 |

No build lacked a homography on any primary frame.

**Pairs, primary frames** (median paired point-gap difference A minus B, frame and block intervals, blocks, median
build-to-build gap on the same frames, paired line-error difference with its frame interval; px):

| Window, A against B | Frames (blocks) | Difference | Frame interval | Block interval | Reading | Build-to-build gap | Line-error difference | D-line ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s04_w00, cuda_a against mac_mps | 106 (11) | -1.13 | [-3.32, 0.16] | [-3.12, 0.16] | no difference resolved | 9.94 | 0.00 [-0.10, 0.06] | no |
| s02_w03, cuda_a against mac_mps | 70 (7) | -5.39 | [-8.19, -2.01] | [-8.32, -1.92] | cuda_a closer | 9.23 | -0.77 [-1.22, -0.41] | yes |
| s02_w03, cuda_a against mac_cpu | 70 (7) | -6.35 | [-8.69, -3.21] | [-7.68, -4.38] | cuda_a closer | 7.47 | -0.84 [-1.05, -0.18] | yes |
| s02_w03, mac_cpu against mac_mps | 70 (7) | +0.52 | [-0.69, 1.86] | [-0.49, 2.07] | no difference resolved | 4.73 | -0.05 [-0.29, 0.00] | no |
| s02_w04, cuda_a against mac_mps | 36 (4) | +4.65 | [1.20, 6.70] | [3.39, 7.71] | mac_mps closer | 11.15 | -0.12 [-0.92, 0.89] | no (line error opposite sign) |
| s02_w04, cuda_a against mac_cpu | 36 (4) | +2.79 | [-1.77, 5.18] | [2.22, 4.24] | no difference resolved | 14.94 | -0.23 [-0.92, 0.79] | no |
| s02_w04, mac_cpu against mac_mps | 36 (4) | +1.06 | [0.53, 2.56] | [0.29, 4.08] | mac_mps closer | 4.89 | +0.05 [0.00, 0.22] | yes (4 blocks, coarse) |

On all truth frames (the sensitivity row: 140, 72 and 62 frames), the readings hold on s04_w00 (no difference) and
s02_w03 (cuda_a closer than mac_mps by 4.88 and than mac_cpu by 6.35, both D-line ready), and every s02_w04 pair
becomes "no difference resolved".

**The reading, in the design's terms (section 5).**

- Court device: on s02_w03 no difference is resolved between the CPU and MPS courts; on s02_w04 the MPS court is closer
  to this truth than the CPU court by 1.06 px (D-line ready, but on 36 frames in 4 blocks, and not on all 62 truth
  frames). Mixed, so **no change**: D175 stands.
- CUDA's court: closer to this truth than both Mac courts on s02_w03, not resolved against MPS on s04_w00, and farther
  than MPS on s02_w04 (not D-line ready). Not closer on every window, so **no change**.
- A lean toward the selecting build's court: **does not show a lean, does not rule it out** (s04_w00 shows no
  difference between the MPS and CUDA courts).

The minimum honest claim (section 4): on these frames, no court output is consistently closer to camera truth than
another across the three windows; the one resolved difference that holds on both frame sets, cuda_a closer on
gsw_ind_2016_s02_w03 by 5.4 to 6.4 px, is on a window whose truth was picked and initialised through a CUDA build.

**What the measure shows about itself (report only; nothing re-read).** On almost every primary frame only one of the
three court points lies on the frame (95 of 106, 70 of 70, 34 of 36): the broadcast camera is zoomed in. So the point
gap, fixed as primary in section 2, mostly measures how the homographies extrapolate beyond the image; its medians
(38 px on s04_w00, 220 to 266 px on gsw_ind_2016) are far larger than the on-frame line error (6.7 px on s04_w00,
18.7 to 21.0 px on gsw_ind_2016). Every paired line-error difference is under 1.0 px. Separately, on the gsw_ind_2016
windows every build's court lines sit about 19 to 21 px from the R1.3 truth, against 6.7 px on s04_w00 from the Mac
truth, while both truths sit 6.3 to 7.5 px (median) from the painted key. A measure on on-frame court points, or the
line error as the deciding measure, would be a new registration; none is started here.
