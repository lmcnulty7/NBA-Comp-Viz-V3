# P8: which court output is closest to camera truth? Design before any number

Written 2026-11-26 (UTC), before any number, for queue item 3 after D179. Report only; JSON only (no model, no video
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

Reading of a pair, fixed now: A is **closer** than B when the median paired difference is at most -1.0 px and its
frame-bootstrap interval lies below zero; B is closer in the mirror case; otherwise **no difference resolved**. 1.0 px
is about a quarter of the smallest build-to-build court gap (3.8 px, Mac CPU against MPS), so a smaller difference
would not separate devices that are themselves 4 to 10 px apart.

## 3. Windows, builds and frames (fixed now)

**gsw_bkn_2015_s04_w00 first.** Camera truth: the P8 workspace's labels/gsw_bkn_2015/gsw_bkn_2015_s04_w00.jsonl (140
frames; the gsw_bkn_2015 fit, reprojection 6.52 px), on the same clip every build read. Builds: Mac MPS (the
registered r23 build, builds/r23), Mac detector-CPU (M6, builds/e1m6_r23_detcpu) and cuda_a (the CUDA check's result).
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
truth (the broken fit, D98) is not used. Builds: Mac MPS, Mac CPU, Mac detector-CPU (s02_w03 only;
tools.p8_truth_v2.DEVICE_BUILDS) and cuda_a. Here the court ran on three devices (Mac CPU, MPS, CUDA), so these two
windows carry the Mac CPU against MPS question.

The R1.3 clips were cut on Colab, so frame indices could differ from the Mac clips the builds read. Alignment check,
fixed now and run first: each R1.3 truth row carries the boxes of the A100 build at that frame. For offsets of -1, 0
and +1 build steps (6 frames at this window's stride), the share of truth-row boxes that have a box with IoU at least
0.9 in the Mac CPU build at frame + offset. The window is **aligned** when the share at offset 0 is at least 0.9 and
higher than at both neighbours; otherwise it is skipped and recorded. (Seen while writing this design, one row only:
frame 24 of s02_w03 carries the box [929, 351, 1024, 552], exactly the Mac CPU build's box at frame 24. The check
decides, not this.)

## 4. The selection caveat

Camera truth is fit to SportVU positions and the painted lines, so it does not copy any build's court output. But the
frames that reach it were chosen through one build's projected feet: for gsw_bkn_2015, the V3 label build on MPS (the
court model there is V3, not the r23 court of the builds compared); for gsw_ind_2016, the A100 build (court on CUDA).
Selection can favour the court output of the selecting build. The reading in section 5 treats a pattern that follows
the selecting platform as a bias signal, not as accuracy. Accepted frames only; three windows on two games is a
description, not a platform ranking.

## 5. What each outcome would change

- **Court device for reference builds (gsw_ind_2016 windows, Mac CPU against Mac MPS).** If CPU's court is closer on
  both aligned windows: propose (a D line, the lead's) running the court model on CPU in reference builds too, at about
  twice the build time (full CPU 142 s against detector-only CPU 72 s on s02_w03, P8_PLATFORM.md). If MPS is closer on
  both: D175 stands as is. Otherwise (mixed, or no difference resolved): no change; the court device's effect is below
  what this truth resolves.
- **CUDA's court.** If CUDA is closer than the Mac outputs: the split route of D179 (2) loses its point (CUDA's court
  would be the better one), and the Colab platform question reopens for the court only. If farther: it confirms D179 (1).
- **Is the existing truth biased toward the MPS court?** The bias signal is: on gsw_bkn_2015_s04_w00 (truth selected
  through an MPS build) MPS is closer than CUDA, while on the gsw_ind_2016 windows (truth selected through a CUDA build)
  CUDA is closer than MPS. That pattern would say truth leans toward the court of the build that selected its frames:
  the Mac truth of the GSW games and hou_cle_2016 would carry an MPS-court lean, which adds weight to the CPU truth
  rebuild queued under P8_truth_arenas.md section 0. Any other pattern is no evidence of that bias.
- Nothing is adopted or rescored here; every change above is a separate decision.

## 6. Output and guards

Output: <P8 root>/court_truth/result.json, holding the alignment checks, per-build and per-pair rows of both frame
sets, both intervals and the readings, with no local path. Guards: refuse_cv_test and tools.p8_common.guard on both
games and every window before any file is read. It reads JSON only, so no heavy-job lock is needed.

## 7. Command (after review; not run yet)

    <frozen CV python> -m tools.p8_court_truth run --cuda-result <P8 data folder>/cuda_check/result

The frozen CV Python is used because `sportvu.bench.line_error_px` imports the frozen court code and OpenCV. Every input
named in section 3 was checked to exist (file existence only, no number read) before this design was sent to review.
