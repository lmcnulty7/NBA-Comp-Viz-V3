# P8 recall-cause supplement: D208's 32 dropped windows read the build's way (design, version 1, 2026-11-27)

Report only. Registered before any number of this item. It is the supplement D208 9.6 and P8_recallcause.md 9.6
planned: reported beside P8_recallcause.md 9.3, never replacing it. Nothing is fitted, tuned or adopted; hou_cle_2016
stays unspent and gsw_sac_2015 apart (the allow-list {gsw_cha_2016, gsw_bkn_2015} of P8_nmsarm.md 12.5 is asserted on
every game and window name, besides tools.p8_common.guard and refuse_cv_test).

## 1. Question

D208 classified the no-own-box misses (D205's P-i and D-i) of 88 windows. Its alignment rule dropped 32 windows (1,291
misses, 20 of them on ties; P8_recallcause.md 9.2). D208 read the clips the a3run way: frame 0 and every stride frame in
sequence, with no seek. Posefoot's stage A (P8_posefoot.md 14.2 to 14.5, branch r-p8-posefoot) then showed that the
build's own read path (tools/p8_posefoot.py build_read: the CAP_FFMPEG open with its fallback, the pre-gate pass and
its intervals, videoseq.SeqReader with its seeks) reproduces 151 of the 156 windows, with none misaligned, and that the
windows a3align called misaligned read aligned under it. So D208's 32 may have been misread, not only over-dropped on
ties. What are their misses' D208 classes when the raw candidates are read the build's way?

What the stored records already show (stage A's per-window records, reference root posefoot/alignment/<game>.json, from
105d8f92): none of the 32 is among stage A's five undetermined windows, and stage A misaligned none, so stage A calls all
32 aligned. In the build's read, 21 of the 32 start their tracking reads with a seek past frame 0 (the a3run read never
seeks); so do some of D208's kept windows.

## 2. What runs

Per window, under one lock job ("p8 recallsupp", frozen CV python, V3_P8_ROOT the reference root, HF_HUB_OFFLINE=1 and
PYTORCH_ENABLE_MPS_FALLBACK=1 as stage A requires):
1. D205's rows recomputed in order (tools.p8_recallcause.d205_rows, with D205's hash, cache and count checks) and checked
   equal to D205's stored rows of the window; D205's and D202's records pinned as in D208 (SHA-256 and run
   20261127T171848Z).
2. The frames read with build_read (the gate device from the window's build record, gate v2 as stage A loads it, the
   build's max_frames), predicting on the universe frames only. build_read's read validation must pass (every sidecar
   frame read); a window that fails it is undetermined, left out and counted.
3. On each universe frame, D208's capture unchanged (tools.p8_recallcause.Raw: the predictor's preprocess and inference,
   then the four NMS calls, each on its own copy of the tensor): R_post, R_pre, other classes, the conf 0.25 boxes.
4. Each miss classified by D208's registered classes and order (P8_recallcause.md 8.3 to 8.5 and 8.10: C1a_own,
   C1a_other, C1b, C1c, C2, the three C3 classes, C4a, C4b, other), with D208's functions imported, not copied.
5. Raw candidates written to the reference root's recallsupp/<window>.json; recallcause/ is never written.

The capture code is D208's with build_read in place of its read_window. The code imports build_read from
tools/p8_posefoot.py, so it is written after r-p8-posefoot reaches v4-research (or after the lead says to copy the three
read functions with credit and a test that they equal posefoot's).

## 3. Gates, before any class is counted

- **G1, the read check on kept windows:** on every stored frame of two of D208's kept windows, build_read's R_pre and
  R_post equal the stored ones in recallcause/ exactly (same count, one to one, every value equal):
  gsw_cha_2016_s05_w00 (tracking reads start at frame 0; 111 stored frames) and gsw_bkn_2015_s02_w01 (the build seeks to
  frame 300 first; 36 stored frames). Both must pass. A failure stops the item: the two reads then disagree on a window
  D208 kept, which would put D208's kept windows in question too, and the lead decides.
- **G2:** D205's row gate passes on each of the 32 windows, and "other" is 0.
- Recorded, not gated: build_read's backend, seeks and read validation per window; D208's own alignment shares (R_post
  at f against the sidecar at f - stride, f and f + stride) recomputed on the build's read, with D208's strict verdict
  beside, to show whether the read or the rule dropped each window.

## 4. Reported (fixed now)

(1) The 32 windows' misses by D208 class and group (C1 to C4, other), P-i and D-i separately and pooled, each with its
window-bootstrap interval (2,000 resamples, seed 0, tools.p8_resetfix.resample_picks over the sorted windows), the
threshold sub-share of C1c and the other-class counts, as D208 reports them. Beside 9.3.

(2) Per window: stage A's verdict (all 32 aligned, section 1), build_read's validation, its seeks, and D208's strict
alignment verdict on the build's read.

(3) **The supplement's reading:** D208's main-loss rule (P8_recallcause.md 8.10: the largest of C1 to C4 is "the main
loss" when its window-bootstrap lower bound is at least 0.5, else "largest, not main"), applied to P-i and D-i on the
88 kept windows plus the 32 (all of which stage A aligned), 120 windows in all, with the 88-window classes taken from
D208's run as recorded and the 32's from this run. Its pooled C1 share is reported with its interval. This reading
covers all misses of the 120 windows; it sits beside 9.3, which stands as D208's registered result on 88 windows.
Windows undetermined under G2 or the read validation are left out of (3) and counted.

D208's per-miss classes on the 88 windows were not stored (its record holds the tables), so (3) pools at the level of
counts per window: the 88 windows' class counts are recomputed from recallcause/ with D208's code, and must equal D208's
recorded tables per unit before they are pooled (a gate, G3). This uses no detector.

(4) The 32 alone against the 88 (group shares side by side), read descriptively.

## 5. Limits

- The 32 were dropped for a reason that may correlate with the scene (20 failed on ties); their classes describe those
  windows, and pooling assumes nothing about why they were dropped.
- build_read reruns the pre-gate gate on the same device type; a borderline gate score could move an interval edge
  (P8_posefoot.md 14.2). The read validation catches a lost sidecar frame, not an extra read in a gap; G1 is the check
  on the detections.
- The truth-bias limit of D202 holds as in D208.

## 6. Cost

Detector predict and NMS capture on the 32 windows' universe frames (989 frames, D208 1,291 misses), the read path's
decode and pre-gate pass on those windows, G1's 147 stored frames, and D205's rows: about 15 to 20 lock minutes (D208 ran
3,631 frames in 2,243 s; stage A read 156 windows in 1,236 s). Cap 40 minutes; G3 reads stored files only. 0 Colab
units. hou_cle_2016 unspent.

## 7. Writes

The reference root's recallsupp/ folder (raw candidates, not committed) and reports/research/p8_recallcause_supp.json,
checked with find_private; nothing else (an AST test checks the targets). builds/, labels/, label_build/, truth_accept/,
cache/ and recallcause/ are watched by manifest before and after; any change stops the run with nothing reported.

## 8. Code note (2026-11-27; written while the design is in review; nothing run)

- The read functions (SeekCount, pregate_intervals, main_reads, open_like_build, build_read) are copied from
  tools/p8_posefoot.py at 53c44192 with credit, by the lead's decision (posefoot reaches v4-research only when its item
  closes). A test checks that each equals posefoot's at that commit, docstrings aside.
- D208 classified misses inline in its run loop. tools/p8_recallsupp.py reproduces that loop body as one function,
  classify_window, importing every rule it calls from tools/p8_recallcause.py. G3 is its check: on the 88 kept windows
  it must give D208's recorded counts for every class, unit and game.
- The stored candidates carry no anchor ids, so R_post's boxes are matched into R_pre by exact equality (both runs use
  the same rule), as P8_nmsarm.md section 3 does.
- The allow-list is implemented in the tool itself, since tools/p8_nmsarm.py is not on v4-research.
