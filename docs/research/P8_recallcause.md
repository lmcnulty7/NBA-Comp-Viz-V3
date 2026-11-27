# P8: where the build loses the boxes of players with no box of their own (registered 2026-11-27, before any number)

Status: design only, report only. No number in this document has been computed. Nothing is fitted, tuned or adopted;
no rule changes; nothing here is a gate for an arm. Code, after review: tools/p8_recallcause.py with tests.

## 1. Question

D205 split the missed row one to one on the development standard rows (D202; D193 (2)). Two of its classes are
players with no box of their own within 5 ft: P-i (0.041 of in-frame players on gsw_cha_2016 and gsw_bkn_2015 pooled;
the nearest box is a neighbour's, at about 3.7 ft) and D-i (0.044; the nearest box is a neighbour's at about 7.5 ft).
For each such miss: did r23's raw detector have a box for the player before tracking, and if so, where in the build
was it lost? D189 and D190 asked the same of far misses on the old truths, with the E1 corpus boxes (D189: r23 raw had
a box within 5 ft for 37 percent of far misses, order-free; D190: no loss class reached 0.010 in-frame on every pool
game, the "never tracked nearby" class later traced to the tracker reset bug, P8_a3input.md, fixed since D193).

## 2. What is stored, and what is not

- **The fixed-tracker composite** (reference root builds/r23; D202's build) stores only the tracker's output: the
  sidecar's boxes, track ids and feet per processed frame, the trajectories and the identity file. It stores no raw
  detections, before or after NMS.
- **The E1 corpus** (P8 workspace, tools/p8_e1data.py) stores r23's post-NMS player boxes (conf 0.40, NMS IoU 0.45,
  CPU) on the frames of the old Mac MPS truth, run in another runtime (OpenCV 5.1.0, torch 2.15.0; the builds use
  4.13.0 and 2.2.2; D190's open point). It covers only the frames that truth accepted: for gsw_cha_2016 at most about a
  third of multipool v1's frames (D202's common-frames split: 912 of the current truth's frames were also accepted by
  the MPS truth), and it holds nothing before NMS.
- **a3trace and a3input** (P8 workspace placement/) store, for gsw_cha_2016_s05_w00 only (1,053 frames), r23's
  predict boxes per frame in the build environment and the tracker's per-frame input counts.
- Nothing stores the boxes NMS suppressed.

So the question cannot be answered from stored files on the D205 universe: classes need raw boxes on every frame of
the universe, and the NMS class needs the boxes before NMS. The design recomputes them (section 4), with its cost
stated first (section 7).

## 3. Universe and the count gate

The universe is D205's P-i and D-i misses on gsw_cha_2016 and gsw_bkn_2015 (current truth, multipool v1). D205's
per-player record (reference root one2one/<game>.json) stores each row's class but not its frame or player, so the
tool recomputes the classes on the same scored frames with tools.p8_one2one's own functions (frame_classes, through
the same collect inputs and checks), keeping each row's window, frame and SportVU player. **Count gate:** per window,
the counts of every class must equal D205's record exactly, and the record's run, hashes and gate fields must match;
otherwise nothing is reported. hou_cle_2016 gets the count gate only and no raw run; it stays unspent.

## 4. Raw detections, recomputed (frozen CV python, under the lock)

On every universe frame (a frame holding at least one P-i or D-i miss), decoded from the window clip with
videoseq.SeqReader as the build decodes it (D190's decode lesson), r23 (models/player_detector.pt, sha256 checked) on
the reference platform (CPU) through ultralytics predict with the build's settings (classes [config.PERSON_CLASS],
conf config.PLAYER_CONF 0.40, the build's image size), twice from one forward pass:
- **R_pre:** the candidates before NMS (predict with NMS IoU 1.0, so only exact duplicates fall);
- **R_post:** R_pre through ultralytics' own NMS at config.PLAYER_IOU 0.45 (the same function and arguments the build's
  predict call uses), so R_post is what the tracker receives.
Gates, before any class is counted:
- **Build-environment gate:** on gsw_cha_2016_s05_w00, R_post on every frame equals a3trace's stored predict boxes
  (same count, one to one within 1 px, scores within 0.01), and its per-frame count equals a3input's first update-call
  input (8,766 boxes over the window); the versions equal a3input's (OpenCV 4.13.0, torch 2.2.2, ultralytics 8.4.48).
- **Corpus cross-check, report only:** on universe frames the corpus also holds, the share of frames whose R_post
  equals the corpus boxes within 1 px (D190's runtime question, measured, not gated).

## 5. Classes (each miss gets the first that applies)

For a miss of player j at frame f, with the SportVU players' positions sv and the frame's H_truth: a raw box's foot is
its bottom centre through H_truth (the corpus convention of D189). Raw boxes are assigned to the frame's ten players
one to one by D205's two-stage rule (2.5 ft, then the leftovers at 5 ft), separately for R_post and for R_pre.

| class | rule | points to | could a detector training arm move it |
|---|---|---|---|
| C1, no raw detection near | j gets no R_pre box in R_pre's assignment, and no unassigned R_pre box lies within 5 ft of j | detector recall (the player is not detected at conf 0.40) | yes |
| C2, suppressed by NMS or merged | j gets an R_pre box but no R_post box: a candidate within 5 ft existed and NMS removed it (it overlapped a neighbour's kept box at IoU 0.45 or more) | NMS settings (IoU, class-aware), or boxes that overlap too much to survive NMS | partly (better-separated boxes); mainly a post-processing setting |
| C3, dropped by the tracker | j gets an R_post box, and no sidecar box matches it (one to one, IoU 0.5 or more); split: late start (a sidecar box within 2.5 ft of j appears within the next 1.0 s of processed frames) or not confirmed | tracker confirmation (activation thresholds, track buffer, new-track rules) | no |
| C4, kept but given to another | j's R_post box matches a sidecar box, but in D205's assignment that sidecar box's foot is claimed by another player or lies beyond 5 ft of j (the stabilised foot moved away, or the track carries a neighbour) | identity (a merged or switched track) or the foot stabiliser | no |
| other | none of the above (for example, j's box survives tracking and lies within 5 ft but D205's assignment gave it to another player) | the scorer's assignment | no |

## 6. Reported, and readings (fixed now)

Per class, for P-i and D-i separately, pooled over gsw_cha_2016 and gsw_bkn_2015 and per game (shown, not read): the
count, the share of the class's misses, the rate over in-frame players, and a window-cluster bootstrap interval of
the share (2,000 resamples, seed 0, windows in sorted name order). Also: C3's late-start and not-confirmed split; for
C1, the share where a box at conf 0.25 to 0.40 exists in R_pre (predicted below the build's threshold; requires the
same pass at conf 0.25, then filtered, and is report only).

Reading, on the pooled shares of P-i and D-i together: the class with the largest share is named, with its direction
from the table; a class holding at least half of the misses is "the main loss". If C1 is the main loss, the misses are
mostly not detected, and a detector training arm is the lever (its ceiling: C1's rate over in-frame players); if C2,
NMS; if C3, the tracker; if C4, identity or the foot stabiliser. Each rate is a ceiling for the lever, not a forecast.

## 7. Cost and guards

- No stored-files stage: D205's classes can only be recomputed through its collect path, which calls the pinned
  score_window, so the count gate and the universe come from the same lock job as the raw pass.
- One job under the lock "p8 recallcause": the count gate on the three games (as D205's collect, about 4 lock minutes
  for all three in D205), then the build-environment gate window (1,053 frames), then the raw pass on the universe
  frames of gsw_cha_2016 and gsw_bkn_2015. The universe frames are at most the two current truths' accepted frames
  (2,924 and 2,984, D199); r23 predict on CPU with sequential decode runs at about 0.25 to 0.35 s per frame (the build's
  0.44 s per processed frame includes the tracker and the court), so at most about 25 to 35 minutes, with the
  conf-0.25 variant from the same pass. Estimate 30 to 45 minutes in all; cap 60 minutes, checked between windows; at
  the cap it stops, reports nothing, and is not retried without a new go.
- Writes: one output path, reports/research/p8_recallcause.json, after find_private; the raw boxes go to the reference
  root's recallcause/ folder (not committed). Inputs opened read-only; a test patches the writers. Guards:
  tools.p8_common.guard and tools.research_testset.refuse_cv_test on every game, stem and SportVU id; V3_HEAVY_HELD;
  a clean `git status -- tools sportvu` with HEAD recorded; the scorecard pin compared on every field; the D202 and D205
  records pinned by SHA-256 and run id; manifests of the roots read (labels, label_build, truth_accept,
  cache/scorecard, builds/r23) before and after; identity tested on content (time maps by frames and spans).
- Nothing held out is read; hou_cle_2016 gets the count gate only.

## 8. Version 2 (2026-11-27; the review of 15cd5512, GO WITH CHANGES; before any code or number)

Where this section differs from sections 1 to 7, it rules.

**8.1 The candidates (replaces section 4's two sets).** Per universe frame, the predictor's own preprocess and
inference run once (the predictor that model.predict sets up for r23 with the build's arguments; imgsz is the
predictor's own), giving the raw prediction tensor. `assert not model.end2end` (an end-to-end head has no NMS step).
ultralytics.utils.nms.non_max_suppression is then called on that tensor with its own arguments, three times:
- **as built:** conf 0.40, iou 0.45, max_det 300, classes [0], return_idxs=True: R_post, what the tracker receives;
- **all candidates:** conf 0.40, iou 1.0, max_det 30000, classes [0], return_idxs=True: R_pre (at IoU 1.0 duplicates are
  kept, so R_pre holds every candidate at conf 0.40 or more up to its own cap);
- **other classes:** conf 0.40, iou 0.45, classes None, filtered after to the non-player classes (for C1b);
plus the as-built call at conf 0.25 for the threshold sub-share (8.4). Boxes are scaled to the frame with
ultralytics.utils.ops.scale_boxes as postprocess does. Recorded per frame: the candidate counts of each call, and
every frame where a call hit its max_det or its max_nms cap; recorded once: the NMS backend (torchvision if loaded,
else ultralytics' TorchNMS) and the versions.

**8.2 Decoding and alignment (replaces section 4's decode).** Each window is read as tools/p8_a3run.py reads it:
every stride frame in sequence from frame 0 with videoseq.SeqReader, predicting only on universe frames and on the
alignment frames. Alignment, per window: on its first five universe frames, R_post is computed at f and at the stride
frames before and after; the share of the sidecar's boxes that R_post matches one to one at IoU 0.5 must be higher at
f than at both neighbours and at least 0.9 on average at f; else the window's misses are left out and counted.
Gates before any class is counted (both must pass):
- gsw_cha_2016_s05_w00: R_post equals a3trace's stored predict boxes on every frame (same count, one to one, every
  coordinate within 0.01 px and score within 0.0001, the record's rounding), and the counts sum to a3input's 8,766;
- gsw_bkn_2015_s04_w00: R_post equals tools/p8_a3run.py's stored predict boxes (placement/a3run/gsw_bkn_2015_s04_w00.json)
  on every frame, to the same tolerance.
Both gates run the full sequence of their window with the same code path; the versions must equal a3input's.

**8.3 C2 by clusters (replaces C2's rule).** Each suppressed candidate (in R_pre, not kept in R_post) belongs to its
suppressor: the highest-score kept R_post box at IoU 0.45 or more with it. Player j is C2 when it has no R_post box of
its own and some suppressed candidate c has all three: c's foot within 2.5 ft of j; c's foot nearer j than any other
player; and c's suppressor's foot nearer another player than j. Reported for C2: the candidate's score and its IoU
with the suppressor. A duplicate of a kept box does not make j detected: a candidate whose suppressor is j's own box,
or whose suppressor's nearest player is j, does not count.

**8.4 C1 split (replaces C1).** For j with no own R_post box and not C2:
- **C1a, the foot model:** an R_pre box covers j's image position (j's SportVU position through H_truth's inverse
  lies inside the box, its bottom extended by a tenth of its height) but its foot maps beyond 5 ft of j;
- **C1b, another class:** an r23 box of another class at conf 0.40 or more has its foot within 5 ft of j;
- **C1c, a recall miss:** neither.
The threshold lever, report only: the share of C1c misses with a player-class box at conf 0.25 to 0.40 whose foot is
within 2.5 ft of j and nearer j than any other player.

**8.5 C4 split and the leftover (replaces C4 and "other").** For j with an own R_post box that matches a sidecar box
one to one at IoU 0.5: **C4a,** the sidecar box's stabilised foot is within 5 ft of j but D205's assignment gives it to
another player (identity or the scorer's pairing, not separated); **C4b,** the stabilised foot lies beyond 5 ft of j.
"Other" is a pure leftover, reported with its count. C3's late start uses the sidecar's own frame order (its processed
frames within the next 1.0 s).

**8.6 The gate on D205's rows (replaces section 3's count gate).** The tool recomputes D205's rows in D205's order
(its collect path, through tools.p8_one2one) and requires every row's class and both sensitivities to equal D205's
stored row, for every window, besides the counts; D205's records (reference root one2one/<game>.json for
gsw_cha_2016 and gsw_bkn_2015) are pinned by SHA-256 and run id. hou_cle_2016 is not recomputed (D205 gated it): it
stays unspent.

**8.7 Readings (replaces section 6's reading).** P-i and D-i are read separately, the pool of the two beside them.
A class is "the main loss" of P-i or D-i only when the window-bootstrap lower bound of its share is at least 0.5; else
the largest class is named "largest, not main". Each class rate over in-frame players is an upper bound on how far a
lever could lower missed, with the ghost cost of more boxes not measured.

**8.8 Other changes.** Section 1's D190 reference is to its finding only; the runtime question it left open was
settled for this pipeline by D192 (the bug was the tracker, not the detector's runtime). Cap 75 minutes. Writes: the
reference root's recallcause/ folder (the raw candidates per frame, not committed) and
reports/research/p8_recallcause.json; nothing else, with a test that patches the writers.

**8.9 Tool (2026-11-27, written, not run).** tools/p8_recallcause.py run (one lock job) and
tests/test_p8_recallcause.py. Two details fixed with the code: C3's late start is read on the D205 rows of player j at
the later scored frames within 1.0 s (a sidecar foot within 2.5 ft of j there); with no such frame the miss is
"C3_undetermined", reported apart. The gate windows read the composite's processed frames, which equal the old
builds' (equal frame sets, P8_resetfix.md 14.3). Tests: box matching and the stored-box equality, suppression clusters
and C2 (a duplicate of j's own box does not count), the class order, the alignment rule, the share table, groups and
the main-loss rule, and an AST check that the only writes are the record and the raw folder.

**8.10 Version 2.1 (2026-11-27; the code check of c43915b4, NO-GO for the run, the design stands; before any
number).** Where this differs from 8.1 to 8.9, it rules.
- **The NMS calls (B1).** ultralytics' non_max_suppression rewrites its input tensor in place, so each of the four
  calls gets its own fresh clone of the raw tensor. A test checks that the raw tensor is unchanged and that repeated
  calls equal a call on a fresh copy.
- **Alignment (B2, replaces 8.2's alignment).** R_post is predicted at f only and compared with the sidecar at
  f - stride, f and f + stride; the rule (higher at f than at both neighbours, at least 0.9 on average at f) is
  unchanged.
- **8.2's tolerance, worded as coded.** R_post's coordinates are rounded to 0.01 px and its score to 0.00001 (the
  stored records' rounding), and every rounded value must equal the stored one (within 1e-6), one to one.
- **C1a split (M2, replaces 8.4's C1a).** C1a_own: a covering R_pre box (as in 8.4) whose kept box, or whose
  suppressor, no player is assigned to in D205's two-stage assignment on R_post, so it is j's own box with its foot
  misplaced. C1a_other: every covering box is another player's detection (its kept box or its suppressor is
  assigned to another player). C1a_own comes first.
- **C1b (M1, replaces 8.4's C1b).** Class 1 (referee) only. Boxes of class 2, 3 and 4 (ball, rim, number) within 5 ft
  of each miss are counted and reported apart; they assign no class.
- **"Other."** By the class order every miss gets a class before "other" (a miss with no own box falls in C1 or C2; one
  with an own box in C3 or C4), so "other" is expected to be 0; any count there is reported as a code fault.
- **C3 late start (M3).** Matched on the SportVU pid of each row (mom.index.q[q]["pid"][moment]), not on the array
  position.
- **D205's records (M4, completes 8.6).** Pinned as constants: gsw_cha_2016.json SHA-256 12de9125...97005,
  gsw_bkn_2015.json 501f61d5...27747, both run 20261127T171848Z; D202's score records must carry the same run id and
  the same windows. Per window, before scoring, D205's own checks run: the sidecar, time map and labels (and the
  identity file where recorded) must hash as D202 recorded them, tools.p8_placecause.cache_gaps must find nothing,
  and the scorer's in_frame, missed, nr_boxes and nr_ghosts must equal D202's record. Any failure stops the run.
- **The row gate per window (M6, completes 8.6).** Each window's recomputed rows must equal D205's stored rows of that
  window before its raw pass; the run stops at the first window that differs. The whole-game comparison stays.
- **Readings (M5, replaces 8.7's main-loss rule).** The main loss is read on the groups C1, C2, C3 and C4, each with its
  window-bootstrap interval (2,000 resamples, seed 0); the largest group is "the main loss" of P-i or D-i only when
  its lower bound is at least 0.5, else "largest, not main". The subclasses inside each group are shown with their own
  counts, shares and intervals; no subclass is named the main loss.
- **Reads.** An undecodable frame anywhere in a window's read stops the run, as in tools/p8_a3run.py.
- **Suppressor.** A kept box suppresses a candidate only at IoU above 0.45, as NMS does (8.3 said "0.45 or more").


## 9. Result (2026-11-27; run at 7ba557d2, reports/research/p8_recallcause.json)

**9.1 The run.** One lock job ("p8 recallcause"), frozen CV python (opencv 4.13.0, torch 2.2.2, ultralytics 8.4.48;
NMS backend torchvision), 2,243 s of the 75-minute cap; per-stage times were not recorded. Before the run the B1 check
(test_nms_once_does_not_change_its_input, which pytest skips under the repository python) was run under the frozen CV
python and passed; the same check showed that a bare non_max_suppression call changes its input.
Gates, all passed:
- D205's rows: recomputed equal to the stored rows in every window, 28,427 rows (gsw_cha_2016) and 29,370
  (gsw_bkn_2015), records at the pinned SHA-256 and run 20261127T171848Z, with D205's per-window hash, cache and count
  checks;
- build environment: gsw_cha_2016_s05_w00, 1,053 frames, 0 differing, 8,766 boxes; gsw_bkn_2015_s04_w00, 1,102 frames,
  0 differing;
- no max_det or max_nms cap was hit; "other" is 0 in every unit.

**9.2 Dropped windows (registered rule 8.10).** 32 of the 120 windows that hold universe misses failed the alignment
rule; their 1,291 misses are left out (3,591 kept). By game: gsw_bkn_2015 945 of
2,745 misses (34 percent), gsw_cha_2016 346 of 2,137 (16 percent).
Why they failed, on the run's own stored shares (window means of the alignment frames):
- 20 windows (768 misses, 59 percent of the dropped misses) fail only on a tie: the share at f is 0.9
  or more and a neighbour equals it, none is higher;
- 7 (335) have the share at f under 0.9;
- 5 (188) have the share at f at 0.9 or more and a neighbour above it.
That ties are likely where boxes move little between stride frames is inference; no record measures motion.
9 dropped windows had fewer than five alignment frames (1 to 3; 6 had one), because fewer than five of their
universe frames have both neighbours processed; any version 2.2 rule fixes a minimum number of alignment frames.
Against tools/p8_a3align.py's stored verdicts, where both cover a
window (reports/research/p8_recallcause_aligndry.json, which also holds the counts above):

| game | windows with misses | dropped | misses in dropped | dropped also in a3align | of those, a3align aligned | kept | kept in a3align | of those, a3align misaligned |
|---|---|---|---|---|---|---|---|---|
| gsw_cha_2016 | 56 | 10 | 346 | 9 | 6 | 46 | 42 | 1 |
| gsw_bkn_2015 | 64 | 22 | 945 | 20 | 14 | 42 | 36 | 0 |

So 20 of the 29 dropped windows that a3align covers are windows a3align called aligned, and a3align called only one
kept window misaligned. Per window, the dropped list, its misses, its mean shares and a3align's verdict are in the dry
run record.

**9.3 Shares on the kept windows, as registered (8.10).** Group shares of the no-own-box misses, window-bootstrap 95
percent intervals (2,000 resamples, seed 0); the main loss is read on the pool of each unit:

| unit | misses (windows) | C1 | C2 | C3 | C4 | other | reading |
|---|---|---|---|---|---|---|---|
| P-i cha | 892 (46) | 0.553 [0.498, 0.621] | 0.163 [0.104, 0.217] | 0.094 [0.065, 0.126] | 0.191 [0.162, 0.225] | 0 | (read on the pool) |
| P-i bkn | 778 (39) | 0.629 [0.578, 0.675] | 0.144 [0.112, 0.178] | 0.060 [0.037, 0.086] | 0.167 [0.136, 0.207] | 0 | (read on the pool) |
| P-i pooled | 1,670 (85) | 0.588 [0.543, 0.632] | 0.154 [0.120, 0.188] | 0.078 [0.058, 0.100] | 0.180 [0.157, 0.207] | 0 | C1: main loss |
| D-i cha | 899 (44) | 0.704 [0.649, 0.753] | 0.102 [0.074, 0.134] | 0.132 [0.096, 0.172] | 0.061 [0.044, 0.079] | 0 | (read on the pool) |
| D-i bkn | 1,022 (41) | 0.789 [0.736, 0.837] | 0.070 [0.049, 0.097] | 0.087 [0.063, 0.115] | 0.054 [0.035, 0.074] | 0 | (read on the pool) |
| D-i pooled | 1,921 (85) | 0.749 [0.706, 0.788] | 0.085 [0.067, 0.107] | 0.108 [0.085, 0.134] | 0.057 [0.045, 0.072] | 0 | C1: main loss |
| both cha | 1,791 (46) | 0.629 [0.581, 0.679] | 0.132 [0.094, 0.171] | 0.113 [0.084, 0.146] | 0.126 [0.108, 0.146] | 0 | (read on the pool) |
| both bkn | 1,800 (42) | 0.719 [0.678, 0.756] | 0.102 [0.084, 0.122] | 0.076 [0.058, 0.098] | 0.103 [0.081, 0.126] | 0 | (read on the pool) |
| both pooled | 3,591 (88) | 0.674 [0.638, 0.711] | 0.117 [0.096, 0.140] | 0.094 [0.074, 0.115] | 0.114 [0.100, 0.130] | 0 | C1: main loss |

As registered, on the kept windows, C1 (no box of its own after NMS, and not a suppressed candidate of its own) is
the main loss of both P-i (0.588, lower bound 0.543) and D-i (0.749, lower bound 0.706). The subclasses, pooled (count: share
[interval]); no subclass is named the main loss:

| subclass | P-i pooled | D-i pooled | both pooled |
|---|---|---|---|
| C1a_own | 49: 0.029 [0.019, 0.041] | 56: 0.029 [0.020, 0.039] | 105: 0.029 [0.022, 0.037] |
| C1a_other | 324: 0.194 [0.169, 0.223] | 768: 0.400 [0.361, 0.436] | 1,092: 0.304 [0.275, 0.332] |
| C1b | 18: 0.011 [0.006, 0.016] | 144: 0.075 [0.045, 0.106] | 162: 0.045 [0.029, 0.063] |
| C1c | 591: 0.354 [0.315, 0.392] | 471: 0.245 [0.206, 0.289] | 1,062: 0.296 [0.263, 0.331] |
| C2 | 257: 0.154 [0.120, 0.188] | 164: 0.085 [0.067, 0.107] | 421: 0.117 [0.096, 0.140] |
| C3_late | 109: 0.065 [0.048, 0.085] | 160: 0.083 [0.063, 0.107] | 269: 0.075 [0.058, 0.093] |
| C3_not_confirmed | 14: 0.008 [0.004, 0.013] | 30: 0.016 [0.010, 0.021] | 44: 0.012 [0.009, 0.016] |
| C3_undetermined | 8: 0.005 [0.002, 0.009] | 18: 0.009 [0.005, 0.014] | 26: 0.007 [0.005, 0.010] |
| C4a | 220: 0.132 [0.113, 0.155] | 0: 0.000 [0.000, 0.000] | 220: 0.061 [0.051, 0.072] |
| C4b | 80: 0.048 [0.037, 0.060] | 110: 0.057 [0.045, 0.072] | 190: 0.053 [0.043, 0.063] |

The threshold sub-share (8.4): of the C1c misses, the share with a player box at conf 0.25 to 0.40 within 2.5 ft of j
and nearer j than any other player is 0.0085 for P-i (591 C1c), 0.0361 for D-i (471) and 0.0207 pooled (1,062).
Other-class boxes within 5 ft of a miss, counted apart (M1), pooled over both units: ball 64, rim 0, number 454. C2:
421 misses, median candidate score 0.758, median IoU with its suppressor 0.559.

**The drop and these shares.** The registered result holds on the 88 kept windows, 3,591 of 4,882 misses;
no statement here covers the full universe. The drop is not random: it is 34 percent of bkn's misses and
16 percent of cha's, and most dropped windows fail on a tie (9.2), so the kept windows may lean toward windows
where boxes move more between frames (inference). The shares may be biased by that; the direction is unknown. The
worst case for the pool of both units: if no dropped miss were C1, C1 would be 2,421 of 4,882, 0.496, a share under
0.5. P-i and D-i cannot be bounded apart from the record (it counts each dropped window's misses,
not their P-i and D-i split).

**9.4 Post hoc (not registered; descriptive).**
- C1a_other, a position inside only another player's detection, is the largest C1 subclass of D-i (0.400) and the
  second of P-i (0.194). Its geometry: j's image foot point lies inside a box (to a tenth of its height below its
  bottom) whose own foot maps more than 5 ft from j and which the assignment gives to another player, so that
  player's box spans j's position in the image; the reading is that j stands behind that player, partly or fully
  hidden, and the detector returns one box for the two. That is inference from the geometry; no per-miss occlusion
  is measured; C1c, no candidate at all near j, is 0.354 of P-i and 0.245 of D-i.
- The threshold lever is small: lowering the conf threshold to 0.25 adds a box near j for under 4 percent of C1c.
- C4a is 0 in D-i, as the class definitions require (a D-i miss has no box foot within 5 ft).
- C1a_own, j's own box with its foot mapped beyond 5 ft, is 0.029 in both units.
Whether these hold on the dropped windows is open (9.5).

**9.5 Proposed alignment rule, version 2.2 (2026-11-27; a proposal for the lead's decision after ml-review; not
registered, nothing rerun).**
- Level of a pair of frames: Hungarian pairs between R_post(f) and the sidecar boxes at f' on IoU, the mean IoU over
  max(len R_post(f), len sidecar(f')) with every unmatched box counted as 0. The window's level at f' is its mean over
  the alignment frames (the first five universe frames with both neighbours processed, as now).
- Misaligned only when the level at f - stride or at f + stride beats the level at f by more than 0.05 (a3align's
  margin), or the level at f is under 0.75; ties read aligned.
- It needs no detector run: R_post at every universe frame and the raw candidates of every window, the dropped ones
  included, are stored in the reference root's recallcause/ folder, and the sidecar boxes are stored.
Dry run (tools/p8_recallcause_aligndry.py; post hoc). Neither record stores the mean IoU, so the dry run checks the
margin, floor and tie logic on stored shares of boxes matched at IoU 0.5, not the statistic:
Limits of this dry run: on p8_a3align.json it is partly circular (a3align's own thresholds and statistic decide
which windows count as aligned), and a3align's verdicts are not ground truth for this run's decode; the 0.75 floor
was applied to shares, not to the mean IoU.
- on p8_a3align.json (a3align's paired S_k against S_0 at offsets -1 and +1, its S_0 at offset 0), pool games only:
  of the windows a3align calls aligned, the registered rule fails 12 of 49 (cha, 7 with a tie) and 8 of 54 (bkn, 4
  with a tie); the proposal fails 3 of 49 and 1 of 54. Of the windows a3align calls misaligned, the registered rule
  flags 4 of 4 (cha) and 8 of 8 (bkn); the proposal flags 4 of 4 and 6 of 8 (a3align's misaligned set includes
  sub-stride suspects, which no offset rule can see);
- on this run's own stored shares, with the proposal's thresholds applied to the shares: it would keep 23 of the 32
  dropped windows (6 cha with 165 misses, 17 bkn with 726) and drop none of the kept ones; the mean IoU could keep
  fewer.
If version 2.2 is adopted, its result is reported beside 9.3, which stands as the registered result.

**9.6 Plan (2026-11-27; not registered).** Posefoot's stage A will give an independent alignment verdict for all 156
windows under its own registered rule. A supplement may then score the 32 dropped windows with those verdicts,
reported beside 9.3; the decision is the lead's after stage A.
