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
