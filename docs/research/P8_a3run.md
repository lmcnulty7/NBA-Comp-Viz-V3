# P8 build loss: r23 in the build's runtime, detector against tracker (registered before any run)

Written 2026-11-26 after P8_a3align.md. That check put 98.8 percent of reachable a3 in aligned windows, so a3 is not a
frame mismatch. The buildloss review then ran its own unregistered spot check, which is now seen and named as a
forking path:

- At IoU 0.7, the corpus "ours" boxes match the r23 CPU sidecar best at offset 0.
- In gsw_bkn_2015_s04_w00, 99.2 percent of sidecar boxes match an "ours" box, but "ours" averages 9.46 boxes a frame
  against the sidecar's 5.04.
- The corpus ran in a different runtime from the builds: OpenCV 5.1.0 and torch 2.15.0, against OpenCV 4.13.0 and
  torch 2.2.2.

The open question is therefore detector runtime against tracker.

This is a lock job, because it runs model code. It is report only: nothing is built, trained or changed, and no unit
is spent. Tool: tools/p8_a3run.py, with the same rules in its docstring. Outputs go to the P8 workspace; the record
is reports/research/p8_a3run.json.

## Windows (fixed now)

- **The a3-heavy windows of P8_trackreset.md:** gsw_bkn_2015_s04_w00; gsw_cha_2016_s05_w00 and s05_w03;
  gsw_sac_2015_s01_w00, s09_w01 and s07_w00.
- **One control per game:** among the game's scored windows that P8_a3align.md found aligned and that are not heavy,
  the one with the most corpus frames. That gives gsw_bkn_2015_s07_w00, gsw_cha_2016_s07_w00 and gsw_sac_2015_s04_w00.

## The run (each window, in the build's runtime)

- **Runtime.** The anaconda interpreter the builds use (ultralytics 8.4.48, torch 2.2.2, OpenCV 4.13.0), with the
  detector on CPU.
- **Weights.** The main checkout's models/player_detector.pt, with its sha256 checked as r23's.
  config.PLAYER_DETECTOR_WEIGHTS is set to it, as the E1R build child does.
- **Frames.** The window clip is read with videoseq.SeqReader in the build's order. First comes the pre-gate pass:
  every config.pregate_params(fps) stride frame from 0 to the end, read for the decoder state only. Then come the
  reference-platform r23 sidecar's processed frames, in order.
- **(1) Predict.** r23 model.predict on each processed frame: player class, confidence 0.40, NMS 0.45, CPU, using its
  own YOLO instance.
- **(2) Track.** detect.tracker.PlayerTracker(device="cpu").update on the same frames in order, exactly as
  build_trajectories.py calls it, with camera-cut resets and exception handling included. After each update, the
  tool also records the BoT-SORT tracker's unconfirmed tracks (tracked but not yet activated) and its count of lost
  tracks.

## Reading rule (fixed now)

- **Reproduction check.** Per window, the share of processed frames where (2) equals the sidecar: the same box count,
  and every box matched one to one within 2 px. The tracker test is read only for windows at 0.95 or more; any other
  window is reported as not reproduced.
- **Runtime test,** on corpus frames. (1)'s boxes per frame are compared with the corpus "ours" boxes, and the tool
  counts the share of "ours" boxes that a (1) box matches one to one at IoU 0.7 or more. The runtime differs, over the
  heavy windows pooled, when (1) has fewer than 0.9 times the "ours" boxes or matches under 0.9 of them.
- **Tracker test,** on all processed frames.
  - Yield is (2)'s boxes over (1)'s boxes.
  - A (1) box is dropped when no (2) box reaches IoU 0.5 with it. The tool reports the share of dropped boxes that an
    unconfirmed track covers at IoU 0.5 or more.
  - The tracker explains the loss when the heavy windows' pooled yield is at most 0.80 and the controls' at least
    0.95, both on reproduced windows.
- **Readings.**
  - Tracker explains and the runtime does not: a tracker arm follows (activation or confirmation). It needs its own
    registration, then a rebuild under the lock, scored by the registered scorer.
  - The runtime differs and the tracker does not explain it: a3 is mainly a runtime difference between the corpus and
    the builds. Every corpus-based raw and teacher analysis carries it, and no build arm follows.
  - Both: both are reported with their shares.
  - Neither: a3 is unexplained.

Why the thresholds:

- 0.9 is a tenth of the boxes, the size of loss that would matter for a3.
- 0.80 and 0.95 separate the heavy windows' known low yield from the controls' known yield of about 1.0
  (P8_trackreset.md). Here they are measured against the build-runtime predict, not against the corpus.

## Ladder and cost

- **Smoke first.** About 2 lock minutes: 20 frames of gsw_bkn_2015_s04_w00 (`run --window ... --frames 20`). Its
  gate: the run completes, the weights hash and versions are recorded, (2) reproduces the sidecar on those frames, and
  the unconfirmed and lost fields are filled. If any of these fails, nothing more runs and the failure is reported.
- **The full run.** The 9 windows hold 6,937 processed frames. At about 0.7 to 0.9 s a frame for the two passes, plus
  model loading and the pre-gate decode pass, it takes about 1.5 to 2 lock hours.
  - Run one window per lock acquisition.
  - Then run `report` in the system Python, with no lock.
  - No units.

Commands, run from the worktree root, one window per acquisition. <frozen CV python> is the anaconda interpreter the
builds use, as in tools/p8_frozen.py:

    python3 -m export.heavy --name "p8 a3run smoke" -- <frozen CV python> -m tools.p8_a3run run --window gsw_bkn_2015_s04_w00 --frames 20
    python3 -m export.heavy --name "p8 a3run <window>" -- <frozen CV python> -m tools.p8_a3run run --window <window>
    /usr/local/bin/python3 -m tools.p8_a3run report

## Forking paths

- a3, its heavy windows, their low yield (P8_trackreset.md) and their alignment (P8_a3align.md) are already seen.
- The review's spot check is already seen: "ours" holds about twice the sidecar's boxes on s04_w00, and the runtimes
  differ.
- The controls were picked by a rule from the alignment record, not by outcome.

## Results

Not run yet. The job waits for the heavy-job lock, which the gsw_cha_2016 truth rebuild holds.
