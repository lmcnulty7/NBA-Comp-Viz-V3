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

## Amendment 1 (2026-11-26, registered before rerunning)

The smoke passed its gate, with 20 of 20 frames reproduced. The 9-window chain then completed gsw_bkn_2015_s04_w00
(662 s). gsw_cha_2016_s05_w00 failed all three attempts with "frame 6888 unreadable", at about 11 minutes each, and
the chain stopped there as designed. That window has a 444-frame gap between processed frames (6090 to 6534). The
registered read order turned the gap into a seek, and in this clip the seek does not land where the build's reads
did.

The fix:

- Before each processed frame, if the decoder would jump more than SEEK_MIN_GAP (300) frames, every stride frame since
  the previous processed frame is read first. No seek then happens inside a window.
- The build reads every stride frame inside its pre-gate intervals, gate-rejected frames included. A long gap between
  processed frames is therefore most likely read sequentially by the build as well.
- Only gsw_cha_2016_s05_w00 and gsw_cha_2016_s07_w00 (gap 4044 to 4452) have such gaps. For the other seven windows
  the order is unchanged, so gsw_bkn_2015_s04_w00's completed run stands. Its result has not been read.
- The registered reproduction check decides whether each replay matches the build.
- A window that still fails after its retries is recorded as missing, and the pooled tests use the windows that ran.

## Results, first full report (2026-11-26, read order of amendment 1)

The chain ended at 08:54. Eight of the nine windows have records. gsw_sac_2015_s04_w00 has none: it failed
all three attempts, and the chain output that would name its error was not kept (only the chain's tail was saved, a
logging slip). The report ran and recorded that window as missing, as amendment 1 provides. Record:
reports/research/p8_a3run_amend1.json, a copy of the report as written by this run.

| Window | Reproduced | (1) predict boxes | (2) track boxes | Yield | Dropped under an unconfirmed track | (1) against "ours" on corpus frames |
| --- | --- | --- | --- | --- | --- | --- |
| bkn s04_w00 (heavy) | 100% | 8,939 | 4,707 | 0.53 | 138 of 4,228 | 1,319 of 1,325 matched; 1,327 against 1,325 boxes |
| cha s05_w00 (heavy) | 100% | 8,766 | 2,657 | 0.30 | 3 of 6,103 | 84 of 85 |
| cha s05_w03 (heavy) | 100% | 1,689 | 387 | 0.23 | 4 of 1,302 | 20 of 20 |
| sac s01_w00 (heavy) | 100% | 5,370 | 1,696 | 0.32 | 37 of 3,671 | 152 of 152 |
| sac s09_w01 (heavy) | 100% | 3,701 | 1,763 | 0.48 | 40 of 1,937 | 155 of 157 |
| sac s07_w00 (heavy) | 100% | 8,851 | 7,262 | 0.82 | 196 of 1,585 | 1,686 of 1,695 |
| bkn s07_w00 (control) | 0.1% | 7,757 | 5,117 | 0.66 | 133 of 2,636 | 1,351 of 1,562 |
| cha s07_w00 (control) | 0.0% | 8,557 | 8,211 | 0.96 | 345 of 345 | 1,345 of 1,523 |
| sac s04_w00 (control) | missing | | | | | |

Heavy windows pooled:

- yield 0.495;
- 2.2 percent of dropped boxes lie under an unconfirmed track;
- (1) holds 0.9997 times the "ours" boxes and matches 0.995 of them at IoU 0.7.

### Reading by the registered rule

- Runtime: it does not differ. On the heavy windows the build runtime's detector reproduces the corpus runtime's boxes
  (0.9997 of the count, 0.995 matched).
- Tracker: not readable. The rule needs the controls reproduced at a yield of 0.95 or more, and no control reproduced.
- So the rule reads "neither". That reading comes from the controls failing their own reproduction check, not from
  the tracker test.

What the heavy windows show, all six reproduced exactly:

- The build's tracker outputs about half of what its detector finds there (yield 0.23 to 0.82, pooled 0.50).
- The dropped boxes are almost never unconfirmed tracks (2.2 percent).
- With the runtime ruled out, the a3 loss in these windows sits in the tracker. The "never confirmed" mechanism is not
  supported: 98 percent of dropped boxes are not unconfirmed tracks at the end of their frame. Which tracker step drops
  them is not measured.

Why the controls failed: every heavy window starts at processed frame 0, and every control starts later (318, 234,
162). After the pre-gate pass, the replay sought straight to the first processed frame. The build sought to its
pre-gate interval's start and read on from there. The two decoders therefore started differently. This also explains
the controls' lower match against "ours" (0.87): the replay's frames there are not the corpus's frames either.

## Amendment 2 (registered before rerunning the controls)

When the first processed frame is not 0, the replay seeks to frame 0, where a seek lands exactly. It then reads every
stride frame up to the first processed frame, with no other seek. The three controls are run again in this order.
Their first outputs are kept in the P8 workspace as <window>__amend1.json, and the first report as
reports/research/p8_a3run_amend1.json. The six heavy windows are unchanged, because they start at 0, where both orders
agree. The reproduction check and every rule stay as registered.

## Results after amendment 2 (computed 2026-11-26)

The three controls ran again with the sequential start from frame 0, first attempt each, in 602, 628 and 393 s of
lock. gsw_sac_2015_s04_w00 now ran (it had failed before). The report was rewritten. Record:
reports/research/p8_a3run.json; the first report stays as p8_a3run_amend1.json.

| Control | Reproduced | (1) predict | (2) track | Yield | (1) against "ours" (matched) |
| --- | --- | --- | --- | --- | --- |
| gsw_bkn_2015_s07_w00 | 0.1% | 7,773 | 5,094 | 0.66 | 1,555 of 1,562 |
| gsw_cha_2016_s07_w00 | 0.0% | 8,532 | 6,945 | 0.81 | 1,519 of 1,523 |
| gsw_sac_2015_s04_w00 | 2.3% | 5,232 | 4,989 | 0.95 | 1,869 of 1,879 |

The heavy windows are unchanged: reproduced 100 percent, yield 0.495, and runtime matched 0.995.

### Reading by the registered rule (final)

- Runtime: it does not differ, on the heavy windows (matched 0.995) or now on the controls (0.996).
- Tracker: still not readable. No control reproduces the sidecar.
- So the rule reads "neither".

What changed with amendment 2: the controls' predict now matches the corpus "ours" boxes (0.996, up from 0.87), so the
replay now decodes the frames the corpus decoded. The sidecars still differ from both. In these three windows the
build's own decode therefore differs from a sequential decode of the clip. The likely cause is the build's seek to its
pre-gate interval start, which lands off by a few frames; this is not tested. It fits the 14-window decode offset of
P8_a3align.md, now seen on the build side.

The heavy-window result stands on its own: those replays reproduce the build exactly, and the tracker outputs about
half of the detector's boxes there. P8_a3trace.md locates where.

## Review amendments (2026-11-26, after the ml-review and the a3input result; the registered text above is unchanged)

1. **"Why the controls failed" (first full report) is a hypothesis.** It is the hypothesis amendment 2 tested: that
   the replay's direct seek to the first processed frame made the controls fail. Outcome: with a sequential start from
   frame 0, the controls' predict match against the corpus rose from 0.87 to 0.996. Reproduction of the sidecars
   stayed at 0.1, 0.0 and 2.3 percent. So the replay now decodes the corpus's frames, but the build's own decode
   still differs in those windows, and the seek hypothesis explains the corpus match, not the reproduction failure.
2. **The end-of-frame claims are qualified.** Two statements in "What the heavy windows show" rest on the tracker's
   state at the end of a frame: "The dropped boxes are almost never unconfirmed tracks (2.2 percent)" and "The 'never
   confirmed' mechanism is not supported". After a reset that state follows two update calls per frame, not one
   (docs/research/P8_a3input.md, cause confirmed). The second call removes the new tracks the first one created.
   "Not supported" is withdrawn: the never-confirmed mechanism is what happens, through the double update. See the
   amendment in P8_a3trace.md.

