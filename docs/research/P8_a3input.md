# P8 build loss: the tracker runs twice per frame after a reset (registered before the lock run)

Written 2026-11-26 after P8_a3trace.md. This document reports a code reading and stored-data check (free, done
first), then registers a short lock run that confirms the cause on one window. Report only: nothing is built or
changed, and no frozen or installed file is edited. Tool: tools/p8_a3input.py, with the same rules in its docstring.
Record: reports/research/p8_a3input.json.

## The free step: what PlayerTracker passes, against the predict call

The static arguments, as the lead asked:

- **The frame.** Both calls receive the same numpy array from SeqReader: BGR, uint8, full 1280 by 720. PlayerTracker
  makes no copy, resize or crop.
- **Arguments.** Both pass classes [0], conf 0.40, iou 0.45, device cpu and verbose false, and both use the
  checkpoint's own imgsz, half, augment, max_det, agnostic_nms and vid_stride. Model.predict's defaults are the same
  for both (conf 0.25, batch 1, rect true).
- **What Model.track adds:** mode "track", batch 1, conf (kept at 0.40), and the tracker callbacks. Nothing in
  DetectionPredictor depends on mode.

There is no colour-order or input-size difference. The difference is in the callbacks, and it is a bug:

- detect/tracker.py PlayerTracker.reset() sets self.model.predictor = None. It does this on a camera cut, and on a
  tracker exception.
- ultralytics 8.4.48 engine/model.py Model.track runs `if not hasattr(self.predictor, "trackers"):
  register_tracker(self, persist)`. With the predictor gone, the next call registers the tracker callbacks again.
  Model.add_callback appends, never replaces, and the new predictor takes the model's callback lists.
- So after the first reset, trackers/track.py on_predict_postprocess_end runs twice per frame.
  - The first call updates the tracker with every detection, then replaces the frame's results with the tracked boxes.
  - The second call updates the same tracker again, now with only those tracked boxes.
  - A new detection becomes an unconfirmed track in the first call. In the second call it finds no match and is
    removed.
  - Each further reset adds one more call.
- The stored trace already shows this. In P8_a3trace.md's record for gsw_cha_2016_s05_w00, the tracker's frame_id
  rises by 2 per processed frame on 1,038 of 1,052 steps, after the window's one reset.
- This fits every earlier finding:
  - only tracks that existed at the reset survive (the fresh tracker's first frame activates everything);
  - new players are never confirmed;
  - track ids are few;
  - almost nothing is left unconfirmed at the end of a frame;
  - the loss depends on the window, through when its reset happens and how the scene changes afterwards.

## A correction to P8_a3trace.md

That trace captured only the last BYTETracker.update call of each frame. Its class G, "not in the tracker's input", is
the second call's input (the already-reduced tracked boxes), not the detector's output. Its step "G" is an artefact of
the doubled call; the detections did reach the tracker in the first call. That branch is under review, so the
correction is recorded here and will be applied there at merge.

## The lock check (registered now)

- **Window:** gsw_cha_2016_s05_w00, in the build's runtime with the detector on CPU, using P8_a3run.md's read order
  with both amendments.
- **Variant A, as built:** PlayerTracker unchanged.
- **Variant B, reset fixed in process:** after PlayerTracker.reset, the tracker callbacks are removed from the model's
  callback lists. Model.track then registers them exactly once again.
- **Recorded per frame and variant:** every BYTETracker.update call (its input count and frame id), and the output
  boxes.

## Reading rule (fixed now)

- **Reproduction.** A must reproduce the sidecar on at least 95 percent of frames (the same box count, every box
  within 2 px). Otherwise nothing is read.
- **Doubling confirmed.** In A, at least 95 percent of frames after the first reset hold two or more update calls,
  and in B no frame holds more than one.
- **Yield.** The detector's output per frame is the first update call's input in B. Yield is the output boxes over
  that, for each variant.
- **Cause confirmed** when doubling is confirmed, B's yield is at least 0.90 and A's is at most 0.50. Then a build arm
  follows: the reset fix as a tools-side patch, with no frozen file edited, in the same way as P8_MPS_CLIP_FIX. It is
  registered on its own and rebuilt on the reference platform under the lock, then scored by the registered scorer.
  Every build that had a reset carries this bug, including r23's registered baselines. Paired comparisons such as E1
  against r23 carried it on both sides.
- **Doubling confirmed but B under 0.90:** the doubling is part of the cause, not all of it. This is reported.
- **Doubling not confirmed:** the code reading is wrong. This is reported, and nothing follows.

## Cost

- Two tracker passes over 1,053 frames, at about 0.6 s a pass a frame, plus the read passes: about 20 to 25 lock
  minutes.
- No separate predict pass is needed: B's first update call gives the detector's output.
- The run queues behind the truth-cpu driver. The lead pauses the driver for it.

Commands, where <frozen CV python> is the anaconda interpreter the builds use:

    python3 -m export.heavy --name "p8 a3input" -- <frozen CV python> -m tools.p8_a3input run
    /usr/local/bin/python3 -m tools.p8_a3input report

## Forking path

The yield, drop and trace results (P8_a3run.md, P8_a3drop.md, P8_a3trace.md) are all seen, and so is the stored
frame_id doubling. This run confirms a mechanism found by reading the code. It is not a blind test.

## Results

Not run yet.
