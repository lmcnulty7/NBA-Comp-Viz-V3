# P8 build loss: an instrumented replay of the tracker on one a3-heavy window (registered before any run)

Written 2026-11-26 after P8_a3drop.md. On the six a3-heavy windows, 62 percent of the build tracker's dropped detections
are high-confidence detections of real players that never become tracks nearby. Yet the replay records show that they
are not unconfirmed tracks at the end of their frame either, so the step that loses them is not known.

This run traces every step on one window. It is a lock job (model code). It is report only: nothing is built or
changed, and no frozen or installed file is edited. Instrumentation works only by wrapping methods inside the running
process. Tool: tools/p8_a3trace.py, with the same rules in its docstring. Record: reports/research/p8_a3trace.json.

## Filters between model.track and the sidecar (the free step, done first)

I read every filter between model.track's output and the sidecar. There is none that could drop boxes:

- **ultralytics trackers/track.py, on_predict_postprocess_end.** The tracker receives result.boxes. The class filter
  (classes=[0]) was already applied by NMS. Its output rows carry the matched detection's index. The results are then
  re-indexed to those rows and their boxes replaced by the track boxes. There is no area, aspect, court or count
  filter. The tracker is built with frame_rate 30, so track_buffer 30 means 30 processed frames.
- **detect/tracker.py, PlayerTracker.update.** It calls model.track with classes=[PERSON_CLASS], conf 0.40, iou 0.45
  and persist. _parse keeps every box that has an id. A LinAlgError, or the re-identification AttributeError, resets
  the tracker and returns no tracks. A camera cut (80 percent of 4 or more ids gone) resets the tracker after the frame.
  There is no other filter.
- **build_trajectories.py.** Every track PlayerTracker returns goes into _frame_record. The foot stabiliser changes
  feet, not boxes. There is no box filter.

So no wrapper filter explains the dropped boxes, and there was nothing to measure on the replay records. The
instrumented run checks this again (class F below).

## The run

- **Window:** gsw_cha_2016_s05_w00, where 6,103 of 8,766 detections are dropped.
- **Runtime and frames:** the build's runtime (anaconda, ultralytics 8.4.48, torch 2.2.2, OpenCV 4.13.0), with the
  detector on CPU. Frames are read in P8_a3run.md's order with both amendments: sequential reads from frame 0, and no
  seek inside the window.
- **Wrapped, in this process only:**
  - BYTETracker.update: the detections it receives (box, confidence, index) and the frame id. After the call, it
    records the tracked tracks (id, activated or not, state, start frame, detection index, box), the lost count and
    the output rows.
  - BOTSORT.get_dists, together with matching.linear_assignment: each association call's track ids, detection
    indices, threshold and matches.
  - BYTETracker.remove_duplicate_stracks: the tracks it removes.
  - PlayerTracker.reset: every reset, with its frame.
- **Predict alongside:** r23 predict at 0.40 runs on every frame, as in P8_a3run.md.
- **Check:** the traced output must equal the sidecar (same as P8_a3run.md's reproduction check). The share of frames
  where it does is reported.

## Reading classes (fixed now)

For every dropped predict box (no output box at IoU 0.5 or more):

- **G, not in the tracker's input:** no input detection reaches IoU 0.9 or more with it.
- **F, matched but not output:** an association matched its detection index, but no output row carries that index.
- **E, removed as a duplicate:** remove_duplicate_stracks removed the new track it started.
- **D, new and unconfirmed:** it started a new track that is not output on its first frame.
- **U, unmatched with no new track:** it was not matched, and no new track was recorded for it.
- **H, other.**

Reading rule:

- The class that holds at least 50 percent of the dropped boxes names the step:
  - G: the tracker's input differs from predict, so an input check comes next;
  - F: a loss in the wrapper or the callback, so a code-fix arm;
  - E: a duplicate-removal arm;
  - D: an activation arm;
  - U or H: the trace is incomplete; this is reported and nothing follows.
- If no class reaches 50 percent, every class is reported with its share, and nothing follows by itself.
- Any arm needs its own registration, then a rebuild under the lock, scored by the registered scorer.

## Cost and order

- About 12 to 15 lock minutes for one window: 1,053 processed frames at about 0.6 s, plus the read passes.
- It queues behind the truth-cpu driver, which holds the lock for about 9 hours.
- Then `report` runs in the system Python, with no lock.

Commands, from the worktree root, where <frozen CV python> is the anaconda interpreter the builds use:

    python3 -m export.heavy --name "p8 a3trace" -- <frozen CV python> -m tools.p8_a3trace run
    /usr/local/bin/python3 -m tools.p8_a3trace report

## Forking path

The a3run yield and the a3drop classes are seen. The window was chosen as the heavy window with the largest share of
dropped detections: 6,103 of 8,766, with only 3 of them under unconfirmed tracks.

## Results

Not run yet.
