# P8 build loss: what the tracker drops on the a3-heavy windows (registered before any number)

Written 2026-11-26 after the first full a3run report (P8_a3run.md). On the six a3-heavy windows the build's tracker
outputs about half of its own detector's boxes, and only 2.2 percent of the dropped boxes are unconfirmed tracks. This
measurement characterises the dropped boxes. It reads stored files only: the a3run replay records, the truth rows and
SportVU. It is report only: no model, no lock, no unit. Tool: tools/p8_a3drop.py, with the same rules in its
docstring. Record: reports/research/p8_a3drop.json.

## The tracker's real gates (read first)

Read from ultralytics 8.4.48 trackers/byte_tracker.py and bot_sort.py, and from botsort.yaml.

- **Thresholds that never bind.** The build passes detections at confidence 0.40. So track_high_thresh (0.25) and
  new_track_thresh (0.25) never bind, and the low-score second association (0.1 to 0.25) never receives a box.
- **First association.** Tracked and lost tracks are matched against detections at a cost of 1 - IoU x score
  (fuse_score is true). A pair matches when the cost is at most match_thresh 0.8, that is IoU x score at least 0.2
  with the Kalman-predicted, GMC-warped track box. BoT-SORT's proximity mask acts only with re-identification, which
  is off.
- **Unconfirmed tracks.** These were created on the previous frame. They are matched against the detections left over,
  at a cost of at most 0.7 (IoU x score at least 0.3). An unconfirmed track that finds no match is removed.
- **New tracks.** A detection left over starts a new track. The new track is not output on its first frame (except on
  the tracker's own first frame), and the replay records it as an unconfirmed box.
- **Duplicate removal.** remove_duplicate_stracks drops the younger of a tracked and a lost track whose boxes overlap
  at IoU above 0.85.
- **Output.** Every activated tracked track is output with its Kalman state box (track.result), not the detection box.

## Registered definitions

- Windows: the six a3-heavy windows of P8_a3run.md.
- Frames: replay frames that are scored frames with a truth row (tools.p8_footbias.load_frames), because the SportVU
  test needs the truth camera.
- A dropped box is a predict box (1) with no kept box (2) at IoU 0.5 or more in its frame, as P8_a3run.md defines it.

Measured per dropped box:

- its confidence;
- its best IoU with a kept box of the frame, and whether that kept box itself matches another predict box at IoU 0.5
  or more;
- its foot (bottom centre) through the truth camera, as the scorer maps a foot, and the distance to the nearest of the
  10 SportVU players;
- whether a kept box lies within two processed frames before or after at IoU 0.3 or more;
- its box-height bin and its camera third.

Classes, the first that holds:

1. **Duplicate.** A kept box at IoU 0.3 or more that matches another predict box at IoU 0.5 or more. The player is
   kept through another box.
2. **Displaced track box.** A kept box at IoU 0.3 or more that matches no predict box at IoU 0.5 or more. The track's
   output box has moved off its own detection.
3. **Low confidence.** Confidence under 0.5. With fuse_score, a detection below 0.5 needs IoU of at least 0.4 with
   the predicted track box to pass the first gate, and 0.6 to confirm an unconfirmed track.
4. **Unmatched real player.** The foot lies within 2.5 ft of a SportVU player. Split into 4a, a gap (a kept box within
   two frames at IoU 0.3 or more), and 4b, never tracked nearby.
5. **Near a player.** The foot lies 2.5 to 3.0 ft from the nearest player.
6. **False positive.** The foot lies more than 3.0 ft from every player (the scorer's ghost rule).

Per class: the count and the share of dropped boxes, per window and pooled. The same is reported over the dropped
boxes whose foot lies within 2.5 ft of a far-miss player of the frame: an in-frame player with no build foot_stab
within 5 ft. Those are the players the a3 loss is made of.

## Reading rule (fixed now)

A class leads to a tracker arm when it holds at least 25 percent of the pooled dropped boxes:

- class 2: a tracker-state arm (output the matched detection box, or change GMC);
- class 3: a score-fusion arm (fuse_score off, or a looser match_thresh);
- class 4a: an association arm (match_thresh, track_buffer);
- class 4b: an activation arm (output a new track on its first frame, or new_track_thresh);
- classes 1, 5 and 6: nothing, because the drop loses no player, or none that can be scored.

Any arm needs its own registration, then a rebuild on the reference platform under the lock, scored by the registered
scorer.

## Forking path

The a3run yield result is already seen: a pooled yield of 0.50, with 2.2 percent of dropped boxes under unconfirmed
tracks. The classes were written knowing it.

## Results

Not computed yet.
