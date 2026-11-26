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

## Results (computed 2026-11-26, after the definitions above were committed in e2ca4cd9)

Stored files only, no lock, about 7 seconds. load_frames' count check passed. Record: reports/research/p8_a3drop.json.

There are 932 dropped boxes on the scored frames of the six heavy windows. Their median confidence is 0.89.

| Class | Count | Share | Among the 578 dropped boxes on a far-miss player |
| --- | --- | --- | --- |
| 1 duplicate | 78 | 8.4% | 30 (5.2%) |
| 2 displaced track box | 0 | 0% | 0 |
| 3 low confidence | 22 | 2.4% | 7 (1.2%) |
| 4a real player, gap | 43 | 4.6% | 21 (3.6%) |
| 4b real player, never tracked nearby | 581 | 62.3% | 520 (90.0%) |
| 5 near a player | 58 | 6.2% | 0 |
| 6 false positive | 150 | 16.1% | 0 |

Class 4b by window: 387 of 619 on gsw_bkn_2015_s04_w00, 38 of 70 and 10 of 18 on the two gsw_cha_2016 windows, and
76 of 98, 46 of 59 and 24 of 68 on the three gsw_sac_2015 windows. By third, the dropped boxes split 293 near, 397
middle and 242 far. By height, 242 are under 150 px, 511 are 150 to 200, 165 are 200 to 250 and 14 are 250 and over.

## Reading by the registered rule

Class 4b holds 62.3 percent of the dropped boxes, so the rule leads to an activation arm. These are high-confidence
detections of real players (foot within 2.5 ft of SportVU) with no kept track within two processed frames, and they
make up 90 percent of the dropped boxes that sit on the a3 players.

A caution before that arm is designed (post hoc, descriptive, from the same replay records):

- The activation arm as written assumes these detections start new tracks that are never output. The replay does not
  support that. At the end of their frame, only 2.2 percent of all dropped boxes on these windows lie under an
  unconfirmed track (P8_a3run.md).
- No kept box lies off its detection: every kept box on every heavy window matches a predict box at IoU 0.5 or more.
- The windows hold few distinct track ids (33 over 1,053 frames on gsw_cha_2016_s05_w00) and few lost tracks (0.3 to
  2.9 on average).
- So most of these detections neither match a kept track, nor start a track, nor sit under a lost one, by what the
  replay recorded. Reading the code, an unmatched detection at 0.40 always starts a track (new_track_thresh 0.25). The
  replay's recorded fields therefore do not explain where these detections go.
- Outputting new tracks on their first frame would most likely not recover them.

Proposal for the lead, before any tracker arm: one instrumented replay of a heavy window under the lock, about 10 to 20
minutes, on gsw_cha_2016_s05_w00 or gsw_bkn_2015_s04_w00. It would record, per frame, the detections BoT-SORT's update
receives, the first-association matches, the unconfirmed matches, the new tracks created, and the tracks removed by
remove_duplicate_stracks. That shows the step where the detections vanish. It needs its own registration. The arm the
rule names would follow only if the instrumented run confirms an activation cause.

## Review amendment (2026-11-26, after the ml-review and the a3input result; the registered text above is unchanged)

The caution in the reading section rests on the replay's state at the end of each frame. That covers the 2.2 percent
under unconfirmed tracks, and the conclusion that the detections "neither match a kept track, nor start a track, nor
sit under a lost one". After a reset, each frame runs two update calls (docs/research/P8_a3input.md, cause confirmed).

- In the first call, these detections do start new, unconfirmed tracks.
- In the second call, which receives only the tracked boxes, those tracks find no match and are removed, so the end of
  the frame shows none of them.

The caution's statement that the recorded fields "do not explain where these detections go" is answered there. The
registered reading (an activation arm) stays as registered. The fix that follows is the reset fix that P8_a3input.md
names, not an activation parameter. See the amendment in P8_a3trace.md.

