# P8: what the P-ii boxes are (design, version 1, registered before any number)

Version 1, 2026-11-27. Report only: nothing is fitted, tuned, trained or built, no truth or score changes, and no Colab
unit is used. No held-out game, no D85 test game and no test arena is read (CLAUDE.md rule 4). hou_cle_2016 gets the
count gate only and stays unspent. Committed before the tool exists or computes any number; it goes to review before
any code runs.

## 1. Question

D205 split the missed row one to one. Its largest class is P-ii, 0.1141 [0.1060, 0.1231] of in-frame players pooled on
gsw_cha_2016 and gsw_bkn_2015: no box within 2.5 ft after the scorer's one-to-one assignment, and a box no one claims
lying 2.5 to 5 ft away (median 3.2 ft). D205 could not tell whether that box is the player's own box placed too far, a
false box, or another player's box. D206 then found that a population shift of the foot moves missed by only about
0.004 even in sample, consistent with per-box scatter, a P-ii offset unlike the hits', or ghost or neighbour boxes, not
separated there.

This item separates them by the box's track. For each P-ii player-frame, the stage B box carries a track id. Over the
rest of the window, is that track paired within 2.5 ft with this player, with someone else, or with no one? And how do
the boxes of each class look: height against what the camera predicts, truncation at the frame edge, overlap with other
boxes?

## 2. Known before this item

- D205 (P8_one2one.md section 11): the two-stage split, rates over in-frame players, pooled: P-ii 0.1141, P-i 0.0405,
  D-i 0.0440, D-iii-free 0.0143. P-ii's stage B box lies a median 3.20 and 3.21 ft away. Its count gate against D202
  passed on every window, and its copies equalled the scorer's one-to-one functions on every frame.
- D204: placement misses are carried by C3a's shared-box clause. The IoU clause (boxes overlapping at IoU 0.3 or more),
  which reads only the build, is not enriched (0.92 and 0.95).
- D206 (P8_footfix.md section 15): the two registered median foot forms reach about -0.004 in sample. Stage A hits
  set the medians, at 86 percent of fit rows.
- D203: team errors concentrate where another player is within 3 ft. Tracks paired with two or more player ids within
  2.5 ft hold 81 percent of team errors. That flag does not separate switches, merges and scorer mismatches.
- The build's identity step merges fragments into canonical tracks (the identity file's idmap). The scorer maps every
  box's tid through it (cid = idmap.get(tid, tid)) before team and identity rows.

## 3. Universe, inputs and the count gate

- **Games:** gsw_cha_2016 and gsw_bkn_2015, measured. hou_cle_2016: count gate only, with no class computed and no
  geometry read.
- **Inputs (stored files only, as D205):**
  - the reference root's builds/r23 composite (sidecars and identity files);
  - each game's current truth labels (work_ms for the pool, the reference root for hou_cle_2016);
  - the clips' sizes;
  - the scorecard cache (referee detections);
  - the SportVU moments and the mirror;
  - D202's score records (work_ref/rebaseline/score__<game>__current.json) and D205's records (work_ref/one2one/<game>.json).
  - The sidecar, time map, labels and identity hashes must equal D202's per window, and the cache must be complete
    (tools.p8_placecause.cache_gaps), or the run stops.
- **Frames:** the pinned scorer's universe, as D205: tools.p8_scorecard.score_window, truth mode camera, run once per
  window for its scored frames and its counts. In-frame players use the 10 px margin. Referee boxes follow the scorer's
  rule (IoU at least 0.5 to a cached referee detection of the frame).
- **Count gate, per window, before any class is kept:**
  - D202's fields (in_frame, missed, far, nr_boxes, missed_1to1, unmatched_1to1) equal D202's record, as D205's gate.
  - For the pool games, the recomputed main-split counts of every class (hit, P-i, P-ii, P-iii-ref, D-i, D-iii-ref,
    D-iii-free, D-iii-none) equal D205's record. D205 stores its rows by window index in window order.
  - Any difference stops the run.

## 4. Definitions

**The assignment** is D205's main two-stage assignment, called through tools.p8_one2one: two_stage(feet, sv, all ten
players, non-referee boxes, _assign_all). Stage A is the scorer's one-to-one 2.5 ft assignment, and stage B the 5 ft
assignment of the leftovers. frame_classes gives each in-frame player's class. It is run on every scored frame of the
window and keeps two things:
- the stage B box of every P-ii player;
- every stage A pair (box, player), for all ten players, in frame or not.

**A track** is the box's canonical id, cid = idmap.get(tid, tid), from the window's identity file. A box without a tid
gets the class "no track", which is expected to be empty.

**Pairing evidence** for a P-ii box b (track c) of player j (SportVU id p) on frame f comes from the window's other
scored frames, f excluded. On each other frame, c's stage A partner, if any, is a SportVU id. Let n_own be the number
of other frames where c's stage A partner is p, and n_other the number where it is another id.

**Classes (exhaustive, fixed now):**
- **own:** n_own at least 1 and n_other 0. The track is this player's on other frames and no one else's.
- **another's:** n_other at least 1 and n_own 0. The track is paired with other players only.
- **switch:** both at least 1. The track is paired with this player on some frames and with others on others. This is
  reported apart, with its split by majority (own-majority, other-majority, tied).
- **ghost:** both 0. The track is never paired within 2.5 ft in the window (beyond frame f itself, where it is
  unclaimed by definition). Split for description:
  - ghost, referee-flagged: the track's box is flagged referee on at least one scored frame of the window;
  - ghost, short: the track appears on fewer than 3 scored frames of the window;
  - ghost, persistent: the rest.
- **no track:** the box has no tid.

The ghost class here is "never paired within 2.5 ft". It is not the scorer's ghost row (a box more than 3.0 ft from
every player on the frame). Every P-ii box lies within 5 ft of its player, so a box can be a ghost here and not in the
ghost row. The share of each class's boxes that are scorer ghosts on the frame (more than 3.0 ft from all ten players)
is reported beside it.

**Report-only fields per P-ii box:**
- the gap in seconds to the nearest other frame giving the class's evidence (own or another's);
- for another's, whether the partner on those frames is in frame on f, and its distance from p on f;
- the depth and lateral offset of the box's foot under T from p, along T's camera ray (tools.p8_placement.
  depth_basis, the camera ground point from sportvu.camera.pose_from_h(T)).

## 5. Box geometry

Per P-ii box, and for reference per stage A box (hit) on the same scored frames:

- **Height against expected.** Box height h = y2 - y1. The expected height e is the pixel distance between the
  projections, through the truth camera's pose (tools.p8_scorecard.project3d with the pose of sportvu.camera.
  pose_from_h(T)), of the box's foot point on the court (T(p_foot)) at height 0 and at height 6.5 ft. The ratio is
  h / e. The nominal 6.5 ft cancels in the comparison with hits, which is what is read: each class's ratio distribution
  against the hits'. Bins, fixed now: under 0.75 (short: cut, crouched or partly hidden), 0.75 to 1.25, over 1.25
  (tall: a merged or doubled box). A frame with no pose gives no ratio, and these are counted.
- **Truncation.** The box touches the frame within 2 px: bottom edge (y2 at least H - 2), side or top. Reported as two
  shares.
- **Overlap.** The largest IoU with any other box on the frame, with the share at least 0.3 (D204's C3a IoU clause).
  Also the share whose area lies at least half inside another box (intersection over the P-ii box's area at least 0.5).

Per class and game: the median and the 25th and 75th percentiles of h / e; the bin shares; the truncation and overlap
shares, beside the hits'. Box height in px and the third of the box (the scorer's third_of under T) are reported too.

## 6. Measures and intervals

- Per game and pooled: each class's share of P-ii boxes and its rate over in-frame players.
- Per class: the geometry of section 5, and the hits' as reference.
- Window-cluster bootstrap, 2,000 resamples, seed 0. Windows are resampled whole within each game, and pooled by
  summing each game's resample (tools.p8_one2one.boot's convention).
- Counts: windows, scored frames, in-frame players, P-ii boxes, and tracks per class.

## 7. What each share would mean (descriptive; no rule decides anything)

| class | reading | where it points |
|---|---|---|
| own | the player's own box, placed 2.5 to 5 ft off on this frame | a foot or per-box problem: the box bottom, the foot point or the box itself on that frame. D206 found no population shift reaching the bar, so a fix here would need a per-box signal. The geometry says which: short boxes point to cut or hidden feet, normal boxes to the foot point |
| ghost | a box whose track is never anyone's within 2.5 ft | detection precision: a false box near the player. The player's own box is then missing from within 5 ft, so the miss is also a detection-recall miss that the stage B match hid |
| another's | a box whose track belongs to other players on other frames | tracking or identity: a neighbour's box, a doubled or merged box, or a track that moved between players. Overlap and tall boxes point to merges, the partner's distance to crowding |
| switch | a track paired with this player and others | identity: a track switch within the window (D203's flag), not separable here from a scorer mismatch in a crowd |

Each share is a ceiling for its route, not a forecast. A class's share says where the P-ii row sits. It says nothing
about how much a fix would move missed, since a fix also moves hits (D206).

## 8. What it can and cannot tell

- Track evidence comes from the build's own identity step. A wrong merge turns a ghost or another's box into "switch"
  or "own". A missed merge turns an own box into a ghost (short). The ghost split names the short tracks for this
  reason.
- "Paired within 2.5 ft on other frames" uses the same truth and the same stage A as the miss. A player whose own box is
  always 2.5 to 5 ft off is never paired with it, so that box reads ghost or another's. Own is therefore a lower bound
  on own boxes, and the ghost-persistent count includes such boxes. As a description, the share of ghost-persistent
  boxes whose track stays within 5 ft of p on at least half the other frames where both appear is reported. It is not a
  class.
- Scored frames are every sixth video frame with an accepted truth, so tracks are seen on those frames only.
- The build's and the truth's errors are not separable and not independent (D202, D205): the truth frames are selected
  by these feet.
- Two games, one arena, one broadcaster.

## 9. Cost and guards

- Stored files only, on the Mac, 0 units. One collect step per game under the lock ("p8 piibox <game>", frozen CV
  python, V3_P8_ROOT set to the reference root): gsw_cha_2016, then gsw_bkn_2015, then hou_cle_2016 (count gate only).
  D205's collect took 55 to 59 s per pool game and 113 s on hou_cle_2016. Here it is about 1 to 2 minutes per pool game
  (the pose per frame and the track tables add a little) and 2 minutes on hou_cle_2016, with the manifest hashing on top.
  About 8 lock minutes in all, cap 30 minutes. The report runs in system python with no lock.
- Every step runs as one background chain and is reported the moment it ends.
- Guards on every game, SportVU id and window: tools.p8_common.guard and tools.research_testset.refuse_cv_test. The
  tool refuses to compute classes or geometry on hou_cle_2016.
- Writes:
  - work_ref/piibox/<game>.json;
  - reports/research/p8_piibox.json.
  - SHA-256 manifests of work_ref's and work_ms's labels, builds and cache before and after each step stop the run on
    any difference.
- tools/ and sportvu/ clean at the run's head; the scorer pin checked.
- Retries: a mechanical failure is retried at most twice, then stopped and reported.
