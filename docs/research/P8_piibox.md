# P8: what the P-ii boxes are (design, version 2, registered before any number)

Version 2, 2026-11-27, after the design review of version 1 (777806c3, GO WITH CHANGES). It replaces version 1, and
section 10 maps each finding to its change. Report only: nothing is fitted, tuned, trained or built, no truth or score changes, and no Colab
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
- **Guards:** tools.p8_common.guard and tools.research_testset.refuse_cv_test on every game, SportVU id, video id
  and window.
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
  - For the pool games, the sorted list of P-ii stage B distances equals D205's (stage_b_ft, within 1e-9 ft).
  - Each P-ii player's stage B box from two_stage equals frame_classes' record: stage B, and the same distance.
  - Every label row's camera fields (camera_C, camera_rvec, camera_f) reproduce its H_truth through sportvu.camera.
    h_px2ft, both normalised to H[2][2] = 1, within 1e-6 at every entry.
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

**Pairing evidence** for a box b (track c) of player j (SportVU id p) on frame f comes from the window's other scored
frames, f excluded. On each other frame:
- c's stage A partner, if any, is a SportVU id. n_own counts the other frames where it is p, and n_other the frames
  where it is another id (in frame or not).
- n_ownB counts the other frames where c is p's stage B box (p is P-ii there and its stage B box is c's).

**Classes (exhaustive, in this order, fixed now):**
1. **own:** n_own at least 1 and n_other 0. The track is this player's within 2.5 ft on other frames and no one
   else's.
2. **own, with crossings:** both at least 1, and n_other is at most 0.2 of n_own + n_other. This is read with own.
3. **another's, with crossings:** both at least 1, and n_own is at most 0.2 of n_own + n_other. This is read with
   another's.
4. **switch:** both at least 1, and neither side at most 0.2.
5. **another's:** n_other at least 1 and n_own 0.
6. **own, misplaced:** n_own and n_other both 0, and n_ownB at least 1. The track is this player's stage B box on other
   frames: the player's own box, always placed 2.5 to 5 ft off. This is read with own, as the foot or per-box route.
7. **ghost:** n_own, n_other and n_ownB all 0. The track is never paired within 2.5 ft in the window, and is never this
   player's stage B box (frame f excluded). Split in this order:
   - ghost, referee-flagged: a box of the track is flagged referee on at least one scored frame of the window;
   - ghost, short: the track appears on fewer than 3 scored frames of the window (scored frames only, f included);
   - ghost, persistent: the rest.
8. **no track:** the box has no tid. This is a stopping counter: one such box stops the run.

The 0.2 bar is applied to SportVU ids pooled as "this player" against "anyone else". The full (n_own, n_other, n_ownB)
distribution of every class is kept as a report-only field.

The ghost class here is "never paired within 2.5 ft". It is not the scorer's ghost row (a box more than 3.0 ft from
every player on the frame). Every P-ii box lies within 5 ft of its player, so a box can be a ghost here and not in the
ghost row. The share of each class's boxes that are scorer ghosts on the frame (more than 3.0 ft from all ten players)
is reported beside it.

**Report-only fields per P-ii box:**
- the gap in seconds to the nearest other frame giving the class's evidence (own or another's);
- for another's (with or without crossings), the most frequent other partner: whether it is in frame on f, and its
  distance from p on f;
- the 5 ft proximity share: for another's (both kinds), own, misplaced and ghost, persistent, the share of boxes whose
  track lies within 5 ft of p under T on at least half the other frames where the track appears;
- the depth and lateral offset of the box's foot under T from p, along the camera ray from camera_C's ground point
  (tools.p8_placement.depth_basis with (camera_C x, camera_C y)).

**The hits baseline.** Every stage A box of an in-frame player (class hit) on the same scored frames is classified the
same way, with p its own partner and f its frame. Each class's share among hits is reported beside its share among
P-ii boxes, with the same bootstrap. For hits, n_ownB counts frames where the track is p's stage B box, as for P-ii.

## 5. Box geometry

Per P-ii box, and for reference per stage A box (hit) on the same scored frames:

- **Height against expected.** Box height h = y2 - y1. The expected height e is the pixel distance between the
  projections of the box's foot point on the court (T(p_foot)) at height 0 and at height 6.5 ft. They are projected
  through the label row's own camera (camera_C, camera_rvec, camera_f) by tools.p8_scorecard.project3d, the camera the
  row's H_truth was made from (checked in the gate). There is no pose fallback. The ratio is h / e. The nominal 6.5 ft
  cancels in the comparison with hits, which is what is read: each class's ratio distribution against the hits'. Bins,
  fixed now: under 0.75 (short: cut, crouched or partly hidden), 0.75 to 1.25, over 1.25 (tall: a merged or doubled
  box).
- **Truncation.** The box touches the frame within 2 px: bottom edge (y2 at least H - 2), side or top. Reported as two
  shares.
- **Overlap.** The largest IoU with any other box on the frame, with the share at least 0.3 (D204's C3a IoU clause).
  Also the share whose area lies at least half inside another box (intersection over the P-ii box's area at least 0.5).

Per class and game: the median and the 25th and 75th percentiles of h / e; the bin shares; the truncation and overlap
shares, beside the hits'. Box height in px and the third of the box (the scorer's third_of under T) are reported too.

## 6. Measures and intervals

- Per game and pooled: each class's share of P-ii boxes and its rate over in-frame players; beside it, each class's
  share among hit boxes.
- Per class: the geometry of section 5, and the hits' as reference.
- Window-cluster bootstrap, 2,000 resamples, seed 0. Windows are resampled whole within each game, and pooled by
  summing each game's resample (tools.p8_one2one.boot's convention).
- Counts: windows, scored frames, in-frame players, P-ii boxes, and tracks per class.

## 7. What each share would mean (descriptive; no rule decides anything)

| class | reading | where it points | what moving this class needs for missed |
|---|---|---|---|
| own (with own, with crossings and own, misplaced) | the player's own box, placed 2.5 to 5 ft off on this frame | a foot or per-box problem: the box bottom, the foot point or the box itself on that frame. D206 found no population shift reaching the bar, so a fix here would need a per-box signal. The geometry says which: short boxes point to cut or hidden feet, normal boxes to the foot point | move the box (its foot) inside 2.5 ft of its player; the own share is a ceiling for that route, not a forecast, since a fix also moves hits (D206) |
| ghost | a box whose track is never anyone's within 2.5 ft | detection precision: a false box near the player. The player's own box is then missing from within 5 ft, so the miss is also a detection-recall miss that the stage B match hid | a precision change removes the box and changes the ghost row; missed moves only if the player's own box is recalled |
| another's (with another's, with crossings) | a box whose track belongs to other players on other frames | tracking or identity: a neighbour's box, a doubled or merged box, or a track that moved between players. Overlap and tall boxes point to merges, the partner's distance to crowding | an identity fix changes the team and identity rows; missed moves only if a merged box is split or the player's own box is recalled |
| switch | a track paired with this player and others | identity: a track switch within the window (D203's flag), not separable here from a scorer mismatch in a crowd | as another's: team and identity rows; missed only through a split or own-box recall |

A class's share says where the P-ii row sits, read beside the same class's share among hits. A class as common among
hits as among P-ii boxes says less about the P-ii row than one concentrated in it.

## 8. What it can and cannot tell

- Track evidence comes from the build's own identity step. A wrong merge turns a ghost or another's box into "switch"
  or "own". A missed merge turns an own box into a ghost (short). The ghost split names the short tracks for this
  reason.
- "Paired within 2.5 ft on other frames" uses the same truth and the same stage A as the miss. A player whose own box is
  always 2.5 to 5 ft off is never paired with it in stage A. Version 2 catches it as own, misplaced when the track is
  the player's stage B box on another frame. Where a neighbour or the frame's other boxes take that stage B slot, it
  still reads ghost or another's, so the own classes remain a lower bound on own boxes. The 5 ft proximity share of
  another's and ghost, persistent (section 4) describes that remainder. It is not a class.
- Scored frames are every sixth video frame with an accepted truth, so tracks are seen on those frames only.
- The build's and the truth's errors are not separable and not independent (D202, D205): the truth frames are selected
  by these feet.
- Two games, one arena, one broadcaster.

## 9. Cost and guards

- Stored files only, on the Mac, 0 units. One collect step per game under the lock ("p8 piibox <game>", frozen CV
  python, V3_P8_ROOT set to the reference root): gsw_cha_2016, then gsw_bkn_2015, then hou_cle_2016 (count gate only).
  D205's collect took 55 to 59 s per pool game and 113 s on hou_cle_2016. Here it is about 1 to 2 minutes per pool game
  (the camera projection per box, the hits baseline and the track tables add a little) and 2 minutes on hou_cle_2016, with the manifest hashing on top.
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

## 10. Changes from version 1 (the review of 777806c3, GO WITH CHANGES)

1. Stage B evidence: n_ownB is recorded on the other frames, and own, misplaced (n_own 0, n_other 0, n_ownB at least
   1) is registered as the foot or per-box route. The 5 ft proximity share is also reported for another's (section 4).
2. The hits baseline: every stage A hit box of an in-frame player is classified the same way, with its class shares
   beside P-ii's and the same bootstrap (sections 4 and 6).
3. A minority rule is fixed now: own, with crossings (n_other at most 0.2 of the paired frames) is read with own. It is
   applied symmetrically: another's, with crossings (n_own at most 0.2) is read with another's, and switch keeps the
   rest. The full (n_own, n_other, n_ownB) distribution is kept as a report-only field (section 4).
4. Readings: a column "what moving this class needs for missed" is added. "Ceiling for its route" is kept for own only
   (section 7).
5. Expected height comes from each label row's camera_C, camera_rvec and camera_f via project3d, and the depth basis
   from camera_C's ground point. The pose fallback is dropped, and the gate checks that the camera fields reproduce
   H_truth (sections 3, 4 and 5).
6. Smalls:
   - the ghost order is referee-flagged, then short, then persistent;
   - the gate also matches the sorted stage_b_ft list per window;
   - the video id is guarded;
   - no track is a stopping counter;
   - "short" counts scored frames only.

## 11. Amendment (2026-11-27, after the code check of 45841d80, GO WITH CHANGES; before any number)

1. **Section 4, another's with stage B evidence.** When n_own is 0, n_other at least 1 and n_ownB at least 1, the 0.2
   rule is applied with the own evidence taken as n_ownB:
   - n_other at most 0.2 of (n_ownB + n_other) reads **own, misplaced, with crossings**, read with own (the foot or
     per-box route);
   - n_ownB at most 0.2 of the same reads **another's**;
   - otherwise it reads **switch**.

   The class joins the list after another's and before own, misplaced, and it joins the own group ("own (all four)").
2. **The camera check** is relative: every entry of the two normalised matrices agrees within 1e-9 of its size (numpy
   allclose, rtol 1e-9, atol 1e-15). An absolute 1e-6 would be loose for the third-row entries, which are about 6e-5.

## 12. Amendment (2026-11-27, after run 1 stopped; before any number)

Run 1 stopped after 10 s on gsw_cha_2016's first window with a KeyError: a player of class hit had no entry in the
two-stage assignment. Nothing was written under work_ref/piibox/, and no file under the watched folders of work_ref or
work_ms changed after the run started (checked by modification time).

Cause: "hit" is the scorer's nearest-foot rule (some foot, referee boxes included, within 2.5 ft), not a stage A match.
So a hit player can have no stage A box: the nearest foot is a referee's, or another player claimed that box one to one.
Such a player then has either no match or only a stage B box. Section 4's hits baseline ("every stage A box of an
in-frame player of class hit") does not cover these players. They are now counted, not classified, as two report-only
counters: hits without a stage A box, and hits with a stage B box only. No class, geometry or gate changes. The P-ii
check (a P-ii player's box must be a stage B match at frame_classes' distance) stays a stop.
