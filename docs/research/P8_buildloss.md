# P8 detection row: where the build loses r23's raw boxes (registered before any number)

Written 2026-11-26 after P8_detrow.md. The lead took its post hoc proposal: ignoring the class order, r23's raw
detector had a box within 5 ft for 37 percent of far misses. That is 0.020 to 0.038 of in-frame players, above the bar
on every pool game. Fixing such a loss is tracking or gating code, not GPU work.

This measurement is report only. No model runs, and nothing is built, trained or changed. Tool:
tools/p8_buildloss.py, with the same rules in its docstring. Record: reports/research/p8_buildloss.json. This document
and the tool were committed before the tool produced any number.

## The build path, read first (what is recorded where)

- **Gate v2** (build_trajectories.py, --pregate and the per-frame gate). A rejected frame is skipped whole and gets no
  sidecar row. The scorer's universe needs a sidecar row, so a gate drop never produces a far miss in this
  population; it lowers coverage instead. No gate class is therefore possible here. The lead's "gate v2 drop" split
  is empty by construction.
- **Detector and tracker** (detect/tracker.py). One ultralytics model.track call at confidence 0.40, with BoT-SORT
  (track_high_thresh 0.25, new_track_thresh 0.25, track_buffer 30, match_thresh 0.8).
  - Only tracks the tracker outputs reach the sidecar. Raw detections are not stored.
  - In ultralytics 8.4.48 (trackers/byte_tracker.py), a new track is activated at once only on the tracker's first
    frame. Later it starts unconfirmed and is output after a second match.
  - A camera-cut detector (detect/camera_cut.py) resets the tracker when 80 percent of 4 or more tracks vanish.
    Resets are not recorded, but new track ids show them.
- **Foot stabilisation** (detect/footpoint.py). For each track it keeps an EMA of the foot (alpha 0.6). It also
  applies an occlusion correction: when a box is shorter than 0.85 of its median height, the foot moves down to that
  median height. The sidecar records foot_raw, foot_stab and foot_corrected.
- **Sidecar rows.** Each processed frame records boxes with tid, bbox, foot_raw, foot_stab and foot_corrected.

## Registered definitions

Population:

- The pool games only (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015).
- Far misses (no foot_stab within 5 ft under the truth camera) for which an M4 r23 raw box at 0.40 has its
  bottom-centre foot, through the truth camera, within 5 ft. This is order-free. The raw box used is the one whose
  foot is nearest.
- M4's raw boxes exist only on the corpus frames, so hou_cle_2016 cannot be measured this way.
- Check: per game, the count must equal P8_detrow.md's order-free "raw had a box" count (786, 261, 431), or the tool
  stops.

Classes, a partition, in this order:

- **a1, activation lag.** No sidecar box at IoU 0.5 or more with the raw box in this frame, and none at IoU 0.3 or
  more in the previous processed frame. In the next processed frame there is a box at IoU 0.3 or more whose track id
  is in neither this frame nor the previous one: the tracker started the track later.
- **a2, track gap.** No sidecar box at IoU 0.5 or more in this frame. But there is a box at IoU 0.3 or more in the
  previous processed frame, or one in the next with a track id that was already present. A track existed and was not
  output here: lost, cut or left unmatched. The lead's "(b) a track dropped or cut there" shows up here, because a box
  that is present in the frame would be scored.
- **a3, never tracked nearby.** No sidecar box at IoU 0.5 or more here, and none at IoU 0.3 or more in the previous
  or next processed frame. Either the tracker never output this player around here, or the build's own detector did
  not fire on the decoded video. The stored data cannot tell those two apart.
- **b, tracked box differs.** A sidecar box at IoU 0.5 or more exists, but its foot_raw is also more than 5 ft away
  through the truth camera. The tracker's box state is at fault, not the stabiliser.
- **c, foot stabilisation moved it.** A sidecar box at IoU 0.5 or more exists, its foot_raw is within 5 ft, and its
  foot_stab is not. Split into c_occlusion (foot_corrected true) and c_ema (false).
- **e, other.** Anything else; expected to be empty.
- **(d) decode or frame mismatch.** Not separable per player. It is sized from M6's stand-in on gsw_bkn_2015, with
  both sides on CPU: 1,356 of 19,511 raw boxes at 0.40 (6.9 percent) had no sidecar box at IoU 0.5. That figure bounds
  tracking and decode together, and the decode part is mixed into a1 to a3.

Report only: for a1 and a2, the frames whose track ids look reset (80 percent or more of 4 or more previous ids
absent).

Measures: per class and per game, the count, the share of the population and the rate among in-frame players, with
window intervals (2,000 resamples, seed 0, windows within the game).

## Reading rule (fixed now)

A class matters at a rate of at least 0.010 among in-frame players on every pool game. What each leads to:

- **a1:** a tracker activation arm, for example activating new tracks at once or a lower new_track_thresh.
- **a2:** a tracker buffer or association arm (track_buffer, match_thresh, fuse_score).
- **a3:** a check of the build's own detector on the decoded video, which is a model run under the lock.
- **b:** a tracker box-state arm.
- **c:** a foot_stab arm (the EMA weight or the occlusion correction).

Any arm needs its own registration and a rebuild under the heavy-job lock, scored on the reference platform.

## Forking path

The order-free 37 percent figure, and every number in P8_detrow.md, are already seen.

## Results (computed 2026-11-26, after the definitions above were committed in 1f18d81f)

Stored files only, no lock, about 12 seconds. Checks passed: load_frames' count check against the registered score
files, and the population equals P8_detrow.md's order-free count on every game (786, 261 and 431). Record:
reports/research/p8_buildloss.json.

Rate among in-frame players [window interval]; share of the population:

| Class | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 |
| --- | --- | --- | --- |
| a1 activation lag | 0.0013; 3.4% | 0.0013; 6.5% | 0.0011; 5.6% |
| a2 track gap | 0.0059 [0.0033, 0.0093]; 15.4% | 0.0044 [0.0026, 0.0062]; 22.6% | 0.0029 [0.0021, 0.0039]; 14.4% |
| a3 never tracked nearby | 0.0230 [0.0007, 0.0666]; 59.9% | 0.0060 [0.0004, 0.0170]; 30.3% | 0.0096 [0.0014, 0.0220]; 47.1% |
| b tracked box differs | 0.0024; 6.2% | 0.0021; 10.7% | 0.0022; 10.7% |
| c occlusion correction | 0.0030 [0.0017, 0.0046]; 7.9% | 0.0029 [0.0014, 0.0046]; 14.6% | 0.0020 [0.0012, 0.0029]; 9.7% |
| c EMA | 0.0027 [0.0019, 0.0037]; 7.1% | 0.0030 [0.0019, 0.0045]; 15.3% | 0.0026 [0.0018, 0.0035]; 12.5% |
| e other | 0 | 0 | 0 |

No a1 or a2 player-frame falls on a frame whose track ids look reset. (d), the bound from M6: 6.9 percent of r23's raw
boxes had no sidecar box on gsw_bkn_2015 with both sides on CPU, tracking and decode together.

## Reading by the registered rule

No class matters: none reaches 0.010 of in-frame players on every pool game.

- a3, never tracked nearby, is the largest class. It is 0.023 on gsw_bkn_2015 but 0.006 on gsw_cha_2016 and 0.0096 on
  gsw_sac_2015, so it falls short of the bar on two games.
- Every other class stays at 0.006 or below on every game.
- By the rule, no tracking, gating or foot-stabilisation arm is selected.

What the numbers say beyond the rule (post hoc, not tested):

- The loss is mostly "no box at all in the build". Together a1, a2 and a3 are 0.030, 0.012 and 0.014 of in-frame
  players. That would be above the bar on every game if the rule had pooled them, which it did not.
- Within that loss, a3 dominates, and it is the part the stored data cannot attribute. Either the tracker never output
  a track, for example an unconfirmed track that never got its second match, or the build's detector on the decoded
  video did not fire where the CPU run on the corpus JPEG did.
- a3's window intervals are very wide (up to [0.0007, 0.067] on gsw_bkn_2015). The loss sits in a few windows, not
  across the game. That fits a window-level cause, such as decode, scene type or a tracker state, better than a
  per-player one. It is not measured here.
- Foot stabilisation (c) loses 0.005 to 0.006 of in-frame players per game, split about evenly between the occlusion
  correction and the EMA. It is small but steady, with narrow intervals.

Proposal for the lead (it needs its own registration; the rule did not select it): find what a3 is. The cheapest
discriminating step is a model run on the windows that carry most of a3. It would run r23 on the decoded video frames
the build used, the same frames as the corpus JPEGs, on CPU, both raw and through model.track. That separates "the
detector did not fire on the video" from "the tracker did not output it". It needs the heavy-job lock (model code)
for minutes per window, and no units.

## Version 2 (2026-11-26; registered after version 1's numbers were seen, before version 2's)

The ml-review of P8_detrow.md raised three points that change this design. Version 1's result (3e90ac33) had already
been computed and read when they arrived, so version 2 is registered with version 1's numbers seen, and that is its
forking path. Version 1 stays as committed. Tool: python3 -m tools.p8_buildloss measure2; record:
reports/research/p8_buildloss_v2.json; per-player rows stay in the P8 workspace (placement/buildloss_v2.json).

1. Distances, recorded per player. The far-miss argument only shows that every kept foot_stab lies beyond 5 ft. A
   foot_stab shift just over 5 ft minus the raw distance is enough to make a miss, so classes are read with the
   numbers:
   - d_raw: the nearest M4 raw bottom-centre at 0.40;
   - d_video: below;
   - d_build: the nearest foot_stab;
   - for the box matched at IoU 0.5 or more, its foot_raw and foot_stab distances.
   Class c (foot stabilisation) is still defined by the matched box's foot_raw within 5 ft and its foot_stab beyond.
   The doc reports their medians.
2. Reachable share. The order-free rates are upper bounds on any missed gain. Only a recovered foot within 2.5 ft can
   lower the missed row; a box recovered 2.5 to 5 ft away becomes a placement miss. The reading rule therefore counts,
   for each class, only the players with d_raw at most 2.5 ft. A class matters when that reachable rate among in-frame
   players is at least 0.010 on every pool game. The rule reads the point estimate, and window intervals (2,000
   resamples, seed 0) of the reachable rates are reported beside it.
3. Decode, sized per player. M6's 6.9 percent was every raw-only box on one game, per box, including real build drops,
   so it does not size decode on this population. The data can size it.
   - The E1 corpus preds ("ours") are r23 at 0.40 on CPU, run on each decoded clip frame before that frame was written
     as the JPEG. The clip is the same window clip the build read (tools/p8_e1data.py run).
   - So class d, decode, is: no sidecar box at IoU 0.5 or more, and no "ours" box within 5 ft on the decoded frame.
     The box then exists only on the JPEG, which differs from the video frame by the JPEG compression and the corpus
     environment's library versions.
   - Order: a matched box gives b or c as in version 1; otherwise d when the decoded-frame detector has no box within
     5 ft; otherwise a1, a2 or a3. A player in a1 to a3 now had a box from the detector on the decoded frame, so the
     loss is the tracker's.

What each class leads to is unchanged from version 1. d leads to no build-side arm: the detector input differs, not
the build. The tool fails loudly if a record or corpus file it needs is missing, and it calls refuse_cv_test at the
game load.

Repository note: test_every_p8_tool_runs_the_cv_guard needs a literal refuse_cv_test( call in every tools/p8_*.py.
tools/p8_buildloss.py has one. On this branch the test still lists tools/p8_detrow.py, p8_footbias.py, p8_footpos.py
and p8_placement_h.py, which the lead fixes on v4-research (30c30de3 and the P8_detrow.md merge). This branch does not
touch them.

### Version 2 results (computed 2026-11-26, after f7290d99)

Stored files only, no lock, about 17 seconds. Checks passed: the scorer counts reproduce, the population equals
P8_detrow.md's order-free count on every game (786, 261, 431), and every population frame has corpus preds. Record:
reports/research/p8_buildloss_v2.json.

Reachable players (raw foot within 2.5 ft): 519 of 786, 101 of 261 and 239 of 431. The raw distance quartiles are
1.19, 1.92 and 3.02 ft; 1.82, 3.26 and 4.37 ft; 1.14, 2.24 and 3.88 ft.

Count; reachable count; reachable rate among in-frame players:

| Class | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 |
| --- | --- | --- | --- |
| d decode (no box on the decoded frame) | 8; 3; 0.0001 | 3; 1; 0.0001 | 5; 1; 0.0000 |
| a1 activation lag | 25; 13; 0.0006 | 17; 4; 0.0003 | 24; 16; 0.0008 |
| a2 track gap | 116; 77; 0.0038 [0.0017, 0.0066] | 56; 24; 0.0018 [0.0008, 0.0029] | 58; 28; 0.0013 [0.0008, 0.0021] |
| a3 never tracked nearby | 470; 374; 0.0183 [0.0002, 0.0538] | 79; 51; 0.0038 [0.0002, 0.0112] | 202; 161; 0.0076 [0.0010, 0.0180] |
| b tracked box differs | 49; 7; 0.0003 | 28; 3; 0.0002 | 46; 3; 0.0001 |
| c occlusion correction | 62; 40; 0.0020 [0.0011, 0.0029] | 38; 16; 0.0012 [0.0004, 0.0022] | 42; 25; 0.0012 [0.0006, 0.0020] |
| c EMA | 56; 5; 0.0002 | 40; 2; 0.0002 | 54; 5; 0.0002 |
| e other | 0 | 0 | 0 |

For class c, the matched box's median foot_raw distance is 3.6, 4.0 and 4.1 ft, and its median foot_stab distance is
5.5, 5.6 and 5.4 ft. Stabilisation moves these feet by under 2 ft. Most of them were already beyond 2.5 ft raw, so
only the occlusion-correction part is mostly reachable.

## Version 2 reading by the registered rule

No class matters. No reachable rate reaches 0.010 on every pool game. a3 (never tracked nearby) is 0.018 on
gsw_bkn_2015 but 0.004 and 0.008 on the other two games. No tracking, gating or foot-stabilisation arm is selected.

Decode is now sized on this population: 8, 3 and 5 players. The detector on the decoded clip frame (corpus "ours")
had a box near almost every population player. So almost all of the "no box in the build" loss is not decode; it
happens between the detector and the sidecar.

What the numbers say beyond the rule (post hoc, descriptive):

- a3's reachable players sit in a few windows.
  - gsw_bkn_2015: 363 of 374 are in gsw_bkn_2015_s04_w00. That is the window M6 and M7 already singled out: on CPU
    the r23 build holds 4.27 boxes a frame there and misses 784 of 1,385 players.
  - gsw_cha_2016: 36 of 51 are in gsw_cha_2016_s05_w00.
  - gsw_sac_2015: 69, 47 and 23 of 161 are in s01_w00, s09_w01 and s07_w00.
  - Outside those windows a3 is small. The loss is a window-level failure of the tracker path, not a per-player one.
- detect/tracker.py returns no tracks for a frame when the tracker raises (a Kalman LinAlgError or the ultralytics
  re-identification AttributeError) and resets it. A run of such frames, or repeated resets, would empty the sidecar
  where the detector still fires. This is a hypothesis; the stored data were not checked for it.

Proposal for the lead (needs its own registration; stored files, minutes, no lock): for the windows carrying a3,
count processed frames with zero sidecar boxes, track-id turnover between frames, and the build logs' reset and
exception lines where they are stored. If empty frames or resets explain them, a fix is tracker robustness code (catch
and keep tracks, or avoid the reset), scored by a rebuild of those windows on the reference platform under the lock.
