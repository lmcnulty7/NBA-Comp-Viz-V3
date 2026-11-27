# P8 NMS arm: a looser NMS IoU threshold on r23 (design, version 1, 2026-11-27)

Report only. The arm changes no frozen file and no default; nothing here is adopted into the build. No lock and no
compute until this design is reviewed. hou_cle_2016 stays unspent: no hou window is built, scored or read. gsw_sac_2015
stays apart. Held-out games are refused by tools.p8_common.guard and tools.research_testset.refuse_cv_test on every
game and window list.

## 1. Question

Does a looser NMS IoU threshold on the r23 detector (the build's is config.PLAYER_IOU 0.45; a higher threshold
suppresses less) recover C2 misses on gsw_cha_2016 and gsw_bkn_2015 without adding more ghosts than it recovers?

Mechanism (D208, P8_recallcause.md 8.3): a C2 miss is a player with no box of its own after NMS whose own candidate
(foot within 2.5 ft of the player, nearer the player than anyone else) was suppressed by a kept box that belongs to a
neighbour. The median IoU of that candidate with its suppressor is 0.559 and its median score 0.758 (D208). At an IoU
threshold above that IoU the candidate survives NMS. The cost: overlapping duplicates of one player also survive, and
the tracker may turn them into ghost tracks or switch identities.

## 2. Upper bound (from D208 and D205)

- C2 is 0.154 of P-i misses and 0.085 of D-i misses on D208's 88 kept windows (421 misses).
- P-i is 0.041 and D-i 0.044 of in-frame players (D205, both games pooled, all windows).
- If every C2 miss became a hit: missed falls by at most about 0.154 x 0.041 + 0.085 x 0.044 = 0.010 (about 5 percent
  of D202's 0.214). C2's candidate foot is within 2.5 ft of the player, so a kept and tracked candidate is a hit
  unless the foot stabiliser moves it.
- Limits: the shares are on kept windows only (D208 9.3); C2 is not measured in the 32 dropped windows or for the other
  miss classes (P-ii, the D-iii classes), where an extra box could also help. The bound is a ceiling, not a forecast.

Expected effect, with reasoning: below the ceiling. Not every C2 candidate survives at a given threshold (the IoU
spread is unknown above the median), BoT-SORT activates a new track only on a second match and may hand the recovered
box to the neighbour's track, and duplicates add ghosts. A guess of 0.003 to 0.007 on missed, with a ghost rise of
unknown size, is the prior; the pilot measures both.

## 3. Stored candidates and whether re-NMS replays the build

Stored (P8_recallcause.md 8.1; reference root recallcause/<window>.json, 120 windows, 3,631 frames, 36 MB): per
universe frame, R_post (conf 0.40, IoU 0.45, max_det 300, class 0), R_pre (conf 0.40, IoU 1.0, max_det 30000, class 0:
every candidate), other classes and conf 0.25 boxes, as [x1, y1, x2, y2, score, cls] in frame pixels at full float
precision. The anchor indices were not stored; R_post's boxes are a subset of R_pre's with identical coordinates (the
same anchors of the same tensor, scaled the same way: the code check of 7ba557d2 found the indices a subset, and
all 2,232 R_post boxes of the two build-environment windows' stored frames appear exactly in R_pre), so kept and suppressed
candidates are told apart by exact equality.

Re-NMS replays NMS, not the build:
- **NMS on stored frames.** ultralytics' non_max_suppression at IoU t over the conf-0.40 class-0 candidates is
  torchvision.ops.nms on those candidates in letterbox pixels, then max_det. R_pre holds every such candidate (no cap
  was hit), so NMS(R_pre, t) equals the build's NMS at t up to two known differences: IoU is computed in letterbox
  pixels before scale_boxes clips boxes to the frame (a translation and a uniform scale keep IoU; the clip does not), and
  float rounding at a threshold tie. Stage 0a checks this at t = 0.45, where the answer is known (R_post).
- **Not the build.** Only universe frames are stored (3,631 of the 38,727 processed frames of the two games), and the
  tracker needs every processed frame. BoT-SORT inside model.track also reads the image (its camera-motion
  compensation), and the foot stabiliser, fragment linker and team step read the tracks. So the stored candidates
  can screen an arm (stage 0b), but scoring through the pinned scorer needs the build itself rerun.

## 4. The tracker must rerun; cost

Yes. detect.tracker.PlayerTracker calls model.track with iou=config.PLAYER_IOU, so detection, NMS and BoT-SORT run in
one call per processed frame, and everything the scorer reads (the sidecar's tracked boxes, the identity file, the
trajectories) is downstream of it. The arm is a full window build per threshold, through the same path as the
composite builds: sportvu.rebuild's build subprocess routed through tools/p8_device.py with P8_DEVICE_DETECTOR=cpu
and P8_TRACKER_RESET_FIX=1, plus one new flag (section 11), P8_DETECTOR_IOU=t, which sets PlayerTracker's iou and is
written to the window's device record. No frozen file and no default changes.

The reset fix is on in every arm window, including windows that had no reset in the composite: a looser NMS may move
the camera-cut resets, and the fix is byte-neutral where no reset occurs (P8_resetfix.md, the equivalence windows).

Cost: 0.44 s per processed frame on the CPU detector (P8_resetfix.md section 8; the hou truth-only build ran at 0.94
of that estimate). gsw_cha_2016 has 18,083 processed frames in 73 windows and gsw_bkn_2015 20,644 in 83 (the composite
builds/r23 sidecars), so one arm on both games is about 4.7 lock hours. Section 7's ladder spends less.

## 5. Arms (fixed now)

Two arms, IoU threshold 0.55 and 0.70, everything else as built (conf 0.40, max_det 300, class 0, CPU detector, reset
fix). 0.70 is ultralytics' own default; 0.55 sits just under C2's median candidate IoU (0.559), so it keeps about the
lower half of C2 candidates by IoU, 0.70 more. No other threshold is tried in this item, and none is chosen after a
number (a failed arm gets no neighbour threshold).

## 6. Decision metric

The pinned scorer (tools.p8_scorecard.score_window, checked through tools.p8_pin on every field), the decided truths
(multipool v1 for both games, D199), each arm window paired with the same window of D202's record
(work_ref/rebaseline/score__<game>__current.json, run 20261127T171848Z). Rows, all over in-frame players:
- **missed** = missed / in_frame (D202's row);
- **ghost row** = nr_ghosts / in_frame (non-referee boxes with no player within 2.5 ft), on the same denominator as
  missed, so a fall in one and a rise in the other compare directly; ghosts per box is reported too, with its
  denominator, since more boxes change it;
- **net error** = (missed + nr_ghosts) / in_frame. "The ghost rise exceeds the missed fall" is exactly a net error rise;
- **one-to-one net error** = unmatched_1to1 / in_frame (D202's field), the duplicates guard: a nearest-within-2.5-ft
  rule lets a duplicate box next to a tracked player count as neither ghost nor miss, so the nearest rows alone can
  improve without anyone found (the duplicates count is reported).
Differences are arm minus baseline, pooled over windows as ratio of sums, with a window-cluster bootstrap (2,000
resamples, seed 0, tools.p8_resetfix.resample_picks over sorted windows, both games resampled together); per game
the same. Guards, reported with margins: team row (non-inferiority -0.01) and position p90 (+0.10 ft).

## 7. Ladder, gates, adoption and kill rules (fixed now)

**Stage 0a, replay check (no detector; under the lock "p8 nmsarm replay" for seconds, since it loads torch; it reads only
recallcause/ files).** On every stored frame of
gsw_cha_2016_s05_w00 (111 frames) and gsw_bkn_2015_s04_w00 (131), the two build-environment windows,
torchvision.ops.nms on R_pre at 0.45 then max_det 300 must equal the stored R_post: same count, one to one, every value
equal. Gate: exact on every frame of both windows. If it fails only on frames with an edge-clipped box, stage 0b leaves
those frames out and counts them; any other failure stops stage 0b (its fallback is a detector pass, which needs a
dated amendment).

**Stage 0b, screen (under the lock "p8 nmsarm stage0", about 10 minutes: it recomputes D205's rows, as D208 did,
to name each miss's player; no detector).** On D208's 88 kept windows, for each arm t:
- R(t): C2 misses (D208's rule, 8.3) whose own candidate survives NMS(R_pre, t);
- A(t): boxes NMS(R_pre, t) adds over R_post that are not such a candidate, split into those within 2.5 ft of a player
  who already has an R_post box (duplicates) and those with no in-frame player within 2.5 ft (ghost candidates).
- Gate G0, per arm: R(t) at least 0.25 of the 421 C2 misses (below that the ceiling on missed is under about 0.0025,
  not measurable on these windows); an arm under it is killed without a build. A(t) is reported, not gated: the
  tracker may drop or keep added boxes, and universe frames over-represent crowded frames.
- The recount of C2 must equal D208's 421, else stage 0b stops.

**Stage 1a, build identity (lock "p8 nmsarm identity", about 0.3 h).** Build gsw_cha_2016_s05_w00 (reset window),
gsw_bkn_2015_s04_w00 (reset window) and gsw_cha_2016_s00_w00 (no reset; built without the fix in the composite) with
P8_DETECTOR_IOU=0.45 into a fresh root. Gate: each window's sidecar boxes, identity and trajectories equal the
composite's on content (not file bytes), and scoring them reproduces D202's window rows on every field. A failure
stops the arm.

**Stage 1b, pilot (lock "p8 nmsarm pilot", about 2.4 h for both arms).** Pilot windows, fixed now: in each game's
windows sorted by name, every index divisible by 4 (gsw_cha_2016 19 windows, 4,445 frames; gsw_bkn_2015 21 windows,
4,713 frames). Both arms that pass G0 are built on them and scored. Per arm:
- P1: missed difference below 0 (point, pooled);
- P2 (kill): net error difference at or above 0 (point, pooled), that is, the ghost rise at least the missed fall: the
  arm is killed;
- P3 (power): the pilot's bootstrap half-width of the net error difference, scaled by the square root of 40 / 116 to
  the 116 confirmation windows, must be under the absolute pilot net error difference; else the arm stops as
  inconclusive (the full run could not show it).
If both arms pass, the confirmation runs for the one with the lower pilot net error difference (ties: 0.55). If none
passes, the item ends: "no arm passed the pilot".

**Stage 2, confirmation (lock "p8 nmsarm confirm", about 3.8 h).** The chosen arm on the 116 other windows
(gsw_cha_2016 54, 13,638 frames; gsw_bkn_2015 62, 15,931 frames). The decision uses these 116 windows only, so the arm
choice made on the pilot does not score itself; all 156 are reported beside.

**Adoption rule (stage 2, all must hold): ADOPTION CANDIDATE.**
- A1: the missed difference's 95 percent interval lies wholly below 0;
- A2: the net error difference's interval lies wholly below 0;
- A3: the one-to-one net error difference is at or below 0 (point);
- A4: in each game the net error difference is at or below 0 (point);
- guards: team difference at least -0.01, position p90 difference at most +0.10 ft (points).
"Adoption candidate" means only that the arm earns a registered held-back test on hou_cle_2016 in a later item; it is
not adopted into the build here.

**Kill rule:** at stage 1b or 2, a net error difference at or above 0 (pooled point) kills the arm: KILL, no other
threshold, no rescue run. **Inconclusive:** neither (for example A1 holds and A2's interval crosses 0): "not shown", no
further compute in this item.

## 8. Alignment

The decision metric needs no alignment rule. Arm and baseline are both builds made by the build itself on the same
clips in the same read order, differing only in the NMS threshold, and both are scored by the pinned scorer against
the same truth with the same time maps; the replay-to-sidecar alignment that D208 had to check (its replay decoded
the clips separately) does not enter. Stage 1a checks that the arm path reproduces the composite exactly.

Stage 0b reads D208's stored replay candidates, so its window set is D208's registered 88 kept windows. Posefoot
section 14's rule is not reused for the decision for the reason above. If posefoot's stage A verdicts exist when stage
0b runs, stage 0b is also reported on the windows posefoot calls aligned, beside the registered screen, never in place
of it; G0 is read on the 88 windows.

## 9. What would change the plan

- Stage 0a fails beyond edge-clipped frames: stage 0b needs a detector pass (a dated amendment, about 40 lock minutes,
  D208's cost).
- Both arms fail G0: the item ends with the screen; C2 is not recoverable by NMS alone at these thresholds.
- Stage 1a fails: the arm path does not reproduce the composite; the failing file is reported and nothing is built.
- The pilot passes P1 but fails P2 at both thresholds: duplicates cost more than C2 gains; a later design would need
  a different lever (for example class-aware or score-aware suppression), not a third threshold.
- An adoption candidate: a later item registers a hou_cle_2016 held-back test before any hou build.

## 10. Cost in lock minutes

| stage | what | lock |
|---|---|---|
| 0a | replay check on stored files | seconds |
| 0b | the screen on 88 kept windows | about 10 min |
| 1a | identity builds, 3 windows (2,590 frames) and scoring | about 20 min |
| 1b | pilot, 2 arms x 40 windows (9,158 frames each) and scoring | about 140 min |
| 2 | confirmation, 1 arm x 116 windows (29,569 frames) and scoring | about 230 min |

At most about 6.7 lock hours, 0 Colab units; about 2.8 hours if the item stops after the pilot. The build rate is
0.44 s per processed frame; scoring is about 4 minutes per 40 windows (D202 scored 156 windows in about 11 minutes).
A stage that runs past 1.5 times its estimate stops and is re-estimated.

## 11. Code to write after review

- tools/p8_device.py: the P8_DETECTOR_IOU flag (a number in (0.45, 1.0], or unset), applied by patching
  PlayerTracker.__init__'s iou in the build child, recorded as "detector_iou" in the device record; check_cached
  refuses a root that mixes thresholds. Tests: the flag parses, refuses bad values, and leaves the command unchanged
  when unset.
- tools/p8_nmsarm.py: replay (0a), screen (0b), build (1a, 1b, 2; fresh roots per arm under the P8 data folder, never
  builds/r23, which only the truth rebuild's compose writes), score (D202's scorer path, cache_gaps before each window,
  the processed frame set of each arm window equal to the baseline's so the scorecard cache applies) and report.
  Records: reports/research/p8_nmsarm_<stage>.json, each checked with find_private.
- tests/test_p8_nmsarm.py: re-NMS on synthetic boxes, the pilot split, the gates, the adoption, kill and
  inconclusive outcomes, the guards on game and window lists, and an AST check of the write targets.
