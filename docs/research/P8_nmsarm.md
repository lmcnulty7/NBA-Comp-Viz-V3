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

## 12. Version 1.1 (2026-11-27; the design review of 328853f0, GO WITH CHANGES; before any code)

Where this section differs from sections 1 to 11, it rules. Only stages 0a and 0b may run under it; stage 1 (1a on)
needs a further dated amendment after stage 0b's result, and its code is not written before then.

**12.1 The review's finding.** The reviewer's re-NMS on the stored candidates of the 88 kept windows added 766 boxes at
0.55 and 1,299 at 0.70, against at most 421 C2 recoveries (about 200 at 0.55), and many added boxes sit on the same
person. The nearest-rule rows (missed, ghosts) cannot see a duplicate next to a tracked player, so a pilot read on them
would favour the arm with most duplicates.

**12.2 Stage 0b gates (replaces section 7's stage 0b).** Per arm t, on the universe frames of D208's 88 kept windows,
against re-NMS at 0.45 as the baseline (equal to the stored R_post wherever stage 0a's check holds; agreement is
counted on every frame), the change in:
- **one-to-one net** (the primary row): tools.p8_resetfix.unmatched_1to1, imported, the count D202 and D205 checked
  against the scorer's own counts; the raw boxes are r23's player class (class 0), so none is a referee box;
- **missed plus ghosts**: missed counts in-frame players with no box foot within sportvu.eval.MISS_FT; ghosts count
  boxes with no player of all ten within sportvu.eval.GHOST_FT (3.0 ft), as tools.p8_scorecard.score_window counts
  them; both constants are imported, not typed;
- **duplicates**: tools.p8_resetfix.frame_extra's count (in-frame players with two or more feet within 2.5 ft).
Feet are the raw boxes' bottom centres through H_truth (p8_recallcause.feet_of), as D208 did; in-frame uses
sportvu.eval.FRAME_MARGIN_PX on the clip's size; the players and H_truth come from D205's rows, recomputed and checked
equal to D205's stored rows per window, as in D208. Gates, per arm:
- K1: an arm whose pooled one-to-one net change is not below 0 is killed;
- K2: 0.70 is kept only if its one-to-one net change is below 0.55's (its marginal change over 0.55 is negative);
- G0 (section 7, unchanged): R(t) at least 0.25 of the 421 C2 misses;
- the C2 recount must equal D208's 421, and the D205 row gate must pass on each of the 120 windows with universe
  misses (the windows with stored candidates), else stage 0b stops.
At most one arm goes on: 0.70 if it passes K1, K2 and G0, else 0.55 if it passes K1 and G0, else none and the item
ends. Reported beside, not gating: the same rows on the 32 dropped windows, a window-bootstrap interval for each
pooled change, and R(t) and A(t) as in section 7.
Limit, read with it: universe frames are the frames with at least one P-i or D-i miss. Other frames have none of
the misses C2 can recover and still pay for the added boxes, so the screen leans in the arm's favour: a kill is safe;
a pass does not show that the build gains.

**12.3 If an arm survives (replaces parts of sections 6 and 7).** The one-to-one net error (unmatched_1to1 /
in_frame) is the primary row for P2, the kill rule and A2; missed, ghosts and net error are reported. With one arm
the arm choice is moot. Added:
- a non-inferiority guard on duplicates per in-frame player: the difference at most +0.005 (D202's baseline is 0.0516
  over all 156 windows, so about a tenth of it);
- a mechanism check: the in-frame players hit in the arm and missed in the baseline, counted by D205's class and,
  inside D208's universe, by D208's class. A missed fall larger than the measured ceiling (12.7), or arm-only hits
  carried mostly (more than half) by P-ii, is flagged "duplicates suspected"; a flagged result is not an adoption
  candidate.

**12.4 Stage 1a (replaces section 7's stage 1a; not coded until stage 1 is amended).** The identity builds at 0.45 on
the three windows of section 7 plus one D208-dropped window that did not fail on a tie, gsw_cha_2016_s02_w01 (a
neighbour above f, five alignment frames, 77 misses, 208 frames, no reset). A positive control: gsw_cha_2016_s05_w00
at 0.55 must differ from the composite and its device record must carry detector_iou 0.55. Unit tests: model.track
receives iou=t on the first call and after PlayerTracker.reset(); the iou patch applies whether or not a device flag
is set.

**12.5 Allow-list.** ALLOWED = {gsw_cha_2016, gsw_bkn_2015} is asserted on every game name, every window name and every
score call, besides tools.p8_common.guard and refuse_cv_test (guard allows hou_cle_2016 and gsw_sac_2015 today).
Tests show that a hou or sac game or window is refused.

**12.6 Noise floor.** Unknown before a pilot, and expected to be dominated by tracker churn: any NMS change perturbs
BoT-SORT everywhere (activations, identities, resets), so per-window differences carry churn beyond the C2 mechanism.
The build itself is deterministic (P8_resetfix.md, the equivalence windows), so there is no run-to-run noise. P3
(replaces section 7's): the pilot's bootstrap half-width of the one-to-one net error difference, scaled by the square
root of n_pilot / n_confirm, must be under 0.7 times the absolute pilot difference on that row.

**12.7 Ceiling (replaces section 2's figure).** Measured: D208's 421 C2 misses on the kept windows, 0.0073 of the
57,797 in-frame players of both games (D202's record, all windows), and 0.0098 of the 42,801 in-frame players of the
88 kept windows. Section 2's 0.010 was an extrapolation of shares to all windows; the measured count stands instead.
C2 in the 32 dropped windows is not measured.

**12.8 Windows with no in-frame player.** 35 of the 156 windows have in_frame 0 and no scored box in D202's record (no
truth frame with a player in frame), so the scorer reads nothing there and an arm cannot change their rows. Stage 1
neither builds nor scores them. The split of section 7 is recomputed on the 121 scorable windows (every index divisible
by 4 in each game's sorted list): pilot gsw_cha_2016 14 of 56 windows (4,603 frames), gsw_bkn_2015 17 of 65 (6,320);
confirmation 42 (12,078) and 48 (12,325).

**12.9 Truth-bias limit.** D202's limit holds here with more force: the truth frames were selected with r23's own
boxes at NMS 0.45, so the rows lean toward r23 as built. An arm differs from the baseline exactly in its boxes, and its
score against a truth chosen with the baseline's boxes may favour the baseline where the two differ; neither the size
nor the direction of that effect is measured. Stage 0b's feet also use H_truth.

**12.10 Cost (replaces section 10).** Stage 0a: seconds of lock. Stage 0b: about 10 lock minutes (D205's recompute,
as in D208; no detector). If an arm survives and stage 1 is amended: 1a about 25 minutes (four identity windows,
2,798 frames, and the positive control, 1,053 frames); pilot of one arm about 85 minutes (10,923 frames); confirmation
about 190 minutes (24,403 frames). At most about 5.0 lock hours, 0 Colab units.

**12.11 The code check of ec68f896 (2026-11-27; GO WITH CHANGES; before any run).** Changes, none of which moves K1,
K2, G0 or the recount:
- raw boxes get the scorer's referee rule (a box overlapping a scorecard-cache referee detection at IoU at least
  sportvu.bench.REF_IOU, as d205_rows does when the sidecar has no class; a sidecar with classes stops the run), so the
  one-to-one and ghost rows equal the scorer's; the flagged boxes are counted per threshold;
- recallcause/ is in the watched manifests, and its digest is written into both records; the screen stops if it changed
  since the replay;
- C2 is counted on every frame; a frame that stage 0a found differing only at the frame edge (the differing boxes and
  the candidates overlapping them touch the edge) is left out of the arm rows, R(t) and A(t) only;
- the screen requires the replay record's head to equal the current HEAD, at IoU 0.45 and max_det 300; both stages
  assert torch 2.2.2 and record torchvision;
- both stages run tests/test_p8_nmsarm.py under the interpreter that runs them (the frozen CV python, so the NMS test
  runs), require it to pass with nothing skipped, and record the result.


## 13. Stage 0 result (2026-11-27; run at 18a13d7b; reports/research/p8_nmsarm_replay.json and p8_nmsarm_screen.json)

**13.1 Stage 0a, the replay.** One lock job ("p8 nmsarm replay", 42 s). torchvision NMS on the stored R_pre at 0.45
with max_det 300 equals the stored R_post on every stored frame: gsw_cha_2016_s05_w00 0 of 111 frames differing,
gsw_bkn_2015_s04_w00 0 of 131. The stage ran tests/test_p8_nmsarm.py under the frozen CV python first (11 passed, none
skipped, the NMS test included). torch 2.2.2, torchvision 0.17.2. Passed.

**13.2 Stage 0b, the screen.** One lock job ("p8 nmsarm stage0", 132 s), after the same unit tests (11 passed). The
D205 row gate passed on all 120 windows with stored candidates; the C2 recount on the kept windows is 421, equal to
D208's. Re-NMS at 0.45 equals the stored R_post on 3,630 of 3,631 universe frames (the one other frame lies outside the
two check windows; it stays in the rows, scored against re-NMS at 0.45 like every frame). The scorer's referee rule
flags 563 raw boxes at 0.45, 568 at 0.55 and 571 at 0.70 (all 120 windows). Baseline on the kept windows' 2,642 universe
frames: in-frame players 25,979, one-to-one net 13,306, missed plus ghosts 11,001, duplicates 1,248.

Changes, arm minus re-NMS at 0.45, on D208's 88 kept windows (sums over universe frames; window-bootstrap 95 percent
intervals, 2,000 resamples, seed 0):

| arm | one-to-one net | missed plus ghosts | duplicates | missed | ghosts | boxes | C2 recoverable | added: duplicates, ghost candidates, other |
|---|---|---|---|---|---|---|---|---|
| 0.55 | -134 [-215, -55] | -222 [-298, -150] | +175 [+119, +236] | -313 | +91 | +540 | 218 of 421 | 273, 133, 143 |
| 0.70 | -14 [-93, +65] | -274 [-367, -192] | +420 [+300, +549] | -486 | +212 | +1,078 | 322 of 421 | 491, 253, 227 |

Gates as registered (12.2): 0.55 passes K1 (-134 below 0) and G0 (218 at least 0.25 of 421); 0.70 passes K1 (-14)
and G0 (322) but fails K2 (-14 is not below 0.55's -134). **The arm that goes on is 0.55.**

Beside, not gating: the 32 dropped windows (989 universe frames):

| arm | one-to-one net | missed plus ghosts | duplicates | missed | C2 recoverable |
|---|---|---|---|---|---|
| 0.55 | -56 | -98 | +46 | -139 | 89 of 151 |
| 0.70 | +14 | -106 | +125 | -195 | 120 of 151 |

**13.3 Reading, with its limits.**
- The screen leans in the arm's favour: it reads only universe frames (frames with a P-i or D-i miss, where C2 can be
  recovered) and raw boxes before any tracker. Its pass shows nothing about the build; stage 1 decides that.
- Mechanism-check warning (registered for the pilot, 12.3; not a finding): at 0.55 the missed count falls by
  313, more than the 218 recoverable C2 misses, so about 95 of the fall comes from players outside C2. That fits added boxes
  landing near P-ii players (a box 2.5 to 5 ft away already) rather than finding lost ones. Duplicates rise by 175
  [119, 236]. The pilot's mechanism check (arm-only hits by class) tests this.
- On the one-to-one row 0.70 gains almost nothing over the baseline (-14, interval across 0): its extra boxes beyond 0.55
  are mostly duplicates (+420 against +175).

## 14. Version 1.2: stage 1a and the pilot, 0.55 only (2026-11-27; after stage 0, before any stage 1 code or lock)

Where this section differs from sections 7, 10 and 12, it rules. The confirmation (stage 2) is not written; it waits
on the pilot's gate and its own amendment.

**14.1 The build path.** tools/p8_nmsarm_stage1.py builds windows through sportvu.rebuild's build subprocess
(tools.p8_frozen.safe_build with sportvu.autolabel.gate_args, as tools/p8_resetfix.py does), its subprocess rewritten to
a child of this module when P8_NMSARM_DETECTOR is set. The child refuses any device setting other than
P8_DEVICE_DETECTOR=cpu with P8_TRACKER_RESET_FIX=1 (the reference platform, D175), applies tools.p8_device's device
patch, then its own IoU patch: PlayerTracker.__init__ is wrapped to pass iou=t, whatever other wrapper is in place,
so the patch does not depend on a device flag. The device record carries "detector_iou": t, and the detector record
r23's SHA-256. Outputs go to new folders of the reference root's builds/ (nmsarm_iou045_identity,
nmsarm_iou055_control, nmsarm_iou055_pilot); builds/r23 is never written. Every window name passes the allow-list.

**14.2 Stage 1a.** Identity: gsw_cha_2016_s05_w00, gsw_bkn_2015_s04_w00, gsw_cha_2016_s00_w00 and gsw_cha_2016_s02_w01
at t = 0.45. Gate, per window: the sidecar equal to the composite's as canonical JSON without "source", the identity
and trajectories files byte-identical (tools.p8_resetfix.compare_window's checks), and the scored rows (score_pair)
equal to D202's record on every field of tools.p8_rebaseline.GATE_FIELDS. Positive control, after the identity gate:
gsw_cha_2016_s05_w00 at t = 0.55. Gate: its device record says 0.55, its sidecar differs from the composite's, and its
sidecar holds more boxes than the composite's. A failure stops the arm.

**14.3 The pilot (0.55 only).** The 31 pilot windows of 12.8 (index divisible by 4 in each game's sorted list of
scorable windows), built at 0.55 after stage 1a passes, then scored: per window, cache_gaps on the arm folder finds
nothing, the arm's processed frames equal the composite's, score_pair gives both sides on the same scored frames, and
the baseline side equals D202's record on every GATE_FIELDS field. Rows over the 31 windows, both games pooled,
arm minus baseline with a paired window bootstrap (2,000 resamples, seed 0; tools.p8_rebaseline.change_rows, which
also gives team and position p90), plus duplicates per in-frame player.

**14.4 Pilot gate (fixed now).**
- P1: the missed difference below 0 (point);
- P2, the kill rule: the one-to-one net error difference at or above 0 (point): KILL;
- P3: the one-to-one net error difference's bootstrap half-width times the square root of 31 / 90 under 0.7 times its
  absolute value; else the arm stops as inconclusive;
- duplicate guard: duplicates per in-frame player up by at most 0.005;
- mechanism check, flag "duplicates suspected" when either holds: the missed fall per in-frame player exceeds 0.0098
  (the larger measured C2 share, 12.7), or more than half of the arm-only hits (players missed in the baseline and hit
  in the arm on the same scored frame) are D205 class P-ii;
- PASS only when P1, P3, the guard and the check all hold and P2 does not kill; a pass allows the confirmation's
  amendment and nothing else.
Reported beside: arm-only hits and arm-only misses by D205 class, and, among P-i and D-i arm-only hits on frames with
stored candidates, how many are D208's C2 (12.2's rule on those candidates); team and position p90 differences;
ghosts per box with its denominator.

**14.5 Cost.** Stage 1a: about 25 lock minutes (4 identity windows, 2,798 frames; the control, 1,053 frames; scoring).
Pilot: about 85 minutes (10,923 frames) plus about 5 minutes of scoring. Each stage stops at 1.5 times its estimate.

**14.6 The code check of abffaf2e (2026-11-27; GO WITH CHANGES; before any stage 1 run or pilot number).** Where this
differs from 14.1 to 14.5, it rules.
- **The mechanism ceiling (replaces 14.4's 0.0098; the lead's decision).** The ceiling is the count of C2 misses
  recoverable at 0.55 on the 31 pilot windows (12.2's rule on the stored candidates, kept and dropped windows, every
  universe frame), over the pilot's in-frame players. cmd_score computes it per window and records it; the gate reads it
  from that record. Reason: 0.0098 counted C2 misses recoverable at any threshold, not at the arm's. Disclosed: the
  reviewer saw in stage 0's per-window rows that the pre-tracker missed fall on these 31 windows is about 0.0067 per
  in-frame player, above the expected new ceiling (about 0.005). This amendment is made knowing that, on definitional
  grounds, before any pilot build, score or outcome exists.
- **Watches.** builds/r23 and the labels, label_build, truth_accept and cache folders of the reference root and of
  work_ms are hashed before and after every build and score step; any change stops the step and nothing is recorded.
- **Stale records.** The identity and control records a later step needs must carry the current HEAD; the gate re-checks
  the stage order and requires the pilot record's head to equal HEAD; a window already present in a stage's folder, or
  returned as cached by the build, is refused (a restart moves the folder aside first).
- **Caps.** The build cap (1.5 times 0.44 s per processed frame) is checked before every attempt, retries included;
  scoring has its own cap at 1.5 times its estimate (identity about 2.5 minutes, control 0.5, the pilot about 11: the
  pilot scores each side twice, once through score_pair for the rows and once through score_window for the
  observations, since score_pair does not return them).
- All four identity windows and the control stay (the lead's decision).


## 15. Stage 1 result (2026-11-27; run at 8929a031; reports/research/p8_nmsarm_identity.json, _control.json, _pilot.json and _pilot_gate.json)

**15.1 Stage 1a.** Identity at 0.45 (build 1,207 s against an estimate of 1,231 s; scoring 36 s): all
four windows (gsw_bkn_2015_s04_w00, gsw_cha_2016_s00_w00, gsw_cha_2016_s02_w01, gsw_cha_2016_s05_w00) equal the composite (sidecar as
canonical JSON, identity and trajectories byte for byte), and both scored sides equal D202's record on every gate
field. Positive control at 0.55 on gsw_cha_2016_s05_w00 (build 460 s): its device record says 0.55 and its
sidecar differs, holding 8,518 boxes against the composite's 8,371. Both gates passed: the arm path reproduces the
composite at 0.45 and moves the boxes at 0.55.

**15.2 The pilot (0.55).** 31 windows built in 5,131 s (estimate 4,806 s, ratio 1.07) and scored in
134 s; every baseline side equals D202's record. Arm minus baseline over 15,656 in-frame players, paired window
bootstrap (2,000 resamples, seed 0):

| row | change | 95 percent interval |
|---|---|---|
| missed | -0.0065 | [-0.0092, -0.0034] |
| detection (no foot within 5 ft) | -0.0050 | [-0.0067, -0.0031] |
| placement | -0.0015 | [-0.0034, +0.0004] |
| ghosts per in-frame player | +0.0033 | [+0.0012, +0.0056] |
| ghosts per box | +0.0016 | [-0.0003, +0.0037] |
| net error (nearest rule) | -0.0031 | [-0.0069, +0.0007] |
| one-to-one net error (primary) | -0.0017 | [-0.0059, +0.0026] |
| position p50 (ft) | -0.0031 | [-0.0104, +0.0083] |
| position p90 (ft) | -0.0046 | [-0.0241, +0.0142] |
| near-field bias (ft) | +0.0003 | [-0.0119, +0.0164] |
| team | +0.0151 | [-0.0129, +0.0445] |
| duplicates per in-frame player | +0.0036 | [+0.0019, +0.0050] |

**15.3 The gate, as registered (14.4 with 14.6).**
- P1, missed falls: yes (-0.0065).
- P2, kill: no; the one-to-one net error change is -0.0017, below 0.
- P3, power: fails. The projected half-width on the 90 confirmation windows is 0.0025, above 0.7 times
  |-0.0017| = 0.0012.
- Duplicate guard: holds (+0.0036, at most +0.005).
- Mechanism check: the flag fires. Missed falls by 0.0065 per in-frame player against a ceiling of
  0.0038 (59 C2 misses recoverable at 0.55 over 15,656 in-frame players). The P-ii share of arm-only
  hits is 66 of 192, under half, so that part of the check does not fire.

**Outcome: STOP.** The arm stops at the pilot: no confirmation, no other threshold and nothing rerun (the protocol:
never scale up to rescue a failed gate). hou_cle_2016 stays unspent.

**15.4 Reading.**
- The missed fall is larger than suppression recovery can explain. Arm-only hits are 192 (P-i 69, P-ii 66, D-i 55, D-iii-free 2), against
  91 arm-only misses. On frames with stored candidates, 52 of the P-i and D-i arm-only hits are D208's C2.
  Of the 124 P-i and D-i arm-only hits, the classes C2 belongs to, 52 are C2, and P-ii players (a box 2.5 to 5 ft away
  already) gain almost as many (66). The reading, post hoc and not separated: part of the fall comes from tracker or
  duplicate effects (a second box near a player, or a track that moves), not from recovered suppressed candidates.
- Net error is not shown to fall: the one-to-one row -0.0017 [-0.0059, +0.0026] and the nearest-rule row -0.0031
  [-0.0069, +0.0007] both span 0, while ghosts per in-frame player rise (+0.0033 [+0.0012, +0.0056]) and duplicates
  rise.
- The screen's pass did not carry to the build, as 13.3 warned: before the tracker, on universe frames only, the
  one-to-one net fell clearly; through the tracker, on all scored frames, it does not.
- The team row (+0.0151 [-0.0129, +0.0445]) is reported and not read: its interval spans 0.
- Each number is on the 31 pilot windows of two games against a truth selected with r23's boxes at 0.45 (12.9).

**15.5 Cost.** Lock time (the lock's own durations): stage 0 174 s (replay 42, screen 132); stage 1a 1,841 s (builds
1,742, scoring 99); the pilot 5,326 s (build 5,170, scoring 156); 7,341 s, about 2.0 lock hours, 0 Colab units.
