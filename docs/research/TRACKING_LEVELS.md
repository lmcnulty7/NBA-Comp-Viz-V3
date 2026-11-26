# Broadcast tracking: levels of claim, living roadmap

The lead revises this file at every loop iteration (PLAN.md D172): status per step with its D line, the current
route, a ranked queue of open bets, and the lessons that set the priors for the next design. The first draft of
2026-11-20 is kept unchanged at the end, so every change of plan can be traced against it. Every training step
follows docs/research/TRAINING_PROTOCOL.md.

## Status on 2026-11-25

Target (D172, Lucien): Levels 1 to 3. Level 4 is closed (D159) and on watch: a cheap weekly check for a legitimate
source of uncut modern broadcasts, reopened only if one appears.

| Level | State | What blocks it now |
| --- | --- | --- |
| 1, generalizes | Open | Footage: 8 synced 2015-16 broadcasts outside the spent held-out set, 5 arenas (P8_step0.md), most already used for development; the Mac and Colab builds not yet compared (D161 (1)) |
| 2, complete player tracking | Open | No detector beats r23 yet: E1 failed on misses (D171); team labels from embeddings failed (D154) |
| 3, analytics-grade | Not started, apart from ball probes | Identity and the ball: WASB's released ball model is weak here (E7a pooled F1 0.1807, D165) and E7b could not run as released (D166 (2)) |
| 4, new seasons | Closed, on watch | Public 2025-26 uploads are edited, so clock placement fails (D157, D159); League Pass is excluded (D128) |

### Level 1 steps

| Step (first draft) | Status | Record |
| --- | --- | --- |
| 1. Colab yes | Done; Colab is driven from Claude Code, cap 60 units for Levels 1 and 2 | D137, D164, D168, D169, D170 |
| 2. One-cell Colab check against the Mac scorecard | Open, next on Colab | D161 (1) |
| 3. Timing alignment | Changed: truth v2 failed its bar, the frozen offset rule stays; the play-by-play fallback is not built | D166 (1) |
| 4. More synced footage | Done as a search, short of the goal: 10 confirmed full broadcasts, 2 new games, 1 new scored arena (hou_cle_2016) | P8_step0.md, D146, D153 |
| 5. Frame gate without held-out frames | Open | |
| 6. Pre-register the Level 1 test | Open; when sources run dry, seal what exists with its detectable effect stated in advance (D172) | D172 |
| 7. Cross-arena accuracy on development games | Open | |
| 8. Score once on the sealed set | Open | |

### Level 2 steps

| Step (first draft) | Status | Record |
| --- | --- | --- |
| 1. Error analysis by cause | Partly done: baseline rows on 3 training games (missed 0.26, ghosts 0.118 to 0.129); the breakdown by cause is the E1 post-mortem | P8_baseline.md, P8_E1.md |
| 2. Detector labels from outside teachers | Done: YOLOX-X on SportsMOT qualified (+0.0419 recall at our precision); E1 corpus 52,478 labels on 5,592 frames | P8_teachers.md, D155 |
| 3. Fine-tune the detector | Paused: E1 not adopted (D182); E1b stopped at its transfer gate before any unit (D183): the reachable misses are mostly foot placement, which survives the build at about 0.3 | D171, D182, D183, P8_E1.md, P8_E1R.md, P8_E1B.md |
| 4. Team labels from appearance | Failed: CLIP and DINOv2 lower the team row 14 to 21 points; colour stays | D153, D154, tracking.md |
| 5. Near-camera bias correction | Step 1 measured (D184): the foot offset grows along the camera ray and varies by game; no correction fitted on two games passes on the third | D184, P8_placement.md |
| 6. Fill missed player-frames | Waits on step 3 | |
| 7. Score once | Open | |

### Level 3 steps

| Step (first draft) | Status | Record |
| --- | --- | --- |
| 1. Identity from jerseys (E4) | Not started; needs a GPU cap (D172) | |
| 2. The ball | Probed: WASB released weights, pooled F1 0.1807 on projected truth (E7a); E7b stopped (2 frames per view, WASB needs 3); human ball labels on broadcast need the NUST-NBA181 videos, which need Lucien's email | D165, D166 (2), D163 (3) |
| 3 to 5. Data format, equivalence test, score | Open | |

## Current route and queue

Ranked by expected information per unit of cost; the lead re-ranks it each iteration.

1. Placement route (D183 to D186), 0 units: the missed row's placement part has no correction that passes on
   every game (step 1 SCATTERED; step 1b MIXED, gsw_cha_2016's truth suspect; the short-foot bias under the r23
   court's H is consistent but correcting it raises missed and ghosts on gsw_sac_2015). On the position rows the
   same shift lowers p50 by 0.25 to 0.28 ft and near-field bias by about 1 ft on every pool game (D187). Now: the
   shift on output positions only (detection unchanged), design reviewed before hou_cle_2016 is scored once. The missed row is reported as detection (no foot
   within 5 ft, 0.052 to 0.091) and placement (2.5 to 5 ft, 0.15 to 0.22); whether Level 2's 5 percent target should
   be restated against the split is Lucien's call.
2. Camera truth for okc_sac_2015 on the reference platform (D178, running): bars A, B, C fixed before the build.
   nyk_bos_2016 and por_okc_2016 stay out (hand-measured layouts). Then the gsw_cha_2016 CPU truth rebuild to test
   whether the existing MPS truth moved; if it moved, rebuild the other in-use truth before any cross-arena tuning.
3. Court output against camera truth: done (D181), no change; court device stays as in D175.
4. Colab as a build helper: only through a tested split route (detector on CUDA, court on the reference device),
   since CUDA matches neither Mac build (D179). Transfers go through Google Drive (D177).
5. Widen the fixed-MPS parity check (P8_MPS_CLIP_FIX, D174 (5)) to more windows and games: a 3x cost saver for
   exploratory builds if it passes a new registered rule.
6. Level 2 error analysis by cause on development games, then the near-camera bias correction (Mac).
7. Footage at low cost, continuously; NUST-NBA181 waits on Lucien's email (D163 (3)).
8. Level 4 watch, weekly.
9. Level 3 (E4 identity, E7 ball) once Level 2's detector question settles; it needs a cap proposal for GPU.

## Lessons and priors

- Measure the platform before the model. On this Mac, ultralytics 8.4.48 on MPS pins the x of the most confident
  boxes to 720 px (D174), for r23 and E1 alike. With only the detector moved to CPU, E1 minus r23 on missed turns from
  +0.0330 to -0.0147 on gsw_bkn_2015 (M6), and E1's raw CPU boxes miss fewer players than r23's (M4); the other games
  are not rebuilt, so how much of the registered rise the fault carries is the leading explanation, not a
  measurement (E1R measures it). Every new detector or library version gets a device parity check on the corpus
  JPEGs (minutes) before any build is scored.
- Check what a measure measures before reading its size. The court gap of D179 (3.8 to 10.6 px between platforms) used
  three fixed court points, and on 90 to 100 percent of truth frames two of them are off the image, so it largely
  measures extrapolation; on-frame line errors differ by under 1 px (D181). A build's platform is still recorded per
  window.
- A remote runtime loses its outputs if it is not polled; chain the result upload into the runtime job itself and poll
  every 5 minutes, even mid-conversation (D179, about 1.1 units lost).
- Agreement pseudo-labels (two models must both box a player) can teach unlabelled players as background; for E1 the
  label audit found this explains only a small part of the rise (D173).
- A pass on the development pool can fail in a new arena: E1 seed 0 passed both GSW pools but only 2 of 3 folds on
  hou_cle_2016 (D182). Keep a non-GSW gate in every detector design. Seed 0 point estimates favoured E1 on ghosts and
  position everywhere while missed barely moved (post hoc): the label recipe, not the platform, is the next suspect.
- Measure transfer to the scored row before sizing a label set: E1's raw gains kept only 0.24 to 0.38 (placement)
  and 0.36 to 0.62 (presence) at build level, so a label set with a -0.080 raw ceiling was worth about -0.020 at
  build level (D183). The cheapest falsifier was a per-game ceiling from stored files, seconds of compute.
- Most misses are not missing boxes: on the reference platform 64 to 75 percent of r23's build-level misses have a
  stabilised foot 2.5 to 5 ft away, offset along the camera ray (D184). Split detection from placement before
  choosing a fix; a detector cannot move the placement row.
- Name which build a truth came from before reading a disagreement with it: camera truth was initialised from and
  fitted to the V3 label build, not the scored r23 court, and a review caught a doc that assumed otherwise (D185).
- A validation metric computed on the training labels (E1: mAP50 0.99) says nothing about accuracy against SportVU.
- E1's design had a ceiling near its noise floor: only 1,102 of 55,147 in-frame players (0.020) were new positives,
  while the mean-box rule threw away most of the 4,081 foot-placement corrections. A blind design review on
  2026-11-25 (training-design-reviewer, no outcome read) called it NO-GO for that reason, no pilot, and validation
  on pseudo-labels. Next designs start with a label oracle: score the labels themselves with the scorer's rules.
- Appearance embeddings (CLIP, DINOv2) are worse team features than colour on these broadcasts (D154).
- A flat objective makes a clock offset unstable across builds; reshaping it helped but not enough (D166).
- Public modern uploads are edited, so the game clock cannot place plays in them (D157, D159).
- Colab GPU runtimes are reclaimed about 30 minutes after the last client call, even with a running cell; poll every
  8 minutes or less (D171). Measured cost about 1.15 units a T4-hour or less.

## First draft (2026-11-20), kept as written


Written 2026-11-20. A working draft, not a pre-registration: every test below gets its own pre-registration in
`docs/research/tracking.md` before it runs. Gates mark the points where a finding decides the next step, so the
plan is expected to change as it runs.

Today's numbers are from the README scorecard section (`reports/scorecard/<game>__r23.txt`): on the held-out game
the median position error is 1.92 ft and the ninetieth percentile 4.98 ft, coverage is 99.3%, 25.1% of players in
view are missed, 15.1% of boxes sit on no player and team labels are 81.9% right; the two held-out arenas sit at
about 2.2 / 5.5 ft, and one of them has 39.7% coverage.

| Level | Claim it would support | Milestones | Main blocker |
| --- | --- | --- | --- |
| 0, today | A measured illustration: 1.92 ft median error on one held-out game, but a quarter of the players missed | Done | |
| 1, generalizes | Position accuracy holds across arenas and seasons, reproducibly | Position target met in every held-out arena; 10 or more scored held-out games with intervals; same numbers on the Mac and on Colab | The P8 platform gap; more broadcast footage synced to SportVU |
| 2, complete player tracking | It finds nearly every player and team | Missed players and boxes on no player each 5% or less; team labels 95% or more; near-camera bias within 0.5 ft | Detector and team models need retraining on a GPU |
| 3, analytics-grade | The site's models give the same answers on broadcast tracking as on SportVU | Stable player identity; ball tracking; the frozen models agree within stated margins on the same games | Identity and ball are the hardest unsolved parts |
| 4, a stand-in for new seasons | Tracking analytics for seasons nobody sells tracking for | Current-season broadcasts run; checked against the partial truth that exists (public shot locations, play-by-play) | No full truth exists for modern games |

Rules for every step: the test is written down before it runs; a sealed set is scored once; heavy GPU work runs
on Colab and heavy CPU work on the Mac one job at a time; a review comes before and after every score.

## Level 1: it generalizes

1. Lucien's yes for Colab work. Every step below needs it.
2. The one-cell Colab check: does Colab reproduce the stored scorecard (1.92 / 4.98 ft on the held-out game)?
   Gate: if yes, Colab is the reference platform; if not, find the cause (video codec, detector version) first.
3. Fix the video-to-data timing alignment (PLAN.md D100) and measure its error on games where it is already known.
   Gate: if the clock-reading solve stays fragile, add a fallback that aligns on play-by-play events.
4. Harvest more footage: 2015-16 games that have both SportVU and a findable broadcast, about 30. Split them into
   development games for tuning and a new sealed set of 10 or more games across 5 or more arenas, hashed before
   any score. Gate: if fewer games can be found, use what exists and state the power in advance.
5. Retrain the frame gate without frames of any held-out game, which removes a known leak.
6. Pre-register the Level 1 test: position error at most 2.0 ft median and 5.0 ft at the ninetieth percentile in
   every arena, with game-bootstrap intervals, and coverage of at least 50%.
7. Improve cross-arena accuracy on development games only: more arenas in the court model's SportVU-derived
   labels, and the shelved camera model tested again. Gate: if one arena stays bad, diagnose it (lens, camera
   height) and iterate on development games, never on the sealed set.
8. Score once on the sealed set and publish the scorecard.

## Level 2: complete player tracking

1. Error analysis on development games: missed players by cause (blocked from view, far and small, camera cut)
   and boxes on no player by cause (referees, bench, crowd). Gate: the largest cause picks the fix (retrain the
   detector, change the tracker, or filter).
2. Detector labels: SportVU positions projected into the frames, on development games only. Teachers are outside
   models, never the pipeline's own output (LABEL_SCHEMA.md); a sample is checked by hand. Gate: if projected
   boxes are poor far from the camera, add a public detector trained on SportsMOT as a second teacher.
3. Fine-tune the detector on a Colab GPU, pre-registered against the current detector: missed players and boxes on
   no player each 5% or less.
4. Team labels (tracking-E1): clusters of each track's appearance within each possession, 95% or more right.
   It needs the synced games of Level 1 step 4.
5. Near-camera bias: a learned correction of where the feet are placed, by depth into the court, within 0.5 ft.
6. Fill missed player-frames (tracking-E2), only after step 3. Gate: if missed players are already 5% or less,
   drop it or reframe it.
7. Score once on the sealed set.

## Level 3: analytics-grade

1. Player identity: jersey numbers read from crops by a vision-language model, matched to the rosters and
   substitutions in play-by-play. Measured against SportVU player ids, with identity switches counted densely,
   not as today's sparse lower bound.
2. The ball: a ball detector trained on SportVU-projected ball labels. Gate: if it fails in crowded frames, infer
   who holds the ball from player positions and play-by-play events instead.
3. The data format: check that broadcast output fills what each frozen model needs (for example, the shot model
   needs the shooter and the defenders at release).
4. Pre-register an equivalence test: on the same games, each frozen model's metric from broadcast tracking must
   sit within a stated margin of its metric from SportVU. Gate: for a model that fails, add broadcast-like noise
   to SportVU to find which input error drives it, then go back to that Level 2 or Level 3 component.
5. Score once on the sealed set.

## Level 4: a stand-in for new seasons

1. Find current-season broadcasts. Gate: if only highlights or condensed games exist, limit the claims to the
   sampled possessions.
2. Check how much broadcasts have changed since 2016, against partial truth: the public shot coordinates against
   where the pipeline puts the shooter at the shot, and play-by-play consistency (who shot, made or missed, when).
   Gate: past a pre-set margin over the 2015-16 error, retrain on modern frames anchored to the shot locations.
3. Run the models and publish modern-season results, each with error bars carried from Level 3.
4. Turn on the visitor beacon, so that the reach of the site can be measured.

## Limits on every level

- One broadcast camera gives about 1.5 ft median error at best, against about 1 inch for an arena camera system
  (README), so matching the vendor's accuracy is never a claim this project can make.
- Ground truth exists only for 2015-16 (SportVU), so every accuracy number beyond that season rests on partial
  checks.

## In short

Level 1 is mostly operations (Colab, a reliable timing alignment, more synced games, a new sealed set). Level 2
is the main modelling work (detector and team labels), led by the error analysis. Level 3 is the claim that
matters, that the site's models give the same answers on broadcast data, and it needs identity and the ball;
failures there go back to Level 2. Level 4 needs modern footage and can only be checked against partial truth.
What needs Lucien: Colab and GPU time, finding the footage, and any compute spend.
