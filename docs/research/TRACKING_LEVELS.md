# Broadcast tracking: levels of claim and a first-draft plan

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
