# P8 Level 2 map: where each row stands (2026-11-27, D202 to D214)

A synthesis for one decision: which Level 2 targets to keep, and whether to restate the 5 percent missed target
(D184 (b) put that question to Lucien). Nothing here is new. Every number is quoted from a D line in PLAN.md or from a
committed record, named beside it. The last section is reasoning and is labelled so.

## 0. What the rows are

- **Rows.** These are D202's development standard rows (D193 (2)): r23 on the fixed tracker, gsw_cha_2016 and
  gsw_bkn_2015 on the multipool v1 truth. They are not the reported baseline (D198). Record:
  reports/research/p8_rebaseline.json. Intervals are window-cluster bootstraps.
- **Two limits on every row (D202).**
  - Truth frames were selected by fits on the same r23 boxes, so the rows are optimistic and lean toward r23.
  - Multipool is a better fit, not a truth shown to be nearer the real positions. The truth change alone lowered missed
    by 0.083 on gsw_cha_2016 and 0.033 on gsw_bkn_2015, on the same build.
- **Games.** Only two development games are in use.
  - gsw_sac_2015 stays apart (D197, D201). The gsw_ind_2016 rebuild is deferred (D200).
  - hou_cle_2016 is kept unspent for a later arm's held-back test (D203). Only its D202 aggregate is quoted: missed
    0.158, ghosts per box 0.071, net error 0.224, team 0.756.
- **Targets.** They come from TRACKING_LEVELS.md, first draft, Level 2:
  - missed players 5 percent or less;
  - boxes on no player (ghosts) 5 percent or less;
  - team labels 95 percent or more;
  - near-camera bias within 0.5 ft.

  Position p50 and p90 have a Level 1 target (at most 2.0 ft and 5.0 ft in every arena), not a Level 2 one.
- **Budget.** Colab is capped at 60 units for Levels 1 and 2 (TRACKING_LEVELS.md, Level 1 step 1). Every item from D202
  to D214 cost 0 units.

## 1. The rows against their targets

| row | gsw_cha_2016 | gsw_bkn_2015 | pooled | target |
| --- | --- | --- | --- | --- |
| missed | 0.208 [0.191, 0.228] | 0.220 [0.205, 0.236] | 0.214 [0.203, 0.226] | 0.05 or less |
| of which detection (no box within 5 ft) | 0.052 [0.047, 0.059] | 0.065 [0.057, 0.075] | 0.059 [0.054, 0.065] | (D184 (a): reported apart) |
| of which placement (a box 2.5 to 5 ft away) | 0.156 [0.142, 0.171] | 0.155 [0.144, 0.167] | 0.155 [0.146, 0.165] | (D184 (a): reported apart) |
| ghosts per box | 0.125 [0.111, 0.141] | 0.118 [0.106, 0.132] | 0.121 [0.111, 0.132] | 0.05 or less |
| team | 0.810 [0.786, 0.834] | 0.857 [0.842, 0.872] | 0.834 [0.818, 0.849] | 0.95 or more |
| near-field bias (ft) | 1.217 [1.134, 1.326] | 1.595 [1.484, 1.697] | 1.411 [1.319, 1.502] | within 0.5 |
| position p50 (ft) | 1.566 [1.484, 1.649] | 1.593 [1.536, 1.664] | 1.580 [1.527, 1.636] | Level 1: 2.0 or less |
| position p90 (ft) | 3.304 [3.150, 3.482] | 3.430 [3.293, 3.591] | 3.370 [3.260, 3.493] | Level 1: 5.0 or less |
| net error (missed plus non-referee ghosts, per in-frame player) | 0.327 [0.297, 0.361] | 0.328 [0.304, 0.355] | 0.327 [0.308, 0.349] | none |
| one-to-one net error (report only, D195) | 0.406 [0.373, 0.445] | 0.414 [0.386, 0.446] | 0.410 [0.387, 0.435] | none |

All values are from reports/research/p8_rebaseline.json.

## 2. Missed, by class

D205 splits the missed row with a two-stage assignment: the scorer's 2.5 ft one-to-one match first, then the leftovers
at 5 ft. Rates are over in-frame players, pooled on cha and bkn (reports/research/p8_one2one.json). The classes add to
D202's 0.155 and 0.059.

| class | what it is | pooled rate | cha | bkn |
| --- | --- | --- | --- | --- |
| P-ii | an unclaimed box 2.5 to 5 ft away | 0.114 [0.106, 0.123] | 0.118 | 0.111 |
| P-i | no box left within 5 ft; the nearest is a neighbour's, about 3.7 ft away | 0.041 [0.037, 0.044] | 0.038 | 0.043 |
| D-i | no box within 5 ft; the nearest is a neighbour's, about 7.5 ft away | 0.044 [0.039, 0.049] | 0.037 | 0.050 |
| D-iii-free | the nearest box is unclaimed but beyond 5 ft | 0.014 [0.012, 0.017] | 0.015 | 0.014 |
| referee boxes (P-iii-ref, D-iii-ref) | the nearest box is a referee's | 0.001 (D205) | see record | see record |

D205 calls each rate a ceiling, not a forecast.

### 2.1 P-ii, the largest class (0.114)

**What it is.** D207: the box 2.5 to 5 ft from the missed player is, by its track:
- the player's own track, 0.541 (hits 0.595);
- a switching track, 0.328;
- another player's, 0.119;
- a ghost, 0.012.

Box geometry barely separates these boxes from hits' (height ratio 0.964 against 0.985). Post hoc, in D207: ordinary own
boxes placed 2.5 to 5 ft off on that frame.

D204, on the whole placement row: in 0.230 of placement misses the nearest box is another player's one-to-one match,
against about 0.04 of hits. Post hoc, in D204: a quarter to two fifths of placement misses look like players scored
against a neighbour's box, plausibly because their own box is missing, merged or misplaced.

**Tried:**

| D line | what | outcome |
| --- | --- | --- |
| D184 (older truth) | a leave-one-game-out foot correction by depth | scattered: no correction passes on every game (-0.0123, -0.0146, +0.0041 on gsw_sac_2015); none adopted |
| D188 (older truth) | shift output court positions only | STOP: p50 falls 0.23 to 0.28 ft, but near-third p90 rises beyond +0.05 ft on every game; detection unchanged by construction |
| D206 | a median foot shift per box-height and depth cell (F1) | killed at stage 0: missed fell 0.0039 and 0.0035 against the -0.010 bar, in sample |
| D210, stage 0a | the raw box bottom instead of the smoothed foot | no nearer the truth (rescue over the null +0.016 and +0.034 against 0.10) |
| D210, stage 2 | a per-box ankle foot from ViTPose-B | fails rules 1, 2, 4 and 5 on both games (rescue +0.047 and +0.063 against 0.10; hits lost 0.42 and 0.41); the pose line stops |
| D212 | is the row truth-side (timing, camera)? | not shown truth-side and not shown build-side. Timing offsets vary beyond the null, but own P-ii is only weakly enriched (1.31 and 1.13 against 1.5). The build's court model (B) is in sample on these games, so a null cannot read as build-side |

**Open.** TRACKING_LEVELS.md (current route) records every per-box foot route tried as closed. No record names a next
lever for this class or costs one. D212 weighed waiting for an independent court model, but no record designs or costs
one.

### 2.2 P-i and D-i, the no-box classes (0.041 and 0.044)

**Where they are lost.** D208 and D211, on all 4,882 misses on 120 windows (D211, post hoc beside D208's registered 9.3):
- the detector stage (C1) carries 0.693 [0.667, 0.719] (P-i 0.615, D-i 0.765);
- NMS suppression carries 0.154 (P-i) and 0.085 (D-i) on D208's 88 kept windows.
- D211 corrects D208 (its 9.7): D208's read does not reproduce the build where the build seeks. D208's 9.3 stands, and
  the re-read sensitivity moves C1 from 0.674 to 0.688 and C3 from 0.094 to 0.073.

**Tried:**

| D line | what | outcome |
| --- | --- | --- |
| D171 | E1, fine-tuned detector, seed 0 (older truth) | not adopted: missed rose +0.0278 [0.0115, 0.0534]; 17.30 units |
| D182 (older truth) | E1 on the reference platform | not adopted: missed -0.0067 pooled, but 2 of 3 folds on hou_cle_2016; gain below the expected 0.010 to 0.016 |
| D183 (older truth) | E1b, fine-tune r23 on teacher boxes | stopped before any unit: transfer-adjusted ceiling -0.0199 pooled and -0.0079 on gsw_cha_2016, against -0.020 and -0.010. Detector fine-tuning paused |
| D209 | NMS threshold 0.55 | inconclusive at the pilot (P3 unpowered). Missed -0.0065 [-0.0092, -0.0034], ghosts per player +0.0033, duplicates +0.0036; net error not shown to fall. Missed fell more than the 0.0038 recoverable ceiling (mechanism flag). A confirmation to rescue it is barred |
| D213 | survey of label sources for crowded players | ranking is reasoning: the held teacher first, then Iter-Deformable-DETR (CrowdHuman) |

**Open.** The held teacher's reach on C1 misses: docs/research/P8_c1reach.md. This design is on branch r-p8-c1reach
and is not merged, so its numbers are design-time bounds:
- Bound (its section 3, as corrected by 10.1): if a student reached every C1 miss, E1's transfer would give a
  build-level fall in missed of 0.0153 on gsw_cha_2016, 0.0348 on gsw_bkn_2015 and 0.0252 pooled. 10.1 says that bound
  has no loss term. What matters is the reach beyond what the held teacher's labels already give, under a union label
  set (the held teacher's labels plus the new source's extra boxes).
- Stage 1 (its 10.2): it measures the held teacher's reach and cost and checks the measurement. It decides nothing by
  itself.
- Stage 2's increment over the held teacher decides (its 10.6):
  - a training-arm design is justified only if the increment is at most -0.0042 on gsw_cha_2016 with its interval
    below -0.0021, below 0 pooled, and the ghost proxy is at most twice the held teacher's;
  - otherwise the reading is "not shown" or "not resolvable with these truths", the detector line stays paused, or the
    CrowdHuman-detector line is killed (an increment above -0.0021).
- Cost: a smoke of about 2 lock minutes, then about 35 lock minutes (cap 60), 0 units (its sections 9 and 10.11).

Rejected options, with their costs:
- E1 seeds 1 and 2, about 30 units (D182);
- the E1b pilot, about 2.3 units for roughly a 1 in 3 chance (D183).

### 2.3 D-iii-free and referee classes (0.014 and 0.001)

No item has targeted them. No lever is recorded.

## 3. Ghosts per box (0.121 against 0.05)

**Known:**
- D207: about 0.571 of P-ii boxes are also scorer ghosts. They are about 0.577 of the pooled ghost row, so the missed
  and ghost rows share one error (D195's pattern). Without them, ghosts per box would be about 0.051. That is a ceiling
  only: moving these boxes is not shown possible.
- D195 (older truth, after resets): extra boxes are mostly on court near a player (0.58 within 3 to 6 ft), 0.23 on
  frames that are not a live view, 0.07 referee-like. No court-region filter fits, and no single frame rule fits all
  three games.

**Tried:**
- D198 (older truth: gsw_bkn_2015's fixed CPU truth), the reset fix on gsw_bkn_2015: ghost share per box +0.0008
  [-0.0024, +0.0033].
- D209, NMS 0.55: ghosts per player +0.0033 [+0.0012, +0.0056].
- D182, E1 (older truth, not adopted): ghosts -0.0160.

**Open.** No ghost-specific lever is recorded. The ghost row is mostly the P-ii boxes (D207, section 2.1).

## 4. Team (0.834 against 0.95)

**Known:**
- D203: one bin carries by the registered rule: another SportVU player within 3 ft, 26 percent of errors at 1.81 times
  the rate. It does not separate tracking from colour assignment.
- D214: team labels are per canonical track. The oracle bounds on cha and bkn (current 0.810 and 0.857) are:
  - perfect colour per track (O1): 0.905 and 0.898;
  - plus splits at sustained identity swaps (O2_swap): 0.947 and 0.942;
  - O2_swap is an oracle rule value, not a reachable fix.
  Record: reports/research/p8_teambound.json.

**Tried:**

| D line | what | outcome |
| --- | --- | --- |
| D153, D154 (older truth) | team labels from CLIP and DINOv2 embeddings (E3) | not adopted: both arms lower the team row (14 to 21 points, TRACKING_LEVELS.md Level 2 step 4) |
| D214 | upper bound from label-level fixes, report only | no arm licensed. Tracking headroom (0.042 and 0.044 against 0.030) is fragile: 0.0299 and 0.025 at L = 2.0. Colour clears on cha only (0.095 against 0.041, bar 0.050). The combined gain (0.137 and 0.085, bar 0.056) is not fragile, stated as a fact, not a pattern |

**Open:**
- D214, report only: colour headroom is not limited by which frames are cropped (O1c 0.894 and 0.887).
- The scorer's gate applied after its assignment changes about 1 percent of team rows.
- No team arm is designed or costed. hou_cle_2016 is held for its held-back test (D203).

## 5. Near-camera bias (1.411 ft against 0.5 ft) and position

**Known:** D184 (older truth): the placement offset runs along the camera ray (median depth -1.1 to -1.9 ft), with no
timing signature, and varies by game.

**Tried:**
- D184 (older truth), a depth correction: scattered, no correction passes on every game.
- D188 (older truth), output shift: STOP on near-third p90 growth, which D188 counts as harm. The scored near-third bias
  varies by game (0.010, 0.367, 0.213 ft, post hoc).
- D206, F1: lowered near-field bias and p50 under the build's court model (B) by about 0.15 ft, while missed barely
  moved. Killed on missed, in sample.
- D210, pose foot: the line stops.
- D182, E1 (older truth): near-field bias -0.1589 ft. E1 was not adopted.

**Open.** No lever is recorded. D188 keeps hou_cle_2016 for an arm "with a new reason".

**Position p50 and p90** (1.58 and 3.37 ft pooled) are below Level 1's bounds on these development rows. That is not
a Level 1 result: Level 1 asks for every held-out arena.

## 6. Reasoning (not measured; for the target decision)

This section reads the records above. It computes nothing new and ranks nothing.

**Which targets look reachable with the current truths and data.**
- **Missed, 0.05.**
  - The detection part alone is 0.059 pooled, above 0.05, before any placement miss.
  - The no-box classes (P-i and D-i) have one open lever. Its design-time bound at perfect reach is about 0.025 pooled
    and 0.015 on gsw_cha_2016 (P8_c1reach.md section 3, unmerged). 10.1 corrects it: what counts is the reach beyond the
    held teacher, under union labels, which is no larger.
  - The placement part (0.155) has no open lever.
  - So the combined 5 percent target does not look reachable on these rows with any recorded lever.
- **Ghosts, 0.05.** About 0.58 of the row is the same boxes as P-ii (D207), and the rest would be about 0.051. The
  ghost target is therefore tied to the placement row rather than independent of it.
- **Team, 0.95.**
  - The oracle bound under label-level fixes sits 0.003 and 0.008 below 0.95 at the rule value.
  - The colour oracle's headroom clears its bar on cha only and is not limited by which frames are cropped (O1c); no
    fix is shown to reach it.
  - No arm is licensed yet.
  - Of the four targets, this one is the nearest to its bound. But the bound is an oracle, and no real fix is shown to
    reach it.
- **Near-camera bias, 0.5 ft.** Three correction forms failed or stopped (D184, D188, D206), and the pose foot failed
  (D210). The bias varies by game, and no lever is recorded.

**Where the truths themselves limit what can be shown.**
- **The build's court model (B) is in sample.** On the pool games r23's court model was trained on labels written
  through an earlier SportVU-fitted truth (D212). A truth-side placement error can be found through B,
  but a null cannot clear the build.
  So whether the placement row is partly the truth's own error stays open (D212: not shown either way).
- **Truth selected by r23.** The truth frames are r23's own selections (D202). The rows are optimistic, and a new
  detector is judged on frames r23 chose. A truth change alone moved missed by 0.083 on cha (D202), which is larger
  than any arm's measured gain in this map. The largest is D198's report-only -0.060 on gsw_bkn_2015, on its fixed CPU
  truth.
- **Oracle-only ceilings.** Two of the ceilings are oracles, not reachable fixes: team O2_swap (D214) and the P-ii and
  ghost ceilings (D205, D207).
- **Transfer.** Raw label gains kept only 0.24 to 0.38 (placement) and 0.36 to 0.62 (presence) at build level (D183).
  Every detector bound here passes through those fractions, which E1b found uncertain by about 0.0035 per game
  (P8_c1reach.md section 8, unmerged).
- **Few games.** Two development games and one unspent new-arena game. D200 notes that only with four games does a
  game-bootstrap interval stop being the range of the game effects.
- **The 2.5 ft criterion and one camera.** Position p90 is 3.37 ft, and TRACKING_LEVELS.md puts one
  broadcast camera at about 1.5 ft median error at best. A player counts as missed when no box sits within 2.5 ft. So
  part of the placement row may measure single-camera position error, not an absent player. D184 (a) already reports
  detection and placement apart. This is a reading, not a measurement.
  - An alternative reading (D204): in 0.230 of placement misses the nearest box is another player's one-to-one match.
    On that reading, part of the row is an own box missing, merged or misplaced, not position error.

**Options Lucien could pick (unranked):**
- **(a) Keep every Level 2 target as written.** Report the gap with its causes. Level 2 then stays open with no
  recorded lever for placement, ghosts or near-camera bias.
- **(b) Restate the missed target against the split (D184 (b)).**
  - A detection target, for the row a detector moves.
  - A separate placement target, or placement judged by position error rather than by the 2.5 ft criterion.
  - Ghosts could be restated beside it, since D207 ties them to placement.
- **(c) Restate targets as bounds from the records.** For example, team against D214's oracle bound and missed
  against the C1 bound, each stated as a ceiling, not a claim.
- **(d) Work on the truths before the targets.** An independent court model (raised in D212) or more development
  games (the gsw_ind_2016 rebuild, D200: about 7.5 lock hours, with a check that already fails). Then decide the
  targets.
- **(e) Pursue the one licensed next step and decide after it.** c1reach stage 1 (about 35 lock minutes, 0 units,
  unmerged design) measures the held teacher's reach and decides nothing by itself. Stage 2's increment over the held
  teacher, under union labels, then decides whether a detector arm is worth units (P8_c1reach.md 10.6).
- **(f) Close Level 2's modelling for some rows and report them as measured limits.** For example placement and
  near-camera bias, keeping the team and detection lines open.
