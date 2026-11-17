# winprob-E1: a pregame strength drift term from past results (Stern's mu, without odds)

Draft written 2026-11-17, before the test score. Pre-registration: docs/research/winprob.md, section 6, "winprob-E1".
Data rules: PLAN.md D79 (development seasons 2020-21 to 2023-24), D89 (the 2025-26 test season, and 2024-25 read as
Elo state only). Code: tools/research_winprob.py and tools/research_winprob_elo.py. Every number below is copied from
reports/research/winprob-E1-dev.json. Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of
games (seed 0); paired differences use the same resampled games for both arms. ECE and calibration slopes are point
estimates and are labelled so.

**Status: development stage done; the test season is not scored yet.**

## What is tested

The frozen live win probability model (a logistic on margin, time, possession and two margin-time terms) knows
nothing about the teams. The experiment adds a pregame expected margin mu_hat from an Elo rating built only from
games that ended before tip-off, entered as Stern's drift term mu_hat * sqrt(seconds remaining / 2880) plus mu_hat
alone (arm e1). Ablation: mu_hat alone, without the drift term. Control: the frozen logistic refit with the frozen
code (arm baseline).

Metric: mean bucket Brier over the five regulation buckets (48-36, 36-24, 24-12, 12-4 and 4-0 minutes left), with
per-bucket Brier and log loss beside it. Adoption threshold, fixed before any run: e1 minus baseline is at most
-0.002, its paired 95 percent interval lies below zero, and no bucket gets worse beyond its interval (the lower end
of that bucket's paired interval above zero). "No bucket" is read as written, overtime included (PLAN.md D91); the
regulation-only reading is printed beside the verdict with a note on whether it would differ. Overtime is not part of
the mean bucket Brier, as in the frozen lane.

## Gate

The baseline arm, refit by the frozen trainer's own fit function on the development seasons under the frozen 5
grouped folds (seed 0), reproduced the stored out-of-fold predictions of all 2292413 training events: largest
difference 2.98e-8 (tolerance 1e-6; the stored values are 32-bit floats), no fold mismatch. Under those folds the
baseline's mean bucket Brier is 0.1468, its Brier 0.1645 and its log loss 0.4845, the frozen report's numbers.

## Elo (chosen on development seasons only)

- Grid: K in 10, 15, 20, 25, 30; home advantage 0, 25, 50, 75, 100 Elo points; between-season pull to the mean 0,
  0.25, 0.5, 0.75. Fixed before any run, not tuned: the FiveThirtyEight margin of victory multiplier and 1/28 point
  per Elo point.
- Forward chaining: one run from 2020-21 (every team starts at 1500; a burn-in season), the pregame win chance of
  each game of 2021-22, 2022-23 and 2023-24 scored by Brier using only earlier games.
- Chosen: K 15, home advantage 50, pull to the mean 0.25. Pregame Brier 0.2213 (2021-22), 0.2280 (2022-23), 0.2128
  (2023-24), mean 0.2207.
- On the scored development games the pregame Elo Brier is 0.2205 [0.2156, 0.2252] (3903 games). Against a constant
  home win rate in each season: 0.2206 [0.2122, 0.2291] against 0.2479 [0.2456, 0.2502] (2021-22), 0.2281 [0.2198,
  0.2357] against 0.2446 [0.2422, 0.2470] (2022-23), 0.2127 [0.2046, 0.2210] against 0.2481 [0.2449, 0.2512]
  (2023-24).

## Leak checks

- Elo prefix check: for 25 random games, the schedule cut after the game with the game's own final score scrambled
  gives the same pregame value for every game up to it. No violation on the development chain or on any development
  season run.
- The frozen prefix_invariance_violations, unchanged, with its column list extended by mu_hat and mu_drift, on 60
  games of 2022-23 (5 probed, 3 cuts each), with the scrambled finals fed into Elo: no violation.
- Every development season rebuilt through the scoring path equals the cached frame (margin, time, mu_hat, mu_drift
  all identical), and the play-by-play final margin agrees with the schedule's final score for every game (1298,
  1302 and 1303 of 1298, 1302 and 1303).

## Development result (forward chaining, 3903 games of 2021-22 to 2023-24)

Each season is scored by arms fitted on the seasons before it, with Elo carried game by game through it.

| arm | mean bucket Brier | Brier | log loss |
| --- | --- | --- | --- |
| baseline | 0.1474 [0.1436, 0.1512] | 0.1653 [0.1613, 0.1692] | 0.4867 [0.4767, 0.4968] |
| e1 | 0.1393 [0.1349, 0.1435] | 0.1555 [0.1507, 0.1601] | 0.4642 [0.4519, 0.4759] |
| ablation | 0.1399 [0.1354, 0.1443] | 0.1560 [0.1512, 0.1606] | 0.4655 [0.4533, 0.4774] |

Paired differences:

| difference | mean bucket Brier | 48-36 | 36-24 | 24-12 | 12-4 | 4-0 | OT (reported only) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e1 minus baseline | -0.0082 [-0.0103, -0.0059] | -0.0208 [-0.0252, -0.0165] | -0.0124 [-0.0159, -0.0092] | -0.0053 [-0.0076, -0.0031] | -0.0020 [-0.0033, -0.0007] | -0.0003 [-0.0008, 0.0002] | 0.0018 [-0.0027, 0.0061] |
| ablation minus baseline | -0.0075 [-0.0099, -0.0052] | -0.0206 [-0.0243, -0.0170] | -0.0124 [-0.0157, -0.0093] | -0.0050 [-0.0076, -0.0024] | -0.0010 [-0.0030, 0.0010] | 0.0012 [-0.0001, 0.0024] | 0.0091 [-0.0011, 0.0192] |
| e1 minus ablation | -0.0006 [-0.0009, -0.0004] | -0.0002 [-0.0009, 0.0005] | 0.0000 [-0.0002, 0.0002] | -0.0004 [-0.0007, 0.0000] | -0.0010 [-0.0017, -0.0003] | -0.0015 [-0.0022, -0.0007] | -0.0074 [-0.0132, -0.0015] |

- By the pre-registered rule, e1 passes on development: -0.0082, interval below zero and below -0.002, no regulation
  bucket worse. The gain sits where it was expected, in the first half (48-36: -0.0208).
- Each season alone: -0.0085 [-0.0126, -0.0042] (2021-22), -0.0041 [-0.0077, -0.0005] (2022-23), -0.0120 [-0.0153,
  -0.0085] (2023-24). 2022-23 is the weakest, and its interval reaches up to -0.0005, above the threshold.
- Ablation: the drift form adds a small amount beyond mu_hat alone (-0.0006 [-0.0009, -0.0004]), almost all of it in
  the last 12 minutes and overtime. A plausible reason: mu_hat alone keeps the same pull toward the pregame favourite
  to the last second, where the drift term shrinks it as the clock runs out.
- Secondary, the frozen protocol (5 grouped folds inside the development seasons, all 5059 games; not the forward
  test, no verdict): e1 minus baseline -0.0079 [-0.0096, -0.0062], ablation minus baseline -0.0073 [-0.0092,
  -0.0055].
- Calibration (point estimates): in the forward folds e1's early-bucket slopes are 0.93 (48-36) and 0.95 (36-24)
  against the baseline's 0.97 and 0.97, and 0.871 and 0.855 in the 48-36 bucket of 2021-22 and 2022-23. The strength
  term sharpens early predictions, and forward in time it is a little over-confident there. Under the within-season
  folds the same slopes are 1.002 and 1.006. This is winprob-E2's territory, not a criterion here.

## Commitments before the test score

Fixed now, before score_once is called, after the development result above. The test season is 2025-26, 1316 games
(PLAN.md D89).

1. **The verdict is the pre-registered rule, read once on e1 minus baseline.** Adopted when the mean bucket Brier
   difference is at most -0.002, its paired 95 percent interval lies below zero, and no bucket's interval lies above
   zero. "No bucket" is read as written: every bucket, overtime included (PLAN.md D91, fixed after the pre-score
   review, before the score). The regulation-only reading (the five buckets of the metric) is printed beside the
   verdict with whether it would differ; it does not replace the verdict. On development both readings agree (pass).
2. **Power.** Scaled from the development interval (half-width 0.0022 at 3903 games), the test half-width is about
   0.0038 at 1316 games. A development-sized effect (-0.008) would be detected; an effect of -0.004, like 2022-23's,
   would give an interval reaching close to zero.
3. **Readings, fixed in advance.**
   - Pass: "adopted as a research result". The frozen lane, its report and the site do not change; whether the model
     on the site gains the term is a separate decision of the lead (a new D line).
   - Interval below zero, difference above -0.002: "a real gain smaller than the size worth a new input: not
     adopted". If the interval also straddles -0.002, the report says the size is not settled at 1316 games, and
     the verdict is not reread.
   - Interval includes zero: "no detectable gain on the test season: not adopted". With a half-width near 0.004,
     gains smaller than that are not ruled out, and the report says so; it is not evidence that the term hurts.
   - e1 worse (interval above zero), or any bucket (overtime included) worse beyond its interval: "fail", naming the
     bucket.
4. **Ablation** is read descriptively only: the drift form "adds" on the test season if e1 minus ablation lies below
   zero, else "no detectable difference". No verdict rests on it.
5. **The control arm is not a new score of the frozen lane.** The baseline's test numbers are the control arm of this
   experiment; they do not replace or sit beside the frozen held-out result of reports/winprob.md.
6. **Every per-arm metric** carries a 95 percent interval from the same 2000-resample bootstrap by game, so nothing is
   read from the test games a second time. ECE and slopes are point estimates. Per-bucket log loss and its paired
   differences carry intervals from the same weights and are reported with no verdict. The pregame Elo Brier of the
   test games is reported as a description.
7. **Data faults are reported, not fixed.** The scorer reports the games listed (1316), without play-by-play, dropped
   by the frozen builders (the clock rule), scored, and the rows or games with a non-finite value; a non-finite value
   that reaches the verdict refuses the verdict, and so does any violation of the Elo prefix check on the test season.
   The ids of test games dropped by the builders and the clock rule's counts by cause are recorded. The scorer checks
   that the frozen loader read only the pinned play-by-play file, and records the code commit, the development
   state-frame sha256 and the gate file sha256 in the ledger entry. It also reports, for every test game, whether the play-by-play final
   margin equals the schedule's final score. If that agreement is below 1.0, the disagreeing count is reported beside
   the verdict, the verdict stands as computed, and there is no rescore. If the scorer crashes, the one score is used
   up (the ledger keeps the started entry) and the experiment is reported as not scored.
8. **Caveats written now.** Elo parameters were chosen on the same check seasons the development stage scores, so the
   development result is a little optimistic (3 parameters on about 3900 games). Brill, Yurko and Wyner (2025) find
   that play-by-play dependence makes intervals narrow even with a game bootstrap, so a result within about one
   half-width of -0.002 is read with care. Elo knows nothing of injuries, rest or trades; early-season ratings are
   mostly last season's.
   A Brier pass does not mean the early predictions are calibrated. On development, forward in time, e1's 48-36
   slope was 0.871 (2021-22) and 0.855 (2022-23), over-confident, but 1.094 (2023-24), under-confident, so the
   picture is mixed. The test slopes (point estimates) are shown beside the verdict. Any use on the site waits for
   winprob-E2 or a D line.
9. **2024-25** enters only as Elo state (D89): its results move the ratings, nothing is fitted, tuned or scored on it,
   and the report gives only the count of games read.

## Test season (2025-26)

Not scored yet.
