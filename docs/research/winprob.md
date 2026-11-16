# Research dossier: winprob (live win probability)

- Lane: winprob (analytics/winprob, frozen at tag freeze-v1)
- Date: 2026-11-16
- Verdict: works. The logistic model is a close cousin of Stern's Brownian-motion model and is well calibrated out of
  fold, but it has no pregame team strength, and on the held-out season it is slightly over-confident (slopes about
  0.96).

## 1. What we built

- Question: the home team's chance of winning from the game state at each play-by-play event (reports/winprob.md).
- Data: 2292413 events from 5059 games (2020-21 to 2023-24) for training; 596840 events from 1304 games of 2024-25 held
  out. No closing spreads: spread coverage on training games is 0.000 (reports/winprob.md, Inputs).
- Model: logistic on margin, seconds remaining, possession and two margin-time interactions including margin over root
  time (improvement-search variant 1), chosen over a constant and isotonic-calibrated boosting by mean bucket Brier.
- Numbers: mean bucket Brier 0.1468 out of fold, 0.1486 held out; overall Brier 0.1645 [0.1616, 0.1683] and 0.1658
  [0.1590, 0.1732]; log loss 0.4845 [0.4767, 0.4945] and 0.4879 [0.4700, 0.5077] (reports/winprob.md, Stage 8 table).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Stern, H. S. (1994). A Brownian Motion Model for the Progress of Sports Scores. Journal of the American Statistical Association 89(427): 1128-1134. | [doi:10.1080/01621459.1994.10476851](https://doi.org/10.1080/01621459.1994.10476851); [JSTOR 2290943](http://www.jstor.org/stable/2290943) | The home win probability is Phi((l + (1 - t) mu) / sqrt((1 - t) sigma^2)) for lead l at time fraction t (Section 2, p. 1129). Our margin over root time feature is Stern's l / sqrt(1 - t) term on a logit link. Stern names two missing inputs, team strength (replace mu by a point spread) and possession; we have possession, not team strength. Fitted on 493 NBA games of 1991-92 (abstract). |
| Deshpande, S. K. and Jensen, S. T. (2016). Estimating an NBA player's impact on his team's chances of winning. Journal of Quantitative Analysis in Sports 12(2): 51-72. | [arXiv:1604.03186](https://arxiv.org/abs/1604.03186) | Uses an in-game win probability model as the basis for player value; a downstream use our winprob tables could serve. |
| Yeh, C.-K., Rice, G. and Dubin, J. A. (2022). Evaluating Real-Time Probabilistic Forecasts With Application to National Basketball Association Outcome Prediction. The American Statistician 76(3): 214-223. | [arXiv:2010.00781](https://arxiv.org/abs/2010.00781), [doi:10.1080/00031305.2021.1967781](https://doi.org/10.1080/00031305.2021.1967781) | Pointwise-in-time logistic models on score difference and pregame relative strength were not significantly worse than ESPN's published feed (2018-19, Section 6). Our model is one pooled logistic with time interactions, with no relative strength. |

## 3. Benchmark

| ours | published | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| Overall Brier 0.1645 out of fold (2020-21 to 2023-24), 0.1658 held out (2024-25), one row per play-by-play event, no pregame strength (reports/winprob.md) | Brier per game second on 2018-19 and 2019-20: dynamic Bayes without pregame strength 0.1663 and 0.1736; adjusted with pregame strength 0.1568 and 0.1635; ESPN feed 0.1550 and 0.1621 (Maddox, Sides and Harvill 2022, Table 3) | Maddox et al. 2022 | Weighting (every second vs every event, so late-game seconds with many events weigh differently), seasons, train window (2012-13 to 2017-18) | roughly comparable: same league, outcome and score; the row without pregame strength is the closest match to ours |
| Held-out calibration slopes 1.001, 0.955, 0.959, 0.957, 0.955 by time bucket (reports/winprob.md) | ESPN NBA feed 2018 to 2024: 6.3 percent of games land in a benchmark tail that holds 5 percent under ideal calibration, an excess of 1.3 percent [0.4, 2.3] extreme losing-team peaks (Pipping-Gamon and Wyner 2026, Section 5, abstract) | Pipping-Gamon and Wyner 2026 | Different diagnostic (pathwise peak-on-loss vs per-bucket slope), different forecaster (ESPN) | not directly comparable; both point the same way (NBA live forecasts slightly over-confident) |
| Bucket Brier 24-12 minutes 0.1502, 12-4 minutes 0.1043 held out (reports/winprob.md) | ESPN Brier at 24 and 12 minutes remaining: 0.1666 and 0.1181 (2018-19), 0.1738 and 0.1311 (2019-20) (Maddox et al. 2022, Table 4) | Maddox et al. 2022 | Buckets of time vs single time points | not directly comparable: a bucket averages over easier late states |

Stern (1994) reports fit by quarter-level summaries (Table 1), not a Brier or log loss, so it has no row here.

## 4. Measured weaknesses (from our reports only)

- No team strength: closing spread coverage 0.000 because odds were skipped (reports/winprob.md, Inputs). The early
  buckets carry the cost: 48-36 minute Brier 0.2306 against the constant's 0.2471 out of fold, the smallest gain of any
  bucket (reports/winprob.md, candidate table).
- Held-out over-confidence: slopes drop to about 0.96 in four of five buckets and ECE rises in every bucket, for example
  36-24 minutes 0.0208 against 0.0085 out of fold (reports/winprob.md, Stage 8 table).
- Last 4 minutes worse on held-out: Brier 0.0582 [0.0504, 0.0666] against 0.0530 [0.0496, 0.0564] (reports/winprob.md).
- Possession is known for 0.781 of training events only (reports/winprob.md, Inputs).
- No external benchmark: the NBA.com WinProbabilityPBP column is empty (reports/winprob.md, Benchmark).

## 5. Recent work that targets these weaknesses

- Team strength: Maddox, Sides and Harvill (2022, arXiv:2207.05114) add a time-weighted pregame win probability and
  cut per-second Brier from 0.1663 to 0.1568 (2018-19) and 0.1736 to 0.1635 (2019-20) (Table 3). Their Table 5 shows
  an Elo pregame probability, computable from past results alone, gives an in-game Brier of 0.1605 against 0.1598 for a
  commercial rating, so a no-odds strength prior loses little. Yeh et al. (2022) find relative strength is the most
  important predictor early in the game (Section 2.1, Figure 2). Both attack our first weakness.
- Over-confidence: Pipping-Gamon and Wyner (2026, arXiv:2601.18774) give a pathwise calibration benchmark (peak win
  probability of the eventual loser) and find the ESPN NBA feed has too many extreme losing peaks. It is a diagnostic
  for the second weakness that per-bucket slopes cannot see.
- Honest intervals: Brill, Yurko and Wyner (2025, arXiv:2406.16171, accepted to JQAS) show by simulation that
  play-by-play dependence inflates estimator variance and lowers effective sample size, and that ordinary bootstrap
  intervals undercover. Our game-level bootstrap is the right unit, but our intervals may still be narrow; this bears
  on how firmly any small gain below is claimed.

## 6. Pre-registered experiments

Shared rules: development data are the play-by-play seasons 2020-21 to 2023-24 (whether 2024-25 joins development is the
lead's decision: it was scored once at freeze-v1, fitting on it scores nothing, but it is no longer untouched). Frozen
code runs only with an explicit `--out` and `--report` under a scratch folder (CLAUDE.md rule 6). The D71 test set is
scored once per experiment; the 2025-26 season is the candidate the data survey is confirming. Intervals: bootstrap by
game, 2000 resamples, 95 percent percentile intervals, paired on the same resampled games. Every new feature passes the
frozen `prefix_invariance_violations` check (no feature at event k changes when the play-by-play is cut after k).

### winprob-E1: a pregame strength drift term from past results (Stern's mu, without odds)

- Hypothesis: adding a pregame expected margin mu_hat, computed only from games played before tip-off (an Elo rating
  with home advantage and between-season regression, parameters chosen on development seasons), entered as Stern's
  drift term mu_hat * sqrt(seconds remaining / 2880) plus mu_hat alone, lowers mean bucket Brier on the D71 test set,
  mostly in the 48-36 and 36-24 buckets.
- Baseline: the frozen logistic (margin, time, possession, margin x root time, margin over root time), refit with
  frozen code on the development seasons.
- Metric: mean bucket Brier (the frozen selection metric, so the comparison is like for like), with per-bucket Brier
  and log loss reported beside it. Why: the gain is expected early in the game, which the bucket mean weights equally
  with the late buckets.
- Adoption threshold: mean bucket Brier improves by at least 0.002 with the paired 95 percent interval excluding zero,
  and no bucket gets worse beyond its interval. Reason for 0.002: the published pregame adjustment moved overall Brier
  by about 0.010 (Maddox et al. 2022, Table 3); a fifth of that is the smallest gain worth a new input.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval of the difference in mean bucket Brier.
- Ablation: mu_hat as a main effect only (no root-time drift term), to test whether Stern's functional form matters.
- Data: development seasons for Elo parameters (forward chaining: fit on earlier seasons, check on the next) and for
  the logistic fit; D71 test set scored once. Elo for the test season is updated game by game from test-season results
  as they happen, which is time-forward and not fitting.
- Compute: Elo over about 6000 games in seconds; logistic on about 2.3 million rows in minutes. Under 30 minutes on the
  16 GB Mac, CPU, under the heavy-job lock.
- Risks: early-season Elo is mostly last season's rating, so roster changes blur it; injuries and rest are not in Elo;
  the effective sample size is smaller than the row count (Brill et al.), so a gain near the threshold should be
  read with care.
- What a coach gains: the first-half numbers stop treating every opponent as average, so "down 6 to a top team" and
  "down 6 to a weak team" read differently.
- Backlog: expected gain 4 (it targets the measured early-game weakness, and the published gain is about 0.01 Brier);
  effort 2 (Elo and two features; no new data).

### winprob-E2: a forward-in-time temperature for the over-confidence

- Hypothesis: the held-out over-confidence (slopes about 0.96) is a forward-in-time effect that random grouped folds
  hide. A single logit temperature, fitted on forward predictions (fit on 2020-21 to 2022-23, predict 2023-24), brings
  the D71 test-set slopes closer to 1 and lowers ECE without hurting Brier.
- Baseline: the frozen logistic refit with frozen code on the development seasons, no temperature.
- Metric: mean ECE over the five regulation buckets (10 bins, the frozen definition) as primary; mean bucket Brier as a
  guard; the peak-on-loss diagnostic of Pipping-Gamon and Wyner (2026) reported as a secondary, descriptive check.
  Why: the weakness is calibration, which Brier mixes with sharpness.
- Adoption threshold: mean bucket ECE falls with the paired 95 percent interval excluding zero, and mean bucket Brier
  does not rise (upper end of its paired interval at or below +0.0005).
- Paired test: bootstrap by game, 2000 resamples, 95 percent intervals of the ECE and Brier differences.
- Ablation: temperature fitted on random grouped folds (the frozen protocol) instead of forward predictions; if it
  comes out near 1, the forward split is what carries the effect.
- Data: development seasons only for the temperature; D71 test set scored once. If E1 is adopted first, E2 runs on top
  of the E1 model and says so.
- Compute: one extra logistic refit and a one-parameter fit; under 15 minutes on the 16 GB Mac, CPU.
- Risks: one forward season gives one temperature, which may not transfer to the test season; a temperature fixes a
  global slope but not a bucket-specific one (the 4-0 bucket was under-confident out of fold, slope 1.065); this is a
  fit, so it runs inside the experiment milestone, never under `export/` or `web/`.
- What a coach gains: a 90 percent shown on screen means about 90 percent, which matters most when deciding whether a
  lead is safe enough to rest starters.
- Backlog: expected gain 2 (the miscalibration is small and every interval overlaps); effort 1 (one parameter, one
  forward split).

## 7. Sources checked (all fetched 2026-11-16)

- Stern 1994: [scanned article](https://www.stat.berkeley.edu/users/aldous/157/Papers/stern.pdf) (JSTOR cover page, pages 1128 and 1129 read as page images); DOI from the publisher URL in search results, [publisher page](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476851) refused the request.
- Deshpande and Jensen 2016: [arXiv abstract](https://arxiv.org/abs/1604.03186).
- Yeh, Rice and Dubin 2022: [arXiv abstract](https://arxiv.org/abs/2010.00781), [PDF](https://arxiv.org/pdf/2010.00781) (Sections 2.1 and 6), [RePEc record of the journal version](https://ideas.repec.org/a/taf/amstat/v76y2022i3p214-223.html).
- Maddox, Sides and Harvill 2022 (NBA): [arXiv abstract](https://arxiv.org/abs/2207.05114), [PDF](https://arxiv.org/pdf/2207.05114) (Tables 3, 4 and 5). Preprint; no journal reference listed.
- Pipping-Gamon and Wyner 2026: [arXiv abstract](https://arxiv.org/abs/2601.18774), [PDF v4](https://arxiv.org/pdf/2601.18774) (abstract, Section 5). Preprint.
- Brill, Yurko and Wyner 2025: [arXiv abstract](https://arxiv.org/abs/2406.16171) (v5; comments say accepted to JQAS).
