# Research dossier: load (fatigue and absence)

- Lane: load (analytics/load, frozen at tag freeze-v1)
- Date: 2026-11-16
- Verdict: split. The absence model works and holds on held-out games; the in-game speed decline model barely beats
  the training mean on held-out games (gain 0.002), so past load says little about how much a player slows down.

## 1. What we built

- Questions: does a player's speed decline across a game, and does the player miss the team's next game
  (reports/load.md).
- Data: 2015-16 SportVU, 11894 player-games with a decline from 604 training games, 12239 with a known next-game
  absence; 344 decline rows and 292 absence rows from 17 held-out games (reports/load.md).
- Targets: decline = average speed of the last stint minus the first, stints of at least 1 minute, ft per s; absence =
  not on the roster of the team's next loaded game. Absence is not injury (reports/load.md, Target definitions).
- Model: ridge (decline) and logistic (absence) on days rest, back-to-back, games in last 7 days, minutes and distance
  over the last 3 games, previous decline and previous average speed; all from previous games only.
- Numbers: decline MSE 0.4929 [0.4740, 0.5178] against the mean's 0.5086 out of fold, 0.5164 against 0.5180 held out.
  Absence log loss 0.2764 [0.2644, 0.2883] against the base rate's 0.3244 out of fold, 0.2776 against 0.3085 held out
  (reports/load.md, Stage 8 table).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Lewis, M. (2018). It's a hard-knock life: Game load, fatigue, and injury risk in the National Basketball Association. Journal of Athletic Training 53(5): 503-509. | [doi:10.4085/1062-6050-243-17](https://doi.org/10.4085/1062-6050-243-17) | Multilevel logistic of injury on time-lagged minutes, rest and game load (2012-15, 627 players, 1663 injury events): odds of injury fell 15.96 percent per day of rest and rose 2.87 percent per 96 minutes played; baseline risk and slopes varied by player. Our absence model uses the same kind of lagged schedule features, without player random effects and without an injury label. |
| Esteves, P. T., Mikolajec, K., Schelling, X. and Sampaio, J. (2021). Basketball performance is affected by the schedule congestion: NBA back-to-backs under the microscope. European Journal of Sport Science 21(1): 26-35. | [doi:10.1080/17461391.2020.1736179](https://doi.org/10.1080/17461391.2020.1736179) | Rest cycles (back-to-back, 1, 2, 3 or more days) as the unit of schedule load, 2016-17 team games; winning was significantly more likely with one day of rest than on a back-to-back. Our rest buckets follow the same idea at player level. |
| Scanlan, A. T., Tucker, P. S., Dascombe, B. J., Berkelmans, D. M., Hiskens, M. I. and Dalbo, V. J. (2015). Fluctuations in Activity Demands Across Game Quarters in Professional and Semiprofessional Male Basketball. Journal of Strength and Conditioning Research 29(11): 3006-3015. | [doi:10.1519/JSC.0000000000000967](https://doi.org/10.1519/JSC.0000000000000967) | Video time-motion study: activity velocities drop across quarters, and the authors attribute the drop to a mix of physiological, tactical and game-related factors, with stoppage time related to activity. This is the reading our decline result supports. |
| Stephen, A., Yep, M. and Fain, G. (2021). Tired of Misattribution, Modeling Player Fatigue in the NBA. arXiv preprint. | [arXiv:2112.14649](https://arxiv.org/abs/2112.14649) | NBA tracking distance per minute (Second Spectrum, via nbastatr) as an exertion proxy; within-game exertion explains under 1 percent of box-score deviation (highest adjusted R squared 0.0013). Closest in data and conclusion to our decline result. |

## 3. Benchmark

| ours | published | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| Decline model gain over the training mean: 0.016 (ft per s)^2 out of fold, 0.002 held out (reports/load.md, Stage 8 text) | Exertion explains under 1 percent of variation in box-score performance; highest adjusted R squared 0.0013 (Stephen et al. 2021, Results, exertion and performance) | Stephen et al. 2021 | Direction (load predicting speed decline vs exertion predicting box-score output), provider (SportVU 2015-16 vs Second Spectrum), metric (MSE gain vs adjusted R squared) | not directly comparable; both find past or current load explains very little |
| Mean decline by rest bucket: back-to-back -0.477 [-0.520, -0.439], one day -0.437 [-0.466, -0.411], two or more -0.457 [-0.495, -0.419] ft per s (reports/load.md, Candidate 1) | Win likelihood rose significantly from back-to-back to one day of rest (Esteves et al. 2021, abstract) | Esteves et al. 2021 | Outcome (player speed decline vs team win), season (2015-16 vs 2016-17), unit (player-game vs team-game) | not directly comparable; same direction (back-to-back is the worst bucket in both) |
| Absence log loss 0.2764 out of fold, 0.2776 held out (reports/load.md) | none: Lewis (2018) reports odds ratios from a multilevel model, not a predictive score | Lewis 2018 | Outcome (any absence vs injury), seasons (2015-16 vs 2012-15) | no comparable published number |

## 4. Measured weaknesses (from our reports only)

- Decline barely beats the mean on held-out games: MSE 0.5164 against 0.5180, gain 0.002 against 0.016 out of fold, and
  the held-out MSE is above the out-of-fold interval (reports/load.md, Stage 8).
- The decline target mixes fatigue with role and game state: a blowout fourth quarter, a closing lineup, or short
  late defensive stints all read as decline (reports/load.md, Target definitions and their limits).
- Absence is not injury: rest, coach's decision, suspension, trade and tracking misses read the same; the injuries
  table starts in 2021-22 and never overlaps the tracking era (reports/load.md, Inputs).
- Absence calibration is uneven in the middle bins: mean p 0.142 against an observed 0.101, and 0.351 against 0.440
  (reports/load.md, calibration table); ECE 0.0146 against the base rate's 0.0042.
- Small held-out sample: 344 decline rows and 292 absence rows, point estimates only (reports/load.md).
- No cross-check against league tracking aggregates: the SpeedDistance tables were absent (reports/load.md).

## 5. Recent work that targets these weaknesses

- Role versus fatigue: Lewis (2018) finds that baseline risk and load slopes vary between players, which suggests a
  player-level effect is the missing piece; Scanlan et al. (2015) and Stephen et al. (2021) find activity and output
  changes within a game are mostly not explained by exertion. Together they point to modelling the stable player part
  of decline before asking what load adds (load-E1).
- Absence is not injury, and selection: Yu and Hu (2026, arXiv:2603.26935) show that conditioning on game
  participation creates a healthy-worker survivor bias that can reverse the sign of a workload-injury association,
  and correct it with inverse probability weighting in a piecewise exponential model (78594 player-games, 771
  players, 2439 injury events). Any injury model we build must report its coefficients as associations, not effects.
- Match fatigue time course: Pernigoni et al. (2024, Journal of Sports Sciences 42(18): 1727-1758,
  doi:10.1080/02640414.2024.2409555), a systematic review of 44 studies, report sprint and jump decrements at match end
  that can persist 24 to 48 hours. It supports a next-game window for absence features, not a within-game speed proxy.

## 6. Pre-registered experiments

Shared rules: development data are the games D71 leaves for development. Intervals: bootstrap by game, 2000 resamples,
95 percent percentile intervals, paired on the same resampled games. Frozen code runs only with an explicit `--out`
and `--report` under a scratch folder (CLAUDE.md rule 6). Nothing here fits under `export/` or `web/`.

### load-E1: a shrunk player history for decline

- Hypothesis: most predictable decline is a stable player trait (role). Adding the player's empirical-Bayes shrunk mean
  of all their previous development-game declines (weight n / (n + k), k chosen in inner grouped folds, previous games
  only) to the frozen ridge lowers decline MSE on the D71 test set.
- Baseline: the frozen ridge on the seven frozen features, refit with frozen code on the development games, with the
  training-mean floor beside it.
- Metric: decline MSE, (ft per s)^2, the frozen metric, plus the gain over the floor. Why: like-for-like with the
  frozen result, which is what failed.
- Adoption threshold: paired MSE difference (new minus frozen ridge) with its whole 95 percent interval below zero on
  the D71 test set, and a gain over the floor of at least 0.010. If the test set holds under about 1000 decline rows,
  the run reports the interval and the verdict says underpowered, not adopted.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval.
- Ablation: the shrunk player mean alone (no schedule features). If it matches the full model, the lane's honest
  statement becomes "decline is role, schedule adds nothing measurable".
- Data: the frozen player_games table for development games (already written by the frozen run under
  analytics/load/<season>/, read-only); D71 test-set rows built by frozen feature code, one game at a time; scored once.
- Compute: ridge on about 12000 rows in seconds; if the D71 test set is new tracking games, building their player
  stints means reading their frames one game at a time under the heavy-job lock (time per game not yet measured). Under 1 hour on the 16 GB Mac, CPU, for an
  internal test set.
- Risks: players with few previous games get little from the prior, and early-season rows have none; if the test set
  is carved from 2015-16 the same players appear in development, which is what the prior needs but limits the claim to
  this season; a gain may come from role and say nothing about fatigue, and the write-up must say so.
- What a coach gains: an honest split between "this player always fades late because of how they are used" and "this
  player fades more after heavy schedules", which decides whether rest or rotation is the lever.
- Backlog: expected gain 3 (directly targets the failed model; the gain may be modest); effort 1 (reads an existing
  table, adds one feature).

### load-E2: injury-labelled absence on the play-by-play seasons

- Hypothesis: with an injury label from the section 15 injuries table (2021-22 onward), a logistic model on schedule
  and minutes features built from play-by-play stints predicts next-game injury absence better than the base rate, and
  all-cause absence (the frozen target) hides a weaker injury signal.
- Baseline: the frozen absence logistic (same features where play-by-play provides them: days rest, back-to-back,
  games in last 7 days, minutes over last 3 games; no distance or speed) refit with frozen code on development seasons,
  and the base rate.
- Metric: log loss of next-game injury absence, with Brier and calibration slope beside it. Why: the frozen lane uses
  log loss, and calibration was its visible weak spot.
- Adoption threshold: injury-absence log loss below the base rate with the paired 95 percent interval excluding zero,
  and calibration slope within 0.9 to 1.1 on the D71 test set.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval of the log loss difference.
- Ablation: all-cause absence target against injury-only target on the same rows, to measure how much rest and coach's
  decisions blur the frozen target.
- Data: development seasons 2021-22 to 2023-24 (2024-25 at the lead's decision, as for the other play-by-play lanes);
  the D71 test set, with the 2025-26 season as the candidate the data survey is confirming. Requires the injuries table
  to cover the test season; the data survey must confirm it before any run.
- Compute: roughly 27000 player-games a season (an estimate: 1230 games, about 11 players a side), logistic in seconds; stint features already exist under
  collect/stints. Under 30 minutes on the 16 GB Mac, CPU, under the heavy-job lock.
- Risks: injury-report practice can change between seasons, so labels may shift (the label audit checks rates by
  season); healthy-worker bias (Yu and Hu 2026) means coefficients are not causal; this drops tracking, so it
  answers absence only, not in-game decline.
- What a coach gains: a next-game injury-absence risk that is separated from planned rest, which is what a medical
  staff would act on.
- Backlog: expected gain 3 (new label and far larger sample; the signal may still be small); effort 3 (new join of the
  injuries table, a label audit, and data confirmation).

## 7. Sources checked (all fetched 2026-11-16)

- Lewis 2018: [DOI resolver](https://doi.org/10.4085/1062-6050-243-17) (resolves to Journal of Athletic Training 53(5), p. 503; the publisher page and PubMed Central refused the request), [BISp record with abstract](https://lida.sport-iat.de/ta/Record/4050303?lng=en).
- Esteves et al. 2021: [Crossref record](https://api.crossref.org/works/10.1080/17461391.2020.1736179) (abstract included).
- Scanlan et al. 2015: [Crossref record](https://api.crossref.org/works?query.bibliographic=Fluctuations+in+Activity+Demands+Across+Game+Quarters+in+Professional+and+Semiprofessional+Male+Basketball&rows=1); abstract read in the Consensus search record.
- Stephen, Yep and Fain 2021: [arXiv abstract](https://arxiv.org/abs/2112.14649), [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2112.14649) (Data, Results sections).
- Yu and Hu 2026: [arXiv abstract](https://arxiv.org/abs/2603.26935). Preprint, not peer reviewed.
- Pernigoni et al. 2024: [Crossref record](https://api.crossref.org/works?query.bibliographic=Match-related+fatigue+in+basketball+a+systematic+review+Pernigoni&rows=1); abstract read in the Consensus search record.
