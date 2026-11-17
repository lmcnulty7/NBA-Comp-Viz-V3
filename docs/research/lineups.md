# Research dossier: lineups (lineup and player ratings)

- Lane: lineups (analytics/lineups, frozen at tag freeze-v1)
- Date: 2026-11-16
- Verdict: inconclusive. RAPM is never worse than raw plus-minus on MSE, but our validation (season-ahead Spearman over
  about 20 recurring lineups) has too little power to show ranking skill either way: every interval spans zero.

## 1. What we built

- Question: net rating of lineups and players, adjusted for teammates and opponents (reports/lineups.md).
- Data: play-by-play stints, 2020-21 to 2023-24 for fitting, 2024-25 held out; 180243 stints after validation, 6363
  games (reports/lineups.md, Inputs).
- Model: RAPM, a possession-weighted ridge on signed player indicators with an unpenalised home intercept, alpha 3000
  by grouped CV by game; candidates plus_minus and rapm_synergy (reports/lineups.md, Protocol).
- Numbers: season-ahead Spearman for lineups present in both seasons 0.065, 0.222, 0.041 (mean 0.109) on the three
  validation seasons; held-out 0.065 [-0.488, 0.482] on 21 lineups. RAPM MSE 169.2 [76.6, 279.3] against the
  plus-minus floor's 230.7 on held-out (reports/lineups.md, Stage 8 table; reports/results.md, Summary).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Sill, J. (2010). Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing. MIT Sloan Sports Analytics Conference. | [proceedings page](https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing) | Our objective is Sill's equation (1): possession-weighted squared error plus lambda times the squared norm of the ratings. Sill chose lambda by CV and scored predicted game margins on held-out March and April games; we score season-ahead lineup ranks instead. |
| Deshpande, S. K. and Jensen, S. T. (2016). Estimating an NBA player's impact on his team's chances of winning. Journal of Quantitative Analysis in Sports 12(2): 51-72. | [arXiv:1604.03186](https://arxiv.org/abs/1604.03186), [doi:10.1515/jqas-2015-0027](https://doi.org/10.1515/jqas-2015-0027) | Bayesian regression of change in win probability on the ten players per shift. We keep point margin per 100 possessions as the target. |
| Grassetti, L., Bellio, R., Di Gaspero, L., Fonseca, G. and Vidoni, P. (2021). An extended regularized adjusted plus-minus analysis for lineup management in basketball using play-by-play data. IMA Journal of Management Mathematics 32(4): 385-409. | [doi:10.1093/imaman/dpaa022](https://doi.org/10.1093/imaman/dpaa022) | Joint player and lineup effects fitted by Bayesian methods, posteriors used to rank lineups (EuroLeague 2018-19). Our rapm_synergy is the ridge analogue; it did not beat plain RAPM on our validation. |
| Hvattum, L. M. (2019). A comprehensive review of plus-minus ratings for evaluating individual players in team sports. International Journal of Computer Science in Sport 18(1): 1-23. | [doi:10.2478/ijcss-2019-0001](https://doi.org/10.2478/ijcss-2019-0001) | Map of the plus-minus literature; used to place our ridge model in the family. |

## 3. Benchmark

| ours | published | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| Held-out season-ahead lineup Spearman 0.065 [-0.488, 0.482], 21 lineups; RAPM MSE 169.2 (net rating per 100, squared) (reports/lineups.md) | Test RMSE of predicted game margin 11.47 points (R squared 0.169) for regularized APM on 3 seasons; 12.01 (R squared 0.088) unregularized; 12.58 = predicting the mean (Sill 2010, pp. 5-6) | Sill 2010 | Metric (game-margin RMSE vs lineup rank correlation), unit (game vs lineup), split (within-season March and April vs next season), seasons (2006-07 to 2008-09 vs 2020-21 to 2024-25) | not directly comparable: different target and split; we never computed a game-margin RMSE |
| same as above | Mean season game-margin RMSE on held-out March and April games: filtered RAPM 12.897, Dummy RAPM 12.856, over 16 seasons; out-of-sample R squared 0.173 and 0.178 (Watts et al. 2026, Table 2) | Watts, Pipping-Gamon and Wyner 2026 | Same differences as the row above; 16 seasons 2007 to 2025 | not directly comparable: different target and split |
| Ridge penalty alpha 3000 chosen by grouped CV (reports/lineups.md) | lambda 3000 optimal with 3 seasons, 2000 with 1 season (Sill 2010, p. 6) | Sill 2010 | Our solver is the same weighted objective (possessions as weights, per-100 target), but our possessions are estimated as duration / 28.8 when the stint table has no count | roughly comparable as a setting, not a performance number |
| same as row 1 | L-RAPM beats raw lineup ratings on next-week possession RMSE; about 5 percent better for lineups never seen before (Petridis and Pelechrinis 2026, Section 4, Figure 4) | Petridis and Pelechrinis 2026 | Within-season weekly expanding window, 2023-24, possession-level RMSE of points per possession | not directly comparable: different metric and horizon |

No paper we found reports a season-ahead Spearman on recurring lineups, so the lane's own validation metric has no
published counterpart.

## 4. Measured weaknesses (from our reports only)

- Too few recurring lineups: 18, 27 and 26 lineups present in both seasons in the three validation pairs, 21 on
  held-out (reports/lineups.md). Every Spearman interval, validation and held-out, contains zero.
- Selection on a noisy metric: rapm (mean Spearman 0.109) was chosen over plus_minus (0.105) by 0.004, far inside every
  paired interval, for example rapm minus plus_minus -0.017 [-0.672, 0.725] (reports/lineups.md, candidate tables).
- Raw lineup net rating barely persists across seasons: Pearson 0.120, 0.102, 0.253 and -0.129 on the held-out pair
  (reports/lineups.md, out-of-season correlation table). Rosters and roles change, so a season-ahead target mixes
  rating error with real change.
- Possessions are estimated from duration (duration_s / 28.8) when the stint table has no count (reports/lineups.md,
  Protocol).

## 5. Recent work that targets these weaknesses

- Small lineup samples: Petridis and Pelechrinis (2026, arXiv:2601.15000) shrink each lineup's rating toward the sum of
  its players' RAPM from the previous season and report gains that grow as the lineup sample shrinks. This attacks our
  first weakness directly: it gives a rating for every lineup, not only recurring ones, and it is validated on
  possessions, not on 20 lineups.
- Underpowered validation: Sill (2010) and Watts, Pipping-Gamon and Wyner (2026, arXiv:2608.19454) validate on the
  margin of every held-out game, which uses thousands of games instead of a few dozen lineups. Watts et al. also show
  that how excluded low-minute players are represented changes held-out RMSE (by 0.042 points, 13 of 16 seasons), a
  detail that matters here because the frozen code rates an unseen player as 0.
- Hierarchical priors: Gong and Chen (2024, JQAS 20(3): 193-217, doi:10.1515/jqas-2022-0120) report improved
  prediction of player plus-minus with a hierarchical Bayesian model that pools by position. This is a candidate prior
  for E2 if the player-sum prior is too weak for new players.

## 6. Pre-registered experiments

Shared rules: development data are the play-by-play seasons 2020-21 to 2023-24 (whether 2024-25 joins development is the
lead's decision: it was scored once at freeze-v1, fitting on it scores nothing, but it is no longer untouched). Frozen
code runs only with an explicit `--out` and `--report` under a scratch folder (CLAUDE.md rule 6). The D71 test set is
scored once per experiment; the 2025-26 season is the candidate the data survey is confirming. Intervals: bootstrap by
game, 2000 resamples, 95 percent percentile intervals, paired on the same resampled games.

### lineups-E1: score RAPM on every game's margin, not on 20 recurring lineups

- Hypothesis: on the D71 test set, RAPM player ratings fitted on the development seasons predict each game's margin
  (summed over its stints, Sill's construction) better than raw plus-minus ratings, and the paired difference excludes
  zero. If true, the lane moves from inconclusive to a tested result.
- Baseline: the frozen rapm (alpha 3000) and the frozen plus_minus candidate, both refit with frozen code on the
  development seasons. Floor: home-court intercept only.
- Metric: game-margin RMSE over all test games, and out-of-sample R squared against the floor. Why: it uses every game,
  so its interval is narrow enough to decide; it is the published metric (Sill 2010, Watts et al. 2026), so our number
  becomes comparable.
- Adoption threshold: RAPM is reported as validated if the paired RMSE difference (RAPM minus plus_minus) has its whole
  95 percent interval below zero and RAPM's R squared against the floor is above zero with its interval above zero.
  Otherwise the lane stays inconclusive and says so.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval of the RMSE difference.
- Ablation: players with no development-season rating set to 0 (frozen behaviour, analytics/lineups/train.py line 260) against one shared replacement-level
  indicator per side, the Watts et al. count idea, with its penalty chosen by grouped CV inside development seasons.
- Data: development seasons for fitting and for a within-season dry run (October to February fit, March and April
  scored, inside 2023-24 only) to check the code; D71 test set scored once.
- Compute: sparse ridge on about 800 player columns; each fit takes seconds; 2000 bootstrap resamples of per-game
  errors need no refit. Under 30 minutes on the 16 GB Mac, CPU, under the heavy-job lock.
- Risks: season-ahead ratings miss roster changes and rookies (the ablation measures this); a game margin also reflects
  garbage time and fouling, so RMSE has a floor near 11 to 13 points in the published work; a pass on margin does not
  validate lineup ranks, and the write-up must say which claim it supports.
- What a coach gains: a player rating with a tested out-of-sample error, so "player A helps more than player B" carries
  an interval that was checked on games the model never saw.
- Backlog: expected gain 4 (it turns an inconclusive lane into a decided one with a published, comparable metric);
  effort 2 (frozen ratings already exist; only a new scorer over stints is needed).

### lineups-E2: lineup ratings shrunk toward the sum of their players (L-RAPM)

- Hypothesis: within the D71 test season, lineup offensive and defensive ratings shrunk toward league average plus the
  sum of the five players' RAPM from the development seasons predict next-week points per possession better than raw
  lineup ratings, most of all for lineups with under 100 possessions.
- Baseline: raw lineup ratings (the plus_minus construction at lineup level) and the frozen rapm_synergy candidate,
  refit with frozen code on the development seasons.
- Metric: weekly expanding-window RMSE of points per possession, summarised as relative improvement over the baseline
  (Petridis and Pelechrinis 2026, equation 1), reported in possession bins (under 25, 25 to 100, over 100). Why: it
  scores every lineup that plays, which is where our current metric has no power.
- Adoption threshold: relative RMSE improvement at least 1 percent overall with the paired interval excluding zero,
  and no bin worse than the baseline beyond its interval.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval of the RMSE difference.
- Ablation: prior from player RAPM (main) against a prior of league average only (pure shrinkage toward zero), which
  isolates what player information adds.
- Data: development seasons for player priors and for choosing the lineup penalty (expanding windows inside 2023-24);
  D71 test set scored once.
- Compute: one sparse ridge per week (about 25 weeks), seconds each; under 1 hour on the 16 GB Mac, CPU.
- Risks: the published gain is small (about 5 percent for unseen lineups; the overall gain is shown only in a figure) and comes from a single season; possession counts here
  are estimated from duration, which adds noise to points per possession; a defensive sign error is possible (the
  paper's own defensive prior convention is ambiguous), so the sign is unit-tested before any run.
- What a coach gains: a rating for a lineup after 30 possessions that is better than its raw net rating, which is the
  decision a coach actually faces with a new rotation.
- Backlog: expected gain 3 (directly attacks small lineup samples, but published gains are small); effort 3 (new
  weekly evaluation loop and prior construction).

## 7. Sources checked (all fetched 2026-11-16)

- Sill 2010, full text: [PDF copy on a course site](https://supermariogiacomazzo.github.io/STOR538_WEBSITE/Articles/Basketball/Basketball_Sill.pdf) (pages 4 to 6 read); [conference page](https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing).
- Deshpande and Jensen 2016: [arXiv abstract](https://arxiv.org/abs/1604.03186).
- Grassetti et al. 2021: [publisher page via DOI](https://academic.oup.com/imaman/article/32/4/385/5924258).
- Hvattum 2019: [Crossref record](https://api.crossref.org/works?query.bibliographic=A+comprehensive+review+of+plus-minus+ratings+for+evaluating+individual+players+in+team+sports+Hvattum&rows=1); abstract read in the Consensus search record.
- Petridis and Pelechrinis 2026, L-RAPM: [arXiv abstract](https://arxiv.org/abs/2601.15000), [PDF](https://arxiv.org/pdf/2601.15000) (Sections 3 and 4).
- Watts, Pipping-Gamon and Wyner 2026, Dummy RAPM: [arXiv abstract](https://arxiv.org/abs/2608.19454), [PDF](https://arxiv.org/pdf/2608.19454) (Table 2, Section 5.1).
- Gong and Chen 2024: [RePEc record](https://econpapers.repec.org/RePEc:bpj:jqsprt:v:20:y:2024:i:3:p:193-217:n:1003) (abstract only; the publisher page refused the request, so no number from it is quoted).
