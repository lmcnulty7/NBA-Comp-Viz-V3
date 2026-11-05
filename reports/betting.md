# Betting market model (analytics/betting/train.py)

Written 2026-11-05T09:03:29Z by `python -m analytics.betting.train --fixtures --report reports/betting.md`.

**Fixture output, not a result.** This run used the synthetic games and odds of tests/_fx_betting.py; it proves the pipeline runs end to end and nothing else. Odds history is a paid feed (The Odds API through collect/extras.py --odds) that has not been collected.

## Inputs

| source | value |
| --- | --- |
| fixtures | analytics/fixtures.py |

## Join coverage (odds to games: team nickname key, game date within one day of the UTC tip minus 8 hours)

| count | value |
| --- | --- |
| odds rows | 2880 |
| odds games (game_key) | 80 |
| closing rows after resolving duplicates | 80 |
| duplicate closing snapshots resolved (nearest tip kept) | 0 |
| games with more than one book (median point, mean de-vigged probability) | 80 |
| games with no is_close snapshot (latest pre-tip used) | 0 |
| games with no pre-tip snapshot (dropped) | 0 |
| spread out of [-30, 30] (nulled) | 0 |
| total out of [150, 300] (nulled) | 0 |
| games in the games table | 80 |
| matched to a closing line | 80 (100.0 percent) |
| games without odds | 0 |
| odds without a game | 0 |
| matched with spread / total / moneyline | 80 / 80 / 80 |
| team names given / inferred from dates / ids unresolved | 26 / 0 / 0 |
| collector held-out rows dropped | 0 |

## Protocol

Unit: one game. Targets: home_cover (home margin plus closing home spread above 0; null on a push or without a spread) and clv = margin forecast minus the market margin forecast (minus the closing home spread).
Features (previous games only): home_net10, away_net10, home_net30, away_net30, home_rest_days, away_rest_days, home_b2b, away_b2b, home_flag, home_pf_std, home_pa_std, away_pf_std, away_pa_std. Guard: analytics.schema.assert_time_forward plus the betting target list and a correlation check against the final margin (over 0.99 refused) on every design matrix.
Held-out unit: season 2021-22, 40 games set aside before any fit; `analytics.eval.refuse_heldout` checked 40 training game ids.
Training seasons: 2020-21 (40 games, 40 with a cover label).
Selection: out-of-fold under 5 grouped folds by game inside the training seasons (seed 0); cover model by out-of-fold log loss, margin model by out-of-fold MSE; intervals are 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

Staking rule, fixed in code before scoring (train.py constants): flat 1 unit on the home cover when p_cover > 0.55, on the away cover when 1 - p_cover > 0.55, else no bet; a win returns the collected spread price when the row carries one, else the nominal 0.91 units (vig 4.5 percent a side, about -110); a loss costs one unit, a push returns the stake; ROI is profit over units staked. betting_games carries no spread price, so the run reports the nominal return.

## Candidates (out-of-fold, 40 games)

| candidate | cover log loss | Brier | margin MSE | bets | profit (units) | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| market | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] | 790.9 [368.6, 1460.1] | 0 | 0.00 [0.00, 0.00] | n/a |
| logistic | 0.8247 [0.5993, 1.1117] | 0.2676 [0.2018, 0.3437] | 790.9 [368.6, 1460.1] | 37 | 5.02 [-5.53, 15.59] | 0.136 [-0.146, 0.433] |
| ridge | 0.7976 [0.6298, 1.0497] | 0.2732 [0.2124, 0.3591] | 873.8 [480.7, 1412.3] | 35 | 3.20 [-9.19, 13.57] | 0.091 [-0.270, 0.380] |

Selected: cover model **market**, margin model **market**.

## Profit and loss per season, selected models

| season | mode | games | labelled | log loss | Brier | margin MSE | bets | profit | ROI | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020-21 | cv | 40 | 40 | 0.6931 | 0.2500 | 790.9 | 0 | 0.00 | n/a | n/a | n/a |
| 2021-22 | heldout | 40 | 40 | 0.6931 | 0.2500 | 1054.3 | 0 | 0.00 | n/a | n/a | n/a |

The row marked held-out is season 2021-22 scored ONCE by the selected models refit on every training row; it was never fitted or tuned on. Training-season rows are out-of-fold under the grouped folds.

## Outputs

Tables under a temporary directory (no --out given), `analytics/betting/<season>/`: betting_games, betting_predictions, betting_summary (docs/contract/betting.md).

## Status

Fixture output, not a result. The real run on the Mac, once odds have been bought and collected with `ODDS_API_KEY=... python -m collect.extras --odds --seasons ...`, is one command:

    python -m analytics.betting.train --data "$DATA_ROOT" --out "$OUT_ROOT" --report reports/betting.md --seed 0 --bootstrap 1000 --heldout-season 2024-25

Standing caveat: odds history is a paid feed not yet collected; until it is, every number here is fixture output.
