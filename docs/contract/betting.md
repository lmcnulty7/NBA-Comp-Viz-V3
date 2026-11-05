# Betting market model: tables (analytics/betting)

Unit: one NBA game. Written under `OUT_ROOT/analytics/betting/<season>/` by `python -m analytics.betting.train`
(betting_games alone by `python -m analytics.betting.features --out OUT_ROOT`). Never into the repository: the
inputs derive from a paid odds feed (docs/DATA_CONTRACT.md section 14) and no raw odds row or derived table is
committed. Nothing here is a result until that feed has been collected; the fixture run proves the pipeline only.

Sign conventions: `closing_spread_home` is the home side's line as the feed prints it (negative when the home
side is favoured; -5.5 means the home side must win by 6). `home_margin` is home score minus away score. The home
side covers when `home_margin + closing_spread_home > 0`, pushes when it is 0. The market's margin forecast is
`-closing_spread_home`, and `clv = margin_forecast - (-closing_spread_home)`: positive when the model expects the
home side to do better than the line says.

## betting_games: `OUT_ROOT/analytics/betting/<season>/betting_games.parquet`

| column | type | meaning |
| --- | --- | --- |
| `game_id` | string | NBA game id |
| `season` | string | `YYYY-YY` |
| `game_date` | string | `YYYY-MM-DD`, local date of the game |
| `home_team_id`, `away_team_id` | int | stats.nba.com team ids |
| `closing_spread_home` | float, nullable | closing home spread (median over books of the is_close snapshot nearest tip); null when no closing line joined or the value fell outside [-30, 30] |
| `closing_total` | float, nullable | closing total, same rule, range [150, 300] |
| `implied_home_prob` | float, nullable | home win probability from the closing moneyline with the vig removed per book (multiplicative), mean over books |
| `home_net10`, `away_net10` | float | point margin per game over the team's previous 10 games of the season (0 with none) |
| `home_net30`, `away_net30` | float | the same over the previous 30 games |
| `home_rest_days`, `away_rest_days` | float | days since the team's previous game, capped at 7 (7 for a first game) |
| `home_b2b`, `away_b2b` | float | 1 when rest_days is 1 |
| `home_flag` | float | 1 on every row (the row is the home side's game) |
| `home_pf_std`, `home_pa_std`, `away_pf_std`, `away_pa_std` | float | season-to-date points for and against per game, previous games only (0 with none) |
| `home_rapm_sum`, `away_rapm_sum` | float, nullable | sum of player RAPM of the team from `OUT_ROOT/analytics/lineups/<season>/player_rapm*.parquet` when it carries (team_id, rapm, as_of) with `as_of` before the game; null otherwise (the lineups lane has not written dated values) |
| `home_margin` | float | home score minus away score (target; never a feature) |
| `home_cover` | float, nullable | 1 when the home side covers, 0 when not, null on a push or without a spread (target) |
| `heldout` | bool | true for rows of the held-out season (set aside before any fit, scored once) |

The FEATURES constant of `analytics/betting/features.py` lists the thirteen feature columns in order;
`feature_columns(df)` appends the RAPM pair when present with a value. Every feature is computed from games strictly
before `game_date` (rolling values shifted by one game within the team's season). The spread, the total and the
implied probability are lines, not features: they enter only the CLV target and the ridge model's conversion of a
margin forecast into a cover probability. `features.assert_time_forward` refuses a feature frame holding any Phase 0
label (analytics.schema) or any of these target and line columns; `features.assert_no_leak` also refuses a column
whose absolute correlation with `home_margin` exceeds 0.95.

## betting_predictions: `OUT_ROOT/analytics/betting/<season>/betting_predictions.parquet`

| column | type | meaning |
| --- | --- | --- |
| `game_id`, `season` | string | as above |
| `p_cover` | float | the selected cover model's probability that the home side covers |
| `margin_forecast` | float | the selected margin model's forecast of `home_margin` |
| `clv` | float | `margin_forecast - (-closing_spread_home)` (null without a spread) |
| `stake` | float | +1 on the home cover, -1 on the away cover, 0 no bet, by the staking rule of `train.py` (p_cover above 0.55 for a side; VIG 4.5 percent a side) |
| `pnl` | float | +0.91 for a winning unit, -1 for a loss, 0 for a push or no bet |
| `fold` | int | the grouped CV fold that produced the out-of-fold prediction; -1 for the held-out season |
| `heldout` | bool | true for the held-out season rows (scored once) |
| `cover_model`, `margin_model` | string | the candidates selected inside the training seasons (`market`, `logistic`, `ridge`) |

## betting_summary: `OUT_ROOT/analytics/betting/<season>/betting_summary.parquet`

| column | type | meaning |
| --- | --- | --- |
| `season` | string | the season |
| `mode` | string | `cv` (out-of-fold inside the training seasons) or `heldout` (scored once) |
| `n_games` | int | rows of the season |
| `n_labelled` | int | rows with a cover label (a spread and no push) |
| `log_loss`, `brier` | float | of `p_cover` against `home_cover` over the labelled rows |
| `mse` | float | of `margin_forecast` against `home_margin` |
| `n_bets` | int | rows with a non-zero stake |
| `pnl` | float | total profit in units |
| `roi` | float | `pnl` over units staked (null without a bet) |
| `lo`, `hi` | float | 2.5 and 97.5 percentiles of the ROI over bootstrap resamples of whole games |

## Candidates, selection, staking (fixed before scoring)

Candidates: `market` (p_cover 0.5, margin forecast the line), `logistic` (cover on FEATURES, standardised, C=1),
`ridge` (margin on FEATURES, standardised, alpha=1, p_cover = Phi((mu + spread) / sigma) with sigma the training
residual sd). Selection inside the training seasons by out-of-fold log loss (cover) and MSE (margin) under grouped
folds by game (`analytics.eval.grouped_folds`), intervals by bootstrap of whole games. The held-out season (default
2024-25; else the latest present when there are at least two) is scored once by the selected models refit on every
training row. The staking rule is the module constants `STAKE_THRESHOLD`, `VIG`, `WIN_RETURN`, `UNIT` of
`analytics/betting/train.py`, written before any scoring and never tuned.

## Inputs on the Mac

`DATA_ROOT/collect/extras/odds/<season>.parquet` (section 14), `DATA_ROOT/collect/stints/games/<season>.parquet`
(game_id, season, game_date, home_team_id, away_team_id, home_score, away_score, home_win, spread_home nullable,
heldout) or, when absent, the final scores and team ids rebuilt from `DATA_ROOT/collect/stints/playbyplay/<season>.parquet`
(section 13). Optional: `OUT_ROOT/analytics/lineups/<season>/player_rapm*.parquet`.

Join: odds team names and game team ids both become a nickname key (`validate.team_key`); the odds date is the UTC
tip minus 8 hours (every NBA tip lands on its local date), and a game within one day of it matches, nearest first,
each side used once. Coverage counts are logged by `python -m analytics.betting.validate`.
