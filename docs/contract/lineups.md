# Lineups lane contract: stints, lineup ratings, player RAPM (analytics/lineups)

Tables the lineups lane reads and writes. Section numbers refer to docs/DATA_CONTRACT.md. Nothing under
`collect/stints/heldout/` is ever read; the 2015-16 held-out games never fit anything.

## Inputs (DATA_ROOT layout on the owner's Mac)

| path | table | required |
| --- | --- | --- |
| `DATA_ROOT/collect/stints/stints/<season>.parquet` | section 12 stints | yes for validate, features, train |
| `DATA_ROOT/collect/stints/playbyplay/<season>.parquet` | section 13 play-by-play | yes for `analytics.lineups.stints` |
| `DATA_ROOT/collect/stints/games/<season>.parquet` (or `DATA_ROOT/games/<season>.parquet`) | game_id, season, game_date, home_team_id, away_team_id, home_score, away_score, home_win, spread_home | optional: without it team_id is null in lineup_ratings and the builder reads the sides from the substitution rows |
| `DATA_ROOT/collect/stints/starters/<season>.parquet` | game_id (string), team_id (int), player_id (int): the five period-1 starters per team | optional |

Seasons: 2020-21 to 2024-25 (`--seasons` overrides). A file that is missing is skipped with the season.

## stints (section 12 plus added columns), `OUT_ROOT/analytics/lineups/<season>/stints.parquet`

Written by `python -m analytics.lineups.stints` from the play-by-play. The section 12 columns exactly
(`game_id, stint, start_s, end_s, duration_s, period_start, period_end, home_1..home_5, away_1..away_5,
home_ids, away_ids, n_home, n_away, lineup_ok, points_home, points_away, score_home, score_away`) plus:

| added column | type | meaning |
| --- | --- | --- |
| `season`, `split`, `heldout`, `season_type`, `game_date`, `source`, `fetched_at` | | copied from the first play-by-play row of the game when present |
| `endpoint` | string | `playbyplay+substitutions` (the builder), never `gamerotation+playbyplayv2` |

Derivation (also in the module docstring):

- Player keys: NBA.com ids when `player1_id` (out) and `player2_id` (in) are filled on a `substitution` row and
  `player1_id` on other rows. Basketball-reference rows without ids: the names of "A. One enters the game for
  C. Three" and the leading actor of other descriptions ("X. Name makes|misses", "... by X. Name"), mapped by
  the optional `roster` {name: id} else to `zlib.crc32(name) & 0x7FFFFFFF`. Those keys are stable per name,
  not NBA ids: join the section 16 rosters table to replace them.
- Starters per period: the first five players per side to appear in the period's events (actor of a
  non-substitution row, or the player going out in a substitution) before they appear coming in. Period 1
  takes the starters table first and the derivation fills the rest. A player who never appears in a period is
  never found: `n_home` or `n_away` is under 5 for that period and `lineup_ok` is false. More than five
  candidates keeps the first five.
- Cuts: period boundaries (720 s, 300 s overtimes) plus every substitution time; substitutions at the same
  elapsed second are one cut; zero-length intervals are skipped. Points use `start_s < elapsed <= end_s`.
  Times are rounded to 0.1 s.

## lineup_ratings, `OUT_ROOT/analytics/lineups/<season>/lineup_ratings.parquet`

One row per five-man lineup seen in the season, from both sides of the floor.

| column | type | meaning |
| --- | --- | --- |
| `season` | string | the season of the raw rating |
| `lineup_id` | string | the five player ids sorted ascending and joined with `\|` |
| `team_id` | Int64, nullable | the most frequent team id of the lineup (null without a games table) |
| `n_possessions` | float | the lineup's possessions in the season: sum of `duration_s / 28.8` (or of the table's `possessions` column) over its stints, both sides |
| `plus_minus_per100` | float | raw net rating of the season: signed point differential over possessions times 100 |
| `rapm_sum_per100` | float | sum of the lineup's five player RAPM coefficients from the fit named in `fit_seasons` (an unseen player counts 0) |
| `synergy_per100` | float | the lineup's ridge coefficient in the rapm_synergy fit; 0 when the lineup is not gated in the fit seasons |
| `lo`, `hi` | float | 2.5 and 97.5 percentiles of `plus_minus_per100` over `--bootstrap` resamples of the lineup's games with replacement; NaN under two games |
| `insufficient` | bool | `n_possessions` under `--gate` (300) in this season (analytics.eval.min_sample_gate) |
| `n_games` (added) | int | games the lineup appeared in |
| `fit_seasons` (added) | string | the seasons of the fit behind `rapm_sum_per100` and `synergy_per100`, joined with commas |
| `in_sample` (added) | bool | true for a training season (the fit includes the season: descriptive ratings), false for the held-out season (the fit is the training seasons only: a prediction) |

## player_rapm, `OUT_ROOT/analytics/lineups/<season>/player_rapm.parquet`

| column | type | meaning |
| --- | --- | --- |
| `season` | string | the output season (the fit is on `fit_seasons`) |
| `player_id` | int64 | |
| `rapm` | float | the player's coefficient (per 100 possessions, +1 when on the home side, -1 away, home court as the intercept) in the possession-weighted ridge, alpha by grouped CV by game |
| `lo`, `hi` | float | 2.5 and 97.5 percentiles over `--bootstrap` refits on games resampled with replacement; NaN when `--bootstrap 0` |
| `fit_seasons` (added) | string | the seasons fitted on |

For a training season the fit is the expanding window of training seasons up to and including it; for the
held-out season it is every training season.

## Side outputs

- `OUT_ROOT/analytics/lineups/validate.json` (validate CLI): the counts of `analytics.lineups.validate.validate`.
- `OUT_ROOT/analytics/lineups/<season>/features.parquet` (features CLI): the feature frame (keys, the ten
  player columns, possessions, lineup ids, team ids, points_diff and diff_per100).
- `reports/lineups.md` (train CLI): inputs, candidate table, out-of-season correlation, the held-out season
  scored once, the Mac command. A `--fixtures` run says "fixture output, not a result" on every table.

## Possessions and target

The section 12 table carries no possession count. A stint's possessions per team are `duration_s / 28.8`
(100 possessions per team per 48 minutes), or the table's own `possessions` column when it has one. The
target is `(points_home - points_away) / possessions * 100`, the stint's net rating from the home side.
