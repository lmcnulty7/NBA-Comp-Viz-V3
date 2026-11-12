# Lineups lane contract: stints, lineup ratings, player RAPM (analytics/lineups)

Tables the lineups lane reads and writes. Section numbers refer to docs/DATA_CONTRACT.md. Nothing under
`collect/stints/heldout/` is ever read; the 2015-16 held-out games never fit anything.

## Inputs (DATA_ROOT layout on the owner's Mac)

| path | table | required |
| --- | --- | --- |
| `DATA_ROOT/collect/stints/stints/<season>.parquet` | section 12 stints from a collector | optional: when present it is read in place of the built stints for that season |
| `DATA_ROOT/collect/stints/playbyplay/<season>.parquet` | section 13 play-by-play | yes for `analytics.lineups.stints` |
| `DATA_ROOT/collect/stints/games/<season>.parquet` (or `DATA_ROOT/games/<season>.parquet`) | game_id, season, game_date, home_team_id, away_team_id, home_score, away_score, home_win, spread_home | optional: without it team_id is null in lineup_ratings and the builder reads the sides from the substitution rows |
| `DATA_ROOT/collect/stints/starters/<season>.parquet` | starter rows only: game_id (bref code), game (SportVU-style id when derivable), game_date, season, side (home/away), team (bref code), team_id (nullable Int64, null from the basketball-reference box score driver), player_slug, player, player_id (nullable Int64, null from that source) | optional |

Seasons: 2020-21 to 2024-25 (`--seasons` overrides). A file that is missing is skipped with the season.

Step order (run_all and `make ext-lineups`): stints, validate, features, train. The stints step builds the
stint table from the play-by-play and the starters and writes it under OUT_ROOT (below); validate, features
and train then read, per season, the first of: the collected `collect/stints/stints/<season>.parquet`,
`OUT_ROOT/analytics/lineups/<season>/stints.parquet`, `DATA_ROOT/analytics/lineups/<season>/stints.parquet`
(`analytics.lineups.read_stints`). On the Mac only play-by-play and starters exist, so the built table is
the one read. OUT_ROOT defaults to DATA_ROOT.

## stints (section 12 plus added columns), `OUT_ROOT/analytics/lineups/<season>/stints.parquet`

Written by `python -m analytics.lineups.stints` from the play-by-play. The section 12 columns exactly
(`game_id, stint, start_s, end_s, duration_s, period_start, period_end, home_1..home_5, away_1..away_5,
home_ids, away_ids, n_home, n_away, lineup_ok, points_home, points_away, score_home, score_away`) plus:

| added column | type | meaning |
| --- | --- | --- |
| `season`, `split`, `heldout`, `season_type`, `game_date`, `source`, `fetched_at` | | copied from the first play-by-play row of the game when present |
| dtypes | | cast on write to section 12: stint int32, start_s/end_s/duration_s float32, period_* and n_* int8, points_* and score_* int16, player ids Int32, lineup_ok bool |
| `endpoint` | string | `playbyplay+substitutions` (the builder), never `gamerotation+playbyplayv2` |

Derivation (also in the module docstring):

- Player keys: NBA.com ids when `player1_id` (out) and `player2_id` (in) are filled on a `substitution` row and
  `player1_id` on other rows. Basketball-reference rows without ids: the names of "A. One enters the game for
  C. Three" and the leading actor of other descriptions ("X. Name makes|misses", "... by X. Name"), mapped by
  the optional `roster` {name: id} else to `zlib.crc32(name) & 0x7FFFFFFF`. Those keys are stable per name,
  not NBA ids: join the section 16 rosters table to replace them. The name is keyed after `clean_name`:
  UTF-8 text misread as Latin-1 is repaired ("Å\xa0ariÄ\x87" becomes "Šarić") and a trailing Jr., Sr., II,
  III or IV is dropped, since the play-by-play writes "K. Oubre" and the box score starters "Kelly Oubre Jr.".
- Foul sides: basketball-reference files a defensive foul (shooting, personal, take, away from play, flagrant)
  under the fouled team, so a foul row without ids takes the side its player has on most of the game's
  other rows (`fix_foul_sides`); a fouler seen nowhere else keeps the row's side.
- Shifted period labels: a game whose periods do not run 1..max without a gap, or whose period above 4 holds a
  clock above 300 s, is dropped whole before any stint is built (`shifted_period_games`, the winprob lane's
  clock rule). The basketball-reference parser counts quarter header rows, so a repeated header shifts every
  later period up (53 of 6,416 games in 2020-21 to 2024-25). The stints step prints how many it dropped.
- Starters table join (agreed with lane I2; `STARTERS_COLUMNS` in stints.py): the game is matched on
  `game_id` when it is a play-by-play game id, else through the games table (`resolve_starters_games`): the
  `game` column against games.game or games.game_id, then (game_date, home team code of the bref code's last
  three letters) when the games table carries a home team code column (home_team, home, home_abbr or
  home_code), then game_date alone when exactly one game is on that date; an unmatched row is ignored. When
  `team_id` and `player_id` are filled the side is the team id against the games table and the key is
  `player_id`. When they are null (the basketball-reference box score driver writes full names such as
  "Eric Bledsoe") the side is the `side` column and the key is, in order: the `roster` {slug or name: id}
  mapping on player_slug, the full name or the abbreviated name; else `name_key(abbreviated(player))`, the
  crc32 key of "E. Bledsoe", the form the builder gives a basketball-reference play-by-play actor; else
  `name_key(player_slug)`. So a slug-only or full-name starters table seeds period 1 for basketball-reference
  play-by-play (name keys on both sides) and is inert for NBA.com play-by-play with player ids (an integer
  id never meets a name key) unless a roster mapping lifts the names to ids. Rows a bench player can produce
  (technical fouls, ejections, delay of game) and team ids in player1_id (at or above 1610612700, or equal to
  the row's team_id: team rebounds, team turnovers, timeouts) never make a starter.
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
| `team_id` (added) | Int64, nullable | the team the player had most possessions with in the OUTPUT season's lineups, falling back to the fit seasons for a player absent from the output season (null without a games table) |
| `as_of` (added) | string, nullable | the last `game_date` of `fit_seasons`: the rating is known from that date on and valid for games after it |
| `in_sample` (added) | bool | true for a training season (the fit includes it), false for the held-out season |
| `fit_seasons` (added) | string | the seasons fitted on |

Time-forwardness for consumers (betting reads team_id, rapm, as_of): a season-level RAPM is not time-forward
inside its own season. A row with `in_sample` true was fitted on the season it is filed under, so it must not
price a game of that season; a row with `in_sample` false (the held-out season's file) was fitted on earlier
seasons only and is valid from `as_of` for every game of the held-out season. There is no per-date refit:
a consumer that needs a rating for a game of season S uses `player_rapm_prior.parquet` of S (or the held-out
season's `player_rapm.parquet`), both `in_sample` false with `as_of` before S starts, and never a row whose
`as_of` is on or after the game date. A player missing from the table (first season in the data) has no
rating: the consumer imputes (betting: documented on its side).

For a training season the fit is the expanding window of training seasons up to and including it; for the
held-out season it is every training season.

## player_rapm_prior, `OUT_ROOT/analytics/lineups/<season>/player_rapm_prior.parquet`

Same columns as player_rapm, written for every training season after the first from the fit on the seasons
BEFORE it (the validation-pair fit), `in_sample` false, `as_of` the last game date of the previous season.
This is the dated table a consumer may use for games of that season: betting globs `player_rapm*.parquet`
and keeps rows with `in_sample` false whose `as_of` precedes the game date. The first training season has no
prior table; the held-out season's `player_rapm.parquet` already is a prior-seasons-only fit.

## Side outputs

- `OUT_ROOT/analytics/lineups/validate.json` (validate CLI): the counts of `analytics.lineups.validate.validate`.
- `OUT_ROOT/analytics/lineups/<season>/features.parquet` (features CLI): the feature frame (keys, the ten
  player columns, possessions, lineup ids, team ids, points_diff and diff_per100).
- `reports/lineups.md` (train CLI): inputs, candidate table, out-of-season correlation, the held-out season
  scored once, the Mac command. A `--fixtures` run says "fixture output, not a result" in its banner and on
  every section heading, and lists the written tables relative to OUT_ROOT.

## Fixtures

Every CLI of the lane takes `--fixtures` with `--fixture-games` (alias `--n-games`; games per fixture season) and
`--fixture-generator`: `auto` or `shared` is `analytics.fixtures.synth_stints` (called without a players frame, so
its team ids match the betting lane's fixture), `local` is the lane's own synthetic generator
`analytics/lineups/synthetic.py`, kept because it carries a starters table and player ratings large enough for
RAPM to recover (the tests rely on both). Production code never imports `tests/`. The lane splits by season
(`--heldout-season`), so it has no possessions or held-out-every knob. Every ridge solve runs inside
`analytics.eval.fit_threads()`; intervals are formatted by `analytics.eval.fmt_ci`.

## Possessions and target

The section 12 table carries no possession count. A stint's possessions per team are `duration_s / 28.8`
(100 possessions per team per 48 minutes), or the table's own `possessions` column when it has one. The
target is `(points_home - points_away) / possessions * 100`, the stint's net rating from the home side.
