# Data contract: SportVU 2015-16 analytics dataset and extension data (v1.1)

What `sportvu/colab_contract.py` writes to `My Drive/nba_harvest/sportvu_contract/` (sections 1 to 8) and
what the collectors in `collect/` write under `nba_harvest/collect/` (sections 9 to 15), and what every
consumer may assume. Any change to a column, a type or a rule below bumps `CONTRACT_VERSION` in the
script and the `contract_version` field of `manifest.json`. v1.1 adds the provenance columns, the
play-by-play columns from NBA.com, and the shots, aggregates, stints, odds and injuries tables of the
extension data requirements (NBA CV V3: Extension Data Requirements, Nov 2026).

## 1. Source and coverage

- SportVU player tracking (STATS LLC for the NBA) as republished at
  `github.com/linouk23/NBA-Player-Movements`, `data/2016.NBA.Raw.SportVU.Game.Logs/<game>.7z`.
- One archive per game, `<game>` = `MM.DD.YYYY.AWAY.at.HOME` (SportVU team abbreviations, e.g.
  BKN, CHA, PHX). Every archive in the listing is converted.
- Regular season games from 2015-10-27 to 2016-01-23, 25 Hz.
- Court coordinates in feet, x 0..94 and y 0..50 from a corner, the same frame as `court/` and
  `court33`. z is the ball height in feet.
- Not for redistribution: the Drive folder is a private working copy (see its README).

## 2. Dedupe and ordering (identical to `sportvu/fetch.py`)

The source repeats moments across consecutive events. A moment is kept once per
`(period, game clock rounded to 0.01 s, x of its last listed player rounded to 0.01 ft)`; moments
without a game clock or without entities are dropped. Kept moments are sorted by
`(period, game clock descending, timestamp)` and numbered `frame` 0..N-1 within the game.
Positions are rounded to 0.01 ft as `fetch.parse` rounds them.

## 3. Files

| file | content |
| --- | --- |
| `<game>.parquet` | frames table (below) |
| `<game>_pbp.parquet` | play-by-play table (below), only when the fetch and join succeeded |
| `heldout/<game>.parquet`, `heldout/<game>_pbp.parquet` | the same two files for held-out games |
| `manifest.json` | one entry per listed game, rewritten after every game |
| `README.md` | source, coverage, layout, held-out rule, redistribution notice |
| `run_report.json` | the last run's timed steps and verdict |

Parquet files are written with zstd compression by pyarrow. A file is complete when its footer reads;
the writer copies to `<name>.part` and renames, so a `.part` file is never a dataset file.

## 4. Frames table: `<game>.parquet`

One row per tracked player per frame. A frame with all ten players on the floor has ten rows; the
frame-level columns repeat on each of its rows. Rows are ordered by `frame`, then by the source's
player order within the frame (home players first in the source).

| column | type | meaning |
| --- | --- | --- |
| `game` | string | `MM.DD.YYYY.AWAY.at.HOME` |
| `gameid` | string | NBA game id from the source, e.g. `0021500381` |
| `date` | string | `YYYY-MM-DD` |
| `period` | int8 | 1..4, 5+ for overtime |
| `frame` | int32 | 0..N-1 within the game (see dedupe and ordering) |
| `ts_ms` | int64 | source wall-clock timestamp, milliseconds |
| `game_clock_s` | float32 | seconds remaining in the period |
| `shot_clock_s` | float32, nullable | seconds on the shot clock; null when the source has none |
| `n_players` | int8 | players tracked in this frame (rows of this frame) |
| `ball_x`, `ball_y`, `ball_z` | float32, nullable | ball position in feet; null when the frame has no ball entity |
| `team_id` | int32 | NBA team id |
| `team` | string | SportVU abbreviation (GSW, BKN, ...) |
| `home` | bool | the player's team is the home team |
| `player_id` | int32 | NBA player id |
| `player` | string | "First Last" from the source roster |
| `jersey` | string, nullable | jersey number as the source gives it |
| `x`, `y` | float32 | player position in feet |

Per-frame values (clock, ball) are recovered by taking the first row of each `frame`.
Clock gaps: `manifest.json` lists, per quarter, every pair of consecutive frames whose game clock
differs by more than 2 s (`gaps_over_2s`, as `fetch.coverage_report` counts them). Frames are not
interpolated across gaps.

## 5. Play-by-play table: `<game>_pbp.parquet`

basketball-reference play-by-play (`fetch_pbp.fetch_game`, box score code `YYYYMMDD0HOME` with
basketball-reference abbreviations: BRK, CHO, PHO for BKN, CHA, PHX) joined to the frames.

| column | type | meaning |
| --- | --- | --- |
| `game` | string | as above |
| `event` | int32 | 0..M-1 in page order |
| `period` | int8 | period of the event |
| `clock_s` | float32 | game clock of the event, seconds remaining |
| `team` | string, nullable | `away`, `home` or null |
| `desc` | string | event text |
| `points` | int8 | points scored on the event |
| `score` | string, nullable | running `away-home` score |
| `kind` | string | `shot_made`, `shot_missed`, `ft_made`, `ft_missed`, `turnover`, `rebound`, `foul`, `other` |
| `frame` | int32 | the frame of the same period whose `game_clock_s` is nearest, when within 2 s; else -1 |
| `frame_clock_diff_s` | float32, nullable | that clock distance; null when `frame` is -1 |
| `source`, `endpoint`, `season`, `fetched_at` | string | provenance (section 9); `source` is `stats.nba.com` or `basketball-reference.com` |
| `event_id` | int32, nullable | NBA.com EVENTNUM (null for basketball-reference rows) |
| `event_type`, `action_type` | int16, nullable | NBA.com EVENTMSGTYPE and EVENTMSGACTIONTYPE |
| `player1_id`, `player2_id`, `player3_id`, `team_id` | int32, nullable | NBA.com player and team ids of the event |
| `clock_str` | string, nullable | the clock as the source printed it |

`kind` from NBA.com adds `substitution`, `timeout`, `jump_ball`, `period_start`, `period_end`, `violation`,
`ejection`, `instant_replay` to the basketball-reference set; `team` is the side of `team_id` (home or away),
or the side whose description column was filled. `score` is normalised to `away-home`. Free throws are
`ft_made` or `ft_missed` by the MISS marker in the description; points are 3 for a made shot whose
description says 3PT, else 2, and 1 for a made free throw. The multi-season play-by-play of
`collect/stints.py` (section 13) uses the same columns with `frame` -1 and a `game_id` column.

Join recipe: `frames.frame == pbp.frame` for the event's frame; the state of play at any frame is the
last event of the same period with `clock_s >= game_clock_s`. A game whose fetch failed has no pbp
file and `pbp_joined: false` in the manifest with the error; `--retry-pbp` fills it in later without
touching the frames.

## 6. Manifest: `manifest.json`

```
{"contract_version": 1, "contract": "docs/DATA_CONTRACT.md", "source": ..., "coverage": [...],
 "updated": ..., "heldout_rule": ..., "heldout": ["<game>", ...],
 "counts": {"games", "ok", "failed", "source_broken", "pbp_joined", "heldout"},
 "games": {"<game>": {
    "game", "date", "away", "home",
    "split":        "train" | "heldout_game" | "heldout_arena" | "heldout_era" | null,
    "heldout":      true when split is a held-out split,
    "status":       "ok" | "failed" | "source_broken",
    "error":        null or the recorded exception,
    "source_bytes", "moments", "frames_rows",
    "gaps_over_2s": {"count", "seconds", "by_quarter": {"1": [[from_s, to_s], ...], ...}},
    "pbp_joined":   bool,
    "pbp":          null or {"code", "events", "matched", "error"},
    "parquet", "parquet_bytes", "pbp_parquet", "seconds", "written"}}}
```

`status` meanings: `ok` frames are on Drive and read back; `source_broken` the archive did not
extract or parse, or held no moments (truncated at the source; recorded, not retried while the
archive's size is unchanged); `failed` anything else (retried on the next run).

## 7. Held-out games

`split` comes from `sportvu/splits.json`: a game whose SportVU id is listed there takes that split, and
every game at an arena in `heldout_arena.arenas` is `heldout_arena` even when the file does not list it
by name. Held-out games are written under `heldout/` and listed in `manifest.heldout`. No training job
may read anything under `heldout/`: `sportvu.splits.refuse_heldout` rejects their ids, and the file
names carry the id, so a path check is enough.

## 8. Build order and resumability

Games are converted in this order: the training games of `splits.json`, then every other GSW game by
date, then the rest by date. A game whose frames parquet is on Drive and reads back is skipped, so a
run that stops resumes where it was. Raw archives and JSON never stay on the VM after a game.

## 9. Provenance columns (every table the collectors write)

| column | type | meaning |
| --- | --- | --- |
| `source` | string | the host the rows came from: `stats.nba.com`, `basketball-reference.com`, `bref` (the shot chart fallback, section 10a), `the-odds-api.com`, an archive file |
| `endpoint` | string | the endpoint or page, e.g. `shotchartdetail`, `gamerotation+playbyplayv2`, `boxscores/shot-chart/<code>.html` |
| `season` | string | `YYYY-YY` of the request (the odds and injury tables derive it from the date) |
| `fetched_at` | string | UTC `YYYY-MM-DDTHH:MM:SSZ` of the response (from the raw cache record) |

Every response is cached as received under `collect/raw/<endpoint>/<sha1 of the parameters>.json.gz` and
never requested twice. Each family's `manifest.json` records per unit (endpoint, season, id): status
(`ok`, `empty`, `failed`), rows, the exact parameters sent, fetched_at, seconds, error. Season-wide
parquet files are `collect/<family>/<endpoint>/<season>.parquet`; rows that belong to a held-out game of
`sportvu/splits.json` (2015-16, by date and teams) carry `split` and `heldout=true` and are compacted into
`collect/<family>/heldout/<endpoint>/<season>.parquet` instead, never into the fitting files.

## 10. Shots table: `collect/shots/shotchartdetail/<season>.parquet`

One row per field-goal attempt, league-wide, from NBA.com ShotChartDetail (one request per team and
season, PlayerID 0, ContextMeasure FGA). The SportVU-derived fields are null here and filled by the
SportVU join of a later step; the join key is (`sportvu_game`, `period`, `clock_s`) against the frames.

| column | type | meaning |
| --- | --- | --- |
| `game_id` | string | NBA game id, e.g. `0021500381` |
| `event_id` | int32 | GAME_EVENT_ID, the play-by-play `event_id` of the attempt |
| `player_id`, `player`, `team_id`, `team` | int32, string | the shooter and the shooter's team |
| `period` | int8 | period |
| `clock_s` | float32 | seconds remaining in the period |
| `loc_x`, `loc_y` | float32 | NBA.com court coordinates in tenths of a foot from the basket (x across, y toward half court) |
| `shot_distance_ft` | float32 | distance in feet as NBA.com reports it |
| `shot_type` | string | `2PT` or `3PT` |
| `action_type` | string | NBA.com action type, e.g. `Pullup Jump shot` |
| `zone_basic`, `zone_area`, `zone_range` | string | NBA.com shot zones |
| `made` | bool | SHOT_MADE_FLAG |
| `game_date`, `home_team`, `away_team` | string | `YYYY-MM-DD` and NBA.com abbreviations |
| `sportvu_game` | string | `MM.DD.YYYY.AWAY.at.HOME`, the Step 1 game id (meaningful for 2015-16) |
| `split`, `heldout` | string, bool | the held-out split of the game (2015-16 only) |
| `season_type` | string | `Regular Season` or `Playoffs` |
| `release_frame` | int32, nullable | the SportVU frame of the release (filled by the SportVU join) |
| `closest_defender_id` | int32, nullable | defender nearest the shooter at release (SportVU join) |
| `closest_defender_dist_ft` | float32, nullable | that distance in feet (SportVU join) |
| `defender_closing_speed_ftps` | float32, nullable | the closest defender's speed toward the shooter at release (SportVU join) |
| provenance | | section 9 |

## 10a. Shots from basketball-reference: `collect/bref_shots/shotchart/<season>.parquet`

The fallback for section 10 while stats.nba.com is unreachable (`collect/bref_shots.py`): one row per field-goal
attempt of a SportVU game, parsed from the shot chart page `boxscores/shot-chart/<code>.html` (one page per game,
at most 17 requests a minute, the page cached under `collect/raw/bref_shotchart/`). Rows of held-out games are
compacted into `collect/bref_shots/heldout/shotchart/<season>.parquet`. The table carries every section 10 column
so the two union; the ids NBA.com gives (`game_id`, `event_id`, `player_id`, `team_id`) are null here, and the
SportVU-derived fields are null until the SportVU join.

| column | type | meaning |
| --- | --- | --- |
| `game`, `sportvu_game` | string | `MM.DD.YYYY.AWAY.at.HOME` |
| `br_code` | string | the basketball-reference box score code, `YYYYMMDD0HOME` (BRK, CHO, PHO for BKN, CHA, PHX) |
| `player`, `player_slug` | string | the shooter as the page names them, and the basketball-reference player slug (`currystep01`) |
| `team`, `team_br`, `team_side` | string | the shooter's team as the SportVU abbreviation, as basketball-reference writes it, and `home` or `away` |
| `period` | int8 | 1..4, 5+ for overtime |
| `clock_s`, `clock_str` | float32, string | seconds remaining in the period, and the clock as the page printed it |
| `made` | bool | the marker's `make` or `miss` class |
| `shot_type` | string | `2PT` or `3PT` |
| `shot_distance_ft` | float32, nullable | the distance the page states, whole feet |
| `px_left`, `px_top` | float32 | the marker's anchor (its top-left corner) on the 500 px wide chart as the page gives it: 10 px per foot, baseline at the top; the basket centre is at (250, 52.5) on the image and at (239.5, 35.5) in anchor coordinates (`collect.bref_shots.HOOP_PX`, the glyph offset fitted on the saved pages) |
| `court_x`, `court_y` | float32 | the shot in the section 1 court frame attacking the LEFT basket (rim at 5.25, 25): `court_x` = 5.25 + feet from the attacked baseline, `court_y` = 25 + feet across (page left is small y) |
| `basket` | string | `left` for every row as written; `collect.bref_shots.mirror_to_right(x, y)` gives the row for the right basket (94 - x, 50 - y) once the frames say which basket the team attacked in that period |
| `loc_x`, `loc_y` | float32 | the section 10 convention, tenths of a foot from the basket: `loc_x` across (page right positive; the sign against NBA.com is unverified until a game is collected from both sources), `loc_y` toward half court |
| `dist_resid_ft` | float32, nullable | pixel distance from the basket minus `shot_distance_ft`: the per-row check of the pixel frame (whole-foot rounding gives up to 0.5 ft) |
| `desc` | string | the marker's tip text, lines joined with ` \| ` |
| `game_date`, `home_team`, `away_team`, `split`, `heldout`, `season_type` | | as in section 10 (SportVU abbreviations) |
| `game_id`, `event_id`, `player_id`, `team_id`, `release_frame`, `closest_defender_id`, `closest_defender_dist_ft`, `defender_closing_speed_ftps` | nullable | null (section 10 columns kept for the union) |
| provenance | | `source` is `bref`, `endpoint` is `boxscores/shot-chart/<code>.html`, `season` from the game date |

Per game the manifest records `bref_shots` (rows), `markers` (markers on the page), `pbp_shots` (field-goal attempts
in the play-by-play: the Step 1 `<game>_pbp.parquet`, `data/pbp/<code>.json` or the basketball-reference pbp page
with `--fetch-pbp`), `pbp_source`, `by_team`, `made` and `calibration`: the basket pixel fitted from the stated
distances (`hoop_px`, `hoop_shift_px`, `resid_default_ft`, `resid_fit_ft`, `ok`). A unit fails when the fitted
basket sits more than 10 px from the assumed one, which means the page's pixel frame changed. `--report` prints
the recovered count against the play-by-play count per game. The chart draws the half court only: on ten games
(`reports/bref_shots_recovery.md`) every attempt inside it had a marker and the only attempts without one were
heaves from 46 ft and beyond, so `bref_shots` runs 0 to 2 short of `pbp_shots` per game.

## 11. Aggregates table: `collect/<family>/aggregates/<season>.parquet`

The long form of every tracking, matchup, play-type, hustle and lineup endpoint, one row per numeric
metric, for validation of every model. The wide tables (`collect/<family>/<endpoint>/<season>.parquet`)
keep every source column in snake_case.

| column | type | meaning |
| --- | --- | --- |
| entity ids | int32, string, nullable | whichever the endpoint names: `player_id`, `player_name`, `team_id`, `team_abbreviation`, `group_id` (lineup), `off_player_id`, `def_player_id`, `game_id` |
| `season`, `season_type` | string | the request |
| `endpoint` | string | the endpoint |
| split columns | string, nullable | `split_name`, `split_value` (closest-defender or touch-time range), `pt_measure_type`, `player_or_team`, `play_type`, `type_grouping`, `measure_type` |
| `metric` | string | the source column, snake_case (rank columns excluded) |
| `value` | float64 | the value |
| `heldout` | bool, nullable | per-game rows of held-out games (compacted into heldout/) |
| provenance | | section 9 |

Per player-game tracking (`boxscoreplayertrackv3`), per-game matchups (`boxscorematchupsv3`) and rotations
(`gamerotation`) are per-game endpoints, collected with `--per-game`, keyed by `game_id`.

## 12. Stints table: `collect/stints/stints/<season>.parquet`

One row per interval between any substitution of a game, from GameRotation boundaries and the
play-by-play scores (`collect/stints.py`).

| column | type | meaning |
| --- | --- | --- |
| `game_id` | string | NBA game id |
| `stint` | int32 | 0..n-1 within the game |
| `start_s`, `end_s`, `duration_s` | float32 | elapsed game seconds since tip-off (720 s periods, 300 s overtimes) |
| `period_start`, `period_end` | int8 | the periods the stint starts and ends in |
| `home_1` .. `home_5`, `away_1` .. `away_5` | int32, nullable | the player ids on court, ascending; null when fewer than five were found |
| `home_ids`, `away_ids` | string | the same ids joined with `\|` |
| `n_home`, `n_away`, `lineup_ok` | int8, bool | how many players the rotation put on court; `lineup_ok` when both are five |
| `points_home`, `points_away` | int16 | points scored in the stint (events with `start_s < elapsed <= end_s`) |
| `score_home`, `score_away` | int16 | running score at the end of the stint |
| `split`, `heldout`, `season_type`, `game_date` | | as above |
| provenance | | `endpoint` is `gamerotation+playbyplayv2` |

Rotation rows (`collect/stints/rotation/<season>.parquet`) keep GameRotation as returned (`in_time_real`,
`out_time_real` in tenths of a second; the builder reads values under 3000 as seconds).

## 13. Multi-season play-by-play: `collect/stints/playbyplay/<season>.parquet`

The play-by-play columns of section 5 (NBA.com fields included), `frame` -1, plus `game_id`, `split`,
`heldout`, `season_type`, `game_date`. Substitutions are `kind = substitution` with `player1_id` out and
`player2_id` in.

## 14. Odds table: `collect/extras/odds/<season>.parquet` (only if pursued)

| column | type | meaning |
| --- | --- | --- |
| `game_key` | string | `YYYY-MM-DD_AwayTeamName_HomeTeamName` from the odds feed (no NBA id; join on date and names) |
| `commence_time` | string | tip-off, UTC |
| `home_team`, `away_team` | string | names as the feed gives them |
| `book` | string | bookmaker key |
| `market` | string | `h2h` (moneyline), `spreads`, `totals` |
| `side` | string | team name, `Over` or `Under` |
| `point` | float32, nullable | the spread or total line |
| `price` | int32 | American odds |
| `snapshot_at` | string | the snapshot time the feed served |
| `is_close` | bool | true for the snapshot taken `--close-minutes` before tip-off (the closing line) |
| provenance | | `source` `the-odds-api.com`, `endpoint` `historical/odds` |

## 15. Injuries table: `collect/extras/injuries/<season>.parquet` (only if pursued)

| column | type | meaning |
| --- | --- | --- |
| `player`, `team` | string | as the report lists them |
| `date` | string | `YYYY-MM-DD` of the report (or game) |
| `status` | string | Out, Questionable, Doubtful, Probable, Available |
| `reason` | string | the report's reason text |
| other columns | | every other column of the archive file, as named there |
| provenance | | `source` names the archive, `endpoint` the file |

The column mapping (`collect.extras.INJURY_ALIASES`) is by name and is printed per file; check it on the
first file before trusting the table.

## 16. Rosters table: `analytics/rosters/<season>/<sportvu_game>.parquet`

Sections 16 to 20 are the Phase 0 foundation tables. Each is parquet, one file per game, under
`analytics/<table>/<season>/<sportvu_game>.parquet`; a game listed as held out in `sportvu/splits.json`
is written under `analytics/<table>/heldout/<season>/<sportvu_game>.parquet` instead and never fits or
tunes anything. `sportvu_game` is the Step 1 game id (`MM.DD.YYYY.AWAY.at.HOME`, e.g.
`12.16.2015.PHX.at.GSW`), the `game` column of sections 4 and 5. Column names, types, nullability and
the enum vocabularies are encoded in `analytics/schema.py` (`TABLES`, `NULLABLE`, `ENUMS`); `schema.cast`
gives a frame the contract dtypes (nullable ints as pandas `Int32`) and `schema.validate` lists what a
frame gets wrong. Frame indices that could not be localised are `-1`.

One row per player on a game's SportVU roster (the distinct `player_id` of the section 4 frames),
joined to the basketball-reference and play-by-play spellings of the name.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `player_id` | int32 | NBA player id (section 4) |
| `player` | string | "First Last" from the SportVU roster |
| `team_id` | int32 | NBA team id |
| `team` | string | SportVU abbreviation (GSW, BKN, ...) |
| `home` | bool | the player's team is the home team |
| `jersey` | string, nullable | jersey number as the source gives it |
| `bref_slug` | string, nullable | basketball-reference player slug, e.g. `curryst01`; null when unmatched |
| `bref_name` | string, nullable | the name as basketball-reference prints it; null when unmatched |
| `pbp_name` | string, nullable | the name as the play-by-play text prints it; null when unmatched |
| `match_method` | string | how the names were matched: `exact`, `lastname`, `fuzzy`, `none` |

## 17. Shot frames table: `analytics/shot_frames/<season>/<sportvu_game>.parquet`

One row per field-goal attempt of the play-by-play (`kind` in `shot_made`, `shot_missed`, section 5),
localised in the frames of section 4.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int32 | play-by-play event index (section 5 `event`) |
| `period` | int8 | period of the event |
| `clock_s` | float32 | game clock of the event, seconds remaining (the logged clock marks the outcome) |
| `made` | bool | the shot was made (TARGET) |
| `pbp_shooter_name` | string, nullable | shooter's name as the play-by-play prints it (POST-OUTCOME) |
| `shooter_id` | int32, nullable | the shooter's NBA player id from the roster join (section 16); null when unmatched |
| `release_frame` | int32 | frame of the release; `-1` when not found |
| `apex_frame` | int32 | frame of the ball's highest point after release; `-1` when not found (POST-OUTCOME) |
| `outcome_frame` | int32 | frame where the ball falls back through rim height; `-1` when not found (POST-OUTCOME) |
| `shooter_at_release_id` | int32, nullable | the player nearest the ball at the release frame; null when no release frame |
| `identity_match` | bool, nullable | `shooter_at_release_id == shooter_id` (TARGET); null when either side is null |
| `confidence` | float32 | 0 to 1, the localiser's confidence in the release frame |
| `confidence_flag` | string | `high`, `low`, `unmatched` |

Localisation rules. The candidate window is the frames of the same period whose game clock lies between
`clock_s - 0.5 s` and `clock_s + 3.0 s`: the logged clock marks the outcome and the release is earlier,
which is a larger clock value. The release frame is the last frame in the window where the ball is
within 2.5 ft horizontally of the shooter and below 8 ft, after which the ball height rises for at least
0.2 s. The apex frame is the frame of the highest ball `z` after the release. The outcome frame is the
frame where the ball falls back through 10 ft near the rim. A row whose release frame was not found has
the three frame columns at `-1`, `shooter_at_release_id` null and `confidence_flag` `unmatched`.

Validation status. Validated 2026-11-11 by the stage 3 shot audit (reports/shot_audit.md) for "shot event
and shooter correct, release within about 0.4 s". The four localisation constants are pinned: apex search
2 s, rim radius 6 ft, flight 25 frames, high-confidence 0.6. The audit drew 604 training shots (no held-out
game); `pass_mark.share` was 0.9671 (323 of 334 `high` rows within 10 frames, 5 marked bad), against a pass
mark of 0.90 (`analytics.audit_shots --summarise`, docs/contract/phase0_driver.md). What this
validates, for `confidence_flag` `high` rows: the shot event and the shooter are correct and the release frame
is within about 0.4 s (10 frames at 25 Hz) of the release an auditor sees in the court animation. It does
not validate 2-frame (0.08 s) release precision: the animation does not resolve the release that finely, and
the audit's 2 and 5 frame shares are information only. A consumer that needs the release to a few frames
(for example a feature read at the exact release frame) treats that precision as unverified until the
independent check listed under open items in STATUS.md is done.

## 18. Rebound frames table: `analytics/rebound_frames/<season>/<sportvu_game>.parquet`

One row per missed shot of the play-by-play (`kind` `shot_missed`, plus `ft_missed` on the last free
throw when the collector pursues it), localised in the frames of section 4 and joined to the play-by-play
rebound that follows it.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int32 | play-by-play event index of the missed shot |
| `rebound_event` | int32, nullable | play-by-play event index of the rebound (`kind` `rebound`); null when none follows |
| `period` | int8 | period of the event |
| `clock_s` | float32 | game clock of the missed shot, seconds remaining |
| `rim_frame` | int32 | frame where the ball reaches the rim (the section 17 outcome frame); `-1` when not found |
| `rebound_frame` | int32 | frame where a player first secures the ball after the rim frame; `-1` when not found (POST-OUTCOME) |
| `rebounder_id` | int32, nullable | the first player within 2 ft of the ball with ball `z` under 7 ft after the rim frame (TARGET) |
| `pbp_rebounder_name` | string, nullable | rebounder's name as the play-by-play prints it (TARGET) |
| `pbp_rebounder_id` | int32, nullable | the play-by-play rebounder's NBA player id from the roster join (TARGET); null when unmatched or a team rebound |
| `identity_match` | bool, nullable | `rebounder_id == pbp_rebounder_id` (TARGET); null when either side is null |
| `confidence` | float32 | 0 to 1, the localiser's confidence in the rebound frame |
| `confidence_flag` | string | `high`, `low`, `unmatched` |

Localisation rules. The rim frame is the section 17 outcome frame of the same `event`. After the rim
frame, the rebound frame is the first frame where some player is within 2 ft of the ball while the
ball `z` is under 7 ft, and `rebounder_id` is that player. A row with no rim frame has both frame
columns at `-1`, `rebounder_id` null and `confidence_flag` `unmatched`.

## 19. Possessions table: `analytics/possessions/<season>/<sportvu_game>.parquet`

One row per possession, segmented from the play-by-play of section 5 and spanned onto the frames of
section 4. Possessions are numbered 0..P-1 within the game in time order.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | 0..P-1 within the game |
| `period` | int8 | period of the possession |
| `start_frame` | int32 | first frame of the span |
| `end_frame` | int32 | last frame of the span (POST-OUTCOME) |
| `start_clock_s` | float32 | game clock at `start_frame` |
| `end_clock_s` | float32 | game clock at `end_frame` (POST-OUTCOME) |
| `start_event` | int32 | play-by-play event index that opens the possession |
| `end_event` | int32 | play-by-play event index that ends it (POST-OUTCOME) |
| `offense_team_id` | int32, nullable | NBA team id of the offence; null when it could not be assigned |
| `end_kind` | string | `shot_made`, `shot_missed`, `free_throws`, `turnover`, `period_end`, `other` (TARGET) |
| `points` | int16 | points the offence scored on the possession (TARGET) |
| `gap` | bool | the span holds a frame-to-frame game clock jump over 2 s |
| `n_frames` | int32 | frames in the span (`end_frame - start_frame + 1` when there is no gap) (POST-OUTCOME) |

Gap rule. `gap` is true when any two consecutive frames inside `[start_frame, end_frame]` differ in
`game_clock_s` by more than 2 s (the section 4 `gaps_over_2s` rule applied to the span). Frames are not
interpolated across the gap; a trainer that needs a continuous span filters on `gap == false`.

## 20. States table (1 Hz): `analytics/states_1hz/<season>/<sportvu_game>.parquet`

Defined now, filled in Phase 2. One row per possession per second of play: the frame nearest each
whole second of elapsed time inside the possession, with the ten players split into the offence and
the defence of that possession and each side sorted by `player_id` (so `off1_id < off2_id < ... <
off5_id` and likewise for the defence).

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the section 4 frame the state is read from |
| `period` | int8 | period |
| `game_clock_s` | float32 | game clock at `frame` |
| `shot_clock_s` | float32, nullable | shot clock at `frame`; null when the source has none |
| `elapsed_in_possession_s` | float32 | seconds since `start_frame` of the possession |
| `offense_team_id` | int32 | NBA team id of the offence |
| `handler_id` | int32, nullable | the offensive player nearest the ball; null when no ball entity |
| `ball_x`, `ball_y`, `ball_z` | float32 | ball position in feet |
| `off1_id` .. `off5_id` | int32 | offensive player ids, sorted ascending |
| `off1_x`, `off1_y` .. `off5_x`, `off5_y` | float32 | their positions in feet, same order |
| `def1_id` .. `def5_id` | int32 | defensive player ids, sorted ascending |
| `def1_x`, `def1_y` .. `def5_x`, `def5_y` | float32 | their positions in feet, same order |
| `handler_nearest_def_dist_ft` | float32, nullable | distance from the handler to the nearest defender; null when no handler |
| `points_to_end` | int16 | points the offence scores from this state to the end of the possession (TARGET) |
| `possession_end_kind` | string | the section 19 `end_kind` of the possession: `shot_made`, `shot_missed`, `free_throws`, `turnover`, `period_end`, `other` (TARGET) |

The TARGET columns of every table (`points_to_end`, `possession_end_kind` here; `made`, `identity_match`,
`rebounder_id`, `pbp_rebounder_id`, `pbp_rebounder_name`, `end_kind`, `points` in sections 17 to 19) and
the POST-OUTCOME columns of every table (measured at or after the outcome: `apex_frame`, `outcome_frame`,
`pbp_shooter_name` in section 17; `rebound_frame` in section 18; `end_frame`, `end_clock_s`, `end_event`,
`n_frames` in section 19) are excluded from every feature frame by `analytics.schema.assert_time_forward`,
which every trainer calls on its feature frame and which checks the union over all five tables
(`analytics.schema.ALL_EXCLUDED`), whatever table the frame was built for. `analytics.schema.feature_allowlist`
lists the columns a trainer may use: neither targets, nor post-outcome columns, nor the id columns
`sportvu_game`, `event`, `rebound_event`, `possession`, `frame`, `player_id`, `shooter_id`. `release_frame`,
`rim_frame` and the possession `start_*` columns stay allowed.

## 21. Extension contracts: `docs/contract/`

Each extension and infrastructure lane documents its inputs, output tables, columns, dtypes, paths,
protocol and leakage rules in its own page. The pages are the contract for the tables written under
`OUT_ROOT/analytics/<lane>/`; this file stays the contract for the five Phase 0 tables above.

| page | what it specifies |
| --- | --- |
| `docs/contract/fixtures.md` | the shared synthetic generator (`analytics/fixtures.py`): every table above plus labels, aggregates, stints, odds; the held-out routing of fixture games |
| `docs/contract/ingestion.md` | the basketball-reference drivers (`collect/bref_schedule`, `bref_seasons`, `bref_boxscores`, `bref_lineups`, `positions`, `drive`) and the tables they write under `DATA_ROOT/collect/` |
| `docs/contract/phase0_driver.md` | `analytics.run_phase0` (rosters, shot frames, rebound frames, possessions over a Step 1 manifest) and `analytics.audit_shots` |
| `docs/contract/labels.md` | hand labels: `labels/{actions,screens,visited}/<game>.jsonl`, the sanctioned reader `analytics.labels.read_labels`, passes and tombstones |
| `docs/contract/interfaces.md` | the five cross-lane tables: `xfg_predictions`, `epv_states`, `actions_detected`, `coverage_predictions`, `player_rapm` |
| `docs/contract/xfg.md`, `rebounds.md`, `matchups.md`, `gravity.md`, `ghosting.md`, `embeddings.md`, `epv.md`, `actions.md`, `coverage.md`, `decisions.md`, `playcalling.md`, `lineups.md`, `winprob.md`, `load.md`, `betting.md` | one page per extension: unit, features, candidates, selection and held-out protocol, output tables, CLI |

`docs/run_order.md` gives the order to run everything on the Mac; `analytics/registry.py` holds the
dependency graph that `run_all.py` and the Makefile follow.
