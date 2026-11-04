# Data contract: SportVU 2015-16 analytics dataset (v1)

What `sportvu/colab_contract.py` writes to `My Drive/nba_harvest/sportvu_contract/`, and what every
consumer may assume. Any change to a column, a type or a rule below bumps `CONTRACT_VERSION` in the
script and the `contract_version` field of `manifest.json`.

## Source and coverage

- SportVU player tracking (STATS LLC for the NBA) as republished at
  `github.com/linouk23/NBA-Player-Movements`, `data/2016.NBA.Raw.SportVU.Game.Logs/<game>.7z`.
- One archive per game, `<game>` = `MM.DD.YYYY.AWAY.at.HOME` (SportVU team abbreviations, e.g.
  BKN, CHA, PHX). Every archive in the listing is converted.
- Regular season games from 2015-10-27 to 2016-01-23, 25 Hz.
- Court coordinates in feet, x 0..94 and y 0..50 from a corner, the same frame as `court/` and
  `court33`. z is the ball height in feet.
- Not for redistribution: the Drive folder is a private working copy (see its README).

## Dedupe and ordering (identical to `sportvu/fetch.py`)

The source repeats moments across consecutive events. A moment is kept once per
`(period, game clock rounded to 0.01 s, x of its last listed player rounded to 0.01 ft)`; moments
without a game clock or without entities are dropped. Kept moments are sorted by
`(period, game clock descending, timestamp)` and numbered `frame` 0..N-1 within the game.
Positions are rounded to 0.01 ft as `fetch.parse` rounds them.

## Files

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

## Frames table: `<game>.parquet`

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

## Play-by-play table: `<game>_pbp.parquet`

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

Join recipe: `frames.frame == pbp.frame` for the event's frame; the state of play at any frame is the
last event of the same period with `clock_s >= game_clock_s`. A game whose fetch failed has no pbp
file and `pbp_joined: false` in the manifest with the error; `--retry-pbp` fills it in later without
touching the frames.

## Manifest: `manifest.json`

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

## Held-out games

`split` comes from `sportvu/splits.json`: a game whose SportVU id is listed there takes that split, and
every game at an arena in `heldout_arena.arenas` is `heldout_arena` even when the file does not list it
by name. Held-out games are written under `heldout/` and listed in `manifest.heldout`. No training job
may read anything under `heldout/`: `sportvu.splits.refuse_heldout` rejects their ids, and the file
names carry the id, so a path check is enough.

## Build order and resumability

Games are converted in this order: the training games of `splits.json`, then every other GSW game by
date, then the rest by date. A game whose frames parquet is on Drive and reads back is skipped, so a
run that stops resumes where it was. Raw archives and JSON never stay on the VM after a game.
