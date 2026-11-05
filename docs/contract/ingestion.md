# Ingestion contract: the basketball-reference drivers (lane I2)

stats.nba.com is unreachable from the owner's machines (it drops non-browser clients; no fingerprint impersonation
is added), so basketball-reference.com is the reachable source for schedules, shot charts, play-by-play, box
scores, lineups and listed positions. The NBA.com collectors (`collect/nba_stats.py`, `shots.py`, `pbp.py`,
`stints.py`, `context.py`, `tracking.py`, `extras.py`) stay as they are. Everything below runs through
`collect/common.py`: a shared raw cache (`--out/raw/<raw endpoint>/<sha1 of params>.json.gz`, one request per
page ever), a manifest per family with one record per unit (`ok`, `empty`, `failed`, rows, params, fetched_at,
error), `--retry-failed`, `--dry-run`, `--limit`, `--compact-only`, and compaction into one parquet per season
with the rows of the held-out games of `sportvu/splits.json` under `heldout/` (docs/DATA_CONTRACT.md section 9).

Rules shared by every driver:

- at most 17 requests a minute: `--rpm` is capped at 17 whatever is given, 0 or negative meaning the maximum
  (`bref_schedule.capped_throttle`; `bref_shots.main` does the same), and the throttle object is one per run,
  shared by every page of that run;
- the first HTTP 429 or 403 stops the run (`bref_shots.PageError(stop_run=True)` through `run_units`); a jail
  lasts up to a day, so the run is not resumed before the next day; 404 fails only that unit; 5xx and connection
  errors are retried three times with the shared backoff;
- the estimated request count is printed before the run: the pending units (after `--limit`) whose page is not in
  the raw cache (by file presence, no gunzip), and the minutes that means at 17 a minute;
- every fetch, play-by-play pages included, goes through `collect.bref_shots.fetch_html` (the same User-Agent,
  retries and error classes as the SportVU shot chart collector), so one monkeypatch covers every driver in the
  tests; play-by-play pages are parsed by `fetch_pbp.parse_events` (the parsing half of `fetch_pbp.fetch_game`);
- a 200 page without its table (or a shot chart without markers, a pbp page without events) fails the unit with
  `ValueError("<unit>: no table <id> on the page")` and the cached page is discarded, so the unit is never marked
  `empty` for good and `--retry-failed` fetches it again (`bref_schedule.fetch_and_parse`);
- an HTTP error from a play-by-play fetch is classified by the response's status code (`response.status_code`),
  never by a substring of the message (a game code such as `202304290DEN` contains `429`); the same holds in
  `bref_shots.pbp_shot_count` and `collect.pbp.br_events`, where a message without a response falls back to a
  digit-bounded match of `429`;
- `--season-types` accepts `Play-In` besides `Regular Season` and `Playoffs` (`bref_schedule.bref_parser`), and
  `--rpm` prints its own help (basketball-reference, maximum 17);
- no raw data is ever written into the repo: outputs go to `--out` (default `data/collect`, ignored by git;
  `$NBA_COLLECT_OUT`; the Drive folder on Colab);
- team codes are basketball-reference's (`BRK`, `CHO`, `PHO` for the SportVU `BKN`, `CHA`, `PHX`); the `game`
  column of per-game rows is the SportVU-style id `MM.DD.YYYY.AWAY.at.HOME` in SportVU abbreviations
  (`bref_schedule.sportvu_game_id`), so rows join the Step 1 tables by name; `game_id` is the basketball-reference
  code `YYYYMMDD0HOME` (no NBA.com game id is available from this source).

## Run order

1. `python -m collect.bref_schedule --seasons 2015-16 ... 2024-25` (the schedule feeds every per-game driver;
   about nine requests a season).
2. In any order, each resumable on its own:
   - `python -m collect.bref_seasons` (shots 2016-17 to 2020-21, play-by-play 2020-21 to 2024-25 by default; also
     copies the play-by-play tables to `stints/playbyplay/<season>.parquet` for the lineups lane);
   - `python -m collect.bref_boxscores --seasons ...` (also writes `stints/starters/<season>.parquet` for the
     lineups lane, so it goes before that lane's stint builder);
   - `python -m collect.bref_lineups --seasons ...`;
   - `python -m collect.positions --seasons ...`.
3. `python -m collect.drive --sync-to` to copy `data/collect` to the Drive folder (or `--sync-from` on another
   machine), so a run resumed elsewhere sees the same raw cache and manifests.

The 2015-16 SportVU games themselves stay with `collect/bref_shots.py` (shots) and `collect/pbp.py --br-only`
(play-by-play joined to the frames); the drivers here share their raw cache (`raw/bref_shotchart/` for shot
chart pages; `raw/bref_pbp/` for parsed play-by-play events, which bref_seasons reads and also writes next to
the page it keeps under `raw/bref_pbp_page/`; all keyed by `{"code"}`), so a page any of them fetched is never
requested again.

## Drivers, units, outputs

| driver | unit | request per unit | requests a season | output tables |
| --- | --- | --- | --- | --- |
| `collect/bref_schedule.py` | one month page (`leagues/NBA_<endyear>_games-<month>.html`) | 1 | 9 (October to June); 2019-20: 10 (`october-2019` ... `october-2020`, the bubble); 2020-21: 8 (December to July); 2011-12: 7 | `bref_schedule/games/<season>.parquet` |
| `collect/bref_seasons.py` | one game, endpoint `shotchart` (`boxscores/shot-chart/<code>.html`) | 1 | 1,230 regular season (2019-20: 1,059; 2020-21: 1,080) plus about 85 playoff games when `--season-types Playoffs` is given: about 72 min a season at 17 a minute, about 5.7 h for the five default seasons | `bref_seasons/shotchart/<season>.parquet` (+ `heldout/`) |
| `collect/bref_seasons.py` | one game, endpoint `playbyplay` (`boxscores/pbp/<code>.html`) | 1 | as above: about 6,000 for the five default seasons, about 5.9 h | `bref_seasons/playbyplay/<season>.parquet` (+ `heldout/`), copied byte for byte to `stints/playbyplay/<season>.parquet` and `stints/heldout/playbyplay/<season>.parquet` |
| `collect/bref_boxscores.py` | one game (`boxscores/<code>.html`) | 1 | as above, about 72 min a season | `bref_boxscores/starters/<season>.parquet`, `bref_boxscores/players/<season>.parquet` (+ `heldout/`); the opening lineups to `stints/starters/<season>.parquet` and `stints/heldout/starters/<season>.parquet` |
| `collect/bref_lineups.py` | one team-season (`teams/<TEAM>/<endyear>/lineups/`) | 1 | 30, about 2 min | `bref_lineups/lineups/<season>.parquet` |
| `collect/positions.py` | one season (`leagues/NBA_<endyear>_per_game.html`) | 1 | 1 | `positions/positions/<season>.parquet`, union in `--out/positions.parquet` |

Every unit's page is cached under `raw/`: `bref_schedule`, `bref_shotchart` (shared with `bref_shots`),
`bref_pbp_page` (the play-by-play page) plus `bref_pbp` (its parsed events, the record `bref_shots --fetch-pbp`
and `collect.pbp` write and read), `bref_boxscore`, `bref_lineups`, `bref_per_game`. Manifest keys are `<endpoint>|<season>|<unit id>` (`games|2015-16|december`,
`shotchart|2016-17|201610250CLE`, `playbyplay|...`, `starters|...|<code>`, `lineups|2015-16|GSW`,
`positions|2015-16|2015-16`).

### Schedule (`bref_schedule/games/<season>.parquet`)

| column | type | meaning |
| --- | --- | --- |
| `game_id` | string | basketball-reference code `YYYYMMDD0HOME` (from the Box Score link, else the row's `csk`, else built from date and home) |
| `date` | string | `YYYY-MM-DD` |
| `home`, `away` | string | basketball-reference team codes |
| `home_score`, `away_score` | Int32, nullable | null until the game is played |
| `season` | string | `YYYY-YY` |
| `season_type` | string | `Regular Season`, `Play-In` (a Notes cell saying "Play-In Game", 2020-21 on), `Playoffs` (rows after the "Playoffs" divider row) |
| `month` | string | the month page the row came from |
| `start_time`, `overtimes`, `attendance`, `arena`, `notes` | | as the page prints them (`attendance` float32) |
| `split` | string, nullable | the `sportvu/splits.json` split of a 2015-16 game (`train`, `heldout_game`, `heldout_arena`, `heldout_era`), null otherwise |
| provenance | | `source` `basketball-reference.com`, `endpoint` `leagues/NBA_<endyear>_games-<month>.html`, `fetched_at` |

The schedule is a listing, not fitting data, so it carries no `heldout` column and nothing is compacted apart;
`games_for_season(fam, season, season_types=("Regular Season",), played_only=True)` returns the game dicts
(`game_id, date, home, away, home_score, away_score, season, season_type, split, heldout, game`) with the held-out
tag computed on the way out. `teams_for_season` lists a season's team codes from its schedule (else the fixed 30).

### Shots (`bref_seasons/shotchart/<season>.parquet`)

The docs/DATA_CONTRACT.md section 10a table exactly (`bref_shots.COLUMNS`, `bref_shots.rows_for_game`): source
`bref`, court feet attacking the left basket, `loc_x`/`loc_y` as NBA.com, the calibration check of the pixel frame
per page (a page whose fitted hoop is more than 10 px from `HOOP_PX` fails the unit). `game_id` is filled with
the basketball-reference code (bref_shots leaves it null), `season_type` from the schedule.

### Play-by-play (`bref_seasons/playbyplay/<season>.parquet`)

The section 13 columns: the section 5 play-by-play columns (`game`, `event`, `period`, `clock_s`, `team`, `desc`,
`points`, `score`, `kind`, `frame` always -1, `frame_clock_diff_s` null, provenance with `source`
`basketball-reference.com` and `endpoint` `boxscores/pbp/<code>.html`, the NBA.com fields `event_id`,
`event_type`, `action_type`, `player1_id`, `player2_id`, `player3_id`, `team_id`, `clock_str` all null) plus
`game_id`, `split`, `heldout`, `season_type`, `game_date`. Events come from `fetch_pbp.parse_events` (the parser
every play-by-play path in the repo uses), so `kind` is its vocabulary (`shot_made`, `shot_missed`, `ft_made`,
`ft_missed`, `turnover`, `rebound`, `foul`, `other`; no `substitution` kind, basketball-reference prints
substitutions as `other` rows whose text starts with the player "enters the game for"). Source order per game:
the cached page (`raw/bref_pbp_page/`), the cached events of another collector (`raw/bref_pbp/`),
`data/pbp/<code>.json` (seeded into the cache), then the page under the throttle. After compaction the
per-season tables (fitting and held-out) are copied byte for byte to `stints/playbyplay/<season>.parquet` and
`stints/heldout/playbyplay/<season>.parquet`, which the lineups lane reads.

### Box scores (`bref_boxscores/starters/<season>.parquet` and `players/<season>.parquet`)

| column | type | meaning |
| --- | --- | --- |
| `game_id` | string | basketball-reference code |
| `team` | string | basketball-reference team code (from the table id `box-<TEAM>-game-basic`, or `box_<team>_basic` on older pages) |
| `player_slug`, `player` | string | the player link's slug (`data-append-csv`) and printed name |
| `starter` | bool | one of the first five rows of the team's basic table (before the Reserves divider) |
| `minutes` | float32 | `MM:SS` as decimal minutes |
| `home` | bool | the team is the schedule's home team |
| `season_type`, `game_date`, `split`, `heldout` | | from the schedule |
| provenance | | `endpoint` `boxscores/<code>.html` |

`starters` holds every player who played (minutes not null); `players` holds every row, with `mp_str`, `reason`
(Did Not Play, Did Not Dress, Inactive, ...) and the basic columns `fg, fga, fg_pct, fg3, fg3a, fg3_pct, ft, fta,
ft_pct, orb, drb, trb, ast, stl, blk, tov, pf, pts, plus_minus` (float32, blank cells null). A unit fails when the
page's teams are not the schedule's or a team has other than five starters.

After compaction the opening lineups are written to `stints/starters/<season>.parquet` (fitting games) and
`stints/heldout/starters/<season>.parquet` (held-out games), the table the lineups lane reads
(`bref_boxscores.stints_starters`, `STINTS_COLUMNS`): starter rows only, ten a game.

| column | type | meaning |
| --- | --- | --- |
| `game_id` | string | basketball-reference code |
| `game` | string | SportVU-style id `MM.DD.YYYY.AWAY.at.HOME` in SportVU abbreviations (from the game's home and away rows) |
| `game_date` | string | `YYYY-MM-DD` |
| `season` | string | `YYYY-YY` |
| `side` | string | `home` or `away` |
| `team` | string | basketball-reference team code |
| `team_id` | Int64, nullable | NBA.com team id: null from this source |
| `player_slug`, `player` | string | basketball-reference slug and printed name |
| `player_id` | Int64, nullable | NBA.com player id: null from this source |

The lineups lane keys on `side` plus `player_slug` (or a name key) when `team_id` and `player_id` are null.

### Lineups (`bref_lineups/lineups/<season>.parquet`)

| column | type | meaning |
| --- | --- | --- |
| `season`, `team` | string | |
| `rank` | Int32 | the page's Rk |
| `players` | string | the five slugs sorted and joined with `\|` |
| `player_names` | string | the printed names in the same order |
| `p1` .. `p5` | string | the sorted slugs |
| `n_players` | int8 | 5 on a well-formed row |
| `minutes` | float32 | MP as decimal minutes |
| `pace` | float32 | |
| `net_rating` | float32 | the PTS column of the "Per 100 Poss (Net)" block: team minus opponents per 100 possessions with the lineup on the floor (`data-stat` `diff_pts`; `net_rtg`, `pts_per_poss`, `pts` accepted as renames) |
| `stats_json` | string | every other numeric column of the row under its `data-stat` name (so a page change never changes the schema) |
| provenance | | `endpoint` `teams/<TEAM>/<endyear>/lineups/` |

### Positions (`positions/positions/<season>.parquet`, union `--out/positions.parquet`)

| column | type | meaning |
| --- | --- | --- |
| `player_slug`, `player` | string | from the player link (`data-stat` `player`, or `name_display` on the 2024 redesign) |
| `season` | string | |
| `position` | string | basketball-reference's listed position (`PG`, `SG`, `SF`, `PF`, `C`, or two joined with a hyphen) |
| `team` | string | basketball-reference team code (`team_id`, or `team_name_abbr` on the 2024 redesign); the total rows of traded players (`TOT`, `2TM`, ...) are dropped, so a traded player has one row per team |
| `age`, `games` | Int16 | `games` from `g`, or `games` on the 2024 redesign |
| `mp_per_g` | float32 | |
| provenance | | `endpoint` `leagues/NBA_<endyear>_per_game.html` |

`positions_by_player` keeps one row per (player, season), the team with the most games. The name join to the
tracking player ids goes through the rosters table (docs/DATA_CONTRACT.md section 16, `analytics/rosters.py`):
rosters carry `bref_slug` per SportVU `player_id` and game, matched by name once by `analytics.rosters.match_names`
(exact, lastname, fuzzy), so positions join rosters on `(bref_slug = player_slug, season)` and never by name
(`positions.join_rosters(positions, rosters)`; the season is derived from `sportvu_game` when the rosters frame
has no season column). A roster row with `bref_slug` null (`match_method` none) gets no position.

### Drive (`collect/drive.py`)

`drive_root()` resolves the synced folder: `$NBA_DRIVE_ROOT`, else
`~/Library/CloudStorage/GoogleDrive-*/My Drive/nba_harvest`, else the Colab mount. `python -m collect.drive
--sync-to [DATA_ROOT]` copies the local `collect/` root (`data/collect` or `--local`) to `DATA_ROOT/collect`;
`--sync-from` the other way; `--only raw bref_shots` limits the copy; `--dry-run` lists. rsync-like semantics in
Python: a file is copied when missing at the destination or when the sizes differ (same size: skipped, no content
comparison), nothing is ever deleted, `.part` files are skipped, each copy lands through a `.part` rename.

## Tests (`tests/test_collect_bref.py`, 12 tests, about 6 s, no network)

An autouse fixture replaces `bref_shots.fetch_html` and `requests.get` with functions that raise, so a forgotten
patch can never reach the network; each test then patches `fetch_html` with a fake that serves the fixture pages
by URL.

Fixture pages under `tests/fixtures/bref/pages/`, hand-built in the pages' real markup with a few rows each (the
`*.html` glob of `tests/test_bref_shots.py` does not reach the subdirectory): three schedule month pages (December
with the SportVU held-out game, the held-out arena and a training game; February with GSW at OKC; April with a
Play-In note, the Playoffs divider and an unplayed row), a box score (PHO at GSW: two basic tables, starters,
reserves, Did Not Play rows, a commented-out advanced table), a lineups page (the 5-man table inside an HTML
comment with a repeated header row, a 4-man table to ignore), a per-game page (a traded player's TOT and team
rows, a repeated header, a two-position player) and a play-by-play page in `fetch_pbp`'s layout. The shot charts
are the existing synthetic pages `tests/fixtures/bref/<code>.synthetic.html`.

- each parser on its fixture page, including the older box score table ids, the 2024 per-game column renames
  (`name_display`, `team_name_abbr`, `games`, `games_started`, `2TM`), a lineup cell without links, and the
  `ValueError` on a page without its table (parser level for all four, run level for the schedule, box scores and
  lineups: the unit fails, the page is not cached, `--retry-failed` fetches it again);
- the play-by-play error classification by status code (a URL carrying `202304290DEN` is not a 429; a real 429
  response is) in `bref_shots.pbp_shot_count` and `collect.pbp.br_events`;
- unit listing (`--dry-run`, `--limit`), the estimated request count, resume (a rerun requests nothing,
  `--retry-failed` asks again), the shared raw cache (bref_shots on a game bref_seasons fetched requests nothing),
  the compacted outputs and the held-out tagging of 2015-16 games through `colab_contract.split_of` from date and
  teams (held-out rows under `heldout/`, `train` and null splits in the fitting files);
- the 17 a minute cap (`--rpm 40` waits at 60/17 s, `--rpm 0` too, in bref_seasons and bref_shots) and the stop
  on the first 429 (no further request; the later units never enter the manifest), on a shot chart page and on a
  play-by-play page, and a 403;
- the stints copies (starters in the STINTS_COLUMNS layout, play-by-play byte for byte), the positions union and
  the rosters join, `Play-In` as a season type, the Drive sync semantics (skipped counts over the `--only` tops).

## Assumptions

- The fixture pages were written by hand from the pages' markup as known here (no page could be fetched in this
  environment); the parsers key on `data-stat` names and table ids and accept the known renames
  (`box_<team>_basic`, `team_name_abbr`, `2TM`), and the lineup net rating falls back through four names. The first
  real run should check one unit of each driver by eye.
- Month lists per season are fixed (`MONTHS_BY_SEASON`): a missing month page is a failed unit (404), not an error.
- Play-In games are tagged from the Notes column; if a page prints them otherwise they land as Regular Season.
- The box score driver takes the first five rows of each basic table as the starters, as basketball-reference
  lists them; a page that lists other than five fails the unit rather than guessing.
- `bref_seasons` uses its own family directory (`bref_seasons/`) so its manifest keys (basketball-reference codes)
  never collide with `bref_shots` (SportVU ids); the 2015-16 shots table the xFG baseline reads stays
  `bref_shots/shotchart/2015-16.parquet`.
- Request estimates count one page per unit; a 5xx retry costs extra requests under the same throttle, and a
  `data/pbp/<code>.json` file spares a request the estimate still counts (it is not a cached page).
