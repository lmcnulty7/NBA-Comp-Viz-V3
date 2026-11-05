# Run order: from collection to every extension, on a Mac

Everything below runs from the repo root with `python3` (pandas, pyarrow, numpy, scipy, scikit-learn).
`DATA_ROOT` is the parent folder of `sportvu_contract/` and `collect/`; on the Mac it is the Google Drive
`nba_harvest` folder. `DATA_ROOT/sportvu_contract/` is the Step 1 contract (docs/DATA_CONTRACT.md section 3:
the per-game frames and play-by-play parquets, `heldout/`, `manifest.json`), the collector tables live under
`DATA_ROOT/collect/` and the Phase 0 and extension tables under `OUT_ROOT/analytics/` (OUT_ROOT defaults to
DATA_ROOT). `analytics.run_phase0` alone takes `--data` as the contract folder itself; `run_all.py --phase0`
finds it under DATA_ROOT and accepts a DATA_ROOT that is the contract folder. Quote the path: Drive paths
have spaces. Times are rough: network steps assume basketball-reference's limit of
17 requests a minute (every bref collector throttles itself and backs off on the first 429, so never run
two at once); compute steps assume a laptop CPU on one 2015-16 season (about 1,230 games, 8 tracked
GSW games in the contract).

`make help` lists the shortcuts; `python run_all.py --list` prints the registry (analytics/registry.py)
with which packages are present; `python run_all.py --collectors` prints the collector commands.
Every extension step runs on fixtures without any data (`make ext-xfg`), which is how to check a machine
before the long steps.

## The order

| # | step | command | inputs | outputs | approximate time | what it needs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | tests on fixtures | `make test-fast` | nothing | pass or fail | 2 to 5 minutes | python3 with the packages above; ffmpeg for the video tests |
| 2 | Drive sync | `python -m collect.drive --sync-from "DATA_ROOT" [--only bref_shots ...] [--dry-run]` (`--where` prints the resolved Drive root; `--sync-to` copies the other way) | `DATA_ROOT/collect/` on Drive (default: the Drive nba_harvest folder) | the local `data/collect/` | minutes (local copy) | Google Drive mounted; no network calls. The Step 1 contract stays at `DATA_ROOT/sportvu_contract/` and is read in place |
| 3 | schedule | `python -m collect.bref_schedule --seasons 2015-16 --out DATA_ROOT/collect` | network | `collect/bref_schedule/<season>.parquet` | 1 minute a season (one page a month) | basketball-reference reachable, 17 a minute |
| 4 | season tables | `python -m collect.bref_seasons --seasons 2015-16 --out DATA_ROOT/collect` | network | `collect/bref_seasons/<season>.parquet` | 2 to 5 minutes a season | same |
| 5 | positions | `python -m collect.positions --seasons 2015-16 --out DATA_ROOT/collect` | network | `collect/positions/<season>.parquet` | about 2 minutes a season | same |
| 6 | box scores | `python -m collect.bref_boxscores --seasons 2015-16 --out DATA_ROOT/collect` | schedule table, network | `collect/bref_boxscores/<season>.parquet` | about 75 minutes a season (1,230 pages at 17 a minute) | same; resumable, rerun to continue |
| 7 | lineups | `python -m collect.bref_lineups --seasons 2015-16 --out DATA_ROOT/collect` | network | `collect/bref_lineups/<season>.parquet` | about 2 minutes a season (one page a team) | same |
| 8 | shot charts | `python -m collect.bref_shots --contract-dir DATA_ROOT --fetch-pbp` | manifest, network | `collect/bref_shots/shotchart/<season>.parquet` | about 1 minute per 17 games; a full season about 75 minutes | same |
| 9 | play-by-play fill | `python -m collect.pbp --contract-dir DATA_ROOT --br-only` | manifest, network | `<game>_pbp.parquet` for games whose join failed | 1 minute per 17 games | same; only the NBA.com path is blocked |
| 10 | NBA.com tables (optional) | `python -m collect.shots`, `collect.tracking`, `collect.context`, `collect.stints` | network | `collect/shots/`, `collect/tracking/`, `collect/context/`, `collect/stints/` | hours; 30 requests a minute | a network that stats.nba.com accepts (a browser session); not reachable from the cloud |
| 11 | extras (optional) | `ODDS_API_KEY=... python -m collect.extras --odds`; `python -m collect.extras --injuries FILE.csv` | API key or downloaded CSV | `collect/extras/odds/`, `collect/extras/injuries/` | minutes | an odds API key; an injury CSV |
| 12 | clock gaps | `python -m analytics.gaps --manifest "DATA_ROOT/sportvu_contract/manifest.json" --out reports/sportvu_gaps.md` | manifest | `reports/sportvu_gaps.md` | seconds | nothing else |
| 13 | Phase 0 tables | `python -m analytics.run_phase0 --data "DATA_ROOT/sportvu_contract" --out "OUT_ROOT" --steps rosters,shot_frames,rebound_frames,possessions` (or `make phase0 DATA="DATA_ROOT"`, which runs 12 then 13 with exactly these arguments) | frames, pbp, manifest, bref shot charts | `OUT_ROOT/analytics/rosters/`, `shot_frames/`, `rebound_frames/`, `possessions/` per game | about 1 minute a tracked game | steps 2 and 8 |
| 14 | xFG baseline (Phase 1) | `python -m analytics.train_xfg --shots DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet --report reports/xfg_baseline.md` | shot charts | `reports/xfg_baseline.md` | about 1 minute with 200 bootstrap draws | step 8 |
| 15 | xfg | `make ext-xfg DATA=DATA_ROOT` | shot charts, shot frames | `reports/xfg.md`, features under OUT_ROOT | 1 to 3 minutes | steps 8, 13 |
| 16 | rebounds | `make ext-rebounds DATA=DATA_ROOT` | rebound frames, shot frames, frames, rosters | `reports/rebounds.md` | 1 to 3 minutes | step 13 |
| 17 | matchups | `make ext-matchups DATA=DATA_ROOT` | frames, possessions, rosters, shot frames, xfg output | `reports/matchups.md` | 2 to 5 minutes (8 tracked games) | steps 13, 15 |
| 18 | gravity | `make ext-gravity DATA=DATA_ROOT` | frames, possessions, shot frames, rosters | `reports/gravity.md` | 2 to 5 minutes | step 13 |
| 19 | ghosting | `make ext-ghosting DATA=DATA_ROOT` | frames, possessions, rosters | `reports/ghosting.md` | 5 to 15 minutes (sequence model on a CPU) | step 13 |
| 20 | embeddings | `make ext-embeddings DATA=DATA_ROOT` | frames, possessions, rosters, positions | `reports/embeddings.md` | 5 to 15 minutes | steps 5, 13 |
| 21 | epv (validate, states, features, train) | `make ext-epv DATA=DATA_ROOT` | frames, pbp, possessions, rosters | `analytics/states_1hz/`, `reports/epv.md` | 5 to 15 minutes | step 13 |
| 22 | actions | `make ext-actions DATA=DATA_ROOT` | frames, pbp, possessions, rosters | `reports/actions.md` | 2 to 5 minutes | step 13 |
| 23 | coverage | `make ext-coverage DATA=DATA_ROOT` | frames, possessions, rosters, actions output | `reports/coverage.md` | 2 to 5 minutes | steps 13, 22 |
| 24 | decisions | `make ext-decisions DATA=DATA_ROOT` | states, shot frames, pbp, xfg and epv outputs | `reports/decisions.md` | 1 to 3 minutes | steps 15, 21 |
| 25 | playcalling | `make ext-playcalling DATA=DATA_ROOT` | states, possessions, pbp, epv, actions and coverage outputs (coverage_predictions) | `reports/playcalling.md` | 2 to 5 minutes | steps 21, 22, 23 |
| 26 | lineups (validate, stints, features, train) | `make ext-lineups DATA=DATA_ROOT` | stints or multi-season pbp, bref lineups, schedule | `reports/lineups.md` | 2 to 10 minutes a season | steps 3, 7 (10 when the NBA.com stints are wanted) |
| 27 | winprob | `make ext-winprob DATA=DATA_ROOT` | pbp, schedule, box scores | `reports/winprob.md` | 1 to 5 minutes a season | steps 3, 6, 9 |
| 28 | load | `make ext-load DATA=DATA_ROOT` | tracking aggregates, box scores, schedule, injuries | `reports/load.md` | 1 to 3 minutes | steps 3, 6 (10, 11 optional) |
| 29 | betting | `make ext-betting DATA=DATA_ROOT` | odds, schedule, box scores, lineups output | `reports/betting.md` | 1 to 3 minutes | steps 11, 26 |
| 30 | everything at once | `make all DATA=DATA_ROOT` (fixtures: `make all-fixtures`) | all of the above | `reports/run_all.md`, `logs/run_all/<ext>_<step>.log` | sum of 15 to 29, about 1 to 2 hours | steps 2 to 13 |
| 31 | site | `make site` | `portfolio/hiring_manager_page.html` | `site/index.html` | seconds | nothing |

Add `OUT=/path` to any make line to put step outputs, logs and reports under that root instead of the
repo. Every step receives `--data DATA_ROOT` (or `--fixtures`), `--seed N` and `--out OUT_ROOT` when set;
train also gets `--report` and `--bootstrap` when given. `run_all.py` stops at the first failed step unless
`--continue-on-error` (`make all` passes it), skips packages that are not importable and the dependents of
a failed extension with a printed reason, and ends with `RUN OK` or `RUN FAILED`. Its exit code is nonzero
when any step failed. A fixtures run without `OUT` writes its reports to `reports/fixtures/` so the tracked
`reports/<ext>.md` are never overwritten by fixture output; `logs/`, `reports/fixtures/`, `.fixture_root/`
and `site/` are gitignored.

## Dependencies

Extensions run in registry order with these edges honoured (`analytics.registry.order_for`):

- matchups depends on xfg
- decisions depends on xfg and epv
- playcalling depends on epv, actions and coverage (it reads coverage_predictions)
- coverage depends on actions
- betting depends on lineups

Full order with every extension selected: xfg, rebounds, matchups, gravity, ghosting, embeddings, epv,
actions, coverage, decisions, playcalling, lineups, winprob, load, betting. `run_all.py --ext matchups`
runs matchups alone and prints a note that xfg was not selected; `--with-deps` pulls the dependencies in.
`make collect-bref` only prints the collector commands; run them one at a time by hand.

Phases (registry `phase`): 0 gaps and run_phase0; 1 xfg, rebounds; 2 matchups, gravity, ghosting,
embeddings; 3 epv, actions, coverage; 4 decisions, playcalling, lineups; 5 winprob, load, betting.
`python run_all.py --data DATA_ROOT --phase 2` runs one phase.

## Reachability

Reachable from any network: the basketball-reference collectors (bref_schedule, bref_seasons,
bref_boxscores, bref_lineups, positions, bref_shots, pbp with `--br-only`) and the local Drive sync.
Not reachable from the cloud or many home networks: shots, tracking, context, stints and probe, which
call stats.nba.com; it refuses non-browser clients. The extensions that list those tables (load, lineups
when the NBA.com stints are wanted, betting for odds) run on their basketball-reference substitutes or
on fixtures until the tables exist.
