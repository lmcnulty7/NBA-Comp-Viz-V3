# Phase 0 driver contract (`analytics/run_phase0.py`, `analytics/audit_shots.py`)

The driver runs the Phase 0 localisers of docs/DATA_CONTRACT.md sections 16 to 19 over every game of a
Step 1 manifest (`sportvu/colab_contract.py`). It calls the module functions (`rosters.build`,
`shot_frames.run`, `rebound_frames.run`, `possessions.run`, `gaps.write_report`), never a subprocess, and
leaves output routing to the modules: a game that `sportvu/splits.json` holds out is written under
`heldout/` by each module itself. Every localiser is deterministic with thresholds fixed in the contract, so
a held-out game is read and written but nothing is fit or tuned on it.

## Command

```
python -m analytics.run_phase0 --data DATA_ROOT [--out OUT_ROOT] [--games G ...] [--limit N]
    [--steps rosters,shot_frames,rebound_frames,possessions,gaps] [--retry-failed] [--dry-run] [--workers 1]
    [--bref-dir DIR] [--fixtures]
```

| option | meaning |
| --- | --- |
| `--data` | DATA_ROOT: the Step 1 folder (`manifest.json`, `<game>.parquet`, `<game>_pbp.parquet`, `heldout/...`) |
| `--out` | OUT_ROOT, default DATA_ROOT: tables under `OUT_ROOT/analytics/<table>/`, reports under `OUT_ROOT/reports/` |
| `--games` | only these SportVU game ids; an id not in the manifest is listed as skipped |
| `--limit` | at most N eligible games, in manifest order (training games first, then by date, as Step 1 wrote it) |
| `--steps` | comma list; `gaps` runs once over the manifest, the other four per game |
| `--retry-failed` | rerun the steps `phase0_manifest.json` marks failed (otherwise they stay failed and the run reports RUN FAILED) |
| `--rebuild` | comma list of per-game steps to rerun even when their output reads back (for example `rosters` after a run whose `--bref-dir` held no pages); other steps are reused as usual |
| `--dry-run` | list eligible games, input availability and the steps that would run or be reused; write nothing |
| `--bref-dir` | cached basketball-reference shot chart pages: `<DIR>/<code>.html` as `collect.bref_shots --save-pages` writes them, or the collector's raw cache `<DIR>/raw/bref_shotchart/<sha1 key>.json.gz` (so DIR is the collector's `--out`: `DATA_ROOT/collect` on Drive or `data/collect` on the Mac, not the `bref_shots/` table folder under it; the driver logs a WARNING when DIR holds no page for the first 5 games). When a page exists for a game, `rosters.build` fills `bref_slug` and `bref_name`; without it, or for a game without a page, those two columns are null, the step record says `bref_page: false`, and the summary counts the games without one. The embeddings and EPV features resolve shooter ids through `rosters.bref_slug`, so pass it on the real run |
| `--workers` | games in parallel through `multiprocessing.Pool` when more than 1. Default 1: Drive FUSE on a Mac serialises and sometimes corrupts concurrent writers, so keep 1 when OUT_ROOT is on Google Drive; with more than 1 the module reports (`localisation_*.md`) may interleave lines |
| `--fixtures` | only without `--data` (with `--data` the flag does nothing): write a synthetic DATA_ROOT (`tests/_fx_phase0.py`: `analytics.fixtures.write_fixture_root` when present, which puts the Step 1 layout under `<tmp>/sportvu_contract`, else the local fallback) to a temp folder and run on it; also copy the summary to the repo's `reports/phase0_driver.md` (fixture output only, never real data) |

Eligible games: manifest `status` `ok`, `pbp_joined` true, and the manifest's `heldout` flag agreeing with
`sportvu/splits.json` (`run_phase0.routed_heldout`, the same `split_of` the modules route by). Every other game
is listed under `skipped` with a reason (`status failed: ...`, `status source_broken: ...`,
`no play-by-play (pbp_joined false: ...)`, `not in the manifest`, `manifest says heldout X but
sportvu/splits.json routes it to the ... folder`). A skipped id that is neither in the Step 1 manifest nor skipped
by the current run (a typo in an earlier `--games`) is dropped from `skipped` on the next run. The last one exists because a game the manifest holds out
must never land in a training folder: Step 1 computes the flag with the same function, so a disagreement
means the manifest or the splits file changed and somebody has to look.

Inputs are located with `sportvu.colab_contract.game_paths` (the split routing), falling back to the other
folder and then to the manifest's own `parquet` and `pbp_parquet` paths.

The last line printed is `RUN OK` (no eligible game has a failed step and the gaps report wrote) or
`RUN FAILED`; the exit code is 0 or 1 accordingly. A missing manifest or data folder also ends with
`RUN FAILED` (exit 1) after one line naming the error. Anything that goes wrong outside a step (a malformed game
id, a manifest entry without a home, an unreadable page) marks every requested step of that game failed with the
error and the run goes on.

## Outputs

| path | content |
| --- | --- |
| `OUT_ROOT/analytics/<table>/<season>/<game>.parquet` | the section 16 to 19 tables of a training game |
| `OUT_ROOT/analytics/<table>/heldout/<season>/<game>.parquet` | the same for a held-out game |
| `OUT_ROOT/analytics/phase0_manifest.json` | the run state, below |
| `OUT_ROOT/reports/phase0_summary.md` | the summary, below |
| `OUT_ROOT/reports/sportvu_gaps.md` | `analytics.gaps.write_report` over the Step 1 manifest |
| `OUT_ROOT/reports/localisation_shots.md`, `localisation_rebounds.md` | the per-game lines the two modules append |
| `reports/phase0_driver.md` (repo) | the summary of a `--fixtures` run |

`<season>` is `2015-16` for a game dated July 2015 to June 2016 (`run_phase0.season_of_game`, the rule the
localisers use).

## `phase0_manifest.json`

Rewritten atomically after every game, so a partial run is readable and a rerun resumes.

```
{
 "version": 1,
 "data_root": "...", "out_root": "...", "steps": ["rosters", "shot_frames", "rebound_frames", "possessions", "gaps"],
 "updated": "YYYY-MM-DD HH:MM:SS",
 "counts": {"games": P, "ok": n, "failed": n, "skipped": n},
 "skipped": {"<game>": "<reason>", ...},
 "gaps": <analytics.gaps.summary(manifest)> or null,     "gaps_error": "..." only when the gaps step failed
 "games": {
  "<game>": {
   "game", "date", "away", "home", "split", "heldout", "season",
   "frames": "<frames parquet read>", "pbp": "<pbp parquet read>",
   "final_score": {"away": a, "home": h, "total": a + h} or null   (the last non-null "away-home" score of the events)
   "points_check": {"possession_points": n, "final_total": n, "ok": bool} or null
   "status": "ok" | "failed" | "pending",   failed when any recorded step (this run or an earlier one) failed;
                                            pending only while the game is being run (a run that dies leaves it, and
                                            the next run reports it failed with "the run ended before this game finished")
   "seconds": s,
   "steps": {
    "<step>": {
     "status": "ok" | "failed",
     "path": "analytics/<table>/[heldout/]<season>/<game>.parquet" relative to OUT_ROOT, or null,
     "rows": n or null,
     "validate": {...} or null,     the module's numbers, see below
     "schema_problems": [...] or null,   analytics.schema.validate of the written table (first five)
     "seconds": s, "written": "YYYY-MM-DD HH:MM:SS",
     "error": null or "ExcType: message", "traceback": "..." only when failed,
     "reused": true   only on a record carried over from an earlier run
    }
   }
  }
 }
}
```

`validate` per step:

| step | keys |
| --- | --- |
| rosters | `n_players`, `pbp_total`, `pbp_matched`, `bref_page` (a shot chart page was found under `--bref-dir`), `bref_total`, `bref_matched`, `bref_slug_filled` (players with a slug), `methods` {exact, lastname, fuzzy, none}, `unmatched` (first ten names) |
| shot_frames | `shot_frames.validate`: `n_shots`, `n_release_found`, `share_release_found`, `n_identity_known`, `identity_agreement`, `share_high`; plus `n_identity_agree`, `n_high`, `n_low`, `n_unmatched` counted by the driver |
| rebound_frames | `rebound_frames.validate`: `n_misses`, `n_rebound_found`, `share_rebound_found`, `n_identity_known`, `identity_agreement`, `share_high`; plus the same four driver counts |
| possessions | `possessions.summary`: `n`, `by_end_kind`, `points_total`, `n_gap`, `mean_duration_s`, `share_offense_known`, `n_dropped` |

Resume rule (`run_phase0.pending_steps`): a step is reused when its last record is `ok` and the file at
`path` exists and its parquet footer reads; a step whose record is `failed` is reused as failed unless
`--retry-failed`; anything else (no record, missing or torn output) runs again. Records of steps outside the
current `--steps` are kept untouched, so a `--steps shot_frames` run followed by a full run only builds the other
three tables, and a failed record from an earlier run still fails the game. Each step's `path` is the one the
module itself reported (`shot_frames.out_path`, `rebound_frames.run`'s `out`, `possessions.run`'s `path`,
`rosters.out_path_for`), checked against `schema.table_path`; a module writing anywhere else fails the step. A game whose manifest
entry is no longer eligible (for example `pbp_joined` turned false) moves from `games` to `skipped`.

## `phase0_summary.md`

Sections, in order: Games (manifest size, processed, ok, failed, held-out among processed, skipped);
Per-step totals (games ok, games failed, rows, seconds per step); rosters (players, play-by-play names
matched / total, `share_pbp_matched`, shot chart names matched / total, `share_bref_matched`, players with
`bref_slug` filled, games with and without a shot chart page: without one `bref_slug` and `bref_name` are null); shot_frames (`n_shots`, `n_release_found`,
`share_release_found`, `n_identity_known`, `n_identity_agree`, `identity_agreement` over all shots with both
ids, `n_high`, `share_high`); rebound_frames (the same over misses); possessions (count, `points_total`,
possessions with a clock gap, dropped, by `end_kind`, the points-sum check: games whose events carry a
final score, games where the possession points equal it, the mismatching games); Clock gaps
(`analytics.gaps.summary` of the Step 1 manifest, overall and by split); Skipped games (game, reason);
Per game (game, split, the four step statuses, `n_shots`, `share_release_found`, `n_misses`,
`share_rebound_found`, possessions, `points / final`, seconds); Failed steps (game, step, error) when any;
then the verdict line. Shares in the totals are ratios of the summed counts, not means of per-game shares.

## Audit sample (`analytics/audit_shots.py`)

```
python -m analytics.audit_shots --data OUT_ROOT --n 200 --seed 0 --out OUT_ROOT/analytics/audit/shots_sample.parquet [--all] [--force]
python -m analytics.audit_shots --summarise OUT_ROOT/analytics/audit/shots_sample.parquet [--report PATH] [--tolerance 2]
```

Draw: the shot_frames tables of the training games only (`OUT_ROOT/analytics/shot_frames/<season>/*.parquet`,
never `heldout/`, and a game `sportvu/splits.json` holds out is refused even if a file sits in the training
folder); `n // games` shots per game (at least 1, so an `n` below the number of training games gives one shot
per game and more than `n` rows), drawn without replacement with `numpy.random.default_rng(seed)` from the
shots whose `release_frame` was found (`--all` includes the unfound rows); a game with fewer candidates
contributes what it has and is listed under `games_short`. Rows are sorted by game and event. The sample is
the only home of the audit verdicts, so a draw over an existing sample that holds any filled audit column
(or that cannot be read) is refused with exit code 2 unless `--force`.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int32 | play-by-play event index (section 17) |
| `period` | int8 | period of the shot |
| `clock_s` | float32 | logged clock of the shot (the outcome) |
| `release_frame`, `apex_frame`, `outcome_frame` | int32 | the localiser's frames, `-1` when not found |
| `shooter_id` | int32, nullable | roster id of the shooter |
| `confidence_flag` | string | `high`, `low`, `unmatched` |
| `audited_release_frame` | int32, nullable | empty; the auditor's release frame from the court replay (set exactly when `audited_ok` is true) |
| `audited_ok` | bool, nullable | empty; true for a real shot whose release the auditor placed, false for a bad row (below) |
| `auditor_note` | string, nullable | empty; free text |

`--summarise` reads a filled sample back and reports, over the rows with `audited_release_frame` set: the
share within the tolerance (10 frames by default, `TOLERANCE_FRAMES`, about 0.4 s) of `release_frame`, the mean absolute difference, the share
of `audited_ok`, and the agreement per `confidence_flag` and per game, each group with its count of bad rows
(`audited_ok` false); `--report` writes the same as markdown.

**Pass-mark statistic.** `pass_mark.share` (printed as `pass-mark statistic`, the first table of the
report) is the number the stage 3 pass mark is compared with: over every audited row with
`confidence_flag == "high"` (a frame or a verdict filled), the share whose `audited_release_frame` is within
the tolerance of `release_frame` and is not marked bad. A high-confidence row marked bad counts as a miss in
the denominator, so sending hard shots to bad cannot raise it. `pass_mark` also holds `rows`,
`within_tolerance` and `bad`. The older `agreement_within_tolerance` and the per-flag `share` leave bad rows
out and stay for reference only.

**Tighter tolerances, information only.** The tolerance is 10 frames because an auditor reading the court
animation confirms a release to about 10 frames, not 2. `information_only` (the report's "Tighter
tolerances" table, the CLI's `information only, not pass/fail` lines) repeats `pass_mark` share and count
and `agreement_within_tolerance` at each of `INFO_TOLERANCES` (2 and 5 frames), skipping the graded
tolerance. These numbers are not compared with any pass mark. `--tolerance N` changes the graded tolerance
(older tests pass 2).

### Filling the sample: the label tool's audit mode

```
python -m analytics.label_tool --data DATA_ROOT --audit OUT_ROOT/analytics/audit/shots_sample.parquet [--port 8765] [--no-backup]
```

walks the sample row by row (`analytics.label_tool.AuditSession`, keys in `docs/labeling_guide.md`
section 7). Each verdict sets the three audit columns of one row, exactly one of:

| verdict (key) | `audited_release_frame` | `audited_ok` | meaning |
| --- | --- | --- | --- |
| accept (`a`) | `release_frame` | `True` | a real shot and the detected frame is its release |
| release (`m`, then `enter`) | the marked frame | `True` | a real shot released at the marked frame (the localiser may be off by any amount) |
| bad (`x`) | `<NA>` | `False` | no release to compare: not a shot at this time, the ball or the shooter cannot be followed, or the frames are broken |

So `audited_release_frame = <NA>` with `audited_ok = False` is a bad row, and a frame is set exactly when
`audited_ok = True`. In `--summarise` terms: a bad high-confidence row is a miss in the pass-mark statistic,
counts against `share_audited_ok` and in the `bad` counts, and stays out of the older agreement numbers; a
release moved by 3 frames or more is outside the 2-frame mark. `auditor_note` is optional
free text (whitespace collapsed, at most 500 characters, empty means `<NA>`).

A row is audited when either of the two columns is filled; the tool resumes at the first row with both
empty, and a later verdict on a row replaces the earlier one. A blank `release_frame`, `apex_frame` or
`outcome_frame` (a CSV round trip turns `-1` into a blank) is read as `-1`, not found. The served window is the shot's period,
100 frames either side of `release_frame` (a row without one, from `--all`, gets `clock_s - 2.5` to
`clock_s + 4.0`), and an audited frame must be a frame of the game in the shot's period.

Writes: after every verdict the whole sample is written to a temporary file next to it, fsynced and moved
over the sample with `os.replace`, so a crash loses at most the verdict being written. Row order, every
other column and the arrow schema (with its pandas metadata) are kept; audit columns of another type (a CSV
round trip) are written back as int32, bool and string. Before the first write of each session the sample
is copied to `<sample>.<UTC time>.bak`, a new file per session that never replaces an earlier backup
(`--no-backup` skips it). The session records a SHA-256 of the sample's bytes when it reads it and after each
of its own writes; a sample whose bytes differ at the next write is not overwritten (409; restart the tool).
The hash, not the modification time, is compared, so a sync client touching the time is not a change. A
failed write is a 500 and leaves the sample as it was. Nothing else is written: the server has no label
endpoints. A file that is not a sample, or a missing one, stops the tool with a one-line error (exit 2).

Held-out: the audit decides whether the localisation constants are pinned as validated, which is a
selection, so held-out games stay out exactly as in the draw. The tool refuses a sample holding any game
that `analytics.eval.is_heldout_game` holds out (every held-out token of `sportvu/splits.json`, the
held-out arena included); without `sportvu/splits.json` it does not start.
