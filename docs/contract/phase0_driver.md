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
    [--fixtures]
```

| option | meaning |
| --- | --- |
| `--data` | DATA_ROOT: the Step 1 folder (`manifest.json`, `<game>.parquet`, `<game>_pbp.parquet`, `heldout/...`) |
| `--out` | OUT_ROOT, default DATA_ROOT: tables under `OUT_ROOT/analytics/<table>/`, reports under `OUT_ROOT/reports/` |
| `--games` | only these SportVU game ids; an id not in the manifest is listed as skipped |
| `--limit` | at most N eligible games, in manifest order (training games first, then by date, as Step 1 wrote it) |
| `--steps` | comma list; `gaps` runs once over the manifest, the other four per game |
| `--retry-failed` | rerun the steps `phase0_manifest.json` marks failed (otherwise they stay failed and the run reports RUN FAILED) |
| `--dry-run` | list eligible games, input availability and the steps that would run or be reused; write nothing |
| `--workers` | games in parallel through `multiprocessing.Pool` when more than 1. Default 1: Drive FUSE on a Mac serialises and sometimes corrupts concurrent writers, so keep 1 when OUT_ROOT is on Google Drive; with more than 1 the module reports (`localisation_*.md`) may interleave lines |
| `--fixtures` | without `--data`, write a synthetic DATA_ROOT (`tests/_fx_phase0.py`) to a temp folder and run on it; also copy the summary to the repo's `reports/phase0_driver.md` (fixture output only, never real data) |

Eligible games: manifest `status` `ok` and `pbp_joined` true. Every other game is listed under `skipped`
with a reason (`status failed: ...`, `status source_broken: ...`, `no play-by-play (pbp_joined false: ...)`,
`not in the manifest`).

Inputs are located with `sportvu.colab_contract.game_paths` (the split routing), falling back to the other
folder and then to the manifest's own `parquet` and `pbp_parquet` paths.

The last line printed is `RUN OK` (no eligible game has a failed step and the gaps report wrote) or
`RUN FAILED`; the exit code is 0 or 1 accordingly.

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
   "status": "ok" | "failed",      failed when any step failed
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
| rosters | `n_players`, `pbp_total`, `pbp_matched`, `methods` {exact, lastname, fuzzy, none}, `unmatched` (first ten names) |
| shot_frames | `shot_frames.validate`: `n_shots`, `n_release_found`, `share_release_found`, `n_identity_known`, `identity_agreement`, `share_high`; plus `n_identity_agree`, `n_high`, `n_low`, `n_unmatched` counted by the driver |
| rebound_frames | `rebound_frames.validate`: `n_misses`, `n_rebound_found`, `share_rebound_found`, `n_identity_known`, `identity_agreement`, `share_high`; plus the same four driver counts |
| possessions | `possessions.summary`: `n`, `by_end_kind`, `points_total`, `n_gap`, `mean_duration_s`, `share_offense_known`, `n_dropped` |

Resume rule (`run_phase0.pending_steps`): a step is reused when its last record is `ok` and the file at
`path` exists and its parquet footer reads; a step whose record is `failed` is reused as failed unless
`--retry-failed`; anything else (no record, missing or torn output) runs again. A game whose manifest
entry is no longer eligible (for example `pbp_joined` turned false) moves from `games` to `skipped`.

## `phase0_summary.md`

Sections, in order: Games (manifest size, processed, ok, failed, held-out among processed, skipped);
Per-step totals (games ok, games failed, rows, seconds per step); rosters (players, play-by-play names
matched / total, `share_pbp_matched`, `match_method` counts); shot_frames (`n_shots`, `n_release_found`,
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
python -m analytics.audit_shots --data OUT_ROOT --n 200 --seed 0 --out OUT_ROOT/analytics/audit/shots_sample.parquet [--all]
python -m analytics.audit_shots --summarise OUT_ROOT/analytics/audit/shots_sample.parquet [--report PATH] [--tolerance 2]
```

Draw: the shot_frames tables of the training games only (`OUT_ROOT/analytics/shot_frames/<season>/*.parquet`,
never `heldout/`, and a game `sportvu/splits.json` holds out is refused even if a file sits in the training
folder); `n // games` shots per game, drawn without replacement with `numpy.random.default_rng(seed)` from the
shots whose `release_frame` was found (`--all` includes the unfound rows); a game with fewer candidates
contributes what it has and is listed under `games_short`. Rows are sorted by game and event.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int32 | play-by-play event index (section 17) |
| `period` | int8 | period of the shot |
| `clock_s` | float32 | logged clock of the shot (the outcome) |
| `release_frame`, `apex_frame`, `outcome_frame` | int32 | the localiser's frames, `-1` when not found |
| `shooter_id` | int32, nullable | roster id of the shooter |
| `confidence_flag` | string | `high`, `low`, `unmatched` |
| `audited_release_frame` | int32, nullable | empty; the auditor's release frame from the court replay |
| `audited_ok` | bool, nullable | empty; the auditor's verdict on the row |
| `auditor_note` | string, nullable | empty; free text |

`--summarise` reads a filled sample back and reports, over the rows with `audited_release_frame` set: the
share within the tolerance (2 frames by default) of `release_frame`, the mean absolute difference, the share
of `audited_ok`, and the agreement per `confidence_flag` and per game; `--report` writes the same as markdown.
