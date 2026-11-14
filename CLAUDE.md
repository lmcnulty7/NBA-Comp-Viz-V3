# CLAUDE.md

Rules for every Claude session and subagent in this repository. PLAN.md is locked and is the source of truth for
what gets built; STATUS.md's closing record is background on the frozen group 1 analytics. This file says how to
work here without breaking either.

## What this is

An analytics product on player tracking, with a broadcast-video tracking pipeline as a measured stand-in for vendor
tracking. The current job (PLAN.md, M0 to M11) adds a Python export layer (`export/`) and a static site (`web/`) on
top of frozen models. Python produces data, the web only reads it, through one JSON contract: `export/schemas.py`.

## Hard rules

1. Never push, publish or deploy. `git push` is denied in `.claude/settings.json`. All work stays on `v3-finish`
   or a teammate branch that the lead merges. Merging to `main`, tagging a release, changing the repository's
   settings and setting up any host belong to Lucien.
2. Only the lead (the main session) edits PLAN.md. It is locked: changes are new D lines, milestone status lines
   and the morning summary. Never reopen a human-made D line. Never start a Parked item.
3. Frozen code is not touched. `git diff --stat freeze-v1 HEAD -- analytics sportvu` must show only
   `analytics/results.py` (2 lines, the stage 9 parser). Add nothing under `analytics/`: a test requires exactly
   the 15 lanes there. New Python goes under `export/`, `tools/` and `tests/`.
4. Held-out data is never fitted, tuned or scored again. The 17 held-out games are the files under `heldout/`
   directories (16 games at CLE plus 12.16.2015.PHX.at.GSW); `sportvu/splits.json` is the source. The possession
   review shows held-out games only. A model prediction shown for a player comes from out-of-fold rows only
   (fold >= 0), never from in-sample rows (fold -2) or held-out rows (fold -1). Descriptive player tables (the
   gravity table, embedding neighbours, observed rates) come from training games only and are labelled
   descriptive. No new held-out metric is computed anywhere.
5. No fit call under `export/` or `web/`: nothing there fits, tunes, calibrates, selects or imputes. The one place
   that reruns frozen fitting code is `tools/` (D45, D49), on training games only, and its output is used only
   when `tools.frozen_common.require_passed` says it reproduces the frozen record.
6. Never run a lane command line with its defaults. Every `analytics.<lane>` and `run_all.py` command writes to
   DATA_ROOT and to `reports/<lane>.md` unless told otherwise, and would overwrite the only copy of the frozen
   outputs and reports. The same holds for `sportvu.bench` and the tracked scorecards. If frozen code must run,
   pass an explicit `--out` and `--report` under a scratch folder outside DATA_ROOT and outside `reports/`.
7. One heavy job at a time. The Mac has 16 GB and has crashed before. Take the machine-wide lock for anything
   that reads DATA_ROOT at scale, runs model code, runs the whole test suite, builds the site or drives a browser:
   `with export.heavy.heavy_job("name"):` in Python, or `python3 -m export.heavy --name "name" -- <command>`.
   Read tracking frames one game at a time and release them before the next.
8. Site numbers come only from `export/`. Every number on the site traces to a file under `reports/` or to a
   table named in `web/data/manifest.json`. No number is typed into web copy, slides or data by hand.
9. Nothing private in a committed file or a data file: no absolute local path, no email address, no cloud drive
   account name. `export.schemas.find_private` is the check. Never glob `reports/`: many old files there hold
   local paths. Read named files only.
10. Never delete anything outside git-tracked worktrees and branches. Never delete the `story-page` branch (it
    holds unpushed work found nowhere else).
11. Do not launch a desktop application or a browser: no `open`, no `osascript`, no Chrome, no Safari. Browser
    tests run only through `npm run check`, with Playwright's own bundled headless builds.
12. No em dashes and no en dashes in anything you write: code, comments, copy, docs, commit messages.

## Environment

- Python is `/usr/local/bin/python3` (3.13). Never `/opt/anaconda3` and never bare `pytest`: the anaconda
  site-packages shadow the repository's `tests/`. Do not install `textstat` or `ultralytics` into the system
  Python for the same reason.
- `DATA_ROOT` is set in the shell and points at the Drive mount with the real data. Quote it: the path has spaces.
  It is the parent of `sportvu_contract/` and `analytics/`.
- This folder is inside iCloud Documents sync. Do not create `node_modules`, build output, generated site data or
  worktrees inside it. They live under `~/Developer/nba-v3-worktrees` (D44). The main checkout is the source of
  truth for git and holds the local broadcast outputs under `data/sportvu` (ignored, absent from worktrees); a
  worktree reads them through the `V3_CV_ROOT` environment variable.
- Node 23 and npm. Web commands run inside `web/` of a worktree outside iCloud.

## Commands

- Fast tests: `/usr/local/bin/python3 -m pytest -q tests -m "not slow"` from the repository root. Baseline on
  2026-11-13 before any change: 866 passed, 1 skipped, 63 deselected.
- Contract: `python3 -m export.schemas --out web/lib/types/contract.schema.json`, `--check <file>`,
  `--validate <data root>`. Synthetic data for web work: `python3 -m export.sample --out web/data`.
- Site data: `python3 -m export.site_data` (one command, end to end, under the heavy-job lock).
- Web: `npm ci`, `npm run check`, `npm run build:public`, `npm run build:private` inside `web/`.

## Layout

- `export/schemas.py`: the contract. `export/site_data.py`: reads reports and DATA_ROOT tables, writes `web/data`.
  `export/demo_bridge.py`: broadcast windows to contract frames, frozen predict only. `export/heavy.py`: the lock.
  `export/sample.py`: synthetic data that validates against the contract, never a result.
- `tools/frozen_models.py`: rebuilds frozen model objects and two outputs that were never stored (D45).
- `web/`: Next.js static export in TypeScript. Pages read data only through `lib/data/source.ts`.
  `web/data`, `web/out`, `web/media-private` and `node_modules` are ignored.
- `reports/`: generated lane reports, `reports/results.md`, `reports/scorecard/`. `analytics/`, `sportvu/`:
  frozen at tag `freeze-v1`.

## Milestones (D38)

Each milestone of PLAN.md closes in this order: `/code-review` on its changes, then the ml-reviewer
(`.claude/agents/ml-reviewer.md`), then the verifier (`.claude/agents/verifier.md`), then the lead writes a status
line in PLAN.md and commits on `v3-finish`. Reviewers and the verifier are read-only; the verifier may not edit a
test or a check, and nobody weakens a check to make it pass.

## Team (D39)

The lead owns PLAN.md, merges, decisions and milestone closes. Teammates work in their own worktree, on the paths
they were given, commit on their own branch, and report the branch name and what they verified. A teammate never
edits PLAN.md, never merges, never pushes, never touches frozen code, and never runs a heavy job outside the lock.
When something the task needs is missing or a rule blocks it, say so in the report instead of working around it.

## Words

- Ghosting is an average defense, never the best or optimal one.
- The broadcast demo is an illustration (D16): analytics results are SportVU-sourced and say so.
- Decisions and matchups are shown as tested with no signal. Failed and inconclusive models stay visible.
- A value shows its interval or says it is a point estimate. Fixture output is never presented as a result.
- Site copy: plain words, short sentences, reading grade 9 or lower.
