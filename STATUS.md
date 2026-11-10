# STATUS.md: extension build, 2026-11-05 (unattended)

## Runbook progress

WAITING ON YOU (stage 3 shot audit): audit the 604 shots in the label tool's audit mode, then
`mkdir -p gates && touch gates/audit-done`. Start the tool with:

```bash
python3 -m analytics.label_tool --data "$DATA_ROOT" --audit "$DATA_ROOT/analytics/audit/shots_sample.parquet"
```

and open http://127.0.0.1:8765/. Keys are in docs/labeling_guide.md section 7 (a accept, m mark the release frame
then enter to save, x bad). Verdicts are written into the sample after every shot and the tool resumes at the first
unaudited shot, so it can be done over several sittings. Only the 334 high-confidence shots are graded against the
pass mark; the 222 low and 48 unmatched are reported, not graded. If time is short, audit the high ones first: the
page shows each shot's confidence, and n skips a shot without saving.

Stage 1 passed: PR #5 and PR #6 merged 2026-11-07. Stage 2 passed 2026-11-07. Two later decisions are open
(odds and NBA.com, under Open gates).

Open PRs from you, not touched by the loop (merge when ready): #8 stage 4a --skip-heldout,
#9 stage 9 results, #10 stage 6 torch models. The loop does not redo stage 4a while #8 is open.

Updated 2026-11-09 by the autonomous loop (docs/mac_runbook.md).

- **Current stage:** 3, audit sample drawn 2026-11-09; waiting for gates/audit-done.
- **Background jobs:** none.
- **Play-by-play gaps at the source:** eight 2021-10-27 games (home teams BOS, BRK, LAC, MIL, NOP, OKC, POR, TOR)
  fail with "no play-by-play events on the page", also on the --retry-failed rerun. The fetched page for
  202110270BOS has the pbp table with a caption and no rows, so basketball-reference has no events for them; the
  same date's ORL and PHO games parsed normally. Their box scores collected normally.

- **Open gates (not blocking yet):** odds for betting (put ODDS_API_KEY in ~/.zshrc and create gates/odds-ready,
  or create gates/odds-skip) and NBA.com tables (gates/nbacom-done or gates/nbacom-skip). Neither blocks anything
  before stage 5 phase 5 (betting) and the embeddings aggregates; the loop carries on meanwhile.
- **Branch:** the Mac checkout is on track-fix.

### Stage 3: Phase 0 on every tracked game (done 2026-11-07)

Commands as in the runbook, with `--bref-dir "$DATA_ROOT/collect"` (corrected, see below). Logs:
logs/loop/phase0.run1.log, phase0.log, phase0_rosters.log. Summary on Drive: DATA_ROOT/reports/phase0_summary.md.
Its totals include the 17 held-out games; the figures below are recomputed from phase0_manifest.json over the 604
training games only (rule 1).

| metric | value |
| --- | --- |
| games processed / failed / skipped | 621 / 0 / 5 (the 5 are source_broken in Step 1) |
| training games | 604 |
| shots | 101803 |
| share of shots with a release frame | 0.6220 |
| shooter identity agreement (shots with both ids) | 0.9538 |
| high-confidence shots | 34528 (0.3392 of shots) |
| misses / share with a rebound frame | 59762 / 0.8331 |
| rebounder identity agreement | 0.6902 |
| points check: possession points equal the final score | 537 of 604 |
| rosters with basketball-reference names (all 621 games) | 621 games; 12599 players with bref_slug; shot chart names matched 0.9950 |

- Points check: all 67 training mismatches undercount (possession points below the final score): 36 by 2, 11 by
  1, 8 by 3, 5 by 4, the rest 5 to 10. Likely points inside the 138 possessions dropped as shorter than a frame or
  across clock gaps; worth a look before the possession-based lanes in stage 5.
- Rebounder identity agreement (0.69) is low next to the shooter's (0.95); the rebound localiser's identity is
  weaker evidence than its frame.
- Fixed on the way (tests in each commit): transient Drive read errors retried (f10db36); a nameless player_id 0
  crashed name matching (f10db36); the runbook's `--bref-dir data/collect/bref_shots` held no pages, so rosters had
  no bref_slug until rebuilt with `--rebuild rosters --bref-dir "$DATA_ROOT/collect"` (06cfc7a).
- reports/sportvu_gaps.md: 626 games, 369 with a tracking gap over 10 s.

### Stage 3: audit sample (drawn 2026-11-09)

Pass mark approved (gates/audit-mark-approved, 2026-11-09 21:12) with the sample line unchanged, and PR #7 merged
first. `python3 -m analytics.audit_shots --data "$DATA_ROOT" --n 200 --seed 0` wrote
DATA_ROOT/analytics/audit/shots_sample.parquet (log: logs/loop/audit_sample.log).

| check | value |
| --- | --- |
| shots / games | 604 / 604 training games, one shot each |
| confidence_flag high / low / unmatched | 334 / 222 / 48 |
| held-out SportVU games in the sample (splits.json heldout_game, heldout_arena, heldout_era) | 0 |
| games at CLE | 0 (21 CLE road games, which are training games) |
| audit tests (tests/test_label_tool_audit.py and the audit_shots tests) | 51 passed |
| audit tool smoke run on the real sample | page and shots 1, 301 and 604 served; sample file unchanged |

### Stage 3: audit pass mark (approved 2026-11-09)

- **Pass mark:** among audited high-confidence shots (confidence_flag high), pass_mark.share, bad rows counted as misses:
  the detected release frame is within 2 frames of the audited release frame on at least 90 percent. This is
  `pass_mark.share` of `audit_shots --summarise` (tolerance 2 frames, analytics/audit_shots.py TOLERANCE_FRAMES). The low
  row is reported, not graded.
- **If it fails:** the four localisation constants (apex search 2 s, rim radius 6 ft, flight 25 frames,
  high-confidence 0.6) are not pinned as validated; the loop writes what failed here and stops stage 3.
- **Sample:** `python3 -m analytics.audit_shots --data "$DATA_ROOT" --n 200 --seed 0`, training games only, shots
  with a release frame. Note: the sampler takes n // games shots per game with at least 1, so with 604 training
  games `--n 200` draws one shot per game, **604 shots**, about three times the runbook's 3 hours (roughly half
  will be high confidence). To audit about 200, change this line to "200 games, one shot each" before creating the
  gate file and the loop adds a game cap to the sampler (with a test) before drawing.

### Stage 2 check (passed 2026-11-07)

Units per collector and season, read from each manifest under DATA_ROOT/collect. No collector had an empty unit;
the only failures are the eight play-by-play games above, rerun once with --retry-failed.

| season | schedule pages | positions | play-by-play ok / failed | box scores ok | schedule games |
| --- | --- | --- | --- | --- | --- |
| 2015-16 | 9 | 1 | | | 1316 |
| 2016-17 | | 1 | | | |
| 2020-21 | 8 | | 1165 / 0 | 1165 | 1171 |
| 2021-22 | 9 | | 1309 / 8 | 1317 | 1323 |
| 2022-23 | 9 | | 1314 / 0 | 1314 | 1320 |
| 2023-24 | 9 | | 1313 / 0 | 1313 | 1319 |
| 2024-25 | 9 | | 1315 / 0 | 1315 | 1321 |

- Play-by-play and box scores cover six fewer games a season than the schedule: the six play-in games, which those
  collectors do not request (default season types).
- Files present: collect/stints/playbyplay/<season>.parquet for 2020-21 to 2024-25 (526883, 593321, 599722,
  592078 and 601833 events); collect/stints/starters/<season>.parquet for the same seasons (10 starters a game:
  11650, 13170, 13140, 13130, 13150 rows); collect/positions.parquet (1070 rows, 2015-16 528 and 2016-17 542).
- Box scores also wrote the full player tables (players/<season>: 31394, 33649, 33834, 34677, 34940 rows).
- Logs: logs/loop/bref_schedule.log, positions.log, bref_seasons.run1.log, bref_seasons.log,
  bref_seasons.retry.log, bref_boxscores.log (gitignored, on the Mac).
- Optional, not collected: odds (paid key) and NBA.com tables. Both are open gates (above).

### Stage 0 (passed 2026-11-07)

DATA_ROOT is exported in ~/.zshrc and points at the Drive nba_harvest folder, which holds
sportvu_contract/manifest.json and collect/. The first run failed three ways; each was fixed with a test (PR #5):

| check | first run | after PR #5 |
| --- | --- | --- |
| make test-fast | could not run: no pytest, then no bs4, then test_makefile_targets_match_registry failed | 612 passed, 1 skipped |
| make all-fixtures | 46 of 47 steps OK; betting train refused --data without --out | 47 of 47 OK |

- requirements.txt now lists pytest, beautifulsoup4, pandas, scipy, threadpoolctl and requests. Running the
  stage 0 install command put pytest 9.2.0 and beautifulsoup4 4.15.0 into /usr/local/bin/python3. Use that
  python3, not /opt/anaconda3: anaconda's site-packages holds a stray `tests` package that shadows the repo's.
- The Makefile no longer takes the exported DATA_ROOT as its fixture root. Before the fix, `make fixtures` would
  have written synthetic data into the real Drive folder.

Goal: the code for every remaining part of the extension plan, on top of Phase 0 and the xFG location
baseline. Integration branch `analytics-phase0`, based on `track-fix`. Twenty-one lanes ran in parallel in
one shared working copy (worktrees did not fit the disk allowance): fifteen extensions and six
infrastructure lanes. Every lane was reviewed by a fresh cold reviewer, fixed, and re-checked; three
repo-wide cold reviews (leakage and held-out rules, interface consistency, code quality and tests) ran at
the end. The last round of fixes ran on Opus after the Fable usage limit was reached. Nothing here is a result: every number in every `reports/<ext>.md` is fixture output from the
synthetic generator, labelled as such in the file.

## Rules held

- Held-out games in `sportvu/splits.json` never fit or tune anything. Every trainer calls
  `analytics.eval.refuse_heldout` on its training game ids and `analytics.schema.assert_time_forward` on its
  feature frame; `tests/test_splits.py` lists every `analytics/*/train.py` and fails on any analytics file
  that fits without the guard. Folds follow one convention: >= 0 out-of-fold, -1 held-out, -2 in-sample. Held-out rows are set aside before any fit and scored once by the selected
  candidate only.
- No raw SportVU, play-by-play, NBA.com or odds data is committed. Fixture pages under
  `tests/fixtures/bref/pages/` are small hand-built excerpts. Nothing scrapes: ingestion tests mock every
  fetch and an autouse guard raises on any unpatched network call.
- Baselines first. A deep model stands in only where the plan names one (ghosting sequence model, EPV
  sequence model, embeddings autoencoder), as an sklearn MLP because torch is absent here, gated by a
  baseline comparison fixed in advance. On fixtures every gate kept the baseline.
- Fixture numbers are never presented as results. Reports say so on every numeric section.

## Assumptions taken without asking

- Shared working copy instead of worktrees; lanes owned disjoint files and ran no git; the lead committed
  by path after each lane's tests passed, with WIP snapshot commits in between (the stop hook required a
  clean tree).
- The twenty subagent cap meant the integration tester started after the first lane finished.
- Cross-lane path decisions, taken when producers and consumers disagreed: EPV writes
  `OUT_ROOT/analytics/epv/<season>/<game>.parquet` (states) and `epv_possessions.parquet`; actions writes
  `OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet`; coverage and playcalling read those;
  held-out rows of every producer sit under `heldout/<season>/`; hand labels are read only through
  `analytics.labels.read_labels` with `labeller_pass == 1` and `label_source == "hand"`; the starters table
  from basketball-reference box scores carries `side` and `player_slug` with null NBA ids, and the lineups
  builder keys on those.
- Gravity ships the ridge expectation and a zone-only shooting prior; the boosting candidate is a
  diagnostic (its reviewer showed a player-identifying prior lets boosting absorb the quantity measured).
- Betting and win probability share one closing-line rule (`analytics.betting.validate.closing_lines`
  then `join_games`): per-book latest pre-tip `is_close` snapshot, median over books, an 8 hour shift and a
  one-day tolerance on the date join.
- `analytics/possessions.py` now turns the season over in July, like the other Phase 0 modules (it said
  September).
- The HistGradientBoosting fits are pinned to one thread (threadpoolctl) because OpenMP spin-waits on the
  shared four-core box made fits hang; `--threads` raises it where offered.

## Lanes

| lane | files | tests | review | needs real data |
| --- | --- | --- | --- | --- |
| I1 fixtures | analytics/fixtures.py, docs/contract/fixtures.md | 24 | PASS, mediums fixed (league-stable ids, aggregates and stints share rosters, localiser confidence) | nothing |
| I2 ingestion | collect/bref_schedule.py, bref_seasons.py, bref_boxscores.py, bref_lineups.py, positions.py, drive.py; docs/contract/ingestion.md | 12 (plus 28 existing collect tests) | FAIL then fixed (starters table contract, status-code error classes, 17/min cap, tableless pages fail not empty, network guard) | basketball-reference pages: schedule 9 pages a season; shot charts, play-by-play and box scores 1 page a game (about 72 min a season each at 17 a minute); lineups 30 a season; positions 1 a season |
| I3 Phase 0 driver | analytics/run_phase0.py, audit_shots.py; docs/contract/phase0_driver.md; reports/phase0_driver.md | 19 | PASS, mediums fixed (subset-run resume, module paths, refusal test, --bref-dir fills bref_slug) | the Step 1 manifest and frames on Drive |
| I4 labeling tool | analytics/labels.py, label_tool.py, label_tool.html; docs/contract/labels.md; docs/labeling_guide.md | 29 | FAIL then PASS on re-review (tombstones, pass filter, visited record, held-out guard, server hardening) | about 27 hours of hand labelling on the Mac |
| I5 run_all | run_all.py, Makefile, analytics/registry.py, docs/run_order.md, pytest.ini | 18 | FAIL then fixed (--out and --seed to every step, quoted paths, Phase 0 layout, coverage edge, log capture) | nothing |
| I6 integration | analytics/interfaces.py, tests/test_integration.py, docs/contract/interfaces.md, reports/integration.md | 111 (fast part; slow part runs every lane on fixtures) | its loop found: rebounds features CLI, embeddings --folds, EPV output path, betting NaN RAPM join; all fixed | nothing |
| xfg | analytics/xfg/ | 13 (plus 7 baseline) | PASS, mediums fixed (selected candidate only on held-out, clock join tested and one-to-many, no event_id branch) | shot charts, shot frames for every tracked game, frames |
| rebounds | analytics/rebounds/ | 12 | PASS, mediums fixed (symmetric miss selection, frame-window tests) | rebound frames, shot frames, frames, rosters |
| matchups | analytics/matchups/ | 15 | PASS, mediums fixed (basket vote per team and period, one-game-per-defender notice, join tests, heldout column from xfg) | frames, possessions, rosters, shot frames, xfg output |
| gravity | analytics/gravity/ | 11 | FAIL then fixed (ridge shipped, zone-only prior, planted-pull recovery test, fallback section in the report) | frames, possessions, shot frames, rosters, shot charts |
| ghosting | analytics/ghosting/ | 9 | PASS, mediums fixed (output under analytics/ghosting, frame-stride cut) | frames, possessions; torch for the GRU that replaces the MLP stand-in |
| embeddings | analytics/embeddings/ | 14 | PASS, mediums fixed (positions file joined through rosters and collapsed to G/F/C, mirrored and gap tests) | frames, possessions, rosters, shot charts, NBA.com aggregates (blocked), positions table |
| epv | analytics/epv/ (states, validate, features, train, eda) | 19 | PASS, mediums fixed (MLP paired against the base candidate, in-sample MSE shown) | frames, play-by-play, possessions, rosters |
| actions | analytics/actions/ | 16 | FAIL then fixed (training restricted to visited possessions, one-to-one label matching, detections cast through analytics.labels) | hand labels under DATA_ROOT/labels, frames, possessions |
| coverage | analytics/coverage/ | 13 | PASS, mediums fixed (selected model only on held-out, by-construction note, unsure rows set aside, strict label reader, per-season team rates) | hand screen labels, frames, rosters, possessions, actions output |
| decisions | analytics/decisions/ | 19 | PASS, mediums fixed (basket per team and period, partial xfg coverage, held-out xfg path, mean quality statistic, drive in the option set) | possessions, shot frames, frames, epv and xfg outputs |
| playcalling | analytics/playcalling/ | 16 | FAIL then fixed (read_labels, action ids never overwritten, score-margin control, recovery test on a planted association) | possessions, labels, actions, coverage and epv outputs |
| lineups | analytics/lineups/ (stints, validate, features, train, eda) | 24 | FAIL then fixed (team ids never players, starters join through the games table, team_id from the output season, player_rapm_prior) | multi-season play-by-play, starters and games tables, bref lineups |
| winprob | analytics/winprob/ | 15 | FAIL then fixed (spread join delegated to betting, finals dropped before features, overtime tests, held-out season out of EDA) | multi-season play-by-play, games, odds |
| load | analytics/load/ | 12 | PASS, mediums fixed (smoothing never bridges a dropout, hand-built history test) | frames, manifest, rosters; aggregates optional |
| betting | analytics/betting/ | 19 | PASS, mediums fixed (per-book closing snapshot, time-forward test, team-name inference with votes, partial RAPM join imputed) | odds history (paid feed), games table, lineups output |

Every lane also has `notebooks/eda_<ext>.ipynb` (hand-written nbformat 4, tables only, no stored outputs),
`docs/contract/<ext>.md` and `reports/<ext>.md`. Each `validate`, `features` and `train` module exposes
`main(argv)` with `--fixtures | --data DATA_ROOT [--out OUT_ROOT] [--report ...] [--folds] [--seed]
[--bootstrap]`; `run_all.py` and the Makefile drive them in dependency order.

## Where it stands

`python3 -m pytest -q tests -m "not slow"` passes: 609 passed, 2 skipped (one needs the SportVU archives,
one a local tracking data file), in about 3 minutes. `python3 -m pytest -q tests -m slow` runs every lane's
validate, features and train on fixtures, then on a shared fixture DATA_ROOT, then through `run_all.py`,
checks the five interface tables, and scans every lane output for the held-out game outside `heldout/`.
On the final tree it passes: 63 passed in about 6 minutes. The per-lane detail is in
`reports/integration.md` (fixture output, not a result).

Open items, none blocking the Mac run: the odds feed and the NBA.com tables remain uncollected
(stats.nba.com drops non-browser clients, odds need an API key); the GRU, the EPV sequence model and the
embeddings autoencoder stand in as sklearn MLPs until torch is installed; `analytics.fixtures` keeps its
star on the floor all game, so the gravity on/off sign test runs on the lane's planted-pull generator; the
four shot-localisation constants (apex search 2 s, rim radius 6 ft, flight 25 frames, high-confidence 0.6)
are still to pin in the contract; the lineups lane keeps its own synthetic stints generator because the
shared one does not carry enough lineup signal for its recovery test.

## Final reviews

Three repo-wide cold reviews ran after every lane passed its own review. All three returned PASS with no
blocker; every medium they raised was fixed and the fixes were re-tested.

- **Leakage and held-out rules.** No fit, imputer, calibration, alpha choice, selection, gate or EDA fit
  touches a held-out game in any lane; every deep model is gated by a baseline comparison fixed in advance.
  Fixed: seven lanes printed held-out scores for non-selected candidates (now the selected candidate only,
  plus a labelled constant floor); in-sample rows wrote fold 0 (now -2, and consumers refuse anything but
  fold >= 0 on training rows); the xFG baseline's smoke run had scored the real held-out game (report
  regenerated without it, the smoke mode no longer scores held-out rows); the held-out arena rule is now
  part of `sportvu.splits.heldout_tokens`, so a lane that forgets the `.at.CLE` suffix still trips the guard;
  later seasons than the held-out one are dropped in win probability and betting; the integration test now
  scans every lane output for the real held-out id.
- **Interface consistency.** A full `run_all.py --fixtures --all` run finished 47 of 47 steps and all five
  interface files passed their checks. Fixed: EPV and decisions read the Phase 0 tables and the 1 Hz states
  from OUT_ROOT first, then DATA_ROOT; the registry and `docs/run_order.md` now name the collector tables
  the lanes actually read; decisions skips the xFG feature file; consumer lists in the interface registry.
- **Code quality and tests.** Hygiene held: no raw data, no network in tests, no em or en dashes. Fixed:
  helpers every lane had copied now live once in `analytics/eval.py` (held-out by id, thread pinning, fast
  calibration slope, interval and table formatting) and `analytics/schema.py` (season of a game, July rule);
  production code no longer imports `tests/` (synthetic generators a lane needs moved into the lane);
  dormant fallback generators deleted from `tests/_fx_*`; import-time thread environment writes removed;
  every fixture CLI accepts `--fixture-games`, `--fixture-possessions`, `--fixture-heldout-every`; report
  sections carrying numbers are each labelled fixture output, with no scratch paths; the integration test's
  tautology, environment gates and skip-on-missing-output were replaced by assertions; `run_all.py`
  re-raises Ctrl-C; the wall-clock test is marked slow; the slow EPV tests now take seconds.

## Next on the Mac (where the full data is)

`DATA_ROOT` is the parent of `sportvu_contract/` and `collect/` (the Drive `nba_harvest` folder). Quote it:
Drive paths have spaces. Full detail and time estimates are in `docs/run_order.md`.

1. Check the machine on fixtures: `make test-fast` then `make all-fixtures`.
2. Collectors, one at a time (17 requests a minute): `python -m collect.bref_schedule --seasons 2015-16 --out "$DATA_ROOT/collect"`, then `collect.bref_boxscores`, `collect.bref_lineups`, `collect.positions`, `collect.bref_seasons` (later seasons for lineups, winprob, betting), and `python -m collect.pbp --contract-dir "$DATA_ROOT/sportvu_contract" --br-only` for games whose play-by-play join failed.
3. Phase 0 over the whole manifest: `make phase0 DATA="$DATA_ROOT"` (runs `analytics.gaps` then `analytics.run_phase0 --data "$DATA_ROOT/sportvu_contract" --out "$DATA_ROOT" --steps rosters,shot_frames,rebound_frames,possessions`; add `--bref-dir "$DATA_ROOT/collect"` (the folder holding `raw/bref_shotchart/`) to fill `bref_slug`; `--dry-run` first; reruns resume).
4. Hand labels: `python -m analytics.label_tool --data "$DATA_ROOT" --game 11.14.2015.BKN.at.GSW --sample 100 --seed 0 --port 8765`, open `http://127.0.0.1:8765/`; relabel with `--labeller-pass 2`; check with `--kappa`.
5. Every extension in dependency order: `make all DATA="$DATA_ROOT"` (or one at a time, `make ext-xfg DATA="$DATA_ROOT"` and so on; `OUT=/path` moves outputs, logs and reports). Reports land in `reports/<ext>.md`, logs in `logs/run_all/`.
6. Optional: `ODDS_API_KEY=... python -m collect.extras --odds --seasons ...` before `make ext-betting`; NBA.com tables (`collect.shots`, `collect.tracking`, `collect.context`, `collect.stints`) from a network stats.nba.com accepts.
