# Mac runbook: from the reviewed code to final results

This is the order to finish the project on the Mac with Claude Code. Each stage has the commands to run
and a prompt to paste into Claude Code. Stages run in order; a stage that ends in a check does not hand
over to the next until the check passes. Background is in STATUS.md, `docs/run_order.md` and the
contract pages under `docs/contract/`.

## Rules for every session

Claude Code on the Mac should follow these in every stage. Each prompt below says "Follow the rules in
docs/mac_runbook.md", which points here.

1. Held-out games (sportvu/splits.json, every game at CLE, and the 2024-25 season for the multi-season
   lanes) never fit, tune or select anything. Until stage 8, nobody reads a held-out number: every real
   train run uses `--skip-heldout` (added in stage 4).
2. Never commit raw data: no SportVU frames, play-by-play, NBA.com tables, odds, labels or parquet outputs.
   Commit code, contract pages and reports only.
3. One basketball-reference collector at a time (17 requests a minute). Stop on a 429 and wait an hour.
4. Reports carry real numbers only after stage 6. Until then they say "fixture output, not a result".
5. A failing check stops the stage. Write what failed into STATUS.md instead of working around it.
6. No em dashes in anything written.

## Stage 0: set up (15 minutes)

In Terminal:

```bash
cd ~/NBA-Comp-Viz-V3 && git fetch origin && git checkout analytics-phase0 && git pull
```

```bash
python3 -m pip install -r requirements.txt
```

```bash
export DATA_ROOT="$HOME/Library/CloudStorage/GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest"
```

Put that last line in `~/.zshrc` so every terminal has it. `python3 -m collect.drive --where` prints the
Drive root it finds, to check the path. Then start Claude Code in the repo and paste:

```text
Read STATUS.md and docs/mac_runbook.md. Follow the rules in docs/mac_runbook.md. Stage 0: confirm DATA_ROOT
points at the Drive nba_harvest folder (it must hold sportvu_contract/manifest.json and collect/), run
`make test-fast`, then `make all-fixtures`, and report pass or fail with the first error lines of anything
that failed. Change nothing.
```

## Stage 1: merge the branch (10 minutes)

```text
Follow the rules in docs/mac_runbook.md. Stage 1: open a pull request from analytics-phase0 into track-fix
with a summary drawn from STATUS.md. Do not merge it; I will.
```

Merge it on GitHub, then `git checkout track-fix && git pull`. Every later stage works on track-fix.

## Stage 2: collect the remaining tables (about 13 hours of collector time, overnight)

Already done in October: the 2015-16 shot charts and the Step 1 play-by-play fill. What is left, one
command at a time, each resumable (rerun the same command to continue; `--dry-run` shows what is
pending). `caffeinate -i` keeps the Mac awake.

```bash
caffeinate -i python3 -m collect.bref_schedule --seasons 2015-16 2020-21 2021-22 2022-23 2023-24 2024-25 --out "$DATA_ROOT/collect"
```

```bash
caffeinate -i python3 -m collect.positions --seasons 2015-16 2016-17 --out "$DATA_ROOT/collect"
```

```bash
caffeinate -i python3 -m collect.bref_seasons --pbp-seasons 2020-21 2021-22 2022-23 2023-24 2024-25 --shots-seasons --out "$DATA_ROOT/collect"
```

```bash
caffeinate -i python3 -m collect.bref_boxscores --seasons 2020-21 2021-22 2022-23 2023-24 2024-25 --out "$DATA_ROOT/collect"
```

Rough times: schedule and positions a few minutes; play-by-play and box scores about 6 hours each.
Optional: odds for betting need a paid key from the-odds-api.com
(`ODDS_API_KEY=... python3 -m collect.extras --odds --seasons 2020-21 2021-22 2022-23 2023-24 2024-25`);
without it the betting lane stays code only. NBA.com tables are optional and only work from a network
stats.nba.com accepts.

When they finish, paste:

```text
Follow the rules in docs/mac_runbook.md. Stage 2 check: for each collector (bref_schedule, positions,
bref_seasons, bref_boxscores) read its manifest under $DATA_ROOT/collect and report units ok, empty and
failed per season. Rerun failed units once with --retry-failed, one collector at a time. Confirm these exist:
collect/stints/playbyplay/<season>.parquet and collect/stints/starters/<season>.parquet for 2020-21 to
2024-25, collect/positions.parquet. Write the counts into STATUS.md.
```

## Stage 3: Phase 0 on every tracked game, then the shot audit (about 10 hours, then 3 hours by hand)

```bash
caffeinate -i python3 -m analytics.gaps --manifest "$DATA_ROOT/sportvu_contract/manifest.json" --out reports/sportvu_gaps.md
```

```bash
caffeinate -i python3 -m analytics.run_phase0 --data "$DATA_ROOT/sportvu_contract" --out "$DATA_ROOT" --steps rosters,shot_frames,rebound_frames,possessions --bref-dir data/collect/bref_shots --workers 1
```

Add `--dry-run` first to see the plan. If `data/collect/bref_shots` is not on this Mac, use
`"$DATA_ROOT/collect/bref_shots"`. Reruns resume; `--retry-failed` redoes failed games.

```text
Follow the rules in docs/mac_runbook.md. Stage 3: read $DATA_ROOT/reports/phase0_summary.md and report games
processed, failed and skipped, share of shots with a release frame, identity agreement, rebound share and the
points check. Before drawing the audit sample, write into STATUS.md the pass mark we will hold the audit to
(I propose: release frame within 2 frames on at least 90 percent of high-confidence shots). Then run
`python3 -m analytics.audit_shots --data "$DATA_ROOT" --n 200 --seed 0`, export the sample to CSV for me, and
tell me how to check each shot in the label tool (python3 -m analytics.label_tool --data "$DATA_ROOT"
--game <game>, scrub to the frame).
```

Fill in `audited_release_frame`, `audited_ok` and `auditor_note` for each shot, then:

```text
Follow the rules in docs/mac_runbook.md. Stage 3 check: convert my filled CSV back to the sample parquet, run
python3 -m analytics.audit_shots --summarise <sample> --report reports/shot_audit.md, and compare with the pass
mark in STATUS.md. If it passes, pin the four localisation constants (apex search 2 s, rim radius 6 ft,
flight 25 frames, high-confidence 0.6) in docs/DATA_CONTRACT.md section 17 as validated. If it fails, stop and
write what failed into STATUS.md. Commit reports and docs only.
```

## Stage 4: hold-out switch and hand labels (about 1 hour of Claude work, then about 27 hours by hand)

```text
Follow the rules in docs/mac_runbook.md. Stage 4a: add a --skip-heldout flag to every analytics/*/train.py
and to analytics/train_xfg.py. With the flag, held-out rows are set aside exactly as now but never scored,
and the report says "held-out games not scored (--skip-heldout)". Pass the flag through run_all.py and add a
SKIP_HELDOUT=1 make variable. Add one test per lane and run make test-fast. Commit on a branch and open a PR
into track-fix.
```

Merge that PR. Then label. Pick 8 training games spread across opponents (any game not held out; the tool
refuses held-out games when sampling). For each game:

```bash
python3 -m analytics.label_tool --data "$DATA_ROOT" --game 11.14.2015.BKN.at.GSW --sample 50 --seed 0 --port 8765
```

Open http://127.0.0.1:8765/ and follow `docs/labeling_guide.md` (seven actions, seven coverages, keys,
decision rules). After all 8 games, relabel every tenth sampled possession with `--labeller-pass 2` as the
guide describes, then:

```bash
python3 -m analytics.label_tool --data "$DATA_ROOT" --kappa
```

Check: kappa at least 0.7 for actions and for coverages. Below that, tighten the definitions in the guide
and relabel the disagreements before stage 6 runs actions, coverage or playcalling.

## Stage 5: real runs, phase by phase, held-out not scored (about 1 to 2 hours of compute)

Run one phase, read it, then the next:

```bash
python3 run_all.py --data "$DATA_ROOT" --phase 1 --continue-on-error --skip-heldout
```

```text
Follow the rules in docs/mac_runbook.md. Stage 5, phase <N>: read reports/run_all.md and logs/run_all/ for
this phase. For each lane: validate counts (rows dropped and why), the out-of-fold candidate table, the
selected candidate and whether any check or gate fired. Fix data problems in code with a test, never by
looking at held-out rows. Report per lane: selected model, out-of-fold metric with interval, anything that
looks wrong. Commit code and reports (they now carry real training-fold numbers; replace the "fixture
output" banner with "real data, training folds only; held-out not scored").
```

Repeat for phases 2, 3, 4 and 5. Phase 1 is xfg and rebounds; also run the Phase 1 baseline
`python3 -m analytics.train_xfg --shots "$DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet" --report reports/xfg_baseline.md --skip-heldout`.
Phase 2 is matchups, gravity, ghosting, embeddings. Phase 3 is epv, actions, coverage (needs the stage 4
labels). Phase 4 is decisions, playcalling, lineups. Phase 5 is winprob, load, betting (betting needs odds).

## Stage 6: deep models where the gate opens (a day, only if needed)

The ghosting GRU, the EPV sequence model and the embeddings autoencoder stand in as sklearn MLPs. Each lane's
report says whether its baseline gate opened.

```text
Follow the rules in docs/mac_runbook.md. Stage 6: for each of ghosting, epv and embeddings, read whether the
gate opened in its report. Only where it opened: pip install torch, replace the MLP stand-in with the model the
contract names (ghosting GRU over the same window and roll-forward; EPV sequence model over the last five
states; embeddings autoencoder), keep the same keep rule fixed in advance, add tests, rerun that lane with
--skip-heldout. Where the gate stayed shut, change nothing and say so in STATUS.md.
```

## Stage 7: freeze (1 hour)

```text
Follow the rules in docs/mac_runbook.md. Stage 7: for every lane, write the selected model, its settings and
every threshold into its docs/contract page under "Frozen for the held-out run", and list them in STATUS.md.
Commit, tag the commit freeze-v1 and push the tag. After this no code or setting changes before stage 8.
```

## Stage 8: score the held-out games once (1 to 2 hours of compute)

```bash
git checkout freeze-v1 && python3 run_all.py --data "$DATA_ROOT" --all --continue-on-error
```

```bash
python3 -m analytics.train_xfg --shots "$DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet" --report reports/xfg_baseline.md
```

```text
Follow the rules in docs/mac_runbook.md. Stage 8: this was the single held-out run at tag freeze-v1. For each
lane, copy the held-out numbers into its report next to the out-of-fold ones, with intervals, and say
plainly where held-out is worse than out-of-fold. Do not change any model or setting. Commit the reports on
a branch from track-fix and open a PR.
```

## Stage 9: write up (half a day)

```text
Follow the rules in docs/mac_runbook.md. Stage 9: write reports/results.md: one section per extension with
the question, the data, the selected model, out-of-fold and held-out numbers with intervals, what failed and
what it means, then a one-page summary at the top. Update portfolio/hiring_manager_page.html with the real
numbers (keep every claim to what the reports show) and run make site. Rewrite STATUS.md as the closing
record. Open a PR.
```

## Blocked on you

- The Mac runs in stages 2, 3, 5 and 8 (overnight jobs for 2 and 3).
- The audit in stage 3 (about 3 hours) and the labels in stage 4 (about 27 hours).
- An odds API key, or a decision to leave betting as code only.
- Whether to try the NBA.com tables from a browser session or go without them.
