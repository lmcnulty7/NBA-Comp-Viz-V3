# Results summary: `analytics/results.py` and `reports/results.md`

Stage 9 prep (docs/mac_runbook.md). One file that summarises every lane report: an overview table with one row per
lane, then one summary table per lane. Code: `analytics/results.py`; tests: `tests/test_results.py` (in
`make test-fast`). It reads markdown only: no DATA_ROOT, no parquet, no network.

```
python3 -m analytics.results [--reports reports] [--out reports/results.md] [--strict] [--check]
make results
```

| flag | effect |
| --- | --- |
| `--reports DIR` | folder of the lane reports (default `reports`) |
| `--out FILE` | the summary (default `reports/results.md`) |
| `--strict` | exit 1 when any lane field is unparsed or a lane report is missing; without it, unparsed fields are written as such and the exit is 0 |
| `--check` | write nothing; exit 1 when FILE is missing or differs from what would be generated (the generated-at time is read from FILE, so only content counts) |

Paths in the generated text (the source reports, the command) are printed relative to the working directory when they
lie under it, else absolute, so `--check` gives the same answer however `--reports` and `--out` are spelled. A run
whose content is unchanged keeps the old generated-at time and leaves the file byte for byte as it was, so
`make results` with nothing new is a no-op diff.

## 1. What it reads

`reports/<ext>.md` for the fifteen extensions of `analytics.registry.EXTENSION_NAMES`, in registry order, each
written by `analytics/<ext>/train.py` (`write_report`). `reports/xfg_baseline.md` (the Phase 1 baseline of
`analytics/train_xfg.py`) is read when present and skipped silently when absent. A missing lane report is a row
with status `report missing` and counts as unparsed.

## 2. How a report is read

Nothing is computed. Every value is copied as written (a cell such as `0.6843 [0.6653, 0.7095]` becomes value
`0.6843` and interval `[0.6653, 0.7095]`; an interval stated as separate lo and hi columns is copied as `[lo, hi]`).
The extraction is one explicit rule pair per lane in `analytics.results.LANES`: a training rule and a held-out rule,
each keyed on the section headings and sentences that lane's writer emits (the fixture tag is stripped from headings
before matching). When a rule does not find what it expects it raises, and the field is shown as
`unparsed (<reason>)`. Every held-out rule looks for the first level-2 heading that starts with `Held-out` (a writer
may say "scored once" or "not scored" after it), never for the rest of the heading. Examples of what the rules read:

| lane | selected model from | primary metric |
| --- | --- | --- |
| xfg | `Selected: **x**` and the `selected` column of the candidate table | out-of-fold log loss |
| rebounds | `Selected: x.` | per-miss log loss |
| matchups | `Selected candidate: x.` outside the held-out block; fallback `Selected candidate: x (` in the held-out section | possession points MSE |
| gravity | the `SHIPPED` role in the candidate table (first run) | MSE of nearest-defender distance |
| ghosting | `Selected ghost:` in Decision | 1.0 s position error |
| embeddings | `Chosen: **x**` in Decision | 5-NN position accuracy (leave-one-out) |
| epv | `Selected: **x**` | MSE expected points, bucket all |
| actions | `Selected model: **x**` | F1 ball_screen |
| coverage | `Selected: x.` and the `selected` column | macro F1 |
| decisions | `Selected: **x**.` on the prior line, with or without a parenthetical | mean Spearman of decision quality vs on-court rating |
| playcalling | `Selected candidate: x.` outside the held-out block; fallback `the selected model (the x)` or a `x (selected)` row | possession points MSE (ridge adjustment) |
| lineups | `Mean validation Spearman: ... Selected: **x**.` | the mean out-of-season Spearman over every validation pair, from that line (no interval stated); each pair's Spearman and MSE are listed separately |
| winprob | `Selected: **x**` | mean bucket Brier (the selection metric; no interval stated) |
| load | `Selected candidate: x.` lines outside the held-out block (one decline model, one absence model unless the absence baseline did not run); fallback `MSE, selected model (x)` and `log loss, selected model (x)` in the held-out section | y_decline MSE |
| betting | `Selected: cover model **x**, margin model **y**` | cover log loss |
| xfg_baseline | the logistic model (train_xfg has a constant floor and no selection rule) | log loss, in-sample or out-of-fold as the heading says |

Degenerate runs the writers document, read as valid records rather than unparsed:

| shape | writer | how it is summarised |
| --- | --- | --- |
| one training game (no grouped fold) | rebounds, matchups, gravity, ghosting, epv, actions, coverage, decisions, playcalling, betting: the candidate heading carries the mode `in-sample`, or the protocol sentence says IN-SAMPLE | every training metric name gets ` (in-sample)` and the first gate is `metrics in-sample` with the report's words; the heading is found by its prefix (`Candidates (`, `Metrics (`), never by `out-of-fold` |
| nothing fitted | embeddings: fewer than 3 sufficient player-seasons, a report with only Inputs and Status, the Status body `fewer than 3 sufficient player-seasons: nothing fitted` | selected `none (<that text>)`, metric `5-NN position accuracy n/a (nothing fitted)`, gate `nothing fitted`; any other embeddings report without a Decision section stays unparsed |
| one training season | lineups: `One training season only: no validation pair, plus_minus selected by default.` | the mean Spearman as written (n/a) with that sentence as its note |
| absence model not run | load: `No player-game has a known absent_next` | decline model only; the sentence is a gate |
| lo and hi both n/a | decisions, playcalling (one game) | no interval, shown as "no interval stated" |

Every other early exit in the lane trainers (no training game, no training rows, one game where a lane needs two
folds) raises SystemExit and writes no report, so the lane shows `report missing` or the previous report. The load
trainer crashes in its one-training-game run (`bootstrap_by_game needs at least one game`) before writing a report.

Lines of the form `Selected candidate: <name exactly as in the out-of-fold table>.`, outside every held-out section,
are preferred wherever they appear (the --skip-heldout change adds one before the held-out heading of matchups,
playcalling and load, with and without the flag). More than one such line where one is expected is unparsed. Older
matchups, playcalling and load reports name their selection only inside the held-out section; written without that
section and without the new line, they show the selected model as unparsed: the summary never infers a selection from
the numbers. Other optional parts: the epv reason line (`Why the constant: ...` or `Why: <model> has the lowest ...`),
the load absence model (`No player-game has a known absent_next` makes it a gate, not a metric).

## 3. The record (`LaneResult`)

| field | meaning |
| --- | --- |
| `lane`, `path` | the lane id and the report it came from |
| `written_at`, `command` | from the report's `Written <time> by \`<command>\`` line |
| `status` | the data status, section 4 |
| `selected` | the selected model, in the report's words |
| `oof` | the primary training-fold metric: name, value, interval (empty when the report states none) |
| `oof_extra`, `candidates` | secondary metrics of the selected model; every candidate's primary metric |
| `heldout` | the held-out state (section 5) with the copied metric and its extras when scored |
| `gates` | (label, text): gates and checks that fired, the text copied from the report (deep candidate kept or dropped, paired bootstraps, sample gates, targets, sign tests) |
| `notes` | context lines copied from the report |
| `problems` | (field, reason) for each unparsed field; any entry makes the lane unparsed |

## 4. Data status

| status | when |
| --- | --- |
| `fixture output, not a result` | the report contains "fixture output, not a result" anywhere (every `--fixtures` run) |
| `smoke run on fixture pages, not a result` | xfg_baseline written with `--smoke` |
| `real data, training folds only; held-out not scored` | no fixture banner, and the stage 5 banner of docs/mac_runbook.md or the `--skip-heldout` phrase is present |
| `real data, held-out scored once` | no fixture banner, held-out numbers stated |
| `real data` | no fixture banner and no held-out numbers: embeddings (it scores no held-out rows) or a lane whose held-out field is unparsed (section 5) |
| `report missing` | the file does not exist |

Mixed states are shown per lane. Whenever any lane is fixture output (or the smoke run), the generated block opens
with the banner **Fixture output, not a result.** and the count of such lanes, and each such lane's heading carries
`(fixture output, not a result)`. With only real inputs neither appears.

## 5. Held-out states

| state | when |
| --- | --- |
| a copied metric | the lane's held-out rule found the selected model's held-out numbers; "point estimate" when the report states no interval |
| `not scored (--skip-heldout)` | the report contains the exact phrase `held-out games not scored (--skip-heldout)` |
| `not scored (stage 5 banner: held-out not scored)` | the report carries the stage 5 banner without that phrase |
| `not run yet` | the report says the held-out rows were set aside and not scored (the baseline smoke run) |
| `absent` | the report has no held-out section: expected only for embeddings; for every other lane, without the --skip-heldout phrase or the stage 5 banner, it is also unparsed ("no held-out section or row"), as is a betting report whose P&L table has no held-out season row and a load report whose held-out section says no player-game was loaded |
| `unparsed (<reason>)` | the held-out rule failed; or the report says held-out was not scored yet states held-out numbers; or the report has held-out numbers but its training-fold numbers could not be read (a held-out number is never shown alone) |

## 6. The output file and the markers

The generated text sits between `<!-- results:begin -->` and `<!-- results:end -->`. Regeneration replaces only
that span; everything before and after (the stage 9 prose, written by hand) is kept byte for byte. A file without
markers keeps its text: when its first line is a `# ` title the block goes under that line, otherwise at the top. A
file with an unpaired or repeated marker is refused and left unchanged; that includes hand text that quotes a marker
string literally (write it in other words, for example "the results begin marker"). A new file is `# Results`, the block, and a comment saying where hand
text goes.

Inside the block: the fixture banner when it applies, a generated-by line (time, canonical command), a count of
unparsed lanes when any, the source table (lane, report, status, written-at, command), the overview table (lane,
status, selected model, out-of-fold metric [interval], held-out, gates and checks fired, each gate as its label plus
the deciding words of the report's sentence) and one `### <lane>` table per lane with every field of section 3.

## 7. Adding or changing a lane

When a lane's `write_report` changes a heading or a selection sentence, `tests/test_results.py`
(`test_every_committed_report_parses`) fails on the regenerated report: update that lane's two rule functions in
`analytics/results.py` and the table in section 2. A new lane needs a `LANES` entry in registry order.
