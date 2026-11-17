# The site polish loop (PLAN.md P4)

A loop in the style of the Darwin Godel Machine: propose one change to the site, build it, score it with machine
checks only, keep it when the site is better by more than the noise of the measurement, log every try. Nobody
grades a candidate by eye. Under D76 the harness exists now and the loop runs only after P3 and the methods page
are merged, so it polishes the site that ships.

## Files

- `web/scripts/polish-score.mjs`: the scorer. One JSON per scored site.
- `web/scripts/polish-loop.mjs`: the driver (`baseline`, `start`, `candidate`, `status`).
- `web/scripts/lib/polish.mjs`: weights, cost, gate, keep rule, route sample, bundle size, copy grade.
- `web/scripts/lib/polish-guard.mjs`: what a candidate may change.
- `web/scripts/lib/polish-check.mjs`: npm run check for one variant.
- `web/tests/unit/polish.test.ts`: tests of all of the above, in the unit step of npm run check.
- `reports/polish/baseline-public.json`, `reports/polish/baseline-private.json`: the baseline of the site.
- `polish-runs/<run>/` at the repository root (ignored by git): the state, the scores and `log.jsonl` of one run.

## What is measured

The scorer serves `web/out` on 127.0.0.1 under its base path (the server of `serve-out.mjs`) and uses only
Playwright's bundled headless Chromium and WebKit. It samples one page per family of pages (home, review, one
game, one possession, prep, one player, results, the 404 page, and on the private build the talk and one slide),
the first in natural order, so the sample is the same on every build of the same data.

| Metric | How | In the cost |
| --- | --- | --- |
| `a11yWeighted` | axe-core 4.14.0 (pinned dev dependency, injected into the page), WCAG 2.2 A and AA rules and axe's best practices, every sampled page at 390 and 1280 px in both browsers, reduced motion. Sum of impact weight (minor 1, moderate 2, serious 3, critical 4) times nodes. | 0.25 |
| `firstViewChromiumMs` | Each sampled page loaded 7 times in a fresh context at 1280 px; per page the median time to the largest paint; mean over pages. Chromium runs on an emulated connection and processor (40 ms latency, 10 Mbps down, 4 times slower processor), so the time follows the bytes and the script a page needs. | 0.20 |
| `firstViewWebkitMs` | The same in WebKit (the load event where it reports no largest paint). WebKit has no such emulation and runs as is, in tens of milliseconds, where the spread from run to run is near a sixth of the value; so it weighs least. | 0.05 |
| `firstLoadJsKb` | Gzip size of every script the page's HTML loads (the nomodule fallback aside), mean over the sampled pages. Read from `out/`, no browser. | 0.20 |
| `pageWeightKb` | Bytes of every response before the load event, median over the loads, mean over pages and browsers. | 0.10 |
| `readingGrade` | Flesch-Kincaid grade (the counter of `check-reading.mjs`) of every sentence of `web/content`, weighted by its words; labels and presenter notes left out. | 0.20 |

The score also records, outside the cost: time to DOMContentLoaded and to the load event per page and browser,
the CSS size per page, the problems of `check-copy.mjs` and `check-reading.mjs`, the grades of the cold open and
the slides, the outcome of npm run check, every console error and uncaught error, and the number and a hash of
the built pages.

## The cost

    cost = sum over metrics of weight x (value + offset) / (reference + offset)

Lower is better. The reference is the median of the baseline runs, so the baseline costs 1.000; a metric 10%
lower moves the cost by a tenth of its weight. The offset is 1 for the accessibility count, which can be zero,
and 0 for the rest. Why these weights: accessibility and speed are what P4 names first, so they carry half; the
script size is the part of speed a change controls most directly; the grade 9 rule of the copy is a hard rule of
the site, and the grade below it still matters to a reader; page weight is partly counted in the first view
already, so it carries the least.

## The gate

A candidate is rejected whatever its cost when any of these holds (`gateReasons` in `lib/polish.mjs`):

- npm run check was not run or failed. Public: all thirteen steps of `npm run check`. Private: the steps that do
  not depend on the variant, the private build at the root of a host, and the browser suite on it.
- `check-copy.mjs` or `check-reading.mjs` reports a problem.
- A sampled page answers with another status than 200, throws an uncaught error or logs a console error.
- Accessibility: a violation (browser, width, page family, axe rule) the base did not have, or more nodes for one
  it had. Fixing one is fine.
- The data build or the set of built pages differs from the base: a polish change keeps the data and the pages.
- The change touches a protected path (below).

## The keep rule

The noise of a variant is the spread (largest minus smallest) of the cost over the baseline runs of the unchanged
site, and never less than 0.01, because three runs in one sitting understate it: the baseline runs spread by
0.0009 for each variant, while the unchanged site, scored again later as a comment-only candidate, came out 0.0037
lower (the self-test below). Near three times that drift is the floor. A candidate is kept when every gate
passes, no variant's cost rises by more than its noise, and at least one variant's cost falls by more than its
noise. Its scores then become the base the next candidate must beat.

## What a candidate may change

Only `web/app`, `web/components`, `web/lib`, `web/content` and `web/public`, and inside them never the data
contract (`web/lib/types`) or the site data (`web/public/data`). Everything that judges the site is protected:
all of `web/scripts` (the checks, the copy rules, the scorer, the driver, the guard), `web/tests`, `tests`,
`web/eslint.config.mjs`, the Playwright and TypeScript configuration, `web/next.config.ts`, the package files,
`docs/SCRIPT.md` (the approved copy), this file, `reports`, `export`, `tools`, the frozen code, PLAN.md,
CLAUDE.md and `.claude`. The driver refuses such a patch before anything is built, and npm run check refuses it
again: the driver sets `POLISH_BASE` to the loop branch, and a test in the unit step fails when the change since
that branch touches a protected path.

## Running it

Every command but `status` builds the site and drives browsers, so it runs under the heavy-job lock, from the
root of a worktree outside iCloud, one command per lock. Real data: `SITE_DATA_DIR` and `MEDIA_PRIVATE_DIR` name
the shared data and clips folders of the worktrees.

    # once, on the merged site: a loop branch polish/<run>/loop and its baseline (3 scored runs per variant)
    python3 -m export.heavy --name "polish start" -- node web/scripts/polish-loop.mjs start --run r1 --base v3-finish

    # per proposed change: a patch (git diff) made by a proposer outside the repository
    python3 -m export.heavy --name "polish candidate" -- \
      node web/scripts/polish-loop.mjs candidate --run r1 --patch <file> --message "Lazy-load the court on review"

    node web/scripts/polish-loop.mjs status --run r1

`start --baseline reports/polish` reuses a stored baseline instead of measuring again, when no site file changed
between the commit it was scored on and the base. A candidate is a branch `polish/<run>/cNNN` from the loop
branch with the patch as one commit; a kept one moves the loop branch to it (fast-forward), a rejected one stays
as a branch for the record. Every candidate, kept or rejected, is one line of `log.jsonl` with its files, gate
reasons, costs, noise and decision. The driver never pushes and never merges into another branch; the lead
reviews the loop branch and merges it like any other branch (CLAUDE.md, Milestones). When a run ends, copy its
`log.jsonl` into `reports/polish/` to keep it.

A proposer (a person or an agent) writes each patch from the site's source and the latest score file. One change
per patch, so the log says what each change was worth. The proposer never edits a protected path and never sees
a held-out number it could optimise against: the score holds only page timings, sizes, accessibility findings
and copy grades.

## Baseline (2026-11-16)

`reports/polish/baseline-public.json` and `reports/polish/baseline-private.json` hold the values, the per-page
detail and the noise, scored on the real data of the worktrees after npm run check passed for each variant. Both
have no accessibility violation, no page failure and no navigation retry; the cost spread over three runs is
0.0009 for each.

## Self-test (2026-11-16)

`reports/polish/selftest-log.jsonl` is the log of a run on the public variant with two candidates, made to show
the driver works: c001 edited `web/scripts/check-copy.mjs` and was refused by the guard before anything was
built; c002 added a comment to `web/lib/format.ts`, passed the gate and was rejected because its cost (0.9963)
did not beat the noise (the floor was 0.005 then; this result is why it is 0.01 now). Its branches were removed
after the test.
