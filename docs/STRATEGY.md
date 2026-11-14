# STRATEGY.md: a proof-of-concept front end and an analytics product, joined by one data contract

Written 2026-11-04 (Lucien's decision). State of the repo it describes: commit `0ca0e4c`
(2026-10-06), ROADMAP.md Phase 2 open at R2.4 (see `docs/STATE_AND_EXTENSIONS.md`). This document
states the strategy; the ROADMAP.md and CLAIMS.md amendments it needs are proposals until Lucien
approves them (listed in the session notes of 2026-11-04, not yet applied). No em dashes.

## The two parts

**1. The CV front end is a proof of concept.** Broadcast video becomes tracking data for a few
games, validated against SportVU. "Working" means the Stage 1 targets in CLAIMS.md ("Stage 1
definition of done") are met on the held-out game AND the held-out arena, under the adoption rule
as amended 2026-10-04. It does not mean every production game is re-tracked, and it does not mean
100% of anything. Today the best measured build meets position error on the held-out game (p50 1.84
/ p90 4.95 ft, `reports/scorecard/gsw_phx_2016__r23_cam.txt`, 2026-10-06) and misses it on the
held-out arena (2.17 / 5.49 and 2.26 / 5.45 ft, `reports/scorecard/cle_*__r23_cam.txt`); missed
players 25.1%, ghost boxes 15.1% and team labels 81.8% fail their rows on every build.

**2. The analytics layer is the product.** Shot quality, matchup assessment, lineups, spacing,
possession value, win probability, strategy recommendations. It runs now on data that already
exists and is not limited by what the CV can do today: SportVU 2015-16 logs (8 games under
`data/sportvu/`, 636 listed by `sportvu/manifest.py`), basketball-reference play-by-play (16 games
under `data/pbp/`, `fetch_pbp.py`), and other public sources listed in `docs/DATA_SOURCES.md` with
their verification status. Later, any game with a video feed runs through the CV and feeds the same
analytics.

**3. The bridge is one schema.** Every source (SportVU, nba.com aggregates, play-by-play, CV
output) converts into the schema in `docs/DATA_CONTRACT.md`, and the analytics read only that
schema. Every row names its source and carries a quality field; for CV rows the quality field is the
scorecard id of the build that produced them (`reports/scorecard/<game>__<build>.json`).

**4. The swap test decides when the CV is good enough for an analysis.** On the proof-of-concept
games (the SportVU-synced games in `sportvu/splits.json`), the same analysis runs on SportVU rows
and on CV rows through the same code. Matching conclusions, by a criterion fixed before looking,
show the CV is good enough for that analysis and no other. The design is proposed in the
2026-11-04 session notes (section E) and becomes a plan item once approved.

## What changes

| Before (ROADMAP.md, 2026-10-03) | After (this strategy) |
|---|---|
| "Extend nothing in the stats layer until Phase 5"; Stage 2 and Stage 3 "start only when Stage 1 hits its targets" | An analytics track runs now, in parallel, on public data; Stage 1 continues unchanged as the CV proof of concept |
| CV finish line: Phases 2 to 5 complete, 14 production games re-harvested (R5.2), then "Stage 2 planning starts in a new document" | CV finish line: Stage 1 targets met on the held-out game and the held-out arena, the R5.1 side-by-side demo, and a passed swap test on at least one analysis. R5.2 re-harvest becomes optional, run when an analysis needs a game that only video can supply |
| CLAIMS.md kill list: "Ball tracking, shot-quality models" | Ball tracking from video stays unplanned (no CV ball row). Shot-quality and other ball-dependent models are in scope for the analytics layer on sources that carry the ball (SportVU `moments[].ball`), and every such result names its source |
| Numbers describe the pipeline | Numbers describe a (source, analysis) pair. A result on SportVU rows is a result about those games; a result on CV rows is also a statement about the CV build named in its quality field |

## What does not change

- **Measurement discipline.** CLAIMS.md stays the contract: a result ships with its artifact path,
  its n and its caveats, or it does not ship. Corrections are recorded, never deleted.
- **Held-out splits.** `sportvu/splits.json` governs. The held-out game gsw_phx_2016, the held-out
  arena CLE and the held-out era gsw_nyk_curry54 are never trained, tuned or thresholded on, for
  CV models and for analytics models alike. The swap test runs on held-out games, so any analytics
  model used in it must have been fitted on other games.
- **Adoption rules.** A CV stage ships only by the ROADMAP.md rule (every pass/fail row it touches
  improves on the point estimate; none is worse beyond the paired window-bootstrap 95% interval;
  coverage floor 50%). Analytics models adopt by a rule of the same shape, written into CLAIMS.md
  before the first result: a stated baseline, a held-out set by game, an interval, and a named
  artifact.
- **The single scorecard** (`python -m sportvu.bench`) stays the only source of CV quality numbers,
  and its id travels with every CV row into the analytics.
- **Provenance caveats.** CLAIMS.md standing caveat 4 (public YouTube uploads for research and
  education, no video redistributed, derived data and fair-use stills only) extends to every new
  source: data stays under `data/`, gitignored; download scripts, manifests and licence notes are
  committed (ROADMAP.md "Data: how much, and why it is free").
- **Kill-list items that do not depend on the ball** stay killed: league-wide or era-normalised
  conclusions on the GSW-centric 2012-2017 sample (CLAIMS.md B4, kill list); closeout tendency as a
  headline metric. Lifting one requires a dated [HUMAN] line in CLAIMS.md.

## How the parts are held to the same standard

| Question | CV front end | Analytics layer |
|---|---|---|
| What is the truth? | SportVU on the held-out game and arena | A held-out set of games from the same source, plus the swap test where CV rows are used |
| What is the baseline? | V3, then the adopted build | A stated simple model (for example the start-type-matched baseline already used by `tier2_credit.py`) |
| What is the unit of error? | Window-cluster bootstrap over observations both builds score | Game-cluster bootstrap, as `data/pbp/tier2_credit.json` already does (2,000 resamples, 95% percentile) |
| Where does the number live? | `reports/scorecard/` | `reports/analytics/<analysis>__<source>.{json,txt}` (proposed path) |
| Who decides adoption? | Lucien at the [HUMAN] gates | Lucien, per analysis, against the rule written in CLAIMS.md |

## Open items this strategy creates (for Lucien)

Listed in the 2026-11-04 session notes, section G: approve the ROADMAP.md and CLAIMS.md diffs;
fix the swap-test analysis, games and match criterion before anyone looks at results; choose the
first analytics extension; decide what "nba.com aggregate" rows may be used for before their terms
and access are verified (`docs/DATA_SOURCES.md` marks them "to check").
