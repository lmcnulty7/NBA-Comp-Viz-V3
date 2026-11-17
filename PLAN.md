# PLAN.md: finishing V3

## Morning summary for Lucien (2026-11-14, written by Claude Code)

**Update 2026-11-15.** Lucien settled the reserved items in D60 to D65 and Claude Code carried them out (D66, D67):
v3-finish merged into main (24cab9f) and pushed, the public site live on GitHub Pages without SportVU frame data,
the private site live on Cloudflare Pages behind Access. M8 and M9 are done (status lines below). Still open: M11
(Lucien's adjustment list, the C14 two-minute test, C12 load time in real browsers, then the v3-site-1 tag).

**Where it stands.** M0 to M7 and M10 are closed on branch v3-finish, each with an independent ml-review and verifier
pass. Nothing was pushed, merged into main, published or deployed. Every check is green on the real data: npm run
check (all 13 steps) and the Python fast suite (count in the M11 status line).

**What was built.**
- export/: the data contract (export/schemas.py), the site export (17 held-out games, 3,162 possessions, 405
  player pages, scorecard, results) and the demo bridge (broadcast build r23 on 25 windows of the held-out game).
- tools/: the frozen models were never saved at freeze-v1, so they were rebuilt from the frozen code on training
  games and kept only because they reproduce every stored held-out prediction exactly (xFG 2,939, EPV 44,703,
  ghosting 1,741,995; D45, D49). Record: reports/export/frozen_checks.json.
- web/: the site (home cold open, possession review, prep tab, results, private /talk slides), public and private
  builds, with checks for copy, reading grade, contrast, data, video and 1,100+ browser tests.
- Docs: README rewritten, prototype numbers tagged, docs/SCRIPT.md, CLAIMS caveat 4 amendment drafted.

**What the demo shows (an illustration).** Run on broadcast instead of sensor positions, xFG moves 0.032 on average
(43 shots), EPV 0.026 points (942 states), and the 1.0 s ghosts move 3.30 ft (71 possessions). 48.7 percent of
player-frames had to be filled from sensor data because the broadcast tracker had no matched box there.

**Things you should know.**
- PLAN.md's tracking table was wrong in two places; the site follows the files (D43, D52): near-field bias is not
  met for either build, and the arena line mixed two rows.
- The camera model was not adopted for the site (D42): r23 is shown, r23_cam beside it as a candidate.
- Reviews caught real problems that are fixed: the demo once recomputed a held-out ghost error (D58), and the review
  page once said no model saw these games (the frame gate trained on frames of the PHX game).
- The Fable usage limit stopped the team at about 02:00; the build continued on Opus 5.5 (D57).

**Self-decided D lines (each with its options and reason below):** D38 reviewer agents; D39 agent team; D40
untracked files; D41 worktree cleanup; D42 camera model not adopted; D43 site follows the scorecard files; D44
build outside iCloud; D45 rebuild frozen models; D46 extra tooling; D47 prep definitions and minimum samples; D48
featured rule; D49 what "reproduces" means; D50 demo constants; D51 shot-making label; D52 corrections; D53 point
estimates; D54 web toolchain; D55 prototype tags; D56 CLAIMS amendment and SCRIPT approval; D57 usage-limit stop;
D58 ghosting swap item; D59 web review fixes.

**Waiting for you (reserved by D37).**
1. Merge v3-finish into main (C3), and set the repository description.
2. Read and accept or edit: the CLAIMS.md caveat 4 amendment, docs/SCRIPT.md, the site name placeholder ("Pixels to
   Possessions", D54).
3. M9 deploy, with two questions only you can settle:
   - The site data is built from the Drive data, which a GitHub Action cannot reach, so the public build needs
     prebuilt data (for example built here and published from out/).
   - The public site as planned ships 5 Hz player positions of 17 SportVU games; the SportVU data is marked "not for
     redistribution" (docs/DATA_CONTRACT.md). Decide what the public build carries.
   - The public out/ is about 524 MB in 21,000 files: fine for Pages limits, slow to upload.
4. Cloudflare Pages and Access for the private site (D29), and who gets access.
5. M11: your adjustment list, the C14 two-minute test with a real person, C12 load time in real browsers, the tag.
6. Where things are: worktrees and build data under ~/Developer/nba-v3-worktrees (_site-data, _artifacts,
   _media-private); rebuild everything with python3 -m export.demo_bridge then python3 -m export.site_data, then
   npm run build:public or build:private in web/ (CLAUDE.md, Commands).


Mode: LOCKED 2026-11-13 19:30 PST. Changes go through a new D line with a reason; build per Milestones in Claude Code.
Version 15, 2026-11-13 (locked; Parked section added).

## Spec (approved 2026-11-13)
V3 is a basketball analytics product built on player tracking, with a working broadcast-video tracking pipeline
as a measured stand-in for the tracking data a team buys from vendors. A short demo shows broadcast video
becoming court positions on a game no model saw, scored against SportVU ground truth, then runs the same frozen
models (xFG, EPV, ghosting) on broadcast positions beside sensor positions to show how far the answers move. The
main page is a possession review for coaching staff: pick a possession and see shot quality, possession value over
time, and where an average defense would have stood. A second tab supports game and season prep with player views
from the season-level models (shot-making over expected, gravity, rebounding, absence risk, similar players).
Every number names its source, whether it is held out, and its interval; models that found no signal are shown as
tested and rejected. The public site carries stills; the full demo with clips sits on a private, access-controlled copy.

## Tracking state (measured, repo at 149e5c4)
Have: gate, court r23 (+ camera model, not adopted), V3 detector + BoT-SORT, unsupervised teams; rebuild on
25 SportVU-synced windows of the held-out game (gsw_phx_2016, 9,500 frames) and 2 held-out-arena games; one scorecard.
| row | target | best build | pass |
| position p50/p90, held-out game | 2.0/5.0 ft | 1.84/4.95 (r23_cam) | yes |
| position, held-out arena | within 1.5x | 2.17/5.49, 2.26/5.45 | no |
| near-field bias | +-0.5 ft | within 0.5 on all held-out games (r23_cam) | yes |
| missed players | <=5% | 25.1% | no |
| ghost boxes | <=5% | 15.1% | no |
| team labels | >=95% | 81.8% | no |
CV rows lack ball, persistent player identity (21.7 id switches per possession) and reliable team; the frozen
models read all three. No adapter writes CV output into the contract frames table; swap_test() is a stub.

## Done checklist (approved 2026-11-13)
Deliverables
- [ ] C1 Private site (access by Lucien only): demo, possession review (A), prep tab (B).
- [ ] C2 Public GitHub Pages site: same pages, clips replaced by stills and a request-access note.
- [ ] C3 Repo: default branch shows current work; README rewritten to the D17 headline with a 2-minute top
      section; PROJECT_SUMMARY and PIPELINE label prototype numbers; strategy docs committed; stale worktrees
      removed; repo description set.
- [ ] C4 "How it was built" page or slides (D14).
Numbers shown
- [ ] C5 Tracking scorecard vs Stage 1 targets on the held-out game and arena, pass/fail per row, build per D20.
- [ ] C6 Demo comparison: xFG and EPV from CV vs SportVU positions for every shot and possession in the demo
      windows, mean absolute difference and n stated, labelled an illustration (D16).
- [ ] C7 Model results table from reports/results.md: held-out value, interval, verdict, failed models included.
Product
- [ ] C8 Possession review covers held-out games only, so every prediction shown is out of sample.
- [ ] C9 Prep tab: player views with intervals and a minimum-sample rule; similar players from embeddings.
- [ ] C10 Ghosting labelled "average defense", never "best"; decisions and matchups shown as no signal.
Quality bar
- [ ] C11 Every number on the site traces to a file under reports/ (generated, not hand typed); tests pass.
- [ ] C12 Works in Chrome and Safari at phone and laptop width; first view loads in under 3 s.
- [ ] C13 CLAIMS caveat 4 amended with a date (D18, D21).
- [ ] C14 2-minute test: someone new to the project, given only the public link, can say what it does, how well
      the tracking works, and one model result. Run once with a real person before the interview.
Out of scope this week: detection, team and identity fixes; group 2; gate v2 retrain; formal swap-test pass/fail;
re-tracking the 14 games; resume (D15).

## Site structure (approved, B3)
Principle: Python produces data, the web only reads it, through one versioned JSON contract. The same shapes
later come from a live service, so the review page can become the bench tool and the prep tab the season tool.

Python side (repo root)
- export/site_data.py: reads reports/ and DATA_ROOT tables, writes web/data/ (gitignored, regenerated):
  manifest.json (schema_version, build id, source per file); games/<game>/possessions.json (index: id, clock,
  result, xFG, EPV start/end, featured flag); games/<game>/<possession>.json (5 Hz positions to 0.1 ft, ball,
  team, EPV curve, ghost defenders, shot xFG, sources); players.json (B views with intervals, min-sample rule,
  similar players); scorecard.json; results.json; demo/swap.json (CV vs SportVU per shot and possession, fill share).
- export/schemas.py: pydantic models = the contract; emits JSON Schema; TS types generated from it.
- export/demo_bridge.py: CV windows -> contract frames (D27), runs frozen predict only, never fits.
Web side (web/, Next.js static export, TypeScript)
- app/page.tsx: headline (D17) + scroll-driven cold open (D25) + entry points.
- app/review/[game]/[possession]/page.tsx: possession review (A), static params from the index; guided tour on
  featured possessions.
- app/prep/page.tsx and app/prep/[player]/page.tsx: game and season prep (B).
- app/results/page.tsx: model results table and tracking scorecard.
- app/talk/[slide]/page.tsx: interview slides (D35), private variant only; keyboard navigation, presenter notes,
  live court components inside slides; slide copy in content/talk/*.md so text edits stay small.
- components/court/: one canvas court renderer (players, ball, ghosts, truth overlay, recurring marker), used by
  the demo, the review and later the live view.
- components/charts/ (EPV curve, xFG, interval bars), components/evidence/ (show-the-evidence toggle).
- lib/data/source.ts: DataSource interface (games, possession, players); StaticSource reads web/data now;
  LiveSource (later) streams the same shapes. Pages never fetch directly.
- lib/types/: generated from export/schemas.py; never edited by hand.
- media-private/ (outside public/, gitignored): clips, copied in only by the private build.
Builds and checks
- VARIANT=public|private at build time; public build fails if any video file is in out/.
- npm run check: typecheck, lint, schema validation of web/data, no numeric literals in copy, Playwright at
  390 and 1280 px, reading grade and jargon checks on the cold open copy.
- Public: GitHub Actions builds and deploys to Pages. Private: Cloudflare Pages + Access (D29).
Later, not this week: FastAPI service running CV + frozen models on a feed, emitting the same JSON; app moves off
static export for live pages only.

## Options
Settled by D25 to D30.

## Milestones (draft 1; ordered, not dated, one loop runs them in order)
[ASK] = Claude Code stops and asks Lucien (D30). Every milestone ends with its DoD checked and a status line here.

M0 Setup and hygiene
- New branch v3-finish from track-fix; commit PLAN.md first.
- Commit docs/STRATEGY.md, STATE_AND_EXTENSIONS.md, DATA_SOURCES.md; [ASK] keep or drop STRATEGY_PROPOSALS.md
  and DATA_CONTRACT_local_draft.md; portfolio/story/shots stays untracked.
- List stale worktrees (9 under .claude/worktrees, nba-v3-prep, the stage 8 worktree); [ASK] before removing.
- [ASK] Camera model (D20): short why-explanation from the scorecards, Lucien decides, logged as a D line.
DoD: git status clean apart from gitignored data; worktree list shows only active ones; D20 logged; pytest
passes (count recorded here).
Status M0 (2026-11-14): closed. DoD met: git status clean; the worktree list shows the main checkout and the five
teammate worktrees of D39; D20 logged as D42; pytest 907 passed, 1 skipped, 63 deselected (866 passed before any change).
Reviews: /code-review 10 findings, all fixed; ml-reviewer PASS with 5 major and 13 minor findings, settled in D49 to D53
or queued as contract changes before M1 closes; verifier PASS, 4 checks it found loosened by the review fixes were
tightened again. Self-decided so far: D38 to D53.

M1 Data contract and exports (Python)
- export/schemas.py (pydantic, the contract), JSON Schema out, TS type generation script.
- export/site_data.py writes web/data/: manifest, 17 held-out game indexes, one file per possession, players.json,
  scorecard.json, results.json.
- Per-possession values come from the frozen models' stored held-out outputs. [ASK] if any per-row held-out
  output is missing: producing it means a predict-only run at freeze-v1, never a refit or a new metric.
- Featured possessions (about 10) picked by a written rule (stratified by result and xFG bucket, fixed seed);
  Lucien may swap picks.
- Player minimum-sample rule set here and logged.
DoD: one command builds web/data end to end; every file validates against the schema; tests on fixtures; values in
results.json equal reports/results.md (test); no local paths or account address in any file (report_paths rule);
no fit call anywhere in export/ (test); manifest names the source of every file; per-game size recorded.
Status M1 (2026-11-14): closed. One command (python3 -m export.site_data) builds the data in about 15 s, peak 1 GB:
17 held-out games, 3,162 possessions, players.json (447 players, 405 shown, 42 hidden), scorecard.json, results.json,
3,185 files, 57.2 MB, per-game sizes in the manifest; every file validates; results.json and scorecard.json equal
their sources (tests); no private string; no fit call (test). Featured: 10 of 10 cells from 336 eligible.
Rebuilds (D45, D49): xFG 2,939 of 2,939, EPV 44,703 of 44,703, ghosting 1,741,995 of 1,741,995 held-out predictions
reproduced (max difference 2.9e-08, 0, 0); absence and embeddings equal their reports; tracked copy in
reports/export/frozen_checks.json. Reviews: code review found three minor issues, fixed; ml-review found one
blocker in M2 (below), fixed; verifier PASS, pytest 1138 passed, 2 skipped.

M2 Demo bridge (Python)
- export/demo_bridge.py on the 25 gsw_phx_2016 windows with the D20 build: match CV boxes to SportVU per frame,
  fill missed players, drop ghosts, record fill share (D27).
- Frozen predict only for xFG (shots in the windows), EPV (states) and ghosting, on CV and on SportVU positions.
- demo/swap.json (per item deltas, mean absolute difference, n, fill share); demo/frames.json (boxes, court
  positions and truth per frame, synced to clip time).
- Clips (media-private/, default cap 15 s each, 60 s total, adjustable at M11) and one still per scene (public).
DoD: files validate; n and fill share stated in the data; tests on fixture windows; no fit call; clip lengths
within cap; scorecard.json names the D20 build.
Status M2 (2026-11-14): closed. Build r23 on 25 windows: xFG n 43 shots, mean absolute difference 0.032
(probability); EPV n 942 states, 0.026 points; ghosting n 71 possessions, ghosts move 3.30 ft on average; fill share
0.487 overall (stated per possession and per item). Sensor-side predictions equal the stored held-out predictions.
Four clips, 47.7 s, each under 15 s; four public stills committed. Reviews: ml-review blocker (the ghosting item
recomputed the held-out ghost error) fixed by D58; verifier code review found three issues (outcome-bearing shot
columns read, one decode check, a note), fixed; verifier PASS on every DoD item.

M3 Web foundation
- web/ scaffold: Next.js static export, TypeScript strict, ESLint, Node pinned (.nvmrc), npm, lockfile.
- lib/types generated; lib/data/source.ts with StaticSource; design tokens (offense, defense, ghost, truth, ball
  colours; type scale; dark mode); layout and nav (Home, Review, Prep, Results).
- components/court canvas renderer; VARIANT=public|private; media-private copy step for private only.
- npm run check: typecheck, lint, schema validation, no numeric literals in copy, no video in public out/,
  Playwright at 390 and 1280 px.
DoD: both variants build to out/; check passes; the court draws a real held-out possession matching a Python
reference frame (snapshot test); out/ served under the Pages base path works locally.
Status M3 (2026-11-14): closed. Next.js 16.4.6 static export (D54). On the real data: public build 21,167 files,
523.65 MB; private build 21,233 files, 548.66 MB with 4 clips; npm run check passes all 13 steps; the court matches
export/court_reference.py within 0.5 px on a real held-out possession (full and half view, Chromium and WebKit, 390
and 1280); out/ works under /NBA-Comp-Viz-V3. Verifier PASS.

M4 Home and cold open
- [ASK] Rewrite SCRIPT.md for V3 (D25 scene list); Lucien approves before copy goes in.
- Headline (D17); scroll steps: clip (private) or still (public), boxes, court map, truth overlay with the error in
  feet; then the swap comparison labelled an illustration (C6, D16); entry points to Review, Prep, Results.
- Public variant: still plus request-access note (D21).
DoD: C6 met; reading grade at most 9 and jargon check pass on the copy; works at 390 and 1280; reduced-motion
fallback; keyboard reachable; no em dashes.
Status M4 (2026-11-14): closed. SCRIPT.md approved under D56. C6 met on real data (illustration badge; n 43, 942,
71; mean absolute difference 0.032, 0.026 points, 3.304 ft; fill share 0.487; every figure checked against
swap.json by a test). Cold-open grade 2.6 to 7.0, jargon and forbidden-claim checks pass; 390 and 1280 with no
sideways scroll; reduced motion and keyboard tested; no dashes. Each panel says its values are point estimates.

M5 Possession review (A)
- Game picker; possession list with filters (result, xFG bucket, featured); possession page with court playback
  (play, pause, scrub, speed), EPV curve synced to time, shot marker with xFG and interval, ghost defenders toggle
  labelled "average defense", evidence toggle with sources.
- Guided tour across the featured possessions.
DoD: C8 and C10 met; a script loads every possession's data without error; playback at least 30 fps on a laptop
(measured); tour completes on phone and laptop.
Status M5 (2026-11-14): closed. C8 and C10 met (tests). All 3,162 possessions load (240,429 frames, 3.6 s).
Playback 59.4 redraws a second in headless Chromium at 1280 (not a physical laptop). The tour walks its 10 stops on
Chromium and WebKit at 390 and 1280. The ml-review blocker (a held-out sentence that overstated) was fixed.

M6 Prep tab (B)
- Player index (search, sort); player page: shot-making over expected, gravity, rebounding, absence risk, similar
  players; minimum-sample note.
DoD: C9 met; every metric shows an interval or says it is a point estimate; hidden players counted and stated.
Status M6 (2026-11-14): closed. 405 player pages, 42 of 447 hidden and stated; every value shows its interval or
says point estimate; D51 baseline error shown on the player page and in the shot-making sort; absence risk is not
called an injury forecast.

M7 Results page
- Tracking scorecard vs Stage 1 targets (C5) and model results (C7), failed models included, links to the reports.
DoD: C5 and C7 met; values equal the source files (test).
Status M7 (2026-11-14): closed. Scorecard (r23 adopted, r23_cam candidate on the court rows with paired
differences) and the 11-lane model table with failed and inconclusive lanes and 4 pending lanes; a browser test
checks every shown value against scorecard.json and results.json on both builds.

M8 Repo and docs
- README rewritten (D17 headline, 2-minute top section, links, honest scorecard); PROJECT_SUMMARY and PIPELINE
  label prototype numbers; repo description set.
- [ASK] CLAIMS caveat 4 amendment text (C13).
- [ASK] Merge v3-finish into main so the default branch shows current work.
DoD: C3 and C13 met; no unlabelled prototype number in a public doc (grep check).
Status M8 (2026-11-14): done except the items D37 reserves. README rewritten (D17 headline, short version, honest
r23 scorecard); PROJECT_SUMMARY and PIPELINE tagged [prototype] (D55) with a test; strategy docs committed; stale
worktrees removed; CLAIMS caveat 4 amendment drafted and committed for review (D56, C13). Waiting for Lucien: the
merge of v3-finish into main, and setting the repository description (account settings).
Status M8 (2026-11-15): done (D61, D62). CLAIMS caveat 4 carries the amendment plus the D61 line (3ddf6d0). Before
the merge, python3 -m tools.check_no_frame_data found no frame data and no clip in the 12,301 tracked files, and
the 278 commits pushed hold no video, table or SportVU data file and no blob over 5 MB. v3-finish was merged into
main with a merge commit, 24cab9f (main's one own commit, f726066, was already on v3-finish as ee02210, so the
merged tree equals v3-finish). On main: pytest 1155 passed, 3 skipped, 65 deselected; npm run check passed all 14
steps. main and v3-finish pushed; repository description set. C3 and C13 met.

M9 Deploy
- [ASK] GitHub Actions deploy of the public variant to Pages.
- [ASK] Cloudflare Pages + Access for the private variant after the free tier is verified; Lucien adds viewers.
DoD: public URL returns 200 with no video files; private URL requires sign-in; first view under 3 s on both (C12).
Status M9 (2026-11-14): not started; every item is reserved by D37 (deploying, host and account setup). Facts
for that step are in the morning summary.
Status M9 (2026-11-15): done (D60, D63, D64, D67). Public: the public variant (21,167 files, 471.75 MB, frames step
clean) is the only content of the orphan gh-pages branch (dddb265, with .nojekyll), served by GitHub Pages at
https://lmcnulty7.github.io/NBA-Comp-Viz-V3/. Live: 200 on home, review, a game, prep, results, data and stills;
404 on /talk/ and /clips/; no video file on the branch; sampled live data holds no positions. First view on this
Mac with an empty cache, median of 3 (largest paint, Chromium and WebKit): home 284 and 324 ms, review 280 and
231 ms, a game 300 and 320 ms, prep 180 and 179 ms, results 200 and 231 ms (web/scripts/measure-load.mjs).
Private: Cloudflare Pages project nba-v3-private, packaged by web/scripts/package-cloudflare.mjs to 18,071 files
(private browser suite on the packaged folder: 609 passed, 31 skipped, 0 failed), deployed only after Access
answered on both hostnames. https://nba-v3-private.pages.dev and every *.nba-v3-private.pages.dev preview return
302 to the Access sign-in page (pages, data, clips and /talk/ alike); one allow rule for Lucien's account address,
one-time PIN only. The private first view cannot be timed from here through Access; it is part of C12 in M11.

M10 Slides (C4)
- /talk route in the private site (D35), built from the approved SCRIPT.md; slide copy in content/talk/*.md;
  arrow-key navigation, presenter notes view, at least one slide with a live court playback.
DoD: C4 met; every number traces to web/data or reports/; /talk absent from the public build (check); runs
offline from out/ on the private variant.
Status M10 (2026-11-14): closed. 11 slides on the private build (7 scenes, 4 on how it was built), arrow keys and
presenter notes, live court on three slides; every number is a token resolved from web/data; /talk returns 404 on
the public build (test and build guard); every slide loads with no host but 127.0.0.1.

M11 Freeze, polish, acceptance
- Feature freeze; Lucien's adjustment list worked through; C14 2-minute test with a real person; final check run.
- [ASK] Tag v3-site-1 and push.
DoD: C1 to C14 each checked with its evidence named here.
Status M11 (2026-11-14): waiting for Lucien. The final check ran (npm run check on real data, all steps passed;
pytest 1141 passed, 2 skipped, 65 deselected; 866 passed before any change). Open: Lucien's adjustment list, the C14 two-minute test with a real person, C12 load time in a real
Chrome and Safari after deploy, and the tag and push (D37).
Status M11 (2026-11-15): Lucien found the site adequate, with no adjustment list (D68). README links the public site
and says the public copy carries no frame data; the repository website field points at the public site; v3-site-1
tags the merge on main that is deployed. Open: C12 load time in real browsers and phones, and the C14 two-minute
test with a real person; Lucien runs both later and the results are recorded here.
Status M11 (2026-11-16): C12 load time checked by Lucien on the deployed sites: first views fast in Chrome and Safari
on the laptop and on Lucien's phone, private site included (Lucien's report, no timings given). With the measured public
first views on this Mac (largest paint 179 to 324 ms, M9 status) and the browser suite at phone and laptop width
in Chromium and WebKit, C12 is counted met. Open: the C14 two-minute test.
Status M11 (2026-11-16): closed except C14, which Lucien deferred (D69). Every other criterion of C1 to C14 is met
with its evidence in the milestone status lines above.

## Research after V3 (D70 to D81, branch v4-research)

R1 Test set (D71, D77). Status (2026-11-16): done. reports/research/testset.json (4b76ab33), fixed and hashed before any
result: Set A, the 10 untouched 2015-12-05 games (a misnamed upload the frozen parser skipped; docs/research/DATA.md),
Set B, 10 Liga ACB 2025-26 games (out-of-distribution check), and the 2025-26 season pending for the play-by-play
lanes (D79). tools/research_testset.py is the only door; its guard tests fail on any other read, a second score or an
edited ledger. Set A ingested by tools/research_ingest.py (archive hashes checked, frozen conversion, 20
basketball-reference pages under D78; outcome files sealed, D81).
R2 Dossiers (D70, D74). Status (2026-11-16): done (772564e9). docs/research/<lane>.md for the 11 lanes and the broadcast
tracking pipeline: foundations, benchmark with comparability, measured weaknesses, recent work, 23 pre-registered
proposals plus the pilot; every citation fetched at the source, unverifiable claims dropped and listed.
docs/research/BACKLOG.md ranks them; top three argued: decisions-E1, winprob-E1, lineups-E1.
R3 Pilot xfg-E1 (cross Venn-Abers). Status (2026-11-16): done, not adopted. Gate reproduced the frozen out-of-fold
predictions (2.97e-8); development games (607): CVAP minus isotonic -0.00052 [-0.00081, -0.00024], passes; Set A
(10 games, scored once): +0.00067 [-0.00133, +0.00229], fails the +0.0005 bound, inconclusive at 10 games (D80).
Interval width p95 0.0077 and 0.0062: xFG stays a point estimate. reports/research/xfg-E1.md. Closed with the
ml-reviewer before and after the score (PASS, majors settled in D80, D81), /code-review (fixes in f8bb108a, dc463e97),
and the verifier (PASS, pytest 1201 passed, 2 skipped, 65 deselected).
R5 Refresh process. Status (2026-11-16): done. docs/research/REFRESH.md: sources (links checked), queries, triage,
dated log; a monthly schedule is proposed and not created (needs Lucien's yes, D73).
R4 Methods and literature page (D72). Status (2026-11-17): done on v4-research (merge 0419da9c), not deployed. Route
/methods in both variants from methods.json (export/research_methods.py: every text copied from a dossier or
BACKLOG.md, every number of ours traced to a report; check-methods finds all 1181 page numbers in methods.json);
xfg-E1 shown with both stages (development met, test not met, not adopted); README section "Methods and literature".
Contract 1.2.0 shared with P3 (D82). Closed with /code-review (10 findings, fixed by the owner of each file), the
ml-reviewer (PASS, four majors fixed: slope intervals, control-arm labels, the 1.1.0 manifest test, fold wording) and
the verifier (items 1 to 5 met; one em dash in a test input fixed in cde9a3a1). pytest 1338 passed, 3 skipped;
npm run check public passed (552 browser tests), private build and browser suite passed (633).

Parked items worked in parallel (D75, D76), status (2026-11-17):
- P3 options 1 and 2: built and merged (0419da9c), capped sample only. export/genai.py with claude-haiku-5-5; every
  number of a text must be a value of its page as shown, a model's value must be called one in its sentence, a lane
  that did not work is never named; checked when written, in the contract validation and on the page. Sample: 25
  of 30 texts kept (v2 prompts); the full run (405 players, 3162 possessions, about $3.38 projected) needs Lucien's yes.
- P4 harness: built and merged; the loop has not run. Baseline public a11y 0, Chromium first view 366 ms, JS 147.26 KB,
  reading grade 4.165; noise floor 0.01. The guard keeps every check out of a candidate's reach. Before a loop:
  a new baseline on the merged site (the stored one predates it); each candidate costs about 15 to 20 minutes of lock.
- P8 Step 0: docs/research/P8_FEASIBILITY.md (2914c37d): no-go as written, go for a two-phase P8 (A: measure with
  scorecard v2 on a new whole-arena test set; B: experiments). Needs Lucien: the done condition, video rights, a
  GPU budget (25 to 40 GPU-hours) and the targets. P8 is preferred over P5, which is not run on its own.
- P6: docs/research/P6_DIRECTION_BRIEF.md written; the choice is Lucien's.
- P7: waits for the interview date.

After V3 (not in this plan): resume (D15), full story page, group 2, detection and teams, live service.

## Parked (not in scope; each needs Lucien's yes and a D line before any work)
- P1 Proposed D38 (Claude Code decides it at the start of M0 under D37): M0 also writes CLAUDE.md and two read-only agents,
  .claude/agents/ml-reviewer.md (leakage, held-out use, numbers traceable to reports/ or web/data) and
  .claude/agents/verifier.md (runs pytest and npm run check, confirms each DoD item, may not edit tests or
  checks). Every milestone closes with /code-review plus ml-reviewer on its changes, then the verifier.
- P9 Proposed D39 (Claude Code decides it at the start of M0 under D37, with P1): build M0 to M11 with an agent team instead of one loop.
  Lead = the main session: owns PLAN.md, merges, runs every [ASK], closes milestones. Teammates in their own
  worktrees: T1 Python export and demo bridge (M1, M2); T2 web (M3, then M4 to M7 against schema fixtures until
  T1's data lands); T3 writing (SCRIPT.md draft for M4 approval, README draft for M8); T4 reviewer (P1 agents,
  read-only, reviews each merge); T5 researcher (P2, docs/research only). Only one heavy data job (DATA_ROOT
  reads, predict runs) at a time: the 16 GB Mac already crashed once (STATUS 2026-11-11). Teammates never edit
  PLAN.md, never push, never touch frozen group 1 code.
- P2 Researcher session, in its own worktree after M0: writes only docs/research/<topic>.md (source, key idea,
  meaning for this repo, proposed experiment, gain, effort) and docs/research/BACKLOG.md ranked by gain over
  effort. Topics: EPV, shot quality, ghosting and defense, broadcast tracking and re-identification, public
  basketball datasets, GenAI for sports analytics. Needs a literature MCP server in Claude Code.
- P3 GenAI component. Options: (1) scouting reports on the prep tab and (2) possession explanations on the
  review page, both generated at build time with a check that every number matches web/data (fits the static
  site); (3) natural-language possession search (needs a live server, after the interview); (4) VLM jersey
  reading for identity (CV upgrade, after the interview). Recommendation: 1 and 2, after M11 or as an
  amendment.
- P4 After M11: DGM-style site polish loop on branches, scored only by machine checks (accessibility and
  performance audits, load time, bundle size, reading grade, tests); check scripts read-only to the loop.
- P5 After the interview: DGM-style CV improvement loop scored on training games only, adopted under D5,
  held-out scored once at the end.
- P8 After V3 ships: CV to full operation with an agent team (supersedes P5 if chosen). Done means: every
  Stage 1 row passes on the held-out game and arena, new scorecard rows for ball, identity and events
  (shots, rebounds) pass targets set before work starts, and the swap test passes for each analysis it feeds
  (D3). Prerequisites: a training-only dev split for all tuning; scorecard extended to the new rows and made
  read-only to the team; GPU budget (Colab or the A100); labels from the SportVU-synced games (players, ids,
  ball). Team: a lead that owns the plan, merges and the scorecard runs; one teammate per subsystem in its own
  worktree (detection, teams and identity, ball, court and camera, events); a read-only judge that scores
  every candidate the same way. Held-out scored once at the end.
  Step 0 (Claude Code, plan mode, no training): feasibility study. Error analysis of the scorecards (where misses,
  ghosts and team errors happen), published single-camera broadcast results as ceilings, available labelled
  games with video, GPU hours per experiment; output: proposed targets, ranked experiments, and a go or no-go
  for Lucien.
- P6 After the interview: map the next version to Lucien's direction. Open questions: sports as destination
  or vehicle; daily work (models, CV/ML systems, advising, product); what "real-time for coaches" means
  (team job, product, research); 1-year vs 5-year horizon.
- P7 Interview date: tell Claude Code when known, to set milestone days and the polish block.

## How to raise a new idea (no planning chat needed)
Open a Claude Code session in plan mode (or a separate worktree), describe the idea, get a proposal with
options and what each changes, then add it here as a Parked item or, once agreed, a D line. The build loop
never changes scope on its own.

## Decision log (one line each; settled items are not reopened)
Inherited from the 2026-11-13 handoff:
- D1 Identity (amended 2026-11-13 by the approved spec): an analytics product on player tracking; the broadcast CV is a measured stand-in for vendor tracking. Was: tracking first, analytics as applications. Reason: D22.
- D2 Audience is DS hiring managers; compared to Hawk-Eye on output format and honesty, not accuracy. One 720p camera cannot match multi-camera tracking.
- D3 CV is a proof of concept on a few games; analytics run on public data now; CV rows feed an analysis only after its swap test passes. CV quality never blocks analytics.
- D4 "Working" = Stage 1 targets, not 100%. A reachable finish line.
- D5 r23 adopted under the amended rule (worse only if the paired bootstrap interval excludes zero).
- D6 Held-out scored once per lane group (freeze-v1 done, freeze-v2 later). Keeps generalisation numbers credible.
- D7 Betting and NBA.com skipped. No odds key; NBA.com blocks scripts.
- D8 Ghosting keeps the MLP (GRU 0.11 ft worse); fits on 1 Hz rows, evaluated at 5 Hz. Game diversity beat frames per game.
- D9 Matchups frozen with ridge; claim is "no reliable signal". Re-tagging not worth it.
- D10 No history rewrite for the email. Already public as the contact address.
- D11 Public home is GitHub Pages; code changes in Claude Code, one loop in the main checkout, parallel work in worktrees.
Decided 2026-11-13:
- D12 Deadline: presentable within one week (interview), unlimited hours. The interview sets the finish line.
- D13 Site order: CV tracking demo first as the hook, then interactive analytics that act on model results. The demo explains the premise in seconds.
- D14 The site is the product; how it was built goes in a separate page or slides. Keeps the product page clean for users.
- D15 Resume updates come last. They depend on what ships.
- D16 The CV demo is an illustration of the end state (analytics fed by broadcast tracking alone), not a claim that the project's analytics come from CV. Analytics results stay SportVU-sourced and say so.
- D17 Headline is the pipeline, measured end to end ("from broadcast pixels to possession value, measured against ground truth at every step"). Both parts carry weight; the failing rows read as rigor.
- D18 Short broadcast clips go on the site (option A), kept short. Strongest hook; CLAIMS caveat 4 gets a dated amendment.
- D19 Interview details withheld until scope is set. Keeps the plan from fitting one role.
- D20 Camera model (R2.4) is decided in Claude Code after a short why-explanation there. That session has the repo context.
- D21 Clips live only on a private site Lucien grants access to; public Pages shows a still and a request-access note. Keeps clips off the public web.
- D22 For sports orgs the analytics are the product; the CV demo is a stand-in for vendor tracking (Second Spectrum, Hawk-Eye) that teams buy. Matches how teams actually get tracking.
- D23 Users: main page is possession review for coaching staff (A, in-game half of the long-term goal); second tab is game and season prep (B, strategy half); fan view dropped. A is the only one the CV can feed this week.
- D24 Spec and done checklist approved as written (C1 to C14, out-of-scope line included). That line also settles swap test (demo-scale only), group 2 (after the interview), detection/teams (out), gate v2 (no retrain).
- D25 Storytelling S2: demo as scroll-driven cold open; SCRIPT.md rewritten for V3 into the slides; one visual language and evidence toggles across A and B; full story page after the interview. Court Ledger name retired. Puts the style where the hook is and keeps the week.
- D26 Site built with Next.js static export and TypeScript (B3). Polish now and a structure that grows into live tools.
- D27 Demo bridge (a): CV positions, missed players filled from SportVU, ghosts dropped by the match, fill share shown per possession; the swap measures position error only. Keeps n and stays honest when the fill share is visible.
- D28 Site data: every held-out possession (about 3,400, estimate) plus about 10 featured; B from training-fold outputs above a minimum sample, no team views. All shown predictions are out of sample.
- D29 Private host: Cloudflare Pages behind Cloudflare Access (email one-time code); free tier verified before setup. Real access control, clips stay private.
- D30 Claude Code rules: decides alone inside C1 to C14; stops and asks on the camera model, frozen group 1 code or held-out rescoring, any retraining, pushing to main, publishing or host setup, CLAIMS edits, new dependencies beyond Next, d3 and canvas, anything outside the checklist. One loop; PLAN.md committed first; each milestone ends with a checked definition of done and a status line; site numbers only from export/.
- D31 Site structure approved as written.
- D32 Milestones are ordered, not dated; the interview date is handled after the plan. Lucien chose detail now, dates later.
- D33 No scope trimmed for B3's extra day; full detail now, adjustments later.
- D34 Lucien works in React with Claude Code's help. B3 stays.
- D35 Slides are a /talk route in the private site. Reuses the live court; text edits are code changes, kept small via markdown copy files.
- D36 Plan locked by Lucien. Building starts at M0 in Claude Code (D11).
- D37 Autonomy (amends D30): Claude Code decides every [ASK] itself except (a) pushing or merging to main, (b) publishing, deploying or setting up any host or account, (c) deleting anything outside git-tracked worktrees and branches. Each self-made decision is logged as a D line marked "(self-decided)" with the options considered and why, so Lucien can reverse it; all work stays on v3-finish until he approves the merge. Lucien chose not to be woken for reversible calls; external and public actions stay his.
Self-decided under D37 by Claude Code, starting 2026-11-13 (each one can be reversed by a new D line):
- D38 (self-decided) P1 adopted as written: CLAUDE.md plus two read-only agents (.claude/agents/ml-reviewer.md, verifier.md); every milestone closes with /code-review, then the ml-reviewer on its changes, then the verifier. Options: adopt; reviewers without CLAUDE.md; skip. Why: nobody watches the build overnight, so leakage and each DoD need a check that did not write the code. Undo by deleting three files.
- D39 (self-decided) P9 adopted with three changes. Team: lead (main session: PLAN.md, merges, every [ASK], milestone closes); T1 Python (M1, M2), the only teammate that runs data jobs; T2 web (M3, then M4 to M7 on contract fixtures until T1's data lands); T3 writing (SCRIPT.md, README); T4 reviewers (D38); T5 researcher (the P2 scope only: docs/research). Changes: worktrees live outside iCloud (D44); every heavy job takes one machine-wide lock (export/heavy.py), so the one-heavy-job rule is enforced, not trusted; T5 starts last and is dropped first if the build needs the time. Options: one loop; team as proposed; team without T5. Why: Python and web meet only at the JSON contract, so they can run side by side once export/schemas.py exists.
- D40 (self-decided) Untracked files at M0. Committed as they are: docs/STRATEGY.md, STATE_AND_EXTENSIONS.md, DATA_SOURCES.md (each already states its date and the commit it describes), data/matchup_eval/labels.json and labels_discarded_20260716_2343.json (hand verdicts a tracked script reads), reports/broadcast_wishlist.csv (input named by a tracked report), .claude/settings.json and .claude/agents/. Left out of git through .gitignore, not deleted: docs/STRATEGY_PROPOSALS.md (never applied, overtaken by D7, D20 and D24, states swap thresholds nobody approved), docs/DATA_CONTRACT_local_draft.md (an older fork of the committed contract, kept on disk as the bridge's reference), docs/IDEAS.md (personal notes, the precursor of Parked), data/gate_sheet/emb_cache_clip.pkl (a cache), portfolio/story/shots (regenerable), .claude/worktrees and .claude/settings.local.json. Options per file: commit, ignore, delete (delete is reserved by D37c). Why: status must be clean and nothing personal or unapproved should ride into a public repository on the merge.
- D41 (self-decided) Removed nine stale worktrees (eight under .claude/worktrees, not nine as M0 says, plus ../nba-v3-prep; the stage 8 worktree was already gone) and the six worktree-agent-* branches. Checked first: every worktree clean, the six branch commits patch-identical to commits in track-fix (git cherry), the two detached ones and prep/stages-4-6 ancestors of track-fix, only caches inside. Every other branch is kept, including story-page, whose four unpushed commits exist nowhere else (the old SCRIPT.md). Options: remove all; remove worktrees but keep branches; leave. Why: M0's DoD, and no work is lost.
- D42 (self-decided) D20 camera model: not adopted for the site. The scorecard and the demo use build r23 (adopted, D5); r23_cam is shown beside it as a measured candidate with its paired differences. Why-explanation: r23_cam adds one self-calibrated camera position per game and a 4-parameter per-frame fit to r23's weights; boxes and tracks are identical. Held-out game: position 1.92/4.98 to 1.84/4.95 ft (paired p50 change 0.08, interval 0.03 to 0.10), near-third bias -0.67 to -0.27 ft. Held-out arena: mixed (cle_nyk p90 5.46 to 5.49; cle_gsw p50 2.22 to 2.26 and near bias 0.12 to 0.26). Training games: no measurable change. The adoption rule (ROADMAP.md lines 76 to 82) needs every touched row to improve on the held-out game and the arena; three do not, so R2.4 stayed unticked. Options: A show r23; B show r23_cam as the build; C r23 as the build with r23_cam as a labelled row (chosen). Why: the two builds differ only on held-out footage, so taking the better one is a selection on held-out data, and the rule as written says no; the position row passes on the point estimate with either. Reverse by changing SCORECARD_BUILD in export/site_data.py.
- D43 (self-decided) Where PLAN.md's Tracking state table differs from the scorecard files, the site follows the files. Near-field bias is "not met" in every scorecard, for r23 and for r23_cam (far third +1.27 to +2.00 ft; only the near third comes inside 0.5 ft, and only with r23_cam), not "yes". The arena line mixes two rows: generalisation (arena within 1.5x of the held-out game) passes, the absolute position target fails on both arena games. The held-out game passes position on the point estimate only (p90 4.98 against 5.0). Held-out possessions number 3,162, not about 3,400. Options: copy the plan's table; follow the files (chosen). Why: C11. The locked table is left as written.
- D44 (self-decided) Build location. iCloud Desktop and Documents sync is on for this folder (707 evicted files and many " 2" duplicates already in it), so node_modules, build output, generated site data and every teammate worktree live under ~/Developer/nba-v3-worktrees, next to the existing ~/Developer/nba-comp-viz-data. The main checkout stays the source of truth for git and for data/sportvu. Options: build in place; symlink heavy folders out; worktrees outside iCloud (chosen). Why: sync conflicts create duplicate files that would enter the site build, and a symlinked node_modules breaks the bundler.
- D45 (self-decided) Frozen model objects and two outputs are rebuilt by rerunning frozen code, never refitted to anything new. Found: no fitted model was ever saved (fit and scoring ran in one process), and the out-of-fold absence probabilities and the embeddings tables were never written. Without them M2 cannot predict on broadcast positions and the prep tab has no absence risk and no similar players. tools/frozen_models.py reruns the freeze-v1 code unchanged, on training games only, with the frozen settings and seed 0, writes outside DATA_ROOT and outside reports/, and keeps a result only if it reproduces the frozen record: the stored held-out predictions for xFG, EPV and ghosting (predictions compared with predictions; no outcome is read and nothing is scored), the reported out-of-fold numbers for absence and embeddings. A rebuild that fails its check is not used and the site says the output was not stored. Options: (a) this; (b) drop the model comparison from the demo and the two prep views (fails C6 and C9); (c) wait for Lucien. Why: M1's "predict-only run at freeze-v1" assumed saved models; a deterministic rerun that reproduces the stored predictions is the same model, held-out is not scored again (D6 holds), and export/ still holds no fit call.
- D46 (self-decided) Tooling beyond Next, d3 and canvas (D30): pydantic (the plan names it), json-schema-to-typescript and ajv as web dev dependencies (the plan asks for generated types and schema validation), Playwright with its own bundled headless Chromium and WebKit (the plan names Playwright; WebKit stands in for Safari in C12). No system browser or desktop app is launched. Options: hand-written types and a Python-only validator; these tools (chosen). Why: generated types are the contract, and all four are build-time only.
- D47 (self-decided) Prep tab definitions and minimum samples, fixed before any player value was computed. Shot-making: makes above expected per 100 attempts against the lane's location and contest model, which does not know the shooter; at least 100 attempts. Gravity: the stored player table with the lane's own 300-possession gate. Rebounding: rebounds per 100 chances with the model's expectation beside it; at least 300 chances. Absence: mean out-of-fold probability of the frozen logistic with the observed rate beside it; at least 10 games. Similar players: the lane's 5 nearest neighbours with its 300-possession gate. Intervals: bootstrap over the player's own games, 200 resamples, seed 0 (the gravity lane's convention). A player page exists when at least one value clears its minimum; hidden players are counted and stated. Options for shot-making: the frozen xFG model's residual (near zero by construction, because that model has a shooter prior); the contest model (chosen, labelled a non-selected candidate with its calibration error stated). Why: "over expected" has to mean an expectation that does not already know the player; the same trap rules out a rebounding residual.
- D48 (self-decided) Featured possessions rule (M1). Eligible: a held-out possession that ends in a made or missed field goal with a located release, an xFG, an EPV curve of at least 5 states, ghosts on at least 80 percent of its frames, no gap flag, 10 players tracked, 6 to 24 s long. One is drawn at random (numpy default_rng, seed 0, ids in sorted order) from each of the 10 cells of result (made, missed) by xFG fifth; an empty cell is reported, not refilled. The rule reads no model error. Lucien may swap picks. Options: hand picks; top or bottom by error; stratified random (chosen, as M1 asks).
- D49 (self-decided) What "reproduces" means in D45, fixed before any rebuild was used (found by the M0 ml-review). Tolerance caps per artifact: xFG 1e-6 in probability, EPV 1e-5 points, ghosting 1e-3 ft; absence and embeddings must equal the lane report to half a unit of its last printed digit. Every stored held-out prediction must be compared (n equals the stored row count). A check vouches for its files by hash and stops covering a file written after it. A rebuild may not call a frozen entry point that scores held-out outcomes. The rules live in tools/frozen_common.py with tests; the tracked copy of every check record is reports/export/frozen_checks.json. Options: leave the bar to each rebuild; fix it in one place (chosen). Why: "the same model" must mean the same thing for every rebuild and must be checkable from a tracked file.
- D50 (self-decided) Demo bridge constants, fixed before the bridge produced a number (found by the M0 ml-review). Matching distance: the scorecard's own, read from the method block of the scorecard file, never tuned. Interpolation: linear between broadcast samples at most 0.5 s apart; everything else is filled from the sensor data and counted. Scenes: chosen by a rule that reads no position error (the longest stretches in which every row has a synced moment, at most 4, each at most 12 s from its start, window order breaking ties); wherever the site shows a scene's error it shows the whole-game scorecard value beside it. Options: hand-picked scenes; a written rule (chosen). Why: the demo game is the held-out game, so nothing here may be chosen by looking at how the numbers come out.
- D51 (self-decided) Shot-making label, amending D47 before any player value was computed (found by the M0 ml-review). The location and contest baseline has a structured error: out of fold it under-predicts the easiest shots and over-predicts the middle band (the calibration table of reports/xfg.md), so rim-heavy shot diets read high. The player view therefore carries, as typed data read from reports/xfg.md, the baseline's name, the fact that it was not the selected model, its calibration error and its calibration rows, and for each player the part of his value that the baseline's own error explains for his mix of shots (his attempts in each calibration bin times that bin's observed minus predicted rate). The page shows both numbers. Options: drop the view; one caveat line; make the baseline's error explicit per player (chosen). Why: a caveat a reader cannot size is not a caveat.
- D52 (self-decided) Corrections to earlier self-decided lines (M0 review). D43: with r23 the near third is inside 0.5 ft on cle_gsw_2016 too (-0.12 ft); with r23_cam it is inside on all three games; the far third is outside for both builds on every game. D42: the cle_gsw near bias moves from -0.12 to +0.26 ft (the signs were dropped); the candidate build is shown on the court rows only (position, near-field bias, coverage, generalisation), because a replay keeps the earlier build's track linking, and its paired differences are against r23_replay, which equals r23 on those rows; r23's arena numbers come from the best of three training seeds of its recipe (ROADMAP_LOG.md line 129), a caveat the site's scorecard now carries. D48: the 10 cells are visited by result (made, then missed), then xFG fifth (1 to 5), with one generator and one draw per cell; a pick swapped by hand is marked as hand-picked in the data.
- D53 (self-decided) Per-prediction values are point estimates. An xFG for one shot, an EPV for one state and a ghost position have no interval of their own (the frozen lanes store none), so the site labels them point estimates and shows the lane's held-out metric with its interval beside them; this is how M5's "shot marker with xFG and interval" is met. Verdict sentences are copied word for word from the hand-written summary table of reports/results.md and shown as quotes from the write-up; the verdict class and every metric shown beside it come from the generated tables. Options: compute a per-shot interval (a new held-out computation, refused); label as point estimates (chosen).
- D54 (self-decided) Web toolchain as create-next-app 16.4.6 picks it, pinned exactly: next 16.4.6, react 19.3.0, TypeScript 5.9.3 (7.0 ships no JavaScript compiler API), ESLint 9.39.5 (10 does not support the Node 23 installed here), plus ajv, json-schema-to-typescript and @playwright/test 1.65.1 under D46. Plain CSS, no Tailwind. Fonts (Newsreader, Libre Franklin, IBM Plex Mono, OFL 1.1) self-hosted. Node here is an x86_64 build under Rosetta, so scripts start Python through arch -arm64. The site name "Pixels to Possessions" is a placeholder for Lucien. Why: the combination the framework's own template chooses, proven by a build.
- D55 (self-decided) Prototype numbers in README.md, PROJECT_SUMMARY.md and PIPELINE.md carry the tag [prototype] on the same line (in a table, the same cell); each labelled file has a dated note; tests/test_public_docs.py enforces it and that no tag is dropped. Options: rewrite the files; tag (chosen). Why: M8's grep check needs a detectable label, and a rewrite would change history the reader may compare against.
- D56 (self-decided) Two [ASK] items of D30 decided under D37: the CLAIMS.md caveat 4 amendment text (C13) is drafted, dated 2026-11-14, marked for Lucien's review and committed; docs/SCRIPT.md is approved for use as the cold-open copy (M4) after an independent fact check and edit, with its numbers as tokens resolved from web/data. Lucien may rewrite either; neither is published until Lucien merges. Options: wait for Lucien (blocks M4 and C13); decide and mark for review (chosen).
- D57 (self-decided) The Fable usage limit stopped every teammate at about 02:00 on 2026-11-14; the session continued on Opus 5.5. Teammate work committed before the stop was merged as is (all branches held their commits; site data and writing had already applied their review fixes). The ghosting rebuild was killed with its agent and was restarted detached. Reviews that did not run (bridge, models, web) run at their milestone close instead.
- D58 (self-decided) The ghosting item of the demo swap is how far the 1.0 s ghosts move when the model reads broadcast positions instead of sensor positions (mean distance between the two predicted ghosts per possession; the sensor side is the 0 ft reference). The first version measured each ghost against the real defender at the predicted moment, which is the lane's held-out error computed again (found by the M1/M2 ml-review); a test now holds that the real position there is never read. The per-possession matched distance in swap.json is a description of the broadcast positions, read under D50 like the scene errors and never used to choose anything. A memory-projection test in tools/ was relaxed by its author after the real EPV run (the gate now reads the projected footprint peak); it is a safety gate, not a correctness check, and the live memory watch still stops a run.
- D59 (self-decided) Web review fixes (M3 to M7, M10): the forbidden-claim patterns now run over every content file; presenter notes of the talk slides are left out of that one rule because they repeat docs/SCRIPT.md word for word and quote phrases such as "never passes" in the negative. Readouts show stored values only (the EPV readout shows the stored state at or before the time). Headline numbers of the demo are rounded once, in the export, to three decimals; the site never rounds. Options for the notes: edit SCRIPT.md notes; exempt notes from that rule (chosen).
Decided by Lucien 2026-11-14 (settling the items D37 reserved):
- D60 (Lucien) The public site carries no SportVU frame data (option b). It keeps the cold open with stills, results, scorecard, prep tab and per-possession values (xFG, EPV curve, outcome); court playback and any per-frame positions are private only. Reason: the source is marked not for redistribution.
- D61 (Lucien) CLAIMS.md caveat 4 keeps the drafted amendment and adds: "The public site carries no SportVU frame data; player positions appear only on the private copy." docs/SCRIPT.md and the name "Pixels to Possessions" are accepted as they are for now.
- D62 (Lucien) Merge v3-finish into main with a merge commit (no squash) and set the repository description.
- D63 (Lucien) Public deploy: build on this Mac and publish the built site to a gh-pages branch served by GitHub Pages.
- D64 (Lucien) Private site: Cloudflare Pages behind Cloudflare Access on Lucien's account; the only allowed email is Lucien's Colab and Drive account address, signing in with a one-time code.
- D65 (Lucien) D60 to D64 lift D37's reservation for these steps only.
- D66 (self-decided) How D60 was carried out. export/public_data.py writes the public data root from the full one (per-frame arrays emptied, ghosts null, the 780 shot locations measured by tracking removed, demo truth points removed; everything else byte for byte) and the public build runs it itself. The public possession page shows an empty half court with the shot chart mark and the access note in place of playback; the cold open's truth and swap scenes show the still, because both drew sensor positions. The per-possession mean ghost deviation is not shown publicly (ghosts are null there). A frames step in npm run check scans every file of the public out/ for tracks. Options: a second export path; a transform of the full root (chosen, one source of truth).
- D67 (self-decided) How D64 was carried out. The free Cloudflare Pages plan takes 20,000 files per site; the private build has 21,233. web/scripts/package-cloudflare.mjs copies out/ without __next._full.txt on the 3,162 possession routes, the one payload no client navigation fetches in Chromium or WebKit (18,071 files; the private browser suite passes on the packaged folder). Options: drop all four payloads per possession route (failed, 404s in the tour); keep index.txt and the tree (failed on WebKit); drop only _full (chosen); a paid plan (an account decision). The project was created on classic Pages (wrangler --force) because wrangler 4.162 otherwise routes Pages to Workers, and D64 names Pages. Order: the project was created empty, a one-line placeholder with no data was deployed to confirm Access answered on the production and preview hostnames, and the private build was deployed only after that, so frame data was never served without sign-in. The placeholder preview (preview-check) stays behind Access; it holds no data.
- D68 (Lucien) The site is adequate as deployed, with no adjustment list. Link the public site from the README and the repository's website field, and tag v3-site-1 now on the deployed state and push it. C12 and C14 are run later and recorded as evidence; changes they lead to go to a later tag. Reason: the tag names what is deployed before anything else changes.
- D69 (Lucien) C14, the two-minute test with a real person, is skipped for now; Lucien will raise it when ready. M11 closes without it. Reason: Lucien's call, to move on to the next work.
- D70 (Lucien) P2 is widened: the research session turns papers into tested results, not notes. For every lane in reports/results.md it writes a dossier and pre-registered experiment proposals. Experiments are run as their own milestones under the existing rules (training games only, ml-reviewer, verifier, frozen code untouched). Reason: notes alone show nothing; a tested result with a verdict shows expert practice.
- D71 (Lucien) A new untouched test set for all post-V3 experiments. The 17 held-out games are spent and are never scored again. The new set is fixed, hashed and committed before any experiment runs, and scored once per experiment. Reason: any new method scored on the spent held-out games would be contaminated.
- D72 (Lucien) A "Methods and literature" page on the site and a matching README section: per lane, the foundations it follows, our number next to the published one, and every experiment tried with its verdict, negative ones included. Reason: visible proof that the work builds on the literature and keeps up with it.
- D73 (Lucien) Work happens on a new branch v4-research from v3-finish. No push, publish, deploy, tag or scheduled job without Lucien's yes. Under D37, Claude Code decides everything else itself and logs it as "(self-decided)" with the options weighed. Reason: research work stays separate from the deployed V3 until Lucien approves it.
- D74 (self-decided) Order of the D70 to D73 work. A data survey (docs/research/DATA.md) runs first, before the D71 test set is fixed: the set prefers truly new data (another season, provider or release, verified at the source) and carves games out of the 2015-16 training games only where nothing new serves a lane. The dossiers run in parallel with the survey (reading papers runs no experiment). Options: fix the set from the 2015-16 release first (the first attempt found every training game used by the frozen models and the 5 leftover games empty, and its claim that no other tracking source exists was not checked, so it is withdrawn); survey first (chosen). Agent team as in D39: one data surveyor and five dossier writers, docs only; the lead fixes the set, runs the pilot and owns PLAN.md.
- D75 (Lucien) Work on every other parked item in parallel with P2, in an order that builds each on the strongest foundation. Reason: Lucien wants as much finished as possible, without weakening what each item rests on.
- D76 (self-decided) The order of the parked items under D75. P3 starts now on branch p3-genai, options 1 (scouting reports on the prep tab) and 2 (possession explanations) only, generated at build time with a check that every number in the text matches web/data; options 3 (search, needs a live server) and 4 (jersey reading, a tracking change) stay with P8. The generator runs on a small sample to prove it; the full run, which spends money on Lucien's API key and sends derived numbers to the model provider, waits for his yes. P4 builds its harness now on branch p4-polish (machine-check scorer, loop driver, the checks read-only to the loop) and runs its loop only after P3 and the methods page are merged, so it polishes the site that ships. P8 Step 0 (the feasibility study, no training) starts when the tracking dossier exists, because it builds on it; P5 does not start on its own, since P8 supersedes it if chosen and Step 0 ends with the go or no-go that decides between them. P6 gets a one-page brief of its open questions; the conversation itself is after the interview. P7 needs the interview date from Lucien. Options: everything at once (P4 would polish a site about to change, and P8 would plan without the dossier); strictly one at a time (slow); this staged order (chosen). One heavy job at a time holds for every branch.
- D77 (self-decided) The D71 test set, fixed and hashed in reports/research/testset.json before any experiment result existed. Set A, the scoring set of every tracking lane: the 10 games of 2015-12-05 in the public SportVU release, which the frozen models never read (the upload stored them under names with a stray folder prefix, so the frozen parser skipped them; docs/research/DATA.md), with their archive blob hashes. Set B, an out-of-distribution check reported beside each verdict and never the adoption test: the 10 Liga ACB 2025-26 games of SkillCorner's open data (MIT License file). The play-by-play lanes use the 2025-26 NBA season, hashed into the file when its source is settled. Development games are every 2015-16 training game. The 17 held-out games stay spent. tools/research_testset.py is the only door: dev_games, refuse_test, and score_once, which allows one ledger entry per experiment; tests/test_research_testset.py fails if other code opens the file, names a test game, or calls score_once twice or in a loop. Options: a 100-game carve-out from the training games, stratified by team and date, with every model refit without it (power to detect about 0.003 in xFG log loss between close models, but the frozen models and V3's design choices did use those games, so it is not untouched; dropped); the untouched sets (chosen; 10 games, about 1,700 shots, so intervals are wider, which every verdict states); waiting for new tracking (none public after 2015-16 for the NBA). The 5 broken archives are empty in every copy found.
- D78 (Lucien) Proceed with basketball-reference as V3 did: fetch the shot charts and play-by-play of the test games (and later the 2025-26 season under the site's rate limit), and let P3 make its sample model calls with numbers derived from them. Reason: Lucien accepts the Sports Reference terms risk as V3 did; the survey's reading of the terms (docs/research/DATA.md) stays on record.
- D79 (self-decided) Development data of the play-by-play lanes: 2020-21 to 2023-24 only. 2024-25 was their held-out season, scored once at the freeze, so under rule 4 it is never fitted, tuned or scored again; the 2025-26 season is their D71 test set. Options: let 2024-25 join development (more data, but it was scored once); keep it out (chosen).
- D80 (self-decided) xfg-E1 commitments fixed before its single test score, after the ml-review of the pilot (PASS, no blocker, four majors). (1) Power: Set A has 10 games; scaled from the development interval, the 95 percent half-width on the test set is about 0.002, so a pass needs a clear effect and a fail whose interval straddles the threshold is read as "inconclusive at this sample size", not as evidence that CVAP is worse. The verdict stays the pre-registered one (upper bound at most +0.0005 on development and on the test set). A 10-game percentile bootstrap under-covers, so a t interval on the 10 per-game mean differences is reported beside it as a sensitivity check; if the two disagree, the report says so. (2) Every per-arm metric gets a 95 percent interval from a 2000-resample bootstrap by game, so nothing is read from the test games a second time. (3) The interval of a shot is GM(p1) - GM(p0) over the 3 inner folds (the wider of the natural choices); the site rule (p95 under 0.01 means "point estimate") must hold on the development games and on the test set; the per-fold p95 is reported too. (4) Set B is not scored for xfg-E1: the ACB data has no path to the frozen features (shooter prior ids, FIBA court, contest features would need a new adapter), and the report says so. The ledger is keyed by experiment and source, writes a started entry before scoring, and only grows (tested). (5) The ingest reports row counts only, never a make count of a test game. (6) Timing: the dossier's pre-registration file time (13:08) precedes the development result (13:10); the code changed after that result in reporting only, and the development stage is rerun on the committed code to show its numbers unchanged. Options: score now as written; settle the review's points first (chosen).
- D81 (self-decided) After the post-score ml-review of xfg-E1 (PASS, two majors). (1) The frozen tools the ingest runs wrote outcome aggregates of the Set A games under the research data folder (a made count per game in the collector manifest, possession end kinds and final scores in the phase 0 manifest, and its reports); they were never printed or opened, and xfg-E1's code was fixed before that data existed, so the pilot is unaffected. They are sealed: tools/research_ingest.py SEALED_FILES and ingest.json list them, and no research tool may open them; only a score_once scorer reads a test outcome. Options: delete them (the frozen loader may need them; nothing is deleted outside git); strip them (edits frozen outputs); seal and record (chosen). (2) The ledger test now walks every committed version of reports/research/scoring_ledger.json: each is a prefix of the next and a scored entry never changes. The pins hash is required. (3) The pilot report states the control arm as the frozen recipe refit, not a new score of the frozen lane, and gives every value its interval.
- D82 (self-decided) Merging the methods page (D72) and P3 (D76), which each raised the contract to 1.2.0 for a different addition. Neither was deployed, so they become one 1.2.0: methods.json, genai/<kind>.json and the manifest carry 1.2.0 only; every other file may carry 1.1.0 or 1.2.0. A data root is therefore a 1.2.0 root (methods.json is required). The merge runs on branch v4-integrate, its checks before v4-research takes it; the review findings on the merge are fixed by the teammate who owns each file. Options: 1.3.0 for the second addition (a version number for a state that never shipped); one 1.2.0 (chosen).
- D83 (Lucien, 2026-11-17) Yes to all of the asks after R4: push v4-research and v3-finish; deploy the methods page and P3's texts to both sites; P3's full run (about $3.38, sends each page's numbers and player names to the model provider); the 2025-26 basketball-reference fetch (about 2,500 pages); the monthly research refresh job; the P4 loop; P8 re-scoped as in docs/research/P8_FEASIBILITY.md (two phases, 25 to 40 GPU-hours, 10 more third-party uploads under CLAIMS.md caveat 4); the next experiments decisions-E1, winprob-E1, lineups-E1.
- D84 (self-decided) How D83 is carried out. Order: push first; then in parallel the 2025-26 fetch (network only, outside the lock), P3's full run (model calls over the network, a few MB of memory, outside the lock; every text still passes every check), decisions-E1 (needs no new data) and P8 Phase A (measurement only, no training); the P4 loop runs after the full texts are in, so it polishes the site that ships; the deploy follows the full checks on the merged result, and again if the loop adopts a candidate. winprob-E1 and lineups-E1 start when the 2025-26 season is hashed into testset.json. Kept out of the yes: basketball-reference content as training labels for P8 (its terms forbid training AI models, P8_FEASIBILITY.md item 6) stays unused until Lucien decides it by name; P8 Phase B GPU sessions run in Lucien's Colab and wait for him. Options: everything in sequence (a day of lock time); parallel where nothing is shared (chosen).
- D85 (self-decided, under D83) P8's prerequisites (docs/research/P8_FEASIBILITY.md section 7). (1) The CV test set is whole arenas, fixed by a rule that reads no result: the channel games dated without a scorebug check, Miami (Pelicans at Heat, Thunder at Heat), Toronto (Spurs at Raptors) and Staples Center (Clippers at Lakers), 4 games in 3 arenas; video ids and SportVU archive blob hashes go into reports/research/testset.json under one more guarded key before any experiment, each game scored once per experiment. (2) Done means the section 6.2 rows meet their targets and every Stage 1 row of 6.1 improves under the adoption rule (D5) on that set; every Stage 1 row passing is the stretch, not the bar. Options: every row passes (the study finds it likely unreachable); targets plus improvement (chosen). (3) Development: the 4 R1.3 games plus the confirmed new games of other arenas, game bootstrap with the window bootstrap beside it, 3 seeds for any trained model. (4) Phase A camera truth runs on the Mac under the lock, overnight after the deploy and the P4 loop, so no Colab session is needed for Phase A; Phase B GPU runs use Lucien's Colab. (5) Labels follow D84: no basketball-reference content as training labels.
- D86 (self-decided) decisions-E1 after its reviews (ml-reviewer PASS, /code-review). (1) Its pre-registration ties H1 to the per-game Spearman but defines metric (1) as the odd/even half-season correlation. The verdict is read as the metric was written (r 0.420, H1 fails), and the report puts beside it, labelled post hoc, that stepped down to one game the target's reliability is about 0.04. Options: reread H1 at the per-game unit after seeing the number (a change after the result); read as written and show the gap (chosen). (2) The game-bootstrap percentile ranges of the reliability statistics are kept as pre-registered and labelled as not valid intervals; a valid interval is added post hoc and reads no verdict. (3) Set A is not scored: its 10 games share one date, so no player has two halves and none reaches 300 decisions in a game; the scorer refuses before the ledger is touched, so the one entry is kept.
- D87 (self-decided) P8 Phase A after its reviews (ml-reviewer PASS, /code-review). (1) The overnight CV runner (tools/p8_night.py) runs the frozen detection and OCR steps with the anaconda Python (V3_CV_PYTHON), the only one holding ultralytics and easyocr, which CLAUDE.md keeps out of the system Python; pytest and every other command stay on /usr/local/bin/python3. This is the same split V3's CV work used. (2) Before the first overnight run: the CV test-set door matches substrings (ids, video ids, paths, stems, arena tags), the guards scan every tracked script, testset.json's CV source cannot change after 1670e3d6, the ball tolerance is written once from development games before any ball model, the pin covers the truth-building code, and r23 is flagged in-sample on the 4 R1.3 games with its rows reported apart from the new arenas. Options: run now and fix later (results would rest on weak doors); fix first (chosen).
- D88 (self-decided) Scorecard v2 is adopted (reports/research/p8_scorecard_pin.json status adopted, this commit), after the P8 Phase A reviews and fixes (0533e25d). From here the scorer, its constants, the truth-building code and the ball tolerance file change only with a PLAN.md line. A future CV test scorer reads a test video only through tools/p8_fetch.cv_test_video inside its score_once scorer. The overnight runner scores the old truth (b4) on the gate games gsw_bkn_2015 and nyk_bos_2016; plan 36.9 lock hours, run in 8-hour nights after the deploy and the P4 loop.
- D89 (self-decided) The 2025-26 test season (reports/research/testset.json pbp_2025_26, branch r-pbp2526) and the winprob-E1 gap season, after the ml-reviewer (merge after fixes). (1) Selection: the 1,316 games the frozen V3 collectors select with their defaults (1,230 regular season, 85 playoff, the NBA Cup final; the 6 Play-In games left out), the same rule that built the development seasons 2020-21 to 2023-24, against the old pending text "regular season" only. The rule reads no outcome and no result exists yet. Neither pre-registration says regular season; lineups-E1 scores "all test games". Options: regular season only (would differ from the development seasons); V3's selection (chosen). (2) winprob-E1 Elo state: 2024-25 results are read as time-forward Elo state only, game by game, with every Elo parameter chosen on 2020-21 to 2023-24, the same footing the pre-registration gives the test season's own results; nothing is fitted, tuned or scored on 2024-25 and it stays out of development (D79). It is read through one door function that returns ids, dates, teams and final scores of 2024-25 and is called only by the Elo state builder. Options: skip 2024-25 and regress end-of-2023-24 Elo across two off-seasons (a stale prior that would weaken the test); read 2024-25 as state (chosen). Fixed before any winprob-E1 code is written.
- Status 2026-11-17: the 2025-26 season is fixed as test source pbp_2025_26 (1,316 games, 3 file pins; merge b58062ba, D89). lineups-E1 scored once on it and passes as pre-registered, as a game-margin claim given each game's stints (RAPM minus lineup plus-minus RMSE -1.15 [-1.59, -0.77]; RAPM R squared against home court 0.098 [0.065, 0.132]; reports/research/lineups-E1.md); lineup ranks stay inconclusive; a possible data fault is reported beside it (1298 of 1300 games agree with the schedule). ml-reviewer PASS before and after the score; merge 6e1b2ea4. winprob-E1 is in development.
- D90 (self-decided) P3 scouting texts after two stops on the 20% rule. Cause of the first: claude-haiku-5-5 thinks by default, and on the scouting-v3 prompt the thinking used the whole 4,000-token budget before any text (14 of 16 drops empty). Thinking off ended the truncations but broke the rules more often (21 of 50 dropped: grade, numbers not on the page, signs written as words). Chosen: thinking on with a set budget and room left for the text, prompt and every check unchanged, the setting recorded with each text, the dropped items asked once more under the new setting, the same 20% stop and $6 cap. Options: a scouting-v4 prompt (a new prompt version after seeing failures, slower to prove); a larger model for scouting (about $2, near the cap); stop at 107 of 405 players (most prep pages without a text). Prompt caching of the fixed instruction block is not switched on: it would save about $0.27 a possession run. The possession texts (2,928) stay as written: they passed every check.
- D91 (self-decided) winprob-E1 bucket guard, fixed before its one score after the pre-score ml-review. The pre-registration says "no bucket gets worse beyond its interval"; the code checked the five regulation buckets only, following the frozen selection metric, which reports overtime but does not select on it. Read as written, as D86 did for decisions-E1: the guard covers every bucket, overtime included; the regulation-only reading is printed beside the verdict with a note on whether it would differ. Options: regulation only (a reading the pre-registration does not contain); as written (chosen).

## Open questions
- None.
