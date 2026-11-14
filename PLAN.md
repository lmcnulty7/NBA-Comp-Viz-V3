# PLAN.md: finishing V3

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

M2 Demo bridge (Python)
- export/demo_bridge.py on the 25 gsw_phx_2016 windows with the D20 build: match CV boxes to SportVU per frame,
  fill missed players, drop ghosts, record fill share (D27).
- Frozen predict only for xFG (shots in the windows), EPV (states) and ghosting, on CV and on SportVU positions.
- demo/swap.json (per item deltas, mean absolute difference, n, fill share); demo/frames.json (boxes, court
  positions and truth per frame, synced to clip time).
- Clips (media-private/, default cap 15 s each, 60 s total, adjustable at M11) and one still per scene (public).
DoD: files validate; n and fill share stated in the data; tests on fixture windows; no fit call; clip lengths
within cap; scorecard.json names the D20 build.

M3 Web foundation
- web/ scaffold: Next.js static export, TypeScript strict, ESLint, Node pinned (.nvmrc), npm, lockfile.
- lib/types generated; lib/data/source.ts with StaticSource; design tokens (offense, defense, ghost, truth, ball
  colours; type scale; dark mode); layout and nav (Home, Review, Prep, Results).
- components/court canvas renderer; VARIANT=public|private; media-private copy step for private only.
- npm run check: typecheck, lint, schema validation, no numeric literals in copy, no video in public out/,
  Playwright at 390 and 1280 px.
DoD: both variants build to out/; check passes; the court draws a real held-out possession matching a Python
reference frame (snapshot test); out/ served under the Pages base path works locally.

M4 Home and cold open
- [ASK] Rewrite SCRIPT.md for V3 (D25 scene list); Lucien approves before copy goes in.
- Headline (D17); scroll steps: clip (private) or still (public), boxes, court map, truth overlay with the error in
  feet; then the swap comparison labelled an illustration (C6, D16); entry points to Review, Prep, Results.
- Public variant: still plus request-access note (D21).
DoD: C6 met; reading grade at most 9 and jargon check pass on the copy; works at 390 and 1280; reduced-motion
fallback; keyboard reachable; no em dashes.

M5 Possession review (A)
- Game picker; possession list with filters (result, xFG bucket, featured); possession page with court playback
  (play, pause, scrub, speed), EPV curve synced to time, shot marker with xFG and interval, ghost defenders toggle
  labelled "average defense", evidence toggle with sources.
- Guided tour across the featured possessions.
DoD: C8 and C10 met; a script loads every possession's data without error; playback at least 30 fps on a laptop
(measured); tour completes on phone and laptop.

M6 Prep tab (B)
- Player index (search, sort); player page: shot-making over expected, gravity, rebounding, absence risk, similar
  players; minimum-sample note.
DoD: C9 met; every metric shows an interval or says it is a point estimate; hidden players counted and stated.

M7 Results page
- Tracking scorecard vs Stage 1 targets (C5) and model results (C7), failed models included, links to the reports.
DoD: C5 and C7 met; values equal the source files (test).

M8 Repo and docs
- README rewritten (D17 headline, 2-minute top section, links, honest scorecard); PROJECT_SUMMARY and PIPELINE
  label prototype numbers; repo description set.
- [ASK] CLAIMS caveat 4 amendment text (C13).
- [ASK] Merge v3-finish into main so the default branch shows current work.
DoD: C3 and C13 met; no unlabelled prototype number in a public doc (grep check).

M9 Deploy
- [ASK] GitHub Actions deploy of the public variant to Pages.
- [ASK] Cloudflare Pages + Access for the private variant after the free tier is verified; Lucien adds viewers.
DoD: public URL returns 200 with no video files; private URL requires sign-in; first view under 3 s on both (C12).

M10 Slides (C4)
- /talk route in the private site (D35), built from the approved SCRIPT.md; slide copy in content/talk/*.md;
  arrow-key navigation, presenter notes view, at least one slide with a live court playback.
DoD: C4 met; every number traces to web/data or reports/; /talk absent from the public build (check); runs
offline from out/ on the private variant.

M11 Freeze, polish, acceptance
- Feature freeze; Lucien's adjustment list worked through; C14 2-minute test with a real person; final check run.
- [ASK] Tag v3-site-1 and push.
DoD: C1 to C14 each checked with its evidence named here.

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

## Open questions
- None.
