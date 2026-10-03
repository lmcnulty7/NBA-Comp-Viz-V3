# ROADMAP_LOG

Pass-by-pass record for ROADMAP.md (newest at the bottom; summary at the top on stop).
Phase A and B history is in FIX_LOG.md.

## STOP (2026-10-03, pass 3): [HUMAN] gate reached (Lucien confirms the Stage 1 targets)
- Done: R0.1 (tag v3-frozen on main 869f7a0, local only; README status section), R0.2 (CLAIMS.md "Stage 1 definition of done"), R0.3 (`python -m sportvu.bench`, V3 scorecard). Blocked: none.
- Waiting on Lucien: confirm or change the targets in CLAIMS.md "Stage 1 definition of done" after reading reports/scorecard/gsw_phx_2016__v3.txt. Decide with it: (a) the 3 px court line target sits below the truth fit's own 6.0 px median residual, so the current truth H cannot confirm it (loosen it, or accept it is only checkable once R1.3 gives a better truth); (b) the held-out era game (gsw_nyk_curry54, 2013) has no public SportVU log, so its rows need another truth or drop out of the generalisation row; (c) whether to push the v3-frozen tag to origin.
- Current scorecard (V3 on gsw_phx_2016, the 15% testable frames only, optimistic): court line 0.0% of frames <= 3 px (p50 26.7 px, p95 57.0 px); position error p50 7.33 ft, p90 15.35 ft; near-field bias near -1.02 ft, far +1.67 ft; coverage 98.6% of 850 live wide s (V3 never rejects); missed 25.1%; ghosts 15.1% non-referee; team labels 81.8%; 21.7 id switches per possession; jersey-read rate and generalisation not measurable.

2026-10-03: ROADMAP.md written; replaces FIX_PLAN Phases C and D. First item: R0.1.

## Pass 1 (2026-10-03): R0.1 freeze V3
- Annotated tag `v3-frozen` on main HEAD 869f7a0 (the commit track-fix branched from); local only, not pushed to origin (main is 21 commits ahead of origin/main as well).
- README.md: new "Status" section after the intro: what V3 is, its SportVU numbers with artifact paths and caveats (reports/sportvu_phx.txt, CLAIMS A8 proposed), what is frozen, what the rebuild reuses, and a table of where each scoreboard step lives (sportvu/*, outputs, R0.3 bench marked not built). No code change; pytest 52 passed.

## Pass 2 (2026-10-03): R0.2 Stage 1 definition of done in CLAIMS.md
- CLAIMS.md: new section before Tier A with the ROADMAP targets table (status PROPOSED until the gate after R0.3), a V3 baseline column from reports/sportvu_phx.* with its caveat line (testable 15% only, optimistic), the measurement protocol (held-out game gsw_phx_2016, held-out arena one of CLE/OKC/NYK named after R1.1, held-out era gsw_nyk_curry54) and the single scorecard path reports/scorecard/<game>__<build>.{json,txt} from `python -m sportvu.bench` (R0.3 must write there).
- Open question raised for Lucien: the public SportVU logs cover 2015-10-27..2016-01-23 only, so the held-out era game (2013) has no tracking truth; how its rows are measured is undecided. pytest 52 passed.

## Pass 3 (2026-10-03): R0.3 scorecard CLI
- sportvu/bench.py + tests/test_bench.py (7 tests): `python -m sportvu.bench data/sportvu/build --name v3` -> reports/scorecard/gsw_phx_2016__v3.{json,txt}. The truth H is each testable frame's camera and every build's feet are re-matched to SportVU through it (5 ft), so later builds are scored on the same 1,447 frames; build-independent references cached in data/sportvu/bench/ (gate v2 wide scores on 8,824 live frames, referee-class detections on testable frames); untestable frames by cause in the artifact. Reproduces B5: position p50 7.33 / p90 15.35 ft (B5 7.31 / 15.38), near -1.02 / far +1.67 ft.
- New rows for V3: court line 0.0% <= 3 px (p50 26.7 px); coverage 98.6%; ghosts 15.1% without referees (only 280 of 12,637 boxes overlap a referee detection and 208 of those are within 3 ft of a player, so referees are not the main ghost source); 21.7 id switches per possession over 56 shot-clock possessions (24.9 at a 5 ft match gate; fragment linking moves 22.1 to 21.7); jersey and generalisation marked not measurable with reasons. CLAIMS.md Stage 1 V3 column and README table now read from the scorecard.
- Finding for the gate: the truth fit residual is 6.0 px median (p90 10.8), above the 3 px line target. The line field was tried as a floor and does not resolve a few px on 720p (ridge support within 2 px: 0.047 for H_truth vs 0.045 for H_truth shifted 6 px). pytest 59 passed.
