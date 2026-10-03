# ROADMAP_LOG

Pass-by-pass record for ROADMAP.md (newest at the bottom; summary at the top on stop).
Phase A and B history is in FIX_LOG.md.

2026-10-03: ROADMAP.md written; replaces FIX_PLAN Phases C and D. First item: R0.1.

## Pass 1 (2026-10-03): R0.1 freeze V3
- Annotated tag `v3-frozen` on main HEAD 869f7a0 (the commit track-fix branched from); local only, not pushed to origin (main is 21 commits ahead of origin/main as well).
- README.md: new "Status" section after the intro: what V3 is, its SportVU numbers with artifact paths and caveats (reports/sportvu_phx.txt, CLAIMS A8 proposed), what is frozen, what the rebuild reuses, and a table of where each scoreboard step lives (sportvu/*, outputs, R0.3 bench marked not built). No code change; pytest 52 passed.

## Pass 2 (2026-10-03): R0.2 Stage 1 definition of done in CLAIMS.md
- CLAIMS.md: new section before Tier A with the ROADMAP targets table (status PROPOSED until the gate after R0.3), a V3 baseline column from reports/sportvu_phx.* with its caveat line (testable 15% only, optimistic), the measurement protocol (held-out game gsw_phx_2016, held-out arena one of CLE/OKC/NYK named after R1.1, held-out era gsw_nyk_curry54) and the single scorecard path reports/scorecard/<game>__<build>.{json,txt} from `python -m sportvu.bench` (R0.3 must write there).
- Open question raised for Lucien: the public SportVU logs cover 2015-10-27..2016-01-23 only, so the held-out era game (2013) has no tracking truth; how its rows are measured is undecided. pytest 52 passed.
