# ROADMAP_LOG

Pass-by-pass record for ROADMAP.md (newest at the bottom; summary at the top on stop).
Phase A and B history is in FIX_LOG.md.

2026-10-03: ROADMAP.md written; replaces FIX_PLAN Phases C and D. First item: R0.1.

## Pass 1 (2026-10-03): R0.1 freeze V3
- Annotated tag `v3-frozen` on main HEAD 869f7a0 (the commit track-fix branched from); local only, not pushed to origin (main is 21 commits ahead of origin/main as well).
- README.md: new "Status" section after the intro: what V3 is, its SportVU numbers with artifact paths and caveats (reports/sportvu_phx.txt, CLAIMS A8 proposed), what is frozen, what the rebuild reuses, and a table of where each scoreboard step lives (sportvu/*, outputs, R0.3 bench marked not built). No code change; pytest 52 passed.
