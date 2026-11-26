# P8 ghost source after a reset (registered 2026-11-26, before any number)

Status: registered before any number; stored files only, no model runs, no lock, 0 units. Code: tools/p8_ghostsrc.py
(run), tests/test_p8_ghostsrc.py. Record: reports/research/p8_ghostsrc.json.

## Why

The reset-fix forecasts (P8_resetfix.md sections 12 to 19) show that the r23 boxes the bug drops after a reset are
often ghosts: of the excess r23 boxes after a first reset, about 0.18, 0.65 and 0.19 lie more than 3 ft from every
SportVU player (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015). On gsw_cha_2016 that makes the fix's net error worse in
about 91 percent of forecast draws, and the lead kept the per-game condition that this fails. This study asks what
those ghost boxes are, so the next arm can target them.

## Population

On the scored frames of the registered score files (the MPS-truth universe, tools.p8_footbias.load_frames, its
counts checked against the score files), the r23 corpus player boxes at 0.40 ("ours"), on frames from the r23 build's
first reset-like frame on (tools.p8_resetfix.first_reset_index), whose bottom centre through H_truth lies more than
3.0 ft from all 10 SportVU players. This is the population the forecasts used. Comparison populations, report only:
the same ghosts before the first reset, and the sidecar's non-referee ghosts after it. Games: gsw_cha_2016 first, then
gsw_bkn_2015 and gsw_sac_2015 (hou_cle_2016 has no corpus boxes).

## Classes (fixed now)

Each ghost box gets every flag; its class is the first that applies, in the lead's order:

| class | rule | what it would lead to |
|---|---|---|
| A, not a standard live view | the frame's truth camera_f in the game's top decile of scored frames (a1, zoom); or its gate v2 wide score in the game's bottom decile of scored frames (a2, marginal wide); or a cut (scorecard cache corr below the frozen cut_hist_corr 0.6) within the 1.0 s before it (a3) | a frame filter arm, its own registered arm |
| B, off court under the build's H | the box's bottom centre through the sidecar row's H (H_truth when the row has no accepted H, counted) lies more than 2 ft outside the 94 by 50 ft court; near (within 8 ft: sideline, bench) or far (crowd) | a court-region filter arm (drop boxes whose foot is off court under the build's H), its own registered arm, possibly combined with the reset fix |
| C, referee-like | an r23 corpus referee box (score 0.25 or more) or a scorecard cache referee detection overlaps it at IoU 0.3 or more | a referee-handling arm (the class or the scorer's IoU rule) |
| D, on court, unmatched | none of the above; near (3 to 6 ft from the nearest player) or far (more than 6 ft) | no filter arm without a further check: near ones bear on foot placement, far ones on on-court false positives or truth and sync error |

Note on A: every scored frame has accepted camera truth and passed gate v2's wide threshold, and the time maps
cover running clock only, so dead-ball replays are mostly outside this population already; A looks for the views
that remain (zoomed, marginal, just after a cut). Since frames after a reset follow a camera cut by construction, a3
will be common after a reset; its sub-flag is reported apart.

## Reported

Per game (gsw_cha_2016 first) and pooled: the number of after-reset ghosts; counts and shares by class and sub-class;
each flag's count regardless of order; the same for the two comparison populations; the collateral of each class's
filter: r23 boxes after a reset that are not ghosts but carry the class's flag (a filter on that flag would drop
them too); the thresholds used (camera_f top decile, wide bottom decile).

## Reading (fixed now)

A class holding at least 0.30 of the after-reset ghosts on gsw_cha_2016, or pooled, is named as the next arm with
what it leads to (the table above), together with its collateral. No class is acted on here; any arm is its own
registered design. If no class reaches 0.30, the study says the ghosts are spread and names the largest class.
