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

## Result (2026-11-26, stored files, no lock, 11 s)

Thresholds: camera_f top decile 3,013, 3,326 and 3,616 (gsw_cha_2016, gsw_bkn_2015, gsw_sac_2015); wide score bottom
decile 0.947, 0.932, 0.928. SportVU positions on the scored frames (in the record's meta since the rerun of
2026-11-26, report only; every other number unchanged): x -4.1 to 94.2 ft and y 0.5 to 51.4 ft (gsw_cha_2016), x -0.9
to 99.3 and y -2.1 to 51.6 (gsw_bkn_2015), x -2.4 to 97.1 and y 1.1 to 50.1 (gsw_sac_2015), so the court rule's
frame (0 to 94 by 0 to 50 ft) is right. No after-reset ghost needed H_truth for class B (every such row had an accepted H).

After-reset r23 ghosts (the forecasts' population), by class:

| game | ghosts | A, not a standard live view | B, off court | C, referee-like | D near (3 to 6 ft) | D far (over 6 ft) |
|---|---|---|---|---|---|---|
| gsw_cha_2016 | 73 | 28 (0.38; all a3, within 1 s after a cut) | 0 | 7 (0.10) | 29 (0.40) | 9 (0.12) |
| gsw_bkn_2015 | 162 | 20 (0.12; a1 18, a2 19 by flag) | 0 | 11 (0.07) | 109 (0.67) | 22 (0.14) |
| gsw_sac_2015 | 62 | 21 (0.34; all a2, marginal wide) | 0 | 4 (0.06) | 34 (0.55) | 3 (0.05) |
| pooled | 297 | 69 (0.23) | 0 | 22 (0.07) | 172 (0.58) | 34 (0.11) |

Comparison (report only): the same r23 ghosts before the first reset (9,111 pooled) split A 0.23, B under 0.01, C
0.09, D near 0.55, D far 0.13; the sidecar's non-referee ghosts after a reset (80 pooled) split A 0.14, D near 0.78,
D far 0.09. This comparison takes the sidecar boxes' raw bbox bottom centre, like the r23 boxes, not the build's
foot_stab, which is why it counts 80 sidecar ghosts where the scorer (foot_stab; the pairing check's table) counts 71. Collateral after a reset (r23 boxes within 3 ft of a player that carry the flag): A 10 of 81 on
gsw_cha_2016, 133 of 975 on gsw_bkn_2015, 85 of 361 on gsw_sac_2015; B none; C 0, 17, 6.

**Reading by the registered rule.** Two classes reach 0.30: A on gsw_cha_2016 (0.38) and D (0.52 on gsw_cha_2016,
0.69 pooled).
- A leads to a frame filter arm. On gsw_cha_2016 it is entirely a3: boxes in the first second after a camera cut. A
  filter on a3 would have dropped 28 of its 73 after-reset ghosts and 10 of its 81 good boxes there. On the other two
  games A is zoom (gsw_bkn_2015) and marginal wide (gsw_sac_2015), not cuts, so one frame rule does not cover all three.
- D leads to no filter arm without a further check. Most of it is D near: boxes 3 to 6 ft from the nearest player.
  Post hoc, not registered: those are likely real players whose box foot lies 3 to 6 ft off: the scorer then counts the
  player as missed (no foot within 2.5 ft) and the box as a ghost, so the net-error row counts one placement error
  twice. The further check is to measure that pairing directly; it adds two conditions to the D-near count, a missed
  player and a one-to-one assignment within 6 ft (the pairing check below).
- B is empty: none of these ghosts stands off court, so a court-region filter would not help. C is small (0.07).

**Limits.** The after-reset populations are small (73, 162 and 62 boxes, a handful of windows per game), so the
shares are rough. The corpus boxes use the raw bottom centre, not the build's stabilised foot, which can push a real
player's box past 3 ft more often than the build would. After-reset ghosts look much like ordinary r23 ghosts (the
before-reset split is close), apart from gsw_cha_2016's cut-adjacent share. Post hoc and unregistered: the D near
reading (a placement error counted twice) is an explanation, not a measurement.

## Pairing check (registered 2026-11-26, before any number; the lead's next item)

Question: how much of gsw_cha_2016's forecast rise in net error is one real player counted twice, missed (no foot
within 2.5 ft) and its own box counted a ghost (more than 3 ft from every player)? It changes no rule; P8_resetfix.md
version 2.3 stands as registered, and this only explains its forecast. Stored files, no lock. Code:
tools/p8_ghostsrc.py pairs; record reports/research/p8_ghostsrc_pairs.json.

Definitions, per scored frame (the same MPS-truth universe), for a set of non-referee boxes with feet in truth court ft:
missed players (in frame, no foot within 2.5 ft), ghost boxes (foot more than 3.0 ft from all 10 players), and
double-count pairs: a one-to-one assignment (Hungarian, by distance) between ghost boxes and missed players with the
pair's distance at most 6.0 ft (also reported at 4.5 ft). E = (missed + ghosts) / in-frame players. Resolving a pair
(the box placed within 2.5 ft) would lower E by 2 / in-frame players per pair.

Box sets, after the r23 build's first reset (and, for context, before it):
- sidecar: the build's non-referee boxes, foot_stab (the scorer's own counts);
- r23 raw: the corpus r23 player boxes at 0.40, raw bottom centre (the forecasts' population);
- r23 stab: the same r23 boxes, but each one that matches a sidecar box one to one at IoU 0.5 or more takes that
  sidecar box's foot_stab; the rest keep the raw bottom centre.

Reported per game, gsw_cha_2016 first, for each r23 set against the sidecar on after-reset frames: in-frame players,
missed, ghosts, pairs (6.0 and 4.5 ft), E; the observed full-fix change dE = E(r23) - E(sidecar) (the forecasts' f = 1,
unscaled); the part of it from added pairs, 2 (pairs(r23) - pairs(sidecar)) / in-frame; and the explained share
(that part over dE, when dE is above zero). Reading (fixed now): on gsw_cha_2016, if the explained share is at least
0.5 under both r23 sets, the forecast's net-error rise is mostly placement counted twice; if under 0.5 under both, it
is mostly not; otherwise the two foot choices disagree and both are reported.

### Pairing check result (2026-11-26, stored files, no lock, 9 s)

Record: reports/research/p8_ghostsrc_pairs.json. After the r23 build's first reset:

| game | in-frame players | set | missed | ghosts | pairs (6.0 ft) | pairs (4.5 ft) | E |
|---|---|---|---|---|---|---|---|
| gsw_cha_2016 | 149 | sidecar | 107 | 8 | 7 | 5 | 0.772 |
| | | r23 raw | 79 | 73 | 31 | 17 | 1.020 |
| | | r23 stab | 79 | 70 | 29 | 18 | 1.000 |
| gsw_bkn_2015 | 1,188 | sidecar | 707 | 49 | 42 | 30 | 0.636 |
| | | r23 raw | 269 | 162 | 93 | 76 | 0.363 |
| | | r23 stab | 251 | 162 | 93 | 71 | 0.348 |
| gsw_sac_2015 | 447 | sidecar | 296 | 14 | 13 | 4 | 0.694 |
| | | r23 raw | 118 | 62 | 45 | 30 | 0.403 |
| | | r23 stab | 109 | 59 | 40 | 22 | 0.376 |

On gsw_cha_2016 the full fix (f = 1, unscaled) raises E by +0.248 (raw) or +0.228 (stab); resolving the added pairs
would lower E by 0.322 and 0.295, explained shares 1.30 and 1.29. **Reading by the registered rule: mostly placement
counted twice.** A share above 1 means the measure is not an additive split of dE: resolving a pair removes a missed
player and a ghost box at once, and the fix also changes other counts. Post hoc, not registered: if the added pairs
were placed within 2.5 ft, E would be about 0.07 below the sidecar; if those boxes were dropped instead, E would stay
about 0.08 to 0.09 above it. On gsw_bkn_2015 and gsw_sac_2015 E falls with the fix (-0.27 to -0.32), so no share is
computed.

Context, report only: before the first reset, pairs are common in every set. Pairs per ghost: sidecar about 0.70
(1,670 of 2,336 on gsw_cha_2016, 1,547 of 2,239 on gsw_bkn_2015, 1,571 of 2,250 on gsw_sac_2015); r23 raw 0.57 to 0.62
and r23 stab 0.56 to 0.60. So the net-error row may count many placement errors twice throughout, not only after a
reset. Before the first reset the explained shares are raw 0.45, 0.42 and 0.33, stab 0.34, 0.33 and 0.12
(gsw_cha_2016, gsw_bkn_2015, gsw_sac_2015).

Limits: gsw_cha_2016's after-reset population is 149 in-frame players in a few windows; the shares rest on 24 (raw) and 22
(stab) added pairs. The pairing is one-to-one by distance within 6 ft and does not prove that a paired box is that player's box.
This explains the version 2.3 forecast; it changes no rule (P8_resetfix.md section 18 stands).

Review notes (2026-11-26, the ml-review of 014b4d06, PASS; wording outside the registered text): "account for"
became "resolving them would lower E by" with the post hoc placed and dropped readings and the note that a share above 1
is not an additive split; the D-near reading is labelled post hoc; "may count" with the separate sidecar and r23
ratios and the before-reset shares; the SportVU range now comes from the record (meta, rerun with every other number
unchanged); the comparison's raw-foot count (80) against the scorer's (71) is explained; the pairing check's extra
conditions and the 24 and 22 added pairs are stated.
