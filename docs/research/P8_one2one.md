# P8: a one-to-one view of the missed row (design, version 2, registered before any number)

Version 2, 2026-11-27, after the design review of version 1 (commit 078fccb2, GO WITH CHANGES); it replaces version 1
and section 10 maps each finding to its change. Report only: nothing is fitted, tuned, trained or built, no truth or
score changes, and no Colab unit is used. No held-out game, no D85 test game and no test arena is read (CLAUDE.md
rule 4). Committed before the tool computes any number.

## 1. Question

D202's missed row counts an in-frame SportVU player as missed when no box foot lies within 2.5 ft (the nearest foot,
not one to one), and splits it into placement (nearest foot 2.5 to 5 ft) and detection (none within 5 ft). D204 found
placement misses carried by a box nearest to two players. This item asks, for every missed player, whether a box is
left for it once the scorer's one-to-one assignment has given boxes to the players they fit within 2.5 ft, and if so how
far it is. That separates a player with no box left (a detection problem as far as this view can tell) from a player
with an unclaimed box placed too far (a foot or geometry problem as far as this view can tell).

## 2. Known before this item

- D204's report-only share: 0.230 [0.208, 0.251] of placement misses (pooled) have a nearest box that is another
  in-frame player's one-to-one match, against 0.036 to 0.042 of hits. That used the scorer's score_window matching
  (sportvu.truth._match: Hungarian on raw distances over all boxes, referees included, then pairs kept within 2.5 ft).
  It is related to P-i below but not equal to it: a different gate order, a different box set and a different player
  set.
- D202's records carry, per window, missed_1to1 (tools.p8_resetfix.one_to_one_missed: in-frame players only, against
  all box feet including referees, within 2.5 ft) and unmatched_1to1 (tools.p8_resetfix.unmatched_1to1: non-referee box
  feet against all ten players, within 2.5 ft, counting unmatched in-frame players plus unmatched non-referee boxes);
  the latter gives net_error_1to1, 0.410 [0.387, 0.435] pooled on gsw_cha_2016 and gsw_bkn_2015.
- D202's rows: missed 0.214, placement 0.155, detection 0.059 (pooled); Level 2's development targets are about 0.05
  missed and 0.05 ghosts per box.

## 3. Universe, inputs and the count gate

- Games: gsw_cha_2016 and gsw_bkn_2015 on multipool v1 (D199), measured separately and pooled; hou_cle_2016 (fixed CPU
  truth) gets the count gate only, and no class or share is computed for it, so it stays unspent (D188, D202 section
  10.7, D204).
- Inputs: exactly D202's score run (reports/research/p8_rebaseline.json, run 20261127T171848Z), read as
  tools/p8_placecause.py's reviewed collect step reads them (D204): each window's sidecar, time map and labels hash
  required in work_ref/rebaseline/score__<game>__current.json and equal (N2), the identity file compared when hashed;
  the scorecard cache checked complete before scoring (N1); the pinned scorer (tools.p8_scorecard.score_window, truth
  mode camera) called per window as D202 called it; its observation frames are the scored frames.
- Count gate, first, one game at a time, per window, against D202's score record: in_frame, missed, far, nr_boxes,
  missed_1to1 and unmatched_1to1, the last two by calling the scorer's own functions (tools.p8_resetfix
  one_to_one_missed and unmatched_1to1, as frame_extra calls them) on every scored frame and summing. Any difference
  stops the tool before any share is computed.
- The record stores the git head it ran from and the tool's own timings beside the lock times.

## 4. The assignment

Main convention (net_error_1to1's): non-referee box feet (the scorer's referee rule) against every SportVU player of the
frame (all ten, in frame or not), Hungarian by distance (scipy linear_sum_assignment) on a cost matrix where pairs beyond
the gate cost 1e6, a pair counting as a match only when within the gate.

- Stage A: exactly the scorer's 2.5 ft assignment (unmatched_1to1's).
- Stage B: only the players still unmatched after stage A against only the non-referee boxes still unmatched, with the
  same rule at a 5 ft gate. A stage B match is "a box no one claims within 2.5 ft, lying 2.5 to 5 ft away".
- unmatched_1to1 returns counts, not pairs, so the tool holds a pairs-returning copy, assign_1to1(feet, sv, is_ref,
  gate), whose lines are unmatched_1to1's with the pairs kept. Equivalence check on every scored frame, at both gates:
  assign_1to1's unmatched in-frame players plus unmatched non-referee boxes must equal unmatched_1to1(..., miss_ft=2.5)
  at 2.5 ft and unmatched_1to1(..., miss_ft=5.0) at 5.0 ft; a difference stops the tool. Stage B applies the same
  function to the reduced sets.

Sensitivities, reported beside the main split, never ruled:
- Single-stage 5 ft: one assignment at 5 ft over all players and non-referee boxes (version 1's split).
- In-frame-only convention for P-i (missed_1to1's): in-frame players only, against all box feet including referees,
  stage A at 2.5 ft (one_to_one_missed's assignment, held as a pairs-returning copy checked against
  one_to_one_missed's count on every frame) and stage B at 5 ft on the remainder.

## 5. Classes, exhaustive

For an in-frame player j with nearest-foot distance n (all boxes, D202's missed rule):

Placement misses (2.5 < n <= 5 ft):
- P-i, no box left: j is unmatched after both stages, and j's nearest box was claimed by another player (in either
  stage; the other player in frame or not).
- P-ii, an unclaimed box 2.5 to 5 ft away: j is matched in stage B.
- P-iii-ref: j is unmatched after both stages and its nearest box is a referee box (outside the assignment).
- Impossibility counters, each stopping the tool if non-zero: (1) j matched in stage A (that box would make n <= 2.5);
  (2) P-iii-free, j unmatched after both stages with its nearest box a non-referee box claimed by no one (by stage B's
  optimality that pair, within 5 ft and both unmatched, would have been matched).

Detection misses (n > 5 ft, or no box on the frame):
- D-i, neighbour's box nearest: j's nearest box was claimed by another player.
- D-iii-ref: the nearest box is a referee box. D-iii-free: the nearest box is an unclaimed non-referee box (possible:
  it lies beyond 5 ft). D-iii-none: no box on the frame.
- Impossibility counter (3): j matched in either stage (no box lies within 5 ft); stops the tool if non-zero.

Hits (n <= 2.5) are counted for context only: matched in stage A or not.

## 6. Measures

Per game and pooled (gsw_cha_2016 and gsw_bkn_2015): each class's count, its share of the placement (or detection)
misses, and its rate over in-frame players (the classes add up to D202's placement and detection rows, 0.155 and 0.059
pooled), each with a window-cluster bootstrap (2,000 resamples, seed 0, windows resampled whole within each game).
Report only, beside them: for P-ii, the median distance of the stage B box; for P-i and D-i, whether the claiming player
is in frame and the median distance between j and the claiming player; the share of P-i and D-i players with an
unclaimed non-referee box 5 to 10 ft away; and the two sensitivities of section 4.

## 7. What each share would mean for the Level 2 targets (descriptive; no rule decides anything)

| Class | What it says | Which fix could move it, and its ceiling |
| --- | --- | --- |
| P-i and D-i (no box left; a neighbour's box nearest) | once boxes go to the players they fit, none is left within 5 ft of this player | detection recall where players overlap (separating merged boxes, keeping a box through occlusion, the tracker's confirmation); a foot or geometry fix moves P-i only where the player's own box lies beyond 5 ft (the 5 to 10 ft report gives a sense of how often an unclaimed box lies there). Ceiling: the class's rate |
| P-ii (an unclaimed box 2.5 to 5 ft away) | a box no one claims within 2.5 ft lies 2.5 to 5 ft from the player | the foot model, the box bottom or the court geometry (the D186 to D188 line, which failed non-inferiority on detection). Its ceiling and its reading as the player's "own box" depend on the assignment choice (section 4's sensitivities) |
| P-iii-ref and D-iii-ref (a referee box nearest) | referee and player boxes interact near the player | the referee rule or class confusion; ceiling: its rate |
| D-iii-free (an unclaimed box beyond 5 ft nearest) | the nearest box is no one's and far | detection or a gross placement error; ceiling: its rate |
| D-iii-none (no box on the frame) | an empty frame for the detector | detection on that frame type |

A ceiling is the most the class could lower the missed row if a fix removed all of it without new misses or ghosts;
it is not a forecast.

## 8. What it can and cannot tell

It can tell how many missed players are left without a box once the scorer's one-to-one assignment has run, and how
many have an unclaimed box placed too far, on these truths and this build. It cannot tell why a box is missing (merged,
occluded, never detected), whose a box really is, or whether a misplaced box is the build's error or the truth's; the
build's and the truth's errors are not independent (D204 section 3: truth frames were selected by, and the multipool
camera fitted to, the same detector's feet). One-to-one assignment is a convention: another rule could give a box to the
other player of a close pair, which is why the sensitivities are reported.

## 9. Cost and guards

Stored files only, the pinned scorer per window, no model, no video decode beyond clip sizes. Under the heavy-job lock
anyway (cheap), one game at a time ("p8 one2one <game>"); D204's collect took 53 to 57 s of lock per measured game and
106 s for hou_cle_2016: estimate 2 minutes a game, cap 10 minutes a game; a report step in system python, no lock. 0
Colab units. tools.p8_common.guard and tools.research_testset.refuse_cv_test on the tags, SportVU ids, video ids and
every window; the reference root's and work_ms's labels, builds and cache hashed before and after each game. Code:
tools/p8_one2one.py with tests on synthetic frames, reviewed before it runs. Record reports/research/p8_one2one.json,
with the git head; per-game rows and gate records under work_ref/one2one/ (M3).

## 10. Changes from version 1 (review findings)

1. Two-stage assignment: stage A is the scorer's 2.5 ft assignment exactly, stage B the still-unmatched players against
   the still-unmatched non-referee boxes at 5 ft; P-ii means an unclaimed box 2.5 to 5 ft away; the single-stage 5 ft
   split is a sensitivity.
2. The equivalence check runs at both gates on every frame, against unmatched_1to1 at 2.5 and at 5.0 ft.
3. P-iii-free is impossible by optimality and is the third impossibility counter; D-iii-free stays.
4. The conventions are stated (missed_1to1: in-frame players against all boxes including referees; D204's 0.230: _match
   at 2.5 ft); P-i under the in-frame-only convention is a sensitivity.
5. "A foot or geometry fix moves P-i only where the player's own box lies beyond 5 ft"; the 5 to 10 ft unclaimed-box
   share is reported; the P-ii row notes that its ceiling and "own box" reading depend on the assignment.
6. D204's 0.230 and D202's missed_1to1, unmatched_1to1 and net_error_1to1 stated as known; hou_cle_2016 gate only; N1
   and N2 reused.

**Code check fixes (2026-11-27, the check of f6a02bad, GO; before the run).** M1: on every frame, stage A's pairs must
equal assign_1to1's pairs at 2.5 ft, or the tool stops. M2: the three impossibility counters are counted for both
sensitivities as well as the main split, and any non-zero counter stops the tool. M3: section 9 names work_ref/one2one/.

