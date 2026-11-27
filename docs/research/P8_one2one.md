# P8: a one-to-one view of the missed row (design, registered before any number)

Written 2026-11-27 for PLAN.md D204's next item. Report only: nothing is fitted, tuned, trained or built, no truth or
score changes, and no Colab unit is used. No held-out game, no D85 test game and no test arena is read (CLAUDE.md
rule 4). Committed before the tool computes any number.

## 1. Question

D202's missed row counts an in-frame SportVU player as missed when no box foot lies within 2.5 ft (the nearest foot,
not one to one), and splits it into placement (nearest foot 2.5 to 5 ft) and detection (none within 5 ft). D204 found
placement misses carried by a box nearest to two players: about a quarter of placement misses have a nearest box that
is another player's one-to-one match. This item asks, for every missed player: does the player have a box of its own
in a one-to-one assignment, and if so how far is it? That separates a missing own box (a detection problem, whatever
the foot does) from an own box placed too far (a foot or geometry problem).

## 2. Universe, inputs and the count gate

- Games: gsw_cha_2016 and gsw_bkn_2015 on multipool v1 (D199), measured separately and pooled; hou_cle_2016 (fixed
  CPU truth) gets the count gate only, and no class or share is computed for it, so it stays unspent (D188, D202
  section 10.7, D204).
- Inputs: exactly D202's score run (reports/research/p8_rebaseline.json, run 20261127T171848Z), read as
  tools/p8_placecause.py's reviewed collect step reads them (D204): each window's sidecar, time map, labels and
  identity file checked against work_ref/rebaseline/score__<game>__current.json's SHA-256 (all three of sidecar,
  time map and labels required); the scorecard cache checked complete before scoring; the pinned scorer
  (tools.p8_scorecard.score_window, truth mode camera) called per window as D202 called it; its observation frames are
  the scored frames.
- Count gate, first, one game at a time, per window, against D202's score record: in_frame, missed, far and nr_boxes
  (as D204), and also missed_1to1 and unmatched_1to1, the latter two computed by calling the scorer's own functions
  (tools.p8_resetfix.one_to_one_missed and unmatched_1to1, as tools.p8_resetfix.frame_extra calls them) on every scored
  frame and summing. Any difference stops the tool before any share is computed.
- The run record stores the git head it ran from and the tool's own timings beside the lock times.

## 3. The one-to-one assignment, from the scorer's code

The one-to-one rule is the one net_error_1to1 uses (tools.p8_resetfix.unmatched_1to1, P8_resetfix.md section 21):
non-referee box feet (the scorer's referee rule) against every SportVU player of the frame (all ten, in frame or not),
Hungarian by distance (scipy linear_sum_assignment) on a cost matrix where pairs farther than the gate cost 1e6, and a
pair counts as a match only when its distance is at most the gate.

That function returns counts, not pairs, and its gate is 2.5 ft; a match at 2.5 to 5 ft needs the same rule at a 5 ft
gate. The tool therefore holds a pairs-returning copy of the function's assignment, assign_1to1(feet, sv, is_ref,
gate), whose lines are those of unmatched_1to1 with the pairs kept, and an equivalence check run on every scored frame
before any class is counted: at gate 2.5 ft, assign_1to1's unmatched in-frame players plus unmatched non-referee boxes
must equal unmatched_1to1's return value exactly; a difference stops the tool. The classes use the same function at
gate 5.0 ft (the "NEAR_FT" of D184's split). No other matching is reimplemented: the nearest-foot distances, the
in-frame rule and the hit, placement and detection classes are D204's, which reproduced D202 on every window.

## 4. Classes, exhaustive

For an in-frame player j with nearest-foot distance n (all boxes, D202's missed rule) and its one-to-one match at the
5 ft gate, at distance m (none if unmatched):

Placement misses (2.5 < n <= 5 ft):
- P-i, no own box: j is unmatched at 5 ft, and j's nearest box is another player's one-to-one match (in frame or not).
- P-ii, own box misplaced: j is matched at 2.5 < m <= 5 ft.
- P-iii, other, in two exhaustive parts: P-iii-ref, j is unmatched and its nearest box is a referee box (outside the
  assignment); P-iii-free, j is unmatched and its nearest box is a non-referee box matched to no one.
- m <= 2.5 cannot occur for a placement miss (that box would make n <= 2.5); the tool counts it and stops if it is
  ever non-zero.

Detection misses (n > 5 ft, or no box on the frame):
- D-i, neighbour's box nearest: j's nearest box is another player's one-to-one match.
- D-iii, other: D-iii-ref (nearest box is a referee box), D-iii-free (nearest box matched to no one), D-iii-none (no
  box on the frame).
- An own box within 5 ft cannot exist for a detection miss (D-ii is empty by definition); counted and stops if
  non-zero.

Hits (n <= 2.5) are counted for context: matched within 2.5 ft at the 2.5 ft gate, or not (a hit whose nearest box went
to someone else), reported only.

## 5. Measures

Per game and pooled (gsw_cha_2016 and gsw_bkn_2015), each class's count, its share of the placement (or detection)
misses, and its rate over in-frame players (so the classes add up to D202's placement and detection rows, 0.155 and
0.059 pooled), each with a window-cluster bootstrap (2,000 resamples, seed 0, windows resampled whole within each game).
Report only, beside them: for P-ii, the median m; for P-i and D-i, whether the neighbour whose match took the box is
in frame, and the median distance between j and that neighbour.

## 6. What each share would mean for the Level 2 targets (descriptive; no rule decides anything)

Level 2's development targets are about 0.05 missed and 0.05 ghosts per box, against D202's 0.214 and 0.121 pooled.

| Class | What it says | Which fix could move it, and its ceiling |
| --- | --- | --- |
| P-i and D-i (no own box; a neighbour's box nearest) | the player has no box of its own in a one-to-one sense; the nearest box belongs to someone else | detection recall where players overlap (separating merged boxes, keeping a box through occlusion, the tracker's confirmation); a foot or geometry fix cannot move it. Ceiling: the class's rate |
| P-ii (own box 2.5 to 5 ft away) | the player has a box, placed too far | the foot model, the box bottom, or the court geometry (the D186 to D188 line, which failed non-inferiority on detection); ceiling: P-ii's rate |
| P-iii-ref and D-iii-ref (a referee box nearest) | referee and player boxes interact near the player | the referee rule or class confusion; ceiling: its rate |
| P-iii-free and D-iii-free (an unmatched box nearest, the player unmatched) | an assignment edge case, expected rare | none as such; reported to show the split is exhaustive |
| D-iii-none (no box on the frame) | an empty frame for the detector | detection on that frame type |

A ceiling is the most the class could lower the missed row if a fix removed all of it without new misses or ghosts;
it is not a forecast.

## 7. What it can and cannot tell

It can tell how many missed players have no box of their own in the scorer's one-to-one sense and how many have one
placed too far, on these truths and this build. It cannot tell why a box is missing (merged, occluded, never detected)
or whether a misplaced box is the build's error or the truth's; the build's and the truth's errors are not independent
(D204 section 3: truth frames were selected by, and the multipool camera fitted to, the same detector's feet). The one
to one assignment is a convention: a different rule could give a box to the other player of a close pair.

## 8. Cost and guards

Stored files only, the pinned scorer per window, no model, no video decode beyond clip sizes. Under the heavy-job lock
anyway (cheap), one game at a time ("p8 one2one <game>"), as D204's collect took 53 to 57 s of lock per measured game
and 106 s for hou_cle_2016: estimate 2 minutes a game, cap 10 minutes a game; a report step in system python, no lock.
0 Colab units. tools.p8_common.guard and tools.research_testset.refuse_cv_test on the tags, SportVU ids, video ids and
every window; the reference root's and work_ms's labels, builds and cache hashed before and after each game. Code:
tools/p8_one2one.py with tests on synthetic frames (the assignment's equivalence with unmatched_1to1, each class, the
impossibility counters, the gate), reviewed before it runs. Record reports/research/p8_one2one.json, with the git head.
