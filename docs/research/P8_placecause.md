# P8: placement misses by cause on the decided truths (design, registered before any number)

Written 2026-11-27 for PLAN.md D202's next item. Report only: nothing is fitted, tuned, trained or built, no truth or
score changes, no correction is proposed or adopted, and no Colab unit is used. No held-out game, no D85 test game and
no test arena is read (CLAUDE.md rule 4). This document is committed before the tool computes any number.

## 1. Question

D202's development standard rows on the decided truths (gsw_cha_2016 and gsw_bkn_2015 on multipool v1, D199;
hou_cle_2016 on its reset-fixed CPU truth) put r23's pooled missed at 0.214, of which placement (the nearest stabilised
foot 2.5 to 5 ft from SportVU, D184's split) is 0.155 and detection 0.059. Which observable conditions do the placement
misses carry, how much of the placement row does each account for, and what would each pattern point to?

## 2. Earlier work, and what is new

- D184 (P8_placement.md step 1): on the earlier truths, placement was 64 to 75 percent of r23's misses; the offset runs
  along the camera ray, short of the player, with no timing signature; no leave-one-game-out correction passed: SCATTERED.
- D185 (step 1b): part of the offset was a disagreement between the r23 court's H and the truth camera (MIXED);
  gsw_cha_2016's truth camera was the first suspect.
- D186 (P8_footbias.md): the stabilised foot lands short toward the camera by 0.3 to 1.4 ft, growing with depth;
  corrections did not lower missed on every game.
- D187 and D188 (P8_footpos.md, P8_outpos.md): a depth shift helped position rows but failed detection
  non-inferiority, and the output-position arm closed with STOP.

What is new now: (a) the truths changed. D184 to D188 scored on MPS truths built with the tracker bug and the frozen fit
(gsw_cha_2016's later shown broken in its CPU form, P8_cha_fit.md); D202's truths are reset-fixed CPU label builds with
the multi-pool fit for gsw_cha_2016 and gsw_bkn_2015. The truth change alone lowered missed by 0.083 on gsw_cha_2016 and
0.033 on gsw_bkn_2015, mostly placement (D202), so the placement misses that remain are a different population. (b)
The scored build changed too: r23 with the tracker reset fixed (the composite builds/r23, D193). (c) D184 to D188
measured the direction, size, timing and correctability of the offset; none measured the conditions a miss occurs
under (frame edge, crowding, box shape, resets, truth quality). This item measures those, and only describes.

Limits carried from D202 and stated with every number: each truth's frames were selected by fits on the same r23
label build's boxes, so absolute rates lean toward r23; multipool v1 is a better fit by the acceptance measures, not a
truth shown to be better.

## 3. Universe and the count gate

- Games and inputs: exactly D202's score run (reports/research/p8_rebaseline.json, run 20261127T171848Z): for each game
  the current truth's labels (gsw_cha_2016 and gsw_bkn_2015 in work_ms, hou_cle_2016 in work_ref), the r23 composite's
  sidecars, the scorecard cache's wide frames and referee entries, time maps, SportVU moments and mirrors, as each score
  record names them, with each file's SHA-256 checked against the record's hashes before use.
- Per in-frame player (the pinned scorer's universe: truth rows on live wide frames that the build processed, players
  projected inside the frame with the 10 px margin), the nearest box foot_stab through the truth H (nearest, not one to
  one, as the scorer's missed rule): hit (at most 2.5 ft), placement miss (2.5 to 5 ft), detection miss (none within
  5 ft). The construction is tools.p8_placement.load_game's, pointed at these inputs.
- Count gate, first: per window, the in-frame, missed and detection counts must equal the D202 score record's in_frame,
  missed and far exactly, on every window of the three games; any difference stops the tool before any share is
  computed.
- gsw_cha_2016 and gsw_bkn_2015 are pooled, as D202's rows; hou_cle_2016 is reported apart, as D202 does.

## 4. Conditions (attributes of a player-frame), fixed now

Each is computed for every in-frame player, from stored files only, with thresholds fixed here before any number; none
is chosen or tuned on outcomes. Percentile thresholds are taken over all of a game's rows of the named quantity, never
split by hit or miss.

- C1 Distance from the camera: the scorer's camera third (near, middle, far; tools.p8_scorecard.third_of) and, for
  misses, the depth component of the offset along the camera ray (tools.p8_placement.depth_basis), as D184. Context:
  D202's near-field bias, 1.41 ft pooled.
- C2 Truncation at the frame edge: the nearest box's bounding box within 2 px of the left, right or bottom image border,
  or the player's projected foot pixel (through the truth H) within 20 px of one of those borders.
- C3 Occlusion or overlapping boxes: the nearest box overlaps another box with intersection over union at least 0.3, or
  another SportVU player (either team) stands within 3.0 ft, or the nearest box is also the nearest box of another
  in-frame player on the same frame.
- C4 Box height and foot model: the nearest box's pixel height over the expected height of a 6.5 ft player at the
  SportVU position (tools.p8_e1_postmortem.foot_and_height) outside [0.75, 1.25]; or the stabilised foot more than 8 px
  from the raw foot (foot_stab against foot_raw, the foot model moved it).
- C5 After a reset or cut: the frame at or after a reset in its window (the reset log of an arm window of the
  composite), or within 30 processed frames after a reset-like frame of the sidecar's track ids
  (tools.p8_trackreset.reset_like, which also marks camera cuts).
- C6 Truth-side error proxy: the truth row's per-frame fit residual (feet_resid_ft) at or above the game's 80th
  percentile over all its truth rows, or its line matches (line_matches) at or below the 20th percentile.

## 5. Measures

Per game, pooled (cha and bkn), and hou_cle_2016 apart; each with a window-cluster bootstrap (2,000 resamples, seed 0,
windows resampled whole, as D202):

- For each condition: its share among placement misses, among hits and among detection misses; the enrichment ratio
  (share among placement misses over share among hits); and the placement-miss rate (over in-frame players) with and
  without the condition.
- Exclusive attribution, in this fixed order: C2, C3, C5, C6, C4, C1 (near third only), none. Each placement miss is
  counted under the first condition it carries; the shares sum to 1 and give the fraction of the placement row (0.155
  pooled) under each. The order is fixed now and is not changed after any number; the non-exclusive shares above are
  the order-free view.
- C1 by third: the placement-miss rate and the median depth offset of placement misses per third.

## 6. Reading rule and what each pattern points to (fixed now; descriptive)

A condition "carries" placement misses when, on gsw_cha_2016 and on gsw_bkn_2015 separately, its enrichment ratio's
window interval lies above 1.5 and its share among placement misses is at least 0.10. A condition that carries on one
game only is reported as such. hou_cle_2016's reading is reported beside, never used to decide.

| Condition carries | Points to (for a later, separately registered design; nothing here acts on it) |
| --- | --- |
| C1 (near third, or depth offset growing toward the camera) | the geometric foot bias along the camera ray already seen (D186), now on the new truths: foot-model or court geometry by distance |
| C2 | the detector and the foot at the frame edge (a truncated box has no visible contact point) |
| C3 | crowding: box overlap or several players sharing one box; association and box separation |
| C4 | box extent or the foot model (a short, tall or moved foot) |
| C5 | tracker behaviour after resets and cuts that the reset fix did not remove |
| C6 | truth-side error: part of the placement row is the truth's, not r23's, which lowers its floor as a target |
| none (the "none" share is the largest) | a scattered offset not tied to these conditions (D184's SCATTERED reading persists) |

## 7. What is not done

Nothing is fitted: no correction, threshold search, model or selection; every threshold is in section 4. No score,
truth, build or registered row changes. D184 to D188's results stand.

## 8. Cost

Stored files only (labels, sidecars, cache JSON, time maps, moments; no video decode, no model), system python, no
lock, as D184 to D188 ran (step 1b took about 22 s; this reads three games' D202 inputs once and bootstraps 2,000
times): estimate 10 minutes, cap 30 minutes. 0 Colab units. Code: tools/p8_placecause.py with tests on synthetic
inputs (the conditions, the exclusive order, the count gate), written and reviewed before it runs; record
reports/research/p8_placecause.json and a dated result section here.
