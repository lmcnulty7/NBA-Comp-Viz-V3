# P8: placement misses by cause on the decided truths (design, version 2, registered before any number)

Version 2, 2026-11-27, after the design review of version 1 (commit 9d5bf978, GO WITH CHANGES); it replaces version 1,
and section 10 maps each finding to its change. Report only: nothing is fitted, tuned, trained or built, no truth or
score changes, no correction is proposed or adopted, and no Colab unit is used. No held-out game, no D85 test game and
no test arena is read (CLAUDE.md rule 4). Committed before the tool computes any number.

## 1. Question

D202's development standard rows on the decided truths (gsw_cha_2016 and gsw_bkn_2015 on multipool v1, D199;
hou_cle_2016 on its reset-fixed CPU truth) put r23's pooled missed at 0.214, of which placement (the nearest stabilised
foot 2.5 to 5 ft from SportVU, D184's split) is 0.155 and detection 0.059. Under which observable conditions do the
placement misses occur, how much of the placement row is attributable to each, and what would each pattern point to?

## 2. Known before this item, and what is new

Known, stated as known (not findings of this item):
- D184 (P8_placement.md step 1): placement was 64 to 75 percent of r23's misses on the earlier truths; the offset runs
  along the camera ray, short of the player, with the median depth falling from near to far on every game; no timing
  signature; no leave-one-game-out correction passed: SCATTERED. D202's 0.155 of 0.214 (about 72 percent placement)
  matches that share.
- D185 (step 1b): MIXED; part of the offset was a disagreement between the r23 court's H and the truth camera.
- D186 (P8_footbias.md): the stabilised foot lands short toward the camera by 0.3 to 1.4 ft, growing with depth;
  corrections did not lower missed on every game. D187 and D188 (P8_footpos.md, P8_outpos.md): a depth shift helped
  position rows but failed detection non-inferiority; the output-position arm closed with STOP.
- D202: near-field bias 1.41 ft pooled.

New now: (a) the truths changed: reset-fixed CPU label builds with the multi-pool fit for gsw_cha_2016 and gsw_bkn_2015
(D199), where D184 to D188 used MPS truths built with the tracker bug and the frozen fit; the truth change alone lowered
missed by 0.083 on gsw_cha_2016 and 0.033 on gsw_bkn_2015, mostly placement (D202), so the placement misses that remain
are a different population. (b) The scored build is r23 with the tracker reset fixed (the composite builds/r23, D193).
(c) D184 to D188 measured the offset's direction, size, timing and correctability; none measured the conditions a miss
occurs under. This item measures those, and only describes.

Limits carried from D202, stated with every number: each truth's frames were selected by fits on the same r23 label
build's boxes, so absolute rates lean toward r23; multipool v1 is a better fit by the acceptance measures, not a truth
shown to be better.

## 3. What it can and cannot tell

It can tell which observable conditions co-occur with placement misses more often than with hits, on these truths and
this build, and what share of the placement row is attributable to each under the stated arithmetic. It cannot
apportion a miss between the build and the truth: every distance is the build's foot through the truth's camera, and
both can be wrong on the same frame. The build's and the truth's errors are not independent either: the truth's
frames were selected by, and the multi-pool camera fitted to, the same detector's feet, which is why D is partly
mechanical. It cannot see frames the truth fit rejected (they are outside the scorer's universe), so conditions that
make the truth reject a frame are undercounted. The box-based conditions (C2's box clause, C3a, C4) describe the nearest
box, which for a placement miss is often another player's box, not the missed player's own. The conditions overlap, and an
association is not a cause: a condition that carries is a place to look in a later, separately registered design, not
a diagnosis. It cannot tell whether the decided truths are closer to the real positions than the earlier ones.

## 4. Universe and the count gate

- Games and inputs: exactly D202's score run (reports/research/p8_rebaseline.json, run 20261127T171848Z): each game's
  current truth labels (gsw_cha_2016 and gsw_bkn_2015 in work_ms, hou_cle_2016 in work_ref), the r23 composite's
  sidecars (work_ref/builds/r23), the scorecard cache's wide frames, cuts and referee entries, time maps, SportVU
  moments and mirrors; each window's sidecar, time map, labels and identity file has its SHA-256 checked against
  work_ref/rebaseline/score__<game>__current.json's hashes before use (the cache files are not compared: D202's score
  step completed missing cache entries after hashing them; the count gate guards what the scorer reads from them).
- Per window, the pinned scorer (tools.p8_scorecard.score_window, truth mode camera) is called as D202 called it, and
  the scored frames are its observation frames. On those frames, per in-frame player (projected inside the frame with
  the 10 px margin), the nearest box foot_stab through the truth H (nearest, not one to one, as the scorer's missed
  rule): hit (at most 2.5 ft), placement miss (2.5 to 5 ft), detection miss (none within 5 ft).
- Count gate, first, one game at a time: per window, the in-frame, missed, detection (far) and non-referee box counts
  must equal work_ref/rebaseline/score__<game>__current.json's in_frame, missed, far and nr_boxes exactly, on every
  window; any difference stops the tool before any share is computed.
- gsw_cha_2016 and gsw_bkn_2015 are measured, separately and pooled. hou_cle_2016 (self-decided, as for the team study):
  the count gate only; no condition share is computed, so hou_cle_2016 stays unspent for later arms (D188, and D202's
  section 10.7 list of what is seen about it).

## 5. Conditions, fixed now

Thresholds are fixed here; none is chosen on outcomes. Percentiles are over all of a game's truth rows, never split by
hit or miss. Where a clause reads the truth (the truth H or the truth camera), it is marked truth-side; the others read
only the build. The scorer's definitions are reused where they exist.

- C1 Distance from the camera: the player in the near third (the scorer's camera third, tools.p8_scorecard.third_of).
  Described, not ruled: the median depth offset of placement misses along the camera ray per third
  (tools.p8_placement.depth_basis), the known short-foot bias (D184, D186).
- C2 Truncation at the frame edge: (build) the nearest box's bounding box within 2 px of the left, right or bottom
  image border; or (truth-side) the player's projected foot pixel within 20 px of one of those borders.
- C3a Overlap, build side: the nearest box overlaps another box with intersection over union at least 0.3, or the
  nearest box is also the nearest box of another in-frame player on the same frame.
- C3b Crowding, truth side: another SportVU player within 3.0 ft (the scorer's crowding, its first bin).
- C4 Box height and foot model: (truth-side) the nearest box's pixel height over the expected height of a 6.5 ft player
  at the SportVU position under the truth camera (tools.p8_e1_postmortem.foot_and_height) outside [0.75, 1.25]; or
  (build) the stabilised foot more than 8 px from the raw foot.
- C5 After a reset or cut, two signals with one horizon, reported separately: C5a, a reset logged in the window (the
  reset log of an arm window of the composite) at most 30 processed frames before the frame; C5b, a reset-like frame of
  the sidecar's track ids (tools.p8_trackreset.reset_like) at most 30 processed frames before. Report only beside them:
  the scorer's since-cut under 1.0 s (its first since_cut bin).
- C6 Truth-side proxy: the truth row's line matches (line_matches) at or below the game's 20th percentile; reported by
  third as well.
- D Frame-level feet disagreement (build or truth; not separable): the truth row's feet_resid_ft at or above the
  game's 80th percentile. Reported with the same measures, not in the truth-side reading and not in the attribution.

C3 means C3a or C3b; C5 means C5a or C5b. Every clause's enrichment is reported beside its condition, not ruled.

## 6. Measures

For gsw_cha_2016, gsw_bkn_2015 and the two pooled, each with a window-cluster bootstrap (2,000 resamples, seed 0,
windows resampled whole, as D202):

- For each condition (C1, C2, C3, C4, C5, C6, D, and each clause): its share among placement misses, among hits and
  among detection misses; the enrichment ratio (share among placement misses over share among hits); the
  placement-miss rate (over in-frame players) with and without it and their rate ratio RR; and its attributable
  fraction of the placement row, AF = (share among placement misses) times (1 - 1/RR), with its interval.
- Union coverage: the share of placement misses and of hits carrying at least one of C1 to C6. The hits' union
  coverage is the baseline against which the placement misses' "none" share is read.
- Exclusive shares: over all 720 orders of the six conditions C1 to C6, each condition's smallest and largest share of
  placement misses when each miss is counted under the first condition it carries (the largest is its non-exclusive
  share, the smallest its share carrying it alone). No single order is reported.
- Report only, not ruled: the share of placement misses whose nearest box is another in-frame player's one-to-one
  match (the scorer's one-to-one match within 2.5 ft).

## 7. Reading rule (fixed now; descriptive)

A condition "carries" placement misses when, on gsw_cha_2016 and on gsw_bkn_2015 separately, its enrichment ratio's
window interval lies above 1.5 and its share among placement misses is at least 0.10. A condition that carries on one
game only is reported as such. SCATTERED means no condition carries by this rule (D184's reading persists). What a
condition that carries would point to, for a later, separately registered design (nothing here acts on it):

| Condition | Points to |
| --- | --- |
| C1 | the geometric foot bias by distance already known (D186), now on the new truths |
| C2 | the detector and the foot at the frame edge (a truncated box has no visible contact point) |
| C3a | box separation and association: overlapping boxes, one box for two players |
| C3b | crowding in the scene, whatever the boxes do |
| C4 | box extent or the foot model |
| C5 | tracker behaviour after resets or cuts that the reset fix did not remove |
| C6 | truth-side weakness (few line matches), where part of the row may be the truth's |
| D | a frame where feet and the camera disagree, by the build or the truth (not separable); reported with the same measures, never part of the verdict |

The verdict (SCATTERED, or carried by named conditions) reads C1 to C6 only (C1, C2, C3a, C3b, C4, C5, C6); D is
reported and never ruled.

## 8. Cost

Under the heavy-job lock anyway (cheap; the scorer reads clips' sizes and caches), one game at a time, lock name
"p8 placecause <game>": about 4 to 6 minutes a game (D202's score step took 654.6 s for five pairs, most of it computing
referee entries now cached), cap 20 minutes a game; then a report step in system python, no lock. 0 Colab units.
Code: tools/p8_placecause.py with tests on synthetic inputs (each condition, the attribution arithmetic, the 720-order
bounds, the count gate), reviewed before it runs. Record reports/research/p8_placecause.json and a dated result section.

## 9. Guards

tools.p8_common.guard and tools.research_testset.refuse_cv_test on the game tags, the SportVU ids, the video ids and
every window, before anything is read. The score records, truths, sidecars and caches are read, never written; the
reference root's and work_ms's labels, builds/r23 and cache are hashed before and after each game, and a change stops
the tool. Writes: reports/research/p8_placecause.json, work_ref/placecause/ (per-game rows), and this document.

## 10. Changes from version 1 (review findings)

1. feet_resid_ft relabelled "frame-level feet disagreement (build or truth; not separable)" (D) and taken out of the
   truth-side reading; line_matches is the only truth-side proxy (C6), reported by third; section 3 says the item cannot
   apportion error between build and truth and cannot see rejected frames.
2. SCATTERED means no condition carries; union coverage among hits is the baseline for "none"; attributable fractions
   with intervals; exclusive shares as the bounds over all 720 orders, the single fixed order dropped.
3. hou_cle_2016: the count gate only, no condition shares; it stays unspent.
4. C3 split into C3a (build) and C3b (truth); truth-side clauses of C2 and C4 marked; the one-to-one-match share
   reported, not ruled.
5. C5: one 30-frame horizon for the logged and the reset-like signals, reported separately; the scorer's since-cut
   beside them.
6. D184's depth by third, D186, D188 and D202's 72 percent placement share stated as known; the scorer's third,
   crowding and since-cut reused; C1's depth clause kept as a description only.
7. The gate names work_ref/rebaseline/score__<game>__current.json and adds nr_boxes; the lock is taken, one game at a
   time.
8. The guard and refuse_cv_test calls named on tags, SportVU ids, video ids and windows.
9. Each clause's enrichment reported beside its condition, not ruled.

**Re-check fixes (2026-11-27, the review of 85731c25, GO; before the run).** N1: before scoring a window the tool stops
if its cache lacks a live frame's wide score, a needed referee frame or the cut signal. N2: the sidecar, time map and
labels hashes must be present in D202's record (no silent skip). N3: the verdict reads C1 to C6 only; D is reported,
not ruled. N4: hou_cle_2016 compares counts only, with no condition, depth or percentile computed. Section 3 adds that
build and truth errors are not independent and that the box-based conditions describe the nearest box.


## 11. Result (2026-11-27, run from fda5e5b5; record reports/research/p8_placecause.json)

**Run.** Three collect steps, one game at a time under the lock: gsw_cha_2016 57 s, gsw_bkn_2015 53 s, hou_cle_2016
106 s (count gate only), lock times from the heavy-job lines of work_ref/logs/placecause.log (the tool's own timings,
inside the lock, are 50.8, 47.1 and 100.2 s); the report step in system python, about 2 s. The count gate passed on every window (73, 83
and 93 windows: in_frame, missed, far and nr_boxes equal to D202's score records, and the scorer's own in_frame and
missed equal to the tool's); the manifests of work_ref and work_ms were unchanged. hou_cle_2016: no condition was
computed; it stays unspent.

**Verdict by the registered rule (C1 to C6): carried by C3a, on both games, through its shared-box clause, which reads
the truth (nearest and in-frame are defined through SportVU and the truth H) and shares the nearest-box rule with the
outcome; the IoU clause, which reads only the build, is not enriched (0.92 [0.83, 1.05] and 0.95 [0.85, 1.06]).** C3a
(the nearest box overlaps another box with IoU at least 0.3, or the nearest box is also the nearest box of another
in-frame player) has enrichment 1.76 [1.67, 1.87] on gsw_cha_2016 and 1.78 [1.65, 1.91] on gsw_bkn_2015, with shares
among placement misses 0.457 and 0.511; the shared-box clause alone has enrichment 2.60 [2.43, 2.80] and 2.49
[2.23, 2.76] (shares 0.388 and 0.442). No other condition carries; the verdict is not SCATTERED.

Beside the verdict, the attributable fractions (pooled) locate it: the shared-box clause 0.232 [0.206, 0.255], C3a 0.221
[0.193, 0.249], the IoU clause -0.007 [-0.016, 0.004]. The one-to-one view (report only): 0.230 [0.208, 0.251] of
placement misses (0.215 [0.183, 0.249] on gsw_cha_2016, 0.246 [0.218, 0.272] on gsw_bkn_2015) have a nearest box that is
another in-frame player's one-to-one match, against 0.036 to 0.042 of hits, enrichment 5.85 [5.19, 6.57] pooled.

**Attributable fractions** (share among placement misses times (1 - 1/RR); pooled gsw_cha_2016 and gsw_bkn_2015, window
intervals; per game in the record):

| Condition | AF pooled | Share among placement misses | Share among hits | Enrichment |
| --- | --- | --- | --- | --- |
| C1 near third | -0.022 [-0.045, 0.002] | 0.264 | 0.290 | 0.91 [0.84, 0.98] |
| C2 frame edge | 0.002 [-0.000, 0.004] | 0.010 | 0.008 | 1.30 [0.97, 1.69] |
| C3 (C3a or C3b) | 0.193 [0.163, 0.220] | 0.501 | 0.327 | 1.54 [1.47, 1.60] |
| C3a overlap (shared-box clause reads the truth) | 0.221 [0.193, 0.249] | 0.484 | 0.274 | 1.77 [1.69, 1.85] |
| C3b crowding (truth) | 0.016 [-0.000, 0.031] | 0.173 | 0.166 | 1.05 [0.95, 1.15] |
| C4 box height or foot | 0.126 [0.102, 0.149] | 0.422 | 0.315 | 1.34 [1.26, 1.42] |
| C5 reset or cut | 0.004 [0.001, 0.009] | 0.012 | 0.007 | 1.66 [1.29, 2.21] |
| C6 few line matches (truth) | 0.054 [0.036, 0.073] | 0.250 | 0.194 | 1.29 [1.19, 1.40] |

Per game, C3a's AF is 0.204 [0.171, 0.241] (gsw_cha_2016) and 0.239 [0.194, 0.278] (gsw_bkn_2015). The fractions overlap
and do not add.

**Union coverage:** 0.851 of placement misses carry at least one of C1 to C6, against 0.723 of hits (pooled; 0.856
against 0.729 on gsw_cha_2016, 0.846 against 0.718 on gsw_bkn_2015); so 0.149 of placement misses carry none.

**Exclusive shares, bounds over all 720 orders** (pooled; smallest when the condition comes last, largest when first):
C1 [0.065, 0.264], C2 [0.003, 0.010], C3 [0.185, 0.501], C4 [0.092, 0.422], C5 [0.002, 0.012], C6 [0.043, 0.250].

**D, frame-level feet disagreement (build or truth; not separable; reported, not ruled):** share among placement misses
0.288 against 0.179 among hits, enrichment 1.61 [1.50, 1.72], AF 0.109 [0.095, 0.126] (pooled).

Report only, not ruled: C4's clauses: the truth-side height clause is enriched (2.15 [1.87, 2.46] pooled) but covers
only 0.037 of placement misses; the foot-move clause covers 0.405 at 1.32 [1.24, 1.40]. C5a and C5b coincide on every
row of both games: the logged and reset-like frame sets are identical in 72 of 73 windows (gsw_cha_2016) and 80 of 83
(gsw_bkn_2015), and in the other windows the extra reset-like frames lie inside the 30-frame horizon of a logged reset
(checked on the sidecars after the run). The scorer's since-cut under 1 s is not enriched (0.85 [0.72, 1.01]). C1's depth description, placement
misses' median depth offset along the camera ray: near +1.15 and +1.30 ft (beyond the player), middle -2.12 and -1.90
ft, far -1.91 and -2.46 ft (gsw_cha_2016, gsw_bkn_2015), the known pattern (D184, D186). C6 by third: 0.24 to 0.26 of
placement misses against 0.17 to 0.20 of hits in every third.

Post hoc, not ruled: (a) the near third is under-represented among placement misses (C1 enrichment 0.91, interval
below 1 pooled), against the expectation that near-field bias would concentrate misses near the camera; (b) C3b
(another SportVU player within 3 ft) at 1.05 [0.95, 1.15] argues against scene crowding at 3 ft as the driver; the
verdict mostly restates that a quarter to two fifths of placement misses are assigned a neighbour's box, plausibly
because the player's own box is missing, merged or misplaced. Telling those apart needs a separately registered design.
Nothing here is adopted, fitted or acted on.

Dated note (2026-11-27, after the ml-review of 01b4502f): section 5 labels C3a "Overlap, build side"; that label was
wrong for its shared-box clause, which defines "nearest" and "in-frame" through SportVU and the truth H, and so reads
the truth. The registered text of section 5 is left unchanged; section 11 uses the corrected label. Next time the
record also stores the git head it ran from (this run's head, fda5e5b5, is in the commit history and the log).
