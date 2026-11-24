# P8 truth v2: a more stable per-window clock offset rule (pre-registration)

Written 2026-11-23 under PLAN.md D161 (2), before any v2 number was computed. Development games only. Code:
tools/p8_truth_v2.py (new; sportvu/ and every other frozen file untouched).

## What stays as it is

The frozen truth (sportvu/truth.py: solve_window_offset, window_truth; the camera truth built on it by
sportvu/autolabel.py and tools/p8_frozen.py) and every result scored on it stay as recorded. Truth v2 is a versioned
rule beside it: nothing already scored is rescored under v2 without its own registration (D161), and no file in the P8
workspace's labels/ is rewritten.

## The problem (D99, D100, docs/research/P8_PLATFORM.md)

The frozen rule takes, for each window, the offset on a grid of 0.04 s over plus or minus 1.5 s that minimises the
median over frames of the median distance from each projected foot to its nearest SportVU player. That objective is
flat near its minimum: with ten players on the court almost any foot is near someone, and a median of medians moves in
steps. Small build differences (MPS against CPU against Colab's CUDA, or another court model) move the argmin by 0.3 to
0.8 s, and the truth candidates collapse (P8_PLATFORM.md: on gsw_ind_2016_s02_w03 the MPS, CPU and detector-on-CPU
builds solved 0.90, 1.42 and 0.94 s; Colab solved 0.62 s).

## The v2 rule (fixed here)

For one window, with the same inputs as the frozen rule (each sidecar frame with a time-map clock, a homography and at
least 4 boxes; feet projected through the frame's homography; the game's mirror; at least 20 such frames, else no
offset, as frozen), on the same grid (minus 1.5 to plus 1.5 s in 0.04 s steps):

1. Objective. At each offset, for each frame with a SportVU moment within the frozen clock tolerance (0.12 s): a
   one-to-one Hungarian match of the projected feet to the 10 SportVU players; the frame's score is the mean of the
   matched distances after dropping the worst 20 percent of pairs (rounded down to whole pairs). J(offset) is the mean
   of the frame scores. One-to-one matching makes a foot pay for the player it takes, and means instead of medians
   remove the steps.
2. Estimate. Not the argmin: the weighted centre of the near-minimum set S = {offsets with J <= J_min + DELTA}, DELTA =
   0.10 ft, each offset weighted by (J_min + DELTA - J). On a flat bottom the argmin jumps; the centre of the bottom does
   not. DELTA is a fixed constant chosen here, not tuned on any number.
3. Edge flag. When S contains the grid end (minus 1.5 or plus 1.5 s), the window is flagged at_edge (its true offset may
   lie outside the grid); the estimate is still reported.

## Data (development games only)

The 3 pooled R1.3 games (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015) and gsw_ind_2016 reported apart (D98). No held-out
or D85 game, no hou_cle_2016 in any bar.

Build pairs that exist today (all built with frozen code; no new build for this registration):

- Court-model pairs, every window of the 4 games that has both: the V3 label build (label_build/, the build the frozen
  truth was solved on) and the r23 rebuild (builds/r23/). Same clip, time map, detector and device (MPS); only the court
  model differs. This is the only build difference available at scale today, and the primary measure.
- Device pairs, gsw_ind_2016 only (the D99 diagnosis builds in _p8-diag): s02_w03 built on MPS, on CPU and with only the
  detector on CPU; s02_w04 built on MPS and on CPU. Five builds of two windows: reported, not a bar.
- CUDA: when D161 (1) runs, its builds are added to the device pairs and reported the same way. Colab's frozen offsets
  from the R1.3 run are printed beside the device pairs for reference only (they come from the frozen rule).

## Measures

- Stability: for each pair, the absolute difference of the window's offset between the two builds, under the frozen
  rule (recomputed here with sportvu.truth.solve_window_offset, unchanged) and under v2. Reported: median absolute
  difference, share of windows over 0.2 s, maximum, per game and pooled over the 3 games; at_edge counts.
- Accuracy guard: on the V3 label builds of the 3 pooled games, the camera-truth candidates (sportvu.camera's
  frames_from_rows over sportvu.truth.truth_for_frame rows, the inputs of the camera fit, as P8_PLATFORM.md counted them)
  at each window's frozen offset and at its v2 offset. A stable rule that moves offsets away from where the data agree
  would lose candidates.
- Check: the recomputed frozen offsets on the V3 label builds must equal the stored labels/<game>_offsets.json; any
  window that differs is reported and left out of the bar.

## Bar (fixed before any number)

v2 passes when both hold on the 3 pooled games:

1. Stability, court-model pairs: the median absolute offset difference under v2 is at most half that under the frozen
   rule, and the share of windows with a difference over 0.2 s is lower under v2, with the 95 percent window bootstrap
   interval (2000 resamples, seed 0) of the paired reduction (frozen share minus v2 share) wholly above zero.
2. Accuracy guard: total candidates at the v2 offsets are at least 95 percent of total candidates at the frozen offsets.

If v2 passes, it is recorded as truth v2's offset rule and used only by work registered after this; if it fails, the
frozen rule stays the only one and v2 is reported as tried. gsw_ind_2016 and the device pairs are reported beside the
bar, and CUDA is added when it exists. Nothing in this test trains or tunes a model.

## Result (run 2026-11-23, tools/p8_truth_v2.py at 29d5afbe; truth_v2/truth_v2_result.json in the P8 workspace)

**v2 fails the registered bar, so the frozen rule stays the only offset rule.** v2 is more stable than the frozen rule
and finds more camera-truth candidates, but its pooled median offset difference between builds is not half the frozen
rule's, which the bar required.

Court-model pairs (V3 label build against r23 rebuild, same clip, time map, detector and device), windows whose
recomputed frozen offset equals the stored one:

| Game | Windows (left out: no frozen offset, fewer than 20 usable frames on the V3 build) | Frozen: median abs difference, share over 0.2 s, max | v2: median, share over 0.2 s, max | v2 at edge (V3, r23) | Candidates on V3 at frozen / v2 offsets |
| --- | --- | --- | --- | --- | --- |
| gsw_bkn_2015 | 77 (6) | 0.44 s, 0.7273, 2.12 s | 0.321 s, 0.6364, 1.755 s | 14, 5 | 3,235 / 3,771 (1.1657) |
| gsw_cha_2016 | 64 (9) | 0.54 s, 0.7812, 2.16 s | 0.356 s, 0.6406, 2.265 s | 14, 1 | 2,579 / 3,122 (1.2105) |
| gsw_sac_2015 | 73 (9) | 0.52 s, 0.7945, 2.44 s | 0.478 s, 0.726, 2.086 s | 19, 2 | 3,278 / 3,624 (1.1056) |
| Pooled, 3 games | 214 | 0.52 s, 0.7664, 2.44 s | 0.367 s, 0.6682, 2.265 s | | 9,092 / 10,517 (1.1567) |
| gsw_ind_2016 (apart, D98) | 89 (3) | 0.6 s, 0.7978, 1.48 s | 0.356 s, 0.7303, 1.831 s | 14, 0 | 3,903 / 4,310 (1.1043) |

Bar checks (pooled): median at most half the frozen rule's: no (0.367 against 0.26); share over 0.2 s lower: yes;
paired reduction of that share, window bootstrap interval [0.0234, 0.1682]: wholly above zero, yes; candidates at least
95 percent: yes (1.1567). One check fails, so v2 fails.

Device pairs, gsw_ind_2016 (D99 diagnosis builds; report-only; Colab's frozen offsets from reports/research/p8_platform.json beside them):

| Window | Frozen offsets | v2 offsets (none at edge) | Colab, frozen rule |
| --- | --- | --- | --- |
| s02_w03 | MPS 0.9, CPU 1.42, detector on CPU 0.94 (spread 0.52 s) | 1.206, 1.074, 1.061 (spread 0.145 s) | 0.62 |
| s02_w04 | MPS -1.34, CPU -0.78 (0.56 s) | 0.158, 0.523 (0.365 s) | -0.06 |

CUDA builds are added here when D161 (1) runs.

Reading:

- Under the frozen rule, two builds of the same window that differ only in the court model give offsets a median 0.52 s
  apart, inside the 0.3 to 0.8 s D99 reported across devices, and 77 percent of windows move by more than 0.2 s. The
  offset is not a property the frozen truth pins down.
- v2 narrows that (median 0.52 to 0.367 s, 10 points fewer windows over 0.2 s, an interval above zero) and its offsets
  give 16 percent more camera-truth candidates, but it does not halve the spread, and it hits the grid edge on 14 to 19
  V3 windows per game, so a wider grid or a different objective would be the next registration, not a change made
  here.
- Left out: 27 windows (6, 9, 9 and 3) whose recomputed frozen offset differs from the stored one. Cause, traced by the
  ML review: all 27 have fewer than 20 usable frames on the V3 build (0 to 16), where frozen window_truth stores a
  fallback of 0.0 (sportvu/truth.py:115) and the tool's solve_frozen returns None (tools/p8_truth_v2.py:98). On every
  solvable window the recomputed frozen offset equals labels/<game>_offsets.json. They have no V3 offset under either
  rule, so the bar numbers are the same with or without the check (rerun by the reviewer: n 214, 0.52 against 0.367,
  interval [0.0234, 0.1682], ratio 1.1567). They are listed in the result file.

Lock time: 0.18 hours (4 measure jobs of 177, 131, 148 and 194 s; the device job 6 s; from logs/tv2_chain.log in the P8
workspace).
