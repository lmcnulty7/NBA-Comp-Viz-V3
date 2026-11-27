# P8 reset-fix build arm: design (version 1, 2026-11-26)

Status: design only, registered before any build or score. No compute has run for this arm. The only numbers below
come from stored files (the r23 reference-platform builds, their identity files, the E1 corpus boxes and the scorer
universe) through `python3 -m tools.p8_resetfix estimate` (about 10 seconds, no model code, no lock; record
reports/research/p8_resetfix_estimate.json). Nothing trains; the design follows docs/research/TRAINING_PROTOCOL.md
because the arm spends lock hours and feeds a decision.

## 1. Question and mechanism

Cause (confirmed, docs/research/P8_a3input.md): detect/tracker.py `PlayerTracker.reset()` sets
`self.model.predictor = None`. In ultralytics 8.4.48, `Model.track` registers the tracker callbacks whenever the
predictor has no `trackers`, and `Model.add_callback` appends. After a reset the model therefore holds two copies of
`on_predict_start` and `on_predict_postprocess_end`; every later frame runs two BoT-SORT updates, the second on the
first update's tracked boxes only, so detections that are not already tracks never confirm. Each further reset adds
one more copy. On gsw_cha_2016_s05_w00, as built, all 1,038 steps after the reset at frame 312 ran two or more updates
and the tracker kept 0.303 of the detector's boxes; with the reset fixed in process it ran one update per step (0
doubled frames) and kept 0.955 (same 8,766 detector boxes in both).

Scope: any build made with detect/tracker.PlayerTracker under ultralytics 8.4.48 that had at least one reset carries
the bug (a code fact). Whether r23's registered baselines and the recorded held-out numbers had resets, and how much
the bug moved them, is not tested here or by this arm. Held-out numbers are not rebuilt or re-scored (CLAUDE.md rule
4); this arm builds and scores the four development games only. Reset timing differs between detectors (the cut rule
counts each build's own tracked ids), so the bug need not cancel in paired comparisons such as E1 against r23.

Question: built on the reference platform with the fix, does r23 miss fewer in-frame players (missed row, and its
detection row) without harming ghosts, team, position and bias?

## 2. The patch (no frozen file edited)

A tools-side switch in the style of `P8_MPS_CLIP_FIX`:

- `P8_TRACKER_RESET_FIX=1` (tools/p8_device.py). Values "1", "0" or unset; anything else is refused. With it set,
  `is_set()` is true, so `install_proxy()` routes the build through `python -m tools.p8_device build_trajectories.py
  ...`, and the child calls `tools.p8_resetfix.apply_reset_fix()` before the build script runs (in
  `device_record`). The window's `<window>_device.json` carries `"reset_fix": true` and
  `"reset_fix_patched": "detect.tracker.PlayerTracker.reset"`. `check_cached()` refuses to reuse a window whose
  record's reset_fix differs from the flag, so a folder never mixes arms.
- The fix (tools/p8_resetfix.py), applied in the child process only:

```python
TRACKER_EVENTS = ("on_predict_start", "on_predict_postprocess_end")
TRACKER_MODULE = "ultralytics.trackers.track"

def is_tracker_callback(c) -> bool:
    f = getattr(c, "func", None)
    return f is not None and getattr(f, "__module__", "") == TRACKER_MODULE and \
        getattr(f, "__name__", "") in TRACKER_EVENTS

def strip_tracker_callbacks(callbacks: dict) -> int:
    n = 0
    for ev in TRACKER_EVENTS:
        keep = [c for c in callbacks.get(ev, []) if not is_tracker_callback(c)]
        n += len(callbacks.get(ev, [])) - len(keep)
        if ev in callbacks:
            callbacks[ev] = keep
    return n

def apply_reset_fix() -> str:
    import detect.tracker as DT
    orig = DT.PlayerTracker.reset
    def reset(self):
        orig(self)                                   # the frozen reset runs unchanged
        cbs = getattr(getattr(self, "model", None), "callbacks", None)
        RESET_LOG["resets"].append({"stripped": strip_tracker_callbacks(cbs) if isinstance(cbs, dict) else None})
    DT.PlayerTracker.reset = reset
    return "detect.tracker.PlayerTracker.reset"
```

  After the frozen reset, every tracker callback is removed; the next `Model.track` call finds no `trackers` on the
  new predictor and registers exactly one copy again. Other callbacks (ultralytics defaults) are kept. This is the
  a3input variant B code, moved into a module and keyed on the callback's module and name instead of identity with
  the imported functions (identical in effect; the real partials are recognised, checked by a test).
- Build-time evidence: each reset appends `{"stripped": n}` to a log written at exit as `<window>_resetfix.json` in
  config.TRACKING_DIR; sportvu.rebuild.build moves every `<window>_*` file next to the window's outputs. With the fix
  every entry must be 2 (one start, one postprocess callback), and the number of entries must equal the identity
  file's `camera_cut_resets`.
- Tests (tests/test_p8_resetfix.py, pass on the system interpreter; the last one also runs under the frozen CV
  interpreter): a fake model with the three ultralytics behaviours the bug needs shows 1, 1, 1 updates per frame
  before a reset, 2 after one reset and 3 after two (the bug), and 1 per frame after every reset with the fix;
  `strip_tracker_callbacks` keeps non-tracker callbacks; `apply_reset_fix` on a stand-in `detect.tracker` keeps the
  frozen reset (predictor None), logs 2 stripped per reset and writes the log; the flag parses; the real ultralytics
  8.4.48 partials are recognised (skipped where ultralytics is not importable).

## 3. Builds

- Platform: the reference platform (D175), `P8_DEVICE_DETECTOR=cpu`, court and gate on auto, r23
  (models/player_detector.pt, sha256 4c886d5f...), through the E1R build path (tools.p8_frozen.safe_build), the
  child under the heavy-job lock, with `P8_TRACKER_RESET_FIX=1` added and nothing else changed.
- Baseline (no fix), already built and never rebuilt: builds/e1m6_r23_detcpu (gsw_bkn_2015) and
  builds/m9_r23_detcpu__<game> (gsw_cha_2016, gsw_sac_2015, hou_cle_2016).
- Arm folders: builds/resetfix_r23_detcpu__<game>.
- Which windows are built, and why that is enough: the fix changes code only inside `PlayerTracker.reset`. In a
  window whose baseline identity file records 0 resets, no reset ran, so the fixed build makes the same calls in the
  same order. CPU builds repeat byte for byte (E1R smoke gate 1: gsw_cha_2016_s00_w00 rebuilt through the E1R path was
  byte-identical to M9's build). So a no-reset window's fixed build equals its baseline build, and only windows with
  `camera_cut_resets` of 1 or more are built:

| game | scored windows | windows with a reset | resets | processed frames in those windows | all processed frames |
|---|---|---|---|---|---|
| gsw_bkn_2015 | 83 | 17 | 19 | 5,987 (0.29) | 20,644 |
| gsw_cha_2016 | 73 | 12 | 12 | 5,274 (0.29) | 18,083 |
| gsw_sac_2015 | 82 | 10 | 10 | 3,684 (0.20) | 18,710 |
| hou_cle_2016 | 93 | 23 | 23 | 10,076 (0.42) | 23,813 |

  The claim is tested, not assumed: an equivalence sample of 2 no-reset windows per game (fixed now: the first and
  the last no-reset scored window in sorted name order) is built with the fix and must be byte-identical to the
  baseline (frames, trajectories, identity). For scoring, a no-reset window of the arm resolves to the baseline
  folder's files, read only and never copied; the score file lists for each window whether it was built or resolved
  to the baseline.
- Pairing checks, as E1R (tools/p8_e1r.py check_records, check_frames, gate-empty sets), with one change: the device
  records must agree on `requested` and differ only in reset_fix (true in the arm, absent in the baseline). Equal
  processed frame sets and identical court H per window (the fix does not touch the gate, the court models or the
  frame reader; the pre-gate pass runs before any tracking). Equal gate-empty window sets.
- Per built window, also checked: the sidecar rows are identical to the baseline's up to and including the
  baseline's first reset-like frame (`tools.p8_trackreset.reset_like`; before the first reset the two builds run the
  same code); the reset log has `camera_cut_resets` entries, each 2.

## 4. Expected effect and upper bound (stored files, pool games, the scorer universe's corpus frames)

The fix can change only frames after a window's first reset. Rates are per in-frame player on the corpus frames of
the scored universe (the MPS-truth universe in use when the estimate ran): 20,457, 13,284 and 21,144 in-frame players
on 2,073, 1,353 and 2,151 frames.

| | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 | pool |
|---|---|---|---|---|
| missed (no foot within 2.5 ft) | 0.2525 | 0.2952 | 0.2060 | |
| in-frame players after a window's first reset (share) | 0.058 | 0.011 | 0.021 | 0.033 |
| missed among those players | 0.594 | 0.718 | 0.660 | |
| ceiling: every miss after a reset found (share of all in-frame) | 0.0345 | 0.0081 | 0.0140 | 0.0202 |
| bound: misses after a reset that an r23 corpus box (0.40) reaches within 2.5 ft | 0.0242 | 0.0038 | 0.0090 | 0.0134 |
| far misses (no foot within 5 ft), all frames | 0.0915 | 0.0751 | 0.0522 | 0.0724 |

- Missed: the bound is the raw-reachable after-reset share, 0.024, 0.004, 0.009 (pooled 0.013). It slightly
  understates what the tracker can return, since the tracker takes boxes from 0.25 and the corpus count uses 0.40.
  Expected: most of the bound, because a3input's fixed variant kept 0.955 of the detector's boxes after the reset
  against 0.303. Expected missed change about -0.015 to -0.022 on gsw_bkn_2015, about -0.002 to -0.004 on
  gsw_cha_2016, about -0.006 to -0.009 on gsw_sac_2015, pooled about -0.009 to -0.013. Direction: down.
- Detection row (far misses): down, by part of the same players. Far misses after a reset are not split out by the
  estimate; the far-miss rows an r23 box reaches within 5 ft (0.038, 0.020, 0.021, all frames) bound it from above.
  The placement row (2.5 to 5 ft) can move either way: some far misses become placement misses when a new track's
  stabilised foot lands 2.5 to 5 ft away.
- Ghosts (non-referee): up or flat. The fix returns boxes the detector already made; some are not players near
  SportVU positions. On these frames the r23 corpus boxes at 0.40 have a ghost rate of 0.147, 0.258, 0.151 against
  the sidecar's 0.133, 0.199, 0.130, and the corpus has 0.045, 0.026, 0.031 more boxes per in-frame player game-wide.
  If the returned after-reset boxes carry the corpus rate, ghosts rise by under 0.005; that is the expectation.
  Heavy windows after a reset (1.5 to 3 tracked players per frame as built) could carry more.
- Position p50, p90 and near-field bias: flat to slightly worse. More players match; the new matches are fresh
  tracks whose stabilised foot (EMA 0.6) starts cold, and they sit in the frames after a camera cut. They enter within
  2.5 ft by definition of a match, so p50 should barely move; p90 can rise a little. Bias: no expected direction.
- Team: flat. New tracks after a reset get team labels from the same window clustering; the denominator grows by the
  new matches.
- Coverage: up or flat (more frames with boxes after a reset).
- hou_cle_2016: not estimated (no corpus frames). 23 of its 93 windows hold a reset and 0.42 of its processed frames
  sit in those windows (0.24 after a first reset), more than any pool game, so its effect may be the largest; no
  number is forecast.

Noise floor: the E1R window-interval half-widths on missed are about 0.0052 pooled, 0.0055, 0.0057 and 0.0109 per pool
game, and about 0.013 to 0.015 on hou_cle_2016. The expected pooled effect (0.009 to 0.013) is about two half-widths;
gsw_bkn_2015's (0.015 to 0.022) is three or more; gsw_cha_2016's and gsw_sac_2015's are below their floors and are not
expected to clear zero alone. The upper bound (pooled 0.020) is above the floor, so the arm is worth running.

## 5. Rule (fixed now)

D5 with D158 as registered (a row is worse when its window or its game interval lies wholly on the bad side of zero;
it improves on its point estimate) decides what "worse" and "improves" mean. The registered pool pass ("every row but
coverage improves") is replaced for this arm, with this reason: the fix acts on one cause of the missed row, and the
other rows have no expected improvement (section 4); requiring them to improve has little power and would fail a
correct fix on noise (TRAINING_PROTOCOL step 1; ml-training-rigor item 9). The rows other than missed are held by
non-inferiority margins instead.

Scoring: the pinned score_window (D88), truth camera, foot_stab, through tools.p8_e1_score.window_stats; the pool
through tools.p8_e1_score.report() called unchanged on the three pool score files (rows, diffs, window and game
bootstrap with 2,000 resamples and seed 0, d5_verdict). The arm is "new", the baseline is "r23".

Pool passes when all of these hold:
1. Missed improves pooled (point estimate below zero) and is worse (D158) in no pool game and not pooled.
2. The detection row improves pooled on its point estimate.
3. No row is worse (D158), pooled or in any pool game.
4. Non-inferiority, on the bad-side end of the pooled window interval of the difference (arm minus baseline):
   ghosts at most +0.010; position p50 at most +0.05 ft; p90 at most +0.10 ft; near-field bias at most 0.10 ft in its
   bad direction; team at least -0.03. These are E1b's registered margins (docs/research/P8_E1B.md, untargeted
   rows), about the size of the change the detector device alone made to r23 (M9: ghosts +0.003 to +0.010), also
   used by the foot-position arm (P8_footpos.md); kept for comparability.
5. Coverage holds its floor (as registered).
6. Every check of section 3 passes. A failed check makes the pool partial, and a partial pool never passes.

Labels: if the pool passes and missed's pooled window interval lies wholly below zero, the report says "the fix
reduces missed players on the pool"; if it passes on the point estimate only, "the fix does not harm and points to
fewer misses". gsw_cha_2016's and gsw_sac_2015's own intervals are reported, not required.

hou_cle_2016 gate (the non-GSW test, as E1R stage 2): runs only if the pool passes. Scored as the registered report
scores hou_cle_2016 (window bootstrap only): missed improves on its point estimate; no row worse by its window
interval; the five margins above on its window interval; coverage floor; every check passes. If it passes, the report
says "the fix holds on the new arena"; if not, "the pool result does not carry to the new arena", and the failing row
is named.

Kill rules per stage are in section 8.

## 6. Truth (decided now)

Both truths are scored; the CPU-detector camera truth decides.

- Why: D191 found gsw_cha_2016's truth MOVED on the CPU detector (accepted frames +80 percent) and made every MPS-truth
  analysis provisional where it depends on that truth; the other three games' CPU truth is being rebuilt now. The
  truth rebuild holds the lock for about 7 to 8 more hours and the builds below need the same lock, so no build can
  start before it ends: scoring on MPS truth "now" would not save time.
- Primary: each game on its CPU-detector truth once it has passed acceptance (section 5, check C, of
  P8_truth_arenas.md). Both arms are scored on it, so the baseline rows on that truth are new numbers; this design
  registers that rescore of the r23 baseline for these pairs only.
- Secondary, reported beside it: both arms on the MPS truth, for continuity with every recorded P8 baseline row.
- If a game's CPU truth fails acceptance, the truth in use for that game stays MPS (D191) and that game is scored on
  it, labelled so in the report. If a game's CPU truth is not yet accepted when its builds finish, its score waits;
  there is no switch to MPS to save time.
- If the primary and secondary verdicts disagree, the primary stands and the disagreement is reported with the rows
  that differ.
- The estimate in section 4 used the MPS-truth universe; the CPU truth accepts more frames, so the rates may shift.

## 7. Reset counts change with the fix (handling)

In a3input, variant B (fixed) saw 2 resets on gsw_cha_2016_s05_w00 against 1 as built. Reading, labelled post hoc: as
built, after the first reset few ids confirm (1.5 to 3 tracked players per frame in heavy windows), and the frozen cut
rule needs 4 or more previous ids with 80 percent of them gone, so it rarely fires again; with the fix more ids
confirm, so the rule can fire again at a later real cut. A second reset in the fixed arm is the frozen rule acting on a
working tracker, so it is part of the arm, not a confound. Handling:

- Not a gate and not adjusted for. The frame sets, the court H and the scored frames do not depend on resets, so
  pairing is unaffected.
- Reported per window and per game for both arms: `camera_cut_resets`, the reset log's entry count, and the windows
  where they differ.
- Checked: the reset log has one entry of 2 per reset in the arm's identity file (the fix ran at every reset).
- Reported as a secondary breakdown, not a rule: the rows on the windows where the reset counts differ, and on the
  others.
- A window that has no reset in the baseline cannot gain one in the arm (before the first reset the builds are
  identical), so the built set of section 3 is complete. The equivalence sample tests this.

## 8. Ladder, budget and kill rules

Lock hours at 0.44 s per processed frame on the CPU detector (E1R and M9 build times). The truth rebuild ends first
(about 7 to 8 hours from now). 0 Colab units.

| stage | what | lock hours | gate (fails stop the arm) |
|---|---|---|---|
| 0 | the build and score commands written (tools/p8_resetfix.py build, score, report), tests, guard checks; reviewed | 0 | tests pass; find_private clean; guards refuse spent and test games |
| 1 smoke | build gsw_cha_2016_s05_w00 (the a3input window) and gsw_cha_2016_s00_w00 (no reset) with the fix; score s05_w00 against the baseline | about 0.2 | s00_w00 byte-identical to M9's; s05_w00 device record reset_fix true and detector cpu; reset log 2 entries, each 2; sidecar boxes over the window's 1,053 processed frames equal a3input variant B's output (8,371; variant A's 2,657 equals the stored sidecar's) and the identity file shows 2 resets (B's); rows identical up to the first reset; every pairing check passes |
| 2 pool | the 39 reset windows of the three pool games (14,945 frames) plus 6 equivalence windows; score on both truths as section 6 | about 2.0 to 2.3 build, up to 0.5 scoring | every check; equivalence windows byte-identical; then the pool rule of section 5 |
| 3 hou | the 23 reset windows of hou_cle_2016 (10,076 frames) plus 2 equivalence windows; score | about 1.3 to 1.4 build, up to 0.2 scoring | only if stage 2 passes; the hou gate |

Total about 4.2 to 4.6 lock hours, against about 9.9 for full rebuilds of four games. Scoring time is an allowance;
the smoke measures it.

Kill and fallback rules, fixed now:
- Smoke gate fails: nothing more is built; the failure is reported with the failing file.
- An equivalence window differs from its baseline: the "only reset windows change" claim fails; the arm stops and the
  report says so. The fallback (full four-game rebuilds of both arms, about 20 lock hours, since the baseline itself
  would then not repeat) needs a new dated version of this design, not a silent switch.
- Pool rule fails: hou_cle_2016 is not built. The report names the failing condition and gives the diagnosis of
  section 9.
- No retry on a build crash beyond two attempts per window (as the a3 run); a third failure stops the stage.

No seeds: the builds are deterministic (E1R smoke gate 1), so one build per arm and window.

## 9. What would change the plan

- Pool and hou pass: the fix is proposed as part of the reference build for every later arm (a lead decision and a D
  line); later baselines would be rebuilt with it, and every earlier r23-baseline comparison stays as recorded.
- Missed improves but a margin fails (most likely ghosts or p90): the fix is still a correct tracker, but its returned
  boxes carry a cost; the next step is a measurement of where the new ghosts or position errors sit (heavy windows,
  frames after a cut), not a change of margin.
- Missed does not improve on gsw_bkn_2015, against an expected 0.015 to 0.022: the bound or the mechanism is wrong at
  build level; one cheap measurement first, box yield per window after the first reset in both arms (stored outputs,
  no lock).
- Pool passes, hou fails: the result does not carry to the new arena; report the hou rows by reset and no-reset
  windows.

## 10. What was already seen (forking paths)

- The cause and the fixed yield on gsw_cha_2016_s05_w00 (a3input); that window is in the pool and in the smoke.
- The section 4 rates on the pool games' corpus frames, including missed rows by game (recorded M6 and M9 rows).
- hou_cle_2016: only its reset counts and frame counts (structure, read from identity and sidecar files); no outcome
  on hou was read.
- The 14 windows with a corpus decode offset (D190) do not bear on this arm: both arms are builds scored against
  truth, and the corpus is used only for the section 4 estimate.
- Note (2026-11-26, re-review B3): the decision rule was changed (versions 2.2 and 2.3) after forecasts derived from
  stored MPS-truth rows (sections 13, 17 and 19); no outcome on the rebuilt truth was seen.

## 11. Ledger line (to add when the arm runs)

`resetfix | reset-fix build arm, r23 CPU, reset windows only | <lock hours> | <pool and hou outcome against section 5> |
<lesson>`

Note (2026-11-26, re-review B3): the ledger line will name the rule of section 18 (version 2.3) and say "rule changed
after forecasts derived from stored MPS-truth rows".

## 12. Version 2, part 1 (2026-11-26): two stored-files measurements, registered before any number

The training-design-reviewer returned GO WITH FIXES on e5858f14. Its key finding: camera truth frames come from the
label build's tracker boxes, so the same bug thins truth after a reset. The lead paused the CPU truth rebuild, and the
truth's label builds will be rebuilt with the fix after this arm's stage 0 passes. Before design v2 and before any
lock time, two measurements run on stored files only (no model code, no lock). Both are report only: they feed the
design and change no registered rule. Code: tools/p8_resetfix.py (truthreset, simulate; the estimate now keeps per-window
counts split at the first reset). Records: reports/research/p8_resetfix_truthreset.json and p8_resetfix_simulate.json.

**(a) Truth acceptance before and after the first reset** (review finding 3).
- Units: the MPS truth of gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 and hou_cle_2016 (the P8 workspace's label builds
  and labels), and gsw_cha_2016's CPU truth (the reference root's label builds and labels).
- Per label window with a label-build sidecar: processed frames are the sidecar rows; accepted frames are the rows of
  labels/<game>/<window>.jsonl; the first reset is the first reset-like frame of the label build's sidecar
  (tools.p8_trackreset.reset_like); "after" is that frame and every later one. A window whose identity file counts a
  reset that no reset-like frame locates is counted apart and kept out of the before and after rates.
- Reported per unit: acceptance rate before and after the first reset, pooled over windows with a located reset; their
  ratio; the window-cluster interval of the difference (2,000 resamples, seed 0, drawn as tools.p8_e1_score.boot
  draws); the rate in windows with no reset; the share of processed frames after a reset against the share of
  accepted frames after a reset, and their ratio (the under-representation factor).

**(b) A simulation of the expected change under the v2 rule** (review findings 1, 4 and 5).
- Base: the reference-platform r23 rows of the registered score files on the MPS truth (M6's r23 side for
  gsw_bkn_2015, M9's CPU side for gsw_cha_2016 and gsw_sac_2015), per window, used for both arms. The per-window
  counts of the estimate are recomputed on the same scored frames and must equal the score files' in-frame and missed
  counts per window (checked; the scored frames all have r23 corpus boxes).
- The arm: in each window with a reset (identity camera_cut_resets at least 1), and on frames from its first reset-like
  frame on: missed falls by K ~ Binomial(R, f), R the missed players an r23 corpus box (0.40) reaches within 2.5 ft;
  the detection row's far misses fall by K5 ~ Binomial(R5, f), R5 the far misses an r23 box reaches within 5 ft;
  A ~ Binomial(E, f) non-referee boxes are added, E the r23 corpus boxes minus the sidecar's non-referee boxes on
  those frames (0 if negative), of which G ~ Binomial(A, q) are ghosts, q the game's ghost share of that excess (r23
  corpus ghosts minus sidecar non-referee ghosts, over E, after the first reset, clipped to 0 to 1).
- f is uniform on 0.5 to 0.9 per draw (centre 0.7, the lead's "about 0.7"); 500 draws, seed 0.
- Not simulated, held at zero difference: position p50 and p90, near-field bias and team (the scorer pairs these on
  players matched in both arms, so arm-only matches do not enter them; changes to players matched in both are not
  modelled). The one-to-one missed row is taken equal to missed.
- The rule is design v2's (tools.p8_resetfix.v2_verdict and pool_verdict, written before this run): missed and the
  detection row are judged by D158 "worse" (window or game interval wholly above zero) per game and pooled and must
  improve pooled; ghosts, p50, p90, bias and team only by their margins (+0.010, +0.05 ft, +0.10 ft, +0.10 ft, -0.03)
  on the bad-side end of the window interval, per game and pooled; coverage floor; the one-to-one missed row must
  improve pooled.
- Intervals: for speed, the draws use count-row intervals computed from the same resample indices as
  tools.p8_e1_score.boot. At the centre (f = 0.7, expected counts) the real rows, diffs and boot are run, and the fast
  values must equal them exactly (else the run stops).
- Reported: the pool pass probability, the pooled-only and per-game pass probabilities, the probability that missed's
  pooled window interval lies wholly below zero, how often each condition fails, and forecasts (5th, 50th and 95th
  percentiles) of the pooled missed, detection and ghost differences and of the ghost interval's upper end, pooled
  and by game.
- Limits fixed now: this is a forecast on the MPS-truth scored frames, which the bug thins after resets (part a); a
  truth rebuilt with the fix holds more post-reset frames, so the scored effect can be larger. hou_cle_2016 is not
  simulated (no corpus boxes). The numbers decide nothing by themselves.

## 13. Results of part 1 (2026-11-26, stored files, no lock)

One code fix after registration and before any number of (b): a scored window with no scored frame has no counts in
the estimate's per-window record, which stopped the first simulation run with a KeyError; such a window now counts as
zero (its score-file counts are zero too, checked). The estimate's rerun reproduces every recorded rate of version 1
exactly (same in-frame, missed, ghost and reset counts) and adds the per-window and after-reset fields.

**(a) Truth acceptance before and after the first reset.** Record: reports/research/p8_resetfix_truthreset.json.

| unit | windows with a reset (located) | acceptance before the first reset | after it | before over after | interval of before minus after | no-reset windows | share of processed frames after a reset | share of accepted frames after a reset | under-representation |
|---|---|---|---|---|---|---|---|---|---|
| gsw_bkn_2015, MPS truth | 18 (18) | 0.159 | 0.007 | 22.1 | [0.068, 0.232] | 0.101 | 0.115 | 0.008 | 14.0 |
| gsw_cha_2016, MPS truth | 14 (14) | 0.080 | 0.005 | 17.1 | [0.032, 0.105] | 0.102 | 0.237 | 0.015 | 16.2 |
| gsw_sac_2015, MPS truth | 12 (12) | 0.146 | 0.014 | 10.7 | [0.015, 0.200] | 0.129 | 0.137 | 0.016 | 8.4 |
| hou_cle_2016, MPS truth | 21 (21) | 0.223 | 0.037 | 6.0 | [0.082, 0.301] | 0.239 | 0.251 | 0.050 | 5.0 |
| gsw_cha_2016, CPU truth | 12 (12) | 0.154 | 0.014 | 11.0 | [0.026, 0.243] | 0.162 | 0.170 | 0.017 | 9.7 |

Reading (report only): in every unit, frames after a label build's first reset are accepted as camera truth 6 to 22
times less often than frames before it, and the window intervals lie wholly above zero. Frames after a reset are 5 to
16 times rarer in truth than in the label builds. The rate before a reset is close to the rate in windows with no reset
(except gsw_cha_2016 on MPS, 0.080 against 0.102). Not separated here: after a camera cut the shot can also change to
one where truth is harder (a closer or moving view), so part of the gap may be the scene, not the bug. The fix applied
to the label builds is the test of that.

**(b) Simulation under the v2 rule.** Record: reports/research/p8_resetfix_simulate.json. The centre's fast intervals
equal tools.p8_e1_score's rows, diffs and boot exactly (missed, ghosts and coverage, point, window and game interval;
position, bias and team zero).

| | value |
|---|---|
| pool pass probability (pooled verdict and every game) | 0.93 |
| pooled verdict alone | 1.00 |
| per game: gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 | 1.00, 0.93, 1.00 |
| missed's pooled window interval wholly below zero | 1.00 |
| the only failing condition | gsw_cha_2016 ghosts over its +0.010 margin, 0.07 of draws |
| pooled missed difference (5th, 50th, 95th percentile) | -0.0118, -0.0095, -0.0070 |
| pooled detection-row difference | -0.0125, -0.0100, -0.0074 |
| pooled ghost difference | +0.0009, +0.0013, +0.0018 |
| upper end of the pooled ghost window interval | +0.0019, +0.0029, +0.0039 |
| upper end of the ghost window interval, gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 (median) | +0.0039, +0.0077, +0.0016 |
| missed difference by game (median) | -0.0172, -0.0028, -0.0064 |
| ghost share of the excess r23 boxes after a reset (q) | 0.18, 0.66, 0.19 |

Reading (report only): on the MPS-truth frames, the expected change passes the v2 rule in 93 percent of draws; the
risk is gsw_cha_2016's own ghost margin, because the few boxes the fix returns there after a reset (99 excess boxes
over the game's scored frames) are mostly ghosts. The forecast leaves out changes to position, bias and team, and the
truth thinning of part (a): on a truth rebuilt with the fix, more post-reset frames are scored, so both the missed
change and the ghost change can be larger than forecast.

## 14. Design version 2 (2026-11-26; supersedes sections 2 to 8 where it says so; before any lock time)

Changes from the training-design-reviewer's GO WITH FIXES on e5858f14 and the lead's decisions. Sections 1 to 11 stay
as registered; where this section differs, it rules. Code: tools/p8_resetfix.py, stage 0 commands (plan, build,
smoke-check, equiv, pin-truth, score, report), with tests in tests/test_p8_resetfix.py. Stage 0 is reviewed before any
lock time.

### 14.1 Scope (review findings 3 and 12; replaces the scope paragraph of section 1)

- Any build made with detect/tracker.PlayerTracker under ultralytics 8.4.48 that had at least one reset carries the
  bug (a code fact). That includes the label builds the camera truth comes from: part (a) shows frames after a label
  build's first reset are accepted as truth 6 to 22 times less often, so every truth in use today is thinned where the
  fix acts.
- Whether r23's registered baselines and the recorded held-out numbers had resets, and how much the bug moved them, is
  not tested here. Held-out numbers are not rebuilt or re-scored (CLAUDE.md rule 4); no held-out identity file is read.
  This arm builds and scores the four development games only.
- The development games' resets are known (their identity files were read for the plan and the estimate), so the arm
  is not blind on where it acts. Its outcome rows on the rebuilt truth are unseen.
- Reset timing differs between detectors, so the bug need not cancel in paired comparisons such as E1 against r23.

### 14.2 One root (finding 2; replaces section 3's folders)

Everything is built and scored in one reference root (work_ref of the P8 data folder, `V3_P8_ROOT`), so the clips,
the truth, the scorecard caches and both arms share one place:

1. Truth: the label builds of all four games are rebuilt there with `P8_TRACKER_RESET_FIX=1` (the truth-rebuild plan,
   amended by its owner; it starts only after this arm's smoke passes), then the frozen fit and acceptance
   (tools.p8_truth_accept, check C included). The existing reference-root label builds of gsw_cha_2016 (its accepted
   CPU truth) and hou_cle_2016 (partial) carry the bug; `tools.p8_device.check_cached` refuses to reuse a window
   whose record's reset_fix differs, so that plan moves them aside (moved, never deleted).
2. Baseline: r23 without the fix, as tools.p8_truth_cpu_r23 places it in builds/r23: copies of the P8 workspace's
   reference-platform r23 builds, only for windows whose clip is byte-identical (SHA-256) to the reference root's,
   whose device record asks for the reference platform with no fix, and whose detector record is r23
   (gsw_cha_2016: 73 of 73 already placed). A refused window would need a no-fix r23 build in the reference root; that
   is not in this design and would need a dated amendment.
3. Plan: `plan --game G` reads the reference root's baseline identity files and writes the reset windows (built), the
   equivalence windows (first and last no-reset window by name) and the SHA-256 of every baseline output. The reset
   windows are recomputed there; section 3's table came from the workspace's builds.
4. Arm: builds/resetfix_r23_detcpu__<game>, built by `build` through tools.p8_frozen.safe_build with the arm's own
   child (`P8_DEVICE_DETECTOR=cpu`, `P8_TRACKER_RESET_FIX=1`, r23 weights, r23 court weights as E1R, TRACKER_REID
   unset and refused if set), each window up to three attempts.
5. Caches: the scorecard caches are build-independent (gate v2 wide score on live frames, referee detections on truth
   frames, the cut signal). Before scoring, `cache-copy` copies a window's cache file from the P8 workspace when the
   two clips are byte-identical and the reference root has none (no lock); the score step computes only the entries still missing (mostly referee detections on truth
   frames the new truth adds) and records the seconds per window (cache_seconds).

### 14.3 What changes in a fixed build (finding 7; corrects sections 2, 3 and 7)

The fix changes code only inside PlayerTracker.reset. A window with no reset in the baseline makes the same calls in
the same order, so its outputs are identical (tested by the equivalence windows). In a window with a reset, the
sidecar rows up to and including the frame of the first reset are identical; after it the tracker differs, and every
whole-window output (trajectories, identity, team clusters, any step that reads the whole window) can change, also for
frames before the first reset. Section 2's "only after the first reset" holds for sidecar rows only, and the prefix
check stays on sidecar rows. The check uses the arm's own first reset frame from its reset log, not a reset-like
frame read from ids.

### 14.4 The patch's log (finding 10; adds to section 2)

Each reset appends {"frame", "stripped"} to `<window>_resetfix.json`: the frame index of the update call that
triggered it (a cut or an exception) and the callbacks it removed. The log also records TRACKER_REID (must be unset),
the tracker config file name and the library versions the build process loaded (opencv, torch, ultralytics; must be
4.13.0, 2.2.2, 8.4.48, a3input's). Every window of the arm must show: entries equal to the identity file's
camera_cut_resets, each with a frame and stripped 2; device record reset_fix true on the reference platform; detector
record r23.

### 14.5 Smoke (finding 9; replaces section 8's smoke row)

`build --game gsw_cha_2016 --smoke` builds gsw_cha_2016_s00_w00 (no reset) and gsw_cha_2016_s05_w00 (the a3input
window) with the fix; `smoke-check` passes only if all hold:
- s00_w00's frames, trajectories and identity are byte-identical to the baseline's;
- s05_w00's clip is byte-identical to the workspace clip a3input read;
- on every one of a3input variant B's 1,053 frames, the arm's sidecar holds the same boxes as B's output: same count
  and a one-to-one match with every coordinate within 1 px (B's boxes are rounded to 0.01 px, the sidecar truncates
  to whole pixels), and the frame sets are equal;
- the arm's reset count equals B's (2);
- the records and the log pass 14.4 (versions included);
- s05_w00's sidecar rows match the baseline's up to the arm's first reset frame.
The smoke scores nothing (the truth it would need is not rebuilt yet). About 0.2 lock hours. If it fails, nothing more
is built and truth-rebuild does not start its fixed label builds.

### 14.6 Truth (findings 3 and 11; replaces section 6)

- The decision truth is the camera truth rebuilt with the fix (14.2, step 1). The MPS truth is not scored for this
  arm: part (a) shows it is thinned after resets, the frames the fix changes, and it lives in another root.
- `pin-truth --game G` runs before any score: it refuses unless the game's acceptance record has a final "use"
  ("use", "apart" or "unused" from tools.p8_truth_accept) and every label-build window's device record shows reset_fix
  true with the detector on cpu; it writes the SHA-256 of every label file and the fit result.
- Nothing is scored until every pool game's truth is pinned final; every score checks the pins and refuses changed
  files. Only a game with "use" is scored. A pool game whose truth is "apart" or "unused" leaves the pool partial, and
  a partial pool never passes; fewer than two pool games in use means no pool.
- Claim limit from shared inputs: the label builds use the r23 detector and, now, the fixed tracker, the same as the
  arm. Truth frames after a reset are accepted where the fixed label build has boxes, which are mostly frames where the
  fixed arm has boxes too; that can favour the arm. The bug-carrying truths favoured the baseline the same way.
  Bracket, report only: every pair is also scored on the frames that the workspace's bug-carrying truth also accepted
  (same clip by SHA-256; the new truth's rows for those frames). The arm is judged on all new-truth frames; the
  bracket shows how much of the result rests on frames the fix itself let into truth.

### 14.7 Rule (findings 1 and 6; replaces section 5)

Implemented by tools.p8_resetfix.v2_verdict and pool_verdict (tested); tools.p8_e1_score's report() and its adopt are
not used. Rows, point differences and intervals come from tools.p8_e1_score.rows, diffs and boot (2,000 resamples,
seed 0); the detection and one-to-one rows use the same resample indices.

- Missed and the detection row (far misses: no foot within 5 ft): D158 "worse" when the window or the game interval
  lies wholly above zero, per game and pooled. Pooled, both must improve (point below zero). Per game they need only
  not be worse.
- Ghosts, position p50, p90, near-field bias, team: judged only by their margin, on the bad-side end of the window
  interval, per game and pooled: ghosts at most +0.010, p50 at most +0.05 ft, p90 at most +0.10 ft, bias at most
  +0.10 ft, team at least -0.03. D158 "worse" is not applied to them. No interval fails the row.
- One-to-one missed (a box can find at most one player): must improve pooled (point below zero).
- Coverage: the arm holds the 0.50 floor (or the baseline's value when the baseline is under it), per game and pooled.
- The pool passes when the pooled verdict and every pool game's verdict pass and the pool is not partial.
- Labels: "the fix reduces missed players on the pool" when the pool passes and missed's pooled window interval lies
  wholly below zero; "the fix does not harm and points to fewer misses" when it passes on the point estimate only.
- hou_cle_2016 (the non-GSW gate): built and scored only after the pool passes; the same rule on its window interval,
  with missed required to improve; "the fix holds on the new arena" or "the pool result does not carry to the new
  arena".
- Report only, per game and pooled: boxes, duplicates and ghosts per in-frame player for both arms; the one-to-one
  missed rate; the position error (median, p90, count) of players matched only in the arm, which the paired position
  rows leave out; players matched only in the baseline; reset counts per window in both arms; the bracket of 14.6.

### 14.8 Expected effect (adds to section 4)

Part (b) forecasts, on the MPS-truth frames: pooled missed -0.007 to -0.012, detection row -0.007 to -0.012, ghosts
+0.001 to +0.002, pool pass probability 0.93, the risk gsw_cha_2016's ghost margin. On the truth rebuilt with the fix,
more post-reset frames are scored (part a: 5 to 16 times under-represented today), where the baseline misses most, so
the baseline's missed rate on the new truth is expected to be higher than recorded and the arm's gain larger; the
ghost change can grow the same way. Position rows: arm-only matches do not enter them; their error is reported.

### 14.9 Ladder, budget and kill rules (findings 2, 11 and 13; replaces section 8)

Lock hours at 0.44 s per processed frame on the CPU detector; the truth figures are the truth-rebuild plan's (section
0 of P8_truth_arenas.md and its section 15 run, plus 15 percent), listed because scoring waits for them.

| stage | what | lock hours | gate (a failure stops the arm) |
|---|---|---|---|
| 0 | stage 0 code and tests; reviewed | 0 | tests pass; find_private clean; the review |
| 1 smoke | 14.5 | about 0.2 | 14.5 |
| truth (its own plan) | label builds with the fix for the four games, fit, acceptance | about 11.5 (gsw_cha_2016 2.7, gsw_bkn_2015 2.9, gsw_sac_2015 2.6, hou_cle_2016 3.3) | its own acceptance; final status per game |
| 2 pool builds | the reset windows of the three pool games (39 by the workspace builds, 14,945 frames; recomputed by plan) and 6 equivalence windows | about 2.0 | builds pass 14.4; equiv passes (byte identity, prefix) |
| 2 pool caches and score | cache copies (no lock), missing cache entries, both arms on the pinned truth, bracket | about 0.5 to 1.5 (caches; measured on the first game) plus 0.2 | pins; pairing checks (records, frame sets, court H, in-frame counts) |
| 2 report | the v2 rule | 0 | the rule |
| 3 hou | its reset windows (23 by the workspace build, 10,076 frames) and 2 equivalence windows, caches, score | about 1.3 plus 0.3 to 0.6 | only after the pool passes; the hou gate |

This arm: about 4.5 to 5.8 lock hours, 0 Colab units, after the truth plan's about 11.5. Stage 2's builds need no
truth and may run before or between the truth plan's jobs, one heavy job at a time, as the lead orders them.

Caps and kill rules, fixed now:
- Any stage over 1.5 times its estimate stops before its next job, and the lead decides.
- Cache time: if the first pool game's cache seconds pass 0.75 lock hours, scoring stops and is re-estimated.
- Smoke, equivalence or prefix failure: the arm stops (as section 8; the fallback to full rebuilds needs a new
  version).
- A build that fails three attempts stops the stage.
- Pool rule fails: hou_cle_2016 is not built; the report names the failing condition.
- No seeds: CPU builds repeat byte for byte (E1R smoke gate 1; tested again by the equivalence windows).

### 14.10 What a pass may claim (finding 12)

If the pool and hou pass: "On camera truth rebuilt with the fix for three GSW home games and one HOU game, r23 built
with the tracker reset fixed misses fewer in-frame players than r23 as built, with ghosts, position, bias and team
within their margins." It does not claim: anything about held-out games or the D85 test set; how much the bug moved any
recorded number (they stand as recorded); any detector other than r23; any arena or broadcaster beyond these four
games; that the gain does not rest on frames the fix let into truth (the bracket shows that part).

## 15. Version 2.1 (2026-11-26; the lead's decisions with truth-rebuild, folded into the review fixes)

Two changes, both approved by the lead, to fit the truth plan's check C (P8_truth_arenas.md section 16 version 4):

1. **The unfixed baseline moves to builds/r23_nofix** (replaces 14.2 step 2). `baseline-copy --game G` copies the P8
   workspace's reference-platform r23 builds (tools.p8_placement.BUILDS) into the reference root's builds/r23_nofix
   for every label window (one per time map) that passes the rules of tools.p8_truth_cpu_r23: built from the
   workspace clip, that clip byte-identical to the reference root's, device record on the reference platform with no
   clip fix and no reset fix, detector record r23 (gate-empty windows: clips byte-identical). It writes
   builds/r23_nofix_source_<game>.json; `plan` refuses a game whose baseline is not complete. Nothing is built for the
   baseline. builds/r23 becomes check C's folder, composed by the truth plan's own step (tools.p8_truth_cpu_r23
   compose, P8_truth_arenas.md section 16 version 4, r-p8-truth-cpu cf840892), per game and only after this arm's
   resetfix/equiv__<game>.json passes: the arm's build for each window in resetfix/plan__<game>.json "reset_windows"
   (a list of window stems), the workspace's reference-platform r23 build for every other window. Nothing in this arm
   writes builds/r23. The acceptance code (tools.p8_truth_v2 measure-dev,
   tools.p8_truth_accept) stays unchanged and reads builds/r23 as before.
2. **hou_cle_2016's builds run before the pool verdict** (changes 14.9's stage 3). Check C on hou_cle_2016 needs its
   fixed reset-window builds. Building them reads no outcome; hou scoring (and so any hou row) stays gated on a pool
   pass, enforced in `score`. `stage_allowed` now gates hou builds on the smoke only and hou scoring on the pool.

Order (one heavy job at a time, each under its own lock): (1) baseline-copy gsw_cha_2016, then the smoke (14.5);
(2) the truth plan's label builds with the fix and fits, four games; (3) baseline-copy, plan, arm builds and equiv for
the three pool games and hou_cle_2016 (they need no truth, so they may interleave with step 2 once a game's label
clips and time maps exist), then the truth plan's compose per game; (4) the truth plan's check C and acceptance; (5) pin-truth, cache-copy and scoring of the pool, then the report;
(6) hou scoring only if the pool passes. The budget of 14.9 is unchanged: hou's builds move earlier, they are not
added.

## 16. Version 2.2 (2026-11-26; the review of 84440284 with version 2.1: GO WITH FIXES; registered before the rerun)

Fixes A1 to A7 and the lead's decisions. Where this section differs from sections 14 and 15, it rules.

**A1, sidecar identity (blocked the smoke).** build_trajectories.py stores the absolute clip path in frames.json as
"source", and the baseline (built from the workspace clip) and the arm (built from the reference root's copy) differ
there by construction. compare_window now compares frames.json as canonical JSON with "source" left out
(trajectories and identity stay byte for byte; neither holds a path). Separately, every arm window must name the
reference root's clip as its source, and that clip must hash as the workspace clip (arm_source_problem; in build,
equiv, the smoke and score). Checked on stored files: gsw_cha_2016_s00_w00's workspace sidecar and the reference
root's copy with its source swapped compare equal (435 rows), and a one-box change makes them differ.

**A2, the bracket becomes a guard (lead decision).** The fixed label builds and the arm share detector and tracker
output (only court and gate settings differ), so the new truth selects frames by the arm's own boxes.
- The pool passes only if, on the frames both truths accepted (the workspace's bug-carrying truth and the fixed
  truth), the pooled missed point estimate is also below zero.
- The label "the fix reduces missed players on the pool" also needs the bracket's pooled missed window interval
  wholly below zero.
- Report only: per window, the arm's sidecar boxes (bbox and track id) against the fixed label build's on frames both
  processed (label_build_consistency).
- Scope: the earlier r23 rows on the CPU truth were also scored on truth selected by r23's own boxes (the shared
  input noted in P8_truth_arenas.md section 15); this is not new to this arm.

**A3, simulation rerun and adoption rule (lead decision, registered now).**
- Rerun (stored files, no lock; `simulate --version 2.2`, record reports/research/p8_resetfix_simulate_v22.json):
  as part 1 (b), except (i) in each window with a reset, every count after the first reset is taken u times, u the
  game's under-representation factor of part (a) on the MPS truth (14.0, 16.2, 8.4): the baseline gains (u - 1) times
  its after-reset counts (in-frame, missed, far, ghosts, boxes), and the counts the arm acts on (R, R5, E and the
  excess ghosts) become u times; (ii) q is drawn per draw and game from Beta(g + 1, E - g + 1), g and E the game's
  excess ghosts and excess boxes after the first reset (unscaled); the centre uses its mean. 500 draws, seed 0; the
  centre is checked against the real rows, diffs and boot as before. Reported: the pass probability, the adoption
  probability under the rule below (the bracket guard taken as passing, since it cannot be simulated), how often each
  condition fails, and the gsw_cha_2016 ghost forecast.
- Adoption rule: if the pool passes every condition except one or more per-game ghost margins, the fix is adopted as
  a correctness fix: it removes a bug and returns real detector boxes, so the rebuilt pipeline is the reference from
  then on. The claim "no harm on ghosts" is then not made, and the ghost cost is reported per game. The fix is not
  adopted if pooled missed fails, the bracket guard fails, any pooled margin fails, or any other condition fails.
  (tools.p8_resetfix.adoption; tested.) hou_cle_2016 scoring stays gated on a full pool pass.

**A4, truth pin.** The pin also covers labels/<game>_offsets.json. It refuses unless the acceptance record, the fit
result, the offsets and every label file post-date every fixed label build (file modification times), and the
acceptance record's count of label windows equals the fixed label windows (pin_order_problems).

**A5, clips.** plan records every label window's clip SHA-256; build, equiv and score refuse a window whose clip
changed.

**A6, check C and budget.** truth-rebuild's supersede step moves the 73 unfixed gsw_cha_2016 copies out of
builds/r23 before its compose. hou_cle_2016's reset-window builds are now unconditional (about 1.3 lock hours plus 2
equivalence windows, 0.06), so the arm's builds are about 3.3 to 3.5 lock hours whatever the pool shows; with the
smoke (0.2), the caches (0.5 to 1.5 for the pool; hou's 0.3 to 0.6 only after a pool pass) and the scoring (0.2),
about 4.2 to 5.8 lock hours.

**A7.**
- A no-reset window (the arm resolves to the baseline) is scored once and its result reused for both sides.
- Stage 0 is gated on the real-ultralytics check passing under the frozen CV interpreter, not skipped:
  `stage0-check` (imports only, no model, no lock) checks that the fix recognises ultralytics 8.4.48's own tracker
  callbacks, that the versions are the pinned ones, that PlayerTracker has reset and update, that the weights are
  r23 and that TRACKER_REID is unset; it writes resetfix/stage0.json, and the smoke refuses to start without it.
- The tests are rerun on the committed head.

## 17. Result of the version 2.2 simulation (2026-11-26, stored files, no lock)

Record: reports/research/p8_resetfix_simulate_v22.json. The centre's fast intervals equal the real rows, diffs and
boot (checked). As registered in section 16 (A3):

| | value |
|---|---|
| pool pass probability | 0.00 |
| adoption probability (bracket guard taken as passing) | 0.00 |
| per game pass: gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 | 0.004, 0.00, 0.43 |
| failing conditions (share of draws) | pooled ghost margin 1.00; gsw_cha_2016 ghost 1.00; gsw_bkn_2015 ghost 1.00; gsw_sac_2015 ghost 0.57 |
| pooled missed difference (5th, 50th, 95th) | -0.108, -0.087, -0.064 |
| pooled ghost difference | +0.010, +0.014, +0.019 |
| upper end of the pooled ghost window interval | +0.023, +0.030, +0.038 (margin +0.010) |
| gsw_cha_2016 ghost window upper end | +0.068, +0.091, +0.117 |
| gsw_cha_2016 ghost difference at the centre | +0.036, interval [+0.001, +0.090] |
| q (Beta mean): gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 | 0.18, 0.65, 0.19 |

Reading, as registered: if the truth rebuilt with the fix lets post-reset frames in at the rates part (a) implies, the
arm's missed gain is large (pooled about -0.09) but its ghost cost breaks the pooled ghost margin in every draw, so
under section 16's rule the fix would not be adopted. The ghost cost comes from the r23 boxes the bug now drops after
a reset: a fifth of the excess on gsw_bkn_2015 and gsw_sac_2015, two thirds on gsw_cha_2016, are more than 3 ft from
every SportVU player.

Post hoc, not ruled (labelled as such): the registered scaling multiplies the scorer universe's after-reset counts by
the label builds' under-representation factor, and the two measure different things (the r23 build's resets against
the label build's, players against frames). After scaling, the share of in-frame players after a first reset becomes
0.46, 0.16 and 0.15 (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015), against 0.13, 0.17 and 0.11 of processed frames after
a first reset in the r23 builds. So the scaling overshoots on gsw_bkn_2015 by about 3.5 times and is close on the
other two; gsw_cha_2016's ghost failure does not depend on the overshoot. A scaling to the processed-frame share is a
possible sensitivity check; it is not run, since it was not registered.

## 18. Version 2.3 (2026-11-26; the lead's decision, registered before any build or score and before the reruns)

The lead weighed three options after section 17: keep the rule (the fix would be rejected for returning real
detector boxes, some of them ghosts), widen the ghost margin (arbitrary), or judge a bug fix by net error (chosen).
Reason: the fix restores what the pipeline was designed to do; the bug suppressed boxes by accident. For Level 2 a
missed player and a ghost box are two errors that count the same. This is a design change made before any outcome on
truth; the rule changed after forecasts derived from stored MPS-truth rows (wording of the 2026-11-26 re-review). It
replaces sections 14.7 and 16 (A3) where they differ; everything else stands.

**Rule (tools.p8_resetfix.v23_verdict and v23_pool; tested).**
- Net error E = (missed players + non-referee ghost boxes) / in-frame players: two counts over one denominator
  (missed as the scorer counts it, ghosts as the ghost row's numerator). Window and game intervals use the same
  resample indices as tools.p8_e1_score.boot.
- Pooled: E improves with its window interval wholly below zero; missed improves with its window interval wholly below
  zero; one-to-one missed improves (point below zero).
- Bracket guard: on frames both truths accepted, the pooled missed and the pooled E point estimates are below zero.
- Position p50, p90, near-field bias and team keep their margins (+0.05 ft, +0.10 ft, +0.10 ft, -0.03; bad-side end of
  the window interval), per game and pooled. Coverage keeps its floor, per game and pooled.
- Per game (my reading, stated so the re-review can check it): E and missed must not be worse (window interval not
  wholly above zero); no improvement is required of a single game.
- Ghosts (the ghost row, per game and pooled) and the detection row are reported with their intervals as a known
  cost and are not gated. Any claim of "no ghost harm" is withdrawn.
- Adoption is the pool pass; section 16's correctness-fix branch is superseded.
- hou_cle_2016 is scored only after a pool pass (as version 2.2), and then judged on its window interval: E and missed
  improve on the point estimate and are not worse; position, bias and team within their margins; coverage floor.
- Labels: a pass reads "the fix lowers net error and missed players on the pool; ghosts are reported as a cost",
  with a note when the bracket's missed interval does not lie wholly below zero.

**Simulation reruns under version 2.3 (stored files, no lock), both registered now.** As section 16 (A3) in every
other respect (500 draws, seed 0, f uniform on 0.5 to 0.9, q per draw from the game's Beta, the bracket guard taken as
passing, the centre checked against the real rows, diffs and boot):
1. Registered scaling: the after-reset counts taken u times, u the under-representation factor of part (a)
   (`simulate --version 2.3 --scaling under`, record reports/research/p8_resetfix_simulate_v23_under.json).
2. Sensitivity: u chosen per game so that the after-reset share of in-frame players equals the share of processed
   frames after a first reset in the r23 builds, u = p (1 - a) / (a (1 - p)) with a the after-reset share of in-frame
   players and p the after-reset share of processed frames (`--scaling processed`, record
   p8_resetfix_simulate_v23_processed.json).
Reported for each: the pass probability, the forecast of the pooled E difference and of its window interval's upper
end, E by game, missed, and the ghost cost. Neither run decides anything by itself.

**Notes (2026-11-26, after the version 2.3 simulations; nothing else in this section changed).** Lead decision: the
per-game condition stays (E and missed not worse in any game), because dropping it after a forecast showed it binds
would tailor the rule to the forecast, and "not worse in any game" is this project's standard (D158, E1R's all-folds
gate); if the fix worsens one game's net error, it is not yet known to help. Citation fix: the compose step in force
is r-p8-truth-cpu 55c94cc8 (the same code as 29b97e62), not cf840892 as section 15 says.

**Next work item, named and not run.** After a reset, many of r23's extra boxes are more than 3 ft from every
SportVU player (about two thirds on gsw_cha_2016, a fifth on the other two). Are they non-court shots that gate v2
lets through, the bench, or fans? That becomes its own stored-files study, registered after this arm.

## 19. Results of the version 2.3 simulations (2026-11-26, stored files, no lock)

Records: reports/research/p8_resetfix_simulate_v23_under.json and p8_resetfix_simulate_v23_processed.json. The
centres' fast intervals equal the real rows, diffs and boot (checked in both runs). Scaling factors u (gsw_bkn_2015,
gsw_cha_2016, gsw_sac_2015): registered 14.0, 16.2, 8.4; sensitivity 2.4, 18.0, 5.7.

| | registered scaling | sensitivity (processed share) |
|---|---|---|
| pool pass probability | 0.09 | 0.07 |
| per game pass: gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 | 1.00, 0.09, 1.00 | 1.00, 0.09, 1.00 |
| failing conditions (share of draws) | gsw_cha_2016 net error worse 0.91; pooled net error interval not wholly below zero 0.02 | gsw_cha_2016 net error worse 0.91; pooled net error interval not wholly below zero 0.62 |
| pooled net error difference (5th, 50th, 95th) | -0.073, -0.059, -0.043 | -0.022, -0.017, -0.013 |
| upper end of its pooled window interval | -0.005, -0.003, -0.001 | -0.002, +0.000, +0.003 |
| net error by game (median): gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 | -0.105, +0.010, -0.035 | -0.030, +0.011, -0.025 |
| pooled missed difference (median) | -0.087 | -0.037 |
| pooled ghost difference (median), its upper end (median) | +0.014, +0.030 | +0.013, +0.033 |
| gsw_cha_2016 ghost upper end (median) | +0.091 | +0.097 |

Reading (report only): under the net-error rule the pooled missed and net-error gains are large under the registered
scaling, but gsw_cha_2016's own net error gets worse in about 91 percent of draws under either scaling, because there
the boxes the fix returns after a reset are about two thirds ghosts, so it adds more ghost boxes than it finds missed
players. That one condition (section 18's per-game "not worse", which I wrote as my reading of the lead's rule)
decides the forecast: without it, the pool would pass in about 0.98 of draws under the registered scaling and about
0.38 under the sensitivity. The ghost source (section 18's next item) bears directly on it.

## 20. Note (2026-11-26, the lead's decision, self-decided; PLAN.md D193): what this arm decides

The tracker bug is a defect, not a design choice. From now on every development arm builds both its baseline and its
candidate with the fixed tracker (P8_TRACKER_RESET_FIX=1), matching the truth rebuilt on the fixed tracker. This arm's
verdict decides one thing only: whether the fixed r23 pipeline replaces r23's registered rows as the reported
reference baseline. If the pool passes (section 18), it does. If it fails, r23's registered rows stay as recorded and
are reported as carrying the bug, the fixed pipeline stays the development standard, and the net-error loss on
gsw_cha_2016 is the ghost-source study's question (docs/research/P8_ghostsrc.md on r-p8-ghostsrc).

Also from the re-review: B1, a built window's source problem is no longer overwritten by its record problems
(arm_window_problems; tested); B4, the simulation's centre check also checks the net-error row against a plain loop
over the same resample indices.

## 21. Version 2.4 (2026-11-26; after the first smoke and the truth-cpu review; before any score)

**The first smoke (2026-11-26, 598 lock seconds) failed by its registered gate.** smoke-check: stage0_passed,
arm_sources, no_reset_byte_identical, clip_byte_identical and every_frame_matches_B (1,053 of 1,053 frames, 8,371
boxes in both) passed; resets_equal_B, records_pass and prefix_equal failed, all because gsw_cha_2016_s05_w00's reset
log was empty while its identity file counts 2 resets. Cause (code reading; post hoc until the rerun confirms it): the
build child runs tools/p8_resetfix.py as __main__, and tools.p8_device imports it again as tools.p8_resetfix, so the
patched reset appended to one module's log while the exit writer, registered from __main__, wrote the other's. Fix:
the log, its start and its writer all go through the canonical tools.p8_resetfix module (_log); a test loads a second
copy of the module, starts the log there and checks that the written log holds the resets the canonical patch
recorded. The smoke's two windows are moved aside (not deleted) before a rerun, on the lead's go; the rerun is retry
1 of 2 (lead). The truth's label builds take another path (`python -m tools.p8_device`, which imports this module
canonically); a test runs tools.p8_device.main from a second copy of p8_device with a stand-in build that resets twice
and checks that the written log holds both resets. The failed smoke is also in TRAINING_PROTOCOL.md's ledger.

**Truth pin checks the label builds' reset logs (truth-cpu review).** The label device record is written before the
build starts, so label_fix_problems also requires, per label window: <window>_resetfix.json, TRACKER_REID unset, the
pinned versions, a frame and stripped 2 on every reset, and as many resets as the identity file's camera_cut_resets
(log_problems, the same rules as the arm's minus the detector record). Tested.

**One-to-one net error, report only (lead's decision).** Per scored frame, non-referee boxes (the scorer's referee
rule) and the 10 players are matched one to one within 2.5 ft (Hungarian, by distance); the count is unmatched
in-frame players plus unmatched non-referee boxes, divided by in-frame players (a box on a player just outside the
frame margin is matched, not a false box). Reported pooled and per game with window intervals (and the game interval
pooled), beside E; it does not gate. It gives the double-count-free view for the write-up and is the candidate primary
metric for future arms, to be registered in their own designs. Section 18's gate stands as registered.

**Smoke retry 1 (2026-11-26, 601 lock seconds, peak 1.0 GB): passes.** Every gate of 14.5 holds: stage0_passed,
arm_sources, no_reset_byte_identical (gsw_cha_2016_s00_w00 equals the r23_nofix baseline), clip_byte_identical,
every_frame_matches_B (1,053 of 1,053 frames, 8,371 boxes in both), resets_equal_B (2 and 2), records_pass and
prefix_equal. gsw_cha_2016_s05_w00's reset log holds 2 resets (frames 312 and 6,534, each stripped 2; the first is
the reset a3input found at frame 312); s00_w00's holds none. This confirms the cause of the first smoke's failure (the
empty log of the second module copy). Record: <reference root>/resetfix/smoke_check.json.


## 22. Result page template (2026-11-26, before any score)

`python3 -m tools.p8_resetfix render` writes the result section from the report record alone
(reports/research/p8_resetfix_result.md; every number comes from resetfix/report.json, none is typed): the pool
verdict, adoption, label, failing conditions and bracket guard; the pooled rows (E, missed, one-to-one missed, the
margin rows and coverage, then the reported rows: ghosts, the detection row, one-to-one net error) with baseline, arm,
difference, window and game intervals and the gate; the same per game; the ghost cost per game and pooled; the
bracket's rows; the report-only rows (boxes, duplicates and ghosts per in-frame player for both arms, arm-only matches
and their error, matched only in the baseline, label-build consistency, reset counts by game); hou_cle_2016's section
(or "not scored" before a pool pass); and a closing part written by hand afterwards (against the forecasts of
sections 17 and 19, diagnosis by cause, the lesson, the ledger row), which quotes only numbers that are in a record.
Tested on a synthetic record.

## 23. Note (2026-11-27, before any score; the lead's decision, self-decided): gsw_cha_2016's fixed truth fails acceptance

truth-rebuild reports that gsw_cha_2016's camera truth rebuilt with the fix fails P8_truth_arenas.md section 5's checks
A (median reprojection 15.18 px against a bar of 10.0) and B (971 accepted frames against a bar of 1,000), so its final
status will be "unused". The lead weighed stopping the arm's builds (saving about 2 lock hours) against continuing,
and decided to continue: the fixed builds of gsw_cha_2016, gsw_bkn_2015 and gsw_sac_2015 and their equivalence checks
go ahead as planned, because D193 needs fixed-tracker builds of both sides for every later arm. Registered now, before
any score:

1. Once the accept phase confirms gsw_cha_2016's status as "unused", this arm's verdict is fixed by the rule of
   sections 14.6 and 18: "no pass: partial pool (gsw_cha_2016 truth unused)". A partial pool never passes, so the fixed
   r23 pipeline does not replace r23's registered rows as the reported reference baseline (section 20).
2. gsw_bkn_2015 and gsw_sac_2015, if their truth is accepted with "use", are pinned and scored as registered
   (pin-truth, cache-copy, score, report and render unchanged), and their rows are reported only: they decide nothing.
3. hou_cle_2016 stays unscored (it is scored only after a pool pass).

The code already enforces all three: score refuses a game whose pinned status is not "use"; report marks the pool
partial when a pool game's truth is not in use and sets the pass to false; hou scoring needs a pool pass in
resetfix/report.json.

**Addendum (2026-11-27, before any score; the lead's record of the final statuses).** The accept phase gives
(work_ref/truth_accept): gsw_cha_2016 "unused" (fails A and B); gsw_bkn_2015 "use" (C: median 0.48 s, share over
0.2 s 0.688, n share 77 of 83); gsw_sac_2015 "apart" (C fails: median 0.64 s against the bar 0.54, share over 0.2 s
0.822 against 0.7945). Only one pool game is in use, and under section 14.6 fewer than two pool games in use means no
pool. The arm's verdict is therefore fixed as "no pass: no pool (one pool game in use; cha unused, sac apart)". The
fixed r23 pipeline does not replace r23's registered rows as the reported baseline (section 20); it stays the
development standard (D193). gsw_bkn_2015 alone is pinned and scored as registered, reported only, with the bracket and
the report-only rows; gsw_sac_2015 is not scored (its truth is "apart"), and hou_cle_2016 stays unscored.

Two mechanics, registered with this addendum: (1) score's gate requires every pool game's truth to be pinned final, so
pin-truth is also run for gsw_cha_2016 and gsw_sac_2015 to record their final non-use statuses and file hashes; nothing
of theirs is scored. (2) report had no path for fewer than two pool games in use (it stopped); it now writes a no-pool
record (no_pool_report): the verdict text above, adoption false, and for the game in use its rows, its version 2.3
per-game reading (deciding nothing), the bracket and the report-only rows; render prints the verdict text first.
Tested on a synthetic record with one game in use.

## 24. Result (2026-11-27): no pass, no pool; gsw_bkn_2015 reported only

The generated result page is reports/research/p8_resetfix_result.md (from the reference root's resetfix/report.json,
copied to reports/research/p8_resetfix.json); every number below is in those records or in the score file's record.

**What ran.** Pins: gsw_bkn_2015 "use" (83 label windows with the fix, checks A, B and C pass), gsw_cha_2016 "unused"
(A and B fail) and gsw_sac_2015 "apart" (C fails), each with its label builds' device records and reset logs checked
(v2.4). Cache copies: 83 of 83 gsw_bkn_2015 windows from the workspace (same clips), so no cache entry was computed
(cache seconds 0). Score: gsw_bkn_2015 only, 146 s (the score record's seconds), 83 windows, all also scored on
the bracket's frames. Builds before it: 1.80 lock hours for the three pool games (section 23's chain), smoke 0.33.

**Verdict, fixed in advance (section 23 addendum): no pass: no pool (1 pool game in use; cha unused, sac apart).**
Not adopted; r23's registered rows stay the reported baseline and carry the bug; the fixed pipeline stays the
development standard (D193). hou_cle_2016 is not scored.

**gsw_bkn_2015, reported only** (arm minus baseline on the truth rebuilt with the fix; window intervals): net error E
0.428 to 0.379, -0.049 [-0.087, -0.012]; missed 0.313 to 0.253, -0.060 [-0.107, -0.016]; one-to-one missed -0.060
[-0.108, -0.016]; one-to-one net error -0.042 [-0.076, -0.010]; detection row -0.067 [-0.120, -0.017]; ghosts +0.0008
[-0.0024, +0.0033]; position p50 -0.002 and p90 -0.006 ft, bias about 0, team +0.007 [-0.003, +0.019]; coverage equal.
Its version 2.3 per-game reading passes every condition; it decides nothing. The truth was built from the same fixed
label builds as the arm (detector and tracker shared; sidecar boxes equal on all 20,644 shared frames), which can
favour the arm (section 14.6). The whole difference sits in 9 of the 83 windows (windows with a nonzero difference in
missed or ghosts).

**Bracket** (the frames the bug-carrying workspace truth also accepted: 13,769 of the 28,554 in-frame players): E
-0.020 [-0.061, 0.000], missed -0.023 [-0.071, 0.000], ghosts -0.0002, one-to-one net error -0.017 [-0.052, 0.000].
On these frames, which a truth built on the buggy tracker selected, the point estimates still improve, with
intervals that reach zero (windows with a nonzero difference: 9 of 83 on the full truth, 2 of 83 on the bracket).

**Report only.** Boxes per in-frame player 0.836 to 0.914, ghosts per in-frame player 0.115 to 0.126, duplicates
0.044 to 0.050; 2,101 players are matched only in the arm (their error p50 1.59 ft, p90 3.45 ft, close to the
baseline's 1.58 and 3.41 on players matched in both); 32 only in the baseline. The arm's sidecar boxes equal the fixed
label build's on all 20,644 shared frames, which confirms the shared input of section 16 (A2): the new truth selects
frames by the arm's own boxes.

**Against the forecast (section 19).** The version 2.3 simulations forecast gsw_bkn_2015's E change at a median of
-0.105 (registered scaling) and -0.030 (processed-share sensitivity), and missed at -0.136 and -0.039. The observed
-0.049 and -0.060 lie between the two, nearer the sensitivity, as section 17's post hoc note on the registered
scaling's overshoot on gsw_bkn_2015 suggested. gsw_bkn_2015's own ghost forecasts were a centre
difference of +0.0128 (registered scaling) and +0.0031 (sensitivity), with window upper ends at a median of +0.0251
and +0.0083 (reports/research/p8_resetfix_simulate_v23_under.json and _processed.json); the ghost row came out at
+0.0008 [-0.0024, +0.0033], below both. Post hoc, not registered: the r23 boxes that the fixed truth lets in after a
reset may be mostly players, since the truth accepts frames whose boxes fit the court.

**Lesson.** A truth built from the same detector and tracker as the arm must pass acceptance on its own before an arm
can be judged on it: two of three pool games' fixed truths failed acceptance here (one on registration and coverage,
one on offset stability), which left no pool. The fix itself is confirmed in the build (smoke, equivalence) and, on
the one game in use, lowers missed and net error; the ghost share per box is unchanged (+0.0008 [-0.0024, +0.0033]),
while ghost boxes per in-frame player rise from 0.115 to 0.126 (report only, point values), and net error still falls.
That is a report, not a verdict.

Review notes (2026-11-27, the ml-review of d2e7638f..8b83899e, FAIL on wording): "without a ghost cost" withdrawn (section
18 withdrew any claim of no ghost harm) in favour of the per-box and per-player ghost figures; the forecast comparison
now uses gsw_bkn_2015's own ghost forecasts; the shared-input sentence and the counts of windows with a nonzero
difference (now in the report record, windows_differing) are added; the lock time and peak memory, which came from the
heavy-job log rather than a record, are dropped; render prints the per-game reading as deciding nothing and points to
this section; report only reports games whose pinned truth is "use".


## 25. Note (2026-11-27, the lead's decision, self-decided; before running): hou_cle_2016 builds for truth acceptance only

hou_cle_2016's fixed truth is pending only because check C needs its reset-window composite (builds/r23, composed by
the truth plan from this arm's fixed builds), and that needs hou_cle_2016's reset-fix builds and equivalence check. So
baseline-copy, plan, build (under the lock, named "resetfix truth-only build hou_cle_2016") and equiv run for
hou_cle_2016 alone, as one chain, for truth acceptance only. hou_cle_2016 is not pinned or scored for this arm: the
arm is closed (D198, section 24) and its hou stage stays unrun. Expected from the workspace builds: 23 reset windows,
10,076 frames, about 1.3 lock hours (cap 1.5 times).

**Result (2026-11-27, truth acceptance only).** baseline-copy 93 of 93; plan 23 reset windows and 10,076 frames (as
expected) plus 2 equivalence windows (336 frames). Build: 25 windows, all on the first attempt, 4,384 window seconds
(1.22 lock hours) against about 4,680 (1.3 h), ratio 0.94. Equiv passes: both equivalence windows (s00_w02, s10_w09)
identical to the r23_nofix baseline, every reset window passes the prefix, source and clip checks. Resets, baseline
against arm: 23 against 25 (s03_w01 and s08_w00 each 1 to 2). Records: <reference root>/resetfix/build__hou_cle_2016.json
and equiv__hou_cle_2016.json. Nothing of hou_cle_2016 is pinned or scored for this arm.
