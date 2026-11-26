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

## 11. Ledger line (to add when the arm runs)

`resetfix | reset-fix build arm, r23 CPU, reset windows only | <lock hours> | <pool and hou outcome against section 5> |
<lesson>`

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
