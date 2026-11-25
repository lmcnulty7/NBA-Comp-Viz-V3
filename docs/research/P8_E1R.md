# P8 E1R: E1 seed 0 on the reference platform (design, before any build)

Version 2, 2026-11-25, after training-design-reviewer returned GO WITH FIXES on version 1 (commit 989da1fb). Version 1
stays in git history; the changes are listed at the end. When this version was committed, no score-m9 output existed
and none had been read (the M9 r23 builds were still running; no e1/m9__*.json file had been written).

Evaluation only: no training, no Colab, no units. Nothing is built for E1R until the lead confirms this version after
the reviewer's second look. The D171 verdict stands whatever E1R shows.

## Question

Does E1 seed 0 pass the registered rule against r23 when both run on the reference platform of D175 (player detector
and tracker on CPU, court and gate as registered)? The registered E1 result (D171) was scored on MPS builds, which
D174 found carry a box fault (ultralytics clip_boxes on MPS on this Mac).

## What E1R can and cannot decide

- Only seed 0 exists. D171 stopped seeds 1 and 2, and D85 (3) asks 3 seeds for any trained model: tools/p8_e1_score.py
  adopts E1 only when all 3 seeds are scored and each seed's pool adopts. So E1R cannot adopt E1, and the registered
  report's verdict field will read "1 of 3 seeds", adopt false.
- The claim E1R can make is exactly "seed 0 passes on the reference platform", or that it does not. It never says
  "E1 beats r23", and it claims no mechanism: on MPS the rises on gsw_cha_2016 and gsw_sac_2015 were broad, not one
  window, and M7 predicts MPS would, if anything, favour E1 (MPS lost more of r23's raw boxes than E1's).
- Written now: a pass alone does not justify spending about 30 units on seeds 1 and 2. That spend would be weighed
  against a new r23-initialised arm (E1b, docs/research/P8_E1.md post-mortem implications). If E1R fails, E1 as built
  is closed on both platforms.

## Arms and builds

- Reference platform: P8_DEVICE_DETECTOR=cpu (tools/p8_device.py), court and gate on auto (MPS), P8_MPS_CLIP_FIX
  unset; the frozen build path of tools.p8_frozen.safe_build with the gate v2 arguments and court model r23.
- r23 arm: the production detector (sha256 4c886d5f...). Its folders resolve only to builds/e1m6_r23_detcpu
  (gsw_bkn_2015, M6) and builds/m9_r23_detcpu__<game> (gsw_cha_2016, gsw_sac_2015, hou_cle_2016, M9), never to
  builds/r23.
- E1 arm: the seed 0 best.pt of the fold named after the game it left out: on a pool game, check_provenance(game,
  game, 0, weights_of(game)), so a fold is never built on its own training games. Folders:
  builds/e1m6_e1_gsw_bkn_2015_detcpu (M6) and builds/e1r_e1_<game>_detcpu for gsw_cha_2016 and gsw_sac_2015; for
  Stage 2, builds/e1r_e1_<fold>_detcpu__hou_cle_2016 for each fold. The build child records the weights' sha256, the
  model, the fold and the seed in each window's detector record, and the device record as tools/p8_device writes it.
- Windows: the label windows of each game with a registered r23 sidecar: 83, 73 and 82 on the pool games (238) and 93
  on hou_cle_2016.
- New code: tools/p8_e1r.py (build, score, report) with tests, written and reviewed before any build. Nothing
  registered is written; E1R's score files are <fold>__seed0__<game>.json in the P8 workspace's e1r/ folder, which
  tools.p8_e1_score report never reads.

## Checks on every scored window (the score stops on any failure)

- Both arms: the device record resolves the detector to cpu; the detector record's sha256 is r23's (4c886d5f1e60...)
  on the r23 arm and, on the E1 arm, the sha256 in that fold's complete.json (gsw_bkn_2015 d4ffed33564b...,
  gsw_cha_2016 0579a034..., gsw_sac_2015 bdad8b76...); on E1R's own builds the recorded fold is the expected one.
- Equal processed frame sets between the arms, and in every frame an identical court H (the court runs on MPS in both
  arms and is deterministic and independent of the detector, so a difference flags drift). This is stricter than the
  95 percent shared-frames kill rule of version 1, which it replaces for E1R pairs.
- Equal gate-empty window sets between the arms. If they differ, the pool is marked partial and never passes
  (p8_e1_score.complete() would otherwise drop such a window from both arms silently).
- Printed per game and arm: in-frame players, boxes (non-referee), duplicates (in-frame players with two or more box
  feet within 2.5 ft, on the scored frames), team rows (n), the paired matched-player count, and frame_sets_differ
  (0 by the check above).

## Rule (registered, unchanged) and the claim

- Scoring: the pinned score_window (D88), truth camera, foot_stab, through tools.p8_e1_score.window_stats.
- Primary verdict: tools.p8_e1_score.report() called unchanged on E1R's three score files, so rows, boot (window and
  game bootstrap, 2,000 resamples, seed 0) and d5_verdict are the registered ones: a row is worse when its window or
  its game interval lies wholly on the bad side of zero, improves on the point estimate; the pool passes when every
  row but coverage improves, none is worse and coverage holds its floor; a partial pool never passes. The game
  interval with 3 games is about the range of the three per-game values.
- Blind verdict, fixed now: on the two-game pool of gsw_cha_2016 and gsw_sac_2015, d5_verdict(point_blind,
  wiv_blind, coverage_blind, None)["adopt"], called unchanged, where point_blind = diffs, wiv_blind = boot over the
  pool's windows and coverage_blind = rows(...)["coverage"]: full D5 with the window interval only (a 2-game game
  bootstrap says nothing).
- Not fully blind: gsw_bkn_2015's build-level rows are known (M6). On the two blind games, missed is partly seen: M4
  scored their raw CPU detections on the corpus frames of the pool windows (gsw_cha_2016 -0.0086, gsw_sac_2015
  -0.0162). Their team, ghost, position and bias rows at build level are unseen.
- Readings, fixed now:
  - Primary passes and blind passes: "seed 0 passes on the reference platform". Stage 2 runs.
  - Primary passes and blind fails: the pass rests on the game already seen and is not supported on the blind games;
    reported as a fail of the claim. Stage 2 does not run.
  - Primary fails and blind passes: a fail. The blind games alone do not override the registered pool.
  - Primary fails and blind fails: a fail; E1 as built is closed on both platforms.

## Stage 2: hou_cle_2016 (the only fully blind, non-GSW test)

- Runs only if the full claim holds (primary and blind both pass).
- Each fold's E1 seed 0 against r23 (M9's hou_cle_2016 build), all checks above, scored as the registered report
  scores hou_cle_2016 (window bootstrap only), with d5_verdict(point, window interval, coverage, None)["adopt"] per
  fold.
- Fixed now: all 3 folds must pass. If they do, the report says "seed 0 passes on the reference platform, including
  the new arena", and the result goes to the lead as the input for choosing between seeds 1 and 2 and E1b; it triggers
  no training by itself. If any fold fails, the report says the pass does not carry to the new arena.

## Smoke stage, with written gates (about 10 to 15 lock minutes)

Before any full E1R build:

1. Rebuild gsw_cha_2016_s00_w00 with r23 through the E1R path into a scratch folder,
   builds/e1r_smoke_r23__gsw_cha_2016. Gate: its _frames.json, _trajectories.json and _identity.json are
   byte-identical to builds/m9_r23_detcpu__gsw_cha_2016's (CPU repeatability, never measured before).
2. Build the same window with fold gsw_cha_2016 into builds/e1r_e1_gsw_cha_2016_detcpu. Gate: its records pass every
   window check above.
3. Score that pair with every check above. Gate: no check fails.
4. Rescore M6's gsw_bkn_2015 folders with the E1R scorer. Gate: it reproduces reports/research/p8_e1_m6.json exactly
   (83 windows; missed r23 0.252530 and E1 0.237816 to 6 decimals, every other row equal).

If any gate fails, nothing more is built and the failure is reported.

## What was already seen (forking paths)

- E1R was chosen after M4 and M6 favoured E1 on CPU (D175 (2)). M4: raw CPU detections, E1 minus r23 missed -0.0156
  pooled, below zero on every game. M6: gsw_bkn_2015 CPU builds, missed -0.0147 [-0.0197, -0.0088], ghosts -0.0168,
  position p50 -0.0843, near-field bias -0.1275, team +0.0188 [-0.0084, 0.0468], coverage equal.
- On MPS, gsw_bkn_2015_s04_w00 carried 609 of the 676 net extra E1 misses of gsw_bkn_2015; on CPU that window is close
  to even (E1 735 missed against r23 784), and without it gsw_bkn_2015's CPU difference is -0.0132.
- M9's r23-only rows on the three other games will be seen before E1R is built (r23 alone, not a comparison). This
  version was committed before any of them existed.
- The camera truth was fitted from r23's MPS builds (D98, D99). Both CPU arms share it, so the pairing holds; any bias
  it carries favoured r23 on MPS.

## Expected effect and noise floor

- Expected, from M4 and M6: missed improves by about 0.010 to 0.016. Ghosts, position and near-field bias improved on
  gsw_bkn_2015 on CPU and in the registered MPS result. Team's point estimate was positive on every game in the
  registered result and on gsw_bkn_2015 on CPU, with intervals across zero. Coverage is expected equal.
- Noise floor: on one game M6's missed window interval is about plus or minus 0.0055. The window bootstrap is kept as
  registered; the reviewer checked a segment bootstrap and found it similar. The main risk to a pass is a row whose
  point estimate turns the wrong way (team is the closest), not missed.

## Cost (lock hours, Mac CPU, no units)

- Smoke: about 10 to 15 lock minutes.
- Stage 1: E1 CPU builds of gsw_cha_2016 (18,083 processed frames) and gsw_sac_2015 (18,710), about 2.3 hours each at
  M6's 0.4501 s a frame (the reviewer measured 0.431), about 4.6 lock hours, plus a few minutes of scoring a game.
- Stage 2: three folds on hou_cle_2016 (23,813 frames each), about 8.9 lock hours, only if the claim holds.
- At most about 13.5 lock hours; one game per lock acquisition, released between games and between building and
  scoring.

## Kill criteria (stop building, say why, report what exists)

- A smoke gate fails.
- Any provenance check fails (weights, run record, complete.json, fold, seed), or a window records other weights, a
  fold other than the expected one, or a detector device other than CPU.
- A window build fails twice on the same window.
- Any per-window check of the score fails (records, frame sets, court H, gate-empty sets).
- The r23 arm's M9 check fails (the registered builds do not reproduce their score files).
- No interim result stops Stage 1: both pool games are built and scored whatever the first one shows.

## Caveats carried over

- In sample for the court model: r23's court model was trained on the R1.3 games' labels; both arms share it.
- The truth is the Mac camera truth (D99), which favours frames where the pipeline already works.
- The team row is the frozen best A/B mapping per window, an upper bound on a fixed mapping.
- One seed, one checkpoint choice (best.pt by mAP50-95 on pseudo-label validation, as registered), no retraining.

## Changes from version 1

1. E1R has its own build and scoring code; the r23 arm never resolves to builds/r23; the E1 fold is the one named after
   the game (version 1 relied on tools/p8_e1_m6.py, whose weights_for always returns fold gsw_bkn_2015); the fold is
   recorded per window; records and hashes are checked per window at score time; a regression test against
   p8_e1_m6.json.
2. The blind rule is full d5_verdict with the window interval; missed on the blind games is marked partly seen; four
   readings written.
3. Stage 2 runs only if the full claim holds, and needs all 3 folds to pass.
4. Equal frame sets and identical court H asserted per window.
5. A smoke stage with written gates.
6. The claim's wording, no mechanism, and the units note.
7. Equal gate-empty sets, else partial.
8. Printed per game and arm counts.
9. The window bootstrap kept, with the reviewer's segment check noted.
10. The order of registration and M9's output stated at the top.
11. Budget noted with the reviewer's measured rate.
12. The camera-truth note.

## Amendment, 2026-11-25 (after the second review, committed before any E1R build)

No rule changes. Code and notes only:

1. Stage order is enforced in tools/p8_e1r.py: a non-smoke Stage 1 build refuses to start unless e1r/smoke_gate1.json
   and e1r/smoke_gate4.json say passes: true and the smoke score file exists (it is written only when every check of
   smoke gate 3 passed); a hou_cle_2016 build refuses unless e1r/p8_e1r.json has primary_pass and blind.pass both
   true. The report emits Stage 2 only when the claim passed, with the fixed wording ("seed 0 passes on the reference
   platform, including the new arena" or "the pass does not carry to the new arena"). Tested.
2. The score stops if a window's in-frame player count differs between the arms, and the device record must request
   exactly {detector: cpu, court: auto, gate: auto}. Counts are printed for hou_cle_2016 too.
3. Duplicates are counted on the scored frames only (the live wide frames the scorer used), with referee boxes
   included: the sidecars carry no class, so this count is descriptive and is not the scorer's non-referee count.
4. Smoke order: gate 4 first (it builds nothing), then gates 1, 2 and 3.
5. Notes, given M9 (D176), which was merged after version 2 and read after it was committed:
   - The missed bar. On the reference platform r23 misses 0.2952 on gsw_cha_2016 and 0.2060 on gsw_sac_2015. For
     missed to improve, E1 must fall below those, a gain of about 0.057 and 0.050 against its registered MPS rates
     (0.3525 and 0.2556), where r23's own gain from the platform was 0.030 and 0.027.
   - The team row. r23's team rate moved with the platform by +0.0376, -0.0111 and +0.0200 (M9), about the size of
     E1's registered team edge (+0.0114 pooled), so on the blind games team's point estimate is near a coin flip. The
     report lists the failing rows (primary_failing_rows, blind.failing_rows). A fail that rests only on team's point
     estimate is still a fail under the registered rule, but it says little for choosing between seeds 1 and 2 and
     E1b.
   - The two priors in this design conflict: M4 (raw CPU detections) predicts a pass on missed, while M7 says MPS
     should, if anything, have favoured E1 (MPS lost more of r23's raw boxes), which would leave less room for E1 to
     gain on CPU. Neither is a mechanism claim.
6. Ledger: E1R's line goes into docs/research/TRAINING_PROTOCOL.md's ledger when it finishes (stage, lock hours,
   outcome against its gate, lesson).
