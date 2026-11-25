# P8 E1R: E1 seed 0 on the reference platform (design, before any build)

Written 2026-11-25 under PLAN.md D175 (2). Evaluation only: no training, no Colab, no units. Nothing is built for E1R
until this design has been through training-design-reviewer and the lead has registered it. The D171 verdict stands
whatever E1R shows.

## Question

Does E1 seed 0 beat r23 when both run on the reference platform of D175 (player detector and tracker on CPU, court
and gate as registered)? The registered E1 result (D171) was scored on MPS builds, which D174 found carry a box
fault (ultralytics clip_boxes on MPS on this Mac). E1R asks the registered question again on builds without the
fault, with the registered rule unchanged.

## What E1R can and cannot decide

- Only seed 0 exists. D171 stopped seeds 1 and 2, and D85 (3) asks 3 seeds for any trained model: tools/p8_e1_score.py
  adopts E1 only when all 3 seeds are scored and each seed's pool adopts. So E1R cannot adopt E1. Its report will
  say "1 of 3 seeds scored" and the verdict field will be false, as the registered code makes it.
- What it can decide: whether seed 0's pool passes the registered rule on the reference platform. If it passes, the
  D171 failure is shown to be a platform effect for seed 0, and training seeds 1 and 2 becomes a candidate next step.
  That step needs Colab units, Lucien's Colab (D137), and its own registration. If it fails, E1 as built is closed on
  both platforms.
- Adoption can never follow from one seed under the registered rule, and E1R does not change that rule.

## Arms and builds

- r23 arm: the production detector (sha256 4c886d5f...) on the reference platform. gsw_bkn_2015 is M6's build
  (builds/e1m6_r23_detcpu); gsw_cha_2016, gsw_sac_2015 and hou_cle_2016 are M9's builds
  (builds/m9_r23_detcpu__<game>), running now. Both were built by tools/p8_e1_m6.py with P8_DEVICE_DETECTOR=cpu.
- E1 arm: each fold's seed 0 best.pt (provenance by tools.p8_e1_score.check_provenance, the run record, complete.json
  and the sha256 checked; d4ffed33..., 0579a034..., bdad8b76...) built the same way. gsw_bkn_2015 by fold gsw_bkn_2015
  exists (builds/e1m6_e1_gsw_bkn_2015_detcpu, M6). To build: gsw_cha_2016 by fold gsw_cha_2016 and gsw_sac_2015 by
  fold gsw_sac_2015, into new folders builds/e1r_e1_<fold>_detcpu. Each fold's weights run only on the game it left
  out (D156). The M6 tool's folder allowlist would be extended to exactly these folders.
- Windows: the label windows of each game with a registered r23 sidecar, the registered set: 83, 73 and 82 windows on
  the pool games (238), and 93 on hou_cle_2016.
- Nothing registered is written: the E1R score files and report get their own names (e1r/...), which
  tools.p8_e1_score report never reads.

## Rule (registered, unchanged)

- Scoring: the pinned score_window (D88), truth camera, foot_stab, through tools.p8_e1_score.window_stats; where the
  two arms' processed frame sets differ, both are scored on the shared frames, as tools.p8_e1_score score does.
- Rows and verdict: tools.p8_e1_score.rows, boot and d5_verdict, called unchanged. Position p50 and p90, near-field
  bias, missed, ghosts (non-referee), team (majority mapping, an upper bound) and coverage. Paired E1 minus r23; window
  bootstrap and game bootstrap (2,000 resamples, seed 0). D5 as amended with D158: a row is worse when its window or
  its game interval lies wholly on the bad side of zero; it improves on the point estimate; the pool passes when every
  row but coverage improves, none is worse, and coverage holds its floor (0.50, or does not fall if r23 is below). A
  pool with any missing window is partial and never passes.
- Apart, as registered: hou_cle_2016 scored with every fold's model, window bootstrap only, reported apart (D156).
  gsw_ind_2016 stays apart and is not scored (D98), as in the registered result.

## What was already seen (forking paths)

- E1R was chosen after M4 and M6 favoured E1 on CPU (D175 (2) asks this to be said). M4: raw CPU detections on the
  corpus JPEGs, E1 minus r23 missed -0.0156 pooled, below zero on every game (gsw_bkn_2015 -0.0197, gsw_cha_2016
  -0.0086, gsw_sac_2015 -0.0162). M6: the gsw_bkn_2015 CPU builds, missed -0.0147 [-0.0197, -0.0088], ghosts
  -0.0168, position p50 -0.0843, near-field bias -0.1275, team +0.0188 [-0.0084, 0.0468], coverage equal.
- gsw_bkn_2015 is therefore not blind: all its build-level rows on the reference platform are known. gsw_cha_2016 and
  gsw_sac_2015 are blind at the build level, but their raw-detection direction is known from M4 (the corpus frames are
  the camera-truth frames the scorer uses). The r23 arm's M9 rows on the three other games will also be seen before
  E1R is built (r23 alone, not a comparison).
- On MPS, one window (gsw_bkn_2015_s04_w00) carried 609 of the 676 net extra E1 misses of gsw_bkn_2015. On CPU that
  window is close to even (E1 735 missed against r23 784), and without it gsw_bkn_2015's CPU difference is -0.0132.
- How the rule treats this: the primary verdict is the registered rule on the full 3-game pool, gsw_bkn_2015
  included, because that is the registered pool. Reported beside it, fixed now, is the same computation on the two
  blind games alone (gsw_cha_2016 and gsw_sac_2015: rows, window interval and D5 checks with the window interval only,
  since a 2-game game bootstrap says nothing). E1R reports "seed 0 passes on the reference platform" only if the
  primary pool passes and, on the blind pool, no row is worse by its window interval and missed improves on its point
  estimate. If the primary pool passes and the blind pool does not meet that, the report says the pass rests on the
  game already seen and is not supported on the blind games.

## Expected effect and noise floor

- Expected, from M4 and M6: missed improves by about 0.010 to 0.016 (M4 raw pooled -0.0156; M6 build -0.0147 on
  gsw_bkn_2015; M4's smallest game, gsw_cha_2016, -0.0086). Ghosts, position and near-field bias improve (all three
  improved in the registered MPS result and on gsw_bkn_2015 on CPU). Team's point estimate was positive on every game
  in the registered result and on gsw_bkn_2015 on CPU; its intervals cross zero. Coverage is expected equal: the
  detector does not change which frames are processed in M6.
- Noise floor: on gsw_bkn_2015, M6's missed window interval is about plus or minus 0.0055 around the difference; on a
  3-game pool the window interval should be narrower. The game interval with 3 games is about the range of the three
  per-game differences (D158 notes it is weak), so a game that moves the other way could make the game interval cross
  zero without making any row worse. The rule needs only the point estimate to improve, so the main risk to a pass is
  a row whose point estimate turns the wrong way (team is the closest: +0.0188 on gsw_bkn_2015 with an interval across
  zero), not the missed row.
- What would surprise: missed not improving on either blind game at the build level, given M4. That would point back
  at the build path (tracking or foot_stab), not at the device.

## Cost (lock hours, Mac CPU, no units)

- Stage 1, the pool: E1 CPU builds of gsw_cha_2016 (18,083 processed frames) and gsw_sac_2015 (18,710), at M6's
  rate of 0.4501 s a frame for E1: about 2.3 and 2.3 hours, 4.6 lock hours, plus a few minutes of scoring a game. The
  r23 arm costs nothing more (M6 and M9).
- Stage 2, hou_cle_2016 apart: the three folds' E1 builds of hou_cle_2016 (23,813 frames each), about 3.0 hours
  each, 8.9 lock hours. Fixed now: Stage 2 runs only if the Stage 1 primary pool passes, since hou_cle_2016 informs
  whether a pass generalises to a new arena and changes no verdict.
- At most about 13.5 lock hours; one game per lock acquisition, released between games and between building and
  scoring.

## Kill criteria (stop building, say why, report what exists)

- Any provenance check fails (weights, run record, complete.json, fold, seed), or a window records other weights or a
  detector device other than CPU.
- A window build fails twice on the same window.
- The r23 arm's registered-side check fails in M9 (the registered builds do not reproduce their score files), or M9
  shows a frame-set problem that makes shared-frame scoring cover less than 95 percent of a game's registered frames.
- No interim result stops Stage 1: both pool games are built and scored whatever the first one shows, so the verdict
  is not chosen by when it was looked at.

## Caveats carried over

- In sample for the court model: r23's court model was trained on the R1.3 games' labels; both arms share it.
- The truth is the Mac camera truth (D99), which favours frames where the pipeline already works.
- The team row is the frozen best A/B mapping per window, an upper bound on a fixed mapping.
- One seed, one checkpoint choice (best.pt by mAP50-95 on pseudo-label validation, as registered), no retraining.
