# P8 build loss: tracker resets and empty frames on the a3-heavy windows (registered before any number)

Written 2026-11-26 after build loss version 2 (P8_buildloss.md). Its largest loss class, a3 ("never tracked
nearby"), sits in a few windows, and decode is negligible there. This check asks whether tracker exceptions, resets or
empty frames explain it. It is report only: no model runs, nothing is built or changed, and no unit is spent. Tool:
tools/p8_trackreset.py, with the same rules in its docstring. Record: reports/research/p8_trackreset.json. The
alignment check the lead first asked for (r-p8-a3align) was dropped before it started; nothing of it exists.

## What an exception does (detect/tracker.py, read first)

- **Kalman LinAlgError.** If model.track raises numpy.linalg.LinAlgError (Kalman, "MPS precision"), a warning is
  logged, reset() runs, and update returns no tracks for that frame.
- **The re-identification bug.** If model.track raises an AttributeError whose text contains "cpu" (the ultralytics
  8.4 native re-identification bug), the same happens: a warning, reset(), and no tracks for the frame.
- **Anything else is raised,** and the build fails.
- **Camera cuts.** After a normal frame, the camera-cut detector flags a cut when 80 percent or more of 4 or more
  previous track ids vanished. reset() then runs, after that frame's tracks were returned.
- **What reset() does.** It drops the predictor, so the next frame starts a fresh BoT-SORT. That fresh tracker's first
  frame activates every detection at once. reset() also clears the cut detector, adds one to n_resets, and offsets
  later track ids.
- **What is stored.** build_trajectories.py stores n_resets as identity["camera_cut_resets"] in each window's
  _identity.json, which counts cut resets and exception resets together. Log lines are not stored for successful
  builds: sportvu.rebuild.build keeps stderr only on failure. So an exception shows only as an empty frame followed by
  new track ids.

## Registered definitions

Population and inputs:

- Reachable a3: build loss version 2's per-player rows of class a3 whose raw foot lies within 2.5 ft (pool games).
- Inputs are stored P8-workspace files only:
  - the reference-platform r23 sidecars and identity files;
  - the registered score files' scored windows;
  - the E1 corpus preds ("ours": r23 at 0.40 on the decoded clip frame);
  - build loss version 2's per-player rows.
- The tool stops if any of these is missing. Guards run on every game.

Windows:

- a3-heavy windows, per game: rank the game's windows by their reachable a3 count. The heavy set is the smallest
  leading set that holds at least 80 percent of the game's reachable a3.
- The comparison set is every other scored window of the pool games.

Measures per processed frame (every sidecar row of a window):

- the box count; a frame is empty when it is 0;
- on corpus frames: the "ours" box count at 0.40. The detector fired when it is 1 or more, and box yield is the
  sidecar box count over the "ours" box count.

Measures per window:

- processed frames, and the share that are empty;
- the share of empty frames among corpus frames where the detector fired;
- runs of consecutive empty frames (how many, and the longest);
- mean track-id turnover (the share of a frame's ids absent from the previous frame);
- reset-like frames (80 percent or more of 4 or more previous ids absent);
- identity's camera_cut_resets and median_players_per_frame;
- the median box yield where the detector fired.

Per reachable a3 player-frame, one label: on an empty frame; on a reset frame (its frame, or one of the two processed
frames before it, is reset-like); or neither.

## Reading rule (fixed now)

- If at least 50 percent of the pool's reachable a3 lies on empty or reset frames, a tracker robustness arm follows.
  That arm needs its own registration, then a rebuild of the affected windows on the reference platform under the
  lock, scored by the registered scorer.
- Otherwise the loss is in tracks the tracker does not output, on frames where it outputs others. Resets and
  exceptions do not explain it, and the next step would be the model run proposed in P8_buildloss.md (model.track on
  the decoded frames of those windows, under the lock).
- Why 50 percent: the arm should fix the majority of what it targets. Below half, an exception or reset fix would
  leave most of a3 in place.

## Forking path

The heavy windows are chosen after a3 was seen: build loss version 2 already named gsw_bkn_2015_s04_w00,
gsw_cha_2016_s05_w00, and gsw_sac_2015's s01_w00, s09_w01 and s07_w00. The rule above defines them mechanically, but
the hypothesis came from looking at them.

## Results

Not computed yet.
