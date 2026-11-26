# P8 build loss: corpus-to-build frame alignment (registered before any number)

Written 2026-11-26 after P8_trackreset.md. That check found no tracker resets under the a3 loss. It also corrected
build loss version 2: the version 2 decode class compared the JPEG with the corpus's own decoded frame, not with the
build's decode. This check asks whether the corpus decode (where M4's raw boxes ran) and the build's decode show the
same frame at the same index, window by window. It also asks how much of a3 falls where they do not.

This is report only: no model runs, nothing is built or changed, and no unit is spent. Tool: tools/p8_a3align.py,
with the same rules in its docstring. Record: reports/research/p8_a3align.json.

## What each side records (read first)

- **Corpus.** plan.json and manifest.json record, per frame, the window and the clip frame index. The plan rows also
  carry the truth row's q and moment.
- **Sidecar.** Each processed frame records the clip frame index. Each file also records the source clip's file name,
  fps and stride.
- **No timestamps or hashes.** Neither side records a decoded timestamp or a content hash.
- **Same clip, different readers.** Both indices count frames of the same window clip, label_build/<window>.mp4. The
  corpus reads it by sequential grab from frame 0 (tools/p8_e1data.py run). The build reads it through
  videoseq.SeqReader with the pre-gate jumps (build_trajectories.py).
- **So the index cannot decide alignment.** A direct index comparison only shows that both name the same index.
  Whether the decoded content at that index matches must be inferred from the boxes. As checks, the tool records
  whether each sidecar names the same clip file, and whether the clip file was modified after the corpus preds were
  written.

## Registered definitions

Windows and the match share:

- Windows: every scored window of the pool games that holds corpus frames with M4 raw boxes. Guards run on every
  game.
- Offsets: for each corpus frame f that the sidecar processed, and each offset k of -2, -1, 0, +1 or +2 processed
  frames (the sidecar frame k places away from f in the window's sorted processed frames), every M4 r23 raw box at
  0.40 gets its largest IoU with a sidecar box of that frame.
- S_k is the share of raw boxes whose largest IoU is at least 0.5, pooled over the window's corpus frames. It is
  compared with S_0 on the same frames.

Two ways a window counts as misaligned:

- **Offset-misaligned:** some k other than 0 has S_k at least S_0 + 0.05.
- **Sub-stride suspect:** among raw boxes whose largest same-frame IoU is at least 0.3, the median of that IoU is
  below 0.75. This catches an offset shorter than one stride (6 video frames), which cannot show up as k not equal to
  0. A tracked box is the detection box updated by the Kalman filter, so in an aligned window this median is
  expected near 0.9.

A window is misaligned if either holds, and aligned otherwise.

- a3 is build loss version 2's class a3 with the raw foot within 2.5 ft, from the per-player rows in the P8 workspace.
  The tool stops if that file is missing.

Why the thresholds:

- 0.05 is a clear margin over the same-frame share. A truly aligned window should match best at offset 0, because
  players move between processed frames 0.1 s apart.
- 0.75 sits between the about 0.9 expected for a matched track box and the 0.5 to 0.6 that a shift of a few video
  frames would give for moving players.

## Reading rule (fixed now)

- **(i)** If at least 50 percent of reachable a3 lies in misaligned windows, a3 is mostly a measurement artefact of a
  corpus-to-build frame mismatch. The detection-row upper bound shrinks by that share, and no build arm follows.
- **(ii)** Otherwise a3 is mostly in aligned windows. Then P8_buildloss.md's model run is justified: model.track on
  those windows' decoded frames, under the lock. That run needs its own registration and waits for the lock.
- The split between offset-misaligned and sub-stride-suspect windows is reported either way.

## Forking path

a3 and its heavy windows are already seen (P8_buildloss.md version 2, P8_trackreset.md). The hypothesis that two
high-yield gsw_sac_2015 windows are misaligned came from looking at them.

## Results (computed 2026-11-26, after the definitions above were committed in 616a205e)

Stored files only, no lock, about 2 seconds. Record: reports/research/p8_a3align.json.

Checks: every one of the 175 measured windows (62, 53, 60) names the same clip file on both sides, and no clip file
was modified after the corpus preds were written.

Typical window: the median S_0 is 0.974, 0.978 and 0.979, and the median same-frame IoU is 0.913, 0.906 and 0.905
(bkn, cha, sac).

Misaligned windows: 8 on gsw_bkn_2015, 4 on gsw_cha_2016 and 2 on gsw_sac_2015, 14 of 175 in all.

- In almost all of them the build's frame one processed frame earlier (usually 6 video frames; k = -1, sometimes -2) matches the corpus frame better
  than the same index. In s02_w05, for example, S_-1 is 0.987 against S_0 0.923, and the median same-frame IoU is 0.68
  in s02_w05. The build's decode shows at index f - 6 what the corpus decode shows at f.
- Two windows (bkn s05_w01 and s06_w00) are flagged by the sub-stride check alone. cha s01_w02 has few boxes
  (S_0 0.27).

The a3-heavy windows are all aligned:

| Window | S_0 | Best other offset | Median same-frame IoU |
| --- | --- | --- | --- |
| gsw_bkn_2015_s04_w00 | 0.533 | 0.526 (k = +1) | 0.925 |
| gsw_cha_2016_s05_w00 | 0.165 | 0.153 | 0.928 |
| gsw_cha_2016_s05_w03 | 0.100 | 0.100 | 0.941 |
| gsw_sac_2015_s01_w00 | 0.359 | 0.346 | 0.914 |
| gsw_sac_2015_s09_w01 | 0.608 | 0.589 | 0.909 |
| gsw_sac_2015_s07_w00 | 0.958 | 0.921 | 0.919 |

Reachable a3 by alignment:

| Game | Misaligned | Aligned |
| --- | --- | --- |
| gsw_bkn_2015 | 5 (all sub-stride) | 369 |
| gsw_cha_2016 | 2 (offset) | 49 |
| gsw_sac_2015 | 0 | 161 |
| Pooled | 7 (1.2 percent) | 579 |

## Reading by the registered rule

(ii): a3 is mostly in aligned windows (98.8 percent), so it is not an artefact of a corpus-to-build frame mismatch.
P8_buildloss.md's model run is justified: model.track on the heavy windows' decoded frames, under the lock, to see
whether the tracker creates new tracks there and never confirms them. It needs its own registration and waits for the
lock.

In the heavy windows the build's boxes that do exist match the raw boxes well (median IoU 0.91 to 0.94, best at
offset 0). The build has fewer of them in four of the six (bkn s04_w00, cha s05_w00 and s05_w03, sac s01_w00); in sac s09_w01
the boxes are there (yield 0.91) but do not match at IoU 0.5 at any offset, unexplained. Whether the build's detector
fired there is open (runtime and decode differences, see the P8_buildloss.md correction); the run must use the build
environment, raw detections and model.track both, to split the detector from the tracker. In gsw_sac_2015_s07_w00 (S_0 0.96) the 23 a3 player-frames are something
else, not measured here.

Beyond the rule (post hoc, and it reaches past a3): 14 of 175 pool windows (8 percent) show the corpus decode one
stride behind the build's decode at the same frame index. Inferred (not checked): the truth rows (camera truth, SportVU moment)
are attached to frame indices of the build-side decode: the V3 label build reads with the same SeqReader path. Corpus JPEGs in those
windows may therefore show a frame about 0.1 s off from their truth rows. That bears on every analysis that scored the
corpus JPEGs against SportVU:

- E1's source 2 labels and the E1b bound and oracle;
- M1 to M4 of the E1 post-mortem;
- P8_detrow.md's raw and teacher classes.

Its size there is not measured. A registered check could rescore those analyses without the 14 windows. A fix of the
corpus decode would be its own decision.
