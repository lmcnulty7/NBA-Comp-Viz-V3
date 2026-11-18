# P8: why the Mac's camera truth falls short of Colab's

Written 2026-11-18 under PLAN.md D98 and D99. A five-minute read. Every number in the tables is in
`reports/research/p8_platform.json`, written by `python3 -m tools.p8_platform_summary` from the diagnosis files. The
game is gsw_ind_2016, an R1.3 training game: development data, not a result.

## The problem

Night 1 relabelled the R1.3 games on this Mac with the frozen code. On gsw_ind_2016 it accepted far fewer frames of
camera truth than the Colab R1.3 run did, from a smaller candidate pool, and its camera fit was worse.

| Game total | Colab (A100) | Mac (MPS) |
| --- | --- | --- |
| Accepted camera-truth frames | 3552 | 970 |
| Frames in the camera fit's pool | 5683 | 3900 |
| Camera reprojection error, px | 6.53 | 18.35 |

The inputs are the same: all 11 video sections hash to Colab's input manifest. The build code and model weights have
not changed since that run. Both runs use fp32, image size 640, confidence 0.40 and NMS IoU 0.45. The differences are
the device (CUDA on Colab, MPS here) and library versions (Colab's ultralytics was not recorded).

## What the local builds show

Two windows were built again on this Mac, once per device. Candidates are frames with at least 6 matched players,
the input of the camera fit.

| Window | Colab candidates | Mac MPS | Mac CPU | Boxes per frame, MPS | Boxes per frame, CPU |
| --- | --- | --- | --- | --- | --- |
| s02_w04 (125 frames) | 73 | 2 | 20 | 6.88 | 7.78 |
| s02_w03 (145 frames) | 89 | 30 | 30 | 7.43 | 8.47 |

1. **MPS sees fewer players.** CPU finds about one more player box per frame, and only 39 of 125 frames of s02_w04
   have the same box count on both devices. The court model's output moves too: a median gap of 4.65 px between the
   two devices' court mappings. MPS is deterministic; a second MPS build matched the first exactly.
2. **The clock offset decides the rest.** Each window's SportVU clock offset is solved by the frozen code from the
   build's own players and court mapping. Small changes move it: on 44 of 81 windows the Mac's offset differs from
   Colab's by more than 0.2 s. A wrong offset matches players to the wrong instant, and most frames lose their 6
   matches. On s02_w03, a CPU build given Colab's offset (0.62 s) has 88 candidates, against Colab's 89. The same
   offset gives an MPS build 45.

So the Mac's lower truth share, and gsw_ind_2016's poor camera fit, come from the platform, not from the game. Fewer
MPS boxes also raise the Mac's missed-player rate.

## The detector carries it

s02_w03 was built a third time with the player detector on CPU and the court model on MPS. Candidates are counted at
fixed clock offsets, to separate the build from the offset solve.

| s02_w03 build | Boxes per frame | Offset solved | Candidates at 0.30 s | at 0.62 s (Colab's) | at 0.90 s | at 1.20 s |
| --- | --- | --- | --- | --- | --- | --- |
| MPS | 7.43 | 0.9 | 65 | 45 | 30 | 16 |
| CPU | 8.47 | 1.42 | 98 | 88 | 55 | 35 |
| Detector CPU, court MPS | 8.47 | 0.94 | 97 | 88 | 55 | 35 |

- With the detector on CPU, all 145 frames have exactly the boxes of the full CPU build. The court mapping stays
  exactly that of MPS (gap 0.0 px), and the candidates match full CPU. The court model's device does not matter here.
- The build took 72 s against 142 s for full CPU on the same window. Full CPU took 122 s against 26 s for MPS on
  s02_w04, so a detector-only CPU build is about 2.4 times MPS (estimate from these two windows).
- The offset solve still misses: each build lands on its own offset (0.9, 1.42, 0.94), none at Colab's 0.62 s,
  and the candidates there are far fewer than at 0.62 s. The detector fix restores the boxes, not the offset rule.

## Pending

- The same two windows built on Colab's GPU (the cell is ready for Lucien): does CUDA match CPU?

## Options

1. **Detector on CPU** (`tools/p8_device.py`, prepared, not used yet): set `P8_DEVICE_DETECTOR=cpu`, keep the court
   model and the gate on MPS. No frozen file changes. At about 2.4 times MPS, the remaining build hours grow from
   about 12 to about 29 (estimates). A workspace never mixes device settings, so this run starts a fresh one and
   relabels the R1.3 games too.
2. **Change the offset rule** (for example, the offset with the most candidates). This is a new truth rule, so it is
   Lucien's decision, not the team's.
3. **Keep MPS** and report the truth share beside every row, with gsw_ind_2016 left out (PLAN.md D98).
