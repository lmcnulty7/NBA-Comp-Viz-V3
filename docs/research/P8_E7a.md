# P8 E7a: a first measurement of the ball row with WASB's basketball model (PLAN.md D155 (2))

Status: stopped at step 1, the license check, on 2026-11-22. No weights were downloaded, no frame was read and no
model ran. Nothing below is a result.

## License finding

Read at the source (github.com/nttcom/WASB-SBDT, default branch, last commit 923462ca of 2023-11-23):

- The basketball WASB weights are `wasb_basketball_best.pth.tar` on Google Drive (id
  1nfECuSyJvPUmz3njZCdFERSQQbERt8FU), linked from MODEL_ZOO.md (row "WASB (Ours)", column "Basketball") and fetched
  by `src/setup_scripts/setup_weights.sh` line 54. An HTTP HEAD request gives 6,102,633 bytes.
- MODEL_ZOO.md says only: "We provide model weights of SOTA methods (including WASB) for each dataset." It names no
  license or terms for them. GET_STARTED.md ("Model Preparation") says only "Download pretrained models listed in
  MODEL_ZOO.md". The README has an MIT badge for the repository.
- LICENSE.md is the MIT License, "Copyright (c) 2023 NTT Communications Corporation". Its grant covers "this
  software and associated documentation files (the "Software")". The weights are not in the repository; they are
  hosted on Google Drive, and neither LICENSE.md nor any other file in the repository says whether the grant extends
  to them.
- The weights were trained on the WASB basketball labels over NUST-NBA181 video (WASB's basketball dataset config
  lists 2018-19 game ids), which its authors release "upon request and only for academic research"
  (docs/research/P8_datasets.md).

Reading: no license explicitly covers the weights. As the task instructs for that case, the finding is recorded
(docs/DATA_SOURCES.md, row "WASB basketball model weights") and E7a stops before any download. Whether the
repository's MIT grant is taken to cover weights its authors distribute for it, or the authors are asked, is a
decision for the lead or Lucien. Steps 2 to 4 (the registration of frames, truth, in-frame and occlusion rule,
metric, tolerance and intervals, then the runs) have not started.
