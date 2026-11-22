# P8 E1 on Colab: what Lucien runs (PLAN.md D137, D155, D156)

E1 trains a player detector on the E1 corpus (docs/research/P8_teachers.md, "E1 corpus"), leave one game out: three
folds, each named by the game it holds out, each trained on the other two games. Colab is Lucien's; the agents prepare
and never run it. Scoring happens afterwards on the Mac (tools/p8_e1_score.py), on each fold's held-out game, pooled
with the game bootstrap against r23, and on hou_cle_2016 apart.

## 1. Upload (once)

On the Mac, the lead packs the corpus into one tar per game (Drive loses files when many small ones sync):

    python3 -m tools.p8_e1_pack pack --dest <a local staging folder>

That folder holds common.tar (meta, manifest, plan), gsw_bkn_2015.tar, gsw_cha_2016.tar, gsw_sac_2015.tar (about
1.9 GB in all) and pack_manifest.json. Copy all five files into My Drive/nba_harvest/p8_e1/corpus/ in the Colab
account. When Drive shows the upload finished, check it from the Mac mount by member count:

    python3 -m tools.p8_e1_pack verify --dest "<the Drive mount>/My Drive/nba_harvest/p8_e1/corpus"

Every tar must report found = expected. A short tar is copied again; nothing trains on a short corpus (the Colab run
checks the counts too and stops).

## 2. Run (one fold per session)

Runtime: T4 GPU. Paste the one cell from notebooks/colab_p8_e1.ipynb (also below), set FOLD to one of
gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 and PIN to the commit the lead names, and run it. All logic is in the repo
script; the cell only mounts Drive, resets the repo to PIN and calls it.

```python
#@title P8 E1, one fold (all logic in tools/p8_e1_colab.py at the pinned commit)
FOLD = "gsw_bkn_2015"      # the held-out game: gsw_bkn_2015, gsw_cha_2016 or gsw_sac_2015
SEED = 0                   # one seed first (D156)
PIN = "<commit the lead names>"
import os
from google.colab import drive
drive.mount("/content/drive")
if not os.path.exists("/content/repo"):
    !git clone https://github.com/lmcnulty7/NBA-Comp-Viz-V3.git /content/repo
%cd /content/repo
!git fetch origin && git reset --hard {PIN} && git log --oneline -1
!pip -q install "ultralytics==8.4.48"
!python -m tools.p8_e1_colab colab --fold {FOLD} --seed {SEED}
drive.flush_and_unmount()    # block until Drive has the result tar (an async upload lost files before)
print("done: check My Drive/nba_harvest/p8_e1/runs/e1_" + FOLD + "_seed" + str(SEED) + ".tar")
```

The run copies the common tar and the fold's two training games' tars to local disk, checks their member counts, trains
(yolov8m.pt, 100 epochs, image size 640, batch 16, patience 20; validation for early stopping is the last 10 percent of
each training game's windows), and writes one tar to My Drive/nba_harvest/p8_e1/runs/: weights (train/weights/best.pt
and last.pt), results, data.yaml, the split lists and run_record.json (fold, held-out game, training games, corpus meta,
seed, recipe, library versions; D150 condition 4).

## 3. Time (estimates, b1 scaling: 0.5 T4-hours per 654 images at the same epochs)

| Fold (held out) | Training images (before the validation split) | T4-hours | Colab units at 1.19 an hour |
| --- | --- | --- | --- |
| gsw_bkn_2015 | 3,517 | about 2.7 | about 3.2 |
| gsw_cha_2016 | 4,226 | about 3.2 | about 3.8 |
| gsw_sac_2015 | 3,441 | about 2.6 | about 3.1 |

About 8.5 T4-hours for the three folds of one seed. These are estimates (P8_step0_trackB.md section 6); early stopping
can end a fold sooner. A session that disconnects writes nothing; rerun the cell for that fold.

## 4. After

Download the three result tars to the Mac (the lead, outside the repository) and score them under the lock:

    <frozen CV python> -m tools.p8_e1_score rebuild --model <fold> --weights <best.pt> --game <fold>
    <frozen CV python> -m tools.p8_e1_score rebuild --model <fold> --weights <best.pt> --game hou_cle_2016
    <frozen CV python> -m tools.p8_e1_score score --model <fold> --game <fold>      (and hou_cle_2016)
    python3 -m tools.p8_e1_score report

The report pools the three held-out games against r23 on the same windows (window and game bootstrap) and applies D5
with the Stage 1 rows; hou_cle_2016 is reported apart; gsw_ind_2016 stays apart (D98).
