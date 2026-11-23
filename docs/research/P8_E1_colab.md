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

Every tar must match the manifest in bytes and in member count. A short tar is copied again; nothing trains on a
short corpus (the Colab run checks bytes and counts again and stops).

## 2. Run (one fold per session)

Runtime: T4 GPU. Paste the one cell from notebooks/colab_p8_e1.ipynb (also below), set FOLD to one of
gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 and leave PIN at 1f14c1ae, the merge of the review fixes on v4-research (named by the lead on 2026-11-23). All logic is in
the repo script; the cell only mounts Drive, resets the repo to PIN and runs it, stopping before "done" if it fails.

```python
#@title P8 E1, one fold (all logic in tools/p8_e1_colab.py at the pinned commit)
FOLD = "gsw_bkn_2015"      # the held-out game: gsw_bkn_2015, gsw_cha_2016 or gsw_sac_2015
SEED = 0                   # one seed first (D156)
PIN = "1f14c1ae"
import os
from google.colab import drive
drive.mount("/content/drive")
if not os.path.exists("/content/repo"):
    !git clone https://github.com/lmcnulty7/NBA-Comp-Viz-V3.git /content/repo
%cd /content/repo
!git fetch origin && git reset --hard {PIN} && git log --oneline -1
!pip -q install "ultralytics==8.4.48"
import subprocess, sys
subprocess.run([sys.executable, "-m", "tools.p8_e1_colab", "colab", "--fold", FOLD, "--seed", str(SEED)], check=True)
drive.flush_and_unmount()    # block until Drive has the result tar (an async upload lost files before)
print("done: My Drive/nba_harvest/p8_e1/runs/e1_" + FOLD + "_seed" + str(SEED) + ".tar and its .complete.json")
```

The run checks the common tar and the fold's two training games' tars on the Drive mount (bytes and member counts)
and extracts them from there straight to local disk (no separate copy step). Before training it runs the project guard
on every game and id, checks that each training game's images, labels and manifest count agree, and runs Ultralytics'
own label check on the train and validation lists (every label found, no image corrupt). It trains (yolov8m.pt, 100
epochs, image size 640, batch 16, patience 20; validation for early stopping is the last 10 percent of each training
game's windows), writes complete.json (the sha256 of best.pt and last.pt), and writes one tar to
My Drive/nba_harvest/p8_e1/runs/ with weights (train/weights/best.pt and last.pt), results, data.yaml, the split lists,
run_record.json (fold, held-out game, training games, corpus meta, seed, recipe, library versions, label check counts;
D150 condition 4) and complete.json, plus <run>.complete.json beside the tar. A result tar of the same name already
there is renamed to <name>.prev-<time> before training, never deleted.

## 3. Time (estimates, b1 scaling: 0.5 T4-hours per 654 images at the same epochs)

| Fold (held out) | Training images (before the validation split) | T4-hours | Colab units at 1.19 an hour |
| --- | --- | --- | --- |
| gsw_bkn_2015 | 3,517 | about 2.7 | about 3.2 |
| gsw_cha_2016 | 4,226 | about 3.2 | about 3.8 |
| gsw_sac_2015 | 3,441 | about 2.6 | about 3.1 |

About 8.5 T4-hours for the three folds of one seed. These are estimates (P8_step0_trackB.md section 6); early stopping
can end a fold sooner. A session that disconnects writes nothing; rerun the cell for that fold.

## 4. After

Download the result tars to the Mac (the lead, outside the repository), unpack each into its own folder and score
under the lock. The weights must stay in <run>/train/weights/: the scorer reads <run>/run_record.json and
<run>/complete.json and refuses weights of another fold, seed or hash.

    <frozen CV python> -m tools.p8_e1_score rebuild --model <fold> --seed 0 --weights <run>/train/weights/best.pt --game <fold>
    <frozen CV python> -m tools.p8_e1_score rebuild --model <fold> --seed 0 --weights <run>/train/weights/best.pt --game hou_cle_2016
    <frozen CV python> -m tools.p8_e1_score score --model <fold> --seed 0 --game <fold>      (and hou_cle_2016)
    python3 -m tools.p8_e1_score report

The report gives each seed's pool and marks the verdict "1 of 3 seeds" until all 3 seeds of every fold are scored;
adoption needs all 3 (D85 (3)). A pool with a missing window is marked partial and never adopts.

The report pools the three held-out games against r23 on the same windows (window and game bootstrap) and applies D5
with the Stage 1 rows; hou_cle_2016 is reported apart; gsw_ind_2016 stays apart (D98).
