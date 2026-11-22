# P8 step 0, Track B: outside teacher detectors

Written 2026-11-21 under PLAN.md D131 and D132. Code: `tools/p8_teachers.py` (rules in its docstring, fixed before
any teacher ran). Outputs stay in the P8 teacher workspace, outside the repository: `results.json` (qualification)
and `misses_results.json` (the follow-up below). No training, no new human labels.

## Qualification on the human box truth

50 box-truth frames (453 player boxes, 2 games, 7 clips; none dropped by the guard). Matching as
`colab_autolabel.py --qualify`: greedy, IoU 0.5. Every model on CPU. Interval: 2,000 frame bootstrap replicates.

| Model | Recall at our precision (threshold) | Teacher minus ours [interval] | Qualifies |
| --- | --- | --- | --- |
| Our detector, player boxes, conf 0.40 | precision 0.9661 [0.9487, 0.9823], recall 0.9426 [0.9152, 0.9663] | | comparator |
| YOLOX-X SportsMOT (MixSort) | 0.9845 (0.1471) | +0.0419 [+0.0229, +0.0644] | yes |
| RF-DETR-L (COCO) | 0.0353 (0.8588) | -0.9073 [-0.9326, -0.6882] | no |
| YOLO11x (COCO) | 0.0596 (0.8456) | -0.8830 [-0.9202, -0.5245] | no |

YOLOX-X's boxes away from every box of ours: 13, of which 12 are true players (precision 0.9231 [0.7273, 1.0]).
Licenses: RF-DETR-L Apache-2.0; MixSort code MIT, its SportsMOT weights non-commercial (CC BY-NC 4.0); YOLO11x
AGPL-3.0. Open: whether the two box-truth games are in SportsMOT's training videos is not checked.

## Misses on camera-truth frames (D131 source 2)

The question: on development games with camera truth, how many SportVU players does our detector miss, and how many
of those can YOLOX-X boxes recover? That is the size of the pseudo-label set that can target our misses.

Rule (`misses-plan`, `misses-run`, `misses-score`):

- Games: the accepted camera-truth frames of gsw_bkn_2015 and gsw_ind_2016 in the P8 workspace. gsw_ind_2016 is
  reported apart (D98: its Mac camera truth is weak, reprojection 18.35 px against 6.52 for gsw_bkn_2015).
- Frames: 400 a game, sampled by time (one at random in each of 400 equal strata of the accepted frames in time
  order, seed 20261121), never by a model's output.
- Targets: SportVU players whose feet, projected through the stored camera truth, fall inside the image.
- A box's foot is its bottom centre, sent to the court through the camera truth. Our detector (player boxes, conf
  0.40) finds a target when a Hungarian match puts a foot within 2.5 ft of it (CONFIRM_FT, frozen
  `sportvu/autolabel.py`). The rest are misses. YOLOX-X at 0.1471 is matched the same way on its own; a miss it
  matches is recovered.
- Check before scoring: the stored label matches were reproduced from the moments file and the camera truth to within
  0.013 ft (gsw_bkn_2015, 1,320 matches) and 0.012 ft (gsw_ind_2016, 421 matches).

| | gsw_bkn_2015 | gsw_ind_2016 (D98, apart) |
| --- | --- | --- |
| Accepted frames / sampled | 2,075 / 400 | 970 / 400 |
| SportVU players in frame | 3,953 | 3,963 |
| Found by our detector | 2,903 | 2,150 |
| Missed by our detector | 1,050, share 0.2656 [0.2497, 0.2818] | 1,813, share 0.4575 [0.4449, 0.4701] |
| Found by YOLOX-X | 3,202 (miss share 0.1900) | 2,287 (miss share 0.4229) |
| Misses recovered by YOLOX-X | 400, share 0.3810 [0.3437, 0.4175] | 273, share 0.1506 [0.1333, 0.1681] |
| Recovered per frame | 1.0 | 0.6825 |
| Boxes: ours / YOLOX-X | 3,738 / 3,980 | 3,809 / 3,983 |
| CPU seconds: ours / YOLOX-X | 37.4 / 348.5 | 39.5 / 351.5 |

Intervals are 2,000 frame bootstrap replicates.

Reading:

- A miss here is a foot more than 2.5 ft from SportVU, not only a missing box. Ours draws 3,738 boxes for 3,953
  players in frame on gsw_bkn_2015, so most of its 1,050 misses there are boxes whose foot lands too far away
  (occlusion, foot point or camera truth error), not players with no box at all. The recovered boxes are those
  where YOLOX-X's foot lands within 2.5 ft and ours does not.
- On gsw_bkn_2015 YOLOX-X recovers 38 percent of our misses, about one pseudo-label a frame. Over all 2,075 accepted
  frames that projects to about 2,075 boxes for this game (1.0 a frame times 2,075; a projection from the sample, not
  a count).
- On gsw_ind_2016 both models miss far more (0.4575 and 0.4229), consistent with its weak camera truth: there the
  truth, not the detectors, likely sets the miss rate. Its numbers should not size the label set.
- Camera-truth frames are accepted only when enough of our detector's boxes match SportVU, so these frames favour
  scenes where our detector already works; the miss share on all live frames is likely higher.

## E1 corpus (D155 (1), built 2026-11-22; nothing trained)

tools/p8_e1data.py ran the YOLOX-X teacher and our detector (both on CPU, under the heavy-job lock, 9 chunks) over
every accepted Mac camera-truth frame of the 3 R1.3 games, in time order, and built D131 sources 1 and 2 by the rules
above: our player boxes at 0.40, YOLOX-X at 0.1471, agreement at IoU 0.5 with the mean box (source 1), teacher boxes
whose foot is within 2.5 ft of an in-frame SportVU player our detector's feet miss (source 2). A source 2 box with IoU
at least 0.5 to a source 1 box is written once. gsw_ind_2016 (D98) and hou_cle_2016 (new-arena check) are not in it.
Window clips were decoded from frame 0 without seeking. The corpus lives in the P8 workspace (e1_corpus/), never in
the repository; meta.json records the games, teacher, weights and their sha256, thresholds, device, library versions,
code commit (da1097af) and the SportVU and camera-truth versions per game (D150 condition 4).

| Game | Frames | SportVU players in frame | Our misses | Source 1 boxes (a frame) | Source 2 boxes (a frame) | Source 2 also in source 1 | Labels written |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsw_bkn_2015 | 2,075 | 20,505 | 5,606 | 18,863 (9.0906) | 2,125 (1.0241) | 1,522 | 19,466 |
| gsw_cha_2016 | 1,366 | 13,450 | 5,326 | 12,516 (9.1625) | 1,123 (0.8221) | 980 | 12,659 |
| gsw_sac_2015 | 2,151 | 21,192 | 5,221 | 19,997 (9.2966) | 1,935 (0.8996) | 1,579 | 20,353 |
| All | 5,592 | 55,147 | 16,153 | 51,376 | 5,183 | 4,081 | 52,478 |

Consistency with the 400-frame gsw_bkn_2015 sample (a check, not a gate): source 1 9.0906 a frame against the
sample's 9.065 [8.9675, 9.1575]; source 2 1.0241 a frame against 1.0; our miss share 5,606 of 20,505 (0.2734) against
0.2656 [0.2497, 0.2818]. The full pass agrees with the sample.

What the counts say: most source 2 boxes (4,081 of 5,183) overlap an agreement box, so they are players both models
boxed whose foot our detector placed more than 2.5 ft from SportVU; only 1,102 boxes (603, 143 and 356) are players
our detector did not box in agreement with the teacher. The miss row moves through box placement as much as through
detection, as the source 2 reading above already said.

Compute: 1.21 CPU hours for YOLOX-X (4,354.7 s) and 0.106 for our detector (380.0 s); 1.35 lock hours in all (9 jobs,
including decoding and writing images). Disk: 1.94 GB (1,940,398,183 bytes; images 1.8 GB, labels 22 MB).

Training config, not run (Colab is Lucien's, D137): tools/p8_e1_colab.py, one seed first, the production detector's
recipe (yolov8m.pt, 100 epochs, image size 640, batch 16, patience 20), one class (player), a whole-window
validation split (last 10 percent of each game's windows) for early stopping only; about 0.5 x 5,592 / 654 = 4.3
T4-hours a seed by b1's scaling (an estimate). Adoption stays the pinned scorecard with a game-bootstrap interval (D5,
D150 (3)). Note for the lead: the corpus games are the same R1.3 games the scorecard baseline pools, so a model trained
on them and scored on them is in sample there; hou_cle_2016 is the out-of-corpus check.
