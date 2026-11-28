# P8 label sources for crowded, overlapping players (survey, 2026-11-27)

A survey only: nothing was downloaded, signed up for, run or trained. Every fact below was read at its source on
2026-11-27 by read-only fetch agents (WebFetch) or through the Hugging Face hub API; the URL is given with each fact.
Quotes are short and verbatim as returned. Where a page states nothing, the entry says "not stated". The ranking and
the proposed measurement are reasoning, not results; the measurement is a later design under
docs/research/TRAINING_PROTOCOL.md.

## 1. Why

- D208 and its supplement (P8_recallcause.md 9.3, P8_recallcause_supp.md section 10, post hoc on all 120 windows):
  the detector stage (C1: no box of its own after NMS, and not a suppressed candidate of its own) holds about 0.69 of
  the no-own-box misses (0.693 for both units; P-i 0.615, D-i 0.765).
- C1a_other, a player whose image position lies inside only another player's box, is about 0.40 of D-i (D208 9.3,
  0.400 on the kept windows). That is the case one box covering two players produces.
- D183 paused detector fine-tuning until a label source reaches these misses. E1b's teacher (YOLOX-X trained on
  SportsMOT) reached 31.2 percent of r23's raw misses, four fifths of it foot placement on players r23 already boxes
  (D183); its reach on C1 specifically was not measured.
- The NMS arm (P8_nmsarm.md section 15) showed that keeping more of r23's own candidates adds duplicates as fast as it
  finds players. A label source has to teach separate boxes for overlapping players, not looser suppression.

What a source must give for C1, in order of weight: (a) a separate box for each person even when partly hidden behind
another (the property C1a_other needs); (b) a full-body (amodal) box, so the foot sits where the feet are, not where
the visible part ends; (c) a view close to the broadcast wide shot; (d) players separable from referees, coaches and
spectators, or at least no harm from them.

## 2. Conditions (D150, Lucien, 2026-11-22)

(1) A licence row in docs/DATA_SOURCES.md before any download; non-commercial is acceptable; nothing redistributed.
(2) Screening before any training: every clip of a spent held-out game (12.16.2015 PHX at GSW; the Cleveland arena,
sportvu/splits.json heldout_game and heldout_arena) and of a home game of MIA, TOR, LAC or LAL in any season (D85,
D129) is dropped; a set whose games cannot be identified is used with that residual risk disclosed. (3) Adoption
unchanged: the pinned scorecard on training games with a game-bootstrap interval excluding zero (D5); the D85 test set
once at the end of P8. (4) Every training run records the datasets and versions it used.

## 3. Outside labelled datasets

| Set | Licence (source) | Separate box for an occluded person; extent | Size; view; resolution | Held-out and D85 screen |
|---|---|---|---|---|
| CrowdHuman | "non-commercial research and educational purposes"; "You will NOT distribute the above images" (crowdhuman.org/download.html) | yes, full, visible and head boxes: "If the individual is partly occluded, the annotator is required to complete the invisible part and draw a full bounding box" (arxiv.org/html/1805.00123, 3.2) | 15,000 train, 4,370 val, 5,000 test images; 22.64 persons per training image; 2.40 person pairs per image at IoU above 0.5 (Table 4); web images by search keyword, "numerous viewpoints", keywords include "students playing basketball" (3.1); resolution not stated | web images, games not identifiable; held-out risk negligible, disclosed |
| SportsMOT | "CC BY-NC 4.0"; "You agree not to distribute the SportsMOT dataset without prior written permission" (github.com/MCG-NJU/SportsMOT) | yes, players only: "The annotators are asked to predict the bounding box of the athlete in the case of occlusion," "as long as the athletes have a visible part of body" (arxiv.org/html/2304.05170v2, 3.1) | 240 clips of basketball, football and volleyball, 45 train and 45 val labelled; basketball averages 845.4 frames and 10 tracks per clip; "720P resolution, 25 FPS"; "Only the overhead shots of sports game scenes are used" (README, paper) | sequence names look like YouTube ids; games not listed; already in DATA_SOURCES.md and used with the risk disclosed (D150 (2)) |
| SHOT | annotations "CC BY-NC 4.0"; footage "excluded" (huggingface.co/datasets/muyu111/basketball, LICENSE) | each of the 10 players has its own id and box on keyframes (README); box extent under occlusion not stated; boxes propagated by MixSort between keyframes (arxiv.org/html/2509.20715v3) | 1,861 cases on the card (1,979 clips in the paper), 5 views, NBA via Tencent Sports, 1080p | team codes in clip names (for example "CLEvsGS-49-view1-5"); already downloaded and screened conservatively (P8_datasets.md: CLE, PHX with GSW, MIA, TOR, LAC, LAL dropped) |
| E-BARD detection | "cc-by-4.0" tag; footage rights not addressed (huggingface.co/datasets/GabrieleGiudici/E-BARD-detection, github.com/GabrieleGiudic/E-BARD) | "22,210 manual, object-level annotations"; occlusion rule not stated | 1,800 frames, 30 per game, 60 games of 2024-25; BARD footage 720p | game ids; already downloaded and screened (P8_datasets.md: 51 of 60 games kept) |
| MOT17, MOT20 | "CC BY-NC-SA 3.0" (motchallenge.net/data/MOT17/, /MOT20/) | yes, full extent: "If the person is partially occluded, the extent is estimated" (arxiv.org/html/1603.00831, II-B); no visible box, a visibility ratio per box | MOT17 7 train sequences, 5,316 frames, mostly 1920x1080; MOT20 8,931 train frames, 127 pedestrians per frame; street, mall and station scenes, elevated views | no sport; none |
| CityPersons | no licence in the repository; images under the Cityscapes terms, non-commercial, no redistribution (github.com/cvgroup-njust/CityPersons, cityscapes-dataset.com/license/) | pedestrians and riders: full box from a head-to-feet line at "a fixed aspect ratio (0.41)", plus a visible box from the mask; groups get one box (arxiv.org/html/1702.05693, 3.1) | 5,000 images, about 7 persons per image; car-mounted street views | none |
| DeepSportradar instants | "CC BY-NC-ND 4.0" (GitHub README); Kaggle metadata says CC BY-NC-SA 4.0, a conflict to resolve with the owner | one mask per person, visible part only ("leading each other to be split in disconnected parts", arxiv.org/html/2208.08190v1, 3.3): no full-body extent | 223 train images (1,674 annotations); fixed sideline arena cameras of the French LNB Pro A, 2 to 5 megapixels | French arenas named; none of ours |
| TeamTrack (basketball part) | "MIT License", text names "Software" (github.com/AtomScott/TeamTrack) | occlusion rule not stated | fisheye 6K indoor and drone 4K outdoor views of university games | not broadcast; none |

Not verifiable on 2026-11-27: Roboflow Universe projects (P8_datasets.md: HTTP 403 to automated reads);
huggingface.co/datasets/koppolusameer/basketball-coco-player-detection (no licence and no source stated on the hub
listing). The hub also lists CrowdHuman mirrors tagged cc-by-nc-4.0 (for example sshao0516/CrowdHuman); the official
site's terms above govern.

## 4. Off-the-shelf detectors trained on crowds

| Detector | Code licence (source) | Weights: training data; access | How it handles overlap; CrowdHuman numbers |
|---|---|---|---|
| Iter-Deformable-DETR (CVPR 2022) | README: "released under MIT License"; no LICENSE file found (github.com/zyayoung/Iter-Deformable-DETR) | CrowdHuman; R-50 and Swin-L checkpoints on Google Drive (README) | end to end without NMS, queries accepted progressively (arxiv.org/abs/2203.07669); README table: R-50 AP 92.1, MR-2 41.5, JI 84.0; Swin-L AP 94.1, MR-2 37.7, JI 87.1 |
| CrowdDet (CVPR 2020) | "MIT License", "Copyright (c) 2020 Megvii Technology" (github.com/Purkialo/CrowdDet, now xg-chu/CrowdDet) | CrowdHuman (benchmark of its table); Google Drive and Baidu | "let each proposal predict a set of correlated instances", "EMD Loss and Set NMS" (arxiv.org/abs/2003.09163); README: RCNN EMD with RM AP 0.9097, MR 0.4102, JI 0.8271 |
| ByteTrack YOLOX-X (mot17) | "MIT License" (github.com/ifzhang/ByteTrack) | "Train on CrowdHuman, MOT17, Cityperson and ETHZ"; Google Drive | ordinary NMS; overlap handled in association ("low score detection boxes ... recover true objects"); no CrowdHuman detection numbers |
| MixSort YOLOX-X (SportsMOT), the current teacher | "MIT License" (github.com/MCG-NJU/MixSort) | yolox_x_sports_train (by its name SportsMOT train, the README does not say); already held in the P8 workspace (DATA_SOURCES.md) | ordinary NMS; association by appearance; no CrowdHuman numbers |
| Deep-EIoU | no licence found (github.com/hsiangwei0903/Deep-EIoU) | not stated | not described on the page; excluded for want of a licence |

None of the detector repositories states a licence for its weights; each was trained on non-commercial data
(CrowdHuman, MOT, Cityscapes, SportsMOT), so D150 (1)'s non-commercial use covers them and the DATA_SOURCES.md row
must name the training data's terms. Iter-Deformable-DETR's licence rests on its README sentence alone (no LICENSE
file), a gap to record in its row.

## 5. Ranking by expected reach on C1 misses per cost (reasoning, not measured)

Reach is unmeasured for every candidate on C1. The ranking weighs (a) to (d) of section 1 against the cost to get a
first measurement.

1. **The held teacher, YOLOX-X SportsMOT (MixSort).** In domain (broadcast-style basketball, players only), trained on
   labels that box an occluded athlete in full. No acquisition cost: weights and code already pass D150 and D131. Its
   C1 reach is the cheapest unknown to remove, and the bar any new source must beat.
2. **Iter-Deformable-DETR, CrowdHuman (R-50, then Swin-L).** Built for exactly the C1a_other case (overlapping people
   without NMS), full-body boxes learned from CrowdHuman's amodal labels. Out of domain (web images), and it boxes
   every person (spectators, referees, coaches), so reach must be read against SportVU positions. One weights
   download after a DATA_SOURCES.md row.
3. **CrowdDet, CrowdHuman.** The same target by a different mechanism (one proposal, several boxes; set NMS); a clean
   MIT licence file. Ranked under Iter-Deformable-DETR on its own table's numbers.
4. **ByteTrack YOLOX-X (CrowdHuman, MOT17, CityPersons, ETHZ).** Amodal crowd training but ordinary NMS; a cheap
   second baseline beside the held teacher (same architecture), to separate the effect of crowd training data from
   that of the overlap mechanism.
5. **SHOT and E-BARD labels for fine-tuning r23.** In domain and already downloaded and screened, every player boxed,
   but their extent rule under occlusion is not stated and any use costs a training run under the protocol; first
   check, on stored files, how often two of their boxes overlap at an IoU like C1a_other's.
6. **CrowdHuman labels for fine-tuning r23.** The richest amodal overlap labels, out of domain, a training run.
7. Lower: MOT17 and MOT20 (amodal, no sport), CityPersons (fixed aspect, street), DeepSportradar (visible-part masks,
   no full extent, ND licence conflict), TeamTrack (not broadcast). Deep-EIoU is out (no licence).

## 6. The cheapest measurement of reach (a proposal; a later design under the training protocol)

Measure, on stored C1 frames, how many C1 misses a teacher reaches, for the held teacher (1) and the top new one (2),
before any training or label build.
- **C1 misses.** The supplement's classification, recomputed from the stored candidates (recallcause/ for its 46
  certified windows, recallsupp/ for its 74 re-read ones) with tools/p8_recallsupp.py's classify_window, gives each C1
  miss's window, frame, SportVU pid and subclass (C1a_own, C1a_other, C1b, C1c). No detector; about the supplement's
  D205 row time under the lock.
- **Frames.** The C1 misses' frames, decoded with the build's read path (build_read), as the supplement read them.
- **Reach.** A C1 miss is reached when the teacher has a box whose foot (bottom centre through H_truth, as D208)
  lies within 2.5 ft of the player and is the player's in a one-to-one assignment of teacher feet to all ten SportVU
  players, so a box shared with a neighbour counts once. Reported per subclass, with a window bootstrap.
- **Cost rows, beside reach.** The teacher's boxes with no player within 3 ft (a ghost proxy) and its duplicates
  (players with two feet within 2.5 ft), the NMS arm's lesson.
- **Reading.** Reach on raw frames is a ceiling: E1's measured raw-to-build transfer (D183: placement 0.24 to 0.38,
  presence 0.36 to 0.62) applies before any build effect is claimed; the design must set its bar after that
  adjustment, and against r23 itself (the conf 0.25 boxes already reach under 0.04 of C1c, D208).
- **Cost.** The held teacher: one pass on the C1 frames on the CPU detector platform, no download. Iter-Deformable-DETR:
  a DATA_SOURCES.md row, one weights download, one pass. A window-stratified sample of the C1 frames keeps each pass
  near 30 lock minutes; the design fixes the sample before any number. hou_cle_2016 stays unspent.

## 7. Sources (all read 2026-11-27)

CrowdHuman: https://www.crowdhuman.org/, https://www.crowdhuman.org/download.html, https://arxiv.org/abs/1805.00123,
https://arxiv.org/html/1805.00123. MOT: https://motchallenge.net/data/MOT17/, https://motchallenge.net/data/MOT20/,
https://arxiv.org/html/1603.00831, https://arxiv.org/html/2003.09003. CityPersons: https://github.com/cvgroup-njust/CityPersons,
https://arxiv.org/html/1702.05693, https://www.cityscapes-dataset.com/license/. SportsMOT: https://github.com/MCG-NJU/SportsMOT,
https://arxiv.org/html/2304.05170v2. DeepSportradar: https://github.com/DeepSportradar/instance-segmentation-challenge,
https://arxiv.org/html/2208.08190v1, Kaggle metadata https://www.kaggle.com/api/v1/datasets/view/deepsportradar/basketball-instants-dataset.
E-BARD: https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection, https://github.com/GabrieleGiudic/E-BARD.
SHOT: https://huggingface.co/datasets/muyu111/basketball (and its LICENSE), https://arxiv.org/html/2509.20715v3.
TeamTrack: https://github.com/AtomScott/TeamTrack, https://atomscott.github.io/TeamTrack/. Detectors:
https://github.com/zyayoung/Iter-Deformable-DETR, https://arxiv.org/abs/2203.07669, https://github.com/Purkialo/CrowdDet,
https://arxiv.org/abs/2003.09163, https://github.com/ifzhang/ByteTrack, https://github.com/MCG-NJU/MixSort,
https://github.com/hsiangwei0903/Deep-EIoU. Hugging Face hub search (queries "basketball player detection" and
"crowdhuman", through the hub API). Failed or blocked: the Kaggle HTML page (JavaScript only), the Deep-EIoU paper on
openaccess.thecvf.com (HTTP 403), Iter-Deformable-DETR's LICENSE paths (HTTP 404), Roboflow Universe (403, per
P8_datasets.md). The detector quotes came through WebFetch's page summaries; the two that a design would rest on
(Iter-Deformable-DETR's licence sentence and its CrowdHuman table) are to be re-read verbatim before its
DATA_SOURCES.md row. A second, independent read the same day gave the same licence sentence and table and added that
the GitHub API reports "license": null for that repository (https://api.github.com/repos/zyayoung/Iter-Deformable-DETR);
the sister repository megvii-research/Iter-E2EDET has a LICENSE file (https://github.com/megvii-research/Iter-E2EDET).
