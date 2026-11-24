# P8 E7b: WASB's basketball model on DeepSportRadar's annotated balls (PLAN.md D165)

Status: stopped before any image was read, on 2026-11-23; the lead chose option 1 (stop), to be logged as D166. The set cannot be scored as WASB expects: each annotated
view holds two frames, and WASB's input is three consecutive frames. Written and committed before any image was
opened; the facts below come from the archive's file names and JSON annotations only.

## What the set holds (from the JSON and file names)

Source: DeepSportRadar basketball instants, basketball-instants-dataset.zip (Kaggle version 4, CC BY-NC-SA 4.0,
sha256 5a586195ac72ff853063aa85418578c64007ad8e547c38b2feee98ee43b38f0b; reports/research/p8_datasets_fetch.json).

- 364 instants (37 games, 15 French arenas, FIBA), each with 2 fixed Keemotion camera views (camcourt1, camcourt2),
  2336 x 1752 px, with per-view calibration (KK, R, T, distortion kc; centimetres, z pointing down).
- One ball annotation per instant: a 3D centre (cm), the view it was annotated in (`image` 0 or 1, read here as
  camcourt1 or camcourt2), and a `visible` flag: 320 visible, 44 not.
- Projected through that view's calibration (pinhole plus the kc distortion), every visible ball lands inside its
  image: 320 images from 37 games would form the image set. The projected ball radius (a 24 cm ball, radius 12 cm,
  focal length over depth) has median 12.4 px (10th to 90th percentile 9.1 to 15.9 px) at full resolution.
- Frames per view: the annotated frame (`_0.png`) and one more, 40 ms later (`_40.png`, 312 of the 320) or 80 ms
  later (`_80.png`, 8). The index lists a third offset near 1,000 ms for most instants, but no such image is in the
  archive. There is no third frame within the clip spacing WASB uses.

## Why it cannot be scored as WASB expects

WASB's released basketball model (configs model/wasb.yaml) takes 3 consecutive video frames stacked as 9 channels
and outputs 3 heatmaps, and its tracker runs over a sequence. A DeepSportRadar view gives 2 frames. Any triplet
would have to repeat one of them ([f0, f0, f40], [f0, f40, f40]) or mix in the 80 ms frame, which is an input the
model never saw in training (its motion cue is what separates the ball from look-alikes), so the result would
measure WASB on a made-up input, not WASB as released. Other differences, which would have been registered, are
smaller: fixed high-mounted cameras rather than broadcast views; a 4:3 image warped into the 512 x 288 input by the
repository's affine (scale 512 / 2336, about 0.22, with the frame letterboxed), giving a ball radius near 2.7 px at
the model's input.

## Options for the lead

1. Stop E7b here (this file's status).
2. Register a variant that repeats a frame (for example [f0, f0, f40], scored on f0), labelled "not as released",
   with its tolerance fixed from the annotation (for example the per-image projected ball radius at full resolution,
   or twice it), F1, precision, recall, the share of images with any WASB point, and a game bootstrap over the 37
   games. Its reading would bound WASB's single-frame detection on fixed cameras; it could not say how far WASB's
   miss rate on our broadcasts (E7a: no point on 364 of 806 in-image frames) is the model's own.
3. Look for human ball annotation on broadcast clips with consecutive frames (WASB's own test set needs the
   NUST-NBA181 videos, which are on request, D163 (3)).

Nothing was trained, tuned or run on DeepSportRadar, and no image was read.

## WASB's own basketball numbers, read at the source

Source: Tarashima, Haq, Wang and Tagawa, "Widely Applicable Strong Baseline for Sports Ball Detection and Tracking",
BMVC 2023, arXiv 2311.05237 (PDF read 2026-11-23); the repository (github.com/nttcom/WASB-SBDT, 923462ca) says its
evaluation commands "reproduce the results of Table 2 and Table 3" of the paper (GET_STARTED.md) and holds no results
table of its own.

- Test set (Table 1, "Basketball"): 11 games, 432 clips, 31,104 frames; resolution "1920 x 1080" ("the majority of
  image resolution in the dataset"); FPS "N/A"; average ball displacement between consecutive frames 33.9 +/- 21.4 px.
  Training: 70 games, 3,392 clips, 244,224 frames. Video from the NBA dataset of reference [91] (the NUST-NBA181
  clips); "we manually annotated ball positions to 45% (81/181 games) of the whole video clips, resulting in 275,328
  annotated images composed of 3,824 video clips"; "camera frequently moves and zooms in rapidly to follow where play
  happens".
- Metric (section 5.1): "With a distance threshold tau [pixel], for each frame we calculate the distance between a
  predicted ball position and a ground truth to classify the prediction into true positive, true negative, false
  positive or false negative." Table 2: "We set the distance threshold tau = 4 [pixel] to compute F1, Accuracy (Acc.)
  and Average Precision (AP), all of which are shown as percentages." In the code the distance is taken in the
  original image's pixels (the prediction mapped back through the inverse affine, src/detectors/postprocessor.py,
  against the annotation, src/utils/evaluator.py). Input: "each image is resized to 288 x 512" (section 5.2).
- Results, Basketball (Table 2): WASB (Step=3): F1 80.6, Acc. 71.3, AP 71.5, 30.2 FPS. WASB (Step=1): F1 82.6,
  Acc. 73.4, AP 77.1, 22.3 FPS. E7a ran the Step=3 setting.

What this says for E7a, as context only: on its own broadcast test data, with hand-clicked centres and a 4 px
tolerance at 1080p, the released model reaches F1 80.6. E7a, on 2015-16 broadcasts at 720p and 59.94 fps scored
against the projected SportVU ball, gives pooled F1 0.0064 at 2.67 px (4 px at 1080p scaled to 720p) and 0.1807 at
the registered 32.07 px, and no point at all on 364 of 806 in-image frames. The gap mixes domain shift (season,
broadcast, frame rate, our 12-frame segments) with truth error; these numbers do not separate the two.

## E-BARD's detection set: consecutive frames? (file names and annotation JSON only)

From E-BARD-detection/all.zip (cc-by-4.0, reports/research/p8_datasets_fetch.json), read without opening an image:

- 1,800 images, 1280 x 720 (COCO metadata), named `<away>-vs-<home>-<game id>_<index>`, 30 per game over 60 games
  (59 regular season 2024-25 and the 2024-25 Cup final). YOLO classes (data.yaml): basketball, hoop, player,
  referee; 1,496 ball boxes on 1,465 images.
- The index runs from 0 to 5,995 (per game maximum 692 at the 10th percentile, 3,084 median, 5,495 at the 90th).
  Within a game, 720 pairs of images have indices one apart; runs of consecutive indices reach 30 (all 30 images of
  4 games). 313 of the 1,465 ball images have both neighbours (index minus 1 and plus 1) in the set, 318 have the
  two before and 315 the two after.
- Whether one index step is one video frame is not stated in the files (the card says only "1,800 frames,
  systematically sampled at 30 frames per game" from "60 full-game recordings"). If it is, those 313 images would
  give WASB its 3 consecutive frames with a hand-drawn ball box as truth (the box centre as the point, a tolerance
  from the box size), on NBA broadcast at 720p.

Candidate for a later registration, not registered and not run: E-BARD's ball images whose index neighbours exist,
once the E-BARD paper or its authors confirm what the index counts. Its 2024-25 frames fall under D149 (detector
use only, nothing into a play-by-play lane) and its D150 screen drops the MIA, TOR, LAC and LAL home games and the
unresolved Cup final (reports/research/p8_datasets_fetch.json, home_screen).

## What E-BARD's detection index counts: not stated at the source (2026-11-23, under D37)

Read at the source, without contacting anyone:

- The four Hugging Face cards (GabrieleGiudici/E-BARD-detection, -TeamAttribution, -JerseyNumberRecognition,
  -ObjectClassification) name the paper ("E-BARD: A Multi-Task Extension of the Basketball Action Recognition Dataset
  for Player Detection, Team Attribution and Jersey Number Recognition") but give no link to it. On sampling, the
  detection card says only "Contains 1,800 frames, systematically sampled at 30 frames per game" and "Extracted from
  60 full-game recordings from the 2024-2025 NBA season". The jersey card says its stubs are "16 consecutive frames"
  of ByteTrack tracks, which is about a different set.
- The project page (bodai.unibs.it, "E-BARD") describes the contributions and links the code at
  github.com/GabrieleGiudic/E-BARD; it says nothing about frame sampling.
- The repository (license file "Creative Commons Attribution 4.0 International (CC BY 4.0)", last push 2026-03-09)
  holds detection training, evaluation and a YOLO-to-COCO converter (detection/src/to_coco.py), which reads images
  already extracted. No file extracts or samples frames from video, and its JSON refers to detection images only by
  the same `<game>_<index>` names (for example `../detection/data/coco/test/sac-vs-hou-0022400862_004924.jpg`).
- The paper itself was not found: an arXiv search for "E-BARD" and for BARD with basketball returns no match, and
  none of the pages above links a PDF.

So no source says whether one index step is one video frame. As the lead set, E7c is not registered and nothing
was run. A later step would need the paper, the authors' answer, or the BARD clips with a documented frame mapping.
