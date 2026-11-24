# P8 E7b: WASB's basketball model on DeepSportRadar's annotated balls (PLAN.md D165)

Status: stopped before any image was read, on 2026-11-23. The set cannot be scored as WASB expects: each annotated
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
