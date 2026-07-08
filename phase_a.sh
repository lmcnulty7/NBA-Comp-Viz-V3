#!/bin/bash
# Phase A: full-length runs on the 5 remaining/extended clips (free footage).
# Chain per clip: build (pregate) -> segment -> matchups. Then alignment + join.
set -u
cd "/Users/lucienmcnulty/Documents/Data Science projects/NBA Comp Viz New Project Version 3"
PY=/opt/anaconda3/bin/python
V="/Users/lucienmcnulty/Documents/Data Science projects/Basketball_Defensive_Vision/data/raw"
CLIPS="curry_q1_clip curry_classic_clip clip_40m00_48m00 clip_55m00_63m00 clip_70m00_78m00"

for c in $CLIPS; do
  echo "=== CLIP START $c $(date +%H:%M:%S) ==="
  $PY build_trajectories.py --source "$V/$c.mp4" --start 0 --max-frames 99999 \
      --stride 3 --pregate --no-video || { echo "=== CLIP FAILED $c (build)"; continue; }
  $PY segment_possessions.py --trajectories "data/tracking/${c}_trajectories.json" \
      || { echo "=== CLIP FAILED $c (segment)"; continue; }
  $PY matchup_metrics.py --clip "$c" --no-video \
      || { echo "=== CLIP FAILED $c (matchups)"; continue; }
  echo "=== CLIP DONE $c $(date +%H:%M:%S) ==="
done

echo "=== ALIGNMENT ==="
$PY align_outcomes.py --clips $CLIPS || echo "=== ALIGN FAILED"
echo "=== PHASE A COMPLETE $(date +%H:%M:%S) ==="
