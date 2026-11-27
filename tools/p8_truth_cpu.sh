#!/bin/bash
# tools/p8_truth_cpu.sh: CPU-detector camera truth built with the tracker reset fix, and its acceptance
# (docs/research/P8_truth_arenas.md section 16 version 4; PLAN.md D192).
#
#     V3_CV_PYTHON=<anaconda python> bash tools/p8_truth_cpu.sh truth  [GAME ...]
#     V3_CV_PYTHON=<anaconda python> bash tools/p8_truth_cpu.sh accept [GAME ...]
#
# truth, per game in the order gsw_cha_2016, hou_cle_2016, gsw_bkn_2015, gsw_sac_2015: move the game's label builds
# made without the fix and the truth fitted from them aside, kept (tools.p8_truth_fixprep supersede; clips and window
# time maps stay); link; label one job a section (V3 court, gate v2, detector on CPU, P8_TRACKER_RESET_FIX=1); check
# that every label window carries the fix (tools.p8_truth_fixprep label-check); fit; the report-only comparisons with
# the MPS truth (tools.p8_truth_moved, tools.p8_truth_moved_dh).
# accept, per game, after the reset-fix arm's builds and passing equivalence for that game: builds/r23 composed fixed
# with fixed (tools.p8_truth_cpu_r23 compose: the arm's builds on reset windows, the workspace baseline elsewhere),
# then, only if every label window has one, check C (tools.p8_truth_v2 measure-dev) and the acceptance checks
# (tools.p8_truth_accept), both unchanged. A game not ready is reported and skipped; rerun accept later.
#
# One heavy-job lock per model step (label, fit, measure-dev); the file-only steps take none. Resumable: a step with a
# done marker (or a measure-dev record) is skipped and built windows are cached (a cached window built without the fix
# is refused by tools.p8_device.check_cached). A failed step is retried at most twice, then the driver stops. A game
# whose lock time in this driver passes its cap (1.5 times its estimate) stops the driver before its next step. A file
# named STOP in the reference root stops it before the next step and before each game. Nothing is written to the P8
# workspace (work). Start guards (version 4.2): P8_DEVICE, P8_DEVICE_COURT, P8_DEVICE_GATE and P8_MPS_CLIP_FIX unset;
# the build interpreter's opencv, torch and ultralytics equal tools.p8_resetfix.PINNED_VERSIONS; for the truth phase,
# resetfix/smoke_check.json shows "passes": true.
set -u -o pipefail
MPS="${V3_P8_MPS_ROOT:-$HOME/Developer/nba-v3-worktrees/_p8-data/work}"
R="${V3_P8_REF_ROOT:-$HOME/Developer/nba-v3-worktrees/_p8-data/work_ref}"
CV="${V3_CV_PYTHON:?set V3_CV_PYTHON to the interpreter that has ultralytics and easyocr (CLAUDE.md: the anaconda one)}"
SYS=/usr/local/bin/python3
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$R/logs/truth_fix.log"
PHASE="${1:-}"; shift || true
ORDER="gsw_cha_2016 hou_cle_2016 gsw_bkn_2015 gsw_sac_2015"
cap_of() {   # lock seconds per phase and game: 1.5 times the section 16 version 4 estimate
  case "$PHASE:$1" in
    truth:gsw_cha_2016) echo 14630;; truth:hou_cle_2016) echo 19190;; truth:gsw_bkn_2015) echo 16470;; truth:gsw_sac_2015) echo 15300;;
    accept:gsw_cha_2016|accept:hou_cle_2016|accept:gsw_bkn_2015|accept:gsw_sac_2015) echo 450;;
    *) echo 0;;
  esac
}
GAMES="${*:-$ORDER}"
cd "$REPO" || exit 2
same=$("$SYS" -c "import os,sys; print(int(os.path.realpath(sys.argv[1]) == os.path.realpath(sys.argv[2])))" "$MPS" "$R")
[ "$same" = 0 ] || { echo "the reference root and the P8 workspace are the same folder" >&2; exit 2; }
"$SYS" -m tools.p8_truth_cpu_r23 detector-check || { echo "models/player_detector.pt is not the r23 detector" >&2; exit 2; }
case "$PHASE" in truth|accept) ;; *) echo "usage: p8_truth_cpu.sh truth|accept [GAME ...]" >&2; exit 2;; esac
[ -z "${TRACKER_REID:-}" ] || { echo "TRACKER_REID must be unset (P8_resetfix.md)" >&2; exit 2; }
"$SYS" -c "from tools.p8_device import RESET_FIX_FLAG, reset_fix_requested; from tools.p8_resetfix import apply_reset_fix" 2>/dev/null \
  || { echo "the reset fix code (tools.p8_resetfix, P8_TRACKER_RESET_FIX in tools.p8_device) is not on this branch" >&2; exit 2; }
for v in P8_DEVICE P8_DEVICE_COURT P8_DEVICE_GATE P8_MPS_CLIP_FIX; do   # only the detector flag and the reset fix are set
  [ -z "${!v:-}" ] || { echo "$v must be unset (section 16 version 4.2)" >&2; exit 2; }
done
"$CV" -c "import sys, cv2, torch, ultralytics
from tools.p8_resetfix import library_versions, PINNED_VERSIONS
v = library_versions(); print('build libraries', v)
sys.exit(0 if v == PINNED_VERSIONS else 1)" \
  || { echo "the build interpreter's library versions are not tools.p8_resetfix.PINNED_VERSIONS" >&2; exit 2; }
if [ "$PHASE" = truth ]; then
  "$SYS" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('passes') is True else 1)" "$R/resetfix/smoke_check.json" 2>/dev/null \
    || { echo "the reset-fix smoke has not passed (resetfix/smoke_check.json)" >&2; exit 2; }
fi
mkdir -p "$R/logs"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
held() { grep -h "^\[heavy\] 'p8 fixtruth $PHASE $1 .*done in" "$LOG" 2>/dev/null | sed -E 's/.*done in ([0-9]+) s.*/\1/' | awk '{s+=$1} END {print s+0}'; }
done_marker() { [ -f "$1" ]; }
mk() { echo "$R/meta/done/$1.json"; }
step() {   # step <game> <name> <done file> <module> <args...>: one lock, at most two retries
  local game="$1" name="$2" marker="$3" mod="$4"; shift 4
  if done_marker "$marker"; then say "skip $name of $game (done)"; return 0; fi
  local cap; cap=$(cap_of "$game")
  for attempt in 1 2 3; do
    if [ -f "$R/STOP" ]; then say "STOP file present; stopping before $name of $game"; exit 3; fi
    local h; h=$(held "$game"); if [ "$h" -gt "$cap" ]; then say "KILL: $game lock time $h s passed its cap $cap s"; exit 4; fi
    say "start $game $name attempt $attempt"
    "$SYS" -m export.heavy --name "p8 fixtruth $PHASE $game $name" -- env V3_P8_ROOT="$R" P8_DEVICE_DETECTOR=cpu P8_TRACKER_RESET_FIX=1 "$CV" -m "$mod" "$@" \
      2>&1 | grep -v "Warning\|warn(\|pandas\|NUMEXPR\|NNPACK" | tee -a "$LOG" | tail -3
    local rc=${PIPESTATUS[0]}
    say "end $game $name attempt $attempt rc=$rc"
    [ "$rc" -eq 0 ] && return 0
  done
  return 1
}
link_from_mps() {   # a channel game (not R1.3): its sections, moments, time maps, layout and mirror from the workspace
  local game="$1" moments
  moments=$("$SYS" -c "import json,sys; r=json.load(open('reports/research/p8_registry.json')); g=r.get('games', r); print(g[sys.argv[1]]['sportvu'])" "$game") || return 1
  mkdir -p "$R/video" "$R/sportvu" "$R/sync" "$R/meta" "$R/reports"
  local n=0 src d
  for src in "$MPS/video/${game}"_s[0-9][0-9].mp4 "$MPS/sportvu/${moments}_moments.json" "$MPS/sync/${game}"_s[0-9][0-9]_local_timemap.json \
             "$MPS/meta/${game}_layout.json" "$MPS/reports/sportvu_direction_${game}.json"; do
    [ -e "$src" ] || { say "FAIL link: missing $(basename "$src")"; return 1; }
    case "$src" in
      */video/*) d="$R/video";; */sportvu/*) d="$R/sportvu";; */sync/*) d="$R/sync";; */meta/*) d="$R/meta";; *) d="$R/reports";;
    esac
    [ -e "$d/$(basename "$src")" ] || { ln -s "$src" "$d/$(basename "$src")"; n=$((n + 1)); }
  done
  say "linked $n files of $game from the workspace"
}
sections_of() { ls "$R/video" | sed -n "s/^\(${1}_s[0-9][0-9]\)\.mp4$/\1/p" | sort; }

say "=== reset-fix truth driver, phase $PHASE: $GAMES"
for TAG in $GAMES; do
  [ "$(cap_of "$TAG")" -gt 0 ] || { say "FAIL: $TAG is not in this plan"; exit 2; }
  if [ -f "$R/STOP" ]; then say "STOP file present; stopping before $TAG"; exit 3; fi     # before supersede moves anything
  if [ "$PHASE" = truth ]; then
    "$SYS" -m tools.p8_truth_fixprep supersede "$TAG" --ref-root "$R" 2>&1 | tee -a "$LOG" || { say "FAIL supersede $TAG"; exit 6; }
    case "$TAG" in
      gsw_*) if ! done_marker "$(mk "link__${TAG}")"; then
               env V3_P8_ROOT="$R" P8_DEVICE_DETECTOR=cpu P8_TRACKER_RESET_FIX=1 "$CV" -m tools.p8_frozen link "$TAG" 2>&1 \
                 | grep -v "Warning\|warn(\|pandas\|NUMEXPR" | tee -a "$LOG" | tail -2
               done_marker "$(mk "link__${TAG}")" || { say "FAIL link $TAG"; exit 5; }
             fi;;
      *) link_from_mps "$TAG" || exit 5;;
    esac
    SECTIONS=$(sections_of "$TAG")
    say "$TAG sections: $(echo $SECTIONS | wc -w)"
    for s in $SECTIONS; do
      [ -e "$R/sync/${s}_local_timemap.json" ] || { say "skip $s (no time map)"; continue; }
      step "$TAG" "label $s" "$(mk "label__${TAG}__${s}")" tools.p8_frozen label "$TAG" --section "$s" \
        || { say "KILL: label $s failed three times; stop and diagnose"; exit 10; }
    done
    "$SYS" -m tools.p8_truth_fixprep label-check "$TAG" --ref-root "$R" 2>&1 | tee -a "$LOG" \
      || { say "KILL: a label window of $TAG lacks the reset fix or the reference devices"; exit 7; }
    step "$TAG" "fit" "$(mk "fit__${TAG}")" tools.p8_frozen fit "$TAG" || { say "FAIL fit $TAG"; exit 11; }
    say "=== $TAG truth built; lock held: $(held "$TAG") s"
    "$SYS" -m tools.p8_truth_moved "$TAG" --mps-root "$MPS" --cpu-root "$R" 2>&1 | tee -a "$LOG" || say "truth_moved $TAG failed"
    "$SYS" -m tools.p8_truth_moved_dh "$TAG" --mps-root "$MPS" --cpu-root "$R" 2>&1 | tail -25 | tee -a "$LOG" || say "truth_moved_dh $TAG failed"
  else
    [ -f "$(mk "fit__${TAG}")" ] && "$SYS" -m tools.p8_truth_fixprep label-check "$TAG" --ref-root "$R" >/dev/null \
      || { say "CHECK C NOT RUN for $TAG: no fixed truth yet"; continue; }
    if ! "$SYS" -m tools.p8_truth_cpu_r23 compose "$TAG" --mps-root "$MPS" --ref-root "$R" 2>&1 | tee -a "$LOG"; then
      say "CHECK C NOT RUN for $TAG: the reset-fix arm's plan or passing equivalence is not there yet, or the composition stopped"
      continue
    fi
    if "$SYS" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['complete'] else 1)" "$R/builds/r23_checkc_${TAG}.json"; then
      step "$TAG" "measure-dev" "$R/truth_v2/dev/${TAG}.json" tools.p8_truth_v2 measure-dev "$TAG" || { say "FAIL measure-dev $TAG"; exit 14; }
      env V3_P8_ROOT="$R" "$SYS" -m tools.p8_truth_accept "$TAG" 2>&1 | tee -a "$LOG" | tail -4 || say "truth_accept $TAG failed"
    else
      say "CHECK C NOT RUN for $TAG: some V3 label windows have no verified r23 build (builds/r23_checkc_${TAG}.json); acceptance pending"
    fi
  fi
  say "=== $TAG phase $PHASE done; lock held: $(held "$TAG") s"
done
say "=== reset-fix truth driver, phase $PHASE, done"
