#!/bin/bash
# tools/p8_truth_cpu.sh: the CPU-detector camera truth rebuilds of docs/research/P8_truth_arenas.md section 16
# (hou_cle_2016, then gsw_bkn_2015, then gsw_sac_2015), the same steps as gsw_cha_2016 (section 15): link, label one
# job a section (V3 court, gate v2, detector on CPU), fit; then the report-only comparisons with the MPS truth
# (tools.p8_truth_moved, tools.p8_truth_moved_dh), which read stored files and take no lock.
#
#     V3_CV_PYTHON=<anaconda python> bash tools/p8_truth_cpu.sh [GAME ...]        (default: the registered order)
#
# One heavy-job lock per model step. Resumable: a step with a done marker is skipped and built windows are cached.
# A failed step is retried at most twice, then the driver stops. A game whose lock time passes its cap (1.5 times its
# estimate, section 16) stops the driver before its next step. A file named STOP in the reference root stops it before
# the next step. Nothing is written to the MPS root (work): hou_cle_2016's inputs are linked from it read only.
set -u -o pipefail
MPS="${V3_P8_MPS_ROOT:-$HOME/Developer/nba-v3-worktrees/_p8-data/work}"
R="${V3_P8_REF_ROOT:-$HOME/Developer/nba-v3-worktrees/_p8-data/work_ref}"
CV="${V3_CV_PYTHON:?set V3_CV_PYTHON to the interpreter that has ultralytics and easyocr (CLAUDE.md: the anaconda one)}"
SYS=/usr/local/bin/python3
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$R/logs/truth_cpu.log"
ORDER="hou_cle_2016 gsw_bkn_2015 gsw_sac_2015"
cap_of() { case "$1" in hou_cle_2016) echo 19190;; gsw_bkn_2015) echo 16470;; gsw_sac_2015) echo 15300;; *) echo 0;; esac; }
GAMES="${*:-$ORDER}"
cd "$REPO" || exit 2
mkdir -p "$R/logs"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
held() { grep -h "^\[heavy\] 'p8 cpu truth $1 .*done in" "$LOG" 2>/dev/null | sed -E 's/.*done in ([0-9]+) s.*/\1/' | awk '{s+=$1} END {print s+0}'; }
done_marker() { [ -f "$R/meta/done/$1.json" ]; }
step() {   # step <game> <name> <done marker> <p8_frozen args...>: at most two retries
  local game="$1" name="$2" marker="$3"; shift 3
  if done_marker "$marker"; then say "skip $marker (done)"; return 0; fi
  local cap; cap=$(cap_of "$game")
  for attempt in 1 2 3; do
    if [ -f "$R/STOP" ]; then say "STOP file present; stopping before $marker"; exit 3; fi
    local h; h=$(held "$game"); if [ "$h" -gt "$cap" ]; then say "KILL: $game lock time $h s passed its cap $cap s"; exit 4; fi
    say "start $marker attempt $attempt"
    "$SYS" -m export.heavy --name "p8 cpu truth $game $name" -- env V3_P8_ROOT="$R" P8_DEVICE_DETECTOR=cpu "$CV" -m tools.p8_frozen "$@" \
      2>&1 | grep -v "Warning\|warn(\|pandas\|NUMEXPR\|NNPACK" | tee -a "$LOG" | tail -3
    local rc=${PIPESTATUS[0]}
    say "end $marker attempt $attempt rc=$rc"
    [ "$rc" -eq 0 ] && return 0
  done
  return 1
}
link_from_mps() {   # a channel game (not R1.3): its sections, moments, time maps, layout and mirror from the MPS root
  local game="$1" moments
  moments=$("$SYS" -c "import json,sys; r=json.load(open('reports/research/p8_registry.json')); g=r.get('games', r); print(g[sys.argv[1]]['sportvu'])" "$game") || return 1
  mkdir -p "$R/video" "$R/sportvu" "$R/sync" "$R/meta" "$R/reports"
  local n=0 src
  for src in "$MPS/video/${game}"_s[0-9][0-9].mp4 "$MPS/sportvu/${moments}_moments.json" "$MPS/sync/${game}"_s[0-9][0-9]_local_timemap.json \
             "$MPS/meta/${game}_layout.json" "$MPS/reports/sportvu_direction_${game}.json"; do
    [ -e "$src" ] || { say "FAIL link: missing $(basename "$src")"; return 1; }
    local d
    case "$src" in
      */video/*) d="$R/video";; */sportvu/*) d="$R/sportvu";; */sync/*) d="$R/sync";; */meta/*) d="$R/meta";; *) d="$R/reports";;
    esac
    [ -e "$d/$(basename "$src")" ] || { ln -s "$src" "$d/$(basename "$src")"; n=$((n + 1)); }
  done
  say "linked $n files of $game from the MPS root"
}

say "=== CPU truth driver start: $GAMES"
for TAG in $GAMES; do
  [ "$(cap_of "$TAG")" -gt 0 ] || { say "FAIL: $TAG is not in this plan"; exit 2; }
  case "$TAG" in
    gsw_*) if ! done_marker "link__${TAG}"; then
             env V3_P8_ROOT="$R" P8_DEVICE_DETECTOR=cpu "$CV" -m tools.p8_frozen link "$TAG" 2>&1 \
               | grep -v "Warning\|warn(\|pandas\|NUMEXPR" | tee -a "$LOG" | tail -2
             done_marker "link__${TAG}" || { say "FAIL link $TAG"; exit 5; }
           fi;;
    *) link_from_mps "$TAG" || exit 5;;
  esac
  SECTIONS=$(ls "$R/video" | sed -n "s/^\(${TAG}_s[0-9][0-9]\)\.mp4$/\1/p" | sort)
  say "$TAG sections: $(echo $SECTIONS | wc -w)"
  for s in $SECTIONS; do
    [ -e "$R/sync/${s}_local_timemap.json" ] || { say "skip $s (no time map)"; continue; }
    step "$TAG" "label $s" "label__${TAG}__${s}" label "$TAG" --section "$s" || { say "KILL: label $s failed three times; stop and diagnose"; exit 10; }
  done
  step "$TAG" "fit" "fit__${TAG}" fit "$TAG" || { say "FAIL fit $TAG"; exit 11; }
  say "=== $TAG truth built; lock held: $(held "$TAG") s"
  "$SYS" -m tools.p8_truth_moved "$TAG" --mps-root "$MPS" --cpu-root "$R" 2>&1 | tee -a "$LOG" || say "truth_moved $TAG failed"
  "$SYS" -m tools.p8_truth_moved_dh "$TAG" --mps-root "$MPS" --cpu-root "$R" 2>&1 | tail -25 | tee -a "$LOG" || say "truth_moved_dh $TAG failed"
done
say "=== CPU truth driver done"
