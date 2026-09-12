"""JSON-only re-render of one matchup possession via matchup_metrics.render_review.
Never calls mm.main() (which rewrites tracked *_matchups.json)."""
import json, sys
from pathlib import Path
import matchup_metrics as mm
import config

stem = sys.argv[1] if len(sys.argv) > 1 else "gsw_hou_duel_s01"
target = int(sys.argv[2]) if len(sys.argv) > 2 else 729
out = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).resolve().parent / "build" / "c5_raw.mp4")

frames, poss, cols = mm.load(stem)
stride = poss.get("stride", 3)
dt = stride / 30.0
corr_path = config.PROJECT_ROOT / "data" / "pbp" / f"{stem}_orientation.json"
corrections = json.loads(corr_path.read_text()) if corr_path.exists() else {}

span = None
for s in poss["spans"]:
    if s["kind"] != "halfcourt" or not s.get("metrics_eligible"):
        continue
    corr = corrections.get(str(s["set_start_frame"]))
    if corr:   # exactly as mm.main(): corrected roles supersede the per-span vote
        s["offense_team"] = corr["offense_team"]
        s["defense_team"] = corr["defense_team"]
        s["confidence"] = max(s.get("confidence") or 0, corr["orientation_conf"])
    if s["set_start_frame"] == target:
        span = s
assert span is not None, f"span {target} not found/eligible"
assert span.get("offense_team") is not None
assert span.get("confidence", 0) >= config.C3_MIN_SPAN_CONF

rec, asg, invalid = mm.possession_metrics(frames, span, dt)
print("core", rec["core_start_frame"], rec["core_end_frame"], "offense", rec["offense_team"],
      "conf", rec["confidence"], "pairs/frame", rec["coverage_pairs_per_frame"],
      "used", rec["frames_used"], "excluded", rec["frames_excluded_team_gt5"])
p = Path(out)
if p.exists():
    p.unlink()
mm.render_review(frames, [rec], {rec["core_start_frame"]: asg},
                 {rec["core_start_frame"]: invalid}, cols, out, 30.0 / stride)
print("wrote", out)
