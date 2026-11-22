"""tools/p8_nba2526_fetch.py: download the NBA 2025-26 full games that the identity check confirmed (P8 Level 4;
PLAN.md D142, D143; docs/research/P8_level4.md section 13).

Whole videos at 720p avc1 through tools.p8_fetch.download (harvest_driver.YTDLP_FMT and the proof-of-origin token
arguments), one at a time, into <P8 workspace>/../level4/nba2526/video/ outside the repository. Each file is checked
with ffprobe: h264, 720 rows, and a duration within DURATION_SLACK_S of the metadata probe's duration. A video that
fails the check is recorded and its file is not kept. The record is level4/nba2526/fetch.json. Network only: no
model, no frames read, so no heavy-job lock.

    python3 -m tools.p8_nba2526_fetch
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys

from tools import p8_nba2526_search as S
from tools.p8_common import atomic_write, guard
from tools.p8_fetch import download

WHO = "tools.p8_nba2526_fetch"
DURATION_SLACK_S = 2.0


def confirmed() -> list[dict]:
    """The confirmed candidates of tools.p8_nba_identity (D142), each with its metadata-probe duration."""
    res = json.loads((S.out_dir() / "nba_identity_candidates.json").read_text())
    probes = S.load("probes.json", {})
    out = []
    for vid, r in sorted(res.items()):
        if r["status"] == "confirmed" and r["game"] == r["title_game"]:
            out.append({"video_id": vid, "game_id": r["game"],
                        "probe_duration_s": (probes.get("youtube:%s" % vid) or {}).get("duration")})
    return out


def ffprobe_line(path) -> str:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,r_frame_rate:format=duration", "-of", "compact=p=0", str(path)],
                       capture_output=True, text=True)
    return " | ".join(l for l in r.stdout.strip().splitlines() if l)


def check_facts(p: dict, probe_duration_s) -> bool:
    return bool(p.get("ok")) and probe_duration_s is not None and p.get("duration_s") is not None \
        and abs(p["duration_s"] - probe_duration_s) <= DURATION_SLACK_S


def main() -> int:
    from tools.research_testset import refuse_cv_test
    vdir = S.out_dir() / "video"
    rec_path = S.out_dir() / "fetch.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    for c in confirmed():
        vid = c["video_id"]
        refuse_cv_test([vid, c["game_id"]], WHO)
        guard([vid], WHO)
        if rec.get(vid, {}).get("verified"):
            continue
        out = vdir / ("%s_%s.mp4" % (c["game_id"], vid))
        p = download(vid, out)
        ok = check_facts(p, c["probe_duration_s"])
        if not ok and out.exists():
            out.unlink()                                   # a file that fails the check is not kept
        rec[vid] = {"game_id": c["game_id"], "file": out.name, "verified": ok, "probe_duration_s": c["probe_duration_s"],
                    "codec": p.get("codec"), "width": p.get("width"), "height": p.get("height"), "fps": p.get("fps"),
                    "duration_s": p.get("duration_s"), "bytes": out.stat().st_size if out.exists() else None,
                    "ffprobe": ffprobe_line(out) if out.exists() else None, "error": p.get("error"),
                    "date": dt.date.today().isoformat()}
        atomic_write(rec_path, json.dumps(rec, indent=1))
        print("%s %s verified=%s %s" % (c["game_id"], vid, ok, rec[vid]["ffprobe"] or rec[vid]["error"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
