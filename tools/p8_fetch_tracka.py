"""tools/p8_fetch_tracka.py: download whole the Track A broadcasts confirmed under D138 that PLAN.md D146 names
(P8 step 0, Track A). Download and ffprobe only: no build, no camera fit (Phase A belongs to p8-teachers).

Each game and video id passes tools.research_testset.refuse_cv_test and tools.p8_common.guard before any request.
The video goes through tools.p8_fetch.download (harvest_driver.YTDLP_FMT, 720p avc1, and the proof-of-origin token
arguments) to <P8 workspace>/video/<SportVU game>.mp4, beside the development videos. The check is
tools.p8_nba2526_fetch's: h264, 720 rows, and a duration within 2 s of the metadata duration read just before.
A file that fails the check is not kept. Records: <P8 workspace>/meta/d146_fetch.json, and the same facts without
any path in reports/research/p8_d146_fetch.json.

    python3 -m tools.p8_fetch_tracka
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time

from tools import p8_identity as I
from tools.p8_common import REPO, atomic_write, guard, p8_root
from tools.p8_fetch import download
from tools.p8_nba2526_fetch import check_facts, ffprobe_line

WHO = "tools.p8_fetch_tracka"
GAMES = {"10.31.2015.GSW.at.NOP": "SXT3wqb_zho", "01.15.2016.CLE.at.HOU": "XDdGwBq0KB8"}   # PLAN.md D146


def metadata_duration(vid: str, tries: int = 3) -> float:
    for k in range(tries):
        try:
            return I.duration_of("youtube", vid)
        except (IndexError, ValueError):
            if k == tries - 1:
                raise
            time.sleep(30)


def main() -> int:
    from tools.research_testset import refuse_cv_test
    items = [x for g, v in GAMES.items() for x in (g, v)]
    refuse_cv_test(items, WHO)
    guard(items, WHO)
    meta = p8_root() / "meta" / "d146_fetch.json"
    rec = json.loads(meta.read_text()) if meta.exists() else {}
    for game, vid in GAMES.items():
        if rec.get(game, {}).get("verified"):
            continue
        dur = metadata_duration(vid)
        out = p8_root() / "video" / ("%s.mp4" % game)
        p = download(vid, out)
        ok = check_facts(p, dur)
        if not ok and out.exists():
            out.unlink()
        rec[game] = {"video_id": vid, "verified": ok, "metadata_duration_s": dur, "codec": p.get("codec"),
                     "width": p.get("width"), "height": p.get("height"), "fps": p.get("fps"),
                     "duration_s": p.get("duration_s"), "bytes": out.stat().st_size if out.exists() else None,
                     "ffprobe": ffprobe_line(out) if out.exists() else None,
                     "error": (p.get("error") or "").strip()[-300:] or None, "date": dt.date.today().isoformat(),
                     "rule": "PLAN.md D146; identity confirmed under D138"}
        atomic_write(meta, json.dumps(rec, indent=1))
        print("%s %s verified=%s %s" % (game, vid, ok, rec[game]["ffprobe"] or rec[game]["error"]), flush=True)
    atomic_write(REPO / "reports" / "research" / "p8_d146_fetch.json", json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
