"""P8 step 0, Track A: the committed record of the search and the identity check (PLAN.md D128 to D138).

Reads the search parts (tools.p8_search) and the identity records (tools.p8_identity) in the P8 workspace and writes
reports/research/p8_search.json: per game its class and candidate ids, per video its identity verdict, and the yield
tables of docs/research/P8_step0.md. Video ids and verdicts only; no title, no path. Every confirmed video is run
through refuse_cv_test again before it is written.

    python3 -m tools.p8_step0_report [--out reports/research/p8_search.json]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

from tools.p8_common import atomic_write, p8_root

WHO = "tools.p8_step0_report"
DEV_ARENAS = ("GSW", "NYK", "OKC", "POR")
REPO = Path(__file__).resolve().parents[1]


def build() -> dict:
    from sportvu import manifest as M
    from tools.p8_common import dev_games
    from tools.p8_search import summary
    from tools.research_testset import refuse_cv_test
    s = summary()
    rows = s["rows"]
    old84 = {r["game"] for r in rows if r["in_old_84"]}
    known = {g["video_id"] for g in dev_games().values()}
    ident = {}
    for p in sorted((p8_root() / "identity").glob("*/identity.json")):
        r = json.loads(p.read_text())
        if r.get("status") == "rejected" and not any(m.get("frames") for m in r.get("moments", [])):
            r["status"] = "not_downloaded"                  # pass 1 recorded these as rejected (fixed in 8e0c1d6b)
        ident[(r["source"], r["video_id"])] = r
    for b in sorted((p8_root() / "identity").glob("batch*.json")):   # guarded videos have only a batch row
        for r in json.loads(b.read_text()).get("rows", []):
            if r.get("status") == "guarded":
                ident.setdefault((r["source"], r["video_id"]), r)
    videos = {}
    for r in rows:
        if r["class"] != "training":
            continue
        for c in r["candidates"]:
            v = videos.setdefault((c["source"], c["id"]), {"source": c["source"], "id": c["id"], "length": c["length"],
                                                          "height": c.get("height"), "found_for": []})
            v["found_for"].append(r["game"])
    for k, v in videos.items():
        rec = ident.get(k, {})
        v["status"] = rec.get("status", "not_checked")
        v["game"] = rec.get("game")
        v["known_dev_video"] = v["id"] in known
        if v["status"] == "confirmed":
            refuse_cv_test([v["id"], v["game"]], WHO)
    conf = [v for v in videos.values() if v["status"] == "confirmed"]
    games_conf = sorted({v["game"] for v in conf})
    new_games = sorted({v["game"] for v in conf if not v["known_dev_video"]} - {v["game"] for v in conf if v["known_dev_video"]})
    home = lambda g: g.split(".")[-1]
    tables = {
        "by_class": s["by_class"],
        "videos_by_source_and_status": {src: dict(collections.Counter(v["status"] for v in videos.values() if v["source"] == src))
                                        for src in sorted({v["source"] for v in videos.values()})},
        "videos_by_length_and_status": {ln: dict(collections.Counter(v["status"] for v in videos.values() if v["length"] == ln))
                                        for ln in sorted({str(v["length"]) for v in videos.values()})},
        "confirmed_games": games_conf,
        "confirmed_games_not_already_known": new_games,
        "confirmed_full_720p_games": sorted({v["game"] for v in conf if v["length"] == "full" and (v["height"] or 0) >= 720}),
        "confirmed_games_by_home_team": dict(collections.Counter(home(g) for g in games_conf)),
        "new_arenas_outside_development": sorted({home(g) for g in games_conf} - set(DEV_ARENAS)),
        "old_84": {"games": len(old84), "with_candidate": sum(1 for r in rows if r["game"] in old84 and r["candidates"]),
                   "confirmed": sorted(g for g in games_conf if g in old84), "earlier_usable": 6},
        "candidate_games_by_home_team": dict(collections.Counter(home(r["game"]) for r in rows
                                                                 if r["class"] == "training" and r["candidates"])),
    }
    return {"written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "rule": "PLAN.md D128 to D138; tools.p8_search, tools.p8_identity, tools.p8_step0_report",
            "calibration": {k: v for k, v in json.loads((p8_root() / "identity" / "calibration.json").read_text()).items()
                            if k in ("videos", "confirmed_true", "false_confirmations", "passes", "written")},
            "tables": tables,
            "games": [{k: r[k] for k in ("game", "class", "candidates", "metadata_hits_not_probed")} for r in rows],
            "videos": sorted(videos.values(), key=lambda v: (v["source"], v["id"]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("--out", default=str(REPO / "reports" / "research" / "p8_search.json"))
    a = ap.parse_args(argv)
    res = build()
    atomic_write(Path(a.out), json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["tables"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
