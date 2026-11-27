"""tools/p8_sac_timemaps.py: the window time maps of the P8 workspace and the reference root, compared by bytes and by
content (docs/research/P8_sac_checkc.md section 9; descriptive, post hoc: the registered Clock reading has fired).
Stored files only, system python, no lock.

    python3 -m tools.p8_sac_timemaps     -> reports/research/p8_sac_timemaps.json (the only file it writes)

Per game (gsw_sac_2015, and as controls gsw_cha_2016, gsw_bkn_2015, hou_cle_2016) and per window with a time map in
both roots: SHA-256 equal or not; the top-level keys whose values differ; whether the frame sets and the spans are
equal; the largest and median absolute clock difference and the count of period differences over the frames both
hold; each file's modification time and whether it is a symbolic link. Provenance: the window time map is written by
the frozen label step (sportvu.autolabel, build_local_timemap of sportvu.local_sync) when the file is absent, from a
fresh clock read of the window clip; its "ocr_seconds" is that read's wall-clock duration.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_sac_timemaps"
GAMES = ("gsw_sac_2015", "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016")
DATA = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"


def out_path() -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / "p8_sac_timemaps.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_timemaps(a: dict, b: dict) -> dict:
    """Content comparison of two window time maps (pure)."""
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    fa = {int(f): (q, c) for f, q, c in a.get("frames", [])}
    fb = {int(f): (q, c) for f, q, c in b.get("frames", [])}
    com = sorted(fa.keys() & fb.keys())
    dc = [abs(fa[f][1] - fb[f][1]) for f in com]
    return {"keys_differing": keys, "frames_equal_sets": fa.keys() == fb.keys(), "frames_common": len(com),
            "spans_equal": a.get("spans") == b.get("spans"),
            "clock_abs_diff_max": float(max(dc)) if dc else None,
            "clock_abs_diff_median": float(np.median(dc)) if dc else None,
            "period_differences": sum(1 for f in com if fa[f][0] != fb[f][0])}


def mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.lstat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(ws: Path = DATA / "work", ref: Path = DATA / "work_ref") -> dict:
    from tools.p8_common import atomic_write, guard
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(list(GAMES), WHO)
    guard(list(GAMES), WHO)
    out = {"what": "Window time maps, workspace against reference root, by bytes and by content "
                   "(docs/research/P8_sac_checkc.md section 9); descriptive, post hoc", "games": {}}
    for g in GAMES:
        wa = {p.name for p in (ws / "label_build").glob(f"{g}_s*_timemap.json")}
        wb = {p.name for p in (ref / "label_build").glob(f"{g}_s*_timemap.json")}
        rows, keyset = {}, {}
        for name in sorted(wa & wb):
            pa, pb = ws / "label_build" / name, ref / "label_build" / name
            with open(pa, "r") as fh:
                a = json.load(fh)
            with open(pb, "r") as fh:
                b = json.load(fh)
            c = compare_timemaps(a, b)
            c.update({"sha_equal": sha256(pa) == sha256(pb), "mtime_workspace": mtime(pa), "mtime_reference": mtime(pb),
                      "symlink_workspace": pa.is_symlink(), "symlink_reference": pb.is_symlink()})
            rows[name[:-len("_timemap.json")]] = c
            for k in c["keys_differing"]:
                keyset[k] = keyset.get(k, 0) + 1
        sha_diff = [w for w, c in rows.items() if not c["sha_equal"]]
        maxes = [c["clock_abs_diff_max"] for c in rows.values() if c["clock_abs_diff_max"] is not None]
        mts = lambda k: sorted({c[k][:10] for c in rows.values()})
        out["games"][g] = {
            "windows_both": len(rows), "only_workspace": len(wa - wb), "only_reference": len(wb - wa),
            "sha_differing": len(sha_diff), "keys_differing_counts": keyset,
            "frames_equal_sets": sum(c["frames_equal_sets"] for c in rows.values()),
            "spans_equal": sum(c["spans_equal"] for c in rows.values()),
            "clock_abs_diff_max_over_windows": max(maxes) if maxes else None,
            "windows_with_clock_difference": sum(1 for x in maxes if x > 0),
            "period_differences": sum(c["period_differences"] for c in rows.values()),
            "content_differs_beyond_ocr_seconds": [w for w, c in rows.items()
                                                    if set(c["keys_differing"]) - {"ocr_seconds"}],
            "mtime_dates_workspace": mts("mtime_workspace"), "mtime_dates_reference": mts("mtime_reference"),
            "symlinks": {"workspace": sum(c["symlink_workspace"] for c in rows.values()),
                         "reference": sum(c["symlink_reference"] for c in rows.values())},
            "windows": rows}
        s = out["games"][g]
        print(g, json.dumps({k: s[k] for k in ("windows_both", "sha_differing", "keys_differing_counts",
                                               "frames_equal_sets", "spans_equal", "clock_abs_diff_max_over_windows",
                                               "period_differences", "content_differs_beyond_ocr_seconds",
                                               "mtime_dates_workspace", "mtime_dates_reference", "symlinks")}))
    atomic_write(out_path(), json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
