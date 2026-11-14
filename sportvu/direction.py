"""
sportvu/direction.py: court mirror and clock offset per game for the new SportVU games (ROADMAP R1.2).

New games have no production build, so (as in B4) the mapped running spans are rebuilt LOCALLY in
short windows (sportvu.rebuild: cut, build_trajectories with production settings), each window
gets its own OCR time map (an ffmpeg -ss cut can start a keyframe early), and the rebuilt raw
positions are compared with SportVU. For every window and every mirror (sportvu.sync.MIRRORS) the
residual clock offset within +-1.5 s is solved (sportvu.truth.solve_window_offset); the statistic
is the median nearest-SportVU-player distance, and a second one uses only positions more than
FAR_FROM_MID_FT from the mid-line, where a y-flip must show. A game is resolved when at least
VOTE_SHARE of its windows pick the same mirror on the far statistic; otherwise it is ambiguous.

  python -m sportvu.direction gsw_cle_xmas15 [--budget 40]
      -> data/sportvu/build/<section>_w<k>.* (gitignored), reports/sportvu_direction_<game>.json
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import config
from sportvu.rebuild import cut, build, plan_windows
from sportvu.sync import SYNC_DIR, MIRRORS, FAR_FROM_MID_FT, SportVUIndex, apply_mirror
from sportvu.truth import frame_clock, solve_window_offset, CLOCK_TOL_S

BUDGET_S = 40.0         # seconds of running clock rebuilt per section (direction needs far less than truth)
# Its own cache: the 40 s windows share names with sportvu.rebuild's 100 s windows (same section, same index),
# so building them into data/sportvu/build made a later rebuild reuse the shorter clips (found at R2.1).
DIR_BUILD = config.PROJECT_ROOT / "data" / "sportvu" / "build_direction"
MIN_ROWS = 20
VOTE_SHARE = 0.8


def window_rows(window: dict, tm: dict, traj: dict) -> list:
    """[(period, clock, positions (n,2))] for the window's frames with >= 3 raw positions."""
    by = {}
    for tr in traj.values():
        for f, x, y in tr.get("raw", []):
            by.setdefault(int(f), []).append((x, y))
    rows = []
    for f, pts in sorted(by.items()):
        if len(pts) < 3:
            continue
        q, clock = frame_clock(window, tm, f)
        if q is not None:
            rows.append((q, clock, np.array(pts, np.float32)))
    return rows


def far_median(rows: list, index: SportVUIndex, mirror: str, offset: float):
    fx, fy = MIRRORS[mirror]
    d = []
    for q, c, p in rows:
        i = index.at(q, c + offset, CLOCK_TOL_S)
        if i is None:
            continue
        pm = apply_mirror(p, fx, fy)
        pm = pm[np.abs(pm[:, 1] - 25.0) > FAR_FROM_MID_FT]
        if len(pm):
            d.extend(np.sqrt(((pm[:, None, :] - index.q[q]["xy"][i][None]) ** 2).sum(-1)).min(1).tolist())
    return round(float(np.median(d)), 2) if d else None


def solve(rows_by_window: dict, index: SportVUIndex) -> dict:
    per = {}
    for name, rows in rows_by_window.items():
        if len(rows) < MIN_ROWS:
            continue
        cand = {}
        for m in MIRRORS:
            off, med = solve_window_offset(rows, index, m)
            cand[m] = {"offset_s": off, "median_ft": med, "far_ft": far_median(rows, index, m, off)}
        vote = min((m for m in cand if cand[m]["far_ft"] is not None), key=lambda m: cand[m]["far_ft"], default=None)
        per[name] = {"rows": len(rows), "vote_far": vote, "candidates": cand}
    votes = [w["vote_far"] for w in per.values() if w["vote_far"]]
    if not votes:
        return {"status": "too_few_frames", "windows": per}
    mirror = max(set(votes), key=votes.count)
    share = votes.count(mirror) / len(votes)
    med_by_m = {m: round(float(np.median([w["candidates"][m]["median_ft"] for w in per.values()])), 2) for m in MIRRORS}
    far_by_m = {m: round(float(np.median([w["candidates"][m]["far_ft"] for w in per.values() if w["candidates"][m]["far_ft"] is not None])), 2) for m in MIRRORS}
    offs = [w["candidates"][mirror]["offset_s"] for w in per.values()]
    return {"status": "ok" if share >= VOTE_SHARE else "ambiguous",
            "resolution": {"mirror": mirror, "vote_share": round(share, 2), "windows_voting": len(votes),
                           "median_ft_by_mirror": med_by_m, "far_ft_by_mirror": far_by_m,
                           "offsets_s": [round(min(offs), 2), round(max(offs), 2)]},
            "windows": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game")
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    a = ap.parse_args()
    reg = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())[a.game]
    from clock_reader import ClockReader
    from sportvu.local_sync import build_local_timemap
    secs = sorted(p.name.replace("_local_timemap.json", "") for p in SYNC_DIR.glob(a.game + "_s*_local_timemap.json"))
    windows = [w for s in secs for w in plan_windows(s, budget_s=a.budget)]
    DIR_BUILD.mkdir(parents=True, exist_ok=True)
    (DIR_BUILD / (a.game + "_windows.json")).write_text(json.dumps(windows, indent=1))
    print("%s: %d windows, %.0f s of video" % (a.game, len(windows), sum((w["f_end"] - w["f_start"]) / w["fps"] for w in windows)), flush=True)
    reader = ClockReader(reg["layout"])
    t0 = time.time()
    rows_by_window = {}
    for k, w in enumerate(windows):
        snip = cut(w, out_dir=DIR_BUILD)
        r = build(w, snip, out_dir=DIR_BUILD)
        tmp = DIR_BUILD / (w["window"] + "_timemap.json")
        if not tmp.exists():
            tmp.write_text(json.dumps(build_local_timemap(w["section"], reader, vid=snip)))
        trp = DIR_BUILD / (w["window"] + "_trajectories.json")
        if trp.exists():
            rows_by_window[w["window"]] = window_rows(w, json.loads(tmp.read_text()), json.loads(trp.read_text()))
        print("  [%d/%d] %-24s %s rows %d (%.0f s)" % (k + 1, len(windows), w["window"], "cached" if r.get("cached") else ("ok" if r.get("rc") == 0 else "FAILED"),
                                                     len(rows_by_window.get(w["window"], [])), time.time() - t0), flush=True)
    index = SportVUIndex(json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (reg["sportvu"] + "_moments.json")).read_text())["moments"])
    rep = {"game": a.game, "sportvu_game": reg["sportvu"], "budget_s": a.budget, **solve(rows_by_window, index)}
    (config.REPORTS_DIR / ("sportvu_direction_%s.json" % a.game)).write_text(json.dumps(rep, indent=1, default=str))
    res = rep.get("resolution") or {}
    print("%s: %s mirror %s (vote %s of %s windows), offsets %s s, median ft %s, far ft %s" % (
        a.game, rep["status"], res.get("mirror"), res.get("vote_share"), res.get("windows_voting"), res.get("offsets_s"),
        res.get("median_ft_by_mirror"), res.get("far_ft_by_mirror")), flush=True)


if __name__ == "__main__":
    main()
