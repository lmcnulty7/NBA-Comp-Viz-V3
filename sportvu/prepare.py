"""
sportvu/prepare.py: fetch, split and sync the Stage 1 SportVU games (ROADMAP R1.2).

Games are the data/harvest/games.json entries with a "sportvu" field (chosen at the gate after
R1.1). Steps, each resumable and safe to rerun:

  fetch      video via harvest_driver.ensure_decodable (avc1 pinned, decode-verified), split into
             ~10-min sections (harvest_driver.split_sections), SportVU moments via sportvu.fetch
             (data/sportvu/<game>_moments.json, gitignored) with the coverage report.
  sync       OCR time map per section (sportvu.local_sync, the game's scorebug layout must be
             registered first) -> mapped running seconds per game.
  report     reports/sportvu_prepare.{json,txt}: per game the fetch, split, layout, mapped running
             seconds and (once solved) the mirror / offset resolution.

  python -m sportvu.prepare fetch [--games gsw_cle_xmas15,...]
  python -m sportvu.prepare sync  [--games ...]
  python -m sportvu.prepare report
"""
from __future__ import annotations
import argparse, json, time
import config

REG = config.PROJECT_ROOT / "data" / "harvest" / "games.json"


def sportvu_games(only: str | None = None) -> dict:
    reg = json.loads(REG.read_text())
    games = {k: v for k, v in reg.items() if isinstance(v, dict) and v.get("sportvu")}
    if only:
        keep = set(only.split(","))
        games = {k: v for k, v in games.items() if k in keep}
    return games


def do_fetch(games: dict) -> None:
    from harvest_driver import ensure_decodable, split_sections
    from sportvu import fetch
    for tag, g in games.items():
        t = time.time()
        ensure_decodable(tag)
        secs = split_sections(tag)
        mom = fetch.SV_DIR / (g["sportvu"] + "_moments.json")
        if not mom.exists():
            parsed = fetch.parse(fetch.download(g["sportvu"]))
            mom.write_text(json.dumps(parsed))
            rep = fetch.coverage_report(parsed)
            (config.REPORTS_DIR / ("sportvu_fetch_%s.json" % g["sportvu"])).write_text(json.dumps(rep, indent=1))
        print("  %-16s video ok, %d sections, SportVU moments ok (%.0f s)" % (tag, len(secs), time.time() - t), flush=True)


def do_sync(games: dict) -> None:
    from clock_reader import ClockReader
    from harvest_driver import HARVEST_VIDEO
    from sportvu.local_sync import build_local_timemap
    from sportvu.sync import SYNC_DIR
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    for tag, g in games.items():
        if not g.get("layout"):
            print("  %-16s skipped: no scorebug layout registered yet" % tag, flush=True)
            continue
        reader = ClockReader(g["layout"])
        for vid in sorted(HARVEST_VIDEO.glob(tag + "_s*.mp4")):
            out = SYNC_DIR / (vid.stem + "_local_timemap.json")
            if out.exists():
                continue
            tm = build_local_timemap(vid.stem, reader, vid=vid)
            out.write_text(json.dumps(tm))
            print("  %-20s reads %d (ok %d) spans %d mapped %.1f s (%.0f s)" % (
                vid.stem, tm["reads"], tm["reads_ok"], tm["running_spans"], tm["mapped_video_s"], tm["ocr_seconds"]), flush=True)


def do_report(games: dict) -> None:
    from harvest_driver import HARVEST_VIDEO
    from sportvu.sync import SYNC_DIR
    rows = {}
    for tag, g in games.items():
        secs = sorted(HARVEST_VIDEO.glob(tag + "_s*.mp4"))
        tms = [json.loads(p.read_text()) for p in sorted(SYNC_DIR.glob(tag + "_s*_local_timemap.json"))]
        cov = config.REPORTS_DIR / ("sportvu_fetch_%s.json" % g["sportvu"])
        dirp = config.REPORTS_DIR / ("sportvu_direction_%s.json" % tag)
        rows[tag] = {"sportvu": g["sportvu"], "split": g.get("split"), "video_id": g["video_id"],
                     "video": (HARVEST_VIDEO / (tag + ".mp4")).exists(), "sections": len(secs), "layout": g.get("layout"),
                     "sportvu_moments": json.loads(cov.read_text())["moments"] if cov.exists() else None,
                     "sections_mapped": len(tms), "mapped_running_s": round(sum(t["mapped_video_s"] for t in tms), 1),
                     "clock_reads": sum(t["reads"] for t in tms), "clock_reads_ok": sum(t["reads_ok"] for t in tms),
                     "direction": json.loads(dirp.read_text()).get("resolution") if dirp.exists() else None}
    rep = {"item": "ROADMAP R1.2", "games": rows}
    (config.REPORTS_DIR / "sportvu_prepare.json").write_text(json.dumps(rep, indent=1))
    L = ["SPORTVU GAME PREPARATION (ROADMAP R1.2): %d games" % len(rows)]
    for tag, r in rows.items():
        d = r["direction"] or {}
        L.append("  %-16s %-14s video %s sections %2d layout %-18s mapped %6.1f s in %d sections | mirror %s offset %s" % (
            tag, r["split"], "ok" if r["video"] else "--", r["sections"], r["layout"] or "-", r["mapped_running_s"],
            r["sections_mapped"], d.get("mirror", "-"), d.get("offsets_s", "-")))
    txt = "\n".join(L)
    (config.REPORTS_DIR / "sportvu_prepare.txt").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("fetch", "sync", "report"))
    ap.add_argument("--games", default=None)
    a = ap.parse_args()
    games = sportvu_games(a.games)
    {"fetch": do_fetch, "sync": do_sync, "report": do_report}[a.step](games)


if __name__ == "__main__":
    main()
