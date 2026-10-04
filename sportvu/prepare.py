"""
sportvu/prepare.py: fetch, split and sync the Stage 1 SportVU games (ROADMAP R1.2).

Games are the data/harvest/games.json entries with a "sportvu" field (chosen at the gate after
R1.1). Steps, each resumable and safe to rerun:

  fetch      video via harvest_driver.ensure_decodable (avc1 pinned, decode-verified), split into
             ~10-min sections (harvest_driver.split_sections), SportVU moments via sportvu.fetch
             (data/sportvu/<game>_moments.json, gitignored) with the coverage report.
  layout     pick the scorebug layout: sample LAYOUT_SAMPLES frames across the game and try every
             registered clock_reader layout (plausible read: period 1..4, clock <= 720 s). A box that
             merely overlaps the bug can read plausible digits, so the top CONSISTENCY_TOP layouts are
             then judged on consistency: each sample is paired with the frame 1 s later, and a pair
             counts when both reads agree on the period and the clock moved by 0 s (stopped) or
             1 +- 0.5 s (running). The most consistent layout wins if its share reaches MIN_LAYOUT_SHARE
             and is written to games.json; otherwise the game needs a new calibration.
  sync       OCR time map per section (sportvu.local_sync, the game's scorebug layout must be
             registered first) -> mapped running seconds per game.
  report     reports/sportvu_prepare.{json,txt}: per game the fetch, split, layout, mapped running
             seconds and (once solved) the mirror / offset resolution.

  python -m sportvu.prepare fetch [--games gsw_cle_xmas15,...]
  python -m sportvu.prepare layout [--games ...]
  python -m sportvu.prepare sync  [--games ...]
  python -m sportvu.prepare report
"""
from __future__ import annotations
import argparse, json, time
import config

REG = config.PROJECT_ROOT / "data" / "harvest" / "games.json"
LAYOUT_SAMPLES = 40
MIN_LAYOUT_SHARE = 0.25            # live play with a visible bug; replays and graphics read nothing
CONSISTENCY_TOP = 3


def sportvu_games(only: str | None = None) -> dict:
    reg = json.loads(REG.read_text())
    games = {k: v for k, v in reg.items() if isinstance(v, dict) and v.get("sportvu") and not v.get("excluded")}
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


def sample_frames(tag: str, n: int = LAYOUT_SAMPLES) -> list:
    """n (frame, frame 1 s later) pairs spread over the game's sections (first and last section
    skipped: pregame, postgame)."""
    import cv2
    from harvest_driver import HARVEST_VIDEO
    secs = sorted(HARVEST_VIDEO.glob(tag + "_s*.mp4"))
    secs = secs[1:-1] or secs
    per = max(1, n // len(secs))
    out = []
    for vid in secs:
        cap = cv2.VideoCapture(str(vid)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for k in range(per):
            f = int(total * (k + 0.5) / per)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, f + int(round(fps)))
            ok2, fr2 = cap.read()
            if ok and ok2:
                out.append((fr, fr2))
        cap.release()
    return out


def do_layout(games: dict) -> None:
    from clock_reader import ClockReader, LAYOUTS
    readers, reg = {}, json.loads(REG.read_text())
    for tag, g in games.items():
        pairs = sample_frames(tag)
        plaus = lambda p, c: p is not None and 1 <= p <= 4 and c is not None and 0 <= c <= 720
        shares = {}
        for lay in LAYOUTS:
            readers.setdefault(lay, ClockReader(lay))
            shares[lay] = round(sum(plaus(*readers[lay].read(a)[:2]) for a, _ in pairs) / max(len(pairs), 1), 3)
        top = sorted((l for l in shares if shares[l] >= MIN_LAYOUT_SHARE), key=lambda l: -shares[l])[:CONSISTENCY_TOP]
        consist = {}
        for lay in top:
            good = 0
            for a, b in pairs:
                p1, c1, _ = readers[lay].read(a)
                p2, c2, _ = readers[lay].read(b)
                if plaus(p1, c1) and plaus(p2, c2) and p1 == p2:
                    d = c1 - c2
                    good += abs(d) < 0.25 or 0.5 <= d <= 1.5
            consist[lay] = round(good / max(len(pairs), 1), 3)
        best = max(consist, key=consist.get) if consist else max(shares, key=shares.get)
        verdict = best if consist.get(best, 0) >= MIN_LAYOUT_SHARE else None
        if verdict:
            reg[tag]["layout"] = verdict
        else:
            reg[tag].pop("layout", None)
        reg[tag]["layout_check"] = {"pairs": len(pairs), "plausible_share": {l: shares[l] for l in top},
                                    "consistent_share": consist, "chosen": verdict}
        print("  %-16s %s | consistent %s | plausible %s" % (tag, verdict or "NEEDS CALIBRATION", consist,
                                                         {l: shares[l] for l in top}), flush=True)
    REG.write_text(json.dumps(reg, indent=1) + "\n")


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
                     "home": g["home"],
                     "direction": ({"status": json.loads(dirp.read_text())["status"], **(json.loads(dirp.read_text()).get("resolution") or {})}
                                   if dirp.exists() else None)}
    # a game ambiguous on its own takes its arena's mirror when every other game there resolved ok to
    # the same mirror and its own leading vote agrees (the mirror depends on the court model's labelling
    # and on SportVU's axes in that arena from the main camera side, both fixed per arena; B3 used the
    # same cross-check across sections). R1.3's per-frame truth fit re-checks it.
    for tag, r in rows.items():
        d = r["direction"]
        if not d or d["status"] != "ambiguous":
            continue
        peers = [o["direction"] for t, o in rows.items() if t != tag and o["home"] == r["home"] and o["direction"]]
        ok = [p for p in peers if p["status"] == "ok"]
        if ok and len({p["mirror"] for p in ok}) == 1 and ok[0]["mirror"] == d["mirror"]:
            d["status"] = "ok_by_arena"
            d["rule"] = "arena consistency: %s at the same arena resolved %s" % (
                ", ".join(t for t, o in rows.items() if t != tag and o["home"] == r["home"] and o["direction"] and o["direction"]["status"] == "ok"), d["mirror"])
    reg = json.loads(REG.read_text())
    excluded = {k: {"sportvu": v["sportvu"], "video_id": v["video_id"], "reason": v["excluded"]}
                for k, v in reg.items() if isinstance(v, dict) and v.get("sportvu") and v.get("excluded")}
    rep = {"item": "ROADMAP R1.2", "games": rows, "excluded": excluded,
           "caveats": ["mirror and offset come from short local rebuilds (sportvu.direction, 40 s of running clock per section), not full-game builds",
                       "the clock OCR resolves 1 s; per-window residual offsets within +-1.5 s are expected and are re-solved per window at R1.3",
                       "mapped running seconds are spans where consecutive 1 Hz clock reads agree with the elapsed video time within 1 s",
                       "ok_by_arena: ambiguous on its own windows (the y-flip is the weak axis, as in B3), resolved by the other game at the same arena; R1.3's per-frame truth fit re-checks it",
                       "windows inside SportVU clock gaps cannot vote (cle_gsw_2016 has 830 s of gaps)"]}
    (config.REPORTS_DIR / "sportvu_prepare.json").write_text(json.dumps(rep, indent=1))
    L = ["SPORTVU GAME PREPARATION (ROADMAP R1.2): %d games" % len(rows)]
    for tag, r in rows.items():
        d = r["direction"] or {}
        L.append("  %-16s %-14s video %s sections %2d layout %-18s mapped %6.1f s in %d sections | mirror %s (%s, vote %s of %s windows) offset %s" % (
            tag, r["split"], "ok" if r["video"] else "--", r["sections"], r["layout"] or "-", r["mapped_running_s"],
            r["sections_mapped"], d.get("mirror", "-"), d.get("status", "-"), d.get("vote_share", "-"), d.get("windows_voting", "-"), d.get("offsets_s", "-")))
        if d.get("rule"):
            L.append("  %-16s   resolved by %s" % ("", d["rule"]))
    for k, v in excluded.items():
        L.append("  %-16s EXCLUDED  %s" % (k, v["reason"]))
    L += ["", "caveats:"] + ["  - " + c for c in rep["caveats"]]
    txt = "\n".join(L)
    (config.REPORTS_DIR / "sportvu_prepare.txt").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("fetch", "layout", "sync", "report"))
    ap.add_argument("--games", default=None)
    a = ap.parse_args()
    games = sportvu_games(a.games)
    {"fetch": do_fetch, "layout": do_layout, "sync": do_sync, "report": do_report}[a.step](games)


if __name__ == "__main__":
    main()
