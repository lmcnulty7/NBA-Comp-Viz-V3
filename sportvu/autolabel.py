"""
sportvu/autolabel.py: SportVU auto-labels on every synced wide frame of the TRAINING games (ROADMAP R1.3).

Games are the data/harvest/games.json entries with split "train" (gate after R1.1); any other split
is refused, so nothing from the held-out game, arena or era can become a label. Per game:

  1. windows   every running span (>= MIN_SPAN_S) of the game's local time maps, cut and built with
               production settings except the gate: gate v2 at its v2 threshold (models/
               trained_head_v2, thresholds.json "v2"). Outputs under data/sportvu/label_build/.
  2. pairs     window OCR time map, residual clock offset and the B4 ICP matching (sportvu.truth.
               window_truth: feet <-> SportVU pairs, >= 6 inliers), under the game's mirror from R1.2.
               The B4 homography itself is NOT used: it fits the feet, not the court
               (reports/sportvu_truth_line_check.txt).
  3. camera    the truth H is a camera model (sportvu.camera, decided 2026-10-04): one camera position
               per game from a pooled fit over the game's frames, per frame rotation + zoom fitted to
               the pairs and refined onto the painted-line ridges.
  4. accept    a frame is accepted when the camera fit has >= MIN_LINE_MATCHES painted-line matches, A1
               line support >= MIN_SUPPORT (V3's typical support is 0.09..0.10), a feet residual median
               <= MAX_FEET_RESID_FT (feet vs SportVU noise is about 1.2..1.7 ft median on the training
               games) and a zoom off its bounds. Thresholds chosen on the training games
               (reports/sportvu_camera_<game>.json). Every other frame is counted under its rule
               (gate_v2_skipped, unmapped, no_moment, untestable_boxes, untestable_noH, untestable_match,
               untestable_inliers, camera_no_init, line_matches_low, support_low, feet_resid_high,
               zoom_at_bound).
  5. labels    data/sportvu/labels/<game>/<window>.jsonl, one line per accepted frame: the window video
               and frame, H_truth (px -> ft), (a) the 13x7 grid keypoints and the 1 ft line samples
               (court/snap_track CH_MID) projected through H_truth, kept when on the frame, (b) boxes
               whose truth-projected foot is within CONFIRM_FT of a SportVU player (Hungarian), with
               team id, player id and jersey, (c) the court region of the frame (court x under the
               frame's lower centre: left, middle or right third).
  6. sheet     data/sportvu/labels/<game>_sheet.jpg: SHEET_N accepted frames with the projected lines.
  7. images    with --save-images, every accepted frame as data/sportvu/labels/<game>/img/<window>_<frame>.jpg
               (the court model trains on these; broadcast stills, never committed).
Per-game results go to data/sportvu/labels/<game>_result.json so games can run in parallel processes;
--manifest-only merges them into the manifest.

Manifest: reports/sportvu_labels_manifest.{json,txt} (committed) with accepted frames per game, per
arena and per court region, and the rejection counts per rule per game. Labels stay gitignored.

  python -m sportvu.autolabel [--games gsw_bkn_2015,...] [--max-windows N] [--save-images]
  python -m sportvu.autolabel --manifest-only
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import config
from sportvu.rebuild import cut, build, plan_windows
from sportvu.sync import SYNC_DIR, SportVUIndex, MIRRORS, apply_mirror
from sportvu.truth import window_truth

REG = config.PROJECT_ROOT / "data" / "harvest" / "games.json"
OUT = config.PROJECT_ROOT / "data" / "sportvu" / "labels"
WIN_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "label_build"
MAX_FEET_RESID_FT = 2.0
MIN_LINE_MATCHES = 30
MIN_SUPPORT = 0.12
CONFIRM_FT = 2.5
SHEET_N = 12
EVERY_SPAN = 1e9          # plan_windows budget: every running span, not a sample


def training_games(only: str | None = None) -> dict:
    reg = json.loads(REG.read_text())
    games = {k: v for k, v in reg.items() if isinstance(v, dict) and v.get("sportvu") and not v.get("excluded")}
    if only:
        keep = set(only.split(","))
        for k in keep:
            if k in games and games[k].get("split") != "train":
                raise SystemExit("refusing %s: split %r is held out (FIX_LOOP hard rule)" % (k, games[k].get("split")))
        games = {k: v for k, v in games.items() if k in keep}
    return {k: v for k, v in games.items() if v.get("split") == "train"}


def gate_args() -> tuple:
    thr = json.loads(config.THRESHOLDS_PATH.read_text())["v2"]
    return ("--gate-head", str(config.HEAD_V2_PATH), "--gate-thr", str(thr))


def project(P, pts_ft):
    import cv2
    return cv2.perspectiveTransform(np.asarray(pts_ft, np.float32).reshape(-1, 1, 2), np.asarray(P, np.float64)).reshape(-1, 2)


def region_of(H_truth, w: int, h: int) -> str:
    from court.court33 import COURT_LENGTH_FT as L
    x = float(project(np.asarray(H_truth, np.float64).reshape(3, 3), [[w / 2.0, h * 0.75]])[0][0])
    return "left" if x < L / 3 else ("right" if x > 2 * L / 3 else "middle")


def frame_labels(row: dict, side_row: dict, sv_xy, sv_pid, sv_team, jersey: dict, mirror: str, w: int, h: int) -> dict:
    from court.grid import GRID_FT
    from court.snap_track import CH_MID
    from sportvu.truth import _match
    Ht = np.array(row["H_truth"], np.float64).reshape(3, 3)
    P = np.linalg.inv(Ht)
    def on(px):
        return (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h) & np.isfinite(px).all(1)
    g = project(P, GRID_FT); gk = on(g)
    ln = project(P, CH_MID); lk = on(ln)
    fx, fy = MIRRORS[mirror]
    sv = apply_mirror(np.asarray(sv_xy, np.float32), fx, fy)
    feet = np.array([b["foot_stab"] for b in side_row["boxes"]], np.float32).reshape(-1, 2)
    ct = project(Ht, feet) if len(feet) else np.zeros((0, 2), np.float32)
    boxes = []
    for bi, j, d in (_match(ct, sv, CONFIRM_FT) if len(ct) else []):
        b = side_row["boxes"][bi]
        boxes.append({"bbox": b["bbox"], "foot": [round(v, 1) for v in b["foot_stab"]], "pid": int(sv_pid[j]),
                      "team": int(sv_team[j]), "jersey": jersey.get(str(int(sv_pid[j]))), "d_ft": round(d, 2)})
    return {"grid": [[int(i), round(float(x), 1), round(float(y), 1)] for i, (x, y) in enumerate(g) if gk[i]],
            "lines": [[round(float(x), 1), round(float(y), 1)] for x, y in ln[lk]],
            "boxes": boxes, "region": region_of(Ht, w, h)}


def label_game(tag: str, g: dict, max_windows: int | None = None, save_images: bool = False) -> dict:
    import cv2
    from clock_reader import ClockReader
    from sportvu.local_sync import build_local_timemap
    if g.get("split") != "train":
        raise SystemExit("refusing %s: held out" % tag)
    mirror = json.loads((config.REPORTS_DIR / ("sportvu_direction_%s.json" % tag)).read_text())["resolution"]["mirror"]
    mom = json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (g["sportvu"] + "_moments.json")).read_text())
    index = SportVUIndex(mom["moments"])
    jersey = {k: v.get("jersey") for k, v in mom.get("players", {}).items()}
    secs = sorted(p.name.replace("_local_timemap.json", "") for p in SYNC_DIR.glob(tag + "_s*_local_timemap.json"))
    windows = [w for s in secs for w in plan_windows(s, budget_s=EVERY_SPAN)]
    if max_windows:
        windows = windows[:max_windows]
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / tag).mkdir(parents=True, exist_ok=True)
    reader = ClockReader(g["layout"])
    from sportvu import camera as cam
    from qc.track_qc import line_overlay
    from videoseq import SeqReader
    counts, regions, accepted, sheet_pick = {}, {}, 0, []
    t0 = time.time()
    built = []                         # phase A: build, OCR, ICP pairs per window
    for k, w in enumerate(windows):
        snip = cut(w, out_dir=WIN_DIR)
        r = build(w, snip, out_dir=WIN_DIR, extra_args=gate_args())
        side_p = WIN_DIR / (w["window"] + "_frames.json")
        tm_p = WIN_DIR / (w["window"] + "_timemap.json")
        if not tm_p.exists():
            tm_p.write_text(json.dumps(build_local_timemap(w["section"], reader, vid=snip)))
        if not side_p.exists():
            counts["build_failed"] = counts.get("build_failed", 0) + 1
            continue
        sc, tm = json.loads(side_p.read_text()), json.loads(tm_p.read_text())
        live = {int(f) for f, _, _ in tm["frames"]}
        counts["gate_v2_skipped"] = counts.get("gate_v2_skipped", 0) + len(live - {r_["frame"] for r_ in sc["frames"]})
        off, _, rows = window_truth(w, sc, tm, index, mirror)
        for row in rows:
            if row["status"] != "ok":
                counts[row["status"]] = counts.get(row["status"], 0) + 1
        cap = cv2.VideoCapture(str(snip)); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        frs = cam.frames_from_rows(w, sc, rows, index, mirror, W, H)
        counts["untestable_inliers"] = counts.get("untestable_inliers", 0) + sum(1 for r_ in rows if r_["status"] == "ok") - len(frs)
        built.append((w, snip, sc, off, frs))
        print("  A [%d/%d] %-26s %s candidates %d (%.0f s)" % (k + 1, len(windows), w["window"],
              "cached" if r.get("cached") else ("ok" if r.get("rc") == 0 else "FAILED"), len(frs), time.time() - t0), flush=True)
    allf = [f for _, _, _, _, frs in built for f in frs]
    g_fit = cam.fit_game(allf)               # phase B: one camera position per game
    if g_fit.get("status") != "ok":
        return {"game": tag, "status": "camera_fit_failed", "camera": g_fit, "rejections": counts}
    C = np.array(g_fit["C"])
    lo, hi = np.log(cam.F_BOUNDS_PX[0]) + 0.01, np.log(cam.F_BOUNDS_PX[1]) - 0.01
    for w, snip, sc, off, frs in built:      # phase C: per-frame camera truth, acceptance, labels
        srow = {r_["frame"]: r_ for r_ in sc["frames"]}
        sr = SeqReader(cv2.VideoCapture(str(snip)))
        lines = []
        for fr in frs:
            ok_img, img = sr.read(fr["frame"])
            fit = cam.refine_lines(C, fr, cam.fit_frame(C, fr), img) if ok_img else {"status": "no_image"}
            if fit.get("status") != "ok":
                rule = "camera_no_init"
            elif fit.get("lines", 0) < MIN_LINE_MATCHES:
                rule = "line_matches_low"
            elif not (lo < np.log(fit["f"]) < hi):
                rule = "zoom_at_bound"
            elif fit["resid_ft"] > MAX_FEET_RESID_FT:
                rule = "feet_resid_high"
            else:
                sup = line_overlay(img, fit["H"], fr["boxes"])["support"]
                rule = None if sup is not None and sup >= MIN_SUPPORT else "support_low"
            if rule:
                counts[rule] = counts.get(rule, 0) + 1
                continue
            row = {**fr["row"], "H_truth": fit["H"]}
            q, i = row["q"], row["moment"]
            lab = frame_labels(row, srow[row["frame"]], index.q[q]["xy"][i], index.q[q]["pid"][i], index.q[q]["team"][i],
                               jersey, mirror, fr["w"], fr["h"])
            if save_images:
                (OUT / tag / "img").mkdir(exist_ok=True)
                img_rel = OUT / tag / "img" / ("%s_%06d.jpg" % (w["window"], row["frame"]))
                cv2.imwrite(str(img_rel), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lab["image"] = str(img_rel.relative_to(config.PROJECT_ROOT))
            lines.append(json.dumps({"game": tag, "window": w["window"], "video": str(snip.relative_to(config.PROJECT_ROOT)),
                                     "frame": row["frame"], "section": w["section"], "section_frame": w["f_start"] + row["frame"],
                                     "q": q, "clock": row["clock"], "offset_s": off, "H_truth": fit["H"], "truth": "camera",
                                     "camera_C": g_fit["C"], "camera_f": fit["f"], "feet_resid_ft": fit["resid_ft"],
                                     "line_matches": fit.get("lines"), **lab}))
            counts["accepted"] = counts.get("accepted", 0) + 1
            regions[lab["region"]] = regions.get(lab["region"], 0) + 1
            sheet_pick.append((str(snip), row["frame"], fit["H"]))
        (OUT / tag / (w["window"] + ".jsonl")).write_text("\n".join(lines) + ("\n" if lines else ""))
        accepted += len(lines)
    print("  C %s: accepted %d (%.0f s)" % (tag, accepted, time.time() - t0), flush=True)
    contact_sheet(tag, sheet_pick)
    return {"game": tag, "sportvu": g["sportvu"], "home": g["home"], "mirror": mirror, "windows": len(windows),
            "camera": {k: v for k, v in g_fit.items() if k != "cost"},
            "accepted": accepted, "regions": regions, "rejections": {k: v for k, v in counts.items() if k != "accepted"}}


def contact_sheet(tag: str, picks: list) -> None:
    import cv2
    from court.snap_track import CH_A, CH_B
    if not picks:
        return
    sel = [picks[int(i)] for i in np.linspace(0, len(picks) - 1, min(SHEET_N, len(picks)))]
    tiles = []
    for vid, f, Hl in sel:
        cap = cv2.VideoCapture(vid); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read(); cap.release()
        if not ok:
            continue
        P = np.linalg.inv(np.array(Hl, np.float64).reshape(3, 3))
        a, b = project(P, CH_A), project(P, CH_B)
        for p0, p1 in zip(a, b):
            if np.isfinite(p0).all() and np.isfinite(p1).all() and (np.abs(p0) < 4000).all() and (np.abs(p1) < 4000).all():
                cv2.line(fr, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (0, 0, 255), 2)
        tiles.append(cv2.resize(fr, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(OUT / (tag + "_sheet.jpg")), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


def write_manifest(results: list) -> None:
    by_arena = {}
    for r in results:
        a = by_arena.setdefault(r["home"], {"games": 0, "accepted": 0, "regions": {}})
        a["games"] += 1; a["accepted"] += r["accepted"]
        for k, v in r["regions"].items():
            a["regions"][k] = a["regions"].get(k, 0) + v
    rep = {"item": "ROADMAP R1.3", "rules": {"truth": "camera model (sportvu/camera.py)", "min_icp_inliers": 6,
                                             "min_line_matches": MIN_LINE_MATCHES, "min_support": MIN_SUPPORT,
                                             "max_feet_resid_ft": MAX_FEET_RESID_FT, "confirm_ft": CONFIRM_FT,
                                             "gate": "v2", "games": "split == train only"},
           "by_arena": by_arena, "games": results,
           "caveats": ["candidate frames come from the B4 ICP matching, which starts from the V3 court H: frames where V3's H is too far off to match 6 players are lost (counted as untestable_*), so labels lean toward views V3 already handles",
                       "the camera truth is checked against the painted lines by A1 line support, an indicator, not a ground truth; feet vs SportVU noise is about 1.2..1.7 ft median",
                       "one camera position per game: an upload that switches broadcast feeds (gsw_cha_2016) mixes camera positions and loses more frames",
                       "labels are on the local h264 copies of the harvest uploads, not on the production sections"]}
    (config.REPORTS_DIR / "sportvu_labels_manifest.json").write_text(json.dumps(rep, indent=1))
    L = ["SPORTVU AUTO-LABELS (ROADMAP R1.3): %d training games" % len(results)]
    for a, v in by_arena.items():
        L.append("  arena %s: %d games, %d accepted frames, regions %s" % (a, v["games"], v["accepted"], v["regions"]))
    for r in results:
        L.append("  %-16s accepted %6d in %d windows | regions %s" % (r["game"], r["accepted"], r["windows"], r["regions"]))
        L.append("  %-16s   rejected: %s" % ("", ", ".join("%s %d" % kv for kv in sorted(r["rejections"].items(), key=lambda kv: -kv[1]))))
    L += ["", "caveats:"] + ["  - " + c for c in rep["caveats"]]
    txt = "\n".join(L)
    (config.REPORTS_DIR / "sportvu_labels_manifest.txt").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", default=None)
    ap.add_argument("--max-windows", type=int, default=None, help="smoke test: label only the first N windows per game")
    ap.add_argument("--save-images", action="store_true", help="write every accepted frame as a jpg (training input)")
    ap.add_argument("--manifest-only", action="store_true", help="merge the per-game results into the manifest")
    a = ap.parse_args()
    if a.manifest_only:
        write_manifest([json.loads(p.read_text()) for p in sorted(OUT.glob("*_result.json"))])
        return
    games = training_games(a.games)
    for tag, g in games.items():
        res = label_game(tag, g, a.max_windows, a.save_images)
        if a.max_windows:
            print(json.dumps(res, indent=1)); continue
        (OUT / (tag + "_result.json")).write_text(json.dumps(res, indent=1))
    if not a.max_windows and not a.games:
        write_manifest([json.loads(p.read_text()) for p in sorted(OUT.glob("*_result.json"))])


if __name__ == "__main__":
    main()
