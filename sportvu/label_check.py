"""sportvu/label_check.py: paint check and acceptance rule for the R1.3 auto-labels.

The labels' court lines come from the camera-model truth H (sportvu/camera.py). The contact sheets of run
r13_20261004_1846 show most frames on the paint and some 20-40 px off (gsw_bkn_2015_s00_w00 frame 120
passed every R1.3 rule with 34 line matches); the A1 line support cannot tell those apart. This check
measures each label against the one court marking a GSW broadcast shows in saturated colour: the gold key.

Per label: build a gold-paint mask (HSV), take the gradient magnitude of the blurred mask as an edge field,
project the key's lane lines and free-throw line (both keys; the baseline edge is left out because the GSW
baseline apron is gold too) through H_truth, and find the edge peak along each 1 ft sample's normal within
EDGE_RADIUS_PX (court/snap_track.match_lines). The label's paint error is the median distance from the
projected samples to their peaks.

Rules (rejections counted per rule per game):
  paint_edge_high       measured and the median edge distance > EDGE_MAX_PX
  paint_edge_unmatched  a key is mostly in frame but fewer than MIN_EDGE_MATCHES samples find an edge
                        (the label is more than EDGE_RADIUS_PX off, or the key is hidden)
Labels with no key in frame are kept and counted as key_out_of_frame (unverified).
EDGE_MAX_PX is chosen on the training games (all GSW), never on a held-out set.

Outputs: data/sportvu/labels/<game>_paint.jsonl (one row per label, with keep) and
reports/sportvu_label_check.{json,txt}. Training games only, like autolabel.
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np
import config

LABELS = config.PROJECT_ROOT / "data" / "sportvu" / "labels"
EDGE_RADIUS_PX = 30       # the gold edge field has no clutter, so a wide search is safe
MIN_EDGE_MATCHES = 8
EDGE_MAX_PX = 12.0        # gsw_bkn_2015 sample of 400: p50 7.1, p75 10.5, p90 14.3; frames seen off by eye are above it
MIN_KEY_PX = 1500         # projected key area in the frame (full-res px) that counts as "in frame"
MIN_IN_SHARE = 0.6        # ... with at least this share of the key inside the frame
# OpenCV HSV (H 0-180): GSW key paint is H 20-24, S 150-255; the maple is H 17-19, S 70-100.
GOLD_H, GOLD_S, GOLD_V = (19, 30), 150, 100
LANE17 = [((0, 17), (19, 17)), ((94, 17), (75, 17))]
LANE33 = [((19, 33), (0, 33)), ((75, 33), (94, 33))]
FT_EDGES = [((19, 17), (19, 33)), ((75, 17), (75, 33))]
KEY_EDGES = LANE17 + LANE33 + FT_EDGES


def chords(segs, step_ft: float = 1.0):
    """(A, B, MID) 1 ft chords of court segments, the template format of court/snap_track.match_lines."""
    a, b = [], []
    for p1, p2 in segs:
        p1, p2 = np.asarray(p1, np.float64), np.asarray(p2, np.float64)
        ts = np.linspace(0, 1, max(1, int(np.linalg.norm(p2 - p1) / step_ft)) + 1)
        a += [p1 + t0 * (p2 - p1) for t0 in ts[:-1]]
        b += [p1 + t1 * (p2 - p1) for t1 in ts[1:]]
    A, B = np.array(a, np.float32), np.array(b, np.float32)
    return A, B, ((A + B) / 2.0).astype(np.float32)


PAINT_CHORDS = chords(KEY_EDGES)


def gold_mask(img_bgr):
    import cv2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= GOLD_H[0]) & (h <= GOLD_H[1]) & (s >= GOLD_S) & (v >= GOLD_V)).astype(np.float32)


def paint_edge_field(img_bgr):
    """Gradient magnitude of the gold-paint mask, on the ridge field's scale (peaks ~50-100)."""
    import cv2
    m = cv2.GaussianBlur(gold_mask(img_bgr) * 255.0, (0, 0), 1.5)
    gx, gy = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    return (np.hypot(gx, gy) / 10.0).astype(np.float32)


def edge_offsets(Hc, field, w: int, h: int, ch=PAINT_CHORDS, radius: int = EDGE_RADIUS_PX) -> np.ndarray:
    """Distances (px) from each projected template sample to its edge peak; Hc maps court ft -> image."""
    import cv2
    from court.snap_track import match_lines
    mid, peaks = match_lines(Hc, field, w, h, radius, chords=ch)
    if not len(peaks):
        return np.zeros(0)
    p = cv2.perspectiveTransform(mid.reshape(-1, 1, 2).astype(np.float64), Hc).reshape(-1, 2)
    return np.linalg.norm(p - peaks, axis=1)


def key_in_frame(Hc, w: int, h: int) -> bool:
    import cv2
    for x0, x1 in ((0, 19), (75, 94)):
        key = cv2.perspectiveTransform(np.array([[[x0, 17]], [[x1, 17]], [[x1, 33]], [[x0, 33]]], np.float64), Hc).reshape(-1, 2)
        if not np.isfinite(key).all() or np.abs(key).max() > 20000:
            continue
        area = abs(cv2.contourArea(key.astype(np.float32)))
        clip = cv2.intersectConvexConvex(key.astype(np.float32), np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32))[0]
        if clip >= MIN_KEY_PX and area > 0 and clip / area >= MIN_IN_SHARE:
            return True
    return False


def key_edges(img_bgr, H_truth) -> dict:
    """Paint error of one label: median and p90 edge distance (px) over the matched key-edge samples."""
    h, w = img_bgr.shape[:2]
    Hc = np.linalg.inv(np.asarray(H_truth, np.float64).reshape(3, 3))
    d = edge_offsets(Hc, paint_edge_field(img_bgr), w, h)
    if len(d) >= MIN_EDGE_MATCHES:
        return {"status": "measured", "n": int(len(d)), "edge_px": round(float(np.median(d)), 2),
                "edge_px_p90": round(float(np.percentile(d, 90)), 2)}
    return {"status": "unmatched" if key_in_frame(Hc, w, h) else "no_key", "n": int(len(d))}


def verdict(r: dict) -> str | None:
    """Rejection rule a paint-check row trips, or None to keep."""
    if r["status"] == "measured":
        return "paint_edge_high" if r["edge_px"] > EDGE_MAX_PX else None
    if r["status"] == "unmatched":
        return "paint_edge_unmatched"
    return None if r["status"] == "no_key" else "no_image"


def _check_row(row: dict) -> dict:
    import cv2
    out = {"window": row["window"], "frame": row["frame"], "region": row.get("region")}
    img = cv2.imread(str(config.PROJECT_ROOT / row["image"])) if row.get("image") else None
    out.update({"status": "no_image"} if img is None else key_edges(img, row["H_truth"]))
    out["reject"] = verdict(out)
    out["keep"] = out["reject"] is None
    return out


def check_game(game: str, workers: int = 8) -> list:
    from multiprocessing import Pool
    rows = [json.loads(l) for f in sorted(glob.glob(str(LABELS / game / "*.jsonl"))) for l in open(f)]
    with Pool(workers) as pool:
        res = pool.map(_check_row, rows, chunksize=32)
    (LABELS / (game + "_paint.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in res))
    return res


def summarise(game: str, res: list) -> dict:
    meas = np.array([r["edge_px"] for r in res if r["status"] == "measured"])
    st, rej, kept_reg = {}, {}, {}
    for r in res:
        st[r["status"]] = st.get(r["status"], 0) + 1
        if r["reject"]:
            rej[r["reject"]] = rej.get(r["reject"], 0) + 1
        else:
            kept_reg[r["region"]] = kept_reg.get(r["region"], 0) + 1
    q = lambda v: round(float(np.percentile(meas, v)), 2) if len(meas) else None
    return {"game": game, "labels": len(res), "status": st, "edge_px": {"p50": q(50), "p75": q(75), "p90": q(90)},
            "share_le_px": {str(t): round(float((meas <= t).mean()), 3) if len(meas) else None for t in (4, 6, 8, 10, 12, 15)},
            "rejected": rej, "kept": sum(kept_reg.values()), "kept_unverified": st.get("no_key", 0), "kept_regions": kept_reg}


def write_report(summaries: list) -> None:
    rep = {"item": "ROADMAP R1.3 verification: label vs painted key", "rules": {
        "edge_radius_px": EDGE_RADIUS_PX, "min_edge_matches": MIN_EDGE_MATCHES, "edge_max_px": EDGE_MAX_PX,
        "key_in_frame": {"min_key_px": MIN_KEY_PX, "min_in_share": MIN_IN_SHARE}, "gold_hsv": [GOLD_H, GOLD_S, GOLD_V]},
        "games": summaries, "total_kept": sum(s["kept"] for s in summaries), "total_labels": sum(s["labels"] for s in summaries)}
    (config.REPORTS_DIR / "sportvu_label_check.json").write_text(json.dumps(rep, indent=1))
    L = ["R1.3 LABEL PAINT CHECK: distance from each label's projected key edges (lane lines, free-throw line)",
         "to the gold paint edge, px at 1280x720. Rule: reject when the median > %.0f px, or when the key is in" % EDGE_MAX_PX,
         "frame and its edges are not found within %d px. Labels with no key in frame are kept, unverified." % EDGE_RADIUS_PX]
    for s in summaries:
        L.append("  %-14s labels %5d -> kept %5d (%2.0f%%, %d unverified) | rejected %s" % (
            s["game"], s["labels"], s["kept"], 100.0 * s["kept"] / max(s["labels"], 1), s["kept_unverified"], s["rejected"]))
        L.append("                 edge px p50 %s p75 %s p90 %s | share <= px %s" % (
            s["edge_px"]["p50"], s["edge_px"]["p75"], s["edge_px"]["p90"], s["share_le_px"]))
        L.append("                 kept regions %s" % s["kept_regions"])
    L += ["  total kept %d of %d" % (rep["total_kept"], rep["total_labels"]), "", "caveats:",
          "  - the key is the only colour-painted marking checked: a label can sit on the key and be off elsewhere",
          "    (pinhole camera, no lens distortion); the far end of the court is not checked",
          "  - the gold edge is not exactly the line centre (line width, blur): about 4-5 px of the median is",
          "    measurement floor (only ~10% of bkn labels are within 4 px)",
          "  - players and gold uniforms (IND) occlude or add gold edges; the median over samples limits that",
          "  - unverified labels (no key in frame, mostly mid-court) keep the camera-truth error of the rest"]
    (config.REPORTS_DIR / "sportvu_label_check.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    games = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())
    train = [t for t, g in games.items() if g.get("split") == "train" and (LABELS / t).is_dir()]
    sel = a.games or train
    bad = [t for t in sel if t not in train]
    if bad:
        raise SystemExit("refusing %s: not a labelled training game" % bad)
    write_report([summarise(t, check_game(t, a.workers)) for t in sel])


if __name__ == "__main__":
    main()
