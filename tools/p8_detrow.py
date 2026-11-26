"""tools/p8_detrow.py: causes of the detection row (docs/research/P8_detrow.md). Report only: no model runs, nothing is
built, trained or changed.

    python3 -m tools.p8_detrow measure   -> <P8 workspace>/placement/detrow.json and reports/research/p8_detrow.json

The detection row is Level 2's movable part of the missed row (P8_placement.md step 3): in-frame players with no
foot_stab within 5 ft under the truth camera ("far misses"), 0.052 to 0.091 of in-frame players on the pool and 0.063
on hou_cle_2016.

Inputs, stored P8-workspace files only (nothing under DATA_ROOT, no model, no video decode):
tools.p8_footbias.load_frames (the reference-platform r23 sidecars, the scorer's universe, truth rows, cache, time
maps, SportVU moments, mirror, and the count check against the registered score files); for the pool games, M4's raw
CPU predictions on the corpus JPEGs (e1/m4_preds__<game>.json: r23 player boxes down to 0.10) and the E1 corpus preds
(the YOLOX-X SportsMOT teacher down to 0.05). hou_cle_2016 has no corpus frames, so its raw and teacher classes are
not measured.

Rules, fixed before any number:
- Games: the pool gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015; hou_cle_2016 reported apart as a check (it chooses
  nothing). Guards on every game, SportVU id and window.
- Population: in-frame players (the scorer's 10 px margin through the truth camera) whose nearest foot_stab, sent
  through the truth camera as the scorer maps it, lies more than 5 ft away (sportvu.bench.MATCH_FT), or who have no
  box at all.
- Expected box of a SportVU player: the pinhole camera of the truth H (tools.p8_e1_postmortem.foot_and_height):
  bottom centre at the projected foot, height that of a 6.5 ft player, width 0.4 of the height. Frames without a
  camera pose give no expected box.
- Cause classes, mutually exclusive, the first that holds:
  1. edge: less than 90 percent of the expected box's area lies inside the image;
  2. occluded: the expected box's largest IoU with the expected box of another in-image player (foot projected inside
     the image) is at least 0.3;
  3. small: expected height under 150 px, or the player is in the far camera third (tools.p8_scorecard.third_of with
     sportvu.eval.near_side of the truth H);
  4. raw had a box: on a corpus frame, some M4 r23 raw box at the operating point (0.40) has its bottom-centre foot,
     through the truth camera, within 5 ft (the build lost it: tracking, gating or stabilisation);
  5. teacher only: on a corpus frame, some teacher box at 0.1471 has its foot within 5 ft, and no r23 raw box at 0.40
     does;
  6. none: on a corpus frame, none of the above;
  7. not measured: past classes 1 to 3, on a frame with no corpus raw boxes (every hou_cle_2016 frame), or a frame
     without a camera pose that no later class reaches.
- Check: per game, the far-miss count equals step 1's (reports/research/p8_placement.json "far"), or the tool stops.
- Per class, per game and pooled: the count, the share of far misses and the rate among in-frame players, with window
  intervals (2,000 resamples, seed 0, windows resampled whole within each game).
- Report only: each attribute's prevalence among far misses without the ordering (edge, occluded, small, raw had a box,
  teacher reaches, r23 raw at 0.10 to 0.40 within 5 ft); and the sub-split of classes 5 and 6 with an r23 raw box
  between 0.10 and 0.40 within 5 ft (a threshold-reachable box).
- Reading rule: a class matters when its rate among in-frame players is at least 0.010 on every pool game (the
  adoption bar for a missed gain: a class smaller than that cannot deliver the bar even if fixed completely).
  Class 5 (teacher only) mattering points to a detector change; class 4 (raw had a box) to tracking or gating;
  classes 1 to 3 to hard geometry no measured source reaches; class 6 mattering means a cause nothing measured
  reaches.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.p8_placement import CHECK, POOL, check_games, rnd

WHO = "tools.p8_detrow"
NEAR_FT = 5.0
TEACHER_CONF, TEACHER_FLOOR, DET_CONF, R23_FLOOR = 0.1471, 0.05, 0.40, 0.10
EDGE_INSIDE, OCC_IOU, SMALL_PX = 0.90, 0.30, 150.0
BOX_W_OF_H = 0.4
CLASSES = ("edge", "occluded", "small", "raw_had_box", "teacher_only", "none", "not_measured")
BAR = 0.010
BOOT_N, SEED = 2000, 0


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def expected_boxes(foot_px, heights) -> np.ndarray:
    f = np.asarray(foot_px, float).reshape(-1, 2)
    h = np.asarray(heights, float).reshape(-1)
    half = BOX_W_OF_H * h / 2.0
    return np.column_stack([f[:, 0] - half, f[:, 1] - h, f[:, 0] + half, f[:, 1]])


def inside_share(box, w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if area <= 0:
        return 0.0
    ix = max(min(x2, w) - max(x1, 0.0), 0.0) * max(min(y2, h) - max(y1, 0.0), 0.0)
    return ix / area


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def cause(edge: bool | None, occ: bool | None, small: bool | None, raw: bool | None, teacher: bool | None) -> str:
    """The first class that holds; None means the attribute is not measured on this frame."""
    if edge:
        return "edge"
    if occ:
        return "occluded"
    if small:
        return "small"
    if raw is None:
        return "not_measured"
    if raw:
        return "raw_had_box"
    if teacher:
        return "teacher_only"
    return "none"


def near_any(sv, feet_ft) -> bool:
    f = np.asarray(feet_ft, float).reshape(-1, 2)
    return bool(len(f)) and bool(np.hypot(*(f - np.asarray(sv, float)).T).min() <= NEAR_FT)


def matters(rates: dict) -> bool:
    return all(rates.get(g) is not None and rates[g] >= BAR for g in POOL)


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def game_rows(tag: str) -> tuple:
    from sportvu.eval import near_side
    from tools import p8_teachers as T
    from tools.p8_e1_postmortem import foot_and_height
    from tools.p8_e1data import corpus_root
    from tools.p8_footbias import load_frames, scorer_court
    from tools.p8_scorecard import third_of
    game = load_frames(tag)
    raw, teach = {}, {}
    if tag in POOL:
        raw = json.loads((p8_root() / "e1" / f"m4_preds__{tag}.json").read_text())
        if raw["game"] != tag or raw["device"] != "cpu" or raw["conf_floor"] > R23_FLOOR:
            raise SystemExit(f"{WHO}: m4_preds__{tag}.json is not the CPU run on {tag}")
        raw = raw["preds"]
        root = corpus_root()
        plan = json.loads((root / "plan.json").read_text())
        for k in range(len(plan["chunks"])):
            for key, rec in json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
                if key.startswith(tag + "/"):
                    teach[key.split("/", 1)[1]] = rec[T.MISS_TEACHER]
    rows, in_frame = [], {}
    W, H = 1280, 720
    for fr in game["frames"]:
        n_in = int(fr["inside"].sum())
        in_frame[fr["w"]] = in_frame.get(fr["w"], 0) + n_in
        if not n_in:
            continue
        feet = scorer_court(fr["Ht"], fr["px"])
        sv = fr["sv"]
        foot_px, hts = foot_and_height(sv, fr["Ht"], W, H)
        boxes = expected_boxes(foot_px, hts) if hts is not None else None
        in_img = (np.isfinite(foot_px).all(1) & (foot_px[:, 0] >= 0) & (foot_px[:, 0] < W) & (foot_px[:, 1] >= 0)
                  & (foot_px[:, 1] < H))
        nh = near_side(fr["Ht"])
        stem = f"{fr['w']}_f{int(fr['f']):06d}"
        rp, tp = raw.get(stem), teach.get(stem)
        r40 = T.feet_ft([b for b in rp["r23"] if b[4] >= DET_CONF], fr["Ht"]) if rp else None
        rlo = T.feet_ft([b for b in rp["r23"] if R23_FLOOR <= b[4] < DET_CONF], fr["Ht"]) if rp else None
        tf = T.feet_ft([b for b in tp if b[4] >= TEACHER_CONF], fr["Ht"]) if (rp and tp is not None) else None
        for j in np.where(fr["inside"])[0]:
            d = np.hypot(*(feet - sv[j]).T).min() if len(feet) else np.inf
            if d <= NEAR_FT:
                continue
            if boxes is not None:
                edge = inside_share(boxes[j], W, H) < EDGE_INSIDE
                others = [k for k in np.where(in_img)[0] if k != j]
                occ = max((iou(boxes[j], boxes[k]) for k in others), default=0.0) >= OCC_IOU
                small = bool(hts[j] < SMALL_PX) or third_of(sv[j][1], nh) == "far"
            else:
                edge = occ = None
                small = third_of(sv[j][1], nh) == "far"
            raw_box = None if r40 is None else near_any(sv[j], r40)
            teacher = None if tf is None else near_any(sv[j], tf)
            low = None if rlo is None else near_any(sv[j], rlo)
            rows.append({"w": fr["w"], "cls": cause(edge, occ, small, raw_box, teacher), "edge": edge, "occ": occ,
                         "small": small, "raw": raw_box, "teacher": teacher, "low": low})
    return rows, in_frame, game["registered"], game["windows"]


def summarise(rows: list, in_frame: dict) -> dict:
    n_in = sum(in_frame.values())
    n = len(rows)
    out = {"in_frame": n_in, "far_misses": n, "far_miss_rate": n / n_in if n_in else None, "classes": {}}
    for c in CLASSES:
        k = sum(r["cls"] == c for r in rows)
        out["classes"][c] = {"count": k, "share_of_far_misses": k / n if n else None,
                             "rate_in_frame": k / n_in if n_in else None}
    def prev(field, val=True):
        meas = [r for r in rows if r[field] is not None]
        return {"measured": len(meas), "count": sum(r[field] == val for r in meas),
                "share": (sum(r[field] == val for r in meas) / len(meas)) if meas else None}
    out["report_only_prevalence"] = {"edge": prev("edge"), "occluded": prev("occ"), "small": prev("small"),
                                     "raw_had_box": prev("raw"), "teacher_reaches": prev("teacher"),
                                     "r23_raw_0_10_to_0_40": prev("low")}
    out["report_only_threshold_reachable"] = {c: sum(1 for r in rows if r["cls"] == c and r["low"]) for c in
                                              ("teacher_only", "none")}
    return out


def boot(rows: list, in_frame: dict, n: int = BOOT_N, seed: int = SEED) -> dict:
    """Window bootstrap (windows resampled within each game) of every class's share and in-frame rate."""
    games = sorted({w.rsplit("_s", 1)[0] for w in in_frame})
    byg = {g: sorted(w for w in in_frame if w.startswith(g)) for g in games}
    cnt = {}
    for r in rows:
        cnt.setdefault(r["w"], {c: 0 for c in CLASSES})[r["cls"]] += 1
    rng = np.random.default_rng(seed)
    reps = {c: {"share": [], "rate": []} for c in CLASSES}
    for _ in range(n):
        pick = [w for g in games for w in (byg[g][i] for i in rng.integers(0, len(byg[g]), len(byg[g])))]
        nin = sum(in_frame[w] for w in pick)
        tot = {c: sum(cnt.get(w, {}).get(c, 0) for w in pick) for c in CLASSES}
        nf = sum(tot.values())
        for c in CLASSES:
            reps[c]["share"].append(tot[c] / nf if nf else 0.0)
            reps[c]["rate"].append(tot[c] / nin if nin else 0.0)
    return {c: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
            for c, d in reps.items()}


def cmd_measure(args) -> dict:
    check_games(POOL + CHECK)
    games, allrows, allin = {}, [], {}
    for t in POOL + CHECK:
        rows, in_frame, reg, nwin = game_rows(t)
        games[t] = {**summarise(rows, in_frame), "intervals": boot(rows, in_frame), "registered": reg, "windows": nwin}
        if t in POOL:
            allrows += rows
            allin.update(in_frame)
        print(t, json.dumps(rnd({c: games[t]["classes"][c] for c in CLASSES})))
    step1 = REPO / "reports" / "research" / "p8_placement.json"       # step 1's far-miss counts, the same rule
    if step1.exists():
        far1 = {g: json.loads(step1.read_text())["counts"][g]["far"] for g in POOL + CHECK}
        got = {g: games[g]["far_misses"] for g in POOL + CHECK}
        if far1 != got:
            raise SystemExit(f"{WHO}: far misses {got} differ from step 1's {far1}")
    pooled = {**summarise(allrows, allin), "intervals": boot(allrows, allin)}
    reading = {c: {"rates": {g: games[g]["classes"][c]["rate_in_frame"] for g in POOL},
                   "matters": matters({g: games[g]["classes"][c]["rate_in_frame"] for g in POOL})}
               for c in CLASSES}
    out = rnd({"what": "Causes of the detection row (far misses, no foot within 5 ft under the truth camera; "
                       "docs/research/P8_detrow.md); report only; stored files",
               "games": games, "pooled": pooled, "reading": reading,
               "rule": {"edge_inside": EDGE_INSIDE, "occluded_iou": OCC_IOU, "small_px": SMALL_PX, "near_ft": NEAR_FT,
                        "teacher_conf": TEACHER_CONF, "det_conf": DET_CONF, "bar": BAR,
                        "boot": {"n": BOOT_N, "seed": SEED, "unit": "window within game"}}})
    d = p8_root() / "placement"
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(d / "detrow.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_detrow.json", text)
    print(json.dumps({"pooled": {c: pooled["classes"][c] for c in CLASSES},
                      "matters": {c: v["matters"] for c, v in reading.items()}}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_detrow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()
