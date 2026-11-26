"""tools/p8_e1b_bound.py: the E1b upper bound, r23's raw misses broken down by cause on the E1 corpus frames
(docs/research/P8_E1B.md, section 2). A measurement only: no model runs, nothing trains, no window is built.

    python3 -m tools.p8_e1b_bound bound      -> <P8 workspace>/e1b/bound.json and reports/research/p8_e1b_bound.json
    python3 -m tools.p8_e1b_bound transfer   -> <P8 workspace>/e1b/transfer.json and
                                                reports/research/p8_e1b_transfer.json (the gate of P8_E1B.md section 2b)

Inputs, all in the P8 workspace and none under DATA_ROOT: the E1 corpus (plan.json, preds/chunk_*.json: our detector
at 0.40 and the YOLOX-X SportsMOT teacher down to its 0.05 floor), M4's raw CPU predictions on the corpus JPEGs
(e1/m4_preds__<game>.json: r23 and the fold's E1 seed 0 down to 0.10) and the SportVU moments (sportvu/). Games: the
3 pool games only (development training games of R1.3); tools.p8_common.guard and refuse_cv_test run first.

Rules, fixed before any number (the scorer's, as M3 of the E1 post-mortem, tools.p8_e1_postmortem):
- In-frame player: sportvu.eval's rule, the truth projection strictly inside the image with a 10 px margin.
- A box set reaches a player when its nearest box foot (bottom centre through the camera truth) lies within 2.5 ft
  (MISS_FT). r23 misses a player when its corpus boxes (player class, 0.40) do not reach it: M3's "ours" row exactly
  (the tool stops if the per-game counts differ from e1/postmortem_m3.json).
- Each r23 miss gets one reach class, the first that holds:
  1. "teacher" : the teacher at or above 0.1471 reaches it (a teacher-box label at the qualified threshold could teach
                 it);
  2. "teacher_low": the teacher between 0.05 and 0.1471 reaches it (only a truth-confirmed label could teach it);
  3. "r23_low" : r23 between 0.10 and 0.40 reaches it (M4 boxes; an operating-point gain, not a label);
  4. "placement": some box of the teacher (0.05 floor) or of r23 (0.10 floor) has its foot 2.5 to 5 ft away
                  (sportvu.bench.MATCH_FT): a box is near but its foot is off;
  5. "none" : no box of either model within 5 ft.
- Causes of every in-frame player: camera third (tools.p8_scorecard.third_of with sportvu.eval.near_side); expected
  height of a 6.5 ft player (the pinhole camera of sportvu.camera.pose_from_h, tools.p8_e1_postmortem.foot_and_height),
  bins under 150, 150 to 200, 200 to 250, 250 px and over, "no pose"; image edge (the projected foot within 40 px of
  a border); overlap: the largest IoU of the player's expected box (width 0.4 of its expected height, bottom centre
  at the projected foot) with the expected box of any other of the 10 SportVU players, bins "none" (under 0.1),
  "partial" (0.1 to 0.4), "heavy" (0.4 and over), "no pose".
- Ceilings, as missed rates on the same frames (lower is better): r23 alone; r23 together with the teacher at 0.1471
  (a player is missed only when neither reaches it); adding the teacher's low band; adding r23's low band. The first
  union is the ceiling of a recipe that keeps every r23 find and learns every teacher-box label; it is an upper
  bound, not a forecast.
- What teacher-box labels teach on r23's finds: each player r23 reaches is "kept" (the teacher at 0.1471 reaches it),
  "displaced" (a teacher box at 0.1471 within 2.5 to 5 ft), or "dropped" (no teacher box at 0.1471 within 5 ft; it
  would be taught as background unless masked).
- Report only, E1's raw gain on the left-out game (M4, 0.40): r23 misses that E1 reaches, split by whether r23 had a
  box within 5 ft ("placement") or none ("presence"), and r23 finds that E1 loses.
- Interval: 2,000 window bootstrap resamples, seed 0, corpus windows resampled whole, for the pooled ceilings and the
  reach shares.
- Report only, added after the first run: r23's misses by reach class among those where r23 has a box (0.10 floor)
  within 5 ft, which splits the teacher-reachable misses into placement (r23 boxes the player, its foot is off) and
  presence (r23 has no box near the player).

Transfer-adjusted ceiling (`transfer`, registered 2026-11-26 after the design review and before any of its numbers;
docs/research/P8_E1B.md section 2b). It asks how much of L1's raw headroom a perfect imitator of the labels would keep
at build level, if its raw gains and losses carried into the build as E1's did on the same game.
- Frames: per pool game, the corpus frames that both reference-platform builds processed: r23 (builds/e1m6_r23_detcpu
  for gsw_bkn_2015, builds/m9_r23_detcpu__<game> otherwise) and E1 seed 0 of the fold named after the game
  (builds/e1m6_e1_gsw_bkn_2015_detcpu, builds/e1r_e1_<game>_detcpu). Players: in-frame with the 10 px margin.
- Raw reach (bottom-centre foot within 2.5 ft, nearest): r23 and E1 at 0.40 from M4's CPU run on the corpus JPEGs
  (one run for both, so the transfer fractions compare like with like); the teacher at 0.1471 from the corpus preds.
  Build reach: the nearest foot_stab of the build's sidecar boxes within 2.5 ft (the scorer's foot).
- A raw gain is "presence" when r23 (M4, 0.10 floor) has no box foot within 5 ft of the player, else "placement".
- Transfer fractions, per game, from E1: f_pres = of E1's presence raw gains (r23 raw misses, E1 raw reaches), the
  share where at build level E1 reaches and r23 misses; f_plc the same for placement gains; f_loss = of E1's raw
  losses (r23 raw reaches, E1 raw misses), the share where at build level E1 misses and r23 reaches.
- L1's net, per game: P = r23 raw misses the teacher reaches, presence; Q = the same, placement; L = r23 raw finds the
  teacher does not reach (L1's displaced and dropped players).
- Perfect-imitation build ceiling, per game: C = -(P f_pres + Q f_plc - L f_loss) / N, N the in-frame players.
  Pooled: the three numerators over the three N, each game with its own fractions. Negative means fewer misses.
- Gate, fixed now, on the point estimates: C pooled at most -0.020 AND C at most -0.010 on every game. If any game is
  above -0.010, E1b with the L1 label set stops before any unit is spent.
- Report only: the optimistic student-free variant C_opt = -(P f_pres + Q f_plc) / N (a student that keeps every r23
  find and adds every teacher find, L = 0); half of C (the realistic student of section 2b); E1's raw and build
  differences on the same frames beside the transfer model's prediction for E1 (a check of the model, not a gate);
  window bootstrap intervals (2,000, seed 0, corpus windows resampled whole, fractions recomputed in each resample)
  for C and C_opt per game and pooled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1b_bound"
POOL = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
SEED = 0
BOOT_N = 2000
MISS_FT = 2.5            # sportvu.eval.MISS_FT (tests assert they agree)
NEAR_FT = 5.0            # sportvu.bench.MATCH_FT
TEACHER_CONF = 0.1471    # tools.p8_teachers.MISS_TEACHER_CONF
TEACHER_FLOOR = 0.05     # the corpus teacher run's floor
R23_FLOOR = 0.10         # M4's floor
DET_CONF = 0.40          # config.PLAYER_CONF
BOX_W_OF_H = 0.4
OVERLAP_EDGES = (0.1, 0.4)
REACH = ("teacher", "teacher_low", "r23_low", "placement", "none")
THIRDS = ("near", "middle", "far")
HEIGHTS = ("<150", "150-200", "200-250", ">=250", "no pose")
OVERLAPS = ("none", "partial", "heavy", "no pose")
EDGES = ("True", "False")
L1_ON_FINDS = ("kept", "displaced", "dropped")
CAUSES = (("third", THIRDS), ("height", HEIGHTS), ("edge", EDGES), ("overlap", OVERLAPS))


def check_games(games) -> list:
    games = list(games)
    bad = [g for g in games if g not in POOL]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not a pool game")
    refuse_cv_test(games, WHO)
    guard(games, WHO)
    return games


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def nearest(sv_xy, feet) -> np.ndarray:
    """Per SportVU player, the distance (ft) to the nearest box foot; inf when there is no box."""
    sv = np.asarray(sv_xy, np.float64).reshape(-1, 2)
    ft = np.asarray(feet, np.float64).reshape(-1, 2)
    if len(ft) == 0:
        return np.full(len(sv), np.inf)
    return np.sqrt(((sv[:, None, :] - ft[None, :, :]) ** 2).sum(-1)).min(1)


def reach_class(d_t: float, d_tlo: float, d_rlo: float, d_any: float) -> str:
    """The reach class of one r23 miss (the first that holds; distances in ft)."""
    if d_t <= MISS_FT:
        return "teacher"
    if d_tlo <= MISS_FT:
        return "teacher_low"
    if d_rlo <= MISS_FT:
        return "r23_low"
    if d_any <= NEAR_FT:
        return "placement"
    return "none"


def l1_on_find(d_t: float) -> str:
    if d_t <= MISS_FT:
        return "kept"
    return "displaced" if d_t <= NEAR_FT else "dropped"


def height_bin(hpx) -> str:
    if hpx is None or not np.isfinite(hpx):
        return "no pose"
    if hpx < 150:
        return "<150"
    if hpx < 200:
        return "150-200"
    return "200-250" if hpx < 250 else ">=250"


def expected_boxes(foot_px, heights) -> np.ndarray:
    """Expected boxes [x1, y1, x2, y2] of players standing at foot_px with the given pixel heights."""
    f = np.asarray(foot_px, np.float64).reshape(-1, 2)
    hh = np.asarray(heights, np.float64).reshape(-1)
    half = BOX_W_OF_H * hh / 2.0
    return np.column_stack([f[:, 0] - half, f[:, 1] - hh, f[:, 0] + half, f[:, 1]])


def max_other_iou(boxes) -> np.ndarray:
    """Per box, the largest IoU with any other box of the set."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    n = len(b)
    if n < 2:
        return np.zeros(n)
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
    np.fill_diagonal(iou, 0.0)
    return iou.max(1)


def overlap_bin(v) -> str:
    if v is None or not np.isfinite(v):
        return "no pose"
    if v < OVERLAP_EDGES[0]:
        return "none"
    return "partial" if v < OVERLAP_EDGES[1] else "heavy"


def frame_rows(sv_xy, inside, H_truth, w: int, h: int, feet: dict) -> list:
    """One frame -> a row per in-frame player. feet: {source: box feet (ft)} for ours (0.40), teacher (>= 0.1471),
    teacher_low (0.05 to 0.1471), r23_low (0.10 to 0.40), r23_all (>= 0.10), teacher_all (>= 0.05) and e1 (0.40,
    or None when absent)."""
    from sportvu.eval import near_side
    from tools.p8_e1_postmortem import foot_and_height, is_edge
    from tools.p8_scorecard import third_of
    sv = np.asarray(sv_xy, np.float64).reshape(-1, 2)
    inside = np.asarray(inside, bool)
    foot, hts = foot_and_height(sv, H_truth, w, h)
    ov = max_other_iou(expected_boxes(foot, hts)) if hts is not None else None
    nh = near_side(np.asarray(H_truth, np.float64).reshape(3, 3))
    d = {k: nearest(sv, v) for k, v in feet.items() if v is not None}
    rows = []
    for j in np.where(inside)[0]:
        found = bool(d["ours"][j] <= MISS_FT)
        d_any = min(d["teacher_all"][j], d["r23_all"][j])
        row = {"j": int(j), "missed": not found, "third": third_of(sv[j][1], nh),
               "height": height_bin(None if hts is None else float(hts[j])),
               "edge": str(is_edge(float(foot[j][0]), float(foot[j][1]), w, h)),
               "overlap": overlap_bin(None if ov is None else float(ov[j])),
               "teacher_reaches": bool(d["teacher"][j] <= MISS_FT),
               "teacher_low_reaches": bool(d["teacher_low"][j] <= MISS_FT),
               "r23_low_reaches": bool(d["r23_low"][j] <= MISS_FT)}
        if found:
            row["l1"] = l1_on_find(float(d["teacher"][j]))
        else:
            row["reach"] = reach_class(float(d["teacher"][j]), float(d["teacher_low"][j]), float(d["r23_low"][j]),
                                       float(d_any))
            row["r23_box_within_5ft"] = bool(d["r23_all"][j] <= NEAR_FT)
        if "e1" in d:
            row["e1_found"] = bool(d["e1"][j] <= MISS_FT)
        rows.append(row)
    return rows


def ceilings(rows: list) -> dict:
    """Missed rates of r23 and of r23 joined with each label source, on the rows given."""
    n = len(rows)
    if not n:
        return {}
    m0 = sum(r["missed"] for r in rows)
    m1 = sum(r["missed"] and not r["teacher_reaches"] for r in rows)
    m2 = sum(r["missed"] and not (r["teacher_reaches"] or r["teacher_low_reaches"]) for r in rows)
    m3 = sum(r["missed"] and not (r["teacher_reaches"] or r["teacher_low_reaches"] or r["r23_low_reaches"])
             for r in rows)
    return {"in_frame": n, "r23": m0 / n, "r23_or_teacher": m1 / n, "plus_teacher_low": m2 / n,
            "plus_r23_low": m3 / n}


def tally(rows: list, field: str, levels) -> dict:
    out = {str(k): 0 for k in levels}
    for r in rows:
        if field in r:
            out[str(r[field])] = out.get(str(r[field]), 0) + 1
    return out


def summary(rows: list) -> dict:
    miss = [r for r in rows if r["missed"]]
    finds = [r for r in rows if not r["missed"]]
    out = {"in_frame": len(rows), "r23_missed": len(miss),
           "r23_missed_rate": round(len(miss) / len(rows), 5) if rows else None,
           "misses_by_reach": tally(miss, "reach", REACH),
           "reach_share_of_misses": {k: round(v / len(miss), 4) if miss else None
                                     for k, v in tally(miss, "reach", REACH).items()},
           "placement_or_none_with_an_r23_box_within_5ft": sum(1 for r in miss if r["r23_box_within_5ft"]
                                                               and r["reach"] in ("placement", "none")),
           # report only, added after the first run: does r23 already have a box within 5 ft (its foot off), by class
           "report_only_misses_by_reach_with_an_r23_box_within_5ft":
               tally([r for r in miss if r["r23_box_within_5ft"]], "reach", REACH),
           "ceilings": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in ceilings(rows).items()},
           "r23_finds_under_teacher_labels": tally(finds, "l1", L1_ON_FINDS)}
    for field, levels in CAUSES:
        out[f"by_{field}"] = {
            str(k): {"in_frame": sum(1 for r in rows if r[field] == str(k)),
                     "r23_missed": sum(1 for r in miss if r[field] == str(k)),
                     "misses_by_reach": tally([r for r in miss if r[field] == str(k)], "reach", REACH),
                     "r23_finds_dropped_by_teacher_labels": sum(1 for r in finds if r[field] == str(k)
                                                                and r["l1"] == "dropped")}
            for k in levels}
    e1 = [r for r in rows if "e1_found" in r]
    if e1:
        out["report_only_e1_raw"] = {
            "rows": len(e1),
            "r23_miss_e1_finds": sum(1 for r in e1 if r["missed"] and r["e1_found"]),
            "of_which_placement": sum(1 for r in e1 if r["missed"] and r["e1_found"] and r["r23_box_within_5ft"]),
            "of_which_presence": sum(1 for r in e1 if r["missed"] and r["e1_found"] and not r["r23_box_within_5ft"]),
            "r23_find_e1_loses": sum(1 for r in e1 if not r["missed"] and not r["e1_found"])}
    return out


def boot(groups: dict, n: int = BOOT_N, seed: int = SEED) -> dict:
    """groups: {window: rows} -> window bootstrap intervals of the pooled ceilings and the reach shares."""
    keys = sorted(groups)
    cols = ("in_frame", "m0", "m1", "m2", "m3") + REACH
    arr = []
    for k in keys:
        rows = groups[k]
        c = ceilings(rows)
        nn = c["in_frame"]
        miss = [r for r in rows if r["missed"]]
        arr.append([nn, c["r23"] * nn, c["r23_or_teacher"] * nn, c["plus_teacher_low"] * nn,
                    c["plus_r23_low"] * nn] + [sum(1 for r in miss if r["reach"] == q) for q in REACH])
    a = np.array(arr, float)
    ix = {c: i for i, c in enumerate(cols)}

    def stats(x):
        s = x.sum(0)
        out = {"r23": s[ix["m0"]] / s[ix["in_frame"]], "r23_or_teacher": s[ix["m1"]] / s[ix["in_frame"]],
               "plus_teacher_low": s[ix["m2"]] / s[ix["in_frame"]], "plus_r23_low": s[ix["m3"]] / s[ix["in_frame"]]}
        out["teacher_gain"] = out["r23_or_teacher"] - out["r23"]
        for q in REACH:
            out[f"share_{q}"] = s[ix[q]] / max(s[ix["m0"]], 1e-9)
        return out

    point = stats(a)
    rng = np.random.default_rng(seed)
    reps = [stats(a[rng.integers(0, len(a), len(a))]) for _ in range(n)]
    return {k: {"point": round(float(v), 5),
                "window_interval": [round(float(np.percentile([r[k] for r in reps], 2.5)), 5),
                                    round(float(np.percentile([r[k] for r in reps], 97.5)), 5)]}
            for k, v in point.items()} | {"windows": len(keys)}


# ── run ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_bound(args) -> dict:
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_teachers as T
    from tools.p8_e1_postmortem import scorer_inside
    from tools.p8_e1data import corpus_root
    check_games(POOL)
    root, wr = corpus_root(), p8_root()
    plan = json.loads((root / "plan.json").read_text())
    if sorted(plan["games"]) != sorted(POOL):
        raise SystemExit(f"{WHO}: the corpus plan names {sorted(plan['games'])}, not the pool games")
    guard([v for g in plan["games"].values() for v in (g["sportvu"], g["video_id"])], WHO)
    preds = {}
    for k in range(len(plan["chunks"])):
        preds.update(json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"])
    m3 = json.loads((wr / "e1" / "postmortem_m3.json").read_text())
    games, groups = {}, {}
    for tag, g in plan["games"].items():
        m4 = json.loads((wr / "e1" / f"m4_preds__{tag}.json").read_text())
        if m4["game"] != tag or m4["fold"] != tag or m4["device"] != "cpu" or m4["conf_floor"] > R23_FLOOR:
            raise SystemExit(f"{WHO}: m4_preds__{tag}.json is not the fold's CPU run on its left-out game")
        index = SportVUIndex(json.loads((wr / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        rows_g, teacher_missed = [], 0
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != tag:
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    rec = preds[f"{tag}/{stem}"]
                    wpx, hpx = rec["size"]
                    H = fr["H_truth"]
                    sv = apply_mirror(index.q[fr["q"]]["xy"][fr["moment"]], fx, fy)
                    inside = scorer_inside(sv, H, wpx, hpx)
                    t_all = rec[T.MISS_TEACHER]
                    r_all = m4["preds"][stem]["r23"]
                    feet = {"ours": T.feet_ft([p for p in rec[T.OURS] if p[5] == "player"], H),
                            "teacher": T.feet_ft([p for p in t_all if p[4] >= TEACHER_CONF], H),
                            "teacher_low": T.feet_ft([p for p in t_all if TEACHER_FLOOR <= p[4] < TEACHER_CONF], H),
                            "teacher_all": T.feet_ft([p for p in t_all if p[4] >= TEACHER_FLOOR], H),
                            "r23_low": T.feet_ft([p for p in r_all if R23_FLOOR <= p[4] < DET_CONF], H),
                            "r23_all": T.feet_ft([p for p in r_all if p[4] >= R23_FLOOR], H),
                            "e1": T.feet_ft([p for p in m4["preds"][stem]["e1"] if p[4] >= DET_CONF], H)}
                    rows = frame_rows(sv, inside, H, wpx, hpx, feet)
                    teacher_missed += sum(1 for r in rows if not r["teacher_reaches"])
                    rows_g += rows
                    groups.setdefault(f"{tag}/{fr['window']}", []).extend(rows)
        ref = m3["games"][tag]
        if (len(rows_g), sum(r["missed"] for r in rows_g), teacher_missed) != \
                (ref["ours"]["in_frame"], ref["ours"]["missed"], ref["teacher"]["missed"]):
            raise SystemExit(f"{WHO}: {tag} does not reproduce M3's in-frame, r23 and teacher missed counts")
        games[tag] = summary(rows_g)
        print(tag, json.dumps({k: games[tag][k] for k in ("in_frame", "r23_missed_rate", "reach_share_of_misses",
                                                          "ceilings")}))
    allrows = [r for v in groups.values() for r in v]
    out = {"what": "E1b upper bound: r23's raw misses by cause on the E1 corpus frames (docs/research/P8_E1B.md "
                   "section 2); raw boxes, bottom-centre feet, sportvu.eval rules; not the registered build rows",
           "rules": {"miss_ft": MISS_FT, "near_ft": NEAR_FT, "teacher_conf": TEACHER_CONF,
                     "teacher_floor": TEACHER_FLOOR, "r23_floor": R23_FLOOR, "det_conf": DET_CONF,
                     "box_w_of_h": BOX_W_OF_H, "overlap_edges": list(OVERLAP_EDGES), "edge_px": 40.0,
                     "boot": {"n": BOOT_N, "seed": SEED, "unit": "corpus window"}},
           "check": "per-game in-frame, r23 missed and teacher missed counts equal e1/postmortem_m3.json",
           "games": games, "pooled": summary(allrows), "pooled_boot": boot(groups)}
    d = wr / "e1b"
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(d / "bound.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_e1b_bound.json", text)
    print("pooled", json.dumps({k: out["pooled"][k] for k in ("in_frame", "r23_missed_rate", "reach_share_of_misses",
                                                              "ceilings", "r23_finds_under_teacher_labels")}))
    print("boot", json.dumps(out["pooled_boot"]))
    return out


# ── transfer-adjusted ceiling (section 2b) ─────────────────────────────────────────────────────────────────────────
BUILDS = {"gsw_bkn_2015": ("e1m6_r23_detcpu", "e1m6_e1_gsw_bkn_2015_detcpu"),
          "gsw_cha_2016": ("m9_r23_detcpu__gsw_cha_2016", "e1r_e1_gsw_cha_2016_detcpu"),
          "gsw_sac_2015": ("m9_r23_detcpu__gsw_sac_2015", "e1r_e1_gsw_sac_2015_detcpu")}
GATE_POOLED, GATE_GAME = -0.020, -0.010
TKEYS = ("n", "e1_gain_pres", "e1_gain_pres_kept", "e1_gain_plc", "e1_gain_plc_kept", "e1_loss", "e1_loss_kept",
         "l1_pres", "l1_plc", "l1_loss", "raw_r23_miss", "raw_e1_miss", "bld_r23_miss", "bld_e1_miss")


def transfer_counts(rr, er, tr, r5, rb, eb) -> dict:
    """Per-player booleans of one frame (in-frame players only): r23 raw reach, E1 raw reach, teacher raw reach, r23
    box within 5 ft, r23 build reach, E1 build reach -> the counts of TKEYS."""
    rr, er, tr, r5, rb, eb = (np.asarray(v, bool) for v in (rr, er, tr, r5, rb, eb))
    gain = ~rr & er
    loss = rr & ~er
    kept = eb & ~rb
    return {"n": int(len(rr)),
            "e1_gain_pres": int((gain & ~r5).sum()), "e1_gain_pres_kept": int((gain & ~r5 & kept).sum()),
            "e1_gain_plc": int((gain & r5).sum()), "e1_gain_plc_kept": int((gain & r5 & kept).sum()),
            "e1_loss": int(loss.sum()), "e1_loss_kept": int((loss & ~eb & rb).sum()),
            "l1_pres": int((~rr & tr & ~r5).sum()), "l1_plc": int((~rr & tr & r5).sum()), "l1_loss": int((rr & ~tr).sum()),
            "raw_r23_miss": int((~rr).sum()), "raw_e1_miss": int((~er).sum()),
            "bld_r23_miss": int((~rb).sum()), "bld_e1_miss": int((~eb).sum())}


def fractions(c: dict) -> dict:
    def fr(a, b):
        return c[a] / c[b] if c[b] else 0.0
    return {"f_pres": fr("e1_gain_pres_kept", "e1_gain_pres"), "f_plc": fr("e1_gain_plc_kept", "e1_gain_plc"),
            "f_loss": fr("e1_loss_kept", "e1_loss")}


def ceiling_terms(c: dict) -> dict:
    """Numerators (counts, negative = fewer misses) of C, C_opt and the model's prediction for E1, for one game."""
    f = fractions(c)
    gain = c["l1_pres"] * f["f_pres"] + c["l1_plc"] * f["f_plc"]
    e1 = (c["e1_gain_pres"] * f["f_pres"] + c["e1_gain_plc"] * f["f_plc"] - c["e1_loss"] * f["f_loss"])
    return {"C": -(gain - c["l1_loss"] * f["f_loss"]), "C_opt": -gain, "E1_model": -e1,
            "E1_raw": c["raw_e1_miss"] - c["raw_r23_miss"], "E1_build": c["bld_e1_miss"] - c["bld_r23_miss"]}


def pooled_rates(per_game: dict) -> dict:
    """{game: counts} -> {game: rates, "pooled": rates}; each game uses its own fractions."""
    out, tot, n = {}, {}, 0
    for g, c in per_game.items():
        t = ceiling_terms(c)
        out[g] = {k: v / c["n"] for k, v in t.items()}
        for k, v in t.items():
            tot[k] = tot.get(k, 0.0) + v
        n += c["n"]
    out["pooled"] = {k: v / n for k, v in tot.items()}
    return out


def gate(rates: dict, games) -> dict:
    worst = max(rates[g]["C"] for g in games)
    ok = rates["pooled"]["C"] <= GATE_POOLED and worst <= GATE_GAME
    return {"pooled_C": round(rates["pooled"]["C"], 5), "worst_game_C": round(worst, 5),
            "rule": f"C pooled <= {GATE_POOLED} and C <= {GATE_GAME} on every game", "passes": bool(ok)}


def transfer_boot(per_window: dict, n: int = BOOT_N, seed: int = SEED) -> dict:
    """per_window: {game/window: counts} -> intervals of C and C_opt per game and pooled (fractions recomputed)."""
    keys = sorted(per_window)
    games = sorted({k.split("/")[0] for k in keys})
    arr = np.array([[per_window[k][t] for t in TKEYS] for k in keys], float)
    gidx = {g: np.array([i for i, k in enumerate(keys) if k.startswith(g + "/")]) for g in games}
    rng = np.random.default_rng(seed)
    reps = {g: {"C": [], "C_opt": []} for g in games + ["pooled"]}
    for _ in range(n):
        per = {}
        for g in games:
            ix = gidx[g][rng.integers(0, len(gidx[g]), len(gidx[g]))]
            per[g] = dict(zip(TKEYS, arr[ix].sum(0)))
        r = pooled_rates(per)
        for g in games + ["pooled"]:
            for k in ("C", "C_opt"):
                reps[g][k].append(r[g][k])
    return {g: {k: [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]
                for k, v in d.items()} for g, d in reps.items()} | {"windows": len(keys),
                                                                    "resampling": "windows within each game"}


def cmd_transfer(args) -> dict:
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_teachers as T
    from tools.p8_e1_postmortem import scorer_inside
    from tools.p8_e1data import corpus_root
    check_games(POOL)
    root, wr = corpus_root(), p8_root()
    plan = json.loads((root / "plan.json").read_text())
    if sorted(plan["games"]) != sorted(POOL):
        raise SystemExit(f"{WHO}: the corpus plan names {sorted(plan['games'])}, not the pool games")
    guard([v for g in plan["games"].values() for v in (g["sportvu"], g["video_id"])], WHO)
    preds = {}
    for k in range(len(plan["chunks"])):
        preds.update(json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"])
    per_game, per_window, skipped = {}, {}, {}

    def sidecar(folder: str, window: str) -> dict:
        p = wr / "builds" / folder / f"{window}_frames.json"
        return {int(r["frame"]): r for r in json.loads(p.read_text())["frames"]} if p.exists() else {}

    def bc_ft(boxes, H):
        return T.feet_ft(boxes, H)

    def stab_ft(rows, H):
        import cv2
        if not rows:
            return np.zeros((0, 2))
        px = np.array([b["foot_stab"] for b in rows], np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(px, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)

    for tag, g in plan["games"].items():
        rfold, efold = BUILDS[tag]
        m4 = json.loads((wr / "e1" / f"m4_preds__{tag}.json").read_text())
        if m4["game"] != tag or m4["fold"] != tag or m4["device"] != "cpu" or m4["conf_floor"] > R23_FLOOR:
            raise SystemExit(f"{WHO}: m4_preds__{tag}.json is not the fold's CPU run on its left-out game")
        index = SportVUIndex(json.loads((wr / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        tot = {k: 0 for k in TKEYS}
        cache_w, sr, se = None, {}, {}
        n_skip = 0
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != tag:
                        continue
                    if fr["window"] != cache_w:
                        cache_w, sr, se = fr["window"], sidecar(rfold, fr["window"]), sidecar(efold, fr["window"])
                    a, b = sr.get(int(fr["frame"])), se.get(int(fr["frame"]))
                    if a is None or b is None:
                        n_skip += 1
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    rec = preds[f"{tag}/{stem}"]
                    wpx, hpx = rec["size"]
                    H = fr["H_truth"]
                    sv = apply_mirror(index.q[fr["q"]]["xy"][fr["moment"]], fx, fy)
                    inside = scorer_inside(sv, H, wpx, hpx)
                    svi = np.asarray(sv, np.float64).reshape(-1, 2)[inside]
                    P = m4["preds"][stem]
                    d = {"rr": nearest(svi, bc_ft([p for p in P["r23"] if p[4] >= DET_CONF], H)) <= MISS_FT,
                         "er": nearest(svi, bc_ft([p for p in P["e1"] if p[4] >= DET_CONF], H)) <= MISS_FT,
                         "tr": nearest(svi, bc_ft([p for p in rec[T.MISS_TEACHER] if p[4] >= TEACHER_CONF], H)) <= MISS_FT,
                         "r5": nearest(svi, bc_ft([p for p in P["r23"] if p[4] >= R23_FLOOR], H)) <= NEAR_FT,
                         "rb": nearest(svi, stab_ft(a["boxes"], H)) <= MISS_FT,
                         "eb": nearest(svi, stab_ft(b["boxes"], H)) <= MISS_FT}
                    c = transfer_counts(**d)
                    acc = per_window.setdefault(f"{tag}/{fr['window']}", {k: 0 for k in TKEYS})
                    for k, v in c.items():
                        acc[k] += v
                        tot[k] += v
        per_game[tag] = tot
        skipped[tag] = n_skip
    rates = pooled_rates(per_game)
    out = {"what": "E1b transfer-adjusted build ceiling of the L1 labels (docs/research/P8_E1B.md section 2b); stored "
                   "files only: corpus preds, M4 CPU preds and the reference-platform build sidecars",
           "builds": {g: {"r23": BUILDS[g][0], "e1": BUILDS[g][1]} for g in POOL},
           "corpus_frames_not_processed_by_both_builds": skipped,
           "counts": per_game,
           "fractions": {g: {k: round(v, 4) for k, v in fractions(c).items()} for g, c in per_game.items()},
           "rates": {g: {k: round(v, 5) for k, v in r.items()} for g, r in rates.items()},
           "half_of_C": {g: round(r["C"] / 2, 5) for g, r in rates.items()},
           "gate": gate(rates, POOL),
           "boot": transfer_boot(per_window),
           "report_only": ["C_opt", "half_of_C", "E1_model", "E1_raw", "E1_build", "boot"]}
    d = wr / "e1b"
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(d / "transfer.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_e1b_transfer.json", text)
    print(json.dumps({k: out[k] for k in ("corpus_frames_not_processed_by_both_builds", "fractions", "rates", "gate",
                                          "boot")}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_e1b_bound")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bound")
    sub.add_parser("transfer")
    a = ap.parse_args(argv)
    {"bound": cmd_bound, "transfer": cmd_transfer}[a.cmd](a)


if __name__ == "__main__":
    main()
