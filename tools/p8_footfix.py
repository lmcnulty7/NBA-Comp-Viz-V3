"""tools/p8_footfix.py: a foot-model correction on the decided truths (docs/research/P8_footfix.md, version 1.2).

    <frozen CV python> -m tools.p8_footfix score s0     inside the lock ("p8 footfix s0"): smoke, identity reproduction
                                                        of D202, the fit-row audit (0c), F1 and F2 in sample (0d)
    python3 -m tools.p8_footfix report s0               -> reports/research/p8_footfix_s0.json (G0)
    <frozen CV python> -m tools.p8_footfix score s1     inside the lock ("p8 footfix s1"): leave one game out
    python3 -m tools.p8_footfix report s1               -> reports/research/p8_footfix_s1.json (the rule, the readings)
    python3 -m tools.p8_footfix table                   -> reports/research/p8_footfix_table.json (F1 on both pool games,
                                                        only after stage 1 reads PASS and "feet nearer")
    <frozen CV python> -m tools.p8_footfix score hou --go TEXT   inside the lock ("p8 footfix hou"): once only
    python3 -m tools.p8_footfix report hou              -> reports/research/p8_footfix_hou.json

Two forms, fitted by medians on fit rows (in-frame players paired one to one with non-referee boxes by D205's two-stage
assignment under the truth camera T), toward t = T^-1(SportVU), the pixel where the decided truth images the player:
F1, a depth shift by cell (box height by third under B, the build's court H) applied in B's frame; F2, a vertical pixel
offset scaled by box height (the predicted-null control, section 13.1 (c)). A box whose cell has no
correction keeps its stored foot_stab exactly. Scoring is D202's: the pinned tools.p8_scorecard.score_window through
tools.p8_resetfix.score_pair, the corrected build's own folder and sidecar on both sides. Writes only under the
reference root's footfix/ folder and reports/research/p8_footfix_*.json; SHA-256 manifests of the watched folders
before and after every score step stop the run on any difference. hou_cle_2016 is refused by stages s0 and s1.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_footfix"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
POOL = (CHA, BKN)
FORMS = ("F1", "F2")
CANDIDATE, CONTROL = "F1", "F2"
HEIGHTS = ("<150", "150-200", "200-250", ">=250")
THIRDS = ("near", "middle", "far")
FIT_MIN = 50                                  # D186: a cell with fewer fit rows gets no correction
F2_NULL = 0.01                                # 13.1 (c): predicted |r_bin| at most 0.01
MISS_FT, NEAR_FT = 2.5, 5.0
BOOT_N, SEED = 2000, 0
MISSED_BAR = -0.010                           # rule 1 and G0
MARGINS = {"ghosts_per_box": 0.010, "nearfield_bias_ft": 0.10, "position_p90_ft": 0.10, "near_p90_ft": 0.05}
TEAM_MARGIN = -0.010
ONE2ONE_BAR, ONE2ONE_FRAC = -0.0075, 0.75     # rule 7 (13.3)
CAPS_S = {"s0": 2400, "s1": 1800, "hou": 1200}
SMOKE_TABLE = {h: 0.01 for h in HEIGHTS}      # the smoke's F2 table: exercises the path, reported nowhere
RECORDS = {s: REPO / "reports" / "research" / f"p8_footfix_{s}.json" for s in ("s0", "s1", "hou")}
TABLE_FILE = REPO / "reports" / "research" / "p8_footfix_table.json"
STAGE_GAMES = {"s0": POOL, "s1": POOL, "hou": (HOU,)}


# ── pure: cells, fits, corrections ─────────────────────────────────────────────────────────────────────────────────
def hbin(h: float) -> str:
    for e, lab in zip((150.0, 200.0, 250.0), HEIGHTS):
        if h < e:
            return lab
    return HEIGHTS[3]


def cell(hb: str, third: str) -> str:
    return f"{hb}|{third}"


def median_table(rows: list, key, field: str, min_n: int = FIT_MIN) -> tuple:
    """({key: median} for keys with at least min_n rows whose field is not None, {key: count})."""
    vals = {}
    for r in rows:
        if r.get(field) is not None:
            vals.setdefault(key(r), []).append(r[field])
    counts = {k: len(v) for k, v in vals.items()}
    return {k: float(np.median(v)) for k, v in vals.items() if len(v) >= min_n}, counts


def fit_tables(rows: list) -> dict:
    """F1: median depth toward T's image of the player per cell (rows with a B pose); F2: median ratio per height bin."""
    f1, n1 = median_table([r for r in rows if r.get("third")], lambda r: cell(r["hb"], r["third"]), "d1")
    f2, n2 = median_table(rows, lambda r: r["hb"], "r2")
    return {"F1": f1, "F2": f2, "counts": {"F1": n1, "F2": n2}}


def b_geometry(row: dict, W: int, H: int) -> tuple:
    """(B, ground point G_B, near side of B) of a sidecar frame, or (None, None, None) without an accepted B or pose."""
    from sportvu import bench as Bn
    from sportvu.camera import pose_from_h
    from sportvu.eval import near_side
    if not (Bn.accepted(row) and row.get("H") is not None):
        return None, None, None
    Hb = np.asarray(row["H"], np.float64).reshape(3, 3)
    pose = pose_from_h(Hb, W, H)
    if pose is None:
        return None, None, None
    return Hb, np.asarray(pose[2][:2], float), near_side(Hb)


def to_court(H, px) -> np.ndarray:
    import cv2
    p = np.asarray(px, np.float64).reshape(-1, 1, 2)
    if not len(p):
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)


def to_image(H, q) -> np.ndarray:
    return to_court(np.linalg.inv(np.asarray(H, np.float64).reshape(3, 3)), q)


def depth_basis(ground, p) -> np.ndarray:
    """tools.p8_placement.depth_basis's unit depth vector (camera ground point to p)."""
    from tools.p8_placement import depth_basis as db
    return db(ground, p)[0]


def corrected_feet(boxes: list, form: str, table: dict, geom: tuple) -> list:
    """Per box: the new foot_stab [x, y], or None when the box's cell has no correction (its stored values are kept)."""
    from tools.p8_scorecard import third_of
    out = [None] * len(boxes)
    if form == "F2":
        for i, b in enumerate(boxes):
            h = float(b["bbox"][3] - b["bbox"][1])
            r = table.get(hbin(h), 0.0)
            if r != 0.0:
                out[i] = [float(b["foot_stab"][0]), float(b["foot_stab"][1]) + r * h]
        return out
    if form != "F1":
        raise SystemExit(f"{WHO}: unknown form {form}")
    Hb, Gb, nhb = geom
    if Hb is None:
        return out
    for i, b in enumerate(boxes):
        q = to_court(Hb, [b["foot_stab"]])[0]
        s = table.get(cell(hbin(float(b["bbox"][3] - b["bbox"][1])), third_of(q[1], nhb)), 0.0)
        if s != 0.0:
            p = to_image(Hb, [q - s * depth_basis(Gb, q)])[0]
            out[i] = [float(p[0]), float(p[1])]
    return out


def correct_sidecar(side: dict, form: str, table: dict, W: int, H: int) -> tuple:
    """(a deep copy of the sidecar with foot_stab corrected, the number of boxes moved)."""
    new = copy.deepcopy(side)
    moved = 0
    for row in new["frames"]:
        geom = b_geometry(row, W, H) if form == "F1" else (None, None, None)
        for b, p in zip(row["boxes"], corrected_feet(row["boxes"], form, table, geom)):
            if p is not None:
                b["foot_stab"] = p
                moved += 1
    return new, moved


def copy_problems(orig: dict, new: dict) -> list:
    """Every difference between two sidecars other than box foot_stab values (empty when only feet moved)."""
    bad = []
    if set(orig) != set(new) or any(orig[k] != new[k] for k in orig if k != "frames"):
        bad.append("top-level fields differ")
    if len(orig.get("frames", [])) != len(new.get("frames", [])):
        return bad + ["frame count differs"]
    for a, b in zip(orig["frames"], new["frames"]):
        if set(a) != set(b) or any(a[k] != b[k] for k in a if k != "boxes"):
            bad.append(f"frame {a.get('frame')}: a field other than boxes differs")
        elif len(a["boxes"]) != len(b["boxes"]):
            bad.append(f"frame {a.get('frame')}: box count differs")
        else:
            for x, y in zip(a["boxes"], b["boxes"]):
                if set(x) != set(y) or any(x[k] != y[k] for k in x if k != "foot_stab") or len(y["foot_stab"]) != 2:
                    bad.append(f"frame {a.get('frame')}: a box field other than foot_stab differs")
                    break
        if len(bad) > 5:
            break
    return bad


def fit_rows_frame(feet_px, ft_T, sv, inside, is_ref, bh, t_px, geom, inlier: set, pids) -> list:
    """Fit rows of one scored frame (13.5 (7)): D205's main two-stage pairs (non-referee boxes against all ten players),
    kept for in-frame players. Each row: stage, height bin, B third (None without a B pose), d1 (depth of B(p) from
    B(t) along G_B to B(p)), dB (depth of B(p) from SportVU, D186's quantity, same row), r2 ((t_y - p_y) / h), inlier."""
    from tools.p8_one2one import _assign_all, two_stage
    from tools.p8_scorecard import third_of
    feet_px = np.asarray(feet_px, np.float64).reshape(-1, 2)
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    nonref = [int(b) for b in np.where(~np.asarray(is_ref, bool).reshape(-1))[0]] if len(feet_px) else []
    pairs, _ = two_stage(ft_T, sv, range(len(sv)), nonref, _assign_all)
    Hb, Gb, nhb = geom
    rows = []
    for j in sorted(pairs):
        if not inside[j]:
            continue
        k, dist, stage = pairs[j]
        h = float(bh[k])
        row = {"stage": stage, "hb": hbin(h), "third": None, "d1": None, "dB": None,
               "r2": float((t_px[j][1] - feet_px[k][1]) / h) if h > 0 else None,
               "inlier": int(pids[j]) in inlier, "dist_T": float(dist)}
        if Hb is not None:
            q = to_court(Hb, [feet_px[k]])[0]
            u = depth_basis(Gb, q)
            row["third"] = third_of(q[1], nhb)
            row["d1"] = float((q - to_court(Hb, [t_px[j]])[0]) @ u)
            row["dB"] = float((q - sv[j]) @ u)
        rows.append(row)
    return rows


# ── pure: the rule and the readings ────────────────────────────────────────────────────────────────────────────────
def guard_row(change: float | None, iv, bar: float, upper: bool = True) -> str:
    """'pass', 'harm' (the point beyond the bar) or 'inconclusive' (only the interval's end beyond it), 13.2."""
    if change is None or iv is None:
        return "missing"
    point_bad = change > bar if upper else change < bar
    end_bad = iv[1] > bar if upper else iv[0] < bar
    return "harm" if point_bad else ("inconclusive" if end_bad else "pass")


def rule7(d1to1: float | None, dmissed: float | None) -> bool:
    if d1to1 is None or dmissed is None:
        return False
    return d1to1 <= ONE2ONE_BAR or d1to1 <= ONE2ONE_FRAC * dmissed


def game_rule(ch: dict) -> dict:
    """Section 6 with 13.2 and 13.3 on one held-out game. ch: {row: {"change", "interval"}} (paired changes)."""
    m = ch["missed"]
    r1 = m["change"] is not None and m["interval"] is not None and m["change"] <= MISSED_BAR and m["interval"][1] < 0
    rows = {"ghosts_per_box": guard_row(ch["ghosts_per_box"]["change"], ch["ghosts_per_box"]["interval"],
                                        MARGINS["ghosts_per_box"]),
            "nearfield_bias_ft": guard_row(ch["nearfield_bias_ft"]["change"], ch["nearfield_bias_ft"]["interval"],
                                           MARGINS["nearfield_bias_ft"]),
            "position_p90_ft": guard_row(ch["position_p90_ft"]["change"], ch["position_p90_ft"]["interval"],
                                         MARGINS["position_p90_ft"]),
            "near_p90_ft": guard_row(ch["near_p90_ft"]["change"], ch["near_p90_ft"]["interval"], MARGINS["near_p90_ft"]),
            "team": guard_row(ch["team"]["change"], ch["team"]["interval"], TEAM_MARGIN, upper=False)}
    r7 = rule7(ch["missed_1to1"]["change"], m["change"])
    return {"rule1_missed": r1, "guards": rows, "rule7_one2one": r7,
            "harm": sorted(k for k, v in rows.items() if v == "harm"),
            "inconclusive": sorted(k for k, v in rows.items() if v in ("inconclusive", "missing")),
            "passes": bool(r1 and r7 and all(v == "pass" for v in rows.values()))}


def g0_game(ch: dict) -> dict:
    """G0 on one game (13.4 (4)): in-sample point estimates, missed at most -0.010, rules 2 to 7 by point."""
    pts = {"missed": ch["missed"]["change"] is not None and ch["missed"]["change"] <= MISSED_BAR,
           "ghosts_per_box": ch["ghosts_per_box"]["change"] is not None and
                             ch["ghosts_per_box"]["change"] <= MARGINS["ghosts_per_box"],
           "nearfield_bias_ft": ch["nearfield_bias_ft"]["change"] is not None and
                                ch["nearfield_bias_ft"]["change"] <= MARGINS["nearfield_bias_ft"],
           "position_p90_ft": ch["position_p90_ft"]["change"] is not None and
                              ch["position_p90_ft"]["change"] <= MARGINS["position_p90_ft"],
           "near_p90_ft": ch["near_p90_ft"]["change"] is not None and ch["near_p90_ft"]["change"] <= MARGINS["near_p90_ft"],
           "team": ch["team"]["change"] is not None and ch["team"]["change"] >= TEAM_MARGIN,
           "rule7_one2one": rule7(ch["missed_1to1"]["change"], ch["missed"]["change"])}
    return {"points": pts, "passes": all(pts.values())}


def copying_t(split: dict) -> bool:
    """13.1 (b): for missed or p50, the inlier players' change below 0 and the non-inlier players' 0 or above; and
    (v1.2, 14.1) the non-inlier players' missed change below 0 while their p50 change under B is above 0."""
    for row in ("missed", "position_p50_ft"):
        a, b = split["inlier"][row]["change"], split["non_inlier"][row]["change"]
        if a is not None and b is not None and a < 0 and b >= 0:
            return True
    m, p = split["non_inlier"]["missed"]["change"], split["non_inlier"]["position_p50_ft"]["change"]
    return m is not None and p is not None and m < 0 and p > 0


def reading(per_game: dict, descriptive_split: tuple = ()) -> dict:
    """13.1 (a) with 14.3: per_game {game: {"rule": game_rule, "p50": change, "split": inlier split}} -> the stage 1
    verdict, with explicit fields for rule 1 and rule 7 failures and inconclusive rows. descriptive_split: games whose
    inlier split is descriptive only (reading F), which then reads nothing."""
    passes = all(v["rule"]["passes"] for v in per_game.values())
    fields = {"rule1_failed": sorted(g for g, v in per_game.items() if not v["rule"].get("rule1_missed", True)),
              "rule7_failed": sorted(g for g, v in per_game.items() if not v["rule"].get("rule7_one2one", True)),
              "harm_rows": {g: v["rule"].get("harm", []) for g, v in per_game.items() if v["rule"].get("harm")},
              "inconclusive_rows": {g: v["rule"].get("inconclusive", []) for g, v in per_game.items()
                                    if v["rule"].get("inconclusive")}}
    if not passes:
        harm = any(fields["harm_rows"].values())
        return {"verdict": "FAIL", "reading": "harm" if harm else "not shown non-inferior or missed bar not met",
                **fields, "stage_2": False}
    p50_ok = all(v["p50"] is not None and v["p50"] <= 0 for v in per_game.values())
    copy_t = any(copying_t(v["split"]) for g, v in per_game.items() if g not in descriptive_split)
    nearer = p50_ok and not copy_t
    return {"verdict": "PASS", "reading": "feet nearer" if nearer else "agreement with T only", **fields,
            "p50_under_B_not_rising": p50_ok, "copying_T": copy_t, "split_descriptive_only": sorted(descriptive_split),
            "stage_2": nearer}


def fold_rows(rows_by_game: dict, held: str) -> tuple:
    """The fit rows of the other pool game for a held-out game; stops if any row comes from the held-out game."""
    if held not in POOL:
        raise SystemExit(f"{WHO}: {held} is not a pool game")
    other = [g for g in POOL if g != held]
    rows = [r for g in other for r in rows_by_game[g]]
    if any(str(r.get("w", "")).startswith(held) for r in rows):
        raise SystemExit(f"{WHO}: the fold table for {held} reads its own rows")
    return rows, other[0]


def f2_null(table: dict) -> dict:
    return {"bins": {k: v for k, v in sorted(table.items())},
            "holds": all(abs(v) <= F2_NULL for v in table.values())}


# ── pure: the inlier split ─────────────────────────────────────────────────────────────────────────────────────────
def split_matrix(base: dict, new: dict, inl: dict) -> dict:
    """base/new: {window: {"f:pid": [missed, err or None]}}; inl: {window: set of "f:pid"} -> per class, per window:
    [n, missed base, missed new] and the paired position errors of players matched in both arms."""
    out = {c: {"counts": {}, "err": {}} for c in ("inlier", "non_inlier")}
    for w in sorted(base):
        for c in out:
            out[c]["counts"][w] = [0, 0, 0]
            out[c]["err"][w] = ([], [])
        if set(base[w]) != set(new[w]):
            raise SystemExit(f"{WHO}: {w}: the in-frame players differ between arms")
        for k, (mb, eb) in base[w].items():
            mn, en = new[w][k]
            c = "inlier" if k in inl.get(w, set()) else "non_inlier"
            cnt = out[c]["counts"][w]
            cnt[0] += 1; cnt[1] += int(mb); cnt[2] += int(mn)
            if eb is not None and en is not None:
                out[c]["err"][w][0].append(eb); out[c]["err"][w][1].append(en)
    return out


def split_rows(sm: dict, n: int = BOOT_N, seed: int = SEED) -> dict:
    """Missed change and p50 change per class with window intervals (windows resampled whole)."""
    res = {}
    for c, d in sm.items():
        ws = sorted(d["counts"])
        C = np.array([d["counts"][w] for w in ws], float).reshape(-1, 3)

        def miss(ix):
            s = C[ix].sum(0)
            return (s[2] - s[1]) / s[0] if s[0] else None

        def p50(ix):
            a = np.concatenate([np.asarray(d["err"][ws[i]][0], float) for i in ix]) if len(ix) else np.zeros(0)
            b = np.concatenate([np.asarray(d["err"][ws[i]][1], float) for i in ix]) if len(ix) else np.zeros(0)
            return float(np.median(b) - np.median(a)) if len(a) else None

        rng = np.random.default_rng(seed)
        picks = [rng.integers(0, len(ws), len(ws)) for _ in range(n)] if len(ws) >= 2 else []
        row = {}
        for name, fn in (("missed", miss), ("position_p50_ft", p50)):
            pt = fn(np.arange(len(ws)))
            reps = [v for v in (fn(p) for p in picks) if v is not None]
            row[name] = {"change": pt, "interval": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
                         if reps else None}
        row["players"] = int(C[:, 0].sum()) if len(C) else 0
        res[c] = row
    return res


# ── data helpers (frozen CV python) ────────────────────────────────────────────────────────────────────────────────
def check_stage_games(stage: str, games) -> None:
    allowed = STAGE_GAMES[stage]
    bad = [g for g in games if g not in allowed]
    if bad:
        raise SystemExit(f"{WHO}: stage {stage} refuses {bad}")
    refuse_cv_test(list(games), WHO)
    guard(list(games), WHO)


def footfix_dir() -> Path:
    from tools import p8_rebaseline as RB
    return RB.roots()["ref"] / "footfix"


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                           text=True).stdout.strip()
    return {"head": head, "tools_and_sportvu_clean": not dirty}


@contextlib.contextmanager
def capture_obs(store: dict):
    """Record the pinned score_window's observations (its return is passed through unchanged) while score_pair runs;
    the original function is restored and checked after."""
    from tools import p8_scorecard as S
    orig = S.score_window

    def wrapped(P, tag, w, build_dir, *a, **k):
        full = orig(P, tag, w, build_dir, *a, **k)
        store[(str(build_dir), w["window"])] = full[1]
        return full

    S.score_window = wrapped
    try:
        yield
    finally:
        S.score_window = orig
    if S.score_window is not orig:
        raise SystemExit(f"{WHO}: score_window was not restored")


def compact_obs(obs: list) -> dict:
    """{"f:pid": [missed, err]} of one window's in-frame players."""
    return {f"{int(o['f'])}:{int(o['pid'])}": [bool(o["missed"]), o["err"]] for o in obs}


def game_context(game: str, stage: str) -> dict:
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    check_stage_games(stage, [game])          # hou_cle_2016 is refused by stages s0 and s1
    r = RB.roots()
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    P = F.install_paths()
    root = RB.truths(r)[game]["current"]
    P_t = {**P, "labels": root / "labels"}
    rec = RB.read_json(r["ref"] / "rebaseline" / f"score__{game}__current.json")
    if RB.labels_pin(root, game) != rec["labels_pin"]:
        raise SystemExit(f"{WHO}: {game}'s labels differ from D202's pin")
    mirror = F.mirror_of(P, game)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    comp = P["builds"] / "r23"
    wins = sorted((w for w in F.label_windows(P, game) if (comp / f"{w['window']}_frames.json").exists()),
                  key=lambda w: w["window"])
    stems = [w["window"] for w in wins]
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    if stems != sorted(rec["windows"]):
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's score record")
    ctx = {"mom": mom, "mirror": mirror, "refs": B.References(), "events": S.pbp_events(g["sportvu"]),
           "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
    return {"game": game, "P": P, "P_t": P_t, "rec": rec, "wins": wins, "comp": comp, "ctx": ctx, "root": root}


def check_inputs(gc: dict) -> None:
    """D202's hashes of every window's sidecar, time map, labels and identity, and a complete scorecard cache."""
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import cache_gaps
    P_t, comp, game = gc["P_t"], gc["comp"], gc["game"]
    for w in gc["wins"]:
        s = w["window"]
        for key, p in (("sidecar", comp / f"{s}_frames.json"), ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                       ("labels", P_t["labels"] / game / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
            want = (gc["rec"].get("hashes") or {}).get(s, {}).get(key)
            if want is None and key in ("sidecar", "timemap", "labels"):
                raise SystemExit(f"{WHO}: {s} has no {key} hash in D202's record")
            if want is not None and RB.sha256(p) != want:
                raise SystemExit(f"{WHO}: {s} {key} differs from D202's record")
        why = cache_gaps(P_t, game, w, comp, gc["ctx"]["mom"], gc["ctx"]["mirror"])
        if why:
            raise SystemExit(f"{WHO}: {s}: the scorecard cache lacks {why}")


def clip_size(P: dict, stem: str) -> tuple:
    import cv2
    cap = cv2.VideoCapture(str(P["label_build"] / f"{stem}.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.release()
    return W, H


def write_arm(gc: dict, arm: str, form: str | None, table: dict | None, stems: list | None = None) -> dict:
    """The arm's build folder: per window the corrected frames.json (identity when form is None) and byte copies of the
    window's other files; copy checks. -> {"dir", "sides", "moved", "frames_sha"}."""
    from tools import p8_rebaseline as RB
    d = footfix_dir() / "builds" / arm
    d.mkdir(parents=True, exist_ok=True)
    comp = gc["comp"]
    sides, moved, sha = {}, 0, {}
    for w in gc["wins"]:
        s = w["window"]
        if stems is not None and s not in stems:
            continue
        orig = RB.read_json(comp / f"{s}_frames.json")
        # the identity arm goes through the same path with an empty F1 table: no box moves, no round trip (13.5 (8))
        new, n = correct_sidecar(orig, form or "F1", table if form else {}, *clip_size(gc["P"], s))
        bad = copy_problems(orig, new)
        if bad:
            raise SystemExit(f"{WHO}: {arm} {s}: {bad[:3]}")
        atomic_write(d / f"{s}_frames.json", json.dumps(new))
        for p in sorted(comp.glob(f"{s}_*")):
            if p.name == f"{s}_frames.json":
                continue
            shutil.copyfile(p, d / p.name)
            if RB.sha256(d / p.name) != RB.sha256(p):
                raise SystemExit(f"{WHO}: {arm} {s}: the copy of {p.name} differs")
        if form is None and RB.sha256(d / f"{s}_frames.json") != RB.sha256(comp / f"{s}_frames.json"):
            raise SystemExit(f"{WHO}: {arm} {s}: the identity sidecar is not byte-equal to builds/r23's")
        sides[s], moved, sha[s] = new, moved + n, RB.sha256(d / f"{s}_frames.json")
    return {"dir": d, "sides": sides, "moved": moved, "frames_sha": sha}


def score_arm(gc: dict, arm: dict, t0: float, cap_s: float) -> tuple:
    """Every window of the arm through score_pair (the arm's folder and sidecar on both sides) -> (records, obs)."""
    from tools.p8_resetfix import score_pair
    recs, obs, store = {}, {}, {}
    with capture_obs(store):
        for w in gc["wins"]:
            s = w["window"]
            if s not in arm["sides"]:
                continue
            if time.time() - t0 > cap_s:
                raise SystemExit(f"{WHO}: the cap was reached; nothing is reported")
            side = arm["sides"][s]
            recs[s] = score_pair(gc["P_t"], gc["game"], w, {"r23": arm["dir"], "new": arm["dir"]},
                                 {"r23": side, "new": side}, gc["ctx"])["r23"]
            obs[s] = compact_obs(store.pop((str(arm["dir"]), s)))
    return recs, obs


def moves_together(base: dict, recs: dict) -> bool:
    """13.5 (9): when missed changes against the baseline, missed_1to1 changes too."""
    dm = sum(recs[s]["missed"] for s in recs) - sum(base[s]["missed"] for s in recs)
    d1 = sum(recs[s]["missed_1to1"] for s in recs) - sum(base[s]["missed_1to1"] for s in recs)
    return dm == 0 or d1 != 0


def inliers_window(gc: dict, w: dict) -> tuple:
    """13.1 (b): sportvu.truth.window_truth on the window's label-build sidecar and time map, as the truth step ran it.
    -> ({frame: set of inlier SportVU ids}, {frame: label-build sidecar row}); stops when a frame the truth accepted
    has no recomputed ok row with the same q and moment."""
    from sportvu.truth import window_truth
    P_t, s = gc["P_t"], w["window"]
    sc = json.loads((P_t["label_build"] / f"{s}_frames.json").read_text())
    tm = json.loads((P_t["label_build"] / f"{s}_timemap.json").read_text())
    _, _, rows = window_truth(w, sc, tm, gc["ctx"]["mom"].index, gc["ctx"]["mirror"])
    by_f = {int(r["frame"]): r for r in rows}
    lp = P_t["labels"] / gc["game"] / f"{s}.jsonl"
    labels = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()] if lp.exists() else []
    for lab in labels:
        r = by_f.get(int(lab["frame"]))
        if r is None or r.get("status") != "ok" or int(r["q"]) != int(lab["q"]) or \
                (lab.get("moment") is not None and int(r["moment"]) != int(lab["moment"])):
            raise SystemExit(f"{WHO}: {s} frame {lab['frame']}: the recomputed truth row does not match the label row")
    inl = {f: {int(p["pid"]) for p in r.get("pairs", []) if p.get("inlier")} for f, r in by_f.items()
           if r.get("status") == "ok"}
    return inl, {int(x["frame"]): x for x in sc["frames"]}


def fit_rows_window(gc: dict, w: dict, scored: set, inl: dict, label_rows: dict) -> tuple:
    """Fit rows of one window, and the count of scored frames whose label-build boxes equal builds/r23's."""
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.sync import apply_mirror
    from tools import p8_scorecard as S
    P_t, s, ctx = gc["P_t"], w["window"], gc["ctx"]
    side = json.loads((gc["comp"] / f"{s}_frames.json").read_text())
    srow = {int(x["frame"]): x for x in side["frames"]}
    sidecar_cls = any("cls" in b for x in srow.values() for b in x["boxes"])
    cache = json.loads((P_t["cache"] / "scorecard" / f"{s}.json").read_text())
    refd = {int(k): v for k, v in cache.get("referees", {}).items()}
    W, H = clip_size(gc["P"], s)
    truth, _ = S.truth_rows(P_t, gc["game"], w, "camera", ctx["mom"].index, ctx["mirror"])
    rows, same_boxes, frames = [], 0, 0
    for row in truth:
        f = int(row["frame"])
        x = srow.get(f)
        if x is None or f not in scored:
            continue
        frames += 1
        lb = label_rows.get(f)
        same_boxes += lb is not None and [(b["bbox"], b["foot_stab"]) for b in lb["boxes"]] == \
            [(b["bbox"], b["foot_stab"]) for b in x["boxes"]]
        d = ctx["mom"].index.q[row["q"]]
        sv = apply_mirror(d["xy"][row["moment"]], ctx["fx"], ctx["fy"])
        Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
        t_px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
        m = FRAME_MARGIN_PX
        inside = (t_px[:, 0] > m) & (t_px[:, 0] < W - m) & (t_px[:, 1] > m) & (t_px[:, 1] < H - m)
        feet = np.array([b["foot_stab"] for b in x["boxes"]], np.float32).reshape(-1, 1, 2)
        if not len(feet):
            continue
        ft = cv2.perspectiveTransform(feet, Ht).reshape(-1, 2).astype(np.float64)
        is_ref = np.array([(b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else
                           any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
                           for b in x["boxes"]], bool)
        bh = [float(b["bbox"][3] - b["bbox"][1]) for b in x["boxes"]]
        for fr in fit_rows_frame(feet.reshape(-1, 2), ft, np.asarray(sv, np.float64), inside, is_ref, bh,
                                 t_px.astype(np.float64), b_geometry(x, W, H), inl.get(f, set()), d["pid"][row["moment"]]):
            rows.append({"w": s, "f": f, **fr})
    return rows, {"frames": frames, "label_boxes_equal": same_boxes}


def scored_frames(obs: dict) -> set:
    return {int(k.split(":")[0]) for k in obs}


def inlier_keys(inl: dict, obs: dict) -> set:
    return {k for k in obs if int(k.split(":")[1]) in inl.get(int(k.split(":")[0]), set())}


# ── audit (0c), pure on fit rows ───────────────────────────────────────────────────────────────────────────────────
def same_row_table(rows: list) -> dict:
    """Per cell: medians of d1 (toward T's image) and dB (toward SportVU under B) on the same rows, the share with
    opposite signs, and the stage B (P-ii) rows."""
    out = {}
    for r in rows:
        if r.get("third") is None:
            continue
        out.setdefault(cell(r["hb"], r["third"]), []).append(r)
    res = {}
    for c, rr in sorted(out.items()):
        d1 = np.array([r["d1"] for r in rr]); dB = np.array([r["dB"] for r in rr])
        res[c] = {"n": len(rr), "p_ii_rows": sum(r["stage"] == "B" for r in rr),
                  "median_d1_toward_T": float(np.median(d1)), "median_dB_toward_sportvu_under_B": float(np.median(dB)),
                  "median_difference": float(np.median(d1 - dB)),
                  "opposite_sign_share": float(np.mean(np.sign(d1) != np.sign(dB)))}
    return res


def audit(rows_by_game: dict) -> dict:
    out = {}
    for g, rows in rows_by_game.items():
        p_ii = [r["dist_T"] for r in rows if r["stage"] == "B"]
        edges = np.arange(2.5, 5.01, 0.5)
        out[g] = {"fit_rows": len(rows), "stage_b_share": float(np.mean([r["stage"] == "B" for r in rows])) if rows else None,
                  "inlier_share": float(np.mean([r["inlier"] for r in rows])) if rows else None,
                  "stage_by_inlier": {f"{st}|{'inlier' if il else 'non_inlier'}":
                                      sum(r["stage"] == st and bool(r["inlier"]) == il for r in rows)
                                      for st in ("A", "B") for il in (True, False)},
                  "tables": {"all": fit_tables(rows), "inlier_only": fit_tables([r for r in rows if r["inlier"]]),
                             "non_inlier_only": fit_tables([r for r in rows if not r["inlier"]])},
                  "f2_null_in_sample": f2_null(fit_tables(rows)["F2"]),
                  "same_row": same_row_table(rows),
                  "p_ii_distance_ft": {f"{a:.1f}-{b:.1f}": int(((np.asarray(p_ii) > a) & (np.asarray(p_ii) <= b)).sum())
                                       for a, b in zip(edges[:-1], edges[1:])}}
    return out


A_FT, A_SHARE = 0.3, 0.8          # 14.8 reading A
E_EDGE_RATE = 0.02                 # 14.8 reading E
F_BOX_SHARE = 0.9                  # 14.8 reading F


def audit_readings(aud: dict, in_frame: dict, label_boxes: dict) -> dict:
    """The audit readings A to F of 14.8, registered before the run, per game and overall."""
    per = {}
    for g, a in aud.items():
        allt = a["tables"]["all"]
        n1, t1 = allt["counts"]["F1"], allt["F1"]
        tot = sum(n1.values())
        small = sum(n for c, n in n1.items() if abs(t1.get(c, 0.0)) <= A_FT)
        sr = a["same_row"]
        pii = sum(v["p_ii_rows"] for v in sr.values())
        conflict = [c for c, v in sr.items() if np.sign(v["median_d1_toward_T"]) != np.sign(v["median_dB_toward_sportvu_under_B"])
                    or v["opposite_sign_share"] > 0.5]
        pii_conflict = sum(sr[c]["p_ii_rows"] for c in conflict)
        ti, tn = a["tables"]["inlier_only"]["F1"], a["tables"]["non_inlier_only"]["F1"]
        opp = [c for c in ti if c in tn and np.sign(ti[c]) != np.sign(tn[c]) and ti[c] != 0 and tn[c] != 0]
        opp_rows = sum(n1.get(c, 0) for c in opp)
        edge = a["p_ii_distance_ft"].get("2.5-3.0", 0)
        lb = label_boxes[g]
        per[g] = {"A_share_within_0.3ft": small / tot if tot else None, "A": bool(tot and small / tot >= A_SHARE),
                  "B_conflict_cells": sorted(conflict), "B_p_ii_share": pii_conflict / pii if pii else None,
                  "B": bool(pii and pii_conflict / pii >= 0.5),
                  "C_opposite_cells": sorted(opp), "C_row_share": opp_rows / tot if tot else None,
                  "C": bool(tot and opp_rows / tot > 0.5),
                  "D": f2_null(allt["F2"])["holds"],
                  "E_edge_rate": edge / in_frame[g] if in_frame[g] else None,
                  "E": bool(in_frame[g] and edge / in_frame[g] < E_EDGE_RATE),
                  "F_box_share": lb["label_boxes_equal"] / lb["frames"] if lb["frames"] else None,
                  "F": bool(not lb["frames"] or lb["label_boxes_equal"] / lb["frames"] < F_BOX_SHARE)}
    gs = list(per)
    return {"per_game": per,
            "A_absorbed": all(per[g]["A"] for g in gs),
            "B_frame_conflict_on": [g for g in gs if per[g]["B"]],
            "C_T_residual_dominates_on": [g for g in gs if per[g]["C"]],
            "D_f2_null_holds": all(per[g]["D"] for g in gs),
            "E_bar_out_of_reach_on": [g for g in gs if per[g]["E"]],
            "F_split_descriptive_on": [g for g in gs if per[g]["F"]],
            "closes_toward_T_line": all(per[g]["A"] for g in gs)}


# ── score (frozen CV python, inside the lock) ──────────────────────────────────────────────────────────────────────
def watched_manifest() -> dict:
    from tools import p8_rebaseline as RB
    r = RB.roots()
    return {"ref": RB.manifest(r["ref"], ("labels", "label_build", "builds", "truth_accept", "truth_v2", "rebaseline",
                                         "cache")),
            "ms": RB.manifest(r["ms"], ("labels", "label_build", "builds", "truth_accept", "truth_v2")),
            "ws": RB.manifest(r["ws"], ("labels", "label_build", "truth_accept"))}


def preflight(stage: str) -> dict:
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    gs = git_state()
    if not gs["tools_and_sportvu_clean"]:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    pinned, now = RB.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if any(now[k] != pinned.get(k) for k in now):
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    check_stage_games(stage, STAGE_GAMES[stage])
    return gs


def versions() -> dict:
    import cv2
    return {"cv2": cv2.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}


def save(path: Path, obj) -> str:
    from tools import p8_rebaseline as RB
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(obj))
    return RB.sha256(path)


def cmd_score_s0() -> dict:
    from tools import p8_scorecard as S
    gs = preflight("s0")
    t0, cap = time.time(), CAPS_S["s0"]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t0))
    d = footfix_dir() / "s0"
    before = watched_manifest()
    S.load_fixed_constants()
    out = {"stage": "s0", "run_id": run_id, "git": gs, "versions": versions(), "games": {}, "files": {}}
    gcs = {g: game_context(g, "s0") for g in POOL}
    for g, gc in gcs.items():
        check_inputs(gc)
    # 0a smoke: one window per game, identity and F2 at the smoke table
    for g, gc in gcs.items():
        s = gc["wins"][0]["window"]
        idn = write_arm(gc, f"smoke_identity__{g}", None, None, [s])
        if idn["frames_sha"][s] != gc["rec"]["hashes"][s]["sidecar"]:
            raise SystemExit(f"{WHO}: smoke {s}: the identity sidecar is not byte-equal to builds/r23's")
        recs, _ = score_arm(gc, idn, t0, cap)
        from tools.p8_rebaseline import gate_problems
        bad = gate_problems(recs, {s: {"new": gc["rec"]["windows"][s]}})
        if bad:
            raise SystemExit(f"{WHO}: smoke {s}: {bad}")
        f2 = write_arm(gc, f"smoke_F2__{g}", "F2", SMOKE_TABLE, [s])
        score_arm(gc, f2, t0, cap)
        out["games"].setdefault(g, {})["smoke"] = {"window": s, "identity_equal": True, "f2_moved": f2["moved"]}
    # 0b reproduction: identity on every window equals D202's records
    base = {}
    for g, gc in gcs.items():
        idn = write_arm(gc, f"s0_identity__{g}", None, None)
        if idn["moved"] or any(idn["frames_sha"][s] != gc["rec"]["hashes"][s]["sidecar"] for s in idn["frames_sha"]):
            raise SystemExit(f"{WHO}: {g}: the identity sidecars are not byte-equal to builds/r23's")
        recs, obs = score_arm(gc, idn, t0, cap)
        from tools.p8_rebaseline import gate_problems
        bad = gate_problems(recs, {s: {"new": v} for s, v in gc["rec"]["windows"].items()})
        if bad:
            raise SystemExit(f"{WHO}: {g}: the reproduction gate fails: {bad[:3]}")
        base[g] = {"recs": recs, "obs": obs}
        out["files"][f"identity__{g}"] = save(d / f"score__identity__{g}.json", base[g])
        out["games"][g]["reproduction"] = {"windows": len(recs), "passes": True}
    # 0c fit rows, inliers and the audit, written before the in-sample scoring
    rows_by_game, inl_keys = {}, {}
    for g, gc in gcs.items():
        rows, eq, keys = [], {"frames": 0, "label_boxes_equal": 0}, {}
        for w in gc["wins"]:
            s = w["window"]
            inl, label_rows = inliers_window(gc, w)
            rr, e = fit_rows_window(gc, w, scored_frames(base[g]["obs"][s]), inl, label_rows)
            rows += rr
            eq = {k: eq[k] + e[k] for k in eq}
            keys[s] = sorted(inlier_keys(inl, base[g]["obs"][s]))
            if time.time() - t0 > cap:
                raise SystemExit(f"{WHO}: the cap was reached; nothing is reported")
        rows_by_game[g], inl_keys[g] = rows, keys
        out["games"][g]["label_boxes"] = eq
        out["files"][f"fitrows__{g}"] = save(d / f"fitrows__{g}.json", rows)
        out["files"][f"inliers__{g}"] = save(d / f"inliers__{g}.json", keys)
    out["audit"] = audit(rows_by_game)
    out["files"]["audit"] = save(d / "audit.json", {"run_id": run_id, "audit": out["audit"]})
    # 0d in sample: each form fitted and scored on the same game
    for g, gc in gcs.items():
        tabs = fit_tables(rows_by_game[g])
        for form in FORMS:
            arm = write_arm(gc, f"s0_{form}_in__{g}", form, tabs[form])
            recs, obs = score_arm(gc, arm, t0, cap)
            if not moves_together(base[g]["recs"], recs):
                raise SystemExit(f"{WHO}: {g} {form}: missed moved but missed_1to1 did not (stale sidecar?)")
            out["files"][f"{form}_in__{g}"] = save(d / f"score__{form}_in__{g}.json",
                                                   {"table": tabs[form], "moved": arm["moved"], "recs": recs, "obs": obs})
    after = watched_manifest()
    if after != before:
        raise SystemExit(f"{WHO}: a watched folder changed during the run; nothing is reported")
    out["seconds"] = round(time.time() - t0, 1)
    save(d / "run.json", out)
    print(json.dumps({k: out[k] for k in ("run_id", "seconds")}))
    return out


def cmd_score_s1() -> dict:
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    gs = preflight("s1")
    s0 = RB.read_json(RECORDS["s0"])
    run0 = RB.read_json(footfix_dir() / "s0" / "run.json")
    if s0.get("run_id") != run0["run_id"] or s0.get("g0", {}).get("verdict") != "PASS":
        raise SystemExit(f"{WHO}: stage 1 runs only after stage 0's record reads G0 PASS for the current s0 run")
    if s0.get("audit_readings", {}).get("closes_toward_T_line", True):
        raise SystemExit(f"{WHO}: reading A (absorbed) closes the toward-T line; stage 1 does not run")
    t0, cap = time.time(), CAPS_S["s1"]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t0))
    d = footfix_dir() / "s1"
    before = watched_manifest()
    S.load_fixed_constants()
    rows = {}
    for g in POOL:
        p = footfix_dir() / "s0" / f"fitrows__{g}.json"
        if RB.sha256(p) != run0["files"][f"fitrows__{g}"]:
            raise SystemExit(f"{WHO}: {g}'s fit rows differ from stage 0's record")
        rows[g] = RB.read_json(p)
    out = {"stage": "s1", "run_id": run_id, "s0_run_id": run0["run_id"], "git": gs, "versions": versions(),
           "files": {}, "tables": {}}
    for g in POOL:
        gc = game_context(g, "s1")
        check_inputs(gc)
        fold, other = fold_rows(rows, g)
        tabs = fit_tables(fold)
        bp = footfix_dir() / "s0" / f"score__identity__{g}.json"
        if RB.sha256(bp) != run0["files"][f"identity__{g}"]:
            raise SystemExit(f"{WHO}: {g}'s identity record differs from stage 0's record")
        base = RB.read_json(bp)
        for form in FORMS:
            arm = write_arm(gc, f"s1_{form}_logo__{g}", form, tabs[form])
            recs, obs = score_arm(gc, arm, t0, cap)
            if not moves_together(base["recs"], recs):
                raise SystemExit(f"{WHO}: {g} {form}: missed moved but missed_1to1 did not (stale sidecar?)")
            out["tables"][f"{form}__{g}"] = {"fitted_on": other, "table": tabs[form], "counts": tabs["counts"][form]}
            out["files"][f"{form}_logo__{g}"] = save(d / f"score__{form}_logo__{g}.json",
                                                     {"table": tabs[form], "moved": arm["moved"], "recs": recs, "obs": obs})
    if watched_manifest() != before:
        raise SystemExit(f"{WHO}: a watched folder changed during the run; nothing is reported")
    out["seconds"] = round(time.time() - t0, 1)
    save(d / "run.json", out)
    print(json.dumps({k: out[k] for k in ("run_id", "seconds")}))
    return out


def marker_path() -> Path:
    return footfix_dir() / "hou_once.json"


def cmd_score_hou(go: str) -> dict:
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    if not go or not go.strip():
        raise SystemExit(f"{WHO}: stage 2 needs the lead's go (--go)")
    gs = preflight("hou")
    s1 = RB.read_json(RECORDS["s1"]) if RECORDS["s1"].exists() else {}
    if s1.get("verdict", {}).get("verdict") != "PASS" or not s1.get("verdict", {}).get("stage_2"):
        raise SystemExit(f"{WHO}: stage 2 needs stage 1's PASS of F1 with the reading 'feet nearer'")
    if not TABLE_FILE.exists():
        raise SystemExit(f"{WHO}: the pooled F1 table is not written")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(TABLE_FILE)], cwd=REPO, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(TABLE_FILE)], cwd=REPO, capture_output=True,
                           text=True).stdout.strip()
    if tracked.returncode != 0 or dirty:
        raise SystemExit(f"{WHO}: the pooled F1 table must be committed and unchanged before hou_cle_2016 is read")
    table = RB.read_json(TABLE_FILE)
    if marker_path().exists():
        raise SystemExit(f"{WHO}: hou_cle_2016 was already started ({marker_path().name}); a retry needs the lead's "
                         f"decision")
    t0, cap = time.time(), CAPS_S["hou"]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t0))
    save(marker_path(), {"run_id": run_id, "go": go, "table_sha256": RB.sha256(TABLE_FILE), "git": gs})
    d = footfix_dir() / "hou"
    before = watched_manifest()
    S.load_fixed_constants()
    gc = game_context(HOU, "hou")
    check_inputs(gc)
    out = {"stage": "hou", "run_id": run_id, "git": gs, "versions": versions(), "go": go,
           "table_sha256": RB.sha256(TABLE_FILE), "files": {}}
    idn = write_arm(gc, f"hou_identity__{HOU}", None, None)
    recs0, obs0 = score_arm(gc, idn, t0, cap)
    bad = RB.gate_problems(recs0, {s: {"new": v} for s, v in gc["rec"]["windows"].items()})
    if bad:
        raise SystemExit(f"{WHO}: hou_cle_2016: the reproduction gate fails: {bad[:3]}")
    out["files"]["identity"] = save(d / "score__identity.json", {"recs": recs0, "obs": obs0})
    arm = write_arm(gc, f"hou_F1__{HOU}", "F1", table["table"])
    recs, obs = score_arm(gc, arm, t0, cap)
    if not moves_together(recs0, recs):
        raise SystemExit(f"{WHO}: hou_cle_2016: missed moved but missed_1to1 did not (stale sidecar?)")
    out["files"]["F1"] = save(d / "score__F1.json", {"table": table["table"], "moved": arm["moved"], "recs": recs,
                                                     "obs": obs})
    if watched_manifest() != before:
        raise SystemExit(f"{WHO}: a watched folder changed during the run; nothing is reported")
    out["seconds"] = round(time.time() - t0, 1)
    save(d / "run.json", out)
    print(json.dumps({k: out[k] for k in ("run_id", "seconds")}))
    return out


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def changes(base: dict, new: dict) -> dict:
    """D202's rows as paired changes (tools.p8_rebaseline.change_rows), plus one-to-one missed and near-third p90."""
    from tools import p8_rebaseline as RB
    from tools.p8_outpos import near_p90_boot
    from tools.p8_resetfix import ratio_interval, resample_picks
    ws = sorted(base)
    prev, curr = [base[s] for s in ws], [new[s] for s in ws]
    out = RB.change_rows(prev, curr)
    picks = resample_picks(len(ws), BOOT_N, SEED) if len(ws) >= 2 else None
    bn, bd = [a["missed_1to1"] for a in prev], [a["in_frame"] for a in prev]
    nn, nd = [b["missed_1to1"] for b in curr], [b["in_frame"] for b in curr]
    out["missed_1to1"] = {"change": sum(nn) / sum(nd) - sum(bn) / sum(bd),
                          "interval": ratio_interval(picks, bn, bd, nn, nd) if picks is not None else None}
    npb = near_p90_boot({s: {"r23": base[s], "new": new[s]} for s in ws}, BOOT_N, SEED)
    out["near_p90_ft"] = {"change": npb["diff"], "interval": npb["iv"]}
    if [a["in_frame"] for a in prev] != [b["in_frame"] for b in curr] or \
            [a["nr_boxes"] for a in prev] != [b["nr_boxes"] for b in curr]:
        raise SystemExit(f"{WHO}: in-frame players or boxes differ between arms")
    return out


def arm_report(stage_dir: Path, base_name: str, arm_name: str, inl: dict | None) -> dict:
    from tools import p8_rebaseline as RB
    base = RB.read_json(stage_dir.parent / "s0" / base_name) if (stage_dir.parent / "s0" / base_name).exists() \
        else RB.read_json(stage_dir / base_name)
    arm = RB.read_json(stage_dir / arm_name)
    ch = changes(base["recs"], arm["recs"])
    rep = {"table": arm["table"], "boxes_moved": arm["moved"], "changes": ch}
    if inl is not None:
        sm = split_matrix(base["obs"], arm["obs"], {w: set(v) for w, v in inl.items()})
        rep["inlier_split"] = split_rows(sm)
    return rep


def clean(x):
    from tools.p8_placecause import clean as c
    return c(x)


def write_record(stage: str, out: dict) -> dict:
    from export.schemas import find_private
    res = clean(out)
    text = json.dumps(res, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data: {find_private(text)[:3]}")
    atomic_write(RECORDS[stage], text)
    return res


def cmd_report_s0() -> dict:
    from tools import p8_rebaseline as RB
    d = footfix_dir() / "s0"
    run = RB.read_json(d / "run.json")
    for k, sha in run["files"].items():
        name = {"audit": "audit.json"}.get(k) or (f"fitrows__{k.split('__')[1]}.json" if k.startswith("fitrows") else
                                                  f"inliers__{k.split('__')[1]}.json" if k.startswith("inliers") else
                                                  f"score__{k}.json")
        if RB.sha256(d / name) != sha:
            raise SystemExit(f"{WHO}: {name} differs from the run's record")
    games = {}
    for g in POOL:
        inl = RB.read_json(d / f"inliers__{g}.json")
        games[g] = {form: arm_report(d, f"score__identity__{g}.json", f"score__{form}_in__{g}.json", inl)
                    for form in FORMS}
        for form in FORMS:
            games[g][form]["g0"] = g0_game(games[g][form]["changes"])
    passes = all(games[g][CANDIDATE]["g0"]["passes"] for g in POOL)
    in_frame = {g: sum(v["in_frame"] for v in RB.read_json(d / f"score__identity__{g}.json")["recs"].values())
                for g in POOL}
    readings = audit_readings(run["audit"], in_frame, {g: run["games"][g]["label_boxes"] for g in POOL})
    out = {"what": "docs/research/P8_footfix.md version 1.2, stage 0 (smoke, reproduction, 0c audit, F1 and F2 in "
                   "sample); G0 reads F1 only; F2 is the predicted-null control", "run_id": run["run_id"],
           "git": run["git"], "versions": run.get("versions"), "seconds": run["seconds"],
           "smoke_and_reproduction": {g: run["games"][g] for g in POOL}, "audit": run["audit"], "in_sample": games,
           "g0": {"verdict": "PASS" if passes else "FAIL", "action": "stage 1 may be asked for" if passes else "KILL",
                  "per_game": {g: games[g][CANDIDATE]["g0"] for g in POOL}},
           "f2_null_in_sample": {g: f2_null(games[g][CONTROL]["table"]) for g in POOL},
           "audit_readings": readings}
    if passes and readings["closes_toward_T_line"]:
        out["g0"]["action"] = "closed by reading A (absorbed), whatever G0 says"
    res = write_record("s0", out)
    print(json.dumps(res["g0"], indent=1))
    return res


def cmd_report_s1() -> dict:
    from tools import p8_rebaseline as RB
    d = footfix_dir() / "s1"
    run = RB.read_json(d / "run.json")
    for k, sha in run["files"].items():
        if RB.sha256(d / f"score__{k}.json") != sha:
            raise SystemExit(f"{WHO}: score__{k}.json differs from the run's record")
    s0_rec = RB.read_json(RECORDS["s0"])
    if s0_rec.get("run_id") != run["s0_run_id"]:
        raise SystemExit(f"{WHO}: stage 0's record is not the run stage 1 started from")
    games, per = {}, {}
    for g in POOL:
        inl = RB.read_json(footfix_dir() / "s0" / f"inliers__{g}.json")
        games[g] = {form: arm_report(d, f"score__identity__{g}.json", f"score__{form}_logo__{g}.json", inl)
                    for form in FORMS}
        for form in FORMS:
            games[g][form]["rule"] = game_rule(games[g][form]["changes"])
        f1 = games[g][CANDIDATE]
        per[g] = {"rule": f1["rule"], "p50": f1["changes"]["position_p50_ft"]["change"], "split": f1["inlier_split"]}
    out = {"what": "docs/research/P8_footfix.md version 1.2, stage 1 (leave one game out); the rule and readings read "
                   "F1 only; F2 is the predicted-null control", "run_id": run["run_id"], "s0_run_id": run["s0_run_id"],
           "git": run["git"], "versions": run.get("versions"), "seconds": run["seconds"], "tables": run["tables"], "per_game": games,
           "verdict": reading(per, tuple(s0_rec["audit_readings"]["F_split_descriptive_on"])),
           "f2_null_fold_tables": {g: f2_null(games[g][CONTROL]["table"]) for g in POOL}}
    res = write_record("s1", out)
    print(json.dumps(res["verdict"], indent=1))
    return res


def cmd_table() -> dict:
    from tools import p8_rebaseline as RB
    s1 = RB.read_json(RECORDS["s1"])
    if s1.get("verdict", {}).get("verdict") != "PASS" or not s1["verdict"].get("stage_2"):
        raise SystemExit(f"{WHO}: the pooled table is written only after stage 1 reads PASS and 'feet nearer'")
    run0 = RB.read_json(footfix_dir() / "s0" / "run.json")
    rows = []
    for g in POOL:
        p = footfix_dir() / "s0" / f"fitrows__{g}.json"
        if RB.sha256(p) != run0["files"][f"fitrows__{g}"]:
            raise SystemExit(f"{WHO}: {g}'s fit rows differ from stage 0's record")
        rows += RB.read_json(p)
    tabs = fit_tables(rows)
    out = {"what": "F1 fitted on gsw_cha_2016 and gsw_bkn_2015 pooled (docs/research/P8_footfix.md, stage 2)",
           "s0_run_id": run0["run_id"], "s1_run_id": s1["run_id"], "table": tabs["F1"], "counts": tabs["counts"]["F1"]}
    atomic_write(TABLE_FILE, json.dumps(clean(out), indent=1))
    print(json.dumps(out, indent=1))
    return out


def cmd_report_hou() -> dict:
    from tools import p8_rebaseline as RB
    d = footfix_dir() / "hou"
    run = RB.read_json(d / "run.json")
    for k, sha in run["files"].items():
        if RB.sha256(d / f"score__{k}.json") != sha:
            raise SystemExit(f"{WHO}: score__{k}.json differs from the run's record")
    rep = arm_report(d, "score__identity.json", "score__F1.json", None)
    rule = game_rule(rep["changes"])
    team = rep["changes"].pop("team")
    out = {"what": "docs/research/P8_footfix.md version 1.2, stage 2: hou_cle_2016 scored once with the committed "
                   "pooled F1 table; team reported only as rule 6 needs it (D203)", "run_id": run["run_id"],
           "go": run["go"], "table_sha256": run["table_sha256"], "git": run["git"],
           "versions": run.get("versions"), "seconds": run["seconds"],
           "changes": {k: v for k, v in rep["changes"].items()}, "team_change_for_rule_6": team,
           "rule": rule, "verdict": "PASS" if rule["passes"] else "FAIL",
           "limit": "a FAIL cannot separate the correction from hou_cle_2016's single-pool truth fit"}
    res = write_record("hou", out)
    print(json.dumps({"verdict": res["verdict"], "rule": res["rule"]}, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("stage", choices=("s0", "s1", "hou"))
    s.add_argument("--go", default="")
    r = sub.add_parser("report")
    r.add_argument("stage", choices=("s0", "s1", "hou"))
    sub.add_parser("table")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        {"s0": cmd_score_s0, "s1": cmd_score_s1, "hou": lambda: cmd_score_hou(a.go)}[a.stage]()
    elif a.cmd == "report":
        {"s0": cmd_report_s0, "s1": cmd_report_s1, "hou": cmd_report_hou}[a.stage]()
    else:
        cmd_table()


if __name__ == "__main__":
    sys.exit(main())
