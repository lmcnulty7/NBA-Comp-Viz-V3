"""tools/p8_outpos.py: K1 applied to the output court positions only, scored by a report-only scorer variant
(docs/research/P8_outpos.md). Nothing is built, trained or changed; no sidecar is rewritten; the pinned scorer and its
rows are unchanged and adopt nothing.

    python3 -m tools.p8_outpos pool        pool rows, leave one game out (development evidence) and the frozen table:
                                           reports/research/p8_outpos_pool.json, reports/research/p8_outpos_table.json
    python3 -m tools.p8_outpos score-hou   hou_cle_2016 scored once with the frozen, committed table:
                                           reports/research/p8_outpos_hou.json

The arm: the pipeline's output court position of a box is B(foot_stab), B the r23 court model's per-frame H. The arm
moves that output point by K1 (P8_footbias.md, fit rule unchanged: the median depth offset of the box's cell, box
pixel height by third of the output point under B's near side, removed along the unit vector from the build camera's
ground point G_B = sportvu.camera.pose_from_h(B) to the output point). foot_stab is not touched, so the truth-frame
feet, the matching, missed and ghosts are identical to the baseline by construction; the code asserts it per window.

The report-only scorer variant. The pinned lines (tools/p8_scorecard.py score_window):

    feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2)
    pairs = _match(ct, sv, B.MATCH_FT) if len(ct) else []
    if B.accepted(r) and r.get("H") is not None:
        Hp = np.array(r["H"], np.float64).reshape(3, 3)
        cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hp).reshape(-1, 2)
        for bi, j, _ in pairs:
            e = cp_[bi] - sv[j]
            err_of[j] = (float(np.hypot(*e)), float(e[1] if nh else -e[1]))

The variant runs the same lines and, between cp_ and e, replaces cp_[bi] by its K1-shifted point (no shift when B has
no camera pose). Everything else (ct, pairs, nh, region = third_of(sv[j][1], nh), the rows of tools/p8_e1_score.py)
is unchanged. The pinned rows (baseline) and the variant rows are reported side by side.

Rules, fixed before any number:
- Pool rows (development evidence): for each pool game, K1 fitted on the other two pool games' offset populations
  (tools.p8_footbias.offset_rows and fit, unchanged), scored by the variant on the held-out game against the pinned
  rows. Paired by tools.p8_e1_score.rows, diffs and boot (2,000 resamples, seed 0, windows of the game).
- Frozen table: K1 fitted on all three pool games, written to reports/research/p8_outpos_table.json by `pool`, which
  reads no hou_cle_2016 file. `score-hou` refuses to run unless that file is tracked by git with no uncommitted change,
  and records its sha256 and the commit that last changed it.
- hou_cle_2016 is scored once: `score-hou` refuses if its record already exists.
- Checks in code: per window, the variant's missed, in-frame, non-referee ghost and box counts and its matched keys
  equal the baseline's; the baseline's matched entries equal the registered score file's CPU side to 1e-4 (pool and
  hou_cle_2016); load_frames' count check against the registered score file.
- Decision rule for hou_cle_2016 (variant minus pinned, window intervals over its windows):
  PASS when (1) the position p50 difference is below zero with its window interval wholly below zero; (2) the
  near-field bias difference is below zero with its window interval wholly below zero; (3) the position p90 difference
  has a window interval upper end of at most +0.10 ft; (4) missed and ghosts identical (asserted, the tool stops
  otherwise). Anything else is FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.p8_footbias import fit, load_frames, offset_rows, score_frame, scorer_court
from tools.p8_placement import CHECK, POOL, check_games, depth_basis, hbin, rnd

WHO = "tools.p8_outpos"
P90_MARGIN_FT = 0.10
TABLE = REPO / "reports" / "research" / "p8_outpos_table.json"
HOU_RECORD = REPO / "reports" / "research" / "p8_outpos_hou.json"
POOL_RECORD = REPO / "reports" / "research" / "p8_outpos_pool.json"
POSITION_ROWS = ("position_p50_ft", "position_p90_ft", "nearfield_bias_ft")


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def shift_point(q, ground, cell, table: dict) -> np.ndarray:
    """The K1-shifted output point: q minus table[cell] along the unit vector from the camera's ground point to q."""
    q = np.asarray(q, float)
    if ground is None:
        return q
    u, _ = depth_basis(ground, q)
    return q - table.get(cell, 0.0) * u


def table_to_json(table: dict) -> dict:
    return {f"{a}|{b}": float(v) for (a, b), v in sorted(table.items())}


def table_from_json(d: dict) -> dict:
    return {tuple(k.split("|")): float(v) for k, v in d.items()}


def decide(diff: dict, iv: dict) -> dict:
    def below(row):
        return diff.get(row) is not None and iv.get(row) is not None and diff[row] < 0 and iv[row][1] < 0
    c1, c2 = below("position_p50_ft"), below("nearfield_bias_ft")
    c3 = iv.get("position_p90_ft") is not None and iv["position_p90_ft"][1] <= P90_MARGIN_FT
    return {"p50_below_zero": bool(c1), "nearfield_bias_below_zero": bool(c2), "p90_within_margin": bool(c3),
            "detection_identical": True, "verdict": "PASS" if (c1 and c2 and c3) else "FAIL"}


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────────────────────
def frame_matched(fr: dict, table: dict) -> tuple:
    """(pinned matched, variant matched) of one frame, from one set of pairs."""
    import cv2
    from sportvu.bench import MATCH_FT
    from sportvu.eval import near_side
    from sportvu.truth import _match
    from tools.p8_scorecard import third_of
    feet = np.asarray(fr["px"], np.float32).reshape(-1, 2)
    if not len(feet) or fr["Hb"] is None:
        return {}, {}
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Ht"]).reshape(-1, 2)
    sv = fr["sv"]
    pairs = _match(ct, sv, MATCH_FT)
    cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Hb"]).reshape(-1, 2).astype(np.float64)
    nh = near_side(fr["Ht"])
    pinned, variant = {}, {}
    for bi, j, _ in pairs:
        key = f"{fr['f']}:{j}"
        region = third_of(sv[j][1], nh)
        for out, point in ((pinned, cp_[bi]),
                           (variant, shift_point(cp_[bi], fr["Gb"],
                                                 (hbin(fr["bh"][bi]), third_of(cp_[bi][1], fr["nhb"]))
                                                 if fr["Gb"] is not None else None, table))):
            e = point - sv[j]
            out[key] = [float(np.hypot(*e)), float(e[1] if nh else -e[1]), region]
    return pinned, variant


def window_records(game: dict, table: dict) -> dict:
    wins = {}
    for fr in game["frames"]:
        w = wins.setdefault(fr["w"], {s: {"missed": 0, "in_frame": 0, "nr_ghosts": 0, "nr_boxes": 0, "cov_acc": 1.0,
                                          "cov_live": 1.0, "team": {"n": 0, "majority": 0, "fixed": 0}, "matched": {}}
                                      for s in ("r23", "new")})
        c = score_frame(scorer_court(fr["Ht"], fr["px"]), fr["sv"], fr["inside"], fr["is_ref"])
        pinned, variant = frame_matched(fr, table)
        for side, m in (("r23", pinned), ("new", variant)):
            for k in ("missed", "in_frame", "nr_ghosts", "nr_boxes"):
                w[side][k] += c[k]
            w[side]["matched"].update(m)
    for name, w in wins.items():
        if any(w["r23"][k] != w["new"][k] for k in ("missed", "in_frame", "nr_ghosts", "nr_boxes")) or \
                w["r23"]["matched"].keys() != w["new"]["matched"].keys():
            raise SystemExit(f"{WHO}: {name} detection rows differ between pinned and variant; they must be identical")
    return wins


def check_registered(tag: str, wins: dict) -> None:
    from tools.p8_footpos import same_matched
    from tools.p8_placement import SCORES, SIDE
    sf = json.loads((p8_root() / "e1" / SCORES[tag]).read_text())
    for w, rec in wins.items():
        if not same_matched(rec["r23"]["matched"], sf["windows"][w][SIDE[tag]]["matched"]):
            raise SystemExit(f"{WHO}: {w} pinned matched entries differ from the registered score file")


def score_game(game: dict, table: dict) -> dict:
    from tools.p8_e1_score import boot, diffs, rows
    wins = window_records(game, table)
    check_registered(game["tag"], wins)
    r, d = rows(list(wins.values())), diffs(list(wins.values()))
    iv = boot({w: [v] for w, v in wins.items()})
    keep = POSITION_ROWS + ("missed_rate", "ghost_rate")
    return {"pinned_variant": {k: r[k] for k in keep}, "diff": {k: d[k] for k in keep},
            "iv": {k: iv[k] for k in keep}, "matched": sum(len(v["r23"]["matched"]) for v in wins.values()),
            "windows": len(wins)}


def cmd_pool(args) -> dict:
    check_games(POOL)
    games = {t: load_frames(t) for t in POOL}               # pool only: no hou_cle_2016 file is read here
    offs = {t: offset_rows(games[t])[0] for t in POOL}
    out = {"games": {}}
    for held in POOL:
        table = fit([r for g in POOL if g != held for r in offs[g]])["K1"]
        out["games"][held] = score_game(games[held], table)
        print(held, json.dumps(rnd({k: out["games"][held][k] for k in ("diff", "iv")})))
    frozen = fit([r for g in POOL for r in offs[g]])["K1"]
    counts = {}
    for g in POOL:
        for r in offs[g]:
            counts[f"{r['hb']}|{r['third']}"] = counts.get(f"{r['hb']}|{r['third']}", 0) + 1
    tab = {"what": "K1 table frozen from the 3 pool games (docs/research/P8_outpos.md); median depth offset (ft) per "
                   "cell box height|third under the r23 court's H; tools.p8_footbias.fit, unchanged",
           "games": list(POOL), "table": table_to_json(frozen), "fit_rows_per_cell": dict(sorted(counts.items()))}
    atomic_write(TABLE, json.dumps(rnd(tab), indent=1))
    out = rnd({"what": "K1 on output positions, pool rows leave one game out (development evidence; "
                       "docs/research/P8_outpos.md); pinned rows unchanged, variant report only", **out})
    atomic_write(POOL_RECORD, json.dumps(out, indent=1))
    return out


def table_provenance() -> dict:
    rel = str(TABLE.relative_to(REPO))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO).returncode == 0
    if not (TABLE.exists() and tracked and clean):
        raise SystemExit(f"{WHO}: {rel} must exist, be tracked and have no uncommitted change before hou_cle_2016 "
                         f"is read")
    commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    return {"path": rel, "sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(), "commit": commit}


def cmd_score_hou(args) -> dict:
    if HOU_RECORD.exists():
        raise SystemExit(f"{WHO}: hou_cle_2016 has been scored already ({HOU_RECORD.name}); it is scored once")
    prov = table_provenance()
    table = table_from_json(json.loads(TABLE.read_text())["table"])
    check_games(CHECK)
    game = load_frames(CHECK[0])
    res = score_game(game, table)
    out = rnd({"what": "K1 on output positions, hou_cle_2016 scored once with the frozen pool table "
                       "(docs/research/P8_outpos.md); report-only scorer variant beside the pinned rows",
               "table": prov, **res, "decision": decide(res["diff"], res["iv"]),
               "rule": {"p90_margin_ft": P90_MARGIN_FT, "boot": "tools.p8_e1_score.boot"}})
    atomic_write(HOU_RECORD, json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("diff", "iv", "decision")}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_outpos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool")
    sub.add_parser("score-hou")
    a = ap.parse_args(argv)
    {"pool": cmd_pool, "score-hou": cmd_score_hou}[a.cmd](a)


if __name__ == "__main__":
    main()
