"""tools/p8_outpos.py: K1 applied to the output court positions only, scored by a report-only scorer variant
(docs/research/P8_outpos.md). Nothing is built, trained or changed; no sidecar is rewritten; the pinned scorer and its
rows are unchanged and adopt nothing.

    python3 -m tools.p8_outpos pool        pool rows, leave one game out (development evidence) and the frozen table:
                                           reports/research/p8_outpos_pool.json, reports/research/p8_outpos_table.json
    python3 -m tools.p8_outpos forms       (v2) pool only: K0 against K1 leave one game out, the form decision, the
                                           frozen K0 table and the hou_cle_2016 forecast from committed numbers:
                                           reports/research/p8_outpos_forms.json, reports/research/p8_outpos_table_k0.json
    python3 -m tools.p8_outpos forms3      (v3) pool only: K1, K0 (reference), K1m and K1t leave one game out under
                                           the v3 pool rule, the chosen form or the stop, frozen K1m and K1t tables and
                                           the forecast: reports/research/p8_outpos_forms3.json,
                                           reports/research/p8_outpos_table_k1m.json, reports/research/p8_outpos_table_k1t.json
    python3 -m tools.p8_outpos score-hou   (v3) hou_cle_2016 scored once, every form in the same run, with the frozen,
                                           committed, sha-pinned tables: reports/research/p8_outpos_hou.json, then the
                                           post-run diagnosis in reports/research/p8_outpos_hou_diagnosis.json

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
  otherwise). Anything else is FAIL. (Version 1; replaced by version 2 below before any hou_cle_2016 file was read.)

Version 2 (docs/research/P8_outpos.md, "Design version 2"), registered before its numbers:
- K0, the naive comparator: one global depth shift, the median depth offset of the fit games' whole offset
  population (tools.p8_footbias.offset_rows, no cells), removed along the same ray. Frozen from the 3 pool games into
  reports/research/p8_outpos_table_k0.json.
- Form decision (pool only, leave one game out, the same variant scorer): K1 is kept only if its position p50
  difference is below K0's on every pool game; otherwise the arm is K0. The chosen form is pinned in this file
  (CHOSEN_FORM) with the K0 table's sha before hou_cle_2016 is read.
- Forecast, from committed numbers only (reports/research/p8_footbias.json "cells": hou_cle_2016's per-cell median
  depth and counts; the frozen tables; the pool's leave-one-game-out rows): for a game g and form F with table t,
  R_g(F) = sum over cells of n * (|m| - |m - t|) / sum of n (m the cell's median depth, n its count). Calibration per
  form: k_p50 = sum over pool games of the p50 difference / sum of R_g; k_p90 likewise. Forecast for hou_cle_2016:
  k_p50 * R_hou and k_p90 * R_hou. Near-field bias forecast in depth terms: per third (near, far) the count-weighted
  mean cell median before (|mean m|) and after (|mean (m - t)|); the forecast reads the larger third.
- hou_cle_2016 decision, for the chosen form: (1) position p50 difference at most -0.10 ft with its window interval
  wholly below zero; (2) position p90 window upper end at most +0.10 ft; (3) near-field bias after at most before
  (a guard), with the third it reads reported before and after; (4) missed and ghosts identical (asserted). PASS when
  all hold. K0 and K1 are both scored in the single run; "the cells matter" needs the chosen form to beat the other
  on hou_cle_2016 p50 (development evidence).
- Guards: a once-only marker under the P8 workspace (placement/outpos_hou.started) is written before any hou_cle_2016
  file is read, and score-hou refuses if it or the record exists; both tables' sha256 are pinned here and checked;
  git HEAD is recorded; score-hou refuses if this file, tools/p8_footbias.py, tools/p8_placement.py,
  tools/p8_scorecard.py or tools/p8_e1_score.py has an uncommitted change.
- Report only: position p90 by third and by box-height bin; per game, the frames where near_side(B) differs from
  near_side(truth); after the decision, K1 fitted on hou_cle_2016 in sample as an upper bound and the transfer ratio
  (post-run diagnosis).
  (Version 2's hou_cle_2016 run was never made: the lead counted near-third p90 growth as harm. Its score-hou refuses.)

Version 3 (docs/research/P8_outpos.md, "Design version 3"), registered before its numbers:
- Forms, named now. Reference only, not eligible: K1 (version 1) and K0 (version 2). Candidates: K1m, K1 with the
  near third unshifted (middle and far cells as K1); K1t, K1's middle and far cells plus one near shift s fitted on the
  scored row's own quantity: on the fit games' matched players (truth pairs, error through B) whose output point lies
  in the near third under B, s is the value on the grid -1.50 to +0.50 ft by 0.01 that brings their median
  toward_camera, after the shift along the ray, closest to zero (ties to the smaller |s|).
- Pool rule, every pool game held out, for each candidate: position p50 difference at most -0.10 ft with its window
  interval wholly below zero; position p90 (whole game) window upper end at most +0.10 ft; near-third p90 difference
  (players whose SportVU third is near, paired) window upper end at most +0.05 ft (window bootstrap, 2,000, seed 0);
  near-field bias after at most before; detection identical (asserted). Among the candidates that pass, the arm is
  the one with the lowest mean p50 difference over the pool games; if none passes, the arm stops and hou_cle_2016 is
  kept unspent. K1 and K0 are scored under the same rule for reference.
- Forecast: k_p50 and k_p90 per form calibrated leave one game out on the pool (prediction errors reported); the
  hou_cle_2016 forecast uses k from all three pool games. The forecast holds if the chosen form's hou_cle_2016 p50
  difference lies within plus or minus 0.05 ft of it. The near-field bias forecast in depth terms is uncalibrated and
  is reported beside the pool's scored near bias.
- hou_cle_2016 rule for the chosen form: the pool rule applied to hou_cle_2016 alone.
- Guards: score-hou writes the decision record before computing the in-sample diagnosis, which goes to its own file;
  it refuses unless `git status --porcelain -- tools sportvu` is empty; it records the P8 workspace as "default" and
  refuses unless p8_root() is the default workspace (V3_P8_ROOT unset or equal to it); the forms3 tables and the
  chosen form are pinned in this file after the pool stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

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
TABLE_K0 = REPO / "reports" / "research" / "p8_outpos_table_k0.json"
FORMS_RECORD = REPO / "reports" / "research" / "p8_outpos_forms.json"
FOOTBIAS_RECORD = REPO / "reports" / "research" / "p8_footbias.json"
K1_SHA = "c420b0a642851104894aec03189cce3404c407703347b3fff29dfc96b7cb80ce"     # the table at 18df8e7f
K0_SHA = "21bc22d960410a5b1723bc20133cfd781ee6691ec3cf2be4a2914064fa7f47ba"     # the K0 table frozen by `forms`
CHOSEN_FORM = "K1"        # the form rule's result in reports/research/p8_outpos_forms.json
P50_BAR_FT = -0.10
GUARDED_FILES = ("tools/p8_outpos.py", "tools/p8_footbias.py", "tools/p8_placement.py", "tools/p8_scorecard.py",
                 "tools/p8_e1_score.py")
CELLS = tuple((h, t) for h in ("<150", "150-200", "200-250", ">=250") for t in ("near", "middle", "far"))
TABLE_K1M = REPO / "reports" / "research" / "p8_outpos_table_k1m.json"
TABLE_K1T = REPO / "reports" / "research" / "p8_outpos_table_k1t.json"
FORMS3_RECORD = REPO / "reports" / "research" / "p8_outpos_forms3.json"
HOU_DIAG = REPO / "reports" / "research" / "p8_outpos_hou_diagnosis.json"
CANDIDATES = ("K1m", "K1t")
REFERENCE = ("K1", "K0")
NEAR_P90_MARGIN_FT = 0.05
FORECAST_TOL_FT = 0.05
S_GRID = tuple(round(-1.50 + 0.01 * i, 2) for i in range(201))
K1M_SHA = "abb7a1097744995fb03240bf39a8a687c6090e8f80b1b517bb8782d34559d8ca"     # v3: the K1m table frozen by forms3
K1T_SHA = "5d190844b3d164b63a2c4969f196e460ebe10749861d21053bcd8d28aaad4b60"     # v3: the K1t table frozen by forms3
CHOSEN_FORM_V3 = "STOP"   # v3: forms3 found no candidate passing the pool rule; score-hou refuses


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
    """Version 1, replaced by cmd_score_hou_v2 before any hou_cle_2016 file was read; it refuses to run."""
    raise SystemExit(f"{WHO}: version 1 score-hou is replaced by version 2")
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


# ── version 2 ──────────────────────────────────────────────────────────────────────────────────────────────────────
def marker_path():
    return p8_root() / "placement" / "outpos_hou.started"


def k0_table(rows: list) -> dict:
    """K0: one global median depth for every cell."""
    g = float(np.median([r["depth"] for r in rows])) if rows else 0.0
    return {c: g for c in CELLS}


def form_decision(diffs_by_form: dict) -> str:
    """{form: {game: p50 diff}} -> "K1" when K1's p50 difference is below K0's on every pool game, else "K0"."""
    k1, k0 = diffs_by_form["K1"], diffs_by_form["K0"]
    return "K1" if all(k1[g] < k0[g] for g in POOL) else "K0"


def removed_bias(counts: dict, medians: dict, table: dict) -> float | None:
    """R = sum n * (|m| - |m - t|) / sum n over cells with a median (cells keyed "hb|third")."""
    num = den = 0.0
    for c, n in counts.items():
        m = medians.get(c)
        if m is None or not n:
            continue
        t = table.get(tuple(c.split("|")), 0.0)
        num += n * (abs(m) - abs(m - t))
        den += n
    return num / den if den else None


def bias_forecast(counts: dict, medians: dict, table: dict) -> dict:
    out = {}
    for third in ("near", "far"):
        cs = [c for c in counts if c.endswith("|" + third) and medians.get(c) is not None and counts[c]]
        n = sum(counts[c] for c in cs)
        if not n:
            continue
        before = abs(sum(counts[c] * medians[c] for c in cs) / n)
        after = abs(sum(counts[c] * (medians[c] - table.get(tuple(c.split("|")), 0.0)) for c in cs) / n)
        out[third] = {"before": before, "after": after}
    if not out:
        return {}
    rb = max(out, key=lambda k: out[k]["before"])
    ra = max(out, key=lambda k: out[k]["after"])
    return {"by_third": out, "before": out[rb]["before"], "third_before": rb, "after": out[ra]["after"],
            "third_after": ra}


def decide_v2(diff: dict, iv: dict, bias: dict) -> dict:
    c1 = diff.get("position_p50_ft") is not None and iv.get("position_p50_ft") is not None and \
        diff["position_p50_ft"] <= P50_BAR_FT and iv["position_p50_ft"][1] < 0
    c2 = iv.get("position_p90_ft") is not None and iv["position_p90_ft"][1] <= P90_MARGIN_FT
    c3 = bias.get("after") is not None and bias.get("before") is not None and bias["after"] <= bias["before"]
    return {"p50_at_most_bar_and_below_zero": bool(c1), "p90_within_margin": bool(c2), "bias_not_worse": bool(c3),
            "detection_identical": True, "verdict": "PASS" if (c1 and c2 and c3) else "FAIL"}


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen(path, pinned: str | None) -> str:
    if pinned is None:
        raise SystemExit(f"{WHO}: no sha is pinned for {Path(path).name}; pin it before hou_cle_2016 is read")
    got = sha256_of(path)
    if got != pinned:
        raise SystemExit(f"{WHO}: {Path(path).name} sha256 {got} differs from the pinned {pinned}")
    return got


def check_once(record, marker) -> None:
    for p in (record, marker):
        if Path(p).exists():
            raise SystemExit(f"{WHO}: {Path(p).name} exists; hou_cle_2016 is scored once")


def check_clean(files=GUARDED_FILES, repo=None) -> str:
    repo = repo or REPO
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *files], cwd=repo).returncode != 0
    if dirty:
        raise SystemExit(f"{WHO}: one of {list(files)} has an uncommitted change")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()


def tracked_clean(path, repo=None) -> str:
    """The commit that last changed a tracked, clean file; SystemExit otherwise."""
    repo = repo or REPO
    rel = str(Path(path).relative_to(repo))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=repo).returncode == 0
    if not (Path(path).exists() and tracked and clean):
        raise SystemExit(f"{WHO}: {rel} must exist, be tracked and have no uncommitted change")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=repo, capture_output=True,
                          text=True).stdout.strip()


def frame_matched_forms(fr: dict, tables: dict) -> tuple:
    """(pinned matched, {form: variant matched}, {key: box height bin}) of one frame from one set of pairs."""
    import cv2
    from sportvu.bench import MATCH_FT
    from sportvu.eval import near_side
    from sportvu.truth import _match
    from tools.p8_scorecard import third_of
    feet = np.asarray(fr["px"], np.float32).reshape(-1, 2)
    if not len(feet) or fr["Hb"] is None:
        return {}, {f: {} for f in tables}, {}
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Ht"]).reshape(-1, 2)
    sv = fr["sv"]
    pairs = _match(ct, sv, MATCH_FT)
    cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Hb"]).reshape(-1, 2).astype(np.float64)
    nh = near_side(fr["Ht"])
    pinned, variants, hb = {}, {f: {} for f in tables}, {}
    for bi, j, _ in pairs:
        key = f"{fr['f']}:{j}"
        region = third_of(sv[j][1], nh)
        hb[key] = hbin(fr["bh"][bi])
        e = cp_[bi] - sv[j]
        pinned[key] = [float(np.hypot(*e)), float(e[1] if nh else -e[1]), region]
        cell = (hb[key], third_of(cp_[bi][1], fr["nhb"])) if fr["Gb"] is not None else None
        for f, t in tables.items():
            e = shift_point(cp_[bi], fr["Gb"], cell, t) - sv[j]
            variants[f][key] = [float(np.hypot(*e)), float(e[1] if nh else -e[1]), region]
    return pinned, variants, hb


def score_forms(game: dict, tables: dict, keep_windows: dict | None = None) -> dict:
    """Every form against the pinned rows on one game, with the report-only breakdowns. keep_windows, when given,
    receives each form's window records."""
    from sportvu.eval import near_side
    from tools.p8_e1_score import boot, diffs, rows
    wins, hbs, ns_diff, ns_frames = {}, {}, 0, 0
    for fr in game["frames"]:
        c = score_frame(scorer_court(fr["Ht"], fr["px"]), fr["sv"], fr["inside"], fr["is_ref"])
        pinned, variants, hb = frame_matched_forms(fr, tables)
        hbs.update({f"{fr['w']}/{k}": v for k, v in hb.items()})
        if fr["Hb"] is not None:
            ns_frames += 1
            ns_diff += bool(near_side(fr["Hb"]) != near_side(fr["Ht"]))
        for f in tables:
            w = wins.setdefault(f, {}).setdefault(fr["w"], {s: {"missed": 0, "in_frame": 0, "nr_ghosts": 0,
                                                                "nr_boxes": 0, "cov_acc": 1.0, "cov_live": 1.0,
                                                                "team": {"n": 0, "majority": 0, "fixed": 0},
                                                                "matched": {}} for s in ("r23", "new")})
            for side, m in (("r23", pinned), ("new", variants[f])):
                for k in ("missed", "in_frame", "nr_ghosts", "nr_boxes"):
                    w[side][k] += c[k]
                w[side]["matched"].update(m)
    out = {"near_side_differs": {"frames_with_build_H": ns_frames, "differs": ns_diff}, "forms": {}}
    if keep_windows is not None:
        keep_windows.update(wins)
    keep = POSITION_ROWS + ("missed_rate", "ghost_rate")
    for f, ws in wins.items():
        for name, w in ws.items():
            if any(w["r23"][k] != w["new"][k] for k in ("missed", "in_frame", "nr_ghosts", "nr_boxes")) or \
                    w["r23"]["matched"].keys() != w["new"]["matched"].keys():
                raise SystemExit(f"{WHO}: {name} detection rows differ between pinned and {f}; they must be identical")
        check_registered(game["tag"], ws)
        r, d = rows(list(ws.values())), diffs(list(ws.values()))
        iv = boot({w: [v] for w, v in ws.items()})
        res = {"pinned_variant": {k: r[k] for k in keep}, "diff": {k: d[k] for k in keep},
               "iv": {k: iv[k] for k in keep}, "matched": sum(len(v["r23"]["matched"]) for v in ws.values()),
               "windows": len(ws)}
        # report only: near-field bias by third, p90 by third and by box-height bin
        tb, tn, eb, en, reg, hb_ = [], [], [], [], [], []
        for name, v in ws.items():
            for k in v["r23"]["matched"].keys() & v["new"]["matched"].keys():
                a, b = v["r23"]["matched"][k], v["new"]["matched"][k]
                eb.append(a[0]); en.append(b[0]); tb.append(a[1]); tn.append(b[1]); reg.append(a[2])
                hb_.append(hbs[f"{name}/{k}"])
        eb, en, tb, tn, reg, hb_ = map(np.array, (eb, en, tb, tn, reg, hb_))
        res["bias_by_third"] = {t: {"before": abs(float(np.median(tb[reg == t]))),
                                    "after": abs(float(np.median(tn[reg == t])))}
                                for t in ("near", "middle", "far") if (reg == t).any()}
        nf = {t: v for t, v in res["bias_by_third"].items() if t in ("near", "far")}
        if nf:
            rb = max(nf, key=lambda t: nf[t]["before"]); ra = max(nf, key=lambda t: nf[t]["after"])
            res["bias_reads"] = {"before": nf[rb]["before"], "third_before": rb, "after": nf[ra]["after"],
                                 "third_after": ra}
        res["p90_by_third"] = {t: [float(np.percentile(eb[reg == t], 90)), float(np.percentile(en[reg == t], 90))]
                               for t in ("near", "middle", "far") if (reg == t).any()}
        res["p90_by_height"] = {h: [float(np.percentile(eb[hb_ == h], 90)), float(np.percentile(en[hb_ == h], 90))]
                                for h in ("<150", "150-200", "200-250", ">=250") if (hb_ == h).any()}
        out["forms"][f] = res
    return out


def cmd_forms(args) -> dict:
    check_games(POOL)
    games = {t: load_frames(t) for t in POOL}               # pool only: no hou_cle_2016 file is read here
    offs = {t: offset_rows(games[t])[0] for t in POOL}
    res = {}
    for held in POOL:
        fit_rows = [r for g in POOL if g != held for r in offs[g]]
        tables = {"K1": fit(fit_rows)["K1"], "K0": k0_table(fit_rows)}
        res[held] = score_forms(games[held], tables)
        res[held]["tables"] = {f: table_to_json(t) for f, t in tables.items()}
        print(held, json.dumps(rnd({f: v["diff"]["position_p50_ft"] for f, v in res[held]["forms"].items()})))
    p50 = {f: {g: res[g]["forms"][f]["diff"]["position_p50_ft"] for g in POOL} for f in ("K0", "K1")}
    chosen = form_decision(p50)
    all_rows = [r for g in POOL for r in offs[g]]
    k0 = k0_table(all_rows)
    atomic_write(TABLE_K0, json.dumps(rnd({"what": "K0 table frozen from the 3 pool games (docs/research/P8_outpos.md "
                                                   "version 2): one global median depth offset (ft) under the r23 "
                                                   "court's H, the same value in every cell",
                                           "games": list(POOL), "global": float(np.median([r["depth"] for r in all_rows])),
                                           "fit_rows": len(all_rows), "table": table_to_json(k0)}), indent=1))
    # forecast from committed numbers only
    fb = json.loads(FOOTBIAS_RECORD.read_text())["cells"]
    k1_frozen = table_from_json(json.loads(TABLE.read_text())["table"])
    frozen = {"K1": k1_frozen, "K0": k0}
    fc = {}
    for f in ("K0", "K1"):
        Rg = {g: removed_bias({c: v.get(g) for c, v in fb["count"].items()},
                              {c: v.get(g) for c, v in fb["median_depth"].items()},
                              table_from_json(res[g]["tables"][f])) for g in POOL}
        dp50 = {g: res[g]["forms"][f]["diff"]["position_p50_ft"] for g in POOL}
        dp90 = {g: res[g]["forms"][f]["diff"]["position_p90_ft"] for g in POOL}
        k50 = sum(dp50.values()) / sum(Rg.values())
        k90 = sum(dp90.values()) / sum(Rg.values())
        cnt_h = {c: v.get("hou_cle_2016") for c, v in fb["count"].items()}
        med_h = {c: v.get("hou_cle_2016") for c, v in fb["median_depth"].items()}
        Rh = removed_bias(cnt_h, med_h, frozen[f])
        fc[f] = {"R_pool": Rg, "k_p50": k50, "k_p90": k90, "R_hou": Rh,
                 "forecast_p50_diff": k50 * Rh, "forecast_p90_diff": k90 * Rh,
                 "forecast_passes_p50_bar": bool(k50 * Rh <= P50_BAR_FT),
                 "bias_forecast_depth_terms": bias_forecast(cnt_h, med_h, frozen[f]),
                 "pool_check": {g: {"predicted_p50": k50 * Rg[g], "observed_p50": dp50[g]} for g in POOL}}
    out = rnd({"what": "Outpos version 2, pool only: K0 against K1 leave one game out through the variant scorer, the "
                       "form decision, the frozen K0 table and the hou_cle_2016 forecast from committed numbers "
                       "(docs/research/P8_outpos.md); no hou_cle_2016 file read",
               "games": res, "p50_diff_by_form": p50, "chosen_form": chosen,
               "form_rule": "K1 kept only if its p50 difference is below K0's on every pool game",
               "k0_table_sha256": sha256_of(TABLE_K0), "k1_table_sha256": sha256_of(TABLE),
               "forecast": fc, "forecast_inputs": "reports/research/p8_footbias.json cells (hou_cle_2016 per-cell "
                                                  "median depth and counts), frozen tables, pool rows above"})
    atomic_write(FORMS_RECORD, json.dumps(out, indent=1))
    print(json.dumps({"p50": out["p50_diff_by_form"], "chosen": chosen,
                      "forecast": {f: {k: v[k] for k in ("forecast_p50_diff", "forecast_p90_diff",
                                                         "bias_forecast_depth_terms")} for f, v in out["forecast"].items()}},
                     indent=1))
    return out


def cmd_score_hou_v2(args) -> dict:
    """Version 2, never run: the lead counted near-third p90 growth as harm (version 3). It refuses to run."""
    raise SystemExit(f"{WHO}: version 2 score-hou is replaced by version 3")
    check_once(HOU_RECORD, marker_path())
    head = check_clean()
    prov = {"K1": {"sha256": check_frozen(TABLE, K1_SHA), "commit": tracked_clean(TABLE)},
            "K0": {"sha256": check_frozen(TABLE_K0, K0_SHA), "commit": tracked_clean(TABLE_K0)}}
    if CHOSEN_FORM not in ("K0", "K1"):
        raise SystemExit(f"{WHO}: CHOSEN_FORM is not pinned")
    forms_rec = json.loads(FORMS_RECORD.read_text())
    tracked_clean(FORMS_RECORD)
    if forms_rec["chosen_form"] != CHOSEN_FORM:
        raise SystemExit(f"{WHO}: the pinned form {CHOSEN_FORM} differs from the forms record")
    marker = marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(marker, json.dumps({"head": head, "chosen_form": CHOSEN_FORM}))
    tables = {"K1": table_from_json(json.loads(TABLE.read_text())["table"]),
              "K0": table_from_json(json.loads(TABLE_K0.read_text())["table"])}
    check_games(CHECK)
    game = load_frames(CHECK[0])
    res = score_forms(game, tables)
    ch, other = CHOSEN_FORM, ("K1" if CHOSEN_FORM == "K0" else "K0")
    r = res["forms"][ch]
    decision = decide_v2(r["diff"], r["iv"], r.get("bias_reads", {}))
    cells_matter = {"chosen_p50_diff": r["diff"]["position_p50_ft"],
                    "other_p50_diff": res["forms"][other]["diff"]["position_p50_ft"],
                    "chosen_beats_other": bool(r["diff"]["position_p50_ft"] < res["forms"][other]["diff"]["position_p50_ft"]),
                    "label": "development evidence"}
    # post-run diagnosis: K1 fitted on hou_cle_2016 in sample, an upper bound, and the transfer ratio
    ins = fit(offset_rows(game)[0])["K1"]
    ins_res = score_forms(game, {"K1_in_sample": ins})["forms"]["K1_in_sample"]
    diag = {"label": "post-run diagnosis, computed after the decision; in sample, an upper bound",
            "in_sample_p50_diff": ins_res["diff"]["position_p50_ft"],
            "in_sample_nearfield_bias_diff": ins_res["diff"]["nearfield_bias_ft"],
            "transfer_ratio_p50": (res["forms"]["K1"]["diff"]["position_p50_ft"] / ins_res["diff"]["position_p50_ft"]
                                   if ins_res["diff"]["position_p50_ft"] else None),
            "in_sample_table": table_to_json(ins)}
    out = rnd({"what": "Outpos version 2: hou_cle_2016 scored once, K0 and K1 in the same run, frozen sha-pinned "
                       "tables (docs/research/P8_outpos.md); report-only scorer variant beside the pinned rows",
               "head": head, "tables": prov, "chosen_form": ch, **res, "decision": decision,
               "cells_matter": cells_matter, "post_run_diagnosis": diag,
               "rule": {"p50_bar_ft": P50_BAR_FT, "p90_margin_ft": P90_MARGIN_FT, "boot": "tools.p8_e1_score.boot"}})
    atomic_write(HOU_RECORD, json.dumps(out, indent=1))
    print(json.dumps({"decision": decision, "cells_matter": cells_matter}, indent=1))
    return out


# ── version 3 ──────────────────────────────────────────────────────────────────────────────────────────────────────
def k1m_table(k1: dict) -> dict:
    """K1 with the near third unshifted."""
    return {c: (0.0 if c[1] == "near" else v) for c, v in k1.items()}


def k1t_table(k1: dict, s_near: float) -> dict:
    """K1's middle and far cells plus one near shift for every near cell."""
    out = {c: v for c, v in k1.items() if c[1] != "near"}
    out.update({c: s_near for c in CELLS if c[1] == "near"})
    return out


def fit_near_shift(a, b) -> float:
    """The grid value s whose shifted toward_camera a - s * b has the median closest to zero (ties to smaller |s|)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return 0.0
    return min(S_GRID, key=lambda s: (round(abs(float(np.median(a - s * b))), 9), abs(s), s))


def near_toward_terms(game: dict) -> tuple:
    """(a, b) over matched players whose output point lies in the near third under B: toward_camera now (a), and its
    change per foot of shift along the ray (b), so that toward after a shift s is a - s * b."""
    import cv2
    from sportvu.bench import MATCH_FT
    from sportvu.eval import near_side
    from sportvu.truth import _match
    from tools.p8_scorecard import third_of
    A, Bv = [], []
    for fr in game["frames"]:
        feet = np.asarray(fr["px"], np.float32).reshape(-1, 2)
        if not len(feet) or fr["Hb"] is None or fr["Gb"] is None:
            continue
        ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Ht"]).reshape(-1, 2)
        pairs = _match(ct, fr["sv"], MATCH_FT)
        cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Hb"]).reshape(-1, 2).astype(np.float64)
        nh = near_side(fr["Ht"])
        for bi, j, _ in pairs:
            if third_of(cp_[bi][1], fr["nhb"]) != "near":
                continue
            e = cp_[bi] - fr["sv"][j]
            u, _ = depth_basis(fr["Gb"], cp_[bi])
            A.append(float(e[1] if nh else -e[1]))
            Bv.append(float(u[1] if nh else -u[1]))
    return np.array(A), np.array(Bv)


def near_p90_boot(ws: dict, n: int = 2000, seed: int = 0) -> dict:
    """Paired near-third (SportVU third) p90 difference, variant minus pinned, with its window bootstrap interval."""
    per = {}
    for name, v in ws.items():
        ks = v["r23"]["matched"].keys() & v["new"]["matched"].keys()
        a = [v["r23"]["matched"][k][0] for k in ks if v["r23"]["matched"][k][2] == "near"]
        b = [v["new"]["matched"][k][0] for k in ks if v["r23"]["matched"][k][2] == "near"]
        if a:
            per[name] = (np.array(a), np.array(b))
    keys = sorted(per)
    if not keys:
        return {"diff": None, "iv": None}

    def stat(ix):
        a = np.concatenate([per[keys[i]][0] for i in ix]); b = np.concatenate([per[keys[i]][1] for i in ix])
        return float(np.percentile(b, 90) - np.percentile(a, 90))

    d = stat(range(len(keys)))
    rng = np.random.default_rng(seed)
    reps = [stat(rng.integers(0, len(keys), len(keys))) for _ in range(n)]
    return {"diff": d, "iv": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}


def rule_v3(res: dict) -> dict:
    """One game's result of one form -> the pool rule's checks."""
    d, iv = res["diff"], res["iv"]
    c1 = d.get("position_p50_ft") is not None and iv.get("position_p50_ft") is not None and \
        d["position_p50_ft"] <= P50_BAR_FT and iv["position_p50_ft"][1] < 0
    c2 = iv.get("position_p90_ft") is not None and iv["position_p90_ft"][1] <= P90_MARGIN_FT
    np90 = res.get("near_p90") or {}
    c3 = np90.get("iv") is not None and np90["iv"][1] <= NEAR_P90_MARGIN_FT
    br = res.get("bias_reads") or {}
    c4 = br.get("after") is not None and br.get("before") is not None and br["after"] <= br["before"]
    return {"p50": bool(c1), "p90": bool(c2), "near_p90": bool(c3), "bias_guard": bool(c4),
            "detection_identical": True, "pass": bool(c1 and c2 and c3 and c4)}


def choose_v3(by_form: dict) -> str:
    """by_form: {form: {game: result with "rule" and "diff"}} -> the passing candidate with the lowest mean p50
    difference, or "STOP"."""
    ok = [f for f in CANDIDATES if f in by_form and all(r["rule"]["pass"] for r in by_form[f].values())]
    if not ok:
        return "STOP"
    return min(ok, key=lambda f: (np.mean([r["diff"]["position_p50_ft"] for r in by_form[f].values()]), f))


def default_root_check() -> str:
    import os
    from pathlib import Path as _P
    default = (_P.home() / "Developer" / "nba-v3-worktrees" / "_p8-data" / "work").resolve()
    env = os.environ.get("V3_P8_ROOT")
    if (env and _P(env).expanduser().resolve() != default) or p8_root().resolve() != default:
        raise SystemExit(f"{WHO}: the P8 workspace is not the default one; refusing")
    return "default (<home>/Developer/nba-v3-worktrees/_p8-data/work)"


def porcelain_clean(repo=None) -> None:
    out = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=repo or REPO,
                         capture_output=True, text=True).stdout.strip()
    if out:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted or untracked changes")


def cmd_forms3(args) -> dict:
    check_games(POOL)
    games = {t: load_frames(t) for t in POOL}               # pool only: no hou_cle_2016 file is read here
    offs = {t: offset_rows(games[t])[0] for t in POOL}
    near = {t: near_toward_terms(games[t]) for t in POOL}
    by_form = {f: {} for f in REFERENCE + CANDIDATES}
    extra = {}
    for held in POOL:
        fit_rows = [r for g in POOL if g != held for r in offs[g]]
        k1 = fit(fit_rows)["K1"]
        a = np.concatenate([near[g][0] for g in POOL if g != held])
        b = np.concatenate([near[g][1] for g in POOL if g != held])
        s_near = fit_near_shift(a, b)
        tables = {"K1": k1, "K0": k0_table(fit_rows), "K1m": k1m_table(k1), "K1t": k1t_table(k1, s_near)}
        res, wins = score_forms_with_windows(games[held], tables)
        extra[held] = {"near_side_differs": res["near_side_differs"], "s_near": s_near,
                       "tables": {f: table_to_json(t) for f, t in tables.items()}}
        for f, r in res["forms"].items():
            r["near_p90"] = near_p90_boot(wins[f])
            r["rule"] = rule_v3(r)
            by_form[f][held] = r
        print(held, json.dumps(rnd({f: {"p50": by_form[f][held]["diff"]["position_p50_ft"],
                                         "near_p90": by_form[f][held]["near_p90"], "rule": by_form[f][held]["rule"]["pass"]}
                                    for f in by_form})))
    chosen = choose_v3(by_form)
    # frozen tables from all three pool games
    k1_frozen = table_from_json(json.loads(TABLE.read_text())["table"])
    a = np.concatenate([near[g][0] for g in POOL]); b = np.concatenate([near[g][1] for g in POOL])
    s_all = fit_near_shift(a, b)
    frozen = {"K1": k1_frozen, "K0": table_from_json(json.loads(TABLE_K0.read_text())["table"]),
              "K1m": k1m_table(k1_frozen), "K1t": k1t_table(k1_frozen, s_all)}
    for f, path in (("K1m", TABLE_K1M), ("K1t", TABLE_K1T)):
        atomic_write(path, json.dumps(rnd({"what": f"{f} table frozen from the 3 pool games (docs/research/P8_outpos.md "
                                                   f"version 3)", "games": list(POOL),
                                           **({"s_near": s_all} if f == "K1t" else {}),
                                           "table": table_to_json(frozen[f])}), indent=1))
    # forecast with leave-one-game-out calibration of k
    fb = json.loads(FOOTBIAS_RECORD.read_text())["cells"]
    cnt = lambda g: {c: v.get(g) for c, v in fb["count"].items()}
    med = lambda g: {c: v.get(g) for c, v in fb["median_depth"].items()}
    fc = {}
    for f in REFERENCE + CANDIDATES:
        Rg = {g: removed_bias(cnt(g), med(g), table_from_json(extra[g]["tables"][f])) for g in POOL}
        dp50 = {g: by_form[f][g]["diff"]["position_p50_ft"] for g in POOL}
        dp90 = {g: by_form[f][g]["diff"]["position_p90_ft"] for g in POOL}
        loo = {}
        for g in POOL:
            o = [h for h in POOL if h != g]
            k = sum(dp50[h] for h in o) / sum(Rg[h] for h in o)
            loo[g] = {"k_p50_without_g": k, "predicted": k * Rg[g], "observed": dp50[g], "error": k * Rg[g] - dp50[g]}
        k50 = sum(dp50.values()) / sum(Rg.values())
        k90 = sum(dp90.values()) / sum(Rg.values())
        Rh = removed_bias(cnt("hou_cle_2016"), med("hou_cle_2016"), frozen[f])
        bias_pool = {g: {"depth_terms_forecast": bias_forecast(cnt(g), med(g), table_from_json(extra[g]["tables"][f])),
                         "scored_bias_by_third": by_form[f][g].get("bias_by_third")} for g in POOL}
        fc[f] = {"R_pool": Rg, "loo": loo, "k_p50": k50, "k_p90": k90, "R_hou": Rh,
                 "forecast_p50_diff": k50 * Rh, "forecast_p90_diff": k90 * Rh,
                 "forecast_tolerance_ft": FORECAST_TOL_FT,
                 "bias_forecast_depth_terms_uncalibrated": bias_forecast(cnt("hou_cle_2016"), med("hou_cle_2016"),
                                                                         frozen[f]),
                 "bias_pool_check": bias_pool}
    out = rnd({"what": "Outpos version 3, pool only (third pool look): K1 and K0 for reference, candidates K1m and K1t, "
                       "leave one game out under the v3 pool rule; frozen K1m and K1t tables; forecast "
                       "(docs/research/P8_outpos.md); no hou_cle_2016 file read",
               "games": {g: extra[g] for g in POOL}, "results": by_form, "chosen_form": chosen,
               "candidates": list(CANDIDATES), "reference": list(REFERENCE), "s_near_all_pool": s_all,
               "table_sha256": {"K1": sha256_of(TABLE), "K0": sha256_of(TABLE_K0), "K1m": sha256_of(TABLE_K1M),
                                "K1t": sha256_of(TABLE_K1T)},
               "forecast": fc})
    atomic_write(FORMS3_RECORD, json.dumps(out, indent=1))
    print(json.dumps({"chosen": chosen, "s_near_all_pool": out["s_near_all_pool"],
                      "forecast": {f: {k: v[k] for k in ("forecast_p50_diff", "forecast_p90_diff")}
                                   for f, v in out["forecast"].items()}}, indent=1))
    return out


def score_forms_with_windows(game: dict, tables: dict) -> tuple:
    """score_forms, also returning each form's window records (for the near-third p90 bootstrap)."""
    captured = {}
    res = score_forms(game, tables, keep_windows=captured)
    return res, captured


def cmd_score_hou_v3(args) -> dict:
    check_once(HOU_RECORD, marker_path())
    porcelain_clean()
    head = check_clean()
    root = default_root_check()
    if CHOSEN_FORM_V3 not in CANDIDATES:
        raise SystemExit(f"{WHO}: no v3 form is pinned (CHOSEN_FORM_V3 = {CHOSEN_FORM_V3}); hou_cle_2016 stays unspent")
    paths = {"K1": (TABLE, K1_SHA), "K0": (TABLE_K0, K0_SHA), "K1m": (TABLE_K1M, K1M_SHA), "K1t": (TABLE_K1T, K1T_SHA)}
    prov = {f: {"sha256": check_frozen(pth, sha), "commit": tracked_clean(pth)} for f, (pth, sha) in paths.items()}
    rec3 = json.loads(FORMS3_RECORD.read_text())
    tracked_clean(FORMS3_RECORD)
    if rec3["chosen_form"] != CHOSEN_FORM_V3:
        raise SystemExit(f"{WHO}: the pinned form differs from the forms3 record")
    marker = marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(marker, json.dumps({"head": head, "chosen_form": CHOSEN_FORM_V3, "version": 3}))
    tables = {f: table_from_json(json.loads(pth.read_text())["table"]) for f, (pth, _) in paths.items()}
    check_games(CHECK)
    game = load_frames(CHECK[0])
    res, wins = score_forms_with_windows(game, tables)
    for f, r in res["forms"].items():
        r["near_p90"] = near_p90_boot(wins[f])
        r["rule"] = rule_v3(r)
    ch = CHOSEN_FORM_V3
    fcast = rec3["forecast"][ch]["forecast_p50_diff"]
    got = res["forms"][ch]["diff"]["position_p50_ft"]
    out = rnd({"what": "Outpos version 3: hou_cle_2016 scored once, every form in the same run, frozen sha-pinned "
                       "tables (docs/research/P8_outpos.md); report-only scorer variant beside the pinned rows",
               "head": head, "p8_root": root, "tables": prov, "chosen_form": ch, **res,
               "decision": {"chosen": ch, **res["forms"][ch]["rule"], "verdict": "PASS" if res["forms"][ch]["rule"]["pass"]
                            else "FAIL"},
               "forecast_check": {"forecast_p50_diff": fcast, "observed_p50_diff": got,
                                  "holds": bool(abs(got - fcast) <= FORECAST_TOL_FT), "tolerance_ft": FORECAST_TOL_FT},
               "cells_matter": {"chosen_p50_diff": got, "K0_p50_diff": res["forms"]["K0"]["diff"]["position_p50_ft"],
                                "chosen_beats_K0": bool(got < res["forms"]["K0"]["diff"]["position_p50_ft"]),
                                "label": "development evidence"}})
    atomic_write(HOU_RECORD, json.dumps(out, indent=1))          # the decision is written before the diagnosis
    print(json.dumps({k: out[k] for k in ("decision", "forecast_check", "cells_matter")}, indent=1))
    ins = fit(offset_rows(game)[0])["K1"]
    ins_res = score_forms(game, {"K1_in_sample": ins})["forms"]["K1_in_sample"]
    diag = rnd({"label": "post-run diagnosis, computed after the decision record was written; in sample, an upper bound",
                "in_sample_p50_diff": ins_res["diff"]["position_p50_ft"],
                "in_sample_nearfield_bias_diff": ins_res["diff"]["nearfield_bias_ft"],
                "transfer_ratio_p50_K1": (res["forms"]["K1"]["diff"]["position_p50_ft"] /
                                          ins_res["diff"]["position_p50_ft"] if ins_res["diff"]["position_p50_ft"] else None),
                "in_sample_table": table_to_json(ins)})
    atomic_write(HOU_DIAG, json.dumps(diag, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_outpos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool")
    sub.add_parser("forms")
    sub.add_parser("forms3")
    sub.add_parser("score-hou")
    a = ap.parse_args(argv)
    {"pool": cmd_pool, "forms": cmd_forms, "forms3": cmd_forms3, "score-hou": cmd_score_hou_v3}[a.cmd](a)


if __name__ == "__main__":
    main()
