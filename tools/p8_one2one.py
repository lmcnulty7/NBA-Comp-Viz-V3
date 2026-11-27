"""tools/p8_one2one.py: a one-to-one view of the missed row (docs/research/P8_one2one.md, version 2). Report only.

    <frozen CV python> -m tools.p8_one2one collect GAME   (under the heavy-job lock, "p8 one2one GAME"; V3_P8_ROOT =
        work_ref) -> work_ref/one2one/GAME.json (the count gate; for gsw_cha_2016 and gsw_bkn_2015 the per-player
        classes; hou_cle_2016 the gate only)
    python3 -m tools.p8_one2one report                   (system python, no lock) -> reports/research/p8_one2one.json

The inputs and checks are tools/p8_placecause.py's reviewed collect step (hashes required, the cache complete, the
pinned score_window per window as D202 called it, the count gate), with missed_1to1 and unmatched_1to1 added to the gate
through the scorer's own functions. The main assignment is unmatched_1to1's rule in two stages (2.5 ft, then the
still-unmatched players and non-referee boxes at 5 ft); assign_1to1 is a pairs-returning copy of unmatched_1to1's lines,
checked equal to it at both gates on every scored frame, and assign_inframe a pairs-returning copy of
one_to_one_missed's, checked equal to it at 2.5 ft on every frame.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_one2one"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
MEASURED, GATE_ONLY = (CHA, BKN), (HOU,)
MISS_FT, NEAR_FT, FAR_REPORT_FT = 2.5, 5.0, 10.0
BOOT_N, SEED = 2000, 0
P_CLASSES = ("P-i", "P-ii", "P-iii-ref")
D_CLASSES = ("D-i", "D-iii-ref", "D-iii-free", "D-iii-none")
COUNTERS = ("placement_matched_in_stage_A", "P-iii-free", "detection_matched")


# ── pure: the assignment ───────────────────────────────────────────────────────────────────────────────────────────
def assign_1to1(feet, sv, is_ref, gate: float) -> dict:
    """unmatched_1to1's assignment with the pairs kept: non-referee box feet against every player, Hungarian on a cost
    matrix where pairs beyond the gate cost 1e6, matches within the gate -> {player index: (box index, distance)}."""
    from scipy.optimize import linear_sum_assignment
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    is_ref = np.asarray(is_ref, bool).reshape(-1)
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    keep = np.where(~is_ref)[0] if len(is_ref) else np.zeros(0, int)
    if not len(keep) or not len(sv):
        return {}
    F = feet[keep]
    D = np.sqrt(((sv[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    C = np.where(D <= gate, D, 1e6)
    ri, ci = linear_sum_assignment(C)
    ok = C[ri, ci] <= gate
    return {int(r): (int(keep[c]), float(D[r, c])) for r, c, o in zip(ri, ci, ok) if o}


def unmatched_count(pairs: dict, inside, is_ref) -> int:
    """unmatched_1to1's count from the pairs: unmatched in-frame players plus unmatched non-referee boxes."""
    n_nonref = int((~np.asarray(is_ref, bool)).sum())
    ins = np.where(np.asarray(inside, bool))[0]
    if n_nonref == 0:
        return int(len(ins))
    return int(sum(1 for j in ins if j not in pairs) + (n_nonref - len(pairs)))


def assign_inframe(feet, sv, inside, gate: float) -> dict:
    """one_to_one_missed's assignment with the pairs kept: in-frame players only, against every box foot (referees
    included), Hungarian on the gated cost matrix -> {player index: (box index, distance)}."""
    from scipy.optimize import linear_sum_assignment
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    ins = np.where(np.asarray(inside, bool))[0]
    if not len(ins) or not len(feet):
        return {}
    D = np.sqrt(((sv[ins][:, None, :] - feet[None, :, :]) ** 2).sum(-1))
    C = np.where(D <= gate, D, 1e6)
    ri, ci = linear_sum_assignment(C)
    return {int(ins[r]): (int(c), float(D[r, c])) for r, c in zip(ri, ci) if C[r, c] <= gate}


def two_stage(feet, sv, eligible_players, eligible_boxes, assign) -> tuple:
    """Stage A at 2.5 ft over the eligible players and boxes, then stage B at 5 ft over those still unmatched.
    assign(feet_sub, sv_sub, gate) -> {sub player: (sub box, distance)}. -> ({player: (box, distance, "A" or "B")},
    {box: player})."""
    out, claimed = {}, {}
    players, boxes = list(eligible_players), list(eligible_boxes)
    for stage, gate in (("A", MISS_FT), ("B", NEAR_FT)):
        p_left = [p for p in players if p not in out]
        b_left = [b for b in boxes if b not in claimed]
        if not p_left or not b_left:
            continue
        sub = assign(np.asarray(feet, np.float64).reshape(-1, 2)[b_left], np.asarray(sv, np.float64).reshape(-1, 2)[p_left], gate)
        for pi, (bi, d) in sub.items():
            out[p_left[pi]] = (b_left[bi], d, stage)
            claimed[b_left[bi]] = p_left[pi]
    return out, claimed


def _assign_all(feet_sub, sv_sub, gate):
    return assign_1to1(feet_sub, sv_sub, np.zeros(len(feet_sub), bool), gate)


def classify(n: float, nearest: int | None, box_is_ref, match, claimed: dict, j: int, n_boxes: int) -> str:
    """The class of in-frame player j (section 5). match: (box, distance, stage) or None; claimed: {box: player}."""
    if n_boxes and n <= MISS_FT:
        return "hit"
    if n_boxes and n <= NEAR_FT:
        if match is not None and match[2] == "A":
            return "placement_matched_in_stage_A"
        if match is not None:
            return "P-ii"
        if box_is_ref[nearest]:
            return "P-iii-ref"
        if nearest in claimed and claimed[nearest] != j:
            return "P-i"
        return "P-iii-free"
    if not n_boxes:
        return "D-iii-none"
    if match is not None:
        return "detection_matched"
    if box_is_ref[nearest]:
        return "D-iii-ref"
    if nearest in claimed and claimed[nearest] != j:
        return "D-i"
    return "D-iii-free"


def frame_classes(feet, sv, inside, is_ref) -> dict:
    """Per in-frame player of one frame: the main class, the single-stage and in-frame-only sensitivities and the
    report-only fields."""
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    is_ref = np.asarray(is_ref, bool).reshape(-1)
    nb = len(feet)
    D = np.sqrt(((sv[:, None, :] - feet[None, :, :]) ** 2).sum(-1)) if nb else np.zeros((len(sv), 0))
    nonref = [int(b) for b in np.where(~is_ref)[0]]
    main, claimed = two_stage(feet, sv, range(len(sv)), nonref, _assign_all)
    single = {p: (b, d, "B" if d > MISS_FT else "A") for p, (b, d) in assign_1to1(feet, sv, is_ref, NEAR_FT).items()}
    claimed_single = {m[0]: p for p, m in single.items()}
    ins = [int(j) for j in np.where(np.asarray(inside, bool))[0]]
    inf, claimed_inf = two_stage(feet, sv, ins, range(nb), _assign_all)
    out = {}
    for j in ins:
        n = float(D[j].min()) if nb else np.inf
        k = int(D[j].argmin()) if nb else None
        cls = classify(n, k, is_ref, main.get(j), claimed, j, nb)
        rec = {"class": cls, "single": classify(n, k, is_ref, single.get(j), claimed_single, j, nb),
               "inframe": classify(n, k, np.zeros(nb, bool), inf.get(j), claimed_inf, j, nb)}
        if cls == "P-ii":
            rec["stage_b_ft"] = main[j][1]
        if cls in ("P-i", "D-i"):
            other = claimed[k]
            rec["claimer_in_frame"] = bool(np.asarray(inside, bool)[other])
            rec["claimer_ft"] = float(np.hypot(*(sv[j] - sv[other])))
            rec["unclaimed_5_10ft"] = any(b not in claimed and NEAR_FT < D[j, b] <= FAR_REPORT_FT for b in nonref)
        out[j] = rec
    return out


# ── pure: measures ─────────────────────────────────────────────────────────────────────────────────────────────────
def window_matrix(rows: list, key: str = "class") -> tuple:
    """rows: (window, record) -> (names, n_windows x len(names) counts) with in-frame, placement and detection totals."""
    names = ["in_frame", "placement", "detection", *P_CLASSES, *D_CLASSES]
    nw = 1 + max((w for w, _ in rows), default=-1)
    M = np.zeros((nw, len(names)))
    for w, r in rows:
        c = r[key]
        M[w, 0] += 1
        if c in P_CLASSES or c in COUNTERS[:2]:
            M[w, 1] += 1
        elif c in D_CLASSES or c == COUNTERS[2]:
            M[w, 2] += 1
        if c in names:
            M[w, names.index(c)] += 1
    return names, M


def shares(names: list, tot: np.ndarray) -> dict:
    out = {}
    i_in, i_p, i_d = names.index("in_frame"), names.index("placement"), names.index("detection")
    for c in P_CLASSES + D_CLASSES:
        v = tot[names.index(c)]
        base = tot[i_p] if c in P_CLASSES else tot[i_d]
        out[c] = {"share": v / base if base else np.nan, "rate": v / tot[i_in] if tot[i_in] else np.nan}
    return out


def boot(Ms: list, names: list, n: int = BOOT_N, seed: int = SEED) -> dict:
    """Windows resampled whole within each game, summed -> [2.5, 97.5] percentiles of each share and rate."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        t = sum(M[rng.integers(0, len(M), len(M))].sum(0) for M in Ms)
        draws.append(shares(names, t))
    return {c: {k: [float(np.nanpercentile([d[c][k] for d in draws], 2.5)),
                    float(np.nanpercentile([d[c][k] for d in draws], 97.5))] for k in ("share", "rate")}
            for c in draws[0]}


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def git_head() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                           text=True).stdout.strip()
    return {"head": head, "tools_and_sportvu_clean": not dirty}


def collect(game: str) -> dict:
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.sync import MIRRORS, apply_mirror
    from tools import p8_frozen as F
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_e1_score import window_stats
    from tools.p8_placecause import cache_gaps
    from tools.p8_resetfix import one_to_one_missed, unmatched_1to1
    if game not in MEASURED + GATE_ONLY:
        raise SystemExit(f"{WHO}: {game} is not in this item")
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    head = git_head()
    if not head["tools_and_sportvu_clean"]:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    rec_p = r["ref"] / "rebaseline" / f"score__{game}__current.json"
    score = json.loads(rec_p.read_text())
    root = RB.truths(r)[game]["current"]
    watched = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    mirror = F.mirror_of(P, game)
    fx, fy = MIRRORS[mirror]
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    comp = P["builds"] / "r23"
    P_t = {**P, "labels": root / "labels"}
    wins = sorted((w for w in F.label_windows(P, game) if (comp / f"{w['window']}_frames.json").exists()),
                  key=lambda w: w["window"])
    stems = [w["window"] for w in wins]
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    if sorted(score["windows"]) != stems:
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's score record")
    measured = game in MEASURED
    rows, counters = [], {c: 0 for c in COUNTERS}
    for wi, w in enumerate(wins):
        s = w["window"]
        for key, p in (("sidecar", comp / f"{s}_frames.json"), ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                       ("labels", P_t["labels"] / game / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
            want = (score.get("hashes") or {}).get(s, {}).get(key)
            if want is None and key in ("sidecar", "timemap", "labels"):
                raise SystemExit(f"{WHO}: {s} has no {key} hash in D202's record")
            if want is not None and RB.sha256(p) != want:
                raise SystemExit(f"{WHO}: {s} {key} differs from D202's record")
        why = cache_gaps(P_t, game, w, comp, mom, mirror)
        if why:
            raise SystemExit(f"{WHO}: {s}: the scorecard cache lacks {why}")
        full = S.score_window(P_t, game, w, comp, "camera", mom, mirror, refs, events)
        if "skipped" in full[0]:
            raise SystemExit(f"{WHO}: {s} skipped: {full[0]['skipped']}")
        st = window_stats(full[0])
        scored = {int(o["f"]) for o in full[1]}
        side = json.loads((comp / f"{s}_frames.json").read_text())
        srow = {int(x["frame"]): x for x in side["frames"]}
        sidecar_cls = any("cls" in b for x in srow.values() for b in x["boxes"])
        cache = json.loads((P_t["cache"] / "scorecard" / f"{s}.json").read_text())
        refd = {int(k): v for k, v in cache.get("referees", {}).items()}
        cap = cv2.VideoCapture(str(P_t["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        truth, _ = S.truth_rows(P_t, game, w, "camera", mom.index, mirror)
        c = {"in_frame": 0, "missed": 0, "far": 0, "missed_1to1": 0, "unmatched_1to1": 0}
        for row in truth:
            f = int(row["frame"])
            x = srow.get(f)
            if x is None or f not in scored:
                continue
            sv = apply_mirror(mom.index.q[row["q"]]["xy"][row["moment"]], fx, fy)
            Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
            px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
            m = FRAME_MARGIN_PX
            inside = (px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)
            feet = np.array([b["foot_stab"] for b in x["boxes"]], np.float32).reshape(-1, 1, 2)
            ft = cv2.perspectiveTransform(feet, Ht).reshape(-1, 2).astype(np.float64) if len(feet) else np.zeros((0, 2))
            is_ref = np.array([(b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else
                               any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
                               for b in x["boxes"]], bool)
            svf = np.asarray(sv, np.float64)
            # the scorer's own counts, and the equivalence of the copies at both gates
            u25, u50 = unmatched_1to1(ft, svf, inside, is_ref, 2.5), unmatched_1to1(ft, svf, inside, is_ref, 5.0)
            m25 = one_to_one_missed(ft, svf, inside, 2.5)
            c["unmatched_1to1"] += u25; c["missed_1to1"] += m25
            if unmatched_count(assign_1to1(ft, svf, is_ref, 2.5), inside, is_ref) != u25 or \
                    unmatched_count(assign_1to1(ft, svf, is_ref, 5.0), inside, is_ref) != u50:
                raise SystemExit(f"{WHO}: {s} frame {f}: assign_1to1 differs from unmatched_1to1")
            if int(inside.sum()) - len(assign_inframe(ft, svf, inside, 2.5)) != m25:
                raise SystemExit(f"{WHO}: {s} frame {f}: assign_inframe differs from one_to_one_missed")
            Dm = np.sqrt(((svf[:, None, :] - ft[None, :, :]) ** 2).sum(-1)).min(1) if len(ft) else np.full(len(svf), np.inf)
            c["in_frame"] += int(inside.sum())
            c["missed"] += int((inside & (Dm > MISS_FT)).sum())
            c["far"] += int((inside & (Dm > NEAR_FT)).sum())
            if not measured:
                continue
            for j, rec in frame_classes(ft, svf, inside, is_ref).items():
                if rec["class"] in COUNTERS:
                    counters[rec["class"]] += 1
                rows.append([wi, rec])
        want = score["windows"][s]
        mine = {**c, "nr_boxes": st["nr_boxes"]}
        if any(mine[k] != want.get(k) for k in mine) or st["in_frame"] != c["in_frame"] or st["missed"] != c["missed"]:
            raise SystemExit(f"{WHO}: count gate failed on {s}: mine {mine}, D202 {[want.get(k) for k in mine]}")
        if any(counters.values()):
            raise SystemExit(f"{WHO}: an impossibility counter is not zero: {counters}")
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    out = {"game": game, "design": "docs/research/P8_one2one.md version 2", "git": head,
           "score_record": str(rec_p.relative_to(r["ref"])), "run_id": score.get("run_id"), "windows": len(stems),
           "gate": {"passes": True, "windows": len(stems), "fields": ["in_frame", "missed", "far", "nr_boxes",
                                                                      "missed_1to1", "unmatched_1to1"]},
           "equivalence": "assign_1to1 at 2.5 and 5.0 ft and assign_inframe at 2.5 ft equal the scorer's counts on every frame",
           "counters": counters, "manifest_unchanged": after == watched, "seconds": round(time.time() - t0, 1)}
    if measured:
        out["rows"] = rows
    else:
        out["note"] = "count gate only; no class is computed, so this game stays unspent (design section 3)"
    d = r["ref"] / "one2one"
    d.mkdir(exist_ok=True)
    atomic_write(d / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "windows", "gate", "counters", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


# ── report (system python, no lock) ────────────────────────────────────────────────────────────────────────────────
def summarize(per_game: dict) -> dict:
    out = {}
    sets = {**{g: [g] for g in per_game}, "pooled": list(per_game)}
    for key in ("class", "single", "inframe"):
        res = {}
        for name, gs in sets.items():
            mats = [window_matrix([(w, rec) for w, rec in per_game[g]["rows"]], key) for g in gs]
            names = mats[0][0]
            tot = sum(M.sum(0) for _, M in mats)
            pt, ci = shares(names, tot), boot([M for _, M in mats], names)
            res[name] = {"counts": {n: int(v) for n, v in zip(names, tot)},
                         "classes": {c: {k: {"value": pt[c][k], "interval": ci[c][k]} for k in ("share", "rate")} for c in pt}}
        out[{"class": "main_two_stage", "single": "sensitivity_single_stage_5ft", "inframe": "sensitivity_inframe_only"}[key]] = res
    rep = {}
    for g, d in per_game.items():
        recs = [rec for _, rec in d["rows"]]
        pii = [r["stage_b_ft"] for r in recs if r["class"] == "P-ii"]
        rep[g] = {"P-ii_median_stage_b_ft": float(np.median(pii)) if pii else None}
        for c in ("P-i", "D-i"):
            rr = [r for r in recs if r["class"] == c]
            rep[g][c] = {"n": len(rr), "claimer_in_frame_share": float(np.mean([r["claimer_in_frame"] for r in rr])) if rr else None,
                         "median_claimer_ft": float(np.median([r["claimer_ft"] for r in rr])) if rr else None,
                         "unclaimed_box_5_10ft_share": float(np.mean([r["unclaimed_5_10ft"] for r in rr])) if rr else None}
    return {"splits": out, "report_only": rep}


def report() -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    r = RB.roots()
    recs = {g: json.loads((r["ref"] / "one2one" / f"{g}.json").read_text()) for g in MEASURED + GATE_ONLY}
    s = summarize({g: recs[g] for g in MEASURED})
    out = {"what": "docs/research/P8_one2one.md version 2; report only; nothing fitted",
           "git": {g: recs[g]["git"] for g in recs}, "gates": {g: recs[g]["gate"] for g in recs},
           "counters": {g: recs[g]["counters"] for g in MEASURED}, **s,
           "hou_cle_2016": "count gate only; no class computed (unspent)",
           "limits": "one-to-one assignment is a convention; build and truth errors are not separable and not "
                     "independent; frames the truth rejected are not seen"}
    atomic_write(REPO / "reports" / "research" / "p8_one2one.json", json.dumps(clean(out), indent=1))
    print(json.dumps(clean({k: out[k] for k in ("gates", "counters")}), indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("game")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    collect(a.game) if a.cmd == "collect" else report()


if __name__ == "__main__":
    sys.exit(main())
