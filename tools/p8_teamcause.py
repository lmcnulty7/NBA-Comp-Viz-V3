"""tools/p8_teamcause.py: team errors by cause on the development standard rows (docs/research/P8_teamcause.md,
version 2). Report only: nothing is fitted, tuned or adopted.

    python3 -m tools.p8_teamcause stored        stage A, a readout of the D202 score records and identity files
                                                -> reports/research/p8_teamcause_stored.json
    <frozen CV python> -m tools.p8_teamcause rows    stage B, inside the heavy-job lock ("p8 teamcause rows")
                                                -> reports/research/p8_teamcause.json

Units: gsw_cha_2016 and gsw_bkn_2015 on multipool v1 and their pool. hou_cle_2016 is left out of both stages except
its aggregate team row (already seen in D202), so it stays unspent for a later team arm's held-back test. Each command
writes only its own record and opens every input read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_teamcause"
DATA = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
POOL = (CHA, BKN)
RUN_ID = "20261127T171848Z"
D202_SHA = {CHA: "24c3b7d9d58da5fa5b080a200dc295b42cf4e47cc3af7ffc6a793f07374c32b0",
            BKN: "151c27f549711322bdff4cfbb0294ebc2a4e41604bc30c4182dafc828305498c"}
LOW_ACC = 0.6
CARRY_SHARE, CARRY_RATIO, MIN_WINDOWS, OVERLAP_RHO = 0.25, 1.5, 10, 0.5
BOOT_N, BOOT_SEED = 2000, 0
CAP_S = 1200
GATE_COUNT = ("n", "majority", "fixed")


def out_stored() -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / "p8_teamcause_stored.json"


def out_rows() -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / "p8_teamcause.json"


def read_json(p: Path):
    with open(p, "r") as fh:
        return json.load(fh)


def sha256(p: Path) -> str | None:
    if not Path(p).exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def colour_distances(colors: dict) -> tuple:
    """(BGR Euclidean, CIELAB Euclidean) between the window's two team colours (team_colors_bgr); None without two."""
    if not colors or len(colors) != 2:
        return None, None
    a, b = (np.asarray(v, float) for v in colors.values())
    import cv2
    lab = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float) for c in (a, b)]
    return float(np.linalg.norm(a - b)), float(np.linalg.norm(lab[0] - lab[1]))


def worst_windows(errors: dict, rows: dict) -> list:
    """Section 8.4: windows with team rows ranked by error count (ties by name); the ceiling of a tenth of them."""
    have = sorted((w for w in rows if rows[w] > 0), key=lambda w: (-errors[w], w))
    return have[:math.ceil(0.1 * len(have))] if have else []


def majority_teams(rows: list) -> dict:
    """label -> its majority SportVU team in the window (tie: the smaller team id; section 8.5)."""
    out = {}
    for lab in {r[1] for r in rows}:
        sub = [r[2] for r in rows if r[1] == lab]
        out[lab] = sorted(set(sub), key=lambda t: (-sub.count(t), t))[0]
    return out


def fixed_mapping(rows: list) -> dict:
    """The one-to-one mapping tools.p8_e3.window_team's fixed takes (the first best in its candidate order)."""
    teams = sorted({r[2] for r in rows})
    best, best_n = {}, -1
    for t0 in teams + [None]:
        for t1 in teams + [None]:
            if t0 is not None and t0 == t1:
                continue
            m = {0: t0, 1: t1}
            n = sum(1 for r in rows if m.get(r[1]) == r[2])
            if n > best_n:
                best, best_n = m, n
    return best


def bin_of(value, edges: tuple, labels: tuple):
    if value is None:
        return "none"
    for e, lab in zip(edges, labels):
        if value < e:
            return lab
    return labels[-1]


def tercile_edges(values: list) -> tuple:
    v = [x for x in values if x is not None]
    return (float(np.percentile(v, 100 / 3)), float(np.percentile(v, 200 / 3))) if v else (0.0, 0.0)


def switch_tracks(pairs: list) -> set:
    """Canonical tracks paired with two or more SportVU player ids in the window: pairs (cid, pid)."""
    ids = {}
    for cid, pid in pairs:
        ids.setdefault(cid, set()).add(pid)
    return {c for c, s in ids.items() if len(s) >= 2}


def bin_table(rows: list, feature: str) -> dict:
    """{bin: rows, errors, fixed_errors, abstentions, windows} for one feature over row dicts."""
    out = {}
    for r in rows:
        b = r[feature]
        e = out.setdefault(str(b), {"rows": 0, "errors": 0, "fixed_errors": 0, "abstentions": 0, "windows": set()})
        if r["abstained"]:
            e["abstentions"] += 1
            continue
        e["rows"] += 1
        e["errors"] += r["error"]
        e["fixed_errors"] += r["fixed_error"]
        e["windows"].add(r["window"])
    return {k: {**v, "windows": len(v["windows"])} for k, v in out.items()}


def carry_reading(rows: list, feature: str, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Section 8.6 on the pooled rows: per bin, share of errors, rate ratio with a window-cluster bootstrap of it,
    windows, excess errors, and whether it carries."""
    from tools.p8_resetfix import resample_picks
    lab = [r for r in rows if not r["abstained"]]
    wins = sorted({r["window"] for r in lab})
    widx = {w: i for i, w in enumerate(wins)}
    bins = sorted({str(r[feature]) for r in lab})
    E = np.zeros((len(wins), len(bins))); N = np.zeros((len(wins), len(bins)))
    for r in lab:
        i, j = widx[r["window"]], bins.index(str(r[feature]))
        N[i, j] += 1
        E[i, j] += r["error"]
    tot_e, tot_n = E.sum(), N.sum()
    rate = tot_e / tot_n if tot_n else 0.0
    picks = resample_picks(len(wins), n_boot, seed) if len(wins) >= 2 else None
    out = {}
    for j, b in enumerate(bins):
        e, n = E[:, j].sum(), N[:, j].sum()
        ratio = (e / n) / rate if n and rate else None
        lo = None
        if picks is not None and n:
            se, sn = E[picks, j].sum(1), N[picks, j].sum(1)
            te, tn = E[picks].sum((1, 2)), N[picks].sum((1, 2))
            ok = (sn > 0) & (te > 0)
            vals = (se[ok] / sn[ok]) / (te[ok] / tn[ok])
            lo = float(np.percentile(vals, 2.5)) if len(vals) else None
        nw = int((N[:, j] > 0).sum())
        share = e / tot_e if tot_e else 0.0
        carries = bool(share >= CARRY_SHARE and ratio is not None and ratio >= CARRY_RATIO and lo is not None
                       and lo > 1 and nw >= MIN_WINDOWS)
        out[b] = {"rows": int(n), "errors": int(e), "rate": e / n if n else None, "share_of_errors": share,
                  "ratio": ratio, "ratio_lower": lo, "windows": nw, "excess": float(e - n * rate), "carries": carries}
    return out


def overlap_pairs(rows: list, features: list) -> list:
    """Feature pairs whose rank correlation over the labelled rows exceeds OVERLAP_RHO in absolute value."""
    from scipy.stats import spearmanr
    lab = [r for r in rows if not r["abstained"]]
    codes = {}
    for f in features:
        levels = sorted({str(r[f]) for r in lab})
        codes[f] = [levels.index(str(r[f])) for r in lab]
    out = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            if len(set(codes[a])) < 2 or len(set(codes[b])) < 2:
                continue
            rho = float(spearmanr(codes[a], codes[b]).statistic)
            if abs(rho) > OVERLAP_RHO:
                out.append({"pair": [a, b], "rho": rho})
    return out


def cross_tab(rows: list, a: str, b: str) -> dict:
    out = {}
    for r in rows:
        if r["abstained"]:
            continue
        k = f"{r[a]}|{r[b]}"
        e = out.setdefault(k, {"rows": 0, "errors": 0})
        e["rows"] += 1
        e["errors"] += r["error"]
    return out


def rank_carriers(readings: dict, overlaps: list) -> dict:
    """All carrying bins ranked by excess errors; the top one named; carriers in an overlapping pair not separated."""
    car = [(f, b, v["excess"]) for f, bins in readings.items() for b, v in bins.items() if v["carries"]]
    car.sort(key=lambda x: (-x[2], x[0], x[1]))
    feats = {f for f, _b, _x in car}
    not_sep = [o["pair"] for o in overlaps if o["pair"][0] in feats and o["pair"][1] in feats]
    return {"carriers": [{"feature": f, "bin": b, "excess": x} for f, b, x in car],
            "top": ({"feature": car[0][0], "bin": car[0][1]} if car else None),
            "not_separated": not_sep,
            "reading": "spread: no bin carries" if not car else "carried"}


# ── stage A: readout of stored records ─────────────────────────────────────────────────────────────────────────────
def stored_window_rows(rec: dict, ident_dir: Path) -> list:
    out = []
    for w in sorted(rec["windows"]):
        t = rec["windows"][w]["team"]
        idf = ident_dir / f"{w}_identity.json"
        ident = read_json(idf) if idf.exists() else {}
        teams = (ident.get("identity") or {}).get("teams") or {}
        bgr, lab = colour_distances(ident.get("team_colors_bgr") or {})
        out.append({"window": w, "n": t["n"], "majority": t["majority"], "fixed": t["fixed"],
                    "errors": t["n"] - t["majority"], "accuracy": (t["majority"] / t["n"]) if t["n"] else None,
                    "silhouette": teams.get("silhouette"), "colour_bgr": bgr, "colour_lab": lab,
                    "abstained": teams.get("tracks_abstained"), "balance": teams.get("frame_balance_median"),
                    "team_vetoed": (ident.get("identity") or {}).get("team_vetoed")})
    return out


def readout(wins: list) -> dict:
    from scipy.stats import spearmanr
    have = [w for w in wins if w["n"] > 0]
    err = {w["window"]: w["errors"] for w in wins}
    nrow = {w["window"]: w["n"] for w in wins}
    worst = worst_windows(err, nrow)
    te, tn = sum(err.values()), sum(nrow.values())
    corr, terc = {}, {}
    for f in ("silhouette", "colour_bgr", "colour_lab", "abstained", "team_vetoed"):
        xs = [(w[f], w["accuracy"]) for w in have if w[f] is not None]
        corr[f] = float(spearmanr(*zip(*xs)).statistic) if len(xs) >= 3 and len({x for x, _ in xs}) > 1 else None
        edges = tercile_edges([w[f] for w in have])
        tb = {}
        for w in have:
            b = bin_of(w[f], edges, ("low", "middle", "high"))
            e = tb.setdefault(b, {"windows": 0, "rows": 0, "errors": 0})
            e["windows"] += 1; e["rows"] += w["n"]; e["errors"] += w["errors"]
        terc[f] = {"edges": edges, "bins": tb}
    return {"windows": len(wins), "windows_with_rows": len(have), "windows_without_rows": len(wins) - len(have),
            "rows": tn, "errors": te, "accuracy": (1 - te / tn) if tn else None,
            "fixed_accuracy": (sum(w["fixed"] for w in wins) / tn) if tn else None,
            "windows_below_0_5": sum(1 for w in have if w["accuracy"] < 0.5),
            "windows_at_or_below_0_6": sum(1 for w in have if w["accuracy"] <= LOW_ACC),
            "majority_not_fixed": [w["window"] for w in wins if w["majority"] != w["fixed"]],
            "worst_windows": worst, "worst_error_share": (sum(err[w] for w in worst) / te) if te else None,
            "worst_row_share": (sum(nrow[w] for w in worst) / tn) if tn else None,
            "spearman_vs_accuracy": corr, "terciles": terc,
            "median_colour_lab": float(np.median([w["colour_lab"] for w in have if w["colour_lab"] is not None]))
            if have else None,
            "median_silhouette": float(np.median([w["silhouette"] for w in have if w["silhouette"] is not None]))
            if have else None}


def cmd_stored(args, root: Path | None = None) -> dict:
    from tools.p8_common import atomic_write, guard
    from tools.research_testset import refuse_cv_test
    from export.schemas import find_private
    refuse_cv_test([CHA, BKN, HOU], WHO)
    guard([CHA, BKN, HOU], WHO)
    ref = root or DATA / "work_ref"
    recs = {}
    for g in POOL:
        p = ref / "rebaseline" / f"score__{g}__current.json"
        if sha256(p) != D202_SHA[g]:
            raise SystemExit(f"{WHO}: {p.name} is not the pinned D202 record")
        recs[g] = read_json(p)
        if recs[g].get("run_id") != RUN_ID:
            raise SystemExit(f"{WHO}: {p.name} is not from run {RUN_ID}")
    per = {g: stored_window_rows(recs[g], ref / "builds" / "r23") for g in POOL}
    out = {"what": "Team errors by cause, stage A: a readout of the D202 score records and identity files "
                   "(docs/research/P8_teamcause.md section 8.3); report only",
           "games": {g: readout(per[g]) for g in POOL}, "pooled": readout(per[CHA] + per[BKN]),
           "per_window": per}
    hou = read_json(ref / "rebaseline" / f"score__{HOU}__current.json")
    n = sum(w["team"]["n"] for w in hou["windows"].values())
    out["hou_cle_2016_aggregate_only"] = {"team_accuracy": sum(w["team"]["majority"] for w in hou["windows"].values())
                                          / n if n else None, "note": "already seen in D202; nothing else is read"}
    from tools.p8_placement import rnd
    text = json.dumps(rnd(out), indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(out_stored(), text)
    print(json.dumps({g: {k: out["games"][g][k] for k in ("accuracy", "windows_at_or_below_0_6", "worst_error_share",
                                                            "worst_row_share")} for g in POOL}, indent=1))
    return out


# ── stage B: per-row features (frozen CV python, inside the lock) ──────────────────────────────────────────────────
FEATURES = ("team_id", "light_label", "third", "zone", "crowd_box", "crowd_sv", "edge", "height", "colour_lab",
            "since_cut", "referee_rule", "referee_overlap", "track_age", "pair_distance", "switch_track")


def window_rows(P, game, w, comp, ctx, edges_h=None) -> tuple:
    """Recompute score_window's team rows of one window with each row's features -> (tuples, row dicts)."""
    import cv2
    from sportvu import bench as B
    from sportvu.eval import near_side
    from sportvu.sync import apply_mirror
    from sportvu.truth import _match
    from tools import p8_scorecard as S
    s = w["window"]
    side = read_json(comp / f"{s}_frames.json")
    srow = {r["frame"]: r for r in side["frames"]}
    tm = read_json(P["label_build"] / f"{s}_timemap.json")
    ident = read_json(comp / f"{s}_identity.json")
    idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
    team_of = {int(k): v for k, v in ident.get("team_by_track", {}).items()}
    light = ident.get("light_team")
    _bgr, lab_d = colour_distances(ident.get("team_colors_bgr") or {})
    fps = float(tm["fps"])
    live = sorted({int(f) for f, _, _ in tm["frames"]})
    rows, _off = S.truth_rows(P, game, w, "camera", ctx["mom"].index, ctx["mirror"])
    cache = read_json(P["cache"] / "scorecard" / f"{s}.json")
    wide = {int(k): v for k, v in cache["wide"].items()}
    refd = {int(k): v for k, v in cache.get("referees", {}).items()}
    corr = {int(k): v for k, v in cache.get("corr", {}).items()}
    cframes = sorted(corr)
    cuts = S.cuts_from_corr(cframes, [corr[f] for f in cframes])
    lw = {f for f in live if wide.get(f, 0.0) >= ctx["refs"].wide_thr}
    rows, _np = S.universe(rows, lw, srow)
    sidecar_cls = any("cls" in b for r in side["frames"] for b in r["boxes"])
    cap = cv2.VideoCapture(str(P["label_build"] / f"{s}.mp4"))
    Wd, Hd = int(cap.get(3)), int(cap.get(4))
    cap.release()
    first_seen = {}
    for r in sorted(side["frames"], key=lambda x: x["frame"]):
        for b in r["boxes"]:
            first_seen.setdefault(idmap.get(b["tid"], b["tid"]), r["frame"])
    tuples, out, pid_pairs = [], [], []
    for row in rows:
        f, q, i = row["frame"], row["q"], row["moment"]
        dq = ctx["mom"].index.q[q]
        sv = apply_mirror(dq["xy"][i], ctx["fx"], ctx["fy"]); pid = dq["pid"][i]; team = dq["team"][i]
        Ht = np.array(row["H_truth"], np.float64).reshape(3, 3)
        nh = near_side(Ht)
        boxes = srow[f]["boxes"]
        feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
        ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2) if len(feet) else \
            np.zeros((0, 2), np.float32)
        pairs = _match(ct, sv, B.MATCH_FT) if len(ct) else []
        cr = S.crowding(sv)
        since, cens = S.since_cut(f, cuts, 0, fps)
        for bi, j, dist in pairs:
            b = boxes[bi]
            cid = idmap.get(b["tid"], b["tid"])
            lab = team_of.get(cid)
            if lab is not None:
                tuples.append((cid, lab, int(team[j])))
            pid_pairs.append((cid, int(pid[j])))
            others = [np.hypot(*(ct[k] - ct[bi])) for k in range(len(ct)) if k != bi]
            x1, y1, x2, y2 = b["bbox"][:4]
            is_ref = (b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else \
                any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
            out.append({"window": s, "frame": int(f), "cid": int(cid), "label": lab, "team": int(team[j]),
                        "abstained": lab is None, "team_id": int(team[j]),
                        "light_label": None if lab is None or light is None else bool(int(lab) == int(light)),
                        "third": S.third_of(sv[j][1], nh), "zone": S.zone_of(*sv[j]),
                        "crowd_box": bin_of(min(others) if others else None, (3.0, 6.0), ("<3", "3-6", ">6")),
                        "crowd_sv": bin_of(float(cr[j]), (3.0, 6.0), ("<3", "3-6", ">6")),
                        "edge": bin_of(float(min(x1, y1, Wd - x2, Hd - y2)), (50.0, 150.0), ("<50", "50-150", ">150")),
                        "height_px": float(y2 - y1), "colour_lab_value": lab_d,
                        "since_cut": "censored" if cens else bin_of(since, (1.0, 3.0), ("<1", "1-3", ">3")),
                        "referee_rule": bool(is_ref),
                        "referee_overlap": any(B._iou(b["bbox"], dd[:4]) >= 0.3 for dd in refd.get(f, [])),
                        "track_age": bin_of((f - first_seen.get(cid, f)) / fps, (1.0, 3.0), ("<1", "1-3", ">3")),
                        "pair_distance": "<=2.5" if dist <= 2.5 else "2.5-5"})
    sw = switch_tracks(pid_pairs)
    lab_rows = [(r["cid"], r["label"], r["team"]) for r in out if not r["abstained"]]
    maj, fix = majority_teams(lab_rows), fixed_mapping(lab_rows)
    for r in out:
        r["switch_track"] = r["cid"] in sw
        r["error"] = (not r["abstained"]) and maj[r["label"]] != r["team"]
        r["fixed_error"] = (not r["abstained"]) and fix.get(r["label"]) != r["team"]
    return tuples, out


def manifest(root: Path, subdirs: tuple) -> dict:
    out = {}
    for sd in subdirs:
        d = root / sd
        if d.is_symlink():
            out[f"{sd}/"] = f"link:{d.resolve().name}"
            continue
        for p in sorted(d.rglob("*")) if d.exists() else []:
            if p.is_file():
                out[f"{sd}/{p.relative_to(d)}"] = sha256(p)
    return out


def cmd_rows(args) -> dict:
    import subprocess
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_pin
    from tools import p8_scorecard as S
    from tools.p8_common import REPO, atomic_write, guard, p8_root
    from tools.p8_e3 import window_team
    from tools.p8_placement import rnd
    from tools.research_testset import refuse_cv_test
    from export.schemas import find_private
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    ref, ms = DATA / "work_ref", DATA / "work_ms"
    if p8_root().resolve() != ref.resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    refuse_cv_test(list(POOL), WHO)
    guard(list(POOL), WHO)
    st = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                        text=True, check=True).stdout.strip()
    if st:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    pinned, now = read_json(p8_pin.PIN_FILE), p8_pin.current()
    if [k for k in now if now[k] != pinned.get(k)]:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    recs = {}
    for g in POOL:
        p = ref / "rebaseline" / f"score__{g}__current.json"
        if sha256(p) != D202_SHA[g] or read_json(p).get("run_id") != RUN_ID:
            raise SystemExit(f"{WHO}: {p.name} is not the pinned D202 record")
        recs[g] = read_json(p)
    if not out_stored().exists():
        raise SystemExit(f"{WHO}: run stage A (stored) first")
    watch = lambda: {"ms": manifest(ms, ("labels", "truth_accept")),
                     "ref": manifest(ref, ("label_build", "truth_accept", "cache/scorecard"))}
    before = watch()
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    P_t = {**P, "labels": ms / "labels"}
    comp = P["builds"] / "r23"
    rows_all, gate = [], {}
    for g in POOL:
        gd = F.game_of(g)
        guard([gd["sportvu"]], WHO)
        mirror = F.mirror_of(P, g)
        mom = S.Moments(read_json(P["sportvu"] / f"{gd['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        events = S.pbp_events(gd["sportvu"])
        wins = [w for w in F.label_windows(P, g) if w["window"] in recs[g]["windows"]]
        wins.sort(key=lambda w: w["window"])
        guard([w["window"] for w in wins], WHO)
        if len(wins) != len(recs[g]["windows"]):
            raise SystemExit(f"{WHO}: {g}: the window lists differ")
        for w in wins:                                     # missing cache entries first: stage B computes nothing new
            s = w["window"]
            c = read_json(P["cache"] / "scorecard" / f"{s}.json")
            tm = read_json(P["label_build"] / f"{s}_timemap.json")
            rows_t, _ = S.truth_rows(P_t, g, w, "camera", mom.index, mirror)
            side = read_json(comp / f"{s}_frames.json")
            sidecar_cls = any("cls" in b for x in side["frames"] for b in x["boxes"])
            need_ref = set() if sidecar_cls else {int(x["frame"]) for x in rows_t}
            missing = len({int(f) for f, _, _ in tm["frames"]} - {int(k) for k in c.get("wide", {})}) + \
                len(need_ref - {int(k) for k in c.get("referees", {})}) + (0 if c.get("corr") else 1)
            if missing:
                raise SystemExit(f"{WHO}: {s} has {missing} missing cache entries; stage B computes nothing new")
        for w in wins:
            if time.time() - t0 > CAP_S:
                raise SystemExit(f"{WHO}: the cap was reached; nothing is reported")
            s = w["window"]
            full = S.score_window(P_t, g, w, comp, "camera", mom, mirror, refs, events)
            tuples, rws = window_rows(P_t, g, w, comp, ctx)
            mine = window_team(tuples)
            want = recs[g]["windows"][s]["team"]
            if [tuple(t) for t in full[0]["team_rows"]] != tuples or {k: mine[k] for k in GATE_COUNT} != \
                    {k: want[k] for k in GATE_COUNT}:
                raise SystemExit(f"{WHO}: the gate fails on {s}; nothing is reported")
            gate[s] = True
            for r in rws:
                r["game"] = g
            rows_all += rws
    if watch() != before:
        raise SystemExit(f"{WHO}: a watched folder changed; nothing is reported")
    for g in POOL:                                         # per-game terciles of box height and colour distance
        gr = [r for r in rows_all if r["game"] == g]
        eh, ec = tercile_edges([r["height_px"] for r in gr]), tercile_edges([r["colour_lab_value"] for r in gr])
        for r in gr:
            r["height"] = bin_of(r["height_px"], eh, ("low", "middle", "high"))
            r["colour_lab"] = bin_of(r["colour_lab_value"], ec, ("low", "middle", "high"))
    if not out_stored().exists():
        raise SystemExit(f"{WHO}: run stage A (stored) first; its worst windows are read here")
    worst = set(read_json(out_stored())["pooled"]["worst_windows"])
    units = {g: [r for r in rows_all if r["game"] == g] for g in POOL}
    units["pooled"] = rows_all
    out = {"what": "Team errors by cause, stage B (docs/research/P8_teamcause.md); report only", "head": head,
           "gate_windows": len(gate), "seconds": round(time.time() - t0, 1), "units": {}}
    for u, rr in units.items():
        out["units"][u] = {"rows": sum(1 for r in rr if not r["abstained"]),
                           "errors": sum(r["error"] for r in rr), "abstentions": sum(r["abstained"] for r in rr),
                           "bins": {f: bin_table(rr, f) for f in FEATURES}}
    pooled = rows_all
    readings = {f: carry_reading(pooled, f) for f in FEATURES}
    ov = overlap_pairs(pooled, list(FEATURES))
    out["pooled_reading"] = {"bins": readings, "overlaps": ov,
                             "cross_tabs": {f"{o['pair'][0]}|{o['pair'][1]}": cross_tab(pooled, *o["pair"]) for o in ov},
                             **rank_carriers(readings, ov)}
    out["pooled_without_worst_windows"] = {"worst_windows": sorted(worst),
                                           "bins": {f: bin_table([r for r in pooled if r["window"] not in worst], f)
                                                    for f in FEATURES}}
    text = json.dumps(rnd(out), indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(out_rows(), text)
    print(json.dumps({"gate_windows": len(gate), "reading": out["pooled_reading"]["reading"],
                      "top": out["pooled_reading"]["top"], "seconds": out["seconds"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_teamcause")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stored")
    sub.add_parser("rows")
    a = ap.parse_args(argv)
    {"stored": cmd_stored, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
