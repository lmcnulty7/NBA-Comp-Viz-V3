"""tools/p8_teambound.py: the upper bound on the team row for label-level fixes (docs/research/P8_teambound.md,
versions 1 and 1.1). Report only, stored files only.

    <frozen CV python> -m tools.p8_teambound collect GAME   inside the lock ("p8 teambound GAME"), V3_P8_ROOT = work_ref
    python3 -m tools.p8_teambound report                    -> reports/research/p8_teambound.json

collect: per window, the pinned score_window and a copy of tools.p8_teamcause.window_rows extended with each row's
SportVU id, the box's fragment tid and the crop-frame flag (detect/reid.py's collector rule), gated on score_window's own
team_rows (in order) and on D202's n, majority and fixed; plus the capped-distance pairing diagnostic. report: the
current row, the oracles O1, O2 (all cuts, swap cuts only) at L in 0.5 to 2.0 s, O3, O1c, O1f, the abstained view,
the per-bin split, purity bands, the MDE and bar per fix, and the registered reading. hou_cle_2016 is refused before
any file is read. Writes only work_ref/teambound/<game>.json and the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_teambound"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
POOL = (CHA, BKN)
L_RULE, L_GRID = 1.0, (0.5, 0.75, 1.0, 1.5, 2.0)
SWAP_SPAN_S = 1.0
CROP_EVERY, CROP_CAP, CROP_MIN_PX = 3, 40, 14            # detect/reid.py TorsoCropCollector (config values)
MIN_CROP_FRAMES = 2                                      # config.TEAM_MIN_CROPS
F_RECOVERY, BAR_MIN, HOU_WINDOWS = 0.5, 0.03, 93
N_BOOT, SEED = 2000, 0
CAP_S = 15 * 60
RECORD = REPO / "reports" / "research" / "p8_teambound.json"
GATE_COUNT = ("n", "majority", "fixed")


def refuse_game(game: str) -> None:
    """9.10: hou_cle_2016 (and any non-pool game) is refused before any file is read; guard does not refuse it."""
    if game not in POOL:
        raise SystemExit(f"{WHO}: {game} is refused; this item reads gsw_cha_2016 and gsw_bkn_2015 only")


# ── pure: labels and scoring ───────────────────────────────────────────────────────────────────────────────────────
def majority_map(rows: list) -> dict:
    """label -> its majority SportVU team in the window (tie: the smaller team id; D203 8.5)."""
    out = {}
    for lab in {r[1] for r in rows}:
        sub = [r[2] for r in rows if r[1] == lab]
        out[lab] = sorted(set(sub), key=lambda t: (-sub.count(t), t))[0]
    return out


def correct(rows: list) -> list:
    """Per row: correct under the window's majority mapping (rows: (cid, label, team))."""
    m = majority_map(rows)
    return [m[r[1]] == r[2] for r in rows]


def encode(team: int, teams_sorted: list) -> int:
    """9.8: oracle labels 0/1 by sorted SportVU team id within the window."""
    return teams_sorted.index(team)


def track_majority(teams: list) -> int:
    return sorted(set(teams), key=lambda t: (-teams.count(t), t))[0]


def runs(id_rows: list) -> list:
    """id_rows: [(t_seconds, pid)] sorted -> [(pid, t_first, t_last)] maximal runs of the same pid."""
    out = []
    for t, p in id_rows:
        if out and out[-1][0] == p:
            out[-1] = (p, out[-1][1], t)
        else:
            out.append((p, t, t))
    return out


def cuts(id_rows: list, L: float) -> list:
    """9.6: the cut times. A run is sustained when last - first >= L; a cut is placed at the first row of each sustained
    run that follows a sustained run of a different id."""
    sus = [r for r in runs(id_rows) if r[2] - r[1] >= L]
    out = []
    for a, b in zip(sus, sus[1:]):
        if a[0] != b[0]:
            out.append((b[1], a[0], b[0]))           # (cut time, old id, new id)
    return out


def segment_of(t: float, cut_times: list) -> int:
    return sum(1 for c in cut_times if t >= c)


def tag_cut(t_cut: float, old: int, cid: int, pairs_by_t: list, span: float = SWAP_SPAN_S) -> str:
    """9.2: 'swap' when the old id is paired within 2.5 ft with another canonical track within span after the cut,
    else 'orphan'. pairs_by_t: [(t, cid, pid)] of the window's identity pairs."""
    for t, c, p in pairs_by_t:
        if t_cut <= t <= t_cut + span and p == old and c != cid:
            return "swap"
    return "orphan"


# ── pure: oracles over one window's rows ───────────────────────────────────────────────────────────────────────────
def window_oracles(rows: list, fps: float, L: float = L_RULE) -> dict:
    """rows: dicts with cid, tid, label (None = abstained), team, pid, id_pair, frame, crop. -> correct counts of the
    current labels and of every oracle, on the labelled rows (the denominator is unchanged), and the abstained view."""
    lab = [r for r in rows if r["label"] is not None]
    if not lab:
        return {"n": 0}
    teams = sorted({r["team"] for r in lab})
    cur_t = [(r["cid"], r["label"], r["team"]) for r in lab]
    cur = correct(cur_t)
    cur_map = majority_map(cur_t)
    by_c, by_tid = {}, {}
    for r in lab:
        by_c.setdefault(r["cid"], []).append(r)
        by_tid.setdefault(r["tid"], []).append(r)
    maj_c = {c: track_majority([r["team"] for r in rr]) for c, rr in by_c.items()}
    maj_f = {t: track_majority([r["team"] for r in rr]) for t, rr in by_tid.items()}

    def score(label_of):
        rr = [(r["cid"], label_of(r), r["team"]) for r in lab]
        return correct(rr)

    out = {"n": len(lab), "current": cur}
    out["O1"] = score(lambda r: encode(maj_c[r["cid"]], teams))
    out["O1f"] = score(lambda r: encode(maj_f[r["tid"]], teams))
    pure = {c for c, rr in by_c.items() if len({x["team"] for x in rr}) == 1}
    out["O3"] = score(lambda r: encode(maj_c[r["cid"]], teams) if r["cid"] in pure
                      else encode(cur_map[r["label"]], teams))
    crop_maj = {}
    for c, rr in by_c.items():
        cr = [x["team"] for x in rr if x["crop"]]
        crop_maj[c] = track_majority(cr) if cr else None
    out["O1c"] = score(lambda r: encode(crop_maj[r["cid"]], teams) if crop_maj[r["cid"]] is not None
                       else encode(cur_map[r["label"]], teams))
    # O2: cuts per track at sustained id changes, segments labelled with their majority team
    pairs_by_t = sorted((r["frame"] / fps, r["cid"], r["pid"]) for r in rows if r["id_pair"])
    seg_label, tags = {}, []
    for swap_only in (False, True):
        lab_of = {}
        for c, rr in by_c.items():
            idr = sorted((x["frame"] / fps, x["pid"]) for x in rr if x["id_pair"])
            cs = cuts(idr, L)
            tagged = []
            for t_cut, old, new in cs:
                tagged.append((t_cut, tag_cut(t_cut, old, c, pairs_by_t), frag_boundary(rr, t_cut, fps)))
            if not swap_only:
                tags += [{"tag": tg, "fragment_boundary": fb} for _, tg, fb in tagged]
            ct = [t for t, tg, _ in tagged if (tg == "swap" or not swap_only)]
            segs = {}
            for x in rr:
                segs.setdefault(segment_of(x["frame"] / fps, ct), []).append(x["team"])
            for x in rr:
                lab_of[id(x)] = encode(track_majority(segs[segment_of(x["frame"] / fps, ct)]), teams)
        seg_label["O2_swap" if swap_only else "O2_all"] = score(lambda r, lab_of=lab_of: lab_of[id(r)])
    out.update(seg_label)
    out["cut_tags"] = tags
    # section 5's abstained view (report only): O1 with abstained tracks labelled too; the denominator grows
    allr = [r for r in rows]
    maj_all = {}
    for r in allr:
        maj_all.setdefault(r["cid"], []).append(r["team"])
    t_all = sorted({r["team"] for r in allr})
    ok_all = correct([(r["cid"], encode(track_majority(maj_all[r["cid"]]), t_all), r["team"]) for r in allr])
    out["O1_with_abstained"] = {"correct": int(sum(ok_all)), "n": len(allr)}
    return out


def frag_boundary(rr: list, t_cut: float, fps: float) -> bool:
    """9.2: whether the cut falls where the canonical track's fragment tid changes (its last row before the cut has
    another tid than its first row at or after it)."""
    before = [x for x in rr if x["frame"] / fps < t_cut]
    after = [x for x in rr if x["frame"] / fps >= t_cut]
    if not before or not after:
        return False
    return max(before, key=lambda x: x["frame"])["tid"] != min(after, key=lambda x: x["frame"])["tid"]


def pattern(gains: dict, bars: dict) -> str:
    """9.1: gains[g] = {"colour", "tracking", "combined"}; bars[kind] = B_arm. -> the registered pattern."""
    gs = list(gains)
    ok = {k: {g: gains[g][k] is not None and gains[g][k] >= bars[k] for g in gs} for k in ("colour", "tracking",
                                                                                         "combined")}
    col, trk = all(ok["colour"].values()), all(ok["tracking"].values())
    if col and trk:
        return "both"
    if col:
        return "colour headroom"
    if trk:
        return "tracking headroom"
    c_games = {g for g in gs if ok["colour"][g]}
    t_games = {g for g in gs if ok["tracking"][g]}
    if c_games and t_games and not (c_games & t_games):     # precedence (9.11): split by kind before combined
        return "split by kind"
    if all(ok["combined"].values()):
        return "combined"
    if c_games or t_games:
        return "one game only"
    return "neither"


def mde(gains_w: list, n_draw: int = HOU_WINDOWS, n: int = N_BOOT, seed: int = SEED, f: float = F_RECOVERY) -> float:
    """9.5: gains_w = [(oracle correct - current correct, rows)] per window of cha and bkn. 93 windows drawn with
    replacement, the row-weighted mean of f x gain, 2,000 times; MDE = 2.8 x its standard deviation."""
    a = np.asarray(gains_w, float).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(a), n_draw)
        means.append(f * a[idx, 0].sum() / a[idx, 1].sum() if a[idx, 1].sum() else np.nan)
    return float(2.8 * np.nanstd(means))


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def crop_frames(side: dict, W: int, H: int) -> set:
    """detect/reid.py's collector: every 3rd processed frame (in processed order), every track of it, at most 40 crops
    per fragment, the inner jersey patch at least 14 px on each side. -> {(frame, tid)}."""
    out, count = set(), {}
    for n, row in enumerate(sorted(side["frames"], key=lambda r: r["frame"])):
        if n % CROP_EVERY:
            continue
        for b in row["boxes"]:
            t = int(b["tid"])
            if count.get(t, 0) >= CROP_CAP:
                continue
            x1, y1, x2, y2 = [float(v) for v in b["bbox"][:4]]
            w, h = x2 - x1, y2 - y1
            cx1, cx2 = int(max(0, x1 + 0.22 * w)), int(min(W, x2 - 0.22 * w))
            cy1, cy2 = int(max(0, y1 + 0.08 * h)), int(min(H, y1 + 0.55 * h))
            if cx2 - cx1 < CROP_MIN_PX or cy2 - cy1 < CROP_MIN_PX:
                continue
            count[t] = count.get(t, 0) + 1
            out.add((int(row["frame"]), t))
    return out


def window_rows(P, game, w, comp, ctx) -> tuple:
    """A copy of tools.p8_teamcause.window_rows (the scorer's pairing on the same inputs), extended with the SportVU id,
    the fragment tid, the identity-pair flag, the crop flag and the capped-distance pairing diagnostic (9.7)."""
    import cv2
    from scipy.optimize import linear_sum_assignment
    from sportvu import bench as B
    from sportvu.sync import apply_mirror
    from sportvu.truth import _match
    from tools import p8_scorecard as S
    from tools.p8_teamcause import bin_of, read_json
    s = w["window"]
    side = read_json(comp / f"{s}_frames.json")
    srow = {r["frame"]: r for r in side["frames"]}
    tm = read_json(P["label_build"] / f"{s}_timemap.json")
    ident = read_json(comp / f"{s}_identity.json")
    idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
    team_of = {int(k): v for k, v in ident.get("team_by_track", {}).items()}
    fps = float(tm["fps"])
    live = sorted({int(f) for f, _, _ in tm["frames"]})
    rows, _off = S.truth_rows(P, game, w, "camera", ctx["mom"].index, ctx["mirror"])
    cache = read_json(P["cache"] / "scorecard" / f"{s}.json")
    wide = {int(k): v for k, v in cache["wide"].items()}
    lw = {f for f in live if wide.get(f, 0.0) >= ctx["refs"].wide_thr}
    rows, _np = S.universe(rows, lw, srow)
    cap = cv2.VideoCapture(str(P["label_build"] / f"{s}.mp4"))
    Wd, Hd = int(cap.get(3)), int(cap.get(4))
    cap.release()
    crops = crop_frames(side, Wd, Hd)
    tuples, out, capped_changed = [], [], 0
    for row in rows:
        f, q, i = row["frame"], row["q"], row["moment"]
        dq = ctx["mom"].index.q[q]
        sv = apply_mirror(dq["xy"][i], ctx["fx"], ctx["fy"]); pid = dq["pid"][i]; team = dq["team"][i]
        Ht = np.array(row["H_truth"], np.float64).reshape(3, 3)
        boxes = srow[f]["boxes"]
        feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
        ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2) if len(feet) else \
            np.zeros((0, 2), np.float32)
        pairs = _match(ct, sv, B.MATCH_FT) if len(ct) else []
        if len(ct):                                    # 9.2's diagnostic: distances capped before the assignment
            D = np.sqrt(((ct[:, None, :] - sv[None, :, :]) ** 2).sum(-1))
            ri, ci = linear_sum_assignment(np.where(D <= B.MATCH_FT, D, 1e6))
            alt = {(int(a), int(b)) for a, b in zip(ri, ci) if D[a, b] <= B.MATCH_FT}
            capped_changed += len({(bi, j) for bi, j, _ in pairs} ^ alt)
        cr = S.crowding(sv)
        for bi, j, dist in pairs:
            b = boxes[bi]
            cid = idmap.get(b["tid"], b["tid"])
            lab = team_of.get(cid)
            if lab is not None:
                tuples.append((cid, lab, int(team[j])))
            out.append({"frame": int(f), "cid": int(cid), "tid": int(b["tid"]), "label": lab, "team": int(team[j]),
                        "pid": int(pid[j]), "id_pair": bool(dist <= B.ID_MATCH_FT),
                        "crop": (int(f), int(b["tid"])) in crops,
                        "crowd_sv": bin_of(float(cr[j]), (3.0, 6.0), ("<3", "3-6", ">6"))})
    crop_count = {}
    for (f_, t_) in crops:
        c_ = idmap.get(t_, t_)
        crop_count[c_] = crop_count.get(c_, 0) + 1
    abst = {c for c, v in team_of.items() if v is None}
    return tuples, out, {"fps": fps, "capped_pairs_changed": capped_changed,
                         "abstained_tracks": len(abst),
                         "abstained_crop_less": sum(1 for c in abst if crop_count.get(c, 0) < MIN_CROP_FRAMES),
                         "abstained_rows": sum(1 for r in out if r["label"] is None)}


def collect(game: str) -> dict:
    refuse_game(game)
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_pin
    from tools import p8_piibox as PB
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_e3 import window_team
    from tools.p8_placecause import cache_gaps
    from tools.p8_teamcause import D202_SHA, RUN_ID
    from export.schemas import find_private
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    head = PB.git_head()
    if not head["tools_and_sportvu_clean"]:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    pinned, now = RB.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if any(now[k] != pinned.get(k) for k in now):
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    recp = r["ref"] / "rebaseline" / f"score__{game}__current.json"
    if RB.sha256(recp) != D202_SHA[game]:
        raise SystemExit(f"{WHO}: {recp.name} is not the pinned D202 record")
    rec = RB.read_json(recp)
    if rec.get("run_id") != RUN_ID:
        raise SystemExit(f"{WHO}: {recp.name} is not run {RUN_ID}")
    watched = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    mirror = F.mirror_of(P, game)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    ctx = {"mom": mom, "mirror": mirror, "refs": refs, "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
    comp = P["builds"] / "r23"
    P_t = {**P, "labels": RB.truths(r)[game]["current"] / "labels"}
    wins = sorted((w for w in F.label_windows(P, game) if w["window"] in rec["windows"]), key=lambda w: w["window"])
    stems = [w["window"] for w in wins]
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    if stems != sorted(rec["windows"]):
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's record")
    out_w = {}
    for w in wins:
        s = w["window"]
        if time.time() - t0 > CAP_S:
            raise SystemExit(f"{WHO}: the 15-minute cap was reached; nothing is reported")
        why = cache_gaps(P_t, game, w, comp, mom, mirror)
        if why:
            raise SystemExit(f"{WHO}: {s}: the scorecard cache lacks {why}")
        full = S.score_window(P_t, game, w, comp, "camera", mom, mirror, refs, events)
        if "skipped" in full[0]:
            raise SystemExit(f"{WHO}: {s} skipped: {full[0]['skipped']}")
        tuples, rws, info = window_rows(P_t, game, w, comp, ctx)
        mine, want = window_team(tuples), rec["windows"][s]["team"]
        if [tuple(t) for t in full[0]["team_rows"]] != tuples or \
                {k: mine[k] for k in GATE_COUNT} != {k: want[k] for k in GATE_COUNT}:
            raise SystemExit(f"{WHO}: the gate fails on {s}; nothing is reported")
        if any(len({x["label"] for x in rws if x["cid"] == c}) > 1 for c in {x["cid"] for x in rws}):
            raise SystemExit(f"{WHO}: {s}: a canonical track carries two labels (section 2 says it cannot)")
        out_w[s] = {**info, "rows": rws}
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    out = {"game": game, "design": "docs/research/P8_teambound.md versions 1 and 1.1", "git": head,
           "d202": {"run_id": RUN_ID, "sha256": D202_SHA[game]}, "windows": out_w,
           "manifest_unchanged": after == watched, "seconds": round(time.time() - t0, 1)}
    text = json.dumps(out)
    if find_private(text):
        raise SystemExit(f"{WHO}: the game record holds private data")
    dd = r["ref"] / "teambound"
    dd.mkdir(parents=True, exist_ok=True)
    atomic_write(dd / f"{game}.json", text)
    print(json.dumps({k: out[k] for k in ("game", "manifest_unchanged", "seconds")} | {"windows": len(out_w)}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
ORACLES = ("current", "O1", "O1f", "O1c", "O3", "O2_all", "O2_swap")


def game_tables(rec: dict, L: float = L_RULE) -> dict:
    per_w = {}
    for s, w in rec["windows"].items():
        o = window_oracles(w["rows"], w["fps"], L)
        per_w[s] = o
    n = sum(o["n"] for o in per_w.values())
    acc = {k: (sum(sum(o[k]) for o in per_w.values() if o["n"]) / n if n else None) for k in ORACLES}
    na = sum(o["O1_with_abstained"]["n"] for o in per_w.values() if o["n"])
    acc["O1_with_abstained_report_only"] = (sum(o["O1_with_abstained"]["correct"] for o in per_w.values() if o["n"]) / na
                                            if na else None)
    return {"per_window": per_w, "n": n, "acc": acc}


def boot_gain(per_w: dict, a: str, b: str, n: int = N_BOOT, seed: int = SEED) -> list | None:
    keys = [k for k, o in per_w.items() if o["n"]]
    if len(keys) < 2:
        return None
    g = np.array([[sum(per_w[k][a]) - sum(per_w[k][b]), per_w[k]["n"]] for k in keys], float)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        idx = rng.integers(0, len(g), len(g))
        v.append(g[idx, 0].sum() / g[idx, 1].sum())
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bin_split(rec: dict, per_w: dict) -> dict:
    """9.3: per crowding bin, tracking = O1's leftover errors, colour = current errors - O1's errors."""
    out = {}
    for s, w in rec["windows"].items():
        lab = [r for r in w["rows"] if r["label"] is not None]
        o = per_w[s]
        if not o["n"]:
            continue
        for r, cu, o1 in zip(lab, o["current"], o["O1"]):
            b = out.setdefault(r["crowd_sv"], {"rows": 0, "errors": 0, "tracking": 0, "colour": 0})
            b["rows"] += 1
            b["errors"] += not cu
            b["tracking"] += not o1
            b["colour"] += (not cu) - (not o1)
    return out


def purity_bands(rec: dict) -> dict:
    out = {"1.0": [0, 0], "0.9-1.0": [0, 0], "0.5-0.9": [0, 0]}
    for w in rec["windows"].values():
        lab = [r for r in w["rows"] if r["label"] is not None]
        if not lab:
            continue
        ok = correct([(r["cid"], r["label"], r["team"]) for r in lab])
        by_c = {}
        for r, c in zip(lab, ok):
            by_c.setdefault(r["cid"], []).append((r["team"], c))
        for c, rr in by_c.items():
            t = [x[0] for x in rr]
            pur = t.count(track_majority(t)) / len(t)
            band = "1.0" if pur == 1.0 else ("0.9-1.0" if pur >= 0.9 else "0.5-0.9")
            out[band][0] += len(rr)
            out[band][1] += sum(1 for x in rr if not x[1])
    return {k: {"rows": v[0], "errors": v[1]} for k, v in out.items()}


def report() -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    r = RB.roots()
    recs = {g: RB.read_json(r["ref"] / "teambound" / f"{g}.json") for g in POOL}
    res = {g: game_tables(recs[g]) for g in POOL}
    gains = {g: {"colour": res[g]["acc"]["O1"] - res[g]["acc"]["current"],
                 "tracking": res[g]["acc"]["O2_swap"] - res[g]["acc"]["O1"],
                 "combined": res[g]["acc"]["O2_swap"] - res[g]["acc"]["current"]} for g in POOL}
    pairs = {"colour": ("O1", "current"), "tracking": ("O2_swap", "O1"), "combined": ("O2_swap", "current")}
    mdes, bars = {}, {}
    for k, (a, b) in pairs.items():
        gw = [(sum(res[g]["per_window"][s][a]) - sum(res[g]["per_window"][s][b]), res[g]["per_window"][s]["n"])
              for g in POOL for s in res[g]["per_window"] if res[g]["per_window"][s]["n"]]
        mdes[k] = mde(gw)
        bars[k] = max(BAR_MIN, mdes[k] / F_RECOVERY)
    reading = pattern(gains, bars)
    fragility = {}
    for L in L_GRID:
        rl = {g: game_tables(recs[g], L) for g in POOL}
        gl = {g: {"colour": rl[g]["acc"]["O1"] - rl[g]["acc"]["current"],
                  "tracking": rl[g]["acc"]["O2_swap"] - rl[g]["acc"]["O1"],
                  "combined": rl[g]["acc"]["O2_swap"] - rl[g]["acc"]["current"]} for g in POOL}
        fragility[str(L)] = {"reading": pattern(gl, bars), "gains": gl,
                             "O2_swap": {g: rl[g]["acc"]["O2_swap"] for g in POOL},
                             "O2_all": {g: rl[g]["acc"]["O2_all"] for g in POOL}}
    fragile = len({v["reading"] for v in fragility.values()}) > 1
    per_game = {}
    for g in POOL:
        pw = res[g]["per_window"]
        tags = [t for o in pw.values() if o["n"] for t in o["cut_tags"]]
        per_game[g] = {
            "rows": res[g]["n"], "accuracy": res[g]["acc"],
            "gains": gains[g], "gain_intervals": {k: boot_gain(pw, a, b) for k, (a, b) in pairs.items()},
            "O2_all_minus_O1": res[g]["acc"]["O2_all"] - res[g]["acc"]["O1"],
            "gap_to_0_95_under_O2_swap": 0.95 - res[g]["acc"]["O2_swap"],
            "cuts": {"swap": sum(t["tag"] == "swap" for t in tags), "orphan": sum(t["tag"] == "orphan" for t in tags),
                     "on_fragment_boundary": sum(t["fragment_boundary"] for t in tags)},
            "bin_split_by_crowding": bin_split(recs[g], pw), "purity_bands": purity_bands(recs[g]),
            "abstention": {k: sum(w[k] for w in recs[g]["windows"].values())
                           for k in ("abstained_tracks", "abstained_crop_less", "abstained_rows")},
            "capped_pairs_changed_diagnostic": sum(w["capped_pairs_changed"] for w in recs[g]["windows"].values()),
            "git": recs[g]["git"], "seconds": recs[g]["seconds"]}
    out = {"what": "docs/research/P8_teambound.md versions 1 and 1.1; report only; oracles use the truth",
           "per_game": per_game, "mde_one_game_93_windows": mdes, "bars": bars, "recovery_fraction": F_RECOVERY,
           "reading": reading, "fragile_between_0_5_and_2_0_s": fragile, "fragility": fragility,
           "prior_stated_before_the_run": "colour headroom on gsw_cha_2016, tracking headroom on gsw_bkn_2015"}
    res_c = clean(out)
    text = json.dumps(res_c, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD, text)
    print(json.dumps({"reading": res_c["reading"], "fragile": res_c["fragile_between_0_5_and_2_0_s"],
                      "bars": res_c["bars"], "gains": {g: res_c["per_game"][g]["gains"] for g in POOL}}, indent=1))
    return res_c


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("game")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        refuse_game(a.game)
        collect(a.game)
    else:
        report()


if __name__ == "__main__":
    sys.exit(main())
