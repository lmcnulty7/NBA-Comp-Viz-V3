"""tools/p8_piiside.py: is the own-box P-ii row partly a truth-side error? (docs/research/P8_piiside.md, versions 1.1 and
1.2 with 11.6's asymmetric reading). Report only, stored files only.

    <frozen CV python> -m tools.p8_piiside collect GAME   inside the lock ("p8 piiside GAME"), V3_P8_ROOT = work_ref
    python3 -m tools.p8_piiside report                    -> reports/research/p8_piiside.json

collect: D207's selection recomputed with tools.p8_piibox's functions and gated per window on D202's counts and D207's
ordered (kind, class, height) rows. On every scored frame with an accepted B: the estimation rows (in-frame players
matched one to one to the non-referee boxes' B(foot_raw) within 5 ft, D205's assign_1to1 at 5 ft on B's feet) with the
SportVU velocity over plus or minus 0.2 s (mirror applied), the offset vectors of foot_raw and foot_stab under B, the
offsets against SportVU read 3 moments later and earlier, the third and the frame state; and, for D207's own P-ii and hit
rows, T(foot_raw) - B(foot_raw), B(foot_raw) - SportVU and T(foot_raw) - SportVU. hou_cle_2016 is refused before any
file is read. Writes only work_ref/piiside/<game>.json and the report.
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

WHO = "tools.p8_piiside"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
POOL = (CHA, BKN)
OWN = ("own", "own, with crossings", "own, misplaced, with crossings", "own, misplaced")
MISS_FT, MATCH_FT = 2.5, 5.0
SPEED_DT_MOMENTS = 5                      # tools.p8_scorecard: speed_dt_s 0.2 at 25 Hz
SHIFT_MOMENTS = 3                         # 11.2: plus and minus 0.12 s
MOVING, SLOW_LO, DIR_MIN = 5.0, 1.0, 1.0  # ft/s
MIN_MOVING_ROWS, MIN_SLOW_WINDOW = 30, 10
OFF_S, MIN_OFF_WINDOWS = 0.1, 5
ENRICH, CAM_RATIO = 1.5, 1.5
SHIFT_TOL_S, POWER_BAR, POWER_SHIFT_S, POWER_FRAC = 0.02, 0.8, 0.1, 0.25
LATERAL_MIN = 5.0
EMA_LAG_S = 0.067
N_SIM, SEED, SEED_POWER = 2000, 0, 1
CAP_S = 15 * 60
RECORD = REPO / "reports" / "research" / "p8_piiside.json"
D207_FILES = {CHA: "gsw_cha_2016.json", BKN: "gsw_bkn_2015.json"}
LIMIT = ("not shown; B is not independent of SportVU on these games (r23 trained in sample on R1.3 labels written "
         "through H_truth; ROADMAP_LOG.md pass 13, sportvu/grid_dataset.py, tools.p8_frozen.R13_TAGS, the scorecard's "
         "IN SAMPLE flag); never read as evidence that the row is build-side")


def refuse_game(game: str) -> None:
    if game not in POOL:
        raise SystemExit(f"{WHO}: {game} is refused; this item reads gsw_cha_2016 and gsw_bkn_2015 only")


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def velocity(q_xy_a, q_xy_b, clock_a: float, clock_b: float):
    """(vx, vy) in ft/s from two positions; the game clock counts down, so dt = clock_a - clock_b."""
    dt = clock_a - clock_b
    if dt <= 0:
        return None
    return (np.asarray(q_xy_b, float) - np.asarray(q_xy_a, float)) / dt


def along(o, v) -> float | None:
    """o . v_hat / |v| in seconds (the time offset a pure along-velocity displacement implies)."""
    sp = float(np.hypot(*v))
    if sp <= 0:
        return None
    return float(np.dot(o, v) / sp / sp)


def window_tau(rows: list, key: str = "a_raw") -> float | None:
    """tau_B of one window: the median of a key over its moving rows; None with fewer than MIN_MOVING_ROWS."""
    v = [r[key] for r in rows if r["speed"] >= MOVING and r.get(key) is not None]
    return float(np.median(v)) if len(v) >= MIN_MOVING_ROWS else None


def iqr(x) -> float:
    x = np.asarray([v for v in x if v is not None], float)
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else float("nan")


def simulate_null(wins: dict, tau0: float, game_slow: dict, n: int = N_SIM, seed: int = SEED,
                  inject: tuple | None = None) -> np.ndarray:
    """11.1: per simulation, each window's moving rows get tau0 x speed + (a slow row's offset vector . v_hat), the slow
    row drawn from the same window (at least MIN_SLOW_WINDOW slow rows) or else the game's slow rows of the same third;
    the window's simulated tau is the median of along / speed; the statistic is the across-window IQR. inject: (share,
    shift_s, seed) adds +-shift_s to a random share of windows in each simulation (11.2's power check)."""
    rng = np.random.default_rng(seed)
    irng = np.random.default_rng(inject[2]) if inject else None
    keys = sorted(wins)
    sims = np.zeros((n, len(keys)))
    for wi, k in enumerate(keys):
        w = wins[k]
        mv = [r for r in w["rows"] if r["speed"] >= MOVING]
        slow = np.asarray([r["o_raw"] for r in w["rows"] if SLOW_LO <= r["speed"] < MOVING], float).reshape(-1, 2)
        sp = np.asarray([r["speed"] for r in mv], float)
        vh = np.asarray([np.asarray(r["v"]) / r["speed"] for r in mv], float).reshape(-1, 2)
        if len(slow) >= MIN_SLOW_WINDOW:
            pools = [slow] * len(mv)
        else:
            pools = [np.asarray(game_slow.get(r["third"], []), float).reshape(-1, 2) for r in mv]
        draws = np.zeros((n, len(mv)))
        for j, pool in enumerate(pools):
            if not len(pool):
                draws[:, j] = np.nan
                continue
            idx = rng.integers(0, len(pool), n)
            draws[:, j] = pool[idx] @ vh[j]
        along_s = tau0 + draws / sp[None, :]
        sims[:, wi] = np.nanmedian(along_s, axis=1) if len(mv) else np.nan
    if inject:
        share, shift, _ = inject
        m = max(1, int(round(share * len(keys))))
        for s in range(n):
            pick = irng.choice(len(keys), m, replace=False)
            sims[s, pick] += shift * irng.choice([-1.0, 1.0], m)
    return np.array([np.nanpercentile(r, 75) - np.nanpercentile(r, 25) for r in sims])


def timing_reading(obs_iqr: float, null_iqr: np.ndarray, power: float, shift_ok: bool, n_off: int, enrich: dict) -> dict:
    """11.2, 11.4 and 11.6 on one game."""
    q975 = float(np.percentile(null_iqr, 97.5))
    out = {"iqr_observed": obs_iqr, "iqr_null_97_5": q975, "power": power, "shift_check": shift_ok, "off_windows": n_off}
    if not shift_ok:
        return {**out, "reading": "not resolvable (the synthetic shift was not recovered)"}
    if obs_iqr <= q975:
        if power >= POWER_BAR:
            return {**out, "reading": "answered: no window timing error beyond B's noise; " + LIMIT}
        return {**out, "reading": "not resolvable at this noise (power under 0.8)"}
    if n_off < MIN_OFF_WINDOWS:
        return {**out, "reading": "not resolvable (fewer than 5 off windows)"}
    found = enrich["value"] is not None and enrich["interval"] is not None and enrich["value"] >= ENRICH \
        and enrich["interval"][0] > 1
    return {**out, "enrichment": enrich,
            "reading": "timing found (read conservatively)" if found else "timing not shown; " + LIMIT}


def weighted_median(values, weights) -> float | None:
    pairs = sorted((v, w) for v, w in zip(values, weights) if v is not None and w > 0)
    if not pairs:
        return None
    tot, acc = sum(w for _, w in pairs), 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return float(v)
    return float(pairs[-1][0])


def camera_ratio(own_rows: list, hit_rows: list) -> float | None:
    """11.3 (a): within each third, the median d_TB of own P-ii rows over that of hits on the same frames; the
    per-third ratios combined as their median weighted by own rows."""
    ratios, weights = [], []
    for t in ("near", "middle", "far"):
        o = [r["d_TB"] for r in own_rows if r["third"] == t]
        h = [r["d_TB"] for r in hit_rows if r["third"] == t]
        if o and h and np.median(h) > 0:
            ratios.append(float(np.median(o) / np.median(h)))
            weights.append(len(o))
    return weighted_median(ratios, weights)


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def collect(game: str) -> dict:
    refuse_game(game)
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.sync import MIRRORS, apply_mirror
    from tools import p8_frozen as F
    from tools import p8_one2one as O
    from tools import p8_piibox as PB
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_e1_score import window_stats
    from tools.p8_placecause import cache_gaps
    from tools.p8_resetfix import one_to_one_missed, unmatched_1to1
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
    score = RB.read_json(r["ref"] / "rebaseline" / f"score__{game}__current.json")
    d207 = RB.read_json(r["ref"] / "piibox" / D207_FILES[game])
    want = {}
    for x in d207["rows"]:
        want.setdefault(x["wi"], []).append((x["kind"], x["class"], x["h"]))
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
    if sorted(score["windows"]) != stems or d207.get("windows") != len(stems):
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's or D207's records")

    def sv_at(d, i, pid):
        j = np.where(d["pid"][i] == pid)[0]
        return None if not len(j) else apply_mirror(np.asarray(d["xy"][i][j[0]], float).reshape(1, 2), fx, fy)[0]

    out_w, counters = {}, {"no_B_frames": 0, "no_velocity": 0, "no_shift": 0}
    for wi, w in enumerate(wins):
        s = w["window"]
        if time.time() - t0 > CAP_S:
            raise SystemExit(f"{WHO}: the 15-minute cap was reached; nothing is reported")
        for key, p in (("sidecar", comp / f"{s}_frames.json"), ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                       ("labels", P_t["labels"] / game / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
            wh = (score.get("hashes") or {}).get(s, {}).get(key)
            if wh is None and key in ("sidecar", "timemap", "labels"):
                raise SystemExit(f"{WHO}: {s} has no {key} hash in D202's record")
            if wh is not None and RB.sha256(p) != wh:
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
        ident = json.loads((comp / f"{s}_identity.json").read_text()) if (comp / f"{s}_identity.json").exists() else {}
        idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
        cache = json.loads((P_t["cache"] / "scorecard" / f"{s}.json").read_text())
        refd = {int(k): v for k, v in cache.get("referees", {}).items()}
        tm = json.loads((P_t["label_build"] / f"{s}_timemap.json").read_text())
        fps = float(tm["fps"])
        cap = cv2.VideoCapture(str(P_t["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        truth, _ = S.truth_rows(P_t, game, w, "camera", mom.index, mirror)
        c = {k: 0 for k in ("in_frame", "missed", "far", "missed_1to1", "unmatched_1to1")}
        frames, sv_of, items, est, ctx, offset_s, resid = [], {}, [], [], {}, None, []
        for row in truth:
            f = int(row["frame"])
            x = srow.get(f)
            if x is None or f not in scored:
                continue
            q, i = int(row["q"]), int(row["moment"])
            d = mom.index.q[q]
            sv = np.asarray(apply_mirror(d["xy"][i], fx, fy), np.float64)
            pids = [int(v) for v in d["pid"][i]]
            Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
            px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
            m = FRAME_MARGIN_PX
            inside = (px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)
            stab = [b["foot_stab"] for b in x["boxes"]]
            ft = map_px(Ht, stab) if stab else np.zeros((0, 2))
            is_ref = np.array([(b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else
                               any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
                               for b in x["boxes"]], bool)
            svf = np.asarray(sv, np.float64)
            c["unmatched_1to1"] += unmatched_1to1(ft, svf, inside, is_ref, 2.5)
            c["missed_1to1"] += one_to_one_missed(ft, svf, inside, 2.5)
            Dm = np.sqrt(((svf[:, None, :] - ft[None, :, :]) ** 2).sum(-1)).min(1) if len(ft) else np.full(len(svf), np.inf)
            c["in_frame"] += int(inside.sum())
            c["missed"] += int((inside & (Dm > MISS_FT)).sum())
            c["far"] += int((inside & (Dm > MATCH_FT)).sum())
            cls = O.frame_classes(ft, svf, inside, is_ref)
            nonref = [int(b) for b in np.where(~is_ref)[0]]
            main, _ = O.two_stage(ft, svf, range(len(svf)), nonref, O._assign_all)
            cids = [idmap.get(int(b["tid"]), int(b["tid"])) for b in x["boxes"]]
            frames.append({"f": f, "cid": cids, "A": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "A"},
                           "B": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "B"},
                           "ref": list(is_ref), "ft": ft})
            sv_of[f] = {pids[j]: svf[j] for j in range(len(svf))}
            picked, _ = PB.select_items(cls, main)
            offset_s = row.get("offset_s", offset_s)
            if row.get("feet_resid_ft") is not None:
                resid.append(float(row["feet_resid_ft"]))
            Hb = np.asarray(x["H"], np.float64).reshape(3, 3) if B.accepted(x) and x.get("H") is not None else None
            third = {j: S.third_of(float(svf[j][1]), S_near_side(Ht)) for j in range(len(svf))}
            ctx[f] = {"x": x, "Ht": Ht, "Hb": Hb, "sv": svf, "third": third, "resid": row.get("feet_resid_ft")}
            for j, kind, bk in picked:
                items.append({"kind": kind, "f": f, "pid": pids[j], "cid": cids[bk], "j": j, "bk": bk,
                              "h": float(x["boxes"][bk]["bbox"][3] - x["boxes"][bk]["bbox"][1])})
            if Hb is None:
                counters["no_B_frames"] += 1
                continue
            # estimation rows: one to one under B within 5 ft, in-frame players, T not involved
            raw = map_px(Hb, [b["foot_raw"] for b in x["boxes"]]) if x["boxes"] else np.zeros((0, 2))
            stb = map_px(Hb, stab) if stab else np.zeros((0, 2))
            pairs = O.assign_1to1(raw, svf, is_ref, MATCH_FT)
            G = np.asarray(row["camera_C"][:2], float)
            k = SPEED_DT_MOMENTS
            for j, (bk, dist) in pairs.items():
                if not inside[j]:
                    continue
                a, b_ = max(0, i - k), min(len(d["clock"]) - 1, i + k)
                pa, pb = sv_at(d, a, pids[j]), sv_at(d, b_, pids[j])
                v = velocity(pa, pb, d["clock"][a], d["clock"][b_]) if pa is not None and pb is not None else None
                if v is None:
                    counters["no_velocity"] += 1
                    continue
                speed = float(np.hypot(*v))
                o_raw, o_stab = raw[bk] - svf[j], stb[bk] - svf[j]
                rec = {"f": f, "speed": speed, "v": [float(v[0]), float(v[1])],
                       "o_raw": [float(o_raw[0]), float(o_raw[1])], "third": third[j],
                       "state": x.get("state"), "a_raw": None, "a_stab": None, "a_plus": None, "a_minus": None,
                       "a_pred_ema": None, "lat": None}
                if speed >= DIR_MIN:
                    rec["a_raw"], rec["a_stab"] = along(o_raw, v), along(o_stab, v)
                    rec["a_pred_ema"] = along(stb[bk] - raw[bk], v)
                    for name, sh in (("a_plus", SHIFT_MOMENTS), ("a_minus", -SHIFT_MOMENTS)):
                        ii = i + sh
                        p_sh = sv_at(d, ii, pids[j]) if 0 <= ii < len(d["clock"]) else None
                        if p_sh is None:
                            counters["no_shift"] += 1
                        else:
                            rec[name] = along(raw[bk] - p_sh, v)
                    u = svf[j] - G
                    u = u / max(np.hypot(*u), 1e-9)
                    wv = np.array([-u[1], u[0]])
                    vw = float(np.dot(v, wv))
                    if abs(vw) >= LATERAL_MIN:
                        rec["lat"] = float(np.dot(o_raw, wv) / vw)
                est.append(rec)
        wd = score["windows"][s]
        mine = {**c, "nr_boxes": st["nr_boxes"]}
        if any(mine[k_] != wd.get(k_) for k_ in mine) or st["in_frame"] != c["in_frame"] or st["missed"] != c["missed"]:
            raise SystemExit(f"{WHO}: D202 count gate failed on {s}")
        tracks = PB.track_table(frames)
        got, outc = [], []
        for it in items:
            ev = PB.evidence(tracks[it["cid"]], it["pid"], it["f"], sv_of, fps)
            it["class"] = PB.classify(ev["n_own"], ev["n_other"], ev["n_ownB"], ev["referee_flagged"], ev["present"])
            got.append((it["kind"], it["class"], it["h"]))
            cx = ctx[it["f"]]
            raw_px = cx["x"]["boxes"][it["bk"]]["foot_raw"]
            qT = map_px(cx["Ht"], [raw_px])[0]
            rec = {"kind": it["kind"], "class": it["class"], "f": it["f"], "third": cx["third"][it["j"]],
                   "resid": cx["resid"], "o_T": [float(v_) for v_ in qT - cx["sv"][it["j"]]],
                   "d_TB": None, "d_B": None, "tb": None}
            if cx["Hb"] is not None:
                qB = map_px(cx["Hb"], [raw_px])[0]
                rec["d_TB"] = float(np.hypot(*(qT - qB)))
                rec["d_B"] = float(np.hypot(*(qB - cx["sv"][it["j"]])))
                rec["tb"] = [float(v_) for v_ in qT - qB]
            outc.append(rec)
        if got != want.get(wi, []):
            raise SystemExit(f"{WHO}: D207 gate failed on {s}")
        out_w[s] = {"wi": wi, "in_frame": wd["in_frame"], "offset_s": offset_s,
                    "median_resid_ft": float(np.median(resid)) if resid else None, "rows": est, "outcome": outc}
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    out = {"game": game, "design": "docs/research/P8_piiside.md versions 1.1 and 1.2 with 11.6", "git": head,
           "windows": out_w, "counters": counters, "manifest_unchanged": after == watched,
           "seconds": round(time.time() - t0, 1)}
    dd = r["ref"] / "piiside"
    dd.mkdir(parents=True, exist_ok=True)
    atomic_write(dd / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "counters", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


def S_near_side(Ht):
    from sportvu.eval import near_side
    return near_side(Ht)


def map_px(H, px) -> np.ndarray:
    """Pixels to the court as the scorer maps foot_stab: float32 pixels through a float64 H."""
    import cv2
    p = np.asarray(px, np.float32).reshape(-1, 1, 2)
    if not len(p):
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2).astype(np.float64)


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def boot_ratio(off: list, rest: list, n: int = N_SIM, seed: int = SEED) -> dict:
    """The own P-ii rate in off windows over the rate in the rest; windows resampled whole within each group.
    off/rest: [(own P-ii rows, in-frame players)] per window."""
    def rate(ws):
        a = np.asarray(ws, float).reshape(-1, 2)
        return a[:, 0].sum() / a[:, 1].sum() if len(a) and a[:, 1].sum() else None
    pt_o, pt_r = rate(off), rate(rest)
    pt = pt_o / pt_r if pt_o is not None and pt_r else None
    rng = np.random.default_rng(seed)
    reps = []
    if len(off) >= 2 and len(rest) >= 2:
        for _ in range(n):
            o = [off[i] for i in rng.integers(0, len(off), len(off))]
            r_ = [rest[i] for i in rng.integers(0, len(rest), len(rest))]
            a, b = rate(o), rate(r_)
            if a is not None and b:
                reps.append(a / b)
    return {"value": pt, "interval": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
            if reps else None, "rate_off": pt_o, "rate_rest": pt_r}


def spearman(a, b) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs, float).T
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def game_report(rec: dict) -> dict:
    wins = rec["windows"]
    tau = {k: window_tau(w["rows"]) for k, w in wins.items()}
    have = {k: v for k, v in tau.items() if v is not None}
    tau0 = float(np.median(list(have.values()))) if have else None
    obs = iqr(have.values())
    game_slow = {}
    for w in wins.values():
        for r in w["rows"]:
            if SLOW_LO <= r["speed"] < MOVING:
                game_slow.setdefault(r["third"], []).append(r["o_raw"])
    sim_w = {k: wins[k] for k in have}
    null = simulate_null(sim_w, tau0, game_slow) if have else np.array([np.nan])
    power_iqr = simulate_null(sim_w, tau0, game_slow, seed=SEED, inject=(POWER_FRAC, POWER_SHIFT_S, SEED_POWER)) \
        if have else np.array([np.nan])
    q975 = float(np.percentile(null, 97.5)) if have else float("nan")
    power = float(np.mean(power_iqr > q975)) if have else 0.0
    shifts = {}
    for key, expect in (("a_plus", -SHIFT_MOMENTS / 25.0), ("a_minus", SHIFT_MOMENTS / 25.0)):
        d = [window_tau(wins[k]["rows"], key) - tau[k] for k in have if window_tau(wins[k]["rows"], key) is not None]
        med = float(np.median(d)) if d else None
        shifts[key] = {"expected_s": expect, "median_move_s": med,
                       "ok": med is not None and abs(med - expect) <= SHIFT_TOL_S}
    shift_ok = all(v["ok"] for v in shifts.values())
    off = [k for k, v in have.items() if abs(v - tau0) >= OFF_S]
    own_rows = {k: sum(1 for r in w["outcome"] if r["kind"] == "P-ii" and r["class"] in OWN) for k, w in wins.items()}
    enrich = boot_ratio([(own_rows[k], wins[k]["in_frame"]) for k in off],
                        [(own_rows[k], wins[k]["in_frame"]) for k in have if k not in off])   # windows with a tau only
    timing = timing_reading(obs, null, power, shift_ok, len(off), enrich)
    ema = [window_tau(w["rows"], "a_stab") - tau[k] for k, w in wins.items()
           if tau[k] is not None and window_tau(w["rows"], "a_stab") is not None]
    ema_pred = [window_tau(w["rows"], "a_pred_ema") for w in wins.values() if window_tau(w["rows"], "a_pred_ema") is not None]
    track = {k: window_tau([r for r in w["rows"] if r["state"] == "TRACK"]) for k, w in wins.items()}
    lat = {k: (float(np.median([r["lat"] for r in w["rows"] if r["lat"] is not None]))
               if sum(r["lat"] is not None for r in w["rows"]) >= MIN_MOVING_ROWS else None) for k, w in wins.items()}
    # camera
    outc = [r for w in wins.values() for r in w["outcome"] if r["d_TB"] is not None]
    own = [r for r in outc if r["kind"] == "P-ii" and r["class"] in OWN]
    own_frames = {(r["f"]) for r in own}
    hits_same = [r for r in outc if r["kind"] == "hit" and r["f"] in own_frames]
    ratio = camera_ratio(own, hits_same)
    rng = np.random.default_rng(SEED)
    keys = sorted(wins)
    reps = []
    for _ in range(N_SIM):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        o_ = [r for k in pick for r in wins[k]["outcome"] if r["d_TB"] is not None and r["kind"] == "P-ii"
              and r["class"] in OWN]
        fr = {r["f"] for r in o_}
        h_ = [r for k in pick for r in wins[k]["outcome"] if r["d_TB"] is not None and r["kind"] == "hit" and r["f"] in fr]
        v = camera_ratio(o_, h_)
        if v is not None:
            reps.append(v)
    ratio_iv = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))] if reps else None
    share_own = float(np.mean([r["d_B"] <= MISS_FT for r in own])) if own else None
    hits_by_f = {}
    for r in outc:
        if r["kind"] == "hit":
            hits_by_f.setdefault(r["f"], []).append(r["tb"])
    rng2 = np.random.default_rng(SEED)
    own_n = [r for r in own if hits_by_f.get(r["f"])]
    null_shares = []
    for _ in range(N_SIM):
        ok = 0
        for r in own_n:
            tb = hits_by_f[r["f"]][int(rng2.integers(0, len(hits_by_f[r["f"]])))]
            ok += float(np.hypot(r["o_T"][0] - tb[0], r["o_T"][1] - tb[1])) <= MISS_FT
        null_shares.append(ok / len(own_n) if own_n else np.nan)
    share_own_n = float(np.mean([r["d_B"] <= MISS_FT for r in own_n])) if own_n else None
    null_q = float(np.nanpercentile(null_shares, 97.5)) if own_n else None
    hits_share = float(np.mean([r["d_B"] <= MISS_FT for r in outc if r["kind"] == "hit"])) if outc else None
    cam_a = ratio is not None and ratio_iv is not None and ratio >= CAM_RATIO and ratio_iv[0] > 1
    cam_b = share_own_n is not None and null_q is not None and share_own_n > null_q          # 11.3 (b)
    by_third = {}
    for t in ("near", "middle", "far"):
        o = [r for r in own if r["third"] == t]
        h = [r for r in hits_same if r["third"] == t]
        by_third[t] = {"own_n": len(o), "hits_n": len(h),
                       "median_dTB_own": float(np.median([r["d_TB"] for r in o])) if o else None,
                       "median_dTB_hits": float(np.median([r["d_TB"] for r in h])) if h else None,
                       "median_dB_own": float(np.median([r["d_B"] for r in o])) if o else None,
                       "median_dB_hits": float(np.median([r["d_B"] for r in h])) if h else None}
    camera = {"ratio_within_third": ratio, "ratio_interval": ratio_iv, "a": cam_a,
              "share_own_dB_within_2_5": share_own, "share_own_with_same_frame_hit": share_own_n,
              "null_share_97_5": null_q, "b": cam_b, "hits_dB_within_2_5_T_side_reference": hits_share,
              "own_rows": len(own), "own_rows_with_same_frame_hit": len(own_n), "by_third": by_third,
              "holds": bool(cam_a and cam_b)}
    in_frame = sum(w["in_frame"] for w in wins.values())
    own_total = sum(own_rows.values())
    off_set = set(off)
    in_off = sum(wins[k]["in_frame"] for k in off_set)
    rate_rest = enrich["rate_rest"] or 0.0
    timing_ub = max(0.0, (sum(own_rows[k] for k in off_set) - rate_rest * in_off)) / in_frame if in_frame else None
    camera_ub = sum(1 for r in own if r["d_B"] <= MISS_FT) / in_frame if in_frame else None
    own_rate = own_total / in_frame if in_frame else None
    bounds = {"own_P-ii_rate": own_rate, "timing_upper": timing_ub, "camera_upper": camera_ub,
              "neither_lower_bound_on_build_side": None if own_rate is None else own_rate - (timing_ub or 0) - (camera_ub or 0),
              "reading_of_the_lower_bound": LIMIT}
    resid = {}
    for lo, hi in ((0.0, 1.0), (1.0, 1.5), (1.5, 2.01)):
        rows = [r for w in wins.values() for r in w["outcome"] if r["resid"] is not None and lo <= r["resid"] < hi]
        resid[f"{lo}-{hi}"] = {"own_P-ii": sum(1 for r in rows if r["kind"] == "P-ii" and r["class"] in OWN),
                               "hits": sum(1 for r in rows if r["kind"] == "hit")}
    speed_ctl = {}
    for lo, hi, name in ((0.0, 5.0, "under 5"), (5.0, 10.0, "5 to 10"), (10.0, 1e9, "10 and over")):
        rr = [r for w in wins.values() for r in w["rows"] if lo <= r["speed"] < hi and r["a_raw"] is not None]
        speed_ctl[name] = {"n": len(rr),
                           "median_along_s_raw": float(np.median([r["a_raw"] for r in rr])) if rr else None,
                           "median_stab_minus_raw_s": float(np.median([r["a_stab"] - r["a_raw"] for r in rr
                                                                       if r["a_stab"] is not None])) if rr else None}
    return {"windows": len(wins), "windows_with_tau": len(have), "tau_0": tau0, "tau_by_window": tau,
            "timing": timing, "shift_check": shifts,
            "report_only": {"ema_difference_median_s": float(np.median(ema)) if ema else None,
                            "ema_predicted_from_pixel_offsets_median_s": float(np.median(ema_pred)) if ema_pred else None,
                            "ema_nominal_s": -EMA_LAG_S,
                            "tau_track_only_by_window": track, "lateral_estimator_by_window": lat,
                            "tau_against_offset_s_spearman": spearman([tau[k] for k in wins],
                                                                      [wins[k]["offset_s"] for k in wins]),
                            "own_P-ii_by_residual_bin_descriptive": resid, "build_side_speed_controls": speed_ctl},
            "camera": camera, "bounds": bounds, "counters": rec["counters"], "seconds": rec["seconds"],
            "git": rec["git"]}


def readings(per: dict) -> dict:
    """11.6's asymmetric reading over both games."""
    t = [per[g]["timing"]["reading"] for g in POOL]
    timing = "timing found (read conservatively)" if all(x.startswith("timing found") for x in t) else \
        ("not resolvable" if any(x.startswith("not resolvable") for x in t) else "timing not shown; " + LIMIT)
    camera = "camera found (read conservatively): T's camera is the likelier side at own P-ii boxes" \
        if all(per[g]["camera"]["holds"] for g in POOL) else "camera not shown; " + LIMIT
    neither = not timing.startswith("timing found") and not camera.startswith("camera found")
    return {"timing": timing, "camera": camera, "overall": ("neither: " + LIMIT) if neither else "a truth-side pattern found"}


def report() -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    r = RB.roots()
    recs = {g: RB.read_json(r["ref"] / "piiside" / f"{g}.json") for g in POOL}
    per = {g: game_report(recs[g]) for g in POOL}
    acc = {g: RB.read_json(r["ms"] / "truth_accept" / f"{g}.json")["checks"]["C"] for g in POOL}
    out = {"what": "docs/research/P8_piiside.md versions 1.1 and 1.2 with 11.6's asymmetric reading; report only",
           "limit": LIMIT, "per_game": per, "check_C_game_level_report_only": acc, "readings": readings(per),
           "depth_share_seen_in_review_not_a_test": {"own_P-ii": [0.764, 0.785], "hits": [0.727, 0.725]}}
    res = clean(out)
    text = json.dumps(res, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD, text)
    print(json.dumps({"readings": res["readings"], **{g: {"timing": res["per_game"][g]["timing"]["reading"],
                                                          "camera": res["per_game"][g]["camera"]["holds"]}
                                                      for g in POOL}}, indent=1))
    return res


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
