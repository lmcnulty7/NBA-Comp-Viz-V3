"""tools/p8_posefoot.py: a per-box foot from pose keypoints (docs/research/P8_posefoot.md, version 1.1). Report only.

Stage 0a only in this version (no download, no decode, stored files):

    <frozen CV python> -m tools.p8_posefoot stage0a GAME   inside the lock ("p8 posefoot 0a GAME"), V3_P8_ROOT = work_ref
    python3 -m tools.p8_posefoot report0a                   -> reports/research/p8_posefoot_0a.json

stage0a: D207's boxes recomputed with tools.p8_piibox's functions (every P-ii box and every classified hit box), gated
per window on D207's ordered (kind, class, height) rows and on D202's counts; on each box, the distance to its player
under T (the scorer's mapping) and under B (the sidecar's accepted H) of foot_stab, foot_raw (the per-frame box bottom)
and the same-run null (another box's foot_raw - foot_stab offset on the same frame, scaled by box height, added to this
box's foot_stab), with scorer-ghost flags and the player's SportVU speed. hou_cle_2016 is refused before any file is
read. Writes only work_ref/posefoot/0a/<game>.json and the report.
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

WHO = "tools.p8_posefoot"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
POOL = (CHA, BKN)
MISS_FT, GHOST_FT = 2.5, 3.0
OWN = ("own", "own, with crossings", "own, misplaced, with crossings", "own, misplaced")
SPEED_BINS = (5.0, 10.0)                  # ft/s: under 5, 5 to 10, 10 and over
BOOT_N, SEED = 2000, 0
# the registered rules (section 11.3, 11.4, 11.5)
NULL_MARGIN = 0.10                        # rule 1: R - R_null at least 0.10
MED_BAR_FT = -0.5                         # rule 2
HIT_MED_FT, HIT_LOST = 0.10, 0.05         # rule 4
EST_BAR = -0.010                          # rule 5
RECORD_0A = REPO / "reports" / "research" / "p8_posefoot_0a.json"
D207_RUN_FILES = {CHA: "gsw_cha_2016.json", BKN: "gsw_bkn_2015.json"}


def refuse_game(game: str) -> None:
    """Every stage refuses hou_cle_2016 (and any non-pool game) before any file is read."""
    if game not in POOL:
        raise SystemExit(f"{WHO}: {game} is refused; this item reads gsw_cha_2016 and gsw_bkn_2015 only")


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def map_px(H, px) -> np.ndarray:
    """Pixels to the court as the scorer maps foot_stab: float32 pixels through a float64 H."""
    import cv2
    p = np.asarray(px, np.float32).reshape(-1, 1, 2)
    if not len(p):
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2).astype(np.float64)


def null_partner(heights: list, b: int, ok: list) -> int | None:
    """The box on the frame, other than b, with ok[k] true, whose height is nearest b's (ties: lower index)."""
    best, bd = None, None
    for k, h in enumerate(heights):
        if k == b or not ok[k]:
            continue
        d = abs(h - heights[b])
        if bd is None or d < bd:
            best, bd = k, d
    return best


def null_foot(stab_b, off_k, h_b: float, h_k: float) -> list:
    """b's foot_stab plus k's offset scaled by h_b / h_k."""
    s = h_b / h_k if h_k > 0 else 1.0
    return [float(stab_b[0] + off_k[0] * s), float(stab_b[1] + off_k[1] * s)]


def speed_bin(v) -> str | None:
    if v is None:
        return None
    return "under 5" if v < SPEED_BINS[0] else ("5 to 10" if v < SPEED_BINS[1] else "10 and over")


def rule_set(rows_pii_own: dict, rows_hit: dict, est: dict, cand: str) -> dict:
    """The five rules on one game. rows_*: {"R": .., "R_null": .., "med_T": {value, interval}, "med_B": {value}, ...}."""
    r1 = rows_pii_own["R"] is not None and rows_pii_own["R_null"] is not None and \
        rows_pii_own["R"] - rows_pii_own["R_null"] >= NULL_MARGIN
    mt = rows_pii_own["med_T"]
    r2 = mt["value"] is not None and mt["interval"] is not None and mt["value"] <= MED_BAR_FT and mt["interval"][1] < 0
    r3 = rows_pii_own["med_B"]["value"] is not None and rows_pii_own["med_B"]["value"] <= 0
    r4 = rows_hit["med_T"]["value"] is not None and rows_hit["med_T"]["value"] <= HIT_MED_FT and \
        rows_hit["lost"] is not None and rows_hit["lost"] <= HIT_LOST
    r5 = est["value"] is not None and est["value"] <= EST_BAR
    return {"candidate": cand, "rule1_null_margin": r1, "rule2_median_T": r2, "rule3_median_B": r3,
            "rule4_hits": r4, "rule5_est": r5, "all": bool(r1 and r2 and r3 and r4 and r5)}


def readings_0a(rules: dict) -> dict:
    """Section 11.3's readings (c) and (d) from the per-game rule sets of foot_raw."""
    stab = all(v["rule1_null_margin"] and v["rule2_median_T"] and v["rule3_median_B"] and v["rule5_est"]
               for v in rules.values())
    hybrid_only = any(not v["rule4_hits"] for v in rules.values())
    return {"c_stabiliser_finding": stab, "d_pose_reads_hybrid_only": hybrid_only}


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def collect0a(game: str) -> dict:
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
    d207 = RB.read_json(r["ref"] / "piibox" / D207_RUN_FILES[game])
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
    want = {}
    for x in d207["rows"]:
        want.setdefault(x["wi"], []).append((x["kind"], x["class"], x["h"]))
    rows, counters = [], {"no_null_partner": 0, "no_B": 0}
    for wi, w in enumerate(wins):
        s = w["window"]
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
        frames, sv_of, items = [], {}, []
        for row in truth:
            f = int(row["frame"])
            x = srow.get(f)
            if x is None or f not in scored:
                continue
            d = mom.index.q[row["q"]]
            sv = apply_mirror(d["xy"][row["moment"]], fx, fy)
            pids = [int(v) for v in d["pid"][row["moment"]]]
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
            c["far"] += int((inside & (Dm > 5.0)).sum())
            cls = O.frame_classes(ft, svf, inside, is_ref)
            nonref = [int(b) for b in np.where(~is_ref)[0]]
            main, _ = O.two_stage(ft, svf, range(len(svf)), nonref, O._assign_all)
            cids = []
            for b in x["boxes"]:
                if b.get("tid") is None:
                    raise SystemExit(f"{WHO}: {s} frame {f}: a box has no track id")
                cids.append(idmap.get(int(b["tid"]), int(b["tid"])))
            frames.append({"f": f, "cid": cids, "A": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "A"},
                           "B": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "B"},
                           "ref": list(is_ref), "ft": ft})
            sv_of[f] = {pids[j]: svf[j] for j in range(len(svf))}
            picked, _why = PB.select_items(cls, main)
            Hb = np.asarray(x["H"], np.float64).reshape(3, 3) if B.accepted(x) and x.get("H") is not None else None
            hts = [float(b["bbox"][3] - b["bbox"][1]) for b in x["boxes"]]
            okn = [not bool(is_ref[k]) for k in range(len(x["boxes"]))]
            for j, kind, bk in picked:
                bx = x["boxes"][bk]
                k = null_partner(hts, bk, okn)
                nf = None
                if k is not None:
                    ok_ = x["boxes"][k]
                    off = [ok_["foot_raw"][0] - ok_["foot_stab"][0], ok_["foot_raw"][1] - ok_["foot_stab"][1]]
                    nf = null_foot(bx["foot_stab"], off, hts[bk], hts[k])
                feet = {"stab": bx["foot_stab"], "raw": bx["foot_raw"], "null": nf}
                dT, gT, dB = {}, {}, {}
                for name, p in feet.items():
                    if p is None:
                        dT[name] = gT[name] = dB[name] = None
                        continue
                    q = map_px(Ht, [p])[0]
                    dT[name] = float(np.hypot(*(q - svf[j])))
                    gT[name] = bool(np.sqrt(((svf - q) ** 2).sum(1)).min() > GHOST_FT)
                    dB[name] = float(np.hypot(*(map_px(Hb, [p])[0] - svf[j]))) if Hb is not None else None
                items.append({"wi": wi, "kind": kind, "f": f, "pid": pids[j], "cid": cids[bk], "h": hts[bk],
                              "dT": dT, "gT": gT, "dB": dB,
                              "speed": mom.speed(int(row["q"]), int(row["moment"]), pids[j])})
                counters["no_null_partner"] += k is None
                counters["no_B"] += Hb is None
        wd = score["windows"][s]
        mine = {**c, "nr_boxes": st["nr_boxes"]}
        if any(mine[k] != wd.get(k) for k in mine) or st["in_frame"] != c["in_frame"] or st["missed"] != c["missed"]:
            raise SystemExit(f"{WHO}: D202 count gate failed on {s}")
        tracks = PB.track_table(frames)
        got = []
        for it in items:
            ev = PB.evidence(tracks[it["cid"]], it["pid"], it["f"], sv_of, fps)
            it["class"] = PB.classify(ev["n_own"], ev["n_other"], ev["n_ownB"], ev["referee_flagged"], ev["present"])
            got.append((it["kind"], it["class"], it["h"]))
        if got != want.get(wi, []):
            raise SystemExit(f"{WHO}: D207 gate failed on {s}: the recomputed (kind, class, height) rows differ")
        for it in items:
            rows.append({k: it[k] for k in ("wi", "kind", "class", "h", "dT", "gT", "dB", "speed")})
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    out = {"game": game, "design": "docs/research/P8_posefoot.md version 1.1, stage 0a", "git": head,
           "windows": len(stems), "in_frame_by_window": [score["windows"][s]["in_frame"] for s in stems],
           "nr_boxes_by_window": [score["windows"][s]["nr_boxes"] for s in stems],
           "gate": {"d202": "six fields per window", "d207": "ordered (kind, class, height) rows per window",
                    "passes": True}, "counters": counters, "manifest_unchanged": after == watched,
           "seconds": round(time.time() - t0, 1), "rows": rows}
    dd = r["ref"] / "posefoot" / "0a"
    dd.mkdir(parents=True, exist_ok=True)
    atomic_write(dd / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "windows", "gate", "counters", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def unit_rows(per_game: dict, gs: list, cand: str = "raw", hit_all: dict | None = None) -> dict:
    """The rows of section 11.3 for a unit (one game or both), with window intervals."""
    from tools.p8_piibox import median_iv, per_window, picks_for, ratio_iv, values_by_window
    n_w = [per_game[g]["windows"] for g in gs]
    picks = picks_for(n_w)
    pii = [[x for x in per_game[g]["rows"] if x["kind"] == "P-ii"] for g in gs]
    own = [[x for x in rs if x["class"] in OWN] for rs in pii]
    hit = [[x for x in per_game[g]["rows"] if x["kind"] == "hit"] for g in gs]

    def share(rs_g, fn, need):
        return ratio_iv([per_window([x for x in rs if need(x)], nw, fn) for rs, nw in zip(rs_g, n_w)],
                        [per_window([x for x in rs if need(x)], nw, lambda x: 1) for rs, nw in zip(rs_g, n_w)], picks)

    def delta(rs_g, frame):
        vals = [values_by_window([{"wi": x["wi"], "v": x[frame][cand] - x[frame]["stab"]} for x in rs
                                  if x[frame][cand] is not None and x[frame]["stab"] is not None], nw, "v")
                for rs, nw in zip(rs_g, n_w)]
        return median_iv(vals, picks)

    within = lambda key: (lambda x: x["dT"][key] is not None and x["dT"][key] <= MISS_FT)
    R = share(own, within(cand), lambda x: x["dT"][cand] is not None)
    Rn = share(own, within("null"), lambda x: x["dT"]["null"] is not None)
    lost = share(hit, lambda x: x["dT"][cand] > MISS_FT, lambda x: x["dT"][cand] is not None)
    out = {"P-ii own": {"n": sum(len(x) for x in own), "R": R["value"], "R_interval": R["interval"],
                        "R_null": Rn["value"], "R_null_interval": Rn["interval"],
                        "med_T": delta(own, "dT"), "med_B": delta(own, "dB")},
           "P-ii all": {"n": sum(len(x) for x in pii), "R": share(pii, within(cand), lambda x: x["dT"][cand] is not None)},
           "hits": {"n": sum(len(x) for x in hit), "lost": lost["value"], "lost_interval": lost["interval"],
                    "med_T": delta(hit, "dT"), "med_B": delta(hit, "dB")}}
    # est and the ghost change (section 11.4), coverage 1 for foot_raw
    hits_all = sum(hit_all[g] for g in gs) if hit_all else None
    s_hit = hits_all / out["hits"]["n"] if hits_all and out["hits"]["n"] else 1.0
    num = [per_window(rs, nw, lambda x: -(x["dT"][cand] is not None and x["dT"][cand] <= MISS_FT))
           + s_hit * per_window(hs, nw, lambda x: x["dT"][cand] is not None and x["dT"][cand] > MISS_FT)
           for rs, hs, nw in zip(pii, hit, n_w)]
    den = [np.asarray(per_game[g]["in_frame_by_window"], float) for g in gs]
    out["est"] = ratio_iv(num, den, picks)
    gnum = [per_window(rs, nw, lambda x: int(bool(x["gT"][cand])) - int(bool(x["gT"]["stab"])))
            + s_hit * per_window(hs, nw, lambda x: int(bool(x["gT"][cand])) - int(bool(x["gT"]["stab"])))
            for rs, hs, nw in zip(pii, hit, n_w)]
    out["ghost_change_per_box"] = ratio_iv(gnum, [np.asarray(per_game[g]["nr_boxes_by_window"], float) for g in gs], picks)
    out["s_hit"] = s_hit
    sp = {}
    for b in ("under 5", "5 to 10", "10 and over"):
        sp[b] = share(own, within(cand), lambda x, b=b: speed_bin(x["speed"]) == b and x["dT"][cand] is not None)
    out["R_by_speed"] = sp
    return out


def report0a() -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    r = RB.roots()
    per = {g: RB.read_json(r["ref"] / "posefoot" / "0a" / f"{g}.json") for g in POOL}
    o2o = {g: RB.read_json(r["ref"] / "one2one" / f"{g}.json") for g in POOL}
    hit_all = {g: sum(1 for _, rec in o2o[g]["rows"] if rec["class"] == "hit") for g in POOL}
    res = {g: unit_rows(per, [g], "raw", hit_all) for g in POOL}
    res["pooled"] = unit_rows(per, list(POOL), "raw", hit_all)
    rules = {g: rule_set(res[g]["P-ii own"], res[g]["hits"], res[g]["est"], "foot_raw") for g in POOL}
    out = {"what": "docs/research/P8_posefoot.md version 1.1, stage 0a: foot_raw against foot_stab on D207's boxes; "
                   "report only", "git": {g: per[g]["git"] for g in POOL}, "gates": {g: per[g]["gate"] for g in POOL},
           "counters": {g: per[g]["counters"] for g in POOL}, "seconds": {g: per[g]["seconds"] for g in POOL},
           "N_hits_all (D205 hit class)": hit_all, "rows": res, "rules_foot_raw": rules,
           "readings": readings_0a(rules)}
    out = clean(out)
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD_0A, text)
    print(json.dumps({"rules": out["rules_foot_raw"], "readings": out["readings"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("stage0a")
    c.add_argument("game")
    sub.add_parser("report0a")
    a = ap.parse_args(argv)
    if a.cmd == "stage0a":
        refuse_game(a.game)
        collect0a(a.game)
    else:
        report0a()


if __name__ == "__main__":
    sys.exit(main())
