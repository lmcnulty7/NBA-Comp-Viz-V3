"""tools/p8_nmsarm.py: stages 0a and 0b of the NMS arm (docs/research/P8_nmsarm.md, version 1.1). Report only: no
detector runs, nothing is fitted, tuned or built; stage 1 is not written (it waits on stage 0b's result).

    <frozen CV python> -m tools.p8_nmsarm replay   inside the heavy-job lock ("p8 nmsarm replay"; V3_P8_ROOT = the
                                                   reference root) -> reports/research/p8_nmsarm_replay.json
    <frozen CV python> -m tools.p8_nmsarm screen   inside the lock ("p8 nmsarm stage0") ->
                                                   reports/research/p8_nmsarm_screen.json

Stage 0a: on every stored frame of the two build-environment windows, torchvision NMS on the stored all-candidates set
(R_pre) at the build's 0.45, then max_det, must equal the stored R_post exactly. Stage 0b (version 1.1, 12.2): per arm,
on the universe frames of D208's 88 kept windows, the change in the one-to-one net (the primary row), in missed plus
ghosts and in duplicates against re-NMS at 0.45, with the scorer's functions and constants imported; K1, K2 and G0
decide which arm, if any, goes on. The 32 dropped windows are reported beside.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

WHO = "tools.p8_nmsarm"
ALLOWED = ("gsw_cha_2016", "gsw_bkn_2015")
ARMS = (0.55, 0.70)
BASE_IOU = 0.45
MAX_DET = 300
CHECK_WINDOWS = ("gsw_cha_2016_s05_w00", "gsw_bkn_2015_s04_w00")
C2_KEPT = 421                          # D208's C2 misses on its 88 kept windows (both units)
G0_SHARE = 0.25
RC_SHA = "794f808ab905252e569a67285aec06528e69a6fd10bbb0228eb2b81de2637a9b"   # reports/research/p8_recallcause.json
BOOT_N, BOOT_SEED = 2000, 0
ROWS = ("u1", "mg", "dup", "missed", "ghosts", "in_frame", "boxes", "ref_boxes")
TORCH = "2.2.2"
TEST_FILE = "tests/test_p8_nmsarm.py"


def allowed(names, who: str = WHO) -> list:
    """Version 1.1 (12.5): every game or window name is gsw_cha_2016 or gsw_bkn_2015 (or one of their windows), then
    the P8 guards; SystemExit otherwise."""
    from tools.p8_common import guard
    from tools.research_testset import refuse_cv_test
    names = [str(n) for n in names]
    bad = [n for n in names if not any(n == g or n.startswith(g + "_") for g in ALLOWED)]
    if bad:
        raise SystemExit(f"{who}: {bad[0]} is not in this item's allow-list {ALLOWED}")
    refuse_cv_test(names, who)
    guard(names, who)
    return names


def out_path(stage: str) -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / f"p8_nmsarm_{stage}.json"


def nms_keep(pre: list, iou_t: float, max_det: int = MAX_DET) -> list:
    """The indices of pre that ultralytics' NMS keeps at iou_t (one class, so no class offset): torchvision.ops.nms,
    the backend the build used, on the boxes and scores, then the first max_det, in its order."""
    import torch
    import torchvision
    if not pre:
        return []
    a = torch.tensor([[float(v) for v in b[:4]] for b in pre], dtype=torch.float32)
    s = torch.tensor([float(b[4]) for b in pre], dtype=torch.float32)
    return [int(i) for i in torchvision.ops.nms(a, s, float(iou_t))[:max_det]]


def keep_equals_post(pre: list, keep: list, post: list) -> bool:
    """Stage 0a: the kept boxes equal the stored R_post exactly (same multiset of [x1, y1, x2, y2, score, cls])."""
    return sorted(tuple(pre[k]) for k in keep) == sorted(tuple(b) for b in post)


def post_ids_of(pre: list, post: list) -> list:
    """R_post's boxes as indices into R_pre by exact equality (the anchors are not stored; P8_nmsarm.md section 3)."""
    where = {}
    for i, b in enumerate(pre):
        where.setdefault(tuple(b), []).append(i)
    out = []
    for b in post:
        ids = where.get(tuple(b))
        if not ids:
            raise SystemExit(f"{WHO}: an R_post box is not in R_pre")
        out.append(ids.pop(0))
    return out


def touches_edge(boxes: list, W: int, H: int) -> bool:
    return any(b[0] <= 0 or b[1] <= 0 or b[2] >= W or b[3] >= H for b in boxes)


def differing_boxes(pre: list, keep: list, post: list) -> list:
    """The boxes where re-NMS and R_post differ (in one and not the other) plus every candidate overlapping one of
    them: the boxes an edge clip could have decided (code check item 3)."""
    from tools.p8_recallcause import iou
    a = [tuple(pre[k]) for k in keep]
    b = [tuple(x) for x in post]
    diff = [x for x in a if a.count(x) > b.count(x)] + [x for x in b if b.count(x) > a.count(x)]
    return diff + [c for c in pre if tuple(c) not in diff and any(iou(c, d) > 0 for d in diff)]


def ref_flags(boxes: list, refs: list) -> np.ndarray:
    """The scorer's referee rule for a frame whose sidecar has no class: a box is a referee box when it overlaps one of
    the scorecard cache's referee detections at IoU >= sportvu.bench.REF_IOU (as tools.p8_recallcause.d205_rows)."""
    from sportvu import bench as B
    return np.array([any(B._iou(b[:4], d[:4]) >= B.REF_IOU for d in refs) for b in boxes], bool)


def frame_counts(feet, sv, inside, is_ref=None) -> dict:
    """One frame's rows from raw feet (12.2): the one-to-one net, missed plus non-referee ghosts, duplicates, with the
    scorer's functions and constants; is_ref from ref_flags (the scorer's referee rule)."""
    from sportvu.eval import GHOST_FT, MISS_FT
    from tools.p8_resetfix import frame_extra
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    ins = np.asarray(inside, bool)
    is_ref = np.zeros(len(feet), bool) if is_ref is None else np.asarray(is_ref, bool).reshape(-1)
    ex = frame_extra(feet, sv, ins, is_ref)
    if len(feet):
        D = np.sqrt(((sv[:, None, :] - feet[None, :, :]) ** 2).sum(-1))
        missed = int((ins & (D.min(1) > MISS_FT)).sum())
        ghosts = int(((D.min(0) > GHOST_FT) & ~is_ref).sum())
    else:
        missed, ghosts = int(ins.sum()), 0
    return {"u1": ex["unmatched_1to1"], "mg": missed + ghosts, "dup": ex["duplicates"], "missed": missed,
            "ghosts": ghosts, "in_frame": int(ins.sum()), "boxes": len(feet), "ref_boxes": int(is_ref.sum())}


def c2_candidates(j: int, sv, pre_feet, post_feet, supp: dict) -> list:
    """Every suppressed candidate that makes j a C2 miss (P8_recallcause.md 8.3, as tools.p8_recallcause.is_c2, which
    returns the first): within MISS_FT of j, nearer j than anyone, its suppressor's foot nearer another player."""
    from sportvu.eval import MISS_FT
    from tools.p8_recallcause import nearest_player
    out = []
    for c, k in supp.items():
        if k is None:
            continue
        if float(np.hypot(*(np.asarray(pre_feet[c]) - sv[j]))) > MISS_FT or nearest_player(pre_feet[c], sv) != j:
            continue
        if nearest_player(post_feet[k], sv) != j:
            out.append(c)
    return out


def added_split(added: list, pre_feet, base_feet, sv, inside) -> dict:
    """A(t) (section 7): added boxes that are not a recovered candidate, as duplicates (nearest player within MISS_FT
    already has a base foot within MISS_FT), ghost candidates (no player of all ten within GHOST_FT) or other."""
    from sportvu.eval import GHOST_FT, MISS_FT
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    bf = np.asarray(base_feet, np.float64).reshape(-1, 2)
    out = {"duplicates": 0, "ghost_candidates": 0, "other": 0}
    for c in added:
        d = np.hypot(*(sv - np.asarray(pre_feet[c], np.float64)).T)
        j = int(np.argmin(d))
        if d.min() > GHOST_FT:
            out["ghost_candidates"] += 1
        elif d[j] <= MISS_FT and len(bf) and float(np.hypot(*(bf - sv[j]).T).min()) <= MISS_FT:
            out["duplicates"] += 1
        else:
            out["other"] += 1
    return out


def deltas(per_window: dict, wins: list) -> dict:
    """Pooled arm-minus-baseline sums of each row over the given windows -> {arm: {row: change}}."""
    out = {}
    for t in ARMS:
        a, b = str(t), str(BASE_IOU)
        out[a] = {k: int(sum(per_window[w][a][k] - per_window[w][b][k] for w in wins)) for k in ROWS}
    return out


def delta_interval(per_window: dict, wins: list, arm: str, row: str, n: int = BOOT_N, seed: int = BOOT_SEED):
    """Window-cluster bootstrap of a pooled change (a sum over windows), 95 percent."""
    from tools.p8_resetfix import resample_picks
    wins = sorted(wins)
    if len(wins) < 2:
        return None
    d = np.array([per_window[w][arm][row] - per_window[w][str(BASE_IOU)][row] for w in wins], float)
    s = d[resample_picks(len(wins), n, seed)].sum(1)
    lo, hi = np.quantile(s, [0.025, 0.975])
    return [float(lo), float(hi)]


def verdict(dl: dict, rec: dict, c2_total: int) -> dict:
    """Version 1.1 (12.2): K1 (one-to-one net change below 0), K2 (0.70 only below 0.55's), G0 (R(t) at least 0.25 of
    the C2 misses) -> the gates per arm and the one arm that goes on, or None."""
    g = {}
    for t in ARMS:
        a = str(t)
        g[a] = {"K1": dl[a]["u1"] < 0, "G0": rec[a] >= G0_SHARE * c2_total}
    g["0.7"]["K2"] = dl["0.7"]["u1"] < dl["0.55"]["u1"]
    go = "0.7" if all(g["0.7"].values()) else ("0.55" if all(g["0.55"].values()) else None)
    return {"gates": g, "arm": go}


def _checks(args_stage: str) -> dict:
    """The run guards shared by both stages -> {r, head}."""
    import subprocess
    from tools import p8_rebaseline as RB
    from tools.p8_common import REPO, p8_root
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if p8_root().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    st = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                        text=True, check=True).stdout.strip()
    if st:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    from tools.p8_recallcause import sha256
    if sha256(REPO / "reports" / "research" / "p8_recallcause.json") != RC_SHA:
        raise SystemExit(f"{WHO}: reports/research/p8_recallcause.json is not D208's record")
    import torch
    import torchvision
    if torch.__version__ != TORCH:
        raise SystemExit(f"{WHO}: torch {torch.__version__} is not {TORCH}")
    return {"r": r, "head": head, "stage": args_stage,
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__}}


def watched(r: dict) -> dict:
    """The manifests that must not change during a stage, recallcause/ (the stored candidates) included."""
    from tools import p8_rebaseline as RB
    return {k: RB.manifest(r[k], ("labels", "label_build", "truth_accept", "cache")) for k in ("ref", "ms")} | \
        {"builds_r23": RB.manifest(r["ref"], ("builds/r23",)), "recallcause": RB.manifest(r["ref"], ("recallcause",))}


def digest(man: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()


def run_unit_tests() -> dict:
    """Code check item 5: this item's tests under the interpreter that runs the stage (the frozen CV python, so the
    NMS test runs), recorded in the stage's record; a failure stops the stage."""
    import subprocess
    import sys
    from tools.p8_common import REPO
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
                        "-rs", TEST_FILE], cwd=REPO, capture_output=True, text=True)
    lines = [x for x in p.stdout.strip().splitlines() if x.strip()]
    out = {"file": TEST_FILE, "returncode": p.returncode, "summary": lines[-1] if lines else "",
           "skipped": [x for x in lines if x.startswith("SKIPPED")]}
    if p.returncode != 0 or out["skipped"]:
        raise SystemExit(f"{WHO}: {TEST_FILE} did not pass in full under this python: {out['summary']}")
    return out


def _write(stage: str, out: dict) -> None:
    from export.schemas import find_private
    from tools.p8_common import atomic_write
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(out_path(stage), text)


def cmd_replay(args) -> dict:
    import cv2
    from tools import p8_frozen as F
    from tools.p8_recallcause import raw_dir, read_json
    c = _checks("replay")
    allowed(CHECK_WINDOWS)
    tests = run_unit_tests()
    before = watched(c["r"])
    t0 = time.time()
    P = F.install_paths()
    res = {}
    for s in CHECK_WINDOWS:
        raw = read_json(raw_dir(c["r"]["ref"]) / f"{s}.json")
        cap = cv2.VideoCapture(str(P["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        bad, edge = [], []
        for f in sorted(raw, key=int):
            v = raw[f]
            kk = nms_keep(v["pre"], BASE_IOU)
            if not keep_equals_post(v["pre"], kk, v["post"]):
                (edge if touches_edge(differing_boxes(v["pre"], kk, v["post"]), W, H) else bad).append(int(f))
        res[s] = {"frames": len(raw), "differing": len(bad) + len(edge), "differing_edge": edge, "differing_other": bad}
    passed = all(v["differing"] == 0 for v in res.values())
    after = watched(c["r"])
    if after != before:
        raise SystemExit(f"{WHO}: a watched folder changed; nothing is reported")
    out = {"what": "P8 NMS arm stage 0a: re-NMS on stored candidates against the stored R_post "
                   "(docs/research/P8_nmsarm.md, version 1.1); report only", "head": c["head"],
           "versions": c["versions"], "unit_tests": tests, "recallcause_digest": digest(after["recallcause"]),
           "iou": BASE_IOU, "max_det": MAX_DET, "windows": res, "passed": passed,
           "edge_only": (not passed) and all(not v["differing_other"] for v in res.values()),
           "seconds": round(time.time() - t0, 1)}
    _write("replay", out)
    print(json.dumps({k: out[k] for k in ("windows", "passed", "edge_only")}))
    return out


def cmd_screen(args) -> dict:
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_one2one as O
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
    from tools import p8_scorecard as S
    from tools.p8_common import guard
    c = _checks("screen")
    r = c["r"]
    rep = RC.read_json(out_path("replay")) if out_path("replay").exists() else None
    if not (rep or {}).get("passed") and not (rep or {}).get("edge_only"):
        raise SystemExit(f"{WHO}: stage 0a has not passed")
    if (rep["head"], rep["iou"], rep["max_det"]) != (c["head"], BASE_IOU, MAX_DET):
        raise SystemExit(f"{WHO}: the replay record is not from this head at IoU {BASE_IOU} and max_det {MAX_DET}")
    edge_skip = {s: set(v["differing_edge"]) for s, v in rep["windows"].items()}
    pinned, now = RC.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if [k for k in now if now[k] != pinned.get(k)]:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    one2one = {g: r["ref"] / "one2one" / f"{g}.json" for g in ALLOWED}
    if {g: RC.sha256(p) for g, p in one2one.items()} != RC.D205_SHA:
        raise SystemExit(f"{WHO}: D205's records are not the pinned ones")
    d205 = {g: RC.read_json(p) for g, p in one2one.items()}
    d202 = {g: RC.read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json") for g in ALLOWED}
    if any(d205[g].get("run_id") != RC.RUN_ID or d202[g].get("run_id") != RC.RUN_ID for g in ALLOWED):
        raise SystemExit(f"{WHO}: D205's or D202's records are not from run {RC.RUN_ID}")
    rc = RC.read_json(RC.out_path())
    dropped = set(rc["windows_dropped"])
    kept = sorted(s for s in rc["alignment"] if s not in dropped)
    allowed(kept + sorted(dropped))
    tests = run_unit_tests()
    before = watched(r)
    if digest(before["recallcause"]) != rep.get("recallcause_digest"):
        raise SystemExit(f"{WHO}: recallcause/ changed since the replay")
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    comp = P["builds"] / "r23"
    raw_d = RC.raw_dir(r["ref"])
    per_window, frames_n, agree, skipped = {}, {}, {}, {}
    c2_total = {"kept": 0, "dropped": 0}
    c2_rec = {"kept": {str(t): 0 for t in ARMS}, "dropped": {str(t): 0 for t in ARMS}}
    added = {"kept": {str(t): {"duplicates": 0, "ghost_candidates": 0, "other": 0} for t in ARMS},
             "dropped": {str(t): {"duplicates": 0, "ghost_candidates": 0, "other": 0} for t in ARMS}}
    for g in ALLOWED:
        allowed([g])
        gd = F.game_of(g)
        guard([gd["sportvu"], gd["video_id"]], WHO)          # ids, not names: the P8 guards only
        mirror = F.mirror_of(P, g)
        mom = S.Moments(RC.read_json(P["sportvu"] / f"{gd['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": S.pbp_events(gd["sportvu"]),
               "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        P_t = {**P, "labels": RB.truths(r)[g]["current"] / "labels"}
        wins = sorted((w for w in F.label_windows(P, g) if (comp / f"{w['window']}_frames.json").exists()),
                      key=lambda w: w["window"])
        allowed([w["window"] for w in wins])
        if sorted(d202[g]["windows"]) != [w["window"] for w in wins]:
            raise SystemExit(f"{WHO}: {g}: the windows differ from D202's record")
        for wi, w in enumerate(wins):
            s = w["window"]
            if not (raw_d / f"{s}.json").exists():
                continue
            allowed([s])
            rows = RC.d205_rows(g, P_t, w, comp, ctx, d202[g]["windows"][s], (d202[g].get("hashes") or {}).get(s, {}))
            mine = [[wi, {k: x["rec"][k] for k in ("class", "single", "inframe")}] for x in rows]
            want = [[a, {k: b[k] for k in ("class", "single", "inframe")}] for a, b in d205[g]["rows"] if a == wi]
            if mine != want:
                raise SystemExit(f"{WHO}: the D205 row gate fails on {s}")
            raw = RC.read_json(raw_d / f"{s}.json")
            cache = RC.read_json(P_t["cache"] / "scorecard" / f"{s}.json")
            refd = {int(k): vv for k, vv in cache.get("referees", {}).items()}
            if any("cls" in b for x in RC.read_json(comp / f"{s}_frames.json")["frames"] for b in x["boxes"]):
                raise SystemExit(f"{WHO}: {s}'s sidecar carries classes; the cache referee rule would not be the scorer's")
            cap = cv2.VideoCapture(str(P["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
            uni = [x for x in rows if x["rec"]["class"] in RC.UNIVERSE]
            byf = {}
            for x in uni:
                byf.setdefault(x["frame"], []).append(x)
            part = "kept" if s in kept else "dropped"
            acc = {str(t): {k: 0 for k in ROWS} for t in (BASE_IOU,) + ARMS}
            agree[s] = 0
            for f in sorted(byf):
                edge_f = f in edge_skip.get(s, set())          # arm rows skip it; C2 is counted on every frame
                skipped[s] = skipped.get(s, 0) + edge_f
                v = raw[str(f)]
                x0 = byf[f][0]
                sv, Ht, px = x0["sv"], x0["Ht"], x0["px"]
                m = FRAME_MARGIN_PX
                inside = (px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)
                pre, post = v["pre"], v["post"]
                pre_feet, post_feet = RC.feet_of(pre, Ht), RC.feet_of(post, Ht)
                keep = {str(t): nms_keep(pre, t) for t in (BASE_IOU,) + ARMS}
                agree[s] += keep_equals_post(pre, keep[str(BASE_IOU)], post)
                flags = ref_flags(pre, refd.get(f, []))
                for t, kk in keep.items():
                    if edge_f:
                        continue
                    fc = frame_counts(pre_feet[kk] if kk else np.zeros((0, 2)), sv, inside, flags[kk])
                    for k in ROWS:
                        acc[t][k] += fc[k]
                post_ids = post_ids_of(pre, post)
                supp = RC.suppressors(pre, post, list(range(len(pre))), post_ids)
                main, _ = O.two_stage(post_feet, sv, range(len(sv)), range(len(post)), O._assign_all)
                rec_c = {str(t): set() for t in ARMS}
                for x in byf[f]:
                    j = x["player"]
                    if main.get(j) is not None:
                        continue
                    cands = c2_candidates(j, sv, pre_feet, post_feet, supp)
                    if not cands:
                        continue
                    c2_total[part] += 1
                    if edge_f:
                        continue
                    for t in ARMS:
                        hit = [cc for cc in cands if cc in set(keep[str(t)])]
                        if hit:
                            c2_rec[part][str(t)] += 1
                            rec_c[str(t)].update(hit)
                base_set = set(keep[str(BASE_IOU)])
                for t in ([] if edge_f else ARMS):
                    add = [k for k in keep[str(t)] if k not in base_set and k not in rec_c[str(t)]]
                    sp = added_split(add, pre_feet, pre_feet[sorted(base_set)] if base_set else np.zeros((0, 2)),
                                     sv, inside)
                    for k2, n2 in sp.items():
                        added[part][str(t)][k2] += n2
            frames_n[s] = len(byf)
            per_window[s] = acc
    after = watched(r)
    if after != before:
        raise SystemExit(f"{WHO}: a watched folder changed; nothing is reported")
    if c2_total["kept"] != C2_KEPT:
        raise SystemExit(f"{WHO}: the C2 recount on the kept windows is {c2_total['kept']}, not D208's {C2_KEPT}")
    kw = [s for s in kept if s in per_window]
    dw = sorted(s for s in dropped if s in per_window)
    dl_k, dl_d = deltas(per_window, kw), deltas(per_window, dw)
    v = verdict(dl_k, c2_rec["kept"], c2_total["kept"])
    base_k = {k: int(sum(per_window[s][str(BASE_IOU)][k] for s in kw)) for k in ROWS}
    out = {"what": "P8 NMS arm stage 0b: pre-tracker rows of re-NMS arms on stored candidates "
                   "(docs/research/P8_nmsarm.md, version 1.1); report only", "head": c["head"],
           "arms": list(ARMS), "base_iou": BASE_IOU,
           "kept": {"windows": len(kw), "frames": sum(frames_n[s] for s in kw), "baseline": base_k, "change": dl_k,
                    "intervals": {str(t): {k: delta_interval(per_window, kw, str(t), k) for k in ("u1", "mg", "dup")}
                                  for t in ARMS},
                    "c2": {"misses": c2_total["kept"], "recoverable": c2_rec["kept"]}, "added": added["kept"]},
           "dropped_beside": {"windows": len(dw), "frames": sum(frames_n[s] for s in dw), "change": dl_d,
                              "c2": {"misses": c2_total["dropped"], "recoverable": c2_rec["dropped"]},
                              "added": added["dropped"]},
           "replay_agreement": {"frames": sum(frames_n.values()),
                                "equal_at_0.45": int(sum(agree.values())), "edge_frames_skipped": skipped},
           "referee_boxes_flagged": {str(t): int(sum(per_window[s][str(t)]["ref_boxes"] for s in per_window))
                                     for t in (BASE_IOU,) + ARMS},
           "versions": c["versions"], "unit_tests": tests, "recallcause_digest": digest(after["recallcause"]),
           "verdict": v, "per_window": per_window, "seconds": round(time.time() - t0, 1)}
    _write("screen", out)
    print(json.dumps({"kept_change": dl_k, "c2": out["kept"]["c2"], "verdict": v}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replay")
    sub.add_parser("screen")
    args = ap.parse_args(argv)
    {"replay": cmd_replay, "screen": cmd_screen}[args.cmd](args)


if __name__ == "__main__":
    main()
