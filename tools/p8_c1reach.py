"""tools/p8_c1reach.py: stage 1 of the C1 reach audit (docs/research/P8_c1reach.md, version 1.1). Report only:
nothing is fitted, tuned or trained, and nothing is downloaded. Three steps, two of them inside one lock job:

    A  <frozen CV python> -m tools.p8_c1reach prepare [--smoke]   inside the lock; V3_P8_ROOT = the reference root;
       HF_HUB_OFFLINE=1, PYTORCH_ENABLE_MPS_FALLBACK=1. D205's rows, R0, the sample, build_read with r23's capture (R1)
       and the frames written losslessly with their SHA-256 -> <reference root>/c1reach/inputs[__smoke].json, frames/
    B  <the teachers venv python> -m tools.p8_c1reach teacher [--smoke]   in the same lock job, after A. Hash checks,
       R2, the held teacher (tools.p8_teachers.load_runner, CPU, fixed threads) -> c1reach/teacher[__smoke].json
    C  python3 -m tools.p8_c1reach report [--smoke]   system python, no lock: the statistics ->
       reports/research/p8_c1reach_stage1[_smoke].json

The venv cannot import the P8 guards (they need pandas), so step A runs every guard, including tools.p8_teachers'
record check on R2's box-truth frames, and step B checks only hashes against step A's file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_c1reach"
ALLOWED = ("gsw_cha_2016", "gsw_bkn_2015")
SMOKE_WINDOW = "gsw_cha_2016_s07_w00"
SAMPLE_PER_WINDOW = 8
FLOOR_CONF = 0.25                      # r23's conf 0.25 boxes, the floor (D208's R_low)
TEACHER = "yolox_x_sportsmot"
TEACHER_CONF = 0.1471                  # tools.p8_teachers.MISS_TEACHER_CONF
TEACHER_FLOOR = 0.05
TEACHER_SHA = "58547880fb73b9f9ac5674547781c6a87071906376286da301f9b0e19b50ed1c"
THREADS = 4
R2_N = 20
COURT_L, COURT_W, COURT_MARGIN = 94.0, 50.0, 3.0
POS_CTRL_MIN = 0.90
R1_MAX_FAIL = 0.05
BAR, KILL = -0.0042, -0.0021
CAP_A_S, CAP_B_S = 45 * 60, 20 * 60          # 10.12: step A between windows, step B between frames
CHA_RESIDUAL = 0.0048                  # P8_E1B.md 2b: gsw_cha_2016's transfer-model residual
BOOT_N, BOOT_SEED = 2000, 0
SUBCLASSES = ("C1a_own", "C1a_other", "C1b", "C1c")
ENV_NEEDED = {"HF_HUB_OFFLINE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def allowed(names) -> list:
    """The allow-list and the P8 guards on every game and window name (P8_c1reach.md header)."""
    from tools.p8_common import guard
    from tools.research_testset import refuse_cv_test
    names = [str(n) for n in names]
    bad = [n for n in names if not any(n == g or n.startswith(g + "_") for g in ALLOWED)]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not in this item's allow-list {ALLOWED}")
    refuse_cv_test(names, WHO)
    guard(names, WHO)
    return names


def sample_frames(frames, k: int = SAMPLE_PER_WINDOW) -> list:
    """Section 4: the window's C1 frames sorted; if more than k, those at ranks round(linspace(0, n - 1, k))."""
    fs = sorted(set(int(f) for f in frames))
    if len(fs) <= k:
        return fs
    return sorted({fs[int(round(x))] for x in np.linspace(0, len(fs) - 1, k)})


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def court_keep(feet) -> np.ndarray:
    """10.8: a box is kept when its foot lies inside the court lines plus COURT_MARGIN on every side."""
    f = np.asarray(feet, np.float64).reshape(-1, 2)
    m = COURT_MARGIN
    return (f[:, 0] >= -m) & (f[:, 0] <= COURT_L + m) & (f[:, 1] >= -m) & (f[:, 1] <= COURT_W + m)


def matched(feet, sv) -> set:
    """10.9: players assigned a box by tools.p8_one2one.assign_1to1 at 2.5 ft over all ten players (no referee boxes)."""
    from sportvu.eval import MISS_FT
    from tools.p8_one2one import assign_1to1
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    return set(assign_1to1(feet, sv, np.zeros(len(feet), bool), MISS_FT))


def cost_rows(feet, sv, inside) -> dict:
    """Section 6's cost rows of one box set on one frame: ghost proxy (no player of all ten within GHOST_FT), duplicates
    (in-frame players with two or more feet within 2.5 ft), boxes near no player (nearest beyond 2.5 ft)."""
    from sportvu.eval import GHOST_FT, MISS_FT
    from tools.p8_resetfix import frame_extra
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    if not len(feet):
        return {"boxes": 0, "ghosts": 0, "near_no_player": 0, "duplicates": 0}
    D = np.sqrt(((sv[:, None, :] - feet[None, :, :]) ** 2).sum(-1)).min(0)
    return {"boxes": len(feet), "ghosts": int((D > GHOST_FT).sum()), "near_no_player": int((D > MISS_FT).sum()),
            "duplicates": frame_extra(feet, sv, np.asarray(inside, bool))["duplicates"]}


def frame_stats(fr: dict, teacher_boxes: list) -> dict:
    """One sampled frame -> per C1 miss: reached by the teacher (at TEACHER_CONF, court-filtered), by the floor (r23 at
    0.25: R_post plus R_low), and the positive control (the teacher's reach of r23's hits) and the cost rows."""
    from tools.p8_recallcause import feet_of
    sv = np.asarray(fr["sv"], np.float64)
    Ht = np.asarray(fr["Ht"], np.float64)
    inside = np.asarray(fr["inside"], bool)
    tb = [b for b in teacher_boxes if b[4] >= TEACHER_CONF]
    band = [b for b in teacher_boxes if TEACHER_FLOOR <= b[4] < TEACHER_CONF]
    tf = feet_of(tb, Ht)
    tf = tf[court_keep(tf)] if len(tf) else tf
    bf = feet_of(band, Ht)
    bf = bf[court_keep(bf)] if len(bf) else bf
    post = fr["r23_post"]
    floor = post + [b for b in fr["r23_low"] if b[4] >= FLOOR_CONF]
    pf, ff = feet_of(post, Ht), feet_of(floor, Ht)
    t_m, f_m, p_m = matched(tf, sv), matched(ff, sv), matched(pf, sv)
    tband_m = matched(np.vstack([tf, bf]) if len(bf) else tf, sv)
    hits = [j for j in p_m if inside[j]]
    misses = [{**m, "teacher": m["player"] in t_m, "floor": m["player"] in f_m,
               "teacher_with_band": m["player"] in tband_m} for m in fr["c1"]]
    return {"misses": misses, "hits": len(hits), "hits_reached": sum(1 for j in hits if j in t_m),
            "cost": {"teacher": cost_rows(tf, sv, inside), "r23_040": cost_rows(pf, sv, inside),
                     "r23_floor": cost_rows(ff, sv, inside)}}


def scaled(window_stats: dict, full: dict) -> dict:
    """Section 4: per window, sampled counts scaled to the window's full C1 counts (full / sampled C1 misses)."""
    n_s = sum(1 for m in window_stats["misses"])
    k = (full["n"] / n_s) if n_s else 0.0
    out = {}
    for m in window_stats["misses"]:
        key = (m["unit"], m["subclass"])
        d = out.setdefault(key, {"n": 0.0, "teacher": 0.0, "floor": 0.0, "beyond": 0.0, "unreached_oc": 0.0})
        d["n"] += k
        d["teacher"] += k * m["teacher"]
        d["floor"] += k * m["floor"]
        d["beyond"] += k * (m["teacher"] and not m["floor"])
        if m["subclass"] in ("C1a_other", "C1c") and not m["teacher"] and not m["floor"]:
            d["unreached_oc"] += k
    return out


def ceiling(beyond_D: float, beyond_P: float, f_pres: float, f_plc: float, n: float) -> float:
    """10.1, section 6: C_S = -(R_D f_pres + R_P f_plc) / N (negative = fewer misses at build level)."""
    return -(beyond_D * f_pres + beyond_P * f_plc) / n if n else 0.0


def e1_fractions(per_window_e1: dict, keys: list) -> dict:
    """E1b's fractions from E1's per-window transfer counts, summed over the given windows of one game."""
    from tools.p8_e1b_bound import TKEYS, fractions
    tot = {t: sum(per_window_e1[k][t] for k in keys) for t in TKEYS}
    return fractions(tot)


# ── E1's per-window transfer counts (stored files only; cha and bkn) ───────────────────────────────────────────────
def e1_per_window(work_root: Path) -> dict:
    """tools/p8_e1b_bound.py cmd_transfer's per-window counts, for gsw_cha_2016 and gsw_bkn_2015 only (E1's third game
    is apart), on its stored files; checked against its record's per-game counts by the caller. The corpus preds
    chunks hold every game's predictions and are loaded whole; only cha and bkn frames are used."""
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_e1b_bound as EB
    from tools import p8_teachers as T
    from tools.p8_e1_postmortem import scorer_inside
    from tools.p8_e1data import corpus_root
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    preds = {}
    for k in range(len(plan["chunks"])):
        preds.update(json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"])

    def sidecar(folder, window):
        p = work_root / "builds" / folder / f"{window}_frames.json"
        return {int(r["frame"]): r for r in json.loads(p.read_text())["frames"]} if p.exists() else {}

    def stab_ft(rows, H):
        import cv2
        if not rows:
            return np.zeros((0, 2))
        px = np.array([b["foot_stab"] for b in rows], np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(px, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)

    per_window = {}
    for tag in ALLOWED:
        g = plan["games"][tag]
        allowed([tag])
        rfold, efold = EB.BUILDS[tag]
        m4 = json.loads((work_root / "e1" / f"m4_preds__{tag}.json").read_text())
        index = SportVUIndex(json.loads((work_root / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        cache_w, sr, se = None, {}, {}
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != tag:
                        continue
                    if fr["window"] != cache_w:
                        cache_w, sr, se = fr["window"], sidecar(rfold, fr["window"]), sidecar(efold, fr["window"])
                    a, b = sr.get(int(fr["frame"])), se.get(int(fr["frame"]))
                    if a is None or b is None:
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    rec = preds[f"{tag}/{stem}"]
                    wpx, hpx = rec["size"]
                    H = fr["H_truth"]
                    sv = apply_mirror(index.q[fr["q"]]["xy"][fr["moment"]], fx, fy)
                    inside = scorer_inside(sv, H, wpx, hpx)
                    svi = np.asarray(sv, np.float64).reshape(-1, 2)[inside]
                    P = m4["preds"][stem]
                    d = {"rr": EB.nearest(svi, T.feet_ft([p for p in P["r23"] if p[4] >= EB.DET_CONF], H)) <= EB.MISS_FT,
                         "er": EB.nearest(svi, T.feet_ft([p for p in P["e1"] if p[4] >= EB.DET_CONF], H)) <= EB.MISS_FT,
                         "tr": EB.nearest(svi, T.feet_ft([p for p in rec[T.MISS_TEACHER] if p[4] >= EB.TEACHER_CONF], H))
                         <= EB.MISS_FT,
                         "r5": EB.nearest(svi, T.feet_ft([p for p in P["r23"] if p[4] >= EB.R23_FLOOR], H)) <= EB.NEAR_FT,
                         "rb": EB.nearest(svi, stab_ft(a["boxes"], H)) <= EB.MISS_FT,
                         "eb": EB.nearest(svi, stab_ft(b["boxes"], H)) <= EB.MISS_FT}
                    c = EB.transfer_counts(**d)
                    acc = per_window.setdefault(f"{tag}/{fr['window']}", {k: 0 for k in EB.TKEYS})
                    for k, v in c.items():
                        acc[k] += v
    return per_window


# ── step A: prepare (frozen CV python, inside the lock) ────────────────────────────────────────────────────────────
def c1_dir(ref: Path) -> Path:
    return ref / "c1reach"


def cmd_prepare(args) -> dict:
    import subprocess
    import cv2
    import config
    from sportvu import bench as B
    from sportvu.rebuild import build_stride
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
    from tools import p8_recallsupp as RS
    from tools import p8_scorecard as S
    from tools import p8_teachers as T
    from tools.p8_common import REPO, atomic_write, guard, main_checkout, p8_root
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    bad_env = {k: os.environ.get(k) for k, v in ENV_NEEDED.items() if os.environ.get(k) != v}
    if bad_env:
        raise SystemExit(f"{WHO}: set {ENV_NEEDED} before the run (got {bad_env})")
    r = RB.roots()
    if p8_root().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    st = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu", "reports/research"], cwd=REPO,
                        capture_output=True, text=True, check=True).stdout.strip()
    if st:
        raise SystemExit(f"{WHO}: tools/, sportvu/ or reports/research/ has uncommitted changes")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    pinned, now = RC.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if [k for k in now if now[k] != pinned.get(k)]:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    if RC.sha256(weights) != RC.R23_SHA:
        raise SystemExit(f"{WHO}: the detector is not r23")
    one2one = {g: r["ref"] / "one2one" / f"{g}.json" for g in RC.POOL}
    if {g: RC.sha256(p) for g, p in one2one.items()} != RC.D205_SHA:
        raise SystemExit(f"{WHO}: D205's records are not the pinned ones")
    d205 = {g: RC.read_json(p) for g, p in one2one.items()}
    d202 = {g: RC.read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json") for g in RC.POOL}
    if any(d205[g].get("run_id") != RC.RUN_ID or d202[g].get("run_id") != RC.RUN_ID for g in RC.POOL):
        raise SystemExit(f"{WHO}: D205's or D202's records are not from run {RC.RUN_ID}")
    supp_p = RS.out_path()
    supp = RC.read_json(supp_p)
    g0 = RC.read_json(RS.out_path("g0"))
    certified, reread = g0["certified"], g0["reread"]
    windows = sorted(certified + reread)
    if len(windows) != 120:
        raise SystemExit(f"{WHO}: the supplement's window sets are not 120 windows")
    todo = [SMOKE_WINDOW] if args.smoke else windows
    allowed(windows + [SMOKE_WINDOW])
    # R2's box-truth frames: the record check here (the venv cannot import the guards), hashes for step B
    ws_t = T.workspace()
    pred_rec = json.loads((ws_t / "preds" / f"{TEACHER}.json").read_text())
    if pred_rec.get("weights_sha256") != TEACHER_SHA or sha256_file(ws_t / "weights" / pred_rec["weights"]) != TEACHER_SHA:
        raise SystemExit(f"{WHO}: the teacher's weights or record are not the pinned ones")
    fl = {f["frame"]: f for f in json.loads((ws_t / "frames.json").read_text())["frames"]}
    clip_game, games = T.clip_games()
    r2 = []
    for name in sorted(pred_rec["preds"])[:R2_N]:
        truth_rec = json.loads((T.truth_dir() / f"{Path(name).stem}.json").read_text())
        why = T.check_record(truth_rec, clip_game, games)
        if why:
            raise SystemExit(f"{WHO}: R2 frame {name} fails the record check: {why}")
        r2.append({"frame": name, "image": fl[name]["image_path"], "sha256": sha256_file(Path(fl[name]["image_path"])),
                   "stored": pred_rec["preds"][name]})
    watch = lambda: {k: RB.manifest(r[k], ("labels", "label_build", "truth_accept", "cache")) for k in ("ref", "ms")} | \
        {"builds_r23": RB.manifest(r["ref"], ("builds/r23",)), "recallcause": RB.manifest(r["ref"], ("recallcause",)),
         "recallsupp": RB.manifest(r["ref"], ("recallsupp",))}
    before = watch()
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    comp = P["builds"] / "r23"
    raw = RC.Raw(weights)
    import torch
    import ultralytics
    versions = {"opencv": cv2.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    if versions != RC.VERSIONS:
        raise SystemExit(f"{WHO}: versions {versions} are not {RC.VERSIONS}")
    from gate.backbones import get_backbone
    from gate.trained_head import TrainedHeadGate
    mc = main_checkout() / "models"
    head_p, thr = mc / config.HEAD_V2_PATH.name, json.loads((mc / config.THRESHOLDS_PATH.name).read_text())["v2"]
    gates = {}

    def gate_for(dev):
        if dev not in gates:
            gates[dev] = TrainedHeadGate.load(head_p, backbone=get_backbone("clip", dev), threshold=thr)
        return gates[dev]

    wins_all = {g: sorted((w for w in F.label_windows(P, g) if (comp / f"{w['window']}_frames.json").exists()),
                          key=lambda w: w["window"]) for g in RC.POOL}
    by_name = {w["window"]: (g, wi, w) for g in RC.POOL for wi, w in enumerate(wins_all[g])}
    out_d = c1_dir(r["ref"])
    fr_d = out_d / ("frames_smoke" if args.smoke else "frames")
    fr_d.mkdir(parents=True, exist_ok=True)

    def cap():
        if time.time() - t0 > CAP_A_S:
            raise SystemExit(f"{WHO}: step A passed its {CAP_A_S // 60}-minute cap; nothing is recorded")
    # pass 1 (code check 3): D205's rows, the classes and R0 for every window, before any frame is read
    all_misses, per_window, work = [], {}, {}
    for g in RC.POOL:
        gd = F.game_of(g)
        guard([gd["sportvu"], gd["video_id"]], WHO)
        mirror = F.mirror_of(P, g)
        mom = S.Moments(RC.read_json(P["sportvu"] / f"{gd['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": S.pbp_events(gd["sportvu"]),
               "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        P_t = {**P, "labels": RB.truths(r)[g]["current"] / "labels"}
        for s in [x for x in windows if x.startswith(g + "_")]:
            if args.smoke and s not in todo:
                continue
            cap()
            _, wi, w = by_name[s]
            rows = RC.d205_rows(g, P_t, w, comp, ctx, d202[g]["windows"][s], (d202[g].get("hashes") or {}).get(s, {}))
            mine = [[wi, {k: x["rec"][k] for k in ("class", "single", "inframe")}] for x in rows]
            want = [[a, {k: b[k] for k in ("class", "single", "inframe")}] for a, b in d205[g]["rows"] if a == wi]
            if mine != want:
                raise SystemExit(f"{WHO}: the D205 row gate fails on {s}")
            uni = [x for x in rows if x["rec"]["class"] in RC.UNIVERSE]
            side = RC.read_json(comp / f"{s}_frames.json")
            src = RC.raw_dir(r["ref"]) if s in certified else r["ref"] / "recallsupp"
            stored = {int(f): v for f, v in RC.read_json(src / f"{s}.json").items()}
            ms = RS.classify_window(g, s, uni, rows, stored, side["fps"])
            all_misses += ms
            c1 = [m for m in ms if m["class"] in SUBCLASSES]
            per_window[s] = {"game": g, "n": len(c1),
                             "by": {f"{u}|{c}": sum(1 for m in c1 if m["d205"] == u and m["class"] == c)
                                    for u in RC.UNIVERSE for c in SUBCLASSES}}
            if c1:
                byf = {}
                for x in uni:
                    byf.setdefault(x["frame"], x)
                work[s] = {"g": g, "w": w, "side": side, "stored": stored, "c1": c1,
                           "byf": {f: byf[f] for f in {m["frame"] for m in c1}}}
    r0 = None
    if not args.smoke:                               # R0: the supplement's 120-window C1 counts, unit and game
        mine_u = RS.unit_tables(all_misses)
        theirs = supp["units_120_post_hoc"]
        bad = [f"{u} {c}" for u in mine_u for c in SUBCLASSES
               if mine_u[u]["classes"][c]["count"] != theirs[u]["classes"][c]["count"]]
        if bad:
            raise SystemExit(f"{WHO}: R0 fails: the C1 recount differs from the supplement in {bad[:5]}")
        r0 = "passed"
    # pass 2: the sampled frames read the build's way, r23's capture (R1), the frames written and read back
    frames_out, r1_fail, undetermined = [], [], []
    for s in sorted(work):
        cap()
        g, w, side, stored, c1, byf = (work[s][k] for k in ("g", "w", "side", "stored", "c1", "byf"))
        pick = sample_frames([m["frame"] for m in c1])
        dev = ((RC.read_json(comp / f"{s}_device.json") if (comp / f"{s}_device.json").exists() else {})
               .get("resolved") or {}).get("gate") or "cpu"
        stride = int(side.get("stride") or 1)
        if build_stride(w["fps"]) != stride:
            raise SystemExit(f"{WHO}: {s}: the sidecar stride is not the build's")
        max_frames = (w["f_end"] - w["f_start"]) // stride + 2
        (fr_d / s).mkdir(exist_ok=True)

        def predict(idx, img, s=s):
            p = fr_d / s / f"{int(idx):06d}.png"
            if not cv2.imwrite(str(p), img):
                raise SystemExit(f"{WHO}: {s} frame {idx}: the frame could not be written")
            back = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if back is None or not np.array_equal(back, img):
                raise SystemExit(f"{WHO}: {s} frame {idx}: the PNG does not read back equal")
            return {"post": raw.run(img)["post"], "frame": int(idx), "sha256": sha256_file(p),
                    "png": str(p.relative_to(out_d)), "size": [int(img.shape[1]), int(img.shape[0])]}
        br = RS.build_read(P["label_build"] / f"{s}.mp4", side, max_frames, gate_for(dev).is_court_visible,
                           set(pick), predict)
        if not br["validation"]["passes"] or set(pick) - set(br["preds"]):
            undetermined.append(s)
            continue
        for f in pick:
            v = br["preds"][f]
            if v["frame"] != f:
                raise SystemExit(f"{WHO}: {s}: a capture is keyed to another frame")
            if json.loads(json.dumps(v["post"])) != stored[f]["post"]:
                r1_fail.append({"frame": f"{s}/{f}", "c1": sum(1 for m in c1 if m["frame"] == f)})
                continue
            x = byf[f]
            frames_out.append({"game": g, "window": s, "frame": f, "png": v["png"], "sha256": v["sha256"],
                               "size": v["size"], "sv": np.asarray(x["sv"]).tolist(),
                               "Ht": np.asarray(x["Ht"]).tolist(), "px": np.asarray(x["px"]).tolist(),
                               "inside": [bool(i) for i in _inside(x["px"], v["size"])],
                               "r23_post": stored[f]["post"], "r23_low": stored[f]["low"],
                               "c1": [{"player": m["player"], "pid": m["pid"], "unit": m["d205"],
                                       "subclass": m["class"]} for m in c1 if m["frame"] == f]})
    n_frames = len(frames_out) + len(r1_fail)
    if n_frames and len(r1_fail) > R1_MAX_FAIL * n_frames:
        raise SystemExit(f"{WHO}: R1 fails on {len(r1_fail)} of {n_frames} sampled frames; nothing is reported")
    if watch() != before:
        raise SystemExit(f"{WHO}: a watched folder changed; nothing is recorded")
    out = {"what": "P8 C1 reach stage 1, step A (docs/research/P8_c1reach.md v1.1)", "head": head, "smoke": args.smoke,
           "versions": versions, "supplement_sha256": RC.sha256(supp_p), "r0": r0, "r1_failed": r1_fail, "undetermined": undetermined,
           "per_window": per_window, "frames": frames_out, "r2": r2,
           "teacher": {"name": TEACHER, "weights_sha256": TEACHER_SHA, "conf": TEACHER_CONF},
           "seconds": round(time.time() - t0, 1)}
    atomic_write(out_d / f"inputs{'__smoke' if args.smoke else ''}.json", json.dumps(out))
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()
                      if k in ("r0", "r1_failed", "undetermined", "frames", "per_window", "seconds")}))
    return out


def _inside(px, size) -> list:
    from sportvu.eval import FRAME_MARGIN_PX
    px = np.asarray(px, np.float64).reshape(-1, 2)
    m, (W, H) = FRAME_MARGIN_PX, size
    return list((px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m))


# ── step B: teacher (the teachers venv, inside the same lock job) ──────────────────────────────────────────────────
def cmd_teacher(args) -> dict:
    import cv2
    import torch
    from tools import p8_teachers as T
    from tools.p8_common import atomic_write
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    ref = Path(os.environ["V3_P8_ROOT"]).expanduser().resolve()
    out_d = c1_dir(ref)
    sfx = "__smoke" if args.smoke else ""
    inp = json.loads((out_d / f"inputs{sfx}.json").read_text())
    bad = [f"{x['window']}/{x['frame']}" for x in inp["frames"] if sha256_file(out_d / x["png"]) != x["sha256"]]
    if bad:
        raise SystemExit(f"{WHO}: {len(bad)} frames do not hash as step A wrote them, first {bad[:3]}")
    if any(sha256_file(Path(x["image"])) != x["sha256"] for x in inp["r2"]):
        raise SystemExit(f"{WHO}: an R2 frame does not hash as step A read it")
    ws = T.workspace()
    if sha256_file(ws / "weights" / T.TEACHERS[TEACHER]["weights"]) != TEACHER_SHA:
        raise SystemExit(f"{WHO}: the teacher's weights are not the pinned ones")
    watch_t = lambda: {"preds": sha256_file(ws / "preds" / f"{TEACHER}.json"),
                       "weights": sha256_file(ws / "weights" / T.TEACHERS[TEACHER]["weights"])}
    before_t = watch_t()
    torch.set_num_threads(THREADS)
    t0 = time.time()
    detect, versions = T.load_runner(TEACHER, ws, "cpu")
    rnd = lambda bs: [[round(v, 2) for v in p[:4]] + [round(p[4], 5), p[5]] for p in bs]
    r2_bad = []
    for x in inp["r2"]:
        mine = rnd(detect(cv2.imread(x["image"])))
        if mine != x["stored"]:
            r2_bad.append(x["frame"])
    if r2_bad:
        raise SystemExit(f"{WHO}: R2 fails on {len(r2_bad)} of {len(inp['r2'])} frames (first {r2_bad[:3]})")
    boxes = {}
    for x in inp["frames"]:
        if time.time() - t0 > CAP_B_S:
            raise SystemExit(f"{WHO}: step B passed its {CAP_B_S // 60}-minute cap; nothing is recorded")
        img = cv2.imread(str(out_d / x["png"]))
        if img is None:
            raise SystemExit(f"{WHO}: cannot read {x['png']}")
        boxes[f"{x['window']}/{x['frame']}"] = rnd(detect(img))
    if watch_t() != before_t:
        raise SystemExit(f"{WHO}: the teachers workspace changed during step B; nothing is recorded")
    out = {"what": "P8 C1 reach stage 1, step B (the held teacher)", "teachers_workspace": before_t, "inputs_head": inp["head"], "smoke": args.smoke,
           "versions": {**versions, "opencv": cv2.__version__}, "threads": THREADS, "device": "cpu",
           "weights_sha256": TEACHER_SHA, "r2": {"frames": len(inp["r2"]), "equal": True}, "boxes": boxes,
           "seconds": round(time.time() - t0, 1)}
    atomic_write(out_d / f"teacher{sfx}.json", json.dumps(out))
    print(json.dumps({"frames": len(boxes), "r2": out["r2"], "seconds": out["seconds"]}))
    return out


# ── step C: report (system python, no lock) ────────────────────────────────────────────────────────────────────────
def aggregate(wstats: dict, per_window: dict, games: tuple = ALLOWED) -> dict:
    """Scaled counts per (game, unit, subclass) summed over the given windows."""
    out = {}
    for s, st in wstats.items():
        g = per_window[s]["game"]
        for (u, c), d in scaled(st, per_window[s]).items():
            acc = out.setdefault((g, u, c), {k: 0.0 for k in d})
            for k, v in d.items():
                acc[k] += v
    return out


def ceilings(agg: dict, f: dict, n_in: dict) -> dict:
    """10.1 and 10.7: per game and pooled (cha plus bkn): C of the beyond-floor reach (P-i at f_plc, D-i at f_pres),
    the two extremes, and H (the unreached C1a_other and C1c misses as if all reached)."""
    out, tot = {}, {k: 0.0 for k in ("C", "C_all_plc", "C_all_pres", "H")}
    for g in ALLOWED:
        bD = sum(v["beyond"] for (gg, u, c), v in agg.items() if gg == g and u == "D-i")
        bP = sum(v["beyond"] for (gg, u, c), v in agg.items() if gg == g and u == "P-i")
        hD = sum(v["unreached_oc"] for (gg, u, c), v in agg.items() if gg == g and u == "D-i")
        hP = sum(v["unreached_oc"] for (gg, u, c), v in agg.items() if gg == g and u == "P-i")
        fp, fl = f[g]["f_pres"], f[g]["f_plc"]
        r = {"C": ceiling(bD, bP, fp, fl, n_in[g]), "C_all_plc": ceiling(bD + bP, 0, fl, fl, n_in[g]),
             "C_all_pres": ceiling(bD + bP, 0, fp, fp, n_in[g]), "H": ceiling(hD, hP, fp, fl, n_in[g])}
        out[g] = r
        for k in tot:
            tot[k] += r[k] * n_in[g]
    n = sum(n_in[g] for g in ALLOWED)
    out["pooled"] = {k: v / n for k, v in tot.items()}
    return out


def union_baseline(e1b_C_cha: float) -> dict:
    """10.12: the gap stage 2 would face on gsw_cha_2016, from E1b's C (the held teacher's labels) to the -0.010 bar,
    registered and with the transfer model's residual added."""
    from tools.p8_e1b_bound import GATE_GAME
    with_res = e1b_C_cha + CHA_RESIDUAL
    return {"E1b_C_cha": e1b_C_cha, "E1b_C_cha_with_residual": with_res, "gap": e1b_C_cha - GATE_GAME,
            "gap_with_residual": with_res - GATE_GAME}


def cmd_report(args) -> dict:
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
    from tools.p8_common import REPO, atomic_write
    from tools.p8_e1b_bound import TKEYS
    r = RB.roots()
    out_d = c1_dir(r["ref"])
    sfx = "__smoke" if args.smoke else ""
    inp = json.loads((out_d / f"inputs{sfx}.json").read_text())
    tea = json.loads((out_d / f"teacher{sfx}.json").read_text())
    if tea["inputs_head"] != inp["head"] or tea["weights_sha256"] != TEACHER_SHA:
        raise SystemExit(f"{WHO}: step B's record does not belong to step A's")
    allowed(sorted(inp["per_window"]))
    wstats, hits, hits_r = {}, 0, 0
    cost = {k: {"boxes": 0, "ghosts": 0, "near_no_player": 0, "duplicates": 0} for k in ("teacher", "r23_040", "r23_floor")}
    for x in inp["frames"]:
        st = frame_stats(x, tea["boxes"][f"{x['window']}/{x['frame']}"])
        w = wstats.setdefault(x["window"], {"misses": []})
        w["misses"] += st["misses"]
        hits += st["hits"]; hits_r += st["hits_reached"]
        for k, v in st["cost"].items():
            for kk, vv in v.items():
                cost[k][kk] += vv
    pos = hits_r / hits if hits else None
    total_c1 = sum(v["n"] for v in inp["per_window"].values())
    drop_w = sum(inp["per_window"][s]["n"] for s in inp["undetermined"])
    drop_f = sum(x["c1"] for x in inp["r1_failed"])
    sampled_c1 = sum(len(x["c1"]) for x in inp["frames"])
    windows_used = {x["window"] for x in inp["frames"]}
    represented = sum(inp["per_window"][s]["n"] for s in windows_used)
    out = {"what": "P8 C1 reach stage 1 (docs/research/P8_c1reach.md v1.1); report only; the held-teacher reference set",
           "inputs_head": inp["head"], "smoke": args.smoke, "r0": inp["r0"], "r1_failed": len(inp["r1_failed"]),
           "undetermined": inp["undetermined"], "frames": len(inp["frames"]),
           "denominators": {"c1_misses": total_c1, "in_undetermined_windows": drop_w, "on_r1_failed_frames": drop_f,
                            "sampled": sampled_c1, "in_windows_with_frames": represented,
                            "share_represented": (represented / total_c1) if total_c1 else None},
           "positive_control": {"r23_hits": hits, "teacher_reached": hits_r, "share": pos,
                                "passes": pos is not None and pos >= POS_CTRL_MIN},
           "cost_rows": cost, "versions_teacher": tea["versions"]}
    if not out["positive_control"]["passes"]:
        out["note"] = "the positive control fails; stage 1 reports nothing else"
        _write(REPO, args.smoke, out, atomic_write)
        return out
    # shares per subclass and unit, sampled, with a window bootstrap
    ws = sorted(wstats)
    def share(rows, key):
        n = sum(1 for m in rows)
        return (sum(1 for m in rows if m[key]) / n) if n else None
    tab = {}
    by_g = {g: [i for i, s in enumerate(ws) if s.startswith(g + "_")] for g in ALLOWED}

    def strat_picks(scope):
        """10.12: the share bootstrap stratified by scope: a game's windows within the game; pooled, within each."""
        rng = np.random.default_rng(BOOT_SEED)
        groups = [by_g[scope]] if scope in by_g else [by_g[g] for g in ALLOWED]
        if sum(len(x) for x in groups) < 2:
            return None
        return np.array([np.concatenate([np.asarray(gr)[rng.integers(0, len(gr), len(gr))] for gr in groups if gr])
                         for _ in range(BOOT_N)])
    for scope in ALLOWED + ("pooled",):
        picks = strat_picks(scope)
        for u in ("P-i", "D-i", "both"):
            for c in SUBCLASSES + ("C1",):
                sel = lambda m: ((u == "both" or m["unit"] == u) and (c == "C1" or m["subclass"] == c))
                per_w = [[m for m in wstats[s]["misses"] if sel(m) and (scope == "pooled" or s.startswith(scope + "_"))]
                         for s in ws]
                rows = [m for w in per_w for m in w]
                ent = {"n": len(rows)}
                for key in ("teacher", "floor", "teacher_with_band"):
                    ent[key] = share(rows, key)
                beyond = [dict(m, beyond=m["teacher"] and not m["floor"]) for m in rows]
                ent["beyond"] = share(beyond, "beyond")
                if picks is not None and rows:
                    num = np.array([sum(1 for m in w if m["teacher"] and not m["floor"]) for w in per_w], float)
                    den = np.array([len(w) for w in per_w], float)
                    sn, sd = num[picks].sum(1), den[picks].sum(1)
                    v = sn[sd > 0] / sd[sd > 0]
                    ent["beyond_interval"] = [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] \
                        if len(v) else None
                tab[f"{u}|{c}|{scope}"] = ent
    out["shares"] = tab
    # ceilings with f's uncertainty (E1's windows resampled jointly, 10.5)
    d202 = {g: RC.read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json") for g in ALLOWED}
    n_in = {g: sum(w["in_frame"] for w in d202[g]["windows"].values()) for g in ALLOWED}
    e1 = e1_per_window(r["ws"])
    rec = json.loads((REPO / "reports" / "research" / "p8_e1b_transfer.json").read_text())
    for g in ALLOWED:
        mine = {t: sum(e1[k][t] for k in e1 if k.startswith(g + "/")) for t in TKEYS}
        if mine != rec["counts"][g]:
            raise SystemExit(f"{WHO}: E1's per-window counts do not reproduce E1b's record on {g}")
    e1_keys = {g: sorted(k for k in e1 if k.startswith(g + "/")) for g in ALLOWED}
    f_point = {g: e1_fractions(e1, e1_keys[g]) for g in ALLOWED}
    agg = aggregate(wstats, inp["per_window"])
    point = ceilings(agg, f_point, n_in)
    rng = np.random.default_rng(BOOT_SEED)
    ws_by_g = {g: [s for s in ws if s.startswith(g + "_")] for g in ALLOWED}
    reps, reps_f = [], []
    for _ in range(BOOT_N):
        f_b = {g: e1_fractions(e1, [e1_keys[g][i] for i in rng.integers(0, len(e1_keys[g]), len(e1_keys[g]))])
               for g in ALLOWED}
        w_b = {}
        for g in ALLOWED:
            idx = rng.integers(0, len(ws_by_g[g]), len(ws_by_g[g])) if ws_by_g[g] else []
            for k, i in enumerate(idx):
                w_b[f"{ws_by_g[g][i]}#{k}"] = wstats[ws_by_g[g][i]]
        pw_b = {k: inp["per_window"][k.split("#")[0]] for k in w_b}
        reps.append(ceilings(aggregate(w_b, pw_b), f_b, n_in))
        reps_f.append(ceilings(agg, f_b, n_in))
    iv = lambda rs, g, k: [round(float(np.percentile([x[g][k] for x in rs], 2.5)), 5),
                           round(float(np.percentile([x[g][k] for x in rs], 97.5)), 5)]
    out["ceilings"] = {g: {k: {"point": round(v, 5), "interval": iv(reps, g, k), "f_only_interval": iv(reps_f, g, k)}
                           for k, v in point[g].items()} for g in point}
    out["union_baseline"] = {k: round(v, 5) for k, v in union_baseline(rec["rates"]["gsw_cha_2016"]["C"]).items()}
    out["fractions"] = f_point
    out["in_frame"] = n_in
    out["stage2_bar"] = {"bar": BAR, "kill": KILL, "note": "stage 1 decides nothing (10.2); H is reported"}
    _write(REPO, args.smoke, out, atomic_write)
    print(json.dumps({"positive_control": out["positive_control"], "ceilings": {g: {k: v["point"] for k, v in d.items()}
                                                                                for g, d in out["ceilings"].items()}},
                     indent=1))
    return out


def _write(repo: Path, smoke: bool, out: dict, atomic_write) -> None:
    from export.schemas import find_private
    text = json.dumps(out, indent=1, default=float)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(repo / "reports" / "research" / f"p8_c1reach_stage1{'_smoke' if smoke else ''}.json", text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prepare", "teacher", "report"):
        sp = sub.add_parser(c)
        sp.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    {"prepare": cmd_prepare, "teacher": cmd_teacher, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
