"""tools/p8_nmsarm_stage1.py: stage 1a (identity and positive control) and the 0.55 pilot of the NMS arm
(docs/research/P8_nmsarm.md, versions 1.1 and 1.2, sections 12 and 14). Report only: nothing is fitted, tuned or adopted;
builds go to new folders of the reference root's builds/, never builds/r23. The confirmation is not written.

    <frozen CV python> -m tools.p8_nmsarm_stage1 build --stage identity|control|pilot   inside the lock
    <frozen CV python> -m tools.p8_nmsarm_stage1 score --stage identity|control|pilot   inside the lock
    python3 -m tools.p8_nmsarm_stage1 gate                                             system python, the pilot gate
    (child: python -m tools.p8_nmsarm_stage1 child build_trajectories.py ..., run by the build subprocess only)

Records: <reference root>/nmsarm/build__<stage>.json; reports/research/p8_nmsarm_<stage>.json (scores and gates).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

WHO = "tools.p8_nmsarm_stage1"
ENV = "P8_NMSARM_DETECTOR"                 # the r23 weights path; set only by cmd_build
IOU_ENV = "P8_NMSARM_IOU"
BUILD_SCRIPT = "build_trajectories.py"
ARM_IOU, ID_IOU = 0.55, 0.45
IOUS = (ID_IOU, ARM_IOU)
IDENTITY = ("gsw_cha_2016_s05_w00", "gsw_bkn_2015_s04_w00", "gsw_cha_2016_s00_w00", "gsw_cha_2016_s02_w01")
CONTROL = "gsw_cha_2016_s05_w00"
FOLDERS = {"identity": "nmsarm_iou045_identity", "control": "nmsarm_iou055_control", "pilot": "nmsarm_iou055_pilot"}
STAGE_IOU = {"identity": ID_IOU, "control": ARM_IOU, "pilot": ARM_IOU}
PILOT_MOD = 4
N_PILOT, N_CONFIRM = 31, 90
P3_FACTOR = 0.7
DUP_MARGIN = 0.005
MECH_CEILING = 0.0098                      # 12.7: the larger measured C2 share (kept windows)
SEC_PER_FRAME = 0.44
CAP_FACTOR = 1.5
BOOT_N, BOOT_SEED = 2000, 0


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def scorable(d202_windows: dict) -> list:
    """12.8: the windows with at least one in-frame player in D202's record, sorted."""
    return sorted(s for s, w in d202_windows.items() if w.get("in_frame", 0) > 0)


def split(names: list) -> tuple:
    """12.8: (pilot, confirmation): every index divisible by 4 in the sorted list is a pilot window."""
    names = sorted(names)
    return [s for i, s in enumerate(names) if i % PILOT_MOD == 0], [s for i, s in enumerate(names) if i % PILOT_MOD]


def iou_from_env(env=None) -> float:
    env = os.environ if env is None else env
    v = env.get(IOU_ENV)
    try:
        t = float(v)
    except (TypeError, ValueError):
        raise SystemExit(f"{WHO}: {IOU_ENV} must be one of {IOUS}, not {v!r}")
    if t not in IOUS:
        raise SystemExit(f"{WHO}: {IOU_ENV} must be one of {IOUS}, not {v!r}")
    return t


def patch_iou(t: float, tracker_cls=None) -> str:
    """14.1: wrap PlayerTracker.__init__ (whatever wrapper is already there) so every tracker gets iou=t; the tracker
    passes self.iou to model.track on every update, so it holds after reset()."""
    if tracker_cls is None:
        import detect.tracker as DT
        tracker_cls = DT.PlayerTracker
    init = tracker_cls.__init__

    def tracker_init(self, *a, **k):
        k["iou"] = t
        init(self, *a, **k)
    tracker_cls.__init__ = tracker_init
    return f"{tracker_cls.__module__}.{tracker_cls.__name__}.__init__"


def rewrite(cmd, env=None):
    env = os.environ if env is None else env
    if env.get(ENV) and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[1] == BUILD_SCRIPT:
        return [cmd[0], "-m", "tools.p8_nmsarm_stage1", "child", *cmd[1:]]
    return cmd


def record_problems(dev: dict | None, det: dict | None, t: float) -> list:
    """One arm window's device and detector records against 14.1."""
    from tools.p8_resetfix import REFERENCE, R23_SHA
    bad = []
    if dev is None or dev.get("requested") != REFERENCE or (dev.get("resolved") or {}).get("detector") != "cpu":
        bad.append("device record is not the reference platform with the detector on cpu")
    if dev is not None and dev.get("clip_fix"):
        bad.append("built with the MPS clip fix")
    if dev is None or dev.get("reset_fix") is not True:
        bad.append("device record does not carry reset_fix")
    if dev is None or dev.get("detector_iou") != t:
        bad.append(f"device record does not carry detector_iou {t}")
    if det is None or det.get("sha256") != R23_SHA or det.get("model") != "r23":
        bad.append("detector record is not r23")
    return bad


def same_fields(mine: dict, want: dict) -> list:
    """The GATE_FIELDS where a freshly scored window differs from D202's stored one (compared after a JSON round trip)."""
    from tools.p8_rebaseline import GATE_FIELDS
    norm = lambda x: json.loads(json.dumps(x))
    return [f for f in GATE_FIELDS if norm(mine.get(f)) != norm(want.get(f))]


def side_boxes(side: dict) -> int:
    return sum(len(x["boxes"]) for x in side["frames"])


def arm_only(base_obs: list, arm_obs: list) -> tuple:
    """(players missed in the baseline and hit in the arm, hit in the baseline and missed in the arm), as (frame, pid)
    pairs on the frames both sides scored."""
    b = {(o["f"], o["pid"]): o["missed"] for o in base_obs}
    a = {(o["f"], o["pid"]): o["missed"] for o in arm_obs}
    both = sorted(set(a) & set(b))
    return [k for k in both if b[k] and not a[k]], [k for k in both if a[k] and not b[k]]


def pilot_gate(prev: list, curr: list, mech: dict, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """14.4 on the pilot's paired windows (sorted, baseline and arm records) and the mechanism counts."""
    from tools.p8_rebaseline import change_rows
    from tools.p8_resetfix import ratio_interval, resample_picks
    ch = change_rows(prev, curr, n_boot, seed)
    picks = resample_picks(len(prev), n_boot, seed) if len(prev) >= 2 else None
    bn, bd = [w["duplicates"] for w in prev], [w["in_frame"] for w in prev]
    nn, nd = [w["duplicates"] for w in curr], [w["in_frame"] for w in curr]
    dup = {"change": (sum(nn) / sum(nd) - sum(bn) / sum(bd)) if sum(nd) and sum(bd) else None,
           "interval": ratio_interval(picks, bn, bd, nn, nd) if picks is not None else None}
    d1, iv1 = ch["net_error_1to1"]["change"], ch["net_error_1to1"]["interval"]
    dm = ch["missed"]["change"]
    half = None if iv1 is None else (iv1[1] - iv1[0]) / 2.0
    p3_proj = None if half is None else half * float(np.sqrt(N_PILOT / N_CONFIRM))
    hits = mech.get("arm_only_hits_by_class", {})
    n_hits = sum(hits.values())
    flag_ceiling = dm is not None and -dm > MECH_CEILING
    flag_pii = n_hits > 0 and hits.get("P-ii", 0) > 0.5 * n_hits
    g = {"P1_missed_falls": dm is not None and dm < 0,
         "P2_kill": d1 is None or d1 >= 0,
         "P3_powered": d1 is not None and p3_proj is not None and p3_proj < P3_FACTOR * abs(d1),
         "duplicate_guard": dup["change"] is not None and dup["change"] <= DUP_MARGIN,
         "mechanism_flag": flag_ceiling or flag_pii}
    if g["P2_kill"]:
        outcome = "KILL"
    elif g["P1_missed_falls"] and g["P3_powered"] and g["duplicate_guard"] and not g["mechanism_flag"]:
        outcome = "PASS"
    else:
        outcome = "STOP"
    return {"gates": g, "outcome": outcome, "rows": ch, "duplicates_per_in_frame": dup, "p3_projected_half": p3_proj,
            "mechanism": {**mech, "flag_ceiling": flag_ceiling, "flag_pii": flag_pii}, "windows": len(prev)}


# ── guards and paths ───────────────────────────────────────────────────────────────────────────────────────────────
def allowed(names) -> list:
    """12.5: the allow-list {gsw_cha_2016, gsw_bkn_2015} and the P8 guards on every game and window name."""
    from tools.p8_nmsarm import allowed as allow
    from tools.research_testset import refuse_cv_test
    names = [str(n) for n in names]
    refuse_cv_test(names, WHO)
    return allow(names, WHO)


def score_path(stage: str) -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / f"p8_nmsarm_{stage}.json"


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _write_report(stage: str, out: dict) -> None:
    from export.schemas import find_private
    from tools.p8_common import atomic_write
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(score_path(stage), text)


def stage_problems(stage: str, step: str) -> list:
    """The gates in force: stage 0 chose 0.55; identity before the control, the control before the pilot; a stage's
    build before its score; the pilot's score before its gate."""
    why = []
    scr = _read(score_path("screen"))
    if not scr or (scr.get("verdict") or {}).get("arm") != "0.55":
        why.append("stage 0b did not choose 0.55")
    need = {"identity": [], "control": ["identity"], "pilot": ["identity", "control"]}[stage]
    for s in need:
        rec = _read(score_path(s))
        if not rec or not rec.get("passed"):
            why.append(f"stage {s} has not passed")
    return why


# ── child (frozen CV python, inside the build subprocess) ──────────────────────────────────────────────────────────
def child(argv: list) -> int:
    import runpy
    import sys
    from tools import p8_device as D
    from tools.p8_resetfix import REFERENCE, sha256_file, start_reset_log
    if not argv or argv[0] != BUILD_SCRIPT:
        raise SystemExit(f"{WHO}: usage: python -m {WHO} child {BUILD_SCRIPT} <its arguments>")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    allowed([Path(src).stem])
    req = D.requested()
    if req != REFERENCE or D.clip_fix_requested() or not D.reset_fix_requested():
        raise SystemExit(f"{WHO}: the arm is P8_DEVICE_DETECTOR=cpu with P8_TRACKER_RESET_FIX=1 only")
    if os.environ.get("TRACKER_REID") is not None:
        raise SystemExit(f"{WHO}: TRACKER_REID must be unset")
    t = iou_from_env()
    resolved = D.apply(req)
    patched = patch_iou(t)
    weights = Path(os.environ[ENV])
    import config
    config.PLAYER_DETECTOR_WEIGHTS = weights
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    dev = {**D.device_record(req, resolved), "detector_iou": t, "iou_patched": patched}
    for suffix, rec in (("device", dev), ("detector", {"weights": weights.name, "sha256": sha256_file(weights),
                                                        "model": "r23", "seed": 0})):
        part = config.TRACKING_DIR / f"{stem}_{suffix}.json.part"
        part.write_text(json.dumps(rec))
        os.replace(part, config.TRACKING_DIR / f"{stem}_{suffix}.json")
    start_reset_log(config.TRACKING_DIR / f"{stem}_resetfix.json")
    script = Path(config.PROJECT_ROOT) / BUILD_SCRIPT
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


class _Proxy:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, cmd, *a, **k):
        return self._real.run(rewrite(cmd), *a, **k)


# ── commands ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _common():
    """Lock, reference root, clean tools/ and sportvu/, the pinned D202 records -> (r, head, d202)."""
    import subprocess
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
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
    d202 = {g: RC.read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json") for g in ("gsw_cha_2016", "gsw_bkn_2015")}
    if any(d.get("run_id") != RC.RUN_ID for d in d202.values()):
        raise SystemExit(f"{WHO}: D202's records are not from run {RC.RUN_ID}")
    return r, head, d202


def stage_windows(stage: str, d202: dict) -> list:
    if stage == "identity":
        return list(IDENTITY)
    if stage == "control":
        return [CONTROL]
    out = []
    for g in ("gsw_cha_2016", "gsw_bkn_2015"):
        out += split(scorable(d202[g]["windows"]))[0]
    if len(out) != N_PILOT:
        raise SystemExit(f"{WHO}: {len(out)} pilot windows, not {N_PILOT}")
    return out


def cmd_build(args) -> dict:
    from sportvu import rebuild
    from sportvu.autolabel import gate_args
    from tools import p8_frozen as F
    from tools.p8_common import atomic_write, main_checkout
    from tools.p8_resetfix import R23_SHA, arm_source_problem, sha256_file
    stage = args.stage
    r, head, d202 = _common()
    why = stage_problems(stage, "build")
    if why:
        raise SystemExit(f"{WHO}: stage {stage} may not build: {why}")
    todo = allowed(stage_windows(stage, d202))
    for k in ("P8_DEVICE", "P8_DEVICE_COURT", "P8_DEVICE_GATE", "P8_MPS_CLIP_FIX", "TRACKER_REID"):
        os.environ.pop(k, None)
    os.environ["P8_DEVICE_DETECTOR"] = "cpu"
    os.environ["P8_TRACKER_RESET_FIX"] = "1"
    os.environ[IOU_ENV] = str(STAGE_IOU[stage])
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    if sha256_file(weights) != R23_SHA:
        raise SystemExit(f"{WHO}: the production detector is not r23")
    P = F.install_paths()
    if P["builds"].resolve() != (r["ref"] / "builds").resolve():
        raise SystemExit(f"{WHO}: builds are not the reference root's")
    rebuild.subprocess = _Proxy(rebuild.subprocess)
    os.environ[ENV] = str(weights)
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    out = P["builds"] / FOLDERS[stage]
    if out.resolve() == (P["builds"] / "r23").resolve():
        raise SystemExit(f"{WHO}: refusing to build into builds/r23")
    out.mkdir(parents=True, exist_ok=True)
    wins = {w["window"]: w for g in ("gsw_cha_2016", "gsw_bkn_2015") for w in F.label_windows(P, g)}
    comp = P["builds"] / "r23"
    est = SEC_PER_FRAME * sum(len(_read(comp / f"{s}_frames.json")["frames"]) for s in todo)
    t_start = time.time()
    res = {"stage": stage, "iou": STAGE_IOU[stage], "folder": out.name, "head": head, "estimate_s": round(est),
           "windows": {}}
    for s in todo:
        if time.time() - t_start > CAP_FACTOR * est:
            raise SystemExit(f"{WHO}: stage {stage} passed {CAP_FACTOR} times its estimate; stopped for re-estimate")
        t0 = time.time()
        clip = P["label_build"] / f"{s}.mp4"
        for attempt in (1, 2, 3):
            rr = F.safe_build(P, wins[s], clip, out, extra_args=gate_args())
            if rr["ok"]:
                break
        else:
            raise SystemExit(f"{WHO}: build of {s} failed three times: {rr.get('tail')}")
        bad = record_problems(_read(out / f"{s}_device.json"), _read(out / f"{s}_detector.json"), STAGE_IOU[stage])
        why_src = arm_source_problem(_read(out / f"{s}_frames.json") or {}, s, r["ref"], r["ws"])
        if why_src:
            bad.append(why_src)
        if rr.get("empty") or bad:
            raise SystemExit(f"{WHO}: {s}: {bad or 'every frame gate-rejected'}")
        res["windows"][s] = {"cached": bool(rr.get("cached")), "attempts": attempt,
                             "seconds": None if rr.get("cached") else round(time.time() - t0, 1)}
    res["seconds"] = round(time.time() - t_start, 1)
    d = r["ref"] / "nmsarm"
    d.mkdir(exist_ok=True)
    atomic_write(d / f"build__{stage}.json", json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("stage", "iou", "folder", "seconds", "estimate_s")}))
    return res


def cmd_score(args) -> dict:
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_nmsarm as N0
    from tools import p8_one2one as O
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
    from tools import p8_scorecard as S
    from tools.p8_common import guard
    from tools.p8_placecause import cache_gaps
    from tools.p8_resetfix import compare_window, score_pair
    stage = args.stage
    r, head, d202 = _common()
    why = stage_problems(stage, "score")
    if why:
        raise SystemExit(f"{WHO}: stage {stage} may not be scored: {why}")
    build = _read(r["ref"] / "nmsarm" / f"build__{stage}.json")
    if not build or build.get("head") != head:
        raise SystemExit(f"{WHO}: stage {stage}'s build record is missing or from another head")
    pinned, now = RC.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if [k for k in now if now[k] != pinned.get(k)]:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    todo = allowed(stage_windows(stage, d202))
    if sorted(build["windows"]) != sorted(todo):
        raise SystemExit(f"{WHO}: the build record's windows are not the stage's")
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    comp, arm = P["builds"] / "r23", P["builds"] / FOLDERS[stage]
    raw_d = RC.raw_dir(r["ref"])
    out = {"stage": stage, "iou": STAGE_IOU[stage], "head": head, "windows": {}}
    obs_rows = {}
    for g in ("gsw_cha_2016", "gsw_bkn_2015"):
        mine = [s for s in todo if s.startswith(g + "_")]
        if not mine:
            continue
        gd = F.game_of(g)
        guard([gd["sportvu"], gd["video_id"]], WHO)
        mirror = F.mirror_of(P, g)
        mom = S.Moments(RC.read_json(P["sportvu"] / f"{gd['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": S.pbp_events(gd["sportvu"]),
               "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        P_t = {**P, "labels": RB.truths(r)[g]["current"] / "labels"}
        wins = {w["window"]: w for w in F.label_windows(P, g)}
        for s in mine:
            w = wins[s]
            side_b, side_a = _read(comp / f"{s}_frames.json"), _read(arm / f"{s}_frames.json")
            rec = {"device": _read(arm / f"{s}_device.json")}
            bad = record_problems(rec["device"], _read(arm / f"{s}_detector.json"), STAGE_IOU[stage])
            if bad:
                raise SystemExit(f"{WHO}: {s}: {bad}")
            if stage == "control":
                cw = compare_window(comp, arm, s)
                rec.update({"differs": not all(cw["byte_identical"].values()),
                            "boxes_composite": side_boxes(side_b), "boxes_arm": side_boxes(side_a)})
                rec["passed"] = rec["differs"] and rec["boxes_arm"] > rec["boxes_composite"]
                out["windows"][s] = rec
                continue
            if [int(x["frame"]) for x in side_b["frames"]] != [int(x["frame"]) for x in side_a["frames"]]:
                raise SystemExit(f"{WHO}: {s}: the arm's processed frames are not the composite's")
            for dd in (comp, arm):
                gap = cache_gaps(P_t, g, w, dd, mom, mirror)
                if gap:
                    raise SystemExit(f"{WHO}: {s}: the scorecard cache lacks {gap}")
            pair = score_pair(P_t, g, w, {"r23": comp, "new": arm}, {"r23": side_b, "new": side_a}, ctx)
            diff_b = same_fields(pair["r23"], d202[g]["windows"][s])
            if diff_b:
                raise SystemExit(f"{WHO}: {s}: the baseline side differs from D202's record in {diff_b}")
            rec.update({"baseline": json.loads(json.dumps(pair["r23"])), "arm": json.loads(json.dumps(pair["new"]))})
            if stage == "identity":
                cw = compare_window(comp, arm, s)
                rec["compare"] = cw
                rec["arm_fields_differing"] = same_fields(pair["new"], d202[g]["windows"][s])
                rec["passed"] = all(cw["byte_identical"].values()) and not rec["arm_fields_differing"]
            else:                                       # pilot: the mechanism counts (14.4)
                full_b = S.score_window(P_t, g, w, comp, "camera", mom, mirror, refs, ctx["events"])
                full_a = S.score_window(P_t, g, w, arm, "camera", mom, mirror, refs, ctx["events"])
                hits, losses = arm_only(full_b[1], full_a[1])
                rows = RC.d205_rows(g, P_t, w, comp, ctx, d202[g]["windows"][s], (d202[g].get("hashes") or {}).get(s, {}))
                cls = {(x["frame"], x["pid"]): x for x in rows}
                raw = _read(raw_d / f"{s}.json") if (raw_d / f"{s}.json").exists() else {}
                c2_hits = 0
                for k in hits:
                    x = cls.get(k)
                    if x is None or x["rec"]["class"] not in RC.UNIVERSE or str(k[0]) not in raw:
                        continue
                    v = raw[str(k[0])]
                    pre, post = v["pre"], v["post"]
                    pf, qf = RC.feet_of(pre, x["Ht"]), RC.feet_of(post, x["Ht"])
                    supp = RC.suppressors(pre, post, list(range(len(pre))), N0.post_ids_of(pre, post))
                    main, _ = O.two_stage(qf, x["sv"], range(len(x["sv"])), range(len(post)), O._assign_all)
                    if main.get(x["player"]) is None and N0.c2_candidates(x["player"], x["sv"], pf, qf, supp):
                        c2_hits += 1
                by = lambda ks: {c: sum(1 for k in ks if (cls.get(k) or {"rec": {"class": "unclassed"}})["rec"]["class"] == c)
                                 for c in sorted({(cls.get(k) or {"rec": {"class": "unclassed"}})["rec"]["class"] for k in ks})}
                rec["mechanism"] = {"arm_only_hits": len(hits), "arm_only_misses": len(losses), "hits_by_class": by(hits),
                                    "misses_by_class": by(losses), "c2_hits_on_stored_frames": c2_hits}
                obs_rows[s] = rec["mechanism"]
            out["windows"][s] = rec
    out["passed"] = all(v.get("passed", True) for v in out["windows"].values()) if stage != "pilot" else None
    out["seconds"] = round(time.time() - t0, 1)
    _write_report(stage, out)
    print(json.dumps({"stage": stage, "passed": out["passed"], "windows": len(out["windows"]), "seconds": out["seconds"]}))
    return out


def cmd_gate(args) -> dict:
    """System python: the pilot gate (14.4) from the pilot's score record."""
    rec = _read(score_path("pilot"))
    if not rec:
        raise SystemExit(f"{WHO}: no pilot score record")
    ws = sorted(rec["windows"])
    allowed(ws)
    if len(ws) != N_PILOT:
        raise SystemExit(f"{WHO}: {len(ws)} pilot windows scored, not {N_PILOT}")
    prev, curr = [rec["windows"][s]["baseline"] for s in ws], [rec["windows"][s]["arm"] for s in ws]
    hits = {}
    for s in ws:
        for c, n in rec["windows"][s]["mechanism"]["hits_by_class"].items():
            hits[c] = hits.get(c, 0) + n
    mech = {"arm_only_hits_by_class": hits,
            "arm_only_misses": sum(rec["windows"][s]["mechanism"]["arm_only_misses"] for s in ws),
            "c2_hits_on_stored_frames": sum(rec["windows"][s]["mechanism"]["c2_hits_on_stored_frames"] for s in ws)}
    out = {"what": "P8 NMS arm pilot gate (docs/research/P8_nmsarm.md 14.4); report only", "pilot_head": rec["head"],
           **pilot_gate(prev, curr, mech)}
    _write_report("pilot_gate", out)
    print(json.dumps({"outcome": out["outcome"], "gates": out["gates"]}, indent=1))
    return out


def main(argv=None) -> None:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "child":
        sys.exit(child(argv[1:]))
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "score"):
        sp = sub.add_parser(c)
        sp.add_argument("--stage", required=True, choices=("identity", "control", "pilot"))
    sub.add_parser("gate")
    args = ap.parse_args(argv)
    {"build": cmd_build, "score": cmd_score, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
