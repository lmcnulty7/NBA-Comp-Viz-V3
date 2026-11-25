"""tools/p8_mps_fix.py: M8, the tools-side MPS clip fix (P8_MPS_CLIP_FIX in tools/p8_device.py) checked at build level
against the M6 CPU builds (docs/research/P8_E1.md, "M8", definitions committed before any number). Report only.

    <frozen CV python> -m tools.p8_mps_fix rebuild --model {r23,e1}     inside the heavy-job lock
    <frozen CV python> -m tools.p8_mps_fix compare                       inside the heavy-job lock
    (internal) <python> -m tools.p8_mps_fix child build_trajectories.py <its arguments>

Builds go only to builds/m8_r23_mpsfix and builds/m8_e1_gsw_bkn_2015_mpsfix (any other folder is refused). The child
sets the detector weights and records them, then hands over to tools.p8_device's child, which applies the clip fix
(P8_MPS_CLIP_FIX=1, set by this tool), writes the device record and runs the frozen build unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_mps_fix"
GAME = FOLD = "gsw_bkn_2015"
WINDOWS = tuple(f"gsw_bkn_2015_s0{k}_w00" for k in (4, 5, 6, 7, 8))
FOLDERS = {"r23": "m8_r23_mpsfix", "e1": "m8_e1_gsw_bkn_2015_mpsfix"}
CPU = {"r23": "e1m6_r23_detcpu", "e1": "e1m6_e1_gsw_bkn_2015_detcpu"}
REGISTERED = ("r23", "e1_gsw_bkn_2015_seed0")
TOL_PX = 2.0
BUILD_SCRIPT = "build_trajectories.py"
ENV = "P8_M8_DETECTOR"
M6_CPU_SECONDS = {"e1": 8905.8, "r23": 8754.7, "windows": 80}     # M6 run log (P8_E1.md, M6 results)


def check_guard() -> None:
    refuse_cv_test([GAME], WHO)
    guard([GAME], WHO)


def folder(model: str) -> str:
    if model not in FOLDERS:
        raise SystemExit(f"{WHO}: unknown model {model}")
    name = FOLDERS[model]
    if name in REGISTERED or name in CPU.values() or not name.startswith("m8_"):
        raise SystemExit(f"{WHO}: {name} is not an M8 folder")
    return name


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def box_gap(a: dict, b: dict) -> float:
    """The largest coordinate difference of two sidecar boxes over bbox and foot_stab (px)."""
    return float(max(max(abs(x - y) for x, y in zip(a["bbox"], b["bbox"])),
                     max(abs(x - y) for x, y in zip(a["foot_stab"], b["foot_stab"]))))


def frame_match(a: list, b: list, tol: float = TOL_PX) -> dict:
    """Two frames' sidecar boxes -> same count, all matched within tol (one to one, Hungarian on box_gap), worst gap."""
    from scipy.optimize import linear_sum_assignment
    if len(a) != len(b):
        return {"same_count": False, "ok": False, "worst": None}
    if not a:
        return {"same_count": True, "ok": True, "worst": 0.0}
    G = np.array([[box_gap(x, y) for y in b] for x in a])
    ri, ci = linear_sum_assignment(G)
    worst = float(G[ri, ci].max())
    return {"same_count": True, "ok": worst <= tol, "worst": worst}


def window_match(side_a: dict, side_b: dict, tol: float = TOL_PX) -> dict:
    fa = {int(r["frame"]): r["boxes"] for r in side_a["frames"]}
    fb = {int(r["frame"]): r["boxes"] for r in side_b["frames"]}
    same_frames = fa.keys() == fb.keys()
    rows = [frame_match(fa[f], fb[f], tol) for f in sorted(fa.keys() & fb.keys())]
    worst = [r["worst"] for r in rows if r["worst"] is not None]
    return {"same_frames": same_frames, "frames": len(fa), "frames_other": len(fb),
            "frames_same_count": sum(r["same_count"] for r in rows), "frames_within_tol": sum(r["ok"] for r in rows),
            "worst_gap_px": max(worst) if worst else None,
            "boxes_ok": same_frames and all(r["ok"] for r in rows)}


def rates(rec: dict) -> dict:
    nb = rec["boxes"] - rec["ref_boxes"]
    return {"missed_rate": round(rec["missed"] / rec["in_frame"], 4) if rec["in_frame"] else None,
            "ghost_rate": round((rec["ghosts"] - rec["ref_ghosts"]) / nb, 4) if nb else None,
            "missed": rec["missed"], "in_frame": rec["in_frame"]}


# ── build ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class _Proxy:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, cmd, *a, **k):
        return self._real.run(rewrite(cmd), *a, **k)


def rewrite(cmd, env=None):
    env = os.environ if env is None else env
    if env.get(ENV) and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[1] == BUILD_SCRIPT:
        return [cmd[0], "-m", "tools.p8_mps_fix", "child", *cmd[1:]]
    return cmd


def child(argv: list) -> int:
    from tools import p8_device as D
    if not D.clip_fix_requested():
        raise SystemExit(f"{WHO}: the child needs P8_MPS_CLIP_FIX=1")
    if any(v != "auto" for v in D.requested().values()):
        raise SystemExit(f"{WHO}: M8 builds leave every device on auto (MPS), as registered")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    guard([src], WHO)
    weights = Path(os.environ[ENV])
    import config
    config.PLAYER_DETECTOR_WEIGHTS = weights
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    part = config.TRACKING_DIR / f"{stem}_detector.json.part"
    part.write_text(json.dumps({"weights": weights.name, "sha256": sha(weights), "model": os.environ.get("P8_M8_MODEL")}))
    os.replace(part, config.TRACKING_DIR / f"{stem}_detector.json")
    return D.main(argv)                       # clip fix, device record, the frozen build unchanged


def cmd_rebuild(args) -> dict:
    from sportvu import rebuild
    from sportvu.autolabel import gate_args
    from tools import p8_device as D
    from tools import p8_frozen as F
    from tools.p8_e1_m4 import weights_of
    from tools.p8_e1_score import check_provenance
    check_guard()
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    for k in ("P8_DEVICE", *D.FLAGS.values()):
        if os.environ.get(k):
            raise SystemExit(f"{WHO}: {k} is set; M8 builds leave every device on auto")
    os.environ[D.CLIP_FIX_FLAG] = "1"
    if args.model == "e1":
        weights = weights_of(FOLD).resolve()
        check_provenance(FOLD, GAME, 0, weights)
    else:
        weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    P = F.install_paths()
    rebuild.subprocess = _Proxy(rebuild.subprocess)
    os.environ[ENV], os.environ["P8_M8_MODEL"] = str(weights), args.model
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    out = P["builds"] / folder(args.model)
    out.mkdir(parents=True, exist_ok=True)
    wins = {w["window"]: w for w in F.label_windows(P, GAME)}
    want = sha(weights)
    res = {"model": args.model, "weights_sha256": want, "windows": {}}
    for s in WINDOWS:
        t0 = time.time()
        r = F.safe_build(P, wins[s], P["label_build"] / f"{s}.mp4", out, extra_args=gate_args())
        secs = time.time() - t0
        if not r["ok"]:
            raise SystemExit(f"{WHO}: build of {s} failed: {r.get('tail')}")
        det = json.loads((out / f"{s}_detector.json").read_text())
        dev = json.loads((out / f"{s}_device.json").read_text())
        if det["sha256"] != want or not dev.get("clip_fix"):
            raise SystemExit(f"{WHO}: {s} did not record the weights or the clip fix")
        frames = len(json.loads((out / f"{s}_frames.json").read_text())["frames"])
        res["windows"][s] = {"cached": bool(r.get("cached")), "seconds": None if r.get("cached") else round(secs, 1),
                             "frames": frames, "device": dev}
    atomic_write(p8_root() / "parity" / f"m8_rebuild__{args.model}.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def cmd_compare(args) -> dict:
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    check_guard()
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    g = F.game_of(GAME)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, GAME)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    wins = {w["window"]: w for w in F.label_windows(P, GAME)}
    out = {"game": GAME, "tolerance_px": TOL_PX, "windows": {}, "cost": {}}
    passed = True
    for m in FOLDERS:
        fix_dir, cpu_dir = P["builds"] / folder(m), P["builds"] / CPU[m]
        for s in WINDOWS:
            if not (P["cache"] / "scorecard" / f"{s}.json").exists():
                raise SystemExit(f"{WHO}: {s} has no cached scorecard references; refusing to compute them")
            wm = window_match(json.loads((fix_dir / f"{s}_frames.json").read_text()),
                              json.loads((cpu_dir / f"{s}_frames.json").read_text()))
            sc = {k: rates(S.score_window(P, GAME, wins[s], d, "camera", mom, mirror, refs, events)[0])
                  for k, d in (("mps_fix", fix_dir), ("cpu", cpu_dir))}
            same_rates = all(sc["mps_fix"][k] == sc["cpu"][k] for k in ("missed_rate", "ghost_rate"))
            ok = wm["boxes_ok"] and same_rates
            passed &= ok
            out["windows"][f"{m}/{s}"] = {**wm, "score": sc, "rates_equal_4dp": same_rates, "passes": ok}
        reb = json.loads((p8_root() / "parity" / f"m8_rebuild__{m}.json").read_text())["windows"]
        secs = sum(v["seconds"] for v in reb.values() if v["seconds"] is not None)
        frames = sum(v["frames"] for v in reb.values() if v["seconds"] is not None)
        out["cost"][m] = {"mps_fix_seconds": round(secs, 1), "mps_fix_frames": frames,
                          "mps_fix_seconds_per_frame": round(secs / frames, 4) if frames else None,
                          "cpu_m6_seconds_per_frame": round(M6_CPU_SECONDS[m] / (20644 - 859), 4),
                          "cpu_m6_note": "M6 full-build seconds over its 80 built windows, divided by their 19,785 "
                                         "processed frames (the game's 20,644 less the 3 pilot windows' 859)"}
    out["passes"] = passed
    atomic_write(p8_root() / "parity" / "p8_mps_fix.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "child":
        return child(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m tools.p8_mps_fix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--model", required=True, choices=tuple(FOLDERS))
    sub.add_parser("compare")
    a = ap.parse_args(argv)
    {"rebuild": cmd_rebuild, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
