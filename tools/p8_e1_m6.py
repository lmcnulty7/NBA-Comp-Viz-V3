"""tools/p8_e1_m6.py: E1 post-mortem M6, gsw_bkn_2015 rebuilt with the player detector on CPU, r23 and E1 (fold
gsw_bkn_2015 seed 0) (docs/research/P8_E1.md, "M5 and M6", definitions committed before any number). Report only.

    <frozen CV python> -m tools.p8_e1_m6 rebuild --model {r23,e1} [--windows N]     inside the heavy-job lock
    <frozen CV python> -m tools.p8_e1_m6 score                                         inside the heavy-job lock
    python3 -m tools.p8_e1_m6 drop                                                    M4 raw boxes against the CPU sidecars
    python3 -m tools.p8_e1_m6 report                                                  -> e1/p8_e1_m6.json
    (internal) <python> -m tools.p8_e1_m6 child build_trajectories.py <its arguments>

Never touches the registered result: builds go only to builds/e1m6_r23_detcpu and builds/e1m6_e1_gsw_bkn_2015_detcpu
(any other name is refused), the score of both models only to one combined file, e1/m6__e1__gsw_bkn_2015.json,
which tools.p8_e1_score report never reads. tools.p8_e1_score is imported for its provenance check, window_stats and the bootstrap, never run.

How: as tools.p8_e1_score rebuild (tools.p8_frozen.safe_build, gate v2 arguments, court model r23), with a proxy
around sportvu.rebuild's subprocess that runs the build as `python -m tools.p8_e1_m6 child build_trajectories.py ...`.
The child applies tools.p8_device's patch for P8_DEVICE_DETECTOR=cpu (court and gate left on auto, as registered),
writes the window's device record and detector record, sets config.PLAYER_DETECTOR_WEIGHTS and runs the frozen
build_trajectories.py unchanged with runpy.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

from tools.p8_common import atomic_write, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_m6"
GAME = "gsw_bkn_2015"
FOLD = "gsw_bkn_2015"
MODELS = ("r23", "e1")
BUILD_SCRIPT = "build_trajectories.py"
ENV = "P8_E1_M6_DETECTOR"
DEVICE_ENV = {"P8_DEVICE_DETECTOR": "cpu"}
REGISTERED = {"r23": 0.2605, "e1": 0.2935}        # docs/research/P8_E1.md, per game table (MPS builds)
RUN_LOG = {"source": "run log of 2026-11-24 to 25 (heavy-job output), not recomputed",
           "pilot_seconds": {"e1": 389.9, "r23": 380.8, "windows": 3, "frames": 859},
           "full_build_seconds": {"e1": 8905.8, "r23": 8754.7}}


def tally_records(build: Path, game: str = GAME) -> dict:
    """Counts of each window's device record and detector weights record in an M6 build folder."""
    dev, wts = {}, {}
    for p in sorted(Path(build).glob(f"{game}_*_device.json")):
        k = json.dumps(json.loads(p.read_text())["resolved"], sort_keys=True)
        dev[k] = dev.get(k, 0) + 1
    for p in sorted(Path(build).glob(f"{game}_*_detector.json")):
        k = json.loads(p.read_text())["sha256"][:12]
        wts[k] = wts.get(k, 0) + 1
    return {"device_records": dev, "weights_sha256_prefix": wts}


def build_dir_name(model: str) -> str:
    if model not in MODELS:
        raise SystemExit(f"{WHO}: unknown model {model}; one of {MODELS}")
    name = "e1m6_r23_detcpu" if model == "r23" else f"e1m6_e1_{FOLD}_detcpu"
    check_new_name(name)
    return name


def check_new_name(name: str) -> str:
    """Only the two M6 folder names; never r23 or a registered e1_<fold>_seed<S> folder."""
    if name not in ("e1m6_r23_detcpu", f"e1m6_e1_{FOLD}_detcpu"):
        raise SystemExit(f"{WHO}: {name} is not an M6 build folder")
    return name


def score_file_name(model: str) -> str:
    if model not in MODELS:
        raise SystemExit(f"{WHO}: unknown model {model}")
    return f"m6__{model}__{GAME}.json"


def weights_for(model: str) -> Path:
    if model == "r23":
        return main_checkout() / "models" / "player_detector.pt"
    from tools.p8_e1_m4 import weights_of
    return weights_of(FOLD)


def check_guard() -> None:
    refuse_cv_test([GAME], WHO)
    guard([GAME], WHO)


# ── the build proxy (driver side) and the child ────────────────────────────────────────────────────────────────────
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
        return [cmd[0], "-m", "tools.p8_e1_m6", "child", *cmd[1:]]
    return cmd


def child(argv: list) -> int:
    from tools import p8_device as D
    from tools.p8_e1_score import _sha
    if not argv or argv[0] != BUILD_SCRIPT:
        raise SystemExit(f"{WHO}: usage: python -m tools.p8_e1_m6 child {BUILD_SCRIPT} <its arguments>")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    guard([src], WHO)
    req = D.requested()
    if req["detector"] != "cpu":
        raise SystemExit(f"{WHO}: the child needs P8_DEVICE_DETECTOR=cpu")
    resolved = D.apply(req)
    weights = Path(os.environ[ENV])
    import config
    config.PLAYER_DETECTOR_WEIGHTS = weights
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    for suffix, rec in (("device", {"requested": req, "resolved": resolved}),
                        ("detector", {"weights": weights.name, "sha256": _sha(weights),
                                      "model": os.environ.get("P8_E1_M6_MODEL"), "seed": 0})):
        part = config.TRACKING_DIR / f"{stem}_{suffix}.json.part"
        part.write_text(json.dumps(rec))
        os.replace(part, config.TRACKING_DIR / f"{stem}_{suffix}.json")
    script = Path(config.PROJECT_ROOT) / BUILD_SCRIPT
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_rebuild(args) -> dict:
    import time
    from sportvu import rebuild
    from sportvu.autolabel import gate_args
    from tools import p8_frozen as F
    from tools.p8_e1_score import _sha, check_provenance
    check_guard()
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    for k, v in DEVICE_ENV.items():
        os.environ[k] = v
    for k in ("P8_DEVICE", "P8_DEVICE_COURT", "P8_DEVICE_GATE"):
        os.environ.pop(k, None)
    F.game_of(GAME)
    weights = weights_for(args.model).resolve()
    if args.model == "e1":
        sha = check_provenance(FOLD, GAME, 0, weights)["sha256"]
    else:
        sha = _sha(weights)
        meta = json.loads((Path(os.environ.get("V3_P8_E1") or p8_root().parent / "e1_corpus") / "meta.json").read_text())
        if sha != meta["detector"]["weights_sha256"]:
            raise SystemExit(f"{WHO}: the production detector is not the one the corpus used")
    P = F.install_paths()                            # installs tools.p8_device's proxy; ours wraps it
    rebuild.subprocess = _Proxy(rebuild.subprocess)
    os.environ[ENV], os.environ["P8_E1_M6_MODEL"] = str(weights), args.model
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    out = P["builds"] / build_dir_name(args.model)
    out.mkdir(parents=True, exist_ok=True)
    wins = [w for w in F.label_windows(P, GAME) if (P["builds"] / "r23" / f"{w['window']}_frames.json").exists()]
    if args.windows:
        wins = wins[:args.windows]
    res = {"model": args.model, "game": GAME, "weights_sha256": sha, "windows": 0, "built": 0, "cached": 0,
           "seconds_built": 0.0}
    for w in wins:
        rec_p = out / f"{w['window']}_detector.json"
        if (out / f"{w['window']}_frames.json").exists():
            if not rec_p.exists() or json.loads(rec_p.read_text())["sha256"] != sha:
                raise SystemExit(f"{WHO}: {w['window']} in {out} was built with other detector weights")
        t0 = time.time()
        r = F.safe_build(P, w, P["label_build"] / f"{w['window']}.mp4", out, extra_args=gate_args())
        if not r["ok"]:
            raise SystemExit(f"{WHO}: build of {w['window']} failed: {r.get('tail')}")
        res["windows"] += 1
        res["cached" if r.get("cached") else "built"] += 1
        if not r.get("cached"):
            res["seconds_built"] += time.time() - t0
        if not r.get("cached") and not r.get("empty"):
            got = json.loads(rec_p.read_text())["sha256"] if rec_p.exists() else None
            dev = json.loads((out / f"{w['window']}_device.json").read_text())
            if got != sha or dev["resolved"]["detector"] != "cpu":
                raise SystemExit(f"{WHO}: {w['window']} did not record the weights or the CPU detector")
    res["seconds_built"] = round(res["seconds_built"], 1)
    print(json.dumps(res, indent=1))
    return res


def cmd_score(args) -> dict:
    """The pinned score_window on the CPU build of one model and on the other's, on shared frames, as tools.p8_e1_score
    scores (both CPU builds must exist)."""
    import time
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    from tools.p8_e1_score import _filtered, frame_sets, window_stats
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
    dirs = {m: P["builds"] / build_dir_name(m) for m in MODELS}
    t0 = time.time()
    all_w = F.label_windows(P, GAME)
    expected = [w["window"] for w in all_w if (P["builds"] / "r23" / f"{w['window']}_frames.json").exists()]
    wins, frame_diff, skipped = {}, {}, []
    tmp = p8_root() / "e1" / "m6_shared_frames"
    for w in all_w:
        s = w["window"]
        if s not in expected or not all((d / f"{s}_frames.json").exists() for d in dirs.values()):
            continue
        if not (P["cache"] / "scorecard" / f"{s}.json").exists():
            raise SystemExit(f"{WHO}: {s} has no cached scorecard references; refusing to compute them")
        fs = frame_sets(json.loads((dirs["r23"] / f"{s}_frames.json").read_text()),
                        json.loads((dirs["e1"] / f"{s}_frames.json").read_text()))
        use = dict(dirs)
        if not fs["equal"]:
            frame_diff[s] = {k: (len(v) if k == "shared" else v) for k, v in fs.items() if k != "equal"}
            use = {m: _filtered(dirs[m], s, fs["shared"], tmp / m) for m in MODELS}
        pair = {}
        for m, bd in use.items():
            rec = S.score_window(P, GAME, w, bd, "camera", mom, mirror, refs, events)[0]
            if "skipped" in rec:
                break
            pair["r23" if m == "r23" else "new"] = window_stats(rec)
        if len(pair) == 2:
            wins[s] = pair
        else:
            skipped.append(s)
    out = {"game": GAME, "fold": FOLD, "seed": 0, "device": {"detector": "cpu", "court": "auto", "gate": "auto"},
           "expected_windows": expected, "scored_windows": sorted(wins), "skipped_windows": skipped,
           "frame_sets_differ": frame_diff, "windows": wins, "seconds": round(time.time() - t0, 1)}
    (p8_root() / "e1").mkdir(parents=True, exist_ok=True)
    atomic_write(p8_root() / "e1" / score_file_name("e1"), json.dumps(out))
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}, indent=1))
    return out


def cmd_report(args) -> dict:
    from tools.p8_e1_score import boot, diffs, rows
    check_guard()
    d = p8_root() / "e1"
    f = json.loads((d / score_file_name("e1")).read_text())
    wins = list(f["windows"].values())
    r = rows(wins)
    reg = json.loads((d / f"{FOLD}__seed0__{GAME}.json").read_text())
    reg_wins = [reg["windows"][s] for s in f["scored_windows"] if s in reg["windows"]]
    out = {"what": "E1 post-mortem M6 (docs/research/P8_E1.md): gsw_bkn_2015 rebuilt with the detector on CPU, "
                   "r23 and E1 fold gsw_bkn_2015 seed 0; report only",
           "complete": set(f["scored_windows"]) == set(f["expected_windows"]),
           "windows": len(wins), "frame_sets_differ": len(f["frame_sets_differ"]),
           "rows": {k: {"r23": a, "e1": b} for k, (a, b) in r.items()}, "diff": diffs(wins),
           "window_interval": boot({i: [w] for i, w in enumerate(wins)}),
           "registered_mps_same_windows": {"rows": {k: {"r23": a, "e1": b} for k, (a, b) in rows(reg_wins).items()},
                                           "diff": diffs(reg_wins)},
           "registered_per_game": REGISTERED,
           "run": {**RUN_LOG, "records": {m: tally_records(p8_root() / "builds" / build_dir_name(m)) for m in MODELS}}}
    m5 = d / f"m6_raw_vs_sidecar.json"
    if m5.exists():
        out["m4_raw_vs_cpu_sidecar"] = json.loads(m5.read_text())
    atomic_write(d / "p8_e1_m6.json", json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("complete", "windows", "diff", "window_interval")} |
                     {"rows": {k: out["rows"][k] for k in ("missed_rate", "ghost_rate")},
                      "registered": out["registered_mps_same_windows"]["diff"]}, indent=1, default=float))
    return out


def cmd_drop(args) -> dict:
    """M4's raw boxes at 0.40 against the CPU sidecars on corpus frames, both sides on CPU (tools.p8_e1_m5.match_boxes)."""
    from tools.p8_e1_m5 import OPERATING, add, match_boxes
    from tools.p8_e1data import corpus_root
    check_guard()
    d = p8_root() / "e1"
    m4 = json.loads((d / f"m4_preds__{GAME}.json").read_text())["preds"]
    plan = json.loads((corpus_root() / "plan.json").read_text())
    corpus = {(fr["window"], int(fr["frame"])) for ch in plan["chunks"] for w in ch for fr in w["frames"]
              if fr["game"] == GAME}
    builds = p8_root() / "builds"
    out = {}
    for m in MODELS:
        acc = {}
        for p in sorted((builds / build_dir_name(m)).glob(f"{GAME}_*_frames.json")):
            s = p.name[:-len("_frames.json")]
            for r in json.loads(p.read_text())["frames"]:
                f = int(r["frame"])
                if (s, f) in corpus:
                    raw = [b for b in m4[f"{s}_f{f:06d}"][m] if b[4] >= OPERATING - 1e-9]
                    add(acc, match_boxes(raw, [b["bbox"] for b in r["boxes"]]))
        out[m] = acc
    atomic_write(d / "m6_raw_vs_sidecar.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "child":
        return child(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m tools.p8_e1_m6")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--model", required=True, choices=MODELS)
    r.add_argument("--windows", type=int, default=None)
    s = sub.add_parser("score")
    sub.add_parser("drop")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"rebuild": cmd_rebuild, "score": cmd_score, "drop": cmd_drop, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
