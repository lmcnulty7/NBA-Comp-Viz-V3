"""tools/p8_e1_m5.py: E1 post-mortem M5, a read-only rescore of the registered tracked builds with bottom-centre feet
(docs/research/P8_E1.md, "M5 and M6", definitions committed before any number). Report only; nothing is rebuilt.

    python3 -m export.heavy --name ... -- <frozen CV python> -m tools.p8_e1_m5 rescore --game G     one pool game
    python3 -m tools.p8_e1_m5 report                                                               -> e1/p8_e1_m5.json

rescore: for r23 and the fold's E1 build, the pinned score_window on the sidecars as built (foot_stab; must reproduce
the registered score file) and on scratch copies whose foot_stab is the box's foot_raw (bottom centre), plus sidecar
box counts and the M4 stand-in for boxes dropped between raw detection and sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_m5"
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
SIDES = ("r23", "new")                  # tools.p8_e1_score's names for r23 and the model
FEET = ("stab", "bottom")
OPERATING = 0.40
IOU = 0.5


def check_games(games) -> list:
    games = list(games)
    bad = [g for g in games if g not in GAMES]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not a pool game")
    refuse_cv_test(games, WHO)
    guard(games, WHO)
    return games


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def bottom_sidecar(side: dict) -> dict:
    """A sidecar whose every box has foot_stab replaced by its foot_raw (the tracked box's bottom centre)."""
    out = dict(side)
    out["frames"] = [{**r, "boxes": [{**b, "foot_stab": list(b["foot_raw"])} for b in r["boxes"]]}
                     for r in side["frames"]]
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if inter > 0 and ua > 0 else 0.0


def match_boxes(raw: list, side: list, thr: float = IOU) -> dict:
    """One-to-one Hungarian match on IoU of raw boxes to sidecar boxes, pairs at IoU >= thr -> counts."""
    from scipy.optimize import linear_sum_assignment
    n = 0
    if raw and side:
        M = np.array([[iou(a, b) for b in side] for a in raw])
        ri, ci = linear_sum_assignment(-M)
        n = int(sum(1 for i, j in zip(ri, ci) if M[i, j] >= thr))
    return {"matched": n, "raw_only": len(raw) - n, "sidecar_only": len(side) - n, "raw": len(raw), "sidecar": len(side)}


def add(acc: dict, c: dict) -> None:
    for k, v in c.items():
        acc[k] = acc.get(k, 0) + v


def same_stats(a: dict, b: dict) -> bool:
    """Two window_stats records agree on every count and on the matched players (errors to 1e-9)."""
    keys = ("missed", "in_frame", "nr_ghosts", "nr_boxes", "cov_acc", "cov_live")
    if any(a[k] != b[k] for k in keys) or a["team"] != b["team"] or a["matched"].keys() != b["matched"].keys():
        return False
    return all(abs(a["matched"][k][0] - b["matched"][k][0]) < 1e-9 for k in a["matched"])


# ── rescore (frozen CV python, inside the lock) ─────────────────────────────────────────────────────────────────────
def _bottom_dir(src_dir: Path, stem: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    side = json.loads((src_dir / f"{stem}_frames.json").read_text())
    atomic_write(dest / f"{stem}_frames.json", json.dumps(bottom_sidecar(side)))
    for p in src_dir.glob(f"{stem}_*"):
        if not p.name.endswith("_frames.json") and not (dest / p.name).exists():
            os.symlink(p, dest / p.name)
    return dest


def cmd_rescore(args) -> dict:
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    from tools.p8_e1_score import build_name, score_name, window_stats
    from tools.p8_e1data import corpus_root
    game = check_games([args.game])[0]
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    g = F.game_of(game)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, game)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    sf = json.loads((p8_root() / "e1" / score_name(game, 0, game)).read_text())
    if sf.get("frame_sets_differ"):
        raise SystemExit(f"{WHO}: frame sets differ in the score file; this tool scores whole windows")
    dirs = {"r23": P["builds"] / "r23", "new": P["builds"] / build_name(game, 0)}
    scratch = p8_root() / "e1" / "m5_bottom"
    wins = {w["window"]: w for w in F.label_windows(P, game)}
    m4 = json.loads((p8_root() / "e1" / f"m4_preds__{game}.json").read_text())["preds"]
    plan = json.loads((corpus_root() / "plan.json").read_text())
    corpus_frames = {(fr["window"], int(fr["frame"])) for ch in plan["chunks"] for w in ch for fr in w["frames"]
                     if fr["game"] == game}
    out_w = {}
    boxes = {s: {"frames": 0, "boxes": 0, "scored_frames": 0, "scored_boxes": 0} for s in SIDES}
    drop = {s: {} for s in SIDES}
    for s in sf["scored_windows"]:
        if not (P["cache"] / "scorecard" / f"{s}.json").exists():
            raise SystemExit(f"{WHO}: {s} has no cached scorecard references; refusing to compute them")
        w = wins[s]
        rec_w = {}
        for side, bd in dirs.items():
            full = S.score_window(P, game, w, bd, "camera", mom, mirror, refs, events)
            st = window_stats(full[0])
            if not same_stats(st, sf["windows"][s][side]):
                raise SystemExit(f"{WHO}: {s} {side} with foot_stab does not reproduce the registered score file")
            bt_dir = _bottom_dir(bd, s, scratch / (bd.name + "__bottom"))
            rec = S.score_window(P, game, w, bt_dir, "camera", mom, mirror, refs, events)[0]
            if "skipped" in rec:
                raise SystemExit(f"{WHO}: {s} {side} bottom skipped: {rec['skipped']}")
            rec_w[side] = {"stab": st, "bottom": window_stats(rec)}
            side_j = json.loads((bd / f"{s}_frames.json").read_text())
            scored_f = {int(o["f"]) for o in full[1]}          # scored frames with an in-frame player
            for r in side_j["frames"]:
                f = int(r["frame"])
                boxes[side]["frames"] += 1
                boxes[side]["boxes"] += len(r["boxes"])
                if f in scored_f:
                    boxes[side]["scored_frames"] += 1
                    boxes[side]["scored_boxes"] += len(r["boxes"])
                if (s, f) in corpus_frames:
                    raw = [b for b in m4[f"{s}_f{f:06d}"]["e1" if side == "new" else "r23"] if b[4] >= OPERATING - 1e-9]
                    add(drop[side], match_boxes(raw, [b["bbox"] for b in r["boxes"]]))
        out_w[s] = rec_w
    out = {"game": game, "fold": game, "seed": 0, "windows": out_w, "sidecar_boxes": boxes, "m4_raw_vs_sidecar": drop,
           "note": "sidecars keep tracked boxes only; raw detections and confidences are not stored"}
    atomic_write(p8_root() / "e1" / f"m5__{game}.json", json.dumps(out))
    print(json.dumps({"game": game, "sidecar_boxes": boxes, "m4_raw_vs_sidecar": drop}, indent=1))
    return out


# ── report (system python) ──────────────────────────────────────────────────────────────────────────────────────────
def summarise(pool: dict) -> dict:
    """pool: {game: [window {r23: {stab, bottom}, new: {stab, bottom}}]} -> rows, differences and intervals per foot."""
    from tools.p8_e1_score import boot, diffs, rows
    out = {}
    for foot in FEET:
        per = {g: [{side: w[side][foot] for side in SIDES} for w in ws] for g, ws in pool.items()}
        allw = [w for v in per.values() for w in v]
        out[foot] = {"rows": {k: {"r23": a, "e1": b} for k, (a, b) in rows(allw).items()}, "diff": diffs(allw),
                     "window_interval": boot({f"{g}/{i}": [w] for g, v in per.items() for i, w in enumerate(v)}),
                     "game_interval": boot(per) if len(per) > 1 else None}
    return out


def cmd_report(args) -> dict:
    check_games(GAMES)
    d = p8_root() / "e1"
    files = {}
    for g in GAMES:
        p = d / f"m5__{g}.json"
        if not p.exists():
            raise SystemExit(f"{WHO}: M5 has not run on {g}")
        files[g] = json.loads(p.read_text())
    games = {}
    for g, f in files.items():
        games[g] = {**summarise({g: list(f["windows"].values())}), "sidecar_boxes": f["sidecar_boxes"],
                    "m4_raw_vs_sidecar": f["m4_raw_vs_sidecar"]}
    pooled = summarise({g: list(f["windows"].values()) for g, f in files.items()})
    tot = lambda key: {s: {k: sum(files[g][key][s][k] for g in GAMES) for k in files[GAMES[0]][key][s]} for s in SIDES}
    pooled["sidecar_boxes"], pooled["m4_raw_vs_sidecar"] = tot("sidecar_boxes"), tot("m4_raw_vs_sidecar")
    out = {"what": "E1 post-mortem M5 (docs/research/P8_E1.md): registered tracked builds rescored with foot_stab "
                   "(reproduces the registered rows) and with bottom-centre feet; report only",
           "sides": {"r23": "builds/r23", "e1": "builds/e1_<fold>_seed0, each pool game by the fold that held it out"},
           "games": games, "pooled": pooled}
    atomic_write(d / "p8_e1_m5.json", json.dumps(out, indent=1, default=float))
    show = {foot: {k: {"r23": pooled[foot]["rows"][k]["r23"], "e1": pooled[foot]["rows"][k]["e1"],
                       "diff": pooled[foot]["diff"][k], "w": pooled[foot]["window_interval"][k],
                       "g": pooled[foot]["game_interval"][k]} for k in ("missed_rate", "ghost_rate")} for foot in FEET}
    print(json.dumps({"pooled": show, "boxes": pooled["sidecar_boxes"], "drop": pooled["m4_raw_vs_sidecar"]}, indent=1,
                     default=float))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_e1_m5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rescore")
    r.add_argument("--game", required=True, choices=GAMES)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"rescore": cmd_rescore, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
