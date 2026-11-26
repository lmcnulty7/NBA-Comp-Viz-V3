"""tools/p8_truth_moved_dh.py: report only, post hoc, not ruled. How far each camera truth of a game (the MPS truth and
its CPU-detector rebuild) sits from the scored r23 build's court H, at the build's own foot pixels.

    python3 -m tools.p8_truth_moved_dh GAME --mps-root <P8 root of the existing truth> --cpu-root <reference root>

|d_H| = |B(p) - T(p)| in ft, for every box of the reference-platform r23 sidecar (tools.p8_placement.BUILDS, in the MPS
root) at its foot_stab pixel p: B is the sidecar's per-frame H, T the truth's H_truth on the same frame. Windows: the
scored windows of the game's registered score file (tools.p8_placement.SCORES). Frames: truth-accepted frames whose
sidecar row is accepted with an H (sportvu.bench.accepted). Reported for each truth on all its frames and on the
frames both truths accept, with the median |T_cpu(p) - T_mps(p)| on those common frames. All boxes, not step 1b's
nearest-box player population (docs/research/P8_placement.md), so the values are not step 1b's.
Confound: B and the new truth share the CPU detector (the truth is fitted from CPU-detector label builds), so a
smaller |d_H| under the new truth is partly expected and is not evidence that it is closer to SportVU.

It also checks that the label windows' time maps are the same in both roots (the windows cover the same frames).
Stored files only: no model, no video, no lock. Writes <cpu root>/truth_moved/<game>_dh_posthoc.json (and --out).
First used on gsw_cha_2016 (2026-11-26, P8_truth_arenas.md section 15), where it ran as a one-game script; this
version takes the game from tools.p8_placement's tables and gives the same numbers on that game.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_truth_moved_dh"
APART = ("gsw_ind_2016",)


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def to_court(H, px) -> np.ndarray:
    import cv2
    p = np.asarray(px, np.float64).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)


def med(a):
    return round(float(np.median(a)), 3) if len(a) else None


def measure(windows: dict, accepted=lambda row: True) -> dict:
    """windows: {window: (sidecar {frame: row}, mps truth {frame: H}, cpu truth {frame: H})} -> the report numbers.
    A frame counts for a truth when the truth has it and its sidecar row is accepted with an H and has boxes."""
    out = {k: [] for k in ("mps_all", "cpu_all", "mps_common", "cpu_common", "truth_shift_common")}
    frames = {"mps": 0, "cpu": 0, "common": 0}
    for _, (side, t_m, t_c) in sorted(windows.items()):
        for f in sorted(set(t_m) | set(t_c)):
            sr = side.get(f)
            if sr is None or not accepted(sr) or sr.get("H") is None or not sr.get("boxes"):
                continue
            px = np.array([b["foot_stab"] for b in sr["boxes"]], np.float64).reshape(-1, 2)
            B = to_court(sr["H"], px)
            got = {}
            for k, t in (("mps", t_m), ("cpu", t_c)):
                if f in t:
                    T = to_court(t[f], px)
                    got[k] = (T, np.hypot(*(B - T).T))
                    out[f"{k}_all"] += list(got[k][1])
                    frames[k] += 1
            if len(got) == 2:
                frames["common"] += 1
                out["mps_common"] += list(got["mps"][1])
                out["cpu_common"] += list(got["cpu"][1])
                out["truth_shift_common"] += list(np.hypot(*(got["cpu"][0] - got["mps"][0]).T))
    return {"frames": frames,
            **{f"median_abs_d_H_ft_{k}": med(v) for k, v in out.items() if k != "truth_shift_common"},
            "median_abs_truth_shift_ft_common": med(out["truth_shift_common"]),
            "boxes": {k: len(v) for k, v in out.items()}}


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def truth_h(root: Path, game: str, stem: str) -> dict:
    p = root / "labels" / game / f"{stem}.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return {int(r["frame"]): r["H_truth"] for r in rows}


def run(game: str, mps_root: Path, cpu_root: Path, out: Path | None = None) -> dict:
    from sportvu.bench import accepted
    from tools.p8_placement import BUILDS, SCORES
    if game in APART:
        raise SystemExit(f"{WHO}: {game} stays apart (D98)")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    if game not in BUILDS:
        raise SystemExit(f"{WHO}: {game} has no reference-platform r23 sidecar in tools.p8_placement.BUILDS")
    mps_root, cpu_root = check_outside(Path(mps_root).expanduser()), check_outside(Path(cpu_root).expanduser())
    if mps_root == cpu_root:
        raise SystemExit(f"{WHO}: the two roots are the same folder")
    sf = json.loads((mps_root / "e1" / SCORES[game]).read_text())
    wins = list(sf["scored_windows"])
    refuse_cv_test(wins, WHO)
    guard(wins, WHO)
    bd = mps_root / "builds" / BUILDS[game]
    windows, tm = {}, {"windows": 0, "same_frames": 0}
    for stem in wins:
        a = json.loads((mps_root / "label_build" / f"{stem}_timemap.json").read_text())["frames"]
        bp = cpu_root / "label_build" / f"{stem}_timemap.json"
        if bp.exists():
            tm["windows"] += 1
            tm["same_frames"] += json.loads(bp.read_text())["frames"] == a
        side = {int(r["frame"]): r for r in json.loads((bd / f"{stem}_frames.json").read_text())["frames"]}
        windows[stem] = (side, truth_h(mps_root, game, stem), truth_h(cpu_root, game, stem))
    res = {"game": game, "what": "report only, post hoc, not ruled: median |B(p) - T(p)| in ft at every r23 box "
                                 "foot_stab pixel; B the r23 sidecar H, T each truth's H",
           "sidecar": BUILDS[game], "score_file": SCORES[game], "scored_windows": len(wins),
           "label_window_timemaps_identical": tm, **measure(windows, accepted)}
    text = json.dumps(res, indent=1) + "\n"
    d = cpu_root / "truth_moved"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"{game}_dh_posthoc.json", text)
    if out:
        atomic_write(Path(out), text)
    print(text)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    ap.add_argument("game")
    ap.add_argument("--mps-root", required=True)
    ap.add_argument("--cpu-root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.game, Path(a.mps_root), Path(a.cpu_root), a.out)


if __name__ == "__main__":
    sys.exit(main())
