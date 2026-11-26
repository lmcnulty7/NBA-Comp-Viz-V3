"""tools/p8_truth_moved_dh.py: report only, post hoc, not ruled: |d_H| = |B(p) - T(p)| at the scored r23 build's foot pixels, for the old (MPS)
and the new (CPU-detector) truth of gsw_cha_2016. Stored files only, no model, no video, no lock.

B: the per-frame H of the reference-platform r23 sidecar (builds/m9_r23_detcpu__gsw_cha_2016 in the MPS root, the
sidecar step 1b scored). T: H_truth of each truth's accepted rows. p: foot_stab of every r23 box on the frame.
Windows: the scored windows of e1/m9__r23_detcpu__gsw_cha_2016.json. Frames: truth-accepted frames whose sidecar row
is accepted with an H (sportvu.bench.accepted).
"""
import json
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import guard
from tools.research_testset import refuse_cv_test

TAG = "gsw_cha_2016"
OLD = Path.home() / "Developer/nba-v3-worktrees/_p8-data/work"
NEW = Path.home() / "Developer/nba-v3-worktrees/_p8-data/work_ref"


def to_court(H, px):
    import cv2
    p = np.asarray(px, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)


def truth(root, stem):
    p = root / "labels" / TAG / f"{stem}.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return {int(r["frame"]): r["H_truth"] for r in rows}


def med(a):
    return round(float(np.median(a)), 3) if len(a) else None


def main():
    from sportvu.bench import accepted
    refuse_cv_test([TAG], "tools.p8_truth_moved_dh")
    guard([TAG], "tools.p8_truth_moved_dh")
    sf = json.loads((OLD / "e1" / "m9__r23_detcpu__gsw_cha_2016.json").read_text())
    wins = list(sf["scored_windows"])
    guard(wins, "tools.p8_truth_moved_dh")
    bd = OLD / "builds" / "m9_r23_detcpu__gsw_cha_2016"
    out = {"old_all": [], "new_all": [], "old_common": [], "new_common": [], "truth_shift_common": []}
    frames = {"old": 0, "new": 0, "common": 0}
    tm_same = {"windows": 0, "same_frames": 0}
    for stem in wins:
        a = json.loads((OLD / "label_build" / f"{stem}_timemap.json").read_text())["frames"]
        bp = NEW / "label_build" / f"{stem}_timemap.json"
        if bp.exists():
            tm_same["windows"] += 1
            tm_same["same_frames"] += json.loads(bp.read_text())["frames"] == a
        side = {int(r["frame"]): r for r in json.loads((bd / f"{stem}_frames.json").read_text())["frames"]}
        to, tn = truth(OLD, stem), truth(NEW, stem)
        for f in sorted(set(to) | set(tn)):
            sr = side.get(f)
            if sr is None or not accepted(sr) or sr.get("H") is None or not sr.get("boxes"):
                continue
            px = np.array([b["foot_stab"] for b in sr["boxes"]], np.float64).reshape(-1, 2)
            B = to_court(sr["H"], px)
            dh = {}
            for k, t in (("old", to), ("new", tn)):
                if f in t:
                    T = to_court(t[f], px)
                    dh[k] = (T, np.hypot(*(B - T).T))
                    out[f"{k}_all"] += list(dh[k][1])
                    frames[k] += 1
            if len(dh) == 2:
                frames["common"] += 1
                out["old_common"] += list(dh["old"][1])
                out["new_common"] += list(dh["new"][1])
                out["truth_shift_common"] += list(np.hypot(*(dh["new"][0] - dh["old"][0]).T))
    res = {"what": "report only, post hoc, not ruled: median |B(p) - T(p)| in ft at every r23 box foot_stab pixel",
           "frames": frames, "label_window_timemaps_identical": tm_same,
           **{f"median_abs_d_H_ft_{k}": med(v) for k, v in out.items() if k != "truth_shift_common"},
           "median_abs_truth_shift_ft_common": med(out["truth_shift_common"]),
           "boxes": {k: len(v) for k, v in out.items()}}
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
