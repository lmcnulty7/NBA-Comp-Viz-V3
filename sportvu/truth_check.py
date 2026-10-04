"""
sportvu/truth_check.py: does the per-frame truth H (sportvu.truth, B4) fit the PAINTED court, or only the
players' feet it was fitted on? (found at ROADMAP R1.3, 2026-10-04)

The truth H is an 8-parameter homography fitted to 6..10 feet <-> SportVU pairs, usually bunched in one
part of the court. It fits those feet (0.33 ft median residual) but nothing constrains it away from
them. This check draws the template through H_truth on sampled phx testable frames (contact sheet,
data/sportvu/labels/PHX_B4_truth_check_sheet.jpg, gitignored) and compares the A1 line indicator
(qc.track_qc.line_overlay: median px from painted-line ridges to the projected template; a right H
reads a few px, about 27 px on good clips because of non-line ridges, a wrong one 60+) for H_truth and
for the V3 pipeline H on the same frames.

  python -m sportvu.truth_check -> reports/sportvu_truth_line_check.{json,txt}
"""
from __future__ import annotations
import glob, json
import numpy as np
import config

EVERY = 6        # every 6th testable frame per window


def main():
    import cv2
    from qc.track_qc import line_overlay
    from sportvu.rebuild import BUILD_DIR
    from sportvu.truth import TRUTH_DIR
    from videoseq import SeqReader
    res = []
    for p in sorted(glob.glob(str(TRUTH_DIR / "gsw_phx_2016_*_truth.json"))):
        d = json.load(open(p)); w = d["window"]["window"]
        side = {r["frame"]: r for r in json.load(open(BUILD_DIR / (w + "_frames.json")))["frames"]}
        ok = [r for r in d["frames"] if r["status"] == "ok"][::EVERY]
        if not ok:
            continue
        sr = SeqReader(cv2.VideoCapture(str(BUILD_DIR / (w + ".mp4"))))
        for r in ok:
            good, fr = sr.read(r["frame"])
            if not good:
                continue
            boxes = [b["bbox"] for b in side[r["frame"]]["boxes"]]
            lt, lp = line_overlay(fr, r["H_truth"], boxes), line_overlay(fr, side[r["frame"]]["H"], boxes)
            if lt["dist_px"] is not None and lp["dist_px"] is not None:
                res.append((lt["dist_px"], lp["dist_px"]))
    a = np.array(res)
    rep = {"frames": int(len(a)), "game": "gsw_phx_2016 (B4 testable frames, every %dth)" % EVERY,
           "a1_dist_px_median": {"truth_H": round(float(np.median(a[:, 0])), 1), "v3_H": round(float(np.median(a[:, 1])), 1)},
           "share_truth_farther_than_v3": round(float((a[:, 0] > a[:, 1]).mean()), 3),
           "share_over_60px": {"truth_H": round(float((a[:, 0] > 60).mean()), 3), "v3_H": round(float((a[:, 1] > 60).mean()), 3)},
           "conclusion": "the truth H is about as far from the paint as V3's H: it is valid near the fitted feet, not as a court-level homography",
           "consequences": ["court line error (pipeline template vs truth template) is not measurable with this truth",
                            "missed-player and ghost-box counts project through H_truth away from the fitted feet and carry that error",
                            "position error and near-field bias compare positions only for players the fit matched, near the feet, and stand",
                            "ROADMAP R1.3 court keypoint labels projected through this H_truth would be wrong"]}
    (config.REPORTS_DIR / "sportvu_truth_line_check.json").write_text(json.dumps(rep, indent=1))
    L = ["TRUTH H vs PAINTED LINES (gsw_phx_2016, %d testable frames)" % rep["frames"],
         "  A1 ridge-to-template distance, median px: truth H %.1f | V3 pipeline H %.1f" % (rep["a1_dist_px_median"]["truth_H"], rep["a1_dist_px_median"]["v3_H"]),
         "  truth H farther from the paint than V3's H on %.0f%% of frames; over 60 px: truth %.0f%%, V3 %.0f%%" % (
             100 * rep["share_truth_farther_than_v3"], 100 * rep["share_over_60px"]["truth_H"], 100 * rep["share_over_60px"]["v3_H"]),
         "  conclusion: " + rep["conclusion"]] + ["  - " + c for c in rep["consequences"]]
    txt = "\n".join(L)
    (config.REPORTS_DIR / "sportvu_truth_line_check.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
