"""
sportvu/validate_checks.py: do the Phase A label-free checks track SportVU truth? (FIX_PLAN B6)

On the testable phx window frames (B4/B5), a frame is WRONG when its pipeline-vs-SportVU frame H
error exceeds H_WRONG_FT, or it has a missed player, or a ghost box (the plan's definition; the
three criteria are also reported separately because missed/ghost base rates are high). For each
check (A1 line distance at several cutoffs, A2 scorebug / off-court feet, A3 speed / team count /
cut jump, A4 second-tracker disagreement) precision and recall of "flags the frame" against
"frame is wrong". Pre-set drop rule: recall < MIN_RECALL or precision < MIN_PRECISION.

  python -m sportvu.validate_checks gsw_phx_2016 -> reports/sportvu_check_validation.{json,txt}
"""
from __future__ import annotations
import json, sys
import numpy as np
import config
from sportvu.rebuild import BUILD_DIR
from sportvu.truth import TRUTH_DIR
from sportvu.eval import eval_frame, near_side
from sportvu.sync import SportVUIndex

H_WRONG_FT = 3.0
SEVERITY_FT = (6.0, 10.0, 15.0)   # graded H-error levels reported beside the plan's 3 ft definition
MIN_RECALL, MIN_PRECISION = 0.50, 0.30
A1_CUTOFFS = (30, 40, 50, 60)


def pr(flag: np.ndarray, wrong: np.ndarray) -> dict:
    tp = int((flag & wrong).sum()); fp = int((flag & ~wrong).sum()); fn = int((~flag & wrong).sum())
    return {"flagged": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}


def main():
    import cv2
    game = sys.argv[1] if len(sys.argv) > 1 else "gsw_phx_2016"
    sv_game = sys.argv[2] if len(sys.argv) > 2 else "12.16.2015.PHX.at.GSW"
    index = SportVUIndex(json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (sv_game + "_moments.json")).read_text())["moments"])
    mirror = json.loads((config.REPORTS_DIR / ("sportvu_sync_%s.json" % game)).read_text())["direction_resolution"]["mirror"]
    rows = []   # one per testable frame: truth metrics + check outputs
    for wp in sorted(BUILD_DIR.glob(game + "_s*_w*_windows.json")) or [BUILD_DIR / (game + "_windows.json")]:
        pass
    windows = json.loads((BUILD_DIR / (game + "_windows.json")).read_text())
    for wd in windows:
        stem = wd["window"]
        tp, qp = TRUTH_DIR / (stem + "_truth.json"), config.REPORTS_DIR / "qc" / (stem + ".json")
        if not tp.exists() or not qp.exists():
            continue
        truth = json.loads(tp.read_text()); side = json.loads((BUILD_DIR / (stem + "_frames.json")).read_text())
        srow = {r["frame"]: r for r in side["frames"]}
        qc = {r["frame"]: r for r in json.loads(qp.read_text())["frames"]}
        cap = cv2.VideoCapture(str(BUILD_DIR / (stem + ".mp4"))); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        for r in truth["frames"]:
            if r["status"] != "ok" or r["frame"] not in qc:
                continue
            q, i = r["q"], r["moment"]
            ev = eval_frame(r, srow[r["frame"]], index.q[q]["xy"][i], index.q[q]["pid"][i], index.q[q]["team"][i], mirror, W, H)
            c = qc[r["frame"]]
            rows.append({"window": stem, "frame": r["frame"], "H_err": ev["frame_err_ft"], "missed": ev["missed"], "ghosts": ev["ghosts"],
                         "a1_dist": c.get("overlay", {}).get("dist_px"), "geo_fails": c.get("geometry", {}).get("fails", []),
                         "phys_fails": c.get("physics", {}).get("fails", []), "second": c.get("second", {}).get("disagreement"),
                         "second_fail": bool(c.get("second", {}).get("fails"))})
    n = len(rows)
    H_wrong = np.array([r["H_err"] is not None and r["H_err"] > H_WRONG_FT for r in rows])
    missed = np.array([r["missed"] > 0 for r in rows]); ghost = np.array([r["ghosts"] > 0 for r in rows])
    wrong = H_wrong | missed | ghost
    checks = {}
    a1 = np.array([r["a1_dist"] if r["a1_dist"] is not None else np.nan for r in rows])
    for cut in A1_CUTOFFS:
        checks["A1_lines_over_%dpx" % cut] = np.nan_to_num(a1, nan=0) > cut
    for rule in ("scorebug", "offcourt_feet"):
        checks["A2_" + rule] = np.array([rule in r["geo_fails"] for r in rows])
    for rule in ("speed", "team_count", "cut_jump"):
        checks["A3_" + rule] = np.array([rule in r["phys_fails"] for r in rows])
    checks["A4_second_over_0.30"] = np.array([r["second_fail"] for r in rows])
    sec = np.array([r["second"] if r["second"] is not None else np.nan for r in rows])
    checks["A4_second_over_0.50"] = np.nan_to_num(sec, nan=0) > 0.50
    out = {"frames": n, "wrong_definition": "H_err > %.1f ft OR missed > 0 OR ghosts > 0" % H_WRONG_FT,
           "base_rates": {"wrong_any": round(float(wrong.mean()), 3), "H_wrong": round(float(H_wrong.mean()), 3),
                          "missed": round(float(missed.mean()), 3), "ghost": round(float(ghost.mean()), 3)},
           "drop_rule": "recall < %.2f or precision < %.2f" % (MIN_RECALL, MIN_PRECISION), "checks": {}}
    herr_all = np.array([r["H_err"] if r["H_err"] is not None else 0.0 for r in rows], float)
    out["base_rates"].update({"H_over_%gft" % t: round(float((herr_all > t).mean()), 3) for t in SEVERITY_FT})
    for name, flag in checks.items():
        res = {"vs_wrong_any": pr(flag, wrong), "vs_H_wrong": pr(flag, H_wrong), "vs_missed": pr(flag, missed), "vs_ghost": pr(flag, ghost)}
        for t in SEVERITY_FT:
            res["vs_H_over_%gft" % t] = pr(flag, herr_all > t)
        v = res["vs_wrong_any"]
        res["verdict"] = "keep" if (v["recall"] or 0) >= MIN_RECALL and (v["precision"] or 0) >= MIN_PRECISION else "drop"
        # A1 as a continuous score: Spearman-style rank correlation with H error
        if name.startswith("A1") and name.endswith("40px"):
            ok = ~np.isnan(a1)
            herr = np.array([r["H_err"] for r in rows], float)
            if ok.sum() > 10:
                from scipy.stats import spearmanr
                rho, p = spearmanr(a1[ok], herr[ok]); res["a1_vs_H_err_spearman"] = {"rho": round(float(rho), 3), "p": float(p)}
        out["checks"][name] = res
    (config.REPORTS_DIR / "sportvu_check_validation.json").write_text(json.dumps(out, indent=1))
    L = ["CHECK VALIDATION vs SportVU, %d testable frames; wrong = %s" % (n, out["wrong_definition"]),
         "  base rates: wrong_any %.2f  H>3ft %.2f  missed %.2f  ghost %.2f  | H>6ft %.2f  H>10ft %.2f  H>15ft %.2f" % tuple(out["base_rates"][k] for k in ("wrong_any", "H_wrong", "missed", "ghost", "H_over_6ft", "H_over_10ft", "H_over_15ft")),
         "  NOTE: with wrong_any at 1.00 precision is uninformative; read recall as the flag rate and look at the graded columns",
         "  drop rule: %s" % out["drop_rule"]]
    for name, res in out["checks"].items():
        v, hw = res["vs_wrong_any"], res["vs_H_wrong"]
        sev = " | ".join("H>%gft P %s R %s" % (t, res["vs_H_over_%gft" % t]["precision"], res["vs_H_over_%gft" % t]["recall"]) for t in SEVERITY_FT)
        L.append("  %-24s flagged %4d  vs wrong_any P %s R %s | vs H>3 P %s R %s | %s | %s%s" % (
            name, v["flagged"], v["precision"], v["recall"], hw["precision"], hw["recall"], sev, res["verdict"].upper(),
            ("  (A1 vs H_err spearman rho %s)" % res["a1_vs_H_err_spearman"]["rho"]) if "a1_vs_H_err_spearman" in res else ""))
    txt = "\n".join(L); (config.REPORTS_DIR / "sportvu_check_validation.txt").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
