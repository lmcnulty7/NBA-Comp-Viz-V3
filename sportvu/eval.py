"""
sportvu/eval.py: pipeline vs SportVU on the held-out game (FIX_PLAN B5).

On every testable frame (B4 truth H): position error in ft per matched player under the PIPELINE H
(p50, p90; by court region near / middle / far third relative to the camera, and by axis), frame H
error (median over matched players), missed players (SportVU players whose truth projection lies
inside the frame with no box foot within MISS_FT under H_truth), ghost boxes (boxes whose truth-
projected foot is > GHOST_FT from every SportVU player; referees are not in SportVU, so refs the
detector boxes count as ghosts here, stated), team accuracy (pipeline A/B per canonical track vs
SportVU team of the matched player, under the majority mapping per window). Nearest-defender
agreement (CLAIMS C1 proxy): per possession from matchup_metrics run on the window, the pipeline's
primary man for each defender vs the SportVU opponent most often nearest to that defender over the
possession's frames, tracks mapped to players by majority over the truth pairs.

  python -m sportvu.eval gsw_phx_2016 12.16.2015.PHX.at.GSW   -> reports/sportvu_phx.{json,txt}
"""
from __future__ import annotations
import json, subprocess, sys
import numpy as np
import config
from sportvu.sync import SportVUIndex, apply_mirror, MIRRORS
from sportvu.rebuild import BUILD_DIR
from sportvu.truth import TRUTH_DIR

MISS_FT, GHOST_FT = 2.5, 3.0
FRAME_MARGIN_PX = 10


def region_of(y: float, near_is_high_y: bool) -> str:
    from court.court33 import COURT_WIDTH_FT as W
    third = W / 3.0
    k = int(min(2, y // third))
    k = k if near_is_high_y else 2 - k            # 0 = far from camera .. 2 = near
    return ("far", "middle", "near")[k]


def near_side(H_truth: np.ndarray) -> bool:
    """True if the court's y=50 sideline is the near (camera-side) one: it projects lower on frame."""
    import cv2
    from court.court33 import COURT_LENGTH_FT as L, COURT_WIDTH_FT as W
    P = np.linalg.inv(H_truth)
    pts = cv2.perspectiveTransform(np.array([[L / 2, 0.0], [L / 2, W]], np.float32).reshape(-1, 1, 2), P).reshape(-1, 2)
    return bool(pts[1][1] > pts[0][1])


def eval_frame(row: dict, side_row: dict, sv_xy: np.ndarray, sv_pid: np.ndarray, sv_team: np.ndarray, mirror: str, w: int, h: int) -> dict:
    import cv2
    fx, fy = MIRRORS[mirror]
    sv = apply_mirror(sv_xy, fx, fy)
    Hp = np.array(side_row["H"], np.float64).reshape(3, 3); Ht = np.array(row["H_truth"], np.float64).reshape(3, 3)
    feet = np.array([b["foot_stab"] for b in side_row["boxes"]], np.float32)
    cp = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hp).reshape(-1, 2)   # pipeline court positions
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2)   # truth court positions of the same feet
    nh = near_side(Ht)
    out = {"frame": row["frame"], "matched": [], "frame_err_ft": None, "missed": 0, "ghosts": 0, "in_frame_players": 0}
    errs = []
    for p in row["pairs"]:
        i, j = p["box"], int(np.where(sv_pid == p["pid"])[0][0])
        e = cp[i] - sv[j]
        errs.append(float(np.hypot(*e)))
        out["matched"].append({"box": i, "tid": side_row["boxes"][i]["tid"], "pid": p["pid"], "team": int(sv_team[j]),
                               "err_ft": round(float(np.hypot(*e)), 2), "err_x": round(float(e[0]), 2), "err_y": round(float(e[1]), 2),
                               "err_toward_camera": round(float(e[1] if nh else -e[1]), 2),
                               "region": region_of(float(sv[j][1]), nh), "inlier": p["inlier"]})
    out["frame_err_ft"] = round(float(np.median(errs)), 2) if errs else None
    # missed: SportVU players inside the frame (truth projection) with no box foot within MISS_FT (truth frame)
    px = cv2.perspectiveTransform(sv.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(Ht)).reshape(-1, 2)
    inside = (px[:, 0] > FRAME_MARGIN_PX) & (px[:, 0] < w - FRAME_MARGIN_PX) & (px[:, 1] > FRAME_MARGIN_PX) & (px[:, 1] < h - FRAME_MARGIN_PX)
    out["in_frame_players"] = int(inside.sum())
    if len(ct):
        d_sv = np.sqrt(((sv[:, None, :] - ct[None, :, :]) ** 2).sum(-1)).min(1)
        out["missed"] = int((inside & (d_sv > MISS_FT)).sum())
        d_box = np.sqrt(((ct[:, None, :] - sv[None, :, :]) ** 2).sum(-1)).min(1)
        out["ghosts"] = int((d_box > GHOST_FT).sum())
    else:
        out["missed"] = int(inside.sum())
    out["n_boxes"] = int(len(feet))
    return out


def run_stage(cmd: list) -> int:
    import os
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    return subprocess.run(["/opt/anaconda3/bin/python"] + cmd, cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True).returncode


def nearest_defender_proxy(windows: list, index: SportVUIndex, mirror: str) -> dict:
    """CLAIMS C1 proxy. Per possession (segment_possessions + matchup_metrics on the window): the
    pipeline's primary man for each defender vs the SportVU opponent most often nearest to that
    defender across the possession's testable frames. Tracks map to SportVU players by majority
    over the truth pairs inside the possession."""
    import shutil
    n_def = n_mapped = n_agree = 0; per = []
    for wd in windows:
        stem = wd["window"]
        tp = TRUTH_DIR / (stem + "_truth.json")
        if not tp.exists():
            continue
        mp = BUILD_DIR / (stem + "_matchups.json")
        if not mp.exists():
            for suf in ("_trajectories.json", "_identity.json"):
                shutil.copy(BUILD_DIR / (stem + suf), config.TRACKING_DIR / (stem + suf))
            run_stage(["segment_possessions.py", "--trajectories", str(config.TRACKING_DIR / (stem + "_trajectories.json")), "--fps", str(wd["fps"])])
            run_stage(["matchup_metrics.py", "--clip", stem, "--fps", str(wd["fps"]), "--no-video"])
            for p in config.TRACKING_DIR.glob(stem + "_*"):
                shutil.move(str(p), str(BUILD_DIR / p.name))
            for p in (config.PROJECT_ROOT / "reports" / "viz").glob("*" + stem + "*"):
                p.unlink()
            if not mp.exists():
                continue
        truth = json.loads(tp.read_text()); side = json.loads((BUILD_DIR / (stem + "_frames.json")).read_text())
        srow = {r["frame"]: r for r in side["frames"]}
        ident = json.loads((BUILD_DIR / (stem + "_identity.json")).read_text())
        idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
        ok_rows = [r for r in truth["frames"] if r["status"] == "ok"]
        for poss in json.loads(mp.read_text())["possessions"]:
            f0, f1 = poss["core_start_frame"], poss["core_end_frame"]
            rows = [r for r in ok_rows if f0 <= r["frame"] <= f1]
            if len(rows) < 5:
                continue
            # track -> pid votes inside the possession
            votes = {}
            for r in rows:
                boxes = srow[r["frame"]]["boxes"]
                for pr in r["pairs"]:
                    cid = idmap.get(boxes[pr["box"]]["tid"], boxes[pr["box"]]["tid"])
                    votes.setdefault(cid, {}).setdefault(pr["pid"], 0)
                    votes[cid][pr["pid"]] += 1
            tid2pid = {cid: max(v, key=v.get) for cid, v in votes.items() if max(v.values()) >= 3}
            for d in poss["defenders"]:
                n_def += 1
                dp, mp_ = tid2pid.get(d["defender"]), tid2pid.get(d["primary_man"])
                if dp is None or mp_ is None:
                    continue
                # SportVU: opponent nearest to the defender, mode over the possession's frames
                near = {}
                for r in rows:
                    q, i = r["q"], r["moment"]
                    pid, xy, team = index.q[q]["pid"][i], index.q[q]["xy"][i], index.q[q]["team"][i]
                    k = np.where(pid == dp)[0]
                    if not len(k):
                        continue
                    k = k[0]; opp = np.where(team != team[k])[0]
                    if not len(opp):
                        continue
                    j = opp[np.argmin(np.sqrt(((xy[opp] - xy[k]) ** 2).sum(-1)))]
                    near[int(pid[j])] = near.get(int(pid[j]), 0) + 1
                if not near:
                    continue
                n_mapped += 1
                sv_best = max(near, key=near.get)
                agree = (sv_best == mp_)
                n_agree += agree
                per.append({"window": stem, "defender_pid": dp, "pipeline_primary_pid": mp_, "sportvu_nearest_pid": sv_best,
                            "share_nearest": round(near[sv_best] / sum(near.values()), 2), "agree": bool(agree)})
    return {"defenders": n_def, "mapped": n_mapped, "agree": n_agree, "agreement": round(n_agree / n_mapped, 3) if n_mapped else None,
            "note": "proxy for CLAIMS C1: pipeline primary man vs the SportVU opponent most often nearest; not the human-labeled matchup", "rows": per}


def main():
    import cv2
    game = sys.argv[1] if len(sys.argv) > 1 else "gsw_phx_2016"
    sv_game = sys.argv[2] if len(sys.argv) > 2 else "12.16.2015.PHX.at.GSW"
    mom = json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (sv_game + "_moments.json")).read_text())
    index = SportVUIndex(mom["moments"])
    mirror = json.loads((config.REPORTS_DIR / ("sportvu_sync_%s.json" % game)).read_text())["direction_resolution"]["mirror"]
    windows = json.loads((BUILD_DIR / (game + "_windows.json")).read_text())
    all_matched, frame_errs, missed, ghosts, in_frame, n_boxes_sum, frames = [], [], 0, 0, 0, 0, 0
    team_rows = []        # (window, tid, pipeline_team, sv_team)
    md_pairs = []         # nearest-defender proxy rows
    per_window = {}
    for wd in windows:
        tp = TRUTH_DIR / (wd["window"] + "_truth.json")
        if not tp.exists():
            continue
        truth = json.loads(tp.read_text()); side = json.loads((BUILD_DIR / (wd["window"] + "_frames.json")).read_text())
        srow = {r["frame"]: r for r in side["frames"]}
        ident_p = BUILD_DIR / (wd["window"] + "_identity.json")
        ident = json.loads(ident_p.read_text()) if ident_p.exists() else {}
        idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
        team_of = {int(k): v for k, v in ident.get("team_by_track", {}).items()}
        cap = cv2.VideoCapture(str(BUILD_DIR / (wd["window"] + ".mp4"))); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        wm, wf = [], []
        for r in truth["frames"]:
            if r["status"] != "ok":
                continue
            q, i = r["q"], r["moment"]
            ev = eval_frame(r, srow[r["frame"]], index.q[q]["xy"][i], index.q[q]["pid"][i], index.q[q]["team"][i], mirror, W, H)
            frames += 1; missed += ev["missed"]; ghosts += ev["ghosts"]; in_frame += ev["in_frame_players"]; n_boxes_sum += ev["n_boxes"]
            frame_errs.append(ev["frame_err_ft"]); wf.append(ev["frame_err_ft"])
            for m in ev["matched"]:
                all_matched.append(m); wm.append(m["err_ft"])
                cid = idmap.get(m["tid"], m["tid"])
                if team_of.get(cid) is not None:
                    team_rows.append((wd["window"], cid, team_of[cid], m["team"]))
        per_window[wd["window"]] = {"frames": len(wf), "frame_err_p50": round(float(np.median(wf)), 2) if wf else None,
                                    "player_err_p50": round(float(np.median(wm)), 2) if wm else None}
    errs = np.array([m["err_ft"] for m in all_matched])
    by_region = {}
    for reg in ("far", "middle", "near"):
        e = np.array([m["err_ft"] for m in all_matched if m["region"] == reg])
        ey = np.array([m["err_y"] for m in all_matched if m["region"] == reg])
        ec = np.array([m["err_toward_camera"] for m in all_matched if m["region"] == reg])
        by_region[reg] = {"n": int(len(e)), "p50": round(float(np.median(e)), 2) if len(e) else None,
                          "p90": round(float(np.percentile(e, 90)), 2) if len(e) else None,
                          "bias_y_ft": round(float(np.median(ey)), 2) if len(ey) else None,
                          "bias_toward_camera_ft": round(float(np.median(ec)), 2) if len(ec) else None}
    # team accuracy: majority mapping pipeline label -> SportVU team, per window
    team_acc = {"n": 0, "correct": 0}
    for wname in {t[0] for t in team_rows}:
        rows = [t for t in team_rows if t[0] == wname]
        for lab in {t[2] for t in rows}:
            sub = [t for t in rows if t[2] == lab]
            counts = {}
            for t in sub:
                counts[t[3]] = counts.get(t[3], 0) + 1
            best = max(counts.values())
            team_acc["n"] += len(sub); team_acc["correct"] += best
    team_acc["accuracy"] = round(team_acc["correct"] / team_acc["n"], 3) if team_acc["n"] else None
    nd = nearest_defender_proxy(windows, index, mirror)
    rep = {"game": game, "sportvu_game": sv_game, "mirror": mirror, "nearest_defender_proxy": {k: v for k, v in nd.items() if k != "rows"}, "method": __doc__.strip().split("\n\n")[0],
           "testable_frames": frames, "matched_players": int(len(errs)),
           "position_error_ft": {"p50": round(float(np.median(errs)), 2), "p90": round(float(np.percentile(errs, 90)), 2),
                                 "mean": round(float(errs.mean()), 2), "bias_x_ft": round(float(np.median([m["err_x"] for m in all_matched])), 2),
                                 "bias_y_ft": round(float(np.median([m["err_y"] for m in all_matched])), 2),
                                 "bias_toward_camera_ft": round(float(np.median([m["err_toward_camera"] for m in all_matched])), 2)} if len(errs) else None,
           "by_region": by_region,
           "frame_H_error_ft": {"p50": round(float(np.median(frame_errs)), 2), "p90": round(float(np.percentile(frame_errs, 90)), 2)} if frame_errs else None,
           "missed_players": {"n": missed, "in_frame_players": in_frame, "rate": round(missed / in_frame, 3) if in_frame else None},
           "ghost_boxes": {"n": ghosts, "boxes": n_boxes_sum, "rate": round(ghosts / n_boxes_sum, 3) if n_boxes_sum else None,
                           "caveat": "referees are not in SportVU, so a correctly boxed referee counts as a ghost here"},
           "team_accuracy": team_acc, "per_window": per_window,
           "caveats": ["testable frames only (B4: 15% of window frames); the untestable share is itself a symptom",
                       "truth H derived through the pipeline's own boxes and initialised from its H",
                       "local windowed rebuild of the harvest sections, not the production artifacts (same code, models, h264 source)",
                       "nearest-defender agreement is a proxy for CLAIMS C1, not the human-labeled matchup accuracy"]}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "sportvu_phx.json").write_text(json.dumps(rep, indent=1))
    L = ["SPORTVU EVAL, gsw_phx_2016 (held out) vs 12.16.2015.PHX.at.GSW: %d testable frames, %d matched players" % (frames, len(errs)),
         "  position error (pipeline H vs SportVU): p50 %.2f ft  p90 %.2f ft  bias x %+.2f  bias y %+.2f" % (rep["position_error_ft"]["p50"], rep["position_error_ft"]["p90"], rep["position_error_ft"]["bias_x_ft"], rep["position_error_ft"]["bias_y_ft"]),
         "  by region (camera): " + "  ".join("%s n=%d p50 %s p90 %s toward-camera %s" % (k, v["n"], v["p50"], v["p90"], v["bias_toward_camera_ft"]) for k, v in by_region.items()),
         "  frame H error: p50 %.2f  p90 %.2f ft" % (rep["frame_H_error_ft"]["p50"], rep["frame_H_error_ft"]["p90"]),
         "  missed players: %d of %d in frame (%.1f%%)" % (missed, in_frame, 100 * missed / max(in_frame, 1)),
         "  ghost boxes: %d of %d (%.1f%%; refs count as ghosts)" % (ghosts, n_boxes_sum, 100 * ghosts / max(n_boxes_sum, 1)),
         "  team accuracy: %s (n=%d)" % (team_acc["accuracy"], team_acc["n"]),
         "  nearest-defender proxy (C1): agreement %s on %d of %d defenders mapped" % (nd["agreement"], nd["mapped"], nd["defenders"])]
    txt = "\n".join(L); (config.REPORTS_DIR / "sportvu_phx.txt").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
