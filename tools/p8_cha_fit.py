"""tools/p8_cha_fit.py: why gsw_cha_2016's camera fit broke with the reset-fixed label builds
(docs/research/P8_cha_fit.md). Report only: nothing is retuned and no truth changes.

    python3 -m tools.p8_cha_fit stage1        stored files only (no fit, no model, no video decode, no lock)
        -> <reference root>/cha_fit/stage1.json and reports/research/p8_cha_fit.json

Stage 1 measurements (the design's section 4, fixed before any number):
- S1 arena consistency: C, C0 and reprojection of every recorded camera fit of the in-use games (P8 workspace, MPS;
  reference root, CPU without the fix where it exists, under superseded_nofix; reference root, CPU with the fix), and
  gsw_ind_2016's MPS fit for reference; each C's distance from the median C of the other GSW home fits.
- S2 court identity: for every label window and frame in both gsw_cha_2016 label builds, whether the per-frame court H
  is identical; where not, the change in the closed-form camera decomposition (sportvu.camera.pose_from_h) position.
- S3 where the change sits: per frame, boxes in the sidecar and frames with at least 6 boxes, before and after the
  window's first reset (the fixed build's reset log; before it both builds run the same code), in both builds.
- S4 what the truth kept: accepted rows of both truths (count, feet_resid_ft, line_matches) before and after the first
  reset, and the windows with the largest loss and gain.
- S5 C0 against C for both gsw_cha_2016 fits.

    <frozen CV python> -m tools.p8_cha_fit stage2   (under the heavy-job lock, name "p8 cha fit stage2"; V3_P8_ROOT
        set to the reference root) -> <reference root>/diag/cha_fit/stage2.json and reports/research/p8_cha_fit_stage2.json

Stage 2 (the design's section 8, registered before any stage 2 number; approved by the lead): F0 reproduction of both
recorded fits (stops on failure), F1 seeds 1 to 10 on both builds, F2 two starts on the fixed seed-0 pool (and a
report-only run with 2,000 evaluations), F3 init subsets, F4 pair changes on shared init frames. It reruns the frozen
sportvu.truth.window_truth, sportvu.camera.frames_from_rows and sportvu.camera.fit_game on the stored label sidecars
(no model, no video decode beyond reading clip sizes) and a tools-side copy of fit_game's least-squares call that must
reproduce the frozen C. Diagnosis only: no output is a truth, nothing is written outside diag/cha_fit and the record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_cha_fit"
GAME = "gsw_cha_2016"
GSW = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
OTHERS = ("hou_cle_2016",)
REFERENCE_ONLY = ("gsw_ind_2016",)          # apart (D98); its MPS fit is read for reference only
MIN_BOXES = 6
W_PX, H_PX = 1280, 720


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def first_reset(log) -> int | None:
    rs = (log or {}).get("resets") or []
    frames = [r["frame"] for r in rs if r.get("frame") is not None]
    return min(frames) if frames else None


def side_of(frame: int, first: int | None) -> str:
    return "before" if first is None or frame < first else "after"


def stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 3), "p25": round(float(np.percentile(v, 25)), 3),
            "p75": round(float(np.percentile(v, 75)), 3), "mean": round(float(v.mean()), 3)}


def distance_to(c, others) -> float | None:
    if not others:
        return None
    return round(float(np.linalg.norm(np.asarray(c, float) - np.median(np.asarray(others, float), axis=0))), 2)


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def roots():
    base = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
    return check_outside(base / "work"), check_outside(base / "work_ref")


def fits(work: Path, ref: Path) -> dict:
    out = {}
    for g in GSW + OTHERS + REFERENCE_ONLY:
        for name, p in (("mps", work / "labels" / f"{g}_result.json"),
                        ("cpu_nofix", ref / "superseded_nofix" / "labels" / f"{g}_result.json"),
                        ("cpu_fix", ref / "labels" / f"{g}_result.json")):
            if g in REFERENCE_ONLY and name != "mps":
                continue
            if p.exists():
                r = json.loads(p.read_text())
                c = r.get("camera") or {}
                out[f"{g}:{name}"] = {"game": g, "fit": name, "C": c.get("C"), "C0": c.get("C0"),
                                      "reproj_px_after": c.get("reproj_px_median_after"),
                                      "reproj_px_before": c.get("reproj_px_median_before"),
                                      "inits": c.get("inits"), "pooled": c.get("pooled_frames"),
                                      "accepted": r.get("accepted"), "status": c.get("status")}
    for k, v in out.items():
        if v["game"] in GSW and v["C"]:
            peers = [o["C"] for kk, o in out.items() if kk != k and o["game"] in GSW and o["C"]
                     and not (o["game"] == GAME and o["fit"] == "cpu_fix")]
            v["dist_ft_to_median_C_of_other_gsw_fits_excluding_cha_fix"] = distance_to(v["C"], peers)
    return out


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def windows(lb: Path) -> list:
    return sorted(p.name[:-len("_frames.json")] for p in lb.glob(f"{GAME}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))


def read_tags() -> list:
    """Every game whose files stage 1 reads: gsw_cha_2016 and the fits of S1."""
    return sorted({GAME, *GSW, *OTHERS, *REFERENCE_ONLY})


def stage1() -> dict:
    from sportvu.camera import pose_from_h
    tags = read_tags()
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    work, ref = roots()
    lb_fix, lb_old = ref / "label_build", ref / "superseded_nofix" / "label_build"
    stems = sorted(set(windows(lb_fix)) | set(windows(lb_old)))
    guard(stems, WHO)
    s2 = {"frames_both": 0, "H_identical": 0, "H_differs": 0, "H_none_both": 0, "H_missing_one_side": 0, "dC_ft": []}
    s3 = {b: {s: {"boxes": [], "frames_ge6": 0, "frames": 0} for s in ("before", "after")} for b in ("nofix", "fix")}
    s4 = {b: {s: {"feet_resid_ft": [], "line_matches": [], "n": 0} for s in ("before", "after")} for b in ("nofix", "fix")}
    per_window = {}
    for s in stems:
        log = load(lb_fix / f"{s}_resetfix.json")
        first = first_reset(log)
        fx, ol = load(lb_fix / f"{s}_frames.json"), load(lb_old / f"{s}_frames.json")
        rows_f = {int(r["frame"]): r for r in (fx or {}).get("frames", [])}
        rows_o = {int(r["frame"]): r for r in (ol or {}).get("frames", [])}
        for f in sorted(set(rows_f) & set(rows_o)):
            a, b = rows_o[f].get("H"), rows_f[f].get("H")
            s2["frames_both"] += 1
            if a is None or b is None:
                s2["H_missing_one_side" if (a is None) != (b is None) else "H_none_both"] += 1
                continue
            if np.allclose(a, b, rtol=0, atol=1e-9):
                s2["H_identical"] += 1
            else:
                s2["H_differs"] += 1
                pa, pb = pose_from_h(a, W_PX, H_PX), pose_from_h(b, W_PX, H_PX)
                if pa is not None and pb is not None:
                    s2["dC_ft"].append(float(np.linalg.norm(np.asarray(pa[2]) - np.asarray(pb[2]))))
        for build, rows in (("nofix", rows_o), ("fix", rows_f)):
            for f, r in rows.items():
                c = s3[build][side_of(f, first)]
                n = len(r.get("boxes") or [])
                c["boxes"].append(n); c["frames"] += 1; c["frames_ge6"] += int(n >= MIN_BOXES)
        acc = {}
        for build, root in (("nofix", ref / "superseded_nofix" / "labels" / GAME), ("fix", ref / "labels" / GAME)):
            p = root / f"{s}.jsonl"
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
            acc[build] = len(rows)
            for r in rows:
                c = s4[build][side_of(int(r["frame"]), first)]
                c["n"] += 1; c["feet_resid_ft"].append(r.get("feet_resid_ft")); c["line_matches"].append(r.get("line_matches"))
        per_window[s] = {"first_reset_frame": first, "resets_fix": len((log or {}).get("resets") or []),
                         "accepted_nofix": acc["nofix"], "accepted_fix": acc["fix"]}
    f = fits(work, ref)
    old, new = f.get(f"{GAME}:cpu_nofix", {}), f.get(f"{GAME}:cpu_fix", {})
    s5 = {"nofix": {"C0": old.get("C0"), "C": old.get("C")}, "fix": {"C0": new.get("C0"), "C": new.get("C")},
          "C0_shift_ft": distance_to(new.get("C0"), [old.get("C0")]) if old.get("C0") and new.get("C0") else None,
          "C_shift_ft": distance_to(new.get("C"), [old.get("C")]) if old.get("C") and new.get("C") else None}
    deltas = sorted(per_window.items(), key=lambda kv: kv[1]["accepted_fix"] - kv[1]["accepted_nofix"])
    out = {"what": "docs/research/P8_cha_fit.md stage 1; report only; stored files", "game": GAME,
           "S1_fits": f,
           "S2_court": {**{k: v for k, v in s2.items() if k != "dC_ft"}, "dC_ft": stats(s2["dC_ft"])},
           "S3_boxes": {b: {s: {"frames": c["frames"], "frames_ge6_boxes": c["frames_ge6"], "boxes_per_frame": stats(c["boxes"])}
                            for s, c in v.items()} for b, v in s3.items()},
           "S4_accepted": {b: {s: {"n": c["n"], "feet_resid_ft": stats(c["feet_resid_ft"]), "line_matches": stats(c["line_matches"])}
                               for s, c in v.items()} for b, v in s4.items()},
           "S4_windows_largest_loss": [{"window": k, **v} for k, v in deltas[:8]],
           "S4_windows_largest_gain": [{"window": k, **v} for k, v in deltas[-5:]],
           "S5_C0_vs_C": s5,
           "windows": len(stems), "windows_with_a_reset_fix": sum(1 for v in per_window.values() if v["resets_fix"])}
    text = json.dumps(out, indent=1) + "\n"
    d = ref / "cha_fit"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "stage1.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_cha_fit.json", text)
    print(text)
    return out


# ── stage 2 ────────────────────────────────────────────────────────────────────────────────────────────────────────
LEAVE_FT, GOOD_PX, SEEDS = 5.0, 10.0, tuple(range(1, 11))
C_OLD = (46.90991991411326, 127.7548789042653, -31.373430881051902)   # the unfixed CPU fit's C (stage 1, S1)


def leaves_start(run: dict) -> bool:
    return bool(run.get("C")) and bool(run.get("C0")) and \
        float(np.linalg.norm(np.asarray(run["C"]) - np.asarray(run["C0"]))) >= LEAVE_FT


def good(run: dict) -> bool:
    return run.get("reproj_px_median_after") is not None and run["reproj_px_median_after"] <= GOOD_PX


def inits_of(frames: list) -> list:
    """fit_game's init list: frames whose V3 court H decomposes into a camera (sportvu.camera.fit_game, same rule)."""
    from sportvu.camera import pose_from_h
    out = []
    for fr in frames:
        p = pose_from_h(fr["H_v3"], fr["w"], fr["h"])
        if p is not None and p[2][2] != 0:
            out.append((fr, p))
    return out


def pooled_ls(frames: list, seed: int = 0, start_C=None, max_nfev: int = 200) -> dict:
    """A tools-side copy of sportvu.camera.fit_game's pooled least squares (same inits, same seeded pool, same residual,
    loss, scale and sparsity), with the start position and the evaluation cap as parameters, returning also the final
    cost, the evaluation count and the termination status. With start_C None and max_nfev 200 it must give fit_game's
    C (F0 checks this)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from sportvu.camera import F_SCALE_PX, POOL_MAX, project
    inits = inits_of(frames)
    if len(inits) < 20:
        return {"status": "too_few_inits", "n": len(inits)}
    C0 = np.median(np.array([p[2] for _, p in inits]), axis=0)
    rng = np.random.default_rng(seed)
    pool = [inits[i] for i in sorted(rng.choice(len(inits), size=min(POOL_MAX, len(inits)), replace=False))]
    x0 = [*(C0 if start_C is None else np.asarray(start_C, float))]
    for _, (rv, f, _) in pool:
        x0 += [*rv, np.log(f)]
    x0 = np.array(x0)
    sizes = [len(fr["px"]) for fr, _ in pool]

    def resid(x):
        C = x[:3]
        return np.concatenate([(project(C, x[3 + 4 * k: 6 + 4 * k], np.exp(x[6 + 4 * k]), fr["w"], fr["h"], fr["ft"])
                                - fr["px"]).ravel() for k, (fr, _) in enumerate(pool)])
    S = lil_matrix((2 * sum(sizes), len(x0)), dtype=int)
    row = 0
    for k, n in enumerate(sizes):
        S[row:row + 2 * n, :3] = 1
        S[row:row + 2 * n, 3 + 4 * k: 7 + 4 * k] = 1
        row += 2 * n
    sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=F_SCALE_PX, x_scale="jac", max_nfev=max_nfev)
    return {"C": sol.x[:3].tolist(), "C0": C0.tolist(), "start_C": x0[:3].tolist(), "cost": float(sol.cost),
            "nfev": int(sol.nfev), "ls_status": int(sol.status), "message": str(sol.message),
            "reproj_px_median_after": round(float(np.median(np.abs(sol.fun))), 2), "pooled_frames": len(pool),
            "inits": len(inits), "pool_keys": [(fr["window"], int(fr["frame"])) for fr, _ in pool]}


def summary(run: dict) -> dict:
    out = {k: run.get(k) for k in ("C", "C0", "reproj_px_median_after", "inits", "pooled_frames", "cost", "nfev",
                                   "ls_status", "status")}
    out["dist_C_C0_ft"] = round(float(np.linalg.norm(np.asarray(run["C"]) - np.asarray(run["C0"]))), 2) \
        if run.get("C") and run.get("C0") else None
    out["leaves_start"], out["good"] = leaves_start(run), good(run)
    return out


def pair_key(fr: dict) -> tuple:
    return (np.round(np.asarray(fr["px"]), 4).tobytes(), np.round(np.asarray(fr["ft"]), 4).tobytes())


def readings(f1: dict, f2: dict, f3: dict, f4: dict) -> dict:
    """The section 8 reading rules applied to the stage 2 numbers."""
    fix_leave_good = sum(1 for r in f1["fix"] if r["leaves_start"] and r["good"])
    fix_leave = sum(1 for r in f1["fix"] if r["leaves_start"])
    old_leave = sum(1 for r in f1["nofix"] if r["leaves_start"])
    out = {"F1": {"fix_runs_leaving_start_and_good": fix_leave_good, "fix_runs_leaving_start": fix_leave,
                  "nofix_runs_leaving_start": old_leave, "runs": len(f1["fix"])}}
    if fix_leave_good >= 1 and old_leave == len(f1["nofix"]):
        out["F1"]["reading"] = "H1: the outcome depends on which frames are pooled"
    elif fix_leave == 0 and old_leave == len(f1["nofix"]):
        out["F1"]["reading"] = "the fixed init set holds the fit at its start; F2 and F3 decide why"
    elif old_leave < len(f1["nofix"]):
        out["F1"]["reading"] = "the fit is fragile on both builds; the fix exposed it"
    else:
        out["F1"]["reading"] = "mixed (no registered pattern)"
    a, b = f2["from_C_old"], f2["from_C0"]
    near_old = float(np.linalg.norm(np.asarray(a["C"]) - np.asarray(C_OLD))) <= LEAVE_FT
    out["F2"] = {"from_C_old_ends_near_C_old": near_old, "cost_from_C_old": a["cost"], "cost_from_C0": b["cost"],
                 "reading": ("H2: the pool has a better minimum the frozen start did not reach" if near_old and a["cost"] < b["cost"]
                             else "the pool prefers the near position (H3 or H5, by F3 and F4)")}
    sub_ok = all(f3[k]["leaves_start"] and f3[k]["good"] for k in ("old_inits_only", "no_post_reset"))
    sub_stuck = all(not f3[k]["leaves_start"] for k in ("old_inits_only", "no_post_reset"))
    out["F3"] = {"reading": ("H3: the frames added after resets cause it" if sub_ok and fix_leave == 0
                             else "the added frames are not the cause (H5 or H2)" if sub_stuck else "mixed (no registered pattern)")}
    out["F4"] = {"share_changed": f4["share_changed"],
                 "reading": "H5: the old frames' inputs changed too" if (f4["share_changed"] or 0) > 0.10 else "not H5"}
    return out


def stage2() -> dict:
    import os
    import time
    import cv2
    from sportvu.camera import fit_game, frames_from_rows
    from sportvu.sync import SportVUIndex
    from sportvu.truth import window_truth
    from tools import p8_frozen as F
    refuse_cv_test([GAME], WHO)
    guard([GAME], WHO)
    work, ref = roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ref.resolve():
        raise SystemExit(f"{WHO}: run stage2 with V3_P8_ROOT set to the reference root")
    t0 = time.time()
    P = F.install_paths()
    g = F.game_of(GAME)
    mirror = F.mirror_of(P, GAME)
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    secs = [s for s in F.sections(P, GAME) if (P["sync"] / f"{s}_local_timemap.json").exists()]
    wins = [w for s in secs for w in F.label_windows(P, GAME, s)]
    guard([w["window"] for w in wins], WHO)
    sizes = {}

    def frames_of(side_dir: Path) -> list:
        out = []
        for w in wins:
            side_p, tm_p = side_dir / (w["window"] + "_frames.json"), P["label_build"] / (w["window"] + "_timemap.json")
            if not side_p.exists() or not tm_p.exists():
                continue
            sc, tm = json.loads(side_p.read_text()), json.loads(tm_p.read_text())
            _, _, rows = window_truth(w, sc, tm, index, mirror)
            if w["window"] not in sizes:
                cap = cv2.VideoCapture(str(P["label_build"] / (w["window"] + ".mp4")))
                sizes[w["window"]] = (int(cap.get(3)), int(cap.get(4)))
                cap.release()
            out += frames_from_rows(w, sc, rows, index, mirror, *sizes[w["window"]])
        return out
    builds = {"fix": frames_of(ref / "label_build"), "nofix": frames_of(ref / "superseded_nofix" / "label_build")}
    recorded = {"fix": json.loads((ref / "labels" / f"{GAME}_result.json").read_text())["camera"],
                "nofix": json.loads((ref / "superseded_nofix" / "labels" / f"{GAME}_result.json").read_text())["camera"]}
    out = {"what": "docs/research/P8_cha_fit.md stage 2 (section 8); diagnosis only, no output is a truth", "F0": {}}
    for b, frames in builds.items():
        fz, cp = fit_game(frames, seed=0), pooled_ls(frames, seed=0)
        rec = recorded[b]
        ok = (fz.get("status") == "ok" and np.linalg.norm(np.asarray(fz["C"]) - np.asarray(rec["C"])) <= 0.01
              and fz["inits"] == rec["inits"] and fz["pooled_frames"] == rec["pooled_frames"]
              and abs(fz["reproj_px_median_after"] - rec["reproj_px_median_after"]) <= 0.01
              and np.linalg.norm(np.asarray(cp["C"]) - np.asarray(fz["C"])) <= 0.01)
        out["F0"][b] = {"reproduced": bool(ok), "frozen": {k: fz.get(k) for k in ("C", "inits", "pooled_frames", "reproj_px_median_after")},
                        "recorded": {k: rec.get(k) for k in ("C", "inits", "pooled_frames", "reproj_px_median_after")},
                        "copy": summary(cp)}
    if not all(v["reproduced"] for v in out["F0"].values()):
        out["stopped"] = "F0: a recorded fit is not reproduced; stage 2 reports F0 only"
    else:
        f1 = {b: [summary({**fit_game(fr, seed=s), "seed": s}) | {"seed": s} for s in SEEDS] for b, fr in builds.items()}
        f2 = {"from_C0": summary(pooled_ls(builds["fix"], seed=0)),
              "from_C_old": summary(pooled_ls(builds["fix"], seed=0, start_C=C_OLD)),
              "report_only_from_C0_2000_evaluations": summary(pooled_ls(builds["fix"], seed=0, max_nfev=2000))}
        old_keys = {(fr["window"], int(fr["frame"])) for fr, _ in inits_of(builds["nofix"])}
        firsts = {w["window"]: first_reset(load(ref / "label_build" / (w["window"] + "_resetfix.json"))) for w in wins}
        sub_a = [fr for fr in builds["fix"] if (fr["window"], int(fr["frame"])) in old_keys]
        sub_b = [fr for fr in builds["fix"] if side_of(int(fr["frame"]), firsts.get(fr["window"])) == "before"]
        f3 = {"old_inits_only": summary(fit_game(sub_a, seed=0)) | {"frames": len(sub_a)},
              "no_post_reset": summary(fit_game(sub_b, seed=0)) | {"frames": len(sub_b)}}
        new_by = {(fr["window"], int(fr["frame"])): fr for fr, _ in inits_of(builds["fix"])}
        old_by = {(fr["window"], int(fr["frame"])): fr for fr, _ in inits_of(builds["nofix"])}
        shared = sorted(set(new_by) & set(old_by))
        changed = sum(1 for k in shared if pair_key(new_by[k]) != pair_key(old_by[k]))
        f4 = {"shared_init_frames": len(shared), "changed": changed,
              "share_changed": round(changed / len(shared), 4) if shared else None,
              "inits_fix_only": len(set(new_by) - set(old_by)), "inits_nofix_only": len(set(old_by) - set(new_by))}
        out.update({"F1": f1, "F2": f2, "F3": f3, "F4": f4, "readings": readings(f1, f2, f3, f4)})
    out["seconds"] = round(time.time() - t0, 1)
    text = json.dumps(out, indent=1, default=str) + "\n"
    d = ref / "diag" / "cha_fit"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "stage2.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_cha_fit_stage2.json", text)
    print(json.dumps({k: out.get(k) for k in ("F0", "readings", "stopped", "seconds")}, indent=1, default=str))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    sub.add_parser("stage2")
    a = ap.parse_args(argv)
    stage1() if a.cmd == "stage1" else stage2()


if __name__ == "__main__":
    sys.exit(main())
