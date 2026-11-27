"""tools/p8_sac_checkc.py: why gsw_sac_2015's fixed CPU truth fails check C (docs/research/P8_sac_checkc.md, version
2.1). Stage 1 only: stored files, system python, no model, no lock. Report only.

    python3 -m tools.p8_sac_checkc stage1     -> reports/research/p8_sac_checkc.json (the only file it writes)

Inputs, all opened read-only: the check C records (the P8 workspace's truth_v2/<game>.json, MPS; the reference root's
truth_v2/dev/<game>.json, fixed, and superseded_nofix/truth_v2/dev/gsw_cha_2016.json, CPU unfixed); the identity files,
sidecars and time maps of the four gsw_sac_2015 builds; the reset-fix plan, equivalence records and reset logs; the
fits' stored offsets; the moments file and the mirror source of both roots; gsw_cha_2016's unfixed label builds. It
imports from tools.p8_truth_v2 only the pure stability and game_summary (no cmd_* function), and nothing here writes
under truth_v2/ or truth_accept/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_sac_checkc"
SAC, BKN, CHA = "gsw_sac_2015", "gsw_bkn_2015", "gsw_cha_2016"
GAMES = (SAC, BKN, CHA)
OVER = 0.2 + 1e-9                     # tools.p8_truth_v2.stability's share threshold
EDGE_S = 1.5 - 1e-9                   # sportvu.truth.OFFSET_MAX: a grid-edge hit
MIN_BOXES = 4
SIDES = ("v3", "r23")
DATA = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"


def out_path() -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / "p8_sac_checkc.json"


# facts the design quotes (sections 6.1, 6.2 and 7.1); stage 1 re-derives them and stops on a difference
KNOWN = {
    "mps_reset_windows_sac": ["gsw_sac_2015_s01_w00", "gsw_sac_2015_s02_w00", "gsw_sac_2015_s04_w02",
                              "gsw_sac_2015_s04_w08", "gsw_sac_2015_s05_w04", "gsw_sac_2015_s07_w00",
                              "gsw_sac_2015_s07_w02", "gsw_sac_2015_s07_w08", "gsw_sac_2015_s08_w02",
                              "gsw_sac_2015_s08_w04", "gsw_sac_2015_s09_w01", "gsw_sac_2015_s09_w05"],
    "mps_only_sac": ["gsw_sac_2015_s02_w00", "gsw_sac_2015_s04_w08", "gsw_sac_2015_s05_w04"],
    "cpu_only_sac": ["gsw_sac_2015_s06_w02"],
    "sac_fixed": {"n": 73, "median_abs_s": 0.64, "share_over_0_2_s": 0.8219},
    "sac_mps": {"median_abs_s": 0.52, "share_over_0_2_s": 0.7945},
    "share_windows": (58, 60),
    "own_changes": {                  # (share change, median change), signed, each record on its own windows
        "sac MPS to fixed": (0.0274, 0.12), "bkn MPS to fixed": (-0.0390, 0.04),
        "cha MPS to unfixed": (-0.0193, -0.06), "cha MPS to fixed": (-0.0624, -0.04),
        "cha unfixed to fixed": (-0.0431, 0.02)},
}


# ── read-only helpers ──────────────────────────────────────────────────────────────────────────────────────────────
def read_json(p: Path):
    with open(p, "r") as fh:
        return json.load(fh)


def sha256(p: Path) -> str | None:
    if not Path(p).exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resets_of(p: Path) -> int | None:
    return None if not Path(p).exists() else int(read_json(p)["identity"].get("camera_cut_resets") or 0)


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def checkc_d(rec: dict) -> dict:
    """{window: |frozen V3 - frozen r23|} on check C's set: game_summary's kept windows (the V3 offset equals the
    fit's stored one) with both offsets."""
    out = {}
    for s, r in rec["windows"].items():
        if r["frozen"]["v3"] != r["stored_frozen_v3"]:
            continue
        a, b = r["frozen"]["v3"], r["frozen"]["r23"]
        if a is not None and b is not None:
            out[s] = abs(a - b)
    return out


def stats(d: dict) -> dict:
    """Median and share over 0.2 s of {window: d}, as tools.p8_truth_v2.stability computes them."""
    from tools.p8_truth_v2 import stability
    return stability([(v, 0.0) for v in d.values()])


def window_class(mps_v3: int, mps_r23: int, cpu_v3: int, cpu_r23: int) -> str:
    mps = (mps_v3 or 0) > 0 or (mps_r23 or 0) > 0
    cpu = (cpu_v3 or 0) > 0 or (cpu_r23 or 0) > 0
    return {(True, True): "both", (True, False): "MPS only", (False, True): "CPU only", (False, False): "none"}[(mps, cpu)]


def crossings(d_mps: dict, d_fix: dict, classes: dict) -> dict:
    """Gross crossings of 0.2 s from the MPS record to the fixed record, by class, on the windows both hold."""
    out = {}
    for s in sorted(d_mps.keys() & d_fix.keys()):
        c = classes.get(s, "unclassed")
        e = out.setdefault(c, {"up": [], "down": []})
        if d_mps[s] <= OVER < d_fix[s]:
            e["up"].append(s)
        elif d_fix[s] <= OVER < d_mps[s]:
            e["down"].append(s)
    return {c: {"up": len(v["up"]), "down": len(v["down"]), "net": len(v["up"]) - len(v["down"]),
                "up_windows": v["up"], "down_windows": v["down"]} for c, v in out.items()}


def moves(h: float, a: float, b: float) -> bool:
    """A hybrid value h covers at least half of the change from a (MPS) to b (fixed), in the same direction."""
    if h is None or a is None or b is None or b == a:
        return False
    return (h - a) / (b - a) >= 0.5 - 1e-9          # float tolerance on the halfway mark


def hybrid(d_mps: dict, d_fix: dict, take_fixed: set) -> dict:
    """d per window from the fixed record where the window is in take_fixed, else from the MPS record."""
    return {s: (d_fix[s] if s in take_fixed else d_mps[s]) for s in d_mps.keys() & d_fix.keys()}


def decomposition_reading(clock_differs: bool, mv_i: bool, mv_ii: bool) -> str:
    """Section 6.3's table, in its fixed priority."""
    if clock_differs:
        return "Clock"
    if mv_i and not mv_ii:
        return "Platform"
    if mv_ii and not mv_i:
        return "Fix"
    return "Joint" if (mv_i and mv_ii) else "Neither"


def side_reading(mv_v3: bool, mv_r23: bool) -> str:
    """Section 7.3, for one statistic."""
    if mv_v3 and mv_r23:
        return "both"
    if mv_v3:
        return "V3 side"
    if mv_r23:
        return "r23 side"
    return "not separated"


def combine_side(median_reading: str, share_reading: str) -> str:
    return median_reading if median_reading == share_reading else "not separated"


def within_controls(sac_abs: float, control_abs: list) -> str:
    """Section 7.1's open reading: within when sac's absolute change is at most the largest control's."""
    return "within" if control_abs and sac_abs <= max(control_abs) else "outside"


def qualifying_frames(side: dict, tm: dict, after: int | None = None) -> tuple:
    """(frames the frozen rule uses: a clock, an H and at least 4 boxes; of those, frames at or after `after`)."""
    from sportvu.truth import frame_clock
    n = n_after = 0
    for r in side["frames"]:
        q, _c = frame_clock(None, tm, r["frame"])
        if q is None or r.get("H") is None or len(r["boxes"]) < MIN_BOXES:
            continue
        n += 1
        n_after += after is not None and int(r["frame"]) >= after
    return n, n_after


def first_logged_reset(log: dict | None):
    fr = [r.get("frame") for r in (log or {}).get("resets", []) if r.get("frame") is not None]
    return min(fr) if fr else None


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ── stage 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def clock_inputs(ws: Path, ref: Path, game: str) -> dict:
    """The moments file and the mirror the two roots give the game (tools.p8_truth_v2._game's sources:
    tools.p8_frozen.game_of, the root's sportvu/<moments>.json and tools.p8_frozen.mirror_of)."""
    from tools.p8_frozen import game_of, layout, mirror_of
    sv = game_of(game)["sportvu"]
    out = {}
    for name, root in (("workspace", ws), ("reference", ref)):
        P = layout(root)
        out[name] = {"moments_sha256": sha256(P["sportvu"] / f"{sv}_moments.json"), "mirror": mirror_of(P, game)}
    return out


def check_known(name: str, got, want, tol: float, stop: list) -> None:
    if isinstance(want, (int, float)):
        if got is None or abs(got - want) > tol:
            stop.append(f"{name}: {got} against the quoted {want}")
    elif got != want:
        stop.append(f"{name}: {got} against the quoted {want}")


def run_stage1(ws: Path, ref: Path, expect: dict | None = KNOWN, clock=clock_inputs) -> dict:
    from tools.p8_common import atomic_write, guard
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(list(GAMES), WHO)
    guard(list(GAMES), WHO)
    stop = []
    rec = {"sac_mps": read_json(ws / "truth_v2" / f"{SAC}.json"),
           "sac_fixed": read_json(ref / "truth_v2" / "dev" / f"{SAC}.json"),
           "bkn_mps": read_json(ws / "truth_v2" / f"{BKN}.json"),
           "bkn_fixed": read_json(ref / "truth_v2" / "dev" / f"{BKN}.json"),
           "cha_mps": read_json(ws / "truth_v2" / f"{CHA}.json"),
           "cha_unfixed": read_json(ref / "superseded_nofix" / "truth_v2" / "dev" / f"{CHA}.json"),
           "cha_fixed": read_json(ref / "truth_v2" / "dev" / f"{CHA}.json")}
    d = {k: checkc_d(v) for k, v in rec.items()}
    st = {k: stats(v) for k, v in d.items()}
    out = {"what": "Why gsw_sac_2015's fixed CPU truth fails check C (docs/research/P8_sac_checkc.md), stage 1; "
                   "report only", "summaries": st}

    # 6.8 and 7.2: window sets and summaries
    wins_m, wins_f = set(rec["sac_mps"]["windows"]), set(rec["sac_fixed"]["windows"])
    if wins_m != wins_f:
        stop.append(f"the two gsw_sac_2015 records hold different windows ({len(wins_m)} against {len(wins_f)})")
    for k in ("sac_fixed", "bkn_fixed", "cha_fixed", "cha_unfixed"):
        stored = (rec[k].get("summary") or {}).get("frozen")
        if stored is not None and {x: stored.get(x) for x in ("n", "median_abs_s", "share_over_0_2_s")} != \
                {x: st[k].get(x) for x in ("n", "median_abs_s", "share_over_0_2_s")}:
            stop.append(f"{k}: recomputed summary {st[k]} differs from the stored {stored}")
    common = sorted(d["sac_mps"].keys() & d["sac_fixed"].keys())
    out["sac_checkc_sets"] = {"mps": len(d["sac_mps"]), "fixed": len(d["sac_fixed"]), "equal":
                              d["sac_mps"].keys() == d["sac_fixed"].keys(), "used": "equal sets" if
                              d["sac_mps"].keys() == d["sac_fixed"].keys() else "intersection", "n_used": len(common)}
    dm, df = {s: d["sac_mps"][s] for s in common}, {s: d["sac_fixed"][s] for s in common}
    if expect:
        for key, k in (("sac_fixed", "sac_fixed"), ("sac_mps", "sac_mps")):
            for x, v in expect[key].items():
                check_known(f"{k} {x}", st[k].get(x), v, 1e-4 if "share" in x else 1e-9 if x == "n" else 0.005, stop)
        n_m = sum(1 for v in d["sac_mps"].values() if v > OVER)
        n_f = sum(1 for v in d["sac_fixed"].values() if v > OVER)
        check_known("share windows", (n_m, n_f), tuple(expect["share_windows"]), 0, stop)

    # 6.1: classes from the four builds' identity files
    plan = read_json(ref / "resetfix" / f"plan__{SAC}.json")
    classes, resets = {}, {}
    for s in sorted(wins_f):
        r = {"mps_v3": resets_of(ws / "label_build" / f"{s}_identity.json"),
             "mps_r23": resets_of(ws / "builds" / "r23" / f"{s}_identity.json"),
             "cpu_v3": resets_of(ref / "label_build" / f"{s}_identity.json"),
             "cpu_r23": resets_of(ref / "builds" / "r23_nofix" / f"{s}_identity.json")}
        resets[s] = r
        classes[s] = window_class(*(r[k] for k in ("mps_v3", "mps_r23", "cpu_v3", "cpu_r23")))
    mps_reset = sorted(s for s, r in resets.items() if (r["mps_v3"] or 0) > 0 or (r["mps_r23"] or 0) > 0)
    cpu_reset = sorted(s for s, r in resets.items() if (r["cpu_v3"] or 0) > 0 or (r["cpu_r23"] or 0) > 0)
    if expect:
        check_known("MPS reset windows", mps_reset, sorted(expect["mps_reset_windows_sac"]), 0, stop)
        check_known("CPU reset windows (the plan's)", sorted(plan["reset_windows"]), cpu_reset, 0, stop)
        check_known("MPS only", sorted(s for s, c in classes.items() if c == "MPS only"), expect["mps_only_sac"], 0, stop)
        check_known("CPU only", sorted(s for s, c in classes.items() if c == "CPU only"), expect["cpu_only_sac"], 0, stop)
    out["classes"] = {"counts": {c: sum(1 for v in classes.values() if v == c) for c in ("both", "MPS only", "CPU only",
                                                                                            "none")},
                      "windows": classes, "resets": resets,
                      "mps_reset_frames_stored": False}

    # 6.4: clock by identity
    tm_diff = [s for s in sorted(wins_f) if sha256(ws / "label_build" / f"{s}_timemap.json") !=
               sha256(ref / "label_build" / f"{s}_timemap.json")]
    ck = clock(ws, ref, SAC)
    clock_differs = bool(tm_diff) or ck["workspace"] != ck["reference"]
    out["clock"] = {"time_maps_differing": tm_diff, "inputs": ck, "differs": clock_differs}

    # 6.2: crossings; 6.3 and 7.4: decomposition
    out["crossings"] = crossings(dm, df, classes)
    none = {s for s in common if classes.get(s) == "none"}
    rest = set(common) - none
    hi, hii = stats(hybrid(dm, df, none)), stats(hybrid(dm, df, rest))
    sm, sf = stats(dm), stats(df)
    dec = {}
    for stat in ("share_over_0_2_s", "median_abs_s"):
        mi, mii = moves(hi[stat], sm[stat], sf[stat]), moves(hii[stat], sm[stat], sf[stat])
        dec[stat] = {"mps": sm[stat], "fixed": sf[stat], "hybrid_i": hi[stat], "hybrid_ii": hii[stat],
                     "i_moves": mi, "ii_moves": mii, "reading": decomposition_reading(clock_differs, mi, mii)}
    out["decomposition"] = dec

    # 6.5 and 7.3: side, by cross pairs (only with identical time maps)
    if tm_diff:
        out["side"] = {"reading": "invalid (time maps differ)"}
    else:
        def cross(v3_rec, r23_rec):
            dd = {}
            for s in common:
                a, b = rec[v3_rec]["windows"][s]["frozen"]["v3"], rec[r23_rec]["windows"][s]["frozen"]["r23"]
                if a is not None and b is not None:
                    dd[s] = abs(a - b)
            return stats(dd)
        sv3, sr23 = cross("sac_fixed", "sac_mps"), cross("sac_mps", "sac_fixed")
        per = {}
        for stat in ("share_over_0_2_s", "median_abs_s"):
            per[stat] = side_reading(moves(sv3[stat], sm[stat], sf[stat]), moves(sr23[stat], sm[stat], sf[stat]))
        out["side"] = {"v3_swap": sv3, "r23_swap": sr23, "per_statistic": per,
                       "reading": combine_side(per["median_abs_s"], per["share_over_0_2_s"]),
                       "note": "the r23 side changes platform too (the workspace r23 rebuild is MPS)"}

    # 6.6 and 7.1: controls
    pairs = {"sac MPS to fixed": ("sac_mps", "sac_fixed"), "bkn MPS to fixed": ("bkn_mps", "bkn_fixed"),
             "cha MPS to unfixed": ("cha_mps", "cha_unfixed"), "cha MPS to fixed": ("cha_mps", "cha_fixed"),
             "cha unfixed to fixed": ("cha_unfixed", "cha_fixed")}
    ctl = {}
    for name, (a, b) in pairs.items():
        own = (st[b]["share_over_0_2_s"] - st[a]["share_over_0_2_s"], st[b]["median_abs_s"] - st[a]["median_abs_s"])
        cw = d[a].keys() & d[b].keys()
        ca, cb = stats({s: d[a][s] for s in cw}), stats({s: d[b][s] for s in cw})
        com = (cb["share_over_0_2_s"] - ca["share_over_0_2_s"], cb["median_abs_s"] - ca["median_abs_s"])
        ctl[name] = {"own_windows": {"share": own[0], "median": own[1], "n": (st[a]["n"], st[b]["n"])},
                     "common_windows": {"share": com[0], "median": com[1], "n": len(cw)}}
        if expect:
            ws_, wm_ = expect["own_changes"][name]
            check_known(f"{name} own share change", round(own[0], 4), ws_, 1e-4, stop)
            check_known(f"{name} own median change", round(own[1], 2), wm_, 0.005, stop)
    cabs = {k: [abs(v["common_windows"][k]) for n, v in ctl.items() if n != "sac MPS to fixed"]
            for k in ("share", "median")}
    sac_c = ctl["sac MPS to fixed"]["common_windows"]
    out["controls"] = {"pairs": ctl, "reading_absolute_common": {
        k: within_controls(abs(sac_c[k]), cabs[k]) for k in ("share", "median")}}

    # M3 descriptions: the fits' stored offsets and grid-edge hits
    so_w, so_r = read_json(ws / "labels" / f"{SAC}_offsets.json"), read_json(ref / "labels" / f"{SAC}_offsets.json")
    diffs = [abs(so_w[s] - so_r[s]) for s in sorted(so_w.keys() & so_r.keys())
             if so_w[s] is not None and so_r[s] is not None]
    out["stored_offsets"] = {"n": len(diffs), "median_abs_s": float(np.median(diffs)) if diffs else None,
                             "share_over_0_2_s": float(np.mean([x > OVER for x in diffs])) if diffs else None}
    out["edge_hits"] = {k: {side: sum(1 for r in rec[k]["windows"].values()
                                      if r["frozen"][side] is not None and abs(r["frozen"][side]) >= EDGE_S)
                            for side in SIDES} for k in ("sac_mps", "sac_fixed")}

    # M4: qualifying frames per sidecar variant, one file at a time
    arm = ref / "builds" / f"resetfix_r23_detcpu__{SAC}"
    variants = {"fixed_v3": ref / "label_build", "mps_v3": ws / "label_build", "composite_r23": ref / "builds" / "r23",
                "r23_nofix": ref / "builds" / "r23_nofix", "mps_r23": ws / "builds" / "r23"}
    frames = {}
    for s in sorted(wins_f):
        first = first_logged_reset(read_json(arm / f"{s}_resetfix.json")) \
            if (arm / f"{s}_resetfix.json").exists() else None
        e = {"cpu_first_reset_frame": first}
        for v, folder in variants.items():
            p = folder / f"{s}_frames.json"
            if not p.exists():
                e[v] = None
                continue
            tm_root = ws if v.startswith("mps") else ref
            tm = read_json(tm_root / "label_build" / f"{s}_timemap.json")
            n, n_after = qualifying_frames(read_json(p), tm, first)
            e[v] = {"frames": n, "after_cpu_first_reset": n_after if first is not None else None}
        frames[s] = e
    out["qualifying_frames"] = frames

    # 6.7: equivalence evidence, and gsw_cha_2016's no-reset V3 label builds against the unfixed ones
    out["equivalence"] = {}
    for g in GAMES:
        e = read_json(ref / "resetfix" / f"equiv__{g}.json")
        out["equivalence"][g] = {"passes": e["passes"], "equivalence_windows": {
            w: c["byte_identical"] for w, c in e["equivalence"].items()},
            "reset_windows_prefix_equal": sum(1 for c in e["reset_windows"].values() if c["prefix"]["equal"]),
            "reset_windows": len(e["reset_windows"])}
    same, differ, missing = 0, [], 0
    for p in sorted((ref / "label_build").glob(f"{CHA}_s*_identity.json")):
        s = p.name[:-len("_identity.json")]
        if resets_of(p):
            continue
        old = ref / "superseded_nofix" / "label_build"
        if not (old / f"{s}_frames.json").exists():
            missing += 1
            continue
        ok = all(sha256(ref / "label_build" / f"{s}_{f}.json") == sha256(old / f"{s}_{f}.json")
                 for f in ("frames", "trajectories", "identity"))
        same += ok
        if not ok:
            differ.append(s)
    out["cha_no_reset_v3_vs_unfixed"] = {"identical": same, "differing": differ, "unfixed_missing": missing}

    if stop:                                          # a quoted fact or a summary differs: no reading is reported
        atomic_write(out_path(), json.dumps({"what": out["what"], "stopped": stop}, indent=1))
        raise SystemExit(f"{WHO}: stage 1 stops: {stop}")
    out["stopped"] = stop
    res = rnd(out)
    atomic_write(out_path(), json.dumps(res, indent=1))
    print(json.dumps({"decomposition": {k: v["reading"] for k, v in res["decomposition"].items()},
                      "side": res["side"].get("reading"), "controls": res["controls"]["reading_absolute_common"],
                      "clock_differs": res["clock"]["differs"], "classes": res["classes"]["counts"]}, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_sac_checkc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    a = ap.parse_args(argv)
    if a.cmd == "stage1":
        run_stage1(DATA / "work", DATA / "work_ref")


if __name__ == "__main__":
    main()
