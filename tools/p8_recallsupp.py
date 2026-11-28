"""tools/p8_recallsupp.py: D208's 32 dropped windows read the build's way (docs/research/P8_recallcause_supp.md,
version 1). Report only: nothing is fitted, tuned or adopted; D208's capture and classes, with the build's read path in
place of the a3run read.

    <frozen CV python> -m tools.p8_recallsupp run   inside the heavy-job lock ("p8 recallsupp"; V3_P8_ROOT = the
                                                    reference root; HF_HUB_OFFLINE=1, PYTORCH_ENABLE_MPS_FALLBACK=1)
                                                    -> reports/research/p8_recallcause_supp.json, raw candidates in the
                                                    reference root's recallsupp/

The read functions SeekCount, pregate_intervals, main_reads, open_like_build and build_read are copied from
tools/p8_posefoot.py at 53c44192 (branch r-p8-posefoot, P8_posefoot.md 14.2), which copied the read path from
build_trajectories.py; the lead chose the copy since posefoot has not reached v4-research. A test checks that they
equal posefoot's at that commit, docstrings aside.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

WHO = "tools.p8_recallsupp"
POSEFOOT_COMMIT = "53c44192"
COPIED = ("SeekCount", "pregate_intervals", "main_reads", "open_like_build", "build_read")
CHECK_WINDOWS = ("gsw_cha_2016_s05_w00", "gsw_bkn_2015_s02_w01")      # G1: kept windows, from frame 0 and a seek to 300
ENV_NEEDED = {"HF_HUB_OFFLINE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
CAP_S = 40 * 60
RC_SHA = "794f808ab905252e569a67285aec06528e69a6fd10bbb0228eb2b81de2637a9b"   # reports/research/p8_recallcause.json
ALLOWED = ("gsw_cha_2016", "gsw_bkn_2015")
UNIT_NAMES = ("P-i", "D-i", "both")
SCOPES = ("gsw_cha_2016", "gsw_bkn_2015", "pooled")


# ── copied from tools/p8_posefoot.py at 53c44192 (credit: P8_posefoot.md 14.2; itself from build_trajectories.py) ──
class SeekCount:
    """videoseq.SeqReader's seek rule, counted: a read seeks when the index is behind the decoder or more than
    SEEK_MIN_GAP ahead (videoseq.py)."""
    def __init__(self, reader):
        from videoseq import SEEK_MIN_GAP
        self.reader, self.gap, self.seeks = reader, SEEK_MIN_GAP, []

    def read(self, idx: int):
        if idx < self.reader.next or idx - self.reader.next > self.gap:
            self.seeks.append(int(idx))
        return self.reader.read(idx)


def pregate_intervals(reader, total: int, pg_stride: int, pg_pad: int, is_live, start: int = 0) -> list:
    """build_trajectories.py's pre-gate pass and interval merge, copied (lines 180-197): every pg_stride frame from
    start through reader.read, a live hit when is_live(frame), each hit padded by pg_pad and merged."""
    hits, i = [], start
    while i < total:
        ret, frame = reader.read(i)
        if not ret:
            break
        if is_live(frame):
            hits.append(i)
        i += pg_stride
    intervals = []
    for h in hits:
        a, b = h - pg_pad, h + pg_pad
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = b
        else:
            intervals.append([a, b])
    return intervals


def main_reads(reader, total: int, intervals: list, stride: int, accepted: set, max_frames: int, need: set, predict,
               start: int = 0) -> tuple:
    """build_trajectories.py's tracking loop read path, copied (lines 210-224 and the idx steps): jump to each
    interval's start, read with the same reader, step by stride; a frame counts towards max_frames when the build
    accepted it (it has a sidecar row), which stands in for the gate's per-frame decision. predict(img) on frames in
    need. -> ({frame: prediction}, [frames read])."""
    idx, n, seg_i, out, reads = start, 0, 0, {}, []
    while idx < total and n < max_frames:
        if intervals is not None:
            while seg_i < len(intervals) and idx > intervals[seg_i][1]:
                seg_i += 1
            if seg_i >= len(intervals):
                break
            if idx < intervals[seg_i][0]:
                idx = intervals[seg_i][0]
        ret, frame = reader.read(idx)
        if not ret:
            break
        reads.append(int(idx))
        if idx in need:
            out[idx] = predict(frame)
        if idx in accepted:
            n += 1
        idx += stride
    return out, reads


def open_like_build(clip):
    """build_trajectories.py line 172: cv2.CAP_FFMPEG first, then the default backend."""
    import cv2
    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(clip))
    return cap


def build_read(clip, side: dict, max_frames: int, is_live, need: set, predict) -> dict:
    """The build's own read path (v1.3, 14.2): open as the build opens, the pre-gate pass with the build's gate, the
    intervals, then the tracking loop's reads through one videoseq.SeqReader. Validation: every sidecar frame is read,
    and the frames the loop counts as accepted equal the sidecar's. -> preds, reads, seeks, backend, validation."""
    import config
    from videoseq import SeqReader
    cap = open_like_build(clip)
    try:
        import cv2
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reader = SeekCount(SeqReader(cap))
        pg_stride, pg_pad = config.pregate_params(fps)
        intervals = pregate_intervals(reader, total, pg_stride, pg_pad, is_live)
        pre_seeks = list(reader.seeks)
        stride = int(side.get("stride") or 1)
        accepted = {int(r["frame"]) for r in side["frames"]}
        preds, reads = main_reads(reader, total, intervals, stride, accepted, max_frames, need, predict)
        rs = set(reads)
        valid = accepted <= rs and len(rs & accepted) == len(accepted)
        return {"preds": preds, "n_reads": len(reads), "intervals": intervals, "backend": backend,
                "seeks_pregate": pre_seeks, "seeks_main": reader.seeks[len(pre_seeks):],
                "validation": {"sidecar_frames_read": len(rs & accepted), "sidecar_frames": len(accepted),
                               "passes": bool(valid)}}
    finally:
        cap.release()


# ── pure: D208's per-miss classification, as one function (its cmd_run loop body, P8_recallcause.md 8.3 to 8.10) ────
def ids_of(pre: list, post: list) -> tuple:
    """(pre_ids, post_ids) by exact equality: R_post's boxes are R_pre's (the same anchors), so a box's index in R_pre
    stands for its anchor (the anchors were not stored)."""
    where = {}
    for i, b in enumerate(pre):
        where.setdefault(tuple(b), []).append(i)
    post_ids = []
    for b in post:
        ids = where.get(tuple(b))
        if not ids:
            raise SystemExit(f"{WHO}: an R_post box is not in R_pre")
        post_ids.append(ids.pop(0))
    return list(range(len(pre))), post_ids


def classify_window(game: str, s: str, uni: list, rows: list, res: dict, fps: float) -> list:
    """D208's classification of every universe miss of one window from its raw candidates (res: {frame: {post, pre,
    other, low}}), line for line as tools/p8_recallcause.py cmd_run does it -> D208's miss records."""
    from tools import p8_one2one as O
    from tools import p8_recallcause as RC
    out = []
    for x in uni:
        f, j, sv, Ht = x["frame"], x["player"], x["sv"], x["Ht"]
        rr = res[f]
        post, pre = rr["post"], rr["pre"]
        pre_ids, post_ids = ids_of(pre, post)
        post_feet, pre_feet = RC.feet_of(post, Ht), RC.feet_of(pre, Ht)
        ref_boxes = [b for b in rr["other"] if b[5] == RC.REFEREE_CLASS]
        ref_feet = RC.feet_of(ref_boxes, Ht)
        near_other = {RC.OTHER_NAMES[c]: sum(1 for b, fo in zip(rr["other"], RC.feet_of(rr["other"], Ht))
                                             if b[5] == c and float(np.hypot(*(fo - sv[j]))) <= RC.NEAR_FT)
                      for c in RC.OTHER_NAMES}
        main, claimed = O.two_stage(post_feet, sv, range(len(sv)), range(len(post)), O._assign_all)
        own = main.get(j, (None,))[0]
        supp = RC.suppressors(pre, post, pre_ids, post_ids)
        pos = {a: k for k, a in enumerate(post_ids)}
        pre_owner = []
        for i, a in enumerate(pre_ids):
            k = pos.get(a, supp.get(i))
            pre_owner.append(claimed.get(k) if k is not None else None)
        c2, cidx = RC.is_c2(j, sv, pre_feet, post_feet, supp) if own is None else (False, None)
        side_m, side_ft, late = None, None, None
        if own is not None:
            mm = RC.match_boxes([post[own][:4]], x["side_boxes"])
            side_m = mm.get(0)
            if side_m is not None:
                side_ft = x["side_feet"][side_m]
            else:
                later = [y for y in rows if y["pid"] == x["pid"] and f < y["frame"] <= f + RC.LATE_S * float(fps)]
                if later:
                    late = any(len(y["side_feet"]) and float(np.hypot(*(y["side_feet"] - y["sv"][y["player"]]).T)
                                                             .min()) <= RC.MISS_FT for y in later)
        cls = RC.classify(j, sv, x["px"][j], own, pre, pre_feet, ref_feet, c2, side_m, side_ft, late, pre_owner)
        lowshare = cls == "C1c" and any(
            float(np.hypot(*(np.asarray(fl) - sv[j]))) <= RC.MISS_FT and RC.nearest_player(fl, sv) == j
            for fl in RC.feet_of(rr["low"], Ht))
        out.append({"game": game, "window": s, "frame": f, "player": j, "pid": x["pid"], "d205": x["rec"]["class"],
                    "class": cls, "low_conf_box": bool(lowshare), "other_classes_within_5ft": near_other,
                    "c2_score": (pre[cidx][4] if c2 else None),
                    "c2_iou": (RC.iou(pre[cidx], post[supp[cidx]]) if c2 else None)})
    return out


def unit_tables(misses: list) -> dict:
    """D208's unit tables (P-i, D-i and both; per game and pooled), as tools/p8_recallcause.py builds them."""
    from tools import p8_recallcause as RC
    units = {}
    for name, sel in (("P-i", lambda m: m["d205"] == "P-i"), ("D-i", lambda m: m["d205"] == "D-i"),
                      ("both", lambda m: True)):
        for scope in SCOPES:
            mm = [m for m in misses if sel(m) and (scope == "pooled" or m["game"] == scope)]
            t = RC.share_table([(m["window"], m["class"]) for m in mm])
            t["groups"] = RC.groups_of([(m["window"], m["class"]) for m in mm])
            t["other_is_a_code_fault"] = t["classes"]["other"]["count"] > 0
            t["other_classes_within_5ft"] = {n: sum(m["other_classes_within_5ft"][n] for m in mm)
                                             for n in RC.OTHER_NAMES.values()}
            t["low_conf_share_of_C1c"] = (sum(m["low_conf_box"] for m in mm if m["class"] == "C1c") /
                                          max(1, sum(m["class"] == "C1c" for m in mm)))
            if scope == "pooled":
                t["reading"] = RC.main_loss(t["groups"])
            units[f"{name}__{scope}"] = t
    return units


def counts_differ(mine: dict, d208: dict) -> list:
    """G3: the units and classes whose miss counts differ from D208's recorded tables."""
    from tools.p8_recallcause import CLASSES
    bad = []
    for u in (f"{n}__{s}" for n in UNIT_NAMES for s in SCOPES):
        for c in CLASSES:
            if mine[u]["classes"][c]["count"] != d208[u]["classes"][c]["count"]:
                bad.append(f"{u} {c}")
    return bad


def same_candidates(fresh: dict, stored: dict) -> bool:
    """G1: R_pre and R_post equal the stored ones exactly (after a JSON round trip of the fresh values)."""
    norm = lambda v: json.loads(json.dumps(v))
    return all(norm(fresh[k]) == stored[k] for k in ("pre", "post"))


def allowed(names) -> list:
    """The allow-list {gsw_cha_2016, gsw_bkn_2015} and the P8 guards on every game and window name."""
    from tools.p8_common import guard
    from tools.research_testset import refuse_cv_test
    names = [str(n) for n in names]
    bad = [n for n in names if not any(n == g or n.startswith(g + "_") for g in ALLOWED)]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not in this item's allow-list {ALLOWED}")
    refuse_cv_test(names, WHO)
    guard(names, WHO)
    return names


def out_path() -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / "p8_recallcause_supp.json"


# ── run (frozen CV python, inside the lock) ────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> dict:
    import subprocess
    from sportvu import bench as B
    from sportvu.rebuild import build_stride
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
    from tools import p8_scorecard as S
    from tools.p8_common import REPO, atomic_write, guard, main_checkout, p8_root
    from export.schemas import find_private
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    bad_env = {k: os.environ.get(k) for k, v in ENV_NEEDED.items() if os.environ.get(k) != v}
    if bad_env:
        raise SystemExit(f"{WHO}: set {ENV_NEEDED} before the run (got {bad_env})")
    r = RB.roots()
    if p8_root().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    st = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                        text=True, check=True).stdout.strip()
    if st:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    pinned, now = RC.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if [k for k in now if now[k] != pinned.get(k)]:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    if RC.sha256(weights) != RC.R23_SHA:
        raise SystemExit(f"{WHO}: the detector is not r23")
    if RC.sha256(RC.out_path()) != RC_SHA:
        raise SystemExit(f"{WHO}: reports/research/p8_recallcause.json is not D208's record")
    one2one = {g: r["ref"] / "one2one" / f"{g}.json" for g in RC.POOL}
    if {g: RC.sha256(p) for g, p in one2one.items()} != RC.D205_SHA:
        raise SystemExit(f"{WHO}: D205's records are not the pinned ones")
    d205 = {g: RC.read_json(p) for g, p in one2one.items()}
    d202 = {g: RC.read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json") for g in RC.POOL}
    if any(d205[g].get("run_id") != RC.RUN_ID or d202[g].get("run_id") != RC.RUN_ID for g in RC.POOL):
        raise SystemExit(f"{WHO}: D205's or D202's records are not from run {RC.RUN_ID}")
    rc = RC.read_json(RC.out_path())
    dropped = sorted(rc["windows_dropped"])
    kept = sorted(s for s in rc["alignment"] if s not in rc["windows_dropped"])
    allowed(dropped + kept + list(CHECK_WINDOWS))
    if any(s not in kept for s in CHECK_WINDOWS) or len(dropped) != 32 or len(kept) != 88:
        raise SystemExit(f"{WHO}: the window sets are not D208's")
    watch = lambda: {k: RB.manifest(r[k], ("labels", "label_build", "truth_accept", "cache")) for k in ("ref", "ms")} | \
        {"builds_r23": RB.manifest(r["ref"], ("builds/r23",)), "recallcause": RB.manifest(r["ref"], ("recallcause",))}
    before = watch()
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    comp = P["builds"] / "r23"
    raw = RC.Raw(weights)
    import cv2
    import torch
    import ultralytics
    import config
    versions = {"opencv": cv2.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    if versions != RC.VERSIONS:
        raise SystemExit(f"{WHO}: versions {versions} are not {RC.VERSIONS}")
    from gate.backbones import get_backbone
    from gate.trained_head import TrainedHeadGate
    mc = main_checkout() / "models"
    head_p, thr = mc / config.HEAD_V2_PATH.name, json.loads((mc / config.THRESHOLDS_PATH.name).read_text())["v2"]
    gate_rec = {"head": head_p.name, "head_sha256": RB.sha256(head_p),
                "coefs_sha256": RB.sha256(head_p.with_name(head_p.stem + "_coefs.npz")), "threshold_v2": thr,
                "clip_backbone": config.CLIP_MODEL_NAME, "env": dict(ENV_NEEDED)}
    gates = {}

    def gate_for(dev):
        if dev not in gates:
            gates[dev] = TrainedHeadGate.load(head_p, backbone=get_backbone("clip", dev), threshold=thr)
        return gates[dev]

    def cap():
        if time.time() - t0 > CAP_S:
            raise SystemExit(f"{WHO}: the 40-minute cap was reached; nothing is reported")

    def read_build(w, side, need):
        s = w["window"]
        dev = ((RC.read_json(comp / f"{s}_device.json") if (comp / f"{s}_device.json").exists() else {})
               .get("resolved") or {}).get("gate") or "cpu"
        stride = int(side.get("stride") or 1)
        if build_stride(w["fps"]) != stride:
            raise SystemExit(f"{WHO}: {s}: the sidecar stride is not the build's")
        max_frames = (w["f_end"] - w["f_start"]) // stride + 2
        br = build_read(P["label_build"] / f"{s}.mp4", side, max_frames, gate_for(dev).is_court_visible, set(need),
                        raw.run)
        return br, dev

    wins_all = {g: sorted((w for w in F.label_windows(P, g) if (comp / f"{w['window']}_frames.json").exists()),
                          key=lambda w: w["window"]) for g in RC.POOL}
    by_name = {w["window"]: (g, wi, w) for g in RC.POOL for wi, w in enumerate(wins_all[g])}
    for g in RC.POOL:
        if sorted(d202[g]["windows"]) != [w["window"] for w in wins_all[g]]:
            raise SystemExit(f"{WHO}: {g}: the windows differ from D202's record")
    raw_old = RC.raw_dir(r["ref"])
    # G1: the read check on two kept windows
    g1 = {}
    for s in CHECK_WINDOWS:
        g, wi, w = by_name[s]
        stored = RC.read_json(raw_old / f"{s}.json")
        side = RC.read_json(comp / f"{s}_frames.json")
        br, dev = read_build(w, side, [int(f) for f in stored])
        bad = [int(f) for f in sorted(stored, key=int) if int(f) not in br["preds"] or
               not same_candidates(br["preds"][int(f)], stored[f])]
        g1[s] = {"frames": len(stored), "differing": len(bad), "first_differing": bad[:5], "gate_device": dev,
                 "backend": br["backend"], "seeks_main": br["seeks_main"], "read_validation": br["validation"]}
        if bad or not br["validation"]["passes"]:
            raise SystemExit(f"{WHO}: G1 fails on {s} ({len(bad)} of {len(stored)} frames differ); nothing is reported")
    # per game: contexts, then the 32 (new reads) and the 88 (G3, stored candidates)
    new_misses, old_misses, per_window = [], [], {}
    sup_d = r["ref"] / "recallsupp"
    sup_d.mkdir(exist_ok=True)
    for g in RC.POOL:
        gd = F.game_of(g)
        guard([gd["sportvu"], gd["video_id"]], WHO)
        mirror = F.mirror_of(P, g)
        mom = S.Moments(RC.read_json(P["sportvu"] / f"{gd['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": S.pbp_events(gd["sportvu"]),
               "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        P_t = {**P, "labels": RB.truths(r)[g]["current"] / "labels"}
        for s in [x for x in dropped + kept if x.startswith(g + "_")]:
            cap()
            _, wi, w = by_name[s]
            rows = RC.d205_rows(g, P_t, w, comp, ctx, d202[g]["windows"][s], (d202[g].get("hashes") or {}).get(s, {}))
            mine = [[wi, {k: x["rec"][k] for k in ("class", "single", "inframe")}] for x in rows]
            want = [[a, {k: b[k] for k in ("class", "single", "inframe")}] for a, b in d205[g]["rows"] if a == wi]
            if mine != want:
                raise SystemExit(f"{WHO}: the D205 row gate fails on {s}; nothing is reported")
            uni = [x for x in rows if x["rec"]["class"] in RC.UNIVERSE]
            side = RC.read_json(comp / f"{s}_frames.json")
            if s in kept:                                  # G3: D208's classes from its stored candidates
                stored = RC.read_json(raw_old / f"{s}.json")
                old_misses += classify_window(g, s, uni, rows, {int(f): v for f, v in stored.items()}, side["fps"])
                continue
            uframes = sorted({x["frame"] for x in uni})
            stride = int(side["stride"])
            fset = {int(x["frame"]) for x in side["frames"]}
            al = [f for f in uframes if f - stride in fset and f + stride in fset][:RC.ALIGN_FRAMES]
            br, dev = read_build(w, side, uframes)
            rec = {"misses": len(uni), "gate_device": dev, "backend": br["backend"], "seeks_main": br["seeks_main"],
                   "seeks_pregate": len(br["seeks_pregate"]), "read_validation": br["validation"]}
            if not br["validation"]["passes"] or set(uframes) - set(br["preds"]):
                per_window[s] = {**rec, "status": "undetermined: the build's read path is not reproduced"}
                continue
            res = br["preds"]
            srow = {int(x["frame"]): x for x in side["frames"]}
            shares = [RC.align_shares(res[f]["post"], [b["bbox"] for b in srow[f - stride]["boxes"]],
                                      [b["bbox"] for b in srow[f]["boxes"]], [b["bbox"] for b in srow[f + stride]["boxes"]])
                      for f in al]
            rec.update({"d208_alignment_on_build_read": {"frames": len(al), "shares": shares,
                                                          "aligned": RC.alignment_ok(shares)},
                        "status": "classified"})
            atomic_write(sup_d / f"{s}.json", json.dumps({str(f): {k: v[k] for k in ("post", "pre", "other", "low")}
                                                          for f, v in res.items()}))
            new_misses += classify_window(g, s, uni, rows, res, side["fps"])
            per_window[s] = rec
    # G3, then the tables
    old_units = unit_tables(old_misses)
    bad = counts_differ(old_units, rc["units"])
    if bad:
        raise SystemExit(f"{WHO}: G3 fails: the recomputed 88-window counts differ from D208's in {bad[:5]}")
    if any(m["class"] == "other" for m in new_misses):
        raise SystemExit(f"{WHO}: a miss of the 32 reads 'other' (expected 0); a code fault")
    if watch() != before:
        raise SystemExit(f"{WHO}: a watched folder changed; nothing is reported")
    out = {"what": "P8 recall-cause supplement: D208's 32 dropped windows read the build's way "
                   "(docs/research/P8_recallcause_supp.md); report only; beside P8_recallcause.md 9.3", "head": head,
           "versions": versions, "nms_backend": raw.backend, "gate": gate_rec, "g1": g1, "g3": "passed",
           "windows": per_window,
           "units_32": unit_tables(new_misses),
           "units_120": unit_tables(old_misses + new_misses),
           "counts": {"misses_32": len(new_misses), "misses_88": len(old_misses),
                      "undetermined": sorted(s for s, v in per_window.items() if v["status"] != "classified")},
           "seconds": round(time.time() - t0, 1)}
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(out_path(), text)
    print(json.dumps({u: out["units_120"][f"{u}__pooled"]["reading"] for u in UNIT_NAMES}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    args = ap.parse_args(argv)
    {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
