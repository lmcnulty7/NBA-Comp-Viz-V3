"""tools/p8_rebaseline.py: the development standard rows (D193 (2)) of r23 on the fixed tracker, on the decided truth
set (D199), with the previous truths beside them (docs/research/P8_rebaseline.md, version 2). Report only.

    <frozen CV python> -m tools.p8_rebaseline score     inside the heavy-job lock ("p8 rebaseline score")
    python3 -m tools.p8_rebaseline report               -> reports/research/p8_rebaseline.json

score: tools.p8_frozen.install_paths on the reference root only; each truth through {**P, "labels": <root>/labels};
the pinned score_window through tools.p8_resetfix.score_pair (the build as both sides). Writes only the reference
root's rebaseline/ folder (score records, the common-frames labels) and cache entries in its cache/scorecard; SHA-256
manifests of the other roots' folders before and after stop the run on any difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_rebaseline"
LABEL = "development standard rows (D193 (2)); not the reported baseline (D198)"
LIMITS = ["shared input: each truth's frames were selected by fits on the same r23 label build's boxes, so the rows "
          "are optimistic in absolute terms and biased toward r23",
          "multipool v1 is a better fit by the acceptance measures, not a truth shown to be better (D199)"]
DATA = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
POOL = (CHA, BKN)
GAMES = (CHA, BKN, HOU)
ORDER = (BKN, CHA, HOU)                # gsw_bkn_2015 first: its previous truth is the reproduction gate (10.5)
CAP_S = 3600
BOOT_N, BOOT_SEED = 2000, 0
GATE_FIELDS = ("missed", "in_frame", "nr_ghosts", "nr_boxes", "matched", "far", "missed_1to1", "unmatched_1to1",
               "team", "cov_acc", "cov_live", "duplicates")


def roots() -> dict:
    return {"ws": DATA / "work", "ref": DATA / "work_ref", "ms": DATA / "work_ms"}


def truths(r: dict) -> dict:
    """game -> {truth name: root of its labels}; "current" is the decided truth (D199)."""
    return {CHA: {"current": r["ms"], "previous": r["ws"]}, BKN: {"previous": r["ref"], "current": r["ms"]},
            HOU: {"current": r["ref"]}}


def out_path() -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / "p8_rebaseline.json"


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


def manifest(root: Path, subdirs: tuple) -> dict:
    """{subdir/relative path: sha256} of every file under the given subfolders (a symlinked subfolder is recorded by
    its resolved name and not walked twice)."""
    out = {}
    for sd in subdirs:
        d = root / sd
        if d.is_symlink():
            out[f"{sd}/"] = f"link:{d.resolve().name}"
            continue
        if not d.exists():
            out[f"{sd}/"] = "absent"
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out[f"{sd}/{p.relative_to(d)}"] = sha256(p)
    return out


def labels_pin(root: Path, game: str) -> dict:
    files = sorted((root / "labels" / game).glob(f"{game}_s*.jsonl")) + \
        [root / "labels" / f"{game}_result.json", root / "labels" / f"{game}_offsets.json"]
    return {p.name: sha256(p) for p in files if p.exists()}


def timemap_content(tm: dict) -> tuple:
    return json.dumps(tm.get("frames")), json.dumps(tm.get("spans"))


# ── pure: rows and intervals ───────────────────────────────────────────────────────────────────────────────────────
RATIOS = {"missed": (("missed",), "in_frame"), "detection": (("far",), "in_frame"),
          "placement": (("missed", "-far"), "in_frame"), "ghosts_per_box": (("nr_ghosts",), "nr_boxes"),
          "ghosts_per_player": (("nr_ghosts",), "in_frame"), "net_error": (("missed", "nr_ghosts"), "in_frame"),
          "net_error_1to1": (("unmatched_1to1",), "in_frame"), "coverage": (("cov_acc",), "cov_live")}
MATCHED_ROWS = {"position_p50_ft": "position_p50_ft", "position_p90_ft": "position_p90_ft",
                "nearfield_bias_ft": "nearfield_bias_ft", "team": "team_accuracy"}


def num(w: dict, keys: tuple) -> float:
    return sum((-w[k[1:]] if k.startswith("-") else w[k]) for k in keys)


def ratio_values(wins: list, picks: np.ndarray | None = None) -> dict:
    """Point values of the ratio rows; with picks, their 2.5 and 97.5 percentiles over the resamples."""
    out = {}
    for name, (keys, den) in RATIOS.items():
        n = np.array([num(w, keys) for w in wins], float)
        d = np.array([w[den] for w in wins], float)
        pt = n.sum() / d.sum() if d.sum() else None
        iv = None
        if picks is not None and len(wins):
            sn, sd = n[picks].sum(1), d[picks].sum(1)
            v = sn[sd > 0] / sd[sd > 0]
            iv = [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)] if len(v) else None
        out[name] = {"value": pt, "interval": iv}
    return out


def matched_values(wins: list, picks: np.ndarray | None = None) -> dict:
    """Position p50 and p90, near-field bias and team of one build (both sides the same), with resampled intervals."""
    from tools.p8_e1_score import rows
    pair = [{"r23": w, "new": w} for w in wins]
    pt = rows(pair)
    out = {k: {"value": pt[r][0], "interval": None} for k, r in MATCHED_ROWS.items()}
    if picks is not None and len(wins):
        acc = {k: [] for k in MATCHED_ROWS}
        for pk in picks:
            rr = rows([pair[i] for i in pk])
            for k, r in MATCHED_ROWS.items():
                if rr[r][0] is not None:
                    acc[k].append(rr[r][0])
        for k, v in acc.items():
            out[k]["interval"] = [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)] \
                if v else None
    return out


def unit_rows(wins: list, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Section 4's rows of one unit (windows in sorted name order) with window-cluster intervals."""
    from tools.p8_resetfix import resample_picks
    picks = resample_picks(len(wins), n_boot, seed) if len(wins) >= 2 else None
    out = {**ratio_values(wins, picks), **matched_values(wins, picks)}
    out["counts"] = {"windows": len(wins), "in_frame": int(sum(w["in_frame"] for w in wins)),
                     "nr_boxes": int(sum(w["nr_boxes"] for w in wins))}
    return out


def change_rows(prev: list, curr: list, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Section 10.4: current minus previous on windows both truths score (paired lists, sorted), with a paired window
    bootstrap; the position, bias and team columns cover players matched under both truths."""
    from tools.p8_e1_score import boot, diffs
    from tools.p8_resetfix import ratio_interval, resample_picks
    pair = [{"r23": a, "new": b} for a, b in zip(prev, curr)]
    out = {}
    picks = resample_picks(len(pair), n_boot, seed) if len(pair) >= 2 else None
    for name, (keys, den) in RATIOS.items():
        bn = [num(a, keys) for a in prev]; bd = [a[den] for a in prev]
        nn = [num(b, keys) for b in curr]; nd = [b[den] for b in curr]
        pt = (sum(nn) / sum(nd) - sum(bn) / sum(bd)) if sum(nd) and sum(bd) else None
        out[name] = {"change": pt, "interval": ratio_interval(picks, bn, bd, nn, nd) if picks is not None else None}
    d = diffs(pair)
    wiv = boot({i: [p] for i, p in enumerate(pair)}, n=n_boot, seed=seed) if len(pair) >= 2 else {}
    for k, r in MATCHED_ROWS.items():
        out[k] = {"change": d[r], "interval": wiv.get(r),
                  "players": ("each truth's own matched players, on the windows both score" if k == "team"
                              else "matched under both truths")}
    return out


def gate_problems(mine: dict, arm: dict) -> list:
    """Section 10.5: per window, the eight fields equal to the reset-fix arm's record (windows[*].new)."""
    bad = []
    if set(mine) != set(arm):
        bad.append(f"window sets differ ({len(mine)} against {len(arm)})")
    for s in sorted(set(mine) & set(arm)):
        for f in GATE_FIELDS:
            if mine[s].get(f) != arm[s]["new"].get(f):
                bad.append(f"{s}: {f} differs")
                break
    return bad


# ── score (frozen CV python, inside the lock) ──────────────────────────────────────────────────────────────────────
def _git_clean_and_head() -> tuple:
    import subprocess
    from tools.p8_common import REPO
    st = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                        text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    return st == "", head


def cmd_score(args) -> dict:
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_pin
    from tools import p8_scorecard as S
    from tools.p8_common import atomic_write, guard, p8_root
    from tools.p8_resetfix import score_pair
    from tools.research_testset import refuse_cv_test
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = roots()
    if p8_root().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    refuse_cv_test(list(GAMES), WHO)
    guard(list(GAMES), WHO)
    clean, head = _git_clean_and_head()
    if not clean:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    pinned, now = read_json(p8_pin.PIN_FILE), p8_pin.current()
    bad_pin = [k for k in now if now[k] != pinned.get(k)]          # every field p8_pin.current() reports
    if bad_pin:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin ({bad_pin})")
    if not (r["ms"] / "label_build").is_symlink() or \
            (r["ms"] / "label_build").resolve() != (r["ref"] / "label_build").resolve():
        raise SystemExit(f"{WHO}: work_ms/label_build is not the reference root's label_build")
    watched = {"ms": manifest(r["ms"], ("labels", "label_build", "cache", "truth_accept")),
               "ws": manifest(r["ws"], ("labels", "label_build", "cache", "truth_accept")),
               "ref": manifest(r["ref"], ("labels", "label_build", "truth_accept"))}
    t0 = time.time()
    P = F.install_paths()                                   # the reference root only
    S.load_fixed_constants()
    refs = B.References()
    d = r["ref"] / "rebaseline"
    d.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t0))
    if (d / "score_run.json").exists():                    # an earlier run's record is moved aside, never deleted
        (d / "earlier").mkdir(exist_ok=True)
        (d / "score_run.json").rename(d / "earlier" / f"score_run__{int(t0)}.json")
    tr = truths(r)
    result = {"label": LABEL, "run_id": run_id, "head": head, "scorecard_pin": pinned.get("scorer"), "pairs": {}}

    def stop(msg):
        raise SystemExit(f"{WHO}: {msg}")

    def score_set(game, name, P_t, wins, ctx, comp):
        recs, cache_log, hashes = {}, {}, {}
        for w in wins:
            if time.time() - t0 > CAP_S:
                stop("the 60-minute cap was reached; nothing is reported")
            s = w["window"]
            side = read_json(comp / f"{s}_frames.json")
            tm = read_json(P_t["label_build"] / f"{s}_timemap.json")
            rows_t, _ = S.truth_rows(P_t, game, w, "camera", ctx["mom"].index, ctx["mirror"])
            cp = P_t["cache"] / "scorecard" / f"{s}.json"
            c = read_json(cp) if cp.exists() else {}
            live = {int(f) for f, _, _ in tm["frames"]}
            sidecar_cls = any("cls" in b for x in side["frames"] for b in x["boxes"])
            need_ref = set() if sidecar_cls else {int(x["frame"]) for x in rows_t}
            miss = {"wide": len(live - {int(k) for k in c.get("wide", {})}),
                    "referees": len(need_ref - {int(k) for k in c.get("referees", {})}),
                    "corr_absent": 0 if c.get("corr") else 1}
            hashes[s] = {k: sha256(p) for k, p in (("sidecar", comp / f"{s}_frames.json"),
                                                   ("clip", P_t["label_build"] / f"{s}.mp4"),
                                                   ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                                                   ("labels", P_t["labels"] / game / f"{s}.jsonl"),
                                                   ("identity", comp / f"{s}_identity.json"),
                                                   ("cache", cp))}
            tw = time.time()
            recs[s] = score_pair(P_t, game, w, {"r23": comp, "new": comp}, {"r23": side, "new": side}, ctx)["r23"]
            if any(miss.values()):
                cache_log[s] = {**miss, "window_seconds": round(time.time() - tw, 1)}
        return recs, cache_log, hashes

    for game in ORDER:
        g = F.game_of(game)
        guard([g["sportvu"]], WHO)
        mirror = F.mirror_of(P, game)
        for name, root in tr[game].items():
            if F.mirror_of(F.layout(root), game) != mirror:
                stop(f"{game}: the mirror differs in {root.name}")
            if sha256(F.layout(root)["sportvu"] / f"{g['sportvu']}_moments.json") != \
                    sha256(P["sportvu"] / f"{g['sportvu']}_moments.json"):
                stop(f"{game}: the moments file differs in {root.name}")
        mom = S.Moments(read_json(P["sportvu"] / f"{g['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": S.pbp_events(g["sportvu"]),
               "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        comp = P["builds"] / "r23"
        plan = read_json(r["ref"] / "resetfix" / f"plan__{game}.json")
        arm = P["builds"] / f"resetfix_r23_detcpu__{game}"
        wins = sorted((w for w in F.label_windows(P, game) if (comp / f"{w['window']}_frames.json").exists()),
                      key=lambda w: w["window"])
        guard([w["window"] for w in wins], WHO)
        for w in wins:                                     # the composite equals its planned source
            s = w["window"]
            src = arm if s in plan["reset_windows"] else P["builds"] / "r23_nofix"
            for f in ("frames", "trajectories", "identity"):
                if sha256(comp / f"{s}_{f}.json") != sha256(src / f"{s}_{f}.json"):
                    stop(f"{s}: the composite's {f} is not its planned source's")
                if game in POOL and sha256(r["ms"] / "builds" / "r23" / f"{s}_{f}.json") != sha256(comp / f"{s}_{f}.json"):
                    stop(f"{s}: work_ms's copy of builds/r23 differs")
        copied = 0
        for w in wins:                                     # caches: copy a missing file from the workspace (same clip)
            s = w["window"]
            dst, src_c = P["cache"] / "scorecard" / f"{s}.json", r["ws"] / "cache" / "scorecard" / f"{s}.json"
            if not dst.exists() and src_c.exists() and \
                    sha256(r["ws"] / "label_build" / f"{s}.mp4") == sha256(P["label_build"] / f"{s}.mp4"):
                dst.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(dst, src_c.read_text())
                copied += 1
        result.setdefault("cache_copied", {})[game] = copied
        for name, root in tr[game].items():
            acc = root / "truth_accept" / f"{game}.json"
            if name == "current" and (not acc.exists() or read_json(acc).get("use") != "use"):
                stop(f"{game}: its current truth does not read use")
            P_t = {**P, "labels": root / "labels"}
            for w in wins:                                 # clips and time maps: the reference root's, by content
                s = w["window"]
                if root == r["ws"]:
                    if sha256(r["ws"] / "label_build" / f"{s}.mp4") != sha256(P["label_build"] / f"{s}.mp4"):
                        stop(f"{s}: the workspace clip differs")
                    if timemap_content(read_json(r["ws"] / "label_build" / f"{s}_timemap.json")) != \
                            timemap_content(read_json(P["label_build"] / f"{s}_timemap.json")):
                        stop(f"{s}: the workspace time map differs in content")
            stems_l = {p.name[:-len(".jsonl")] for p in (root / "labels" / game).glob(f"{game}_s*.jsonl")}
            if stems_l != {w["window"] for w in wins}:
                stop(f"{game} {name}: the truth's label windows ({len(stems_l)}) are not the scored windows "
                     f"({len(wins)})")
            pin = labels_pin(root, game)
            if game == BKN and name == "previous":
                arm_pin = read_json(r["ref"] / "resetfix" / f"truth_pin__{game}.json")["files"]
                if pin != arm_pin:
                    stop("gsw_bkn_2015's previous truth is not the reset-fix arm's pin")
            recs, cache_log, hashes = score_set(game, name, P_t, wins, ctx, comp)
            if game == BKN and name == "previous":     # the reproduction gate, before any other number is kept
                bad = gate_problems(json.loads(json.dumps(recs)),
                                    read_json(r["ref"] / "resetfix" / f"score__{game}.json")["windows"])
                if bad:
                    stop(f"the reproduction gate fails: {bad[:3]}")
            rec = {"run_id": run_id, "game": game, "truth": name, "root": root.name, "labels_pin": pin, "windows": recs,
                   "cache": cache_log, "hashes": hashes}
            atomic_write(d / f"score__{game}__{name}.json", json.dumps(rec))
            result["pairs"][f"{game}__{name}"] = {"windows": len(recs), "cache_windows": len(cache_log)}
        if game in POOL:                                   # the common-frames split (10.4)
            common, frame_counts = {}, {}
            for w in wins:
                s = w["window"]
                fr = [set(int(json.loads(x)["frame"]) for x in (root / "labels" / game / f"{s}.jsonl").read_text()
                          .splitlines() if x.strip()) if (root / "labels" / game / f"{s}.jsonl").exists() else set()
                      for root in tr[game].values()]
                common[s] = fr[0] & fr[1]
                for k_, v_ in (("previous_only", len(fr[list(tr[game]).index("previous")] - fr[list(tr[game]).index("current")])),
                               ("current_only", len(fr[list(tr[game]).index("current")] - fr[list(tr[game]).index("previous")])),
                               ("both", len(common[s]))):
                    frame_counts[k_] = frame_counts.get(k_, 0) + v_
            for name, root in tr[game].items():
                lc = d / "labels_common" / name
                (lc / game).mkdir(parents=True, exist_ok=True)
                for extra in (f"{game}_offsets.json", f"{game}_result.json"):
                    if (root / "labels" / extra).exists():
                        atomic_write(lc / extra, (root / "labels" / extra).read_text())
                for w in wins:
                    s = w["window"]
                    src = root / "labels" / game / f"{s}.jsonl"
                    keep = [x for x in (src.read_text().splitlines() if src.exists() else [])
                            if x.strip() and int(json.loads(x)["frame"]) in common[s]]
                    atomic_write(lc / game / f"{s}.jsonl", "".join(x + "\n" for x in keep))
                recs, cache_log, _h = score_set(game, name, {**P, "labels": lc}, wins, ctx, comp)
                atomic_write(d / f"score__{game}__{name}__common.json",
                             json.dumps({"run_id": run_id, "game": game, "truth": name, "common_frames": True,
                                         "windows": recs, "cache": cache_log, "frame_counts": frame_counts}))
    after = {"ms": manifest(r["ms"], ("labels", "label_build", "cache", "truth_accept")),
             "ws": manifest(r["ws"], ("labels", "label_build", "cache", "truth_accept")),
             "ref": manifest(r["ref"], ("labels", "label_build", "truth_accept"))}
    if after != watched:
        stop("a watched folder changed during the run; nothing is reported")
    result["seconds"] = round(time.time() - t0, 1)
    atomic_write(d / "score_run.json", json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def cmd_report(args) -> dict:
    from export.schemas import find_private
    from tools.p8_common import atomic_write
    from tools.p8_placement import rnd
    d = roots()["ref"] / "rebaseline"
    run = read_json(d / "score_run.json")
    sc = {k: read_json(d / f"score__{k}.json") for k in run["pairs"]}
    for g in POOL:
        for n in ("previous", "current"):
            sc[f"{g}__{n}__common"] = read_json(d / f"score__{g}__{n}__common.json")
    stale = [k for k, v in sc.items() if v.get("run_id") != run["run_id"]]
    if stale:
        raise SystemExit(f"{WHO}: score files from another run: {stale}")
    wins = {k: [v["windows"][s] for s in sorted(v["windows"])] for k, v in sc.items()}
    out = {"label": LABEL, "limits": LIMITS, "run_id": run["run_id"], "head": run["head"], "seconds": run["seconds"],
           "cache_copied": run.get("cache_copied"), "per_game": {},
           "pooled_cha_bkn": None, "truth_change": {}, "common_frames": {}, "cache": {}}
    for k, w in wins.items():
        if k.endswith("__common"):
            continue
        out["per_game"][k] = unit_rows(w)
        out["cache"][k] = {"windows": len(sc[k]["cache"]),
                           "window_seconds": round(sum(c["window_seconds"] for c in sc[k]["cache"].values()), 1),
                           "missing": {f: sum(c[f] for c in sc[k]["cache"].values()) for f in ("wide", "referees",
                                                                                                "corr_absent")}}
    pool_w = {s: sc[f"{g}__current"]["windows"][s] for g in POOL for s in sc[f"{g}__current"]["windows"]}
    pooled = [pool_w[s] for s in sorted(pool_w)]                # sorted over all pooled stems (10.3)
    out["pooled_cha_bkn"] = unit_rows(pooled)
    for g in POOL:
        a, b = sc[f"{g}__previous"]["windows"], sc[f"{g}__current"]["windows"]
        both = sorted(a.keys() & b.keys())
        out["truth_change"][g] = {"what_changed": {CHA: "platform (Mac MPS to CPU label builds), the tracker bug's "
                                                        "thinning of post-reset truth frames, and the multipool fit",
                                                   BKN: "the multipool fit only (same CPU label builds)"}[g],
                                  "windows_both": len(both), "windows_previous_only": len(a.keys() - b.keys()),
                                  "windows_current_only": len(b.keys() - a.keys()),
                                  **change_rows([a[s] for s in both], [b[s] for s in both])}
        ca, cb = (sc[f"{g}__{n}__common"]["windows"] for n in ("previous", "current"))
        cboth = sorted(ca.keys() & cb.keys())
        out["common_frames"][g] = {"windows": len(cboth), "frame_counts": sc[f"{g}__current__common"]["frame_counts"],
                                   **change_rows([ca[s] for s in cboth], [cb[s] for s in cboth])}
    res = rnd(out)
    text = json.dumps(res, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data: {find_private(text)[:3]}")
    atomic_write(out_path(), text)
    print(json.dumps({k: {r: v[r]["value"] for r in ("missed", "net_error", "ghosts_per_box")}
                      for k, v in out["per_game"].items()}, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_rebaseline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
