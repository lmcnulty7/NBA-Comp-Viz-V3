"""tools/p8_multistart_stage2.py: stage 2 of the multi-pool camera fit (docs/research/P8_multistart_fit.md, sections 7, 8
and 15). Written, not run; it runs only after its review and the lead's word.

    <frozen CV python> -m tools.p8_multistart_stage2 multipool GAME   (V3_P8_ROOT = work_ref; lock
        "p8 multistart stage2 GAME multipool")
    <frozen CV python> -m tools.p8_multistart_stage2 truth hou_cle_2016    (V3_P8_ROOT = work_ms; lock "... truth")
    <frozen CV python> -m tools.p8_multistart_stage2 accept hou_cle_2016   (V3_P8_ROOT = work_ms; lock "... accept")

Games: hou_cle_2016 (multipool, stability, truth step, acceptance; acceptance stays pending while the reset-fix arm has
no check C composite for it), gsw_sac_2015 (multipool and stability only, report: it stays apart by check C),
gsw_ind_2016 (multipool and stability only, report only, on its MPS label build in the P8 workspace, read only; never a
truth, D98). Stage 2 runs only if gsw_cha_2016's multipool record shows kill rule 4's "stage 2 runs" and stability,
and gsw_bkn_2015's multipool record shows stability (kill rule 3).

The procedure is stage 1's (tools/p8_multistart.py: the same seed sets, chunks, kill rule 1 after chunk 1, J, every
run competing, selection, stability at unfitted feet, stale-run refusal), run by one core here that takes the label
build's root as a parameter; tools/p8_multistart.py's stage 1 functions are not changed. The reference root's protected
files (and, for gsw_ind_2016, the P8 workspace's files for that game) are hashed before and after every step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools import p8_multistart as M
from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_multistart_stage2"
STAGE2 = {"hou_cle_2016": {"label_root": "ref", "truth": True},
          "gsw_sac_2015": {"label_root": "ref", "truth": False},
          "gsw_ind_2016": {"label_root": "work", "truth": False}}


def gate_stage2(ms: Path) -> None:
    """Stage 2 runs only if gsw_cha_2016 passed kill rule 4 and stability, and gsw_bkn_2015 is stable (kill rule 3)."""
    cha, bkn = (ms / "multistart" / g / "multipool.json" for g in ("gsw_cha_2016", "gsw_bkn_2015"))
    if not cha.exists() or not bkn.exists():
        raise SystemExit(f"{WHO}: stage 2 waits for gsw_cha_2016's and gsw_bkn_2015's multipool records")
    c, b = json.loads(cha.read_text()), json.loads(bkn.read_text())
    if not (c.get("kill4") or {}).get("stage2_runs") or not c["stability"]["passes"]:
        raise SystemExit(f"{WHO}: stage 2 does not run: gsw_cha_2016 failed kill rule 4 or stability")
    if not b["stability"]["passes"]:
        raise SystemExit(f"{WHO}: stage 2 does not run: kill rule 3, gsw_bkn_2015 is not stable")


def workspace_manifest(work: Path, game: str) -> dict:
    """SHA-256 of the P8 workspace's files for one game (labels and label sidecars), read only for gsw_ind_2016."""
    files = [*sorted((work / "labels").glob(f"{game}*")), *sorted((work / "labels" / game).glob("*")),
             *sorted((work / "label_build").glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))]
    out = {}
    for p in files:
        if p.is_file():
            out[str(p.relative_to(work))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _check(game: str):
    if game not in STAGE2:
        raise SystemExit(f"{WHO}: {game} is not a stage 2 game")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    return M.roots_ms()


def multipool(game: str) -> dict:
    from sportvu import camera as cam
    from sportvu.sync import SportVUIndex
    from tools import p8_frozen as F
    from tools.p8_cha_fit import inits_of
    work, ref, ms = _check(game)
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ref.resolve():
        raise SystemExit(f"{WHO}: run multipool with V3_P8_ROOT set to the reference root")
    gate_stage2(ms)
    src = work if STAGE2[game]["label_root"] == "work" else ref
    before, before_w, t0 = M.manifest(ref), workspace_manifest(work, game), time.time()
    P = F.install_paths(src)                 # read only: install_paths only creates folders that already exist there
    g = F.game_of(game)
    mirror = F.mirror_of(P, game)
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    secs = [s for s in F.sections(P, game) if (P["sync"] / f"{s}_local_timemap.json").exists()]
    wins = [w for s in secs for w in F.label_windows(P, game, s)]
    refuse_cv_test([w["window"] for w in wins], WHO)
    guard([w["window"] for w in wins], WHO)
    frames = M.build_inputs(P, wins, index, mirror, src / "label_build")
    inits = inits_of(frames)
    init_frames = [fr for fr, _ in inits]
    C0 = np.median(np.array([p[2] for _, p in inits]), axis=0)
    d = ms / "multistart" / game
    d.mkdir(parents=True, exist_ok=True)
    fh = M.keys_hash(frames)
    runs = {}
    for set_name, seeds in M.SEED_SETS.items():
        for seed in seeds:
            rp = d / f"seed_{seed:02d}.json"
            if rp.exists():
                r = json.loads(rp.read_text())
                stale = M.stale_run(r, fh, C0, len(inits))
                if stale:
                    raise SystemExit(f"{WHO}: {rp.name} is stale ({stale}); move it aside before resuming")
                runs[seed] = r
                continue
            t = time.time()
            fz = cam.fit_game(frames, seed=seed)
            pool = M.pool_of(inits, seed)
            resid, _ = M.problem(pool)
            x0 = M.start_vector(pool, C0)
            first, x1 = M.run_chunks(pool, x0, 1)
            diff = float(np.linalg.norm(np.asarray(first[0]["C"]) - np.asarray(fz["C"])))
            if diff > M.REPRO_FT:                    # kill rule 1, before the other 19 chunks
                raise SystemExit(f"{WHO}: kill rule 1, {game} seed {seed}: first chunk {diff:.4f} ft from fit_game")
            rest, x = M.run_chunks(pool, x1, M.CHUNKS - 1)
            chunks = first + rest
            jr = M.common_objective(x[:3], init_frames)
            r = {"seed": seed, "set": set_name, "frames_hash": fh, "C": x[:3].tolist(), "C0": C0.tolist(), "J": jr["J"],
                 "fit_frame_failed": jr["fit_frame_failed"], "report_only_all_inits_median_px": jr["report_only_all_inits_median_px"],
                 "pooled_median_px": chunks[-1]["pooled_median_px"],
                 "pooled_median_px_at_start": round(float(np.median(np.abs(resid(x0)))), 2),
                 "cost": chunks[-1]["cost"], "converged_at_chunk": M.converged_at(chunks), "first_chunk_vs_fit_game_ft": diff,
                 "inits": len(inits), "pooled_frames": len(pool), "chunks": chunks, "seconds": round(time.time() - t, 1)}
            atomic_write(rp, json.dumps(r, indent=1))
            runs[seed] = r
    bad_j = [s for s, r in runs.items() if not np.isfinite(r["J"]) or r["fit_frame_failed"]]
    if bad_j:
        raise SystemExit(f"{WHO}: invalid J (non-finite or fit_frame failures) for seeds {bad_j}")
    sel = {name: M.select([runs[s] for s in seeds]) for name, seeds in M.SEED_SETS.items()}
    sidecars = {}
    for w in wins:
        sp = src / "label_build" / (w["window"] + "_frames.json")
        if sp.exists():
            sidecars[w["window"]] = {int(r["frame"]): r["boxes"] for r in json.loads(sp.read_text())["frames"]}
    counts = {name: {} for name in sel}
    maps = {name: M.foot_maps(r["C"], init_frames, M.per_frame(r["C"], init_frames), sidecars, counts[name])
            for name, r in sel.items()}
    stab = M.stability(maps)
    stab["feet_per_selection"] = counts
    frozen_p = (work if game == "gsw_ind_2016" else ref) / "labels" / f"{game}_result.json"
    frozen_C = json.loads(frozen_p.read_text())["camera"]["C"] if frozen_p.exists() else None
    out = {"game": game, "design": "docs/research/P8_multistart_fit.md sections 7, 8 and 15 (stage 2)",
           "label_build": "P8 workspace (MPS, unfixed), read only; report only" if game == "gsw_ind_2016"
           else "reference root (reset-fixed, CPU detector)",
           "truth_step": STAGE2[game]["truth"], "inits": len(inits), "frames_hash": fh, "C0": C0.tolist(),
           "selected": {name: {k: r[k] for k in ("seed", "C", "J", "pooled_median_px", "pooled_median_px_at_start", "cost",
                                                 "converged_at_chunk", "report_only_all_inits_median_px", "pooled_frames",
                                                 "inits")} for name, r in sel.items()},
           "runs": {s: {k: r[k] for k in ("set", "C", "J", "pooled_median_px", "converged_at_chunk", "seconds")}
                    for s, r in sorted(runs.items())},
           "report_only_converged_per_set": {name: sum(1 for s in seeds if runs[s]["converged_at_chunk"])
                                             for name, seeds in M.SEED_SETS.items()},
           "stability": stab,
           "report_only_pairwise_C_ft": {f"{a}|{b}": round(float(np.linalg.norm(np.asarray(sel[a]["C"]) - np.asarray(sel[b]["C"]))), 3)
                                         for i, a in enumerate(sorted(sel)) for b in sorted(sel)[i + 1:]},
           "report_only_distance_from_frozen_C_ft": round(float(np.linalg.norm(np.asarray(sel["truth"]["C"]) - np.asarray(frozen_C))), 3)
           if frozen_C else None}
    after, after_w = M.manifest(ref), workspace_manifest(work, game)
    out["guard_manifest"] = {"files": len(before), "unchanged": before == after,
                             "workspace_files": len(before_w), "workspace_unchanged": before_w == after_w}
    out["seconds"] = round(time.time() - t0, 1)
    text = json.dumps(out, indent=1, default=str)
    atomic_write(d / "multipool.json", text)
    atomic_write(REPO / "reports" / "research" / f"p8_multistart_multipool_{game}.json", text)
    print(json.dumps({k: out[k] for k in ("game", "selected", "stability", "guard_manifest", "seconds")}, indent=1, default=str))
    if before != after or before_w != after_w:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


def truth(game: str) -> dict:
    """The truth step for hou_cle_2016 only, as stage 1's (tools/p8_multistart.py truth)."""
    from tools import p8_frozen as F
    work, ref, ms = _check(game)
    if not STAGE2[game]["truth"]:
        raise SystemExit(f"{WHO}: {game} has no truth step (section 7)")
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ms.resolve():
        raise SystemExit(f"{WHO}: run the truth step with V3_P8_ROOT set to work_ms")
    gate_stage2(ms)
    rec_p = ms / "multistart" / game / "multipool.json"
    if not rec_p.exists():
        raise SystemExit(f"{WHO}: {game} has no multipool record")
    rec = json.loads(rec_p.read_text())
    if not rec["stability"]["passes"]:
        raise SystemExit(f"{WHO}: {game} is not stable; no multipool truth is written (section 6)")
    if (ms / "labels" / f"{game}_result.json").exists():
        raise SystemExit(f"{WHO}: {game} already has a truth in work_ms")
    before, t0 = M.manifest(ref), time.time()
    M.setup_ms(ref, ms, game)
    P = F.install_paths(ms)
    s = rec["selected"]["truth"]
    result = M.truth_result(s, rec)
    M.check_labels_inside(P, ms, game)
    with M.replaced_fit_game(result, rec["frames_hash"]) as rp:
        res = F.step_fit(P, game)
    M.check_labels_inside(P, ms, game)
    hashes = {str(p.relative_to(ms)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted([*(P["labels"] / game).glob("*.jsonl"), *P["labels"].glob(f"{game}_*")])}
    after = M.manifest(ref)
    out = {"game": game, "truth_version": "multipool v1", "selected": s, "fit_step": res, "fit_game_calls": rp.calls,
           "labels_sha256": hashes, "guard_manifest": {"files": len(before), "unchanged": before == after},
           "seconds": round(time.time() - t0, 1)}
    text = json.dumps(out, indent=1, default=str)
    atomic_write(ms / "multistart" / game / "truth.json", text)
    atomic_write(REPO / "reports" / "research" / f"p8_multistart_truth_{game}.json", text)
    print(json.dumps({k: out[k] for k in ("game", "fit_step", "fit_game_calls", "guard_manifest", "seconds")}, indent=1, default=str))
    if before != after:
        raise SystemExit(f"{WHO}: the reference root's protected files changed")
    return out


def accept(game: str) -> dict:
    """Acceptance for hou_cle_2016 in work_ms: check C needs the reset-fix arm's composite for the game in work_ref; while
    builds/r23_checkc_<game>.json is missing or not complete, the status is recorded as pending and nothing is measured."""
    import shutil
    from tools import p8_truth_accept as TA
    from tools import p8_truth_v2 as TV
    work, ref, ms = _check(game)
    if not STAGE2[game]["truth"]:
        raise SystemExit(f"{WHO}: {game} has no truth step, so no acceptance (section 7)")
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ms.resolve():
        raise SystemExit(f"{WHO}: run accept with V3_P8_ROOT set to work_ms")
    if not (ms / "labels" / f"{game}_result.json").exists():
        raise SystemExit(f"{WHO}: {game} has no multipool truth in work_ms")
    before, t0 = M.manifest(ref), time.time()
    cc = ref / "builds" / f"r23_checkc_{game}.json"
    if not cc.exists() or json.loads(cc.read_text()).get("complete") is not True:
        rec = {"game": game, "truth_version": "multipool v1", "acceptance": {"use": "pending"},
               "reason": "no complete check C composite for this game in the reference root (the reset-fix arm's "
                         "equivalence for it has not passed)", "seconds": round(time.time() - t0, 1)}
    else:
        (ms / "builds" / "r23").mkdir(parents=True, exist_ok=True)
        for src in [*sorted((ref / "builds" / "r23").glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_*")), cc]:
            dst = (ms / "builds" / "r23" / src.name) if src.parent.name == "r23" else (ms / "builds" / src.name)
            if not dst.exists():
                shutil.copy2(src, dst)
        TV.cmd_measure_dev(argparse.Namespace(tag=game))
        rec = {"game": game, "truth_version": "multipool v1", "acceptance": TA.accept(game, ms),
               "seconds": round(time.time() - t0, 1)}
    after = M.manifest(ref)
    rec["guard_manifest"] = {"files": len(before), "unchanged": before == after}
    text = json.dumps(rec, indent=1, default=str)
    atomic_write(ms / "multistart" / game / "accept.json", text)
    atomic_write(REPO / "reports" / "research" / f"p8_multistart_accept_{game}.json", text)
    print(json.dumps({k: rec.get(k) for k in ("game", "acceptance", "reason", "guard_manifest")}, indent=1, default=str)[:2000])
    if before != after:
        raise SystemExit(f"{WHO}: the reference root's protected files changed")
    return rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("multipool", "truth", "accept"):
        sp = sub.add_parser(name)
        sp.add_argument("game")
    a = ap.parse_args(argv)
    {"multipool": multipool, "truth": truth, "accept": accept}[a.cmd](a.game)


if __name__ == "__main__":
    sys.exit(main())
