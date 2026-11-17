"""tools/p8_night.py: the P8 Phase A camera-truth and r23 baseline run on the Mac, one game at a time (PLAN.md D85 (4)).

    python3 -m tools.p8_night plan                       the job list and the lock-hour estimate; runs nothing
    V3_CV_PYTHON=<python with ultralytics and easyocr> \\
        python3 -m tools.p8_night run [--games a,b] [--max-hours H] [--sections N] [--b4 gate|all|none]

Prepared by the P8 measurement teammate; the lead starts it (overnight, after the deploy and the P4 loop). It is
resumable: every job writes its output atomically and records itself in <P8 root>/meta/night_status.json, and a
rerun skips a job whose status is ok and whose output check still passes. A failed job stops the rest of that game
(later jobs depend on it) and the run moves on to the next game.

Each job is one short unit: one section of OCR sync, direction, labelling or rebuild (a few minutes to about 15), one
game's camera fit, or one game's scorecard. Every job that runs a model or reads video at scale holds the heavy-job
lock for its own duration only (python3 -m export.heavy --name "p8 ..." -- ...), so teammates can take the lock
between jobs (CLAUDE.md rule 7). Outputs go to the P8 workspace (tools.p8_common.p8_root), outside DATA_ROOT, outside
reports/ and outside the repository (rule 6). No lane command line runs and nothing is written to DATA_ROOT; the
scorecard reads one play-by-play file per game from it.

Order: the 4 R1.3 games (sections linked from the main checkout; camera truth relabelled here so labels, clips and
builds share one decode), then the confirmed channel games of reports/research/p8_registry.json. CV test games, test
videos and test-arena games are refused (refuse_cv_test) before any job is planned.

--sections N limits every per-section step to a game's first N sections (a smoke run; its fit is --partial and its
status is kept under its own configuration key, so it never counts as a full job); --max-hours stops starting new jobs
after that many hours. --b4 gate (default) adds the costly old-truth (B4) score on the scorecard's two gate games, one
R1.3 and one channel game, which tools.p8_scorecard gate needs. A job is done only while its output check passes:
the step's done marker (meta/done/), a non-partial fit, a scorecard newer than its fit and rebuilds.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.p8_common import REPO, channel_games, dev_games, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_night"
R13 = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_ind_2016", "gsw_sac_2015")
BUILD = "r23"
SCORES = (("r23", "camera"), ("v3", "camera"), ("r23", "camera_paint"))
B4_SCORE = ("r23", "b4")          # the old truth, for the Phase A gate comparison (costly)
B4_MODES = ("gate", "all", "none")  # gate (default): b4 only on the scorecard's gate games, one R1.3 and one channel

# Lock-hour estimate, measured by the smoke run of 2026-11-17 (gsw_bkn_2015, section s00, 161 s of mapped video in 9
# label windows, this Mac, lock held by that job alone): label 481 s, fit 25 s, rebuild r23 331 s, score r23 camera
# 48 s (gate v2, referee and cut caches filled), each later camera score about 7 s, score r23 b4 518 s (the frozen
# B4 ICP offset search). Per second of mapped label-window video: label 3.0 s, rebuild 2.06 s, first score 0.30 s,
# b4 score 3.2 s. OCR sync is the R1.2 record: 4,389 to 5,440 s per game for 5,439 to 6,262 reads (data/sportvu/sync,
# ocr_seconds), about 0.8 s per 1 Hz read; direction builds 40 s of running clock per section at the label rate.
LABEL_S_PER_VIDEO_S = 3.0
REBUILD_S_PER_VIDEO_S = 2.06
SCORE_S_PER_VIDEO_S = 0.30
B4_S_PER_VIDEO_S = 3.2
OCR_S_PER_READ = 0.8
FIT_H, LATER_SCORE_H, LAYOUT_H = 0.15, 0.03, 0.1
DIRECTION_S_PER_SECTION = 40 * LABEL_S_PER_VIDEO_S
CHANNEL_MAPPED_S = 2450.0          # running seconds mapped per game, the R1.2 mean (reports/sportvu_prepare.json)


def b4_for(tag: str, b4: str) -> bool:
    from tools.p8_scorecard import CONSTANTS
    return b4 == "all" or (b4 == "gate" and tag in CONSTANTS["gate_games"])


def jobs_for(tag: str, g: dict, P: dict, n_sections: int, max_sections=None, b4: str = "gate") -> list:
    """[(name, argv for tools.p8_frozen / p8_scorecard, heavy, estimate_h)] in run order."""
    secs = [f"{tag}_s{k:02d}" for k in range(n_sections)][:max_sections]
    mapped = _mapped_s(tag, P) or CHANNEL_MAPPED_S
    frac = len(secs) / max(1, n_sections)
    video_s = mapped * frac
    per_sec_label_h = video_s * LABEL_S_PER_VIDEO_S / 3600 / max(1, len(secs))
    per_sec_rebuild_h = video_s * REBUILD_S_PER_VIDEO_S / 3600 / max(1, len(secs))
    J = [("sweep", ["tools.p8_frozen", "sweep", tag], False, 0.0)]
    if tag in R13:
        J.append(("link", ["tools.p8_frozen", "link", tag], False, 0.0))
    else:
        J += [("split", ["tools.p8_frozen", "split", tag], False, 0.02),
              ("moments", ["tools.p8_frozen", "moments", tag], False, 0.02)]
        if not g.get("layout"):            # a layout fixed by hand in the registry needs no automatic check
            J.append(("layout", ["tools.p8_frozen", "layout", tag], True, LAYOUT_H))
        J += [(f"sync {s}", ["tools.p8_frozen", "sync", tag, "--section", s], True, 600 * OCR_S_PER_READ / 3600) for s in secs]
        J += [(f"direction {s}", ["tools.p8_frozen", "direction", tag, "--section", s], True, DIRECTION_S_PER_SECTION / 3600) for s in secs]
        J.append(("mirror", ["tools.p8_frozen", "mirror", tag], False, 0.02))
    J += [(f"label {s}", ["tools.p8_frozen", "label", tag, "--section", s], True, per_sec_label_h) for s in secs]
    J.append(("fit", ["tools.p8_frozen", "fit", tag] + (["--partial"] if max_sections else []), True, FIT_H))
    J += [(f"rebuild {s}", ["tools.p8_frozen", "rebuild", tag, "--section", s, "--build", BUILD], True, per_sec_rebuild_h) for s in secs]
    J += [(f"score {b} {t}", ["tools.p8_scorecard", "game", tag, "--build", b, "--truth", t], True,
           video_s * SCORE_S_PER_VIDEO_S / 3600 if k == 0 else LATER_SCORE_H) for k, (b, t) in enumerate(SCORES)]
    if b4_for(tag, b4):
        J.append(("score r23 b4", ["tools.p8_scorecard", "game", tag, "--build", B4_SCORE[0], "--truth", B4_SCORE[1]], True,
                  video_s * B4_S_PER_VIDEO_S / 3600))
    return J


def _mapped_s(tag: str, P: dict):
    tms = sorted(P["sync"].glob(f"{tag}_s*_local_timemap.json"))
    if not tms:
        return None
    return sum(json.loads(p.read_text())["mapped_video_s"] for p in tms)


def n_sections(tag: str, g: dict, P: dict) -> int:
    n = len(sorted(P["video"].glob(f"{tag}_s[0-9][0-9].mp4")))
    if n:
        return n
    if tag in R13:
        return {"gsw_cha_2016": 10}.get(tag, 11)
    return int(g.get("duration_s", 6300) // 600) + 1


def plan(games=None, max_sections=None, b4: str = "gate") -> dict:
    from tools.p8_frozen import layout
    P = layout(p8_root())
    dg = dev_games()
    order = [t for t in R13 if t in dg] + sorted(t for t in dg if t not in R13)
    if games:
        order = [t for t in order if t in games]
    refuse_cv_test(order + [dg[t]["sportvu"] for t in order] + [dg[t]["video_id"] for t in order], WHO)
    out = {}
    for t in order:
        J = jobs_for(t, dg[t], P, n_sections(t, dg[t], P), max_sections, b4)
        out[t] = {"jobs": [{"name": n, "argv": a, "heavy": h, "estimate_h": round(e, 3)} for n, a, h, e in J],
                  "lock_h": round(sum(e for _, _, h, e in J if h), 2), "all_h": round(sum(e for *_, e in J), 2)}
    return {"games": out, "lock_h_total": round(sum(g["lock_h"] for g in out.values()), 1),
            "all_h_total": round(sum(g["all_h"] for g in out.values()), 1),
            "pending_channel_games": sorted(channel_games(status="pending"))}


def config_key(max_sections, b4: str) -> str:
    """The run configuration a status entry belongs to: a smoke run (--sections) never marks a full job done."""
    return f"sections={max_sections or 'all'} b4={b4}"


def output_ok(P: dict, tag: str, job: dict, max_sections) -> bool:
    """A job counts as done only while its output still passes this check (the status file alone is not enough)."""
    from tools.p8_frozen import done_path
    a = job["argv"]
    step = a[1]
    if a[0] == "tools.p8_scorecard":
        build, truth = a[a.index("--build") + 1], a[a.index("--truth") + 1]
        card = P["scorecards"] / f"{tag}__{build}__{truth}.json"
        fit = done_path(P, "fit", tag)
        inputs = [fit] + sorted((P["meta"] / "done").glob(f"rebuild__{tag}__*__{BUILD}.json"))
        return card.exists() and fit.exists() and all(card.stat().st_mtime >= p.stat().st_mtime for p in inputs)
    section = a[a.index("--section") + 1] if "--section" in a else None
    build = a[a.index("--build") + 1] if step == "rebuild" else None
    marker = done_path(P, step, tag, section, build)
    if not marker.exists():
        return False
    if step == "fit":
        res = json.loads(marker.read_text()).get("result", {})
        return bool(max_sections) or not res.get("partial", True)
    return True


def _status_path(P: dict) -> Path:
    return P["meta"] / "night_status.json"


def _save(path: Path, status: dict) -> None:
    part = path.with_suffix(".part")
    part.write_text(json.dumps(status, indent=1))
    part.replace(path)


def run(games=None, max_hours=None, max_sections=None, b4: str = "gate") -> int:
    py = os.environ.get("V3_CV_PYTHON")
    if not py or not Path(py).exists():
        raise SystemExit(f"{WHO}: set V3_CV_PYTHON to the Python that has ultralytics and easyocr (the frozen CV code's)")
    from tools.p8_frozen import layout
    P = layout(p8_root())
    for p in P.values():
        p.mkdir(parents=True, exist_ok=True)
    sp = _status_path(P)
    status = json.loads(sp.read_text()) if sp.exists() else {}
    t_start = time.time()
    pl = plan(games, max_sections, b4)
    for tag, gp in pl["games"].items():
        st = status.setdefault(tag, {}).setdefault(config_key(max_sections, b4), {})
        jobs = gp["jobs"]                  # a channel game's list is replanned after its split (the section count)
        k = 0
        while k < len(jobs):
            job = jobs[k]
            name = job["name"]
            if name != "sweep" and st.get(name, {}).get("state") == "ok" and output_ok(P, tag, job, max_sections):
                k += 1
                continue
            if max_hours and (time.time() - t_start) / 3600 > max_hours:
                print(f"{WHO}: --max-hours reached; stopping before {tag} {name}", flush=True)
                _save(sp, status)
                return 0
            log = P["logs"] / f"{tag}__{name.replace(' ', '_')}.log"
            cmd = [py, "-m", *job["argv"]]
            if job["heavy"]:
                cmd = [sys.executable, "-m", "export.heavy", "--name", f"p8 {tag} {name}", "--", *cmd]
            t0 = time.time()
            with open(log, "w") as fh:
                r = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
            ok = r.returncode == 0 and (name == "sweep" or output_ok(P, tag, job, max_sections))
            st[name] = {"state": "ok" if ok else "failed", "rc": r.returncode,
                        "seconds": round(time.time() - t0, 1), "at": dt.datetime.now().isoformat(timespec="seconds"),
                        "log": log.name}
            _save(sp, status)
            print(f"  {tag:14s} {name:28s} {st[name]['state']:6s} {st[name]['seconds']:8.1f} s", flush=True)
            if not ok:
                print(f"{WHO}: {tag} stopped at {name} (see logs/{log.name}); next game", flush=True)
                break
            if name == "split":            # the section count is known now: replan this game
                dg = dev_games()
                jobs = [{"name": n, "argv": a, "heavy": h, "estimate_h": e}
                        for n, a, h, e in jobs_for(tag, dg[tag], P, n_sections(tag, dg[tag], P), max_sections, b4)]
            k += 1
    for tag in pl["games"]:                # leave nothing of a killed or failed build in the checkout
        subprocess.run([py, "-m", "tools.p8_frozen", "sweep", tag], cwd=REPO, capture_output=True)
    _save(sp, status)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("--games", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--sections", type=int, default=None)
    ap.add_argument("--b4", default="gate", choices=B4_MODES,
                    help="score r23 on the old B4 truth: gate (default, the scorecard's two gate games), all or none")
    a = ap.parse_args(argv)
    games = a.games.split(",") if a.games else None
    if a.cmd == "plan":
        pl = plan(games, a.sections, a.b4)
        for t, g in pl["games"].items():
            print(f"{t:14s} jobs {len(g['jobs']):3d}  lock {g['lock_h']:5.2f} h  all {g['all_h']:5.2f} h")
        print(f"total: lock {pl['lock_h_total']} h, all {pl['all_h_total']} h; channel games still pending a scorebug "
              f"check (not planned): {', '.join(pl['pending_channel_games']) or 'none'}")
        return 0
    return run(games, a.max_hours, a.sections, a.b4)


if __name__ == "__main__":
    sys.exit(main())
