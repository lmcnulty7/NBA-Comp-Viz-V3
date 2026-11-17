"""tools/p8_night.py: the P8 Phase A camera-truth and r23 baseline run on the Mac, one game at a time (PLAN.md D85 (4)).

    python3 -m tools.p8_night plan                       the job list and the lock-hour estimate; runs nothing
    V3_CV_PYTHON=<python with ultralytics and easyocr> \\
        python3 -m tools.p8_night run [--games a,b] [--max-hours H] [--sections N]

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

--sections N limits every per-section step to a game's first N sections (a smoke run); --max-hours stops starting
new jobs after that many hours.
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
SCORES = (("r23", "camera"), ("v3", "camera"), ("r23", "b4"), ("r23", "camera_paint"))

# Lock-hour estimate. Bases (docs/research/P8_FEASIBILITY.md section 5.1): b2, this Mac 2.15 s per second of label
# window video for camera-truth labelling (ROADMAP_LOG.md 2026-10-04), which is mostly the V3 window build; the r23
# rebuild is the same build over the same windows; OCR sync measured 4,389 to 5,440 s per R1.2 game for 5,439 to
# 6,262 reads (data/sportvu/sync, ocr_seconds), about 0.8 s per 1 Hz read; direction builds 40 s per section.
LABEL_S_PER_VIDEO_S = 2.15
OCR_S_PER_READ = 0.8
FIT_H, SCORE_H, LAYOUT_H = 0.25, 0.4, 0.1
DIRECTION_S_PER_SECTION = 40 * LABEL_S_PER_VIDEO_S
CHANNEL_MAPPED_S = 2450.0          # running seconds mapped per game, the R1.2 mean (reports/sportvu_prepare.json)


def jobs_for(tag: str, g: dict, P: dict, n_sections: int, max_sections=None) -> list:
    """[(name, argv for tools.p8_frozen / p8_scorecard, heavy, estimate_h)] in run order."""
    secs = [f"{tag}_s{k:02d}" for k in range(n_sections)][:max_sections]
    mapped = _mapped_s(tag, P) or CHANNEL_MAPPED_S
    per_sec_label_h = mapped * LABEL_S_PER_VIDEO_S / 3600 / max(1, n_sections)
    J = []
    if tag in R13:
        J.append(("link", ["tools.p8_frozen", "link", tag], False, 0.0))
    else:
        J += [("split", ["tools.p8_frozen", "split", tag], False, 0.02),
              ("moments", ["tools.p8_frozen", "moments", tag], False, 0.02),
              ("layout", ["tools.p8_frozen", "layout", tag], True, LAYOUT_H)]
        J += [(f"sync {s}", ["tools.p8_frozen", "sync", tag, "--section", s], True, 600 * OCR_S_PER_READ / 3600) for s in secs]
        J += [(f"direction {s}", ["tools.p8_frozen", "direction", tag, "--section", s], True, DIRECTION_S_PER_SECTION / 3600) for s in secs]
        J.append(("mirror", ["tools.p8_frozen", "mirror", tag], False, 0.02))
    J += [(f"label {s}", ["tools.p8_frozen", "label", tag, "--section", s], True, per_sec_label_h) for s in secs]
    J.append(("fit", ["tools.p8_frozen", "fit", tag], True, FIT_H))
    J += [(f"rebuild {s}", ["tools.p8_frozen", "rebuild", tag, "--section", s, "--build", BUILD], True, per_sec_label_h) for s in secs]
    J += [(f"score {b} {t}", ["tools.p8_scorecard", "game", tag, "--build", b, "--truth", t], True,
           SCORE_H if k == 0 else 0.1) for k, (b, t) in enumerate(SCORES)]
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


def plan(games=None, max_sections=None) -> dict:
    from tools.p8_frozen import layout
    P = layout(p8_root())
    dg = dev_games()
    order = [t for t in R13 if t in dg] + sorted(t for t in dg if t not in R13)
    if games:
        order = [t for t in order if t in games]
    refuse_cv_test(order + [dg[t]["sportvu"] for t in order] + [dg[t]["video_id"] for t in order], WHO)
    out = {}
    for t in order:
        J = jobs_for(t, dg[t], P, n_sections(t, dg[t], P), max_sections)
        out[t] = {"jobs": [{"name": n, "argv": a, "heavy": h, "estimate_h": round(e, 3)} for n, a, h, e in J],
                  "lock_h": round(sum(e for _, _, h, e in J if h), 2), "all_h": round(sum(e for *_, e in J), 2)}
    return {"games": out, "lock_h_total": round(sum(g["lock_h"] for g in out.values()), 1),
            "all_h_total": round(sum(g["all_h"] for g in out.values()), 1),
            "pending_channel_games": sorted(channel_games(status="pending"))}


def _status_path(P: dict) -> Path:
    return P["meta"] / "night_status.json"


def _save(path: Path, status: dict) -> None:
    part = path.with_suffix(".part")
    part.write_text(json.dumps(status, indent=1))
    part.replace(path)


def run(games=None, max_hours=None, max_sections=None) -> int:
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
    pl = plan(games, max_sections)
    for tag, gp in pl["games"].items():
        st = status.setdefault(tag, {})
        jobs = gp["jobs"]                  # a channel game's list is replanned after its split (the section count)
        k = 0
        while k < len(jobs):
            job = jobs[k]
            name = job["name"]
            if st.get(name, {}).get("state") == "ok":
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
            st[name] = {"state": "ok" if r.returncode == 0 else "failed", "rc": r.returncode,
                        "seconds": round(time.time() - t0, 1), "at": dt.datetime.now().isoformat(timespec="seconds"),
                        "log": log.name}
            _save(sp, status)
            print(f"  {tag:14s} {name:28s} {st[name]['state']:6s} {st[name]['seconds']:8.1f} s", flush=True)
            if r.returncode != 0:
                print(f"{WHO}: {tag} stopped at {name} (see logs/{log.name}); next game", flush=True)
                break
            if name == "split":            # the section count is known now: replan this game
                dg = dev_games()
                jobs = [{"name": n, "argv": a, "heavy": h, "estimate_h": e}
                        for n, a, h, e in jobs_for(tag, dg[tag], P, n_sections(tag, dg[tag], P), max_sections)]
            k += 1
    _save(sp, status)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("--games", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--sections", type=int, default=None)
    a = ap.parse_args(argv)
    games = a.games.split(",") if a.games else None
    if a.cmd == "plan":
        pl = plan(games, a.sections)
        for t, g in pl["games"].items():
            print(f"{t:14s} jobs {len(g['jobs']):3d}  lock {g['lock_h']:5.2f} h  all {g['all_h']:5.2f} h")
        print(f"total: lock {pl['lock_h_total']} h, all {pl['all_h_total']} h; channel games still pending a scorebug "
              f"check (not planned): {', '.join(pl['pending_channel_games']) or 'none'}")
        return 0
    return run(games, a.max_hours, a.sections)


if __name__ == "__main__":
    sys.exit(main())
