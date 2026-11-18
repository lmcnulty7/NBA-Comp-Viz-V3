"""tools/p8_frozen.py: the frozen R1.2 and R1.3 code, run on P8 development games with every output in the P8 workspace.

    python -m tools.p8_frozen STEP TAG [--section TAG_sNN] [--build NAME]

sportvu/ is frozen (CLAUDE.md rule 3) and its modules write under data/sportvu, data/harvest and reports/ of the
checkout, and refuse any game outside sportvu/splits.json "train". This module changes none of them. It imports their
functions, points their module-level paths at the P8 workspace (install_paths, before any step runs) and replaces only
the two orchestration loops that hold the train-only refusal (autolabel.label_game and direction.main). That refusal
is replaced by tools.p8_common.guard (no CV test game, test video or test arena; no spent held-out game).

Steps, one short job each (the runner, tools.p8_night, holds the heavy-job lock around every job that runs a model):

  link         R1.3 games: link the main checkout's sections, moments and sync maps into the workspace (read only)
  split        channel games: ~10-min sections of video/<tag>.mp4 (ffmpeg stream copy, harvest_driver.split_sections)
  moments      channel games: SportVU moments (sportvu.fetch.download + parse) -> sportvu/<game>_moments.json
  layout       channel games: the scorebug layout, frozen prepare.do_layout scoring, result in meta/<tag>_layout.json
  sync         one section: OCR time map (sportvu.local_sync.build_local_timemap) -> sync/<section>_local_timemap.json
  direction    one section: the 40 s direction windows built with V3 (sportvu.direction's cut, build, window_rows)
  mirror       channel games: the mirror vote over the direction windows (sportvu.direction.solve)
  label        one section: camera-truth candidate windows (autolabel phase A: cut, V3 build with gate v2, time map)
  fit          the camera position and per-frame camera truth (autolabel phases B and C, sportvu.camera), labels in
               labels/<tag>/<window>.jsonl; GSW home games also get the gold-key paint check (sportvu.label_check)
  rebuild      one section: the label windows built again with another court model (--build r23: COURT_GRID_WEIGHTS)

Interpreter: the frozen CV code needs ultralytics and easyocr, which only the anaconda Python has (CLAUDE.md keeps
them out of the system Python). Tests import only the pure parts of this module.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from tools.p8_common import REPO, atomic_write, dev_games, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_frozen"
R23_SHA256_FILE = REPO / "reports" / "r23_verification.json"


# ── workspace paths ─────────────────────────────────────────────────────────────────────────────────────────────────
def layout(root: Path) -> dict:
    """The P8 workspace layout."""
    return {k: root / k for k in ("video", "sportvu", "sync", "direction", "label_build", "labels", "builds", "meta",
                                  "reports", "cache", "scorecards", "logs", "failed", "tmp")}


def install_paths(root: Path | None = None) -> dict:
    """Point the frozen modules' paths at the workspace. Returns the layout. Must run before any step."""
    import config
    import fetch_pbp
    import harvest_driver
    from sportvu import autolabel, direction, fetch, local_sync, rebuild, sync, truth
    P = layout(root or p8_root())
    for p in P.values():
        p.mkdir(parents=True, exist_ok=True)

    def video_path(stem: str):
        return P["video"] / f"{stem}.mp4"

    config.REPORTS_DIR = P["reports"]
    models = main_checkout() / "models" if main_checkout() else None
    if not config.HEAD_V2_PATH.exists() and models and (models / config.HEAD_V2_PATH.name).exists():
        config.HEAD_V2_PATH = models / config.HEAD_V2_PATH.name          # gitignored: read from the main checkout
    harvest_driver.HARVEST_VIDEO = P["video"]
    for mod in (fetch_pbp, rebuild, local_sync, sync):
        mod.video_path = video_path
    fetch.SV_DIR = P["sportvu"]
    for mod in (sync, rebuild, autolabel, direction, local_sync):
        if hasattr(mod, "SYNC_DIR"):
            mod.SYNC_DIR = P["sync"]
    rebuild.BUILD_DIR = P["direction"]          # never used without an explicit out_dir here; kept off the checkout
    direction.DIR_BUILD = P["direction"]
    autolabel.OUT, autolabel.WIN_DIR = P["labels"], P["label_build"]
    truth.TRUTH_DIR = P["cache"] / "b4_truth"
    return P


def games() -> dict:
    """The development games, guarded."""
    return dev_games()


def game_of(tag: str) -> dict:
    g = games().get(tag)
    if g is None:
        raise SystemExit(f"{WHO}: {tag} is not a development game (R1.3 or a confirmed channel game)")
    refuse_cv_test([tag, g["sportvu"], g["video_id"]], WHO)
    guard([g["sportvu"], g["video_id"]], WHO)
    return g


def sections(P: dict, tag: str) -> list:
    return sorted(p.stem for p in P["video"].glob(f"{tag}_s[0-9][0-9].mp4"))


def mirror_of(P: dict, tag: str) -> str:
    """The game's mirror: the workspace's direction result, else the tracked R1.2 report (R1.3 games)."""
    for p in (P["reports"] / f"sportvu_direction_{tag}.json", REPO / "reports" / f"sportvu_direction_{tag}.json"):
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("status") in ("ok", "ok_by_arena") or (d.get("resolution") and tag in R13_TAGS):
                return d["resolution"]["mirror"]
            raise SystemExit(f"{WHO}: {tag} mirror is {d.get('status')}; resolve it before labelling")
    raise SystemExit(f"{WHO}: {tag} has no direction result")


R13_TAGS = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_ind_2016", "gsw_sac_2015")


def layout_of(P: dict, tag: str) -> str:
    """The scorebug layout: the R1.3 registry entry, else the workspace layout check, else a hand calibration in the
    P8 registry (layouts are injected into clock_reader.LAYOUTS by inject_layouts)."""
    g = game_of(tag)
    if g.get("layout"):
        return g["layout"]
    reg = json.loads((REPO / "data" / "harvest" / "games.json").read_text())
    if tag in reg and reg[tag].get("layout"):
        return reg[tag]["layout"]
    p = P["meta"] / f"{tag}_layout.json"
    if p.exists() and json.loads(p.read_text()).get("chosen"):
        return json.loads(p.read_text())["chosen"]
    raise SystemExit(f"{WHO}: {tag} has no scorebug layout yet (run the layout step, or calibrate one by hand)")


def inject_layouts() -> None:
    """Hand calibrations of new scorebugs from reports/research/p8_registry.json "layouts" into clock_reader.LAYOUTS
    (in memory; clock_reader.py is not edited). An existing name is never overwritten."""
    import clock_reader
    reg = json.loads((REPO / "reports" / "research" / "p8_registry.json").read_text())
    for name, box in (reg.get("layouts") or {}).items():
        if name in clock_reader.LAYOUTS:
            continue
        clock_reader.LAYOUTS[name] = {"period_box": tuple(box["period_box"]), "clock_box": tuple(box["clock_box"]),
                                      **({"alternates": [tuple(map(tuple, a)) for a in box["alternates"]]}
                                         if box.get("alternates") else {})}


PERIOD_WORDS = (("FIR", 1), ("SEC", 2), ("THI", 3), ("FOU", 4), ("OV", 5), ("OT", 5))
WORD_ALLOW = "FIRSTECONDHUROVQ"


def period_from_words(text: str):
    """1..5 from a period written as a word (FIRST, SECOND, THIRD, FOURTH, OT or OVERTIME); None otherwise."""
    t = "".join(ch for ch in text.upper() if ch.isalpha())
    for prefix, p in PERIOD_WORDS:
        if t.startswith(prefix):
            return p
    return None


def make_reader(layout_name: str):
    """clock_reader.ClockReader for the layout. A layout whose registry entry says period_words reads the period as a
    word with its own allowlist (the frozen reader reads digits only); the clock is parsed by the frozen code."""
    from clock_reader import ClockReader
    inject_layouts()
    reg = json.loads((REPO / "reports" / "research" / "p8_registry.json").read_text())
    if not (reg.get("layouts") or {}).get(layout_name, {}).get("period_words"):
        return ClockReader(layout_name)

    class WordPeriodReader(ClockReader):
        def _read_boxes(self, frame_bgr, period_box, clock_box):
            _, clock, raw = super()._read_boxes(frame_bgr, period_box, clock_box)
            words = self._ocr(frame_bgr, period_box, WORD_ALLOW)
            return period_from_words(words), clock, f"{words}|{raw}"

    return WordPeriodReader(layout_name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def court_weights(build: str) -> Path:
    """The court model of a named build, checked against its recorded sha256 (r23: reports/r23_verification.json)."""
    if build != "r23":
        raise SystemExit(f"{WHO}: unknown build {build}; only r23 (the frozen court baseline) is defined")
    p = main_checkout() / "models" / "court_grid_r23.pt"
    rec = json.loads(R23_SHA256_FILE.read_text())
    want = {v for k, v in _walk(rec) if "sha256" in k.lower() and isinstance(v, str) and len(v) == 64}
    got = sha256(p)
    if got not in want:
        raise SystemExit(f"{WHO}: court_grid_r23.pt sha256 {got[:12]} is not recorded in reports/r23_verification.json")
    return p


def _walk(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}.{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


# ── safe wrappers around the frozen cut and build ───────────────────────────────────────────────────────────────────
def sweep_checkout(P: dict, stem_glob: str) -> list:
    """Move anything the frozen build left in the checkout for these window stems (data/tracking/<stem>_*,
    reports/viz/team_clusters_<stem>*) into <P8 root>/failed/, so nothing stays under the repository or reports/.
    shutil.move is a rename on one filesystem."""
    import shutil
    import config
    moved = []
    dest = P["failed"] / "checkout_leftovers"
    for src in list(config.TRACKING_DIR.glob(stem_glob + "_*")) + \
            list((config.PROJECT_ROOT / "reports" / "viz").glob("team_clusters_" + stem_glob + "*")):
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest / src.name))
        moved.append(src.name)
    return moved


def safe_cut(P: dict, w: dict, out_dir: Path) -> Path:
    """sportvu.rebuild.cut into a scratch folder, then renamed into place: a killed cut never leaves a partial clip
    that a later run would take as done."""
    from sportvu.rebuild import cut
    final = out_dir / (w["window"] + ".mp4")
    if final.exists():
        return final
    tmp = P["tmp"] / "cut"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / final.name).unlink(missing_ok=True)           # a partial clip of a killed run, in the scratch folder only
    os.replace(cut(w, out_dir=tmp), final)
    return final


# build_trajectories.py exits non-zero rather than write an empty sidecar when it processed no frame. With --pregate,
# a window whose coarse gate pass finds no live frame reads nothing and prints the "Video unreadable" variant of that
# message, so the text alone cannot tell a gate-empty window from a broken clip: a zero-frame exit counts as gate-empty
# only when this process can decode the clip itself (harvest_driver.decodable). A window of running clock can be all
# non-wide footage, e.g. a picture-in-picture interview over live play (gsw_cha_2016_s06_w07, night 1); the frozen
# autolabel counted such windows as build_failed and went on (Colab R1.3 result for gsw_cha_2016: build_failed 2).
ZERO_FRAME_MARKERS = ("processed 0 frames", "successful reads:", "All read frames were gate-rejected",
                      "Video unreadable by cv2")


def zero_frame_exit(r: dict) -> bool:
    return r.get("rc") not in (0, None) and any(m in (r.get("tail") or "") for m in ZERO_FRAME_MARKERS)


def _decodable(path: Path) -> bool:
    import harvest_driver
    return harvest_driver.decodable(path)


def gate_empty_path(out_dir: Path, stem: str) -> Path:
    return out_dir / (stem + "_gate_empty.json")


def safe_build(P: dict, w: dict, snip: Path, out_dir: Path, extra_args: tuple = ()) -> dict:
    """sportvu.rebuild.build with the checkout swept before and after. A window whose every read frame the gate
    rejected is not a failure: it gets a <window>_gate_empty.json marker (so it is not built again) and the step goes
    on, as the frozen autolabel did. Any other build with rc != 0 is a failure: its outputs in out_dir (all but the
    clip and the window time map) move to <P8 root>/failed/ so a rerun builds it again."""
    import shutil
    from sportvu.rebuild import build
    stem = w["window"]
    if gate_empty_path(out_dir, stem).exists():
        return {"ok": True, "cached": True, "empty": True}
    sweep_checkout(P, stem)
    r = build(w, snip, out_dir=out_dir, extra_args=extra_args)
    left = sweep_checkout(P, stem)
    if r.get("cached"):
        return {"ok": True, "cached": True}
    tail = r.get("tail", "") or ""
    if zero_frame_exit(r) and not (out_dir / (stem + "_frames.json")).exists() and not left and _decodable(snip):
        atomic_write(gate_empty_path(out_dir, stem), json.dumps({"window": stem, "rc": r.get("rc"), "tail": tail}))
        return {"ok": True, "empty": True, "rc": r.get("rc")}
    ok = r.get("rc") == 0 and (out_dir / (stem + "_frames.json")).exists() and not left
    if not ok:
        print(f"{WHO}: build of {stem} failed (rc {r.get('rc')}): {tail}", file=sys.stderr, flush=True)
        dest = P["failed"] / out_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob(stem + "_*"):
            if not f.name.endswith(("_timemap.json",)):
                shutil.move(str(f), str(dest / f.name))
    return {"ok": ok, "rc": r.get("rc"), "tail": r.get("tail", ""), "left_in_checkout": left}


def write_timemap(path: Path, section: str, reader, vid: Path) -> None:
    from sportvu.local_sync import build_local_timemap
    if not path.exists():
        atomic_write(path, json.dumps(build_local_timemap(section, reader, vid=vid)))


def done_path(P: dict, step: str, tag: str, section=None, build=None) -> Path:
    name = "__".join(x for x in (step, tag, section, build) if x)
    return P["meta"] / "done" / f"{name}.json"


def fail_if(failures: list, what: str) -> None:
    if failures:
        raise SystemExit(f"{WHO}: {len(failures)} window build(s) failed in {what}, for example {failures[0]}; "
                         "their outputs are in failed/ and a rerun builds them again")


# ── steps ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def step_link(P: dict, tag: str) -> dict:
    """R1.3 games: symlink the main checkout's sections, moments and sync maps (nothing is copied or changed)."""
    if tag not in R13_TAGS:
        raise SystemExit(f"{WHO}: link is for the R1.3 games only")
    g = game_of(tag)
    main = main_checkout()
    vid_src = (main / "data" / "harvest" / "video").resolve()
    n = 0
    for src in sorted(vid_src.glob(f"{tag}_s[0-9][0-9].mp4")):
        dst = P["video"] / src.name
        if not dst.exists():
            dst.symlink_to(src); n += 1
    sv = main / "data" / "sportvu"
    for src in [sv / f"{g['sportvu']}_moments.json", *sorted((sv / "sync").glob(f"{tag}_s*_local_timemap.json"))]:
        dst = (P["sportvu"] if src.name.endswith("_moments.json") else P["sync"]) / src.name
        if not dst.exists():
            dst.symlink_to(src); n += 1
    return {"linked": n, "sections": len(sections(P, tag))}


def step_split(P: dict, tag: str) -> dict:
    import harvest_driver
    game_of(tag)
    return {"sections": len(harvest_driver.split_sections(tag))}


def step_moments(P: dict, tag: str) -> dict:
    from sportvu import fetch
    g = game_of(tag)
    out = P["sportvu"] / f"{g['sportvu']}_moments.json"
    if not out.exists():
        raw = fetch.download(g["sportvu"])
        parsed = fetch.parse(raw)
        atomic_write(P["reports"] / f"sportvu_fetch_{g['sportvu']}.json", json.dumps(fetch.coverage_report(parsed), indent=1))
        atomic_write(out, json.dumps(parsed))
        raw.unlink()                                  # the 100 MB raw JSON is not kept (as sportvu.manifest does)
    return {"moments": out.name}


def step_layout(P: dict, tag: str) -> dict:
    """The frozen layout check (sportvu.prepare.do_layout's scoring) without writing games.json."""
    from clock_reader import ClockReader, LAYOUTS
    from sportvu.prepare import CONSISTENCY_TOP, MIN_LAYOUT_SHARE, sample_frames
    game_of(tag)
    inject_layouts()
    pairs = sample_frames(tag)
    plaus = lambda p, c: p is not None and 1 <= p <= 4 and c is not None and 0 <= c <= 720
    readers = {lay: ClockReader(lay) for lay in LAYOUTS}
    shares = {lay: round(sum(plaus(*readers[lay].read(a)[:2]) for a, _ in pairs) / max(len(pairs), 1), 3) for lay in LAYOUTS}
    top = sorted((l for l in shares if shares[l] >= MIN_LAYOUT_SHARE), key=lambda l: -shares[l])[:CONSISTENCY_TOP]
    consist = {}
    for lay in top:
        good = 0
        for a, b in pairs:
            p1, c1, _ = readers[lay].read(a)
            p2, c2, _ = readers[lay].read(b)
            if plaus(p1, c1) and plaus(p2, c2) and p1 == p2:
                d = c1 - c2
                good += abs(d) < 0.25 or 0.5 <= d <= 1.5
        consist[lay] = round(good / max(len(pairs), 1), 3)
    best = max(consist, key=consist.get) if consist else None
    res = {"pairs": len(pairs), "plausible_share": shares, "consistent_share": consist,
           "chosen": best if best and consist[best] >= MIN_LAYOUT_SHARE else None}
    atomic_write(P["meta"] / f"{tag}_layout.json", json.dumps(res, indent=1))
    return {"chosen": res["chosen"], "consistent_share": consist}


def step_sync(P: dict, tag: str, section: str) -> dict:
    from clock_reader import ClockReader
    from sportvu.local_sync import build_local_timemap
    game_of(tag)
    inject_layouts()
    out = P["sync"] / f"{section}_local_timemap.json"
    if out.exists():
        return {"cached": True}
    tm = build_local_timemap(section, make_reader(layout_of(P, tag)), vid=P["video"] / f"{section}.mp4")
    atomic_write(out, json.dumps(tm))
    return {k: tm[k] for k in ("reads", "reads_ok", "running_spans", "mapped_video_s", "ocr_seconds")}


def _windows(P: dict, section: str, budget_s: float) -> list:
    from sportvu.rebuild import plan_windows
    return plan_windows(section, budget_s=budget_s)


def step_direction(P: dict, tag: str, section: str) -> dict:
    """sportvu.direction's per-window loop for one section (cut, V3 build, window time map)."""
    from clock_reader import ClockReader
    from sportvu.direction import BUDGET_S
    from sportvu.local_sync import build_local_timemap
    from sportvu.rebuild import build, cut
    game_of(tag)
    inject_layouts()
    reader = make_reader(layout_of(P, tag))
    done, failures = 0, []
    for w in _windows(P, section, BUDGET_S):
        snip = safe_cut(P, w, P["direction"])
        r = safe_build(P, w, snip, P["direction"])
        if not r["ok"]:
            failures.append(w["window"]); continue
        if r.get("empty"):
            continue
        write_timemap(P["direction"] / (w["window"] + "_timemap.json"), w["section"], reader, snip)
        done += (P["direction"] / (w["window"] + "_trajectories.json")).exists()
    fail_if(failures, section)
    return {"windows_built": done}


def step_mirror(P: dict, tag: str) -> dict:
    from sportvu.direction import BUDGET_S, solve, window_rows
    from sportvu.sync import SportVUIndex
    g = game_of(tag)
    rows = {}
    for s in sections(P, tag):
        if not (P["sync"] / f"{s}_local_timemap.json").exists():
            continue
        for w in _windows(P, s, BUDGET_S):
            tmp, trp = (P["direction"] / (w["window"] + k) for k in ("_timemap.json", "_trajectories.json"))
            if tmp.exists() and trp.exists():
                rows[w["window"]] = window_rows(w, json.loads(tmp.read_text()), json.loads(trp.read_text()))
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    rep = {"game": tag, "sportvu_game": g["sportvu"], "budget_s": BUDGET_S, **solve(rows, index)}
    atomic_write(P["reports"] / f"sportvu_direction_{tag}.json", json.dumps(rep, indent=1, default=str))
    return {"status": rep["status"], **{k: (rep.get("resolution") or {}).get(k) for k in ("mirror", "vote_share")}}


def label_windows(P: dict, tag: str, section: str | None = None) -> list:
    """autolabel's plan: every running span of the section's (or game's) local time maps."""
    from sportvu.autolabel import EVERY_SPAN
    secs = [section] if section else [s for s in sections(P, tag) if (P["sync"] / f"{s}_local_timemap.json").exists()]
    return [w for s in secs for w in _windows(P, s, EVERY_SPAN)]


def step_label(P: dict, tag: str, section: str) -> dict:
    """autolabel phase A for one section: cut, V3 build with gate v2, window time map."""
    from clock_reader import ClockReader
    from sportvu.autolabel import gate_args
    from sportvu.local_sync import build_local_timemap
    from sportvu.rebuild import build, cut
    game_of(tag)
    inject_layouts()
    reader = make_reader(layout_of(P, tag))
    out, failures = {"windows": 0, "built": 0}, []
    for w in label_windows(P, tag, section):
        snip = safe_cut(P, w, P["label_build"])
        r = safe_build(P, w, snip, P["label_build"], extra_args=gate_args())
        if not r["ok"]:
            failures.append(w["window"]); continue
        out["windows"] += 1
        if r.get("empty"):
            out["gate_empty"] = out.get("gate_empty", 0) + 1; continue
        write_timemap(P["label_build"] / (w["window"] + "_timemap.json"), w["section"], reader, snip)
        out["built"] += 1
    fail_if(failures, section)
    return out


def step_fit(P: dict, tag: str, partial: bool = False) -> dict:
    """autolabel phases B and C (camera position, per-frame camera truth, acceptance, labels) for one game, with the
    frozen constants and functions; plus the gold-key paint check on GSW home games."""
    import cv2
    import numpy as np
    from qc.track_qc import line_overlay
    from sportvu import autolabel as A
    from sportvu import camera as cam
    from sportvu import label_check as LC
    from sportvu.sync import SportVUIndex
    from sportvu.truth import window_truth
    from videoseq import SeqReader
    g = game_of(tag)
    mirror = mirror_of(P, tag)
    mom = json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())
    index = SportVUIndex(mom["moments"])
    jersey = {k: v.get("jersey") for k, v in mom.get("players", {}).items()}
    counts, regions, built = {}, {}, []
    secs = [s for s in sections(P, tag) if (P["sync"] / f"{s}_local_timemap.json").exists()]
    unlabelled = [s for s in secs if not done_path(P, "label", tag, s).exists()]
    if unlabelled and not partial:
        raise SystemExit(f"{WHO}: fit needs every synced section labelled first; missing {unlabelled}")
    if partial:
        secs = [s for s in secs if s not in unlabelled]
    windows = [w for s in secs for w in label_windows(P, tag, s)]
    for w in windows:
        side_p, tm_p = (P["label_build"] / (w["window"] + k) for k in ("_frames.json", "_timemap.json"))
        if not side_p.exists() or not tm_p.exists():
            key = "gate_v2_empty_window" if gate_empty_path(P["label_build"], w["window"]).exists() else "build_failed"
            counts[key] = counts.get(key, 0) + 1
            continue
        sc, tm = json.loads(side_p.read_text()), json.loads(tm_p.read_text())
        live = {int(f) for f, _, _ in tm["frames"]}
        counts["gate_v2_skipped"] = counts.get("gate_v2_skipped", 0) + len(live - {r["frame"] for r in sc["frames"]})
        off, _, rows = window_truth(w, sc, tm, index, mirror)
        for row in rows:
            if row["status"] != "ok":
                counts[row["status"]] = counts.get(row["status"], 0) + 1
        snip = P["label_build"] / (w["window"] + ".mp4")
        cap = cv2.VideoCapture(str(snip)); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        frs = cam.frames_from_rows(w, sc, rows, index, mirror, W, H)
        counts["untestable_inliers"] = counts.get("untestable_inliers", 0) + sum(1 for r in rows if r["status"] == "ok") - len(frs)
        built.append((w, snip, sc, off, frs))
    g_fit = cam.fit_game([f for *_, frs in built for f in frs])
    if g_fit.get("status") != "ok":
        res = {"game": tag, "status": "camera_fit_failed", "camera": g_fit, "rejections": counts}
        atomic_write(P["labels"] / f"{tag}_result.json", json.dumps(res, indent=1))
        raise SystemExit(f"{WHO}: {tag} camera fit failed: {g_fit}")
    C = np.array(g_fit["C"])
    lo, hi = np.log(cam.F_BOUNDS_PX[0]) + 0.01, np.log(cam.F_BOUNDS_PX[1]) - 0.01
    paint_on = g.get("home") == "GSW" or tag in R13_TAGS          # the gold key exists only at GSW (label_check)
    paint, accepted = [], 0
    (P["labels"] / tag).mkdir(parents=True, exist_ok=True)
    for w, snip, sc, off, frs in built:
        srow = {r["frame"]: r for r in sc["frames"]}
        sr = SeqReader(cv2.VideoCapture(str(snip)))
        lines = []
        for fr in frs:
            ok_img, img = sr.read(fr["frame"])
            fit = cam.refine_lines(C, fr, cam.fit_frame(C, fr), img) if ok_img else {"status": "no_image"}
            if fit.get("status") != "ok":
                rule = "camera_no_init"
            elif fit.get("lines", 0) < A.MIN_LINE_MATCHES:
                rule = "line_matches_low"
            elif not (lo < np.log(fit["f"]) < hi):
                rule = "zoom_at_bound"
            elif fit["resid_ft"] > A.MAX_FEET_RESID_FT:
                rule = "feet_resid_high"
            else:
                sup = line_overlay(img, fit["H"], fr["boxes"])["support"]
                rule = None if sup is not None and sup >= A.MIN_SUPPORT else "support_low"
            if rule:
                counts[rule] = counts.get(rule, 0) + 1
                continue
            row = {**fr["row"], "H_truth": fit["H"]}
            q, i = row["q"], row["moment"]
            lab = A.frame_labels(row, srow[row["frame"]], index.q[q]["xy"][i], index.q[q]["pid"][i],
                                 index.q[q]["team"][i], jersey, mirror, fr["w"], fr["h"])
            lines.append(json.dumps({"game": tag, "window": w["window"], "frame": row["frame"], "section": w["section"],
                                     "section_frame": w["f_start"] + row["frame"], "q": q, "clock": row["clock"],
                                     "moment": int(i), "offset_s": off, "H_truth": fit["H"], "truth": "camera",
                                     "camera_C": g_fit["C"], "camera_f": fit["f"], "camera_rvec": list(map(float, fit["rvec"])),
                                     "feet_resid_ft": fit["resid_ft"], "line_matches": fit.get("lines"), **lab}))
            if paint_on:
                pr = {"window": w["window"], "frame": row["frame"], **LC.key_edges(img, fit["H"])}
                pr["reject"] = LC.verdict(pr); pr["keep"] = pr["reject"] is None
                paint.append(json.dumps(pr))
            counts["accepted"] = counts.get("accepted", 0) + 1
            regions[lab["region"]] = regions.get(lab["region"], 0) + 1
        atomic_write(P["labels"] / tag / (w["window"] + ".jsonl"), "\n".join(lines) + ("\n" if lines else ""))
        accepted += len(lines)
    if paint_on:
        atomic_write(P["labels"] / f"{tag}_paint.jsonl", "\n".join(paint) + ("\n" if paint else ""))
    res = {"game": tag, "sportvu": g["sportvu"], "home": g.get("home"), "mirror": mirror, "windows": len(windows),
           "camera": {k: v for k, v in g_fit.items() if k != "cost"}, "accepted": accepted, "regions": regions,
           "paint_check": paint_on, "rejections": {k: v for k, v in counts.items() if k != "accepted"},
           "sections": secs, "partial": partial}
    # every built window's residual clock offset (also windows without an accepted frame), for the event rows
    atomic_write(P["labels"] / f"{tag}_offsets.json", json.dumps({w["window"]: off for w, _, _, off, _ in built}, indent=1))
    atomic_write(P["labels"] / f"{tag}_result.json", json.dumps(res, indent=1))
    return {k: res[k] for k in ("accepted", "regions", "paint_check", "sections", "partial")}


def step_rebuild(P: dict, tag: str, section: str, build_name: str) -> dict:
    """The section's label windows built again with another court model; clips and time maps are shared."""
    from sportvu.autolabel import gate_args
    game_of(tag)
    weights = court_weights(build_name)
    out = P["builds"] / build_name
    out.mkdir(parents=True, exist_ok=True)
    os.environ["COURT_GRID_WEIGHTS"] = str(weights)       # read by build_trajectories' config at import
    if not done_path(P, "label", tag, section).exists():
        raise SystemExit(f"{WHO}: rebuild {section} needs its label step first")
    res, failures = {"windows": 0, "built": 0}, []
    for w in label_windows(P, tag, section):
        snip = P["label_build"] / (w["window"] + ".mp4")
        if not snip.exists():
            continue
        r = safe_build(P, w, snip, out, extra_args=gate_args())
        if not r["ok"]:
            failures.append(w["window"]); continue
        res["windows"] += 1
        if r.get("empty"):
            res["gate_empty"] = res.get("gate_empty", 0) + 1; continue
        res["built"] += 1
    fail_if(failures, f"{section} ({build_name})")
    return res


def atomic_write_mk(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, text)


def step_sweep(P: dict, tag: str) -> dict:
    """Move any frozen-build leftover of this game's windows out of the checkout (start and end of a night run)."""
    return {"moved": sweep_checkout(P, f"{tag}_s[0-9][0-9]_w[0-9][0-9]")}


STEPS = {"sweep": step_sweep, "link": step_link, "split": step_split, "moments": step_moments, "layout": step_layout, "sync": step_sync,
         "direction": step_direction, "mirror": step_mirror, "label": step_label, "fit": step_fit,
         "rebuild": step_rebuild}
PER_SECTION = ("sync", "direction", "label", "rebuild")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("step", choices=sorted(STEPS))
    ap.add_argument("tag")
    ap.add_argument("--section", default=None)
    ap.add_argument("--build", default="r23")
    ap.add_argument("--partial", action="store_true", help="fit: use only the sections labelled so far (a smoke run)")
    a = ap.parse_args(argv)
    game_of(a.tag)
    refuse_cv_test([a.tag, a.section or "", a.build], WHO)
    P = install_paths()
    t0 = time.time()
    args = [P, a.tag]
    if a.step in PER_SECTION:
        if not a.section or not a.section.startswith(a.tag + "_s"):
            raise SystemExit(f"{WHO}: {a.step} needs --section {a.tag}_sNN")
        args.append(a.section)
    if a.step == "rebuild":
        args.append(a.build)
    if a.step == "fit":
        args.append(a.partial)
    res = STEPS[a.step](*args)
    atomic_write_mk(done_path(P, a.step, a.tag, a.section, a.build if a.step == "rebuild" else None),
                    json.dumps({"step": a.step, "tag": a.tag, "section": a.section, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "result": res}, default=str))
    print(json.dumps({"step": a.step, "tag": a.tag, "section": a.section, "seconds": round(time.time() - t0, 1), **res}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
