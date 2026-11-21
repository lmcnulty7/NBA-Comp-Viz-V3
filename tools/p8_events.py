"""tools/p8_events.py: P8-E2, shots and rebounds from the play-by-play and the clock (docs/research/P8_E2.md,
pre-registered in 9f1256e1; docs/research/P8_FEASIBILITY.md sections 6.2 and 8).

    python -m tools.p8_events place                      coverage, gaps and the e2 / e2_m1 builds (light)
    python -m tools.p8_events physical                   row C game-time part from the frozen SportVU tables
    python -m tools.p8_events score TAG --truth camera   rows A (camera) or B (b4) through the pinned scorer
    python -m tools.p8_events report                     pooled rows, intervals, reports/research/p8_e2.json

The method places every play-by-play field-goal attempt and rebound at the video frame whose OCR clock equals the
play-by-play clock (M0), or that clock less one constant lag taken from the other R1.3 games' Colab camera truth
(M1, leave one game out). Spans of the time map whose clock runs backward in game time are dropped first (the
chain rule). Nothing is fitted or trained. The scorer (tools/p8_scorecard.py, pinned by D88) is imported and run
unchanged; its score_window is wrapped at run time only to keep the per-window tallies it already returns, so the
window bootstrap can be drawn. Every tag, SportVU id and video id passes refuse_cv_test and the spent held-out guard.
Outputs go under the P8 workspace (builds/e2, builds/e2_m1, scorecards/, reports/p8_e2/) and one summary without
local paths to reports/research/p8_e2.json.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, dev_games, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_events"
APART = ("gsw_ind_2016",)                 # D98: reported, kept out of pooled rows
KINDS = ("shot", "rebound")
PBP_KINDS = {"shot": ("shot_made", "shot_missed"), "rebound": ("rebound",)}
RUN_TOL_S = 1.0                           # sportvu.local_sync.RUN_TOL_S (checked equal in the tests)
MARGIN_S = 0.5                            # tools.p8_scorecard CONSTANTS["event_live_wide_s"] (checked in the tests)
TOL_S = 1.0                               # tools.p8_scorecard CONSTANTS["event_tol_s"]
GAP_CLASSES = ("period not mapped", "between sections", "video skips game time", "clock ran unread")
BUILDS = {"M0": "e2", "M1": "e2_m1"}
SUMMARY = REPO / "reports" / "research" / "p8_e2.json"


# ── pure helpers (unit tested) ──────────────────────────────────────────────────────────────────────────────────────
def game_s(period: int, clock: float) -> float:
    """Game seconds elapsed at (period, clock remaining): 12-minute quarters, 5-minute overtimes."""
    p = int(period)
    if p <= 4:
        return (p - 1) * 720.0 + (720.0 - clock)
    return 2880.0 + (p - 5) * 300.0 + (300.0 - clock)


def can_follow(a: dict, b: dict, tol: float = RUN_TOL_S) -> bool:
    """Span b may come after span a in video order without game time running backward."""
    return b["period"] > a["period"] or (b["period"] == a["period"] and b["c0"] <= a["c1"] + tol)


def chain(spans: list, tol: float = RUN_TOL_S) -> list:
    """Indices (into spans, which must be in video order) of the chain of largest total video_s in which every
    span may follow the one before it. Ties keep the earliest predecessor."""
    n = len(spans)
    best, prev = [0.0] * n, [-1] * n
    for i in range(n):
        best[i] = float(spans[i]["video_s"])
        for j in range(i):
            if can_follow(spans[j], spans[i], tol) and best[j] + spans[i]["video_s"] > best[i]:
                best[i], prev[i] = best[j] + spans[i]["video_s"], j
    if not n:
        return []
    k = int(np.argmax(best))
    out = []
    while k >= 0:
        out.append(k)
        k = prev[k]
    return out[::-1]


def span_holds(span: dict, period: int, clock: float, margin: float = MARGIN_S) -> bool:
    return span["period"] == period and span["c1"] - margin <= clock <= span["c0"] + margin


def nearest_frame(frames: list, period: int, clock: float, lo: int, hi: int):
    """The time map frame [f, p, c] with lo <= f <= hi, the same period and the nearest clock; None if none."""
    best, bd = None, None
    for f, p, c in frames:
        if lo <= f <= hi and p == period:
            d = abs(c - clock)
            if bd is None or d < bd:
                best, bd = int(f), d
    return best


def classify(period: int, clock: float, chained: list, margin: float = MARGIN_S):
    """None when a chained span holds the event; else its gap class (P8_E2.md section 1.5). chained: spans with
    section, f0, f1, fps, period, c0, c1, in game order."""
    if any(span_holds(s, period, clock, margin) for s in chained):
        return None
    if not any(s["period"] == period for s in chained):
        return "period not mapped"
    g = game_s(period, clock)
    before = [s for s in chained if game_s(s["period"], s["c1"]) <= g]
    after = [s for s in chained if game_s(s["period"], s["c0"]) >= g]
    if not before or not after or before[-1]["section"] != after[0]["section"]:
        return "between sections"
    a, b = before[-1], after[0]
    dg = game_s(b["period"], b["c0"]) - game_s(a["period"], a["c1"])
    dv = (b["f0"] - a["f1"]) / a["fps"]
    return "video skips game time" if dv < dg - RUN_TOL_S else "clock ran unread"


def gap_seconds(chained: list) -> dict:
    """Video and game seconds between consecutive chained spans of the same section, by kind of gap."""
    out = {"dead_ball_video_s": 0.0, "unread_video_s": 0.0, "unread_game_s": 0.0, "skipped_game_s": 0.0,
           "mapped_game_s": round(sum(s["c0"] - s["c1"] for s in chained), 1),
           "mapped_video_s": round(sum((s["f1"] - s["f0"]) / s["fps"] for s in chained), 1)}
    for a, b in zip(chained, chained[1:]):
        if a["section"] != b["section"]:
            continue
        dg = game_s(b["period"], b["c0"]) - game_s(a["period"], a["c1"])
        dv = (b["f0"] - a["f1"]) / a["fps"]
        if dg <= RUN_TOL_S:
            out["dead_ball_video_s"] += dv
        elif dv < dg - RUN_TOL_S:
            out["skipped_game_s"] += dg - max(dv, 0.0)
        else:
            out["unread_video_s"] += dv
            out["unread_game_s"] += dg
    return {k: round(v, 1) for k, v in out.items()}


def place_window(tm: dict, events: list, lag: float = 0.0, margin: float = MARGIN_S) -> list:
    """[{frame, kind, event}] for the events [(event, kind, period, clock)] that a span of the window's own time map
    holds at clock - lag; the frame is that span's nearest time map frame."""
    out = []
    for ev, kind, p, c in events:
        c2 = c - lag
        for s in tm["spans"]:
            if span_holds(s, p, c2, margin):
                f = nearest_frame(tm["frames"], p, c2, s["f0"], s["f1"])
                if f is not None:
                    out.append({"frame": f, "kind": kind, "event": int(ev)})
                    break
    return out


def leave_one_out_lag(offsets_by_game: dict, tag: str):
    """Median of the per-window offsets of every other game; None when there are none."""
    vals = [v for g, vs in offsets_by_game.items() if g != tag for v in vs]
    return round(float(np.median(vals)), 3) if vals else None


def window_tallies(truth: list, pred: list, tol: float = TOL_S) -> list:
    """Observations of one window for the bootstrap: ('t', found) per truth event and ('f', 1) per false event,
    through the pinned scorer's match_events."""
    from tools.p8_scorecard import match_events
    found, false, pairs = match_events(truth, pred, tol)
    hit = {i for i, _, _ in pairs}
    return [("t", int(i in hit)) for i in range(len(truth))] + [("f", 1)] * false


def found_share(obs: list):
    t = [x for k, x in obs if k == "t"]
    return (sum(t) / len(t)) if t else None


def false_per_true(obs: list):
    t = sum(1 for k, _ in obs if k == "t")
    return (sum(1 for k, _ in obs if k == "f") / t) if t else None


def share_within(obs: list):
    """Share of 1s in a list of 0/1 observations (row C)."""
    return (sum(obs) / len(obs)) if obs else None


# ── inputs ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def games_with_maps(P: dict) -> dict:
    """Development games, guarded, split into those with section time maps and those without."""
    every = dev_games()
    ids = [x for t, g in every.items() for x in (t, g["sportvu"], g["video_id"])]
    refuse_cv_test(ids, WHO, homes=[g.get("home") for g in every.values() if g.get("home")])
    guard(ids, WHO)
    have = {t: g for t, g in every.items() if sorted(P["sync"].glob(f"{t}_s[0-9][0-9]_local_timemap.json"))}
    return {"run": have, "no_time_map": sorted(set(every) - set(have))}


def pbp_events(sv_game: str) -> dict:
    """{kind: [(event, kind, period, clock_s)]} of the contract's play-by-play (scoring truth only, D84)."""
    import pandas as pd
    guard([sv_game], WHO)
    d = pd.read_parquet(Path(os.environ["DATA_ROOT"]) / "sportvu_contract" / f"{sv_game}_pbp.parquet",
                        columns=["event", "period", "clock_s", "kind", "frame", "frame_clock_diff_s"])
    out = {k: [(int(e), k, int(p), float(c)) for e, p, c, kd in
               d[["event", "period", "clock_s", "kind"]].itertuples(index=False) if kd in PBP_KINDS[k]] for k in KINDS}
    fg = d[d["kind"].isin(PBP_KINDS["shot"] + PBP_KINDS["rebound"])]
    diff = fg["frame_clock_diff_s"].dropna().abs()
    out["_contract_frame"] = {"events": int(len(fg)), "with_frame": int((fg["frame"] >= 0).sum()),
                              "abs_clock_diff_p50_s": round(float(diff.median()), 3) if len(diff) else None,
                              "abs_clock_diff_max_s": round(float(diff.max()), 3) if len(diff) else None}
    return out


def section_spans(P: dict, tag: str) -> list:
    """Every running span of the game's section time maps, in video order, with its section and fps."""
    out = []
    for p in sorted(P["sync"].glob(f"{tag}_s[0-9][0-9]_local_timemap.json")):
        tm = json.loads(p.read_text())
        for s in tm["spans"]:
            out.append({**s, "section": tm["section"], "fps": float(tm["fps"])})
    return out


def colab_offsets(tags) -> dict:
    """{tag: [per-window offset_s]} of the R1.3 Colab camera truth in the main checkout (training games only)."""
    root = main_checkout() / "data" / "sportvu" / "labels"
    out = {}
    for tag in tags:
        guard([tag], WHO)
        d = root / tag
        vals = []
        for p in sorted(d.glob(f"{tag}_s[0-9][0-9]_w[0-9][0-9].jsonl")) if d.exists() else []:
            line = next((l for l in p.read_text().splitlines() if l.strip()), None)
            if line is not None and json.loads(line).get("offset_s") is not None:
                vals.append(float(json.loads(line)["offset_s"]))
        out[tag] = vals
    return out


def offset_summary(vals: list) -> dict:
    a = np.abs(np.array(vals, float))
    return {"windows": len(vals), "median_s": round(float(np.median(vals)), 3) if vals else None,
            "share_abs_le_1s": round(float((a <= TOL_S).mean()), 3) if vals else None}


# ── place: coverage, gaps and the e2 builds ─────────────────────────────────────────────────────────────────────────
def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        dst.unlink()
    os.symlink(src, dst)


def place_game(P: dict, tag: str, g: dict, lags: dict) -> dict:
    from tools import p8_frozen as F
    guard([tag, g["sportvu"], g["video_id"]], WHO)
    ev = pbp_events(g["sportvu"])
    spans = section_spans(P, tag)
    keep = chain(spans)
    chained = [spans[i] for i in keep]
    on_chain = {(s["section"], s["f0"]) for s in chained}
    cov = {"sections": len({s["section"] for s in spans}), "spans": len(spans), "chained_spans": len(chained),
           "dropped_spans": [{k: s[k] for k in ("section", "f0", "period", "c0", "c1", "video_s")}
                             for i, s in enumerate(spans) if i not in set(keep)],
           "gaps": gap_seconds(chained), "contract_frame": ev["_contract_frame"], "events": {}}
    placed_span = {}
    for k in KINDS:
        cls = {c: 0 for c in GAP_CLASSES}
        n_placed = 0
        for e, _, p, c in ev[k]:
            r = classify(p, c, chained)
            if r is None:
                n_placed += 1
                placed_span[e] = next(f"{s['section']}@{s['f0']}" for s in chained if span_holds(s, p, c))
            else:
                cls[r] += 1
        cov["events"][k] = {"pbp": len(ev[k]), "placed_in_chained_span": n_placed,
                            "placed_share": round(n_placed / len(ev[k]), 3) if ev[k] else None, "gaps": cls}
    # the builds the scorer reads: every label window gets sidecar links and an events file (empty off the chain)
    windows = F.label_windows(P, tag)
    wrote = {m: 0 for m in BUILDS}
    for m, build in BUILDS.items():
        lag = 0.0 if m == "M0" else lags.get(tag)
        bd = P["builds"] / build
        bd.mkdir(parents=True, exist_ok=True)
        for w in windows:
            stem = w["window"]
            for suf in ("_frames.json", "_identity.json"):
                src = P["builds"] / "r23" / (stem + suf)
                src = src if src.exists() else P["label_build"] / (stem + suf)
                if src.exists():
                    _link(src, bd / (stem + suf))
            tm_p = P["label_build"] / (stem + "_timemap.json")
            evs = []
            if lag is not None and tm_p.exists() and (w["section"], w["f0"]) in on_chain:
                tm = json.loads(tm_p.read_text())
                evs = place_window(tm, [x for k in KINDS for x in ev[k]], lag)
            atomic_write(bd / (stem + "_events.json"), json.dumps({"method": m, "lag_s": lag, "events": evs}))
            wrote[m] += len(evs)
    cov["label_windows"] = len(windows)
    cov["label_windows_on_chain"] = sum((w["section"], w["f0"]) in on_chain for w in windows)
    cov["placements_written"] = wrote
    cov["lag_m1_s"] = lags.get(tag)
    return cov, placed_span


def cmd_place(P: dict) -> dict:
    gm = games_with_maps(P)
    r13 = [t for t, g in gm["run"].items() if g.get("source") == "R1.3"]
    colab = colab_offsets(r13)
    lags = {t: leave_one_out_lag(colab, t) for t in r13}
    out = {"date": datetime.date.today().isoformat(), "games": {}, "no_time_map": gm["no_time_map"],
           "colab_offsets": {t: offset_summary(v) for t, v in colab.items()}, "lags_m1_s": lags}
    spans_of = {}
    for tag, g in gm["run"].items():
        cov, placed_span = place_game(P, tag, g, lags)
        out["games"][tag] = cov
        spans_of[tag] = placed_span
        e = cov["events"]
        print("  %-14s spans %3d chained %3d  shots placed %3d/%3d  rebounds placed %3d/%3d" % (
            tag, cov["spans"], cov["chained_spans"], e["shot"]["placed_in_chained_span"], e["shot"]["pbp"],
            e["rebound"]["placed_in_chained_span"], e["rebound"]["pbp"]), flush=True)
    d = P["reports"] / "p8_e2"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "place.json", json.dumps(out, indent=1))
    atomic_write(d / "placed_spans.json", json.dumps(spans_of))
    return out


# ── physical: row C game-time part ──────────────────────────────────────────────────────────────────────────────────
def physical_clocks(sv_game: str) -> dict:
    """{kind: {pbp event: game clock of the frozen SportVU physical frame}} (shot release, rebound secured)."""
    import pandas as pd
    from analytics.shot_frames import season_of_game
    guard([sv_game], WHO)
    D = Path(os.environ["DATA_ROOT"])
    season = season_of_game(sv_game)
    fr = pd.read_parquet(D / "sportvu_contract" / f"{sv_game}.parquet", columns=["frame", "game_clock_s"])
    clock = fr.drop_duplicates("frame").set_index("frame")["game_clock_s"]
    del fr
    shots = pd.read_parquet(D / "analytics" / "shot_frames" / season / f"{sv_game}.parquet", columns=["event", "release_frame"])
    rebs = pd.read_parquet(D / "analytics" / "rebound_frames" / season / f"{sv_game}.parquet", columns=["rebound_event", "rebound_frame"])
    out = {"shot": {}, "rebound": {}}
    for e, f in shots.itertuples(index=False):
        if f >= 0 and f in clock.index:
            out["shot"][int(e)] = float(clock.loc[f])
    for e, f in rebs.itertuples(index=False):
        if pd.notna(e) and f >= 0 and f in clock.index:
            out["rebound"][int(e)] = float(clock.loc[f])
    return out


def cmd_physical(P: dict) -> dict:
    gm = games_with_maps(P)
    spans_of = json.loads((P["reports"] / "p8_e2" / "placed_spans.json").read_text())
    out = {}
    for tag, g in gm["run"].items():
        ev = pbp_events(g["sportvu"])
        ph = physical_clocks(g["sportvu"])
        rows = {}
        for k in KINDS:
            obs = []
            for e, _, p, c in ev[k]:
                span = spans_of[tag].get(str(e))
                if span is not None and e in ph[k]:
                    obs.append({"event": e, "span": span, "d_s": round(c - ph[k][e], 3)})
            placed = sum(1 for e, *_ in ev[k] if str(e) in spans_of[tag])
            rows[k] = {"placed": placed, "with_physical_frame": len(obs), "obs": obs}
        out[tag] = {"physical_by_event": {k: {str(e): v for e, v in ph[k].items()} for k in KINDS}, "rows": rows}
        print("  %-14s shots with release %d of %d placed, rebounds with frame %d of %d placed" % (
            tag, rows["shot"]["with_physical_frame"], rows["shot"]["placed"],
            rows["rebound"]["with_physical_frame"], rows["rebound"]["placed"]), flush=True)
    atomic_write(P["reports"] / "p8_e2" / "physical.json", json.dumps(out))
    return out


# ── score: rows A and B through the pinned scorer ───────────────────────────────────────────────────────────────────
def cmd_score(P: dict, tag: str, mode: str) -> dict:
    from tools import p8_scorecard as S
    refuse_cv_test([tag], WHO)
    gm = games_with_maps(P)
    if tag not in gm["run"]:
        raise SystemExit(f"{WHO}: {tag} is not a development game with time maps")
    out = {"game": tag, "truth": mode, "methods": {}}
    for m, build in BUILDS.items():
        if not (P["builds"] / build).exists():
            raise SystemExit(f"{WHO}: run place first ({build} is missing)")
        seen = []
        orig = S.score_window

        def spy(*a, **k):
            r = orig(*a, **k)
            seen.append((a[2]["window"], r[3]))
            return r
        S.score_window = spy
        try:
            rep = S.score_game(tag, build, mode)
        finally:
            S.score_window = orig
        per_w = {}
        for stem, v2 in seen:
            if not v2:
                continue
            per_w[stem] = {"offset_s": v2.get("window_offset_s"),
                           "truth": {k: v2.get("events_truth", {}).get(k, []) for k in KINDS},
                           "pred": {k: v2.get("events_pred", {}).get(k, []) for k in KINDS}}
        out["methods"][m] = {"build": build, "scorecard": f"scorecards/{tag}__{build}__{mode}.json",
                             "v2_rows": {k: rep["v2_rows"][k] for k in ("shots", "rebounds")},
                             "frozen_rows_note": "the e2 builds link r23 (else V3) sidecars; their frozen rows are not an E2 result",
                             "per_window": per_w}
        r = rep["v2_rows"]
        print("  %-14s %-6s %-5s shots found %s false %s (truth %d)  rebounds found %s (truth %d)" % (
            tag, mode, m, r["shots"]["found_share"], r["shots"]["false_per_true"],
            r["shots"]["truth_events_near_live_wide_frames"], r["rebounds"]["found_share"],
            r["rebounds"]["truth_events_near_live_wide_frames"]), flush=True)
    d = P["reports"] / "p8_e2"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"score_{tag}_{mode}.json", json.dumps(out))
    return out


# ── report: pooled rows and intervals ───────────────────────────────────────────────────────────────────────────────
def _row(groups_w: dict, groups_g: dict, kind: str) -> dict:
    from tools.p8_scorecard import boot_windows
    obs = [o for v in groups_w.values() for o in v]
    r = {"truth_events": sum(1 for k, _ in obs if k == "t"), "found_share": _r(found_share(obs)),
         "found_share_window_ci": boot_windows(groups_w, found_share),
         "found_share_game_ci": boot_windows(groups_g, found_share) if len(groups_g) >= 2 else None}
    if kind == "shot":
        r["false_per_true"] = _r(false_per_true(obs))
        r["false_per_true_window_ci"] = boot_windows(groups_w, false_per_true)
        r["false_per_true_game_ci"] = boot_windows(groups_g, false_per_true) if len(groups_g) >= 2 else None
        r["pass"] = None if r["found_share"] is None else bool(r["found_share"] >= 0.95 and r["false_per_true"] <= 0.05)
    else:
        r["pass"] = None if r["found_share"] is None else bool(r["found_share"] >= 0.90)
    return r


def _r(x):
    return None if x is None else round(float(x), 3)


def scored_rows(score_files: dict, tags: list, m: str) -> dict:
    """Row A or B for one method over the given games: window and game groups of match observations."""
    rows = {}
    for kind in KINDS:
        gw, gg = {}, {}
        for tag in tags:
            sc = score_files.get(tag)
            if sc is None:
                continue
            gobs = []
            for stem, w in sc["methods"][m]["per_window"].items():
                o = window_tallies([tuple(x) for x in w["truth"][kind]], [tuple(x) for x in w["pred"][kind]])
                gw[stem] = o
                gobs += o
            gg[tag] = gobs
        rows[kind] = _row(gw, gg, kind)
        rows[kind]["games"] = [t for t in tags if t in score_files]
    return rows


def offsets_of(sc: dict) -> list:
    return [w["offset_s"] for w in sc["methods"]["M0"]["per_window"].values() if w["offset_s"] is not None]


def full_error_rows(sc: dict, ev: dict, phys: dict, lag) -> dict:
    """Row C full error on scored windows: placement's true clock less the physical frame's clock, per truth event
    the scorer counted, joined to a unique play-by-play event of that (period, clock, kind)."""
    out = {}
    for kind in KINDS:
        key = {}
        for e, _, p, c in ev[kind]:
            key.setdefault((p, c), []).append(e)
        obs, dup, nophys = [], 0, 0
        for stem, w in sc["methods"]["M0"]["per_window"].items():
            off = w["offset_s"]
            if off is None:
                continue
            for p, c in w["truth"][kind]:
                es = key.get((int(p), float(c)), [])
                if len(es) != 1:
                    dup += 1
                    continue
                ph = phys[kind].get(str(es[0]))
                if ph is None:
                    nophys += 1
                    continue
                for m, L in (("M0", 0.0), ("M1", lag)):
                    if L is not None:
                        obs.append((m, stem, c - L + off - ph))
        out[kind] = {}
        for m in ("M0", "M1"):
            g = {}
            for mm, stem, d in obs:
                if mm == m:
                    g.setdefault(stem, []).append(int(abs(d) <= TOL_S))
            from tools.p8_scorecard import boot_windows
            flat = [x for v in g.values() for x in v]
            out[kind][m] = {"n": len(flat), "share_within_1s": _r(share_within(flat)),
                            "window_ci": boot_windows(g, share_within)}
        out[kind]["skipped_not_unique"] = dup
        out[kind]["skipped_no_physical_frame"] = nophys
    return out


def cmd_report(P: dict) -> dict:
    from tools.p8_scorecard import boot_windows
    d = P["reports"] / "p8_e2"
    place = json.loads((d / "place.json").read_text())
    phys = json.loads((d / "physical.json").read_text())
    tags = list(place["games"])
    pooled = [t for t in tags if t not in APART]
    res = {"experiment": "P8-E2", "pre_registration": "docs/research/P8_E2.md (9f1256e1)",
           "date": datetime.date.today().isoformat(), "games_run": tags, "pooled": pooled, "apart": list(APART),
           "no_time_map": place["no_time_map"], "lags_m1_s": place["lags_m1_s"],
           "colab_offsets": place["colab_offsets"], "targets": {"shots": ">= 0.95 found within 1.0 s, <= 0.05 false per true",
                                                                "rebounds": ">= 0.90 found within 1.0 s"},
           "coverage": place["games"], "rows": {}}
    for truth in ("camera", "b4"):
        sf = {t: json.loads((d / f"score_{t}_{truth}.json").read_text()) for t in tags if (d / f"score_{t}_{truth}.json").exists()}
        if not sf:
            continue
        R = {"scored_games": sorted(sf)}
        for m in BUILDS:
            R[m] = {"pooled": scored_rows(sf, pooled, m),
                    **{f"apart_{t}": scored_rows(sf, [t], m) for t in APART if t in sf},
                    **{f"game_{t}": scored_rows(sf, [t], m) for t in sf if t not in APART}}
        R["offsets"] = {t: offset_summary(offsets_of(sf[t])) for t in sf}
        if truth == "camera":
            R["row_c_full_error"] = {}
            for t in sf:
                ev = pbp_events(dev_games()[t]["sportvu"])
                R["row_c_full_error"][t] = full_error_rows(sf[t], ev, phys[t]["physical_by_event"], place["lags_m1_s"].get(t))
        res["rows"][truth] = R
    # row C game-time part: pooled over the pooled games, window (span) and game bootstrap
    C = {}
    for kind in KINDS:
        gw, gg = {}, {}
        for t in pooled:
            for o in phys[t]["rows"][kind]["obs"]:
                gw.setdefault(f"{t}:{o['span']}", []).append(int(abs(o["d_s"]) <= TOL_S))
                gg.setdefault(t, []).append(int(abs(o["d_s"]) <= TOL_S))
        flat = [x for v in gw.values() for x in v]
        ds = [o["d_s"] for t in pooled for o in phys[t]["rows"][kind]["obs"]]
        C[kind] = {"pooled": {"n": len(flat), "share_within_1s": _r(share_within(flat)),
                              "window_ci": boot_windows(gw, share_within), "game_ci": boot_windows(gg, share_within),
                              "pbp_minus_physical_p50_s": _r(np.median(ds)) if ds else None,
                              "abs_p90_s": _r(np.percentile(np.abs(ds), 90)) if ds else None,
                              "placed": sum(phys[t]["rows"][kind]["placed"] for t in pooled)},
                   **{t: {"n": len(phys[t]["rows"][kind]["obs"]), "placed": phys[t]["rows"][kind]["placed"],
                          "share_within_1s": _r(share_within([int(abs(o["d_s"]) <= TOL_S) for o in phys[t]["rows"][kind]["obs"]])),
                          "pbp_minus_physical_p50_s": _r(np.median([o["d_s"] for o in phys[t]["rows"][kind]["obs"]]))
                          if phys[t]["rows"][kind]["obs"] else None} for t in tags}}
    res["rows"]["physical_game_time"] = C
    # coverage pooled
    cov = {}
    for kind in KINDS:
        tot = sum(place["games"][t]["events"][kind]["pbp"] for t in pooled)
        pl = sum(place["games"][t]["events"][kind]["placed_in_chained_span"] for t in pooled)
        cls = {c: sum(place["games"][t]["events"][kind]["gaps"][c] for t in pooled) for c in GAP_CLASSES}
        cov[kind] = {"pbp": tot, "placed": pl, "placed_share": _r(pl / tot) if tot else None, "gaps": cls}
    res["coverage_pooled"] = cov
    from export.schemas import find_private
    text = json.dumps(res, indent=1, default=str)
    if find_private(text):
        raise SystemExit(f"{WHO}: the summary holds a private item: {find_private(text)[:3]}")
    atomic_write(SUMMARY, text + "\n")
    atomic_write(d / "results.json", text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("place")
    sub.add_parser("physical")
    s = sub.add_parser("score")
    s.add_argument("tag")
    s.add_argument("--truth", required=True, choices=["camera", "b4"])
    sub.add_parser("report")
    a = ap.parse_args(argv)
    from tools import p8_frozen as F
    P = F.install_paths(p8_root())
    if a.cmd == "place":
        cmd_place(P)
    elif a.cmd == "physical":
        cmd_physical(P)
    elif a.cmd == "score":
        cmd_score(P, a.tag, a.truth)
    else:
        cmd_report(P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
