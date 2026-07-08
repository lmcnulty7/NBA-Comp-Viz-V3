#!/usr/bin/env python
"""
align_outcomes.py — Tier 2 step 3: anchor-per-possession PBP alignment + outcome join.

For each metrics-eligible possession (C2 set cores, the certified region):
  1. Read TWO clock anchors from the scorebug (near set start, near core end —
     sparse, retryable OCR; eval'd at 100% on readable crops, 0 confident-wrong).
  2. Anchor sanity: same period, clock decreasing, and the clock-rate vs video
     frames must be plausible (game clock runs during live play; a mid-set foul
     pauses it, so tolerance is generous but a nonsensical pair is rejected).
  3. Window the game's PBP events between the two clocks (small slack before
     the set; larger slack after core end — the core trims the last 2 s and the
     possession-ending shot usually lands right at/after retreat onset).
  4. Outcome = offense-team points in the window + the terminating event
     (last shot attempt / turnover); which real team is on offense comes from
     the light-kit ↔ home-team mapping (pre-2017: home wears light).

C2 CROSS-VALIDATION (the acceptance gate for Tier 2): independently of our
geometry, the PBP says who was attempting shots in each aligned window. The
fraction of possessions where PBP's actor team == our predicted offense is a
ground-truth check of C2's offense/defense assignment on real outcomes.

Outputs: data/pbp/<clip>_outcomes.json + reports/tier2_crossval.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import logging

import cv2

import config
from clock_reader import CLIP_LAYOUT, ClockReader
from fetch_pbp import CLIP_GAME, GAMES, LIGHT_IS_HOME, PBP_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("align_outcomes")

CLOCK_RATE_TOL = 6.0    # |Δclock − Δvideo| tolerance (s) — fouls pause the clock


def reconstruct_possessions(events: list[dict]) -> list[dict]:
    """Split a game's PBP into possessions (the fix for window slack straddling
    two possessions and flipping the actor vote — first version failed exactly
    that way). Boundaries: turnover; defensive rebound; possession-ending made
    shot / final FT. And-one FTs stay with the shot's possession. Returns
    [{team, period, clock_start, clock_end, points, events}] — clock DESCENDS
    within a possession (clock_start > clock_end)."""
    poss, cur = [], None

    def close():
        nonlocal cur
        if cur and cur["events"]:
            cur["clock_end"] = cur["events"][-1]["clock_s"]
            cur["points"] = sum(e["points"] for e in cur["events"])
            poss.append(cur)
        cur = None

    def start(team, e):
        nonlocal cur
        cur = {"team": team, "period": e["period"], "clock_start": e["clock_s"], "events": []}

    for e in sorted(events, key=lambda e: (e["period"], -e["clock_s"])):
        if e["team"] is None:
            continue
        if cur is not None and e["period"] != cur["period"]:
            close()
        k = e["kind"]
        if k in ("shot_made", "shot_missed", "turnover"):
            if cur is None or e["team"] != cur["team"]:
                close()
                start(e["team"], e)
            cur["events"].append(e)
            if k == "turnover":
                close()
            # made shots: keep open briefly — an and-one FT by the same team at the
            # same clock belongs here; the next offensive action by anyone closes it
        elif k in ("ft_made", "ft_missed"):
            if cur is None or e["team"] != cur["team"]:
                close()
                start(e["team"], e)
            cur["events"].append(e)
        elif k == "rebound":
            if cur is not None and e["team"] != cur["team"]:
                close()          # defensive rebound = change of possession
            elif cur is not None:
                cur["events"].append(e)   # offensive board extends the possession
        elif cur is not None:
            cur["events"].append(e)       # fouls etc. ride along
    close()
    # PBP stamps events at the possession-ENDING moment, so a one-shot possession
    # would have zero clock width and nothing could overlap it. Possessions
    # partition the clock: each starts where the previous one ended (or at the
    # period start), ends at its own last event.
    prev_end: dict[int, float] = {}
    for p in poss:
        per = p["period"]
        p["clock_start"] = prev_end.get(per, 720.0 if per <= 4 else 300.0)
        prev_end[per] = p["clock_end"]
    return poss


def align_clip(clip: str, fps: float = 30.0):
    poss = json.loads((config.TRACKING_DIR / f"{clip}_possessions.json").read_text())
    ident = json.loads((config.TRACKING_DIR / f"{clip}_identity.json").read_text())
    game = GAMES[CLIP_GAME[clip]]
    pbp = json.loads((PBP_DIR / f"{CLIP_GAME[clip]}.json").read_text())["events"]
    pbp_poss = reconstruct_possessions(pbp)
    light = ident.get("light_team")
    # team id → real team name: light kit = home (pre-2017)
    def real(team_id):
        if team_id is None or light is None:
            return None
        return game["home"] if ((team_id == light) == LIGHT_IS_HOME) else game["away"]

    reader = ClockReader(CLIP_LAYOUT[clip])
    src = config.VIDEO_DIR / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(src))

    out, n_anchor_fail = [], 0
    for span in poss["spans"]:
        if span.get("kind") != "halfcourt" or not span.get("metrics_eligible"):
            continue
        a1 = reader.anchor(cap, span["set_start_frame"])
        a2 = reader.anchor(cap, span["core_end_frame"])
        rec = {k: span[k] for k in ("set_start_frame", "core_end_frame", "attacked_basket",
                                    "offense_team", "confidence")}
        rec["offense_real"] = real(span.get("offense_team"))
        rec["defense_real"] = real(span.get("defense_team"))
        if not a1 or not a2 or a1["period"] != a2["period"]:
            rec["status"] = "anchor_failed"
            n_anchor_fail += 1
            out.append(rec)
            continue
        # clock-rate sanity: video advanced Δf frames; clock should have dropped ≈ Δf/fps
        dv = (a2["frame"] - a1["frame"]) / fps
        dc = a1["clock_s"] - a2["clock_s"]
        if dc < 0 or abs(dc - dv) > CLOCK_RATE_TOL:
            rec.update({"status": "anchor_inconsistent",
                        "anchors": [a1, a2], "d_video_s": round(dv, 1), "d_clock_s": round(dc, 1)})
            n_anchor_fail += 1
            out.append(rec)
            continue
        period = a1["period"]
        # match the anchor range [a2.clock, a1.clock] to the PBP possession with
        # maximal clock overlap — no slack heuristics, possessions partition time
        best, best_ov = None, 0.0
        for p in pbp_poss:
            if p["period"] != period:
                continue
            ov = min(a1["clock_s"], p["clock_start"]) - max(a2["clock_s"], p["clock_end"])
            if ov > best_ov:
                best, best_ov = p, ov
        if best is None:
            rec["status"] = "no_pbp_overlap"
            out.append(rec)
            continue
        span_len = max(a1["clock_s"] - a2["clock_s"], 1e-6)
        term = next((e for e in reversed(best["events"]) if e["kind"] in
                     ("shot_made", "shot_missed", "turnover", "ft_made", "ft_missed")), None)
        rec.update({
            "status": "aligned", "period": period,
            "anchors": [a1, a2],
            "pbp_possession": {"team": best["team"],
                               "clock": [best["clock_start"], best["clock_end"]],
                               "points": best["points"]},
            "overlap_frac_of_span": round(best_ov / span_len, 2),
            "offense_points": best["points"],
            "terminating_event": ({k: term[k] for k in ("clock_s", "team", "kind", "desc")}
                                  if term else None),
            "pbp_offense_real": game.get(best["team"]),
            "offense_agrees": (game.get(best["team"]) == rec["offense_real"]
                               if rec["offense_real"] else None),
        })
        out.append(rec)
    cap.release()
    # two video spans can map to ONE real possession (a gate-skipped replay splits
    # it on video; PBP knows better) — flag so credit metrics never double-count
    seen: dict = {}
    for r in out:
        if r.get("status") == "aligned":
            key = (r["period"], tuple(r["pbp_possession"]["clock"]))
            if key in seen:
                r["duplicate_of_span"] = seen[key]
            else:
                seen[key] = r["set_start_frame"]
    return out, n_anchor_fail


def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor-per-possession PBP alignment.")
    ap.add_argument("--clips", nargs="*", default=["curry_q1_clip", "curry_classic_clip",
                                                   "clip_10m00_18m00", "clip_26m00_34m00"])
    args = ap.parse_args()

    agree = disagree = unknown = aligned = failed = 0
    for clip in args.clips:
        recs, n_fail = align_clip(clip)
        (PBP_DIR / f"{clip}_outcomes.json").write_text(json.dumps(recs, indent=1))
        log.info("── %s ──", clip)
        for r in recs:
            if r["status"] != "aligned":
                failed += 1
                log.info("  poss @%d: %s", r["set_start_frame"], r["status"])
                continue
            aligned += 1
            ok = r["offense_agrees"]
            agree += ok is True
            disagree += ok is False
            unknown += ok is None
            term = r["terminating_event"]
            pc = r["pbp_possession"]["clock"]
            log.info("  poss @%d P%d [pbp %s→%s, ovl %.0f%%]: offense %s (pred) vs %s (PBP) %s | %d pts | ends: %s",
                     r["set_start_frame"], r["period"],
                     f"{int(pc[0])//60}:{int(pc[0])%60:02d}", f"{int(pc[1])//60}:{int(pc[1])%60:02d}",
                     100 * r["overlap_frac_of_span"],
                     r["offense_real"], r["pbp_offense_real"],
                     "✓" if ok else ("✗" if ok is False else "?"),
                     r["offense_points"], term["desc"][:60] if term else "n/a")

    total_checked = agree + disagree
    report = {
        "possessions_aligned": aligned, "anchor_failures": failed,
        "offense_cross_validation": {
            "agree": agree, "disagree": disagree, "unknown": unknown,
            "agreement_rate": round(agree / total_checked, 3) if total_checked else None,
        },
        "acceptance_note": "Tier 2 credit metrics are gated on this cross-validation "
                           "(C2 offense vs independent PBP actor team).",
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "tier2_crossval.json").write_text(json.dumps(report, indent=2))
    log.info("── CROSS-VALIDATION ── %s", json.dumps(report["offense_cross_validation"]))
    log.info("report → %s", config.REPORTS_DIR / "tier2_crossval.json")


if __name__ == "__main__":
    main()
