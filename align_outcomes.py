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
from clock_reader import ClockReader, layout_for_clip
from fetch_pbp import GAMES, PBP_DIR, game_for_clip, light_team_name, video_path

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


def align_clip(clip: str, fps: float = 30.0, anchor_cache: dict | None = None):
    poss = json.loads((config.TRACKING_DIR / f"{clip}_possessions.json").read_text())
    ident = json.loads((config.TRACKING_DIR / f"{clip}_identity.json").read_text())
    code = game_for_clip(clip)
    game = GAMES[code]
    pbp = json.loads((PBP_DIR / f"{code}.json").read_text())["events"]
    pbp_poss = reconstruct_possessions(pbp)
    light = ident.get("light_team")
    # team id → real team name via which REAL team wears the light kit
    # (pre-2017 home default, per-game override for special uniforms)
    light_name = light_team_name(game)
    dark_name = ({game["home"], game["away"]} - {light_name}).pop()
    def real(team_id):
        if team_id is None or light is None:
            return None
        return light_name if team_id == light else dark_name

    reader = ClockReader(layout_for_clip(clip))
    src = video_path(clip)
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(src))

    out, n_anchor_fail = [], 0
    for span in poss["spans"]:
        if span.get("kind") != "halfcourt" or not span.get("metrics_eligible"):
            continue
        f1, f2 = span["set_start_frame"], span["core_end_frame"]
        if anchor_cache and f1 in anchor_cache:
            a1, a2 = anchor_cache[f1]
        else:
            third = (f2 - f1) // 3
            a1 = reader.anchor_multi(cap, f1, [f1 + third, f1 + 2 * third])
            a2 = reader.anchor_multi(cap, f2, [f2 - third // 2])
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
        def consistent(x, y):
            dv = (y["frame"] - x["frame"]) / fps
            dc = x["clock_s"] - y["clock_s"]
            return dc >= 0 and abs(dc - dv) <= CLOCK_RATE_TOL
        if not consistent(a1, a2):
            # one anchor is off (e.g. read during a graphic overlay) — a third
            # anchor mid-core arbitrates which one to replace
            mid = reader.anchor_multi(cap, (f1 + f2) // 2, [])
            if mid and mid["period"] == a1["period"] and consistent(a1, mid):
                a2 = mid
            elif mid and mid["period"] == a2["period"] and consistent(mid, a2):
                a1 = mid
            else:
                rec.update({"status": "anchor_inconsistent", "anchors": [a1, a2]})
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
    return out, n_anchor_fail, {"light": light, "light_name": light_name, "dark_name": dark_name}


def apply_orientation(all_recs: dict, mappings: dict) -> None:
    """Period-orientation correction (measured 2026-07-09: the per-possession
    defense-closer vote is only ~65–75% right on playoff footage, but within a
    (game, period) the attacked basket DETERMINES the offense. A conf-weighted
    majority of C2's own votes per (game, period, basket) — no PBP involved, so
    the cross-validation stays independent — overrides each span's call. This
    promotes the 'same basket all period' canary into the assignment mechanism.)

    Mutates aligned recs in place: offense/defense fields, cluster ids (so the
    matchup engine can correct roles via the per-clip corrections file), and
    recomputed offense_agrees. Orientation can also FILL spans whose team vote
    was unknown. Contradictory periods (same team majority at both baskets)
    are left untouched."""
    from collections import Counter, defaultdict

    votes = defaultdict(Counter)   # (game, period, basket) -> Counter[team_real]
    for clip, (recs, _) in all_recs.items():
        game = game_for_clip(clip)
        for r in recs:
            if r.get("status") == "aligned" and r.get("offense_real"):
                votes[(game, r["period"], r["attacked_basket"])][r["offense_real"]] += r.get("confidence", 0.5)

    orient = {}                    # (game, period) -> {basket: team_real}
    strength = {}
    for (game, period, basket), c in votes.items():
        orient.setdefault((game, period), {})[basket] = c.most_common(1)[0][0]
        strength[(game, period, basket)] = round(c.most_common(1)[0][1] / max(sum(c.values()), 1e-6), 2)
    for key, m in list(orient.items()):
        if len(set(m.values())) < len(m):      # same team both baskets ⇒ unusable
            log.warning("orientation contradiction for %s — left as per-span votes", key)
            orient.pop(key)

    for clip, (recs, _) in all_recs.items():
        game = game_for_clip(clip)
        mp = mappings[clip]
        corrections = {}
        for r in recs:
            if r.get("status") != "aligned":
                continue
            m = orient.get((game, r["period"]), {})
            expected = m.get(r["attacked_basket"])
            if expected is None:
                r["offense_source"] = "span_vote"
                continue
            other = ({GAMES[game]["home"], GAMES[game]["away"]} - {expected}).pop()
            r["orientation_flipped"] = (r["offense_real"] is not None and r["offense_real"] != expected)
            r["offense_source"] = "orientation"
            r["orientation_conf"] = strength[(game, r["period"], r["attacked_basket"])]
            r["offense_real"], r["defense_real"] = expected, other
            if mp["light"] is not None:        # cluster ids for the matchup engine
                off_cid = mp["light"] if expected == mp["light_name"] else 1 - mp["light"]
                r["offense_team"] = off_cid
                corrections[str(r["set_start_frame"])] = {
                    "offense_team": off_cid, "defense_team": 1 - off_cid,
                    "orientation_conf": r["orientation_conf"]}
            r["offense_agrees"] = (r["pbp_offense_real"] == expected
                                   if r.get("pbp_offense_real") else None)
        if corrections:
            (PBP_DIR / f"{clip}_orientation.json").write_text(json.dumps(corrections, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor-per-possession PBP alignment.")
    ap.add_argument("--clips", nargs="*", default=["curry_q1_clip", "curry_classic_clip",
                                                   "clip_10m00_18m00", "clip_26m00_34m00"])
    ap.add_argument("--reuse-anchors", action="store_true",
                    help="Reuse anchors from existing outcomes files (skip OCR).")
    args = ap.parse_args()

    all_recs, mappings = {}, {}
    for clip in args.clips:
        cache = {}
        if args.reuse_anchors and (PBP_DIR / f"{clip}_outcomes.json").exists():
            for old in json.loads((PBP_DIR / f"{clip}_outcomes.json").read_text()):
                if old.get("anchors"):
                    cache[old["set_start_frame"]] = old["anchors"]
        recs, n_fail, mp = align_clip(clip, anchor_cache=cache)
        all_recs[clip] = (recs, n_fail)
        mappings[clip] = mp
    apply_orientation(all_recs, mappings)

    agree = disagree = unknown = aligned = failed = 0
    for clip in args.clips:
        recs, n_fail = all_recs[clip]
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
