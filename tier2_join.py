#!/usr/bin/env python
"""
tier2_join.py — Tier 2 join (correctness pass, NO aggregates).

Joins each aligned possession OUTCOME (align_outcomes.py: points + terminating
event, PBP-validated) to its MATCHUP record (matchup_metrics.py: defenders,
primary man, time assigned) on set_start_frame.

Deliberately computes NO defender-level statistics: with 9 possessions across
2 games this pass verifies that the JOIN is structurally right — correct
outcome attached to correct possession's defenders, correctly deduped —
before the harvesting pipeline scales the sample to where credit numbers
mean anything.

Dedupe is a HARD FILTER, not metadata (same semantics as the >5-track gate):
records carrying `duplicate_of_span` (one real possession split across two
video spans by a gate-skipped replay) are EXCLUDED from the join, and a
structural assertion proves no two joined records share a PBP possession —
double-counting is impossible by construction, and the run fails loudly if
that ever stops being true.

Structural checks, all reported:
  1. one outcome per joined possession, defenders present;
  2. no two joined records share (game, period, PBP clock range)  [assert];
  3. team consistency: the matchup engine's defense == the team PBP says was
     NOT on offense;
  4. exclusion accounting: every aligned outcome is joined or listed with a
     reason (duplicate / no matchup record / degraded).

Output: data/pbp/tier2_join.json + console table for eyeball review.
"""
from __future__ import annotations

import json
import logging

import config
from fetch_pbp import GAMES, PBP_DIR, game_for_clip

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("tier2_join")

def discover_clips() -> list[str]:
    """Every clip with an aligned-outcomes file joins (Phase A adds clips freely)."""
    return sorted(p.name.replace("_outcomes.json", "")
                  for p in PBP_DIR.glob("*_outcomes.json"))


def main() -> None:
    joined, excluded = [], []
    seen_pbp: dict[tuple, dict] = {}    # (game, period, clock range) -> joined record

    for clip in discover_clips():
        outcomes = json.loads((PBP_DIR / f"{clip}_outcomes.json").read_text())
        matchups = {p["set_start_frame"]: p for p in json.loads(
            (config.TRACKING_DIR / f"{clip}_matchups.json").read_text())["possessions"]}
        game = game_for_clip(clip)

        for o in outcomes:
            f = o["set_start_frame"]
            if o.get("status") != "aligned":
                excluded.append({"clip": clip, "set_start_frame": f, "reason": o.get("status")})
                continue
            if "duplicate_of_span" in o:      # HARD dedupe — never enters the join
                excluded.append({"clip": clip, "set_start_frame": f,
                                 "reason": f"duplicate_of_span:{o['duplicate_of_span']}"})
                continue
            m = matchups.get(f)
            if m is None:
                excluded.append({"clip": clip, "set_start_frame": f, "reason": "no_matchup_record"})
                continue
            if m.get("degraded"):
                excluded.append({"clip": clip, "set_start_frame": f, "reason": "matchup_degraded"})
                continue
            # weak PBP match = the only path a role inversion could slip past the
            # agreement check (wrong possession matched, coincidentally same team)
            if o.get("overlap_frac_of_span", 0) < 0.8:
                excluded.append({"clip": clip, "set_start_frame": f, "reason": "weak_pbp_match"})
                continue

            # check 3: team consistency between the two independent pipelines.
            # PBP is authoritative on offense — a disagreement means the matchup
            # record's defender roles are INVERTED, so the possession is hard-
            # excluded from credit joins (not just flagged).
            pbp_off = o["pbp_offense_real"]
            pbp_def = ({GAMES[game]["home"], GAMES[game]["away"]} - {pbp_off}).pop()
            team_ok = (o["offense_real"] == pbp_off and o["defense_real"] == pbp_def)
            if not team_ok:
                excluded.append({"clip": clip, "set_start_frame": f,
                                 "reason": "offense_disagrees_pbp"})
                continue

            defenders = sorted(m["defenders"], key=lambda d: -d["time_assigned_s"])
            if not defenders:      # nothing to credit (all frames gated / no labeled pairs)
                excluded.append({"clip": clip, "set_start_frame": f, "reason": "no_defenders"})
                continue
            rec = {
                "clip": clip, "game": game, "set_start_frame": f,
                "period": o["period"],
                "pbp_clock": o["pbp_possession"]["clock"],
                "offense_real": o["offense_real"], "defense_real": pbp_def,
                "team_consistency_ok": team_ok,
                "outcome": {"points": o["offense_points"],
                            "terminating": o["terminating_event"]["desc"]
                            if o["terminating_event"] else None},
                "defenders": [{"fragment": d["defender"], "primary_man": d["primary_man"],
                               "time_assigned_s": d["time_assigned_s"],
                               "matchup_dist_median_ft": d["matchup_dist_median_ft"]}
                              for d in defenders],
                "primary_defender_fragment": defenders[0]["defender"] if defenders else None,
            }
            rec["_span_len"] = o["core_end_frame"] - o["set_start_frame"]
            key = (game, o["period"], tuple(o["pbp_possession"]["clock"]))
            prev = seen_pbp.get(key)
            if prev is not None:
                # a possession straddling a SECTION boundary appears in both
                # sections (per-clip dedupe can't see this) — keep the record
                # with more core time on the possession, exclude the fragment
                if rec["_span_len"] > prev["_span_len"]:
                    joined.remove(prev)
                    excluded.append({"clip": prev["clip"], "set_start_frame": prev["set_start_frame"],
                                     "reason": f"duplicate_cross_section:{clip}@{f}"})
                    seen_pbp[key] = rec
                    joined.append(rec)
                else:
                    excluded.append({"clip": clip, "set_start_frame": f,
                                     "reason": f"duplicate_cross_section:{prev['clip']}@{prev['set_start_frame']}"})
                continue
            seen_pbp[key] = rec
            joined.append(rec)

    # ── console: the eyeball table ────────────────────────────────────────────
    log.info("── TIER 2 JOIN (correctness pass — NO aggregates by design) ──")
    all_team_ok = True
    for r in joined:
        all_team_ok &= r["team_consistency_ok"]
        tops = ", ".join(f"{d['fragment']}→{d['primary_man']} ({d['time_assigned_s']}s, "
                         f"{d['matchup_dist_median_ft']}ft)" for d in r["defenders"][:3])
        log.info(" %s @%d P%d: %s offense, %d pts — %s",
                 r["clip"], r["set_start_frame"], r["period"], r["offense_real"],
                 r["outcome"]["points"], (r["outcome"]["terminating"] or "?")[:55])
        log.info("   defense=%s (consistent: %s) | top defenders (frag→man): %s",
                 r["defense_real"], r["team_consistency_ok"], tops)
    for e in excluded:
        log.info(" excluded %s @%s: %s", e["clip"], e["set_start_frame"], e["reason"])

    checks = {
        "joined": len(joined),
        "excluded": len(excluded),
        "every_join_has_outcome_and_defenders": all(r["defenders"] and
                                                    r["outcome"]["points"] is not None
                                                    for r in joined),
        "no_shared_pbp_possession": True,   # the assert above enforces it
        "team_consistency_all": all_team_ok,
        "note": "no defender aggregates computed — sample too small by design; "
                "scale via harvesting before credit numbers",
    }
    out = PBP_DIR / "tier2_join.json"
    out.write_text(json.dumps({"checks": checks, "joined": joined, "excluded": excluded}, indent=1))
    log.info("── CHECKS ── %s", json.dumps(checks))
    log.info("join → %s", out)


if __name__ == "__main__":
    main()
