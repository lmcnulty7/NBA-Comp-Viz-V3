#!/usr/bin/env python
"""
tier2_credit.py — Tier 2 team-level defensive credit (PLACEHOLDER SAMPLE).

The reframing this whole chain was built for: not "how far was the defender"
but "did possessions defended by this team run WORSE for the offense than that
offense's own average". Pure aggregation over existing joined data — no video,
no models, safe to run alongside a harvest queue.

Per defense bucket (real team):
  ppp_allowed   points per possession on OUR joined halfcourt possessions
                where this team defended (PBP-validated outcomes)
  baseline      the OFFENSES' own points per possession computed from the full
                game's PBP possessions (reconstruct_possessions — same code the
                alignment uses), possession-weighted across the bucket. This
                controls for opponent quality: stopping the 2016 Warriors is
                not the same as stopping anyone else.
  credit_100    (baseline − ppp_allowed) × 100 — positive = offenses did worse
                than their norm on these defended possessions.

⚠ SAMPLE-SIZE GUARD: numbers print with an explicit not-yet-meaningful banner
until buckets reach MIN_BUCKET (300). Re-run after each harvest join — reads
only data/pbp/tier2_join.json + the games' PBP files, so it is idempotent and
cheap. Output: data/pbp/tier2_credit.json.

Known scope limits (inherited, documented elsewhere): halfcourt set-cores only
(no transition defense), team-level only (player buckets gated on jersey OCR).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

from align_outcomes import reconstruct_possessions
from fetch_pbp import GAMES, PBP_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("tier2_credit")

MIN_BUCKET = 300   # possessions per defense bucket before numbers mean anything


def game_ppp_baselines() -> dict:
    """{(game_code, offense_real): full-game points per possession} from PBP —
    the offense's own norm that defended possessions are compared against."""
    out = {}
    for code, meta in GAMES.items():
        p = PBP_DIR / f"{code}.json"
        if not p.exists():
            continue
        poss = reconstruct_possessions(json.loads(p.read_text())["events"])
        by_team = defaultdict(lambda: [0, 0])
        for x in poss:
            if x["team"] in ("home", "away"):
                by_team[x["team"]][0] += x["points"]
                by_team[x["team"]][1] += 1
        for side, (pts, n) in by_team.items():
            if n:
                out[(code, meta[side])] = pts / n
    return out


def main() -> None:
    join = json.loads((PBP_DIR / "tier2_join.json").read_text())
    baselines = game_ppp_baselines()

    buckets = defaultdict(lambda: {"n": 0, "pts": 0, "exp": 0.0, "games": set()})
    missing_baseline = 0
    for r in join["joined"]:
        key = (r["game"], r["offense_real"])
        if key not in baselines:
            missing_baseline += 1
            continue
        b = buckets[r["defense_real"]]
        b["n"] += 1
        b["pts"] += r["outcome"]["points"]
        b["exp"] += baselines[key]
        b["games"].add(r["game"])

    rows = []
    for team, b in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
        ppp = b["pts"] / b["n"]
        base = b["exp"] / b["n"]
        rows.append({
            "defense": team, "n_possessions": b["n"], "n_games": len(b["games"]),
            "ppp_allowed": round(ppp, 3), "baseline_ppp": round(base, 3),
            "credit_per_100": round((base - ppp) * 100, 1),
            "meaningful": b["n"] >= MIN_BUCKET,
        })

    out = {"rows": rows, "min_bucket": MIN_BUCKET,
           "joined_total": len(join["joined"]),
           "note": "PLACEHOLDER until buckets reach min_bucket — do not report",
           "baseline": "offense's own full-game PPP (PBP), possession-weighted"}
    (PBP_DIR / "tier2_credit.json").write_text(json.dumps(out, indent=1))

    any_meaningful = any(r["meaningful"] for r in rows)
    log.info("── TIER 2 TEAM-LEVEL DEFENSIVE CREDIT %s ──",
             "" if any_meaningful else "(PLACEHOLDER — no bucket at n≥%d yet)" % MIN_BUCKET)
    log.info(" %-8s %5s %6s %12s %13s %11s", "defense", "n", "games", "ppp_allowed",
             "baseline_ppp", "credit/100")
    for r in rows:
        log.info(" %-8s %5d %6d %12.3f %13.3f %+11.1f%s",
                 r["defense"], r["n_possessions"], r["n_games"], r["ppp_allowed"],
                 r["baseline_ppp"], r["credit_per_100"],
                 "" if r["meaningful"] else "   [n too small]")
    if missing_baseline:
        log.info(" (%d joined possessions skipped: no PBP baseline)", missing_baseline)
    log.info("credit → %s", PBP_DIR / "tier2_credit.json")


if __name__ == "__main__":
    main()
