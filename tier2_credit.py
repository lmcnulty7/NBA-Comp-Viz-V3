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
from pathlib import Path

from align_outcomes import reconstruct_possessions
from fetch_pbp import GAMES, PBP_DIR
from tier2_join import sources_fingerprint

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("tier2_credit")

MIN_BUCKET = 300   # possessions per defense bucket before numbers mean anything


def check_join_fresh(join: dict, pbp_dir: Path) -> list[str]:
    """The join must have been built from EXACTLY the outcomes files on disk —
    otherwise credit silently aggregates a stale join (run 10: the packaged
    join predated the run; credit consumed it without complaint). Returns the
    discrepancies; caller refuses to aggregate on any."""
    recorded = join.get("sources")
    if recorded is None:
        return ["join has no sources fingerprint (predates the guard) — re-run tier2_join.py"]
    current = sources_fingerprint(pbp_dir)
    problems = [f"outcomes NOT in join: {c}" for c in sorted(current.keys() - recorded.keys())]
    problems += [f"join references missing outcomes: {c}" for c in sorted(recorded.keys() - current.keys())]
    problems += [f"outcomes changed since join: {c}"
                 for c in sorted(k for k in current.keys() & recorded.keys()
                                 if current[k] != recorded[k])]
    return problems


def credit_rows(buckets: dict) -> list[dict]:
    """Defense buckets → report rows; the n≥MIN_BUCKET gate lives HERE so it
    lifts automatically the moment a bucket crosses — never a manual edit."""
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
    return rows


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
    stale = check_join_fresh(join, PBP_DIR)
    if stale:
        for p in stale:
            log.error(" STALE JOIN: %s", p)
        raise SystemExit("tier2_join.json does not match the outcomes on disk — "
                         "run tier2_join.py first, then re-run tier2_credit.py")
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

    rows = credit_rows(buckets)
    any_meaningful = any(r["meaningful"] for r in rows)

    out = {"rows": rows, "min_bucket": MIN_BUCKET,
           "joined_total": len(join["joined"]),
           "note": ("rows with meaningful=true are reportable; the rest remain placeholders"
                    if any_meaningful else
                    "PLACEHOLDER until buckets reach min_bucket — do not report"),
           "baseline": "offense's own full-game PPP (PBP), possession-weighted"}
    (PBP_DIR / "tier2_credit.json").write_text(json.dumps(out, indent=1))

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
