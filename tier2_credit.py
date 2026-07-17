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
  baseline      the OFFENSES' own points per possession from that game's PBP,
                possession-weighted across the bucket, with two Phase-1
                corrections (CLAIMS.md B2):
                  LEAVE-SAMPLE-OUT — the credited possessions are excluded
                  from their own baseline (otherwise the baseline contains the
                  outcome and mechanically shrinks credit toward zero);
                  CONTEXT-MATCHED — each possession is compared against its
                  OWN start-type norm (start classified from PBP: gained via
                  made shot / dead ball = halfcourt; via live rebound / steal
                  = live). The bias audit showed 35% of joined possessions are
                  live-start, so a blanket halfcourt baseline mismatches too.
  credit_100    (baseline − ppp_allowed) × 100 — positive = offenses did worse
                than their norm on these defended possessions.
  ci95          cluster-bootstrap 95% CI on credit_100, resampling GAMES with
                replacement (possessions within a game are correlated — same
                lineups, same shooting night). Never report the point estimate
                without it (CLAIMS.md B1). Single-game buckets get ci95=null.

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
import random
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


def start_types(poss: list[dict]) -> list[str]:
    """'halfcourt' | 'live' per possession — HOW the team gained the ball
    (Cleaning-the-Glass-style split, from PBP alone). The previous possession's
    terminating event is the mechanism: made shot / made FT / dead-ball
    turnover / period start ⇒ inbound against a set defense ('halfcourt');
    missed shot or FT (⇒ live defensive rebound) or a steal ⇒ 'live'
    (transition OPPORTUNITY — a proxy, not a transition detector)."""
    types = []
    prev = None
    for p in poss:
        if prev is None or prev["period"] != p["period"]:
            types.append("halfcourt")            # period start = dead ball
        else:
            last = prev["events"][-1]
            k = last["kind"]
            if k in ("shot_missed", "ft_missed"):
                types.append("live")             # gained via live def. rebound
            elif k == "turnover" and "steal" in last["desc"].lower():
                types.append("live")
            else:                                # made shot/FT, dead-ball TO, foul
                types.append("halfcourt")
        prev = p
    return types


def possession_rows(code: str, meta: dict) -> list[dict]:
    """One row per PBP possession of a game: offense_real, period, clock key,
    points, start_type. The clock key (period, clock_start, clock_end) is the
    same identity the join records as pbp_clock — how sampled possessions are
    matched for leave-sample-out exclusion."""
    p = PBP_DIR / f"{code}.json"
    if not p.exists():
        return []
    poss = reconstruct_possessions(json.loads(p.read_text())["events"])
    rows = []
    for x, st in zip(poss, start_types(poss)):
        if x["team"] not in ("home", "away"):
            continue
        rows.append({"offense_real": meta[x["team"]], "period": x["period"],
                     "clock": (round(x["clock_start"], 1), round(x["clock_end"], 1)),
                     "points": x["points"], "start_type": st})
    return rows


def baseline_from_rows(rows: list[dict], exclude: set | None = None,
                       halfcourt_only: bool = True) -> dict:
    """{offense_real: PPP} over one game's possession rows. `exclude` holds
    (period, clock) keys of the SAMPLED possessions — leave-sample-out, so the
    baseline never contains the outcomes being credited against it."""
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        if halfcourt_only and r["start_type"] != "halfcourt":
            continue
        if exclude and (r["period"], r["clock"]) in exclude:
            continue
        acc[r["offense_real"]][0] += r["points"]
        acc[r["offense_real"]][1] += 1
    return {off: pts / n for off, (pts, n) in acc.items() if n}


def bootstrap_ci(possessions: list[dict], n_boot: int = 2000,
                 seed: int = 42) -> list[float] | None:
    """Cluster-bootstrap 95% CI on credit_per_100: resample GAMES with
    replacement (the cluster — possessions within a game share lineups and
    shooting variance), recompute pooled (baseline − allowed) × 100 each time.
    Per-possession baselines are treated as fixed (their sampling error is
    second-order next to the outcome variance). None if fewer than 2 games."""
    by_game = defaultdict(list)
    for p in possessions:
        by_game[p["game"]].append(p)
    games = sorted(by_game)
    if len(games) < 2:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [p for g in (rng.choice(games) for _ in games) for p in by_game[g]]
        n = len(sample)
        credit = (sum(p["exp"] for p in sample) - sum(p["points"] for p in sample)) / n * 100
        stats.append(credit)
    stats.sort()
    return [round(stats[int(0.025 * n_boot)], 1), round(stats[int(0.975 * n_boot)], 1)]


def main() -> None:
    join = json.loads((PBP_DIR / "tier2_join.json").read_text())
    stale = check_join_fresh(join, PBP_DIR)
    if stale:
        for p in stale:
            log.error(" STALE JOIN: %s", p)
        raise SystemExit("tier2_join.json does not match the outcomes on disk — "
                         "run tier2_join.py first, then re-run tier2_credit.py")
    # sampled possessions per game, keyed for leave-sample-out exclusion
    sampled = defaultdict(set)          # game -> {(period, clock key)}
    for r in join["joined"]:
        sampled[r["game"]].add((r["period"], tuple(round(c, 1) for c in r["pbp_clock"])))

    # per-game baselines: leave-sample-out, keyed by (offense, start_type) —
    # the audit showed 35% of joined possessions are LIVE-start (early offense
    # flowing into a set), so each possession is compared against its own
    # start-type norm, not a blanket halfcourt one. Keep the naive baseline
    # (all-context, sample-in) too so the correction stays visible.
    base_lso, base_naive, row_lookup = {}, {}, {}
    for code, meta in GAMES.items():
        rows_g = possession_rows(code, meta)
        if not rows_g:
            continue
        excl = sampled.get(code)
        base_lso[code] = {
            "halfcourt": baseline_from_rows(rows_g, exclude=excl, halfcourt_only=True),
            "live": baseline_from_rows([r for r in rows_g if r["start_type"] == "live"],
                                       exclude=excl, halfcourt_only=False),
        }
        base_naive[code] = baseline_from_rows(rows_g, exclude=None, halfcourt_only=False)
        row_lookup[code] = {(r["period"], r["clock"]): r for r in rows_g}

    buckets = defaultdict(lambda: {"n": 0, "pts": 0, "exp": 0.0, "games": set()})
    per_bucket_poss = defaultdict(list)
    naive_exp = defaultdict(float)
    missing_baseline = 0
    unmatched_sample = 0
    for r in join["joined"]:
        key = (r["period"], tuple(round(c, 1) for c in r["pbp_clock"]))
        row = row_lookup.get(r["game"], {}).get(key)
        if row is None:
            # sampled possession didn't match a reconstructed one ⇒ its LSO
            # exclusion silently failed too — drop it and count loudly
            unmatched_sample += 1
            continue
        exp = base_lso[r["game"]][row["start_type"]].get(r["offense_real"])
        if exp is None:
            missing_baseline += 1
            continue
        b = buckets[r["defense_real"]]
        b["n"] += 1
        b["pts"] += r["outcome"]["points"]
        b["exp"] += exp
        b["games"].add(r["game"])
        per_bucket_poss[r["defense_real"]].append(
            {"game": r["game"], "points": r["outcome"]["points"], "exp": exp})
        naive_exp[r["defense_real"]] += base_naive[r["game"]][r["offense_real"]]

    rows = credit_rows(buckets)
    for r in rows:
        r["ci95"] = bootstrap_ci(per_bucket_poss[r["defense"]])
        r["ci_excludes_zero"] = bool(r["ci95"]) and (r["ci95"][0] > 0 or r["ci95"][1] < 0)
        r["baseline_naive_ppp"] = round(naive_exp[r["defense"]] / r["n_possessions"], 3)
    # the funnel selects scoring-friendly possessions (every bucket's allowed
    # PPP runs above its halfcourt baseline) — that offset is COMMON MODE,
    # because every bucket passes the same funnel. credit_rel differences it
    # out against the leave-bucket-out pooled credit; ABSOLUTE credit vs a
    # non-funnel baseline is not a claim this table can support (CLAIMS.md B4).
    for r in rows:
        others = [p for t, lst in per_bucket_poss.items() if t != r["defense"] for p in lst]
        if others:
            pooled = (sum(p["exp"] for p in others) - sum(p["points"] for p in others)) \
                     / len(others) * 100
            r["credit_rel_per_100"] = round(r["credit_per_100"] - pooled, 1)
    any_meaningful = any(r["meaningful"] for r in rows)

    out = {"rows": rows, "min_bucket": MIN_BUCKET,
           "joined_total": len(join["joined"]),
           "note": ("rows with meaningful=true are reportable; the rest remain placeholders"
                    if any_meaningful else
                    "PLACEHOLDER until buckets reach min_bucket — do not report"),
           "baseline": "offense's own START-TYPE-MATCHED PPP from that game's PBP "
                       "(each possession vs its halfcourt/live norm), leave-sample-"
                       "out, possession-weighted (baseline_naive_ppp = old "
                       "all-context sample-in baseline, for comparison)",
           "ci": "cluster bootstrap by game, 2000 resamples, 95% percentile",
           "credit_rel": "credit_per_100 minus leave-bucket-out pooled credit — the "
                         "funnel's common-mode selection offset differenced out; the "
                         "only cross-team comparison this sample supports"}
    (PBP_DIR / "tier2_credit.json").write_text(json.dumps(out, indent=1))

    log.info("── TIER 2 TEAM-LEVEL DEFENSIVE CREDIT %s ──",
             "" if any_meaningful else "(PLACEHOLDER — no bucket at n≥%d yet)" % MIN_BUCKET)
    log.info(" %-8s %5s %6s %12s %13s %11s %18s %8s", "defense", "n", "games",
             "ppp_allowed", "baseline_ppp", "credit/100", "95% CI", "rel/100")
    for r in rows:
        ci = f"[{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]" if r["ci95"] else "(1 game)"
        log.info(" %-8s %5d %6d %12.3f %13.3f %+11.1f %18s %+8.1f%s",
                 r["defense"], r["n_possessions"], r["n_games"], r["ppp_allowed"],
                 r["baseline_ppp"], r["credit_per_100"], ci,
                 r.get("credit_rel_per_100", 0.0),
                 "" if r["meaningful"] else "   [n too small]")
    if missing_baseline:
        log.info(" (%d joined possessions skipped: no PBP baseline)", missing_baseline)
    if unmatched_sample:
        log.warning(" %d sampled possessions did not match a reconstructed PBP "
                    "possession — their LSO exclusion FAILED; investigate before "
                    "trusting baselines", unmatched_sample)
    log.info("credit → %s", PBP_DIR / "tier2_credit.json")


if __name__ == "__main__":
    main()
