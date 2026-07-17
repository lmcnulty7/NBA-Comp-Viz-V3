#!/usr/bin/env python
"""
tier2_bias_audit.py — quantify WHICH possessions survive the funnel (CLAIMS B3).

The join keeps ~50-60% of detected spans, and the losses are not random:
anchors need readable clocks, alignment needs clean PBP overlap, matchups need
un-degraded frames. This audit compares the JOINED possessions against the
same games' full PBP possession population on every dimension we can measure
from PBP alone — so every credit table can cite its own selection profile
instead of implying representativeness.

Dimensions: period | start type (halfcourt/live) | clock phase within period |
outcome points mix | PPP. The PPP row is the headline: sampled-vs-population
PPP gap is the funnel's selection offset (the reason tier2_credit reports
credit_rel, the leave-bucket-out relative credit).

Reads tier2_join.json + PBP; writes reports/tier2_bias_audit.{json,txt}.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict

import config
from fetch_pbp import GAMES, PBP_DIR
from tier2_credit import possession_rows

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("tier2_bias_audit")


def clock_phase(clock_start: float) -> str:
    """Early/mid/late thirds of a 12-min period (clock counts DOWN)."""
    return "early" if clock_start > 480 else ("mid" if clock_start > 240 else "late")


def pts_bin(p: int) -> str:
    return {0: "0", 1: "1", 2: "2"}.get(p, "3+")


def profile(rows: list[dict]) -> dict:
    """Distribution profile of a possession set (population or sample)."""
    n = len(rows)
    if not n:
        return {"n": 0}
    share = lambda c: {k: round(v / n, 3) for k, v in sorted(c.items())}
    return {
        "n": n,
        "ppp": round(sum(r["points"] for r in rows) / n, 3),
        "period": share(Counter(str(r["period"]) for r in rows)),
        "start_type": share(Counter(r["start_type"] for r in rows)),
        "clock_phase": share(Counter(clock_phase(r["clock"][0]) for r in rows)),
        "points_mix": share(Counter(pts_bin(r["points"]) for r in rows)),
    }


def main() -> None:
    join = json.loads((PBP_DIR / "tier2_join.json").read_text())
    sampled_keys = defaultdict(set)
    for r in join["joined"]:
        sampled_keys[r["game"]].add(
            (r["period"], tuple(round(c, 1) for c in r["pbp_clock"])))

    pop_all, pop_hc, pop_lv, samp = [], [], [], []
    for code in sorted(sampled_keys):
        rows = possession_rows(code, GAMES[code])
        pop_all += rows
        pop_hc += [r for r in rows if r["start_type"] == "halfcourt"]
        pop_lv += [r for r in rows if r["start_type"] == "live"]
        samp += [r for r in rows if (r["period"], r["clock"]) in sampled_keys[code]]

    ps = profile(samp)
    # offset vs the start-mix-MATCHED population norm: what the sample "should"
    # score if selection were random within start types — the residual is
    # outcome-shape selection (FT-trip exclusion, made-shot enrichment)
    mix = ps["start_type"]
    matched_ppp = (mix.get("halfcourt", 0) * profile(pop_hc)["ppp"]
                   + mix.get("live", 0) * profile(pop_lv)["ppp"])
    out = {
        "games": len(sampled_keys),
        "population_all": profile(pop_all),
        "population_halfcourt": profile(pop_hc),
        "population_live": profile(pop_lv),
        "sampled_joined": ps,
        "selection_offset_per_100": round((ps["ppp"] - matched_ppp) * 100, 1),
        "note": "sampled = joined possessions matched back to reconstructed PBP; "
                "offset = sampled PPP minus start-mix-matched population PPP — the "
                "common-mode bias that credit_rel differences out",
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "tier2_bias_audit.json").write_text(json.dumps(out, indent=1))

    lines = [f"TIER 2 FUNNEL SELECTION AUDIT — {out['games']} games",
             f"{'':16s}{'pop(all)':>10s}{'pop(halfcourt)':>15s}{'JOINED':>10s}"]
    pa, ph, ps = out["population_all"], out["population_halfcourt"], out["sampled_joined"]
    lines.append(f"{'n':16s}{pa['n']:>10d}{ph['n']:>15d}{ps['n']:>10d}")
    lines.append(f"{'ppp':16s}{pa['ppp']:>10.3f}{ph['ppp']:>15.3f}{ps['ppp']:>10.3f}")
    for dim in ("period", "start_type", "clock_phase", "points_mix"):
        lines.append(f"-- {dim}")
        for k in sorted(set(pa[dim]) | set(ps[dim])):
            lines.append(f"  {k:14s}{pa[dim].get(k, 0):>10.3f}"
                         f"{ph[dim].get(k, 0):>15.3f}{ps[dim].get(k, 0):>10.3f}")
    lines.append(f"selection offset (sampled − halfcourt pop): "
                 f"{out['selection_offset_per_100']:+.1f} pts/100")
    txt = "\n".join(lines)
    (config.REPORTS_DIR / "tier2_bias_audit.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
