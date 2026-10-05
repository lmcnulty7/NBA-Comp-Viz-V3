"""sportvu/compare.py: scorecards of several builds side by side (ROADMAP R2.2 onward).

Reads reports/scorecard/<game>__<build>.json (sportvu.bench) for each build on the held-out game and the
SportVU-synced held-out arena games, and prints one table per game: the court and detection rows, the
better build per row, and the testable n. Every build is scored on the same testable frames and truth.

  python -m sportvu.compare v3 public
      -> reports/scorecard/compare__v3__public.{json,txt}
"""
from __future__ import annotations
import argparse, json
import config
from sportvu.bench import SCORECARD_DIR

# (row, value getter, lower is better?) on the scorecard json's metrics
ROWS = [
    ("position p50 ft", lambda m: m["position_error"]["p50_ft"], True),
    ("position p90 ft", lambda m: m["position_error"]["p90_ft"], True),
    ("near-third bias ft", lambda m: m["near_field_bias"]["near_third_ft"], "abs"),
    ("far-third bias ft", lambda m: m["near_field_bias"]["far_third_ft"], "abs"),
    ("coverage %", lambda m: m["coverage"]["coverage"] and round(100 * m["coverage"]["coverage"], 1), False),
    ("missed players %", lambda m: m["missed_players"]["rate"] and round(100 * m["missed_players"]["rate"], 1), True),
    ("ghost boxes %", lambda m: m["ghost_boxes"]["rate"] and round(100 * m["ghost_boxes"]["rate"], 1), True),
    ("team labels %", lambda m: m["team_labels"]["accuracy"] and round(100 * m["team_labels"]["accuracy"], 1), False),
]


def games() -> list:
    from sportvu import splits
    sp = splits.load()
    return [g for s in ("heldout_game", "heldout_arena") for g, v in sp[s]["games"].items() if v.get("sportvu")]


def better(vals: dict, how) -> str | None:
    have = {b: v for b, v in vals.items() if v is not None}
    if len(have) < 2:
        return None
    key = (lambda b: abs(have[b])) if how == "abs" else (lambda b: have[b] if how else -have[b])
    best = sorted(have, key=key)
    return None if key(best[0]) == key(best[1]) else best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("builds", nargs="+")
    a = ap.parse_args()
    out = {"builds": a.builds, "games": {}}
    L = ["SCORECARDS SIDE BY SIDE: %s (reports/scorecard/<game>__<build>.txt; same testable frames and truth)" % ", ".join(a.builds)]
    for g in games():
        cards = {b: json.loads(p.read_text()) for b in a.builds if (p := SCORECARD_DIR / ("%s__%s.json" % (g, b))).exists()}
        if not cards:
            continue
        any_card = next(iter(cards.values()))
        L += ["", "%s (%s): %d testable frames" % (g, any_card.get("split", "?"), any_card["testable"]),
              "  %-20s %s  better" % ("", "  ".join("%12s" % b for b in a.builds))]
        rows = {}
        for name, get, how in ROWS:
            vals = {b: (get(cards[b]["metrics"]) if b in cards else None) for b in a.builds}
            rows[name] = {"values": vals, "better": better(vals, how)}
            L.append("  %-20s %s  %s" % (name, "  ".join("%12s" % ("-" if vals[b] is None else vals[b]) for b in a.builds),
                                          rows[name]["better"] or "="))
        out["games"][g] = {"split": any_card.get("split"), "testable": any_card["testable"], "rows": rows}
    L += ["", "bias rows: closer to 0 is better; coverage: higher is better (a floor of 50% in the adoption rule)"]
    stem = "compare__" + "__".join(a.builds)
    (SCORECARD_DIR / (stem + ".json")).write_text(json.dumps(out, indent=1))
    (SCORECARD_DIR / (stem + ".txt")).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
