"""analytics.gaps: SportVU clock gaps from a Step 1 manifest (sportvu/colab_contract.py).

The manifest records, per game, every hole of more than 2 s in the tracking clock as
gaps_over_2s = {"count", "seconds", "by_quarter": {"1": [[from_clock_s, to_clock_s], ...], ...}}.
The clock counts down, so a span is [larger clock, smaller clock] in seconds remaining of that
quarter and its length is from minus to. This module turns those spans into tables, finds the games
whose gaps exceed a threshold, flags possessions that overlap a gap, and writes reports/sportvu_gaps.md.

Nothing here reads tracking data or fits anything: held-out games are counted like any other game.

CLI: python -m analytics.gaps --manifest PATH --out reports/sportvu_gaps.md [--threshold 10] [--note TEXT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GAP_FLOOR_S = 2.0                     # the manifest only records gaps over this (sportvu.fetch.GAP_S)
DEFAULT_THRESHOLD_S = 10.0

GAP_COLUMNS = ["sportvu_game", "period", "gap_start_clock_s", "gap_end_clock_s", "gap_s"]
OVER_COLUMNS = ["sportvu_game", "split", "n_gaps_over", "longest_gap_s", "seconds_over"]
POSSESSION_COLUMNS = ["sportvu_game", "period", "start_clock_s", "end_clock_s"]   # section 19 columns used here


def _empty_gaps() -> pd.DataFrame:
    return pd.DataFrame({"sportvu_game": pd.Series([], dtype="object"),
                         "period": pd.Series([], dtype="int8"),
                         "gap_start_clock_s": pd.Series([], dtype="float64"),
                         "gap_end_clock_s": pd.Series([], dtype="float64"),
                         "gap_s": pd.Series([], dtype="float64")})


def game_gaps(manifest: dict) -> pd.DataFrame:
    """One row per recorded gap (over 2 s) in every game of the manifest.

    Columns: sportvu_game, period (int8), gap_start_clock_s (the larger clock, where tracking stops),
    gap_end_clock_s (the smaller clock, where it resumes), gap_s = start - end. Games whose entry has no
    gaps_over_2s (failed, source_broken, or reconciled from Drive without a summary) contribute no rows.
    Sorted by game, period, then clock descending (game order within the quarter)."""
    rows = []
    for game, entry in manifest.get("games", {}).items():
        info = entry.get("gaps_over_2s") or {}
        for q, spans in (info.get("by_quarter") or {}).items():
            for span in spans or []:
                start, end = float(span[0]), float(span[1])
                if start < end:                      # defensive: spans are [larger, smaller] in the manifest
                    start, end = end, start
                rows.append((game, int(q), start, end, start - end))
    if not rows:
        return _empty_gaps()
    df = pd.DataFrame(rows, columns=GAP_COLUMNS)
    df["period"] = df["period"].astype("int8")
    df["gap_s"] = df["gap_s"].round(3)
    df = df.sort_values(["sportvu_game", "period", "gap_start_clock_s"], ascending=[True, True, False])
    return df.reset_index(drop=True)


def games_over(manifest: dict, threshold_s: float = DEFAULT_THRESHOLD_S) -> pd.DataFrame:
    """Games with at least one gap strictly longer than threshold_s.

    Columns: sportvu_game, split, n_gaps_over (gaps over the threshold), longest_gap_s, seconds_over
    (the summed length of the gaps over the threshold, not of every gap). Sorted by seconds_over
    descending, then game."""
    gaps = game_gaps(manifest)
    over = gaps[gaps["gap_s"] > float(threshold_s)]
    if over.empty:
        return pd.DataFrame({"sportvu_game": pd.Series([], dtype="object"),
                             "split": pd.Series([], dtype="object"),
                             "n_gaps_over": pd.Series([], dtype="int64"),
                             "longest_gap_s": pd.Series([], dtype="float64"),
                             "seconds_over": pd.Series([], dtype="float64")})
    agg = (over.groupby("sportvu_game", sort=True)
               .agg(n_gaps_over=("gap_s", "size"), longest_gap_s=("gap_s", "max"), seconds_over=("gap_s", "sum"))
               .reset_index())
    agg["n_gaps_over"] = agg["n_gaps_over"].astype("int64")
    agg["seconds_over"] = agg["seconds_over"].round(3)
    splits = {g: e.get("split") for g, e in manifest.get("games", {}).items()}
    agg.insert(1, "split", agg["sportvu_game"].map(splits).astype("object"))
    agg = agg.sort_values(["seconds_over", "sportvu_game"], ascending=[False, True])
    return agg[OVER_COLUMNS].reset_index(drop=True)


def flag_possessions(possessions: pd.DataFrame, gaps: pd.DataFrame) -> pd.Series:
    """True for each possession whose clock span overlaps a gap of the same game and period.

    possessions carries sportvu_game, period, start_clock_s, end_clock_s (start >= end: the clock counts
    down); gaps is game_gaps output. Overlap is strict: a gap that only touches a possession's endpoint
    does not flag it. The result is a bool Series aligned to possessions.index, named gap_overlap."""
    missing = [c for c in POSSESSION_COLUMNS if c not in possessions.columns]
    if missing:
        raise KeyError("possessions is missing columns %s" % missing)
    out = pd.Series(False, index=possessions.index, name="gap_overlap", dtype=bool)
    if len(possessions) == 0 or gaps is None or len(gaps) == 0:
        return out
    left = pd.DataFrame({"_row": np.arange(len(possessions)),
                         "sportvu_game": possessions["sportvu_game"].astype("object").to_numpy(),
                         "period": possessions["period"].astype("int64").to_numpy(),
                         "p_start": possessions["start_clock_s"].astype("float64").to_numpy(),
                         "p_end": possessions["end_clock_s"].astype("float64").to_numpy()})
    right = pd.DataFrame({"sportvu_game": gaps["sportvu_game"].astype("object").to_numpy(),
                          "period": gaps["period"].astype("int64").to_numpy(),
                          "g_start": gaps["gap_start_clock_s"].astype("float64").to_numpy(),
                          "g_end": gaps["gap_end_clock_s"].astype("float64").to_numpy()})
    m = left.merge(right, on=["sportvu_game", "period"], how="inner")
    if m.empty:
        return out
    hit = (m["g_start"] > m["p_end"]) & (m["g_end"] < m["p_start"])
    flagged = np.zeros(len(possessions), dtype=bool)
    flagged[m.loc[hit, "_row"].to_numpy()] = True
    out[:] = flagged
    return out


def summary(manifest: dict, threshold_s: float = DEFAULT_THRESHOLD_S) -> dict:
    """Counts for the report: games, games_ok (status ok), games_with_gap_over_2s, games_with_gap_over_threshold,
    total_gap_seconds (every recorded gap, from the spans), and by_split with the same counts per split."""
    games = manifest.get("games", {})
    gaps = game_gaps(manifest)
    over = games_over(manifest, threshold_s)
    per_game_gaps = set(gaps["sportvu_game"]) if len(gaps) else set()
    per_game_over = set(over["sportvu_game"]) if len(over) else set()
    seconds_by_game = gaps.groupby("sportvu_game")["gap_s"].sum().to_dict() if len(gaps) else {}

    def block(ids):
        return {"games": len(ids),
                "games_ok": sum(1 for g in ids if games[g].get("status") == "ok"),
                "games_with_gap_over_2s": sum(1 for g in ids if g in per_game_gaps),
                "games_with_gap_over_threshold": sum(1 for g in ids if g in per_game_over),
                "total_gap_seconds": round(float(sum(seconds_by_game.get(g, 0.0) for g in ids)), 1)}

    by_split: dict = {}
    for g, e in games.items():
        by_split.setdefault(str(e.get("split")), []).append(g)
    result = block(list(games))
    result["gaps_over_2s"] = int(len(gaps))
    result["gaps_over_threshold"] = int(over["n_gaps_over"].sum()) if len(over) else 0
    result["threshold_s"] = float(threshold_s)
    result["by_split"] = {s: block(ids) for s, ids in sorted(by_split.items())}
    return result


def _load(manifest_path) -> dict:
    p = Path(manifest_path)
    if p.is_dir():
        p = p / "manifest.json"
    return json.loads(p.read_text())


def _fmt(x: float) -> str:
    x = float(x)
    return str(int(x)) if x == int(x) else ("%.3f" % x).rstrip("0").rstrip(".")


def write_report(manifest_path, out_path, threshold_s: float = DEFAULT_THRESHOLD_S, note: str | None = None) -> dict:
    """Write the markdown gap report for the manifest at manifest_path to out_path; returns summary()."""
    manifest = _load(manifest_path)
    s = summary(manifest, threshold_s)
    over = games_over(manifest, threshold_s)
    games = manifest.get("games", {})
    thr = _fmt(threshold_s)
    floor = _fmt(GAP_FLOOR_S)
    lines = ["# SportVU clock gaps", ""]
    lines.append("Manifest: `%s` (%d games, updated %s, contract v%s)." % (
        manifest_path, s["games"], manifest.get("updated", "unknown"), manifest.get("contract_version", "?")))
    lines.append("")
    lines.append("Scope: this report covers only the games listed in that manifest. A gap is a hole of more than "
                 "%s s in the tracking clock of one quarter, as recorded by the Step 1 converter "
                 "(`gaps_over_2s` in the manifest); the clock counts down, so a gap runs from the larger clock "
                 "to the smaller one. Held-out games are counted here and nothing is fit or tuned." % floor)
    if note:
        lines += ["", note]
    lines += ["", "## Summary", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append("| games in manifest | %d |" % s["games"])
    lines.append("| games with status ok | %d |" % s["games_ok"])
    lines.append("| games with any gap over %s s | %d |" % (floor, s["games_with_gap_over_2s"]))
    lines.append("| games with any gap over %s s | %d |" % (thr, s["games_with_gap_over_threshold"]))
    lines.append("| gaps over %s s | %d |" % (floor, s["gaps_over_2s"]))
    lines.append("| gaps over %s s | %d |" % (thr, s["gaps_over_threshold"]))
    lines.append("| total gap seconds (all gaps over %s s) | %s |" % (floor, _fmt(s["total_gap_seconds"])))
    lines += ["", "## By split", ""]
    lines.append("| split | games | ok | with gap over %s s | with gap over %s s | gap seconds |" % (floor, thr))
    lines.append("|---|---|---|---|---|---|")
    for split, b in s["by_split"].items():
        lines.append("| %s | %d | %d | %d | %d | %s |" % (split, b["games"], b["games_ok"], b["games_with_gap_over_2s"],
                                                       b["games_with_gap_over_threshold"], _fmt(b["total_gap_seconds"])))
    lines += ["", "## Games with a gap over %s s" % thr, ""]
    if over.empty:
        lines.append("None.")
    else:
        lines.append("| game | split | gaps over %s s | longest gap (s) | seconds over %s s |" % (thr, thr))
        lines.append("|---|---|---|---|---|")
        for r in over.itertuples(index=False):
            lines.append("| %s | %s | %d | %s | %s |" % (r.sportvu_game, r.split, r.n_gaps_over,
                                                       _fmt(r.longest_gap_s), _fmt(r.seconds_over)))
    skipped = [g for g, e in games.items() if e.get("status") != "ok"]
    if skipped:
        lines += ["", "Games without gap data (status not ok): %s." % ", ".join(
            "%s (%s)" % (g, games[g].get("status")) for g in skipped)]
    lines.append("")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analytics.gaps", description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="Step 1 manifest.json (or the directory holding it)")
    ap.add_argument("--out", default="reports/sportvu_gaps.md")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_S, help="gap length in seconds (default 10)")
    ap.add_argument("--note", default=None, help="one extra paragraph placed under the scope line")
    a = ap.parse_args(argv)
    s = write_report(a.manifest, a.out, a.threshold, a.note)
    print("wrote %s: %d games, %d with a gap over %s s, %s gap seconds" % (
        a.out, s["games"], s["games_with_gap_over_threshold"], _fmt(a.threshold), _fmt(s["total_gap_seconds"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
