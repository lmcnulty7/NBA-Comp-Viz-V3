"""analytics/winprob/validate.py: checks of the play-by-play, games and odds tables before any feature is built.

    python -m analytics.winprob.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]
        [--fixture-games 20] [--fixture-seasons 2020-21,2021-22]

Checks (every finding is a count, never a row of data):
    schema        the section 13 columns the state builder needs are present
    score         every non-null score string parses to two non-negative ints ("away-home")
    clocks        the clock rule of analytics.winprob.features.clock_rule: a game whose period numbers skip a
                  value or whose overtime holds a clock above 300 s is dropped whole, then any event outside
                  0 <= clock_s <= 720 (300 in overtime) or without a period is dropped. The drops are counted
                  (counts.clock_rule, by cause and season); the check fails only when a season loses more than
                  2% of its events or games. The checks below run on the kept events.
    monotone      inside a game, in (period, event) order, neither running score ever decreases and the
                  period never decreases
    duplicates    repeated (game_id, event) pairs
    games         every play-by-play game has a games row; the games table's final score agrees with the
                  last score of the play-by-play where both exist
    odds          the share of games with a closing home spread (analytics.betting.validate join; the games
                  table's spread_home counts only when there is no odds table)
    heldout       rows flagged heldout (or split starting with heldout) are counted; they never fit anything

validate(...) returns dict(counts, problems, ok); problems is a list of strings naming the check and the
count. main prints the dict as JSON and, with --out, writes it to OUT_ROOT/analytics/winprob/validate.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.winprob import features as F

REQUIRED = ("game_id", "event", "period", "clock_s", "team", "kind", "score", "season")
MAX_DROPPED_SHARE = 0.02   # above this share of a season's events or games, the clock rule fails the run


def validate(pbp: pd.DataFrame, games: pd.DataFrame | None = None, odds: pd.DataFrame | None = None,
             team_names: dict | None = None) -> dict:
    """Every check of the module docstring on the play-by-play (and the games and odds tables when given)."""
    problems: list[str] = []
    counts: dict = {"rows": int(len(pbp)), "games": int(pbp["game_id"].nunique()) if "game_id" in pbp.columns else 0}
    missing = [c for c in REQUIRED if c not in pbp.columns]
    counts["missing_columns"] = missing
    if missing:
        problems.append("schema: missing column(s) %s" % missing)
        return {"counts": counts, "problems": problems, "ok": False}
    counts["seasons"] = sorted(pbp["season"].astype(str).unique().tolist())
    counts["kinds"] = {str(k): int(v) for k, v in pbp["kind"].astype(str).value_counts().items()}

    parsed = [F.parse_score(s) for s in pbp["score"].tolist()]
    nonnull = int(pbp["score"].notna().sum())
    bad_score = sum(1 for s, v in zip(pbp["score"].tolist(), parsed) if (s is not None and s is not pd.NA and not (isinstance(s, float) and np.isnan(s))) and v is None)
    counts["score_nonnull"] = nonnull
    counts["score_unparsed"] = int(bad_score)
    if bad_score:
        problems.append("score: %d non-null score string(s) do not parse to two ints" % bad_score)

    # the clock rule of features.clock_rule: bad games and events are dropped and counted, as features and train do
    pbp, rule = F.clean_playbyplay(pbp)
    counts["clock_out_of_range"] = rule["out_of_range_rows"]
    counts["clock_rule"] = rule
    for season, s in rule["by_season"].items():
        rows_share = s["rows_dropped"] / s["rows"] if s["rows"] else 0.0
        games_share = s["games_dropped"] / s["games"] if s["games"] else 0.0
        if rows_share > MAX_DROPPED_SHARE or games_share > MAX_DROPPED_SHARE:
            problems.append("clocks: season %s drops %d of %d event(s) and %d of %d game(s) under the clock rule, above %.0f%%"
                            % (season, s["rows_dropped"], s["rows"], s["games_dropped"], s["games"], 100 * MAX_DROPPED_SHARE))

    period = pd.to_numeric(pbp["period"], errors="coerce")
    p = pbp.assign(_period=period, _event=pd.to_numeric(pbp["event"], errors="coerce"))
    p = p.sort_values(["game_id", "_period", "_event"], kind="stable")
    away = pd.Series([v[0] if v else np.nan for v in (F.parse_score(s) for s in p["score"].tolist())], index=p.index, dtype=float)
    home = pd.Series([v[1] if v else np.nan for v in (F.parse_score(s) for s in p["score"].tolist())], index=p.index, dtype=float)
    g = p["game_id"].astype(str)
    dec = 0
    for col in (away, home):
        ff = col.groupby(g).ffill()
        d = ff.groupby(g).diff()
        dec += int((d < 0).sum())
    counts["score_decreases"] = dec
    if dec:
        problems.append("monotone: %d event(s) where a running score decreases inside a game" % dec)
    pdec = int((p["_period"].groupby(g).diff() < 0).sum())
    counts["period_decreases"] = pdec
    if pdec:
        problems.append("monotone: %d event(s) where the period decreases inside a game" % pdec)

    dup = int(pbp.duplicated(["game_id", "event"]).sum())
    counts["duplicate_events"] = dup
    if dup:
        problems.append("duplicates: %d repeated (game_id, event) pair(s)" % dup)

    held = pbp["heldout"].fillna(False).astype(bool) if "heldout" in pbp.columns else pd.Series(False, index=pbp.index)
    if "split" in pbp.columns:
        held = held | pbp["split"].astype("string").fillna("").str.startswith("heldout")
    counts["heldout_rows"] = int(held.sum())
    counts["heldout_games"] = int(pbp.loc[held, "game_id"].nunique())

    if games is not None:
        gids = set(pbp["game_id"].astype(str))
        have = set(games["game_id"].astype(str))
        counts["games_without_row"] = len(gids - have)
        if gids - have:
            problems.append("games: %d play-by-play game(s) without a games row" % len(gids - have))
        derived = F.games_from_playbyplay(pbp).set_index("game_id")
        gg = games.drop_duplicates("game_id").set_index(games["game_id"].astype(str))
        both = derived.index.intersection(gg.index)
        if len(both) and {"home_score", "away_score"} <= set(gg.columns):
            mism = int(((derived.loc[both, "home_score"].to_numpy() != gg.loc[both, "home_score"].astype(int).to_numpy())
                        | (derived.loc[both, "away_score"].to_numpy() != gg.loc[both, "away_score"].astype(int).to_numpy())).sum())
            counts["final_score_mismatch"] = mism
            if mism:
                problems.append("games: %d game(s) whose games-table final score differs from the last play-by-play score" % mism)
        if "home_win" in gg.columns:
            counts["home_win_rate"] = float(gg["home_win"].astype(bool).mean())
        gtab = gg.reset_index(drop=True).assign(game_id=gg.index)
        spread = F.closing_spreads(odds, gtab, team_names) if odds is not None else pd.Series(np.nan, index=gtab["game_id"], dtype=float)
        counts["odds_rows"] = int(len(odds)) if odds is not None else 0
        counts["games_with_spread_from_odds"] = int(spread.notna().sum())
        if odds is None and "spread_home" in gg.columns:
            spread = spread.where(spread.notna(), gg["spread_home"].astype(float).to_numpy())
        counts["games_with_closing_spread"] = int(spread.notna().sum())
        counts["spread_coverage"] = float(spread.notna().mean()) if len(spread) else 0.0
    return {"counts": counts, "problems": problems, "ok": not problems}


def parse_args(argv=None):
    """The validate CLI: --fixtures or --data, with --out, --seed and the fixture knobs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="check the synthetic fixtures of analytics.fixtures "
                     "(fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT with collect/stints/playbyplay/<season>.parquet")
    ap.add_argument("--out", default=None, help="OUT_ROOT: writes OUT_ROOT/analytics/winprob/validate.json")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default 0)")
    F.add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Run the checks from the command line, print the JSON result and, with --out, write it."""
    a = parse_args(argv)
    d = F.load_args(a)
    res = validate(d["playbyplay"], d["games"], d["odds"], d.get("team_names"))
    res["files"] = d["files"]
    if a.out:
        p = Path(a.out) / "analytics" / "winprob" / "validate.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1) + "\n")
        res["written"] = str(p)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
