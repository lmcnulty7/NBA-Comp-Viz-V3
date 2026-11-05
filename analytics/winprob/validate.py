"""analytics/winprob/validate.py: checks of the play-by-play, games and odds tables before any feature is built.

    python -m analytics.winprob.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]

Checks (every finding is a count, never a row of data):
    schema        the section 13 columns the state builder needs are present
    score         every non-null score string parses to two non-negative ints ("away-home")
    clocks        0 <= clock_s <= 720 in periods 1 to 4 and 0 <= clock_s <= 300 in overtime
    monotone      inside a game, in (period, event) order, neither running score ever decreases and the
                  period never decreases
    duplicates    repeated (game_id, event) pairs
    games         every play-by-play game has a games row; the games table's final score agrees with the
                  last score of the play-by-play where both exist
    odds          the share of games with a closing home spread
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


def validate(pbp: pd.DataFrame, games: pd.DataFrame | None = None, odds: pd.DataFrame | None = None) -> dict:
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

    period = pd.to_numeric(pbp["period"], errors="coerce")
    clock = pd.to_numeric(pbp["clock_s"], errors="coerce")
    limit = np.where(period <= F.REGULATION_PERIODS, F.PERIOD_S, F.OVERTIME_S)
    bad_clock = int(((clock < 0) | (clock > limit) | clock.isna() | period.isna() | (period < 1)).sum())
    counts["clock_out_of_range"] = bad_clock
    if bad_clock:
        problems.append("clocks: %d event(s) with a clock outside [0, period length] or a missing period" % bad_clock)

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
        spread = F.closing_spreads(odds, gtab) if odds is not None else pd.Series(np.nan, index=gtab["game_id"], dtype=float)
        counts["odds_rows"] = int(len(odds)) if odds is not None else 0
        counts["games_with_spread_from_odds"] = int(spread.notna().sum())
        if "spread_home" in gg.columns:
            spread = spread.where(spread.notna(), gg["spread_home"].astype(float).to_numpy())
        counts["games_with_closing_spread"] = int(spread.notna().sum())
        counts["spread_coverage"] = float(spread.notna().mean()) if len(spread) else 0.0
    return {"counts": counts, "problems": problems, "ok": not problems}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true")
    src.add_argument("--data", help="DATA_ROOT with collect/stints/playbyplay/<season>.parquet")
    ap.add_argument("--out", default=None, help="OUT_ROOT: writes OUT_ROOT/analytics/winprob/validate.json")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    d = F.load_fixtures(a.seed) if a.fixtures else F.load_data(a.data)
    res = validate(d["playbyplay"], d["games"], d["odds"])
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
