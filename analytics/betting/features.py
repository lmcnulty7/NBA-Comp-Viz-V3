"""analytics/betting/features.py: the betting_games frame (docs/contract/betting.md) from the games table joined to
the closing lines, with features that use previous games only.

    python -m analytics.betting.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]

FEATURES (per game, every one from games strictly before game_date, ordered by (game_date, game_id)):
    home_net10, away_net10      point margin per game over the team's previous 10 games (0 with no previous game)
    home_net30, away_net30      the same over the previous 30 games
    home_rest_days, away_rest_days   days since the team's previous game, capped at REST_CAP (the cap for a first game)
    home_b2b, away_b2b          1 when rest_days == 1
    home_flag                   1 on every row (the unit is the home side's game; kept as the spec lists it)
    home_pf_std, home_pa_std, away_pf_std, away_pa_std   season-to-date points for and against per game (0 when none)
RAPM_FEATURES (optional, present only when the lineups lane wrote player_rapm*.parquet under
OUT_ROOT/analytics/lineups/<season>/ with team_id, rapm and as_of; rows with in_sample False are preferred):
    home_rapm_sum, away_rapm_sum  the team's player RAPM sum from the latest as_of strictly before the game; null in
                                  betting_games when no dated row covers the game or the team, imputed to 0 in the
                                  design matrix
    home_has_rapm, away_has_rapm  1 when the sum was found, 0 when imputed (so a model can tell 0 from missing)

Targets and lines, never features: home_margin, home_cover (null on a push), closing_spread_home, closing_total,
implied_home_prob, home_score, away_score, home_win. The cover features exclude the spread; the spread is used only
for the CLV target and the ridge model's conversion to a cover probability.

Guards: assert_time_forward(df) calls analytics.schema.assert_time_forward (the Phase 0 label union) and then refuses
any column of BETTING_TARGETS; assert_no_leak(X, y_margin) additionally refuses a feature whose absolute correlation
with the final margin exceeds LEAK_CORR (a feature built from the final margin is caught even under a new name).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema as _schema
from analytics.betting import validate as _validate

WHO = "betting.features"
FEATURES = ("home_net10", "away_net10", "home_net30", "away_net30", "home_rest_days", "away_rest_days",
            "home_b2b", "away_b2b", "home_flag", "home_pf_std", "home_pa_std", "away_pf_std", "away_pa_std")
RAPM_SUMS = ("home_rapm_sum", "away_rapm_sum")
RAPM_FLAGS = ("home_has_rapm", "away_has_rapm")
RAPM_FEATURES = RAPM_SUMS + RAPM_FLAGS
BETTING_TARGETS = frozenset({"home_margin", "home_cover", "home_score", "away_score", "home_win", "closing_spread_home",
                             "closing_total", "implied_home_prob", "clv", "spread_home", "overround_h2h",
                             "spread_price_home", "p_cover", "margin_forecast", "pnl", "stake"})
ID_COLUMNS = frozenset({"game_id", "season", "game_date", "home_team_id", "away_team_id", "game_key", "heldout",
                        "n_books", "n_close_snapshots", "used_fallback"})
REST_CAP = 7
LEAK_CORR = 0.99   # a disguised copy of the margin sits above 0.999; a real form feature stays under 0.6
LEAK_MIN_ROWS = 30   # the correlation tripwire needs this many finite rows: on fewer, a rolling mean over one or two
                     # prior games is degenerate and correlates with anything (the name guard still applies)
LEAK_MIN_STD = 1e-9  # a constant feature is skipped by the tripwire
GAMES_COLUMNS = ("game_id", "season", "game_date", "home_team_id", "away_team_id", "closing_spread_home", "closing_total",
                 "implied_home_prob") + FEATURES + ("home_margin", "home_cover", "heldout")


def feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    """FEATURES, plus the four RAPM features when df carries a non-null RAPM sum (the pair is imputed to 0 with the
    has_rapm flags in design(), so a partial join never leaves NaN in a design matrix)."""
    cols = list(FEATURES)
    if df is not None and all(c in df.columns for c in RAPM_FEATURES) and df[list(RAPM_SUMS)].notna().any().any():
        cols += list(RAPM_FEATURES)
    return cols


# ── guards ─────────────────────────────────────────────────────────────────────────────────────────
def assert_time_forward(df: pd.DataFrame) -> None:
    """Raise ValueError when df holds a Phase 0 label (analytics.schema) or a betting target or line column."""
    _schema.assert_time_forward(df, "possessions")
    leaked = [c for c in df.columns if c in BETTING_TARGETS]
    if leaked:
        raise ValueError("%s: feature frame holds target or line column(s) %s" % (WHO, leaked))


def assert_no_leak(X: pd.DataFrame, y_margin=None) -> None:
    """assert_time_forward on X, then refuse any column whose absolute correlation with the final margin exceeds
    LEAK_CORR (a disguised copy of the final margin). The correlation tripwire runs only over columns with at least
    LEAK_MIN_ROWS finite rows and a standard deviation above LEAK_MIN_STD; the name guard always runs."""
    assert_time_forward(X)
    if y_margin is None:
        return
    y = np.asarray(y_margin, dtype=float)
    for c in X.columns:
        x = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < LEAK_MIN_ROWS or np.std(x[ok]) <= LEAK_MIN_STD or np.std(y[ok]) <= LEAK_MIN_STD:
            continue
        r = abs(float(np.corrcoef(x[ok], y[ok])[0, 1]))
        if r > LEAK_CORR:
            raise ValueError("%s: feature %r correlates %.3f with the final margin (over %.2f): it is built from the "
                             "outcome" % (WHO, c, r, LEAK_CORR))


# ── features ───────────────────────────────────────────────────────────────────────────────────────
def _team_log(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game): season, game_date, game_id, pf, pa, margin."""
    h = pd.DataFrame({"team": games["home_team_id"], "game_id": games["game_id"], "season": games["season"],
                      "game_date": games["game_date"], "pf": games["home_score"], "pa": games["away_score"]})
    a = pd.DataFrame({"team": games["away_team_id"], "game_id": games["game_id"], "season": games["season"],
                      "game_date": games["game_date"], "pf": games["away_score"], "pa": games["home_score"]})
    log = pd.concat([h, a], ignore_index=True)
    log["margin"] = log["pf"].astype(float) - log["pa"].astype(float)
    log["_d"] = pd.to_datetime(log["game_date"])
    return log.sort_values(["team", "season", "_d", "game_id"]).reset_index(drop=True)


def team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Per (team, game_id): net10, net30, rest_days, b2b, pf_std, pa_std from strictly previous games of the same
    season (every rolling value is shifted by one game)."""
    log = _team_log(games)
    out = []
    for (_team, _season), g in log.groupby(["team", "season"], sort=False):
        prev_margin = g["margin"].shift(1)
        net10 = prev_margin.rolling(10, min_periods=1).mean()
        net30 = prev_margin.rolling(30, min_periods=1).mean()
        prev_d = g["_d"].shift(1)
        rest = (g["_d"] - prev_d).dt.days.astype(float).clip(upper=REST_CAP)
        n_prev = np.arange(len(g), dtype=float)
        pf_std = g["pf"].astype(float).cumsum().shift(1) / np.where(n_prev > 0, n_prev, np.nan)
        pa_std = g["pa"].astype(float).cumsum().shift(1) / np.where(n_prev > 0, n_prev, np.nan)
        out.append(pd.DataFrame({"team": g["team"], "game_id": g["game_id"], "net10": net10.fillna(0.0),
                                 "net30": net30.fillna(0.0), "rest_days": rest.fillna(float(REST_CAP)),
                                 "b2b": (rest == 1).astype(int), "pf_std": pf_std.fillna(0.0), "pa_std": pa_std.fillna(0.0)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["team", "game_id", "net10", "net30", "rest_days", "b2b", "pf_std", "pa_std"])


def rapm_features(games: pd.DataFrame, out_root=None) -> pd.DataFrame:
    """home_rapm_sum, away_rapm_sum, home_has_rapm, away_has_rapm per game when
    OUT_ROOT/analytics/lineups/<season>/player_rapm*.parquet (player_rapm_prior.parquet included) exists with columns
    (team_id, rapm, as_of); the sums are null (flags 0) without such a file, for a game on or before the first as_of
    and for a team with no row at the latest as_of before the game. Rows with in_sample False are preferred when the
    column exists (in-sample fits of the season being predicted are not time-forward); a file without as_of is
    ignored."""
    out = pd.DataFrame({"game_id": games["game_id"], "home_rapm_sum": np.nan, "away_rapm_sum": np.nan})
    if out_root is None:
        return _with_flags(out)
    for season, g in games.groupby("season"):
        files = sorted((Path(out_root) / "analytics" / "lineups" / str(season)).glob("player_rapm*.parquet"))
        if not files:
            continue
        r = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        if not {"team_id", "rapm", "as_of"}.issubset(r.columns):
            continue
        if "in_sample" in r.columns:
            oos = r[~r["in_sample"].fillna(True).astype(bool)]
            r = oos if len(oos) else r
        r = r.dropna(subset=["as_of"])
        r["as_of"] = pd.to_datetime(r["as_of"])
        for i, row in g.iterrows():
            before = r[r["as_of"] < pd.to_datetime(row["game_date"])]
            if before.empty:
                continue
            latest = before[before["as_of"] == before["as_of"].max()]
            sums = latest.groupby("team_id")["rapm"].sum()
            out.loc[out["game_id"] == row["game_id"], "home_rapm_sum"] = float(sums.get(row["home_team_id"], np.nan))
            out.loc[out["game_id"] == row["game_id"], "away_rapm_sum"] = float(sums.get(row["away_team_id"], np.nan))
    return _with_flags(out)


def _with_flags(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["home_has_rapm"] = out["home_rapm_sum"].notna().astype(float)
    out["away_has_rapm"] = out["away_rapm_sum"].notna().astype(float)
    return out


def build_games(joined: pd.DataFrame, heldout_season: str | None = None, out_root=None) -> pd.DataFrame:
    """The betting_games frame from the validate join (games with closing columns) -> GAMES_COLUMNS plus the RAPM
    columns. Rows without a closing spread keep their features but have null home_cover (they cannot be scored
    against the market). heldout is true for rows of heldout_season."""
    g = joined.copy()
    g["game_id"] = g["game_id"].astype(str)
    g["_d"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["_d", "game_id"]).reset_index(drop=True)
    form = team_form(g)
    for side in ("home", "away"):
        f = form.rename(columns={c: "%s_%s" % (side, c) for c in ("net10", "net30", "rest_days", "b2b", "pf_std", "pa_std")})
        g = g.merge(f, left_on=["%s_team_id" % side, "game_id"], right_on=["team", "game_id"], how="left").drop(columns="team")
    g["home_flag"] = 1
    g = g.merge(rapm_features(g, out_root), on="game_id", how="left")
    g["home_margin"] = g["home_score"].astype(float) - g["away_score"].astype(float)
    cover = g["home_margin"] + g["closing_spread_home"]
    g["home_cover"] = np.where(cover > 0, 1.0, np.where(cover < 0, 0.0, np.nan))
    g.loc[g["closing_spread_home"].isna(), "home_cover"] = np.nan
    g["heldout"] = (g["season"].astype(str) == str(heldout_season)) if heldout_season else False
    cols = list(GAMES_COLUMNS) + list(RAPM_FEATURES)
    out = g[cols].copy()
    for c in FEATURES:
        out[c] = out[c].astype(float)
    return out


def design(bg: pd.DataFrame, columns=None) -> pd.DataFrame:
    """The feature matrix of bg (float), guarded: never the spread, never a target."""
    cols = columns or feature_columns(bg)
    X = bg[cols].astype(float)
    for c in RAPM_SUMS:
        if c in X.columns:
            X[c] = X[c].fillna(0.0)          # imputed; the has_rapm flag says which rows
    if X.isna().any().any():
        raise ValueError("%s: design matrix holds NaN in %s" % (WHO, [c for c in X.columns if X[c].isna().any()]))
    assert_no_leak(X, bg["home_margin"] if "home_margin" in bg.columns else None)
    return X


# ── driver ─────────────────────────────────────────────────────────────────────────────────────────
def pick_heldout_season(seasons, requested: str | None) -> str | None:
    """requested when present; else the latest season when there are at least two; else none."""
    seasons = sorted(set(map(str, seasons)))
    if requested and requested in seasons:
        return requested
    if len(seasons) >= 2:
        return seasons[-1]
    return None


def later_seasons(seasons, heldout_season: str | None) -> list[str]:
    """Seasons whose key sorts after the held-out season: they are never fitted on (their games postdate the held-out
    season, so fitting on them would look into the future of the games being scored). Empty without a held-out season."""
    if not heldout_season:
        return []
    return sorted(s for s in set(map(str, seasons)) if s > str(heldout_season))


def drop_later_seasons(joined: pd.DataFrame, heldout_season: str | None) -> tuple[pd.DataFrame, list[str]]:
    """joined without the seasons later than the held-out season -> (frame, the ignored season keys)."""
    ignored = later_seasons(joined["season"].unique(), heldout_season)
    if not ignored:
        return joined, []
    return joined[~joined["season"].astype(str).isin(ignored)].reset_index(drop=True), ignored


def write_games(bg: pd.DataFrame, out_root) -> list[Path]:
    paths = []
    for season, g in bg.groupby("season", sort=True):
        d = Path(out_root) / "analytics" / "betting" / str(season)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "betting_games.parquet"
        g.reset_index(drop=True).to_parquet(p, index=False)
        paths.append(p)
    return paths


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true")
    src.add_argument("--data")
    ap.add_argument("--out", default=None, help="OUT_ROOT: betting_games.parquet per season under analytics/betting/")
    ap.add_argument("--heldout-season", default="2024-25")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    a = parse_args(argv)
    data = _validate.fixture_data(a.seed) if a.fixtures else _validate.load_data(a.data)
    res = _validate.validate(data["odds"], data["games"], data.get("team_names"))
    held = pick_heldout_season(res["joined"]["season"].unique(), a.heldout_season)
    joined, ignored = drop_later_seasons(res["joined"], held)
    bg = build_games(joined, held, a.out)
    if ignored:
        print("%s: seasons later than the held-out season ignored: %s" % (WHO, ", ".join(ignored)))
    assert_time_forward(design(bg))
    print("%s: %d games, %d seasons, held-out season %s, %d rows with a closing spread, features %s%s"
          % (WHO, len(bg), bg["season"].nunique(), held, int(bg["closing_spread_home"].notna().sum()),
             feature_columns(bg), " (fixture output, not a result)" if a.fixtures else ""))
    if a.out:
        for p in write_games(bg, a.out):
            print("wrote", p)
    return bg


if __name__ == "__main__":
    main()
