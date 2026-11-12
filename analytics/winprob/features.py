"""analytics/winprob/features.py: the time-forward state frame of the live win probability model.

    python -m analytics.winprob.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]
        [--fixture-games 20] [--fixture-seasons 2020-21,2021-22]

One row per play-by-play event (docs/DATA_CONTRACT.md sections 5 and 13). Every feature of a row is a
function of the events of the same game up to and including that event, the pre-game odds table and
nothing else:

    margin               home score minus away score after the event (the running score forward-filled
                         inside the game, 0-0 before the first scoring event)
    seconds_remaining    seconds left in regulation: (4 - period) * 720 + clock_s for periods 1 to 4,
                         clock_s of the current overtime period otherwise
    overtime             1 when the period is an overtime period
    period               the period of the event
    possession_home      1 when the home team has the ball after the event, 0 when the away team has it,
                         null when the events so far do not say (before a rebound of a missed shot, at the
                         start of a period, before any event with a team)
    spread_home          the closing spread of the home side from the odds table (negative when the home
                         team is favoured), null when the game has no closing spread
    spread_missing       1 when spread_home is null (the model sees 0 for the spread and this flag)
    margin_x_sqrt_time   margin * sqrt(seconds_remaining), the interaction the spec asks for

The target (home_win) and the final score never enter a feature: assert_no_leak refuses a design matrix
with a forbidden column by name, and prefix_invariance_violations checks by value that a feature at event k
is unchanged when the play-by-play is cut after event k (a feature built from the final margin fails this).
analytics.schema.assert_time_forward runs on every frame this module returns.

Assumptions taken where the spec is silent:
  - the odds table has no NBA id: the closing spread comes from analytics.betting.validate (closing_lines: per
    book the latest pre-tip is_close snapshot, median over books; join_games: both team keys and the tip's US
    local date with one day of tolerance, each row used once); no date-only matching. The games table's
    spread_home column is used only when there is no odds table at all
  - the games table's home_score, away_score and home_win are dropped before any feature is built; the target
    is mapped onto the states after the features, and the prefix probe scrambles the finals of the games row
  - the possession indicator is read from event kinds only (shot_made, rebound, turnover, free throws by
    position in the trip); it stays null when unknown rather than guessed
  - the clock rule (clock_rule, applied by validate, build_states and the trainer alike): a game whose period
    numbers skip a value, or whose overtime holds a clock above 300 s, is dropped whole (its period labels are
    shifted by a collector parsing bug); then any event with a null or out-of-range period or clock is dropped
  - the time buckets of the evaluation are minutes remaining in regulation: 48 to 36, 36 to 24, 24 to 12,
    12 to 4, 4 to 0; overtime rows form their own bucket "OT"
  - the states frame is named winprob_states, which is not one of the five analytics.schema tables, so
    assert_time_forward is called with the closest table name (states_1hz); the guard checks the union of
    all tables' target and post-outcome columns anyway
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.betting import validate as _betting

FEATURES = ("margin", "seconds_remaining", "overtime", "period", "possession_home", "spread_home",
            "spread_missing", "margin_x_sqrt_time")
KEYS = ("game_id", "season", "event")
TARGET = "home_win"
FORBIDDEN = frozenset({"home_win", "home_score", "away_score", "final_margin", "final_score", "score",
                       "points", "p_home_win", "winner", "home_points", "away_points"})
REGULATION_PERIODS = 4
PERIOD_S = 720.0
OVERTIME_S = 300.0
BUCKETS = (("48-36", 36 * 60, 48 * 60 + 1), ("36-24", 24 * 60, 36 * 60), ("24-12", 12 * 60, 24 * 60),
           ("12-4", 4 * 60, 12 * 60), ("4-0", 0, 4 * 60))
BUCKET_NAMES = tuple(b[0] for b in BUCKETS) + ("OT",)
SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25")
FIXTURE_SEASONS = ("2020-21", "2021-22")
FT_RE = re.compile(r"(\d+)\s+of\s+(\d+)")
SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

# NBA team id -> the full name the-odds-api.com uses (analytics.betting.validate.TEAM_NAMES is the source).
TEAM_NAMES = {tid: name for tid, (_abbr, name) in _betting.TEAM_NAMES.items()}


def feature_columns() -> list:
    """The columns the models may be fed, in order."""
    return list(FEATURES)


def assert_no_leak(X) -> None:
    """ValueError when a design matrix carries a column that names the outcome or the score."""
    cols = set(getattr(X, "columns", X))
    bad = sorted(cols & FORBIDDEN)
    if bad:
        raise ValueError("winprob.features: forbidden column(s) in the design matrix: %s" % bad)


def time_bucket(seconds_remaining, overtime) -> pd.Series:
    """Bucket label per row: minutes remaining in regulation, or OT."""
    s = np.asarray(seconds_remaining, dtype=float)
    ot = np.asarray(overtime, dtype=int) == 1
    out = np.full(len(s), "OT", dtype=object)
    for name, lo, hi in BUCKETS:
        out[(~ot) & (s >= lo) & (s < hi)] = name
    out[(~ot) & (s >= BUCKETS[0][2])] = BUCKETS[0][0]
    return pd.Series(out, dtype="string")


# ── score and clock ────────────────────────────────────────────────────────────────────────────────
def parse_score(s):
    """'away-home' -> (away, home) ints, or None when the string is null or malformed."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s is pd.NA:
        return None
    m = SCORE_RE.match(str(s))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def clock_rule(pbp: pd.DataFrame) -> tuple[pd.Series, dict]:
    """The clock rule shared by validate, features and train: (keep mask over pbp's rows, counts).

    Whole games are dropped when their period labels cannot be trusted:
      period_gap       the game's periods do not run 1..max without a gap. The basketball-reference parser
                       (fetch_pbp.parse_events) counts quarter header rows, and the page repeats a header
                       around an event filed under another quarter, so every later period is shifted up
      overtime_clock   a period above 4 holds a clock above 300 s (a regulation quarter labelled overtime)
    Then single events are dropped:
      bad_clock        null or non-numeric period or clock, period below 1, clock below 0, or a clock above
                       the period length (720 s, 300 s in overtime), such as the page's "24:00.0"
    counts: rows and games per cause (a game counts under the first game cause that fits), per season too."""
    def num(col):   # plain float with NaN, also for nullable and arrow columns
        v = pd.to_numeric(pbp[col], errors="coerce")
        return pd.Series(np.asarray(v.to_numpy(dtype=float, na_value=np.nan)), index=pbp.index)
    period, clock = num("period"), num("clock_s")
    gid = pbp["game_id"].astype(str)
    limit = np.where(period > REGULATION_PERIODS, OVERTIME_S, PERIOD_S)
    out_of_range = (clock < 0) | (clock > limit) | clock.isna() | period.isna() | (period < 1)
    ok_period = period.where(period >= 1)
    per_game = pd.DataFrame({"g": gid, "p": ok_period}).dropna().groupby("g")["p"].agg(["max", "nunique"])
    gap_games = set(per_game.index[per_game["max"] != per_game["nunique"]])
    ot_bad = (period > REGULATION_PERIODS) & (clock > OVERTIME_S)
    ot_games = set(gid[ot_bad].unique()) - gap_games
    in_gap = gid.isin(gap_games)
    in_ot = gid.isin(ot_games)
    row_bad = out_of_range & ~in_gap & ~in_ot
    keep = ~(in_gap | in_ot | row_bad)
    season = pbp["season"].astype(str) if "season" in pbp.columns else pd.Series("", index=pbp.index)
    causes = {"period_gap": in_gap, "overtime_clock": in_ot, "bad_clock": row_bad}
    counts = {"rows_in": int(len(pbp)), "rows_kept": int(keep.sum()), "rows_dropped": int((~keep).sum()),
              "games_in": int(gid.nunique()), "games_dropped": int(len(gap_games) + len(ot_games)),
              "out_of_range_rows": int(out_of_range.sum()), "by_cause": {}, "by_season": {}}
    for name, m in causes.items():
        counts["by_cause"][name] = {"rows": int(m.sum()), "games": int(gid[m].nunique()),
                                    "out_of_range_rows": int((m & out_of_range).sum())}
    for s in sorted(season.unique()):
        m = season == s
        counts["by_season"][s] = {"rows": int(m.sum()), "games": int(gid[m].nunique()),
                                  "rows_dropped": int((m & ~keep).sum()),
                                  "games_dropped": int(gid[m & (in_gap | in_ot)].nunique()),
                                  "out_of_range_rows": int((m & out_of_range).sum())}
    return keep, counts


def clean_playbyplay(pbp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """pbp without the rows clock_rule drops, and clock_rule's counts. A no-op on clean events."""
    keep, counts = clock_rule(pbp)
    if bool(keep.all()):
        return pbp, counts
    return pbp.loc[keep.to_numpy()].reset_index(drop=True), counts


def seconds_remaining(period, clock_s) -> np.ndarray:
    """Seconds left in regulation for periods 1 to 4, the overtime clock itself after that."""
    p = np.asarray(period, dtype=int)
    c = np.asarray(clock_s, dtype=float)
    reg = (REGULATION_PERIODS - np.clip(p, 1, REGULATION_PERIODS)) * PERIOD_S + c
    return np.where(p <= REGULATION_PERIODS, reg, c)


# ── possession ─────────────────────────────────────────────────────────────────────────────────────
def _ft_last(desc):
    """True for the last free throw of a trip, False for an earlier one, None for a technical (possession unchanged)."""
    d = str(desc) if desc is not None and desc is not pd.NA else ""
    if "technical" in d.lower():
        return None
    m = FT_RE.search(d)
    return True if not m else int(m.group(1)) >= int(m.group(2))


def possession_after(kind, team, desc, period) -> np.ndarray:
    """Possession indicator after each event of one game, in event order: 1.0 home, 0.0 away, nan unknown.
    Reads only the events up to and including the row (a forward state machine)."""
    n = len(kind)
    out = np.full(n, np.nan)
    cur = np.nan
    last_period = None
    for i in range(n):
        k = kind[i] if kind[i] is not None else ""
        t = team[i]
        if period[i] != last_period:
            cur = np.nan
            last_period = period[i]
        side = None
        if isinstance(t, str) and t in ("home", "away"):
            side = 1.0 if t == "home" else 0.0
        if side is not None:
            other = 1.0 - side
            if k == "shot_made" or k == "turnover":
                cur = other
            elif k == "shot_missed":
                cur = np.nan
            elif k == "rebound":
                cur = side
            elif k in ("ft_made", "ft_missed"):
                last = _ft_last(desc[i])
                if last is not None:
                    cur = side if not last else (other if k == "ft_made" else np.nan)
            elif k == "jump_ball":
                cur = side
        if k in ("period_start", "period_end"):
            cur = np.nan
        out[i] = cur
    return out


# ── odds ───────────────────────────────────────────────────────────────────────────────────────────
def closing_spreads(odds: pd.DataFrame | None, games: pd.DataFrame, team_names: dict | None = None) -> pd.Series:
    """Home closing spread per game_id (index game_id), NaN when the odds table has none for the game.

    Delegates to analytics.betting.validate: closing_lines picks, per (game, book), the book's latest pre-tip
    is_close snapshot and takes the median over books; join_games matches on both team keys and the tip's US
    local date (UTC minus 8 h, one day of tolerance), each odds row and game used at most once. team_names (id ->
    feed name) extends TEAM_NAMES for ids that are not stats.nba.com ids (a fixture); ids still unknown are inferred
    from date co-occurrence there. No date-only matching."""
    out = pd.Series(np.nan, index=pd.Index(games["game_id"].astype(str), name="game_id"), dtype=float)
    if odds is None or len(odds) == 0 or not {"home_team_id", "away_team_id", "game_date"} <= set(games.columns):
        return out
    closing, _ = _betting.closing_lines(odds)
    if len(closing) == 0:
        return out
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    g = g.drop_duplicates("game_id").reset_index(drop=True)
    joined, _ = _betting.join_games(closing, g, {**TEAM_NAMES, **(team_names or {})})
    vals = pd.Series(joined["closing_spread_home"].astype(float).to_numpy(), index=joined["game_id"].astype(str))
    hit = out.index.intersection(vals.index)
    out.loc[hit] = vals.loc[hit].to_numpy()
    return out


# ── games ──────────────────────────────────────────────────────────────────────────────────────────
def games_from_playbyplay(pbp: pd.DataFrame) -> pd.DataFrame:
    """Derive the games table (game_id, season, game_date, home_team_id, away_team_id, home_score, away_score,
    home_win, spread_home null) from the last score of each game in the play-by-play."""
    rows = []
    cols = pbp.columns
    for gid, g in pbp.groupby(pbp["game_id"].astype(str), sort=True):
        g = g.sort_values(["period", "event"], kind="stable")
        last = None
        for s in g["score"].tolist()[::-1]:
            last = parse_score(s)
            if last is not None:
                break
        away, home = last if last is not None else (0, 0)
        tid = g["team_id"] if "team_id" in cols else pd.Series([np.nan] * len(g), index=g.index)
        t = g["team"] if "team" in cols else pd.Series([None] * len(g), index=g.index)
        h = tid[(t == "home") & pd.notna(tid)]
        a = tid[(t == "away") & pd.notna(tid)]
        rows.append({"game_id": gid, "season": str(g["season"].iloc[0]) if "season" in cols else "",
                     "game_date": str(g["game_date"].iloc[0]) if "game_date" in cols else "",
                     "home_team_id": int(h.iloc[0]) if len(h) else -1, "away_team_id": int(a.iloc[0]) if len(a) else -1,
                     "home_score": int(home), "away_score": int(away), "home_win": bool(home > away),
                     "spread_home": np.nan})
    return pd.DataFrame(rows, columns=["game_id", "season", "game_date", "home_team_id", "away_team_id",
                                       "home_score", "away_score", "home_win", "spread_home"])


# ── the state frame ────────────────────────────────────────────────────────────────────────────────
def build_states(pbp: pd.DataFrame, games: pd.DataFrame, odds: pd.DataFrame | None = None,
                 team_names: dict | None = None, spreads: pd.Series | None = None) -> pd.DataFrame:
    """Play-by-play, games and (optional) odds -> the winprob_states frame (docs/contract/winprob.md), one row per
    event, sorted by game_id, period, event. Columns: keys, the features, time_bucket, heldout, and the target
    home_win (the trainer splits the target off before any fit). Never the final score. spreads (game_id -> home
    closing spread, from closing_spreads on the full games table) takes precedence over odds when given, so a
    rebuild from part of a game's events sees the same pre-game spread as the full build."""
    for c in ("game_id", "event", "period", "clock_s", "kind", "score"):
        if c not in pbp.columns:
            raise ValueError("winprob.features: play-by-play lacks column %r" % c)
    for c in ("game_id", "home_win"):
        if c not in games.columns:
            raise ValueError("winprob.features: games table lacks column %r" % c)
    p = clean_playbyplay(pbp)[0].copy()   # the clock rule: no bad clock or shifted period reaches a feature
    p["game_id"] = p["game_id"].astype(str)
    p = p.sort_values(["game_id", "period", "event"], kind="stable").reset_index(drop=True)
    parsed = [parse_score(s) for s in p["score"].tolist()]
    away = pd.Series([v[0] if v else np.nan for v in parsed], dtype=float)
    home = pd.Series([v[1] if v else np.nan for v in parsed], dtype=float)
    grp = p["game_id"]
    away = away.groupby(grp).ffill().fillna(0.0)
    home = home.groupby(grp).ffill().fillna(0.0)
    period = p["period"].astype(int).to_numpy()
    clock = p["clock_s"].astype(float).to_numpy()
    sec = seconds_remaining(period, clock)
    overtime = (period > REGULATION_PERIODS).astype(int)
    poss = np.full(len(p), np.nan)
    kinds = p["kind"].astype(object).where(p["kind"].notna(), None).tolist()
    teams = p["team"].astype(object).where(p["team"].notna(), None).tolist() if "team" in p.columns else [None] * len(p)
    descs = p["desc"].astype(object).where(p["desc"].notna(), None).tolist() if "desc" in p.columns else [None] * len(p)
    for gid, idx in p.groupby("game_id", sort=False).indices.items():
        poss[idx] = possession_after([kinds[i] for i in idx], [teams[i] for i in idx], [descs[i] for i in idx], period[idx])
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    g = g.drop_duplicates("game_id").set_index("game_id")
    target = g[TARGET].astype(bool).astype(int)
    # the finals and the target leave the games table here, before any feature is built
    g = g.drop(columns=[c for c in ("home_score", "away_score", "home_win", "final_margin") if c in g.columns])
    if spreads is not None:
        # pre-game spreads joined once on the full games table (closing_spreads), independent of the events
        spread = pd.Series(spreads).astype(float)
        spread.index = spread.index.astype(str)
        spread = spread.reindex(g.index)
    elif odds is not None:
        spread = closing_spreads(odds, g.reset_index(), team_names)
    elif "spread_home" in g.columns:
        spread = g["spread_home"].astype(float)
    else:
        spread = pd.Series(np.nan, index=g.index, dtype=float)
    sp = p["game_id"].map(spread).astype(float).to_numpy()
    margin = (home - away).to_numpy(dtype=float)
    out = pd.DataFrame({
        "game_id": p["game_id"].to_numpy(),
        "season": (p["season"].astype(str) if "season" in p.columns else p["game_id"].map(g["season"]).astype(str)).to_numpy(),
        "event": p["event"].astype(int).to_numpy(),
        "margin": margin,
        "seconds_remaining": sec.astype(float),
        "overtime": overtime,
        "period": period,
        "possession_home": pd.Series(poss, dtype="float64").astype("Int8"),
        "spread_home": sp,
        "spread_missing": np.isnan(sp).astype(int),
        "margin_x_sqrt_time": margin * np.sqrt(np.maximum(sec, 0.0)),
    })
    out["time_bucket"] = time_bucket(out["seconds_remaining"], out["overtime"]).to_numpy()
    held = p["heldout"].fillna(False).astype(bool).to_numpy() if "heldout" in p.columns else np.zeros(len(p), dtype=bool)
    if "split" in p.columns:
        held = held | p["split"].astype("string").fillna("").str.startswith("heldout").to_numpy(dtype=bool)
    out["heldout"] = held
    missing = sorted(set(out["game_id"]) - set(g.index))
    if missing:
        raise ValueError("winprob.features: %d game(s) of the play-by-play are not in the games table, e.g. %s" % (len(missing), missing[:3]))
    out[TARGET] = out["game_id"].map(target).astype(int).to_numpy()
    schema.assert_time_forward(out.drop(columns=[TARGET]), "states_1hz")
    return out


def design(states: pd.DataFrame, cols=None) -> pd.DataFrame:
    """The numeric design matrix of the feature columns (possession null -> 0 plus a possession_missing flag,
    spread null -> 0 with spread_missing already set)."""
    cols = list(cols) if cols is not None else feature_columns()
    X = pd.DataFrame(index=states.index)
    for c in cols:
        if c == "possession_home":
            v = states[c].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
            X["possession_missing"] = np.isnan(v).astype(float)
            X[c] = np.nan_to_num(v, nan=0.0)
        elif c == "spread_home":
            X[c] = np.nan_to_num(states[c].astype(float).to_numpy(), nan=0.0)
        else:
            X[c] = states[c].astype(float).to_numpy()
    assert_no_leak(X)
    schema.assert_time_forward(X, "states_1hz")
    return X


# ── value-based leakage probe ──────────────────────────────────────────────────────────────────────
def prefix_invariance_violations(pbp: pd.DataFrame, games: pd.DataFrame, odds=None, build=build_states,
                                 n_games: int = 3, cuts: int = 3, seed: int = 0, team_names: dict | None = None) -> list:
    """Cut the play-by-play of a few games after event k, scramble the finals and the outcome of the games row, and
    rebuild: every feature of the rows up to k must be unchanged (events after the state never enter, the games
    table's final score and outcome never enter). Returns the list of (game_id, cut, column) violations; a feature
    built from the final margin lands here whether it read the play-by-play or the games table."""
    rng = np.random.default_rng(seed)
    pbp = clean_playbyplay(pbp)[0]   # probe only games the build keeps
    kw = {"team_names": team_names} if team_names is not None else {}
    if odds is not None:
        # the pre-game spread is joined once on the full games table: the join (team names inferred from date
        # co-occurrence when the ids are unknown) is schedule information, not an event, so no cut may change it
        kw["spreads"] = closing_spreads(odds, games, team_names)
    full = build(pbp, games, odds, **kw)
    cols = [c for c in feature_columns() if c in full.columns]
    gids = sorted(pbp["game_id"].astype(str).unique())
    pick = [gids[i] for i in rng.choice(len(gids), size=min(n_games, len(gids)), replace=False)]
    out = []
    for gid in pick:
        pg = pbp[pbp["game_id"].astype(str) == gid].sort_values(["period", "event"], kind="stable")
        n = len(pg)
        for k in sorted(set(int(v) for v in rng.integers(2, max(3, n - 1), size=cuts))):
            cut = pg.iloc[:k]
            gk = games[games["game_id"].astype(str) == gid].copy()
            # scrambled finals: a feature that reads the games table's final score or outcome changes too
            for c in ("home_score", "away_score"):
                if c in gk.columns:
                    gk[c] = gk[c].astype(int) + int(rng.integers(1, 50))
            gk[TARGET] = ~gk[TARGET].astype(bool)
            part = build(cut, gk, odds, **kw)
            ref = full[full["game_id"] == gid].iloc[:k]
            if len(part) != len(ref):
                out.append((gid, k, "<rows>"))
                continue
            for c in cols:
                a = part[c].to_numpy(dtype=float, na_value=np.nan)
                b = ref[c].to_numpy(dtype=float, na_value=np.nan)
                if not bool(np.all(np.isclose(a, b, equal_nan=True))):
                    out.append((gid, k, c))
    return out


# ── io ─────────────────────────────────────────────────────────────────────────────────────────────
def load_data(data_root, seasons=SEASONS) -> dict:
    """DATA_ROOT/collect/stints/playbyplay/<season>.parquet (required for at least one season),
    DATA_ROOT/collect/extras/odds/<season>.parquet and DATA_ROOT/collect/stints/games/<season>.parquet when present
    -> dict(playbyplay, games, odds, files)."""
    root = Path(data_root)
    pbp, games, odds, files = [], [], [], []
    for s in seasons:
        f = root / "collect" / "stints" / "playbyplay" / ("%s.parquet" % s)
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if "season" not in d.columns:
            d["season"] = s
        pbp.append(d)
        files.append(str(f))
        fg = root / "collect" / "stints" / "games" / ("%s.parquet" % s)
        if fg.exists():
            gd = pd.read_parquet(fg)
            if "season" not in gd.columns:
                gd["season"] = s
            games.append(gd)
            files.append(str(fg))
        else:
            games.append(games_from_playbyplay(d))
        fo = root / "collect" / "extras" / "odds" / ("%s.parquet" % s)
        if fo.exists():
            odds.append(pd.read_parquet(fo))
            files.append(str(fo))
    if not pbp:
        raise SystemExit("winprob: no play-by-play under %s/collect/stints/playbyplay for seasons %s" % (root, list(seasons)))
    return {"playbyplay": pd.concat(pbp, ignore_index=True), "games": pd.concat(games, ignore_index=True),
            "odds": pd.concat(odds, ignore_index=True) if odds else None, "files": files, "team_names": {}}


def load_fixtures(seed: int = 0, n_games: int = 20, seasons=FIXTURE_SEASONS) -> dict:
    """The synthetic play-by-play, games and odds of analytics.fixtures (synth_stints, synth_odds) in the load_data
    layout, with the fixture team-name map the spread join needs. Fixture output, not a result."""
    from analytics.fixtures import synth_odds, synth_stints
    d = synth_stints(seed, n_games=n_games, seasons=tuple(seasons))
    odds = synth_odds(seed, d["games"])
    return {"playbyplay": d["playbyplay"], "games": d["games"], "odds": odds, "files": ["<fixtures seed %d>" % seed],
            "team_names": _betting.fixture_team_names(d["games"])}


def add_fixture_args(ap, seasons: str = ",".join(FIXTURE_SEASONS)) -> None:
    """The fixture knobs every winprob CLI shares: --fixture-games (games per season) and --fixture-seasons."""
    ap.add_argument("--fixture-games", type=int, default=20, help="with --fixtures: games per fixture season (default 20)")
    ap.add_argument("--fixture-seasons", default=seasons, help="with --fixtures: fixture seasons, comma separated "
                    "(default %s)" % seasons)


def load_args(a) -> dict:
    """The inputs a parsed CLI asks for: load_fixtures with the fixture knobs under --fixtures, else load_data."""
    if a.fixtures:
        return load_fixtures(a.seed, n_games=a.fixture_games, seasons=tuple(s for s in a.fixture_seasons.split(",") if s))
    return load_data(a.data)


def write_states(states: pd.DataFrame, out_root) -> list:
    """OUT_ROOT/analytics/winprob/<season>/winprob_features.parquet per season (the full feature frame of this module,
    docs/contract/winprob.md) -> the paths written. The trainer writes winprob_states beside it."""
    paths = []
    for season, s in states.groupby("season", sort=True):
        p = Path(out_root) / "analytics" / "winprob" / str(season) / "winprob_features.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        s.reset_index(drop=True).to_parquet(p, index=False)
        paths.append(str(p))
    return paths


def parse_args(argv=None):
    """The features CLI: --fixtures or --data, with --out, --seed and the fixture knobs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="build from the synthetic fixtures of analytics.fixtures "
                     "(fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT with collect/stints/playbyplay/<season>.parquet")
    ap.add_argument("--out", default=None, help="OUT_ROOT; the states table goes under OUT_ROOT/analytics/winprob/<season>/")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default 0)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Build the state frame from the command line, print a JSON summary and, with --out, write it."""
    a = parse_args(argv)
    d = load_args(a)
    pbp, clock = clean_playbyplay(d["playbyplay"])
    states = build_states(pbp, d["games"], d["odds"], d.get("team_names"))
    summary = {"rows": int(len(states)), "games": int(states["game_id"].nunique()),
               "clock_rule": {k: clock[k] for k in ("rows_dropped", "games_dropped", "by_cause")},
               "seasons": sorted(states["season"].unique().tolist()), "features": feature_columns(),
               "spread_coverage": float(1 - states["spread_missing"].mean()) if len(states) else 0.0,
               "possession_known": float(states["possession_home"].notna().mean()) if len(states) else 0.0,
               "written": write_states(states, a.out) if a.out else []}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
