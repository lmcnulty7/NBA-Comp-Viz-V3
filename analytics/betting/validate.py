"""analytics/betting/validate.py: the odds table (docs/DATA_CONTRACT.md section 14) checked and reduced to one
closing line per game and market, then joined to the games table.

    python -m analytics.betting.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]

Steps
  1. check_schema: the section 14 columns, markets in {h2h, spreads, totals}, prices are non-zero integers
     with absolute value at least 100, points present for spreads and totals.
  2. closing_lines: per (game_key, book) the book's own latest pre-tip is_close snapshot (books are staggered);
     a game whose books have no is_close snapshot falls back to each book's latest pre-tip snapshot and is counted
     under fallback_preclose. Earlier closing snapshots of a book are the duplicates: counted per book, never
     silently dropped (duplicate_snapshots; books_without_close for a book lacking one while others have it).
     Books are combined by the median point (home spread, total, spread price) and the mean de-vigged home
     probability.
  3. vig removal per market and book: implied(p_i) = American price to probability, then p_i / sum(p) over the
     market's sides (the multiplicative method). Hand case: -110/-110 -> 0.5/0.5; -150/+130 -> 0.5798/0.4202.
  4. ranges: a home spread within [-30, 30], a total within [150, 300]; rows outside are counted and dropped.
  5. join to games: team names from the feed become a nickname key (team_key: 'Los Angeles Lakers', 'LA Lakers'
     and 'Lakers' all give 'lakers'); game ids give the same key through TEAM_NAMES (stats.nba.com ids). The
     feed's commence_time is UTC, so a 7pm ET tip is dated the next day: the join accepts a game whose
     game_date is within one day of the tip's UTC date (minus 8 hours), nearest first, each side used once.
     A games table whose ids are not stats.nba.com ids (a fixture, another provider) can pass team_names
     (id -> name); ids still unknown are inferred from date co-occurrence (a date with exactly one game on
     each side votes home id -> home name, away id -> away name; the majority name wins) and counted.
     Coverage counts (games with a closing spread, with a total, with a moneyline, unmatched on both sides)
     are logged and returned.

Nothing here writes raw odds rows into the repository: --out receives a small json of counts only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WHO = "betting.validate"
ODDS_COLUMNS = ("game_key", "commence_time", "home_team", "away_team", "book", "market", "side", "point", "price",
                "snapshot_at", "is_close")
MARKETS = ("h2h", "spreads", "totals")
SPREAD_RANGE = (-30.0, 30.0)
TOTAL_RANGE = (150.0, 300.0)
TIP_SHIFT_HOURS = 8        # UTC tip minus this is the US local calendar date for every NBA tip time
MIN_VOTES = 2              # date co-occurrence votes an inferred team name needs (and a strict majority)
DATE_TOLERANCE_DAYS = 1

# stats.nba.com team ids -> (abbreviation, name as the odds feed prints it)
TEAM_NAMES = {
    1610612737: ("ATL", "Atlanta Hawks"), 1610612738: ("BOS", "Boston Celtics"), 1610612739: ("CLE", "Cleveland Cavaliers"),
    1610612740: ("NOP", "New Orleans Pelicans"), 1610612741: ("CHI", "Chicago Bulls"), 1610612742: ("DAL", "Dallas Mavericks"),
    1610612743: ("DEN", "Denver Nuggets"), 1610612744: ("GSW", "Golden State Warriors"), 1610612745: ("HOU", "Houston Rockets"),
    1610612746: ("LAC", "Los Angeles Clippers"), 1610612747: ("LAL", "Los Angeles Lakers"), 1610612748: ("MIA", "Miami Heat"),
    1610612749: ("MIL", "Milwaukee Bucks"), 1610612750: ("MIN", "Minnesota Timberwolves"), 1610612751: ("BKN", "Brooklyn Nets"),
    1610612752: ("NYK", "New York Knicks"), 1610612753: ("ORL", "Orlando Magic"), 1610612754: ("IND", "Indiana Pacers"),
    1610612755: ("PHI", "Philadelphia 76ers"), 1610612756: ("PHX", "Phoenix Suns"), 1610612757: ("POR", "Portland Trail Blazers"),
    1610612758: ("SAC", "Sacramento Kings"), 1610612759: ("SAS", "San Antonio Spurs"), 1610612760: ("OKC", "Oklahoma City Thunder"),
    1610612761: ("TOR", "Toronto Raptors"), 1610612762: ("UTA", "Utah Jazz"), 1610612763: ("MEM", "Memphis Grizzlies"),
    1610612764: ("WAS", "Washington Wizards"), 1610612765: ("DET", "Detroit Pistons"), 1610612766: ("CHA", "Charlotte Hornets"),
}
_ABBR_TO_ID = {abbr: tid for tid, (abbr, _) in TEAM_NAMES.items()}
_CITY_WORDS = {"los", "angeles", "la", "golden", "state", "new", "york", "orleans", "san", "antonio", "oklahoma",
               "city", "portland", "trail"}


# ── team keys ──────────────────────────────────────────────────────────────────────────────────────
def team_key(value, names: dict | None = None) -> str:
    """A join key for a team given its stats.nba.com id, abbreviation or any spelling of its name; names
    (id -> name) extends TEAM_NAMES for ids of another provider or a fixture.

    Known ids and abbreviations map to the nickname; names are lower-cased, split into words, the city words
    dropped and the nickname words joined ('Portland Trail Blazers' -> 'trailblazers', 'LA Clippers' ->
    'clippers'). An unknown integer id becomes 'team<id>' so a fixture with made-up ids still joins to odds
    whose names are 'Team <id>' or 'team<id>'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)) or (isinstance(value, str) and value.strip().isdigit()):
        tid = int(value)
        if names and tid in names:
            value = names[tid]
        elif tid in TEAM_NAMES:
            value = TEAM_NAMES[tid][1]
        else:
            return "team%d" % tid
    s = str(value).strip()
    if s.upper() in _ABBR_TO_ID:
        s = TEAM_NAMES[_ABBR_TO_ID[s.upper()]][1]
    words = re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
    if words and words[0] == "team" and len(words) == 2 and words[1].isdigit():
        return "team" + words[1]
    nick = [w for w in words if w not in _CITY_WORDS]
    if words and words[-1] == "blazers":
        nick = ["trailblazers"]
    return "".join(nick) if nick else "".join(words)


# ── prices ─────────────────────────────────────────────────────────────────────────────────────────
def american_to_prob(price) -> np.ndarray:
    """American odds -> implied probability (with the vig still in): -110 -> 0.5238, +130 -> 0.4348."""
    p = np.asarray(price, dtype=float)
    out = np.where(p < 0, -p / (-p + 100.0), 100.0 / (p + 100.0))
    return out


def remove_vig(probs) -> np.ndarray:
    """Implied probabilities of the sides of one market -> probabilities that sum to one (multiplicative method).
    The overround (sum minus one) is the book's margin on the market."""
    p = np.asarray(probs, dtype=float)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("remove_vig: implied probabilities must be positive and finite, got %r" % (p,))
    return p / s


def _parse_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


# ── checks ─────────────────────────────────────────────────────────────────────────────────────────
def check_schema(odds: pd.DataFrame) -> list[str]:
    """Problems with the odds frame against section 14; empty when it conforms."""
    problems = []
    for c in ODDS_COLUMNS:
        if c not in odds.columns:
            problems.append("missing column %r" % c)
    if problems:
        return problems
    bad_market = sorted(set(odds["market"].astype(str).unique()) - set(MARKETS))
    if bad_market:
        problems.append("unknown market(s) %s; allowed %s" % (bad_market, list(MARKETS)))
    price = pd.to_numeric(odds["price"], errors="coerce")
    if price.isna().any():
        problems.append("%d non-numeric price(s)" % int(price.isna().sum()))
    elif (price.abs() < 100).any():
        problems.append("%d price(s) with absolute value under 100 (American odds)" % int((price.abs() < 100).sum()))
    needs_point = odds["market"].isin(["spreads", "totals"])
    if pd.to_numeric(odds.loc[needs_point, "point"], errors="coerce").isna().any():
        problems.append("%d spreads/totals row(s) without a point" % int(pd.to_numeric(odds.loc[needs_point, "point"], errors="coerce").isna().sum()))
    if _parse_time(odds["commence_time"]).isna().any():
        problems.append("%d unparseable commence_time(s)" % int(_parse_time(odds["commence_time"]).isna().sum()))
    if _parse_time(odds["snapshot_at"]).isna().any():
        problems.append("%d unparseable snapshot_at(s)" % int(_parse_time(odds["snapshot_at"]).isna().sum()))
    return problems


# ── closing lines ──────────────────────────────────────────────────────────────────────────────────
def _market_frame(rows: pd.DataFrame) -> dict:
    """One (game, book, snapshot) group of rows -> spread_home, total, implied_home_prob and the h2h overround."""
    out = {"spread_home": np.nan, "total": np.nan, "implied_home_prob": np.nan, "overround_h2h": np.nan,
           "spread_price_home": np.nan}
    home_key = team_key(rows["home_team"].iloc[0])
    sp = rows[rows["market"] == "spreads"]
    if len(sp):
        hs = sp[sp["side"].map(team_key) == home_key]
        if len(hs):
            out["spread_home"] = float(pd.to_numeric(hs["point"].iloc[0]))
            out["spread_price_home"] = float(pd.to_numeric(hs["price"].iloc[0]))
    tot = rows[rows["market"] == "totals"]
    if len(tot):
        out["total"] = float(pd.to_numeric(tot["point"].iloc[0]))
    ml = rows[rows["market"] == "h2h"]
    if len(ml) >= 2:
        keys = ml["side"].map(team_key).to_numpy()
        implied = american_to_prob(pd.to_numeric(ml["price"]).to_numpy())
        out["overround_h2h"] = float(implied.sum() - 1.0)
        fair = remove_vig(implied)
        m = keys == home_key
        if m.any():
            out["implied_home_prob"] = float(fair[m][0])
    return out


def closing_lines(odds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """-> (one row per game_key, counts). Columns: game_key, commence_time, home_team, away_team, closing_spread_home,
    closing_total, implied_home_prob, overround_h2h, spread_price_home, n_books, n_close_snapshots, used_fallback,
    snapshot_at.

    Per (game_key, book): the book's own latest pre-tip is_close snapshot (books are staggered in the feed, so the
    instant differs by book); its earlier is_close snapshots are the duplicates, counted per book. A book without an
    is_close pre-tip snapshot while another book has one is dropped and counted (books_without_close). When no book
    has one, every book's latest pre-tip snapshot stands in (used_fallback, counted per game). Books are then
    combined: median spread, total and spread price, mean de-vigged home probability. snapshot_at is the latest of
    the chosen book snapshots."""
    df = odds.copy()
    df["_tip"] = _parse_time(df["commence_time"])
    df["_at"] = _parse_time(df["snapshot_at"])
    df["is_close"] = df["is_close"].fillna(False).astype(bool)
    df["book"] = df["book"].astype(str)
    counts = {"rows": int(len(df)), "games": int(df["game_key"].nunique()), "duplicate_snapshots": 0,
              "fallback_preclose": 0, "no_pretip_snapshot": 0, "multi_book_games": 0, "books_without_close": 0}
    rows = []
    for key, g in df.groupby("game_key", sort=True):
        tip = g["_tip"].iloc[0]
        pre = g[g["_at"] <= tip]
        if pre.empty:
            counts["no_pretip_snapshot"] += 1
            continue
        any_close = bool(pre["is_close"].any())
        used_fallback = not any_close
        if used_fallback:
            counts["fallback_preclose"] += 1
        chosen_parts, snaps = [], []
        for book, b in pre.groupby("book", sort=True):
            close = b[b["is_close"]] if any_close else b
            if close.empty:
                counts["books_without_close"] += 1
                continue
            b_snaps = sorted(close["_at"].unique())
            counts["duplicate_snapshots"] += len(b_snaps) - 1
            chosen_parts.append(close[close["_at"] == b_snaps[-1]])
            snaps.append(b_snaps[-1])
        if not chosen_parts:
            counts["no_pretip_snapshot"] += 1
            continue
        if len(chosen_parts) > 1:
            counts["multi_book_games"] += 1
        per_book = pd.DataFrame([_market_frame(part) for part in chosen_parts])
        rows.append({"game_key": key, "commence_time": g["commence_time"].iloc[0], "home_team": g["home_team"].iloc[0],
                     "away_team": g["away_team"].iloc[0],
                     "closing_spread_home": float(per_book["spread_home"].median()),
                     "closing_total": float(per_book["total"].median()),
                     "implied_home_prob": float(per_book["implied_home_prob"].mean()),
                     "overround_h2h": float(per_book["overround_h2h"].mean()),
                     "spread_price_home": float(per_book["spread_price_home"].median()),
                     "n_books": len(chosen_parts), "n_close_snapshots": len(set(snaps)), "used_fallback": used_fallback,
                     "snapshot_at": max(snaps).strftime("%Y-%m-%dT%H:%M:%SZ")})
    out = pd.DataFrame(rows, columns=["game_key", "commence_time", "home_team", "away_team", "closing_spread_home",
                                      "closing_total", "implied_home_prob", "overround_h2h", "spread_price_home",
                                      "n_books", "n_close_snapshots", "used_fallback", "snapshot_at"])
    counts["closing_rows"] = int(len(out))
    return out, counts


def check_ranges(closing: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop closing rows whose spread or total lies outside the plausible range; count them."""
    sp, tot = closing["closing_spread_home"], closing["closing_total"]
    bad_sp = sp.notna() & ~sp.between(*SPREAD_RANGE)
    bad_tot = tot.notna() & ~tot.between(*TOTAL_RANGE)
    out = closing.copy()
    out.loc[bad_sp, "closing_spread_home"] = np.nan
    out.loc[bad_tot, "closing_total"] = np.nan
    return out, {"spread_out_of_range": int(bad_sp.sum()), "total_out_of_range": int(bad_tot.sum()),
                 "with_spread": int(out["closing_spread_home"].notna().sum()),
                 "with_total": int(out["closing_total"].notna().sum()),
                 "with_moneyline": int(out["implied_home_prob"].notna().sum())}


# ── join ───────────────────────────────────────────────────────────────────────────────────────────
def _unknown_ids(games: pd.DataFrame, names: dict) -> set:
    ids = set(pd.concat([games["home_team_id"], games["away_team_id"]]).dropna().astype(int).tolist())
    return {t for t in ids if t not in TEAM_NAMES and t not in names}


def infer_team_names(closing: pd.DataFrame, games: pd.DataFrame, names: dict | None = None) -> dict:
    """id -> feed name for game ids that neither TEAM_NAMES nor names know, from date co-occurrence: a local date
    with exactly one odds game and exactly one games row (within DATE_TOLERANCE_DAYS) votes its home and away
    pairs; an id is resolved only with at least MIN_VOTES votes and a strict majority for one name. Raises ValueError
    when an inferred name collides with another inferred name or with a name already known (TEAM_NAMES or names):
    two ids cannot share a team. Returns only the inferred pairs (empty when nothing is unknown)."""
    names = dict(names or {})
    unknown = _unknown_ids(games, names)
    if not unknown or closing.empty:
        return {}
    c_date = (_parse_time(closing["commence_time"]) - pd.Timedelta(hours=TIP_SHIFT_HOURS)).dt.tz_localize(None).dt.normalize()
    g_date = pd.to_datetime(games["game_date"]).dt.normalize()
    c_by = closing.groupby(c_date.to_numpy()).indices
    g_by = games.groupby(g_date.to_numpy()).indices
    votes: dict = {}
    for d, ci in c_by.items():
        if len(ci) != 1:
            continue
        hits = [gi for off in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1)
                for gi in g_by.get(d + pd.Timedelta(days=off), [])]
        if len(hits) != 1:
            continue
        crow, grow = closing.iloc[ci[0]], games.iloc[hits[0]]
        for tid, name in ((grow["home_team_id"], crow["home_team"]), (grow["away_team_id"], crow["away_team"])):
            if pd.notna(tid) and int(tid) in unknown:
                votes.setdefault(int(tid), {}).setdefault(str(name), 0)
                votes[int(tid)][str(name)] += 1
    out: dict = {}
    for tid, v in votes.items():
        name, n = max(v.items(), key=lambda kv: (kv[1], kv[0]))
        total = sum(v.values())
        if n >= MIN_VOTES and n * 2 > total:
            out[tid] = name
    known_keys = {team_key(n) for n in names.values()} | {team_key(n[1]) for n in TEAM_NAMES.values()}
    seen: dict = {}
    for tid, name in sorted(out.items()):
        k = team_key(name)
        if k in known_keys:
            raise ValueError("%s: inferred team name %r for id %d collides with a known team" % (WHO, name, tid))
        if k in seen:
            raise ValueError("%s: ids %d and %d both infer the team name %r" % (WHO, seen[k], tid, name))
        seen[k] = tid
    return out


def join_games(closing: pd.DataFrame, games: pd.DataFrame, team_names: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Attach the closing line to each game -> (games with the closing columns, coverage counts).

    Match on (home key, away key) and a game_date within DATE_TOLERANCE_DAYS of the tip's local date; the nearest
    date wins and every odds row and game is used at most once. team_names (id -> name) extends TEAM_NAMES; ids
    still unknown are inferred (infer_team_names) and the count reported."""
    names = dict(team_names or {})
    inferred = infer_team_names(closing, games, names)
    names.update(inferred)
    c = closing.copy()
    c["_hk"] = c["home_team"].map(team_key)
    c["_ak"] = c["away_team"].map(team_key)
    c["_date"] = (_parse_time(c["commence_time"]) - pd.Timedelta(hours=TIP_SHIFT_HOURS)).dt.tz_localize(None).dt.normalize()
    g = games.copy()
    g["_hk"] = g["home_team_id"].map(lambda t: team_key(t, names))
    g["_ak"] = g["away_team_id"].map(lambda t: team_key(t, names))
    g["_date"] = pd.to_datetime(g["game_date"]).dt.normalize()
    cand = g.reset_index().merge(c, on=["_hk", "_ak"], suffixes=("", "_odds"))
    cand["_diff"] = (cand["_date"] - cand["_date_odds"]).abs().dt.days
    cand = cand[cand["_diff"] <= DATE_TOLERANCE_DAYS].sort_values(["_diff", "index", "game_key"])
    cand = cand.drop_duplicates("index").drop_duplicates("game_key")
    cols = ["closing_spread_home", "closing_total", "implied_home_prob", "overround_h2h", "spread_price_home",
            "n_books", "n_close_snapshots", "used_fallback", "game_key"]
    out = games.copy()
    idx = cand["index"].to_numpy()
    for col in cols:
        if col == "game_key":
            s = pd.Series([None] * len(out), index=out.index, dtype=object)
        elif col == "used_fallback":
            s = pd.Series([pd.NA] * len(out), index=out.index, dtype="boolean")
        elif col in ("n_books", "n_close_snapshots"):
            s = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int32")
        else:
            s = pd.Series(np.nan, index=out.index, dtype=float)
        s.iloc[idx] = cand[col].to_numpy()
        out[col] = s
    matched = out["game_key"].notna()
    counts = {"games": int(len(games)), "odds_games": int(len(closing)), "matched": int(matched.sum()),
              "games_without_odds": int((~matched).sum()),
              "odds_without_game": int(len(closing) - matched.sum()),
              "matched_with_spread": int(out["closing_spread_home"].notna().sum()),
              "matched_with_total": int(out["closing_total"].notna().sum()),
              "matched_with_moneyline": int(out["implied_home_prob"].notna().sum()),
              "coverage": float(matched.mean()) if len(games) else 0.0,
              "team_names_given": len(team_names or {}), "team_names_inferred": len(inferred),
              "team_ids_unresolved": len(_unknown_ids(games, names))}
    return out, counts


# ── games table ────────────────────────────────────────────────────────────────────────────────────
def games_from_playbyplay(pbp: pd.DataFrame) -> pd.DataFrame:
    """Final scores from a section 13 play-by-play: the last non-null away-home score per game, the home and away
    team ids from the events' team_id by side, the date from game_date."""
    rows = []
    for gid, g in pbp.groupby("game_id", sort=True):
        g = g.sort_values(["period", "event"]) if "event" in g.columns else g.sort_values("period")
        score = g["score"].dropna()
        if score.empty:
            continue
        try:
            away, home = (int(x) for x in str(score.iloc[-1]).split("-"))
        except ValueError:
            continue
        side_ids = {}
        if "team_id" in g.columns and "team" in g.columns:
            for side in ("home", "away"):
                ids = g.loc[g["team"] == side, "team_id"].dropna()
                side_ids[side] = int(ids.iloc[0]) if len(ids) else None
        rows.append({"game_id": str(gid), "season": str(g["season"].iloc[0]) if "season" in g.columns else None,
                     "game_date": str(g["game_date"].iloc[0])[:10] if "game_date" in g.columns else None,
                     "home_team_id": side_ids.get("home"), "away_team_id": side_ids.get("away"),
                     "home_score": home, "away_score": away, "home_win": home > away, "spread_home": None,
                     "heldout": bool(g["heldout"].fillna(False).astype(bool).any()) if "heldout" in g.columns else False})
    return pd.DataFrame(rows)


def load_data(data_root) -> dict:
    """Read DATA_ROOT -> dict(odds, games, files). Every season parquet under collect/extras/odds is read; the games
    table comes from collect/stints/games/<season>.parquet when present, else from collect/stints/playbyplay."""
    root = Path(data_root)
    files = {"odds": sorted((root / "collect" / "extras" / "odds").glob("*.parquet")),
             "games": sorted((root / "collect" / "stints" / "games").glob("*.parquet")),
             "playbyplay": sorted((root / "collect" / "stints" / "playbyplay").glob("*.parquet"))}
    if not files["odds"]:
        raise SystemExit("%s: no odds parquet under %s (collect/extras.py --odds has not been run; the feed is paid)"
                         % (WHO, root / "collect" / "extras" / "odds"))
    odds = pd.concat([pd.read_parquet(p) for p in files["odds"]], ignore_index=True)
    if files["games"]:
        games = pd.concat([pd.read_parquet(p) for p in files["games"]], ignore_index=True)
    elif files["playbyplay"]:
        games = pd.concat([games_from_playbyplay(pd.read_parquet(p)) for p in files["playbyplay"]], ignore_index=True)
    else:
        raise SystemExit("%s: no games or playbyplay parquet under %s" % (WHO, root / "collect" / "stints"))
    return {"odds": odds, "games": games, "files": {k: [str(p) for p in v] for k, v in files.items()}}


@functools.lru_cache(maxsize=4)
def _fixture_data_cached(seed: int, n_games: int, seasons: tuple) -> dict:
    from tests._fx_betting import SOURCE, synth_odds, synth_stints, team_names
    d = synth_stints(seed, n_games=n_games, seasons=seasons)
    return {"odds": synth_odds(seed, d["games"]), "games": d["games"], "files": {"fixtures": [SOURCE]},
            "stints": d["stints"], "playbyplay": d["playbyplay"], "team_names": team_names()}


def fixture_data(seed: int = 0, n_games: int = 40, seasons=("2020-21", "2021-22")) -> dict:
    """The synthetic odds and games of tests/_fx_betting.py (generated once per (seed, n_games, seasons) in a
    process; the frames come back as copies, so callers may mutate them)."""
    d = _fixture_data_cached(int(seed), int(n_games), tuple(seasons))
    return {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in d.items()}


# ── driver ─────────────────────────────────────────────────────────────────────────────────────────
def validate(odds: pd.DataFrame, games: pd.DataFrame, team_names: dict | None = None) -> dict:
    """The whole check -> dict(problems, closing, joined, counts). Raises ValueError on a schema problem.
    team_names (id -> feed name) is for a games table whose ids are not stats.nba.com ids."""
    problems = check_schema(odds)
    if problems:
        raise ValueError("%s: odds table does not match section 14: %s" % (WHO, problems))
    games = games.copy()
    dup = games["game_id"].astype(str).duplicated()
    if dup.any():
        raise ValueError("%s: games table has %d duplicate game_id(s), e.g. %s" % (WHO, int(dup.sum()), sorted(games.loc[dup, "game_id"].astype(str).unique())[:3]))
    if "heldout" in games.columns:
        held = games["heldout"].fillna(False).astype(bool)
        dropped_heldout = int(held.sum())
        games = games[~held].reset_index(drop=True)
    else:
        dropped_heldout = 0
    closing, counts = closing_lines(odds)
    closing, rng_counts = check_ranges(closing)
    joined, join_counts = join_games(closing, games, team_names)
    counts = {"closing": {**counts, **rng_counts}, "join": join_counts, "games_heldout_dropped": dropped_heldout}
    return {"problems": problems, "closing": closing, "joined": joined, "counts": counts}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="run on the synthetic fixtures of tests/_fx_betting.py")
    src.add_argument("--data", help="DATA_ROOT with collect/extras/odds and collect/stints")
    ap.add_argument("--out", default=None, help="OUT_ROOT; counts go to OUT_ROOT/analytics/betting/validate.json")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    data = fixture_data(a.seed) if a.fixtures else load_data(a.data)
    res = validate(data["odds"], data["games"], data.get("team_names"))
    summary = {"who": WHO, "mode": "fixtures" if a.fixtures else "data", "files": data["files"], "counts": res["counts"],
               "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "note": "fixture output, not a result" if a.fixtures else "counts over the collected odds table"}
    text = json.dumps(summary, indent=2)
    if a.out:
        out = Path(a.out) / "analytics" / "betting"
        out.mkdir(parents=True, exist_ok=True)
        (out / "validate.json").write_text(text)
    print(text)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
