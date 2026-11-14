"""analytics/lineups/stints.py: section 12 stints from section 13 play-by-play substitutions.

    python -m analytics.lineups.stints --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seasons ...]

Why a second builder: collect/stints.py builds stints from NBA.com GameRotation, which is not reachable from
every network. The reachable source is the play-by-play itself (NBA.com PlayByPlayV2 or basketball-reference
pages, docs/DATA_CONTRACT.md sections 5 and 13), whose substitutions say who left and who entered but never
who started a period. This module derives the rest.

Player keys. A substitution row is `kind == substitution` with `player1_id` out and `player2_id` in when the
NBA.com columns exist. A player id at or above 1610612700, or equal to the row's team_id, is a TEAM id (NBA.com
writes it into player1_id for team rebounds, team turnovers and timeouts) and never a player. Rows a bench
player can produce (technical fouls, ejections, delay of game) never make a starter. Without ids (basketball-reference rows) the description "A. One enters the game for
C. Three" gives the names, the actor of any other row is the leading "X. Name" of its description (before
"makes", "misses", "enters" or after "by"), and names become stable integer keys: the id of `roster`
(a {name: player_id} mapping, when given) else zlib.crc32 of the name (positive, under 2**31, documented in
docs/contract/lineups.md; the section 16 rosters table maps names to ids when the owner joins them). The name
is keyed after clean_name: UTF-8 misread as Latin-1 is repaired and a trailing Jr., Sr., II, III or IV is
dropped, because the play-by-play writes "K. Oubre" where the box score starters have "Kelly Oubre Jr.".
Basketball-reference files a defensive foul (shooting, personal, take) under the fouled team, so a name-keyed
foul row takes the side its player has on the game's other rows (fix_foul_sides).

Starters per period. For each period and side, the starters are the first five players to appear in that
period's events (the actor of a non-substitution event, or the player going OUT in a substitution) before
they have appeared as a player coming IN, in event order. For period 1 the starters table
(DATA_ROOT/collect/stints/starters/<season>.parquet, STARTERS_COLUMNS: starter rows with side, team, a bref
game code, full names and null ids from the basketball-reference box score driver; starters_by_side maps the
game through the games table and the name to the builder's key) is taken first and the derivation fills what
it lacks.
A player who never appears in a period is never found: that side then has fewer than five known players and
every stint of that period carries n_home or n_away under 5 and lineup_ok false (validate.py drops and counts
them). More than five candidates (a data error) keeps the first five by appearance.

Stints. Cuts are the period boundaries (720 s periods, 300 s overtimes) and every substitution time; one row
per interval between consecutive cuts with positive length; substitutions at the same elapsed second form one
cut. On-court sets are updated at each substitution (the out player removed when present, the in player
added). Points per side are the points of events with start_s < elapsed <= end_s; score_home and score_away
are the running totals at end_s. Columns are those of section 12 plus season, split, heldout, season_type,
game_date, source, endpoint, fetched_at copied from the play-by-play when present; endpoint is
`playbyplay+substitutions`.
"""
from __future__ import annotations

import argparse
import re
import zlib

import numpy as np
import pandas as pd

from analytics.lineups import PLAYER_COLUMNS, SEASONS, add_fixture_args, load_inputs, out_dir

PERIOD_S, OT_S = 720.0, 300.0
ENDPOINT = "playbyplay+substitutions"
_ENTERS = re.compile(r"^(?P<inn>.+?) enters the game for (?P<out>.+?)\s*$")
_ACTOR_LEAD = re.compile(r"^(?P<n>[A-Z][\w'.\-]*\.? [\w'.\-]+(?: [\w'.\-]+)?) (?:makes|misses|enters)")
_ACTOR_BY = re.compile(r"\bby (?P<n>[A-Z][\w'.\-]*\.? [\w'.\-]+(?: [\w'.\-]+)?)(?: \(|$)")


def elapsed_s(period: int, clock_s: float) -> float:
    """Seconds since tip-off for (period, seconds remaining in the period)."""
    period = int(period)
    before = sum(PERIOD_S if p <= 4 else OT_S for p in range(1, period))
    return before + (PERIOD_S if period <= 4 else OT_S) - float(clock_s)


def period_of(elapsed: float) -> int:
    """The period of a game-elapsed second (5 and up for overtimes)."""
    if elapsed < 4 * PERIOD_S:
        return int(elapsed // PERIOD_S) + 1
    return 5 + int((elapsed - 4 * PERIOD_S) // OT_S)


SUFFIXES = frozenset({"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV"})


def repair_text(name: str) -> str:
    """Undo UTF-8 read as Latin-1 ("Å\xa0ariÄ\x87" -> "Šarić"); other text is returned unchanged."""
    name = str(name)
    if not any(0x80 <= ord(c) <= 0xFF for c in name) or any(ord(c) > 0xFF for c in name):
        return name
    try:
        return name.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return name


def clean_name(name: str) -> str:
    """The form a name is keyed on: mojibake repaired, whitespace collapsed, a trailing Jr./Sr./II/III/IV dropped
    (the basketball-reference play-by-play writes "K. Oubre" where the box score has "Kelly Oubre Jr.")."""
    parts = repair_text(name).split()
    if len(parts) > 2 and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def name_key(name: str, roster: dict | None = None) -> int:
    """Stable integer key of a basketball-reference name: roster id when known, else crc32 under 2**31 of
    clean_name(name)."""
    raw = " ".join(str(name).split())
    name = clean_name(name)
    if roster:
        for cand in (raw, name):
            if cand in roster:
                return int(roster[cand])
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


def bref_substitution(desc: str) -> tuple[str, str] | None:
    """('in name', 'out name') from "A. One enters the game for C. Three", else None."""
    m = _ENTERS.match(str(desc).strip())
    return (m.group("inn").strip(), m.group("out").strip()) if m else None


def bref_actor(desc: str) -> str | None:
    """The player a basketball-reference description is about (best effort), else None."""
    d = str(desc).strip()
    m = _ACTOR_LEAD.match(d) or _ACTOR_BY.search(d)
    return m.group("n").strip() if m else None


def _has(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and v is not pd.NA


TEAM_ID_MIN = 1610612700
STARTERS_COLUMNS = ("game_id", "game", "game_date", "season", "side", "team", "team_id", "player_slug", "player", "player_id")


def _player(v, team_id=None):
    """int(v) when v is a player id; None when absent, a team id (NBA.com puts the team id, 1610612xxx, in
    player1_id for team rebounds, team turnovers and timeouts) or equal to the row's team_id."""
    if not _has(v):
        return None
    v = int(v)
    if v >= TEAM_ID_MIN or (_has(team_id) and v == int(team_id)):
        return None
    return v


def abbreviated(name: str) -> str:
    """"Eric Bledsoe" -> "E. Bledsoe" (the basketball-reference play-by-play form); a name already in that form,
    or a single token, is returned unchanged."""
    parts = repair_text(name).split()
    if len(parts) < 2 or re.match(r"^[A-Z]\.$", parts[0]):
        return " ".join(parts)
    return "%s. %s" % (parts[0][0], " ".join(parts[1:]))


def event_players(row: dict, roster: dict | None = None) -> tuple[str, int | None, int | None]:
    """(kind, out_or_actor_key, in_key) of one play-by-play row: ids when the NBA.com columns are filled,
    else keys from the description."""
    kind = str(row.get("kind") or "")
    p1, p2 = row.get("player1_id"), row.get("player2_id")
    desc = row.get("desc") or ""
    sub = bref_substitution(desc)
    tid = row.get("team_id")
    p1, p2 = _player(p1, tid), _player(p2, tid)
    if kind == "substitution" or sub is not None:
        if p1 is not None and p2 is not None:
            return "substitution", p1, p2
        if sub is not None:
            return "substitution", name_key(sub[1], roster), name_key(sub[0], roster)
        return "substitution", p1, p2
    if p1 is not None:
        return kind, p1, None
    actor = bref_actor(desc)
    return kind, (name_key(actor, roster) if actor else None), None


def derive_starters(events: list[dict], side: str, known=()) -> list[int]:
    """The first five players of `side` to appear in `events` (one period, in order) before appearing as a
    substitute coming in; `known` (the starters table) comes first."""
    starters = list(dict.fromkeys(int(k) for k in known))
    came_in: set[int] = set()
    for e in events:
        if e["side"] != side:
            continue
        kind, a, b = e["kind"], e["a"], e["b"]
        if kind == "substitution":
            if a is not None and a not in came_in and a not in starters:
                starters.append(a)
            if b is not None:
                came_in.add(b)
        elif e.get("bench_ok"):
            continue                               # a technical foul or an ejection can be a bench player's
        elif a is not None and a not in came_in and a not in starters:
            starters.append(a)
    return starters[:5]


_BENCH_OK = re.compile(r"technical|ejection|T\.FOUL|Non-Unsportsmanlike|Delay", re.I)


def bench_ok(kind: str, desc: str) -> bool:
    """True for rows a player not on the floor can produce (technical fouls, ejections, delay-of-game)."""
    return kind == "ejection" or bool(_BENCH_OK.search(str(desc or "")))


def _side_of(row: dict, home_team_id) -> str | None:
    t = row.get("team")
    if isinstance(t, str) and t in ("home", "away"):
        return t
    tid = row.get("team_id")
    if _has(tid) and _has(home_team_id):
        return "home" if int(tid) == int(home_team_id) else "away"
    return None


def fix_foul_sides(rows: list[dict]) -> None:
    """Basketball-reference play-by-play files a defensive foul under the fouled team, so a name-keyed foul row's
    `team` is often not the fouler's side. Such a row takes the side its player has on most of the game's other
    rows; a fouler seen nowhere else (or tied) keeps the row's side. In place."""
    votes: dict = {}
    for e in rows:
        if e["kind"] == "foul" or e["side"] not in ("home", "away"):
            continue
        for k in (e["a"], e["b"]):
            if k is not None:
                votes.setdefault(k, {"home": 0, "away": 0})[e["side"]] += 1
    for e in rows:
        if e["kind"] != "foul" or not e["by_name"] or e["a"] is None:
            continue
        v = votes.get(e["a"])
        if v is not None and v["home"] != v["away"]:
            e["side"] = "home" if v["home"] > v["away"] else "away"


def build_game_stints(pbp: pd.DataFrame, starters: dict | None = None, home_team_id=None,
                      roster: dict | None = None) -> pd.DataFrame:
    """One game's play-by-play (section 13 columns; `team` home/away, or team_id with home_team_id) ->
    section 12 stints. `starters` is {"home": [...ids], "away": [...]} for period 1 (optional)."""
    if len(pbp) == 0:
        return pd.DataFrame()
    game_id = str(pbp["game_id"].iloc[0]) if "game_id" in pbp.columns else str(pbp["game"].iloc[0])
    order = pbp.assign(_el=[elapsed_s(p, c) for p, c in zip(pbp["period"], pbp["clock_s"])])
    order = order.sort_values(["_el", "event"] if "event" in order.columns else ["_el"], kind="stable")
    rows = []
    for r in order.to_dict("records"):
        kind, a, b = event_players(r, roster)
        rows.append({"t": float(r["_el"]), "period": int(r["period"]), "side": _side_of(r, home_team_id), "kind": kind,
                     "a": a, "b": b, "points": int(r["points"]) if _has(r.get("points")) else 0,
                     "bench_ok": bench_ok(kind, r.get("desc")), "by_name": not _has(r.get("player1_id"))})
    fix_foul_sides(rows)
    n_periods = max(e["period"] for e in rows)
    cuts = {0.0}
    for p in range(1, n_periods + 1):
        cuts.add(elapsed_s(p, 0.0))
    subs = [e for e in rows if e["kind"] == "substitution" and e["side"] in ("home", "away")]
    for e in subs:
        cuts.add(e["t"])
    cuts = sorted(cuts)
    # on-court sets at each cut: walk periods, derive starters, apply substitutions
    on_at: dict[float, dict] = {}
    for p in range(1, n_periods + 1):
        p_start, p_end = elapsed_s(p, PERIOD_S if p <= 4 else OT_S), elapsed_s(p, 0.0)
        ev = [e for e in rows if e["period"] == p]
        known = starters if (p == 1 and starters) else {}
        on = {s: derive_starters(ev, s, known.get(s, ())) for s in ("home", "away")}
        t_prev = p_start
        on_at[p_start] = {s: sorted(on[s]) for s in on}
        for e in [x for x in subs if x["period"] == p]:
            if e["t"] != t_prev:
                on_at[e["t"]] = {s: sorted(on[s]) for s in on}
            s = e["side"]
            if e["a"] is not None and e["a"] in on[s]:
                on[s].remove(e["a"])
            if e["b"] is not None and e["b"] not in on[s]:
                on[s].append(e["b"])
            on_at[e["t"]] = {s2: sorted(on[s2]) for s2 in on}
            t_prev = e["t"]
        on_at.setdefault(p_end, {s: sorted(on[s]) for s in on})
    sc = sorted(((e["t"], e["side"], e["points"]) for e in rows if e["points"] and e["side"]), key=lambda x: x[0])
    sc_t = np.array([x[0] for x in sc], dtype=float)
    sc_h = np.cumsum([x[2] if x[1] == "home" else 0 for x in sc], dtype=int)
    sc_a = np.cumsum([x[2] if x[1] == "away" else 0 for x in sc], dtype=int)

    def points_in(t0, t1):                       # events with t0 < t <= t1
        i0, i1 = np.searchsorted(sc_t, t0, side="right"), np.searchsorted(sc_t, t1, side="right")
        h = int(sc_h[i1 - 1]) - (int(sc_h[i0 - 1]) if i0 else 0) if i1 else 0
        a = int(sc_a[i1 - 1]) - (int(sc_a[i0 - 1]) if i0 else 0) if i1 else 0
        return h, a
    meta = {c: pbp[c].iloc[0] for c in ("season", "split", "heldout", "season_type", "game_date", "source", "fetched_at") if c in pbp.columns}
    out, hp, ap, n = [], 0, 0, 0
    current = {"home": [], "away": []}
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        if t0 in on_at:
            current = on_at[t0]
        if t1 <= t0:
            continue
        home, away = current["home"], current["away"]
        ph, pa = points_in(t0, t1)
        hp += ph; ap += pa
        row = {"game_id": game_id, "stint": n, "start_s": round(t0, 1), "end_s": round(t1, 1), "duration_s": round(t1 - t0, 1),
               "period_start": period_of(t0), "period_end": period_of(max(t1 - 0.1, t0))}
        for k in range(5):
            row["home_%d" % (k + 1)] = home[k] if k < len(home) else None
            row["away_%d" % (k + 1)] = away[k] if k < len(away) else None
        row.update({"home_ids": "|".join(map(str, home)), "away_ids": "|".join(map(str, away)), "n_home": len(home), "n_away": len(away),
                    "lineup_ok": len(home) == 5 and len(away) == 5, "points_home": ph, "points_away": pa, "score_home": hp, "score_away": ap,
                    **meta, "endpoint": ENDPOINT})
        out.append(row)
        n += 1
    return cast_section12(pd.DataFrame(out))


def cast_section12(df: pd.DataFrame) -> pd.DataFrame:
    """The section 12 dtypes: stint int32, times float32, periods int8, counts int8, points and scores int16,
    player ids Int32 (nullable), lineup_ok bool."""
    if len(df) == 0:
        return df
    for c in PLAYER_COLUMNS:
        df[c] = df[c].astype("Int32")
    for c, t in (("stint", "int32"), ("start_s", "float32"), ("end_s", "float32"), ("duration_s", "float32"), ("period_start", "int8"),
                 ("period_end", "int8"), ("n_home", "int8"), ("n_away", "int8"), ("points_home", "int16"), ("points_away", "int16"),
                 ("score_home", "int16"), ("score_away", "int16"), ("lineup_ok", "bool")):
        df[c] = df[c].astype(t)
    return df


def starter_key(row: dict, roster: dict | None = None):
    """The key of one starters row: player_id when filled, else the roster id of player_slug, the full name or the
    abbreviated name, else name_key(abbreviated(player)) (the key the builder gives the basketball-reference
    play-by-play name "E. Bledsoe"), else name_key(player_slug). None when nothing names the player."""
    pid = row.get("player_id")
    if _has(pid):
        return int(pid)
    player = row.get("player") if isinstance(row.get("player"), str) and row.get("player") else None
    slug = row.get("player_slug") if isinstance(row.get("player_slug"), str) and row.get("player_slug") else None
    if roster:
        for cand in (slug, player, abbreviated(player) if player else None):
            if cand and cand in roster:
                return int(roster[cand])
    if player:
        return name_key(abbreviated(player))
    if slug:
        return name_key(slug)
    return None


def resolve_starters_games(starters: pd.DataFrame, games: pd.DataFrame | None) -> pd.DataFrame:
    """starters with a `pbp_game_id` column: the play-by-play game id of each row. A bref box score code in
    `game_id` (YYYYMMDD0HOME) is mapped through the games table when game_id is not itself a games id: by `game`
    against games.game or games.game_id, else by (game_date, home team code) when games carries a home team
    code column (home_team, home, home_abbr or home_code), else by game_date alone when exactly one game of
    the games table is on that date. Unmatched rows keep game_id."""
    out = starters.copy()
    out["pbp_game_id"] = out["game_id"].astype(str)
    if games is None or len(games) == 0 or "game_id" not in games.columns:
        return out
    gids = set(games["game_id"].astype(str))
    miss = ~out["pbp_game_id"].isin(gids)
    if not miss.any():
        return out
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    if "game" in out.columns and "game" in g.columns:
        by_game = dict(zip(g["game"].astype(str), g["game_id"]))
        hit = out.loc[miss, "game"].astype(str).map(by_game)
        out.loc[miss, "pbp_game_id"] = hit.fillna(out.loc[miss, "pbp_game_id"])
        miss = ~out["pbp_game_id"].isin(gids)
    code_col = next((c for c in ("home_team", "home", "home_abbr", "home_code") if c in g.columns), None)
    if miss.any() and "game_date" in g.columns:
        date = g["game_date"].astype(str).str[:10]
        if code_col is not None and "game_date" in out.columns:
            by_dh = dict(zip(zip(date, g[code_col].astype(str)), g["game_id"]))
            codes = out.loc[miss, "game_id"].astype(str).str[9:]
            dates = out.loc[miss, "game_date"].astype(str).str[:10]
            hit = pd.Series([by_dh.get((d, c)) for d, c in zip(dates, codes)], index=out.index[miss])
            out.loc[miss, "pbp_game_id"] = hit.fillna(out.loc[miss, "pbp_game_id"])
            miss = ~out["pbp_game_id"].isin(gids)
        if miss.any() and "game_date" in out.columns:
            counts = date.value_counts()
            by_date = {d: gid for d, gid in zip(date, g["game_id"]) if counts[d] == 1}
            hit = out.loc[miss, "game_date"].astype(str).str[:10].map(by_date)
            out.loc[miss, "pbp_game_id"] = hit.fillna(out.loc[miss, "pbp_game_id"])
    return out


def starters_by_side(starters: pd.DataFrame, game_id: str, home_team_id, away_team_id=None, roster: dict | None = None,
                     games: pd.DataFrame | None = None) -> dict:
    """{"home": keys, "away": keys} of one game's period-1 starters from the starters table (STARTERS_COLUMNS,
    agreed with lane I2: starter rows only, team_id and player_id null from the basketball-reference box score
    driver). Rows match the game on `pbp_game_id` (resolve_starters_games, when `games` is given), else on
    `game_id`, else on `game`. The side is the team id against home_team_id / away_team_id when filled, else the
    `side` column. The key is starter_key(row, roster)."""
    if starters is None or len(starters) == 0:
        return {}
    if "pbp_game_id" not in starters.columns and games is not None:
        starters = resolve_starters_games(starters, games)
    s = pd.DataFrame()
    for col in ("pbp_game_id", "game_id", "game"):
        if col in starters.columns:
            s = starters[starters[col].astype(str) == str(game_id)]
            if len(s):
                break
    out: dict = {}
    for r in s.to_dict("records"):
        side = None
        tid = r.get("team_id")
        if _has(tid) and home_team_id is not None and int(tid) == int(home_team_id):
            side = "home"
        elif _has(tid) and away_team_id is not None and int(tid) == int(away_team_id):
            side = "away"
        elif isinstance(r.get("side"), str) and r["side"] in ("home", "away"):
            side = r["side"]
        key = starter_key(r, roster)
        if side is None or key is None:
            continue
        out.setdefault(side, [])
        if key not in out[side]:
            out[side].append(key)
    return out


REGULATION_PERIODS = 4
OVERTIME_S = 300.0


def shifted_period_games(playbyplay: pd.DataFrame) -> set:
    """Games whose period labels cannot be trusted, dropped whole before any stint is built (the winprob lane's
    clock rule, docs/contract/winprob.md): the periods do not run 1..max without a gap, or a period above 4 holds
    a clock above 300 s. The basketball-reference parser counts quarter header rows, and a page that repeats a
    header around an event filed under another quarter shifts every later period up (53 of 6,416 games in
    2020-21 to 2024-25), which would give the wrong starters and quarter boundaries."""
    if len(playbyplay) == 0 or "period" not in playbyplay.columns:
        return set()
    gcol = "game_id" if "game_id" in playbyplay.columns else "game"
    period = pd.to_numeric(playbyplay["period"], errors="coerce")
    gid = playbyplay[gcol].astype(str)
    ok = period >= 1
    per = pd.DataFrame({"g": gid[ok], "p": period[ok]}).groupby("g")["p"].agg(["max", "nunique"])
    bad = set(per.index[per["max"] != per["nunique"]])
    if "clock_s" in playbyplay.columns:
        clock = pd.to_numeric(playbyplay["clock_s"], errors="coerce")
        bad |= set(gid[(period > REGULATION_PERIODS) & (clock > OVERTIME_S)].unique())
    return bad


def build_stints(playbyplay: pd.DataFrame, games: pd.DataFrame | None = None, starters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every game of a play-by-play table -> stints (section 12 columns plus the carried fields). Games from
    shifted_period_games are dropped whole; their ids are in ``attrs["dropped_shifted_period_games"]``."""
    if len(playbyplay) == 0:
        return pd.DataFrame()
    gcol = "game_id" if "game_id" in playbyplay.columns else "game"
    dropped = shifted_period_games(playbyplay)
    if dropped:
        playbyplay = playbyplay.loc[~playbyplay[gcol].astype(str).isin(dropped)]
    teams = {}
    if games is not None and len(games):
        teams = {str(r.game_id): (r.home_team_id, r.away_team_id) for r in games.itertuples()}
    if "home_team_id" in playbyplay.columns:
        for g, grp in playbyplay.groupby(gcol):
            teams.setdefault(str(g), (grp["home_team_id"].iloc[0], None))
    if starters is not None and len(starters):
        starters = resolve_starters_games(starters, games)
    frames = []
    for g, grp in playbyplay.groupby(gcol, sort=True):
        h, a = teams.get(str(g), (None, None))
        if h is None and "team_id" in grp.columns and "team" in grp.columns:      # the sides name the ids
            hid = grp.loc[grp["team"] == "home", "team_id"].dropna()
            aid = grp.loc[grp["team"] == "away", "team_id"].dropna()
            h = int(hid.iloc[0]) if len(hid) else None
            a = int(aid.iloc[0]) if len(aid) else None
        st = starters_by_side(starters, str(g), h, a)
        frames.append(build_game_stints(grp, st, h))
    out = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    out.attrs["dropped_shifted_period_games"] = sorted(dropped)
    return out


def summarize(stints: pd.DataFrame) -> dict:
    """Stint and game counts, the lineup_ok share and the summed points of a stints table."""
    if len(stints) == 0:
        return {"stints": 0, "games": 0, "lineup_ok_share": float("nan")}
    return {"stints": int(len(stints)), "games": int(stints["game_id"].nunique()),
            "lineup_ok_share": float(stints["lineup_ok"].mean()),
            "points_home": int(stints["points_home"].sum()), "points_away": int(stints["points_away"].sum())}


def parse_args(argv=None):
    """The CLI's arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic stints (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT (default DATA_ROOT; nothing is written for --fixtures without it)")
    ap.add_argument("--seasons", nargs="*", default=list(SEASONS), help="--data only: seasons to read (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default %(default)s)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Build stints from the fixture or DATA_ROOT play-by-play, write them per season under OUT_ROOT and print a summary."""
    a = parse_args(argv)
    inputs = load_inputs(a.fixtures, a.data, a.seasons, a.seed, a.fixture_generator, a.n_games, stints=False)
    stints = build_stints(inputs["playbyplay"], inputs["games"], inputs["starters"])
    out_root = a.out or (a.data if not a.fixtures else None)
    written = []
    if out_root and len(stints):
        for season, grp in stints.groupby("season"):
            p = out_dir(out_root, str(season)) / "stints.parquet"
            grp.reset_index(drop=True).to_parquet(p, index=False)
            written.append(str(p))
    s = summarize(stints)
    s["dropped_shifted_period_games"] = len(stints.attrs.get("dropped_shifted_period_games", []))
    s["written"] = written
    s["fixture"] = inputs["fixture"]
    print("stints from play-by-play%s: %s" % (" (fixture output, not a result)" if inputs["fixture"] else "", s))
    return s


if __name__ == "__main__":
    main()
