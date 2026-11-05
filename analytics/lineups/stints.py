"""analytics/lineups/stints.py: section 12 stints from section 13 play-by-play substitutions.

    python -m analytics.lineups.stints --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seasons ...]

Why a second builder: collect/stints.py builds stints from NBA.com GameRotation, which is not reachable from
every network. The reachable source is the play-by-play itself (NBA.com PlayByPlayV2 or basketball-reference
pages, docs/DATA_CONTRACT.md sections 5 and 13), whose substitutions say who left and who entered but never
who started a period. This module derives the rest.

Player keys. A substitution row is `kind == substitution` with `player1_id` out and `player2_id` in when the
NBA.com columns exist. Without ids (basketball-reference rows) the description "A. One enters the game for
C. Three" gives the names, the actor of any other row is the leading "X. Name" of its description (before
"makes", "misses", "enters" or after "by"), and names become stable integer keys: the id of `roster`
(a {name: player_id} mapping, when given) else zlib.crc32 of the name (positive, under 2**31, documented in
docs/contract/lineups.md; the section 16 rosters table maps names to ids when the owner joins them).

Starters per period. For each period and side, the starters are the first five players to appear in that
period's events (the actor of a non-substitution event, or the player going OUT in a substitution) before
they have appeared as a player coming IN, in event order. For period 1 the starters table
(DATA_ROOT/collect/stints/starters/<season>.parquet: game_id, team_id, player_id, with team_id mapped to a
side by the games table or the substitution rows) is taken first and the derivation fills what it lacks.
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
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.lineups import PLAYER_COLUMNS, SEASONS, load_inputs, out_dir

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
    if elapsed < 4 * PERIOD_S:
        return int(elapsed // PERIOD_S) + 1
    return 5 + int((elapsed - 4 * PERIOD_S) // OT_S)


def name_key(name: str, roster: dict | None = None) -> int:
    """Stable integer key of a basketball-reference name: roster id when known, else crc32 under 2**31."""
    name = " ".join(str(name).split())
    if roster and name in roster:
        return int(roster[name])
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


def event_players(row: dict, roster: dict | None = None) -> tuple[str, int | None, int | None]:
    """(kind, out_or_actor_key, in_key) of one play-by-play row: ids when the NBA.com columns are filled,
    else keys from the description."""
    kind = str(row.get("kind") or "")
    p1, p2 = row.get("player1_id"), row.get("player2_id")
    desc = row.get("desc") or ""
    sub = bref_substitution(desc)
    if kind == "substitution" or sub is not None:
        if _has(p1) and _has(p2):
            return "substitution", int(p1), int(p2)
        if sub is not None:
            return "substitution", name_key(sub[1], roster), name_key(sub[0], roster)
        return "substitution", int(p1) if _has(p1) else None, int(p2) if _has(p2) else None
    if _has(p1):
        return kind, int(p1), None
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
        elif a is not None and a not in came_in and a not in starters:
            starters.append(a)
    return starters[:5]


def _side_of(row: dict, home_team_id) -> str | None:
    t = row.get("team")
    if isinstance(t, str) and t in ("home", "away"):
        return t
    tid = row.get("team_id")
    if _has(tid) and _has(home_team_id):
        return "home" if int(tid) == int(home_team_id) else "away"
    return None


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
                     "a": a, "b": b, "points": int(r["points"]) if _has(r.get("points")) else 0})
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
    scoring = sorted(((e["t"], e["side"], e["points"]) for e in rows if e["points"] and e["side"]), key=lambda x: x[0])
    meta = {c: pbp[c].iloc[0] for c in ("season", "split", "heldout", "season_type", "game_date", "source", "fetched_at") if c in pbp.columns}
    out, hp, ap, n = [], 0, 0, 0
    current = {"home": [], "away": []}
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        if t0 in on_at:
            current = on_at[t0]
        if t1 <= t0:
            continue
        home, away = current["home"], current["away"]
        ph = sum(pts for t, s, pts in scoring if t0 < t <= t1 and s == "home")
        pa = sum(pts for t, s, pts in scoring if t0 < t <= t1 and s == "away")
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
    df = pd.DataFrame(out)
    for c in PLAYER_COLUMNS:
        df[c] = df[c].astype("Int32")
    return df


def starters_by_side(starters: pd.DataFrame, game_id: str, home_team_id, away_team_id=None) -> dict:
    """{"home": ids, "away": ids} of one game from the starters table (game_id, team_id, player_id)."""
    if starters is None or len(starters) == 0:
        return {}
    s = starters[starters["game_id"].astype(str) == str(game_id)]
    out = {}
    for tid, grp in s.groupby("team_id"):
        if home_team_id is not None and int(tid) == int(home_team_id):
            out["home"] = [int(x) for x in grp["player_id"]]
        elif away_team_id is not None and int(tid) == int(away_team_id):
            out["away"] = [int(x) for x in grp["player_id"]]
    return out


def build_stints(playbyplay: pd.DataFrame, games: pd.DataFrame | None = None, starters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every game of a play-by-play table -> stints (section 12 columns plus the carried fields)."""
    if len(playbyplay) == 0:
        return pd.DataFrame()
    gcol = "game_id" if "game_id" in playbyplay.columns else "game"
    teams = {}
    if games is not None and len(games):
        teams = {str(r.game_id): (r.home_team_id, r.away_team_id) for r in games.itertuples()}
    if "home_team_id" in playbyplay.columns:
        for g, grp in playbyplay.groupby(gcol):
            teams.setdefault(str(g), (grp["home_team_id"].iloc[0], None))
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
    return pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()


def summarize(stints: pd.DataFrame) -> dict:
    if len(stints) == 0:
        return {"stints": 0, "games": 0, "lineup_ok_share": float("nan")}
    return {"stints": int(len(stints)), "games": int(stints["game_id"].nunique()),
            "lineup_ok_share": float(stints["lineup_ok"].mean()),
            "points_home": int(stints["points_home"].sum()), "points_away": int(stints["points_away"].sum())}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT (default DATA_ROOT; nothing is written for --fixtures without it)")
    ap.add_argument("--seasons", nargs="*", default=list(SEASONS))
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    inputs = load_inputs(a.fixtures, a.data, a.seasons, a.seed)
    stints = build_stints(inputs["playbyplay"], inputs["games"], inputs["starters"])
    out_root = a.out or (a.data if not a.fixtures else None)
    written = []
    if out_root and len(stints):
        for season, grp in stints.groupby("season"):
            p = out_dir(out_root, str(season)) / "stints.parquet"
            grp.reset_index(drop=True).to_parquet(p, index=False)
            written.append(str(p))
    s = summarize(stints)
    s["written"] = written
    s["fixture"] = inputs["fixture"]
    print("stints from play-by-play%s: %s" % (" (fixture output, not a result)" if inputs["fixture"] else "", s))
    return s


if __name__ == "__main__":
    main()
