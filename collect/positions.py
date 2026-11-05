"""collect/positions.py: listed positions per player-season from basketball-reference.com
(https://www.basketball-reference.com/leagues/NBA_<endyear>_per_game.html).

One unit per season, one request each. The per-game table lists every player once per team (a player traded
during the season has a total row, team TOT or 2TM/3TM, plus one row per team):

  <table id="per_game_stats"> <thead> <tr><th data-stat="ranker">Rk</th><th data-stat="player">Player</th><th data-stat="pos">Pos</th>
      <th data-stat="age">Age</th><th data-stat="team_id">Tm</th> ... </tr> </thead>   (2024-25 pages: data-stat="team_name_abbr", "Team")
    <tbody> <tr class="full_table"><td data-stat="ranker">1</td>
      <td data-stat="player" data-append-csv="curryst01" csk="Curry,Stephen"><a href="/players/c/curryst01.html">Stephen Curry</a></td>
      <td data-stat="pos">PG</td><td data-stat="age">27</td><td data-stat="team_id"><a href="/teams/GSW/2016.html">GSW</a></td>
      <td data-stat="g">79</td> ... <td data-stat="mp_per_g">34.2</td> ... </tr>
  (the 2024 redesign renames player to name_display, team_id to team_name_abbr, g and gs to games and games_started;
  both spellings are read)
Total rows are dropped: every row of the output names a real team, and the position is the same on a player's rows.

Output: --out/positions/positions/<season>.parquet per season and, after compaction, their union in one file,
--out/positions.parquet (player_slug, player, season, position, team, age, games, mp_per_g, provenance). Positions
are basketball-reference's listed ones (PG, SG, SF, PF, C, or two joined with a hyphen).

The name join to tracking player ids goes through analytics/rosters (docs/DATA_CONTRACT.md section 16): the rosters
table carries bref_slug per SportVU player_id and game, matched by name through analytics.rosters.match_names, so
positions join rosters on (bref_slug = player_slug, season) and never by name; join_rosters(positions, rosters) does
it. A player whose roster row has bref_slug null (match_method none) gets no position.

  python -m collect.positions --seasons 2015-16 2016-17
"""
from __future__ import annotations
from pathlib import Path

from collect import common
from collect import bref_schedule as sched

FAMILY = "positions"
ENDPOINT = "positions"
RAW_ENDPOINT = "bref_per_game"
SOURCE = "basketball-reference.com"
MAX_RPM = sched.MAX_RPM
DEFAULT_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TOTAL_TEAMS = {"TOT", "2TM", "3TM", "4TM", "5TM"}
COLUMNS = ("player_slug", "player", "season", "position", "team", "age", "games", "mp_per_g", "source", "endpoint", "fetched_at")


def per_game_url(season: str) -> str:
    return "%s/leagues/NBA_%d_per_game.html" % (sched.HOST, sched.end_year(season))


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────
def _first(cells: dict, *stats):
    """The first of several data-stat spellings present in the row (page redesigns rename columns)."""
    return next((cells[s] for s in stats if s in cells), None)


def parse_per_game(html: str, where: str = "") -> list[dict]:
    """Every player row of the per-game table -> dict(player_slug, player, position, team, age, games, mp_per_g, is_total).
    ValueError when the page has no per-game table."""
    tables = sched.require_table(html, r"^per_game_stats$", "id=per_game_stats", where)
    rows = []
    body = tables[0].find("tbody") or tables[0]
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = sched.row_cells(tr)
        pc = _first(cells, "player", "name_display")
        slug = sched.slug_of(pc)
        if pc is None or slug is None:
            continue
        team_cell = _first(cells, "team_id", "team_name_abbr")
        team = sched.team_of_link(team_cell) or (team_cell.get_text(" ", strip=True) if team_cell is not None else None)
        g_cell = _first(cells, "g", "games")
        age, mp = sched.cell_num(cells, "age"), sched.cell_num(cells, "mp_per_g")
        g = sched.cell_num(cells, g_cell["data-stat"]) if g_cell is not None else None
        rows.append({"player_slug": slug, "player": pc.get_text(" ", strip=True), "position": sched.cell_text(cells, "pos"), "team": team,
                     "age": int(age) if age is not None else None, "games": int(g) if g is not None else None, "mp_per_g": mp,
                     "is_total": team in TOTAL_TEAMS})
    return rows


def rows_for_season(season: str, players: list[dict], fetched_at: str):
    import pandas as pd
    recs = [{k: p[k] for k in ("player_slug", "player", "position", "team", "age", "games", "mp_per_g")} | {"season": season, "source": SOURCE,
             "endpoint": "leagues/NBA_%d_per_game.html" % sched.end_year(season), "fetched_at": fetched_at} for p in players if not p["is_total"]]
    df = pd.DataFrame(recs, columns=list(COLUMNS))
    for c in ("age", "games"):
        df[c] = df[c].astype("Int16")
    df["mp_per_g"] = df["mp_per_g"].astype("float32")
    for c in ("player_slug", "player", "season", "position", "team", "source", "endpoint", "fetched_at"):
        df[c] = df[c].astype("string")
    return df.drop_duplicates(["player_slug", "team"]).reset_index(drop=True)


def positions_by_player(df):
    """One row per (player_slug, season): the position of the row with the most games (ties: the first)."""
    d = df.sort_values(["player_slug", "season", "games"], ascending=[True, True, False], kind="stable")
    return d.drop_duplicates(["player_slug", "season"]).reset_index(drop=True)


def join_rosters(positions, rosters):
    """The rosters table (analytics/rosters, section 16: player_id, bref_slug, sportvu_game, ...) joined to a position per
    player-season on (bref_slug, season); the season is derived from sportvu_game when the rosters frame has none.
    Returns the rosters rows with position (null when the slug is null or unknown that season)."""
    import pandas as pd
    r = rosters.copy()
    if "season" not in r.columns:
        from collect.bref_shots import season_of
        r["season"] = [season_of("%s-%s-%s" % (g[6:10], g[0:2], g[3:5])) for g in r["sportvu_game"].astype(str)]
    p = positions_by_player(positions)[["player_slug", "season", "position"]].rename(columns={"player_slug": "bref_slug"})
    p["bref_slug"], p["season"] = p["bref_slug"].astype(object), p["season"].astype(object)
    r["bref_slug"], r["season"] = r["bref_slug"].astype(object), r["season"].astype(object)
    out = r.merge(p, on=["bref_slug", "season"], how="left")
    out["position"] = out["position"].astype("string")
    return out


# ── one unit ───────────────────────────────────────────────────────────────────────────────────────
def do_season(unit: dict, fam: common.Family, throttle: common.Throttle) -> tuple[int, dict]:
    season = unit["season"]
    rec = sched.page_record(fam, RAW_ENDPOINT, unit["params"], per_game_url(season), throttle)
    players = parse_per_game(rec["body"], season)
    df = rows_for_season(season, players, rec["fetched_at"])
    if len(df):
        fam.write_part(ENDPOINT, season, season, df)
    return len(df), {"fetched_at": rec["fetched_at"], "endpoint": "leagues/NBA_%d_per_game.html" % sched.end_year(season),
                     "players": int(df["player_slug"].nunique()) if len(df) else 0, "total_rows_dropped": sum(p["is_total"] for p in players),
                     "positions": {k: int(v) for k, v in df.groupby("position").size().items()} if len(df) else {}}


def write_union(fam: common.Family) -> Path | None:
    """positions/<season>.parquet of every season -> --out/positions.parquet (the one-file table the plan names)."""
    import pandas as pd
    files = sorted((fam.dir / ENDPOINT).glob("*.parquet")) if (fam.dir / ENDPOINT).exists() else []
    if not files:
        return None
    path = fam.dir.parent / "positions.parquet"
    common.write_parquet(pd.concat([common.read_parquet(f) for f in files], ignore_index=True), path)
    return path


def units_for(seasons: list[str]) -> list[tuple[str, dict]]:
    return [(common.Manifest.key(ENDPOINT, s, s), {"season": s, "params": {"season": s}}) for s in seasons]


def main(argv=None) -> str:
    ap = sched.bref_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); print("union:", write_union(fam)); return "compact only"
    units = units_for(a.seasons)
    throttle = sched.capped_throttle(a.rpm)
    sched.estimate_requests(fam, units, a, lambda u: (RAW_ENDPOINT, u["params"]))
    final = common.run_units(fam, units, lambda u: do_season(u, fam, throttle), a, "basketball-reference positions, %d seasons" % len(units))
    if not a.dry_run:
        print("union:", write_union(fam), flush=True)
    return final


if __name__ == "__main__":
    main()
