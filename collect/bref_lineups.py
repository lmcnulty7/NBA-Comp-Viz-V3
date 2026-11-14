"""collect/bref_lineups.py: per-season five-man lineup tables from basketball-reference.com
(https://www.basketball-reference.com/teams/<TEAM>/<endyear>/lineups/).

One unit per team-season (30 a season, the teams from the collected schedule or the fixed list), one request each.
The page carries the 5-man, 4-man, 3-man and 2-man tables (mostly inside HTML comments); this driver keeps the
five-man one:

  <table id="lineups_5-man_"> <thead> <tr class="over_header"> ... <th colspan="..">Per 100 Poss (Net)</th> ... </tr>
    <tr><th data-stat="ranker">Rk</th><th data-stat="lineup">Lineup</th><th data-stat="mp">MP</th><th data-stat="pace">Pace</th>
        <th data-stat="diff_fg">FG</th> ... <th data-stat="diff_pts">PTS</th> ... </tr> </thead>
    <tbody> <tr><th data-stat="ranker">1</th>
      <td data-stat="lineup" csk="curryst01;greendr01;..."><a href="/players/c/curryst01.html">S. Curry</a> | <a ...>D. Green</a> | ...</td>
      <td data-stat="mp">812:09</td><td data-stat="pace">99.8</td><td data-stat="diff_fg">+3.1</td> ... <td data-stat="diff_pts">+13.6</td> ... </tr>
The net columns are the team minus its opponents per 100 possessions with the lineup on the floor; net_rating is
the PTS column of that block (data-stat diff_pts, with net_rtg, pts_per_poss and pts accepted as renames). Every
other numeric column of the row is kept as JSON in stats_json under its data-stat name, so a column the page adds or
renames never changes the parquet schema.

Output: --out/bref_lineups/lineups/<season>.parquet, one row per lineup: season, team, rank, players (the five slugs
sorted and joined with |), player_names, p1..p5 (the sorted slugs), minutes, pace, net_rating, stats_json,
provenance. Team-season aggregates carry no game, so there is no held-out tagging. At most 17 requests a minute;
the first HTTP 429 or 403 stops the run.

  python -m collect.bref_lineups --seasons 2015-16 2016-17 [--teams GSW CLE]
"""
from __future__ import annotations
import json
import re

from collect import common
from collect import bref_schedule as sched

FAMILY = "bref_lineups"
ENDPOINT = "lineups"
RAW_ENDPOINT = "bref_lineups"
SOURCE = "basketball-reference.com"
MAX_RPM = sched.MAX_RPM
DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TABLE_ID_RE = r"^lineups_5-man_?$"
NET_RATING_STATS = ("diff_pts", "net_rtg", "pts_per_poss", "pts")
COLUMNS = ("season", "team", "rank", "players", "player_names", "p1", "p2", "p3", "p4", "p5", "n_players", "minutes", "pace", "net_rating",
           "stats_json", "source", "endpoint", "fetched_at")


def lineups_url(team: str, season: str) -> str:
    return "%s/teams/%s/%d/lineups/" % (sched.HOST, team, sched.end_year(season))


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────
def parse_lineup_cell(cell) -> tuple[list[str], list[str]]:
    """(slugs, names) of a lineup cell: the player links, else the csk slugs with the text split on |."""
    slugs, names = [], []
    for a in cell.find_all("a", href=True):
        m = re.search(r"/players/\w/([^/.]+)\.html", a["href"])
        if m:
            slugs.append(m.group(1)); names.append(a.get_text(" ", strip=True))
    if not slugs and cell.get("csk"):
        slugs = [s for s in re.split(r"[;|,]\s*", cell["csk"]) if s]
        names = [n.strip() for n in cell.get_text(" ", strip=True).split("|") if n.strip()]
    return slugs, names


def parse_lineups(html: str, where: str = "") -> list[dict]:
    """Every row of the five-man table -> dict(rank, slugs, names, minutes, pace, net_rating, stats). ValueError when the
    page has no five-man table."""
    tables = sched.require_table(html, TABLE_ID_RE, "id=lineups_5-man_", where)
    rows = []
    body = tables[0].find("tbody") or tables[0]
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = sched.row_cells(tr)
        if "lineup" not in cells:
            continue
        slugs, names = parse_lineup_cell(cells["lineup"])
        if not slugs:
            continue
        stats = {k: sched.cell_num(cells, k) for k in cells if k not in ("ranker", "lineup", "mp")}
        stats = {k: v for k, v in stats.items() if v is not None}
        net = next((stats[k] for k in NET_RATING_STATS if k in stats), None)
        rank = sched.cell_num(cells, "ranker")
        rows.append({"rank": int(rank) if rank is not None else None, "slugs": slugs, "names": names,
                     "minutes": sched.minutes_of(sched.cell_text(cells, "mp")), "pace": stats.get("pace"), "net_rating": net, "stats": stats})
    return rows


def rows_for_team(team: str, season: str, lineups: list[dict], fetched_at: str):
    import pandas as pd
    recs = []
    for r in lineups:
        order = sorted(range(len(r["slugs"])), key=lambda i: r["slugs"][i])
        slugs = [r["slugs"][i] for i in order]
        names = [r["names"][i] if i < len(r["names"]) else "" for i in order]
        rec = {"season": season, "team": team, "rank": r["rank"], "players": "|".join(slugs), "player_names": "|".join(names),
               "n_players": len(slugs), "minutes": r["minutes"], "pace": r["pace"], "net_rating": r["net_rating"],
               "stats_json": json.dumps(r["stats"], sort_keys=True), "source": SOURCE,
               "endpoint": "teams/%s/%d/lineups/" % (team, sched.end_year(season)), "fetched_at": fetched_at}
        for k in range(5):
            rec["p%d" % (k + 1)] = slugs[k] if k < len(slugs) else None
        recs.append(rec)
    df = pd.DataFrame(recs, columns=list(COLUMNS))
    df["rank"] = df["rank"].astype("Int32")
    df["n_players"] = df["n_players"].astype("int8")
    for c in ("minutes", "pace", "net_rating"):
        df[c] = df[c].astype("float32")
    for c in ("season", "team", "players", "player_names", "p1", "p2", "p3", "p4", "p5", "stats_json", "source", "endpoint", "fetched_at"):
        df[c] = df[c].astype("string")
    return df


# ── one unit ───────────────────────────────────────────────────────────────────────────────────────
def do_team(unit: dict, fam: common.Family, throttle: common.Throttle) -> tuple[int, dict]:
    team, season = unit["team"], unit["season"]
    rec, lineups = sched.fetch_and_parse(fam, RAW_ENDPOINT, unit["params"], lineups_url(team, season), throttle, lambda html: parse_lineups(html, "%s %s" % (team, season)))
    df = rows_for_team(team, season, lineups, rec["fetched_at"])
    if len(df):
        fam.write_part(ENDPOINT, season, team, df)
    return len(df), {"fetched_at": rec["fetched_at"], "endpoint": "teams/%s/%d/lineups/" % (team, sched.end_year(season)),
                     "minutes": float(df["minutes"].sum()) if len(df) else 0.0, "not_five": int((df["n_players"] != 5).sum()) if len(df) else 0}


def units_for(fam: common.Family, seasons: list[str], teams: list[str] | None = None) -> list[tuple[str, dict]]:
    out = []
    for s in seasons:
        for t in (teams or sched.teams_for_season(fam, s)):
            out.append((common.Manifest.key(ENDPOINT, s, t), {"season": s, "team": t, "params": {"team": t, "season": s}}))
    return out


def main(argv=None) -> str:
    ap = sched.bref_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    ap.add_argument("--teams", nargs="*", help="basketball-reference team codes (default: the season's schedule, else all 30)")
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    units = units_for(fam, a.seasons, a.teams)
    throttle = sched.capped_throttle(a.rpm)
    sched.estimate_requests(fam, units, a, lambda u: (RAW_ENDPOINT, u["params"]))
    return common.run_units(fam, units, lambda u: do_team(u, fam, throttle), a, "basketball-reference lineups, %d team-seasons" % len(units))


if __name__ == "__main__":
    main()
