"""collect/probe.py: the earliest season each stats.nba.com endpoint family returns data (the doc's open question).

One request per season per family, from --from (1996) to the last completed season, through the shared raw
cache. The per-game families (play-by-play, rotation, matchups v3, player tracking v3, win probability)
first list the season's games (leaguegamelog) and probe its first game, so they cost two requests a
season. A 400 or an empty result set counts as "no data" for that season; the first season with rows is
the family's earliest season. Output: --out/probe/probe.json and a printed table.

  python -m collect.probe                        # every family, 1996-97 .. last completed season, about 450 requests
  python -m collect.probe --families shots synergy hustle --from 2010
  python -m collect.probe --save-fixtures tests/fixtures/collect     # trimmed real responses for the tests
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from collect import common
from collect.nba_stats import NBAStats, ClientError, result_sets, games_from_log, dump_fixture, params_for

GSW, LEBRON = 1610612744, 2544   # a team and a player that exist in every probed season (LeBron from 2003-04)

FAMILIES: dict[str, dict] = {
    "shots":        {"endpoint": "shotchartdetail", "params": lambda s, g: {"Season": s, "TeamID": GSW, "PlayerID": 0}},
    "pbp":          {"endpoint": "playbyplayv2", "params": lambda s, g: {"GameID": g}, "per_game": True},
    "ptshot":       {"endpoint": "leaguedashplayerptshot", "params": lambda s, g: {"Season": s}},
    "ptstats":      {"endpoint": "leaguedashptstats", "params": lambda s, g: {"Season": s, "PtMeasureType": "SpeedDistance"}},
    "shotdefend":   {"endpoint": "playerdashptshotdefend", "params": lambda s, g: {"Season": s, "PlayerID": LEBRON, "TeamID": 0}},
    "ptreb":        {"endpoint": "playerdashptreb", "params": lambda s, g: {"Season": s, "PlayerID": LEBRON, "TeamID": 0}},
    "playertrack":  {"endpoint": "boxscoreplayertrackv3", "params": lambda s, g: {"GameID": g}, "per_game": True},
    "matchups":     {"endpoint": "leagueseasonmatchups", "params": lambda s, g: {"Season": s, "DefTeamID": GSW}},
    "matchups_v3":  {"endpoint": "boxscorematchupsv3", "params": lambda s, g: {"GameID": g}, "per_game": True},
    "synergy":      {"endpoint": "synergyplaytypes", "params": lambda s, g: {"SeasonYear": s}},
    "hustle":       {"endpoint": "leaguehustlestatsplayer", "params": lambda s, g: {"Season": s}},
    "lineups":      {"endpoint": "leaguedashlineups", "params": lambda s, g: {"Season": s, "TeamID": GSW}},
    "rotation":     {"endpoint": "gamerotation", "params": lambda s, g: {"GameID": g}, "per_game": True},
    "winprob":      {"endpoint": "winprobabilitypbp", "params": lambda s, g: {"GameID": g}, "per_game": True},
    "roster":       {"endpoint": "leaguedashplayerstats", "params": lambda s, g: {"Season": s}},
}


def row_count(payload: dict) -> int:
    sets = result_sets(payload)
    if sets:
        return int(sum(len(df) for df in sets.values()))
    for k in ("boxScoreMatchups", "boxScorePlayerTrack"):          # v3 shapes
        box = payload.get(k) or {}
        return int(sum(len(t.get("players", [])) for t in (box.get("homeTeam") or {}, box.get("awayTeam") or {}) if t))
    return 0


def earliest(results: dict) -> str | None:
    """results {season: rows or 'error: ...'} -> the first season with rows > 0."""
    for season in sorted(results):
        v = results[season]
        if isinstance(v, int) and v > 0:
            return season
    return None


def probe(stats: NBAStats, families: list[str], seasons: list[str], season_type: str = "Regular Season") -> dict:
    out = {"season_type": season_type, "probed_at": common.utc_now(), "families": {}}
    first_game: dict[str, str | None] = {}
    for fam in families:
        spec = FAMILIES[fam]
        res = {}
        for season in seasons:
            game = None
            if spec.get("per_game"):
                if season not in first_game:
                    try:
                        games = games_from_log(stats.get("leaguegamelog", {"Season": season, "SeasonType": season_type}))
                        first_game[season] = games[0]["game_id"] if games else None
                    except ClientError as e:
                        if e.stop_run:
                            raise
                        first_game[season] = None
                game = first_game[season]
                if game is None:
                    res[season] = "error: no game listed"
                    continue
            try:
                res[season] = row_count(stats.get(spec["endpoint"], spec["params"](season, game)))
            except ClientError as e:
                if e.stop_run:
                    raise
                res[season] = "error: %s" % str(e)[:120]
            print("%-12s %s %s" % (fam, season, res[season]), flush=True)
        out["families"][fam] = {"endpoint": spec["endpoint"], "earliest_season": earliest(res), "by_season": res}
    return out


def table(out: dict) -> str:
    lines = ["%-12s %-26s %-10s %s" % ("family", "endpoint", "earliest", "seasons with rows / probed")]
    for fam, r in out["families"].items():
        n = sum(1 for v in r["by_season"].values() if isinstance(v, int) and v > 0)
        lines.append("%-12s %-26s %-10s %d / %d" % (fam, r["endpoint"], r["earliest_season"] or "none", n, len(r["by_season"])))
    return "\n".join(lines)


def save_fixtures(stats: NBAStats, out: dict, dest: Path, max_rows: int = 12) -> list[str]:
    """For each family, the earliest season's cached response trimmed to a few rows -> dest/<endpoint>.json."""
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for fam, r in out["families"].items():
        season = r["earliest_season"]
        if not season:
            continue
        spec = FAMILIES[fam]
        game = None
        if spec.get("per_game"):
            game = games_from_log(stats.get("leaguegamelog", {"Season": season, "SeasonType": out["season_type"]}))[0]["game_id"]
        params = params_for(spec["endpoint"], spec["params"](season, game))
        p = dest / (spec["endpoint"] + ".json")
        dump_fixture(stats.cache, spec["endpoint"], params, p, max_rows)
        written.append(str(p))
    return written


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(common.DEFAULT_OUT))
    ap.add_argument("--families", nargs="*", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--from", dest="first", type=int, default=1996, help="first season start year (default 1996)")
    ap.add_argument("--to", dest="last", type=int, help="last season start year (default: last completed)")
    ap.add_argument("--rpm", type=float, default=30.0)
    ap.add_argument("--save-fixtures", help="also write trimmed responses to this directory")
    a = ap.parse_args(argv)
    fam = common.Family(a.out, "probe")
    stats = NBAStats(fam.raw, common.Throttle(a.rpm))
    seasons = common.seasons_from(a.first, a.last)
    out = probe(stats, a.families, seasons)
    (fam.dir / "probe.json").write_text(json.dumps(out, indent=1) + "\n")
    print(table(out))
    if a.save_fixtures:
        print("fixtures:", save_fixtures(stats, out, Path(a.save_fixtures)))
    print("%d requests made, results in %s" % (stats.requests_made, fam.dir / "probe.json"))
    return str(fam.dir / "probe.json")


if __name__ == "__main__":
    main()
