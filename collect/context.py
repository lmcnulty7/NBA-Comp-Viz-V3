"""collect/context.py: matchups, play types, hustle, lineups and rotations (collection plan step 6).

Per season (regular season by default), league-wide:
  leagueseasonmatchups     one unit per defending team (every defender x offensive player pair of its games)
  synergyplaytypes         one unit per play type x players/teams x offensive/defensive (44 per season)
  leaguehustlestatsplayer  one unit per season
  leaguedashlineups        five-man lineups, one unit per team x MeasureType (Base, Advanced)
Per game, only with --per-game: boxscorematchupsv3 (defender x offensive player per game) and gamerotation
(each player's in and out times per game; the stints collector reads the same cache).
Each endpoint family starts in a different season (see collect.probe and the PR notes); a season before
the start returns empty units, recorded as "empty" in the manifest, never as failures.

  python -m collect.context --seasons 2015-16 2016-17 2017-18
  python -m collect.context --seasons 2017-18 --per-game
"""
from __future__ import annotations
from collect import common
from collect.nba_stats import NBAStats, unit, table_unit, season_games, matchups_v3_rows, SYNERGY_PLAY_TYPES

FAMILY = "context"
DEFAULT_SEASONS = common.seasons_from(2015)
IDS = ["player_id", "player_name", "team_id", "team_abbreviation", "team_name", "group_id", "group_name",
       "off_player_id", "off_player_name", "def_player_id", "def_player_name", "off_team_id", "def_team_id", "game_id"]
LONG_KEEP = ["season_type", "play_type", "type_grouping", "player_or_team", "measure_type", "heldout"]


def season_units(season: str, st: str) -> list:
    tag = st.replace(" ", "")
    out = []
    for team in common.TEAM_IDS:
        out.append(unit("leagueseasonmatchups", season, "%s_def%d" % (tag, team), {"Season": season, "SeasonType": st, "DefTeamID": team}, {"season_type": st}))
    for pt in SYNERGY_PLAY_TYPES:
        for who in ("P", "T"):
            for grp in ("offensive", "defensive"):
                out.append(unit("synergyplaytypes", season, "%s_%s_%s_%s" % (tag, pt, who, grp),
                                {"SeasonYear": season, "SeasonType": st, "PlayType": pt, "PlayerOrTeam": who, "TypeGrouping": grp},
                                {"season_type": st, "play_type": pt, "player_or_team": who, "type_grouping": grp}))
    out.append(unit("leaguehustlestatsplayer", season, tag, {"Season": season, "SeasonType": st}, {"season_type": st}))
    for team in common.TEAM_IDS:
        for mt in ("Base", "Advanced"):
            out.append(unit("leaguedashlineups", season, "%s_%d_%s" % (tag, team, mt), {"Season": season, "SeasonType": st, "TeamID": team, "MeasureType": mt},
                            {"season_type": st, "measure_type": mt}))
    return out


def game_units(stats: NBAStats, fam: common.Family, season: str, st: str) -> list:
    out = []
    for g in season_games(stats, fam, season, st):
        f = {"season_type": st, "game_date": g["date"], "split": g["split"], "heldout": g["heldout"]}
        out.append(unit("boxscorematchupsv3", season, g["game_id"], {"GameID": g["game_id"]}, f, rows_fn=matchups_v3_rows))
        out.append(unit("gamerotation", season, g["game_id"], {"GameID": g["game_id"]}, f, rows_fn=rotation_rows))
    return out


def rotation_rows(payload: dict) -> list[dict]:
    """gamerotation has two result sets (AwayTeam, HomeTeam): one table with a home flag."""
    from collect.nba_stats import result_sets
    rows = []
    for name, df in result_sets(payload).items():
        df = df.copy()
        df["home"] = name.lower().startswith("home")
        rows += df.to_dict("records")
    return rows


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    ap.add_argument("--per-game", action="store_true", help="also boxscorematchupsv3 and gamerotation per game")
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    stats = NBAStats(fam.raw, common.Throttle(a.rpm))
    do = lambda u: table_unit(stats, fam, u, long_ids=IDS, long_keep=LONG_KEEP)
    final = common.run_units(fam, [u for s in a.seasons for st in a.season_types for u in season_units(s, st)], do, a, "context: season units")
    if a.per_game:
        final = common.run_units(fam, [u for s in a.seasons for st in a.season_types for u in game_units(stats, fam, s, st)], do, a, "context: per-game units")
    return final


if __name__ == "__main__":
    main()
