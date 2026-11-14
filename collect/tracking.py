"""collect/tracking.py: NBA.com tracking aggregates from 2015-16 on (collection plan step 4).

Per player-season (and per team-season where the endpoint has a team mode), league-wide:
  leaguedashplayerptshot   all shots, then one unit per closest-defender range and per touch-time range
  leaguedashptstats        one unit per PtMeasureType (SpeedDistance, Drives, Possessions, Passing, Rebounding,
                           Defense, CatchShoot, PullUpShot, ElbowTouch, PostTouch, PaintTouch, Efficiency), players and teams
  leaguedashplayerstats    the season roster (every player with minutes), also the player list for the per-player units
  playerdashptshotdefend   one unit per player (defended FG% by shot class)
  playerdashptreb          one unit per player (rebound chances, contested, distance)
Per player-game, only with --per-game: boxscoreplayertrackv3 for every game of the season (speed, distance,
touches, passes per player per game; about 1,230 units per season).
Every unit writes its wide table and its long form into aggregates/<season>.parquet (the contract aggregates
table: entity, season, endpoint, split, metric, value). 2015-16 per-game rows of held-out games go to heldout/.

  python -m collect.tracking                                  # 2015-16 .. last completed season
  python -m collect.tracking --seasons 2015-16 --per-game     # adds the per-game tracking box scores
"""
from __future__ import annotations
from collect import common
from collect.nba_stats import (NBAStats, unit, table_unit, season_games, player_track_v3_rows, PT_MEASURE_TYPES,
                               CLOSE_DEF_RANGES, TOUCH_TIME_RANGES)

FAMILY = "tracking"
DEFAULT_SEASONS = common.seasons_from(2015)
PLAYER_IDS = ["player_id", "player_name", "player_last_team_id", "team_id", "team_abbreviation", "group_set", "group_value"]
LONG_KEEP = ["season_type", "split_name", "split_value", "player_or_team", "pt_measure_type", "game_id", "heldout"]


def league_units(season: str, st: str) -> list:
    out = [unit("leaguedashplayerstats", season, st.replace(" ", ""), {"Season": season, "SeasonType": st}, {"season_type": st})]
    tag = st.replace(" ", "")
    out.append(unit("leaguedashplayerptshot", season, tag + "_all", {"Season": season, "SeasonType": st},
                    {"season_type": st, "split_name": "none", "split_value": "all"}))
    for r in CLOSE_DEF_RANGES:
        out.append(unit("leaguedashplayerptshot", season, tag + "_def_" + r.split(" ")[0], {"Season": season, "SeasonType": st, "CloseDefDistRange": r},
                        {"season_type": st, "split_name": "CloseDefDistRange", "split_value": r}))
    for r in TOUCH_TIME_RANGES:
        out.append(unit("leaguedashplayerptshot", season, tag + "_touch_" + r.replace(" ", "")[5:9], {"Season": season, "SeasonType": st, "TouchTimeRange": r},
                        {"season_type": st, "split_name": "TouchTimeRange", "split_value": r}))
    for m in PT_MEASURE_TYPES:
        for who in ("Player", "Team"):
            out.append(unit("leaguedashptstats", season, "%s_%s_%s" % (tag, m, who), {"Season": season, "SeasonType": st, "PtMeasureType": m, "PlayerOrTeam": who},
                            {"season_type": st, "pt_measure_type": m, "player_or_team": who}))
    return out


def player_units(fam: common.Family, season: str, st: str) -> list:
    """One playerdashptshotdefend and one playerdashptreb unit per player of the season roster part."""
    p = fam.part_path("leaguedashplayerstats", season, st.replace(" ", ""))
    if not p.exists():
        return []
    roster = common.read_parquet(p)
    out = []
    for pid, tid in zip(roster["player_id"], roster["team_id"]):
        for ep in ("playerdashptshotdefend", "playerdashptreb"):
            out.append(unit(ep, season, "%s_%d" % (st.replace(" ", ""), int(pid)),
                            {"Season": season, "SeasonType": st, "PlayerID": int(pid), "TeamID": int(tid)},
                            {"season_type": st, "player_id": int(pid)}))
    return out


def game_units(stats: NBAStats, fam: common.Family, season: str, st: str) -> list:
    out = []
    for g in season_games(stats, fam, season, st):
        out.append(unit("boxscoreplayertrackv3", season, g["game_id"], {"GameID": g["game_id"]},
                        {"season_type": st, "game_date": g["date"], "split": g["split"], "heldout": g["heldout"]}, rows_fn=player_track_v3_rows))
    return out


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    ap.add_argument("--per-game", action="store_true", help="also boxscoreplayertrackv3 for every game of the seasons")
    ap.add_argument("--skip-players", action="store_true", help="skip the per-player units (shot defend, rebounding)")
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    stats = NBAStats(fam.raw, common.Throttle(a.rpm))
    do = lambda u: table_unit(stats, fam, u, long_ids=PLAYER_IDS, long_keep=LONG_KEEP)
    phase1 = [u for s in a.seasons for st in a.season_types for u in league_units(s, st)]
    final = common.run_units(fam, phase1, do, a, "tracking: league-wide units")
    if not a.skip_players:
        phase2 = [u for s in a.seasons for st in a.season_types for u in player_units(fam, s, st)]
        final = common.run_units(fam, phase2, do, a, "tracking: per-player units")
    if a.per_game:
        phase3 = [u for s in a.seasons for st in a.season_types for u in game_units(stats, fam, s, st)]
        final = common.run_units(fam, phase3, do, a, "tracking: per-game units")
    return final


if __name__ == "__main__":
    main()
