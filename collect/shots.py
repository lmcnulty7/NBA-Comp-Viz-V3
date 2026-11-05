"""collect/shots.py: every field-goal attempt, league-wide, from stats.nba.com ShotChartDetail (collection plan step 2).

Default seasons: 2015-16 and the five after it (2016-17 .. 2020-21). One unit per (season, season type, team):
ShotChartDetail with TeamID set and PlayerID 0 returns every attempt of that team's games on offense, so
30 units cover a season (about 200,000 rows). Rows become the shots table of docs/DATA_CONTRACT.md: the
SportVU-derived fields (release_frame, closest_defender_id, closest_defender_dist_ft,
defender_closing_speed_ftps) are left null here and filled by the later SportVU join. 2015-16 rows of the
held-out games of sportvu/splits.json carry heldout=true and are compacted into heldout/.

  python -m collect.shots                                   # 2015-16 .. 2020-21, regular season
  python -m collect.shots --seasons 2015-16 --season-types "Regular Season" Playoffs
"""
from __future__ import annotations
from collect import common
from collect.nba_stats import NBAStats, unit, table_unit

FAMILY = "shots"
DEFAULT_SEASONS = [common.season_str(y) for y in range(2015, 2021)]
NULL_SPORTVU = ("release_frame", "closest_defender_id", "closest_defender_dist_ft", "defender_closing_speed_ftps")


def convert(df, u: dict):
    """ShotChartDetail rows -> the shots contract columns (plus every source column, snake_case)."""
    import pandas as pd
    out = pd.DataFrame({
        "game_id": df["game_id"].astype(str),
        "event_id": df["game_event_id"].astype("int32"),
        "player_id": df["player_id"].astype("int32"),
        "player": df["player_name"],
        "team_id": df["team_id"].astype("int32"),
        "team": df["team_name"],
        "period": df["period"].astype("int8"),
        "clock_s": (df["minutes_remaining"].astype(float) * 60 + df["seconds_remaining"].astype(float)).astype("float32"),
        "loc_x": df["loc_x"].astype("float32"),
        "loc_y": df["loc_y"].astype("float32"),
        "shot_distance_ft": df["shot_distance"].astype("float32"),
        "shot_type": df["shot_type"].astype(str).str.extract(r"^(\dPT)")[0],
        "action_type": df["action_type"],
        "zone_basic": df["shot_zone_basic"],
        "zone_area": df["shot_zone_area"],
        "zone_range": df["shot_zone_range"],
        "made": df["shot_made_flag"].astype(int) == 1,
        "game_date": df["game_date"].astype(str).str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True),
        "home_team": df["htm"],
        "away_team": df["vtm"],
    })
    for c in NULL_SPORTVU:
        out[c] = pd.Series([None] * len(out), dtype="float32" if c != "closest_defender_id" else "Int32")
    out["sportvu_game"] = out.apply(lambda r: "%s.%s.%s.%s.at.%s" % (r.game_date[5:7], r.game_date[8:10], r.game_date[:4], r.away_team, r.home_team), axis=1) if len(out) else pd.Series(dtype=str)
    if u["season"] == "2015-16" and len(out):
        out["split"] = out.apply(lambda r: common.heldout_split(r.game_date, r.away_team, r.home_team), axis=1)
    else:
        out["split"] = None
    out["heldout"] = out["split"].notna()
    out["season_type"] = u["fields"]["season_type"]
    for c in ("source", "endpoint", "season", "fetched_at"):
        out[c] = df[c]
    return out


def units(seasons, season_types):
    out = []
    for season in seasons:
        for st in season_types:
            for team in common.TEAM_IDS:
                out.append(unit("shotchartdetail", season, "%s_%d" % (st.replace(" ", ""), team),
                                {"Season": season, "SeasonType": st, "TeamID": team, "PlayerID": 0, "ContextMeasure": "FGA"},
                                fields={"season_type": st}, set_name="Shot_Chart_Detail"))
    return out


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    stats = NBAStats(fam.raw, common.Throttle(a.rpm))
    return common.run_units(fam, units(a.seasons, a.season_types), lambda u: table_unit(stats, fam, u, convert), a, "shots")


if __name__ == "__main__":
    main()
