"""collect/stints.py: multi-season play-by-play with substitutions, rebuilt into stints (collection plan step 7).

Default: the five last completed seasons, regular season. One unit per game, three outputs:
  playbyplay/<season>.parquet   every PlayByPlayV2 event in the contract play-by-play columns (no frames:
                                frame is -1 outside the SportVU window) plus game_id
  rotation/<season>.parquet     GameRotation rows (each player's in and out times, tenths of a second)
  stints/<season>.parquet       the contract stints table: the ten player ids on court, start and end, points
                                each side, built from the rotation boundaries and the play-by-play scores
About 1,230 games a season, two requests each. 2015-16 games of sportvu/splits.json held-out games are
tagged heldout and compacted into heldout/. The raw cache is shared with collect.context, so a per-game
context run does not refetch rotations.

  python -m collect.stints                              # five seasons, about 12,300 requests
  python -m collect.stints --seasons 2015-16 --limit 20
"""
from __future__ import annotations
from collect import common
from collect.nba_stats import NBAStats, SOURCE, first_set, season_games, unit
from collect.pbp import nba_events
from collect.context import rotation_rows

FAMILY = "stints"
N_SEASONS = 5
DEFAULT_SEASONS = common.seasons_from(common.last_completed_season() - N_SEASONS + 1)


def period_of(elapsed: float) -> int:
    """Period containing an elapsed game second (720 s regulation periods, 300 s overtimes)."""
    if elapsed < 4 * 720:
        return int(elapsed // 720) + 1
    return 5 + int((elapsed - 4 * 720) // 300)


def build_stints(game_id: str, rotation: list[dict], events: list[dict]) -> list[dict]:
    """Rotation rows (person_id, home, in_time_real, out_time_real) + play-by-play events (period, clock_s,
    team, points) -> one row per interval between any substitution, with the five home and five away
    player ids on court and the points each side scored. Times in tenths of a second (the endpoint's unit;
    a game whose largest time is under 3000 is read as seconds)."""
    if not rotation:
        return []
    scale = 0.1 if max(float(r["out_time_real"]) for r in rotation) > 3000 else 1.0
    spans = [(int(r["person_id"]), bool(r["home"]), float(r["in_time_real"]) * scale, float(r["out_time_real"]) * scale) for r in rotation]
    cuts = sorted({t for _, _, a, b in spans for t in (a, b)})
    scoring = sorted(((common.elapsed_s(e["period"], e["clock_s"]), e["team"], e["points"]) for e in events if e.get("points")), key=lambda x: x[0])
    rows, home_pts, away_pts = [], 0, 0
    for i, (t0, t1) in enumerate(zip(cuts[:-1], cuts[1:])):
        if t1 <= t0:
            continue
        home = sorted(p for p, h, a, b in spans if h and a <= t0 and b >= t1)
        away = sorted(p for p, h, a, b in spans if not h and a <= t0 and b >= t1)
        ph = sum(pts for t, team, pts in scoring if t0 < t <= t1 and team == "home")
        pa = sum(pts for t, team, pts in scoring if t0 < t <= t1 and team == "away")
        home_pts += ph; away_pts += pa
        row = {"game_id": game_id, "stint": i, "start_s": round(t0, 1), "end_s": round(t1, 1), "duration_s": round(t1 - t0, 1),
               "period_start": period_of(t0), "period_end": period_of(max(t1 - 0.1, t0)),
               "n_home": len(home), "n_away": len(away), "home_ids": "|".join(map(str, home)), "away_ids": "|".join(map(str, away)),
               "points_home": ph, "points_away": pa, "score_home": home_pts, "score_away": away_pts, "lineup_ok": len(home) == 5 and len(away) == 5}
        for k in range(5):
            row["home_%d" % (k + 1)] = home[k] if k < len(home) else None
            row["away_%d" % (k + 1)] = away[k] if k < len(away) else None
        rows.append(row)
    return rows


def do_game(stats: NBAStats, fam: common.Family, u: dict) -> tuple[int, dict]:
    import pandas as pd
    g = u["game"]
    rec_pbp = stats.get_record("playbyplayv2", {"GameID": g["game_id"]})
    rec_rot = stats.get_record("gamerotation", {"GameID": g["game_id"]})
    events = nba_events(rec_pbp["body"], g["home_team_id"])
    rotation = rotation_rows(rec_rot["body"])
    fields = {"season_type": u["fields"]["season_type"], "game_date": g["date"], "split": g["split"], "heldout": g["heldout"]}
    pbp = common.add_provenance(pd.DataFrame(events), SOURCE, "playbyplayv2", u["season"], rec_pbp["fetched_at"], game_id=g["game_id"], frame=-1, **fields)
    rot = common.add_provenance(pd.DataFrame(rotation), SOURCE, "gamerotation", u["season"], rec_rot["fetched_at"], **fields)
    st = common.add_provenance(pd.DataFrame(build_stints(g["game_id"], rotation, events)), SOURCE, "gamerotation+playbyplayv2", u["season"], rec_rot["fetched_at"], **fields)
    if len(pbp):
        fam.write_part("playbyplay", u["season"], g["game_id"], pbp)
    if len(rot):
        fam.write_part("rotation", u["season"], g["game_id"], rot)
    if len(st):
        fam.write_part("stints", u["season"], g["game_id"], st)
    bad = int((~st["lineup_ok"]).sum()) if len(st) else 0
    return len(st), {"events": len(events), "rotation_rows": len(rotation), "stints_not_5v5": bad, "fetched_at": rec_rot["fetched_at"]}


def game_units(stats: NBAStats, fam: common.Family, season: str, st: str) -> list:
    out = []
    for g in season_games(stats, fam, season, st):
        key, u = unit("stints", season, g["game_id"], {"GameID": g["game_id"]}, {"season_type": st})
        out.append((key, {**u, "game": g}))
    return out


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    stats = NBAStats(fam.raw, common.Throttle(a.rpm))
    units = [u for s in a.seasons for st in a.season_types for u in game_units(stats, fam, s, st)]
    return common.run_units(fam, units, lambda u: do_game(stats, fam, u), a, "stints: play-by-play + rotation per game")


if __name__ == "__main__":
    main()
