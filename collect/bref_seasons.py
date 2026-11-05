"""collect/bref_seasons.py: whole seasons of shot charts and play-by-play from basketball-reference.com.

collect/bref_shots.py collects the shot charts of the SportVU games (one listing of about 620 games); this driver
extends it to every game of a season, the games coming from the schedule collected by collect.bref_schedule
(games_for_season). Two endpoints, one request per game each:

  shotchart   boxscores/shot-chart/<code>.html   parsed by bref_shots.parse_shot_chart, rows by bref_shots.rows_for_game
              (docs/DATA_CONTRACT.md 10a: source=bref, court feet attacking the left basket, loc_x/loc_y as NBA.com)
  playbyplay  boxscores/pbp/<code>.html          parsed by fetch_pbp.fetch_game, rows in the contract section 13 columns
              (section 5 play-by-play columns with frame -1 and the NBA.com fields null, plus game_id, split, heldout,
              season_type, game_date)
game_id is the basketball-reference code (YYYYMMDD0HOME); game is the SportVU-style id MM.DD.YYYY.AWAY.at.HOME in
SportVU abbreviations, so rows join the Step 1 tables by name. 2015-16 games are tagged with their sportvu/splits.json
split (sportvu.colab_contract.split_of from date and teams) and the held-out rows are compacted under heldout/.

Defaults: shots for 2016-17 to 2020-21 (--shots-seasons), play-by-play for 2020-21 to 2024-25 (--pbp-seasons);
--seasons is accepted as a shorthand for both. The raw cache is shared with bref_shots (raw/bref_shotchart/,
raw/bref_pbp/, keyed by {"code"}) and collect.pbp, so a page any of them fetched is never requested again; data/pbp/
<code>.json files are reused too. At most 17 requests a minute; the first HTTP 429 or 403 stops the run; the
estimated request count is printed before the run.

  python -m collect.bref_schedule --seasons 2016-17             # first: the schedule
  python -m collect.bref_seasons --shots-seasons 2016-17 --pbp-seasons   # shots only
  python -m collect.bref_seasons --pbp-seasons 2023-24 --shots-seasons --limit 50 --dry-run
Outputs: --out/bref_seasons/shotchart/<season>.parquet and playbyplay/<season>.parquet (held-out rows under heldout/).
"""
from __future__ import annotations
import json
from pathlib import Path

import config
from collect import common
from collect import bref_shots as bs
from collect import bref_schedule as sched

FAMILY = "bref_seasons"
SHOTS = "shotchart"
PBP = "playbyplay"
RAW_SHOTS = bs.RAW_ENDPOINT                    # raw/bref_shotchart/<key of {"code"}>: the page, shared with bref_shots
RAW_PBP = bs.RAW_PBP_ENDPOINT                  # raw/bref_pbp/<key of {"code"}>: the parsed events, shared with bref_shots and collect.pbp
PBP_SOURCE = "basketball-reference.com"
LOCAL_PBP_DIR = Path(config.PROJECT_ROOT) / "data" / "pbp"
MAX_RPM = bs.MAX_RPM
DEFAULT_SHOTS_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]
DEFAULT_PBP_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
PBP_COLUMNS = ("game", "event", "period", "clock_s", "team", "desc", "points", "score", "kind", "frame", "frame_clock_diff_s",
               "source", "endpoint", "season", "fetched_at", "event_id", "event_type", "action_type",
               "player1_id", "player2_id", "player3_id", "team_id", "clock_str",
               "game_id", "split", "heldout", "season_type", "game_date")


def game_dict(g: dict) -> dict:
    """A schedule game -> the {game, date, away, home} dict bref_shots.rows_for_game wants (SportVU abbreviations)."""
    return {"game": g["game"], "date": g["date"], "away": sched.sportvu_abbr(g["away"]), "home": sched.sportvu_abbr(g["home"])}


# ── play-by-play ───────────────────────────────────────────────────────────────────────────────────
def pbp_events(fam: common.Family, code: str, throttle: common.Throttle) -> tuple[list[dict], str, str]:
    """(events, fetched_at, where from): the shared raw cache, data/pbp/<code>.json, or the page through
    fetch_pbp.fetch_game under the throttle (cached afterwards). An HTTP 429 or 403 raises PageError(stop_run=True)."""
    params = {"code": code}
    rec = fam.raw.get(RAW_PBP, params)
    if rec is not None:
        return rec["body"], rec["fetched_at"], "cache"
    local = LOCAL_PBP_DIR / ("%s.json" % code)
    if local.exists():
        events = json.loads(local.read_text())["events"]
        rec = fam.raw.put(RAW_PBP, params, events, "data/pbp/%s.json" % code, 200)
        return events, rec["fetched_at"], "data/pbp/%s.json" % code
    import fetch_pbp
    throttle.wait()
    try:
        events = fetch_pbp.fetch_game(code)
    except Exception as e:                                      # noqa: BLE001 (classify the jail, else record)
        msg = str(e)
        if "429" in msg:
            raise bs.PageError("pbp page %s HTTP 429: basketball-reference rate limit, no more requests this run (jail lasts up to a day)" % code, 429, stop_run=True)
        if "403" in msg:
            raise bs.PageError("pbp page %s HTTP 403: this client is blocked by basketball-reference" % code, 403, stop_run=True)
        raise bs.PageError("pbp page %s: %r" % (code, e))
    rec = fam.raw.put(RAW_PBP, params, events, "%s/boxscores/pbp/%s.html" % (bs.HOST, code), 200)
    return events, rec["fetched_at"], "page"


def pbp_rows(g: dict, events: list[dict], fetched_at: str):
    """fetch_pbp events of one game -> the contract section 13 table (frame -1, NBA.com fields null)."""
    import pandas as pd
    code = g["game_id"]
    recs = []
    for i, e in enumerate(events):
        recs.append({"game": g["game"], "event": i, "period": e["period"], "clock_s": e["clock_s"], "team": e.get("team"),
                     "desc": e.get("desc") or "", "points": e.get("points") or 0, "score": e.get("score"), "kind": e.get("kind") or "other",
                     "frame": -1, "frame_clock_diff_s": None, "source": PBP_SOURCE, "endpoint": "boxscores/pbp/%s.html" % code,
                     "season": g["season"], "fetched_at": fetched_at, "event_id": None, "event_type": None, "action_type": None,
                     "player1_id": None, "player2_id": None, "player3_id": None, "team_id": None, "clock_str": None,
                     "game_id": code, "split": g.get("split"), "heldout": bool(g.get("heldout")), "season_type": g["season_type"], "game_date": g["date"]})
    df = pd.DataFrame(recs, columns=list(PBP_COLUMNS))
    types = {"event": "int32", "period": "int8", "clock_s": "float32", "points": "int8", "frame": "int32", "frame_clock_diff_s": "float32",
             "event_id": "Int32", "event_type": "Int16", "action_type": "Int16", "player1_id": "Int32", "player2_id": "Int32",
             "player3_id": "Int32", "team_id": "Int32", "heldout": "bool"}
    for c, t in types.items():
        df[c] = df[c].astype(t)
    for c in ("game", "team", "desc", "score", "kind", "source", "endpoint", "season", "fetched_at", "clock_str", "game_id", "split", "season_type", "game_date"):
        df[c] = df[c].astype("string")
    return df


# ── units ──────────────────────────────────────────────────────────────────────────────────────────
def do_unit(unit: dict, fam: common.Family, throttle: common.Throttle) -> tuple[int, dict]:
    g, code = unit["game"], unit["game"]["game_id"]
    if unit["endpoint"] == SHOTS:
        rec = sched.page_record(fam, RAW_SHOTS, {"code": code}, bs.shot_chart_url(code), throttle)
        shots = bs.parse_shot_chart(rec["body"])
        cal = bs.calibrate(shots)
        if cal["ok"] is False:
            raise ValueError("%s: the hoop fitted from the tips sits %.1f px from HOOP_PX %s: the page's pixel frame is not the assumed one"
                             % (code, cal["hoop_shift_px"], cal["hoop_px"]))
        df = bs.rows_for_game(game_dict(g), shots, rec["fetched_at"], g.get("split"), g["season_type"])
        df["game_id"] = df["game_id"].astype("string")
        df.loc[:, "game_id"] = code
        if len(df):
            fam.write_part(SHOTS, g["season"], code, df)
        return len(df), {"fetched_at": rec["fetched_at"], "endpoint": "boxscores/shot-chart/%s.html" % code, "markers": len(shots),
                         "made": int(df["made"].sum()) if len(df) else 0, "calibration": cal, "split": g.get("split"), "heldout": bool(g.get("heldout"))}
    events, fetched_at, where = pbp_events(fam, code, throttle)
    df = pbp_rows(g, events, fetched_at)
    if len(df):
        fam.write_part(PBP, g["season"], code, df)
    kinds = {k: int(n) for k, n in df.groupby("kind").size().items()} if len(df) else {}
    return len(df), {"fetched_at": fetched_at, "endpoint": "boxscores/pbp/%s.html" % code, "from": where, "kinds": kinds,
                     "periods": int(df["period"].max()) if len(df) else 0, "split": g.get("split"), "heldout": bool(g.get("heldout"))}


def units_for(fam: common.Family, endpoint: str, seasons: list[str], season_types) -> list[tuple[str, dict]]:
    out = []
    for s in seasons:
        games = sched.games_for_season(fam, s, season_types)
        if not games:
            print("!! no schedule for %s under %s: run python -m collect.bref_schedule --seasons %s first" % (s, fam.dir.parent, s), flush=True)
        for g in games:
            out.append((common.Manifest.key(endpoint, s, g["game_id"]), {"endpoint": endpoint, "season": s, "game": g, "params": {"code": g["game_id"]}}))
    return out


def raw_of(unit: dict) -> tuple[str, dict]:
    return (RAW_SHOTS if unit["endpoint"] == SHOTS else RAW_PBP), unit["params"]


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], [])
    ap.set_defaults(rpm=MAX_RPM)
    ap.add_argument("--shots-seasons", nargs="*", default=None, help="seasons of shot charts (default %s; give it empty for none)" % " ".join(DEFAULT_SHOTS_SEASONS))
    ap.add_argument("--pbp-seasons", nargs="*", default=None, help="seasons of play-by-play (default %s; give it empty for none)" % " ".join(DEFAULT_PBP_SEASONS))
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    shots_seasons = a.seasons if a.seasons else (DEFAULT_SHOTS_SEASONS if a.shots_seasons is None else a.shots_seasons)
    pbp_seasons = a.seasons if a.seasons else (DEFAULT_PBP_SEASONS if a.pbp_seasons is None else a.pbp_seasons)
    units = units_for(fam, SHOTS, shots_seasons, a.season_types) + units_for(fam, PBP, pbp_seasons, a.season_types)
    throttle = sched.capped_throttle(a.rpm)
    sched.estimate_requests(fam, units, a, raw_of)
    return common.run_units(fam, units, lambda u: do_unit(u, fam, throttle), a,
                            "basketball-reference seasons: shots %s, play-by-play %s" % (" ".join(shots_seasons) or "-", " ".join(pbp_seasons) or "-"))


if __name__ == "__main__":
    main()
