"""collect/bref_seasons.py: whole seasons of shot charts and play-by-play from basketball-reference.com.

collect/bref_shots.py collects the shot charts of the SportVU games (one listing of about 620 games); this driver
extends it to every game of a season, the games coming from the schedule collected by collect.bref_schedule
(games_for_season). Two endpoints, one request per game each:

  shotchart   boxscores/shot-chart/<code>.html   parsed by bref_shots.parse_shot_chart, rows by bref_shots.rows_for_game
              (docs/DATA_CONTRACT.md 10a: source=bref, court feet attacking the left basket, loc_x/loc_y as NBA.com)
  playbyplay  boxscores/pbp/<code>.html          parsed by fetch_pbp.parse_events, rows in the contract section 13 columns
              (section 5 play-by-play columns with frame -1 and the NBA.com fields null, plus game_id, split, heldout,
              season_type, game_date)
game_id is the basketball-reference code (YYYYMMDD0HOME); game is the SportVU-style id MM.DD.YYYY.AWAY.at.HOME in
SportVU abbreviations, so rows join the Step 1 tables by name. 2015-16 games are tagged with their sportvu/splits.json
split (sportvu.colab_contract.split_of from date and teams) and the held-out rows are compacted under heldout/.

Defaults: shots for 2016-17 to 2020-21 (--shots-seasons), play-by-play for 2020-21 to 2024-25 (--pbp-seasons);
--seasons is accepted as a shorthand for both. Every page goes through bref_shots.fetch_html under the shared
throttle. The raw cache is shared with bref_shots and collect.pbp: shot chart pages under raw/bref_shotchart/,
play-by-play pages under raw/bref_pbp_page/ and their parsed events under raw/bref_pbp/ (the record the other two
write and read), all keyed by {"code"}, so a page any of them fetched is never requested again; data/pbp/<code>.json
files are reused too. At most 17 requests a minute; the first HTTP 429 or 403 stops the run; the estimated
request count is printed before the run. After compaction the play-by-play tables are copied to
--out/stints/playbyplay/<season>.parquet (held-out rows to --out/stints/heldout/playbyplay/), which the lineups
lane reads.

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
RAW_PBP_PAGE = "bref_pbp_page"                 # raw/bref_pbp_page/<key of {"code"}>: the page as received
STINTS_FAMILY = "stints"
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
def pbp_url(code: str) -> str:
    return "%s/boxscores/pbp/%s.html" % (bs.HOST, code)


def pbp_events(fam: common.Family, code: str, throttle: common.Throttle) -> tuple[list[dict], str, str]:
    """(events, fetched_at, where from), in order: the cached page (raw/bref_pbp_page/), the cached events another
    collector parsed (raw/bref_pbp/), data/pbp/<code>.json, or the page through bref_shots.fetch_html under the
    throttle (429 and 403 stop the run), parsed by fetch_pbp.parse_events; the page and its events are then cached."""
    import fetch_pbp
    params = {"code": code}
    rec = fam.raw.get(RAW_PBP_PAGE, params)
    if rec is not None:
        return fetch_pbp.parse_events(rec["body"]), rec["fetched_at"], "cache"
    rec = fam.raw.get(RAW_PBP, params)
    if rec is not None:
        return rec["body"], rec["fetched_at"], "cache"
    local = LOCAL_PBP_DIR / ("%s.json" % code)
    if local.exists():
        events = json.loads(local.read_text())["events"]
        rec = fam.raw.put(RAW_PBP, params, events, "data/pbp/%s.json" % code, 200)
        return events, rec["fetched_at"], "data/pbp/%s.json" % code
    rec = sched.page_record(fam, RAW_PBP_PAGE, params, pbp_url(code), throttle)
    events = fetch_pbp.parse_events(rec["body"])
    fam.raw.put(RAW_PBP, params, events, pbp_url(code), 200)
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
    import pandas as pd
    g, code = unit["game"], unit["game"]["game_id"]
    if unit["endpoint"] == SHOTS:
        rec = sched.page_record(fam, RAW_SHOTS, {"code": code}, bs.shot_chart_url(code), throttle)
        shots = bs.parse_shot_chart(rec["body"])
        if not shots:
            raise ValueError("%s: no shot markers on the page" % code)
        cal = bs.calibrate(shots)
        if cal["ok"] is False:
            raise ValueError("%s: the hoop fitted from the tips sits %.1f px from HOOP_PX %s: the page's pixel frame is not the assumed one"
                             % (code, cal["hoop_shift_px"], cal["hoop_px"]))
        df = bs.rows_for_game(game_dict(g), shots, rec["fetched_at"], g.get("split"), g["season_type"])
        df["game_id"] = pd.Series([code] * len(df), dtype="string")
        if len(df):
            fam.write_part(SHOTS, g["season"], code, df)
        return len(df), {"fetched_at": rec["fetched_at"], "endpoint": "boxscores/shot-chart/%s.html" % code, "markers": len(shots),
                         "made": int(df["made"].sum()) if len(df) else 0, "calibration": cal, "split": g.get("split"), "heldout": bool(g.get("heldout"))}
    events, fetched_at, where = pbp_events(fam, code, throttle)
    if not events:
        raise ValueError("%s: no play-by-play events on the page" % code)
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


def raw_of(unit: dict):
    if unit["endpoint"] == SHOTS:
        return RAW_SHOTS, unit["params"]
    return [(RAW_PBP_PAGE, unit["params"]), (RAW_PBP, unit["params"])]


def copy_pbp_to_stints(fam: common.Family) -> list[Path]:
    """--out/bref_seasons/playbyplay/<season>.parquet -> --out/stints/playbyplay/<season>.parquet (and heldout/playbyplay/),
    the table the lineups lane reads. Returns the paths written."""
    import shutil
    out = []
    for sub in ("", "heldout"):
        src_dir = fam.dir / sub / PBP if sub else fam.dir / PBP
        if not src_dir.exists():
            continue
        for p in sorted(src_dir.glob("*.parquet")):
            dst = (fam.dir.parent / STINTS_FAMILY / sub / PBP / p.name) if sub else (fam.dir.parent / STINTS_FAMILY / PBP / p.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(p, tmp)
            tmp.replace(dst)
            out.append(dst)
    return out


def main(argv=None) -> str:
    ap = sched.bref_parser(__doc__.split("\n\n")[0], [])
    ap.add_argument("--shots-seasons", nargs="*", default=None, help="seasons of shot charts (default %s; give it empty for none)" % " ".join(DEFAULT_SHOTS_SEASONS))
    ap.add_argument("--pbp-seasons", nargs="*", default=None, help="seasons of play-by-play (default %s; give it empty for none)" % " ".join(DEFAULT_PBP_SEASONS))
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); print("stints copies:", [str(p) for p in copy_pbp_to_stints(fam)]); return "compact only"
    shots_seasons = a.seasons if a.seasons else (DEFAULT_SHOTS_SEASONS if a.shots_seasons is None else a.shots_seasons)
    pbp_seasons = a.seasons if a.seasons else (DEFAULT_PBP_SEASONS if a.pbp_seasons is None else a.pbp_seasons)
    units = units_for(fam, SHOTS, shots_seasons, a.season_types) + units_for(fam, PBP, pbp_seasons, a.season_types)
    throttle = sched.capped_throttle(a.rpm)
    sched.estimate_requests(fam, units, a, raw_of)
    final = common.run_units(fam, units, lambda u: do_unit(u, fam, throttle), a,
                             "basketball-reference seasons: shots %s, play-by-play %s" % (" ".join(shots_seasons) or "-", " ".join(pbp_seasons) or "-"))
    if not a.dry_run:
        print("stints copies:", [str(p) for p in copy_pbp_to_stints(fam)], flush=True)
    return final


if __name__ == "__main__":
    main()
