"""collect/nba_stats.py: a small stats.nba.com client with the parameter sets the endpoints require.

stats.nba.com answers 400 unless EVERY parameter an endpoint knows is present (blank allowed), so ENDPOINTS
carries the full set per endpoint (the lists nba_api sends). Requests go through collect.common.Throttle
and the raw cache: a cached response is never requested again, so a rerun of any collector is offline.
Blocked clients (cloud IPs, missing headers) get a hang or a 403: the client times out, backs off
(5, 10, 20, 40, 80 s) and gives up after five tries with a ClientError the collector records per unit.

  stats = NBAStats(cache, Throttle(rpm=30))
  payload = stats.get("playbyplayv2", {"GameID": "0021500381"})
  frames = result_sets(payload)             # {"PlayByPlay": DataFrame, ...}
"""
from __future__ import annotations
import json, time
from collect.common import BACKOFF_S, RawCache, Throttle, snake

BASE = "https://stats.nba.com/stats/"
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
SOURCE = "stats.nba.com"

_DASH_BLANKS = {"College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "", "Division": "",
                "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "", "Height": "", "Location": "",
                "Outcome": "", "PORound": "", "PlayerExperience": "", "PlayerPosition": "", "SeasonSegment": "",
                "StarterBench": "", "VsConference": "", "VsDivision": "", "Weight": ""}
_DASH_ZEROS = {"LastNGames": 0, "Month": 0, "OpponentTeamID": 0, "Period": 0, "TeamID": 0}
_BOX_V3 = {"StartPeriod": 0, "EndPeriod": 14, "StartRange": 0, "EndRange": 31800, "RangeType": 0}

ENDPOINTS: dict[str, dict] = {
    "playbyplayv2": {"GameID": "", "StartPeriod": 0, "EndPeriod": 14},
    "shotchartdetail": {"ContextMeasure": "FGA", "LastNGames": 0, "LeagueID": "00", "Month": 0, "OpponentTeamID": 0,
                        "Period": 0, "PlayerID": 0, "SeasonType": "Regular Season", "TeamID": 0, "AheadBehind": "",
                        "ClutchTime": "", "ContextFilter": "", "DateFrom": "", "DateTo": "", "EndPeriod": "", "EndRange": "",
                        "GameID": "", "GameSegment": "", "Location": "", "Outcome": "", "PlayerPosition": "", "PointDiff": "",
                        "Position": "", "RangeType": "", "RookieYear": "", "Season": "", "SeasonSegment": "", "StartPeriod": "",
                        "StartRange": "", "VsConference": "", "VsDivision": ""},
    "leaguedashplayerptshot": {"LeagueID": "00", "PerMode": "Totals", "Season": "", "SeasonType": "Regular Season",
                               "CloseDefDistRange": "", "TouchTimeRange": "", "DribbleRange": "", "ShotClockRange": "",
                               "GeneralRange": "", "ShotDistRange": "", **_DASH_ZEROS, **_DASH_BLANKS},
    "playerdashptshotdefend": {"PlayerID": 0, "TeamID": 0, "LeagueID": "00", "Season": "", "SeasonType": "Regular Season",
                               "PerMode": "Totals", "LastNGames": 0, "Month": 0, "OpponentTeamID": 0, "Period": 0,
                               "DateFrom": "", "DateTo": "", "GameSegment": "", "Location": "", "Outcome": "",
                               "SeasonSegment": "", "VsConference": "", "VsDivision": ""},
    "playerdashptreb": {"PlayerID": 0, "TeamID": 0, "LeagueID": "00", "Season": "", "SeasonType": "Regular Season",
                        "PerMode": "Totals", "LastNGames": 0, "Month": 0, "OpponentTeamID": 0, "Period": 0,
                        "DateFrom": "", "DateTo": "", "GameSegment": "", "Location": "", "Outcome": "",
                        "SeasonSegment": "", "VsConference": "", "VsDivision": ""},
    "leaguedashptstats": {"PtMeasureType": "SpeedDistance", "PlayerOrTeam": "Player", "LeagueID": "00", "Season": "",
                          "SeasonType": "Regular Season", "PerMode": "Totals", **_DASH_ZEROS, **_DASH_BLANKS},
    "leaguedashplayerstats": {"MeasureType": "Base", "PerMode": "Totals", "PlusMinus": "N", "PaceAdjust": "N", "Rank": "N",
                              "LeagueID": "00", "Season": "", "SeasonType": "Regular Season", "PORound": 0,
                              "ShotClockRange": "", "TwoWay": "", **_DASH_ZEROS, **{k: v for k, v in _DASH_BLANKS.items() if k != "PORound"}},
    "boxscoreplayertrackv3": {"GameID": "", **_BOX_V3},
    "boxscorematchupsv3": {"GameID": "", **_BOX_V3},
    "leagueseasonmatchups": {"LeagueID": "00", "Season": "", "SeasonType": "Regular Season", "PerMode": "Totals",
                             "DefPlayerID": "", "DefTeamID": "", "OffPlayerID": "", "OffTeamID": ""},
    "synergyplaytypes": {"LeagueID": "00", "PerMode": "Totals", "PlayType": "Isolation", "PlayerOrTeam": "P",
                         "SeasonType": "Regular Season", "SeasonYear": "", "TypeGrouping": "offensive"},
    "leaguehustlestatsplayer": {"LeagueID": "00", "PerMode": "Totals", "Season": "", "SeasonType": "Regular Season",
                                "Month": 0, "OpponentTeamID": 0, "TeamID": "", "PORound": "", "College": "", "Conference": "",
                                "Country": "", "DateFrom": "", "DateTo": "", "Division": "", "DraftPick": "", "DraftYear": "",
                                "Height": "", "Location": "", "Outcome": "", "PlayerExperience": "", "PlayerPosition": "",
                                "SeasonSegment": "", "VsConference": "", "VsDivision": "", "Weight": ""},
    "leaguedashlineups": {"GroupQuantity": 5, "LastNGames": 0, "LeagueID": "00", "MeasureType": "Base", "Month": 0,
                          "OpponentTeamID": 0, "PaceAdjust": "N", "PerMode": "Totals", "Period": 0, "PlusMinus": "N",
                          "Rank": "N", "Season": "", "SeasonType": "Regular Season", "Conference": "", "DateFrom": "",
                          "DateTo": "", "Division": "", "GameSegment": "", "Location": "", "Outcome": "", "PORound": "",
                          "SeasonSegment": "", "ShotClockRange": "", "TeamID": "", "VsConference": "", "VsDivision": ""},
    "gamerotation": {"GameID": "", "LeagueID": "00"},
    "leaguegamelog": {"Counter": 0, "Direction": "ASC", "LeagueID": "00", "PlayerOrTeam": "T", "Season": "",
                      "SeasonType": "Regular Season", "Sorter": "DATE", "DateFrom": "", "DateTo": ""},
    "winprobabilitypbp": {"GameID": "", "RunType": "each second"},
}
SYNERGY_PLAY_TYPES = ("Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup", "Spotup", "Handoff", "Cut",
                      "OffScreen", "OffRebound", "Misc")
PT_MEASURE_TYPES = ("SpeedDistance", "Drives", "Possessions", "Passing", "Rebounding", "Defense", "CatchShoot",
                    "PullUpShot", "ElbowTouch", "PostTouch", "PaintTouch", "Efficiency")
CLOSE_DEF_RANGES = ("0-2 Feet - Very Tight", "2-4 Feet - Tight", "4-6 Feet - Open", "6+ Feet - Wide Open")
TOUCH_TIME_RANGES = ("Touch < 2 Seconds", "Touch 2-6 Seconds", "Touch 6+ Seconds")


class ClientError(Exception):
    """A request that failed after every retry. stop_run=True when the collector should stop (jail risk)."""
    def __init__(self, msg: str, status: int | None = None, stop_run: bool = False):
        super().__init__(msg)
        self.status, self.stop_run = status, stop_run


def params_for(endpoint: str, overrides: dict) -> dict:
    if endpoint not in ENDPOINTS:
        raise KeyError("unknown endpoint %r (add it to collect.nba_stats.ENDPOINTS)" % endpoint)
    p = dict(ENDPOINTS[endpoint])
    bad = set(overrides) - set(p)
    if bad:
        raise KeyError("%s does not take %s" % (endpoint, sorted(bad)))
    p.update(overrides)
    return p


class NBAStats:
    def __init__(self, cache: RawCache, throttle: Throttle, timeout: float = 30.0, tries: int = 5, session=None):
        self.cache, self.throttle, self.timeout, self.tries = cache, throttle, timeout, tries
        self.session = session
        self.requests_made = 0

    def _session(self):
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers.update(HEADERS)
        return self.session

    def get(self, endpoint: str, overrides: dict) -> dict:
        """The response body (dict) for endpoint + overrides, from the raw cache or the network."""
        return self.get_record(endpoint, overrides)["body"]

    def get_record(self, endpoint: str, overrides: dict) -> dict:
        """The cache record {"body", "fetched_at", "params", "url", ...}, fetching when absent."""
        params = params_for(endpoint, overrides)
        rec = self.cache.get(endpoint, params)
        if rec is not None:
            return rec
        url = BASE + endpoint
        last = None
        for attempt in range(self.tries):
            self.throttle.wait(BACKOFF_S[attempt - 1] if attempt else 0.0)
            self.requests_made += 1
            try:
                r = self._session().get(url, params=params, timeout=self.timeout)
            except Exception as e:                              # noqa: BLE001 (timeouts, resets: retry)
                last = "request error: %r" % e
                continue
            if r.status_code == 200:
                try:
                    body = r.json()
                except ValueError:
                    last = "non-JSON body: %s" % r.text[:120]
                    continue
                return self.cache.put(endpoint, params, body, r.url, r.status_code)
            last = "HTTP %d: %s" % (r.status_code, r.text[:160].replace("\n", " "))
            if r.status_code in (400, 404):                     # a parameter problem: retrying cannot help
                raise ClientError("%s %s" % (endpoint, last), r.status_code)
            if r.status_code == 403:                            # blocked client: stop before every unit fails
                raise ClientError("%s %s (this client is blocked by stats.nba.com: run on the Mac or Colab)" % (endpoint, last), 403, stop_run=True)
        raise ClientError("%s failed after %d tries: %s" % (endpoint, self.tries, last))


# ── response shapes ────────────────────────────────────────────────────────────────────────────────
def result_sets(payload: dict) -> dict:
    """{name: DataFrame} for the classic 'resultSets' (list) or 'resultSet' (one) shape; columns snake_case."""
    import pandas as pd
    sets = payload.get("resultSets")
    if sets is None and "resultSet" in payload:
        sets = [payload["resultSet"]]
    out = {}
    for rs in sets or []:
        cols = [snake(h) for h in rs.get("headers", [])]
        out[rs.get("name", "result")] = pd.DataFrame(rs.get("rowSet", []), columns=cols)
    return out


def first_set(payload: dict, name: str | None = None):
    sets = result_sets(payload)
    if name and name in sets:
        return sets[name]
    if not sets:
        raise ValueError("response carries no result set")
    return next(iter(sets.values()))


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> one level, keys joined with '_' and snake_cased; lists are left to the caller."""
    out = {}
    for k, v in d.items():
        key = snake(prefix + "_" + k if prefix else k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif not isinstance(v, list):
            out[key] = v
    return out


def matchups_v3_rows(payload: dict) -> list[dict]:
    """boxscorematchupsv3 -> one row per (defender, offensive player) pair with the statistics flattened."""
    box = payload.get("boxScoreMatchups") or payload.get("boxscoreMatchups") or {}
    game_id = box.get("gameId")
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = box.get(side) or {}
        for p in team.get("players", []):
            base = {"game_id": game_id, "def_team_id": team.get("teamId"), "def_team": team.get("teamTricode"),
                    "def_player_id": p.get("personId"), "def_player": "%s %s" % (p.get("firstName", ""), p.get("familyName", "")),
                    "def_home": side == "homeTeam"}
            for m in p.get("matchups", []):
                row = dict(base)
                row.update({"off_player_id": m.get("personId"), "off_player": "%s %s" % (m.get("firstName", ""), m.get("familyName", ""))})
                row.update(flatten(m.get("statistics", {})))
                rows.append(row)
    return rows


def player_track_v3_rows(payload: dict) -> list[dict]:
    """boxscoreplayertrackv3 -> one row per player with the statistics flattened."""
    box = payload.get("boxScorePlayerTrack") or {}
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = box.get(side) or {}
        for p in team.get("players", []):
            row = {"game_id": box.get("gameId"), "team_id": team.get("teamId"), "team": team.get("teamTricode"),
                   "home": side == "homeTeam", "player_id": p.get("personId"),
                   "player": "%s %s" % (p.get("firstName", ""), p.get("familyName", "")), "position": p.get("position")}
            row.update(flatten(p.get("statistics", {})))
            rows.append(row)
    return rows


def game_id_season(game_id: str) -> str:
    """'0021500381' -> '2015-16' (digits 4-5 are the season's start year within the century)."""
    yy = int(game_id[3:5])
    year = 1900 + yy if yy >= 46 else 2000 + yy
    return "%d-%02d" % (year, (year + 1) % 100)


def dump_fixture(cache: RawCache, endpoint: str, params: dict, dest, max_rows: int = 12) -> dict:
    """A cached response trimmed to max_rows per result set, for tests/fixtures/collect/ (real shape, few rows)."""
    rec = cache.get(endpoint, params)
    if rec is None:
        raise FileNotFoundError("no cached response for %s %s" % (endpoint, params))
    body = json.loads(json.dumps(rec["body"]))
    for rs in body.get("resultSets", []) + ([body["resultSet"]] if "resultSet" in body else []):
        rs["rowSet"] = rs.get("rowSet", [])[:max_rows]
    out = {"endpoint": endpoint, "params": params, "fetched_at": rec["fetched_at"], "trimmed_to": max_rows, "body": body}
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    return out


# ── shared unit runner for table endpoints ─────────────────────────────────────────────────────────
def games_from_log(payload: dict) -> list[dict]:
    """leaguegamelog (PlayerOrTeam=T) -> one dict per game: game_id, date (YYYY-MM-DD), home, away, home_team_id,
    away_team_id, from the two team rows ('GSW vs. PHX' is the home row, 'PHX @ GSW' the away row)."""
    df = first_set(payload, "LeagueGameLog")
    games: dict[str, dict] = {}
    for r in df.itertuples(index=False):
        d = r._asdict()
        g = games.setdefault(str(d["game_id"]), {"game_id": str(d["game_id"]), "date": str(d["game_date"])[:10]})
        side = "home" if " vs. " in str(d["matchup"]) else "away"
        g[side] = d["team_abbreviation"]
        g[side + "_team_id"] = int(d["team_id"])
    return [g for _, g in sorted(games.items()) if "home" in g and "away" in g]


def unit(endpoint: str, season: str, unit_id, params: dict, fields: dict | None = None, set_name: str | None = None, rows_fn=None) -> tuple[str, dict]:
    """(manifest key, unit dict) for one request whose rows become one part."""
    from collect.common import Manifest
    return Manifest.key(endpoint, season, unit_id), {"endpoint": endpoint, "season": season, "id": unit_id, "params": params,
                                                     "fields": fields or {}, "set_name": set_name, "rows_fn": rows_fn}


def table_unit(stats: "NBAStats", fam, u: dict, convert=None, long_ids=None, long_keep=()) -> tuple[int, dict]:
    """Fetch one unit, add provenance, convert, write its part (and its long aggregates part when long_ids is given)."""
    import pandas as pd
    from collect.common import add_provenance, to_long
    rec = stats.get_record(u["endpoint"], u["params"])
    body = rec["body"]
    df = pd.DataFrame(u["rows_fn"](body)) if u.get("rows_fn") else first_set(body, u.get("set_name"))
    df = add_provenance(df, SOURCE, u["endpoint"], u["season"], rec["fetched_at"], **u.get("fields", {}))
    if convert is not None:
        df = convert(df, u)
    if len(df):
        fam.write_part(u["endpoint"], u["season"], u["id"], df)
        if long_ids is not None:
            fam.write_part("aggregates", u["season"], "%s_%s" % (u["endpoint"], u["id"]), to_long(df, long_ids, long_keep))
    return len(df), {"fetched_at": rec["fetched_at"], "endpoint": u["endpoint"]}


def season_games(stats: "NBAStats", fam, season: str, season_type: str) -> list[dict]:
    """The games of a season (leaguegamelog, cached), each tagged with its held-out split when it has one."""
    from collect.common import heldout_split
    payload = stats.get("leaguegamelog", {"Season": season, "SeasonType": season_type})
    games = games_from_log(payload)
    for g in games:
        g["split"] = heldout_split(g["date"], g["away"], g["home"])
        g["heldout"] = g["split"] is not None
    return games
