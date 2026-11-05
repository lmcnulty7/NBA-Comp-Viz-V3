"""collect/bref_shots.py: shot charts of the SportVU games from basketball-reference.com (the stats.nba.com fallback).

stats.nba.com hangs for non-browser clients (Mac and cloud alike, Nov 2026), so ShotChartDetail (collect/shots.py)
cannot be collected. basketball-reference publishes one shot chart page per game,
boxscores/shot-chart/<code>.html, with every field-goal attempt as a positioned marker:

  <div class="shot-area" id="shots-GSW">
    <div style="top:52px;left:252px;" tip="1st Qtr, 11:40.0 remaining<br>Stephen Curry made 2-pointer from 2 ft<br>GSW now leads 2-0"
         class="tooltip make p-currystep01 q-1">...</div>

The marker gives (x, y) in page pixels, make or miss, the player (slug and name), the period and the clock; the
tip also gives the shooter's distance, which is used as a check of the pixel mapping (calibrate). The chart is a
half court 500 px wide (50 ft, 10 px per ft) with the baseline at the top and the basket centre at HOOP_IMG_PX
(250, 52.5) on the image. A marker's top and left place its top-left corner, not the glyph's centre: on the saved
pages (tests/fixtures/bref) the basket fitted from the stated distances sits at (239.5, 35.5) in marker
coordinates, so the glyph centre is MARKER_OFFSET_PX (10.5, 17) past the anchor and HOOP_PX is the anchor the
basket corresponds to. Every mapping below takes the anchor as the page gives it.

Court frame (docs/DATA_CONTRACT.md, section 1 and court/court33.py): x along the length 0..94 ft, y across 0..50 ft,
rims at (5.25, 25) and (88.75, 25). Every shot is written attacking the LEFT basket: court_x = 5.25 + ft from the
baseline, court_y = 25 + ft across (page left -> small y). The side a team really attacked in a period is not on
the page; mirror_to_right(x, y) turns a row around for the other basket once the SportVU frames say so (the join
step). loc_x, loc_y follow the NBA.com ShotChartDetail convention (tenths of a foot from the basket, y toward half
court) so the rows union with collect/shots; the sign of loc_x against NBA.com is unverified (no overlapping game
collected from both sources yet).

Run pattern (collect/common.py): one unit per game, the page cached under --out/raw/bref_shotchart/ and never
re-requested, parts under --out/bref_shots/parts/shotchart/<season>/<game>.parquet, compacted per season with the
held-out games of sportvu/splits.json under heldout/, manifest per unit (status, rows, bref_shots, pbp_shots,
calibration). At most 17 requests a minute (basketball-reference's limit; --rpm is capped); the first HTTP 429
stops the run (the jail lasts up to a day). Rows carry source=bref.

  python -m collect.bref_shots --games 12.16.2015.PHX.at.GSW 11.14.2015.BKN.at.GSW
  python -m collect.bref_shots --contract-dir ~/.../nba_harvest/sportvu_contract         # every Step 1 game
  python -m collect.bref_shots --from-github                                             # every archive in the source listing
  python -m collect.bref_shots --games ... --fetch-pbp     # also fetch the pbp page when no local play-by-play exists
  python -m collect.bref_shots --games ... --save-pages tests/fixtures/bref            # keep the pages for tests
  python -m collect.bref_shots --report                    # shots recovered per game vs the play-by-play shot count
The play-by-play shot count (shot_made + shot_missed) comes from, in order: <game>_pbp.parquet under --contract-dir,
data/pbp/<code>.json (fetch_pbp.py), or the basketball-reference pbp page when --fetch-pbp is given.
"""
from __future__ import annotations
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

import config
from collect import common
from court.court33 import COURT_VERTICES_33, COURT_LENGTH_FT, COURT_WIDTH_FT
from sportvu import colab_contract as cc
from sportvu.manifest import parse_name

FAMILY = "bref_shots"
ENDPOINT = "shotchart"                        # the parts/ and compacted directory name
RAW_ENDPOINT = "bref_shotchart"               # raw/<RAW_ENDPOINT>/<key>.json.gz holds the page as received
RAW_PBP_ENDPOINT = "bref_pbp"
SOURCE = "bref"
HOST = "https://www.basketball-reference.com"
MAX_RPM = 17.0
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.9"}

# ── the chart's pixel frame ────────────────────────────────────────────────────────────────────────
CHART_PX = (500, 472)                         # width, height of the court image
PX_PER_FT = 10.0
HOOP_IMG_PX = (250.0, 52.5)                   # the basket centre on the image: 25 ft across, 5.25 ft from the baseline
MARKER_OFFSET_PX = (10.5, 17.0)               # glyph centre minus marker anchor (top-left), fitted on the saved pages
HOOP_PX = (HOOP_IMG_PX[0] - MARKER_OFFSET_PX[0], HOOP_IMG_PX[1] - MARKER_OFFSET_PX[1])   # (239.5, 35.5): the basket in marker coordinates
RIM_LEFT = tuple(float(v) for v in COURT_VERTICES_33[6])      # (5.25, 25.0)
RIM_RIGHT = tuple(float(v) for v in COURT_VERTICES_33[26])    # (88.75, 25.0)
CALIB_MIN_DIST_FT = 3.0                       # tips round to whole feet: near-rim shots say little about the hoop pixel
CALIB_TOL_PX = 10.0                           # the fitted hoop may differ from HOOP_PX by this much before a unit fails
BREF_TO_SPORTVU = {v: k for k, v in cc.BR_ABBR.items()}

TIP_RE = re.compile(r"^(?P<player>.+?)\s+(?P<made>made|missed)\s+(?P<pts>\d)-pointer(?:\s+from\s+(?P<dist>\d+)\s*ft)?", re.I)
PERIOD_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+(Qtr|quarter|OT|overtime)", re.I)
CLOCK_RE = re.compile(r"(\d+):(\d+(?:\.\d+)?)\s+remaining", re.I)
PX_RE = {k: re.compile(k + r"\s*:\s*(-?\d+(?:\.\d+)?)\s*px") for k in ("top", "left")}

COLUMNS = ("game", "br_code", "game_id", "event_id", "player_id", "player", "player_slug", "team_id", "team", "team_br",
           "team_side", "period", "clock_s", "clock_str", "made", "shot_type", "shot_distance_ft", "px_left", "px_top",
           "loc_x", "loc_y", "court_x", "court_y", "basket", "dist_resid_ft", "desc", "game_date", "home_team", "away_team",
           "sportvu_game", "split", "heldout", "season_type", "release_frame", "closest_defender_id",
           "closest_defender_dist_ft", "defender_closing_speed_ftps", "source", "endpoint", "season", "fetched_at")


class PageError(Exception):
    """A page that could not be fetched. stop_run=True for a 429 or 403: every further request would fail too."""
    def __init__(self, msg: str, status: int | None = None, stop_run: bool = False):
        super().__init__(msg)
        self.status, self.stop_run = status, stop_run


# ── games ──────────────────────────────────────────────────────────────────────────────────────────
def game_from_id(sportvu_id: str) -> dict:
    """'12.16.2015.PHX.at.GSW' -> {game, date, away, home}."""
    g = parse_name(sportvu_id + ".7z")
    if not g:
        raise ValueError("not a SportVU game id: %r (want MM.DD.YYYY.AWAY.at.HOME)" % sportvu_id)
    return g


def season_of(date: str) -> str:
    y, m = int(date[:4]), int(date[5:7])
    return common.season_str(y if m >= 7 else y - 1)


def shot_chart_url(code: str) -> str:
    return "%s/boxscores/shot-chart/%s.html" % (HOST, code)


# ── fetching (the only network code here) ──────────────────────────────────────────────────────────
def fetch_html(url: str, throttle: common.Throttle, timeout: float = 30.0, tries: int = 3) -> str:
    """One page under the throttle. 429 and 403 raise PageError(stop_run=True); 404 raises PageError; connection
    errors and 5xx are retried with the shared backoff."""
    import requests
    last = None
    for attempt in range(tries):
        throttle.wait(common.BACKOFF_S[attempt - 1] if attempt else 0.0)
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
        except Exception as e:                                  # noqa: BLE001 (timeouts, resets: retry)
            last = "request error: %r" % e
            continue
        if r.status_code == 200:
            return r.text
        last = "HTTP %d" % r.status_code
        if r.status_code == 429:
            raise PageError("%s HTTP 429: basketball-reference rate limit, no more requests this run (jail lasts up to a day)" % url, 429, stop_run=True)
        if r.status_code == 403:
            raise PageError("%s HTTP 403: this client is blocked by basketball-reference" % url, 403, stop_run=True)
        if r.status_code == 404:
            raise PageError("%s HTTP 404: no such page" % url, 404)
    raise PageError("%s failed after %d tries: %s" % (url, tries, last))


def page_record(fam: common.Family, code: str, throttle: common.Throttle) -> dict:
    """The raw-cache record of a shot chart page ({"body": html, "fetched_at", ...}), fetched when absent."""
    params = {"code": code}
    rec = fam.raw.get(RAW_ENDPOINT, params)
    if rec is None:
        url = shot_chart_url(code)
        html = fetch_html(url, throttle)
        rec = fam.raw.put(RAW_ENDPOINT, params, html, url, 200)
    return rec


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────
def _px(style: str, key: str) -> float | None:
    m = PX_RE[key].search(style or "")
    return float(m.group(1)) if m else None


def _period(classes: list[str], head: str) -> int | None:
    for c in classes:
        if re.match(r"^q-\d+$", c):
            return int(c[2:])
    m = PERIOD_RE.search(head or "")
    if m:
        n, kind = int(m.group(1)), m.group(2).upper()
        return n if kind in ("QTR", "QUARTER") else 4 + n
    return None


def _shot_areas(soup):
    """The shot-area divs, live or (as basketball-reference does with many tables) inside HTML comments."""
    from bs4 import BeautifulSoup, Comment
    areas = soup.find_all("div", class_="shot-area")
    if areas:
        return areas
    out = []
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "shot-area" in c:
            out.extend(BeautifulSoup(c, "html.parser").find_all("div", class_="shot-area"))
    return out


def parse_shot_chart(html: str) -> list[dict]:
    """Every shot marker on a shot chart page -> dict(team_br, player_slug, player, period, clock_s, clock_str, made,
    shot_type, shot_distance_ft, px_left, px_top, desc, idx). idx is the page order within the team's area."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    shots = []
    for area in _shot_areas(soup):
        team_br = (area.get("id") or "").split("-", 1)[-1].upper() or None
        for i, d in enumerate(area.find_all("div", class_="tooltip")):
            classes = d.get("class") or []
            tip = d.get("tip") or ""
            parts = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"<br\s*/?>", tip)]
            head, body = (parts + ["", ""])[:2]
            m = TIP_RE.match(body)
            if not m and len(parts) > 2:                        # a leading line was added: look for the shot line
                for p in parts[1:]:
                    m = TIP_RE.match(p)
                    if m:
                        body = p
                        break
            cm = CLOCK_RE.search(head)
            clock_s = int(cm.group(1)) * 60 + float(cm.group(2)) if cm else None
            made = "make" in classes or ("miss" not in classes and bool(m and m.group("made").lower() == "made"))
            slug = next((c[2:] for c in classes if c.startswith("p-")), None)
            shots.append({"team_br": team_br, "player_slug": slug,
                          "player": m.group("player") if m else None,
                          "period": _period(classes, head),
                          "clock_s": clock_s, "clock_str": "%s:%s" % (cm.group(1), cm.group(2)) if cm else None,
                          "made": bool(made), "shot_type": ("%sPT" % m.group("pts")) if m else None,
                          "shot_distance_ft": float(m.group("dist")) if m and m.group("dist") is not None else None,
                          "px_left": _px(d.get("style"), "left"), "px_top": _px(d.get("style"), "top"),
                          "desc": " | ".join(p for p in parts if p), "idx": i})
    return shots


# ── the court frame ────────────────────────────────────────────────────────────────────────────────
def to_court(px_left: float, px_top: float, hoop_px=HOOP_PX) -> tuple[float, float]:
    """Marker anchor pixels -> contract court feet, attacking the LEFT basket (rim at RIM_LEFT): x along the length from
    the attacked baseline, y across (page left -> small y)."""
    x = RIM_LEFT[0] + (px_top - hoop_px[1]) / PX_PER_FT
    y = RIM_LEFT[1] + (px_left - hoop_px[0]) / PX_PER_FT
    return round(x, 2), round(y, 2)


def mirror_to_right(x: float, y: float) -> tuple[float, float]:
    """The same shot attacking the RIGHT basket (a half turn of the court about its centre)."""
    return round(COURT_LENGTH_FT - x, 2), round(COURT_WIDTH_FT - y, 2)


def nba_loc(px_left: float, px_top: float, hoop_px=HOOP_PX) -> tuple[float, float]:
    """NBA.com ShotChartDetail LOC_X, LOC_Y: tenths of a foot from the basket, x across (page right positive, sign
    against NBA.com unverified), y toward half court."""
    return round((px_left - hoop_px[0]) * 10.0 / PX_PER_FT, 1), round((px_top - hoop_px[1]) * 10.0 / PX_PER_FT, 1)


def pixel_distance_ft(px_left: float, px_top: float, hoop_px=HOOP_PX) -> float:
    return float(np.hypot(px_left - hoop_px[0], px_top - hoop_px[1]) / PX_PER_FT)


def calibrate(shots: list[dict], search_px: float = 30.0, step_px: float = 0.5) -> dict:
    """Fit the hoop pixel from the distances the tips state (scale fixed at PX_PER_FT): a grid search around HOOP_PX
    for the least mean absolute residual. Returns {"n", "hoop_px", "hoop_shift_px", "resid_default_ft",
    "resid_fit_ft", "ok"}; ok is False when the fitted hoop is farther than CALIB_TOL_PX from HOOP_PX, which means
    the page's pixel frame is not the one this module assumes."""
    pts = np.array([(s["px_left"], s["px_top"], s["shot_distance_ft"]) for s in shots
                    if s["px_left"] is not None and s["px_top"] is not None and s["shot_distance_ft"] is not None
                    and s["shot_distance_ft"] >= CALIB_MIN_DIST_FT], float)
    if len(pts) < 5:
        return {"n": int(len(pts)), "hoop_px": None, "hoop_shift_px": None, "resid_default_ft": None, "resid_fit_ft": None, "ok": None}
    dist_default = np.hypot(pts[:, 0] - HOOP_PX[0], pts[:, 1] - HOOP_PX[1]) / PX_PER_FT
    resid_default = float(np.mean(np.abs(dist_default - pts[:, 2])))
    offs = np.arange(-search_px, search_px + step_px / 2, step_px)
    gx, gy = np.meshgrid(HOOP_PX[0] + offs, HOOP_PX[1] + offs, indexing="ij")
    d = np.hypot(pts[None, None, :, 0] - gx[..., None], pts[None, None, :, 1] - gy[..., None]) / PX_PER_FT
    resid = np.mean(np.abs(d - pts[None, None, :, 2]), axis=-1)
    i, j = np.unravel_index(int(np.argmin(resid)), resid.shape)
    hoop = (float(gx[i, j]), float(gy[i, j]))
    shift = float(np.hypot(hoop[0] - HOOP_PX[0], hoop[1] - HOOP_PX[1]))
    return {"n": int(len(pts)), "hoop_px": [round(hoop[0], 1), round(hoop[1], 1)], "hoop_shift_px": round(shift, 1),
            "resid_default_ft": round(resid_default, 3), "resid_fit_ft": round(float(resid[i, j]), 3), "ok": bool(shift <= CALIB_TOL_PX)}


# ── rows ───────────────────────────────────────────────────────────────────────────────────────────
def rows_for_game(game: dict, shots: list[dict], fetched_at: str, split: str | None, season_type: str = "Regular Season"):
    """Parsed markers of one game -> the bref shots table (docs/DATA_CONTRACT.md, section 10a)."""
    import pandas as pd
    code = cc.br_code(game)
    home_br, away_br = cc.BR_ABBR.get(game["home"], game["home"]), cc.BR_ABBR.get(game["away"], game["away"])
    recs = []
    for s in shots:
        if s["px_left"] is None or s["px_top"] is None or s["period"] is None or s["clock_s"] is None:
            continue
        team_br = s["team_br"]
        side = "home" if team_br == home_br else "away" if team_br == away_br else None
        cx, cy = to_court(s["px_left"], s["px_top"])
        lx, ly = nba_loc(s["px_left"], s["px_top"])
        resid = (pixel_distance_ft(s["px_left"], s["px_top"]) - s["shot_distance_ft"]) if s["shot_distance_ft"] is not None else None
        recs.append({"game": game["game"], "br_code": code, "game_id": None, "event_id": None, "player_id": None,
                     "player": s["player"], "player_slug": s["player_slug"], "team_id": None,
                     "team": BREF_TO_SPORTVU.get(team_br, team_br), "team_br": team_br, "team_side": side,
                     "period": s["period"], "clock_s": s["clock_s"], "clock_str": s["clock_str"], "made": s["made"],
                     "shot_type": s["shot_type"], "shot_distance_ft": s["shot_distance_ft"],
                     "px_left": s["px_left"], "px_top": s["px_top"], "loc_x": lx, "loc_y": ly,
                     "court_x": cx, "court_y": cy, "basket": "left",
                     "dist_resid_ft": round(resid, 2) if resid is not None else None, "desc": s["desc"],
                     "game_date": game["date"], "home_team": game["home"], "away_team": game["away"], "sportvu_game": game["game"],
                     "split": split, "heldout": cc.is_heldout(split), "season_type": season_type,
                     "release_frame": None, "closest_defender_id": None, "closest_defender_dist_ft": None, "defender_closing_speed_ftps": None,
                     "source": SOURCE, "endpoint": "boxscores/shot-chart/%s.html" % code, "season": season_of(game["date"]), "fetched_at": fetched_at})
    df = pd.DataFrame(recs, columns=list(COLUMNS))
    types = {"game_id": "string", "event_id": "Int32", "player_id": "Int32", "team_id": "Int32", "period": "int8", "clock_s": "float32",
             "made": "bool", "shot_distance_ft": "float32", "px_left": "float32", "px_top": "float32", "loc_x": "float32", "loc_y": "float32",
             "court_x": "float32", "court_y": "float32", "dist_resid_ft": "float32", "heldout": "bool", "release_frame": "Int32",
             "closest_defender_id": "Int32", "closest_defender_dist_ft": "float32", "defender_closing_speed_ftps": "float32"}
    for c, t in types.items():
        df[c] = df[c].astype(t)
    for c in ("player", "player_slug", "team", "team_br", "team_side", "clock_str", "shot_type", "split", "desc"):
        df[c] = df[c].astype("string")
    return df.sort_values(["period", "clock_s"], ascending=[True, False], kind="stable").reset_index(drop=True)


# ── the play-by-play shot count to compare with ───────────────────────────────────────────────────
def pbp_shot_count(game: dict, code: str, contract_dir: Path | None, fam: common.Family, throttle: common.Throttle,
                   fetch_pbp_page: bool, sp: dict) -> tuple[int | None, str | None]:
    """(field-goal attempts in the play-by-play, where it came from): the Step 1 <game>_pbp.parquet, data/pbp/<code>.json,
    or the basketball-reference pbp page (cached under raw/bref_pbp/) when fetch_pbp_page is set; (None, None) otherwise."""
    if contract_dir is not None:
        _, pbp_path, _ = cc.game_paths(game, contract_dir, sp)
        if pbp_path.exists():
            import pyarrow.parquet as pq
            kinds = pq.read_table(pbp_path, columns=["kind"]).column("kind").to_pylist()
            return sum(k in ("shot_made", "shot_missed") for k in kinds), str(pbp_path)
    local = config.PROJECT_ROOT / "data" / "pbp" / ("%s.json" % code)
    if local.exists():
        ev = json.loads(local.read_text())["events"]
        return sum(e.get("kind") in ("shot_made", "shot_missed") for e in ev), "data/pbp/%s.json" % code
    if fetch_pbp_page:
        params = {"code": code}
        rec = fam.raw.get(RAW_PBP_ENDPOINT, params)
        if rec is None:
            import fetch_pbp
            throttle.wait()
            try:
                events = fetch_pbp.fetch_game(code)
            except Exception as e:                              # noqa: BLE001 (classify the jail, else record)
                status = getattr(getattr(e, "response", None), "status_code", None) or int((re.search(r"(?<!\d)(429|403)(?!\d)", str(e)) or [0, 0])[1])
                if status == 429:
                    raise PageError("pbp page %s HTTP 429: no more requests this run" % code, 429, stop_run=True)
                raise PageError("pbp page %s: %r" % (code, e))
            rec = fam.raw.put(RAW_PBP_ENDPOINT, params, events, "%s/boxscores/pbp/%s.html" % (HOST, code), 200)
        return sum(e.get("kind") in ("shot_made", "shot_missed") for e in rec["body"]), "boxscores/pbp/%s.html" % code
    return None, None


# ── one game ───────────────────────────────────────────────────────────────────────────────────────
def do_game(unit: dict, fam: common.Family, throttle: common.Throttle, sp: dict, contract_dir: Path | None,
            fetch_pbp_page: bool, save_pages: Path | None) -> tuple[int, dict]:
    game = {k: unit[k] for k in ("game", "date", "away", "home")}
    code = cc.br_code(game)
    split = cc.split_of(game, sp)
    rec = page_record(fam, code, throttle)
    html = rec["body"]
    if save_pages is not None:
        save_pages.mkdir(parents=True, exist_ok=True)
        (save_pages / ("%s.html" % code)).write_text(html)
    shots = parse_shot_chart(html)
    cal = calibrate(shots)
    if cal["ok"] is False:
        raise ValueError("%s: the hoop fitted from the tips sits %.1f px from HOOP_PX %s (residual %.2f ft vs %.2f ft): the page's pixel frame is not the assumed one"
                         % (code, cal["hoop_shift_px"], cal["hoop_px"], cal["resid_fit_ft"], cal["resid_default_ft"]))
    df = rows_for_game(game, shots, rec["fetched_at"], split)
    season = season_of(game["date"])
    if len(df):
        fam.write_part(ENDPOINT, season, game["game"], df)
    n_pbp, pbp_src = pbp_shot_count(game, code, contract_dir, fam, throttle, fetch_pbp_page, sp)
    by_team = {t: int(n) for t, n in df.groupby("team").size().items()} if len(df) else {}
    return len(df), {"br_code": code, "endpoint": "boxscores/shot-chart/%s.html" % code, "fetched_at": rec["fetched_at"],
                     "bref_shots": int(len(df)), "markers": len(shots), "pbp_shots": n_pbp, "pbp_source": pbp_src,
                     "by_team": by_team, "made": int(df["made"].sum()) if len(df) else 0, "calibration": cal,
                     "split": split, "heldout": cc.is_heldout(split)}


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def report(fam: common.Family) -> str:
    """Shots recovered per game against the play-by-play count, from the manifest."""
    lines = ["%-26s %-13s %5s %5s %6s %7s %-14s %s" % ("game", "code", "bref", "pbp", "diff", "resid", "split", "status")]
    tot_b, tot_p = 0, 0
    for key, u in sorted(fam.manifest.doc["units"].items()):
        gid = key.split("|")[-1]
        cal = u.get("calibration") or {}
        b, p = u.get("bref_shots"), u.get("pbp_shots")
        diff = ("%+d" % (b - p)) if (b is not None and p is not None) else "n/a"
        if b is not None and p is not None:
            tot_b += b; tot_p += p
        resid = ("%.2f ft" % cal["resid_default_ft"]) if cal.get("resid_default_ft") is not None else "n/a"
        lines.append("%-26s %-13s %5s %5s %6s %7s %-14s %s" % (gid, u.get("br_code", ""), b if b is not None else "", p if p is not None else "",
                                                             diff, resid, u.get("split") or "", u["status"] + ("" if u["status"] != "failed" else ": " + str(u.get("error", ""))[:60])))
    lines.append("total over games with both counts: bref %d, pbp %d (%+d)" % (tot_b, tot_p, tot_b - tot_p))
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def list_games(a) -> list[dict]:
    games = []
    if a.games:
        games += [game_from_id(g) for g in a.games]
    if a.contract_dir:
        step1 = json.loads((Path(a.contract_dir).expanduser() / "manifest.json").read_text())
        games += [{k: e[k] for k in ("game", "date", "away", "home")} for e in step1["games"].values() if e.get("status") == "ok"]
    if a.from_github:
        games += [{k: g[k] for k in ("game", "date", "away", "home")} for g in cc.list_github()]
    seen, out = set(), []
    for g in games:
        if g["game"] not in seen:
            seen.add(g["game"]); out.append(g)
    return out


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], ["2015-16"])
    ap.set_defaults(rpm=MAX_RPM)
    ap.add_argument("--games", nargs="*", help="SportVU game ids (MM.DD.YYYY.AWAY.at.HOME)")
    ap.add_argument("--contract-dir", help="the Step 1 sportvu_contract directory: every game with status ok, and its _pbp.parquet for the comparison")
    ap.add_argument("--from-github", action="store_true", help="every game archive in the SportVU source listing (one GitHub API call)")
    ap.add_argument("--fetch-pbp", action="store_true", help="fetch the basketball-reference pbp page (same throttle) when no local play-by-play exists for the comparison")
    ap.add_argument("--save-pages", help="also write each page as <code>.html into this directory (tests/fixtures/bref for the tests)")
    ap.add_argument("--report", action="store_true", help="print shots recovered vs play-by-play per game from the manifest and stop")
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.report:
        out = report(fam); print(out); return out
    if a.compact_only:
        print(fam.compact()); return "compact only"
    from sportvu import splits
    sp = splits.load()
    games = list_games(a)
    if not games:
        sys.exit("no games: give --games, --contract-dir or --from-github")
    games = cc.order_games(games, sp)
    units = [(common.Manifest.key(ENDPOINT, season_of(g["date"]), g["game"]), {**g, "params": {"code": cc.br_code(g)}}) for g in games]
    rpm = MAX_RPM if a.rpm <= 0 else min(a.rpm, MAX_RPM)
    if a.rpm > MAX_RPM:
        print("--rpm %.0f capped at %.0f (basketball-reference's limit)" % (a.rpm, MAX_RPM), flush=True)
    throttle = common.Throttle(rpm)
    contract_dir = Path(a.contract_dir).expanduser() if a.contract_dir else None
    save_pages = Path(a.save_pages) if a.save_pages else None
    final = common.run_units(fam, units, lambda u: do_game(u, fam, throttle, sp, contract_dir, a.fetch_pbp, save_pages), a,
                             "basketball-reference shot charts for %d SportVU games" % len(units))
    print(report(fam), flush=True)
    return final


if __name__ == "__main__":
    main()
