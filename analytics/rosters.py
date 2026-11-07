"""Rosters table (docs/DATA_CONTRACT.md section 16): one row per player of a SportVU game, with the
basketball-reference shot chart name and slug and the play-by-play name joined to the SportVU player id.

The SportVU frames parquet (section 4) is the roster of record: player_id, player, team_id, team, home, jersey.
Names from the other two sources are matched to it in this order, and the first step that gives a unique
player wins: exact (full name, case and accent insensitive, periods and suffixes dropped), lastname (unique
within the roster, or within the team when the name carries a team hint), fuzzy (difflib ratio >= 0.85 on the
full name, unique best), else none. Play-by-play names come as "F. Lastname" and match on last name plus
first initial (exact), then on a unique last name alone (lastname), then on a fuzzy last name among players
with that initial (fuzzy). Nothing here is fitted: the lookup is deterministic, so running it on a held-out
game only writes that game's table under analytics/rosters/heldout/.

CLI: python -m analytics.rosters --game 12.16.2015.PHX.at.GSW --frames PATH [--bref-html PATH] [--pbp-json PATH]
     --out-dir analytics/rosters --report reports/rosters_unmatched.md
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ("exact", "lastname", "fuzzy", "none")
METHOD_RANK = {m: i for i, m in enumerate(METHODS)}
FUZZY_MIN = 0.85
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
PARTICLES = ("Van", "Von", "De", "Da", "Di", "La", "Le", "Del", "Della", "St.")
ROSTER_COLS = ["sportvu_game", "player_id", "player", "team_id", "team", "home", "jersey"]
OUT_COLS = ROSTER_COLS + ["bref_slug", "bref_name", "pbp_name", "match_method"]
# "S. Curry", "R. Hollis-Jefferson", "N. Van Exel", "K. O'Neal", "O. Porter Jr."
PBP_NAME_RE = re.compile(r"\b([A-Z])\. ((?:(?:%s) )?[A-Z][^\W\d_]*(?:['-][^\W\d_]+)*(?: (?:Jr\.|Sr\.|II|III|IV))?)"
                         % "|".join(re.escape(p) for p in PARTICLES))

SCHEMA = pa.schema([
    ("sportvu_game", pa.string()), ("player_id", pa.int32()), ("player", pa.string()), ("team_id", pa.int32()),
    ("team", pa.string()), ("home", pa.bool_()), ("jersey", pa.string()), ("bref_slug", pa.string()),
    ("bref_name", pa.string()), ("pbp_name", pa.string()), ("match_method", pa.string()),
])


# ── name normalisation ─────────────────────────────────────────────────────────────────────────────
def fix_mojibake(s: str) -> str:
    """UTF-8 read as latin-1 and re-encoded (the cached basketball-reference pages hold "TeletoviÄ\x87"
    for Teletović): undo it when the round trip succeeds and changes the string."""
    if not s or s.isascii():
        return s
    try:
        fixed = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    return fixed if fixed != s else s


def normalize(name: str) -> str:
    """Lower-case ASCII, accents stripped (NFKD, combining marks dropped), periods removed ("T.J." -> "tj"),
    generation suffixes dropped, one space between tokens."""
    s = unicodedata.normalize("NFKD", fix_mojibake(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", "ignore").decode("ascii").lower().replace(".", "")
    toks = [t for t in re.split(r"[\s,]+", s) if t]
    while len(toks) > 1 and toks[-1] in SUFFIXES:
        toks.pop()
    return " ".join(toks)


def split_name(norm: str) -> tuple[str, str]:
    """(first, last) of a normalised name; the last name keeps a leading particle ("van exel")."""
    toks = norm.split(" ")
    if len(toks) == 1:
        return "", toks[0]
    low_particles = {p.rstrip(".").lower() for p in PARTICLES}
    if len(toks) >= 3 and toks[-2] in low_particles:
        return " ".join(toks[:-2]), " ".join(toks[-2:])
    return " ".join(toks[:-1]), toks[-1]


def is_initial_form(name: str) -> bool:
    """"S. Curry" is; "T.J. Warren" is not: the first token is a single letter followed by a period."""
    return bool(re.match(r"^\s*[A-Za-z]\.\s+\S", name or ""))


# ── the roster ─────────────────────────────────────────────────────────────────────────────────────
def roster_from_frames(frames_path) -> pd.DataFrame:
    """Distinct (player_id, player, team_id, team, home, jersey) rows of a Step 1 frames parquet (section 4),
    read column by column and row group by row group so the x/y columns are never loaded."""
    pf = pq.ParquetFile(str(frames_path))
    cols = ["game", "player_id", "player", "team_id", "team", "home", "jersey"]
    parts = [pf.read_row_group(i, columns=cols).to_pandas().drop_duplicates() for i in range(pf.metadata.num_row_groups)]
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    df = df.rename(columns={"game": "sportvu_game"})
    # one row per player: the first jersey seen wins if the source ever disagrees with itself
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    df = df.sort_values(["home", "team", "player_id"], ascending=[False, True, True]).reset_index(drop=True)
    df["player_id"] = df["player_id"].astype("int32")
    df["team_id"] = df["team_id"].astype("int32")
    df["home"] = df["home"].astype(bool)
    df["jersey"] = df["jersey"].astype(object).where(df["jersey"].notna(), None)
    return df[ROSTER_COLS]


# ── matching ───────────────────────────────────────────────────────────────────────────────────────
def _team_mask(roster: pd.DataFrame, hint) -> np.ndarray:
    """Rows of the roster a team hint allows: "home"/"away", a SportVU abbreviation, a basketball-reference
    abbreviation (PHO, BRK, CHO) or a team id. No hint, or an unknown one, allows every row."""
    if hint is None or (isinstance(hint, float) and np.isnan(hint)):
        return np.ones(len(roster), bool)
    h = str(hint).strip()
    if h.lower() in ("home", "away"):
        return (roster["home"] == (h.lower() == "home")).to_numpy()
    from sportvu.colab_contract import BR_ABBR
    sv = {v: k for k, v in BR_ABBR.items()}.get(h.upper(), h.upper())
    m = (roster["team"].str.upper() == sv).to_numpy()
    if m.any():
        return m
    m = (roster["team_id"].astype(str) == h).to_numpy()
    return m if m.any() else np.ones(len(roster), bool)


def _unique(idx: np.ndarray):
    return int(idx[0]) if len(idx) == 1 else None


def _fuzzy_best(target: str, pool: np.ndarray, cands: np.ndarray):
    scored = sorted(((difflib.SequenceMatcher(None, target, pool[i]).ratio(), int(i)) for i in cands), reverse=True)
    best = [s for s in scored if s[0] >= FUZZY_MIN]
    if best and (len(best) == 1 or best[0][0] > best[1][0]):
        return best[0][1]
    return None


def _match_one(name: str, hint, norm_full, firsts, lasts, roster):
    """(roster row index or None, match_method) for one name: the steps of the module docstring in order."""
    n = normalize(name)
    if not n:
        return None, "none"
    allowed = _team_mask(roster, hint)
    first, last = split_name(n)
    if is_initial_form(name):
        ini = first[:1]
        has_initial = np.array([f.startswith(ini) for f in firsts], bool)
        same_last = (lasts == last) & allowed
        hit = _unique(np.flatnonzero(same_last & has_initial))
        if hit is not None:
            return hit, "exact"
        hit = _unique(np.flatnonzero(same_last))
        if hit is not None:
            return hit, "lastname"
        hit = _fuzzy_best(last, lasts, np.flatnonzero(allowed & has_initial))
        return (hit, "fuzzy") if hit is not None else (None, "none")
    hit = _unique(np.flatnonzero((norm_full == n) & allowed))
    if hit is not None:
        return hit, "exact"
    hit = _unique(np.flatnonzero((lasts == last) & allowed))
    if hit is not None:
        return hit, "lastname"
    hit = _fuzzy_best(n, norm_full, np.flatnonzero(allowed))
    return (hit, "fuzzy") if hit is not None else (None, "none")


def match_names(roster: pd.DataFrame, names: list[str], kind: str, teams: list | None = None) -> pd.DataFrame:
    """DataFrame(name, player_id, team_id, match_method), one row per input name in input order. kind is "bref"
    (full names) or "pbp" ("F. Lastname" forms; a full name in a pbp list still matches as a full name). teams is
    an optional parallel list of team hints (see _team_mask); None entries carry no hint."""
    if kind not in ("bref", "pbp"):
        raise ValueError("kind must be 'bref' or 'pbp', not %r" % (kind,))
    roster = roster.reset_index(drop=True)
    norm_full = np.array([normalize(p) for p in roster["player"]], dtype=object)
    firsts = np.array([split_name(x)[0] for x in norm_full], dtype=object)
    lasts = np.array([split_name(x)[1] for x in norm_full], dtype=object)
    teams = list(teams) if teams is not None else [None] * len(names)
    if len(teams) != len(names):
        raise ValueError("teams must be parallel to names")
    rows = []
    for name, hint in zip(names, teams):
        i, method = _match_one(name, hint, norm_full, firsts, lasts, roster)
        rows.append({"name": name, "player_id": int(roster.at[i, "player_id"]) if i is not None else None,
                     "team_id": int(roster.at[i, "team_id"]) if i is not None else None, "match_method": method})
    out = pd.DataFrame(rows, columns=["name", "player_id", "team_id", "match_method"])
    out["player_id"] = out["player_id"].astype("Int32")
    out["team_id"] = out["team_id"].astype("Int32")
    return out


# ── the two name sources ───────────────────────────────────────────────────────────────────────────
def bref_names(html: str, sportvu_game: str | None = None) -> pd.DataFrame:
    """Distinct (name, slug, team hint) of a basketball-reference shot chart page, in page order. The hint is
    "home"/"away" when sportvu_game is given (the page's team code against the game's home code), else the
    page's own code."""
    from collect.bref_shots import parse_shot_chart
    from sportvu.colab_contract import BR_ABBR
    home = sportvu_game.split(".at.")[-1] if sportvu_game else None
    home_br = BR_ABBR.get(home, home) if home else None
    seen, rows = set(), []
    for s in parse_shot_chart(html):
        if not s.get("player"):
            continue
        name = fix_mojibake(s["player"]).strip()
        key = (name, s.get("player_slug"))
        if key in seen:
            continue
        seen.add(key)
        tb = s.get("team_br")
        hint = (("home" if tb.upper() == home_br else "away") if (tb and home_br) else tb)
        rows.append({"name": name, "slug": s.get("player_slug"), "team": hint})
    return pd.DataFrame(rows, columns=["name", "slug", "team"])


def pbp_names(pbp: dict | list) -> list[str]:
    """Distinct "F. Lastname" strings of a fetch_pbp game json ({"meta", "events": [{"desc", ...}]}) or a plain
    list of events, in first-seen order. The event's "team" is not used as a hint: a foul, steal or block names
    a player of the other team in the same description."""
    events = pbp["events"] if isinstance(pbp, dict) else pbp
    seen, out = set(), []
    for e in events:
        desc = fix_mojibake(e.get("desc") or "")
        for m in PBP_NAME_RE.finditer(desc):
            name = "%s. %s" % (m.group(1), m.group(2))
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


# ── build, unmatched, report ───────────────────────────────────────────────────────────────────────
def _best(a: str, b: str) -> str:
    return a if METHOD_RANK[a] <= METHOD_RANK[b] else b


def build(sportvu_game: str, frames_path, bref_html: str | None = None, pbp_json: dict | str | None = None,
          roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """The rosters table of one game: the frames roster with bref_slug, bref_name, pbp_name filled where a name
    matched and match_method the best of the two sources ("none" when neither named the player). bref_html is
    the page text; pbp_json the loaded dict or the raw json text; roster, when given, replaces reading
    frames_path. When several names of one source land on the same player the one with the better method
    (then the first seen) is kept."""
    ros = (roster if roster is not None else roster_from_frames(frames_path)).copy().reset_index(drop=True)
    ros["sportvu_game"] = sportvu_game
    for c in ("bref_slug", "bref_name", "pbp_name"):
        ros[c] = pd.Series([None] * len(ros), dtype=object)
    methods = {"bref_name": ["none"] * len(ros), "pbp_name": ["none"] * len(ros)}
    by_pid = {int(p): i for i, p in enumerate(ros["player_id"])}

    def fill(matches: pd.DataFrame, col: str, slug_of: dict | None = None):
        for _, r in matches.iterrows():
            if pd.isna(r["player_id"]):
                continue
            i = by_pid[int(r["player_id"])]
            if ros.at[i, col] is not None and METHOD_RANK[methods[col][i]] <= METHOD_RANK[r["match_method"]]:
                continue
            ros.at[i, col] = r["name"]
            methods[col][i] = r["match_method"]
            if slug_of is not None:
                ros.at[i, "bref_slug"] = slug_of.get(r["name"])

    if bref_html:
        bn = bref_names(bref_html, sportvu_game)
        if len(bn):
            m = match_names(ros, list(bn["name"]), "bref", teams=list(bn["team"]))
            fill(m, "bref_name", dict(zip(bn["name"], bn["slug"])))
    if pbp_json is not None:
        pbp = json.loads(pbp_json) if isinstance(pbp_json, str) else pbp_json
        names = pbp_names(pbp)
        if names:
            fill(match_names(ros, names, "pbp"), "pbp_name")
    ros["match_method"] = [_best(a, b) for a, b in zip(methods["bref_name"], methods["pbp_name"])]
    return ros[OUT_COLS].reset_index(drop=True)


def unmatched(rosters: pd.DataFrame, bref_names_: list[str] | None, pbp_names_: list[str] | None) -> list[str]:
    """Names of either source that landed on nobody, in source order (bref first), each once."""
    got = set(rosters["bref_name"].dropna()) | set(rosters["pbp_name"].dropna())
    out = []
    for n in list(bref_names_ or []) + list(pbp_names_ or []):
        if n not in got and n not in out:
            out.append(n)
    return out


def summarize(sportvu_game: str, rosters: pd.DataFrame, bref_list: list[str] | None, pbp_list: list[str] | None) -> dict:
    """The per-game entry write_report expects."""
    bref_list, pbp_list = list(bref_list or []), list(pbp_list or [])
    got_b, got_p = set(rosters["bref_name"].dropna()), set(rosters["pbp_name"].dropna())
    return {"game": sportvu_game, "roster_size": int(len(rosters)),
            "bref_total": len(bref_list), "bref_matched": sum(n in got_b for n in bref_list),
            "pbp_total": len(pbp_list), "pbp_matched": sum(n in got_p for n in pbp_list),
            "methods": {m: int((rosters["match_method"] == m).sum()) for m in METHODS},
            "unmatched": unmatched(rosters, bref_list, pbp_list),
            "unnamed_players": [str(p) for p, m in zip(rosters["player"], rosters["match_method"]) if m == "none"]}


def write_report(results: dict, out_path) -> Path:
    """reports/rosters_unmatched.md from {sportvu_game: summarize(...)}: per game the roster size, bref names
    matched / total, pbp names matched / total and the unmatched names. Names and counts only, never raw data."""
    lines = ["# Rosters: name matching per game", "",
             "Written by `python -m analytics.rosters` (docs/DATA_CONTRACT.md section 16). Counts are distinct names",
             "of each source; a source that was not given for a game shows 0 / 0. The method columns count roster",
             "players by the best match_method over both sources.", "",
             "| game | roster | bref matched / total | pbp matched / total | exact | lastname | fuzzy | none |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in sorted(results):
        r = results[g]
        m = r["methods"]
        lines.append("| %s | %d | %d / %d | %d / %d | %d | %d | %d | %d |" % (
            g, r["roster_size"], r["bref_matched"], r["bref_total"], r["pbp_matched"], r["pbp_total"],
            m["exact"], m["lastname"], m["fuzzy"], m["none"]))
    for g in sorted(results):
        r = results[g]
        lines += ["", "## %s" % g, "",
                  "- roster size: %d" % r["roster_size"],
                  "- basketball-reference shot chart names matched: %d / %d" % (r["bref_matched"], r["bref_total"]),
                  "- play-by-play names matched: %d / %d" % (r["pbp_matched"], r["pbp_total"]),
                  "- unmatched names: %s" % (", ".join(r["unmatched"]) if r["unmatched"] else "(none)"),
                  "- roster players no source named: %s" % (", ".join(r["unnamed_players"]) if r["unnamed_players"] else "(none)")]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ── output location ────────────────────────────────────────────────────────────────────────────────
def season_of_game(sportvu_game: str) -> str:
    from collect.common import season_str
    mm, dd, yyyy = sportvu_game.split(".")[:3]
    y, m = int(yyyy), int(mm)
    return season_str(y if m >= 7 else y - 1)


def out_path_for(sportvu_game: str, out_dir, sp: dict | None = None) -> Path:
    """analytics/rosters/<season>/<game>.parquet, or analytics/rosters/heldout/<season>/<game>.parquet for a game
    of a held-out split (sportvu.colab_contract.split_of over sportvu/splits.json)."""
    from sportvu import splits
    from sportvu.colab_contract import split_of, is_heldout
    sp = sp or splits.load()
    parts = sportvu_game.split(".")
    game = {"game": sportvu_game, "home": parts[-1], "away": parts[3] if len(parts) > 3 else None}
    heldout = is_heldout(split_of(game, sp))
    season = season_of_game(sportvu_game)
    base = Path(out_dir) / "heldout" / season if heldout else Path(out_dir) / season
    return base / ("%s.parquet" % sportvu_game)


def write_table(rosters: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rosters[OUT_COLS].copy()
    for c in ("jersey", "bref_slug", "bref_name", "pbp_name"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False), str(path))
    return path


def _load_results(report: Path) -> dict:
    """Earlier games of the report, kept beside it as json so a run adds its game instead of replacing the file."""
    side = report.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True, help="SportVU game id, MM.DD.YYYY.AWAY.at.HOME")
    ap.add_argument("--frames", required=True, help="Step 1 frames parquet of the game")
    ap.add_argument("--bref-html", help="basketball-reference shot chart page of the game")
    ap.add_argument("--pbp-json", help="fetch_pbp json of the game (data/pbp/<code>.json)")
    ap.add_argument("--out-dir", default=str(ROOT / "analytics" / "rosters"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "rosters_unmatched.md"))
    a = ap.parse_args(argv)
    html = Path(a.bref_html).read_text(encoding="utf-8", errors="replace") if a.bref_html else None
    pbp = json.loads(Path(a.pbp_json).read_text(encoding="utf-8")) if a.pbp_json else None
    rosters = build(a.game, a.frames, bref_html=html, pbp_json=pbp)
    bl = list(bref_names(html, a.game)["name"]) if html else []
    pl = pbp_names(pbp) if pbp else []
    out = write_table(rosters, out_path_for(a.game, a.out_dir))
    report = Path(a.report)
    results = _load_results(report)
    results[a.game] = summarize(a.game, rosters, bl, pl)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.with_suffix(".json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(results, report)
    r = results[a.game]
    print("%s: %d players -> %s; bref %d/%d, pbp %d/%d, unmatched %s" % (
        a.game, len(rosters), out, r["bref_matched"], r["bref_total"], r["pbp_matched"], r["pbp_total"],
        ", ".join(r["unmatched"]) or "(none)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
