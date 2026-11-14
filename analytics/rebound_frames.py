"""analytics/rebound_frames.py: locate the rim frame and the rebound frame of every missed shot
(docs/DATA_CONTRACT.md section 18, the rebound_frames table).

Inputs are the frames table (section 4) and the play-by-play table (section 5) of one game. Every
threshold below is fixed by the contract; nothing here is fitted, and held-out games (sportvu/splits.json)
are only ever run as a smoke test.

    python -m analytics.rebound_frames --game 12.16.2015.PHX.at.GSW --frames FRAMES.parquet \\
        --pbp data/pbp/201512160GSW.json --out-dir analytics/rebound_frames \\
        --report reports/localisation_rebounds.md
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── fixed parameters (contract section 18; never tuned) ───────────────────────────────────────────
BASKETS = ((5.25, 25.0), (88.75, 25.0))   # feet
REBOUND_EVENT_MAX_S = 6.0                 # the rebound event follows the miss by at most this much clock
WINDOW_BEFORE_S = 3.0                     # window starts this much clock before the miss (clock counts down)
WINDOW_AFTER_S = 6.0                      # and ends this much clock after it
RIM_Z_FT = 10.0                           # the ball is "at the rim" once it falls below this height
RIM_RADIUS_FT = 6.0                       # within this horizontal distance of a basket
RIM_FALLBACK_RADIUS_FT = 4.0              # used when the ball never rises above RIM_Z_FT in the window
HOLD_RADIUS_FT = 2.0                      # a player within this horizontal distance of the ball
HOLD_Z_FT = 7.0                           # while the ball is under this height holds it
CONFIDENCE_FRAMES = 75.0                  # 3 s at 25 Hz: confidence falls to 0 at this rim-to-rebound gap
HIGH_CONFIDENCE = 0.5
FLAGS = ("high", "low", "unmatched")

FT_KINDS = ("ft_made", "ft_missed")
HELDOUT_SUBDIR = "heldout"

SCHEMA = pa.schema([
    ("sportvu_game", pa.string()), ("event", pa.int32()), ("rebound_event", pa.int32()),
    ("period", pa.int8()), ("clock_s", pa.float32()), ("rim_frame", pa.int32()),
    ("rebound_frame", pa.int32()), ("rebounder_id", pa.int32()), ("pbp_rebounder_name", pa.string()),
    ("pbp_rebounder_id", pa.int32()), ("identity_match", pa.bool_()), ("confidence", pa.float32()),
    ("confidence_flag", pa.string()),
])
COLUMNS = [f.name for f in SCHEMA]

_REBOUND_RE = re.compile(r"rebound by (.+?)\s*$", re.IGNORECASE)
_FT_COUNT_RE = re.compile(r"(\d+) of (\d+)")


# ── names ─────────────────────────────────────────────────────────────────────────────────────────
def norm_name(s) -> str:
    """Lower-case ASCII letters only: repairs latin-1 mojibake (the basketball-reference page gives
    "TeletoviÄ\\x87"), strips accents and punctuation so "Teletović", the mojibake and "Teletovic" agree."""
    if s is None:
        return ""
    s = str(s)
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if c.isascii() and c.isalpha()).lower()


def pbp_rebounder(desc) -> str | None:
    """'Defensive rebound by D. Green' -> 'D. Green'; 'Offensive rebound by Team' and anything else -> None."""
    if not desc:
        return None
    m = _REBOUND_RE.search(str(desc))
    if not m:
        return None
    name = m.group(1).strip()
    if norm_name(name) == "team":
        return None
    return name


def match_player(name, roster: list[tuple[int, str]]) -> int | None:
    """Match 'D. Green' to the (player_id, player) pairs of the frames by last name, breaking ties with the
    first initial. None when no player or more than one player fits."""
    if not name:
        return None
    parts = str(name).replace(".", " ").split()
    if not parts:
        return None
    if len(parts) > 1:
        initial, last = norm_name(parts[0])[:1], norm_name(" ".join(parts[1:]))
    else:
        initial, last = "", norm_name(parts[0])
    if not last:
        return None
    cands = []
    for pid, full in roster:
        toks = str(full).split()
        if not toks:
            continue
        tail_forms = {norm_name(toks[-1]), norm_name(" ".join(toks[1:]))}
        if last in tail_forms:
            cands.append((int(pid), norm_name(toks[0])[:1]))
    if len(cands) > 1 and initial:
        cands = [c for c in cands if c[1] == initial]
    ids = sorted({c[0] for c in cands})
    return ids[0] if len(ids) == 1 else None


# ── misses and their rebound events ───────────────────────────────────────────────────────────────
def miss_events(events: pd.DataFrame) -> list[int]:
    """Positional indices (into events sorted by event) of every shot_missed and of the last ft_missed of a
    free-throw trip: 'N of M' with N == M, or (no count in the text, e.g. a technical) a miss not followed
    by another free throw of the same period and clock."""
    kinds = events["kind"].tolist(); descs = events["desc"].tolist()
    periods = events["period"].tolist(); clocks = events["clock_s"].tolist()
    out = []
    for i, k in enumerate(kinds):
        if k == "shot_missed":
            out.append(i)
        elif k == "ft_missed":
            m = _FT_COUNT_RE.search(str(descs[i] or ""))
            if m:
                if int(m.group(1)) == int(m.group(2)):
                    out.append(i)
            else:
                j = i + 1
                nxt_ft = (j < len(kinds) and kinds[j] in FT_KINDS and periods[j] == periods[i]
                          and clocks[j] == clocks[i])
                if not nxt_ft:
                    out.append(i)
    return out


def rebound_event_for(events: pd.DataFrame, i: int) -> int | None:
    """Positional index of the next rebound event after row i when it is in the same period, at or after the
    miss on the clock (smaller or equal clock_s) and within REBOUND_EVENT_MAX_S; else None."""
    period, clock = events["period"].iat[i], float(events["clock_s"].iat[i])
    kinds = events["kind"].values; periods = events["period"].values; clocks = events["clock_s"].values
    for j in range(i + 1, len(events)):
        if periods[j] != period:
            return None
        if kinds[j] == "rebound":
            c = float(clocks[j])
            return j if (c <= clock and clock - c <= REBOUND_EVENT_MAX_S) else None
    return None


# ── frames ────────────────────────────────────────────────────────────────────────────────────────
def per_frame(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per frame (the first row of each frame carries the clock and the ball), sorted by frame."""
    cols = ["frame", "period", "game_clock_s", "ball_x", "ball_y", "ball_z"]
    pf = frames[cols].drop_duplicates("frame", keep="first").sort_values("frame", kind="stable")
    return pf.reset_index(drop=True)


def rim_frame_in(win: pd.DataFrame) -> int:
    """First frame of the window where the ball falls below RIM_Z_FT within RIM_RADIUS_FT of a basket after
    having been above RIM_Z_FT earlier in the window; if it never exceeds RIM_Z_FT, the first frame within
    RIM_FALLBACK_RADIUS_FT of a basket. -1 when neither happens."""
    if len(win) == 0:
        return -1
    fr = win["frame"].to_numpy()
    bx = win["ball_x"].to_numpy(float); by = win["ball_y"].to_numpy(float); bz = win["ball_z"].to_numpy(float)
    ok = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz)
    d = np.full(len(win), np.inf)
    for x0, y0 in BASKETS:
        d = np.minimum(d, np.hypot(bx - x0, by - y0))
    above = ok & (bz > RIM_Z_FT)
    if above.any():
        seen = np.concatenate([[False], np.cumsum(above)[:-1] > 0])     # above at a strictly earlier frame
        hit = ok & seen & (bz < RIM_Z_FT) & (d <= RIM_RADIUS_FT)
        return int(fr[np.argmax(hit)]) if hit.any() else -1
    hit = ok & (d <= RIM_FALLBACK_RADIUS_FT)
    return int(fr[np.argmax(hit)]) if hit.any() else -1


def rebound_frame_in(win: pd.DataFrame, players: pd.DataFrame, rim_frame: int) -> tuple[int, int | None]:
    """(rebound_frame, rebounder_id): the first frame after rim_frame (inside the window) where a player is
    within HOLD_RADIUS_FT of the ball horizontally while ball_z < HOLD_Z_FT; the nearest such player."""
    if rim_frame < 0:
        return -1, None
    after = win[(win["frame"] > rim_frame) & (win["ball_z"] < HOLD_Z_FT)]
    if after.empty:
        return -1, None
    pl = players.merge(after[["frame", "ball_x", "ball_y"]], on="frame", how="inner")
    if pl.empty:
        return -1, None
    dist = np.hypot(pl["x"].to_numpy(float) - pl["ball_x"].to_numpy(float),
                    pl["y"].to_numpy(float) - pl["ball_y"].to_numpy(float))
    pl = pl.assign(dist=dist)
    near = pl[pl["dist"] <= HOLD_RADIUS_FT].sort_values(["frame", "dist"], kind="stable")
    if near.empty:
        return -1, None
    r = near.iloc[0]
    return int(r["frame"]), int(r["player_id"])


# ── the table ─────────────────────────────────────────────────────────────────────────────────────
def locate_rebounds(frames: pd.DataFrame, events: pd.DataFrame, sportvu_game: str) -> pd.DataFrame:
    """One row per missed shot (shot_missed, or the last ft_missed of a trip): the contract section 18 table."""
    events = events.sort_values("event", kind="stable").reset_index(drop=True)
    pf = per_frame(frames)
    players = frames[["frame", "period", "player_id", "x", "y"]]
    roster = [(int(p), str(n))
              for p, n in frames[["player_id", "player"]].drop_duplicates().itertuples(index=False)]
    rows = []
    for i in miss_events(events):
        period = int(events["period"].iat[i]); clock = float(events["clock_s"].iat[i])
        j = rebound_event_for(events, i)
        name = pbp_rebounder(events["desc"].iat[j]) if j is not None else None
        pbp_id = match_player(name, roster) if name else None
        win = pf[(pf["period"] == period) & (pf["game_clock_s"] <= clock + WINDOW_BEFORE_S)
                 & (pf["game_clock_s"] >= clock - WINDOW_AFTER_S)]
        rim = rim_frame_in(win)
        reb, rebounder = rebound_frame_in(win, players[players["period"] == period], rim)
        ident = (rebounder == pbp_id) if (rebounder is not None and pbp_id is not None) else None
        if reb < 0 or rim < 0:
            conf = 0.0
        else:
            conf = (1.0 - min(1.0, (reb - rim) / CONFIDENCE_FRAMES)) * (1.0 if ident else 0.5)
        if pbp_id is None or reb < 0:
            flag = "unmatched"
        elif conf >= HIGH_CONFIDENCE and ident:
            flag = "high"
        else:
            flag = "low"
        rows.append({
            "sportvu_game": sportvu_game, "event": int(events["event"].iat[i]),
            "rebound_event": int(events["event"].iat[j]) if j is not None else None,
            "period": period, "clock_s": clock, "rim_frame": rim, "rebound_frame": reb,
            "rebounder_id": rebounder, "pbp_rebounder_name": name, "pbp_rebounder_id": pbp_id,
            "identity_match": ident, "confidence": conf, "confidence_flag": flag,
        })
    return _typed(pd.DataFrame(rows, columns=COLUMNS))


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({
        "sportvu_game": "string", "event": "int32", "rebound_event": "Int32", "period": "int8",
        "clock_s": "float32", "rim_frame": "int32", "rebound_frame": "int32", "rebounder_id": "Int32",
        "pbp_rebounder_name": "string", "pbp_rebounder_id": "Int32", "identity_match": "boolean",
        "confidence": "float32", "confidence_flag": "string",
    })


def to_arrow(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df[COLUMNS], schema=SCHEMA, preserve_index=False)


def validate(rebound_frames: pd.DataFrame) -> dict:
    """Coverage and agreement of one table: how many misses got a rebound frame, how often the frame-level
    rebounder is the play-by-play rebounder when both are known, and the share flagged high."""
    n = int(len(rebound_frames))
    found = int((rebound_frames["rebound_frame"].to_numpy() >= 0).sum()) if n else 0
    ident = rebound_frames["identity_match"]
    known = int(ident.notna().sum()) if n else 0
    agree = float(ident.dropna().astype(bool).mean()) if known else None
    high = int((rebound_frames["confidence_flag"] == "high").sum()) if n else 0
    return {
        "n_misses": n, "n_rebound_found": found, "share_rebound_found": (found / n) if n else None,
        "n_identity_known": known, "identity_agreement": agree, "share_high": (high / n) if n else None,
    }


# ── inputs, outputs, the per-game run ─────────────────────────────────────────────────────────────
FRAME_COLUMNS = ["game", "period", "frame", "game_clock_s", "ball_x", "ball_y", "ball_z",
                 "team_id", "home", "player_id", "player", "x", "y"]


def season_of_game(game: str) -> str:
    """'12.16.2015.PHX.at.GSW' -> '2015-16' (seasons start in July, as collect.common.season_str)."""
    m = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})\.([A-Z]{3})\.at\.([A-Z]{3})$", game)
    if not m:
        raise ValueError("not a SportVU game id: %r (want MM.DD.YYYY.AWAY.at.HOME)" % game)
    month, year = int(m.group(1)), int(m.group(3))
    start = year if month >= 7 else year - 1
    return "%d-%02d" % (start, (start + 1) % 100)


def home_of_game(game: str) -> str:
    return game.rsplit(".", 1)[-1]


def read_frames(path) -> tuple[pa.Table, pd.DataFrame]:
    table = pq.read_table(path, columns=FRAME_COLUMNS)
    return table, table.to_pandas()


def read_events(game: str, path, frames_table: pa.Table) -> pd.DataFrame:
    """Section 5 events from a fetch_pbp.py JSON (joined to the frames with sportvu.colab_contract.join_pbp)
    or from an already joined _pbp.parquet."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        from sportvu import colab_contract
        raw = json.loads(path.read_text())
        ev = raw["events"] if isinstance(raw, dict) else raw
        fp, fc = colab_contract.frame_clocks(frames_table)
        return colab_contract.join_pbp(game, ev, fp, fc).to_pandas()
    return pq.read_table(path).to_pandas()


def split_of_game(game: str) -> str | None:
    from sportvu import colab_contract, splits
    return colab_contract.split_of({"game": game, "home": home_of_game(game)}, splits.load())


def output_path(game: str, out_dir, split: str | None) -> Path:
    from sportvu import colab_contract
    d = Path(out_dir)
    if colab_contract.is_heldout(split):
        d = d / HELDOUT_SUBDIR
    return d / season_of_game(game) / (game + ".parquet")


REPORT_HEADER = """# Rebound localisation (DATA_CONTRACT section 18)

`analytics/rebound_frames.py` locates, for every missed shot, the rim frame (the ball falls below 10 ft within
6 ft of a basket after rising above 10 ft earlier in the window) and the rebound frame (the first later frame
where a player is within 2 ft of a ball under 7 ft), and compares that player with the basketball-reference
rebounder. Every threshold is fixed in the module and none was tuned on any game. One line per `run`, with the
`validate()` numbers.

| run (UTC) | game | split | n_misses | n_rebound_found | share_rebound_found | n_identity_known | identity_agreement | share_high | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    return "%.3f" % v if isinstance(v, float) else str(v)


def append_report(report_path, game: str, split: str | None, stats: dict, note: str = "") -> None:
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or not p.read_text().strip():
        p.write_text(REPORT_HEADER)
    keys = ("n_misses", "n_rebound_found", "share_rebound_found", "n_identity_known",
            "identity_agreement", "share_high")
    line = "| %s | %s | %s | %s | %s |\n" % (
        time.strftime("%Y-%m-%d %H:%M", time.gmtime()), game, split or "none",
        " | ".join(_fmt(stats[k]) for k in keys), note)
    with p.open("a") as fh:
        fh.write(line)


def run(game: str, frames_path, pbp_json_or_parquet, out_dir, report_path, note: str | None = None) -> dict:
    """Build the table for one game, write <out_dir>/<season>/<game>.parquet (held-out games under
    <out_dir>/heldout/<season>/) and append one line to the report. Returns validate() plus the paths."""
    from sportvu import colab_contract
    table, frames = read_frames(frames_path)
    events = read_events(game, pbp_json_or_parquet, table)
    split = split_of_game(game)
    df = locate_rebounds(frames, events, game)
    out = output_path(game, out_dir, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_arrow(df), out, compression="zstd")
    stats = validate(df)
    if note is None:
        note = ("held-out game, used as a smoke test only, no parameter tuned on it"
                if colab_contract.is_heldout(split) else "")
    append_report(report_path, game, split, stats, note)
    return {**stats, "split": split, "out": str(out), "report": str(report_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rebound_frames table for one game (DATA_CONTRACT section 18)")
    ap.add_argument("--game", required=True, help="SportVU game id, MM.DD.YYYY.AWAY.at.HOME")
    ap.add_argument("--frames", required=True, help="frames parquet (contract section 4)")
    ap.add_argument("--pbp", required=True, help="fetch_pbp.py JSON or joined _pbp.parquet (section 5)")
    ap.add_argument("--out-dir", default="analytics/rebound_frames")
    ap.add_argument("--report", default="reports/localisation_rebounds.md")
    a = ap.parse_args(argv)
    res = run(a.game, a.frames, a.pbp, a.out_dir, a.report)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
