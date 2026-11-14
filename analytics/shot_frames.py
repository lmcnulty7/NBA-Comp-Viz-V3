"""analytics/shot_frames.py: locate each play-by-play shot in the SportVU frames (docs/DATA_CONTRACT.md section 17).

One row per shot_made or shot_missed event: the release frame (ball leaves the shooter's hands), the apex
frame (highest ball_z of the flight) and the outcome frame (ball drops below the rim near a basket), plus
the identity check (is the player nearest the ball at release the one play-by-play names?) and a confidence.

Rules (fixed, never tuned on a held-out game):
  window     frames of the event's period with game_clock_s in [clock_s - 0.5, clock_s + 3.0] (the clock counts
             down: the logged time marks the outcome, the release is earlier at a larger clock)
  release    the last window frame where the ball is within 2.5 ft horizontally of the shooter and ball_z < 8 ft,
             after which ball_z rises for at least 0.2 s (5 frames at 25 Hz: non-decreasing over the next 5
             frames and higher at the end of them)
  apex       the frame of maximum ball_z after the release, within the window extended 2 s later
  outcome    the first frame after the apex where ball_z < 10 ft within 6 ft horizontally of either basket
  confidence (1 - min(1, |apex - release - 25| / 25)) * (1 if identity_match else 0.5)
  flag       high when confidence >= 0.6 and identity_match; unmatched when the shooter is unknown or no release
             frame was found; else low. A shot whose shooter matches nobody on the frames roster has its release
             located by the ball alone (nearest player to the ball at the candidate release) and is unmatched.

    python -m analytics.shot_frames --game 12.16.2015.PHX.at.GSW --frames out/heldout/12.16.2015.PHX.at.GSW.parquet \\
        --pbp data/pbp/201512160GSW.json --out-dir analytics/shot_frames --report reports/localisation_shots.md
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

HZ = 25
WINDOW_BEFORE_S, WINDOW_AFTER_S = 0.5, 3.0     # clock_s - 0.5 .. clock_s + 3.0
APEX_EXTRA_S = 2.0                             # the apex and outcome search runs to clock_s - 0.5 - 2.0
HOLD_FT, RELEASE_Z_FT = 2.5, 8.0               # ball near the shooter, below this height
RISE_FRAMES = int(round(0.2 * HZ))             # 5 frames at 25 Hz
RIM_Z_FT, RIM_RADIUS_FT = 10.0, 6.0
BASKETS = ((5.25, 25.0), (88.75, 25.0))
FLIGHT_FRAMES = 25                             # expected release-to-apex, the confidence anchor
HIGH_CONF = 0.6
SHOT_KINDS = ("shot_made", "shot_missed")
HELDOUT_SUBDIR = "heldout"

COLUMNS = ["sportvu_game", "event", "period", "clock_s", "made", "pbp_shooter_name", "shooter_id",
           "release_frame", "apex_frame", "outcome_frame", "shooter_at_release_id", "identity_match",
           "confidence", "confidence_flag"]
_SHOOTER_RE = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s+(makes|misses)\b")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


# ── names ──────────────────────────────────────────────────────────────────────────────────────────
def parse_shooter(desc: str | None) -> tuple[str | None, str | None]:
    """('K', 'Thompson') from 'K. Thompson makes 2-pt layup from 1 ft'; (None, None) when the text differs."""
    m = _SHOOTER_RE.match(desc or "")
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _norm(s: str) -> str:
    if not isinstance(s, str):          # NaN or None: a nameless player (player_id 0 in some frames)
        return ""
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def _last_name(full: str) -> str:
    toks = [t for t in _norm(full).split() if t not in _SUFFIXES]
    return " ".join(toks[1:]) if len(toks) > 1 else (toks[0] if toks else "")


def match_shooter(initial: str | None, last: str | None, roster: pd.DataFrame, side: str | None) -> int | None:
    """player_id of the roster row whose last name is `last` (first initial breaks ties); roster columns
    player_id, player, home. `side` ('home' or 'away') restricts the team when known."""
    if not last:
        return None
    want = _norm(last)
    want_toks = [t for t in want.split() if t not in _SUFFIXES]
    want = " ".join(want_toks)
    pool = roster
    if side in ("home", "away"):
        pool = roster[roster["home"] == (side == "home")]
    for scope in (pool, roster):
        names = scope["player"].astype(str)
        full_last = names.map(_last_name)
        final_tok = full_last.map(lambda s: s.split()[-1] if s else "")
        hits = scope[(full_last == want) | (final_tok == want)]
        if len(hits) > 1 and initial:
            by_init = hits[names.loc[hits.index].map(lambda s: _norm(s)[:1]) == initial.lower()]
            if len(by_init):
                hits = by_init
        if len(hits):
            return int(hits["player_id"].iloc[0])
    return None


# ── per-frame views ────────────────────────────────────────────────────────────────────────────────
class _Frames:
    """Sorted per-frame arrays (period, clock, ball) and the row range of every frame id."""

    def __init__(self, frames: pd.DataFrame):
        order = np.argsort(frames["frame"].to_numpy(), kind="stable")
        self.rows = frames.iloc[order].reset_index(drop=True)
        fr = self.rows["frame"].to_numpy()
        self.frame_ids, first, counts = np.unique(fr, return_index=True, return_counts=True)
        self.first, self.counts = first, counts
        self.period = self.rows["period"].to_numpy()[first].astype(int)
        self.clock = self.rows["game_clock_s"].to_numpy(dtype=float)[first]
        self.ball_x = self.rows["ball_x"].to_numpy(dtype=float)[first]
        self.ball_y = self.rows["ball_y"].to_numpy(dtype=float)[first]
        self.ball_z = self.rows["ball_z"].to_numpy(dtype=float)[first]
        self.px = self.rows["x"].to_numpy(dtype=float)
        self.py = self.rows["y"].to_numpy(dtype=float)
        self.pid = self.rows["player_id"].to_numpy()

    def window(self, period: int, lo: float, hi: float) -> np.ndarray:
        """Indices (into the per-frame arrays) of the period's frames with lo <= clock <= hi, in frame order."""
        return np.where((self.period == period) & (self.clock >= lo) & (self.clock <= hi))[0]

    def players(self, i: int):
        a, n = self.first[i], self.counts[i]
        return self.pid[a:a + n], self.px[a:a + n], self.py[a:a + n]

    def ball_dist(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(player ids, horizontal ball-to-player distances) at frame index i."""
        ids, x, y = self.players(i)
        return ids, np.hypot(x - self.ball_x[i], y - self.ball_y[i])

    def rises(self, i: int) -> bool:
        """ball_z non-decreasing over the RISE_FRAMES frames after i (same period, no gap) and higher at the end."""
        j = i + RISE_FRAMES
        if j >= len(self.ball_z) or self.period[j] != self.period[i]:
            return False
        z = self.ball_z[i:j + 1]
        if np.isnan(z).any():
            return False
        return bool(np.all(np.diff(z) >= 0) and z[-1] > z[0])


# ── the locator ────────────────────────────────────────────────────────────────────────────────────
def _release(fv: _Frames, win: np.ndarray, shooter_id: int | None):
    """(frame index of the release or None, nearest player id at it or None)."""
    for i in win[::-1]:                                   # last window frame first (frames run forward in time)
        if np.isnan(fv.ball_z[i]) or fv.ball_z[i] >= RELEASE_Z_FT:
            continue
        ids, d = fv.ball_dist(i)
        if not len(ids):
            continue
        if shooter_id is not None:
            hit = np.where(ids == shooter_id)[0]
            if not len(hit) or d[hit[0]] > HOLD_FT:
                continue
        elif d.min() > HOLD_FT:
            continue
        if fv.rises(i):
            return int(i), int(ids[int(np.argmin(d))])
    return None, None


def _apex(fv: _Frames, rel: int, period: int, lo: float) -> int | None:
    after = np.where((np.arange(len(fv.ball_z)) > rel) & (fv.period == period) & (fv.clock >= lo))[0]
    after = after[~np.isnan(fv.ball_z[after])]
    return int(after[np.argmax(fv.ball_z[after])]) if len(after) else None


def _outcome(fv: _Frames, apex: int, period: int, lo: float) -> int | None:
    after = np.where((np.arange(len(fv.ball_z)) > apex) & (fv.period == period) & (fv.clock >= lo))[0]
    for i in after:
        if np.isnan(fv.ball_z[i]) or fv.ball_z[i] >= RIM_Z_FT:
            continue
        if any(np.hypot(fv.ball_x[i] - bx, fv.ball_y[i] - by) <= RIM_RADIUS_FT for bx, by in BASKETS):
            return int(i)
    return None


def locate_shots(frames: pd.DataFrame, events: pd.DataFrame, sportvu_game: str) -> pd.DataFrame:
    """The section 17 table: one row per shot_made or shot_missed event of `events` (section 5 columns)
    located in `frames` (section 4 columns)."""
    fv = _Frames(frames)
    roster = fv.rows[["player_id", "player", "home"]].drop_duplicates("player_id").reset_index(drop=True)
    shots = events[events["kind"].isin(SHOT_KINDS)]
    out = []
    for e in shots.itertuples(index=False):
        period, clock = int(e.period), float(e.clock_s)
        initial, last = parse_shooter(e.desc)
        side = e.team if isinstance(e.team, str) else None
        shooter_id = match_shooter(initial, last, roster, side)
        win = fv.window(period, clock - WINDOW_BEFORE_S, clock + WINDOW_AFTER_S)
        rel, near = _release(fv, win, shooter_id)
        apex = outcome = None
        if rel is not None:
            lo = clock - WINDOW_BEFORE_S - APEX_EXTRA_S
            apex = _apex(fv, rel, period, lo)
            if apex is not None:
                outcome = _outcome(fv, apex, period, lo)
        ident = (near == shooter_id) if (shooter_id is not None and near is not None) else None
        if rel is None or apex is None:
            conf = 0.0
        else:
            conf = (1.0 - min(1.0, abs((apex - rel) - FLIGHT_FRAMES) / FLIGHT_FRAMES)) * (1.0 if ident else 0.5)
        if shooter_id is None or rel is None:
            flag = "unmatched"
        elif conf >= HIGH_CONF and ident:
            flag = "high"
        else:
            flag = "low"
        fid = lambda i: int(fv.frame_ids[i]) if i is not None else -1   # noqa: E731
        out.append({
            "sportvu_game": sportvu_game, "event": int(e.event), "period": period, "clock_s": clock,
            "made": e.kind == "shot_made",
            "pbp_shooter_name": ("%s. %s" % (initial, last)) if last else None,
            "shooter_id": shooter_id, "release_frame": fid(rel), "apex_frame": fid(apex),
            "outcome_frame": fid(outcome), "shooter_at_release_id": near, "identity_match": ident,
            "confidence": conf, "confidence_flag": flag})
    return _typed(pd.DataFrame(out, columns=COLUMNS))


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({
        "sportvu_game": "string", "event": "int32", "period": "int8", "clock_s": "float32", "made": "bool",
        "pbp_shooter_name": "string", "shooter_id": "Int32", "release_frame": "int32", "apex_frame": "int32",
        "outcome_frame": "int32", "shooter_at_release_id": "Int32", "identity_match": "boolean",
        "confidence": "float32", "confidence_flag": "string"})


def validate(shot_frames: pd.DataFrame) -> dict:
    n = int(len(shot_frames))
    found = int((shot_frames["release_frame"] >= 0).sum()) if n else 0
    known = shot_frames["identity_match"].dropna() if n else pd.Series([], dtype="boolean")
    return {
        "n_shots": n,
        "n_release_found": found,
        "share_release_found": round(found / n, 4) if n else None,
        "n_identity_known": int(len(known)),
        "identity_agreement": round(float(known.astype(bool).mean()), 4) if len(known) else None,
        "share_high": round(float((shot_frames["confidence_flag"] == "high").mean()), 4) if n else None,
    }


# ── files ──────────────────────────────────────────────────────────────────────────────────────────
def season_of_game(game: str) -> str:
    """'2015-16' from 'MM.DD.YYYY.AWAY.at.HOME' (seasons turn over in July)."""
    mm, _dd, yyyy = game.split(".")[:3]
    y = int(yyyy) if int(mm) >= 7 else int(yyyy) - 1
    return "%d-%02d" % (y, (y + 1) % 100)


def load_events(game: str, pbp_path: Path, table) -> pd.DataFrame:
    """Section 5 events: a fetch_pbp.py JSON joined to the frames, or an already joined pbp parquet."""
    import pyarrow.parquet as pq
    from sportvu import colab_contract as cc
    if str(pbp_path).endswith(".json"):
        events = json.loads(Path(pbp_path).read_text())["events"]
        fp, fc = cc.frame_clocks(table)
        return cc.join_pbp(game, events, fp, fc).to_pandas()
    return pq.read_table(pbp_path).to_pandas()


def out_path(game: str, out_dir: Path, home: str) -> tuple[Path, bool]:
    from sportvu import colab_contract as cc, splits
    split = cc.split_of({"game": game, "home": home}, splits.load())
    held = cc.is_heldout(split)
    d = Path(out_dir) / HELDOUT_SUBDIR if held else Path(out_dir)
    return d / season_of_game(game) / ("%s.parquet" % game), held

REPORT_HEADER = """# Shot localisation (analytics/shot_frames.py)

One line per game: the validate() numbers of analytics/shot_frames/<season>/<game>.parquet. Rules and
thresholds are the fixed values written in analytics/shot_frames.py; nothing is tuned on any game. A line
marked held-out comes from a held-out game of sportvu/splits.json, used as a smoke test only, with no
parameter tuned on it.

| game | split | n_shots | n_release_found | share_release_found | n_identity_known | identity_agreement | share_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
"""


def append_report(report_path: Path, game: str, held: bool, v: dict) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not report_path.exists():
        report_path.write_text(REPORT_HEADER)
    split = "held-out (smoke test only, no parameter tuned on it)" if held else "train"
    fmt = lambda x: "n/a" if x is None else ("%.4f" % x if isinstance(x, float) else str(x))   # noqa: E731
    cells = [game, split] + [fmt(v[k]) for k in ("n_shots", "n_release_found", "share_release_found",
                                                  "n_identity_known", "identity_agreement", "share_high")]
    with report_path.open("a") as f:
        f.write("| " + " | ".join(cells) + " |\n")


def run(game: str, frames_path, pbp_json_or_parquet, out_dir, report_path) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(frames_path, columns=["game", "period", "frame", "game_clock_s", "ball_x", "ball_y",
                                                "ball_z", "team", "team_id", "home", "player_id", "player", "x", "y"])
    events = load_events(game, Path(pbp_json_or_parquet), table)
    frames = table.to_pandas()
    home_rows = frames.loc[frames["home"], "team"]
    home = str(home_rows.iloc[0]) if len(home_rows) else game.split(".")[-1]
    sf = locate_shots(frames, events, game)
    dest, held = out_path(game, Path(out_dir), home)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.to_parquet(dest, index=False)
    v = validate(sf)
    append_report(Path(report_path), game, held, v)
    print(json.dumps({"game": game, "heldout": held, "out": str(dest), **v}))
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--pbp", required=True, help="fetch_pbp.py JSON or a section 5 pbp parquet")
    ap.add_argument("--out-dir", default="analytics/shot_frames")
    ap.add_argument("--report", default="reports/localisation_shots.md")
    a = ap.parse_args(argv)
    run(a.game, a.frames, a.pbp, a.out_dir, a.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
