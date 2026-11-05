"""analytics.fixtures: deterministic synthetic data in the shapes of docs/DATA_CONTRACT.md.

Every lane of the analytics build tests against these frames instead of raw SportVU data. Nothing here
touches the network or raw data; every table is generated from a numpy RandomState seeded by the caller,
so the same seed gives the same bytes. The contract tables (sections 16 to 20) pass
``analytics.schema.validate`` and carry ``analytics.schema.cast`` dtypes.

API (docs/contract/fixtures.md documents the guarantees):
    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None) -> dict of DataFrames
        keys: frames (section 4), events (section 5 plus the v1.1 NBA.com columns), shots (section 10 shape),
        rosters (16), shot_frames (17), rebound_frames (18), possessions (19), states_1hz (20), plus
        meta (ids, date, teams), truth (the hidden effects and action labels) and manifest_entry.
    synth_season(seed, n_games=8, heldout_every=0) -> list of synth_game dicts
    synth_aggregates(seed, n_players=60, seasons=("2015-16", "2016-17")) -> (aggregates, positions)
    synth_stints(seed, n_games=20, seasons=("2020-21",), n_players_per_team=9) -> dict(stints, playbyplay, games)
    synth_odds(seed, games) -> odds DataFrame (section 14)
    synth_labels(seed, game) -> dict(actions, screens)
    write_fixture_root(root, seed=0, n_games=6, heldout_every=3) -> manifest dict (the DATA_ROOT layout)

CLI: python -m analytics.fixtures --out DIR [--seed 0] [--games 6] [--heldout-every 3]

Physics of a synthetic game (all in feet, 25 Hz, court 94 x 50, baskets at (5.25, 25) and (88.75, 25)):
    the ball is carried by a handler (within 1 ft of him, z between 1 and 5 ft), passed between players
    (0.4 to 0.7 s flights under 8 ft), and shot: gathered to 7.5 ft, released, parabola to an apex above
    8 ft, down to the rim at exactly 10 ft. A made shot drops through the net and rests on the baseline
    until the next handler picks it up; a miss bounces off the rim to a rebounder who holds it. Defenders
    track their man with a lag and noise; the "star" (slot 0 of each team) is guarded 2 ft tight and pulls
    the other defenders toward him (gravity). Every game has exactly one 3 s clock gap, inside the setup
    phase of possession ``GAP_POSSESSION(n_possessions)``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import schema  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────────────────────────
HZ = 25
DT = 1.0 / HZ
G_FT = 32.2
COURT_X, COURT_Y = 94.0, 50.0
BASKETS = ((5.25, 25.0), (88.75, 25.0))
PERIOD_S = 720.0
GAP_SECONDS = 3.0
VMAX = 20.0                      # ft/s, players
HELDOUT_GAME = "12.16.2015.PHX.at.GSW"
FETCHED_AT = "2026-01-01T00:00:00Z"
SOURCE = "fixture"
SEASON_2015 = "2015-16"

FIRST_NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet",
               "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango",
               "Uniform", "Victor", "Whiskey", "Xray", "Yankee", "Zulu"]
LAST_NAMES = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
              "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
              "Twentyone", "Twentytwo", "Twentythree", "Twentyfour", "Twentyfive", "Twentysix"]
# synthetic three-letter team codes: none is a real NBA (or basketball-reference) code
TEAM_CODES = ["ABQ", "BOI", "CHT", "DUL", "ERI", "FRS", "GRN", "HLX", "ISL", "JNO", "KNX", "LRD",
              "MOB", "NWK", "OMA", "PEO", "QCY", "RNO", "SPK", "TUL"]
TEAM_NAMES = {"ABQ": "Albuquerque Arrows", "BOI": "Boise Badgers", "CHT": "Chattanooga Comets",
              "DUL": "Duluth Drifters", "ERI": "Erie Eagles", "FRS": "Fresno Foxes", "GRN": "Greenville Giants",
              "HLX": "Halifax Harbors", "ISL": "Islip Islanders", "JNO": "Juneau Jets", "KNX": "Knoxville Knights",
              "LRD": "Laredo Lizards", "MOB": "Mobile Mariners", "NWK": "Newark Nomads", "OMA": "Omaha Owls",
              "PEO": "Peoria Pilots", "QCY": "Quincy Quakes", "RNO": "Reno Rockets", "SPK": "Spokane Spartans",
              "TUL": "Tulsa Titans", "PHX": "Phoenix Suns", "GSW": "Golden State Warriors"}
TEAM_ID_BASE = 1610612700
PLAYER_ID_BASE = 9000
N_ROSTER = 7                     # players per team in a synthetic game (5 on the floor, 2 reserves)
# lineup per period as slots into the 7-man roster; slot 0 is the star, slot 1 the point guard
LINEUPS = {1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3, 5], 3: [0, 1, 2, 3, 4], 4: [0, 1, 2, 6, 4]}
SUBS = {2: (4, 5), 3: (5, 4), 4: (3, 6)}        # period -> (roster index out, roster index in)

ACTION_TYPES = ("ball_screen", "handoff", "off_ball_screen", "isolation", "post_up", "drive", "cut")
ACTION_P = (0.42, 0.10, 0.10, 0.12, 0.08, 0.10, 0.08)
COVERAGES = ("switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over")
P_TURNOVER = 0.12
P_OREB = 0.25
BASE_MAKE = {"layup": 0.55, "mid": 0.36, "three": 0.30}
DEF_DIST_EFFECT = 0.025          # make probability per foot of nearest-defender distance (capped at 8 ft)
STAR_SKILL = 0.08

BOOKS = ("draftkings", "fanduel", "betmgm")
ARCHETYPES = ("guard", "wing", "big")
POSITION_OF = {"guard": "G", "wing": "F", "big": "C"}
DEF_RANGES = ("0-2 Feet - Very Tight", "2-4 Feet - Tight", "4-6 Feet - Open", "6+ Feet - Wide Open")
PLAY_TYPES = ("Isolation", "PRBallHandler", "PRRollman", "Postup", "Spotup", "Handoff", "Cut", "OffScreen",
              "Transition")


def GAP_POSSESSION(n_possessions: int) -> int:
    """The possession index that carries the 3 s clock gap (always the same for a given count)."""
    return max(0, n_possessions // 3)


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0.0)


def _move(start, target, n, vmax=VMAX):
    """Straight-line positions over n frames from start to target (k,2) at min(vmax, needed) speed."""
    start = np.asarray(start, float)
    target = np.asarray(target, float)
    d = target - start
    dist = np.linalg.norm(d, axis=-1)
    speed = np.minimum(vmax, dist / max(n * DT, DT))
    t = np.arange(1, n + 1)[:, None] * DT
    reach = np.minimum(dist[None, :], speed[None, :] * t)
    u = _unit(d)
    return start[None, :, :] + u[None, :, :] * reach[:, :, None]


def _smooth(raw, prev, a=0.18):
    """First-order lag: y[t] = (1-a) y[t-1] + a raw[t], y[-1] = prev. raw (n,k,2), prev (k,2)."""
    from scipy.signal import lfilter
    zi = ((1 - a) * np.asarray(prev, float))[None, :, :]
    y, _ = lfilter([a], [1.0, -(1 - a)], raw, axis=0, zi=zi)
    return y


def _attack(basket):
    """(sign, bx): court x = bx + sign * u where u is the distance from the basket toward half court."""
    return (1.0, basket[0]) if basket[0] < 47 else (-1.0, basket[0])


def _court(basket, u, v):
    sgn, bx = _attack(basket)
    return np.array([bx + sgn * u, 25.0 + v])


def _clock_str(c: float) -> str:
    m = int(c // 60)
    s = c - 60 * m
    return "%d:%04.1f" % (m, s)


def _season_of_date(date: str) -> str:
    y, m = int(date[:4]), int(date[5:7])
    start = y if m >= 7 else y - 1
    return "%d-%02d" % (start, (start + 1) % 100)


def _initial_name(full: str) -> str:
    first, last = full.split(" ", 1)
    return "%s. %s" % (first[0], last)


# ── the game simulator ────────────────────────────────────────────────────────────────────────────
class _Sim:
    """Appends per-phase arrays: positions (n,10,2), ball (n,3), clock (n), shot clock (n)."""

    def __init__(self, rs: np.random.RandomState):
        self.rs = rs
        self.pos = np.zeros((10, 2))
        self.pos_chunks, self.ball_chunks, self.clock_chunks, self.shot_chunks, self.period_chunks = [], [], [], [], []
        self.n = 0                    # frames so far (frame id of the next frame)
        self.clock = PERIOD_S
        self.period = 1
        self.poss_start_clock = PERIOD_S
        self.gap_pending = False
        self.gap_spans = {}           # period -> [[from, to]]

    # -- frames bookkeeping --
    def _append(self, pos, ball, gap_at=None):
        n = len(pos)
        clock = self.clock - np.arange(1, n + 1) * DT
        if gap_at is not None:
            clock[gap_at:] -= GAP_SECONDS
            self.gap_spans.setdefault(self.period, []).append(
                [round(float(clock[gap_at - 1]), 2), round(float(clock[gap_at]), 2)])
        shot = np.maximum(0.0, 24.0 - (self.poss_start_clock - clock))
        self.pos_chunks.append(pos); self.ball_chunks.append(ball); self.clock_chunks.append(clock)
        self.shot_chunks.append(shot); self.period_chunks.append(np.full(n, self.period, dtype=np.int8))
        self.clock = float(clock[-1])
        self.pos = pos[-1].copy()
        self.n += n

    def last_frame(self) -> int:
        return self.n - 1

    def defense_track(self, off_pos, off_idx, def_idx, basket, handler_slot, star_slot, override=None, swap=None):
        """Defender raw targets from offence positions (n,5,2): between the man and the basket, tight on the
        star, with gravity (shading toward the star) and noise; optional per-defender overrides (n,2)."""
        n = len(off_pos)
        b = np.asarray(basket)
        to_basket = _unit(b[None, None, :] - off_pos)
        dist = np.full(5, 5.5)
        dist[handler_slot] = 3.0
        dist[star_slot] = 2.0
        raw = off_pos + to_basket * dist[None, :, None]
        star = off_pos[:, star_slot:star_slot + 1, :]
        shade = 0.12 * (star - raw)
        shade[:, star_slot, :] = 0
        raw = raw + shade
        raw = raw + self.rs.normal(0, 0.35, raw.shape)
        if swap is not None:
            a, c = swap
            raw[:, [a, c], :] = raw[:, [c, a], :]
        if override:
            for slot, path in override.items():
                raw[:, slot, :] = path
        prev = self.pos[def_idx]
        return _smooth(raw, prev)

    def phase(self, n, off_idx, def_idx, off_targets, basket, handler_slot, star_slot, ball_fn,
              override=None, swap=None, gap=False, def_targets=None):
        """One phase of n frames: offence moves straight to off_targets, defence tracks (or moves to
        def_targets when given), the ball follows ball_fn(off_pos, def_pos) -> (n,3)."""
        extra = int(GAP_SECONDS * HZ) if gap else 0
        m = n + extra
        off = _move(self.pos[off_idx], off_targets, m)
        if def_targets is not None:
            dfp = _move(self.pos[def_idx], def_targets, m)
        else:
            dfp = self.defense_track(off, off_idx, def_idx, basket, handler_slot, star_slot, override, swap)
        pos = np.zeros((m, 10, 2))
        pos[:, off_idx, :] = off
        pos[:, def_idx, :] = dfp
        ball = ball_fn(off, dfp)
        gap_at = None
        if gap:
            k = n // 2
            keep = np.r_[np.arange(0, k), np.arange(k + extra, m)]
            pos, ball = pos[keep], ball[keep]
            gap_at = k
        pos[:, :, 0] = np.clip(pos[:, :, 0], 0.5, COURT_X - 0.5)
        pos[:, :, 1] = np.clip(pos[:, :, 1], 0.5, COURT_Y - 0.5)
        self._append(pos, ball, gap_at)


def _dribble_z(n, t0=0.0):
    t = t0 + np.arange(1, n + 1) * DT
    return 2.5 + 1.5 * np.sin(2 * np.pi * 2.5 * t)


def _held_ball(holder_pos, basket, z):
    """Ball at the holder's hands: 0.8 ft toward the basket, given z array."""
    d = _unit(np.asarray(basket)[None, :] - holder_pos) * 0.8
    return np.column_stack([holder_pos + d, z])


def _spots(rs, basket, handler_slot):
    """Five offensive spots in court coordinates: handler at the top, others on wings, corner, dunker."""
    base = [(27.0, 0.0), (22.0, -17.0), (22.0, 17.0), (4.0, -20.0), (5.0, 8.0)]
    order = [handler_slot] + [s for s in range(5) if s != handler_slot]
    out = np.zeros((5, 2))
    for k, slot in enumerate(order):
        u, v = base[k]
        u += rs.uniform(-2, 2); v += rs.uniform(-2, 2)
        out[slot] = _court(basket, u, v)
    return out


def _shot_spot(rs, basket, cls):
    if cls == "layup":
        u, v = rs.uniform(1.5, 3.5), rs.uniform(-2.5, 2.5)
    elif cls == "mid":
        r, ang = rs.uniform(9, 19), rs.uniform(-1.1, 1.1)
        u, v = r * math.cos(ang), r * math.sin(ang)
    else:
        r, ang = rs.uniform(24.5, 27.0), rs.uniform(-0.85, 0.85)
        u, v = r * math.cos(ang), r * math.sin(ang)
    return _court(basket, u, v), (u, v)


def _synth_game_impl(seed, n_possessions, heldout, sportvu_game):
    rs = np.random.RandomState(seed)
    # -- identity --
    if sportvu_game is None:
        if heldout:
            sportvu_game = HELDOUT_GAME
        else:
            away, home = rs.choice(TEAM_CODES, 2, replace=False)
            month = int(rs.choice([11, 12, 1, 2, 3]))
            day = int(rs.randint(1, 29))
            year = 2015 if month >= 10 else 2016
            sportvu_game = "%02d.%02d.%d.%s.at.%s" % (month, day, year, away, home)
    mm, dd, yyyy, away, _at, home = sportvu_game.split(".")
    date = "%s-%s-%s" % (yyyy, mm, dd)
    season = _season_of_date(date)
    gameid = "00215%05d" % (seed % 100000)
    home_id, away_id = TEAM_ID_BASE + (seed % 50) * 2 + 1, TEAM_ID_BASE + (seed % 50) * 2 + 2
    split = "heldout_game" if heldout else "train"
    # rosters: 7 per team; home first. Names are unique within the game.
    names_first = list(FIRST_NAMES)
    rs.shuffle(names_first)
    roster = []
    for t, (tid, code, is_home) in enumerate(((home_id, home, True), (away_id, away, False))):
        for k in range(N_ROSTER):
            i = t * N_ROSTER + k
            roster.append({"player_id": PLAYER_ID_BASE + seed % 1000 * 20 + i, "player": "%s %s" % (names_first[i], LAST_NAMES[i]),
                           "team_id": tid, "team": code, "home": is_home, "jersey": str(k + (0 if is_home else 10))})
    pid = np.array([r["player_id"] for r in roster])
    skill = rs.uniform(-0.06, 0.06, len(roster))
    skill[0] += STAR_SKILL; skill[N_ROSTER] += STAR_SKILL
    # per-period slot -> roster index (0..13)
    slot_roster = {p: [i for i in LINEUPS[p]] + [N_ROSTER + i for i in LINEUPS[p]] for p in range(1, 5)}

    sim = _Sim(rs)
    events, shots, shot_rows, reb_rows, poss_rows, truth_actions, truth_screens = [], [], [], [], [], [], []
    score = [0, 0]                      # away, home
    per_period = [n_possessions // 4 + (1 if p < n_possessions % 4 else 0) for p in range(4)]
    gap_poss = GAP_POSSESSION(n_possessions)
    poss_no = 0
    action_id = 0

    def add_event(kind, team, desc, points=0, frame=None, p1=None, p2=None, etype=0, atype=0):
        if frame is None:
            frame = sim.last_frame()
        clock = float(clocks_of(frame))
        if points:
            score[0 if team == "away" else 1] += points
        events.append({"game": sportvu_game, "event": len(events), "period": sim.period, "clock_s": clock,
                       "team": team, "desc": desc, "points": points, "score": "%d-%d" % (score[0], score[1]),
                       "kind": kind, "frame": int(frame), "frame_clock_diff_s": 0.0,
                       "source": SOURCE, "endpoint": "synth_game", "season": season, "fetched_at": FETCHED_AT,
                       "event_id": len(events), "event_type": etype, "action_type": atype,
                       "player1_id": p1, "player2_id": p2, "player3_id": None,
                       "team_id": None if team is None else (home_id if team == "home" else away_id),
                       "clock_str": _clock_str(clock)})
        return len(events) - 1

    def clocks_of(frame):
        # the clock of a frame id (frames are appended in order; look in the chunks)
        k = frame
        for c in sim.clock_chunks:
            if k < len(c):
                return c[k]
            k -= len(c)
        raise IndexError(frame)

    def ball_of(frame):
        k = frame
        for c in sim.ball_chunks:
            if k < len(c):
                return c[k]
            k -= len(c)
        raise IndexError(frame)

    def pos_of(frame):
        k = frame
        for c in sim.pos_chunks:
            if k < len(c):
                return c[k]
            k -= len(c)
        raise IndexError(frame)

    for period in range(1, 5):
        sim.period = period
        sim.clock = PERIOD_S
        sim.poss_start_clock = PERIOD_S
        sr = slot_roster[period]                   # 10 slots -> roster index
        off_side = "home" if period in (1, 4) else "away"
        # jump-ball-ish placement at the period start
        sim.pos[:5] = np.column_stack([rs.uniform(30, 64, 5), rs.uniform(8, 42, 5)])
        sim.pos[5:] = np.column_stack([rs.uniform(30, 64, 5), rs.uniform(8, 42, 5)])
        period_first_event = len(events)
        if period in SUBS:
            for side in ("home", "away"):
                out_i, in_i = SUBS[period]
                base = 0 if side == "home" else N_ROSTER
                sim.phase(1, list(range(5)), list(range(5, 10)), sim.pos[:5], BASKETS[0], 1, 0,
                          lambda o, d: np.array([[47.0, 25.0, 4.0]]), def_targets=sim.pos[5:])
                add_event("substitution", side, "%s enters the game for %s" % (
                    _initial_name(roster[base + in_i]["player"]), _initial_name(roster[base + out_i]["player"])),
                    p1=int(pid[base + out_i]), p2=int(pid[base + in_i]), etype=8)
        handler_slot = 1                                   # point guard brings it up
        carrier_from_prev = None                           # (slot within team) when a rebound/steal continues
        prev_end_frame = sim.n                             # start_frame of the first possession = first frame
        start_event = period_first_event
        for _k in range(per_period[period - 1]):
            is_home = off_side == "home"
            off_idx = list(range(0, 5)) if is_home else list(range(5, 10))
            def_idx = list(range(5, 10)) if is_home else list(range(0, 5))
            off_tid = home_id if is_home else away_id
            # home attacks the right basket in periods 1 and 2
            basket = BASKETS[1] if (is_home == (period <= 2)) else BASKETS[0]
            star_slot = 0
            handler = carrier_from_prev if carrier_from_prev is not None else 1
            sim.poss_start_clock = sim.clock
            poss_start_frame = prev_end_frame
            gap_here = (poss_no == gap_poss)
            swap = None
            # -- SETUP: bring the ball up --
            spots = _spots(rs, basket, handler)
            n_setup = int(rs.uniform(2.5, 4.0) * HZ)
            sim.phase(n_setup, off_idx, def_idx, spots, basket, handler, star_slot,
                      lambda o, d, h=handler, b=basket: _held_ball(o[:, h], b, _dribble_z(len(o))),
                      gap=gap_here)
            # -- ACTION --
            action = ACTION_TYPES[rs.choice(len(ACTION_TYPES), p=ACTION_P)]
            a_start = sim.n
            screener = None
            coverage = None
            if action == "ball_screen":
                coverage = COVERAGES[rs.randint(len(COVERAGES))]
                screener = 4 if handler != 4 else 3
                swap_after = _ball_screen(sim, rs, off_idx, def_idx, basket, handler, screener, star_slot, coverage)
                swap = swap_after
            elif action == "handoff":
                giver = 4 if handler != 4 else 3
                _handoff(sim, rs, off_idx, def_idx, basket, handler, giver, star_slot)
                screener = giver
                # the receiver of the handoff is the new handler
                handler, screener = _handoff_result(handler, giver)
            elif action == "off_ball_screen":
                cutter = 0 if handler != 0 else 2
                screener = 4 if handler != 4 else 3
                _off_ball_screen(sim, rs, off_idx, def_idx, basket, handler, cutter, screener, star_slot)
                handler = cutter
            elif action == "isolation":
                _isolation(sim, rs, off_idx, def_idx, basket, handler, star_slot)
            elif action == "post_up":
                big = 4 if handler != 4 else 3
                _post_up(sim, rs, off_idx, def_idx, basket, handler, big, star_slot)
                handler = big
            elif action == "drive":
                _drive(sim, rs, off_idx, def_idx, basket, handler, star_slot)
            else:
                cutter = 2 if handler != 2 else 0
                _cut(sim, rs, off_idx, def_idx, basket, handler, cutter, star_slot)
                handler = cutter
            a_end = sim.last_frame()
            truth_actions.append({"sportvu_game": sportvu_game, "possession": poss_no, "action_type": action,
                                  "start_frame": a_start, "end_frame": a_end,
                                  "ball_handler_id": int(pid[sr[off_idx[handler]]]),
                                  "screener_id": None if screener is None else int(pid[sr[off_idx[screener]]]),
                                  "label_source": "synthetic", "labeller_pass": 1})
            if coverage is not None:
                truth_screens.append({"sportvu_game": sportvu_game, "possession": poss_no, "action_id": action_id,
                                      "coverage": coverage,
                                      "on_ball_defender_id": int(pid[sr[def_idx[handler]]]),
                                      "screener_defender_id": int(pid[sr[def_idx[screener]]]),
                                      "label_source": "synthetic", "confidence": "sure"})
            action_id += 1
            # -- PASSES (maybe a turnover) --
            turnover = rs.rand() < P_TURNOVER
            n_pass = int(rs.randint(0, 3)) if not turnover else 1
            last_passer = None
            ended = False
            for _p in range(n_pass):
                receiver = int(rs.choice([s for s in range(5) if s != handler]))
                if turnover:
                    stealer = receiver          # the receiver's defender steps into the lane
                    _pass(sim, rs, off_idx, def_idx, basket, handler, receiver, star_slot, swap, steal=True)
                    steal_frame = sim.last_frame()
                    hname = _initial_name(roster[sr[off_idx[handler]]]["player"])
                    sname = _initial_name(roster[sr[def_idx[stealer]]]["player"])
                    ev = add_event("turnover", off_side, "Turnover by %s (bad pass; steal by %s)" % (hname, sname),
                                   frame=steal_frame, p1=int(pid[sr[off_idx[handler]]]),
                                   p2=int(pid[sr[def_idx[stealer]]]), etype=5, atype=2)
                    poss_rows.append(_poss_row(sportvu_game, poss_no, period, poss_start_frame, steal_frame,
                                               clocks_of(poss_start_frame), clocks_of(steal_frame), start_event, ev,
                                               off_tid, "turnover", 0, gap_here))
                    prev_end_frame = steal_frame
                    start_event = ev + 1
                    carrier_from_prev = stealer
                    ended = True
                    break
                _pass(sim, rs, off_idx, def_idx, basket, handler, receiver, star_slot, swap)
                last_passer = handler
                handler = receiver
                n_hold = int(rs.uniform(0.6, 1.6) * HZ)
                sim.phase(n_hold, off_idx, def_idx, sim.pos[off_idx], basket, handler, star_slot,
                          lambda o, d, h=handler, b=basket: _held_ball(o[:, h], b, _dribble_z(len(o))), swap=swap)
            if ended:
                off_side = "away" if is_home else "home"
                poss_no += 1
                continue
            # -- SHOT (loop for offensive rebounds) --
            shooter = handler
            putback = False
            while True:
                if putback:
                    cls = "layup"
                elif action in ("drive", "cut", "post_up"):
                    cls = "layup" if action != "post_up" else rs.choice(["layup", "mid"])
                else:
                    cls = rs.choice(["layup", "mid", "three"], p=[0.3, 0.3, 0.4])
                spot, (u, v) = _shot_spot(rs, basket, cls)
                rel_frame, apex_frame, out_frame, def_dist = _shoot(sim, rs, off_idx, def_idx, basket, shooter, star_slot, spot, swap, cls)
                dist_ft = math.hypot(u, v)
                p_make = BASE_MAKE[cls] + DEF_DIST_EFFECT * min(def_dist, 8.0) + skill[sr[off_idx[shooter]]]
                made = bool(rs.rand() < float(np.clip(p_make, 0.05, 0.95)))
                pts = 3 if cls == "three" else 2
                sname = _initial_name(roster[sr[off_idx[shooter]]]["player"])
                kind_word = "layup" if cls == "layup" else "jump shot"
                desc = "%s %s %d-pt %s from %d ft" % (sname, "makes" if made else "misses", pts, kind_word, round(dist_ft))
                assist_id = None
                if made and last_passer is not None and not putback:
                    desc += " (assist by %s)" % _initial_name(roster[sr[off_idx[last_passer]]]["player"])
                    assist_id = int(pid[sr[off_idx[last_passer]]])
                ev = add_event("shot_made" if made else "shot_missed", off_side, desc, pts if made else 0,
                               frame=out_frame, p1=int(pid[sr[off_idx[shooter]]]), p2=assist_id,
                               etype=1 if made else 2, atype=5 if cls == "layup" else 1)
                shooter_pid = int(pid[sr[off_idx[shooter]]])
                near_ids = pos_of(rel_frame)
                shot_rows.append({"sportvu_game": sportvu_game, "event": ev, "period": period, "clock_s": clocks_of(out_frame),
                                  "made": made, "pbp_shooter_name": sname, "shooter_id": shooter_pid,
                                  "release_frame": rel_frame, "apex_frame": apex_frame, "outcome_frame": out_frame,
                                  "shooter_at_release_id": shooter_pid, "identity_match": True,
                                  "confidence": 1.0, "confidence_flag": "high"})
                sgn, _bx = _attack(basket)
                shots.append({"game": sportvu_game, "sportvu_game": sportvu_game, "game_id": gameid, "event": ev,
                              "event_id": ev, "period": period, "clock_s": clocks_of(out_frame), "made": made,
                              "loc_x": float(round(10 * v * sgn, 1)), "loc_y": float(round(10 * u, 1)),
                              "court_x": float(spot[0]), "court_y": float(spot[1]),
                              "shot_distance_ft": float(round(dist_ft)), "shot_type": "3PT" if pts == 3 else "2PT",
                              "action_type": {"layup": "Layup Shot", "mid": "Pullup Jump shot", "three": "Jump Shot"}[cls],
                              "zone_basic": {"layup": "Restricted Area", "mid": "Mid-Range", "three": "Above the Break 3"}[cls],
                              "player_id": shooter_pid, "player": roster[sr[off_idx[shooter]]]["player"],
                              "team_id": off_tid, "team": home if is_home else away, "game_date": date,
                              "home_team": home, "away_team": away, "split": split, "heldout": heldout,
                              "season": season, "season_type": "Regular Season", "source": SOURCE,
                              "endpoint": "synth_game", "fetched_at": FETCHED_AT,
                              "release_frame": None, "closest_defender_id": None,
                              "closest_defender_dist_ft": None, "defender_closing_speed_ftps": None})
                if made:
                    poss_rows.append(_poss_row(sportvu_game, poss_no, period, poss_start_frame, out_frame,
                                               clocks_of(poss_start_frame), clocks_of(out_frame), start_event, ev,
                                               off_tid, "shot_made", pts, gap_here))
                    prev_end_frame = out_frame
                    start_event = ev + 1
                    # the ball drops through the net and the other team's point guard picks it up
                    new_off_idx, new_def_idx = def_idx, off_idx
                    _after_make(sim, rs, new_off_idx, new_def_idx, basket, 1)
                    carrier_from_prev = None
                    break
                # miss: rebound
                offensive = rs.rand() < P_OREB
                reb_team_idx = off_idx if offensive else def_idx
                rebounder = int(rs.choice([3, 4]))
                reb_frame = _rebound(sim, rs, off_idx, def_idx, basket, rebounder, offensive, star_slot, swap)
                rname = _initial_name(roster[sr[reb_team_idx[rebounder]]]["player"])
                rev = add_event("rebound", off_side if offensive else ("away" if is_home else "home"),
                                "%s rebound by %s" % ("Offensive" if offensive else "Defensive", rname),
                                frame=reb_frame, p1=int(pid[sr[reb_team_idx[rebounder]]]), etype=4)
                reb_rows.append({"sportvu_game": sportvu_game, "event": ev, "rebound_event": rev, "period": period,
                                 "clock_s": clocks_of(out_frame), "rim_frame": out_frame, "rebound_frame": reb_frame,
                                 "rebounder_id": int(pid[sr[reb_team_idx[rebounder]]]), "pbp_rebounder_name": rname,
                                 "pbp_rebounder_id": int(pid[sr[reb_team_idx[rebounder]]]), "identity_match": True,
                                 "confidence": 1.0, "confidence_flag": "high"})
                if offensive:
                    putback = True
                    shooter = rebounder
                    last_passer = None
                    sim.poss_start_clock = sim.clock           # shot clock reset
                    n_hold = int(rs.uniform(0.4, 0.8) * HZ)
                    sim.phase(n_hold, off_idx, def_idx, sim.pos[off_idx], basket, shooter, star_slot,
                              lambda o, d, h=shooter, b=basket: _held_ball(o[:, h], b, np.full(len(o), 5.0)), swap=swap)
                    continue
                poss_rows.append(_poss_row(sportvu_game, poss_no, period, poss_start_frame, reb_frame,
                                           clocks_of(poss_start_frame), clocks_of(reb_frame), start_event, rev,
                                           off_tid, "shot_missed", 0, gap_here))
                prev_end_frame = reb_frame
                start_event = rev + 1
                carrier_from_prev = rebounder
                break
            off_side = "away" if is_home else "home"
            poss_no += 1
        # a short tail after the last possession of the period
        sim.phase(int(0.6 * HZ), list(range(5)), list(range(5, 10)), sim.pos[:5], BASKETS[0], 1, 0,
                  lambda o, d, b=ball_of(sim.last_frame()): np.tile(b, (len(o), 1)), def_targets=sim.pos[5:])

    # -- assemble the frames table --
    pos = np.concatenate(sim.pos_chunks)                       # (N,10,2)
    ball = np.concatenate(sim.ball_chunks)
    clock = np.concatenate(sim.clock_chunks)
    shotc = np.concatenate(sim.shot_chunks)
    periods = np.concatenate(sim.period_chunks)
    N = len(pos)
    frame_ids = np.arange(N, dtype=np.int32)
    slot_pid = np.zeros((N, 10), dtype=np.int64)
    for p in range(1, 5):
        m = periods == p
        slot_pid[m] = pid[slot_roster[p]]
    pid_to_row = {int(r["player_id"]): r for r in roster}
    flat_pid = slot_pid.ravel()
    rep = lambda a: np.repeat(a, 10)   # noqa: E731
    name_of = {int(r["player_id"]): r["player"] for r in roster}
    jersey_of = {int(r["player_id"]): r["jersey"] for r in roster}
    ts0 = 1450000000000
    ts = ts0 + np.cumsum(np.r_[0, np.round((clock[:-1] - clock[1:]) * 1000).astype(np.int64)
                               + np.where(np.diff(periods) != 0, 900000, 0)])
    ts = ts + np.where(np.diff(np.r_[periods[0], periods]) != 0, 0, 0)
    ts = np.abs(ts)
    frames = pd.DataFrame({
        "game": sportvu_game, "gameid": gameid, "date": date,
        "period": rep(periods).astype(np.int8), "frame": rep(frame_ids).astype(np.int32),
        "ts_ms": rep(ts).astype(np.int64),
        "game_clock_s": rep(clock).astype(np.float32), "shot_clock_s": rep(shotc).astype(np.float32),
        "n_players": np.full(N * 10, 10, dtype=np.int8),
        "ball_x": rep(ball[:, 0]).astype(np.float32), "ball_y": rep(ball[:, 1]).astype(np.float32),
        "ball_z": rep(ball[:, 2]).astype(np.float32),
        "team_id": np.where(np.tile(np.arange(10) < 5, N), home_id, away_id).astype(np.int32),
        "team": np.where(np.tile(np.arange(10) < 5, N), home, away),
        "home": np.tile(np.arange(10) < 5, N),
        "player_id": flat_pid.astype(np.int32),
        "player": pd.Series(flat_pid).map(name_of).to_numpy(),
        "jersey": pd.Series(flat_pid).map(jersey_of).to_numpy(),
        "x": pos[:, :, 0].ravel().astype(np.float32), "y": pos[:, :, 1].ravel().astype(np.float32),
    })
    events_df = _events_frame(events)
    shots_df = _shots_frame(shots)
    rosters_df = schema.cast(pd.DataFrame([{
        "sportvu_game": sportvu_game, "player_id": r["player_id"], "player": r["player"], "team_id": r["team_id"],
        "team": r["team"], "home": r["home"], "jersey": r["jersey"],
        "bref_slug": (r["player"].split()[1][:5] + r["player"].split()[0][:2]).lower() + "01",
        "bref_name": r["player"], "pbp_name": _initial_name(r["player"]), "match_method": "exact"}
        for r in roster]), "rosters")
    shot_frames_df = schema.cast(pd.DataFrame(shot_rows, columns=schema.columns("shot_frames")), "shot_frames")
    rebound_df = schema.cast(pd.DataFrame(reb_rows, columns=schema.columns("rebound_frames")), "rebound_frames")
    poss_df = schema.cast(pd.DataFrame(poss_rows, columns=schema.columns("possessions")), "possessions")
    states_df = _states_1hz(sportvu_game, poss_df, pos, ball, clock, shotc, periods, slot_pid, home_id)
    gaps_by_q = {str(p): sim.gap_spans.get(p, []) for p in range(1, 5)}
    gap_seconds = round(sum(a - b for spans in sim.gap_spans.values() for a, b in spans), 1)
    n_gaps = sum(len(v) for v in sim.gap_spans.values())
    manifest_entry = {
        "game": sportvu_game, "date": date, "away": away, "home": home, "split": split, "heldout": heldout,
        "status": "ok", "error": None, "source_bytes": 0, "moments": int(N), "frames_rows": int(len(frames)),
        "gaps_over_2s": {"count": n_gaps, "seconds": gap_seconds, "by_quarter": gaps_by_q},
        "pbp_joined": True,
        "pbp": {"code": "%s%s%s0%s" % (yyyy, mm, dd, home), "events": int(len(events_df)),
                "matched": int((events_df["frame"] >= 0).sum()), "error": None},
        "parquet": ("heldout/" if heldout else "") + sportvu_game + ".parquet", "parquet_bytes": 0,
        "pbp_parquet": ("heldout/" if heldout else "") + sportvu_game + "_pbp.parquet", "seconds": 0.0,
        "written": FETCHED_AT}
    truth = {"actions": truth_actions, "screens": truth_screens,
             "star_player_ids": [int(pid[0]), int(pid[N_ROSTER])],
             "skill": {int(pid[i]): float(skill[i]) for i in range(len(roster))},
             "gap_possession": gap_poss, "gap_spans": gaps_by_q,
             "make_model": "p = BASE_MAKE[class] + 0.025 * min(nearest_def_dist_ft, 8) + skill[shooter]"}
    meta = {"sportvu_game": sportvu_game, "gameid": gameid, "date": date, "season": season, "home": home, "away": away,
            "home_team_id": home_id, "away_team_id": away_id, "heldout": heldout, "split": split,
            "final_score": "%d-%d" % (score[0], score[1]), "seed": seed, "n_possessions": n_possessions}
    return {"frames": frames, "events": events_df, "shots": shots_df, "rosters": rosters_df,
            "shot_frames": shot_frames_df, "rebound_frames": rebound_df, "possessions": poss_df,
            "states_1hz": states_df, "meta": meta, "truth": truth, "manifest_entry": manifest_entry}


def _poss_row(game, no, period, sf, ef, sc, ec, se, ee, tid, kind, pts, gap):
    return {"sportvu_game": game, "possession": no, "period": period, "start_frame": int(sf), "end_frame": int(ef),
            "start_clock_s": float(sc), "end_clock_s": float(ec), "start_event": int(se), "end_event": int(ee),
            "offense_team_id": int(tid), "end_kind": kind, "points": int(pts), "gap": bool(gap),
            "n_frames": int(ef - sf + 1)}


# ── phases ────────────────────────────────────────────────────────────────────────────────────────
def _held(h, basket, z=None):
    def fn(o, d):
        zz = np.full(len(o), 5.0) if z is None else z(len(o))
        return _held_ball(o[:, h], basket, zz)
    return fn


def _ball_screen(sim, rs, off_idx, def_idx, basket, handler, screener, star_slot, coverage):
    """Screen set for 1 s (screener within 4 ft of the handler's defender), then the handler drives and the
    two defenders move as the coverage says. Returns the assignment swap (a, b) for a switch, else None."""
    cur = sim.pos[off_idx]
    targets = cur.copy()
    h0 = _court(basket, 24.0, 10.0)
    targets[handler] = h0
    d1 = h0 + _unit(np.asarray(basket) - h0) * 3.0
    s0 = d1 + np.array([0.0, -2.5])
    targets[screener] = s0
    n = int(1.0 * HZ)
    sim.phase(n, off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))
    # the screen is set: hold 1.0 s with everyone still
    n_hold = int(1.0 * HZ)
    override = {screener: np.tile(s0 + _unit(np.asarray(basket) - s0) * 2.5, (n_hold, 1))}
    sim.phase(n_hold, off_idx, def_idx, sim.pos[off_idx], basket, handler, star_slot, _held(handler, basket, _dribble_z),
              override=override)
    # the drive
    n_drive = int(1.2 * HZ)
    if coverage == "ice":
        way, end = _court(basket, 18.0, 16.0), _court(basket, 10.0, 18.0)
    else:
        way, end = _court(basket, 22.5, 4.5), _court(basket, 13.0, -2.0)
    half = n_drive // 2
    hpath = np.concatenate([_move(sim.pos[off_idx][handler][None], way[None], half)[:, 0],
                            _move(way[None], end[None], n_drive - half)[:, 0]])
    roll = _court(basket, 5.0, 2.0)
    targets = sim.pos[off_idx].copy()
    targets[screener] = roll if coverage != "switch" else s0
    b = np.asarray(basket)
    over_way = s0 + np.array([0.0, 2.0]) + _unit(s0 - b) * 2.0
    under_way = s0 + _unit(b - s0) * 2.5
    trail = hpath + _unit(hpath - b[None, :]) * 3.0
    front = hpath + _unit(b[None, :] - hpath) * 3.0
    ovr = {}
    swap = None
    d1_pos = sim.pos[def_idx][handler]
    d2_pos = sim.pos[def_idx][screener]
    if coverage == "over":
        ovr[handler] = np.concatenate([_move(d1_pos[None], over_way[None], half)[:, 0], trail[half:]])
    elif coverage == "under":
        ovr[handler] = np.concatenate([_move(d1_pos[None], under_way[None], half)[:, 0], front[half:]])
    elif coverage == "switch":
        ovr[handler] = np.tile(s0 + _unit(b - s0) * 2.0, (n_drive, 1))
        ovr[screener] = front
        swap = (handler, screener)
    elif coverage == "drop":
        ovr[handler] = np.concatenate([_move(d1_pos[None], over_way[None], half)[:, 0], trail[half:]])
        ovr[screener] = _move(d2_pos[None], _court(basket, 7.0, -1.0)[None], n_drive)[:, 0]
    elif coverage == "hedge_show":
        show = hpath[: n_drive // 2] + _unit(b[None, :] - hpath[: n_drive // 2]) * 2.0
        ovr[handler] = np.concatenate([_move(d1_pos[None], over_way[None], half)[:, 0], trail[half:]])
        ovr[screener] = np.concatenate([show, _move(show[-1][None], (roll + _unit(b - roll) * 2.0)[None], n_drive - half)[:, 0]])
    elif coverage == "blitz_trap":
        ovr[handler] = hpath + np.array([0.0, 2.5])
        ovr[screener] = hpath + np.array([0.0, -2.5])
    else:  # ice: the on-ball defender jumps to the screen side of the handler, the big drops baseline side
        ovr[handler] = hpath + np.array([0.0, -3.0])
        ovr[screener] = _move(d2_pos[None], _court(basket, 7.0, 12.0)[None], n_drive)[:, 0]
    off_path = _move(sim.pos[off_idx], targets, n_drive)
    off_path[:, handler, :] = hpath
    ball = _held_ball(hpath, basket, _dribble_z(n_drive))
    dfp = sim.defense_track(off_path, off_idx, def_idx, basket, handler, star_slot, ovr)
    pos = np.zeros((n_drive, 10, 2))
    pos[:, off_idx, :] = off_path
    pos[:, def_idx, :] = dfp
    sim._append(pos, ball)
    # recover 0.8 s
    sim.phase(int(0.8 * HZ), off_idx, def_idx, sim.pos[off_idx], basket, handler, star_slot,
              _held(handler, basket, _dribble_z), swap=swap)
    return swap


def _handoff(sim, rs, off_idx, def_idx, basket, handler, giver, star_slot):
    """The handler passes to the big at the elbow, who hands the ball back to a cutting handler."""
    elbow = _court(basket, 16.0, 8.0)
    targets = sim.pos[off_idx].copy()
    targets[giver] = elbow
    sim.phase(int(1.0 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))
    _pass(sim, rs, off_idx, def_idx, basket, handler, giver, star_slot, None)
    # the handler runs past the giver and takes the ball
    targets = sim.pos[off_idx].copy()
    targets[handler] = elbow + np.array([0.0, -1.5])
    sim.phase(int(0.8 * HZ), off_idx, def_idx, targets, basket, giver, star_slot, _held(giver, basket))
    targets[handler] = _court(basket, 12.0, -2.0)
    sim.phase(int(0.6 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))


def _handoff_result(handler, giver):
    return handler, giver


def _off_ball_screen(sim, rs, off_idx, def_idx, basket, handler, cutter, screener, star_slot):
    """The screener stands on the cutter's defender for 1 s; the cutter runs off it and gets the pass."""
    c0 = _court(basket, 6.0, -18.0)
    targets = sim.pos[off_idx].copy()
    targets[cutter] = c0
    b = np.asarray(basket)
    d_c = c0 + _unit(b - c0) * 3.0
    s0 = d_c + np.array([3.0, 0.0]) * (-1.0 if basket[0] > 47 else 1.0)
    targets[screener] = s0
    sim.phase(int(1.0 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))
    n_hold = int(1.0 * HZ)
    sim.phase(n_hold, off_idx, def_idx, sim.pos[off_idx], basket, handler, star_slot, _held(handler, basket, _dribble_z),
              override={screener: np.tile(s0 + _unit(b - s0) * 2.0, (n_hold, 1))})
    targets = sim.pos[off_idx].copy()
    targets[cutter] = _court(basket, 20.0, -14.0)
    sim.phase(int(1.0 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))
    _pass(sim, rs, off_idx, def_idx, basket, handler, cutter, star_slot, None)


def _isolation(sim, rs, off_idx, def_idx, basket, handler, star_slot):
    """Everyone else clears far from the handler, who holds for 2.5 s then attacks."""
    targets = sim.pos[off_idx].copy()
    far = [(4.0, -21.0), (4.0, 21.0), (24.0, -20.0), (24.0, 20.0)]
    k = 0
    for s in range(5):
        if s != handler:
            targets[s] = _court(basket, *far[k]); k += 1
    targets[handler] = _court(basket, 22.0, 0.0)
    sim.phase(int(2.5 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))


def _post_up(sim, rs, off_idx, def_idx, basket, handler, big, star_slot):
    targets = sim.pos[off_idx].copy()
    targets[big] = _court(basket, 8.0, -7.0)
    sim.phase(int(1.0 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))
    _pass(sim, rs, off_idx, def_idx, basket, handler, big, star_slot, None)
    sim.phase(int(2.0 * HZ), off_idx, def_idx, sim.pos[off_idx], basket, big, star_slot, _held(big, basket, _dribble_z))


def _drive(sim, rs, off_idx, def_idx, basket, handler, star_slot):
    targets = sim.pos[off_idx].copy()
    targets[handler] = _court(basket, 6.0, rs.uniform(-4, 4))
    sim.phase(int(1.5 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))


def _cut(sim, rs, off_idx, def_idx, basket, handler, cutter, star_slot):
    targets = sim.pos[off_idx].copy()
    targets[cutter] = _court(basket, 4.0, rs.uniform(-5, 5))
    sim.phase(int(1.2 * HZ), off_idx, def_idx, targets, basket, handler, star_slot, _held(handler, basket, _dribble_z))
    _pass(sim, rs, off_idx, def_idx, basket, handler, cutter, star_slot, None)


def _pass(sim, rs, off_idx, def_idx, basket, passer, receiver, star_slot, swap, steal=False):
    """Ball flight from the passer to the receiver (or to the receiver's defender who steps into the lane)."""
    start = sim.pos[off_idx][passer]
    if steal:
        target_player = sim.pos[def_idx][receiver]
        mid = 0.5 * (start + sim.pos[off_idx][receiver])
        dist = np.linalg.norm(mid - start)
    else:
        dist = np.linalg.norm(sim.pos[off_idx][receiver] - start)
    n = max(int(0.4 * HZ), min(int(0.7 * HZ), int(dist / 30.0 * HZ)))
    off_t = sim.pos[off_idx].copy()
    ovr = None
    if steal:
        ovr = {receiver: _move(target_player[None], mid[None], n)[:, 0]}

    def ball_fn(o, d):
        end = d[:, receiver] if steal else o[:, receiver]
        t = np.arange(1, n + 1) / n
        xy = start[None, :] + (end - start[None, :]) * t[:, None]
        z = 5.0 + 2.0 * np.sin(np.pi * t)
        return np.column_stack([xy, z])

    sim.phase(n, off_idx, def_idx, off_t, basket, passer, star_slot, ball_fn, override=ovr, swap=swap)


def _shoot(sim, rs, off_idx, def_idx, basket, shooter, star_slot, spot, swap, cls):
    """Move the shooter to the spot, gather (ball 4 -> 7.5 ft over 0.32 s), release, parabola to the rim.
    Returns (release_frame, apex_frame, outcome_frame, nearest defender distance at release)."""
    targets = sim.pos[off_idx].copy()
    targets[shooter] = spot
    dist = np.linalg.norm(spot - sim.pos[off_idx][shooter])
    n_go = max(int(0.3 * HZ), int(dist / 12.0 * HZ))
    sim.phase(n_go, off_idx, def_idx, targets, basket, shooter, star_slot, _held(shooter, basket, _dribble_z), swap=swap)
    # gather: 8 frames, z rising 4 -> 7.5, shooter still
    n_g = 8
    z = np.linspace(4.0, 7.5, n_g)
    sim.phase(n_g, off_idx, def_idx, sim.pos[off_idx], basket, shooter, star_slot,
              lambda o, d: _held_ball(o[:, shooter], basket, z), swap=swap)
    release_frame = sim.last_frame()
    rel_ball = sim.ball_chunks[-1][-1]
    dpos = sim.pos[def_idx]
    def_dist = float(np.min(np.linalg.norm(dpos - sim.pos[off_idx][shooter][None, :], axis=1)))
    # flight to the rim
    b = np.array([basket[0], basket[1]])
    T = 0.9 + 0.3 * (cls == "three") + 0.15 * (cls == "mid")
    n_f = int(round(T * HZ)) + 1           # one frame past the rim so the ball falls below 10 ft
    t = np.arange(1, n_f + 1) * DT
    vz = (10.0 - rel_ball[2] + 0.5 * G_FT * T * T) / T
    zz = rel_ball[2] + vz * t - 0.5 * G_FT * t * t
    xy = rel_ball[None, :2] + (b - rel_ball[:2])[None, :] * np.minimum(t / T, 1.0)[:, None]
    flight = np.column_stack([xy, zz])
    # shooter lands, rebounders crash: slots 3 and 4 of both teams move toward the rim side
    targets = sim.pos[off_idx].copy()
    sim.phase(n_f, off_idx, def_idx, targets, basket, shooter, star_slot, lambda o, d: flight, swap=swap)
    apex_frame = release_frame + 1 + int(np.argmax(zz))
    outcome_frame = sim.last_frame()
    return release_frame, apex_frame, outcome_frame, def_dist


def _after_make(sim, rs, new_off_idx, new_def_idx, basket, handler):
    """The ball drops through the net (0.4 s) and rests on the baseline until the new handler reaches it."""
    b = np.array(basket)
    n_drop = int(0.4 * HZ)
    zz = np.linspace(9.5, 0.3, n_drop)
    rest = b + np.array([-2.0 if basket[0] < 47 else 2.0, 1.0])
    drop = np.column_stack([np.tile(b, (n_drop, 1)), zz])
    sim.phase(n_drop, new_off_idx, new_def_idx, sim.pos[new_off_idx], basket, handler, 0, lambda o, d: drop,
              def_targets=sim.pos[new_def_idx])
    # the new offence turns around: targets near the other end for everyone but the handler
    d = np.linalg.norm(sim.pos[new_off_idx][handler] - rest)
    n_pick = max(int(0.5 * HZ), int(d / (VMAX * 0.9) * HZ))
    targets = sim.pos[new_off_idx].copy()
    targets[handler] = rest
    sgn = -1.0 if basket[0] < 47 else 1.0
    pick = np.column_stack([np.tile(rest, (n_pick, 1)), np.full(n_pick, 0.3)])
    sim.phase(n_pick, new_off_idx, new_def_idx, targets, basket, handler, 0, lambda o, d: pick,
              def_targets=sim.pos[new_def_idx] + np.array([sgn * -8.0, 0.0]))


def _rebound(sim, rs, off_idx, def_idx, basket, rebounder, offensive, star_slot, swap):
    """The ball bounces off the rim to a landing point where the rebounder waits; everyone else is kept
    3 ft away. Returns the rebound frame (first frame with ball z under 7 ft, in the rebounder's hands)."""
    b = np.array(basket)
    sgn, _ = _attack(basket)
    ang = rs.uniform(-1.0, 1.0)
    r = rs.uniform(4.0, 8.0)
    land = b + np.array([sgn * r * math.cos(ang), r * math.sin(ang)])
    team_idx = off_idx if offensive else def_idx
    other_idx = def_idx if offensive else off_idx
    n = int(rs.uniform(0.6, 0.9) * HZ)
    t = np.arange(1, n + 1) / n
    xy = b[None, :] + (land - b)[None, :] * t[:, None]
    zz = 10.0 + 6.0 * np.sin(np.pi * t) * (1 - t) ** 0.3 - 3.5 * t      # peaks ~ 12 then down to 6.5
    zz[-1] = 6.5
    zz[:-1] = np.maximum(zz[:-1], 7.2)
    ball = np.column_stack([xy, zz])
    targets_team = sim.pos[team_idx].copy()
    targets_team[rebounder] = land
    targets_other = sim.pos[other_idx].copy()
    for arr, skip in ((targets_team, rebounder), (targets_other, None)):
        for s in range(5):
            if s == skip:
                continue
            v = arr[s] - land
            dd = np.linalg.norm(v)
            if dd < 3.5:
                arr[s] = land + (_unit(v) if dd > 1e-6 else np.array([sgn, 0.0])) * 3.5
    pos = np.zeros((n, 10, 2))
    pos[:, team_idx, :] = _move(sim.pos[team_idx], targets_team, n, vmax=30.0)
    pos[:, other_idx, :] = _move(sim.pos[other_idx], targets_other, n, vmax=30.0)
    sim._append(pos, ball)
    return sim.last_frame()


# ── derived tables ────────────────────────────────────────────────────────────────────────────────
def _events_frame(events: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(events)
    return df.astype({"event": "int32", "period": "int8", "clock_s": "float32", "points": "int8", "frame": "int32",
                      "frame_clock_diff_s": "float32", "event_id": "Int32", "event_type": "Int16",
                      "action_type": "Int16", "player1_id": "Int32", "player2_id": "Int32", "player3_id": "Int32",
                      "team_id": "Int32"})


def _shots_frame(shots: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(shots)
    return df.astype({"event": "int32", "event_id": "Int32", "period": "int8", "clock_s": "float32", "made": "bool",
                      "loc_x": "float32", "loc_y": "float32", "court_x": "float32", "court_y": "float32",
                      "shot_distance_ft": "float32", "player_id": "int32", "team_id": "int32", "heldout": "bool",
                      "release_frame": "Int32", "closest_defender_id": "Int32",
                      "closest_defender_dist_ft": "float32", "defender_closing_speed_ftps": "float32"})


def _states_1hz(game, poss, pos, ball, clock, shotc, periods, slot_pid, home_id):
    rows = []
    for p in poss.itertuples(index=False):
        sf, ef = int(p.start_frame), int(p.end_frame)
        c0 = clock[sf]
        span = np.arange(sf, ef + 1)
        elapsed = c0 - clock[span]
        total = float(elapsed[-1])
        is_home_off = int(p.offense_team_id) == home_id
        off_slots = np.arange(0, 5) if is_home_off else np.arange(5, 10)
        def_slots = np.arange(5, 10) if is_home_off else np.arange(0, 5)
        s = 0
        while s <= total + 1e-6:
            f = int(span[np.argmin(np.abs(elapsed - s))])
            oid, did = slot_pid[f, off_slots], slot_pid[f, def_slots]
            oo, dd = np.argsort(oid), np.argsort(did)
            opos, dpos = pos[f, off_slots][oo], pos[f, def_slots][dd]
            bxy = ball[f, :2]
            hd = np.linalg.norm(opos - bxy[None, :], axis=1)
            h = int(np.argmin(hd))
            nd = float(np.min(np.linalg.norm(dpos - opos[h][None, :], axis=1)))
            row = {"sportvu_game": game, "possession": int(p.possession), "frame": f, "period": int(periods[f]),
                   "game_clock_s": float(clock[f]), "shot_clock_s": float(shotc[f]),
                   "elapsed_in_possession_s": float(clock[sf] - clock[f]), "offense_team_id": int(p.offense_team_id),
                   "handler_id": int(oid[oo][h]), "ball_x": float(ball[f, 0]), "ball_y": float(ball[f, 1]),
                   "ball_z": float(ball[f, 2])}
            for i in range(5):
                row["off%d_id" % (i + 1)] = int(oid[oo][i])
            for i in range(5):
                row["off%d_x" % (i + 1)] = float(opos[i, 0]); row["off%d_y" % (i + 1)] = float(opos[i, 1])
            for i in range(5):
                row["def%d_id" % (i + 1)] = int(did[dd][i])
            for i in range(5):
                row["def%d_x" % (i + 1)] = float(dpos[i, 0]); row["def%d_y" % (i + 1)] = float(dpos[i, 1])
            row["handler_nearest_def_dist_ft"] = nd
            row["points_to_end"] = int(p.points)
            row["possession_end_kind"] = str(p.end_kind)
            rows.append(row)
            s += 1
    return schema.cast(pd.DataFrame(rows, columns=schema.columns("states_1hz")), "states_1hz")


# ── public API ────────────────────────────────────────────────────────────────────────────────────
def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game: str | None = None) -> dict:
    """One synthetic game: frames, events, shots, rosters, shot_frames, rebound_frames, possessions,
    states_1hz, plus meta, truth and manifest_entry. Deterministic in (seed, n_possessions, heldout, id)."""
    if n_possessions < 4:
        raise ValueError("n_possessions must be at least 4 (one per period)")
    return _synth_game_impl(int(seed), int(n_possessions), bool(heldout), sportvu_game)


def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list[dict]:
    """n_games synthetic games with distinct ids. With heldout_every = k > 0, every k-th game (the k-th,
    2k-th, ...) is held out: the first of them carries the real held-out id 12.16.2015.PHX.at.GSW, any
    later one a synthetic id with heldout True (ids never repeat within a season)."""
    rs = np.random.RandomState(seed * 7919 + 17)
    games, used = [], set()
    first_heldout_done = False
    for k in range(n_games):
        gseed = seed * 1000 + k
        held = heldout_every > 0 and (k + 1) % heldout_every == 0
        gid = None
        if held and not first_heldout_done:
            gid = HELDOUT_GAME
            first_heldout_done = True
        else:
            while gid is None or gid in used:
                away, home = rs.choice(TEAM_CODES, 2, replace=False)
                month = int(rs.choice([11, 12, 1, 2, 3]))
                day = int(rs.randint(1, 29))
                year = 2015 if month >= 10 else 2016
                gid = "%02d.%02d.%d.%s.at.%s" % (month, day, year, away, home)
        used.add(gid)
        games.append(synth_game(gseed, n_possessions=n_possessions, heldout=held, sportvu_game=gid))
    return games


def synth_aggregates(seed: int, n_players: int = 60, seasons=("2015-16", "2016-17")):
    """Long aggregates (section 11) for n_players over the seasons, with a hidden 3-archetype style factor
    (guard, wing, big) that drives every metric; plus a positions frame aligned to the archetypes."""
    rs = np.random.RandomState(seed * 31 + 5)
    arche = rs.randint(0, 3, n_players)
    base_pid = 200000 + (seed % 1000) * 100
    pids = base_pid + np.arange(n_players)
    names = ["%s %s" % (FIRST_NAMES[i % len(FIRST_NAMES)], LAST_NAMES[(i // len(FIRST_NAMES)) % len(LAST_NAMES)])
             for i in range(n_players)]
    team_of = TEAM_ID_BASE + (np.arange(n_players) % 4)
    rows, pos_rows = [], []
    # archetype profiles: guard, wing, big
    prof = {
        "drives": (12.0, 6.0, 2.0), "touches": (75.0, 50.0, 40.0), "avg_speed": (4.6, 4.3, 3.9),
        "dist_miles": (2.6, 2.4, 2.0), "time_of_poss": (5.5, 2.5, 1.2), "catch_shoot_fga": (3.0, 4.0, 1.0),
        "pull_up_fga": (5.0, 2.5, 0.5), "reb": (3.0, 5.5, 9.5), "contested_reb": (0.8, 1.6, 3.4),
        "screen_assists": (0.3, 1.0, 3.5), "deflections": (2.4, 1.8, 1.2), "contested_shots": (6.0, 8.0, 11.0),
        "box_outs": (1.0, 2.5, 5.0), "loose_balls_recovered": (0.7, 0.6, 0.5), "charges_drawn": (0.1, 0.15, 0.2),
    }
    fg_by_range = {  # fg_pct by closest defender range: guards fall off less when tight, bigs are rim-bound
        "guard": (0.38, 0.42, 0.46, 0.50), "wing": (0.40, 0.44, 0.47, 0.49), "big": (0.52, 0.56, 0.58, 0.58)}
    ppp_by_type = {
        "guard": {"Isolation": 0.92, "PRBallHandler": 0.95, "PRRollman": 0.9, "Postup": 0.75, "Spotup": 1.0,
                  "Handoff": 0.9, "Cut": 1.1, "OffScreen": 0.95, "Transition": 1.12},
        "wing": {"Isolation": 0.85, "PRBallHandler": 0.85, "PRRollman": 1.0, "Postup": 0.85, "Spotup": 1.02,
                 "Handoff": 0.9, "Cut": 1.2, "OffScreen": 1.0, "Transition": 1.1},
        "big": {"Isolation": 0.7, "PRBallHandler": 0.6, "PRRollman": 1.15, "Postup": 0.95, "Spotup": 0.9,
                "Handoff": 0.8, "Cut": 1.3, "OffScreen": 0.8, "Transition": 1.05}}
    for season in seasons:
        for i in range(n_players):
            a = ARCHETYPES[arche[i]]
            pos_rows.append({"player_id": int(pids[i]), "season": season, "position": POSITION_OF[a]})
            common = {"player_id": int(pids[i]), "player_name": names[i], "team_id": int(team_of[i]),
                      "team_abbreviation": TEAM_CODES[i % 4], "season": season, "season_type": "Regular Season",
                      "split_name": None, "split_value": None, "pt_measure_type": None, "player_or_team": "Player",
                      "play_type": None, "type_grouping": None, "measure_type": None, "heldout": None,
                      "source": SOURCE, "fetched_at": FETCHED_AT}
            gp = int(rs.randint(50, 82))
            mins = float(rs.uniform(18, 36))

            def add(endpoint, metric, value, **extra):
                rows.append({**common, "endpoint": endpoint, "metric": metric, "value": float(value), **extra})

            for metric, levels in prof.items():
                mu = levels[arche[i]]
                val = max(0.0, mu * rs.lognormal(0, 0.18))
                if metric in ("drives", "touches", "time_of_poss"):
                    add("leaguedashptstats", metric, val, pt_measure_type="Drives" if metric == "drives" else "Possessions")
                elif metric in ("avg_speed", "dist_miles"):
                    add("leaguedashptstats", metric, val, pt_measure_type="SpeedDistance")
                elif metric in ("catch_shoot_fga", "pull_up_fga"):
                    add("leaguedashptstats", metric, val,
                        pt_measure_type="CatchShoot" if metric.startswith("catch") else "PullUpShot")
                elif metric in ("reb", "contested_reb"):
                    add("leaguedashptstats", metric, val, pt_measure_type="Rebounding")
                else:
                    add("leaguehustlestatsplayer", metric, val)
            add("leaguedashptstats", "gp", gp, pt_measure_type="SpeedDistance")
            add("leaguedashptstats", "min", mins, pt_measure_type="SpeedDistance")
            for j, rng in enumerate(DEF_RANGES):
                fga = max(0.2, (6.0 - 1.2 * j) * rs.lognormal(0, 0.2))
                pct = float(np.clip(fg_by_range[a][j] + rs.normal(0, 0.03), 0.2, 0.8))
                add("leaguedashplayerptshot", "fga", fga, split_name="closest_defender", split_value=rng)
                add("leaguedashplayerptshot", "fg_pct", pct, split_name="closest_defender", split_value=rng)
                add("leaguedashplayerptshot", "efg_pct", min(0.9, pct * (1.1 if a != "big" else 1.0)),
                    split_name="closest_defender", split_value=rng)
            for pt in PLAY_TYPES:
                poss = max(0.1, {"guard": 3.0, "wing": 2.0, "big": 1.5}[a] * rs.lognormal(0, 0.4))
                ppp = float(np.clip(ppp_by_type[a][pt] + rs.normal(0, 0.06), 0.3, 1.8))
                add("synergyplaytypes", "ppp", ppp, play_type=pt, type_grouping="offensive")
                add("synergyplaytypes", "poss", poss, play_type=pt, type_grouping="offensive")
                add("synergyplaytypes", "fg_pct", float(np.clip(ppp * 0.45, 0.2, 0.75)), play_type=pt, type_grouping="offensive")
    agg = pd.DataFrame(rows)
    agg = agg[["player_id", "player_name", "team_id", "team_abbreviation", "season", "season_type", "endpoint",
               "split_name", "split_value", "pt_measure_type", "player_or_team", "play_type", "type_grouping",
               "measure_type", "metric", "value", "heldout", "source", "fetched_at"]]
    agg = agg.astype({"player_id": "int32", "team_id": "int32", "value": "float64", "heldout": "boolean"})
    for c in ("split_name", "split_value", "pt_measure_type", "play_type", "type_grouping", "measure_type"):
        agg[c] = agg[c].astype("object").where(agg[c].notna(), None)
    positions = pd.DataFrame(pos_rows).astype({"player_id": "int32"})
    agg.attrs["archetype"] = {int(pids[i]): ARCHETYPES[arche[i]] for i in range(n_players)}
    positions.attrs["archetype"] = agg.attrs["archetype"]
    return agg, positions


def synth_stints(seed: int, n_games: int = 20, seasons=("2020-21",), n_players_per_team: int = 9) -> dict:
    """Stints (section 12), play-by-play (section 13) and a games frame with hidden player ratings
    (offence and defence per player) and a lineup interaction effect (one designated pair per team scores
    an extra 0.12 points per possession when both are on the floor)."""
    rs = np.random.RandomState(seed * 101 + 3)
    n_teams = 6
    team_ids = [TEAM_ID_BASE + 20 + t for t in range(n_teams)]
    team_code = {tid: TEAM_CODES[6 + t] for t, tid in enumerate(team_ids)}
    players = {}            # team -> list of player ids
    off_rating, def_rating, names = {}, {}, {}
    pair = {}
    k = 0
    for tid in team_ids:
        ids = [300000 + (seed % 1000) * 100 + k + i for i in range(n_players_per_team)]
        k += n_players_per_team
        players[tid] = ids
        for i, p in enumerate(ids):
            off_rating[p] = float(rs.normal(0, 0.03))
            def_rating[p] = float(rs.normal(0, 0.03))
            names[p] = "%s %s" % (FIRST_NAMES[(p + i) % len(FIRST_NAMES)], LAST_NAMES[p % len(LAST_NAMES)])
        pair[tid] = (ids[0], ids[1])
    stints, pbp, games = [], [], []
    stint_cols = ["game_id", "season", "stint", "start_s", "end_s", "duration_s", "period_start", "period_end"] + \
        ["home_%d" % i for i in range(1, 6)] + ["away_%d" % i for i in range(1, 6)] + \
        ["home_ids", "away_ids", "n_home", "n_away", "lineup_ok", "points_home", "points_away", "score_home",
         "score_away", "split", "heldout", "season_type", "game_date", "source", "endpoint", "fetched_at"]
    for si, season in enumerate(seasons):
        yy = int(season[:4])
        for g in range(n_games):
            game_id = "002%02d%05d" % (yy % 100, g + 1)
            h, a = rs.choice(n_teams, 2, replace=False)
            home_tid, away_tid = team_ids[h], team_ids[a]
            month = [11, 12, 1, 2, 3][g % 5]
            date = "%d-%02d-%02d" % (yy if month >= 7 else yy + 1, month, 1 + (g * 3) % 27)
            # lineups: a stint boundary every 60 to 240 s
            boundaries = [0.0]
            while boundaries[-1] < 2880.0:
                boundaries.append(min(2880.0, boundaries[-1] + float(rs.uniform(60, 240))))
            cur = {home_tid: sorted(rs.choice(players[home_tid], 5, replace=False).tolist()),
                   away_tid: sorted(rs.choice(players[away_tid], 5, replace=False).tolist())}
            score = [0, 0]      # away, home
            ev_rows = []

            def add_ev(period, clock, team, desc, points, kind, p1=None, p2=None, etype=0):
                side_tid = None if team is None else (home_tid if team == "home" else away_tid)
                if points:
                    score[0 if team == "away" else 1] += points
                ev_rows.append({"game": game_id, "event": len(ev_rows), "period": period, "clock_s": clock, "team": team,
                                "desc": desc, "points": points, "score": "%d-%d" % (score[0], score[1]),
                                "kind": kind, "frame": -1, "frame_clock_diff_s": None, "source": SOURCE,
                                "endpoint": "synth_stints", "season": season, "fetched_at": FETCHED_AT,
                                "event_id": len(ev_rows), "event_type": etype, "action_type": 0,
                                "player1_id": p1, "player2_id": p2, "player3_id": None, "team_id": side_tid,
                                "clock_str": _clock_str(clock), "game_id": game_id, "split": "train", "heldout": False,
                                "season_type": "Regular Season", "game_date": date})

            add_ev(1, 720.0, None, "Start of 1st quarter", 0, "period_start", etype=12)
            for s in range(len(boundaries) - 1):
                t0, t1 = boundaries[s], boundaries[s + 1]
                p0, p1 = int(t0 // 720) + 1, int(max(t0, t1 - 0.01) // 720) + 1
                if s > 0:
                    # one substitution per team at the boundary
                    for tid, side in ((home_tid, "home"), (away_tid, "away")):
                        on = cur[tid]
                        out_p = int(rs.choice(on))
                        bench = [p for p in players[tid] if p not in on]
                        in_p = int(rs.choice(bench))
                        cur[tid] = sorted([p for p in on if p != out_p] + [in_p])
                        clock = 720.0 - (t0 - (p0 - 1) * 720.0)
                        add_ev(p0, clock, side, "%s enters the game for %s" % (_initial_name(names[in_p]), _initial_name(names[out_p])),
                               0, "substitution", p1=out_p, p2=in_p, etype=8)
                hl, al = cur[home_tid], cur[away_tid]
                n_poss = max(1, int(round((t1 - t0) / 14.5)))
                pts = {}
                for tid, side, mine, theirs in ((home_tid, "home", hl, al), (away_tid, "away", al, hl)):
                    ppp = 1.05 + sum(off_rating[p] for p in mine) - sum(def_rating[p] for p in theirs)
                    if pair[tid][0] in mine and pair[tid][1] in mine:
                        ppp += 0.12
                    if side == "home":
                        ppp += 0.02
                    pts[side] = int(rs.poisson(max(0.05, ppp) * n_poss))
                # scoring events spread inside the stint
                for side in ("home", "away"):
                    remaining = pts[side]
                    tid = home_tid if side == "home" else away_tid
                    while remaining > 0:
                        v = 3 if (remaining >= 3 and rs.rand() < 0.3) else min(2, remaining)
                        el = float(rs.uniform(t0 + 0.01, t1))
                        per = int(el // 720) + 1
                        clock = 720.0 - (el - (per - 1) * 720.0)
                        shooter = int(rs.choice(cur[tid]))
                        add_ev(per, round(clock, 1), side, "%s makes %d-pt shot" % (_initial_name(names[shooter]), v), v,
                               "shot_made", p1=shooter, etype=1)
                        remaining -= v
                stints.append({"game_id": game_id, "season": season, "stint": s, "start_s": t0, "end_s": t1,
                               "duration_s": t1 - t0, "period_start": p0, "period_end": p1,
                               **{"home_%d" % (i + 1): hl[i] for i in range(5)},
                               **{"away_%d" % (i + 1): al[i] for i in range(5)},
                               "home_ids": "|".join(str(p) for p in hl), "away_ids": "|".join(str(p) for p in al),
                               "n_home": 5, "n_away": 5, "lineup_ok": True,
                               "points_home": pts["home"], "points_away": pts["away"],
                               "score_home": 0, "score_away": 0, "split": "train", "heldout": False,
                               "season_type": "Regular Season", "game_date": date, "source": SOURCE,
                               "endpoint": "synth_stints", "fetched_at": FETCHED_AT})
            add_ev(4, 0.0, None, "End of 4th quarter", 0, "period_end", etype=13)
            # running score per stint from the event order (events are stored in time order per stint)
            ev_df = pd.DataFrame(ev_rows)
            ev_df["elapsed"] = (ev_df["period"] - 1) * 720.0 + (720.0 - ev_df["clock_s"])
            ev_df = ev_df.sort_values(["elapsed", "event"], kind="stable").reset_index(drop=True)
            ev_df["event"] = np.arange(len(ev_df)); ev_df["event_id"] = ev_df["event"]
            sh = sa = 0
            scores = []
            for r in ev_df.itertuples(index=False):
                if r.points:
                    if r.team == "home":
                        sh += int(r.points)
                    else:
                        sa += int(r.points)
                scores.append("%d-%d" % (sa, sh))
            ev_df["score"] = scores
            pbp.append(ev_df.drop(columns=["elapsed"]))
            sh = sa = 0
            first = len(stints) - (len(boundaries) - 1)
            for st in stints[first:]:
                sh += st["points_home"]; sa += st["points_away"]
                st["score_home"], st["score_away"] = sh, sa
            strength = (sum(off_rating[p] - def_rating[p] for p in players[home_tid]) -
                        sum(off_rating[p] - def_rating[p] for p in players[away_tid])) * 40.0 + 2.5
            games.append({"game_id": game_id, "season": season, "game_date": date, "home_team_id": home_tid,
                          "away_team_id": away_tid, "home_score": sh, "away_score": sa, "home_win": sh > sa,
                          "spread_home": None if rs.rand() < 0.1 else float(round(-strength * 2) / 2)})
    stints_df = pd.DataFrame(stints, columns=stint_cols)
    stints_df = stints_df.astype({"stint": "int32", "start_s": "float32", "end_s": "float32", "duration_s": "float32",
                                  "period_start": "int8", "period_end": "int8", "n_home": "int8", "n_away": "int8",
                                  "lineup_ok": "bool", "points_home": "int16", "points_away": "int16",
                                  "score_home": "int16", "score_away": "int16", "heldout": "bool",
                                  **{"home_%d" % i: "Int32" for i in range(1, 6)},
                                  **{"away_%d" % i: "Int32" for i in range(1, 6)}})
    pbp_df = pd.concat(pbp, ignore_index=True)
    pbp_df = pbp_df.astype({"event": "int32", "period": "int8", "clock_s": "float32", "points": "int8", "frame": "int32",
                            "frame_clock_diff_s": "float32", "event_id": "Int32", "event_type": "Int16",
                            "action_type": "Int16", "player1_id": "Int32", "player2_id": "Int32",
                            "player3_id": "Int32", "team_id": "Int32", "heldout": "bool"})
    games_df = pd.DataFrame(games).astype({"home_team_id": "int32", "away_team_id": "int32", "home_score": "int16",
                                           "away_score": "int16", "home_win": "bool", "spread_home": "Float32"})
    out = {"stints": stints_df, "playbyplay": pbp_df, "games": games_df}
    out["truth"] = {"off_rating": off_rating, "def_rating": def_rating, "pair_effect": {t: list(v) for t, v in pair.items()},
                    "pair_bonus_ppp": 0.12, "players": players, "team_code": team_code}
    return out


def synth_odds(seed: int, games: pd.DataFrame) -> pd.DataFrame:
    """Section 14 odds for the games frame of synth_stints: three books, h2h, spreads and totals, two
    snapshots (open and close), consistent with spread_home (a null spread gets a book line near pick-em)."""
    rs = np.random.RandomState(seed * 17 + 11)
    rows = []
    code_of = lambda tid: TEAM_CODES[(tid - TEAM_ID_BASE) % len(TEAM_CODES)]   # noqa: E731
    for g in games.itertuples(index=False):
        hname, aname = TEAM_NAMES[code_of(int(g.home_team_id))], TEAM_NAMES[code_of(int(g.away_team_id))]
        key = "%s_%s_%s" % (g.game_date, aname.replace(" ", ""), hname.replace(" ", ""))
        commence = "%sT02:00:00Z" % g.game_date
        true_spread = float(g.spread_home) if pd.notna(g.spread_home) else float(rs.normal(0, 1.0))
        total = float(round(2 * (210.0 + rs.normal(0, 6.0))) / 2)
        for snap, is_close in (("%sT14:00:00Z" % g.game_date, False), ("%sT01:30:00Z" % g.game_date, True)):
            for book in BOOKS:
                sp = round(2 * (true_spread + rs.normal(0, 0.5 if is_close else 1.0))) / 2
                p_home = 1.0 / (1.0 + math.exp(sp / 4.0))
                def american(p):
                    p = min(0.95, max(0.05, p)) * 1.045
                    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))
                base = {"game_key": key, "commence_time": commence, "home_team": hname, "away_team": aname,
                        "book": book, "snapshot_at": snap, "is_close": is_close, "source": SOURCE,
                        "endpoint": "synth_odds", "fetched_at": FETCHED_AT, "season": g.season}
                rows.append({**base, "market": "h2h", "side": hname, "point": None, "price": american(p_home)})
                rows.append({**base, "market": "h2h", "side": aname, "point": None, "price": american(1 - p_home)})
                rows.append({**base, "market": "spreads", "side": hname, "point": sp, "price": -110})
                rows.append({**base, "market": "spreads", "side": aname, "point": -sp, "price": -110})
                tt = total + float(rs.choice([-0.5, 0.0, 0.5]))
                rows.append({**base, "market": "totals", "side": "Over", "point": tt, "price": -110})
                rows.append({**base, "market": "totals", "side": "Under", "point": tt, "price": -110})
    df = pd.DataFrame(rows, columns=["game_key", "commence_time", "home_team", "away_team", "book", "market", "side",
                                     "point", "price", "snapshot_at", "is_close", "source", "endpoint", "fetched_at",
                                     "season"])
    return df.astype({"point": "float32", "price": "int32", "is_close": "bool"})


def synth_labels(seed: int, game: dict) -> dict:
    """Action and screen labels of a synth_game dict: the truth the simulator recorded (seed is accepted for
    API symmetry and only orders the rows)."""
    actions = pd.DataFrame(game["truth"]["actions"],
                           columns=["sportvu_game", "possession", "action_type", "start_frame", "end_frame",
                                    "ball_handler_id", "screener_id", "label_source", "labeller_pass"])
    actions = actions.astype({"possession": "int32", "start_frame": "int32", "end_frame": "int32",
                              "ball_handler_id": "int32", "screener_id": "Int32", "labeller_pass": "int8"})
    screens = pd.DataFrame(game["truth"]["screens"],
                           columns=["sportvu_game", "possession", "action_id", "coverage", "on_ball_defender_id",
                                    "screener_defender_id", "label_source", "confidence"])
    screens = screens.astype({"possession": "int32", "action_id": "int32", "on_ball_defender_id": "int32",
                              "screener_defender_id": "int32"})
    return {"actions": actions, "screens": screens}


# ── the DATA_ROOT layout ──────────────────────────────────────────────────────────────────────────
def _write_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in df.to_dict("records"):
            f.write(json.dumps({k: (None if (v is pd.NA or (isinstance(v, float) and math.isnan(v))) else
                                   (int(v) if isinstance(v, (np.integer,)) else v)) for k, v in rec.items()}) + "\n")


def _pq(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path.stat().st_size


def write_fixture_root(root, seed: int = 0, n_games: int = 6, heldout_every: int = 3, n_possessions: int = 40) -> dict:
    """Write the DATA_ROOT layout the lanes read (docs/contract/fixtures.md) and return the Step 1 manifest."""
    root = Path(root)
    games = synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    sv = root / "sportvu_contract"
    manifest = {"contract_version": 1, "contract": "docs/DATA_CONTRACT.md",
                "source": "analytics.fixtures synth_season seed %d" % seed, "coverage": ["2015-16 synthetic"],
                "updated": FETCHED_AT,
                "heldout_rule": "games listed in heldout (and every file under heldout/) are held out: no training "
                                "job may read them", "heldout": [], "games": {}}
    shots_train, shots_held = [], []
    for g in games:
        m = g["meta"]
        gid, held, season = m["sportvu_game"], m["heldout"], m["season"]
        d = sv / "heldout" if held else sv
        entry = dict(g["manifest_entry"])
        entry["parquet_bytes"] = _pq(g["frames"], d / (gid + ".parquet"))
        _pq(g["events"], d / (gid + "_pbp.parquet"))
        manifest["games"][gid] = entry
        if held:
            manifest["heldout"].append(gid)
        (shots_held if held else shots_train).append(g["shots"])
        for table in ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz"):
            _pq(g[table], schema.table_path(root / "analytics", table, season, gid, held))
        labels = synth_labels(seed, g)
        _write_jsonl(root / "labels" / "actions" / (gid + ".jsonl"), labels["actions"])
        _write_jsonl(root / "labels" / "screens" / (gid + ".jsonl"), labels["screens"])
    manifest["counts"] = {"games": len(games), "ok": len(games), "failed": 0, "source_broken": 0,
                          "pbp_joined": len(games), "heldout": len(manifest["heldout"])}
    sv.mkdir(parents=True, exist_ok=True)
    (sv / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if shots_train:
        _pq(pd.concat(shots_train, ignore_index=True), root / "collect" / "bref_shots" / "shotchart" / (SEASON_2015 + ".parquet"))
    if shots_held:
        _pq(pd.concat(shots_held, ignore_index=True), root / "collect" / "bref_shots" / "heldout" / "shotchart" / (SEASON_2015 + ".parquet"))
    agg, positions = synth_aggregates(seed)
    for season, part in agg.groupby("season"):
        _pq(part.reset_index(drop=True), root / "collect" / "tracking" / "aggregates" / (season + ".parquet"))
    for season, part in positions.groupby("season"):
        _pq(part.reset_index(drop=True), root / "collect" / "tracking" / "positions" / (season + ".parquet"))
    st = synth_stints(seed)
    for season, part in st["stints"].groupby("season"):
        _pq(part.reset_index(drop=True), root / "collect" / "stints" / "stints" / (season + ".parquet"))
    for season, part in st["playbyplay"].groupby("season"):
        _pq(part.reset_index(drop=True), root / "collect" / "stints" / "playbyplay" / (season + ".parquet"))
    for season, part in st["games"].groupby("season"):
        _pq(part.reset_index(drop=True), root / "collect" / "stints" / "games" / (season + ".parquet"))
    odds = synth_odds(seed, st["games"])
    for season, part in odds.groupby("season"):
        _pq(part.reset_index(drop=True), root / "collect" / "extras" / "odds" / (season + ".parquet"))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analytics.fixtures", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="DATA_ROOT directory to write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--heldout-every", type=int, default=3)
    ap.add_argument("--possessions", type=int, default=40)
    a = ap.parse_args(argv)
    man = write_fixture_root(a.out, seed=a.seed, n_games=a.games, heldout_every=a.heldout_every, n_possessions=a.possessions)
    print(json.dumps({"out": str(a.out), "games": man["counts"], "heldout": man["heldout"]}))
    return 0


__all__ = ["synth_game", "synth_season", "synth_aggregates", "synth_stints", "synth_odds", "synth_labels",
           "write_fixture_root", "HELDOUT_GAME", "BASKETS", "HZ", "GAP_POSSESSION", "TEAM_CODES", "TEAM_NAMES"]

if __name__ == "__main__":
    sys.exit(main())
