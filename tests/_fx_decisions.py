"""tests/_fx_decisions.py: synthetic games for the decision-quality lane (analytics/decisions).

Prefers the shared generator of lane I1 (analytics/fixtures.py) when it is importable; otherwise the minimal
local generator below with the same signatures and column names:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None)
        -> dict(frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz,
                sportvu_game, heldout)
    synth_season(seed, n_games=8, heldout_every=0) -> list of such dicts
    write_fixture_root(root, seed, n_games, heldout_every) -> the DATA_ROOT layout train.py reads

The local generator is deterministic given the seed. Each possession is a short script on a 94 x 50 ft court at
25 frames per second: the handler holds the ball (ball within 1 ft of the handler, under 8 ft), passes it (the
ball travels between two players over 0.4 s), sometimes drives (10 ft/s toward the basket for 1 s) and shoots
(release at 6 ft, apex 16 ft, back through 10 ft at the rim after 1 s). A made shot ends the possession, a miss
ends it at a defensive rebound, a bad pass to a defender is a turnover. Every player carries a fixed skill
(a function of player_id, so it is the same across games): the probability of picking the option the fixture
itself rates best (shoot when open and close, else pass to the most open teammate). Shot outcomes depend on
distance and openness, so lineups of skilled deciders score more: the sign the lane's validation looks for.
Fixture numbers are never results.
"""
from __future__ import annotations

try:  # the shared generator of lane I1, when it exists
    from analytics.fixtures import synth_game, synth_season  # type: ignore  # noqa: F401
    try:
        from analytics.fixtures import write_fixture_root  # type: ignore  # noqa: F401
    except ImportError:  # pragma: no cover
        write_fixture_root = None  # type: ignore
    SHARED = True
except ImportError:
    SHARED = False

from pathlib import Path

import numpy as np
import pandas as pd

FPS = 25
DT = 1.0 / FPS
RIM_LEFT, RIM_RIGHT = (5.25, 25.0), (88.75, 25.0)
HOME_ID, AWAY_ID = 1001, 1002
HOME_PLAYERS = [100 + i for i in range(8)]
AWAY_PLAYERS = [200 + i for i in range(8)]
SEASON = "2015-16"


def skill(player_id: int) -> float:
    """Fixed per-player probability of choosing the fixture's best option (0.2 to 0.9)."""
    return 0.2 + 0.7 * ((int(player_id) * 37) % 11) / 10.0


def _pmake(dist: float) -> float:
    return float(np.clip(0.68 - 0.016 * dist, 0.2, 0.7))


def _pts(dist: float) -> int:
    return 3 if dist >= 22.0 else 2


def _open(gap: float) -> float:
    return 0.5 + 0.5 * min(gap, 6.0) / 6.0


class _Game:
    def __init__(self, seed, n_possessions, heldout, sportvu_game):
        self.rng = np.random.default_rng(seed)
        self.n_possessions = n_possessions
        self.heldout = heldout
        self.game = sportvu_game or "01.%02d.2016.AAA.at.BBB" % (seed % 28 + 1)
        self.frame = 0
        self.event = 0
        self.rows = []            # per-frame blocks: dict of arrays
        self.events = []
        self.shots = []
        self.shot_frames = []
        self.rebound_frames = []
        self.possessions = []
        self.states = []
        self.lineup = {HOME_ID: list(HOME_PLAYERS[:5]), AWAY_ID: list(AWAY_PLAYERS[:5])}

    # ── per-possession script ─────────────────────────────────────────────────────────────────
    def run(self):
        per_period = int(np.ceil(self.n_possessions / 4))
        clock = 720.0
        period = 1
        for p in range(self.n_possessions):
            if p and p % per_period == 0 and period < 4:
                period += 1
                clock = 720.0
            clock = self.possession(p, period, clock)
        return self

    def possession(self, p, period, clock):
        rng = self.rng
        off_team = HOME_ID if p % 2 == 0 else AWAY_ID
        def_team = AWAY_ID if off_team == HOME_ID else HOME_ID
        right = (off_team == HOME_ID) == (period <= 2)
        rim = np.array(RIM_RIGHT if right else RIM_LEFT)
        inward = np.array([-1.0, 0.0]) if right else np.array([1.0, 0.0])   # from the rim toward half court
        for t in (HOME_ID, AWAY_ID):
            if rng.uniform() < 0.35:
                pool = HOME_PLAYERS if t == HOME_ID else AWAY_PLAYERS
                self.lineup[t] = sorted(int(x) for x in rng.choice(pool, 5, replace=False))
        off = list(self.lineup[off_team])
        dfn = list(self.lineup[def_team])
        # offence positions: distance 6..27 ft from the rim, angle -75..75 deg from the inward axis
        dist = rng.uniform(6, 27, 5)
        dist[0] = rng.uniform(20, 26)                                       # the first handler, up top
        ang = np.deg2rad(rng.uniform(-75, 75, 5))
        ang[0] = np.deg2rad(rng.uniform(-15, 15))
        across = np.array([-inward[1], inward[0]])
        pos_off = rim + dist[:, None] * (np.cos(ang)[:, None] * inward + np.sin(ang)[:, None] * across)
        gaps = rng.uniform(1, 9, 5)
        assign = rng.permutation(5)                                         # defender j guards offence assign[j]
        pos = {"off": pos_off.copy(), "def": np.zeros((5, 2))}
        self._gap = gaps
        self._assign = assign
        self._rim = rim
        self._place_defenders(pos)
        handler = 0
        start_frame = self.frame
        start_clock = clock
        start_event = self.event
        release = None
        n_pass = 0
        shot_dist = None
        outcome = None
        for _ in range(4):
            self._hold(pos, off, dfn, handler, int(rng.integers(20, 40)))
            d_self = float(np.linalg.norm(pos["off"][handler] - rim))
            gap_of = lambda i: float(gaps[int(np.where(assign == i)[0][0])])
            vals = {}
            for i in range(5):
                d = float(np.linalg.norm(pos["off"][i] - rim))
                v = _pts(d) * _pmake(d) * _open(gap_of(i))
                vals[i] = v if i == handler else 0.95 * v
            best = max(vals, key=vals.get)
            choice = best if rng.uniform() < skill(off[handler]) else int(rng.integers(0, 5))
            if choice == handler or n_pass >= 3:
                if d_self > 12 and rng.uniform() < 0.35:
                    self._drive(pos, off, dfn, handler, 25)
                release = self.frame
                shot_dist = float(np.linalg.norm(pos["off"][handler] - rim))
                p_make = _pmake(shot_dist) * (0.6 + 0.4 * _open(gap_of(handler)))
                made = bool(rng.uniform() < p_make)
                apex, out = self._shot(pos, off, dfn, handler)
                outcome = ("made" if made else "missed", release, apex, out, handler, shot_dist)
                break
            if rng.uniform() < 0.07:                                         # a bad pass: turnover
                j = int(rng.integers(0, 5))
                self._pass(pos, off, dfn, handler, j, to_defender=True)
                outcome = ("turnover", None, None, self.frame - 1, handler, None)
                break
            self._pass(pos, off, dfn, handler, choice)
            handler = choice
            n_pass += 1
        kind = outcome[0]
        end_clock = clock - (self.frame - start_frame) * DT
        points = 0
        if kind in ("made", "missed"):
            _, release, apex, out_f, shooter_i, sdist = outcome
            shooter = off[shooter_i]
            made = kind == "made"
            points = _pts(sdist) if made else 0
            ev_clock = clock - (out_f - start_frame) * DT
            self.events.append({"game": self.game, "event": self.event, "period": period, "clock_s": ev_clock,
                                "team": "home" if off_team == HOME_ID else "away",
                                "desc": "P%d %s %d-pt shot from %d ft" % (shooter, "makes" if made else "misses", _pts(sdist), round(sdist)),
                                "points": points, "score": None, "kind": "shot_made" if made else "shot_missed",
                                "frame": out_f, "frame_clock_diff_s": 0.0})
            self.shot_frames.append({"sportvu_game": self.game, "event": self.event, "period": period, "clock_s": ev_clock,
                                     "made": made, "pbp_shooter_name": "P%d" % shooter, "shooter_id": shooter,
                                     "release_frame": release, "apex_frame": apex, "outcome_frame": out_f,
                                     "shooter_at_release_id": shooter, "identity_match": True, "confidence": 0.9,
                                     "confidence_flag": "high"})
            spos = pos["off"][shooter_i]
            rel = spos - rim
            loc_y = float(np.dot(rel, inward)) * 10
            loc_x = float(np.dot(rel, across)) * 10
            self.shots.append({"game": self.game, "sportvu_game": self.game, "period": period, "clock_s": ev_clock,
                               "made": made, "shot_type": "3PT" if _pts(sdist) == 3 else "2PT",
                               "shot_distance_ft": float(round(sdist)), "loc_x": loc_x, "loc_y": loc_y,
                               "court_x": 5.25 + loc_y / 10, "court_y": 25 + loc_x / 10, "player": "P%d" % shooter,
                               "player_id": shooter, "event_id": self.event, "heldout": self.heldout,
                               "split": "heldout_game" if self.heldout else None, "source": "fixture"})
            shot_event = self.event
            self.event += 1
            if made:
                self._after_made(pos, off, dfn)
                end_kind = "shot_made"
            else:
                reb_frame, rebounder = self._after_missed(pos, off, dfn)
                end_kind = "shot_missed"
                self.events.append({"game": self.game, "event": self.event, "period": period,
                                    "clock_s": clock - (reb_frame - start_frame) * DT,
                                    "team": "home" if def_team == HOME_ID else "away",
                                    "desc": "Defensive rebound by P%d" % rebounder, "points": 0, "score": None,
                                    "kind": "rebound", "frame": reb_frame, "frame_clock_diff_s": 0.0})
                self.rebound_frames.append({"sportvu_game": self.game, "event": shot_event, "rebound_event": self.event,
                                            "period": period, "clock_s": ev_clock, "rim_frame": out_f,
                                            "rebound_frame": reb_frame, "rebounder_id": rebounder,
                                            "pbp_rebounder_name": "P%d" % rebounder, "pbp_rebounder_id": rebounder,
                                            "identity_match": True, "confidence": 0.8, "confidence_flag": "high"})
                self.event += 1
        else:
            end_kind = "turnover"
            self.events.append({"game": self.game, "event": self.event, "period": period, "clock_s": end_clock,
                                "team": "home" if off_team == HOME_ID else "away", "desc": "Turnover by P%d (bad pass)" % off[outcome[4]],
                                "points": 0, "score": None, "kind": "turnover", "frame": self.frame - 1, "frame_clock_diff_s": 0.0})
            self.event += 1
        end_frame = self.frame - 1
        self.possessions.append({"sportvu_game": self.game, "possession": p, "period": period, "start_frame": start_frame,
                                 "end_frame": end_frame, "start_clock_s": start_clock, "end_clock_s": end_clock,
                                 "start_event": start_event, "end_event": self.event - 1, "offense_team_id": off_team,
                                 "end_kind": end_kind, "points": points, "gap": False, "n_frames": end_frame - start_frame + 1})
        self._states(p, period, start_frame, end_frame, start_clock, off_team, off, dfn, points, end_kind)
        return end_clock - 1.0                                               # a second of dead time between possessions

    # ── segments (each appends frames) ────────────────────────────────────────────────────────
    def _place_defenders(self, pos):
        rim = self._rim
        for j in range(5):
            i = self._assign[j]
            to_rim = rim - pos["off"][i]
            n = np.linalg.norm(to_rim)
            u = to_rim / n if n > 0 else np.array([1.0, 0.0])
            pos["def"][j] = pos["off"][i] + u * min(self._gap[j], max(n - 1.0, 0.5))

    def _emit(self, pos_seq, ball_seq, off, dfn, off_is_home):
        """pos_seq: (n, 2, 5, 2) offence then defence; ball_seq: (n, 3)."""
        n = len(ball_seq)
        home_ids, away_ids = (off, dfn) if off_is_home else (dfn, off)
        home_pos, away_pos = (pos_seq[:, 0], pos_seq[:, 1]) if off_is_home else (pos_seq[:, 1], pos_seq[:, 0])
        ids = np.concatenate([np.repeat([home_ids], n, axis=0), np.repeat([away_ids], n, axis=0)], axis=1)  # (n, 10)
        xy = np.concatenate([home_pos, away_pos], axis=1)                                                    # (n, 10, 2)
        frames = np.arange(self.frame, self.frame + n)
        self.rows.append({"frame": np.repeat(frames, 10), "player_id": ids.reshape(-1),
                          "x": xy[:, :, 0].reshape(-1), "y": xy[:, :, 1].reshape(-1),
                          "ball_x": np.repeat(ball_seq[:, 0], 10), "ball_y": np.repeat(ball_seq[:, 1], 10),
                          "ball_z": np.repeat(ball_seq[:, 2], 10), "home": np.tile(np.r_[np.ones(5), np.zeros(5)].astype(bool), n)})
        self.frame += n

    def _walk(self, pos, off, dfn, n, still=(), sigma=0.12):
        """n frames of small random motion; returns (n, 2, 5, 2) and leaves pos at the last frame."""
        seq = np.zeros((n, 2, 5, 2))
        steps = self.rng.normal(0, sigma, (n, 5, 2))
        for i in still:
            steps[:, i] = 0
        seq[:, 0] = pos["off"][None] + np.cumsum(steps, axis=0)
        seq[:, 0, :, 0] = np.clip(seq[:, 0, :, 0], 1, 93)
        seq[:, 0, :, 1] = np.clip(seq[:, 0, :, 1], 1, 49)
        for k in range(n):
            pos["off"] = seq[k, 0]
            self._place_defenders(pos)
            seq[k, 1] = pos["def"]
        pos["off"] = seq[-1, 0].copy()
        pos["def"] = seq[-1, 1].copy()
        return seq

    def _hold(self, pos, off, dfn, handler, n):
        seq = self._walk(pos, off, dfn, n, still=(handler,))
        t = np.arange(n) * DT
        ball = np.zeros((n, 3))
        ball[:, :2] = seq[:, 0, handler] + np.array([0.6, 0.0])
        ball[:, 2] = 3.0 + 1.5 * np.sin(2 * np.pi * 2 * t)
        self._emit(seq, ball, off, dfn, off[0] < 200)

    def _pass(self, pos, off, dfn, a, b, to_defender=False):
        n = int(0.4 * FPS)
        src = pos["off"][a].copy()
        seq = self._walk(pos, off, dfn, n, still=(b,) if not to_defender else ())
        dst = seq[-1, 1, b] if to_defender else seq[-1, 0, b]
        t = np.linspace(0, 1, n + 1)[1:]
        ball = np.zeros((n, 3))
        ball[:, :2] = src[None] + (dst - src)[None] * t[:, None]
        ball[:, 2] = 5.0 + 2.5 * np.sin(np.pi * t)
        ball[-1, :2] = dst + np.array([0.3, 0.0])
        ball[-1, 2] = 4.0
        self._emit(seq, ball, off, dfn, off[0] < 200)

    def _drive(self, pos, off, dfn, handler, n):
        rim = self._rim
        seq = np.zeros((n, 2, 5, 2))
        u = rim - pos["off"][handler]
        d0 = np.linalg.norm(u)
        u = u / d0
        step = 10.0 / FPS
        path = pos["off"][handler][None] + u[None] * np.minimum(np.arange(1, n + 1) * step, max(d0 - 4.0, 0))[:, None]
        walk = self._walk(pos, off, dfn, n, still=(handler,))
        walk[:, 0, handler] = path
        for k in range(n):
            pos["off"] = walk[k, 0]
            self._place_defenders(pos)
            walk[k, 1] = pos["def"]
        pos["off"] = walk[-1, 0].copy()
        pos["def"] = walk[-1, 1].copy()
        seq = walk
        ball = np.zeros((n, 3))
        ball[:, :2] = seq[:, 0, handler] + np.array([0.6, 0.0])
        ball[:, 2] = 2.5 + 1.0 * np.sin(np.arange(n) * 0.9)
        self._emit(seq, ball, off, dfn, off[0] < 200)

    def _shot(self, pos, off, dfn, handler):
        n = FPS
        rim = self._rim
        src = pos["off"][handler].copy()
        seq = self._walk(pos, off, dfn, n, still=(handler,))
        t = np.linspace(0, 1, n + 1)[1:]
        ball = np.zeros((n, 3))
        ball[:, :2] = src[None] + (rim - src)[None] * t[:, None]
        ball[:, 2] = 6.0 + 40.0 * t - 36.0 * t ** 2                          # 6 at release, apex near 0.55 s, 10 at the rim
        release = self.frame
        apex = self.frame + int(np.argmax(ball[:, 2]))
        self._emit(seq, ball, off, dfn, off[0] < 200)
        return apex, self.frame - 1

    def _after_made(self, pos, off, dfn):
        n = 6
        seq = self._walk(pos, off, dfn, n)
        ball = np.zeros((n, 3))
        ball[:, :2] = self._rim
        ball[:, 2] = np.linspace(9.0, 0.5, n)
        self._emit(seq, ball, off, dfn, off[0] < 200)

    def _after_missed(self, pos, off, dfn):
        n = 12
        rim = self._rim
        spot = rim + self.rng.uniform(-4, 4, 2)
        spot[0] = float(np.clip(spot[0], 1, 93))
        j = int(np.argmin(np.linalg.norm(pos["def"] - spot, axis=1)))
        seq = self._walk(pos, off, dfn, n)
        path = np.linspace(pos["def"][j], spot, n)
        seq[:, 1, j] = path
        pos["def"][j] = spot
        t = np.linspace(0, 1, n)
        ball = np.zeros((n, 3))
        ball[:, :2] = rim[None] + (spot - rim)[None] * t[:, None]
        ball[:, 2] = 10.0 + 4.0 * t - 11.0 * t ** 2
        self._emit(seq, ball, off, dfn, off[0] < 200)
        return self.frame - 1, dfn[j]

    def _states(self, p, period, start, end, start_clock, off_team, off, dfn, points, end_kind):
        for s in range(0, int((end - start) * DT) + 1):
            f = start + int(round(s * FPS))
            if f > end:
                break
            self.states.append({"possession": p, "frame": f, "period": period, "game_clock_s": start_clock - s,
                                "shot_clock_s": 24.0 - s, "elapsed_in_possession_s": float(s), "offense_team_id": off_team,
                                "off_ids": sorted(off), "def_ids": sorted(dfn), "points_to_end": points, "possession_end_kind": end_kind})

    # ── tables ────────────────────────────────────────────────────────────────────────────────
    def tables(self):
        blocks = {k: np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]}
        n = len(blocks["frame"])
        poss = pd.DataFrame(self.possessions)
        # per-frame period and clock from the possession spans (dead time between possessions is not tracked)
        per = np.zeros(self.frame, dtype=np.int8)
        clk = np.zeros(self.frame, dtype=np.float32)
        for r in self.possessions:
            idx = np.arange(r["start_frame"], r["end_frame"] + 1)
            per[idx] = r["period"]
            clk[idx] = r["start_clock_s"] - (idx - r["start_frame"]) * DT
        pid = blocks["player_id"].astype(np.int32)
        home = blocks["home"]
        frames = pd.DataFrame({"game": self.game, "gameid": "0021500%03d" % (hash(self.game) % 1000), "date": "2016-01-01",
                               "period": per[blocks["frame"]], "frame": blocks["frame"].astype(np.int32),
                               "ts_ms": (1_450_000_000_000 + blocks["frame"] * 40).astype(np.int64),
                               "game_clock_s": clk[blocks["frame"]],
                               "shot_clock_s": np.float32(24.0), "n_players": np.int8(10),
                               "ball_x": blocks["ball_x"].astype(np.float32), "ball_y": blocks["ball_y"].astype(np.float32),
                               "ball_z": blocks["ball_z"].astype(np.float32),
                               "team_id": np.where(home, HOME_ID, AWAY_ID).astype(np.int32),
                               "team": np.where(home, "AAA", "BBB"), "home": home, "player_id": pid,
                               "player": ["P%d" % i for i in pid], "jersey": [str(i % 100) for i in pid],
                               "x": blocks["x"].astype(np.float32), "y": blocks["y"].astype(np.float32)})
        events = pd.DataFrame(self.events, columns=["game", "event", "period", "clock_s", "team", "desc", "points", "score",
                                                    "kind", "frame", "frame_clock_diff_s"])
        rosters = pd.DataFrame([{"sportvu_game": self.game, "player_id": i, "player": "P%d" % i, "team_id": t,
                                 "team": "AAA" if t == HOME_ID else "BBB", "home": t == HOME_ID, "jersey": str(i % 100),
                                 "bref_slug": "p%d" % i, "bref_name": "P%d" % i, "pbp_name": "P%d" % i, "match_method": "exact"}
                                for t, ids in ((HOME_ID, HOME_PLAYERS), (AWAY_ID, AWAY_PLAYERS)) for i in ids])
        states = self._states_table(frames)
        shot_cols = ["sportvu_game", "event", "period", "clock_s", "made", "pbp_shooter_name", "shooter_id", "release_frame",
                     "apex_frame", "outcome_frame", "shooter_at_release_id", "identity_match", "confidence", "confidence_flag"]
        reb_cols = ["sportvu_game", "event", "rebound_event", "period", "clock_s", "rim_frame", "rebound_frame", "rebounder_id",
                    "pbp_rebounder_name", "pbp_rebounder_id", "identity_match", "confidence", "confidence_flag"]
        return {"frames": frames, "events": events, "shots": pd.DataFrame(self.shots), "rosters": rosters,
                "shot_frames": pd.DataFrame(self.shot_frames, columns=shot_cols),
                "rebound_frames": pd.DataFrame(self.rebound_frames, columns=reb_cols),
                "possessions": poss, "states_1hz": states, "sportvu_game": self.game, "heldout": self.heldout}

    def _states_table(self, frames):
        fl = frames.set_index(["frame", "player_id"])[["x", "y"]]
        ball = frames.drop_duplicates("frame").set_index("frame")[["ball_x", "ball_y", "ball_z"]]
        rows = []
        for s in self.states:
            f = s["frame"]
            r = {"sportvu_game": self.game, "possession": s["possession"], "frame": f, "period": s["period"],
                 "game_clock_s": s["game_clock_s"], "shot_clock_s": s["shot_clock_s"],
                 "elapsed_in_possession_s": s["elapsed_in_possession_s"], "offense_team_id": s["offense_team_id"]}
            b = ball.loc[f]
            r["ball_x"], r["ball_y"], r["ball_z"] = float(b["ball_x"]), float(b["ball_y"]), float(b["ball_z"])
            best, best_d = None, np.inf
            for side, ids in (("off", s["off_ids"]), ("def", s["def_ids"])):
                for k, pid in enumerate(ids, 1):
                    x, y = fl.loc[(f, pid)]
                    r["%s%d_id" % (side, k)] = pid
                    r["%s%d_x" % (side, k)] = float(x)
                    r["%s%d_y" % (side, k)] = float(y)
                    if side == "off":
                        d = float(np.hypot(x - r["ball_x"], y - r["ball_y"]))
                        if d < best_d:
                            best, best_d = pid, d
            r["handler_id"] = best
            dx = np.array([r["off%d_x" % k] for k in range(1, 6)])
            hx = [(r["off%d_x" % k], r["off%d_y" % k]) for k in range(1, 6) if r["off%d_id" % k] == best][0]
            dd = [np.hypot(r["def%d_x" % k] - hx[0], r["def%d_y" % k] - hx[1]) for k in range(1, 6)]
            r["handler_nearest_def_dist_ft"] = float(min(dd))
            r["points_to_end"] = s["points_to_end"]
            r["possession_end_kind"] = s["possession_end_kind"]
            del dx
            rows.append(r)
        from analytics import schema
        return pd.DataFrame(rows, columns=schema.columns("states_1hz"))


if not SHARED:
    def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
        """One synthetic game: frames (section 4), events (section 5), shots (10a-like), and the five Phase 0
        tables, plus sportvu_game and heldout."""
        return _Game(seed, n_possessions, heldout, sportvu_game).run().tables()

    def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0) -> list:
        """n_games synthetic games; every heldout_every-th game (1-based) is flagged held out (0: none)."""
        out = []
        for i in range(n_games):
            held = bool(heldout_every) and ((i + 1) % heldout_every == 0)
            name = "%02d.%02d.2016.AAA.at.BBB" % (2 if held else 1, i % 28 + 1)
            out.append(synth_game(seed * 1000 + i, heldout=held, sportvu_game=name))
        return out

    def write_fixture_root(root, seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
        """Write the DATA_ROOT layout: root/sportvu_contract/<game>.parquet and <game>_pbp.parquet (held-out under
        heldout/), root/analytics/<table>/<season>/<game>.parquet (held-out under heldout/<season>/)."""
        from analytics import schema
        root = Path(root)
        games = []
        for i in range(n_games):
            held = bool(heldout_every) and ((i + 1) % heldout_every == 0)
            name = "%02d.%02d.2016.AAA.at.BBB" % (2 if held else 1, i % 28 + 1)
            g = synth_game(seed * 1000 + i, n_possessions=n_possessions, heldout=held, sportvu_game=name)
            sub = root / "sportvu_contract" / ("heldout" if held else "")
            sub.mkdir(parents=True, exist_ok=True)
            g["frames"].to_parquet(sub / (name + ".parquet"), index=False)
            g["events"].to_parquet(sub / (name + "_pbp.parquet"), index=False)
            for t in ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz"):
                p = schema.table_path(root / "analytics", t, SEASON, name, heldout=held)
                p.parent.mkdir(parents=True, exist_ok=True)
                schema.cast(g[t], t).to_parquet(p, index=False)
            games.append(name)
        return games
