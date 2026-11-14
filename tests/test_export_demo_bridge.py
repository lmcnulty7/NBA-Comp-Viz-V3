"""export/demo_bridge.py (PLAN.md M2): the matcher, the interpolation rule, the fill share, the frame mapping check,
the scene rule, and the whole bridge on a synthetic game with synthetic sidecars. Every number produced here is
fixture output, never a result.

No test reads the real DATA_ROOT or the real broadcast outputs: each one builds its own folders. The frozen models
are replaced by stand-ins that fit nothing."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from analytics import fixtures as FX  # noqa: E402
from export import demo_bridge as B  # noqa: E402
from export import heavy  # noqa: E402
from export import schemas as S  # noqa: E402
from tools import frozen_common as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GAME_ID = FX.HELDOUT_GAME                 # the real held-out id, on synthetic rows
TAG = B.GAME                              # the real game tag too, so the command line finds its scorecard
FPS = 60000 / 1001
STRIDE = 6
WIDTH, HEIGHT = 640, 360
PX_PER_FT_X, PX_PER_FT_Y, OFF_X, OFF_Y = 6.0, 3.0, 30.0, 125.0
BOX_W, BOX_H = 14, 34
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ------------------------------------------------------------------------------------------------ stand-ins

class StubXfg:
    """Not a model: a fixed formula on the frozen feature columns, so that positions move the answer."""

    def predict(self, frame):
        gap = np.nan_to_num(frame["closest_defender_dist_ft"].to_numpy(dtype=float), nan=3.0)
        speed = np.nan_to_num(frame["shooter_speed_ftps"].to_numpy(dtype=float), nan=0.0)
        far = frame["distance_ft"].to_numpy(dtype=float)
        return np.clip(0.45 + 0.02 * gap - 0.006 * far - 0.004 * speed, 0.02, 0.98)


class StubEpv:
    """Not a model: one state a second from the start of every possession, valued by where the offense stands."""

    def predict_states(self, frames, possessions, game):
        rows = []
        for p in possessions.itertuples():
            for frame in range(int(p.start_frame), int(p.end_frame) + 1, 25):
                now = frames[frames["frame"] == frame]
                offense = now[now["team_id"] == p.offense_team_id]
                rows.append((game, int(p.possession), frame, 0.8 + 0.004 * float(offense["x"].mean()) - 0.002 * float(now["y"].mean())))
        return pd.DataFrame(rows, columns=["sportvu_game", "possession", "frame", "epv_points"])


class StubGhost:
    """Not a model: a ghost stands where the defender stood one second earlier, moved half a foot."""

    def predict_ghosts(self, frames, possessions, game):
        at = frames.set_index(["frame", "player_id"])[["x", "y", "team_id"]]
        rows = []
        for p in possessions.itertuples():
            if p.gap:
                continue
            for frame in range(int(p.start_frame) + 35, int(p.end_frame) + 1, 5):
                now = frames[(frames["frame"] == frame) & (frames["team_id"] != p.offense_team_id)]
                for player in now["player_id"].to_numpy():
                    then = at.loc[(frame - 25, player)]
                    for horizon, back in ((0.2, 0.9), (1.0, 0.5)):
                        rows.append((game, int(p.possession), frame, int(player), float(then["x"]) + back, float(then["y"]), horizon))
        return pd.DataFrame(rows, columns=["sportvu_game", "possession", "frame", "def_player_id", "ghost_x", "ghost_y", "horizon_s"])


def passed_check(kind: str) -> dict:
    return {"name": kind, "compared": "fixture predictions with fixture predictions", "n": 10, "n_expected": 10,
            "max_abs_diff": 0.0, "tolerance": C.TOLERANCE_CAP[kind], "passed": True, "unit": B.UNITS[kind]}


def stub_models(kinds=S.MODEL_KINDS) -> dict:
    stubs = {"xfg": StubXfg, "epv": StubEpv, "ghosting": StubGhost}
    return {kind: B.Model(kind, predictor=stubs[kind](), check=passed_check(kind)) for kind in kinds}


@pytest.fixture(autouse=True)
def tracked_copy_of_stub_checks(tmp_path, monkeypatch):
    """The bridge uses a model only when its check equals the tracked copy (D49): give it a copy of the stub checks.
    A test that wants the copy to disagree writes its own."""
    path = tmp_path / "frozen_checks.json"
    records = [dict(passed_check(kind), passed=True, reasons=[]) for kind in S.MODEL_KINDS]
    path.write_text(json.dumps({"checks": records}))
    monkeypatch.setattr(B, "TRACKED_CHECKS", path)
    return path


def test_a_model_whose_check_differs_from_the_tracked_copy_is_refused(state, tracked_copy_of_stub_checks):
    doc = json.loads(tracked_copy_of_stub_checks.read_text())
    doc["checks"][0]["max_abs_diff"] = 1.0
    tracked_copy_of_stub_checks.write_text(json.dumps(doc))
    with pytest.raises(B.BridgeProblem, match="not the artifact's own check"):
        B.compare(stub_models(), state)


# ------------------------------------------------------------------------------------------------- fixture

def pan(frame: float) -> float:
    """The camera of the synthetic clips swings left and right: pixels the picture is shifted by at a clip frame."""
    return 40.0 * np.sin(frame / 40.0)


def court_to_pixel(x, y, frame: float):
    return PX_PER_FT_X * x + OFF_X + pan(frame), PX_PER_FT_Y * y + OFF_Y


def pixel_to_court_h(frame: float, mirror: str = "identity") -> list:
    """The homography of a row: pixel to court feet, for the camera position at that clip frame. A build whose
    court frame is the mirror of the sensor's gets the mirrored homography."""
    ax, bx = 1 / PX_PER_FT_X, -(OFF_X + pan(frame)) / PX_PER_FT_X
    ay, by = 1 / PX_PER_FT_Y, -OFF_Y / PX_PER_FT_Y
    if mirror == "flip_xy":
        ax, bx, ay, by = -ax, S.COURT_LENGTH_FT - bx, -ay, S.COURT_WIDTH_FT - by
    return [ax, 0.0, bx, 0.0, ay, by, 0.0, 0.0, 1.0]


WINDOWS = (
    # name, period, clock at clip frame 0, row frames, running-clock spans (first frame, last frame), frames late
    dict(name=f"{TAG}_s00_w00", period=1, clock0=699.0, rows=range(0, 606, STRIDE), spans=((0, 180), (240, 600)), late=0),
    dict(name=f"{TAG}_s01_w00", period=2, clock0=696.64, rows=range(12, 306, STRIDE), spans=((12, 300),), late=2),
)
OFFSET_S = 0.3            # the residual clock offset of every synthetic window
SEEN = (0, 1, 3, 4, 6, 7, 9)     # of the ten players of a frame, in id order, the build sees these seven
ERRORS = np.array([[0.8, 0.4], [-1.1, 0.6], [0.5, -1.4], [1.6, 1.0], [-0.7, -0.9], [0.3, 1.8], [-1.9, 0.2]])


def build_fixture(root: Path, late_in_offsets=None, error_scale: float = 1.0, clips: bool = True,
                  mirror: str = "identity", data_root: Path = None) -> dict:
    """A synthetic game in the DATA_ROOT layout, synthetic broadcast outputs for two windows of it, and a
    repository folder with the scorecard the bridge reads. Returns the folders and the tables. With `data_root`
    the game is the held-out game of a data root that already exists, and only the broadcast side is made."""
    season, cv_root, repo = "2015-16", root / "cv", root / "repo"
    for folder in (cv_root / "build", cv_root / "build_r23", cv_root / "truth", repo / "reports" / "scorecard"):
        folder.mkdir(parents=True, exist_ok=True)
    if data_root is None:
        game = FX.synth_game(3, n_possessions=10, heldout=True, sportvu_game=GAME_ID)
        data_root = root / "data_root"
        lane, charts = data_root / "analytics", data_root / "collect" / "bref_shots" / "heldout" / "shotchart"
        for folder in (data_root / "sportvu_contract" / "heldout", lane / "possessions" / "heldout" / season, charts,
                       lane / "shot_frames" / "heldout" / season, lane / "xfg" / season, lane / "epv" / "heldout" / season,
                       lane / "ghosting" / "heldout" / season):
            folder.mkdir(parents=True, exist_ok=True)
        game["frames"].to_parquet(data_root / "sportvu_contract" / "heldout" / f"{GAME_ID}.parquet", index=False)
        game["possessions"].to_parquet(lane / "possessions" / "heldout" / season / f"{GAME_ID}.parquet", index=False)
        game["shot_frames"].to_parquet(lane / "shot_frames" / "heldout" / season / f"{GAME_ID}.parquet", index=False)
        game["shots"].to_parquet(charts / f"{season}.parquet", index=False)
    else:
        game = {"frames": pd.read_parquet(Path(data_root) / "sportvu_contract" / "heldout" / f"{GAME_ID}.parquet")}
    frames = game["frames"]
    (repo / "reports" / "scorecard" / f"{TAG}__r23.json").write_text(json.dumps({
        "game": TAG, "sportvu_game": GAME_ID, "build": "r23", "build_dir": "data/sportvu/build_r23", "mirror": mirror,
        "testable": 12, "truth_frames": 130, "method": {"match_ft": 5.0, "miss_ft": 2.5},
        "metrics": {"position_error": {"p50_ft": 9.9, "p90_ft": 19.9, "matched_players": 77}}}))

    level = frames.groupby("frame", sort=True).first()
    ids, period, clock = level.index.to_numpy(), level["period"].to_numpy(), level["game_clock_s"].to_numpy(dtype=float)
    plan, offsets = [], {}
    for w in WINDOWS:
        quarter = np.flatnonzero(period == w["period"])
        rows, marks = [], []
        for n, f in enumerate(w["rows"]):
            span = next((s for s in w["spans"] if s[0] <= f <= s[1]), None)
            if span is None:
                marks.append({"frame": f, "status": "unmapped"})
                rows.append({"frame": f, "state": "TRACK", "H": pixel_to_court_h(f + w["late"], mirror), "boxes": []})
                continue
            now = w["clock0"] - f / FPS + OFFSET_S
            moment = int(np.argmin(np.abs(clock[quarter] - now)))
            if abs(clock[quarter][moment] - now) > 0.12:        # as the truth step: no sensor moment near this clock
                marks.append({"frame": f, "status": "no_moment", "q": w["period"], "clock": round(now, 2)})
                rows.append({"frame": f, "state": "TRACK", "H": pixel_to_court_h(f + w["late"], mirror), "boxes": []})
                continue
            marks.append({"frame": f, "status": "ok", "q": w["period"], "clock": round(now, 2), "moment": moment})
            frame_id = ids[quarter[moment]]
            players = frames[frames["frame"] == frame_id].sort_values("player_id")
            boxes = []
            for k, j in enumerate(SEEN):
                if (k == 0 and n % 10 == 3) or (k == 1 and 20 <= n < 28):
                    continue        # one player misses a row now and then, one misses 0.8 s
                player = players.iloc[j]
                error = ERRORS[k] * error_scale * (4.5 if k == 6 else 1.0)       # the seventh box is far off
                px, py = court_to_pixel(player["x"] + error[0], player["y"] + error[1], f + w["late"])
                boxes.append({"tid": k + 1, "bbox": [int(px - BOX_W / 2), int(py - BOX_H), int(px + BOX_W / 2), int(py)],
                              "foot_raw": [px, py], "foot_stab": [px, py], "foot_corrected": False})
            px, py = court_to_pixel(47.0, -30.0, f + w["late"])                     # a referee far beyond the sideline
            boxes.append({"tid": 99, "bbox": [int(px - 7), int(py - 34), int(px + 7), int(py)], "foot_raw": [px, py],
                          "foot_stab": [px, py], "foot_corrected": False})
            rows.append({"frame": f, "state": "TRACK", "H": pixel_to_court_h(f + w["late"], mirror), "boxes": boxes})
        side = {"source": "not copied", "fps": FPS, "stride": STRIDE, "frames": rows}
        for folder in ("build", "build_r23"):
            (cv_root / folder / f"{w['name']}_frames.json").write_text(json.dumps(side))
        (cv_root / "truth" / f"{w['name']}_truth.json").write_text(json.dumps(
            {"window": {"window": w["name"], "fps": FPS, "period": w["period"]}, "offset_s": OFFSET_S, "frames": marks}))
        (cv_root / "build" / f"{w['name']}_timemap.json").write_text(json.dumps({
            "fps": FPS, "stride": STRIDE,
            "spans": [{"f0": a, "f1": b, "period": w["period"], "c0": w["clock0"] - a / FPS, "c1": w["clock0"] - b / FPS}
                      for a, b in w["spans"]]}))
        plan.append({"window": w["name"], "section": TAG, "fps": FPS, "period": w["period"]})
        late = w["late"] if late_in_offsets is None else late_in_offsets
        offsets[w["name"]] = {"offsets": [[w["rows"][0], late]], "uncalibrated": []}
        if clips:
            write_clip(cv_root / "build" / f"{w['name']}.mp4", w, rows)
    (cv_root / "build" / f"{TAG}_windows.json").write_text(json.dumps(plan))
    (cv_root / "replay_offsets.json").write_text(json.dumps(offsets))
    return {"root": root, "data_root": data_root, "cv_root": cv_root, "repo": repo, "game": game, "out": root / "site",
            "artifacts": root / "artifacts", "media": root / "media", "stills": root / "stills"}


def write_clip(path: Path, w: dict, rows: list) -> None:
    """A synthetic window clip: every box of every row drawn as a patterned block that moves with its player, on
    a smooth floor that pans with the camera. The build read row `frame` on clip frame frame + late, so the
    blocks are drawn there."""
    rng = np.random.default_rng(0)
    floor = cv2.GaussianBlur(rng.integers(60, 160, size=(HEIGHT, WIDTH + 200), dtype=np.uint8), (0, 0), 9.0)
    skins = {tid: rng.integers(0, 256, size=(BOX_H + 4, BOX_W + 4, 3), dtype=np.uint8) for tid in list(range(1, 8)) + [99]}
    by_frame = {r["frame"]: r["boxes"] for r in rows}
    known = sorted(f for f, boxes in by_frame.items() if boxes)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for number in range(max(by_frame) + w["late"] + STRIDE + 31):          # half a second of floor after the last row
        shift = int(round(pan(number))) + 100
        image = cv2.cvtColor(np.ascontiguousarray(floor[:, shift:shift + WIDTH]), cv2.COLOR_GRAY2BGR)
        at = number - w["late"]                                   # the row frame this picture belongs to
        before = max((f for f in known if f <= at), default=None)
        after = min((f for f in known if f >= at), default=None)
        if before is not None and after is not None and after - before <= STRIDE:
            share = 0.0 if after == before else (at - before) / (after - before)
            later = {b["tid"]: b["bbox"] for b in by_frame[after]}
            for box in by_frame[before]:
                if box["tid"] not in later:
                    continue
                x1, y1, x2, y2 = (int(round(a + share * (b - a))) for a, b in zip(box["bbox"], later[box["tid"]]))
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, WIDTH), min(y2, HEIGHT)
                if x2 > x1 and y2 > y1:
                    image[y1:y2, x1:x2] = skins[box["tid"]][:y2 - y1, :x2 - x1]
        writer.write(image)
    writer.release()


def stored_from_stubs(fx: dict) -> None:
    """Write the stored held-out predictions of the synthetic game: what the stand-ins give on the sensor table."""
    from analytics.xfg import features as XF
    from analytics.xfg import validate as XV
    tables = B.load_tables(fx["data_root"], GAME_ID)
    shots, _ = XV.clean_shots(tables["shots"])
    shot_frames, _ = XV.clean_contract(tables["shot_frames"], "shot_frames")
    frames, _ = XV.clean_frames(tables["frames"][XV.FRAME_COLUMNS])
    features = XF.build(shots, shot_frames, {GAME_ID: frames})
    lane, season = fx["data_root"] / "analytics", "2015-16"
    stored = pd.DataFrame({"sportvu_game": GAME_ID, "event": features["event"].to_numpy(), "player_id": features["player_id"],
                           "p_location": np.nan, "p_contest": np.nan,
                           "p_selected": StubXfg().predict(features).astype(np.float32), "made": True, "fold": -1, "heldout": True})
    other = stored.assign(sportvu_game="11.11.2015.NYK.at.CHA", fold=2, heldout=False, p_selected=np.float32(0.5))
    pd.concat([other, stored], ignore_index=True).to_parquet(lane / "xfg" / season / "xfg_predictions.parquet", index=False)
    states = StubEpv().predict_states(tables["frames"], tables["possessions"], GAME_ID)
    states.assign(points_to_end=1, fold=-1, heldout=True).drop(columns=["sportvu_game"]).to_parquet(
        lane / "epv" / "heldout" / season / f"{GAME_ID}.parquet", index=False)
    ghosts = StubGhost().predict_ghosts(tables["frames"], tables["possessions"], GAME_ID)
    ghosts.assign(horizon_s=ghosts["horizon_s"].astype(np.float32), fold=-1, heldout=True).drop(columns=["sportvu_game"]).to_parquet(
        lane / "ghosting" / "heldout" / season / f"ghost_frames_{GAME_ID}.parquet", index=False)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    made = build_fixture(tmp_path_factory.mktemp("bridge"))
    stored_from_stubs(made)
    return made


@pytest.fixture(scope="module")
def state(fx):
    card = B.read_scorecard(TAG, "r23", fx["repo"])
    windows = B.load_windows(fx["cv_root"], TAG, B.build_folder(card["build_dir"]))
    return B.prepare(card, B.load_tables(fx["data_root"], GAME_ID), windows, data_root=fx["data_root"])


def run_bridge(fx: dict, monkeypatch, **options) -> dict:
    monkeypatch.setenv("DATA_ROOT", str(fx["data_root"]))
    options.setdefault("models", stub_models())
    options.setdefault("clips", False)
    return B.run(fx["cv_root"], fx["out"], artifacts=fx["artifacts"], media=fx["media"], stills=fx["stills"], game=TAG,
                 build="r23", data_root=fx["data_root"], repo=fx["repo"], say=lambda message: None, **options)


# ---------------------------------------------------------------------------------------------- the matcher

def test_the_local_matcher_is_the_scorecard_matcher():
    truth = pytest.importorskip("sportvu.truth")
    assert B.matcher()[0] is truth._match
    rng = np.random.default_rng(0)
    for n_boxes in (0, 1, 4, 10, 13):
        court = rng.uniform(0, 94, size=(n_boxes, 2))
        players = rng.uniform(0, 94, size=(10, 2))
        players[: min(n_boxes, 6)] = court[: min(n_boxes, 6)] + rng.normal(0, 2.0, size=(min(n_boxes, 6), 2))
        mine, theirs = B.hungarian(court, players, 5.0), truth._match(court, players, 5.0)
        assert mine == theirs
        assert len({box for box, _, _ in mine}) == len(mine) and len({player for _, player, _ in mine}) == len(mine)
        assert all(distance <= 5.0 for _, _, distance in mine)


def test_matching_keeps_pairs_inside_the_distance_and_never_shares_a_player():
    players = np.array([[10.0, 10.0], [20.0, 10.0], [50.0, 25.0]])
    court = np.array([[10.5, 10.0],      # 0.5 ft from the first player
                      [11.0, 10.0],      # 1.0 ft from the first player too: he is taken, the second is 9 ft away
                      [50.0, 29.9],      # 4.9 ft from the third player
                      [80.0, 40.0]])     # nobody near
    pairs = B.hungarian(court, players, 5.0)
    assert [(box, player) for box, player, _ in pairs] == [(0, 0), (2, 2)]
    assert pairs[0][2] == pytest.approx(0.5) and pairs[1][2] == pytest.approx(4.9)
    assert B.hungarian(court, players, 0.4) == []
    assert np.allclose(B.mirror_xy([[10.0, 5.0]], "flip_xy"), [[84.0, 45.0]])
    assert np.allclose(B.mirror_xy(B.mirror_xy([[10.0, 5.0]], "flip_x"), "flip_x"), [[10.0, 5.0]])
    with pytest.raises(B.BridgeProblem):
        B.mirror_xy([[0.0, 0.0]], "sideways")


def test_boxes_are_matched_dropped_or_left_to_the_sensor(state):
    sensor, match_ft = state.sensor, state.card["match_ft"]
    n_rows = n_unmatched_boxes = n_unseen = 0
    for w in state.windows:
        for i in np.flatnonzero(w.synced):
            pair_row, pair_ft, court = w.pair_row[i], w.pair_ft[i], w.court[i]
            rows = sensor.rows_at(int(w.pos[i]))
            hit = pair_row >= 0
            assert len(pair_row) == len(w.rows[i]["boxes"]) and len(np.unique(pair_row[hit])) == hit.sum()   # one to one
            assert np.isin(pair_row[hit], rows).all()                       # only players of the synced moment
            truth = np.column_stack([sensor.x[pair_row[hit]], sensor.y[pair_row[hit]]])
            assert np.allclose(pair_ft[hit], np.hypot(*(court[hit] - truth).T), atol=1e-6)
            assert (pair_ft[hit] <= match_ft).all() and np.isnan(pair_ft[~hit]).all()
            assert not hit[-1] and w.rows[i]["boxes"][-1]["tid"] == 99     # the referee is never a player
            if w.kept[i]:                                                   # a matched player holds the broadcast position
                assert state.used[pair_row[hit]].all()
                assert np.allclose(state.x[pair_row[hit]], court[hit, 0], atol=1e-4)
                assert np.allclose(state.y[pair_row[hit]], court[hit, 1], atol=1e-4)
            free = rows[~np.isin(rows, pair_row) & ~state.used[rows]]       # filled from the sensor data, unchanged
            assert np.array_equal(state.x[free], sensor.x[free]) and np.array_equal(state.y[free], sensor.y[free])
            n_rows, n_unmatched_boxes, n_unseen = n_rows + 1, n_unmatched_boxes + int((~hit).sum()), n_unseen + len(free)
    assert n_rows == state.sync["synced"] and n_unmatched_boxes >= n_rows and n_unseen >= 2 * n_rows
    assert state.matches["matched"] < state.matches["boxes"] < state.matches["player_slots"]
    # the court position of a box is the sensor position it was drawn from, plus the error it was given
    w = state.windows[0]
    first = int(np.flatnonzero(w.synced)[0])
    players = sensor.rows_at(int(w.pos[first]))
    players = players[np.argsort(sensor.player[players])]
    drawn = np.column_stack([sensor.x[players], sensor.y[players]])[list(SEEN[:6])] + ERRORS[:6]
    assert np.allclose(w.court[first][:6], drawn, atol=0.2)


def test_the_matching_distance_comes_from_the_scorecard_and_is_not_an_option(fx, tmp_path):
    card = B.read_scorecard(TAG, "r23", fx["repo"])
    assert card["match_ft"] == 5.0 and card["mirror"] == "identity" and B.build_folder(card["build_dir"]) == "build_r23"
    assert B.build_folder("data/sportvu/replay/r23_cam") == "replay/r23_cam"
    real = json.loads((REPO / "reports" / "scorecard" / f"{B.GAME}__{B.BUILD}.json").read_text())
    assert B.read_scorecard(B.GAME, B.BUILD)["match_ft"] == real["method"]["match_ft"]
    options = B.main.__code__.co_consts
    assert not any(isinstance(text, str) and text.startswith("--") and ("match" in text or "gap" in text) for text in options)
    broken = tmp_path / "reports" / "scorecard"
    broken.mkdir(parents=True)
    (broken / f"{TAG}__r23.json").write_text(json.dumps({"game": TAG, "sportvu_game": GAME_ID, "build": "r23",
                                                         "build_dir": "data/sportvu/build_r23", "method": {}}))
    with pytest.raises(B.BridgeProblem):
        B.read_scorecard(TAG, "r23", tmp_path)
    with pytest.raises(B.BridgeProblem):
        B.read_scorecard(TAG, "nothing", tmp_path)


def test_a_mirrored_build_is_matched_in_the_sensor_frame(state, tmp_path):
    mirrored = build_fixture(tmp_path, clips=False, mirror="flip_xy")
    card = B.read_scorecard(TAG, "r23", mirrored["repo"])
    assert card["mirror"] == "flip_xy"
    other = B.prepare(card, B.load_tables(mirrored["data_root"], GAME_ID), B.load_windows(mirrored["cv_root"], TAG, "build_r23"))
    assert other.matches == state.matches and np.array_equal(other.used, state.used)
    assert np.allclose(other.x, state.x, atol=1e-3) and np.allclose(other.y, state.y, atol=1e-3)
    wrong = dict(card, mirror="identity")                       # read as if it were not mirrored: nothing lines up
    lost = B.prepare(wrong, B.load_tables(mirrored["data_root"], GAME_ID), B.load_windows(mirrored["cv_root"], TAG, "build_r23"))
    assert lost.matches["matched"] < 0.5 * state.matches["matched"]


# ------------------------------------------------------------------------------------- the interpolation rule

def test_interpolation_is_linear_between_samples_at_most_half_a_second_apart():
    assert B.MAX_GAP_S == 0.5
    # samples at window frames 0, 24 (0.4 s later) and 60 (0.6 s after that), synced to sensor frames 100, 110, 125
    ids, xy = B.player_path([0, 24, 60], [100, 110, 125], [[10.0, 20.0], [20.0, 25.0], [50.0, 25.0]], fps=60.0)
    assert ids.tolist() == list(range(100, 111)) + [125]            # nothing between 110 and 125: the gap is too long
    assert np.allclose(xy[0], [10.0, 20.0]) and np.allclose(xy[10], [20.0, 25.0]) and np.allclose(xy[-1], [50.0, 25.0])
    assert np.allclose(xy[5], [15.0, 22.5]) and np.allclose(xy[1], [11.0, 20.5])
    # exactly half a second is interpolated, one frame more is not
    assert len(B.player_path([0, 30], [100, 112], [[0.0, 0.0], [12.0, 0.0]], fps=60.0)[0]) == 13
    assert len(B.player_path([0, 31], [100, 112], [[0.0, 0.0], [12.0, 0.0]], fps=60.0)[0]) == 2
    assert len(B.player_path([0, 30], [100, 112], [[0.0, 0.0], [12.0, 0.0]], fps=FPS)[0]) == 2       # 30 frames at 59.94
    assert len(B.player_path([0, 24], [100, 110], [[0.0, 0.0], [12.0, 0.0]], fps=FPS)[0]) == 11
    # a single sample keeps its own frame, neighbours one sensor frame apart need nothing in between
    assert B.player_path([6], [40], [[1.0, 2.0]], fps=60.0)[0].tolist() == [40]
    assert B.player_path([0, 6], [40, 41], [[1.0, 2.0], [2.0, 2.0]], fps=60.0)[0].tolist() == [40, 41]
    assert len(B.player_path([], [], np.zeros((0, 2)), fps=60.0)[0]) == 0


def test_a_short_miss_is_interpolated_and_a_long_one_is_filled(state):
    """Every player's broadcast path, checked against his own matched samples: a straight line between two
    samples at most half a second apart, the sensor position between two samples farther apart."""
    sensor = state.sensor
    short = long = 0
    for w in state.windows:
        samples = {}
        for i in np.flatnonzero(w.kept):
            for box in np.flatnonzero(w.pair_row[i] >= 0):
                row = int(w.pair_row[i][box])
                samples.setdefault(int(sensor.player[row]), []).append((int(w.frame[i]), row, w.court[i][box]))
        for player, seen in samples.items():
            for (fa, a, pa), (fb, b, pb) in zip(seen, seen[1:]):
                ids = np.arange(sensor.frame[a] + 1, sensor.frame[b])
                between = sensor.player_rows(player, ids)
                assert (between >= 0).all()
                if (fb - fa) / w.fps <= B.MAX_GAP_S:
                    share = ((ids - sensor.frame[a]) / (sensor.frame[b] - sensor.frame[a]))[:, None]
                    line = pa + share * (pb - pa)
                    assert state.used[between].all()
                    assert np.allclose(np.column_stack([state.x[between], state.y[between]]), line, atol=1e-4)
                    short += fb - fa > w.stride
                else:
                    assert not state.used[between].any()
                    assert np.array_equal(state.x[between], sensor.x[between]) and np.array_equal(state.y[between], sensor.y[between])
                    long += 1
    assert short >= 5 and long >= 2          # a one-row miss and a longer miss both happen in the synthetic build


# ---------------------------------------------------------------------------------------------- fill share

def test_fill_share_counts_every_player_frame_inside_the_windows(state):
    sensor, fill = state.sensor, B.overall_fill(state)
    inside = state.row_window >= 0
    assert fill["n_player_frames"] == int(inside.sum()) == 10 * fill["n_frames"]
    assert fill["n_filled"] == int((inside & ~state.used).sum())
    assert fill["fill_share"] == fill["n_filled"] / fill["n_player_frames"]
    assert 0.2 < fill["fill_share"] < 0.8                       # three of ten players have no box at all
    assert not state.used[~inside].any()                        # nothing outside a window is replaced
    assert np.array_equal(state.x[~inside], sensor.x[~inside]) and np.array_equal(state.y[~inside], sensor.y[~inside])
    # a window runs from its first to its last synced row, the unsynced rows in the middle included
    for w in state.windows:
        pos = w.pos[w.synced]
        assert (state.window_of[pos.min():pos.max() + 1] == w.order).all()
        assert (state.window_of == w.order).sum() == pos.max() - pos.min() + 1
    w = state.windows[0]
    gap_rows = np.flatnonzero(~w.synced)
    before, after = w.pos[gap_rows[0] - 1], w.pos[gap_rows[-1] + 1]
    r0, r1 = sensor.rows_between(int(sensor.ids[before]) + 1, int(sensor.ids[after]) - 1)
    assert r1 > r0 and (state.row_window[r0:r1] == w.order).all() and not state.used[r0:r1].any()


def test_fill_share_per_possession_is_recomputable(state):
    rows = B.possession_fill(state)
    possessions = state.tables["possessions"].set_index("possession")
    assert [r.possession for r in rows] == sorted(r.possession for r in rows) and len(rows) >= 3
    for r in rows:
        p = possessions.loc[r.possession]
        mine = (state.sensor.frame >= p["start_frame"]) & (state.sensor.frame <= p["end_frame"]) & (state.row_window >= 0)
        assert r.n_player_frames == int(mine.sum()) > 0
        assert r.n_filled == int((mine & ~state.used).sum())
        assert r.fill_share == pytest.approx(r.n_filled / r.n_player_frames, abs=1e-6)
        assert r.matched_p50_ft is None or 0 < r.matched_p50_ft <= state.card["match_ft"]
    outside = set(possessions.index) - {r.possession for r in rows}
    for possession in outside:
        p = possessions.loc[possession]
        mine = (state.sensor.frame >= p["start_frame"]) & (state.sensor.frame <= p["end_frame"])
        assert not (state.row_window[mine] >= 0).any()


# ------------------------------------------------------------------------------------------ frame mapping

def test_frame_mapping_is_checked_on_the_game_clock(fx, state):
    report = state.sync
    assert report["synced"] == sum(int(w.synced.sum()) for w in state.windows) and report["rows"] > report["synced"]
    assert report["max_clock_gap_s"] <= 0.03 and report["max_nearest_margin_s"] <= 0.011 and report["duplicates"] == 0
    for w in state.windows:
        idx = np.flatnonzero(w.synced)
        assert (state.sensor.period[w.pos[idx]] == w.period[idx]).all()
        assert np.abs(state.sensor.clock[w.pos[idx]] - w.clock[idx]).max() <= 0.03
    card = B.read_scorecard(TAG, "r23", fx["repo"])
    tables = B.load_tables(fx["data_root"], GAME_ID)
    for shift in (2, -3):                                      # a moment index that is off by a few frames
        windows = B.load_windows(fx["cv_root"], TAG, "build_r23")
        windows[0].moment[windows[0].synced] += shift
        with pytest.raises(B.BridgeProblem, match="frame mapping"):
            B.prepare(card, tables, windows)
    windows = B.load_windows(fx["cv_root"], TAG, "build_r23")
    windows[1].period[windows[1].synced] = 3                    # the wrong quarter
    with pytest.raises(B.BridgeProblem):
        B.prepare(card, tables, windows)
    windows = B.load_windows(fx["cv_root"], TAG, "build_r23")
    windows[0].offset_s += 0.2                                  # the time map no longer explains the stored clock
    with pytest.raises(B.BridgeProblem, match="time maps"):
        B.prepare(card, tables, windows)


def test_two_rows_on_one_sensor_frame_keep_the_nearer_one(fx):
    """Where the sensor clock stands still, two rows of the build can have the same nearest sensor frame. The row
    nearer to that frame on the clock is kept, the other gives no position."""
    tables = B.load_tables(fx["data_root"], GAME_ID)
    windows = B.load_windows(fx["cv_root"], TAG, "build_r23")
    w = windows[0]
    start = int(B.Sensor(tables["frames"]).moments[1][w.moment[3]])       # the sensor frame of the fourth row
    frames = tables["frames"].copy()
    clock = frames["game_clock_s"].to_numpy().copy()
    stopped = (frames["frame"] >= start) & (frames["frame"] < start + 10)
    clock[stopped.to_numpy()] = clock[(frames["frame"] == start).to_numpy()][0]   # ten frames without the clock moving
    frames["game_clock_s"] = clock
    sensor = B.Sensor(frames)
    quarter = sensor.clock[sensor.moments[1]]
    for i in np.flatnonzero(w.synced):                          # sync again, as the truth step does: the nearest clock
        w.moment[i] = int(np.argmin(np.abs(quarter - w.clock[i])))
    for i in (5, 6):                                            # too far from every sensor frame: no moment
        w.synced[i], w.moment[i], w.period[i], w.clock[i] = False, -1, 0, np.nan
    assert w.moment[3] == w.moment[4] and w.moment[7] > w.moment[4]
    report = B.sync_rows(sensor, windows, 0.12)
    assert report["duplicates"] == 1 and w.kept[3] and not w.kept[4] and w.synced[4]
    assert report["used"] == report["synced"] - 1
    assert (np.diff(w.pos[w.kept]) > 0).all()


# ------------------------------------------------------------------------------------------- the scene rule

def test_the_scene_rule_takes_the_longest_stretches_and_cuts_them():
    assert B.stretches([0, 6, 12, 18, 30, 36, 42], [True, True, True, False, True, True, True], 6) == [(0, 2), (4, 6)]
    assert B.stretches([0, 6, 18, 24], [True] * 4, 6) == [(0, 1), (2, 3)]         # a missing row ends a stretch
    assert B.stretches([0, 6, 12], [True, False, True], 6) == []                  # one row is not a stretch
    long_frames = list(range(0, 6 * 400, 6))                                     # 40 s
    inputs = [
        {"order": 0, "fps": 60.0, "stride": 6, "frame": list(range(0, 600, 6)), "synced": [True] * 100},        # 9.9 s
        {"order": 1, "fps": 60.0, "stride": 6, "frame": long_frames, "synced": [True] * 400},                   # 39.9 s
        {"order": 2, "fps": 60.0, "stride": 6, "frame": list(range(0, 600, 6)), "synced": [True] * 100},        # 9.9 s
        {"order": 3, "fps": 60.0, "stride": 6, "frame": list(range(0, 300, 6)), "synced": [True] * 50},         # 4.9 s
        {"order": 4, "fps": 60.0, "stride": 6, "frame": list(range(0, 1200, 6)), "synced": [True] * 100 + [False] + [True] * 99},
    ]
    scenes = B.choose_scenes(inputs)
    assert [s["window"] for s in scenes] == [1, 0, 2, 4]          # longest first, the earlier window on a tie
    assert len(scenes) == B.SCENES_MAX == 4
    first = scenes[0]
    assert first["first"] == 0 and first["n_frames"] == 715 and first["duration_s"] == pytest.approx(715 / 60.0)
    assert first["duration_s"] <= B.SCENE_MAX_S < (first["n_frames"] + 6) / 60.0   # one more row would pass 12 s
    assert scenes[1]["n_frames"] == 595 and scenes[3]["first"] == 0 and scenes[3]["last"] == 99
    assert all(s["duration_s"] <= B.SCENE_MAX_S <= B.CLIP_CAP_S for s in scenes)
    assert sum(s["duration_s"] for s in scenes) <= B.SCENES_MAX * B.SCENE_MAX_S <= B.TOTAL_CAP_S
    assert [s["id"] for s in scenes] == ["scene-1", "scene-2", "scene-3", "scene-4"]
    assert B.choose_scenes(inputs, max_scenes=2)[1]["window"] == 0


def test_scene_selection_reads_no_error_value(fx, state, tmp_path):
    inputs = B.scene_inputs(state.windows)
    assert all(set(w) == {"order", "fps", "stride", "frame", "synced"} for w in inputs)
    assert all(isinstance(v, (int, bool)) for w in inputs for key in ("frame", "synced") for v in w[key])
    chosen = B.choose_scenes(inputs)
    # the same game with every broadcast position three times as wrong: other errors, other matches, same scenes
    worse = build_fixture(tmp_path / "worse", error_scale=3.0)
    card = B.read_scorecard(TAG, "r23", worse["repo"])
    other = B.prepare(card, B.load_tables(worse["data_root"], GAME_ID), B.load_windows(worse["cv_root"], TAG, "build_r23"))
    assert other.matches["matched"] < state.matches["matched"]
    assert np.median(other.sample_ft) > np.median(state.sample_ft)
    assert B.choose_scenes(B.scene_inputs(other.windows)) == chosen
    # and with no position at all
    bare = [dict(w) for w in inputs]
    assert B.choose_scenes(bare) == chosen
    assert [s["window"] for s in chosen] == [0, 1, 0]              # 6.0 s, 4.8 s and 3.0 s stretches


# ------------------------------------------------------------------------------------------ the whole bridge

def test_the_bridge_writes_two_valid_files_and_states_n_and_fill_share(fx, monkeypatch):
    report = run_bridge(fx, monkeypatch)
    assert S.validate_file(fx["out"], "demo/swap.json") == [] and S.validate_file(fx["out"], "demo/frames.json") == []
    swap = S.DemoSwap.model_validate_json((fx["out"] / "demo" / "swap.json").read_text())
    assert swap.label == "illustration" and swap.game_id == GAME_ID and swap.game == TAG and swap.build == "r23"
    assert swap.match_ft == 5.0 and swap.max_gap_s == B.MAX_GAP_S and swap.n_windows == 2
    assert swap.fill_share == pytest.approx(report["fill"]["fill_share"], abs=1e-6) and 0 < swap.fill_share < 1
    assert swap.n_frames * 10 == report["fill"]["n_player_frames"]
    by_kind = {s.kind: s for s in swap.summaries}
    assert set(by_kind) == set(S.MODEL_KINDS)
    for kind, summary in by_kind.items():
        items = [i for i in swap.items if i.kind == kind]
        assert summary.available and summary.n == len(items) > 0 and summary.unit == B.UNITS[kind]
        diffs = np.array([i.abs_diff for i in items])
        assert summary.mean_abs_diff == pytest.approx(diffs.mean(), abs=1e-6)
        assert summary.median_abs_diff == pytest.approx(np.median(diffs), abs=1e-6)
        assert diffs.max() > 0                                   # the positions really moved the answer
        assert "stored held-out predictions" in summary.note
        for item in items:
            assert item.abs_diff == pytest.approx(abs(item.broadcast - item.sensor), abs=1e-9)
            assert 0.0 <= item.fill_share <= 1.0 and item.window in {w["name"] for w in WINDOWS}
        assert len({i.key for i in items}) == len(items)
    assert {c.name for c in swap.checks} == set(S.MODEL_KINDS) and all(c.passed for c in swap.checks)
    listed = {p.possession for p in swap.possessions}
    assert {i.possession for i in swap.items} <= listed
    text = " ".join(swap.method)
    for needle in ("position error only", "5 ft", "0.5 s", "filled", "not tuned", "biased low", "No outcome".lower()):
        assert needle in text or needle in text.lower(), needle
    assert f"{report['fill']['n_filled']} of {report['fill']['n_player_frames']}" in text
    paths = [s.path for s in swap.sources]
    assert f"reports/scorecard/{TAG}__r23.json" in paths and all(not p.startswith("/") for p in paths)
    assert any(p.startswith("$DATA_ROOT/analytics/xfg/") for p in paths)
    marked = json.loads((fx["artifacts"] / "demo" / "possessions.json").read_text())
    assert [p["possession"] for p in marked["possessions"]] == sorted(listed) and marked["game_id"] == GAME_ID
    assert all(p["game_id"] == GAME_ID for p in marked["possessions"])
    assert S.find_private((fx["artifacts"] / "demo" / "possessions.json").read_text()) == []
    assert report["written"]["demo/swap.json"] == (fx["out"] / "demo" / "swap.json").stat().st_size


def test_items_are_what_the_models_give_on_each_table(fx, state, monkeypatch):
    run_bridge(fx, monkeypatch)
    swap = S.DemoSwap.model_validate_json((fx["out"] / "demo" / "swap.json").read_text())
    possessions = state.tables["possessions"]
    # EPV: one item per state inside a window, sensor side equal to the stored value
    stored = pd.read_parquet(fx["data_root"] / "analytics" / "epv" / "heldout" / "2015-16" / f"{GAME_ID}.parquet")
    inside = stored[np.array([state.window_at(int(f)) >= 0 for f in stored["frame"]])]
    items = {i.key: i for i in swap.items if i.kind == "epv"}
    assert len(items) == len(inside) < len(stored)
    again = StubEpv().predict_states(state.broadcast_table, possessions, GAME_ID).set_index(["possession", "frame"])
    for row in inside.itertuples():
        item = items[f"state-{row.possession}-{row.frame}"]
        assert item.sensor == pytest.approx(row.epv_points, abs=1e-6)
        assert item.broadcast == pytest.approx(again.loc[(row.possession, row.frame), "epv_points"], abs=1e-6)
        start = int(possessions.loc[possessions["possession"] == row.possession, "start_frame"].iloc[0])
        assert item.t == pytest.approx((row.frame - start) / 25.0, abs=0.006) and item.possession == row.possession
        # the item's fill share counts the possession's states up to this one, a frame outside a window as filled
        frames = [f for f in range(start, row.frame + 1, 25)]
        rows = np.concatenate([np.arange(*state.sensor.rows_between(f, f)) for f in frames])
        assert item.fill_share == pytest.approx(float((~state.used[rows]).mean()), abs=1e-6)
    # xFG: the shot inside each window, sensor side equal to the stored value
    xfg = [i for i in swap.items if i.kind == "xfg"]
    table = pd.read_parquet(fx["data_root"] / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet")
    table = table[table["sportvu_game"] == GAME_ID].set_index("event")
    assert len(xfg) == 2 and {i.window for i in xfg} == {w["name"] for w in WINDOWS}
    for item in xfg:
        event = int(item.key.split("-")[1])
        assert item.sensor == pytest.approx(float(table.loc[event, "p_selected"]), abs=1e-6)
        release = int(state.tables["shot_frames"].set_index("event").loc[event, "release_frame"])
        assert state.window_at(release) >= 0 and state.windows[state.window_at(release)].name == item.window
    # ghosting: one item per possession, the 1.0 s rows inside the windows only; sensor is the 0 ft reference and
    # broadcast is the mean distance between the ghost fed sensor positions and the ghost fed broadcast positions
    ghosts = pd.read_parquet(fx["data_root"] / "analytics" / "ghosting" / "heldout" / "2015-16" / f"ghost_frames_{GAME_ID}.parquet")
    ghosts = ghosts[(ghosts["horizon_s"] > 0.5).to_numpy() & np.array([state.window_at(int(f)) >= 0 for f in ghosts["frame"]])]
    by_possession = {i.possession: i for i in swap.items if i.kind == "ghosting"}
    assert set(by_possession) == set(ghosts["possession"]) and all(i.t is None for i in by_possession.values())
    keys = ["possession", "frame", "def_player_id"]
    moved = StubGhost().predict_ghosts(state.broadcast_table, possessions, GAME_ID)
    moved = moved[(moved["horizon_s"] > 0.5).to_numpy()]
    both = ghosts.merge(moved, on=keys, suffixes=("_s", "_b"))
    for possession, rows in both.groupby("possession"):
        shift = np.hypot(rows["ghost_x_s"] - rows["ghost_x_b"], rows["ghost_y_s"] - rows["ghost_y_b"]).mean()
        item = by_possession[possession]
        assert item.sensor == 0.0
        assert item.broadcast == pytest.approx(shift, abs=1e-5) and item.abs_diff == pytest.approx(shift, abs=1e-5)


def test_ghosting_never_reads_the_real_defender_at_the_target_frame():
    import inspect
    source = inspect.getsource(B.ghost_swap)
    body = source.split('"""', 2)[2]
    assert "rows_of(out" not in body and "sensor.x" not in body and "state.x" not in body


def test_a_shot_counts_only_when_its_lookback_lies_inside_the_window(state):
    import dataclasses
    from analytics.xfg import features as XF
    model = stub_models(("xfg",))["xfg"]
    whole = B.xfg_swap(model, state)
    assert whole.facts["release_in_window"] == whole.facts["items"] == 2 and whole.facts["stored_gap"] <= 1e-6
    item = next(i for i in whole.items if i.window == WINDOWS[0]["name"])
    release = int(state.tables["shot_frames"].set_index("event").loc[int(item.key.split("-")[1]), "release_frame"])
    # the window now begins one second before the release: the release is inside, the touch before it is not
    cut = state.sensor.ids < release - 25
    window_of = np.where(cut & (state.window_of == 0), -1, state.window_of)
    later = dataclasses.replace(state, window_of=window_of, row_window=window_of[state.sensor.row_pos])
    fewer = B.xfg_swap(model, later)
    assert fewer.facts["release_in_window"] == 2 and fewer.facts["items"] == 1
    assert [i.window for i in fewer.items] == [WINDOWS[1]["name"]]
    # the lookback is at least the 0.4 s the speed features read, and the item's fill share covers exactly it
    features_touch = 3.36                                       # seconds the synthetic shooter held the ball
    first = release - max(XF.SPEED_FRAMES, round(features_touch * XF.FPS))
    r0, r1 = state.sensor.rows_between(first, release)
    assert item.fill_share == pytest.approx(float((~state.used[r0:r1]).mean()), abs=1e-6)
    assert (state.row_window[r0:r1] == 0).all()


def test_a_missing_model_is_said_to_be_missing_and_shows_nothing(fx, monkeypatch, tmp_path):
    run_bridge(fx, monkeypatch, models=stub_models(("xfg",)))
    swap = S.DemoSwap.model_validate_json((fx["out"] / "demo" / "swap.json").read_text())
    by_kind = {s.kind: s for s in swap.summaries}
    assert by_kind["xfg"].available and {i.kind for i in swap.items} == {"xfg"}
    for kind in ("epv", "ghosting"):
        assert not by_kind[kind].available and by_kind[kind].n == 0 and by_kind[kind].note
        assert by_kind[kind].mean_abs_diff is None and by_kind[kind].median_abs_diff is None
        assert any(line.startswith(f"{kind}: not available") for line in swap.method)
    assert [c.name for c in swap.checks] == ["xfg"]
    assert not any("epv" in s.path or "ghosting" in s.path for s in swap.sources)
    # the real loaders on an empty artifacts folder: nothing can be loaded, every reason is written down
    empty = tmp_path / "artifacts"
    empty.mkdir()
    models = {kind: B.load_model(kind, empty) for kind in S.MODEL_KINDS}
    assert all(m.predictor is None and m.note and m.check is None for m in models.values())
    assert "no rebuilt artifact" in models["xfg"].note
    run_bridge(fx, monkeypatch, models=models)
    swap = S.DemoSwap.model_validate_json((fx["out"] / "demo" / "swap.json").read_text())
    assert swap.items == [] and swap.checks == [] and not any(s.available for s in swap.summaries)
    assert 0 < swap.fill_share < 1 and len(swap.possessions) >= 3          # the fill share is stated all the same
    # a predictor without a passed check is refused
    unchecked = stub_models(("xfg",))
    unchecked["xfg"].check = dict(passed_check("xfg"), passed=False, max_abs_diff=0.5)
    with pytest.raises(B.BridgeProblem, match="passed rebuild check"):
        run_bridge(fx, monkeypatch, models=unchecked)
    assert B.rebuild_check("epv", {"name": "epv", "passed": False, "status": "stopped"}) is None
    assert B.rebuild_check("epv", passed_check("epv")).passed is True


def test_models_are_loaded_by_name_and_a_refusal_becomes_a_note(fx, monkeypatch, tmp_path):
    """The loaders of the other rebuild tools are imported by name. A stand-in module plays one that loads, one
    whose check did not pass, and one that fails with a local path in its message."""
    package = tmp_path / "standins"
    package.mkdir()
    (package / "standin_rebuild.py").write_text(
        "from tests.test_export_demo_bridge import StubEpv\n"
        "def load(artifacts=None):\n    return StubEpv()\n"
        "def refuse(artifacts=None):\n    raise RuntimeError('ghosting: the rebuild did not reproduce the frozen record')\n"
        "def leak(artifacts=None):\n    raise OSError('cannot read /Users/someone/model.pkl')\n")
    monkeypatch.syspath_prepend(str(package))
    artifacts = tmp_path / "artifacts"
    for kind, record in (("epv", passed_check("epv")),
                         ("ghosting", dict(passed_check("ghosting"), passed=False, max_abs_diff=0.4))):
        (artifacts / kind).mkdir(parents=True)
        (artifacts / kind / "check.json").write_text(json.dumps(record))
    monkeypatch.setattr(B, "LOADERS", {"xfg": ("standin_rebuild", "leak"), "epv": ("standin_rebuild", "load"),
                                       "ghosting": ("standin_rebuild", "refuse")})
    models = {kind: B.load_model(kind, artifacts) for kind in S.MODEL_KINDS}
    assert type(models["epv"].predictor).__name__ == "StubEpv" and models["epv"].note is None and models["epv"].check["passed"]
    assert models["ghosting"].predictor is None and "did not reproduce the frozen record" in models["ghosting"].note
    assert models["xfg"].predictor is None and "someone" not in models["xfg"].note and "OSError" in models["xfg"].note
    monkeypatch.setattr(B, "LOADERS", dict(B.LOADERS, xfg=("no_such_module_anywhere", "load")))
    assert "could not be imported" in B.load_model("xfg", artifacts).note
    run_bridge(fx, monkeypatch, models=models)
    swap = S.DemoSwap.model_validate_json((fx["out"] / "demo" / "swap.json").read_text())
    assert {s.kind: s.available for s in swap.summaries} == {"xfg": False, "epv": True, "ghosting": False}
    assert {c.name: c.passed for c in swap.checks} == {"epv": True, "ghosting": False}      # a failed check is shown as failed
    assert {i.kind for i in swap.items} == {"epv"} and S.find_private(json.dumps(swap.model_dump(mode="json"))) == []


def test_a_sensor_side_value_that_differs_from_the_stored_one_stops_the_run(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path)
    stored_from_stubs(fx)
    path = fx["data_root"] / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet"
    table = pd.read_parquet(path)
    held = (table["sportvu_game"] == GAME_ID).to_numpy()
    table.loc[np.flatnonzero(held)[2], "p_selected"] += np.float32(1e-4)
    table.to_parquet(path, index=False)
    with pytest.raises(B.BridgeProblem, match="do not equal the stored held-out predictions"):
        run_bridge(fx, monkeypatch, models=stub_models(("xfg",)))
    assert not (fx["out"] / "demo" / "swap.json").exists()
    table.loc[held, "fold"] = 0                                 # rows that are not held-out rows
    table.to_parquet(path, index=False)
    with pytest.raises(B.BridgeProblem):
        run_bridge(fx, monkeypatch, models=stub_models(("xfg",)))
    epv = fx["data_root"] / "analytics" / "epv" / "heldout" / "2015-16" / f"{GAME_ID}.parquet"
    pd.read_parquet(epv).iloc[:-1].to_parquet(epv, index=False)  # one stored state short
    with pytest.raises(B.BridgeProblem, match="not the stored held-out states"):
        run_bridge(fx, monkeypatch, models=stub_models(("epv",)))


def test_no_outcome_column_is_read(fx, tmp_path):
    tables = B.load_tables(fx["data_root"], GAME_ID)
    assert not tables["shots"]["made"].any() and not tables["shot_frames"]["made"].any()
    assert fx["game"]["shots"]["made"].any()                    # the labels are on disk; the bridge left them there
    assert not {"points", "end_kind"} & set(tables["possessions"].columns)
    assert "desc" not in tables["shots"].columns
    for column in ("apex_frame", "outcome_frame"):
        if column in tables["shot_frames"].columns:
            assert (tables["shot_frames"][column] == -1).all()
    for kind, banned in (("xfg", "made"), ("epv", "points_to_end")):
        assert banned not in B.stored_predictions(fx["data_root"], GAME_ID, kind).columns
    assert (tables["shots"]["sportvu_game"] == GAME_ID).all() and tables["shots"]["heldout"].all()
    holes = tmp_path / "shots.parquet"
    shots = fx["game"]["shots"].copy()
    shots["made"] = shots["made"].astype("boolean")
    shots.loc[0, "made"] = pd.NA
    shots.to_parquet(holes, index=False)
    with pytest.raises(B.BridgeProblem, match="outcome"):
        B._label_free(holes)


def test_only_a_heldout_game_and_safe_folders_are_accepted(fx, monkeypatch, tmp_path):
    card = fx["repo"] / "reports" / "scorecard" / f"{TAG}__train.json"
    card.write_text(json.dumps({"game": TAG, "sportvu_game": "11.11.2015.NYK.at.CHA", "build": "train",
                                "build_dir": "data/sportvu/build_r23", "method": {"match_ft": 5.0}}))
    monkeypatch.setenv("DATA_ROOT", str(fx["data_root"]))
    common = dict(game=TAG, data_root=fx["data_root"], repo=fx["repo"], models=stub_models(), clips=False, say=lambda m: None)
    with pytest.raises(B.BridgeProblem, match="not a held-out game"):
        B.run(fx["cv_root"], tmp_path / "site", artifacts=tmp_path / "a", build="train", **common)
    with pytest.raises(SystemExit):                             # never under DATA_ROOT
        B.run(fx["cv_root"], fx["data_root"] / "site", artifacts=tmp_path / "a", build="r23", **common)
    with pytest.raises(B.BridgeProblem, match="read only"):     # never inside the broadcast outputs
        B.run(fx["cv_root"], fx["cv_root"] / "site", artifacts=tmp_path / "a", build="r23", **common)
    with pytest.raises(B.BridgeProblem):
        B.run(tmp_path / "nowhere", tmp_path / "site", artifacts=tmp_path / "a", build="r23", **common)
    assert not (tmp_path / "site").exists()


def test_the_command_line_takes_the_lock_writes_the_files_and_reports_a_problem(fx, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.delenv(heavy.HELD_ENV, raising=False)
    monkeypatch.delenv("V3_ARTIFACTS", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(fx["data_root"]))
    out, artifacts = tmp_path / "site", tmp_path / "artifacts"
    args = ["--cv-root", str(fx["cv_root"]), "--out", str(out), "--artifacts", str(artifacts), "--no-media"]
    assert B.main(args) == 0                                     # no rebuilt model in the folder: the data still comes out
    printed = capsys.readouterr().out
    assert "heavy" in printed and "xfg: not available" in printed
    report = json.loads(printed[printed.index("\n{") + 1:printed.rindex("}") + 1])
    assert report["game_id"] == GAME_ID and report["build"] == "r23" and report["media_checked"] is None
    assert report["match_ft"] == B.read_scorecard(B.GAME, B.BUILD)["match_ft"] and report["max_gap_s"] == B.MAX_GAP_S
    assert not any(s["available"] for s in report["summaries"]) and 0 < report["fill"]["fill_share"] < 1
    assert S.validate_file(out, "demo/swap.json") == [] and S.validate_file(out, "demo/frames.json") == []
    assert (artifacts / "demo" / "possessions.json").exists() and not (artifacts / "demo" / "media_check.json").exists()
    assert B.main(args + ["--build", "nothing"]) == 1
    assert "reports/scorecard/gsw_phx_2016__nothing.json is missing" in capsys.readouterr().out
    monkeypatch.delenv("DATA_ROOT")
    assert B.main(args) == 1 and "DATA_ROOT" in capsys.readouterr().out


# -------------------------------------------------------------------------------------- frames.json and clips

def test_scenes_carry_boxes_positions_and_truth_within_the_clip_caps(fx, state, monkeypatch):
    report = run_bridge(fx, monkeypatch)
    doc = S.DemoFrames.model_validate_json((fx["out"] / "demo" / "frames.json").read_text())
    assert doc.game_id == GAME_ID and doc.build == "r23" and "no position error" in doc.scene_rule
    assert [s.id for s in doc.scenes] == ["scene-1", "scene-2", "scene-3"]
    assert [s.window for s in doc.scenes] == [WINDOWS[0]["name"], WINDOWS[1]["name"], WINDOWS[0]["name"]]
    assert all(0 < s.duration_s <= B.SCENE_MAX_S <= B.CLIP_CAP_S for s in doc.scenes)
    assert doc.total_clip_s == pytest.approx(sum(s.duration_s for s in doc.scenes), abs=0.002) and doc.total_clip_s <= B.TOTAL_CAP_S
    assert report["total_clip_s"] == doc.total_clip_s and report["media_checked"] is None
    for scene in doc.scenes:
        w = next(w for w in state.windows if w.name == scene.window)
        assert scene.clip == f"{scene.id}.mp4" and scene.still == f"stills/{scene.id}.jpg"
        assert (scene.width, scene.height) == (WIDTH, HEIGHT) and scene.fps == pytest.approx(FPS, abs=0.01)
        times = [f.t for f in scene.frames]
        assert times[0] == 0.0 and times == sorted(times) and times[-1] < scene.duration_s
        assert np.allclose(np.diff(times), STRIDE / FPS, atol=0.002)             # the build's rows, one stride apart
        assert scene.still_t in times and scene.period == int(w.period[w.synced][0])
        assert scene.duration_s == pytest.approx((round(times[-1] * FPS) + 1) / FPS, abs=0.002)
        errors = []
        for frame in scene.frames:
            assert len(frame.truth) == 10 and all(p.player_id and p.team in ("GSW", "PHX") for p in frame.truth)
            matched = [b for b in frame.boxes if b.status == "matched"]
            assert all(b.player_id is not None for b in matched)
            assert all(b.player_id is None for b in frame.boxes if b.status == "unmatched")
            assert any(b.track == 99 and b.status == "unmatched" for b in frame.boxes)       # the referee
            truth = {p.player_id: p for p in frame.truth}
            placed = {p.player_id: p for p in frame.broadcast if p.player_id is not None}
            assert set(placed) == {b.player_id for b in matched} and len(frame.broadcast) >= len(matched)
            gaps = [np.hypot(p.x - truth[i].x, p.y - truth[i].y) for i, p in placed.items()]
            assert max(gaps) <= 5.0 + 0.02 and frame.error_ft == pytest.approx(np.median(gaps), abs=0.02)
            errors += gaps
            assert all(abs(b.w - BOX_W) <= 1 and abs(b.h - BOX_H) <= 1 for b in frame.boxes)
        assert scene.n_matched == len(errors)
        assert scene.p50_error_ft == pytest.approx(np.median(errors), abs=0.02)
        assert scene.p90_error_ft == pytest.approx(np.quantile(errors, 0.9), abs=0.02)
    notes = " ".join(doc.notes)
    assert "9.9 ft" in notes and "19.9 ft" in notes and "biased low" in notes       # the scorecard value beside it


def test_the_decode_offset_places_the_boxes_and_must_belong_to_this_build(fx, state):
    scenes = B.plan_scenes(state, fx["cv_root"])
    by_window = {state.windows[s["window"]].name: s for s in scenes}
    late = by_window[WINDOWS[1]["name"]]
    assert late["late"] == 2 and late["source_first"] == int(state.windows[1].frame[late["first"]]) + 2
    assert by_window[WINDOWS[0]["name"]]["late"] == 0
    assert all(s["first"] <= s["still_row"] <= s["last"] for s in scenes)
    other = json.loads((fx["cv_root"] / "build" / f"{WINDOWS[1]['name']}_frames.json").read_text())
    path = fx["root"] / "moved" / "build"
    shutil.copytree(fx["cv_root"], fx["root"] / "moved")
    other["frames"][3]["boxes"][0]["bbox"][0] += 1                 # the reference build saw another box
    (path / f"{WINDOWS[1]['name']}_frames.json").write_text(json.dumps(other))
    with pytest.raises(B.BridgeProblem, match="calibrated"):
        B.plan_scenes(state, fx["root"] / "moved")
    state.windows[1].decode, kept = None, state.windows[1].decode
    try:
        with pytest.raises(B.BridgeProblem, match="no decode offsets"):
            B.plan_scenes(state, fx["cv_root"])
    finally:
        state.windows[1].decode = kept


def test_a_clip_is_never_wider_than_the_cap_and_never_enlarged(tmp_path):
    for (width, height), expected in (((1600, 900), (1280, 720, 0.8)), ((320, 180), (320, 180, 1.0))):
        path = tmp_path / f"{width}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        for _ in range(3):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
        writer.release()
        shape = B.clip_shape(path)
        assert (shape["width"], shape["height"], shape["scale"]) == expected and shape["source_frames"] == 3
        assert shape["fps"] == pytest.approx(30.0) and not B.index_first(path)      # as cv2 writes it: index at the end
    with pytest.raises(B.BridgeProblem):
        B.clip_shape(tmp_path / "missing.mp4")
    assert B.timemap_clock([{"f0": 0, "f1": 60, "period": 2, "c0": 100.0, "c1": 99.0}], 30) == (2, 99.5)
    assert B.timemap_clock([{"f0": 0, "f1": 60, "period": 2, "c0": 100.0, "c1": 99.0}], 61) == (None, None)


def test_motion_score_is_highest_at_the_true_alignment():
    rng = np.random.default_rng(1)
    floor = cv2.GaussianBlur(rng.integers(80, 140, size=(120, 400), dtype=np.uint8), (0, 0), 2.0)
    frames, rows = {}, []
    for n in range(120):
        image = floor.copy()
        x = 20 + 2.5 * n                                           # a block that runs to the right
        cv2.rectangle(image, (int(x), 40), (int(x) + 16, 80), 230, -1)
        frames[n] = image
        if n % 6 == 0:
            rows.append((n, [(x, 40.0, 16.0, 40.0)]))
    scores = B.motion_scores(frames, rows)
    assert scores["boxes"] > 5 and B._best(scores["scores"], largest=True) == "0"
    assert scores["scores"]["0"] > 3 * max(scores["scores"]["12"], scores["scores"]["-12"], 0.0)
    late = [(n, [(x + 2.5 * 12, y, w, h) for x, y, w, h in boxes]) for n, boxes in rows]       # boxes 12 frames ahead
    assert B._best(B.motion_scores(frames, late)["scores"], largest=True) == "12"
    assert B.motion_scores(frames, [(n, []) for n, _ in rows])["boxes"] == 0
    exact = B.nearest_shift({j: frames[10 + j] for j in (0, 20, 40)}, frames, 10)
    assert B._best(exact, largest=False) == "0" and exact["0"] == 0.0 and exact["1"] > 0.0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg and ffprobe on PATH")
def test_clips_and_stills_are_cut_where_the_boxes_are(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path / "good")
    stored_from_stubs(fx)
    report = run_bridge(fx, monkeypatch, clips=True)
    assert report["media_checked"] is True
    check = json.loads((fx["artifacts"] / "demo" / "media_check.json").read_text())
    doc = S.DemoFrames.model_validate_json((fx["out"] / "demo" / "frames.json").read_text())
    assert check["passed"] and check["total_clip_s"] <= B.TOTAL_CAP_S and S.find_private(json.dumps(check)) == []
    for scene in doc.scenes:
        clip, still = fx["media"] / scene.clip, fx["stills"] / Path(scene.still).name
        seen = B.probe(clip)
        assert seen["codec"] == "h264" and seen["audio_streams"] == 0 and seen["width"] <= B.CLIP_MAX_WIDTH
        assert seen["duration_s"] == pytest.approx(scene.duration_s, abs=1 / FPS) and seen["duration_s"] <= B.CLIP_CAP_S
        assert seen["frames"] == round(scene.duration_s * FPS) and B.index_first(clip)
        assert still.read_bytes()[:2] == b"\xff\xd8" and cv2.imread(str(still)).shape[:2] == (scene.height, scene.width)
        measured = check["scenes"][scene.id]
        assert measured["passed"] and measured["problems"] == []
        assert min(measured["cut_mean_abs_diff_by_shift"], key=measured["cut_mean_abs_diff_by_shift"].get) == "0"
        motion = measured["motion_inside_minus_around_by_shift"]
        assert max(motion, key=motion.get) == "0" and motion["0"] > 0
    assert check["scenes"]["scene-2"]["decoder_late_frames"] == 2 and check["scenes"]["scene-1"]["decoder_late_frames"] == 0
    # the same clips with a decode offset that is wrong by a third of a second: the boxes miss the players
    wrong = build_fixture(tmp_path / "wrong", late_in_offsets=22)
    stored_from_stubs(wrong)
    with pytest.raises(B.BridgeProblem, match="failed a check"):
        run_bridge(wrong, monkeypatch, clips=True)
    assert not (wrong["out"] / "demo" / "frames.json").exists()
    failed = json.loads((wrong["artifacts"] / "demo" / "media_check.json").read_text())
    assert not failed["passed"] and any("moving regions" in p for s in failed["scenes"].values() for p in s["problems"])


# ------------------------------------------------------------------------------------------- house rules

def test_the_bridge_and_its_test_hold_no_dash_no_private_string_and_no_tuning_knob():
    for path in (REPO / "export" / "demo_bridge.py", Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert chr(0x2013) not in text and chr(0x2014) not in text, path.name
    assert S.find_private((REPO / "export" / "demo_bridge.py").read_text(encoding="utf-8")) == []
    import inspect
    assert set(inspect.signature(B.run).parameters) == {"cv_root", "out", "artifacts", "media", "stills", "game", "build",
                                                        "data_root", "models", "clips", "repo", "say"}
