"""export/sample.py: a small synthetic data root that validates against the contract.

`python3 -m export.sample --out web/data` writes it. Every team, player and number is invented and the manifest
says `synthetic: true`, so the site labels all of it as fixture output. It exists so the web side can be built and
tested before, and without, the real export. Nothing here is a result.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from export import schemas as S

HZ = 5.0
SRC_FIXTURE = [S.Source(path=S.FIXTURE_SOURCE, what="synthetic fixture values, not a result", rows="not_split")]
GAMES = (
    dict(slug="fixture-game-a", game_id="00.00.0000.AAA.at.BBB", label="AAA at BBB (fixture)", date="2000-01-01",
         home="BBB", away="AAA", split="heldout_game"),
    dict(slug="fixture-game-b", game_id="00.00.0000.CCC.at.DDD", label="CCC at DDD (fixture)", date="2000-01-02",
         home="DDD", away="CCC", split="heldout_arena"),
)
RESULTS = ("shot_made", "shot_missed", "turnover", "shot_made", "free_throws", "shot_missed")
EDGES = [0.30, 0.42, 0.50, 0.60]


def r1(v: float) -> float:
    return round(float(v), 1)


def r3(v: float) -> float:
    return round(float(v), 3)


def _dump(model: S.Strict) -> str:
    return json.dumps(model.model_dump(mode="json"), separators=(",", ":"), allow_nan=False) + "\n"


def _possession(game: dict, k: int) -> S.Possession:
    n = 60 + 5 * k
    t = [round(i / HZ, 2) for i in range(n)]
    attack_x = S.RIM_X_FT[k % 2]
    sign = -1.0 if attack_x < 47 else 1.0
    offense, defense = (game["away"], game["home"]) if k % 2 == 0 else (game["home"], game["away"])
    players, ghosts = [], []
    for j in range(5):
        base_y = 8.0 + 8.5 * j
        ox = [r1(47 + sign * (8 + 28 * i / n) + 3 * math.sin(i / 9 + j)) for i in range(n)]
        oy = [r1(base_y + 4 * math.sin(i / 11 + j * 1.7)) for i in range(n)]
        dx = [r1(x + sign * 4.0) for x in ox]
        dy = [r1(y + 1.5 * math.cos(i / 7 + j)) for i, y in enumerate(oy)]
        players.append(S.PlayerTrack(player_id=100 + j, name=f"Fixture Attacker {j + 1}", jersey=str(j + 1),
                                     team=offense, offense=True, x=ox, y=oy))
        players.append(S.PlayerTrack(player_id=200 + j, name=f"Fixture Defender {j + 1}", jersey=str(j + 11),
                                     team=defense, offense=False, x=dx, y=dy))
        gx = [None if i < 7 else r1(dx[i] + sign * 1.5) for i in range(n)]
        gy = [None if i < 7 else r1(dy[i] - 1.0) for i in range(n)]
        ghosts.append(S.GhostTrack(player_id=200 + j, x=gx, y=gy))
    handler = players[0]
    ball = S.BallTrack(x=[r1(v + 1.0) for v in handler.x], y=list(handler.y),
                       z=[r1(4 + 2 * abs(math.sin(i / 3))) for i in range(n)])
    seconds = int(t[-1])
    epv = S.EpvCurve(t=[float(s) for s in range(seconds + 1)],
                     value=[r3(0.95 + 0.25 * math.sin(s / 2.5 + k)) for s in range(seconds + 1)])
    result = RESULTS[k % len(RESULTS)]
    shots = []
    if result in ("shot_made", "shot_missed"):
        i_rel = n - 8
        shots.append(S.Shot(event=10 + k, t=t[i_rel], t_source="release_frame", x=handler.x[i_rel],
                            y=handler.y[i_rel], location_source="tracking_at_release", shooter_id=100,
                            shooter="Fixture Attacker 1", made=result == "shot_made", three=k % 3 == 0,
                            xfg=r3(0.28 + 0.07 * k), has_contest=True))
    start_clock = 700.0 - 40.0 * k
    return S.Possession(
        schema_version=S.SCHEMA_VERSION, game_id=game["game_id"], game_slug=game["slug"], possession=k + 1,
        period=1 + k // 3, hz=HZ, frame0=1000 * (k + 1), t=t,
        game_clock_s=[r1(start_clock - v) for v in t], shot_clock_s=[r1(max(0.0, 24.0 - v)) for v in t],
        offense=offense, defense=defense, attack_x=attack_x, result=result,
        points={"shot_made": 3 if k % 3 == 0 else 2, "free_throws": 1}.get(result, 0), gap=False,
        players=players, ball=ball, epv=epv,
        ghosts=None if k == 2 else S.Ghosts(label="average defense", horizon_s=1.0, tracks=ghosts,
                                            dropped_points=0, mean_deviation_ft=r1(3.0 + 0.2 * k)),
        shots=shots, notes=["Synthetic fixture possession, not a result."], sources=SRC_FIXTURE)


def _index(game: dict, possessions: list) -> S.PossessionIndex:
    rows = []
    for k, p in enumerate(possessions):
        xfg = p.shots[-1].xfg if p.shots else None
        bucket = None if xfg is None else 1 + sum(xfg > e for e in EDGES)
        rows.append(S.PossessionSummary(
            possession=p.possession, period=p.period, start_clock_s=p.game_clock_s[0],
            end_clock_s=p.game_clock_s[-1], duration_s=p.t[-1], offense=p.offense, defense=p.defense,
            result=p.result, points=p.points, gap=p.gap, n_frames=len(p.t), xfg=xfg, xfg_bucket=bucket,
            epv_start=p.epv.value[0], epv_end=p.epv.value[-1], has_ghosts=p.ghosts is not None,
            featured=k in (0, 3), featured_by="rule" if k in (0, 3) else None,
            featured_cell=f"{p.result} fifth {bucket}" if k in (0, 3) else None,
            demo=game["split"] == "heldout_game" and k == 0))
    info = S.GameInfo(**game)
    return S.PossessionIndex(schema_version=S.SCHEMA_VERSION, game=info, xfg_bucket_edges=EDGES, possessions=rows,
                             notes=["Synthetic fixture game, not a result."], sources=SRC_FIXTURE)


def _est(value, lo, hi, n, unit) -> S.Estimate:
    return S.Estimate(value=value, lo=lo, hi=hi, interval="95% bootstrap over the player's games (fixture)",
                      point_estimate=False, n=n, n_unit=unit)


def _players() -> S.PlayersFile:
    views = []
    for j in range(12):
        attempts = 260 - 20 * j
        big = j % 4 != 3 and attempts >= 100
        neighbours = [S.SimilarPlayer(rank=r + 1, player_id=100 + (j + r + 1) % 12,
                                      name=f"Fixture Player {(j + r + 1) % 12 + 1}", distance=r3(0.4 + 0.3 * r))
                      for r in range(5)]
        views.append(S.PlayerView(
            player_id=100 + j, slug=f"fixture-player-{j + 1}", name=f"Fixture Player {j + 1}",
            teams=["AAA"] if j < 6 else ["CCC"], position="GFC"[j % 3], games=20 + j,
            shot_making=S.ShotMaking(
                shown=big, attempts=attempts if big else 40, min_attempts=100,
                over_expected_per_100=_est(r1(2.5 - 0.6 * j), r1(-1.0 - 0.6 * j), r1(6.0 - 0.6 * j), attempts,
                                           "attempts") if big else None,
                fg_pct=r3(0.47 - 0.004 * j) if big else None, expected_fg_pct=r3(0.45) if big else None,
                baseline_error_per_100=r1(1.5 - 0.2 * j) if big else None,
                frozen_model_residual_per_100=r1(0.2) if big else None),
            gravity=S.Gravity(shown=True, possessions=900 - 30 * j, min_possessions=300, games=20 + j,
                              gravity_ft=_est(r3(0.8 - 0.12 * j), r3(0.5 - 0.12 * j), r3(1.1 - 0.12 * j),
                                              900 - 30 * j, "possessions")),
            rebounding=S.Rebounding(shown=big, chances=700 - 25 * j if big else 120, min_chances=300,
                                    rate_per_100=_est(r1(9.0 + j), r1(7.5 + j), r1(10.5 + j), 700 - 25 * j,
                                                      "chances") if big else None,
                                    model_expected_per_100=r1(9.2 + j) if big else None),
            absence_risk=S.AbsenceRisk(shown=True, games=20 + j, min_games=10, absent=j % 4,
                                       kind="model_out_of_fold",
                                       model_probability=_est(r3(0.06 + 0.01 * j), r3(0.04 + 0.01 * j),
                                                              r3(0.09 + 0.01 * j), 20 + j, "games"),
                                       observed_rate=_est(r3((j % 4) / (20 + j)), 0.0, r3(0.2), 20 + j, "games")),
            similar=S.Similar(shown=big, available=True, possessions=1500 - 40 * j, min_possessions=300,
                              neighbours=neighbours if big else [])))
    rules = [S.MinSampleRule(metric=m, min_n=n, unit=u, reason="fixture rule")
             for m, n, u in (("shot_making", 100, "attempts"), ("gravity", 300, "possessions"),
                             ("rebounding", 300, "chances"), ("absence_risk", 10, "games"),
                             ("similar", 300, "possessions"))]
    notes = [S.MetricNote(metric=r.metric, verdict="fixture output, not a result",
                          notes=["Synthetic fixture values."], sources=SRC_FIXTURE) for r in rules]
    baseline = S.Baseline(
        name="fixture baseline", selected=False,
        ece=S.Estimate(value=0.09, lo=0.08, hi=0.1, interval="fixture interval", point_estimate=False),
        calibration=[S.CalibrationRow(lo=round(0.1 * b, 1), hi=round(0.1 * b + 0.1, 1), n=1000 - 90 * b,
                                      predicted=r3(0.1 * b + 0.05), observed=r3(0.1 * b + 0.05 + 0.02 * (b - 4)))
                     for b in range(1, 9)],
        source=SRC_FIXTURE[0])
    checks = [S.RebuildCheck(name=name, compared="fixture numbers with fixture numbers", n=2, max_abs_diff=0.0,
                             tolerance=tolerance, passed=True, unit=unit)
              for name, tolerance, unit in (("absence", 5e-5, "log loss"), ("embeddings", 5e-4, "accuracy"))]
    return S.PlayersFile(schema_version=S.SCHEMA_VERSION, season="fixture season", window="fixture window",
                         n_players=15, n_shown=12, n_hidden=3, rules=rules, metrics=notes,
                         shot_making_baseline=baseline, checks=checks, players=views, sources=SRC_FIXTURE)


def _scorecard() -> S.Scorecard:
    games = [S.ScorecardGame(game="fixture_game_a", game_id=GAMES[0]["game_id"], split="heldout_game",
                             truth_frames=1000, testable=200, untestable_share=0.8),
             S.ScorecardGame(game="fixture_game_b", game_id=GAMES[1]["game_id"], split="heldout_arena",
                             truth_frames=600, testable=90, untestable_share=0.85)]

    def rows(shift: float) -> list:
        # Plainly unreal values, so a page built on the fixture cannot pass for the real scorecard.
        spec = (("position_error", "fixture target", ("met", "not met"),
                 lambda g: {"p50_ft": r1(9.9 + 3.0 * g - shift), "p90_ft": r1(19.9 + 5.0 * g - shift)}),
                ("near_field_bias", "fixture target", ("not met", "not met"),
                 lambda g: {"near_third_ft": r1(-7.0 + shift), "far_third_ft": r1(8.0 - shift)}),
                ("missed_players", "fixture target", ("not met", "not met"), lambda g: {"rate": r3(0.77 + 0.01 * g)}),
                ("ghost_boxes", "fixture target", ("not met", "not met"), lambda g: {"rate": r3(0.66)}),
                ("team_labels", "fixture target", ("not met", "not met"), lambda g: {"accuracy": r3(0.33 - 0.07 * g)}),
                ("identity", "reported", ("reported", "reported"),
                 lambda g: {"id_switches_per_possession": r1(77.0 + 9 * g)}))
        out = []
        for key, target, status, values in spec:
            cells = []
            for g, game in enumerate(games):
                v = values(g)
                cells.append(S.ScorecardCell(
                    game=game.game, split=game.split, status=status[g],
                    passed={"met": True, "not met": False}.get(status[g]),
                    text=", ".join(f"{name} {val}" for name, val in v.items()), values=v, n=1000.0 - 300 * g,
                    n_unit="observations"))
            out.append(S.ScorecardRow(key=key, target=target, cells=cells))
        return out

    builds = [S.ScorecardBuild(build="fixture_a", role="adopted", note="Synthetic adopted build.", games=games,
                               rows=rows(0.0), caveats=["Synthetic fixture scorecard, not a result."]),
              S.ScorecardBuild(build="fixture_b", role="candidate", note="Synthetic candidate build.", games=games,
                               rows=rows(0.1), caveats=["Synthetic fixture scorecard, not a result."])]
    paired = [S.PairedDifference(game="fixture_game_a", metric="p50_ft", baseline="fixture_a", build="fixture_b",
                                 diff=0.5, lo=0.4, hi=0.6, excludes_zero=True),
              S.PairedDifference(game="fixture_game_b", metric="p50_ft", baseline="fixture_a", build="fixture_b",
                                 diff=-0.5, lo=-1.5, hi=0.5, excludes_zero=False)]
    return S.Scorecard(schema_version=S.SCHEMA_VERSION, decision="Fixture decision text.", builds=builds,
                       paired=paired, method={"match_ft": "9.0"}, sources=SRC_FIXTURE)


def _results() -> S.Results:
    def lane(name, question, verdict, cls, oof, held) -> S.LaneResult:
        return S.LaneResult(
            lane=name, question=question, selected_model="fixture model", status="fixture output, not a result",
            out_of_fold=S.MetricValue(text=f"metric {oof} [{oof - 0.01:.4f}, {oof + 0.01:.4f}]", name="metric",
                                      value=oof, lo=round(oof - 0.01, 4), hi=round(oof + 0.01, 4),
                                      point_estimate=False),
            heldout=None if held is None else S.MetricValue(text=f"metric {held} (point estimate)", name="metric",
                                                            value=held, point_estimate=True),
            verdict=verdict, verdict_source=S.FIXTURE_SOURCE, verdict_class=cls, report=S.FIXTURE_SOURCE,
            rows=[S.ReportRow(field="status", value="fixture output, not a result")])

    lanes = [lane("fixture_works", "does the fixture model work?", "works on the fixture", "works", 0.62, 0.63),
             lane("fixture_flat", "is there any signal?", "no signal beats the constant", "no_signal", 1.28, 1.32),
             lane("fixture_failed", "was this the best option?", "failed: no validated signal", "failed", 0.01,
                  -0.01),
             lane("fixture_open", "which lineups are good?", "inconclusive", "inconclusive", 0.11, None)]
    return S.Results(schema_version=S.SCHEMA_VERSION, freeze_tag="fixture", generated="2000-01-01T00:00:00Z",
                     lanes=lanes, pending=[S.PendingLane(lane="fixture_pending", reason="no result yet")],
                     notes=["Synthetic fixture table, not a result."], sources=SRC_FIXTURE)


def _demo() -> tuple:
    scenes, items, poss = [], [], []
    for s in range(2):
        frames = []
        for i in range(30):
            boxes, cv, truth = [], [], []
            for j in range(8):
                px = 120 + 130 * j + 3 * i
                py = 300 + 40 * math.sin(i / 6 + j)
                ghost = j == 7
                boxes.append(S.DemoBox(x=r1(px), y=r1(py), w=46.0, h=110.0, track=j + 1,
                                       status="unmatched" if ghost else "matched",
                                       player_id=None if ghost else 100 + j))
                if not ghost:
                    tx, ty = 20 + 8 * j + 0.4 * i, 10 + 4 * j + 2 * math.sin(i / 6 + j)
                    truth.append(S.DemoPoint(player_id=100 + j, team="AAA" if j < 4 else "BBB", x=r1(tx), y=r1(ty)))
                    cv.append(S.DemoPoint(player_id=100 + j, team="AAA" if j < 4 else "BBB", x=r1(tx + 1.2),
                                          y=r1(ty - 0.9)))
            frames.append(S.DemoFrame(t=round(i / 10, 2), boxes=boxes, broadcast=cv, truth=truth, error_ft=1.5))
        scenes.append(S.DemoScene(id=f"fixture-scene-{s + 1}", window=f"fixture_w0{s}", period=1 + s,
                                  start_clock_s=600.0 - 100 * s, clip=f"fixture-scene-{s + 1}.mp4",
                                  still=f"stills/fixture-scene-{s + 1}.jpg", still_t=1.0, duration_s=3.0, fps=10.0,
                                  width=1280, height=720, n_matched=210, p50_error_ft=1.5, p90_error_ft=4.2,
                                  frames=frames))
        poss.append(S.SwapPossession(possession=s + 1, window=f"fixture_w0{s}", n_player_frames=3000,
                                     n_filled=900 + 150 * s, fill_share=r3((900 + 150 * s) / 3000),
                                     matched_p50_ft=1.5))
        for kind, a, b in (("xfg", 0.44, 0.47), ("epv", 1.02, 0.97), ("ghosting", 3.4, 3.9)):
            items.append(S.SwapItem(kind=kind, window=f"fixture_w0{s}", possession=s + 1, key=f"{kind}-{s}", t=2.0,
                                    sensor=a, broadcast=b + 0.01 * s, abs_diff=r3(abs(b + 0.01 * s - a)),
                                    fill_share=r3((900 + 150 * s) / 3000)))
    summaries = []
    for kind, unit in (("xfg", "probability"), ("epv", "points"), ("ghosting", "ft")):
        diffs = sorted(i.abs_diff for i in items if i.kind == kind)
        summaries.append(S.SwapSummary(kind=kind, n=len(diffs), mean_abs_diff=r3(sum(diffs) / len(diffs)),
                                       median_abs_diff=r3(diffs[len(diffs) // 2]), unit=unit, available=True))
    checks = [S.RebuildCheck(name=m, compared="fixture predictions with fixture predictions", n=100, n_expected=100,
                             max_abs_diff=0.0, tolerance=1e-6, passed=True, unit=u)
              for m, u in (("xfg", "probability"), ("epv", "points"), ("ghosting", "ft"))]
    swap = S.DemoSwap(schema_version=S.SCHEMA_VERSION, label="illustration", game_id=GAMES[0]["game_id"],
                      game="fixture_game_a", build="fixture_a", n_windows=2, n_frames=60, fill_share=0.325,
                      match_ft=9.0, max_gap_s=0.5, summaries=summaries, items=items, possessions=poss, checks=checks,
                      method=["Synthetic fixture comparison, not a result."], sources=SRC_FIXTURE)
    frames = S.DemoFrames(schema_version=S.SCHEMA_VERSION, game_id=GAMES[0]["game_id"], game="fixture_game_a",
                          build="fixture_a", total_clip_s=6.0, scene_rule="Fixture rule: the first two scenes.",
                          scenes=scenes,
                          notes=["Synthetic fixture scenes, not a result."], sources=SRC_FIXTURE)
    return swap, frames


def write_sample(out) -> list[str]:
    """Write the synthetic data root to `out` and return the relative paths written."""
    out = Path(out)
    files: dict[str, tuple[str, str]] = {}

    def put(rel: str, model: S.Strict) -> None:
        text = _dump(model)
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        files[rel] = (type(model).__name__, text)

    manifest_games = []
    for game in GAMES:
        possessions = [_possession(game, k) for k in range(6)]
        index = _index(game, possessions)
        put(f"games/{game['slug']}/possessions.json", index)
        for p in possessions:
            put(f"games/{game['slug']}/{p.possession}.json", p)
        size = sum(len(text.encode()) for rel, (_, text) in files.items() if rel.startswith(f"games/{game['slug']}/"))
        manifest_games.append(S.ManifestGame(**game, n_possessions=len(possessions),
                                             n_featured=sum(r.featured for r in index.possessions), bytes=size))
    put("players.json", _players())
    put("scorecard.json", _scorecard())
    put("results.json", _results())
    swap, frames = _demo()
    put("demo/swap.json", swap)
    put("demo/frames.json", frames)

    entries = []
    for rel, (model, text) in sorted(files.items()):
        entries.append(S.ManifestFile(path=rel, model=model, count=1, bytes=len(text.encode()),
                                      sha256=hashlib.sha256(text.encode()).hexdigest(), sources=SRC_FIXTURE))
    manifest = S.Manifest(schema_version=S.SCHEMA_VERSION, build_id="fixture", generated_at="2000-01-01T00:00:00Z",
                          repo_commit="fixture", freeze_tag="fixture", synthetic=True, season="fixture season",
                          n_games=len(GAMES), n_possessions=sum(g.n_possessions for g in manifest_games),
                          games=manifest_games, files=entries)
    put("manifest.json", manifest)
    return sorted(files)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write a synthetic data root that validates against the contract.")
    parser.add_argument("--out", required=True, help="folder to write (for example web/data)")
    args = parser.parse_args(argv)
    written = write_sample(args.out)
    problems = S.validate_tree(args.out)
    for line in problems:
        print(line)
    print(f"{len(written)} synthetic files written; {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
