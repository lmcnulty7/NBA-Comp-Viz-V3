"""tools/p8_footpos.py: K1 on the position rows (docs/research/P8_footpos.md). Report only: no model runs, nothing is
built, trained or changed.

    python3 -m tools.p8_footpos measure  -> <P8 workspace>/placement/footpos.json and reports/research/p8_footpos.json

Question: P8_footbias.md's K1 (a depth shift of the foot by box height and third, in the r23 court's frame, fitted
leave one game out) did not lower the missed row. The scorer's position rows map the foot through the build's own H,
where that bias was measured. Does K1 improve them on every pool game, and at what cost to missed and ghosts?

Inputs: P8_footbias.md's (tools.p8_footbias.load_frames: the reference-platform r23 sidecars, their registered score
files, truth rows, scorecard cache, time maps, SportVU moments, mirror), stored P8-workspace files only. Pool games
only: gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015. hou_cle_2016 is not read here (its position rows stay unseen for a
later registered arm). Guards on every game, SportVU id and window.

The position rows, exactly as the registered scorer computes them (tools/p8_scorecard.py score_window, quoted):

    feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2)
    pairs = _match(ct, sv, B.MATCH_FT) if len(ct) else []
    if B.accepted(r) and r.get("H") is not None:
        Hp = np.array(r["H"], np.float64).reshape(3, 3)
        cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hp).reshape(-1, 2)
        for bi, j, _ in pairs:
            e = cp_[bi] - sv[j]
            err_of[j] = (float(np.hypot(*e)), float(e[1] if nh else -e[1]))
            rec["matched"].append({"err": ..., "toward_camera": ..., "region": third_of(sv[j][1], nh), ...})

with Ht the truth H, sv all 10 mirrored SportVU players, _match sportvu.truth._match (Hungarian, within MATCH_FT 5 ft),
nh = sportvu.eval.near_side(Ht). The rows (tools/p8_e1_score.py rows, called unchanged here): position p50 and p90 =
percentiles of err over the matched players both arms score (keys frame:sv present in both); near-field bias = the
larger of the near and far thirds' absolute median toward_camera over those players. Missed and non-referee ghosts as
P8_footbias.md (and the scorer). Paired difference corrected minus baseline; window bootstrap by tools.p8_e1_score.boot
(2,000 resamples, seed 0) over each game's windows.

Rules, fixed before any number:
- Baseline: the sidecars as built. Check: every window's baseline matched entries (keys, err, toward, region) equal
  the registered score file's CPU-side "matched" to 1e-4, and the per-game missed and ghost counts equal it (as
  load_frames already checks), or the tool stops.
- Corrected arm: K1 exactly as P8_footbias.md: fitted on the two other pool games' offset populations
  (tools.p8_footbias.fit, cells box height by third, median depth, 0 for a cell under 50 fit rows), applied by
  tools.p8_footbias.corrected_px to every box of the held-out game (q = B(p) moved along the unit vector from the
  build camera's ground point, back to the image by B^-1). The corrected pixels then go through the scorer's lines
  above unchanged: truth H for matching, missed and ghosts; the build's H for err and toward_camera.
- Check: K1's missed and ghost changes reproduce P8_footbias.md's record to 1e-5, or the tool stops.
- Reading rule, on each pool game held out:
  POSITION GAIN if the position p50 difference and the near-field bias difference are both below zero with window
     intervals wholly below zero, on every pool game.
  NON-INFERIOR if, on every pool game, the upper end of the window interval of the missed difference is at most
     +0.010 and that of the non-referee ghost difference at most +0.010.
  PASS if both hold. p90 is reported, not ruled.
- The margins are fixed knowing P8_footbias.md's K1 numbers on gsw_sac_2015 (missed +0.0084 [-0.0002, +0.0165],
  ghosts +0.0083 [+0.0031, +0.0134]): both upper ends exceed +0.010, so NON-INFERIOR cannot hold for K1 as fitted and
  PASS is out of reach. What this measurement can still decide is POSITION GAIN.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.research_testset import refuse_cv_test
from tools.p8_footbias import corrected_px, fit, load_frames, offset_rows, score_frame, scorer_court
from tools.p8_placement import POOL, check_games, rnd

WHO = "tools.p8_footpos"
MISSED_MARGIN, GHOST_MARGIN = 0.010, 0.010
MATCH_TOL = 1e-4


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def matched_of(fr: dict, px) -> dict:
    """The scorer's matched entries of one frame for the given foot pixels: {"f:j": [err, toward, region]}."""
    import cv2
    from sportvu.bench import MATCH_FT
    from sportvu.eval import near_side
    from sportvu.truth import _match
    from tools.p8_scorecard import third_of
    feet = np.asarray(px, np.float32).reshape(-1, 2)
    if not len(feet) or fr["Hb"] is None:
        return {}
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Ht"]).reshape(-1, 2)
    sv = fr["sv"]
    pairs = _match(ct, sv, MATCH_FT)
    cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), fr["Hb"]).reshape(-1, 2)
    nh = near_side(fr["Ht"])
    out = {}
    for bi, j, _ in pairs:
        e = cp_[bi] - sv[j]
        out[f"{fr['f']}:{j}"] = [float(np.hypot(*e)), float(e[1] if nh else -e[1]), third_of(sv[j][1], nh)]
    return out


def same_matched(a: dict, b: dict, tol: float = MATCH_TOL) -> bool:
    if a.keys() != b.keys():
        return False
    return all(abs(a[k][0] - b[k][0]) <= tol and abs(a[k][1] - b[k][1]) <= tol and a[k][2] == b[k][2] for k in a)


def read_rule(per_game: dict) -> dict:
    """per_game: {game: {diff: {row: d}, iv: {row: [lo, hi]}}} -> the three verdicts."""
    def below(g, row):
        d, iv = per_game[g]["diff"].get(row), per_game[g]["iv"].get(row)
        return d is not None and iv is not None and d < 0 and iv[1] < 0
    pos = all(below(g, "position_p50_ft") and below(g, "nearfield_bias_ft") for g in per_game)
    noninf = all(per_game[g]["iv"].get("missed_rate") is not None and per_game[g]["iv"]["missed_rate"][1] <= MISSED_MARGIN
                 and per_game[g]["iv"].get("ghost_rate") is not None
                 and per_game[g]["iv"]["ghost_rate"][1] <= GHOST_MARGIN for g in per_game)
    return {"position_gain": bool(pos), "non_inferior": bool(noninf), "pass": bool(pos and noninf)}


# ── run ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def window_records(game: dict, table: dict | None) -> dict:
    """{window: {"r23": baseline, "new": corrected}} in tools.p8_e1_score's window format."""
    wins = {}
    for fr in game["frames"]:
        w = wins.setdefault(fr["w"], {s: {"missed": 0, "in_frame": 0, "nr_ghosts": 0, "nr_boxes": 0, "cov_acc": 1.0,
                                          "cov_live": 1.0, "team": {"n": 0, "majority": 0, "fixed": 0}, "matched": {}}
                                      for s in ("r23", "new")})
        for side, px in (("r23", fr["px"]), ("new", corrected_px(fr, "K1", table) if table is not None else fr["px"])):
            c = score_frame(scorer_court(fr["Ht"], px), fr["sv"], fr["inside"], fr["is_ref"])
            for k in ("missed", "in_frame", "nr_ghosts", "nr_boxes"):
                w[side][k] += c[k]
            w[side]["matched"].update(matched_of(fr, px))
    return wins


def cmd_measure(args) -> dict:
    from tools.p8_e1_score import boot, diffs, rows
    from tools.p8_placement import SCORES, SIDE
    refuse_cv_test(check_games(POOL), WHO)
    games = {t: load_frames(t) for t in POOL}
    offs = {t: offset_rows(games[t])[0] for t in POOL}
    prior = json.loads((REPO / "reports" / "research" / "p8_footbias.json").read_text())
    out_games = {}
    for held in POOL:
        sf = json.loads((p8_root() / "e1" / SCORES[held]).read_text())
        table = fit([r for g in POOL if g != held for r in offs[g]])["K1"]
        wins = window_records(games[held], table)
        for w, rec in wins.items():
            if not same_matched(rec["r23"]["matched"], sf["windows"][w][SIDE[held]]["matched"]):
                raise SystemExit(f"{WHO}: {w} baseline matched entries differ from the registered score file")
        d = diffs(list(wins.values()))
        r = rows(list(wins.values()))
        iv = boot({w: [v] for w, v in wins.items()})
        ref = prior["logo"]["K1"][held]
        if abs(d["missed_rate"] - ref["net"]) > 1e-5 or abs(d["ghost_rate"] - ref["ghost"]) > 1e-5:
            raise SystemExit(f"{WHO}: {held} K1 missed or ghost change does not reproduce P8_footbias.md")
        out_games[held] = {"rows_baseline_corrected": {k: v for k, v in r.items() if k not in ("team_accuracy", "coverage")},
                           "diff": {k: v for k, v in d.items() if k not in ("team_accuracy", "coverage")},
                           "iv": {k: v for k, v in iv.items() if k not in ("team_accuracy", "coverage")},
                           "matched_both": sum(len(v["r23"]["matched"].keys() & v["new"]["matched"].keys())
                                               for v in wins.values()),
                           "matched_baseline": sum(len(v["r23"]["matched"]) for v in wins.values()),
                           "matched_corrected": sum(len(v["new"]["matched"]) for v in wins.values()),
                           "windows": len(wins)}
        print(held, json.dumps(rnd({k: out_games[held][k] for k in ("diff", "iv", "matched_both")})))
    verdict = read_rule(out_games)
    out = rnd({"what": "K1 on the position rows (docs/research/P8_footpos.md); report only; stored files; pool games "
                       "only, hou_cle_2016 not read",
               "games": out_games, "verdict": verdict,
               "rule": {"position": "p50 and near-field bias differences below zero with window intervals wholly "
                                    "below zero on every pool game",
                        "non_inferior": f"window upper end of missed <= {MISSED_MARGIN} and of ghosts <= "
                                        f"{GHOST_MARGIN} on every pool game", "boot": "tools.p8_e1_score.boot"}})
    dd = p8_root() / "placement"
    dd.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(dd / "footpos.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_footpos.json", text)
    print("verdict", verdict)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_footpos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()
