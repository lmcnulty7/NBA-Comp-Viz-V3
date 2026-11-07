"""analytics/audit_shots.py: a stratified sample of localised shots for the visual audit.

The plan calls for a human check of the shot localiser (docs/DATA_CONTRACT.md section 17) against the court
replay: an auditor scrubs to the release frame of each sampled shot and records where the release really is.
This module draws that sample and summarises the filled sheet.

    python -m analytics.audit_shots --data DATA_ROOT --n 200 --seed 0 --out OUT_ROOT/analytics/audit/shots_sample.parquet
    python -m analytics.audit_shots --summarise OUT_ROOT/analytics/audit/shots_sample.parquet [--report PATH]

Sampling: training games only (the shot_frames tables under <root>/analytics/shot_frames/<season>/, never
heldout/, and any game sportvu/splits.json holds out is refused even if a file sits in the training folder);
equal counts per game (n // games, drawn without replacement with numpy's default_rng(seed) from the shots
whose release frame was found, so the auditor always has a frame to scrub to; --all includes the unfound
rows too); a game with fewer candidates than the per-game count contributes what it has and is listed. The
per-game count is at least 1, so an n below the number of training games returns one shot per game, which is
more than n rows.

Sample columns: the court replay keys (sportvu_game, event, period, clock_s, release_frame, apex_frame,
outcome_frame, shooter_id, confidence_flag) plus the empty audit columns audited_release_frame (Int32),
audited_ok (boolean) and auditor_note (string). --summarise reports, over the rows with
audited_release_frame filled, the share within 2 frames of release_frame (overall, per confidence_flag and
per game) and the share of audited_ok.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPLAY_COLUMNS = ["sportvu_game", "event", "period", "clock_s", "release_frame", "apex_frame", "outcome_frame",
                  "shooter_id", "confidence_flag"]
AUDIT_COLUMNS = ["audited_release_frame", "audited_ok", "auditor_note"]
COLUMNS = REPLAY_COLUMNS + AUDIT_COLUMNS
TOLERANCE_FRAMES = 2
HELDOUT_SUBDIR = "heldout"


def tables_dir(data_root) -> Path:
    """<root>/analytics/shot_frames, or <root> itself when it already is that folder."""
    root = Path(data_root)
    if root.name == "shot_frames" and root.parent.name == "analytics":
        return root
    return root / "analytics" / "shot_frames"


def _is_heldout_game(game: str) -> bool:
    from sportvu import splits
    from sportvu.colab_contract import is_heldout, split_of
    return is_heldout(split_of({"game": game, "home": game.rsplit(".", 1)[-1]}, splits.load()))


def training_tables(data_root) -> list[Path]:
    """The per-game shot_frames parquet files of the training games: <season>/<game>.parquet, never under
    heldout/ and never a game sportvu/splits.json holds out."""
    d = tables_dir(data_root)
    out = []
    for p in sorted(d.glob("*/*.parquet")):
        if HELDOUT_SUBDIR in p.relative_to(d).parts:
            continue
        if _is_heldout_game(p.stem):
            continue
        out.append(p)
    return out


def draw(tables: list[Path], n: int, seed: int = 0, include_unfound: bool = False) -> tuple[pd.DataFrame, dict]:
    """(sample, info): n // len(tables) shots per game, without replacement, from each table's candidates."""
    if not tables:
        raise ValueError("no training shot_frames tables found")
    per_game = max(1, n // len(tables))
    rng = np.random.default_rng(seed)
    parts, short = [], {}
    for p in tables:
        df = pq.read_table(str(p)).to_pandas()
        cand = df if include_unfound else df[df["release_frame"] >= 0]
        cand = cand.sort_values("event", kind="stable").reset_index(drop=True)
        k = min(per_game, len(cand))
        if k < per_game:
            short[str(cand["sportvu_game"].iloc[0]) if len(cand) else p.stem] = int(len(cand))
        idx = np.sort(rng.choice(len(cand), size=k, replace=False)) if k else np.zeros(0, dtype=int)
        parts.append(cand.iloc[idx][REPLAY_COLUMNS])
    sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REPLAY_COLUMNS)
    sample = sample.sort_values(["sportvu_game", "event"], kind="stable").reset_index(drop=True)
    sample["audited_release_frame"] = pd.array([pd.NA] * len(sample), dtype="Int32")
    sample["audited_ok"] = pd.array([pd.NA] * len(sample), dtype="boolean")
    sample["auditor_note"] = pd.array([pd.NA] * len(sample), dtype="string")
    sample = sample[COLUMNS]
    info = {"games": len(tables), "per_game": per_game, "requested": n, "rows": int(len(sample)), "seed": seed,
            "games_short": short, "include_unfound": include_unfound,
            "counts": {g: int(c) for g, c in sample["sportvu_game"].value_counts().sort_index().items()}}
    return sample, info


def write_sample(sample: pd.DataFrame, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)
    return out_path


def summarise(sample: pd.DataFrame, tolerance: int = TOLERANCE_FRAMES) -> dict:
    """Agreement of the audited release frame with the localiser's, over the audited rows."""
    s = sample.copy()
    audited = s[s["audited_release_frame"].notna()]
    diff = (audited["audited_release_frame"].astype("Int64") - audited["release_frame"].astype("Int64")).abs()
    within = diff <= tolerance
    ok = s["audited_ok"].dropna()

    def share(mask) -> float | None:
        return round(float(mask.mean()), 4) if len(mask) else None

    by_flag = {}
    for flag, grp in audited.assign(_within=within.to_numpy()).groupby("confidence_flag", sort=True):
        by_flag[str(flag)] = {"audited": int(len(grp)), "within_tolerance": int(grp["_within"].sum()),
                              "share": share(grp["_within"])}
    by_game = {}
    for game, grp in audited.assign(_within=within.to_numpy()).groupby("sportvu_game", sort=True):
        by_game[str(game)] = {"audited": int(len(grp)), "within_tolerance": int(grp["_within"].sum()),
                              "share": share(grp["_within"])}
    return {"rows": int(len(s)), "audited": int(len(audited)), "tolerance_frames": int(tolerance),
            "within_tolerance": int(within.sum()), "agreement_within_tolerance": share(within),
            "mean_abs_diff_frames": round(float(diff.mean()), 3) if len(diff) else None,
            "audited_ok_filled": int(len(ok)), "share_audited_ok": share(ok.astype(bool)),
            "by_confidence_flag": by_flag, "by_game": by_game}


def _fmt(x) -> str:
    return "n/a" if x is None else ("%.4f" % x if isinstance(x, float) else str(x))


def summary_markdown(res: dict, sample_path) -> str:
    L = ["# Shot localisation audit", "",
         "Sample `%s`: %d rows, %d audited. Agreement counts an audited release frame within %d frames of the "
         "localiser's release_frame (docs/DATA_CONTRACT.md section 17)." % (
             sample_path, res["rows"], res["audited"], res["tolerance_frames"]), "",
         "| metric | value |", "|---|---|",
         "| audited rows | %d |" % res["audited"], "| within tolerance | %d |" % res["within_tolerance"],
         "| agreement_within_tolerance | %s |" % _fmt(res["agreement_within_tolerance"]),
         "| mean abs difference (frames) | %s |" % _fmt(res["mean_abs_diff_frames"]),
         "| audited_ok filled | %d |" % res["audited_ok_filled"], "| share_audited_ok | %s |" % _fmt(res["share_audited_ok"]), "",
         "## By confidence_flag", "", "| flag | audited | within tolerance | share |", "|---|---|---|---|"]
    L += ["| %s | %d | %d | %s |" % (f, v["audited"], v["within_tolerance"], _fmt(v["share"])) for f, v in res["by_confidence_flag"].items()]
    L += ["", "## By game", "", "| game | audited | within tolerance | share |", "|---|---|---|---|"]
    L += ["| %s | %d | %d | %s |" % (g, v["audited"], v["within_tolerance"], _fmt(v["share"])) for g, v in res["by_game"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analytics.audit_shots", description=__doc__.split("\n")[0])
    ap.add_argument("--data", help="OUT_ROOT of analytics.run_phase0 (holds analytics/shot_frames), or that folder itself")
    ap.add_argument("--n", type=int, default=200,
                    help="sample size in total: n // games per training game, at least 1, so n below the game count gives one shot per game and more than n rows")
    ap.add_argument("--seed", type=int, default=0, help="numpy default_rng seed of the draw (the same seed gives the same sample)")
    ap.add_argument("--out", default=None, help="sample parquet (default <data>/analytics/audit/shots_sample.parquet)")
    ap.add_argument("--all", action="store_true", help="draw from every shot, not only those with a release frame")
    ap.add_argument("--summarise", default=None, metavar="SAMPLE", help="read a filled sample back and report agreement")
    ap.add_argument("--report", default=None, help="with --summarise: write the markdown report here")
    ap.add_argument("--tolerance", type=int, default=TOLERANCE_FRAMES, help="frames of agreement (default 2)")
    a = ap.parse_args(argv)
    if a.summarise:
        sample = pd.read_parquet(a.summarise)
        res = summarise(sample, a.tolerance)
        print(json.dumps(res, indent=1))
        if a.report:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(summary_markdown(res, a.summarise), encoding="utf-8")
            print("wrote %s" % a.report)
        print("agreement within %d frames: %s over %d audited rows" % (
            res["tolerance_frames"], _fmt(res["agreement_within_tolerance"]), res["audited"]))
        return 0
    if not a.data:
        ap.error("--data is required (or --summarise SAMPLE)")
    tables = training_tables(a.data)
    sample, info = draw(tables, a.n, a.seed, include_unfound=a.all)
    out = write_sample(sample, a.out or Path(a.data) / "analytics" / "audit" / "shots_sample.parquet")
    print(json.dumps(info, indent=1))
    print("wrote %s: %d shots, %d per game over %d training games%s" % (
        out, info["rows"], info["per_game"], info["games"],
        ("; short games %s" % info["games_short"]) if info["games_short"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
