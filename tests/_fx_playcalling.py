"""tests/_fx_playcalling.py: test-only fixture adapter of the play calling lane (tests/test_playcalling.py).

The shared generator is analytics.fixtures (synth_game, synth_season, synth_labels, re-exported here). The lane's
synthetic inputs (season_inputs, synth_outputs and the local generator with planted PPP associations, under the
_local names) live in analytics/playcalling/synthetic.py, where production code reads them; this module re-exports
them for the tests and adds write_fixture_root, which writes the DATA_ROOT layout plus the upstream lane outputs
(analytics/actions/<season>/actions_detected/<game>.parquet, analytics/coverage/<season>/coverage_predictions.parquet,
analytics/epv/<season>/<game>.parquet) for the data-root tests. Every number derived from these tables is fixture
output, not a result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.fixtures import synth_game, synth_labels, synth_season  # noqa: F401  (the shared generator)
from analytics.playcalling.synthetic import (HELDOUT_GAME, LABELLED_AT, SEASON, is_heldout_fixture, labels_for,  # noqa: F401
                                             season_games, season_inputs, strip_fx, synth_game_local, synth_labels_local,
                                             synth_outputs, synth_season_local)


def _jsonl(path: Path, df: pd.DataFrame) -> None:
    """Append-only label lines as analytics.labels writes them (labeller_pass and labelled_at filled when absent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "labeller_pass" not in df.columns:
        df["labeller_pass"] = 1
    if "labelled_at" not in df.columns:
        df["labelled_at"] = LABELLED_AT
    with path.open("w") as fh:
        for rec in df.to_dict("records"):
            fh.write(json.dumps({k: (None if (v is pd.NA or (isinstance(v, float) and np.isnan(v))) else (int(v) if isinstance(v, np.integer) else v))
                                 for k, v in rec.items()}) + "\n")


def write_fixture_root(root, seed: int = 0, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40, shared: bool = False) -> Path:
    """Write the DATA_ROOT layout: sportvu_contract/<game>.parquet (held-out under heldout/),
    analytics/possessions/<season>/<game>.parquet (+ heldout/<season>/), labels/actions/<game>.jsonl,
    labels/screens/<game>.jsonl, and the upstream outputs in their lanes' layouts under analytics/actions,
    analytics/coverage and analytics/epv (held-out games under heldout/<season>/). shared=True draws the games from
    analytics.fixtures, shared=False from the lane's local generator."""
    root = Path(root)
    games = season_games(seed, n_games, heldout_every, n_possessions, shared)
    preds = {False: [], True: []}
    for k, g in enumerate(games):
        gseed = seed * 100 + k
        lab = labels_for(gseed, g)
        outs = synth_outputs(gseed, g, lab)
        gid = str(g["possessions"]["sportvu_game"].iloc[0])
        held = is_heldout_fixture(g, k, heldout_every)
        sub = "heldout/" if held else ""
        fp = root / "sportvu_contract" / sub / (gid + ".parquet")
        fp.parent.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(fp, index=False)
        pp = root / "analytics" / "possessions" / sub / SEASON / (gid + ".parquet")
        pp.parent.mkdir(parents=True, exist_ok=True)
        schema.cast(strip_fx(g["possessions"]), "possessions").to_parquet(pp, index=False)
        _jsonl(root / "labels" / "actions" / (gid + ".jsonl"), lab["actions"])
        _jsonl(root / "labels" / "screens" / (gid + ".jsonl"), lab["screens"])
        ap = root / "analytics" / "actions" / sub / SEASON / "actions_detected" / (gid + ".parquet")
        ap.parent.mkdir(parents=True, exist_ok=True)
        outs["actions_detected"].to_parquet(ap, index=False)
        cp = root / "analytics" / "actions" / sub / SEASON / "action_candidates" / (gid + ".parquet")
        cp.parent.mkdir(parents=True, exist_ok=True)
        outs["actions_detected"].drop(columns=["action_type"]).to_parquet(cp, index=False)      # a decoy without the needed columns
        ep = root / "analytics" / "epv" / sub / SEASON / (gid + ".parquet")
        ep.parent.mkdir(parents=True, exist_ok=True)
        outs["epv_states"].to_parquet(ep, index=False)
        preds[held].append(outs["coverage_predictions"].assign(heldout=held))
    for held, parts in preds.items():
        if parts:
            d = root / "analytics" / "coverage" / ("heldout/" if held else "") / SEASON
            d.mkdir(parents=True, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_parquet(d / "coverage_predictions.parquet", index=False)
            pd.DataFrame({"team_id": [1], "n": [1]}).to_parquet(d / "coverage_team_rates.parquet", index=False)   # a decoy table
    return root
