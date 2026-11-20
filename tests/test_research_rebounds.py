"""rebounds-E1 machinery (docs/research/rebounds.md, the D125 amendment): the typed temperature against the frozen
single tau, recovery of known parameters, the type index, and guards on the module's source. No real data, no
held-out game, no test game; the module has no Set A path yet (D125: one is written only if development adopts)."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.rebounds import train as T
from tools import research_rebounds as R

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_rebounds.py"


def misses(n=3000, seed=0):
    """Synthetic misses: 1 none row, 5 offense and 5 defense candidates each, with a raw q per row."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n):
        for c in range(11):
            rows.append({"sportvu_game": f"g{m % 40}", "event": m, "candidate_player_id": c,
                         "is_none": int(c == 10), "is_offense": int(c < 5), "q": float(rng.beta(2, 5))})
    return pd.DataFrame(rows)


def test_equal_slopes_and_zero_offsets_give_the_frozen_single_tau():
    f = misses(200)
    key, types = T.miss_key(f), R.type_index(f)
    tau = 1.7
    frozen = T.normalise_within(np.exp(tau * np.log(np.clip(f["q"].to_numpy(), 1e-15, 1.0))), key)
    assert np.allclose(R.typed_p(f["q"], types, key, [tau, tau, tau, 0.0, 0.0]), frozen)
    assert np.allclose(R.typed_p(f["q"], types, key, [tau, 0.0, 0.0], shared=True), frozen)


def test_the_type_index_is_offense_defense_none():
    f = pd.DataFrame({"is_none": [0, 0, 1], "is_offense": [1, 0, 0]})
    assert R.type_index(f).tolist() == [0, 1, 2]


def test_fit_types_recovers_known_slopes_and_offsets():
    f = misses(3000, seed=1)
    key, types = T.miss_key(f), R.type_index(f)
    truth = [1.2, 2.0, 0.8, -0.4, 0.6]
    p = R.typed_p(f["q"], types, key, truth)
    rng = np.random.default_rng(2)
    y = np.zeros(len(f), dtype=bool)
    for m, idx in f.groupby("event").indices.items():
        y[rng.choice(idx, p=p[idx] / p[idx].sum())] = True
    fit = R.fit_types(f["q"], types, key, y, tau=1.0)
    at_truth = -float(np.log(R.typed_p(f["q"], types, key, truth)[y]).mean())
    assert fit["converged"] and fit["inner_log_loss"] <= at_truth + 1e-6        # the optimum is at least the truth
    assert fit["inner_log_loss"] <= fit["start_log_loss"]
    assert np.allclose(fit["params"], truth, atol=0.35)                             # slope and offset trade off a little


def test_the_module_lists_only_the_training_directory_and_has_no_set_a_path():
    text = SOURCE.read_text()
    for banned in ("score_once", "scoring_data_root", "research_data_root", "test_features", "list_root(",
                   "bundles_from_root(", "T.main(", "heldout_once"):
        assert banned not in text, banned
    assert "dev_games(" in text and "refuse_test(" in text and "clean_head()" in text
    tree = ast.parse(text)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("scorer", "final")]
