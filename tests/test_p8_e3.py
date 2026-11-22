"""tools.p8_e3: crop selection, the embedding arm with matched abstention, the team metrics, the paired bootstrap and
the adoption rule, on synthetic data. No model, no video."""

import numpy as np
import pytest

from tools import p8_e3 as E


def test_pick_even_spreads_over_the_track():
    assert E.pick_even(list(range(40)), 8) == [0, 6, 11, 17, 22, 28, 33, 39]
    assert E.pick_even([1, 2], 8) == [1, 2]


def _frag_embs(rng):
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    embs = {}
    for fid in range(6):
        base = a if fid < 3 else b
        embs[fid] = base + 0.05 * rng.standard_normal((4, 3))
    embs[6] = (a + b) / 2 + 0.01 * rng.standard_normal((4, 3))       # an ambiguous track
    embs[7] = a[None, :]                                              # one crop only
    return embs


def test_embedding_arm_splits_teams_and_matches_the_abstention_count():
    rng = np.random.default_rng(0)
    embs = _frag_embs(rng)
    keys = list(range(8))
    out = E.embed_teams(embs, {}, keys, n_abstain=2, min_crops=2, seed=42)
    assert out[7] is None and out[6] is None                          # no feature, then the largest ratio
    assert len({out[k] for k in range(3)}) == 1 and len({out[k] for k in range(3, 6)}) == 1
    assert out[0] != out[3]
    assert sum(v is None for v in out.values()) == 2
    # abstention never falls below the keys that have no feature
    assert sum(v is None for v in E.embed_teams(embs, {}, keys, 0, 2, 42).values()) == 1


def test_canonical_tracks_pool_their_fragments_and_too_few_fragments_abstain():
    rng = np.random.default_rng(1)
    embs = _frag_embs(rng)
    out = E.embed_teams(embs, {1: 0, 2: 0}, [0, 3, 4, 5], 0, 2, 42)
    assert out[0] is not None and out[0] != out[3]
    few = {0: embs[0], 1: embs[3]}
    assert E.embed_teams(few, {}, [0, 1], 0, 2, 42) == {0: None, 1: None}


def test_window_team_majority_and_fixed_mapping():
    rows = [(1, 0, 10)] * 6 + [(2, 1, 10)] * 3 + [(3, 1, 20)] * 1
    w = E.window_team(rows)
    assert w == {"n": 10, "majority": 9, "fixed": 7}                 # majority maps both labels to team 10
    assert E.window_team([]) == {"n": 0, "majority": 0, "fixed": 0}


def _wins(gain):
    return [{"r23": {"n": 100, "majority": 80, "fixed": 70}, "clip": {"n": 100, "majority": 80 + gain, "fixed": 75},
             "dinov2_small": {"n": 100, "majority": 80, "fixed": 70}} for _ in range(4)]


def test_paired_bootstrap_and_the_adoption_rule():
    games = {"a": _wins(8), "b": _wins(7), "c": _wins(6)}
    per_game = {g: E.game_rows(v) for g, v in games.items()}
    pooled = E.game_rows([w for v in games.values() for w in v])
    assert pooled["clip"]["diff_pts"] == pytest.approx(7.0)
    for a in E.ARMS:
        pooled[a]["game_interval"] = E.paired_boot(games, a)
        pooled[a]["window_interval"] = E.paired_boot({i: [w] for i, w in enumerate(sum(games.values(), []))}, a)
    assert pooled["clip"]["game_interval"][0] > 0
    v = E.verdict(pooled, per_game, "clip")
    assert v["adopt"] and all(v["checks"].values())
    v0 = E.verdict(pooled, per_game, "dinov2_small")
    assert not v0["adopt"] and not v0["checks"]["gain_at_least_5_pts"]
    per_game["a"]["clip"]["diff_pts"] = -1.5
    assert not E.verdict(pooled, per_game, "clip")["checks"]["no_game_down_over_1_pt"]


def test_one_group_has_no_interval_and_the_tool_guards():
    assert E.paired_boot({"a": _wins(5)}, "clip") is None
    src = open(E.__file__).read()
    assert "refuse_cv_test(" in src and "game_of(tag)" in src
    assert "gsw_ind_2016" not in E.POOLED and "hou_cle_2016" not in E.POOLED


# ── D154: the colour-on-replay arm ─────────────────────────────────────────────────────────────────────────────────
def test_matched_abstention_for_the_colour_arm():
    lab = {1: 0, 2: 1, 3: 0, 4: 1}
    ratio = {1: 0.1, 2: 0.9, 3: 0.5, 4: 0.2}
    out = E.matched_abstain(lab, ratio, [1, 2, 3, 4, 5], 3)
    assert out == {1: 0, 2: None, 3: None, 4: 1, 5: None}
    assert sum(v is None for v in E.matched_abstain(lab, ratio, [1, 2, 3, 4, 5], 0).values()) == 1


def test_order_trace_counts_a_swapped_window_as_agreeing_up_to_order():
    stored = {1: 0, 2: 1, 3: None, 4: 0}
    t = E.order_trace({1: 1, 2: 0, 3: None, 4: 1}, stored)
    assert t == {"keys": 4, "as_given": 1, "up_to_order": 4, "swapped_window": True}
    t = E.order_trace({1: 0, 2: 1, 3: None, 4: 1}, stored)
    assert t["as_given"] == 3 and t["up_to_order"] == 3 and not t["swapped_window"]


def test_confound_reading():
    below = {a: {"game_interval": [-0.2, -0.1], "window_interval": [-0.2, -0.15]} for a in E.ARMS}
    assert E.confound_reading(-1.0, below) == "embeddings worse"
    assert E.confound_reading(-6.0, below) == "comparison confounded"
    assert E.confound_reading(-3.0, below) == "inconclusive"
    mixed = {**below, "clip": {"game_interval": [-0.2, 0.01], "window_interval": [-0.2, -0.1]}}
    assert E.confound_reading(-1.0, mixed) == "inconclusive"
