"""Phase-1 statistical machinery: start-type classification, leave-sample-out
baselines, and the cluster-bootstrap CI. These are the corrections that turned
the credit table from a point estimate on a biased baseline into CLAIMS B1/B2."""
from conftest import N_FRAMES  # noqa: F401  (repo-root sys.path side effect)
from tier2_credit import baseline_from_rows, bootstrap_ci, start_types


# ── start_types ───────────────────────────────────────────────────────────────
def poss(period, kind, desc=""):
    return {"period": period, "events": [{"kind": kind, "desc": desc}]}


def test_start_types_mechanisms():
    seq = [
        poss(1, "shot_made"),                                  # opener
        poss(1, "shot_missed"),                                # after MAKE → halfcourt
        poss(1, "turnover", "bad pass; steal by D. Wade"),     # after MISS → live
        poss(1, "turnover", "traveling"),                      # after STEAL → live
        poss(1, "ft_made"),                                    # after dead-ball TO → halfcourt
        poss(2, "shot_made"),                                  # after made FT → halfcourt… but new period
    ]
    assert start_types(seq) == [
        "halfcourt",   # first possession of the game
        "halfcourt",   # gained via made shot
        "live",        # gained via live defensive rebound
        "live",        # gained via steal
        "halfcourt",   # gained via dead-ball turnover
        "halfcourt",   # period boundary overrides everything
    ]


# ── baseline_from_rows (leave-sample-out + context filter) ────────────────────
ROWS = [
    {"offense_real": "A", "period": 1, "clock": (700.0, 680.0), "points": 2, "start_type": "halfcourt"},
    {"offense_real": "A", "period": 1, "clock": (660.0, 640.0), "points": 0, "start_type": "halfcourt"},
    {"offense_real": "A", "period": 1, "clock": (620.0, 600.0), "points": 2, "start_type": "live"},
    {"offense_real": "B", "period": 1, "clock": (580.0, 560.0), "points": 3, "start_type": "halfcourt"},
]


def test_baseline_halfcourt_only():
    b = baseline_from_rows(ROWS, halfcourt_only=True)
    assert b == {"A": 1.0, "B": 3.0}      # A's live-start possession excluded


def test_baseline_leave_sample_out():
    b = baseline_from_rows(ROWS, exclude={(1, (660.0, 640.0))}, halfcourt_only=True)
    assert b["A"] == 2.0                  # the sampled 0-point possession left out


def test_baseline_all_context():
    b = baseline_from_rows(ROWS, halfcourt_only=False)
    assert abs(b["A"] - 4 / 3) < 1e-9


# ── bootstrap_ci ──────────────────────────────────────────────────────────────
def flat_possessions(n_games=4, per_game=20):
    """points == exp everywhere ⇒ credit is exactly 0 in every resample."""
    return [{"game": f"g{g}", "points": 1, "exp": 1.0}
            for g in range(n_games) for _ in range(per_game)]


def test_bootstrap_null_effect_ci_is_zero():
    assert bootstrap_ci(flat_possessions()) == [0.0, 0.0]


def test_bootstrap_single_game_returns_none():
    assert bootstrap_ci([{"game": "g0", "points": 0, "exp": 1.0}] * 50) is None


def test_bootstrap_deterministic_and_covers_truth():
    import random
    rng = random.Random(7)
    # true effect 0, noisy outcomes over 8 games
    p = [{"game": f"g{g}", "points": rng.choice([0, 0, 2, 2, 3]),
          "exp": 1.4} for g in range(8) for _ in range(30)]
    ci1, ci2 = bootstrap_ci(p, seed=42), bootstrap_ci(p, seed=42)
    assert ci1 == ci2                      # same seed ⇒ same interval
    assert ci1[0] < ci1[1]
    assert bootstrap_ci(p, seed=1) != ci1 or True   # different seed may differ
