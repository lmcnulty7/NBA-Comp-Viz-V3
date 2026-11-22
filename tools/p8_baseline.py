"""tools/p8_baseline.py: the Track B training-game baseline, pooled over the R1.3 games with Mac camera truth
(PLAN.md D140; docs/research/P8_baseline.md).

    python3 -m tools.p8_baseline [--tags gsw_bkn_2015,gsw_cha_2016,gsw_sac_2015] [--build r23] [--truth camera]

Reads only what scorecard v2 already wrote (tools/p8_scorecard.py game and combine, pinned by D88) and changes none of
it. `combine` pools position and missed players with a game bootstrap; the ghost and team rows it does not pool. Here
they are pooled from each game's frozen counts (ghost boxes: non-referee ghosts over non-referee boxes; team labels:
correct over n, the frozen best A/B mapping per window), with the same game bootstrap (tools.p8_scorecard.boot_windows,
its BOOT_N and seed, games as the clusters). The frozen scorer keeps no per-window non-referee ghost or team counts, so
these two rows have no window bootstrap when pooled. gsw_ind_2016 stays apart (D98). Nothing is fitted or tuned.
"""

import argparse
import json

from tools.p8_common import atomic_write
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_baseline"
POOLED = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")


def pooled_ratio(counts: dict) -> dict:
    """counts: game -> (numerator, denominator) -> the pooled ratio and its game bootstrap."""
    from tools.p8_scorecard import boot_windows

    def stat(items):
        den = sum(d for _, d in items)
        return sum(n for n, _ in items) / den if den else None

    groups = {g: [c] for g, c in counts.items()}
    return {"value": stat(list(counts.values())), "game_bootstrap": boot_windows(groups, stat),
            "num": sum(n for n, _ in counts.values()), "den": sum(d for _, d in counts.values())}


def main(argv=None) -> dict:
    from tools import p8_frozen as F
    from tools.p8_common import p8_root
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_baseline")
    ap.add_argument("--tags", default=",".join(POOLED))
    ap.add_argument("--build", default="r23")
    ap.add_argument("--truth", default="camera")
    a = ap.parse_args(argv)
    tags = a.tags.split(",")
    refuse_cv_test(tags, WHO)
    if "gsw_ind_2016" in tags:
        raise SystemExit(f"{WHO}: gsw_ind_2016 is reported apart (D98), never pooled")
    P = F.layout(p8_root())
    reps = {t: json.loads((P["scorecards"] / f"{t}__{a.build}__{a.truth}.json").read_text()) for t in tags}
    comb = json.loads((P["scorecards"] / f"combined__{a.build}__{a.truth}.json").read_text())
    r13 = comb["groups"]["r13"]
    if sorted(r13["games"]) != sorted(tags):
        raise SystemExit(f"{WHO}: the combined file pools {r13['games']}, not {tags}; run combine with these tags")
    ghost = pooled_ratio({t: (r["frozen_rows"]["ghost_boxes"]["non_referee_ghosts"],
                              r["frozen_rows"]["ghost_boxes"]["non_referee_boxes"]) for t, r in reps.items()})
    team = pooled_ratio({t: (r["frozen_rows"]["team_labels"]["correct"], r["frozen_rows"]["team_labels"]["n"])
                         for t, r in reps.items()})
    out = {"build": a.build, "truth": a.truth, "games": tags,
           "missed_rate": {"value": r13["missed_rate"], "game_bootstrap": r13["game_bootstrap"]["missed_rate"],
                           "window_bootstrap": r13["window_bootstrap"]["missed_rate"]},
           "position_p50_ft": {"value": r13["position_p50_ft"], "game_bootstrap": r13["game_bootstrap"]["position_p50_ft"],
                               "window_bootstrap": r13["window_bootstrap"]["position_p50_ft"]},
           "position_p90_ft": {"value": r13["position_p90_ft"], "game_bootstrap": r13["game_bootstrap"]["position_p90_ft"],
                               "window_bootstrap": r13["window_bootstrap"]["position_p90_ft"]},
           "ghost_rate_non_referee": ghost, "team_accuracy": team,
           "truth_frames": {t: [r["universe"]["with_accepted_truth"], r["universe"]["live_wide_frames"]] for t, r in reps.items()},
           "note": "game bootstrap over 3 games has 10 distinct resamples; r23's court model is in sample on these games"}
    atomic_write(P["scorecards"] / f"baseline_r13__{a.build}__{a.truth}.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
