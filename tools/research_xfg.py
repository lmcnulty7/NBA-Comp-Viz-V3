"""tools/research_xfg.py: the xFG research experiments (PLAN.md D70, D71, D74).

    python3 -m tools.research_xfg features [--artifacts DIR]

features    Builds the frozen xFG feature frame of the 2015-16 training rows once, with the frozen code unchanged
            (analytics.xfg.validate and analytics.xfg.train.assemble, the same path tools.frozen_models uses), and
            writes it to <artifacts>/research/xfg_features_train.parquet. Held-out rows are set aside by the frozen
            assemble and are never written; their labels are blanked before any feature is built. The research
            experiments split this frame into development games and the D71 test games through
            tools.research_testset only.

Writes only under the artifacts folder (never under DATA_ROOT, never under reports/), under the heavy-job lock.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import MemoryGuard, artifacts_dir, assert_safe_output, training_games_only

WHO = "tools.research_xfg"
FEATURES_FILE = "xfg_features_train.parquet"


def research_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research")


def features_path(artifacts=None) -> Path:
    return research_dir(artifacts) / FEATURES_FILE


def build_features(artifacts=None) -> dict:
    """Build and write the training feature frame -> a summary record."""
    import os
    from analytics.xfg import train as T
    from tools.frozen_models import xfg_tables
    root = Path(os.environ["DATA_ROOT"])
    out = features_path(artifacts)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    guard = MemoryGuard()
    train, heldout = T.assemble(xfg_tables(root, guard))
    del heldout                                           # set aside by the frozen assemble; never written
    games = sorted(train["sportvu_game"].astype(str).unique())
    training_games_only(games, WHO)
    train.to_parquet(out, index=False)
    record = {"file": out.name, "rows": int(len(train)), "games": len(games),
              "with_contest": int(train["has_contest"].sum()), "seconds": round(time.time() - started, 1),
              "peak_gb": round(peak_rss_gb(), 2)}
    (out.parent / "xfg_features_train.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("features",))
    ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(argv)
    if a.command == "features":
        with heavy_job("research xfg features"):
            print(json.dumps(build_features(a.artifacts), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
