"""tools/p8_scorebug.py: date a channel upload by one scorebug frame (P8 Phase A, D85 (3)).

    python3 -m tools.p8_scorebug check TAG --at SECONDS --period P --clock M:SS --away N --home N [--by NAME]

The reading (period, game clock, away and home score) is taken by eye from one frame of the upload
(tools.p8_fetch frame TAG SECONDS). It is compared with the contract play-by-play of every candidate SportVU game of
the registry entry: the score after every event at or before that game clock in that period. A candidate matches
when its score equals the reading at that clock (or within MATCH_SLACK_S after it, since a basket and the scorebug
update can be a second apart). The entry becomes "confirmed" with that game when exactly one candidate matches and
"rejected" when none does (the upload is most likely another meeting of the two teams, outside the SportVU window).
The play-by-play is read for this identity check only; nothing is fitted or scored.
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from tools.p8_common import REGISTRY, channel_games, guard, load_registry
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_scorebug"
MATCH_SLACK_S = 2.0


def score_at(events: list, period: int, clock_s: float):
    """(away, home) after every event of the period with a clock at or above clock_s (the clock runs down), carrying
    the last score of earlier periods; None before the first scored event."""
    last = None
    for p, c, score in events:
        if p < period or (p == period and c >= clock_s):
            if score:
                last = score
    if last is None:
        return (0, 0)
    a, h = last.split("-")
    return int(a), int(h)


def matches(events: list, period: int, clock_s: float, away: int, home: int) -> bool:
    return any(score_at(events, period, clock_s + d) == (away, home) for d in (0.0, 1.0, MATCH_SLACK_S)) or \
        any(score_at(events, period, clock_s - d) == (away, home) for d in (1.0, MATCH_SLACK_S))


def pbp(game: str) -> list:
    import pandas as pd
    d = pd.read_parquet(Path(os.environ["DATA_ROOT"]) / "sportvu_contract" / f"{game}_pbp.parquet",
                        columns=["period", "clock_s", "score"])
    d = d.sort_values(["period", "clock_s"], ascending=[True, False], kind="stable")
    return [(int(p), float(c), s) for p, c, s in d.itertuples(index=False)]


def check(tag: str, at_s: float, period: int, clock: str, away: int, home: int, by: str) -> dict:
    refuse_cv_test([tag], WHO)
    g = channel_games()[tag]
    guard([g["video_id"], *g["candidates"]], WHO)
    m, s = clock.split(":")
    clock_s = int(m) * 60 + float(s)
    res = {c: matches(pbp(c), period, clock_s, away, home) for c in g["candidates"]}
    hit = [c for c, ok in res.items() if ok]
    rec = {"video_s": at_s, "read": {"period": period, "clock": clock, "away": away, "home": home},
           "candidates": {c: ("score matches" if ok else "score differs") for c, ok in res.items()},
           "by": by, "date": dt.date.today().isoformat(),
           "method": "one frame read by eye; score after the play-by-play events at that period and clock (tools.p8_scorebug)"}
    reg = load_registry()
    e = reg["games"][tag]
    e["scorebug_check"] = rec
    if len(hit) == 1:
        e.update(status="confirmed", sportvu=hit[0], date="%s-%s-%s" % (hit[0][6:10], hit[0][0:2], hit[0][3:5]))
    else:
        e.update(status="rejected" if not hit else "ambiguous", sportvu=None)
    reg["updated"] = dt.date.today().isoformat()
    REGISTRY.write_text(json.dumps(reg, indent=1) + "\n")
    return {"tag": tag, "status": e["status"], "sportvu": e["sportvu"], **rec}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("tag")
    c.add_argument("--at", type=float, required=True)
    c.add_argument("--period", type=int, required=True)
    c.add_argument("--clock", required=True)
    c.add_argument("--away", type=int, required=True)
    c.add_argument("--home", type=int, required=True)
    c.add_argument("--by", default="P8 measurement teammate (Claude Code), frame read by eye")
    a = ap.parse_args(argv)
    print(json.dumps(check(a.tag, a.at, a.period, a.clock, a.away, a.home, a.by), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
