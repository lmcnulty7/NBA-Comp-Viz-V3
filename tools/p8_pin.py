"""tools/p8_pin.py: the pin of scorecard v2 (P8 Phase A). Pure; it reads source files and touches no game.

    python3 -m tools.p8_pin            print the current pin
    python3 -m tools.p8_pin --write    write reports/research/p8_scorecard_pin.json (the lead's call, in review)

The pin holds the sha256 of the scorer, the frozen-step wrapper and the shared P8 module, of the scorer's CONSTANTS, of
reports/research/p8_constants.json once fix-ball-tolerance has written it, of the frozen files it scores with and the
frozen constants it reads. tests/test_p8_scorecard.py fails when any of them differs from the committed pin, so a
branch that changes them cannot pass unnoticed. The pin's status is "draft" until the lead adopts it in a commit that
also changes PLAN.md; from then on every change of the pin file must come with a PLAN.md change (history test).
"""

import hashlib
import json
import sys

from tools.p8_common import REPO

P8_NO_GAMES = True
PIN_FILE = REPO / "reports" / "research" / "p8_scorecard_pin.json"
SCORER = "tools/p8_scorecard.py"
P8_CODE = ("tools/p8_scorecard.py", "tools/p8_frozen.py", "tools/p8_common.py", "tools/p8_device.py")
CONSTANTS_FILE = "reports/research/p8_constants.json"       # the ball tolerance, written once by fix-ball-tolerance
FROZEN = ("clock_reader.py", "sportvu/bench.py", "sportvu/truth.py", "sportvu/eval.py", "sportvu/sync.py", "sportvu/camera.py",
          "sportvu/autolabel.py", "sportvu/label_check.py", "sportvu/rebuild.py", "sportvu/local_sync.py",
          "sportvu/direction.py", "videoseq.py", "models/thresholds.json")


def file_sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def current() -> dict:
    from tools import p8_scorecard as S
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX, GHOST_FT, MISS_FT
    frozen_constants = {"MATCH_FT": B.MATCH_FT, "ID_MATCH_FT": B.ID_MATCH_FT, "MISS_FT": MISS_FT, "GHOST_FT": GHOST_FT,
                        "FRAME_MARGIN_PX": FRAME_MARGIN_PX, "REF_IOU": B.REF_IOU, "REFEREE_CLASS": B.REFEREE_CLASS,
                        "SHOT_RESET_S": B.SHOT_RESET_S, "MIN_POSS_FRAMES": B.MIN_POSS_FRAMES,
                        "LINE_TARGET_PX": B.LINE_TARGET_PX, "TRUTH_COURT_VALID": B.TRUTH_COURT_VALID}
    return {"scorer": {f: file_sha(f) for f in P8_CODE},
            "fixed_constants_file": {CONSTANTS_FILE: file_sha(CONSTANTS_FILE) if (REPO / CONSTANTS_FILE).exists() else None},
            "constants_sha256": hashlib.sha256(json.dumps(S.CONSTANTS, sort_keys=True).encode()).hexdigest(),
            "constants": S.CONSTANTS,
            "frozen_files": {f: file_sha(f) for f in FROZEN},
            "frozen_constants": frozen_constants}


def main(argv=None) -> int:
    pin = current()
    if "--write" in (argv or sys.argv[1:]):
        old = json.loads(PIN_FILE.read_text()) if PIN_FILE.exists() else {}
        status = old.get("status", "draft")
        PIN_FILE.write_text(json.dumps({"fixed_on": "2026-11-17", "decision": "P8_FEASIBILITY.md section 7 (4); PLAN.md D85",
                                        "status": status, **pin}, indent=1) + "\n")
    print(json.dumps(pin, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
