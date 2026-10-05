"""sportvu/splits.py: the train and held-out splits (ROADMAP R1.4) and the guard every training script calls.

sportvu/splits.json is tracked (game ids, arena ids, video ids and counts only, so a Colab checkout has it).
`python -m sportvu.splits` writes it from data/harvest/games.json, the R1.3 paint-check keep flags and the
held-out decisions in ROADMAP.md:
  heldout_game   gsw_phx_2016 (SportVU scorecard game)
  heldout_arena  every game whose home arena is CLE (SportVU-synced or not)
  heldout_era    gsw_nyk_curry54 (2013, no SportVU; label-free indicators only)
  train          the games with split "train" in games.json (not excluded), all at GSW
and reports/sportvu_splits.{json,txt}: the per-split counts plus an audit of what the existing training
scripts read (held-out ids found in their inputs).

Training code:
    from sportvu import splits
    splits.refuse_heldout(sample_ids, "train_x")  # SystemExit if splits.json is missing or any id is held out
    rows = list(splits.train_labels())           # R1.3 labels of the train games that passed the paint check
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import config

PATH = Path(__file__).resolve().parent / "splits.json"
HELDOUT_GAME, HELDOUT_ARENA, HELDOUT_ERA = "gsw_phx_2016", "CLE", "gsw_nyk_curry54"
HELDOUT_SPLITS = ("heldout_game", "heldout_arena", "heldout_era")
LABELS = config.PROJECT_ROOT / "data" / "sportvu" / "labels"


def load() -> dict:
    if not PATH.exists():
        raise SystemExit("sportvu/splits.json is missing: run `python -m sportvu.splits` (ROADMAP R1.4). "
                         "Training scripts refuse to run without it.")
    return json.loads(PATH.read_text())


def heldout_tokens(sp: dict | None = None) -> set:
    """Every id that marks held-out data: game tags, SportVU game ids and YouTube video ids."""
    sp = sp or load()
    out = set()
    for s in HELDOUT_SPLITS:
        for tag, g in sp[s]["games"].items():
            out |= {tag} | {v for v in (g.get("sportvu"), g.get("video_id")) if v}
    return out


def heldout_hits(items, sp: dict | None = None) -> list:
    toks = heldout_tokens(sp)
    return [str(i) for i in items if any(t in str(i) for t in toks)]


def refuse_heldout(items, who: str, sp: dict | None = None) -> int:
    """Stop a training script whose training items (paths, game tags, sample ids) touch a held-out game.
    Returns how many items were checked."""
    sp = sp or load()
    items = [str(i) for i in items]
    hits = heldout_hits(items, sp)
    if hits:
        ids = sorted(t for t in heldout_tokens(sp) if any(t in h for h in hits))
        raise SystemExit("%s refuses to run: %d of %d training items belong to held-out games %s, e.g. %s. "
                         "Drop them; see sportvu/splits.json." % (who, len(hits), len(items), ids, hits[:3]))
    return len(items)


def train_games(sp: dict | None = None) -> list:
    return sorted((sp or load())["train"]["games"])


def train_labels(sp: dict | None = None):
    """R1.3 label rows of the train games with keep=true in the paint check (sportvu/label_check.py)."""
    sp = sp or load()
    for game in train_games(sp):
        paint = LABELS / (game + "_paint.jsonl")
        if not paint.exists():
            raise SystemExit("%s has no paint check (%s): run `python -m sportvu.label_check`" % (game, paint))
        keep = {(r["window"], r["frame"]) for r in map(json.loads, paint.open()) if r["keep"]}
        for f in sorted(glob.glob(str(LABELS / game / "*.jsonl"))):
            for line in open(f):
                row = json.loads(line)
                if (row["window"], row["frame"]) in keep:
                    yield row


def build() -> dict:
    games = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())
    rep = lambda p: json.loads((config.REPORTS_DIR / p).read_text()) if (config.REPORTS_DIR / p).exists() else {}
    prep = rep("sportvu_prepare.json").get("games", {})
    card = rep("scorecard/gsw_phx_2016__v3.json")
    check = {g["game"]: g for g in rep("sportvu_label_check.json").get("games", [])}
    def sv_name(g):          # the R1.2 games carry it; phx predates the field (same rule as bench.sportvu_name)
        if g.get("sportvu"):
            return g["sportvu"]
        y, m, d = g["date"].split("-")
        name = "%s.%s.%s.%s.at.%s" % (m, d, y, g["away"], g["home"])
        return name if (config.PROJECT_ROOT / "data" / "sportvu" / (name + "_moments.json")).exists() else None
    ids = lambda t, g: {"arena": g["home"], "sportvu": sv_name(g), "video_id": g.get("video_id")}
    sp = {"version": 1, "item": "ROADMAP R1.4",
          "rule": "Training reads this file, fails if it is missing (sportvu.splits.load) and refuses any item "
                  "that carries a held-out game id, SportVU id or video id (sportvu.splits.refuse_heldout). "
                  "Thresholds are chosen on train games only.",
          "train": {"arenas": [], "games": {}, "frames_rule": "R1.3 labels with keep=true in "
                    "data/sportvu/labels/<game>_paint.jsonl (sportvu.label_check)"},
          "heldout_game": {"games": {}, "frames_rule": "SportVU scorecard testable frames (reports/scorecard/gsw_phx_2016__v3.json)"},
          "heldout_arena": {"arenas": [HELDOUT_ARENA], "games": {},
                            "frames_rule": "never labelled; SportVU-synced running seconds from R1.2 (reports/sportvu_prepare.json)"},
          "heldout_era": {"games": {}, "frames_rule": "no SportVU (2013): label-free indicators only, no frame count"},
          "excluded": {}}
    for tag, g in sorted(games.items()):
        if g.get("excluded"):
            sp["excluded"][tag] = g["excluded"]
        elif tag == HELDOUT_GAME:
            sp["heldout_game"]["games"][tag] = {**ids(tag, g), "frames": card.get("testable")}
        elif tag == HELDOUT_ERA:
            sp["heldout_era"]["games"][tag] = {**ids(tag, g), "frames": None}
        elif g.get("home") == HELDOUT_ARENA:
            sp["heldout_arena"]["games"][tag] = {**ids(tag, g), "synced_running_s": prep.get(tag, {}).get("mapped_running_s")}
        elif g.get("split") == "train":
            sp["train"]["games"][tag] = {**ids(tag, g), "frames": check.get(tag, {}).get("kept"),
                                         "labels_before_paint_check": check.get(tag, {}).get("labels")}
    if g_bad := [t for t, g in sp["train"]["games"].items() if g["arena"] == HELDOUT_ARENA]:
        raise SystemExit("train games in the held-out arena: %s" % g_bad)
    sp["train"]["arenas"] = sorted({g["arena"] for g in sp["train"]["games"].values()})
    sp["train"]["frames"] = sum(g["frames"] or 0 for g in sp["train"]["games"].values())
    return sp


def audit(sp: dict) -> list:
    """Held-out ids in what the existing training scripts read. Scripts whose inputs are named without a
    game id (external sets, prototype frames) can only be checked by name."""
    root = config.PROJECT_ROOT
    out = []
    def names(p):
        p = Path(p)
        return [str(q.relative_to(root)) if q.is_relative_to(root) else str(q) for q in p.rglob("*") if q.is_file()] if p.exists() else []
    sheet = root / "data" / "gate_sheet"
    gate_items = [d["tag"] for ix in ("index.json", "index_rest.json") if (sheet / ix).exists()
                  for d in json.loads((sheet / ix).read_text())]
    from gate.labels import load_truth
    try:
        truth = [str(p) for p in load_truth().paths]
    except Exception as e:                                   # noqa: BLE001  (report, do not crash the audit)
        truth = ["<truth/ unreadable: %s>" % e]
    srcs = [("gate_ship_v2.py, gate_retrain_experiment.py", "data/gate_sheet index tags (harvest frames)", gate_items),
            ("train_gate.py", "truth/ (gate.labels.load_truth)", truth),
            ("train_court_kp.py", "config.COURT_KP_LABELS + COURT_KP_DATASET", names(config.COURT_KP_LABELS) + names(config.COURT_KP_DATASET)),
            ("train_line_seg.py", "config.LINE_DATASET", names(config.LINE_DATASET)),
            ("train_player_detector.py, colab_train_player.py", "config.PLAYER_DETECTOR_DATA (external set)", names(Path(config.PLAYER_DETECTOR_DATA).parent)),
            ("train_court_pose.py", "data/court_pose33 + data/court_pose_grid", names(root / "data" / "court_pose33") + names(root / "data" / "court_pose_grid")),
            ("R1.3 labels (sportvu.splits.train_labels)", "data/sportvu/labels kept rows", [r["game"] for r in train_labels(sp)])]
    toks = heldout_tokens(sp)
    for who, what, items in srcs:
        hits = heldout_hits(items, sp)
        by = {t: sum(1 for h in hits if t in h) for t in sorted(toks) if any(t in h for h in hits)}
        out.append({"scripts": who, "reads": what, "items": len(items), "heldout_items": len(hits), "by_id": by})
    return out


def main() -> None:
    sp = build()
    PATH.write_text(json.dumps(sp, indent=1) + "\n")
    aud = audit(sp)
    (config.REPORTS_DIR / "sportvu_splits.json").write_text(json.dumps({"splits": sp, "audit": aud}, indent=1))
    L = ["SPLITS (ROADMAP R1.4): sportvu/splits.json"]
    t = sp["train"]
    L.append("  train          arenas %s | %d games | %d frames (R1.3 labels kept by the paint check)" % (t["arenas"], len(t["games"]), t["frames"]))
    for tag, g in t["games"].items():
        L.append("                   %-16s %5s of %s labels" % (tag, g["frames"], g["labels_before_paint_check"]))
    for s in HELDOUT_SPLITS:
        for tag, g in sp[s]["games"].items():
            if "frames" in g:
                n = "%s frames" % g["frames"] if g["frames"] is not None else "no SportVU, no frame count"
            else:
                n = "%s s synced" % g["synced_running_s"] if g.get("synced_running_s") else "no SportVU (production game)"
            L.append("  %-14s %-16s arena %s | %s" % (s, tag, g["arena"], n))
    L.append("  excluded       %s" % sp["excluded"])
    L += ["", "audit: held-out ids in what the training scripts read (each script now calls refuse_heldout)"]
    for a in aud:
        L.append("  %-46s %6d items | held-out %4d %s" % (a["scripts"][:46], a["items"], a["heldout_items"], a["by_id"] or ""))
    L += ["", "caveats:",
          "  - ids are matched by name: items without a game id (external sets, clip_* and curry_* harvest clips,",
          "    prototype frames) cannot be traced to a game and pass the guard",
          "  - gate v2 (models/trained_head_v2, used to pick wide frames for R1.3 and the scorecard) was trained",
          "    before this guard on harvest frames that include held-out games (audit row above); it is not a",
          "    court or position model, but held-out frame selection by it is optimistic until a retrain without them"]
    (config.REPORTS_DIR / "sportvu_splits.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
