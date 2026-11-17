"""tools/genai_eval.py: the evaluation of the genai lane and the run of genai-E1 (docs/research/genai.md).

    python3 -m tools.genai_eval e1-dev      template and Haiku on the dev items -> reports/research/genai-E1-dev.json
    python3 -m tools.genai_eval e1-final    the one test score of genai-E1 -> reports/research/genai-E1-final.json

The evaluation of a text is export.genai.check_text, unchanged, against the item the text was written from: a text
passes when the checker finds no problem. Every problem is put under one rule (RULES, by the start of its message),
so a failure rate per rule can be read. Around the checker: reading grade (export.genai.fk_grade, rounded as the
site rounds it), words per text (export.genai.words_of), and variety: a sentence frame is a sentence with the item's
names and every number the checker reads blanked out, and an arm's variety is distinct frames over sentences, with
distinct word pairs over words (distinct-2, Li et al. 2016) beside it.

Intervals: a 2000-resample bootstrap of the units (players for scouting, games for possessions), seed 0, 95 percent
percentile interval. A paired difference resamples the units once and takes both arms on the same draw.

Nothing here fits, tunes or calls a model. genai-E1 reads no held-out game: possession items come from training
games (tools.genai_data), and the Haiku arm is scored on scouting items only (its possession texts are about
held-out games and are never read here).
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np

from export import genai as G
from tools import genai_data as D

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "research"
N_BOOT = 2000
SEED = 0
HAIKU = "claude-haiku-5-5"

# Each checker problem under one rule, by the start of its message (export.genai check_numbers, check_status,
# check_lanes, check_words). The order matters: the first match wins.
RULES = (
    ("model_word", r"^number .* is a model's value"),
    ("interval", r"^number .* is a value given without its interval"),
    ("point_estimate", r"^number .* is a point estimate the text does not call one"),
    ("number_value", r"^number "),
    ("number_word", r"^the (?:number word|ordinal|character) "),
    ("direction", r"^'.*' (?:links|says) "),
    ("status", r"^uses .* does not say its model's status"),
    ("lane", r"^names the .* lane"),
    ("grade", r"reads at grade"),
    ("length", r"^\d+ words, not"),
    ("style", r"^(?:jargon|forbidden claim|calls a defense|markup characters|an em dash|not one paragraph)"),
    ("no_text", r"^no text"),
)
RULE_NAMES = tuple(name for name, _ in RULES)
_RULE_RE = tuple((name, re.compile(pattern)) for name, pattern in RULES)


def rule_of(problem: str) -> str:
    for name, pattern in _RULE_RE:
        if pattern.search(problem):
            return name
    return "other"


# ------------------------------------------------------------------------------------------------ one text


def evaluate(text, item: dict) -> dict:
    """The checker's verdict on one text (None: the arm gave no text) and the measures around it."""
    if not text:
        return {"passed": False, "rules": ["no_text"], "problems": ["no text"], "grade": None, "words": None,
                "frames": []}
    problems, _numbers = G.check_text(text, item)
    rules = sorted({rule_of(p) for p in problems})
    return {"passed": not problems, "rules": rules, "problems": problems, "grade": G.round1(G.fk_grade(text)),
            "words": len(G.words_of(text)), "frames": frames_of(text, item)}


def frames_of(text: str, item: dict) -> list:
    """The sentence frames of a text: names blanked to <name>, every number the checker reads to <n>, lower case."""
    names = sorted({n for n in list(item.get("names", [])) + [item["context"].get("name") or ""] if n},
                   key=len, reverse=True)
    spans = []
    for t in G.numbers_in(text, names):
        start = t.start - 1 if t.raw[:1] in ("-", "+", G.MINUS) else t.start   # a sign sits before the digits
        spans.append((start, start + len(t.raw)))
    chars = list(text)
    for start, stop in sorted(spans, reverse=True):
        chars[start:stop] = list(" <n> ")
    masked = "".join(chars)
    for name in names:
        masked = masked.replace(name, " <name> ")
    out = []
    for sentence in G.sentences_of(masked):
        frame = " ".join(sentence.lower().split())
        if frame:
            out.append(frame)
    return out


# ------------------------------------------------------------------------------------------------ an arm


def distinct2(frames: list) -> float:
    """Distinct word pairs over the number of words, as Li et al. 2016 define distinct-2 (Table 2)."""
    pairs = [tuple(f.split()[k:k + 2]) for f in frames for k in range(max(0, len(f.split()) - 1))]
    words = sum(len(f.split()) for f in frames)
    return len(set(pairs)) / words if words else 0.0


def arm_measures(rows: list) -> dict:
    """Measures of one arm over its rows (each the output of evaluate plus `unit`)."""
    n = len(rows)
    texts = [r for r in rows if r["grade"] is not None]
    frames = [f for r in texts for f in r["frames"]]
    out = {"n_items": n, "n_texts": len(texts), "pass_rate": sum(r["passed"] for r in rows) / n if n else None,
           "rule_failure_rate": {name: sum(name in r["rules"] for r in rows) / n if n else None
                                 for name in RULE_NAMES + ("other",)},
           "grade_mean": statistics.fmean(r["grade"] for r in texts) if texts else None,
           "grade_max": max(r["grade"] for r in texts) if texts else None,
           "words_mean": statistics.fmean(r["words"] for r in texts) if texts else None,
           "words_min": min(r["words"] for r in texts) if texts else None,
           "words_max": max(r["words"] for r in texts) if texts else None,
           "sentences": len(frames), "distinct_frames": len(set(frames)),
           "frame_variety": len(set(frames)) / len(frames) if frames else None,
           "distinct_2": distinct2(frames)}
    return out


def _stat(rows: list, name: str):
    m = arm_measures(rows)
    return m[name]


def bootstrap(rows_by_arm: dict, stats: tuple, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95 percent percentile intervals of each stat of each arm, and of each paired difference (second arm minus
    first) of the first two arms, resampling units with replacement. Every arm's rows carry the same units."""
    arms = list(rows_by_arm)
    units = sorted({r["unit"] for r in rows_by_arm[arms[0]]})
    by_unit = {arm: {} for arm in arms}
    for arm in arms:
        for r in rows_by_arm[arm]:
            by_unit[arm].setdefault(r["unit"], []).append(r)
    rng = np.random.default_rng(seed)
    draws = {(arm, s): [] for arm in arms for s in stats}
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        for arm in arms:
            rows = [r for k in pick for r in by_unit[arm].get(units[k], [])]
            m = arm_measures(rows)
            for s in stats:
                draws[(arm, s)].append(m[s])
    out = {"n_boot": n_boot, "seed": seed, "units": len(units), "arms": {}, "differences": {}}
    for arm in arms:
        out["arms"][arm] = {s: _interval(draws[(arm, s)]) for s in stats}
    if len(arms) >= 2:
        a, b = arms[0], arms[1]
        for s in stats:
            diffs = [y - x for x, y in zip(draws[(a, s)], draws[(b, s)]) if x is not None and y is not None]
            out["differences"][f"{b} minus {a}: {s}"] = _interval(diffs)
    return out


def _interval(values) -> list:
    values = [v for v in values if v is not None]
    if not values:
        return [None, None]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


# ------------------------------------------------------------------------------------------------ arms of E1


def template_rows(items: list) -> tuple:
    """(rows, seconds to write and check the texts) of the template arm."""
    from tools.genai_templates import template_text
    started = time.perf_counter()
    texts = [template_text(item) for item in items]
    write_s = time.perf_counter() - started
    rows = []
    for item, text in zip(items, texts):
        row = evaluate(text, item)
        row.update(unit=unit_of(item), key=item["key"])
        rows.append(row)
    return rows, write_s


def template_speed(items: list, n: int = 1000) -> dict:
    """Seconds to write n texts (the items in turn) and to write and check them, on this machine."""
    from tools.genai_templates import template_text
    started = time.perf_counter()
    for k in range(n):
        template_text(items[k % len(items)])
    write_s = time.perf_counter() - started
    started = time.perf_counter()
    for k in range(n):
        item = items[k % len(items)]
        G.check_text(template_text(item), item)
    return {"texts": n, "write_s": write_s, "write_and_check_s": time.perf_counter() - started}


def unit_of(item: dict) -> str:
    return item["game"] if item["kind"] == "possession" else item["key"]


def haiku_rows(items: list) -> list:
    """The Haiku arm as delivered: the text the site snapshot holds for each player, or none (a dropped item)."""
    texts = json.loads(D.snapshot_file("genai/scouting.json").read_text(encoding="utf-8"))
    by_key = {t["key"]: t for t in texts["items"]}
    rows = []
    for item in items:
        text = by_key.get(item["key"])
        if text is not None and text["input_sha256"] != G.input_sha256(item):
            raise D.GenaiDataProblem(f"{item['key']}: the Haiku text was written from another input")
        row = evaluate(text["text"] if text else None, item)
        row.update(unit=unit_of(item), key=item["key"],
                   prompt_version=text["prompt_version"] if text else None)
        rows.append(row)
    return rows


def haiku_calls(items: list) -> dict:
    """Every cached Haiku call for these items (same input hash): tokens, cost at export.genai's price table, and the
    first-attempt pass rate under the pinned checker, by prompt version and call setting."""
    keys = {item["key"]: item for item in items}
    hashes = {item["key"]: G.input_sha256(item) for item in items}
    records = [r for r in D.haiku_records() if r.get("key") in keys and r.get("model") == HAIKU]
    current = [r for r in records if r["input_sha256"] == hashes[r["key"]]]
    tokens = {"input": 0, "output": 0, "calls": 0}
    groups = {}
    for r in current:
        for attempt in r["attempts"]:
            tokens["input"] += attempt["usage"]["input_tokens"]
            tokens["output"] += attempt["usage"]["output_tokens"]
            tokens["calls"] += 1
        setting = f"{r['prompt_version']} | " + G_call(r.get("call"))
        first = r["attempts"][0] if r["attempts"] else None
        ok = bool(first and first.get("text") and first.get("stop_reason") == "end_turn"
                  and not G.check_text(first["text"], keys[r["key"]])[0])
        g = groups.setdefault(setting, {"records": 0, "first_attempt_pass": 0, "kept": 0})
        g["records"] += 1
        g["first_attempt_pass"] += ok
        g["kept"] += bool(r.get("text"))
    cost = G.cost(tokens["input"], tokens["output"])
    return {"items": len(items), "records": len(current), "records_other_input": len(records) - len(current),
            "tokens": tokens, "usd": cost, "usd_per_1000_items_asked": cost / len(items) * 1000 if items else None,
            "price_per_mtok": G.PRICE_PER_MTOK, "by_setting": groups}


def G_call(call) -> str:
    return "legacy (no setting stored)" if not call else json.dumps(call, sort_keys=True)


def haiku_timing() -> dict:
    """Seconds per item of the P3 scouting runs, estimated from the creation times of consecutive cache records of
    one setting (the runs called the model one item at a time). Gaps over 120 s are pauses between runs."""
    records = D.haiku_records()
    by_setting = {}
    for r in records:
        if r.get("model") == HAIKU and r.get("created"):
            by_setting.setdefault((r["prompt_version"], G_call(r.get("call"))), []).append(
                time.mktime(time.strptime(r["created"], "%Y-%m-%dT%H:%M:%SZ")))
    gaps = []
    for stamps in by_setting.values():
        stamps.sort()
        gaps += [b - a for a, b in zip(stamps, stamps[1:]) if 0 < b - a <= 120]
    return {"gaps": len(gaps), "median_s_per_item": statistics.median(gaps) if gaps else None,
            "mean_s_per_item": statistics.fmean(gaps) if gaps else None,
            "note": "an estimate from cache record timestamps (one record per item, written after its last attempt); "
                    "no call was made to time it"}


# ------------------------------------------------------------------------------------------------ E1


STATS = ("pass_rate", "grade_mean", "words_mean", "frame_variety", "distinct_2")


def _strip(rows: list) -> list:
    """Rows without text-derived fields, for the committed json (keys, verdicts, rules, grade, words)."""
    return [{k: r[k] for k in ("key", "unit", "passed", "rules", "grade", "words")} for r in rows]


def e1(split: str, scouting: list, possession: list) -> dict:
    t_rows, t_write = template_rows(scouting)
    h_rows = haiku_rows(scouting)
    p_rows, p_write = template_rows(possession)
    out = {
        "experiment": "genai-E1", "split": split, "checker_git_blob": D.git_blob("export/genai.py"),
        "templates_git_blob": D.git_blob("tools/genai_templates.py"), "code_commit": D.git_head(),
        "scouting": {
            "items": len(scouting),
            "template": arm_measures(t_rows), "haiku": arm_measures(h_rows),
            "haiku_prompt_versions": _count(r["prompt_version"] for r in h_rows),
            "bootstrap_by_player": bootstrap({"template": t_rows, "haiku": h_rows}, STATS),
            "haiku_calls": haiku_calls(scouting),
            "template_speed": template_speed(scouting), "template_write_s_these_items": t_write,
            "rows": {"template": _strip(t_rows), "haiku": _strip(h_rows)},
        },
        "possession": {
            "items": len(possession), "games": len({unit_of(i) for i in possession}),
            "template": arm_measures(p_rows),
            "bootstrap_by_game": bootstrap({"template": p_rows}, STATS),
            "template_speed": template_speed(possession), "template_write_s_these_items": p_write,
            "haiku": "not run: the Haiku possession texts are about held-out games; a Haiku arm on training-game "
                     "possessions needs API spend and waits for a yes (PLAN.md D92)",
            "rows": {"template": _strip(p_rows)},
        },
        "haiku_timing": haiku_timing(),
    }
    return out


def _count(values) -> dict:
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def write(name: str, record: dict) -> Path:
    path = OUT_DIR / name
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _ci(pair, digits=3) -> str:
    return f"[{pair[0]:.{digits}f}, {pair[1]:.{digits}f}]" if pair and pair[0] is not None else "[none]"


def summary(record: dict) -> list:
    """The lines the report copies its numbers from (every number of reports/research/genai-E1.md is one of these)."""
    lines = [f"split {record['split']}, checker blob {record['checker_git_blob'][:12]}, "
             f"templates blob {record['templates_git_blob'][:12]}"]
    for kind, boot_key, arms in (("scouting", "bootstrap_by_player", ("template", "haiku")),
                                 ("possession", "bootstrap_by_game", ("template",))):
        k = record[kind]
        boot = k[boot_key]
        lines.append(f"{kind}: {k['items']} items, {boot['units']} units")
        for arm in arms:
            m, b = k[arm], boot["arms"][arm]
            lines.append(f"  {arm}: pass {m['pass_rate']:.3f} {_ci(b['pass_rate'])}, texts {m['n_texts']}, "
                         f"grade mean {m['grade_mean']:.2f} {_ci(b['grade_mean'], 2)} max {m['grade_max']}, "
                         f"words mean {m['words_mean']:.1f} {_ci(b['words_mean'], 1)} ({m['words_min']} to "
                         f"{m['words_max']}), frames {m['distinct_frames']} of {m['sentences']} sentences, variety "
                         f"{m['frame_variety']:.3f} {_ci(b['frame_variety'])}, distinct-2 {m['distinct_2']:.3f} "
                         f"{_ci(b['distinct_2'])}")
            fails = {r: v for r, v in m["rule_failure_rate"].items() if v}
            lines.append(f"    rule failure rates: {fails or 'none'}")
        for name, pair in boot["differences"].items():
            lines.append(f"  {name} {_ci(pair)}")
        speed = k["template_speed"]
        lines.append(f"  template speed: {speed['texts']} texts written in {speed['write_s']:.3f} s, written and "
                     f"checked in {speed['write_and_check_s']:.2f} s")
    calls = record["scouting"]["haiku_calls"]
    lines.append(f"haiku calls: {calls['records']} records ({calls['records_other_input']} of other inputs left out), "
                 f"{calls['tokens']['calls']} calls, {calls['tokens']['input']} input and {calls['tokens']['output']} "
                 f"output tokens, ${calls['usd']:.4f}, ${calls['usd_per_1000_items_asked']:.2f} per 1,000 items asked")
    for setting, g in calls["by_setting"].items():
        lines.append(f"  {setting}: {g['records']} records, first attempt passes {g['first_attempt_pass']}, kept "
                     f"{g['kept']}")
    lines.append(f"haiku prompt versions delivered: {record['scouting']['haiku_prompt_versions']}")
    t = record["haiku_timing"]
    lines.append(f"haiku timing (estimate): {t['gaps']} gaps, median {t['median_s_per_item']:.1f} s per item, mean "
                 f"{t['mean_s_per_item']:.1f} s")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="genai lane evaluation (docs/research/genai.md)")
    parser.add_argument("stage", choices=("e1-dev", "e1-final", "summary"))
    parser.add_argument("--json", help="for summary: the result json to print")
    args = parser.parse_args(argv)
    if args.stage == "summary":
        for line in summary(json.loads(Path(args.json).read_text(encoding="utf-8"))):
            print(line)
        return 0
    if args.stage == "e1-dev":
        record = e1("dev", D.items("scouting", "dev"), D.items("possession", "dev"))
        print(write("genai-E1-dev.json", record))
        return 0

    def scorer():
        return e1("test", D.items("scouting", "test"), D.items("possession", "test"))

    record = D.score_test_once("genai-E1", scorer)
    print(write("genai-E1-final.json", record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
