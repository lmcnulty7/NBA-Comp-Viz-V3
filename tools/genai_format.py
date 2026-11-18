"""tools/genai_format.py: the compact input of the genai-E2 trained arms (docs/research/genai.md, genai-E2).

The P3 prompt (export/prompts) and export.genai.user_message give the facts as indented JSON with a long instruction
block: about 3,800 tokens per scouting example with its text, too long for LoRA on 16 GB. A trained arm does not need
the rules spelled out: it learns them from its targets, and the checker scores it the same way. So the trained arms
(A1, A2, A3) get this compact input. The few-shot arm (A0) gets the same compact input with RULES (the P3 prompt's
rules cut to this format) and 2 compact worked examples; A0-P3, reported beside it, gets the full P3 prompt and JSON.

    <venv python> -m tools.genai_format --out reports/research/genai-E2-format-tokens.json

counts the tokens of every train example in both forms with the Qwen2.5 tokenizer (no model is loaded).

The compact input holds the same facts as the item, nothing more: one line per fact or per estimate, the shown value
exactly as the page prints it, the unit only where the shown value does not carry it, and the context (names, teams,
position, result, shots, model status) on a few short lines. A value with an interval is one line "field: value
(lo to hi, LEVEL)"; a point estimate is marked "point". The fact's plain description (`about`) is left out: a field
name stands for it. Nothing here reads data; it formats an item built by export.genai.
"""

INSTRUCTION = ("Write one short paragraph in plain words from these facts. Copy each number exactly. Give each "
               "range with its value. Call a point value a point estimate. Use short sentences.")

# The few-shot arm (A0) gets the rules a trained arm learns from its targets: the P3 prompt's rules, cut to the
# compact format. Two worked examples follow it as earlier turns of the chat.
RULES = ("Write one paragraph of 60 to 110 words about the item, from the facts below it. Rules: use only numbers "
         "from the facts, copied exactly, with the same digits, sign and percent sign; write every number as digits, "
         "never as a word. A fact with a range in brackets is given with its range, for example 4.2 (95% range 2.9 "
         "to 6.0). A fact marked point is a point estimate: write the words point estimate once. A model's number "
         "needs the word model, expected or estimate in its sentence. A view under not shown has too small a "
         "sample. Say that descriptive facts describe past games. Keep each sentence short and plain. Never use a "
         "dash, a list or markdown, and never write about decisions, matchups, lineups, passes, drives, screens, "
         "coverage, play calls, betting or odds. Write only the paragraph.")

UNIT_MARK = {"feet": " ft", "points": " pts", "seconds": " s"}


def _value(fact: dict) -> str:
    unit = fact.get("unit", "none")
    return fact["shown"] + UNIT_MARK.get(unit, "")


def fact_lines(item: dict) -> list:
    """One line per fact or estimate. Two runs are folded into one line each: the EPV curve ("time: value" pairs)
    and the similar players ("name distance"). A sample-size fact (".n") whose shown value repeats a count already
    listed is left out: the same digits stand on the count's line."""
    import re
    facts = item["facts"]
    by_field = {f["field"]: f for f in facts}
    shown_counts = {f["shown"] for f in facts if f["kind"] == "fact" and not f["field"].endswith(".n")}
    names = (item["context"].get("similar_players") or []) if item["kind"] == "scouting" else []
    lines, done = [], set()
    curve = [(f, by_field.get(f"epv.value.{f['field'].split('.')[-1]}")) for f in facts
             if re.fullmatch(r"epv\.t\.\d+", f["field"])]
    if curve:
        lines.append("epv curve (s: pts, point): " + ", ".join(f"{t['shown']}: {v['shown']}" for t, v in curve if v))
        done |= {f["field"] for pair in curve for f in pair if f}
    near = []
    for k, name in enumerate(names, 1):
        d = by_field.get(f"similar.neighbours.{k}.distance")
        if d:
            near.append(f"{name} {d['shown']}")
            done |= {f"similar.neighbours.{k}.distance", f"similar.neighbours.{k}.rank"}
    if near:
        lines.append("similar, by rank (style distance, point): " + ", ".join(near))
    for f in facts:
        field = f["field"]
        if field in done:
            continue
        if field.endswith(".n") and f["shown"] in shown_counts:
            done.add(field)
            continue
        if f["kind"] == "value":
            group = f["group"]
            lo, hi, level = (by_field.get(f"{group}.{k}") for k in ("lo", "hi", "interval"))
            rng = f"{_value(lo)} to {_value(hi)}" if lo and hi else ""
            rng += f", {level['shown']}" if level else ""
            lines.append(f"{group}: {_value(f)} ({rng})")
            done |= {f"{group}.{k}" for k in ("value", "lo", "hi", "interval")}
        elif f["kind"] == "point":
            lines.append(f"{field}: {_value(f)} point")
        else:
            lines.append(f"{field}: {_value(f)}")
        done.add(field)
    return lines


def context_lines(item: dict) -> list:
    c = item["context"]
    status = c.get("model_status") or {}
    not_working = {k: v for k, v in status.items() if v != "works"}
    if item["kind"] == "scouting":
        out = [f"player: {c['name']}", f"teams: {', '.join(c.get('teams') or [])}"]
        if c.get("position"):
            out.append(f"position: {c['position']}")
        hidden = [k for k, v in (c.get("shown") or {}).items() if not v]
        if hidden:
            out.append(f"not shown: {', '.join(hidden)}")
    else:
        out = [f"offense: {c['offense']}", f"defense: {c['defense']}", f"result: {c['result']}"]
        for j, shot in enumerate(c.get("shots") or []):
            out.append(f"shot {j}: {shot.get('shooter') or 'unknown'}, {shot.get('kind') or 'shot'}, "
                       f"{'made' if shot.get('made') else 'missed'}")
        if c.get("clock_gap"):
            out.append("clock gap: yes")
    if not_working:
        out.append("model status: " + ", ".join(f"{k} {v.replace('_', ' ')}" for k, v in sorted(not_working.items())))
    return out


def compact_message(item: dict) -> str:
    return "\n".join([f"kind: {item['kind']}"] + context_lines(item) + ["facts:"] + fact_lines(item))


def token_counts(out) -> dict:
    """Token counts of the train examples (input and target) in the compact form and, for the first 200 of each
    kind, in the P3 form; counts only, no text."""
    import json
    from pathlib import Path
    from transformers import AutoTokenizer
    from export import genai as G
    from tools import genai_e2 as X

    tok = AutoTokenizer.from_pretrained(X.MODEL[0], revision=X.MODEL[1])
    record = {"_comment": "genai-E2 input format token counts (tools/genai_format.py); counts only, no text",
              "tokenizer": list(X.MODEL)}
    for kind in ("scouting", "possession"):
        rows = [row for row in X.targets("train") if row[0]["kind"] == kind]
        compact = sorted(len(X.encode_train(tok, item, target)[0]) for item, target in rows)
        full = []
        for item, target in rows[:200]:
            chat = [{"role": "system", "content": G.prompt(kind)[1]}, {"role": "user", "content": G.user_message(item)},
                    {"role": "assistant", "content": target}]
            ids = tok.apply_chat_template(chat, tokenize=True)
            full.append(len(ids["input_ids"] if hasattr(ids, "input_ids") else ids))
        record[kind] = {"examples": len(compact), "compact_mean": sum(compact) / len(compact),
                        "compact_largest": compact[-1], "compact_over_800": sum(n > 800 for n in compact),
                        "compact_over_1024": sum(n > 1024 for n in compact),
                        "p3_form_mean_first_200": sum(full) / len(full), "p3_form_largest_first_200": max(full)}
    Path(out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="token counts of the genai-E2 input forms")
    parser.add_argument("--out", required=True)
    print(token_counts(parser.parse_args().out))
