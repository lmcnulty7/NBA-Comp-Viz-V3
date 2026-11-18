"""tools/genai_e2.py: genai-E2 on Qwen2.5-0.5B-Instruct (docs/research/genai.md, genai-E2 and its amendments).

Runs in the lane's venv (torch, transformers, peft) under the heavy-job lock, never more than 3 hours a stretch:

    python3 -m export.heavy --name genai-e2 -- <venv python> -m tools.genai_e2 <stage> [--max-hours H]

Stages, in order (each resumes from what the earlier ones saved; the state lives outside the repository, under
V3_GENAI_MODELS/runs and V3_GENAI_DATA/e2):
    a0        the few-shot arm (A0) and A0-P3 on the dev items
    sft       A1: LoRA on train items, one checkpoint and one dev score per epoch, up to 3 epochs
    rft       A2: k candidates per pool item from the dev-best A1 checkpoint, the primary checker keeps passes,
              one more epoch on them; a second round only by the pre-registered rule
    a3        A3: the dev-best arm with constrained decoding, on dev
    choose    write the dev choices (reports/research/genai-E2-0.5B-dev.json) before any test item is read
    test      the one test score of every arm (score_test_once "genai-E2-0.5B")

Every score uses the primary checker (export/genai.py of v4-research at PRIMARY, blob checked) and reports the
freeze checker beside it. Nothing reads a held-out game; the items are the frozen genai items (tools.genai_data).
Texts are written only outside the repository; the committed json holds keys, verdicts, rules and measures.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from tools import genai_data as D
from tools import genai_eval as E
from tools import genai_format as F
from tools.genai_templates import template_text

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
SIZE = "0.5B"
EXPERIMENT = f"genai-E2-{SIZE}"
CHECKERS = {"primary": ("db993008", "41ba991ba22d34c51903593845750fa0cfcdf1dd"),
            "freeze": ("4061d2ed", "c10b8a4bf5e82c2440a1d5ed296759ea810482a0")}
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
LR, WARMUP, ACCUM, EPOCHS, SEED = 2e-4, 10, 8, 3, 0
MAX_TRAIN_TOKENS, MAX_NEW_TOKENS, GEN_BATCH = 1024, 320, 8
FEW_SHOT = 2
RFT = {"k": 4, "temperature": 0.8, "top_p": 0.95, "keep_per_item": 2, "possession_pool": 800, "round2_gain": 0.03}
DTYPE = "float32"
MAX_STRETCH_H = 3.0


def models_dir() -> Path:
    return Path(os.environ.get("V3_GENAI_MODELS") or Path.home() / "Developer" / "nba-v3-worktrees" / "_genai-models")


def run_dir() -> Path:
    return models_dir() / "runs" / EXPERIMENT


def out_dir() -> Path:
    return D.data_dir() / "e2"


def _save(name: str, record) -> Path:
    path = out_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _load(name: str):
    path = out_dir() / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------------------------------------------ checkers


_CHECKS = {}


def checker(name: str):
    """check_text of a pinned export/genai.py, loaded from git; its blob is verified first."""
    if name not in _CHECKS:
        commit, blob = CHECKERS[name]
        found = subprocess.run(["git", "rev-parse", f"{commit}:export/genai.py"], cwd=REPO_ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        if found != blob:
            raise SystemExit(f"checker {name}: export/genai.py at {commit} is blob {found}, not the pinned {blob}")
        _CHECKS[name] = E.sensitivity_checker(commit)
    return _CHECKS[name]


def evaluate(text, item: dict, name: str = "primary") -> dict:
    from export import genai as G
    if not text:
        return {"passed": False, "rules": ["no_text"], "grade": None, "words": None, "frames": []}
    problems, _ = checker(name)(text, item)
    return {"passed": not problems, "rules": sorted({E.rule_of(p) for p in problems}),
            "grade": G.round1(G.fk_grade(text)), "words": len(G.words_of(text)), "frames": E.frames_of(text, item)}


def score(items: list, texts: dict) -> dict:
    """Rows and measures of one arm's texts under both checkers (primary decides; freeze beside it)."""
    out = {}
    for name in CHECKERS:
        rows = []
        for item in items:
            row = evaluate(texts.get(item["key"]), item, name)
            row.update(unit=E.unit_of(item), key=item["key"])
            rows.append(row)
        out[name] = rows
    return out


# ------------------------------------------------------------------------------------------------ data


def targets(split: str = "train") -> list:
    """(item, target) of a split: scouting items with their Haiku text in the snapshot, possession items with their
    template text (genai-E2, A1)."""
    snapshot = {t["key"]: t["text"] for t in
                json.loads(D.snapshot_file("genai/scouting.json").read_text(encoding="utf-8"))["items"]}
    rows = [(item, snapshot[item["key"]]) for item in D.items("scouting", split) if item["key"] in snapshot]
    rows += [(item, template_text(item)) for item in D.items("possession", split)]
    return rows


def few_shot(kind: str) -> list:
    """FEW_SHOT train (item, target) pairs of a kind, drawn with a seeded generator."""
    pool = [row for row in targets("train") if row[0]["kind"] == kind]
    return random.Random(f"{SEED}:{kind}").sample(pool, FEW_SHOT)


def messages(item: dict, arm: str, answer: str = None) -> list:
    """The chat of one item: compact (trained arms), few-shot (A0, rules and 2 compact examples) or p3 (A0-P3)."""
    from export import genai as G
    if arm == "p3":
        system = G.prompt(item["kind"])[1]
        chat = [{"role": "system", "content": system}, {"role": "user", "content": G.user_message(item)}]
    elif arm == "few_shot":
        chat = [{"role": "system", "content": F.RULES}]
        for example, target in few_shot(item["kind"]):
            chat += [{"role": "user", "content": F.compact_message(example)}, {"role": "assistant", "content": target}]
        chat.append({"role": "user", "content": F.compact_message(item)})
    else:
        chat = [{"role": "system", "content": F.INSTRUCTION}, {"role": "user", "content": F.compact_message(item)}]
    if answer is not None:
        chat.append({"role": "assistant", "content": answer})
    return chat


def _ids(tok, chat, generation: bool) -> list:
    ids = tok.apply_chat_template(chat, add_generation_prompt=generation, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)


def encode_train(tok, item: dict, target: str):
    prompt = _ids(tok, messages(item, "compact"), True)
    answer = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    return prompt + answer, [-100] * len(prompt) + answer


# ------------------------------------------------------------------------------------------------ model


def load(adapter: Path = None, trainable: bool = False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1])
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL[0], revision=MODEL[1], dtype=getattr(torch, DTYPE)).to("mps")
    if trainable:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    return tok, model


class NumberGuard:
    """A3: a logits processor that lets a run of digits form only a prefix of a value of its item: the fact's shown
    value without sign, percent sign or unit, taken from the facts with a minus sign when the run follows a minus
    sign and from the others when it does not; "100" of "per 100" is allowed too. Qwen2.5 writes each digit as its
    own token, so the guard works on the 10 digit tokens and the separators "." "," ":". Inside a value that is not
    yet complete only its continuations may follow; after a complete value a separator is free (punctuation) and a
    digit may not be glued to it. Every non-ASCII digit token is masked always. It cannot fix a unit, a missing
    bound or a missing word."""

    SEPARATORS = ".,:"
    MINUSES = ("-", "\u2212")

    def __init__(self, tok, items: list, prompt_len: int):
        import torch
        self.tok, self.prompt_len = tok, prompt_len
        self.sets = []
        for item in items:
            neg, pos = set(), {"100"}
            for f in item["facts"]:
                digits = re.sub(r"[^0-9.,:]", "", f["shown"])
                if digits and digits[0].isdigit():
                    (neg if f["shown"].startswith(self.MINUSES) else pos).add(digits)
            self.sets.append((neg, pos))
        vocab = tok.get_vocab()
        self.digit = {str(d): vocab[str(d)] for d in range(10)}
        self.sep = {c: vocab[c] for c in self.SEPARATORS if c in vocab}
        other = [i for t, i in vocab.items() if re.search(r"\d", tok.convert_tokens_to_string([t]))
                 and i not in self.digit.values()]
        self.other_digit = torch.tensor(other, dtype=torch.long)

    def allowed_next(self, text: str, b: int):
        """(allowed digit and separator chars, or None for no restriction beyond digits, forced) for row b."""
        match = re.search(r"[0-9][0-9.,:]*$", text)
        run = match.group(0) if match else ""
        before = text[:match.start()] if match else text
        neg, pos = self.sets[b]
        values = neg if before.endswith(self.MINUSES) else pos
        if not run:
            return {v[0] for v in values}, False
        prefixes = [v for v in values if v.startswith(run)]
        if not prefixes:
            # a value ended with a separator used as punctuation ("0.143."): no digit may be glued to it
            return set(), False
        nxt = {v[len(run)] for v in prefixes if len(v) > len(run)}
        complete = run in values
        if complete:
            return nxt | set(self.SEPARATORS), False
        return nxt, True

    def __call__(self, input_ids, scores):
        import torch
        scores[:, self.other_digit.to(scores.device)] = -math.inf
        for b in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[b, self.prompt_len:], skip_special_tokens=True)
            chars, forced = self.allowed_next(text, b)
            if forced:
                keep = [self.digit.get(c) or self.sep.get(c) for c in chars]
                mask = torch.full_like(scores[b], -math.inf)
                mask[torch.tensor([k for k in keep if k is not None], dtype=torch.long, device=scores.device)] = 0.0
                scores[b] = scores[b] + mask
                continue
            block = [i for c, i in self.digit.items() if c not in chars]
            if block:
                scores[b, torch.tensor(block, dtype=torch.long, device=scores.device)] = -math.inf
        return scores


def generate(tok, model, items: list, arm: str, sample: bool = False, n: int = 1, guard: bool = False,
             log=print, chat_of=None) -> tuple:
    """{key: [texts]} and {key: seconds} for the items, in batches of GEN_BATCH (left padding). `chat_of` gives a
    chat per item in place of messages(item, arm) (the repair loop)."""
    import torch
    from transformers import LogitsProcessorList
    texts, seconds = {}, {}
    model.eval()
    model.config.use_cache = True
    for start in range(0, len(items), GEN_BATCH):
        batch = items[start:start + GEN_BATCH]
        chats = [tok.apply_chat_template(chat_of(item) if chat_of else messages(item, arm), add_generation_prompt=True,
                                         tokenize=False) for item in batch]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
        width = enc["input_ids"].shape[1]
        kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.eos_token_id, num_return_sequences=n)
        kwargs.update(dict(do_sample=True, temperature=RFT["temperature"], top_p=RFT["top_p"]) if sample
                      else dict(do_sample=False))
        if guard:
            kwargs["logits_processor"] = LogitsProcessorList([NumberGuard(tok, [i for i in batch for _ in range(n)],
                                                                          width)])
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, **kwargs)
        torch.mps.synchronize()
        per = (time.perf_counter() - t0) / len(batch)
        for j, item in enumerate(batch):
            rows = out[j * n:(j + 1) * n, width:]
            texts[item["key"]] = [tok.decode(r, skip_special_tokens=True).strip() for r in rows]
            seconds[item["key"]] = per
        log(f"  generated {min(start + GEN_BATCH, len(items))} of {len(items)} ({arm}{', guard' if guard else ''})")
    return texts, seconds


# ------------------------------------------------------------------------------------------------ training


def train(tok, model, rows: list, epochs: int, tag: str, deadline: float, dev_items: list, log=print) -> dict:
    """LoRA on (item, target) rows, AdamW with warm-up and linear decay, ACCUM examples per step. One adapter
    checkpoint and one dev score per epoch; a stretch that reaches its deadline saves and stops (resumable)."""
    import torch
    state_path = run_dir() / tag / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"epoch": 0, "dev": [], "skipped": 0}
    encoded = []
    for item, target in rows:
        ids, labels = encode_train(tok, item, target)
        if len(ids) > MAX_TRAIN_TOKENS:
            continue
        encoded.append((ids, labels))
    state["skipped"] = len(rows) - len(encoded)
    steps_per_epoch = math.ceil(len(encoded) / ACCUM)
    total = steps_per_epoch * epochs
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.0, 1 - s / max(1, total)))
    opt_path = run_dir() / tag / "optimizer.pt"
    if opt_path.exists():
        saved = torch.load(opt_path, weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
    while state["epoch"] < epochs:
        order = list(range(len(encoded)))
        random.Random(f"{SEED}:{tag}:{state['epoch']}").shuffle(order)
        done = state.get("examples_done", 0)
        model.train()
        model.config.use_cache = False
        t0 = time.perf_counter()
        for k in range(done, len(order)):
            ids, labels = encoded[order[k]]
            loss = model(input_ids=torch.tensor([ids], device="mps"), labels=torch.tensor([labels], device="mps")).loss
            (loss / ACCUM).backward()
            if (k + 1) % ACCUM == 0 or k + 1 == len(order):
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                if time.perf_counter() > deadline:
                    _checkpoint(model, optimizer, schedule, tag, dict(state, examples_done=k + 1))
                    log(f"{tag}: stretch deadline at epoch {state['epoch'] + 1}, example {k + 1}; saved")
                    return dict(state, examples_done=k + 1, stopped=True)
            if (k + 1) % 200 == 0:
                log(f"{tag}: epoch {state['epoch'] + 1}, {k + 1} of {len(order)}, loss {float(loss):.3f}")
        state["epoch"] += 1
        state["examples_done"] = 0
        state.setdefault("train_s", []).append(time.perf_counter() - t0)
        _checkpoint(model, optimizer, schedule, tag, state, epoch_dir=f"epoch{state['epoch']}")
        texts, seconds = generate(tok, model, dev_items, "compact", log=log)
        rows_ = score(dev_items, {k: v[0] for k, v in texts.items()})
        _save(f"{tag}-epoch{state['epoch']}-dev-texts.json", {k: v[0] for k, v in texts.items()})
        dev = {"epoch": state["epoch"], "primary": E.arm_measures(rows_["primary"]),
               "freeze": E.arm_measures(rows_["freeze"]), "seconds_per_item": _mean(seconds.values())}
        state["dev"].append({k: v for k, v in dev.items()})
        _checkpoint(model, optimizer, schedule, tag, state, epoch_dir=None)
        log(f"{tag}: epoch {state['epoch']} dev pass {dev['primary']['pass_rate']:.3f}")
        if time.perf_counter() > deadline and state["epoch"] < epochs:
            return dict(state, stopped=True)
    return state


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else None


def _checkpoint(model, optimizer, schedule, tag: str, state: dict, epoch_dir=None) -> None:
    import torch
    folder = run_dir() / tag
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(folder / "latest"))
    if epoch_dir:
        model.save_pretrained(str(folder / epoch_dir))
    torch.save({"optimizer": optimizer.state_dict(), "schedule": schedule.state_dict()}, folder / "optimizer.pt")
    (folder / "state.json").write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")


def best_epoch(tag: str) -> int:
    state = json.loads((run_dir() / tag / "state.json").read_text())
    return max(state["dev"], key=lambda d: (d["primary"]["pass_rate"], -d["epoch"]))["epoch"]


# ------------------------------------------------------------------------------------------------ stages


def dev_items() -> list:
    return D.items("scouting", "dev") + D.items("possession", "dev")


def stage_a0(deadline, log=print) -> dict:
    tok, model = load()
    items = dev_items()
    texts, seconds = generate(tok, model, items, "few_shot", log=log)
    rows = score(items, {k: v[0] for k, v in texts.items()})
    scouting = [i for i in items if i["kind"] == "scouting"]
    p3_texts, p3_seconds = generate(tok, model, scouting, "p3", log=log)
    p3_rows = score(scouting, {k: v[0] for k, v in p3_texts.items()})
    _save("a0-dev-texts.json", {"few_shot": {k: v[0] for k, v in texts.items()},
                                "p3": {k: v[0] for k, v in p3_texts.items()}})
    record = {"a0": _by_kind(rows), "a0_p3": _by_kind(p3_rows), "seconds_per_item": _mean(seconds.values()),
              "p3_seconds_per_item": _mean(p3_seconds.values())}
    _save("a0-dev.json", record)
    return record


def _by_kind(rows: dict) -> dict:
    out = {}
    for name, rs in rows.items():
        for kind in ("scouting", "possession"):
            mine = [r for r in rs if ("/" in r["key"]) == (kind == "possession")]
            if mine:
                out.setdefault(name, {})[kind] = E.arm_measures(mine)
    return out


def stage_sft(deadline, log=print) -> dict:
    tag = "a1"
    latest = run_dir() / tag / "latest"
    tok, model = load(latest if latest.exists() else None, trainable=True)
    return train(tok, model, targets("train"), EPOCHS, tag, deadline, dev_items(), log)


def rft_pool() -> list:
    rows = targets("train")
    scouting = [item for item, _ in rows if item["kind"] == "scouting"]
    possession = [item for item, _ in rows if item["kind"] == "possession"]
    return scouting + random.Random(f"{SEED}:rft").sample(possession, RFT["possession_pool"])


def stage_rft(deadline, round_: int = 1, log=print) -> dict:
    source = run_dir() / "a1" / f"epoch{best_epoch('a1')}" if round_ == 1 else run_dir() / "a2-r1" / "epoch1"
    tag = f"a2-r{round_}"
    cand_name = f"{tag}-candidates.json"
    candidates = _load(cand_name)
    if candidates is None:
        tok, model = load(source)
        pool = rft_pool()
        found, seconds = generate(tok, model, pool, "compact", sample=True, n=RFT["k"], log=log)
        check = checker("primary")
        by_key = {item["key"]: item for item in pool}
        kept = {}
        for key, texts in found.items():
            good = []
            for text in dict.fromkeys(texts):
                if text and not check(text, by_key[key])[0]:
                    good.append(text)
            kept[key] = good[:RFT["keep_per_item"]]
        candidates = {"kept": kept, "seconds_per_item": _mean(seconds.values()),
                      "pool": len(pool), "items_with_a_pass": sum(bool(v) for v in kept.values()),
                      "kept_texts": sum(len(v) for v in kept.values())}
        _save(cand_name, candidates)
        del model
    tok, model = load(run_dir() / tag / "latest" if (run_dir() / tag / "latest").exists() else source, trainable=True)
    by_key = {item["key"]: item for item in rft_pool()}
    rows = [(by_key[k], t) for k, texts in candidates["kept"].items() for t in texts]
    state = train(tok, model, rows, 1, tag, deadline, dev_items(), log)
    state["candidates"] = {k: v for k, v in candidates.items() if k != "kept"}
    return state


def stage_a3(arm_dir: Path, deadline, log=print) -> dict:
    tok, model = load(arm_dir)
    items = dev_items()
    texts, seconds = generate(tok, model, items, "compact", guard=True, log=log)
    rows = score(items, {k: v[0] for k, v in texts.items()})
    _save("a3-dev-texts.json", {k: v[0] for k, v in texts.items()})
    record = {"arm": str(arm_dir.relative_to(run_dir())), "a3": _by_kind(rows), "seconds_per_item": _mean(seconds.values())}
    _save("a3-dev.json", record)
    return record


def repair(tok, model, items: list, first: dict, guard: bool = False, log=print) -> dict:
    """The repair loop (secondary): an item whose first text fails the primary checker is asked once more, with that
    text and its problems in the chat, as P3 asks Haiku (export.genai.call_model). Returns the final texts."""
    check = checker("primary")
    failed = {item["key"]: check(first.get(item["key"]) or "", item)[0] for item in items}
    redo = [item for item in items if failed[item["key"]] or not first.get(item["key"])]

    def chat_of(item):
        problems = failed[item["key"]] or ["no text"]
        return messages(item, "compact", first.get(item["key"]) or "") + [{
            "role": "user", "content": "That text broke these rules:\n- " + "\n- ".join(problems) +
            "\nWrite the paragraph again and fix every one. Copy each number exactly as shown."}]

    second, _ = generate(tok, model, redo, "compact", guard=guard, log=log, chat_of=chat_of) if redo else ({}, {})
    return {item["key"]: second[item["key"]][0] if item["key"] in second else first.get(item["key"]) for item in items}


def choose() -> dict:
    """The dev choices, written before any test item is read: the A1 epoch, whether A2 ran a second round, the
    dev-best fine-tuned arm of each kind, and the arm A3 decodes."""
    a1 = json.loads((run_dir() / "a1" / "state.json").read_text())
    arms = {f"a1-epoch{d['epoch']}": d for d in a1["dev"]}
    for tag in ("a2-r1", "a2-r2"):
        path = run_dir() / tag / "state.json"
        if path.exists():
            for d in json.loads(path.read_text())["dev"]:
                arms[f"{tag}-epoch{d['epoch']}"] = d
    a1_best = f"a1-epoch{best_epoch('a1')}"
    # per kind, the dev pass rate of an arm comes from its saved dev texts
    by_kind = {}
    for arm in arms:
        texts = _load(f"{arm.rsplit('-epoch', 1)[0]}-epoch{arm.rsplit('-epoch', 1)[1]}-dev-texts.json")
        rows = score(dev_items(), texts)
        by_kind[arm] = _by_kind(rows)
    best = {kind: max((a for a in arms if a == a1_best or a.startswith("a2")),
                      key=lambda a: by_kind[a]["primary"][kind]["pass_rate"]) for kind in ("scouting", "possession")}
    overall = max(set(best.values()) | {a1_best}, key=lambda a: sum(by_kind[a]["primary"][k]["pass_rate"]
                                                                       for k in ("scouting", "possession")))
    r1 = by_kind.get("a2-r1-epoch1")
    gain = None
    if r1:
        gain = {k: r1["primary"][k]["pass_rate"] - by_kind[a1_best]["primary"][k]["pass_rate"]
                for k in ("scouting", "possession")}
    a2_final = max((a for a in arms if a.startswith("a2")), default=None)
    return {"a1_epoch": best_epoch("a1"), "a1_best": a1_best, "a2_final": a2_final, "best_by_kind": best,
            "a3_arm": overall,
            "rft_round1_gain": gain, "rft_round2_rule": f"runs if the dev pass rate gain of round 1 is at least "
            f"{RFT['round2_gain']} (mean over kinds)", "dev": by_kind}


def adapter_of(arm: str) -> Path:
    tag, epoch = arm.rsplit("-epoch", 1)
    return run_dir() / tag / f"epoch{epoch}"


def paired(rows_a: list, rows_b: list) -> list:
    """95 percent interval of (b minus a) pass rate, units resampled together (genai_eval.bootstrap)."""
    out = E.bootstrap({"a": rows_a, "b": rows_b}, ("pass_rate",))
    return out["differences"]["b minus a: pass_rate"]


def reading(diff: list, point: float, rule: str) -> str:
    lo, hi = diff
    if rule == "h1":
        return "passes" if point >= 0.10 and lo > 0 else "fails" if hi < 0.10 else "inconclusive"
    if rule == "h3":
        return "non-inferior" if lo >= -0.05 else "inferior" if hi < -0.05 else "inconclusive"
    return "helps" if lo > 0 else "hurts" if hi < 0 else "inconclusive"


def stage_test(choices: dict, log=print) -> dict:
    """Every arm on the test items, scored by both checkers; the hypotheses read on the primary checker."""
    from export import genai as G

    def scorer():
        items = D.items("scouting", "test") + D.items("possession", "test")
        scouting = [i for i in items if i["kind"] == "scouting"]
        texts = {}
        tok, model = load()
        texts["a0"] = {k: v[0] for k, v in generate(tok, model, items, "few_shot", log=log)[0].items()}
        texts["a0_p3"] = {k: v[0] for k, v in generate(tok, model, scouting, "p3", log=log)[0].items()}
        del model
        timing = {}
        arms = {choices["a1_best"], *choices["best_by_kind"].values(), choices["a3_arm"]}
        arms |= {choices["a2_final"]} if choices.get("a2_final") else set()
        for arm in sorted(arms):
            tok, model = load(adapter_of(arm))
            found, seconds = generate(tok, model, items, "compact", log=log)
            texts[arm] = {k: v[0] for k, v in found.items()}
            timing[arm] = _mean(seconds.values())
            if arm == choices["a3_arm"]:
                found, seconds = generate(tok, model, items, "compact", guard=True, log=log)
                texts["a3"] = {k: v[0] for k, v in found.items()}
                timing["a3"] = _mean(seconds.values())
            if arm in choices["best_by_kind"].values():
                texts[f"{arm}+repair"] = repair(tok, model, items, texts[arm], log=log)
            del model
        _save("test-texts.json", texts)
        rows = {arm: score(items, t) for arm, t in texts.items()}
        snapshot = {t["key"]: t["text"] for t in
                    json.loads(D.snapshot_file("genai/scouting.json").read_text(encoding="utf-8"))["items"]}
        cache = {r["key"]: r["_text"] for r in E._cache_text_rows(scouting)}
        rows["haiku"] = score(scouting, {k: snapshot.get(k) for k in [i["key"] for i in scouting]})
        rows["haiku_cache"] = score(scouting, cache)
        return items, rows, timing

    assert EXPERIMENT == "genai-E2-0.5B"
    items, rows, timing = D.score_test_once("genai-E2-0.5B", scorer)

    def kind_rows(arm, kind, checker_name="primary"):
        return [r for r in rows[arm][checker_name] if ("/" in r["key"]) == (kind == "possession")]

    def point(arm_b, arm_a, kind):
        return (E.arm_measures(kind_rows(arm_b, kind))["pass_rate"] - E.arm_measures(kind_rows(arm_a, kind))["pass_rate"])

    hyp = {}
    for kind in ("scouting", "possession"):
        best = choices["best_by_kind"][kind]
        d = paired(kind_rows("a0", kind), kind_rows(best, kind))
        hyp[f"H1 {kind}: {best} minus a0"] = {"point": point(best, "a0", kind), "interval": d,
                                              "reading": reading(d, point(best, "a0", kind), "h1")}
        if choices.get("a2_final"):
            a2 = choices["a2_final"]
            d = paired(kind_rows(choices["a1_best"], kind), kind_rows(a2, kind))
            hyp[f"H2 {kind}: {a2} minus {choices['a1_best']}"] = {
                "point": point(a2, choices["a1_best"], kind), "interval": d,
                "reading": reading(d, point(a2, choices["a1_best"], kind), "h2")}
        d = paired(kind_rows(choices["a3_arm"], kind), kind_rows("a3", kind))
        hyp[f"H4 {kind}: a3 minus {choices['a3_arm']}"] = {"point": point("a3", choices["a3_arm"], kind),
                                                          "interval": d,
                                                          "reading": reading(d, point("a3", choices["a3_arm"], kind),
                                                                             "h4")}
    best = choices["best_by_kind"]["scouting"]
    for haiku in ("haiku", "haiku_cache"):
        d = paired(kind_rows(haiku, "scouting"), kind_rows(f"{best}+repair", "scouting"))
        p = point(f"{best}+repair", haiku, "scouting")
        hyp[f"H3 scouting: {best}+repair minus {haiku}"] = {"point": p, "interval": d, "reading": reading(d, p, "h3")}
    record = {"experiment": EXPERIMENT, "model": MODEL, "checkers": CHECKERS, "choices": {k: v for k, v in
              choices.items() if k != "dev"}, "hypotheses": hyp, "seconds_per_item": timing,
              "arms": {arm: {name: {kind: E.arm_measures([r for r in rs if ("/" in r["key"]) == (kind == "possession")])
                                    for kind in ("scouting", "possession")
                                    if any(("/" in r["key"]) == (kind == "possession") for r in rs)}
                             for name, rs in by.items()} for arm, by in rows.items()},
              "rows": {arm: {name: E._strip(rs) for name, rs in by.items()} for arm, by in rows.items()}}
    path = REPO_ROOT / "reports" / "research" / f"{EXPERIMENT}-final.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="genai-E2 on the 0.5B model (docs/research/genai.md)")
    parser.add_argument("stage", choices=("a0", "sft", "rft", "rft2", "a3", "choose", "test"))
    parser.add_argument("--max-hours", type=float, default=MAX_STRETCH_H)
    parser.add_argument("--arm", help="a3: the adapter folder under the run folder (the dev-best arm)")
    args = parser.parse_args(argv)
    if args.max_hours > MAX_STRETCH_H:
        print(f"a stretch holds the lock for at most {MAX_STRETCH_H} hours", file=sys.stderr)
        return 1
    # stop training 25 minutes before the end, so the dev score of the epoch fits in the stretch
    deadline = time.perf_counter() + args.max_hours * 3600 - 25 * 60
    started = time.perf_counter()
    if args.stage == "a0":
        record = stage_a0(deadline)
    elif args.stage == "sft":
        record = stage_sft(deadline)
    elif args.stage in ("rft", "rft2"):
        record = stage_rft(deadline, 1 if args.stage == "rft" else 2)
    elif args.stage == "a3":
        record = stage_a3(run_dir() / args.arm, deadline)
    elif args.stage == "choose":
        record = choose()
        path = REPO_ROOT / "reports" / "research" / f"{EXPERIMENT}-dev.json"
        path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    else:
        choices = json.loads((REPO_ROOT / "reports" / "research" / f"{EXPERIMENT}-dev.json").read_text())
        record = stage_test(choices)
    print(json.dumps({"stage": args.stage, "wall_s": time.perf_counter() - started,
                      "summary": _short(record)}, indent=1))
    return 0


def _short(record):
    if isinstance(record, dict):
        return {k: _short(v) for k, v in record.items() if k not in ("rows", "frames")}
    return record


if __name__ == "__main__":
    raise SystemExit(main())
