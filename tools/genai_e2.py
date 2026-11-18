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
    rehearse  the exact test path on the dev items, without the ledger
    test      the one test score of every arm (score_test_once "genai-E2-0.5B"); refuses unless the dev json is
              committed and unchanged, tools/ and export/ are clean, and the rehearsal ran on this dev json and commit
    test-resume  the one pre-registered resume entry, only after a crash of the test call

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
# The two pre-registered sizes (docs/research/genai.md; the 1.5B amendment, PLAN.md D101): model, revision and the
# number format the pilot set. set_size() picks one before any stage runs; 0.5B is the default.
SIZES = {"0.5B": (("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"), "float32"),
         "1.5B": (("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"), "bfloat16")}
MODEL = SIZES["0.5B"][0]
SIZE = "0.5B"
EXPERIMENT = f"genai-E2-{SIZE}"
CHECKERS = {"primary": ("db993008", "41ba991ba22d34c51903593845750fa0cfcdf1dd"),
            "freeze": ("4061d2ed", "c10b8a4bf5e82c2440a1d5ed296759ea810482a0")}
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
LR, WARMUP, ACCUM, EPOCHS, SEED = 2e-4, 10, 8, 3, 0
MAX_TRAIN_TOKENS, MAX_NEW_TOKENS, GEN_BATCH = 1024, 320, 8
# The P3 form is about 3,900 prompt tokens: a batch of 8 needs about 7 GB for attention scores alone and ran out of
# memory under the cap (a0, 2026-11-17), so A0-P3 generates one item at a time.
P3_BATCH = 1
SAVE_EVERY = 400          # examples between mid-epoch checkpoints (a crash loses at most this much work)
FEW_SHOT = 2
RFT = {"k": 4, "temperature": 0.8, "top_p": 0.95, "keep_per_item": 2, "possession_pool": 800, "round2_gain": 0.03}
DTYPE = SIZES["0.5B"][1]
MAX_STRETCH_H = 3.0


def models_dir() -> Path:
    return Path(os.environ.get("V3_GENAI_MODELS") or Path.home() / "Developer" / "nba-v3-worktrees" / "_genai-models")


def run_dir() -> Path:
    return models_dir() / "runs" / EXPERIMENT


def out_dir() -> Path:
    return D.data_dir() / ("e2" if SIZE == "0.5B" else f"e2-{SIZE}")


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
    try:
        problems, _ = checker(name)(text, item)
    except Exception as exc:  # a checker that raises on a text fails the text (amendment j), it does not crash
        problems = [f"checker raised {type(exc).__name__}"]
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


_TARGETS = {}


def targets(split: str = "train") -> list:
    """(item, target) of a split that pass the primary checker: scouting items with their Haiku text in the
    snapshot, possession items with their template text (genai-E2, A1; amendment g). The dropped counts per kind are
    in target_drops()."""
    if split not in _TARGETS:
        snapshot = {t["key"]: t["text"] for t in
                    json.loads(D.snapshot_file("genai/scouting.json").read_text(encoding="utf-8"))["items"]}
        rows = [(item, snapshot[item["key"]]) for item in D.items("scouting", split) if item["key"] in snapshot]
        rows += [(item, template_text(item)) for item in D.items("possession", split)]
        check = checker("primary")
        kept = [row for row in rows if not check(row[1], row[0])[0]]
        drops = {kind: sum(1 for item, _ in rows if item["kind"] == kind)
                 - sum(1 for item, _ in kept if item["kind"] == kind) for kind in ("scouting", "possession")}
        totals = {kind: sum(1 for item, _ in rows if item["kind"] == kind) for kind in ("scouting", "possession")}
        _TARGETS[split] = (kept, {"dropped": drops, "candidates": totals})
    return _TARGETS[split][0]


def target_drops(split: str = "train") -> dict:
    targets(split)
    return _TARGETS[split][1]


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
    size = P3_BATCH if arm == "p3" else GEN_BATCH
    for start in range(0, len(items), size):
        batch = items[start:start + size]
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
        log(f"  generated {min(start + size, len(items))} of {len(items)} ({arm}{', guard' if guard else ''})")
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
            loss = answer_loss(model, ids, labels)
            (loss / ACCUM).backward()
            last_loss = float(loss.detach())
            del loss
            if (k + 1) % ACCUM == 0 or k + 1 == len(order):
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                # the first stretch ran out of memory under the cap after about 200 examples of mixed lengths:
                # the MPS cache grows with each new shape, so it is emptied at every optimizer step
                torch.mps.empty_cache()
                if (k + 1) % SAVE_EVERY == 0:
                    _checkpoint(model, optimizer, schedule, tag, dict(state, examples_done=k + 1))
                if time.perf_counter() > deadline:
                    _checkpoint(model, optimizer, schedule, tag, dict(state, examples_done=k + 1))
                    log(f"{tag}: stretch deadline at epoch {state['epoch'] + 1}, example {k + 1}; saved")
                    return dict(state, examples_done=k + 1, stopped=True)
            if (k + 1) % 200 == 0:
                log(f"{tag}: epoch {state['epoch'] + 1}, {k + 1} of {len(order)}, loss {last_loss:.3f}, "
                    f"MPS {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
        state["epoch"] += 1
        state["examples_done"] = 0
        state.setdefault("train_s", []).append(time.perf_counter() - t0)
        _checkpoint(model, optimizer, schedule, tag, state, epoch_dir=f"epoch{state['epoch']}")
        texts, seconds = generate(tok, model, dev_items, "compact", log=log)
        rows_ = score(dev_items, {k: v[0] for k, v in texts.items()})
        _save(f"{tag}-epoch{state['epoch']}-dev-texts.json", {k: v[0] for k, v in texts.items()})
        dev = {"epoch": state["epoch"], **_by_kind(rows_), "mean_pass": mean_pass(_by_kind(rows_)),
               "seconds_per_item": _mean(seconds.values()), "targets": target_drops("train")}
        state["dev"].append({k: v for k, v in dev.items()})
        _checkpoint(model, optimizer, schedule, tag, state, epoch_dir=None)
        log(f"{tag}: epoch {state['epoch']} dev pass, mean over kinds, {dev['mean_pass']:.3f}")
        if time.perf_counter() > deadline and state["epoch"] < epochs:
            return dict(state, stopped=True)
    return state


def answer_loss(model, ids: list, labels: list):
    """The mean cross-entropy of the answer tokens, as the model's own loss with these labels computes it, but with
    logits kept only for the answer positions (logits_to_keep). The full-length logits (151,936 words by every
    position, in float32 for the loss) were the memory peak that stopped the first 1.5B stretch under the cap; the
    answer is the end of the example, so nothing of the loss changes."""
    import torch
    import torch.nn.functional as Fn
    n_answer = sum(1 for x in labels if x != -100)
    if n_answer == 0 or any(x != -100 for x in labels[:len(labels) - n_answer]):
        raise ValueError("the answer must be the end of the example")
    keep = n_answer + 1
    logits = model(input_ids=torch.tensor([ids], device="mps"), logits_to_keep=keep).logits
    target = torch.tensor([ids[len(ids) - n_answer:]], device="mps")
    return Fn.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), target.reshape(-1))


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


def mean_pass(by_kind: dict, name: str = "primary") -> float:
    """The dev pass rate of an arm: the mean of its scouting and possession pass rates (amendment e)."""
    return (by_kind[name]["scouting"]["pass_rate"] + by_kind[name]["possession"]["pass_rate"]) / 2


def best_epoch(tag: str) -> int:
    state = json.loads((run_dir() / tag / "state.json").read_text())
    return max(state["dev"], key=lambda d: (d["mean_pass"], -d["epoch"]))["epoch"]


def dev_arms() -> dict:
    """{arm: dev entry} of the chosen A1 epoch and of each A2 round that ran."""
    a1 = json.loads((run_dir() / "a1" / "state.json").read_text())
    best = best_epoch("a1")
    arms = {f"a1-epoch{best}": next(d for d in a1["dev"] if d["epoch"] == best)}
    for tag in ("a2-r1", "a2-r2"):
        path = run_dir() / tag / "state.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["dev"]:
                arms[f"{tag}-epoch1"] = state["dev"][-1]
    return arms


def dev_best_arm() -> str:
    """The arm A3 decodes (amendment f): the highest mean dev pass rate among the chosen A1 epoch and the last A2
    round that ran; ties go to the earlier arm."""
    arms = dev_arms()
    final_a2 = max((a for a in arms if a.startswith("a2")), default=None)
    pool = [a for a in arms if a.startswith("a1")] + ([final_a2] if final_a2 else [])
    return max(pool, key=lambda a: (arms[a]["mean_pass"], -pool.index(a)))


def round2_allowed() -> tuple:
    """(allowed, gain): RFT round 2 runs only when round 1 raised the mean dev pass rate by RFT["round2_gain"]."""
    arms = dev_arms()
    a1 = next(a for a in arms if a.startswith("a1"))
    if "a2-r1-epoch1" not in arms:
        return False, None
    gain = arms["a2-r1-epoch1"]["mean_pass"] - arms[a1]["mean_pass"]
    return gain >= RFT["round2_gain"], gain


def dev_items() -> list:
    return D.items("scouting", "dev") + D.items("possession", "dev")


def stage_a0(deadline, log=print) -> dict:
    """A0 (few-shot) on every dev item and A0-P3 on the dev players. Resumable: texts already written are kept in
    a0-dev-texts.json, and a stretch that reaches its deadline stops between batches and says so."""
    saved = _load("a0-dev-texts.json") or {"few_shot": {}, "p3": {}}
    items = dev_items()
    scouting = [i for i in items if i["kind"] == "scouting"]
    todo = {"few_shot": [i for i in items if i["key"] not in saved["few_shot"]],
            "p3": [i for i in scouting if i["key"] not in saved["p3"]]}
    seconds = saved.setdefault("seconds", {"few_shot": {}, "p3": {}})
    if any(todo.values()):
        tok, model = load()
        for arm in ("few_shot", "p3"):
            for start in range(0, len(todo[arm]), GEN_BATCH):
                if time.perf_counter() > deadline:
                    _save("a0-dev-texts.json", saved)
                    log("a0: stretch deadline reached; texts so far saved")
                    return {"stopped": True, "few_shot_done": len(saved["few_shot"]), "p3_done": len(saved["p3"])}
                found, secs = generate(tok, model, todo[arm][start:start + GEN_BATCH], arm, log=log)
                saved[arm].update({k: v[0] for k, v in found.items()})
                seconds[arm].update(secs)
                _save("a0-dev-texts.json", saved)
    rows = score(items, saved["few_shot"])
    p3_rows = score(scouting, saved["p3"])
    record = {"a0": _by_kind(rows), "a0_mean_pass": mean_pass(_by_kind(rows)), "a0_p3": _by_kind(p3_rows),
              "seconds_per_item": _mean(seconds["few_shot"].values()),
              "p3_seconds_per_item": _mean(seconds["p3"].values()), "targets": target_drops("train"),
              "few_shot_examples": {kind: [ex["key"] for ex, _ in few_shot(kind)] for kind in ("scouting", "possession")}}
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


def stage_smoke(log=print) -> dict:
    """A check of the machinery, not a result: few-shot generation in one batch of 8 per kind on 16 dev items, the
    same items with the number guard, and 2 items one at a time to compare with the batch (greedy, so the texts
    should agree up to floating point)."""
    import torch
    items = D.items("scouting", "dev")[:8] + D.items("possession", "dev")[:8]
    tok, model = load()
    t0 = time.perf_counter()
    batch, seconds = generate(tok, model, items, "few_shot", log=log)
    batch_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    guarded, guard_seconds = generate(tok, model, items, "few_shot", guard=True, log=log)
    guard_s = time.perf_counter() - t0
    single = {}
    for item in (items[0], items[8]):
        one, _ = generate(tok, model, [item], "few_shot", log=log)
        single[item["key"]] = one[item["key"]][0]
    rows = score(items, {k: v[0] for k, v in batch.items()})
    grows = score(items, {k: v[0] for k, v in guarded.items()})
    record = {"items": len(items), "batch_s": batch_s, "guard_s": guard_s,
              "batch_equals_single": {k: single[k] == batch[k][0] for k in single},
              "passes_primary": sum(r["passed"] for r in rows["primary"]),
              "passes_primary_guard": sum(r["passed"] for r in grows["primary"]),
              "number_value_failures": sum("number_value" in r["rules"] for r in rows["primary"]),
              "number_value_failures_guard": sum("number_value" in r["rules"] for r in grows["primary"]),
              "peak_mps_gb": torch.mps.driver_allocated_memory() / 1e9,
              "note": "machinery check on 16 dev items; not a result"}
    _save("smoke-texts.json", {"batch": {k: v[0] for k, v in batch.items()},
                               "guard": {k: v[0] for k, v in guarded.items()}, "single": single})
    _save("smoke.json", record)
    return record


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
    if round_ == 2:
        allowed, gain = round2_allowed()
        if not allowed:
            raise SystemExit(f"RFT round 2 refused: round 1 changed the mean dev pass rate by {gain}, under "
                             f"{RFT['round2_gain']} (the pre-registered rule)")
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


def stage_a3(deadline, log=print) -> dict:
    arm = dev_best_arm()
    arm_dir = adapter_of(arm)
    tok, model = load(arm_dir)
    items = dev_items()
    texts, seconds = generate(tok, model, items, "compact", guard=True, log=log)
    rows = score(items, {k: v[0] for k, v in texts.items()})
    _save("a3-dev-texts.json", {k: v[0] for k, v in texts.items()})
    record = {"arm": arm, "a3": _by_kind(rows), "seconds_per_item": _mean(seconds.values())}
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
    """The dev choices, written before any test item is read (amendments e, f, h)."""
    arms = dev_arms()
    a1_best = next(a for a in arms if a.startswith("a1"))
    a2_final = max((a for a in arms if a.startswith("a2")), default=None)
    candidates = [a1_best] + ([a2_final] if a2_final else [])
    best = {kind: max(candidates, key=lambda a: (arms[a]["primary"][kind]["pass_rate"], -candidates.index(a)))
            for kind in ("scouting", "possession")}
    allowed, gain = round2_allowed()
    a3 = _load("a3-dev.json")
    return {"a1_epoch": best_epoch("a1"), "a1_best": a1_best, "a2_final": a2_final, "best_by_kind": best,
            "a3_arm": dev_best_arm(), "a3_dev": a3 and {k: v for k, v in a3.items() if k != "arm"},
            "rft_round1_gain_mean_over_kinds": gain, "rft_round2_ran": "a2-r2-epoch1" in arms,
            "rft_round2_allowed": allowed, "targets": target_drops("train"),
            "dev": {arm: {k: v for k, v in d.items() if k in ("primary", "freeze", "mean_pass", "epoch")}
                    for arm, d in arms.items()},
            "a0_dev": _load("a0-dev.json"),
            "a1_epochs": [{k: v for k, v in d.items() if k in ("primary", "freeze", "mean_pass", "epoch",
                                                                "seconds_per_item")}
                          for d in json.loads((run_dir() / "a1" / "state.json").read_text())["dev"]],
            "a1_train_s": json.loads((run_dir() / "a1" / "state.json").read_text()).get("train_s"),
            "a1_skipped_over_1024_tokens": json.loads((run_dir() / "a1" / "state.json").read_text()).get("skipped"),
            "rft_candidates": {tag: {k: v for k, v in (_load(f"{tag}-candidates.json") or {}).items() if k != "kept"}
                               | _kept_by_kind(_load(f"{tag}-candidates.json"))
                               for tag in ("a2-r1", "a2-r2") if _load(f"{tag}-candidates.json")}}


def _kept_by_kind(candidates) -> dict:
    kept = (candidates or {}).get("kept", {})
    out = {}
    for kind in ("scouting", "possession"):
        keys = [k for k in kept if ("/" in k) == (kind == "possession")]
        out[kind] = {"pool_items": len(keys), "items_with_a_pass": sum(bool(kept[k]) for k in keys),
                     "kept_texts": sum(len(kept[k]) for k in keys)}
    return out


def adapter_of(arm: str) -> Path:
    tag, epoch = arm.rsplit("-epoch", 1)
    return run_dir() / tag / f"epoch{epoch}"


def paired(rows_a: list, rows_b: list) -> list:
    """95 percent interval of (b minus a) pass rate, units resampled together (genai_eval.bootstrap)."""
    out = E.bootstrap({"a": rows_a, "b": rows_b}, ("pass_rate",))
    return out["differences"]["b minus a: pass_rate"]


def _as_number_ok(rows: list) -> list:
    """Rows whose `passed` is "no number_value failure", so a pass-rate bootstrap gives minus the failure rate."""
    return [dict(r, passed="number_value" not in r["rules"]) for r in rows]


def reading(diff: list, point: float, rule: str) -> str:
    lo, hi = diff
    if rule == "h1":
        return "passes" if point >= 0.10 and lo > 0 else "fails" if hi < 0.10 else "inconclusive"
    if rule == "h3":
        return "non-inferior" if lo >= -0.05 else "inferior" if hi < -0.05 else "inconclusive"
    return "helps" if lo > 0 else "hurts" if hi < 0 else "inconclusive"


def run_arms(items: list, choices: dict, tag: str, log=print, resume: bool = False) -> tuple:
    """Every arm on `items`. Each arm is scored under both checkers as soon as its texts exist, and its texts and full
    rows are saved outside the repository (amendment j). With `resume`, an arm whose rows are saved is read back and
    not generated again (greedy decoding, so nothing is chosen again). Used by the test scorer, the resume entry and,
    unchanged, by the rehearsal on dev items. Returns (rows, timing, texts)."""
    import gc

    scouting = [i for i in items if i["kind"] == "scouting"]
    texts, timing, rows = {}, {}, {}

    def saved(arm) -> bool:
        if not resume or _load(f"{tag}-rows-{arm}.json") is None:
            return False
        rows[arm] = _load(f"{tag}-rows-{arm}.json")
        texts[arm] = _load(f"{tag}-texts-{arm}.json")
        timing[arm] = (_load(f"{tag}-timing.json") or {}).get(arm)
        log(f"  {arm}: rows saved before the crash, not generated again")
        return True

    def keep(arm, subset, found, seconds=None):
        texts[arm] = found
        timing[arm] = _mean(seconds.values()) if seconds else None
        rows[arm] = score(subset, found)
        _save(f"{tag}-texts-{arm}.json", found)
        _save(f"{tag}-rows-{arm}.json", rows[arm])
        _save(f"{tag}-timing.json", timing)

    def free(model):
        import torch
        del model
        gc.collect()
        torch.mps.empty_cache()

    if not (saved("a0") and saved("a0_p3")):
        tok, model = load()
        if "a0" not in rows:
            found, seconds = generate(tok, model, items, "few_shot", log=log)
            keep("a0", items, {k: v[0] for k, v in found.items()}, seconds)
        if "a0_p3" not in rows and not saved("a0_p3"):
            found, seconds = generate(tok, model, scouting, "p3", log=log)
            keep("a0_p3", scouting, {k: v[0] for k, v in found.items()}, seconds)
        free(model)
    arms = {choices["a1_best"], *choices["best_by_kind"].values(), choices["a3_arm"]}
    arms |= {choices["a2_final"]} if choices.get("a2_final") else set()
    for arm in sorted(arms):
        needs = [arm] + (["a3"] if arm == choices["a3_arm"] else []) + \
                ([f"{arm}+repair"] if arm in choices["best_by_kind"].values() else [])
        todo = [a for a in needs if not saved(a)]
        if not todo:
            continue
        tok, model = load(adapter_of(arm))
        if arm in todo:
            found, seconds = generate(tok, model, items, "compact", log=log)
            keep(arm, items, {k: v[0] for k, v in found.items()}, seconds)
        if "a3" in todo:
            found, seconds = generate(tok, model, items, "compact", guard=True, log=log)
            keep("a3", items, {k: v[0] for k, v in found.items()}, seconds)
        if f"{arm}+repair" in todo:
            keep(f"{arm}+repair", items, repair(tok, model, items, texts[arm], log=log))
        free(model)
    snapshot = {t["key"]: t for t in
                json.loads(D.snapshot_file("genai/scouting.json").read_text(encoding="utf-8"))["items"]}
    from export import genai as G
    haiku = {}
    for item in scouting:
        text = snapshot.get(item["key"])
        if text is not None and text["input_sha256"] != G.input_sha256(item):
            raise SystemExit(f"{item['key']}: the snapshot Haiku text was written from another input")
        haiku[item["key"]] = text["text"] if text else None
    keep("haiku", scouting, haiku)
    keep("haiku_cache", scouting, {r["key"]: r["_text"] for r in E._cache_text_rows(scouting)})
    return rows, timing, texts


def read_hypotheses(rows: dict, choices: dict) -> dict:
    def kind_rows(arm, kind, name="primary"):
        return [r for r in rows[arm][name] if ("/" in r["key"]) == (kind == "possession")]

    def rate(arm, kind):
        return E.arm_measures(kind_rows(arm, kind))["pass_rate"]

    hyp = {}
    for kind in ("scouting", "possession"):
        best = choices["best_by_kind"][kind]
        d, p = paired(kind_rows("a0", kind), kind_rows(best, kind)), rate(best, kind) - rate("a0", kind)
        hyp[f"H1 {kind}: {best} minus a0"] = {"point": p, "interval": d, "reading": reading(d, p, "h1")}
        if choices.get("a2_final"):
            a2, a1 = choices["a2_final"], choices["a1_best"]
            d, p = paired(kind_rows(a1, kind), kind_rows(a2, kind)), rate(a2, kind) - rate(a1, kind)
            hyp[f"H2 {kind}: {a2} minus {a1}"] = {"point": p, "interval": d, "reading": reading(d, p, "h2")}
        base = choices["a3_arm"]
        d, p = paired(kind_rows(base, kind), kind_rows("a3", kind)), rate("a3", kind) - rate(base, kind)
        ok = paired(_as_number_ok(kind_rows(base, kind)), _as_number_ok(kind_rows("a3", kind)))
        fail_a = E.arm_measures(kind_rows(base, kind))["rule_failure_rate"]["number_value"]
        fail_b = E.arm_measures(kind_rows("a3", kind))["rule_failure_rate"]["number_value"]
        nv = [-ok[1], -ok[0]]
        hyp[f"H4 {kind}: a3 minus {base}"] = {
            "point": p, "interval": d, "reading": reading(d, p, "h4"),
            "number_value_failure_point": fail_b - fail_a, "number_value_failure_interval": nv,
            "number_value_reading": "fewer wrong numbers" if nv[1] < 0 else "more wrong numbers" if nv[0] > 0
            else "inconclusive"}
    best = choices["best_by_kind"]["scouting"]
    for haiku, decides in (("haiku_cache", True), ("haiku", False)):
        d = paired(kind_rows(haiku, "scouting"), kind_rows(f"{best}+repair", "scouting"))
        p = rate(f"{best}+repair", "scouting") - rate(haiku, "scouting")
        hyp[f"H3 scouting: {best}+repair minus {haiku}"] = {"point": p, "interval": d, "reading": reading(d, p, "h3"),
                                                           "decides": decides}
    return hyp


def variety(rows: dict, items: list, texts: dict) -> dict:
    """Per arm and kind (amendment k): frame variety and distinct-2 with bootstrap intervals by unit, the share of
    possession texts identical to the template's text of the item, and the share of the arm's sentence frames found
    among the template's frames of the same items. No hypothesis reads them."""
    by_key = {i["key"]: i for i in items}
    template = {k: template_text(i) for k, i in by_key.items()}
    template_frames = {kind: {f for k, i in by_key.items() if i["kind"] == kind for f in E.frames_of(template[k], i)}
                       for kind in ("scouting", "possession")}
    out = {}
    for arm, by in rows.items():
        for kind in ("scouting", "possession"):
            mine = [r for r in by["primary"] if ("/" in r["key"]) == (kind == "possession")]
            if not mine:
                continue
            boot = E.bootstrap({arm: mine}, ("frame_variety", "distinct_2"))["arms"][arm]
            m = E.arm_measures(mine)
            frames = [f for r in mine for f in r["frames"]]
            entry = {"frame_variety": m["frame_variety"], "frame_variety_interval": boot["frame_variety"],
                     "distinct_2": m["distinct_2"], "distinct_2_interval": boot["distinct_2"],
                     "share_frames_in_template": (sum(f in template_frames[kind] for f in frames) / len(frames))
                     if frames else None}
            if kind == "possession":
                keys = [r["key"] for r in mine]
                entry["share_identical_to_template"] = sum(
                    (texts.get(arm) or {}).get(k) == template[k] for k in keys) / len(keys)
            out.setdefault(arm, {})[kind] = entry
    return out


def summarize(rows: dict, timing: dict, choices: dict, items: list, texts: dict) -> dict:
    return {"hypotheses": read_hypotheses(rows, choices), "seconds_per_item": timing,
            "variety": variety(rows, items, texts),
            "arms": {arm: {name: {kind: E.arm_measures([r for r in rs if ("/" in r["key"]) == (kind == "possession")])
                                  for kind in ("scouting", "possession")
                                  if any(("/" in r["key"]) == (kind == "possession") for r in rs)}
                           for name, rs in by.items()} for arm, by in rows.items()},
            "rows": {arm: {name: E._strip(rs) for name, rs in by.items()} for arm, by in rows.items()}}


DEV_JSON = REPO_ROOT / "reports" / "research" / f"{EXPERIMENT}-dev.json"


def set_size(size: str) -> None:
    """Point every stage at one model size: its model, number format, run folder, data folder and dev json."""
    global MODEL, SIZE, EXPERIMENT, DTYPE, DEV_JSON
    MODEL, DTYPE = SIZES[size]
    SIZE, EXPERIMENT = size, f"genai-E2-{size}"
    DEV_JSON = REPO_ROOT / "reports" / "research" / f"{EXPERIMENT}-dev.json"


def dev_json_clean() -> str:
    """The sha256 of the committed dev choices; SystemExit unless git tracks the file and it has no change."""
    rel = str(DEV_JSON.relative_to(REPO_ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT, capture_output=True)
    changed = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=REPO_ROOT, capture_output=True,
                             text=True).stdout.strip()
    if tracked.returncode != 0 or changed:
        raise SystemExit(f"{rel} must be committed and unchanged before the test stage")
    return D.sha256_file(DEV_JSON)


def tree_clean() -> None:
    """SystemExit when tools/ or export/ has an uncommitted change (amendment l)."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "export"], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("tools/ or export/ has uncommitted changes: commit them before the test stage")


def rehearsal_matches(dev_sha: str, head: str) -> None:
    """SystemExit unless the rehearsal ran on this dev json and this commit (amendment l)."""
    rehearsal = _load("rehearse.json")
    if rehearsal is None:
        raise SystemExit("rehearse the test path on the dev items first (stage rehearse)")
    if rehearsal.get("dev_json_sha256") != dev_sha or rehearsal.get("head") != head:
        raise SystemExit("the rehearsal ran on another dev json or another commit: rehearse again")


def stage_rehearse(log=print) -> dict:
    """The test path on the dev items, without the ledger (amendment i): same arms, same scoring, same readings."""
    dev_sha, head = dev_json_clean(), D.git_head()
    choices = json.loads(DEV_JSON.read_text())
    items = dev_items()
    rows, timing, texts = run_arms(items, choices, "rehearse", log)
    record = {"dev_json_sha256": dev_sha, "head": head, **summarize(rows, timing, choices, items, texts)}
    _save("rehearse.json", record)
    return record


def _final(rows, timing, texts, choices, dev_sha, head, items, resume_used: bool) -> dict:
    record = {"experiment": EXPERIMENT, "model": MODEL, "checkers": CHECKERS, "dev_json_sha256": dev_sha,
              "head": head, "resume_used": resume_used,
              "choices": {k: v for k, v in choices.items() if k not in ("dev", "a0_dev", "a1_epochs")},
              **summarize(rows, timing, choices, items, texts)}
    path = REPO_ROOT / "reports" / "research" / f"{EXPERIMENT}-final.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def _test_items() -> list:
    return D.items("scouting", "test") + D.items("possession", "test")


def stage_test(log=print) -> dict:
    """Every arm on the test items, scored once; the hypotheses read on the primary checker. A crash inside the
    scorer leaves a crash record for the one resume entry (amendment j)."""
    tree_clean()
    dev_sha, head = dev_json_clean(), D.git_head()
    rehearsal_matches(dev_sha, head)
    choices = json.loads(DEV_JSON.read_text())
    box = {}

    def scorer():
        box["items"] = _test_items()
        return run_arms(box["items"], choices, "test", log)

    try:
        if SIZE == "0.5B":
            rows, timing, texts = D.score_test_once("genai-E2-0.5B", scorer)
        else:
            rows, timing, texts = D.score_test_once("genai-E2-1.5B", scorer)
    except BaseException as exc:
        if "items" in box:
            _save("test-crash.json", {"dev_json_sha256": dev_sha, "head": head, "error": type(exc).__name__,
                                      "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise
    return _final(rows, timing, texts, choices, dev_sha, head, box["items"], resume_used=False)


def stage_test_resume(log=print) -> dict:
    """The one pre-registered resume entry: only after a crash of the scoring call, on the same commit and dev
    choices; it generates only the arms with no saved rows."""
    crash = _load("test-crash.json")
    if crash is None:
        raise SystemExit(f"no crash record of {EXPERIMENT}: the resume entry is refused")
    tree_clean()
    dev_sha, head = dev_json_clean(), D.git_head()
    if crash["dev_json_sha256"] != dev_sha or crash["head"] != head:
        raise SystemExit("the resume entry needs the commit and the dev choices of the crashed run")
    choices = json.loads(DEV_JSON.read_text())
    box = {}

    def scorer():
        box["items"] = _test_items()
        return run_arms(box["items"], choices, "test", log, resume=True)

    if SIZE == "0.5B":
        rows, timing, texts = D.score_test_once("genai-E2-0.5B-resume", scorer)
    else:
        rows, timing, texts = D.score_test_once("genai-E2-1.5B-resume", scorer)
    return _final(rows, timing, texts, choices, dev_sha, head, box["items"], resume_used=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="genai-E2 on Qwen2.5 0.5B or 1.5B (docs/research/genai.md)")
    parser.add_argument("stage", choices=("smoke", "a0", "sft", "rft", "rft2", "a3", "choose", "rehearse", "test",
                                          "test-resume"))
    parser.add_argument("--max-hours", type=float, default=MAX_STRETCH_H)
    parser.add_argument("--size", choices=sorted(SIZES), default="0.5B")
    args = parser.parse_args(argv)
    set_size(args.size)
    if args.max_hours > MAX_STRETCH_H:
        print(f"a stretch holds the lock for at most {MAX_STRETCH_H} hours", file=sys.stderr)
        return 1
    # a training stage stops 25 minutes before the end, so the dev score of the epoch fits in the stretch; a
    # generation stage stops between batches 3 minutes before it
    reserve = 25 * 60 if args.stage in ("sft", "rft", "rft2") else 3 * 60
    deadline = time.perf_counter() + args.max_hours * 3600 - reserve
    started = time.perf_counter()
    if args.stage == "smoke":
        record = stage_smoke()
    elif args.stage == "a0":
        record = stage_a0(deadline)
    elif args.stage == "sft":
        record = stage_sft(deadline)
    elif args.stage in ("rft", "rft2"):
        record = stage_rft(deadline, 1 if args.stage == "rft" else 2)
    elif args.stage == "a3":
        record = stage_a3(deadline)
    elif args.stage == "choose":
        record = choose()
        DEV_JSON.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    elif args.stage == "rehearse":
        record = stage_rehearse()
    elif args.stage == "test":
        record = stage_test()
    else:
        record = stage_test_resume()
    print(json.dumps({"stage": args.stage, "wall_s": time.perf_counter() - started,
                      "summary": _short(record)}, indent=1))
    return 0


def _short(record):
    if isinstance(record, dict):
        return {k: _short(v) for k, v in record.items() if k not in ("rows", "frames")}
    return record


if __name__ == "__main__":
    raise SystemExit(main())
