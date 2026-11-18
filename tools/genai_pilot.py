"""tools/genai_pilot.py: the feasibility pilot of genai-E2 (docs/research/genai.md), not an experiment.

    python3 -m export.heavy --name genai-pilot -- <venv python> -m tools.genai_pilot --out <json outside the repo>

Runs in the lane's venv (torch, transformers, peft), never the system Python. It measures whether LoRA on the Mac's
GPU (MPS) is practical: load time, seconds per LoRA step, generation speed and peak memory, on Qwen2.5-0.5B-Instruct
(20 LoRA steps on train items) and, if time is left, Qwen2.5-1.5B-Instruct (load, 5 steps, generation). Training
rows are train players only; generation runs on 3 dev items. The texts it writes are checked by export.genai only to
show the loop runs end to end: a pilot pass rate is not a result and is not reported as one. It stops itself before
MAX_S seconds. The models are read from the Hugging Face cache under HF_HOME (outside the repository), pinned by
revision.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

MODELS = {"0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
          "1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")}
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
LR = 2e-4
MAX_S = 18 * 60
MAX_NEW_TOKENS = 320


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def sft_rows(limit: int):
    """(item, target) pairs, scouting and possession in turn: train players with their Haiku text in the snapshot,
    and train possessions with their template text."""
    from tools import genai_data as D
    from tools.genai_templates import template_text
    texts = {t["key"]: t["text"] for t in
             json.loads(D.snapshot_file("genai/scouting.json").read_text(encoding="utf-8"))["items"]}
    scouting = [(item, texts[item["key"]]) for item in D.items("scouting", "train") if item["key"] in texts]
    possession = [(item, template_text(item)) for item in D.items("possession", "train")[:limit]]
    rows = [row for pair in zip(scouting, possession) for row in pair]
    return rows[:limit]


def encode(tok, system: str, item: dict, answer: str = None, compact: bool = True):
    """Token ids (and labels) of one example. compact: the trained arms' input (tools.genai_format, its fixed
    instruction as the system turn); else the full P3 prompt and JSON message, as the few-shot arm gets them."""
    from export import genai as G
    from tools import genai_format as F
    if compact:
        messages = [{"role": "system", "content": F.INSTRUCTION}, {"role": "user", "content": F.compact_message(item)}]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": G.user_message(item)}]
    prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = list(prompt_ids)
    if answer is None:
        return prompt_ids, None
    answer_ids = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def run_model(size: str, steps: int, gen_items: list, train: list, system: str, dtype_name: str, deadline: float,
              log=print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from export import genai as G

    name, revision = MODELS[size]
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    out = {"model": name, "revision": revision, "dtype": dtype_name, "device": "mps"}
    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype).to("mps")
    torch.mps.synchronize()
    out["load_s"] = time.perf_counter() - started
    out["params"] = sum(p.numel() for p in model.parameters())
    # Gradient checkpointing: the first pilot ran out of MPS memory on step 1 (about 3,800 tokens per example, 19 GiB
    # of stored activations). Recomputing each layer in the backward pass changes memory and time, not the result.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    out["gradient_checkpointing"] = True
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    out["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    step_s, step_tokens, losses, peak_driver = [], [], [], 0
    for k in range(steps):
        if time.perf_counter() > deadline:
            out["stopped_early"] = f"deadline after {k} steps"
            break
        item, text = train[k % len(train)]
        ids, labels = encode(tok, system, item, text)
        t0 = time.perf_counter()
        batch = torch.tensor([ids], device="mps")
        loss = model(input_ids=batch, labels=torch.tensor([labels], device="mps")).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        step_s.append(time.perf_counter() - t0)
        step_tokens.append(len(ids))
        losses.append(float(loss.detach().cpu()))
        peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
        log(f"{size} step {k + 1}: {step_s[-1]:.2f} s, {len(ids)} tokens, loss {losses[-1]:.3f}")
    out.update(steps=len(step_s), step_s=step_s, step_tokens=step_tokens, loss=losses,
               mean_step_s_after_first=(sum(step_s[1:]) / max(1, len(step_s) - 1)) if step_s else None)
    model.eval()
    model.config.use_cache = True
    gens = []
    for item in gen_items:
        if time.perf_counter() > deadline:
            out["generation_stopped_early"] = True
            break
        ids, _ = encode(tok, system, item)
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model.generate(torch.tensor([ids], device="mps"), max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
        torch.mps.synchronize()
        dt_s = time.perf_counter() - t0
        new = result[0][len(ids):]
        text = tok.decode(new, skip_special_tokens=True).strip()
        problems, _ = G.check_text(text, item)
        gens.append({"prompt_tokens": len(ids), "new_tokens": int(len(new)), "seconds": dt_s,
                     "tokens_per_s": len(new) / dt_s if dt_s else None, "checker_problems": len(problems)})
        peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
        log(f"{size} generate: {len(new)} tokens in {dt_s:.1f} s, {len(problems)} checker problems")
    out["generation"] = gens
    if gen_items and time.perf_counter() < deadline:
        item = gen_items[0]
        ids, _ = encode(tok, system, item, compact=False)
        t0 = time.perf_counter()
        with torch.no_grad(), model.disable_adapter():
            result = model.generate(torch.tensor([ids], device="mps"), max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
        torch.mps.synchronize()
        dt_s = time.perf_counter() - t0
        new = result[0][len(ids):]
        out["generation_full_prompt_base"] = {"prompt_tokens": len(ids), "new_tokens": int(len(new)), "seconds": dt_s}
        peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
        log(f"{size} base, full P3 prompt: {len(ids)} prompt tokens, {len(new)} new in {dt_s:.1f} s")
    out["peak_mps_driver_gb"] = peak_driver / 1e9
    out["peak_rss_gb_so_far"] = rss_gb()
    del model, optimizer
    torch.mps.empty_cache()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="genai-E2 feasibility pilot")
    parser.add_argument("--out", required=True, help="json written outside the repository")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    if Path(args.out).resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        print("the pilot output goes outside the repository", file=sys.stderr)
        return 1
    from export import genai as G
    from tools import genai_data as D
    started = time.perf_counter()
    deadline = started + MAX_S
    system = G.prompt("scouting")[1]   # the few-shot arm's prompt, timed once on the first scouting item
    train = sft_rows(64)
    gen_items = D.items("scouting", "dev")[:2] + D.items("possession", "dev")[:2]
    record = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prompt": G.prompt("scouting")[0],
              "lora": LORA, "lr": LR, "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), "max_new_tokens": MAX_NEW_TOKENS, "train_rows_available": len(train),
              "runs": []}
    record["runs"].append(run_model("0.5B", args.steps, gen_items, train, system, "float32", deadline))
    if time.perf_counter() < deadline - 5 * 60:
        record["runs"].append(run_model("1.5B", 6, gen_items, train, system, "bfloat16", deadline - 30))
    record["wall_s"] = time.perf_counter() - started
    record["peak_rss_gb"] = rss_gb()
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
