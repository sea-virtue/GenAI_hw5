from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TORCHVISION", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lora import format_prompt
from src.thu_qa.io_utils import ensure_parent, read_jsonl, write_jsonl
from src.thu_qa.text_utils import char_f1, normalize_for_match


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str, adapter_dir: str | None, bf16: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def generate_answer(model, tokenizer, instruction: str, input_text: str, args: argparse.Namespace) -> str:
    prompt = format_prompt(tokenizer, instruction, input_text)
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        output_ids = model.generate(**encoded, **generation_kwargs)
    new_tokens = output_ids[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "exact_match": 0.0, "contains_reference": 0.0, "char_f1": 0.0}
    exact = 0
    contains = 0
    f1_sum = 0.0
    for row in rows:
        pred_norm = normalize_for_match(row["prediction"])
        ref_norm = normalize_for_match(row["reference"])
        exact += int(pred_norm == ref_norm)
        contains += int(bool(ref_norm) and ref_norm in pred_norm)
        f1_sum += char_f1(row["prediction"], row["reference"])
    n = len(rows)
    return {
        "count": n,
        "exact_match": exact / n,
        "contains_reference": contains / n,
        "char_f1": f1_sum / n,
    }


def evaluate_one(name: str, adapter_dir: str | None, examples: list[dict], args: argparse.Namespace) -> dict:
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, adapter_dir, args.bf16)
    rows = []
    for example in tqdm(examples, desc=f"eval {name}"):
        prediction = generate_answer(
            model,
            tokenizer,
            example["instruction"],
            example.get("input", ""),
            args,
        )
        rows.append(
            {
                "id": example.get("id", ""),
                "model": name,
                "instruction": example["instruction"],
                "reference": example["output"],
                "prediction": prediction,
                "source_url": example.get("source_url", ""),
                "char_f1": char_f1(prediction, example["output"]),
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prediction_path = Path(args.output_dir) / f"{name}_predictions.jsonl"
    write_jsonl(prediction_path, rows)
    metrics = compute_metrics(rows)
    metrics["prediction_file"] = str(prediction_path)

    failure_rows = sorted(rows, key=lambda row: row["char_f1"])[: args.num_failure_cases]
    failure_path = Path(args.output_dir) / f"{name}_failure_cases.jsonl"
    write_jsonl(failure_path, failure_rows)
    metrics["failure_file"] = str(failure_path)
    return metrics


def evaluate(args: argparse.Namespace) -> None:
    examples = list(read_jsonl(args.eval_file))
    if args.limit:
        examples = examples[: args.limit]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    summary = {}
    summary["base"] = evaluate_one("base", None, examples, args)
    if args.adapter_dir:
        summary["finetuned"] = evaluate_one("finetuned", args.adapter_dir, examples, args)
        summary["delta"] = {
            key: summary["finetuned"][key] - summary["base"][key]
            for key in ["exact_match", "contains_reference", "char_f1"]
        }

    metrics_path = ensure_parent(Path(args.output_dir) / "metrics.json")
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base and LoRA-finetuned campus QA models.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter-dir", default="outputs/qwen3-0.6b-thuqa-lora")
    parser.add_argument("--eval-file", default="data/processed/eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--num-failure-cases", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
