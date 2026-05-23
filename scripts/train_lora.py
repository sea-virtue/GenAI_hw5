from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.io_utils import read_jsonl


SYSTEM_PROMPT = (
    "你是清华大学校园问答助手。请只依据可靠的校园公开信息回答，"
    "表达清楚、简洁；如果问题缺少依据，请说明需要查询官方来源。"
)


def format_prompt(tokenizer, instruction: str, input_text: str = "") -> str:
    user_content = instruction.strip()
    if input_text.strip():
        user_content += "\n" + input_text.strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_content}\n<|assistant|>\n"


def tokenize_example(tokenizer, row: dict, max_length: int) -> dict:
    prompt = format_prompt(tokenizer, row["instruction"], row.get("input", ""))
    answer = row["output"].strip() + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    overflow = len(prompt_ids) + len(answer_ids) - max_length
    if overflow > 0:
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
        else:
            keep_answer = max(8, max_length // 2)
            prompt_ids = prompt_ids[-(max_length - keep_answer) :]
            answer_ids = answer_ids[:keep_answer]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class CausalCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(args: argparse.Namespace):
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    return model


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "optim": args.optim,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def train(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.model_name)
    train_rows = list(read_jsonl(args.train_file))
    eval_rows = list(read_jsonl(args.eval_file))
    if not train_rows:
        raise ValueError(f"empty training file: {args.train_file}")
    if not eval_rows:
        raise ValueError(f"empty eval file: {args.eval_file}")

    train_dataset = Dataset.from_list(train_rows).map(
        lambda row: tokenize_example(tokenizer, row, args.max_length),
        remove_columns=list(train_rows[0].keys()),
        desc="tokenize train",
    )
    eval_dataset = Dataset.from_list(eval_rows).map(
        lambda row: tokenize_example(tokenizer, row, args.max_length),
        remove_columns=list(eval_rows[0].keys()),
        desc="tokenize eval",
    )

    model = load_model(args)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter and tokenizer to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen campus QA assistant.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--eval-file", default="data/processed/eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen3-0.6b-thuqa-lora")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bf16. Recommended on A100/H100/RTX 40 series.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--optim", default="adamw_torch")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
