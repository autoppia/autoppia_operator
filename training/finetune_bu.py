"""Fine-tune browser-use/bu-30b-a3b-preview with LoRA on SFT trajectory data.

Self-contained script designed to run on a RunPod A100/H100 GPU pod.
Installs dependencies, loads SFT data, trains LoRA adapters, and saves weights.

Usage:
    python -m training.finetune_bu \
        --data data/sft/train.jsonl \
        --val-data data/sft/val.jsonl \
        --output-dir models/bu-30b-lora \
        --epochs 3 --lr 2e-4 --lora-rank 32

Hardware requirement: A100 80GB (4-bit quantisation keeps memory < 40GB).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency bootstrap (for RunPod pods that start with bare PyTorch image)
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "transformers>=4.40",
    "peft>=0.10",
    "bitsandbytes>=0.43",
    "datasets",
    "accelerate>=0.28",
    "trl>=0.8",
    "safetensors",
]


def _ensure_deps() -> None:
    """Install missing Python packages."""
    for pkg in REQUIRED_PACKAGES:
        base = pkg.split(">=")[0].split("==")[0]
        try:
            __import__(base)
        except ImportError:
            logger.info("Installing %s …", pkg)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
            )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sft_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load HuggingFace-messages-format JSONL file."""
    examples: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                examples.append(obj)
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MODEL_ID = "browser-use/bu-30b-a3b-preview"


def train(
    data_path: str,
    output_dir: str,
    val_data_path: Optional[str] = None,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_seq_len: int = 2048,
    bf16: bool = True,
) -> Dict[str, Any]:
    """Run LoRA fine-tuning on the bu-30b model.

    Returns dict with training metrics.
    """
    _ensure_deps()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    # --- Quantisation config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load model + tokeniser ---
    logger.info("Loading model %s (4-bit) …", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    model.config.use_cache = False

    # --- LoRA config ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    # --- Load data ---
    train_examples = load_sft_jsonl(data_path)

    def _format_messages(example: Dict[str, Any]) -> str:
        """Convert messages list to a single string for SFT."""
        parts: List[str] = []
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|end|>")
        return "\n".join(parts)

    train_texts = [_format_messages(ex) for ex in train_examples]
    train_ds = Dataset.from_dict({"text": train_texts})

    val_ds = None
    if val_data_path and os.path.exists(val_data_path):
        val_examples = load_sft_jsonl(val_data_path)
        val_texts = [_format_messages(ex) for ex in val_examples]
        val_ds = Dataset.from_dict({"text": val_texts})

    # --- Training args ---
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_ds else "no",
        save_total_limit=2,
        report_to="none",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    # --- Train ---
    logger.info("Starting training …")
    result = trainer.train()

    # --- Save adapter ---
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Adapter saved to %s", output_dir)

    metrics = {
        "train_loss": result.training_loss,
        "epochs": epochs,
        "lora_rank": lora_rank,
        "trainable_params": trainable,
        "total_params": total,
        "train_examples": len(train_examples),
    }
    # Save metrics
    with open(os.path.join(output_dir, "train_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune bu-30b with LoRA")
    parser.add_argument("--data", default="data/sft/train.jsonl", help="SFT train JSONL")
    parser.add_argument("--val-data", default="data/sft/val.jsonl", help="SFT val JSONL")
    parser.add_argument("--output-dir", default="models/bu-30b-lora", help="Output dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    train(
        data_path=args.data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        bf16=not args.no_bf16,
    )


if __name__ == "__main__":
    main()
