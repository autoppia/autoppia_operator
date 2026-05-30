"""Fine-tune browser-use/bu-30b-a3b-preview with LoRA on SFT trajectory data.

Self-contained script designed to run on a RunPod A100/H100 GPU pod.
Installs dependencies, loads SFT data, trains LoRA adapters, and saves weights.

Usage:
    python -m training.finetune_bu \
        --data data/autocinema_trajectory_harvest/sft/train.jsonl \
        --val-data data/autocinema_trajectory_harvest/sft/val.jsonl \
        --output-dir models/bu-30b-lora \
        --epochs 3 --lr 2e-4 --lora-rank 32

Hardware requirement: A100 80GB (4-bit quantisation keeps memory < 40GB).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency bootstrap (for RunPod pods that start with bare PyTorch image)
# ---------------------------------------------------------------------------
NUMPY_PACKAGE = "numpy<2"
REQUIRED_PACKAGES = [
    "requests>=2.32.3",
    "transformers>=4.55,<5",
    "peft>=0.17.0",
    "bitsandbytes>=0.47.0",
    "datasets>=4.0.0",
    "accelerate>=1.10.0",
    "trl>=0.23.0",
    "safetensors>=0.5.0",
]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
RESTART_SENTINEL = "BU_FINETUNE_DEPS_RESTARTED"
PINNED_TORCH_PACKAGES = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "torchaudio==2.8.0",
]


def _torch_runtime_requires_repair() -> bool:
    try:
        import torch
    except ImportError:
        return True

    version = str(getattr(torch, "__version__", "") or "")
    cuda_version = str(getattr(torch.version, "cuda", "") or "")
    if not version.startswith("2.8."):
        return True
    if not cuda_version.startswith("12.8"):
        return True
    return not bool(torch.cuda.is_available())


def _repair_torch_runtime() -> None:
    logger.info("Installing %s to avoid the NumPy 2.x ABI break in the RunPod image", NUMPY_PACKAGE)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", NUMPY_PACKAGE],
    )
    logger.info("Installing torch 2.8.0/cu128 to match the RunPod 12.8 driver")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--no-cache-dir",
            "--force-reinstall",
            "--index-url",
            TORCH_INDEX_URL,
            *PINNED_TORCH_PACKAGES,
        ]
    )
    importlib.invalidate_caches()


def _restart_current_process() -> None:
    logger.info("Restarting interpreter so freshly installed torch wheels are re-imported cleanly")
    env = os.environ.copy()
    env[RESTART_SENTINEL] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _ensure_deps() -> None:
    """Install or upgrade the training stack to a compatible set."""
    if _torch_runtime_requires_repair():
        if os.environ.get(RESTART_SENTINEL) == "1":
            raise RuntimeError("Torch runtime still mismatched after dependency repair restart")
        _repair_torch_runtime()
        _restart_current_process()
    for pkg in REQUIRED_PACKAGES:
        logger.info("Installing/upgrading %s …", pkg)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", pkg],
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sft_jsonl(path: str) -> list[dict[str, Any]]:
    """Load HuggingFace-messages-format JSONL file."""
    examples: list[dict[str, Any]] = []
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_train_metrics(output_dir: str, payload: dict[str, Any]) -> None:
    _write_json(Path(output_dir) / "train_metrics.json", payload)


class ProgressMetricsCallback:
    def __init__(
        self,
        *,
        output_dir: str,
        base_model: str,
        epochs: int,
        lora_rank: int,
        trainable_params: int,
        total_params: int,
        train_examples: int,
        val_examples: int,
        started_at: str,
    ) -> None:
        self.output_dir = output_dir
        self.base_model = base_model
        self.epochs = epochs
        self.lora_rank = lora_rank
        self.trainable_params = trainable_params
        self.total_params = total_params
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.started_at = started_at

    def __getattr__(self, name: str):
        if not name.startswith("on_"):
            raise AttributeError(name)

        def _noop(*args: Any, **kwargs: Any) -> Any:
            if len(args) >= 3:
                return args[2]
            return None

        return _noop

    def _base_payload(self, state: Any) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "epochs": self.epochs,
            "lora_rank": self.lora_rank,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "train_examples": self.train_examples,
            "val_examples": self.val_examples,
            "started_at": self.started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "max_steps": int(getattr(state, "max_steps", 0) or 0),
            "epoch": getattr(state, "epoch", None),
        }

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        payload = self._base_payload(state)
        payload["status"] = "training"
        _write_train_metrics(self.output_dir, payload)

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        payload = self._base_payload(state)
        payload["status"] = "training"
        payload["phase"] = "epoch_begin"
        _write_train_metrics(self.output_dir, payload)
        return control

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        payload = self._base_payload(state)
        payload["status"] = "training"
        if isinstance(logs, dict):
            if "loss" in logs:
                payload["train_loss"] = logs["loss"]
            if "eval_loss" in logs:
                payload["eval_loss"] = logs["eval_loss"]
            if "learning_rate" in logs:
                payload["learning_rate"] = logs["learning_rate"]
        _write_train_metrics(self.output_dir, payload)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        payload = self._base_payload(state)
        payload["status"] = "save_pending"
        _write_train_metrics(self.output_dir, payload)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        payload = self._base_payload(state)
        payload["status"] = "training"
        payload["phase"] = "epoch_end"
        _write_train_metrics(self.output_dir, payload)
        return control


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MODEL_ID = "browser-use/bu-30b-a3b-preview"


def _load_trainable_model(base_model: str, *, bnb_config: Any, bf16: bool) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    torch = __import__("torch")
    max_memory = None
    if torch.cuda.is_available():
        total_vram_gib = int(torch.cuda.get_device_properties(0).total_memory // (1024**3))
        # accelerate's default 90% reservation leaves this model partly offloaded on A100 80GB.
        # Reserve a small buffer explicitly, but keep the full model on GPU.
        usable_vram_gib = max(total_vram_gib - 2, 1)
        max_memory = {0: f"{usable_vram_gib}GiB", "cpu": "256GiB"}
    common_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        "max_memory": max_memory,
    }

    if getattr(config, "model_type", "") == "qwen3_vl_moe":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(base_model, **common_kwargs)

    return AutoModelForCausalLM.from_pretrained(base_model, **common_kwargs)


def _build_sft_training_args(
    *,
    training_args_cls: type[Any],
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    bf16: bool,
    val_ds_present: bool,
    max_seq_len: int,
) -> Any:
    training_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": bf16,
        "fp16": not bf16,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
    }
    training_args_params = inspect.signature(training_args_cls.__init__).parameters
    eval_value = "epoch" if val_ds_present else "no"
    if "evaluation_strategy" in training_args_params:
        training_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in training_args_params:
        training_kwargs["eval_strategy"] = eval_value
    if "dataset_text_field" in training_args_params:
        training_kwargs["dataset_text_field"] = "text"
    if "max_length" in training_args_params:
        training_kwargs["max_length"] = max_seq_len
    elif "max_seq_length" in training_args_params:
        training_kwargs["max_seq_length"] = max_seq_len
    if "packing" in training_args_params:
        training_kwargs["packing"] = False
    return training_args_cls(**training_kwargs)


def train(
    data_path: str,
    output_dir: str,
    val_data_path: str | None = None,
    base_model: str = MODEL_ID,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_seq_len: int = 2048,
    bf16: bool = True,
) -> dict[str, Any]:
    """Run LoRA fine-tuning on the bu-30b model.

    Returns dict with training metrics.
    """
    _ensure_deps()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments

    trl_module = importlib.import_module("trl")
    SFTTrainer = trl_module.SFTTrainer
    SFTConfig = getattr(trl_module, "SFTConfig", None)

    # --- Quantisation config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load model + tokeniser ---
    logger.info("Loading model %s (4-bit) …", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_trainable_model(base_model, bnb_config=bnb_config, bf16=bf16)
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
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    # --- Load data ---
    train_examples = load_sft_jsonl(data_path)

    def _format_messages(example: dict[str, Any]) -> str:
        """Convert messages list to a single string for SFT."""
        parts: list[str] = []
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|end|>")
        return "\n".join(parts)

    train_texts = [_format_messages(ex) for ex in train_examples]
    train_ds = Dataset.from_dict({"text": train_texts})

    val_ds = None
    val_examples: list[dict[str, Any]] = []
    if val_data_path and os.path.exists(val_data_path):
        val_examples = load_sft_jsonl(val_data_path)
        val_texts = [_format_messages(ex) for ex in val_examples]
        val_ds = Dataset.from_dict({"text": val_texts})

    # --- Training args ---
    os.makedirs(output_dir, exist_ok=True)
    training_args_cls = TrainingArguments
    if SFTConfig is not None:
        sft_config_params = inspect.signature(SFTConfig.__init__).parameters
        if "dataset_text_field" in sft_config_params:
            training_args_cls = SFTConfig
    training_args = _build_sft_training_args(
        training_args_cls=training_args_cls,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        bf16=bf16,
        val_ds_present=bool(val_ds),
        max_seq_len=max_seq_len,
    )

    # --- Trainer ---
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = max_seq_len
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)
    progress_callback = ProgressMetricsCallback(
        output_dir=output_dir,
        base_model=base_model,
        epochs=epochs,
        lora_rank=lora_rank,
        trainable_params=trainable,
        total_params=total,
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    add_callback = getattr(trainer, "add_callback", None)
    if callable(add_callback):
        add_callback(progress_callback)
    _write_train_metrics(
        output_dir,
        {
            "base_model": base_model,
            "epochs": epochs,
            "lora_rank": lora_rank,
            "trainable_params": trainable,
            "total_params": total,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "started_at": progress_callback.started_at,
            "updated_at": progress_callback.started_at,
            "status": "initializing_trainer",
        },
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
        "val_examples": len(val_examples),
        "base_model": base_model,
        "status": "completed",
        "started_at": progress_callback.started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Save metrics
    _write_train_metrics(output_dir, metrics)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune bu-30b with LoRA/QLoRA")
    parser.add_argument("--data", default="data/autocinema_trajectory_harvest/sft/train.jsonl", help="SFT train JSONL")
    parser.add_argument("--val-data", default="data/autocinema_trajectory_harvest/sft/val.jsonl", help="SFT val JSONL")
    parser.add_argument("--base-model", default=MODEL_ID, help="Base model name/path")
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
        base_model=args.base_model,
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
