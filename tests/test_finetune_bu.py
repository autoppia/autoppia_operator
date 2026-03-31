from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from training.finetune_bu import (
    NUMPY_PACKAGE,
    ProgressMetricsCallback,
    _ensure_deps,
    _load_trainable_model,
    _torch_runtime_requires_repair,
)


def test_load_trainable_model_uses_image_text_loader_for_qwen3_vl_moe(monkeypatch) -> None:
    calls: list[tuple[str, str, object, object]] = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(model_type="qwen3_vl_moe")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("causal", args[0], kwargs["quantization_config"], kwargs.get("max_memory")))
            return "causal"

    class FakeAutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("vision", args[0], kwargs["quantization_config"], kwargs.get("max_memory")))
            return "vision"

    fake_transformers = SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoModelForImageTextToText=FakeAutoModelForImageTextToText,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(total_memory=80 * 1024**3),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = _load_trainable_model("browser-use/bu-30b-a3b-preview", bnb_config="bnb", bf16=True)

    assert result == "vision"
    assert calls == [("vision", "browser-use/bu-30b-a3b-preview", "bnb", {0: "78GiB", "cpu": "256GiB"})]


def test_load_trainable_model_uses_causal_loader_for_text_models(monkeypatch) -> None:
    calls: list[tuple[str, str, object, object]] = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(model_type="llama")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("causal", args[0], kwargs["quantization_config"], kwargs.get("max_memory")))
            return "causal"

    class FakeAutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("vision", args[0], kwargs["quantization_config"], kwargs.get("max_memory")))
            return "vision"

    fake_transformers = SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoModelForImageTextToText=FakeAutoModelForImageTextToText,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda index: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = _load_trainable_model("meta-llama/test", bnb_config="bnb", bf16=False)

    assert result == "causal"
    assert calls == [("causal", "meta-llama/test", "bnb", None)]


def test_torch_runtime_requires_repair_when_cuda_runtime_is_incompatible(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _torch_runtime_requires_repair() is True


def test_torch_runtime_requires_repair_accepts_cuda_118_runtime(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.8.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _torch_runtime_requires_repair() is False


def test_ensure_deps_repairs_and_restarts_before_loading_torch(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.delenv("BU_FINETUNE_DEPS_RESTARTED", raising=False)
    monkeypatch.setattr("training.finetune_bu._torch_runtime_requires_repair", lambda: True)
    monkeypatch.setattr("training.finetune_bu._repair_torch_runtime", lambda: calls.append("repair"))
    monkeypatch.setattr("training.finetune_bu._restart_current_process", lambda: calls.append("restart"))

    _ensure_deps()

    assert calls == ["repair", "restart"]


def test_ensure_deps_fails_if_repair_restart_did_not_fix_runtime(monkeypatch) -> None:
    monkeypatch.setenv("BU_FINETUNE_DEPS_RESTARTED", "1")
    monkeypatch.setattr("training.finetune_bu._torch_runtime_requires_repair", lambda: True)

    try:
        _ensure_deps()
    except RuntimeError as exc:
        assert "still mismatched" in str(exc)
    else:
        raise AssertionError("expected _ensure_deps() to fail after a bad repair restart")


def test_repair_torch_runtime_installs_numpy1_before_torch(monkeypatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr("training.finetune_bu.subprocess.check_call", lambda cmd: commands.append(cmd))
    monkeypatch.setattr("training.finetune_bu.importlib.invalidate_caches", lambda: None)

    from training.finetune_bu import _repair_torch_runtime

    _repair_torch_runtime()

    assert commands[0][-1] == NUMPY_PACKAGE
    assert "torch==2.8.0" in commands[1]


def test_training_arguments_compat_uses_eval_strategy_when_needed(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeTrainingArguments:
        def __init__(
            self,
            *,
            output_dir,
            num_train_epochs,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            lr_scheduler_type,
            warmup_ratio,
            bf16,
            fp16,
            logging_steps,
            save_strategy,
            eval_strategy,
            save_total_limit,
            report_to,
            max_grad_norm,
            gradient_checkpointing,
            optim,
        ):
            captured["eval_strategy"] = eval_strategy

    monkeypatch.setattr("training.finetune_bu._ensure_deps", lambda: None)

    class FakeDataset:
        @staticmethod
        def from_dict(payload):
            return payload

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        @staticmethod
        def save_pretrained(output_dir):
            captured["tokenizer_save_pretrained"] = output_dir

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)

        @staticmethod
        def get_nb_trainable_parameters():
            return (1, 2)

        @staticmethod
        def save_pretrained(output_dir):
            captured["save_pretrained"] = output_dir

    class FakeSFTTrainer:
        def __init__(
            self,
            *,
            model,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
        ):
            captured["trainer_eval_dataset"] = eval_dataset
            captured["trainer_processing_class"] = processing_class
            captured["trainer_model"] = model
            captured["trainer_args"] = args

        @staticmethod
        def train():
            return SimpleNamespace(training_loss=0.5)

    monkeypatch.setattr("training.finetune_bu.load_sft_jsonl", lambda path: [{"messages": [{"role": "user", "content": "hi"}]}])
    monkeypatch.setattr("training.finetune_bu._load_trainable_model", lambda *args, **kwargs: FakeModel())
    real_exists = __import__("os").path.exists
    monkeypatch.setattr(
        "training.finetune_bu.os.path.exists",
        lambda path: True if path == "val.jsonl" else real_exists(path),
    )

    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=lambda **kwargs: kwargs,
            TaskType=SimpleNamespace(CAUSAL_LM="causal"),
            get_peft_model=lambda model, config: model,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
            BitsAndBytesConfig=lambda **kwargs: kwargs,
            TrainingArguments=FakeTrainingArguments,
        ),
    )
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(SFTTrainer=FakeSFTTrainer))

    from training.finetune_bu import train

    train(
        data_path="train.jsonl",
        val_data_path="val.jsonl",
        output_dir=str(tmp_path / "models" / "out"),
    )

    assert captured["eval_strategy"] == "epoch"
    assert captured["trainer_eval_dataset"] == {"text": ["<|user|>\nhi\n<|end|>"]}
    assert captured["trainer_processing_class"].__class__.__name__ == "FakeTokenizer"


def test_progress_metrics_callback_writes_live_metrics(tmp_path) -> None:
    callback = ProgressMetricsCallback(
        output_dir=str(tmp_path),
        base_model="browser-use/bu-30b-a3b-preview",
        epochs=3,
        lora_rank=32,
        trainable_params=10,
        total_params=100,
        train_examples=105,
        val_examples=12,
        started_at="2026-03-26T15:55:10Z",
    )

    state = SimpleNamespace(global_step=2, max_steps=21, epoch=0.29)
    callback.on_log(None, state, None, logs={"loss": 1.234, "learning_rate": 2e-4})

    metrics = json.loads((tmp_path / "train_metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "training"
    assert metrics["base_model"] == "browser-use/bu-30b-a3b-preview"
    assert metrics["train_examples"] == 105
    assert metrics["val_examples"] == 12
    assert metrics["global_step"] == 2
    assert metrics["max_steps"] == 21
    assert metrics["train_loss"] == 1.234


def test_progress_metrics_callback_supports_epoch_hooks(tmp_path) -> None:
    callback = ProgressMetricsCallback(
        output_dir=str(tmp_path),
        base_model="browser-use/bu-30b-a3b-preview",
        epochs=3,
        lora_rank=32,
        trainable_params=10,
        total_params=100,
        train_examples=105,
        val_examples=12,
        started_at="2026-03-26T15:55:10Z",
    )

    state = SimpleNamespace(global_step=12, max_steps=84, epoch=1.0)
    control = object()

    assert callback.on_epoch_begin(None, state, control) is control
    metrics = json.loads((tmp_path / "train_metrics.json").read_text(encoding="utf-8"))
    assert metrics["phase"] == "epoch_begin"

    assert callback.on_epoch_end(None, state, control) is control
    metrics = json.loads((tmp_path / "train_metrics.json").read_text(encoding="utf-8"))
    assert metrics["phase"] == "epoch_end"


def test_progress_metrics_callback_noops_unknown_hooks(tmp_path) -> None:
    callback = ProgressMetricsCallback(
        output_dir=str(tmp_path),
        base_model="browser-use/bu-30b-a3b-preview",
        epochs=3,
        lora_rank=32,
        trainable_params=10,
        total_params=100,
        train_examples=105,
        val_examples=12,
        started_at="2026-03-26T15:55:10Z",
    )
    control = object()

    assert callback.on_step_begin(None, None, control) is control
