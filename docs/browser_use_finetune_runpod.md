# Browser Use Fine-Tune RunPod Runbook

## Scope

This repo is prepared for the first Browser Use SFT bootstrap, not a finished fine-tune. The local bridge uses the real replayable Autocinema harvest under `data/autocinema_trajectory_harvest/` and exports only successful replayable episodes into SFT train/val JSONL.

Chosen defaults:

- Base model: `browser-use/bu-30b-a3b-preview`
- Training method: LoRA/QLoRA SFT
- Preferred GPU: `NVIDIA A100 80GB PCIe`
- Existing bootstrap anchor pod: `bu-30b-a3b-bootstrap` / `59dd4g1snevkqs`

Why this path:

- `browser-use/bu-30b-a3b-preview` matches the browser-agent policy direction already used in this repo.
- LoRA/QLoRA is the lowest-risk first pass for a 30B preview model and keeps the first run within a single A100 80GB budget envelope.
- The replayable harvest currently has `170` episodes and `36` successes. The export keeps failures out of the SFT dataset instead of hiding them.

## Local-Only Steps

Run these before spending more GPU money:

```bash
python -m training.format_for_sft \
  --input data/autocinema_trajectory_harvest/episodes.jsonl \
  --summary data/autocinema_trajectory_harvest/summary.json \
  --output-dir data/autocinema_trajectory_harvest/sft

python -m training.finetune_bu --help
python -m training.runpod_job --help
python -m training.serve_model --help
python check.py
```

Verify:

- `data/autocinema_trajectory_harvest/sft/train.jsonl` is non-empty
- `data/autocinema_trajectory_harvest/sft/val.jsonl` is non-empty
- `data/autocinema_trajectory_harvest/sft/manifest.json` reports `browser-use/bu-30b-a3b-preview`
- the train/val counts look sane for the `36` successful episodes

## RunPod Bootstrap

What requires a real RunPod pod:

- downloading the base model
- adapter training
- checkpoint writes under `models/bu-30b-lora/`

Recommended pod profile:

- GPU: `NVIDIA A100 80GB PCIe`
- GPUs: `1`
- Volume: `80 GB`
- Cloud: `COMMUNITY` unless the workload needs stricter isolation
- Budget guardrail: inspect price before launch and stop if it exceeds the local `training.runpod_config.RunPodConfig` envelope

Fallbacks:

- `NVIDIA H100 PCIe` if A100 capacity is unavailable and the price is acceptable
- a smaller card only for environment/bootstrap checks, not for the first real `browser-use/bu-30b-a3b-preview` SFT run

## Access Flow

1. In RunPod, inspect the existing exited pod `59dd4g1snevkqs` as the baseline for image, storage, and wiring.
2. Start or clone a pod with `NVIDIA A100 80GB PCIe`.
3. Use the RunPod UI or `runpodctl` to obtain SSH/connect instructions.
4. SSH into the pod after it reaches `RUNNING`.

Example access flow:

```bash
runpodctl config --apiKey "$RUNPOD_API_KEY"
runpodctl get pod 59dd4g1snevkqs
# then use the SSH endpoint shown by RunPod
ssh root@<runpod-host> -p <runpod-port>
```

Inside the pod:

```bash
cd /workspace
git clone <your-fork-or-origin-url> autoppia_operator
cd autoppia_operator
git checkout daryxx
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## First SFT Run

Copy or sync the repo with the committed SFT artifacts already present under `data/autocinema_trajectory_harvest/sft/`.

Launch training:

```bash
python -m training.finetune_bu \
  --base-model browser-use/bu-30b-a3b-preview \
  --data data/autocinema_trajectory_harvest/sft/train.jsonl \
  --val-data data/autocinema_trajectory_harvest/sft/val.jsonl \
  --output-dir models/bu-30b-lora \
  --epochs 3 \
  --lr 2e-4 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --batch-size 2 \
  --grad-accum 8
```

Or use the orchestration wrapper:

```bash
python -m training.runpod_job \
  --data data/autocinema_trajectory_harvest/sft/train.jsonl \
  --val-data data/autocinema_trajectory_harvest/sft/val.jsonl \
  --output-dir models/bu-30b-lora \
  --gpu-type "NVIDIA A100 80GB PCIe"
```

## Post-Run Checks

Before leaving the pod running:

- confirm `models/bu-30b-lora/train_metrics.json` exists
- confirm `models/bu-30b-lora/adapter_config.json` points back to `browser-use/bu-30b-a3b-preview`
- archive logs and adapter weights
- stop or terminate the pod to avoid idle billing

To serve the adapter later:

```bash
python -m training.serve_model \
  --base-model browser-use/bu-30b-a3b-preview \
  --adapter-path models/bu-30b-lora
```
