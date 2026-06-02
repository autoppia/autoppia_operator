#!/usr/bin/env bash
set -euo pipefail

cd /workspace/autoppia_operator_run
mkdir -p /workspace/autoppia_operator_run/tmp_uploads
export TMPDIR=/workspace/autoppia_operator_run/tmp_uploads
export BU_SKIP_ENSURE_DEPS=1
export DEMO_WEBS_ENDPOINT=http://84.247.180.192
export PYTHONPATH=/workspace/autoppia_operator_run:${PYTHONPATH:-}

rm -rf /workspace/autoppia_operator_run/models/contact-ppo-qwen3-multiseed

nohup python training/rl/contact_ppo_trainer.py \
  --base-model Qwen/Qwen3-30B-A3B \
  --train-seeds 3,13,15,16,17,19,22,25,27,28,29,30,33,35,37,40,41,47,48,50 \
  --dev-seeds 2,9,18,20,23,26,32,34,36,46 \
  --updates 12 \
  --episodes-per-update 6 \
  --episodes-per-seed 3 \
  --max-steps 7 \
  --expert-run-glob '/workspace/autoppia_operator_run/data/autocinema/contact/gold/runs/*.json' \
  --expert-pretrain-updates 1 \
  --expert-batch-size 4 \
  --teacher-bc-weight 2.0 \
  --teacher-bc-max-step 0 \
  --lr 3e-5 \
  --clip-range 0.2 \
  --vf-coef 0.5 \
  --ent-coef 0.02 \
  --ppo-epochs 2 \
  --minibatch-size 4 \
  --max-new-tokens 64 \
  --max-prompt-tokens 2048 \
  --temperature 1.0 \
  --top-p 0.98 \
  --first-step-samples 8 \
  --first-step-temperature 1.15 \
  --first-step-top-p 0.99 \
  --gradient-checkpointing \
  --output-dir /workspace/autoppia_operator_run/models/contact-ppo-qwen3-multiseed \
  > /workspace/autoppia_operator_run/contact_ppo_qwen3_multiseed.log 2>&1 < /dev/null &

echo $! > /workspace/autoppia_operator_run/contact_ppo_qwen3_multiseed.pid
printf 'PID='
cat /workspace/autoppia_operator_run/contact_ppo_qwen3_multiseed.pid
