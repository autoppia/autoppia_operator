# Autocinema Training And Inference

## Clean Inference

Use the clean structured inference path:
- operator: `src/operator/agents/operator.py`
- enable: `WEB_AGENT_RUNTIME=structured`
- target demo endpoint: `DEMO_WEBS_ENDPOINT=http://84.247.180.192`

Single-seed smoke:
```bash
WEB_AGENT_RUNTIME=structured DEMO_WEBS_ENDPOINT=http://84.247.180.192 python scripts/eval/eval_structured_operator.py   --project-id autocinema   --use-case CONTACT   --num-tasks 1   --repeat 1   --seed 1   --max-steps 10   --task-concurrency 1   --out tmp/contact_structured_smoke.json
```

Multi-seed clean eval:
```bash
python scripts/eval/eval_clean_operator_suite.py   --project-id autocinema   --use-case CONTACT   --seeds 1-10   --endpoint http://84.247.180.192   --max-steps 10   --out tmp/contact_clean_suite_10.json
```

## Generalist Training

Preparation wrapper:
- `scripts/training/run_autocinema_training.py`

Campaign builder:
- `scripts/eval/run_autocinema_campaign.py`

Unified cycle wrapper:
- `scripts/training/run_generalist_cycle.py`

Prepare a merged multi-use-case SFT campaign:
```bash
python scripts/training/run_autocinema_training.py   --run-name autocinema_all_campaign   --use-cases all   --existing-pod-id <pod-id>   --epochs 2   --lora-rank 32   --all-use-cases-eval-tasks 3
```

Run the generic cycle and immediately benchmark the clean operator on one use case:
```bash
python scripts/training/run_generalist_cycle.py   --run-name autocinema_all_campaign   --use-cases all   --existing-pod-id <pod-id>   --eval-clean-use-case CONTACT   --eval-clean-seeds 1-10   --endpoint http://84.247.180.192
```

This flow is intentionally simple:
1. Consolidate gold traces by use case.
2. Export per-use-case SFT datasets.
3. Merge them into one multi-use-case adapter campaign.
4. Keep inference fixed and clean.
5. Evaluate the same clean operator on real demo webs.
