# Notes For This Run

- Target repo: `/home/usuario1/daryxx/autoppia/operator/autoppia_operator`
- Branch must remain `daryxx`
- Existing replayable harvest:
  - `data/autocinema_trajectory_harvest/summary.json`
  - `data/autocinema_trajectory_harvest/episodes.jsonl`
- Known current gaps:
  - `training.format_for_sft` crashes on the real harvest because episode structure mismatches
  - `training.export` import fails due to missing `training.schema`
  - `training.runpod_config.RunPodConfig` still defaults to an underpowered cheap GPU
- Chosen starting point:
  - model: `browser-use/bu-30b-a3b-preview`
  - method: LoRA/QLoRA SFT
  - preferred GPU: `NVIDIA A100 80GB PCIe`
- Existing RunPod context from MCP:
  - healthy balance around `$699`
  - existing exited pod: `bu-30b-a3b-bootstrap`
  - existing pod id: `59dd4g1snevkqs`
- The goal is readiness and reproducibility, not claiming training is already finished.
