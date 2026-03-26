# Agent Prompt

You are working on `autoppia_operator` as a Browser Use fine-tuning readiness project.

Your job is to bridge the gap between the current replayable Autocinema harvest and a real first SFT run on RunPod.

Priority order:

1. make the current replayable harvest export cleanly into SFT train/val data
2. fix broken training-module contracts that block that path
3. align base-model and RunPod defaults with the Browser Use plan
4. leave concrete RunPod + SSH runbooks and machine-readable bootstrap instructions

Rules:

- Stay on branch `daryxx`
- Use `browser-use/bu-30b-a3b-preview` as the starting point unless the repo itself forces a different choice
- Prefer LoRA/QLoRA SFT over full fine-tuning
- Do not drift into speculative RL work before the SFT bridge is healthy
- Use the real replayable harvest under `data/autocinema_trajectory_harvest/`, not mock examples
- Do not hide failures in the dataset; use successful episodes for SFT export and record the selection clearly
- Be explicit about what is local-only and what requires a real RunPod pod
- Treat the existing RunPod A100 bootstrap pod id `59dd4g1snevkqs` as a useful anchor in docs and plans
