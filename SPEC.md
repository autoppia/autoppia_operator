# SPEC

## Mission

Turn `autoppia_operator` into a fine-tuning-ready Browser Use training repo on top of the replayable Autocinema harvest that already exists on branch `daryxx`.

This project is not about doing full RL or blindly starting GPU jobs. It is about making the next step operationally clean:

- the current replayable Autocinema harvest can be exported into a real SFT split
- the chosen base model and training method are explicit and consistent
- the RunPod defaults are realistic for this model family
- the repo contains a concrete, reproducible RunPod + SSH execution plan that a human can run immediately

## External Facts To Use

These facts were established before this run and should guide the implementation:

- the current replayable harvest lives under `data/autocinema_trajectory_harvest/`
- `summary.json` currently reports:
  - `project_id = autocinema`
  - `episodes_total = 170`
  - `successes_total = 36`
  - `failures_total = 134`
  - `replayable_episodes_total = 170`
- the current `training.format_for_sft` path is broken against this harvest because the episode structure does not match what the formatter expects
- `training.export` currently fails to import because it references `training.schema`
- the recommended starting model is `browser-use/bu-30b-a3b-preview`
- the preferred first training method is LoRA/QLoRA SFT, not full fine-tuning
- the preferred first GPU target is `NVIDIA A100 80GB PCIe`
- an existing RunPod pod named `bu-30b-a3b-bootstrap` exists with pod id `59dd4g1snevkqs`
- the current RunPod account balance is healthy, around `$699`

## Hard Constraints

- Work only in `/home/usuario1/daryxx/autoppia/operator/autoppia_operator`
- Keep the repo on branch `daryxx`
- Do not weaken or discard the existing replayable harvest just to make exports easier
- Do not claim a training pipeline is ready if the SFT export still crashes on the real harvest
- Do not switch to a different base model without strong evidence
- Do not pretend the repo has already completed fine-tuning; the target is readiness, not a fake finished model
- Keep the plan grounded in actual files and commands that match the repo

## What The Repo Must Contain After This Run

### 1. A working SFT export bridge from the replayable Autocinema harvest

The repo must be able to convert the current replayable dataset into SFT-ready JSONL files for Browser Use fine-tuning.

Required output artifacts under the target repo:

- `data/autocinema_trajectory_harvest/sft/train.jsonl`
- `data/autocinema_trajectory_harvest/sft/val.jsonl`
- `data/autocinema_trajectory_harvest/sft/manifest.json`

The manifest must include at least:

- source dataset paths used
- total episodes seen
- successful episodes used for SFT
- example counts for train and val
- system prompt or format version
- base model target

The export path should use the replayable harvest as input and should not depend on ad hoc manual editing.

### 2. The Browser Use base model decision must be explicit and consistent

The repo should consistently treat `browser-use/bu-30b-a3b-preview` as the starting point for Autocinema browser-agent fine-tuning unless a concrete repo-level blocker is discovered.

If multiple files set base model defaults, align them. If there is a better config abstraction, use it.

### 3. RunPod configuration must be realistic for this model

The repo must not default to an underpowered cheap GPU profile for this job.

The default RunPod recommendation should reflect the chosen model and first-pass training method:

- preferred GPU: `NVIDIA A100 80GB PCIe`
- method: LoRA/QLoRA SFT
- enough disk / budget guardrails for a real bootstrap job

It is acceptable to keep cheaper alternatives documented, but the default path should match the recommended model.

### 4. A concrete RunPod + SSH runbook and machine-readable plan must exist

Add both:

- a human-readable runbook doc
- a machine-readable plan/manifest

Required artifacts:

- `docs/browser_use_finetune_runpod.md`
- `training/runpod_bootstrap_plan.json`

They must cover at least:

- why `browser-use/bu-30b-a3b-preview` is the chosen starting point
- why LoRA/QLoRA is the recommended first training method
- preferred GPU and fallback options
- the existing bootstrap pod id `59dd4g1snevkqs`
- the expected dataset paths for SFT
- the exact local commands to export data and start training
- the expected SSH or RunPod access flow a human should use
- what to verify before spending more GPU money

Be concrete. The runbook should help a human bring up the A100 pod and launch the first SFT run.

### 5. The training package must pass the key local checks

At minimum, the following must work on the checked-out repo without a GPU:

- `training.export` imports successfully
- the real replayable Autocinema harvest can be converted into non-empty SFT train/val JSONL
- the fine-tuning and RunPod CLIs still expose a useful `--help`

## Recommended Workflow

1. Start from the real replayable harvest under `data/autocinema_trajectory_harvest/`
2. Fix the training bridge so the current dataset can be exported cleanly
3. Add or update tests around the broken points before or while fixing them
4. Align model defaults and RunPod defaults with the Browser Use plan
5. Generate the committed SFT artifacts and manifest from the real dataset
6. Write the runbook and machine-readable bootstrap plan

## Definition Of Done

This project is done only when all `.arbos/tests/` pass.

That means all of the following are true:

- `python check.py` still passes
- the repo stays on `daryxx`
- `training.export` imports
- the replayable Autocinema harvest exports into non-empty SFT train/val files
- `data/autocinema_trajectory_harvest/sft/manifest.json` exists and describes the export
- `docs/browser_use_finetune_runpod.md` exists and is concrete
- `training/runpod_bootstrap_plan.json` exists and is concrete
- the default model choice is `browser-use/bu-30b-a3b-preview`
- the default RunPod recommendation is suitable for that model and points first to `NVIDIA A100 80GB PCIe`
