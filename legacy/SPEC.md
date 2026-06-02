# SPEC.md

## Objective

Build a strong, generalist web agent for the IWA benchmark demo-web environment.

In this repo, the immediate benchmark focus is `autocinema`, but the target is broader:
- one agent
- one training method
- one clean inference method
- strong performance across demo webs and use cases

The agent should succeed in the benchmark because it has learned the environment and task distribution well, not because inference is hardcoded per use case.

## Current Goal

Current working goal:
- train and serve an agent that performs very well in the IWA demo-web benchmark
- keep inference clean
- keep training as generic as possible
- use the real benchmark environment for eval whenever possible

## Environment

Use the real demo-web server by default:
- `DEMO_WEBS_ENDPOINT=http://84.247.180.192`
- typical Autocinema frontend: `http://84.247.180.192:8000`

Canonical evaluator used here:
- `autoppia_iwa.src.evaluation.stateful_evaluator.TaskExecutionSession`
- local wrapper in this repo: `src/operator/eval/session.py`

Related repos available in this workspace:
- `/home/usuario1/daryxx/autoppia/operator/autoppia_iwa`
- `/home/usuario1/daryxx/autoppia/operator/autoppia_webs_demo`

These contain:
- task definitions
- evaluator logic
- demo-web implementations
- events and score rules

## Clean System Shape

### Inference

Inference should be as clean as possible.

Target properties:
- no use-case-specific heuristics if avoidable
- no hidden task overrides if avoidable
- one clear runtime path for serving the trained model

Current clean inference path:
- `src/operator/agents/operator.py`
- selected via `WEB_AGENT_RUNTIME=structured`

### Training

Training should be as generic as possible.

Target properties:
- reusable across demo webs
- reusable across use cases
- based on real trajectories / scores / prompts rather than fragile hand-authored logic
- suitable for both SFT-style and RL-style improvement

Current wrappers:
- `scripts/training/run_autocinema_training.py`
- `scripts/eval/run_autocinema_campaign.py`

## RL Research Track

Main current research question:
- is RL using benchmark score as reward, optionally densified with an LLM judge, sufficient to train a strong generalist benchmark agent?

Current RL components already present:
- environment / rollout path in `training/rl/`
- PPO-style trainer
- score-based reward
- reward shaping utilities
- optional LLM judge in `training/rl/judge.py`

Current judge default:
- `gpt-4o`
- controlled by `CONTACT_RL_JUDGE_MODEL`

What we are trying to learn:
- whether evaluator score + optional `gpt-4o` judge give enough signal
- whether the resulting policy can become strong without inference-time heuristics
- whether the same recipe can scale from one use case to many

## GPU / Infra

This repo has an available GPU path through RunPod.

Relevant facts:
- RunPod tooling is already in the repo
- A100 is available in the stack
- useful entrypoints include:
  - `training/runpod_job.py`
  - `training/finetune_bu.py`
  - `training/runpod_config.py`

## Success Criteria

A good end state looks like this:
- clean inference path
- generic training path
- easy eval command
- high score on real IWA demo-web evaluation
- strong performance across multiple use cases, not just one
- minimal or no inference-time task-specific heuristics

## Non-Goals

Not a valid end state:
- passing a task only because of brittle inference hacks
- a method that works only for one demo web or one use case
- claiming success without real eval against the benchmark environment
