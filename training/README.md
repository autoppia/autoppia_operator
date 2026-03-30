# Training Package

`training/` is the reusable library behind the CLI helpers in `scripts/training/`.

What belongs here:
- trajectory models and normalization
- IWAP and S3 ingestion
- SFT and PPO bootstrap export
- PPO-style rollout collection helpers

What does not belong here:
- subnet runtime logic
- `/act` serving code
- one-off shell orchestration

Current structure is coherent:
- `pipeline.py`: ingestion and dataset bundle orchestration
- `normalize.py`, `models.py`, `dataset.py`: normalization primitives and typed records
- `exporters.py`: SFT/PPO export helpers
- `iwap_client.py`, `s3_source.py`: external data access
- `ppo_loop.py`: online rollout collection using the operator as policy
