# Scripts

The root of `scripts/` is reserved for stable operator entrypoints:

- `scripts/sn36_ops.py`: subnet36 operator lifecycle CLI.
- `scripts/deploy_check.py`: `/act` and handshake contract checks.

Everything else is grouped by purpose:

- `scripts/eval/`: task generation and eval comparison helpers.
- `scripts/training/`: dataset preparation, export, and fine-tuning helpers.
- `scripts/sn36/`: manual shell helpers for submission and metagraph inspection.

Obsolete one-off scripts should be removed instead of restored to the root.
