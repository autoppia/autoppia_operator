#!/usr/bin/env bash
set -euo pipefail

repo="${ARBOS_TARGET_REPO:?ARBOS_TARGET_REPO is required}"
cd "$repo"

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "daryxx" ]]; then
  echo "Expected branch daryxx, got: $branch" >&2
  exit 1
fi

python check.py
