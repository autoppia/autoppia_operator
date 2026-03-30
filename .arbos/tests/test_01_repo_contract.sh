#!/usr/bin/env bash
set -euo pipefail

repo="${ARBOS_TARGET_REPO:?ARBOS_TARGET_REPO is required}"
cd "$repo"

python check.py
