#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${TASK_CACHE:-$ROOT/data/task_cache/tasks_5_projects.json}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$ROOT/data/full_eval_score_cheat_${STAMP}}"
PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5.2}"
MAX_STEPS="${MAX_STEPS:-8}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-1}"
TASKS_PER_USE_CASE="${TASKS_PER_USE_CASE:-1}"

if [[ $# -gt 0 ]]; then
  PROJECTS=("$@")
else
  PROJECTS=("autobooks" "autodining" "autocinema")
fi

mkdir -p "$OUT_DIR"

echo "Full eval score-cheat batch"
echo "  cache:          $CACHE"
echo "  out_dir:        $OUT_DIR"
echo "  provider:       $PROVIDER"
echo "  model:          $MODEL"
echo "  max_steps:      $MAX_STEPS"
echo "  concurrency:    $TASK_CONCURRENCY"
echo "  tasks/use_case: $TASKS_PER_USE_CASE"
echo "  projects:       ${PROJECTS[*]}"
echo

python - <<'PY' "$CACHE" "$TASKS_PER_USE_CASE" "${PROJECTS[@]}"
import json
import sys
from collections import defaultdict

cache = sys.argv[1]
tasks_per_use_case = max(1, int(sys.argv[2]))
projects = sys.argv[3:]
with open(cache, "r", encoding="utf-8") as f:
    raw = json.load(f)
items = raw.get("tasks") if isinstance(raw, dict) else raw
by_project = defaultdict(set)
for item in items:
    pid = str(item.get("web_project_id") or "")
    uc = item.get("use_case") or {}
    name = str(uc.get("name") or "") if isinstance(uc, dict) else ""
    if pid and name:
        by_project[pid].add(name)
total = 0
for project in projects:
    uc_count = len(by_project.get(project, set()))
    tasks = uc_count * tasks_per_use_case
    total += tasks
    print(f"  {project}: {uc_count} use cases -> {tasks} task(s)")
print(f"  total planned tasks: {total}")
print("  rough time @ 90-105s/task, concurrency=1:", f"{total * 90 / 3600:.2f}-{total * 105 / 3600:.2f}h")
PY

echo
printf "project\tnum_tasks\tsuccess_rate\tavg_score\tavg_steps\tavg_task_seconds\testimated_cost_usd\tresult_file\n" > "$OUT_DIR/summary.tsv"

for project in "${PROJECTS[@]}"; do
  out_file="$OUT_DIR/${project}.json"
  echo "=== Running $project ==="
  (
    cd "$ROOT"
    python eval.py \
      --provider "$PROVIDER" \
      --model "$MODEL" \
      --task-cache "$CACHE" \
      --web-project-id "$project" \
      --all-use-cases \
      --tasks-per-use-case "$TASKS_PER_USE_CASE" \
      --task-concurrency "$TASK_CONCURRENCY" \
      --max-steps "$MAX_STEPS" \
      --enable-score-cheating \
      --out "$out_file"
  )
  python - <<'PY' "$project" "$out_file" "$OUT_DIR/summary.tsv"
import json
import sys
project, out_file, summary_tsv = sys.argv[1:4]
with open(out_file, "r", encoding="utf-8") as f:
    data = json.load(f)
timing = data.get("timing") or {}
row = [
    project,
    str(int(data.get("num_tasks") or 0)),
    str(data.get("success_rate") or 0.0),
    str(data.get("avg_score") or 0.0),
    str(data.get("avg_steps") or 0.0),
    str(timing.get("avg_task_seconds") or 0.0),
    str(sum(float(ep.get("estimated_cost_usd") or 0.0) for ep in (data.get("episodes") or []))),
    out_file,
]
with open(summary_tsv, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
print(
    f"[summary] {project}: tasks={row[1]} success_rate={row[2]} avg_score={row[3]} "
    f"avg_steps={row[4]} avg_task_seconds={row[5]} cost_usd={row[6]}"
)
PY
  echo
done

echo "Batch complete"
echo "  summary: $OUT_DIR/summary.tsv"
