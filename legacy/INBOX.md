# Manager Update — Cycle 26 (2026-03-15)

## Phase: eval_improved — CRITICAL: Write eval_cycle26_actual.json

### Current State
- Cycle 26, eval_improved phase
- `data/eval_cycle26_actual.json` is **MISSING** — this blocks CHECK
- Best score so far: 7/10 (0.70) — ADD_COMMENT, CONTACT, FILM_DETAIL, FILTER_FILM, LOGIN, LOGOUT, SEARCH_FILM
- Target for cycle 26: DELETE_FILM or EDIT_FILM passing = 8/10 = NEW BEST

### Step 1: Debug DELETE_FILM and EDIT_FILM

```bash
cd /home/usuario1/daryxx/autoppia/operator/autoppia_operator
for UC in DELETE_FILM EDIT_FILM; do
    echo "=== $UC ===" && python eval.py --use-case "$UC" --num-tasks 1 --max-steps 20 2>&1 | tail -50
done
```

Key suspect: `_guard_delete_task_against_unrelated_form_edits` in engine.py:3256
- For DELETE: must navigate to film detail page, find delete button, click, confirm
- For EDIT: must navigate to /films/<id>/edit, fill changed fields, submit
- Both require admin auth — check _guard_auth_task_initial_navigation fires first

### Step 2: Run full eval and write eval_cycle26_actual.json

```bash
cd /home/usuario1/daryxx/autoppia/operator/autoppia_operator
python eval.py --num-tasks 10 --distinct-use-cases --max-steps 15 2>&1 | tee /tmp/eval_c26.txt
```

Then write the file (REQUIRED for CHECK to pass):
```python
import json, re, datetime
text = open('/tmp/eval_c26.txt').read()
CYCLE = 26
m = re.search(r'([0-9]+)/([0-9]+)\s+(?:passed|succeeded|success)', text, re.I)
if m:
    score = int(m.group(1)) / int(m.group(2))
else:
    m2 = re.search(r'success[_ ]rate[:\s]+([0-9.]+)', text, re.I)
    score = float(m2.group(1)) if m2 else 0.0
p = json.load(open('meta/phase.json'))
result = {'cycle': CYCLE, 'score': score, 'measured': True, 'tasks_run': 10,
    'successes': round(score * 10),
    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'note': f'Cycle {CYCLE} measured score={score:.2f}'}
json.dump(result, open(f'data/eval_cycle{CYCLE}_actual.json', 'w'), indent=2)
if score > p.get('best_eval_score', 0):
    p['best_eval_score'] = score
json.dump(p, open('meta/phase.json', 'w'), indent=2)
print(f'Score: {score:.4f} — written to data/eval_cycle{CYCLE}_actual.json')
```

The CHECK will pass once eval_cycle26_actual.json exists with measured=True.
