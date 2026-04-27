# Harvester Pipeline — Branch Comparison
**Branches:** `gul/harvester` vs `harvester/prototype`  
**Common ancestor:** `669eeaf`  
**Date:** 2026-04-24

---

## 1. Executive Summary

1. **Scope de proyectos**: `gul/harvester` cubre 12 proyectos IWA (autocinema + 11 más). `harvester/prototype` cubre 2 (autocinema + autohealth).
2. **Origen de los planes determinísticos**: `gul` extrae los pasos directamente del registry de trayectorias de IWA (`autoppia_iwa.trajectory_registry`). `prototype` tiene los pasos escritos a mano en cada builder.
3. **Selectors**: `gul` tiene un pipeline de enriquecimiento semántico (`trajectory_selectors.py`) que convierte los XPath crudos de IWA en candidatos id/class/text/placeholder. `prototype` usa selectores manuales definidos en `selectors.py`.
4. **Routing de use cases**: `prototype` tiene un sistema de intents (`use_case_intents.py` + `canonical_intent()`) que permite resolver use cases desconocidos mapeando a intents canónicos. `gul` no tiene ese fallback.
5. **Metadata de use cases**: `prototype` tiene `use_case_registry.py` con hints de harvesting, clusters de fallo y targets de gold por use case. `gul` no tiene ese nivel de metadata.
6. **SFT export**: `gul` tiene un módulo dedicado `format_for_sft.py` con versiones v2 y v3 del formato. `prototype` delega a la función que ya estaba en `harvester.py`.
7. **Tests**: `gul` tiene tests por proyecto (un archivo por IWA project). `prototype` tiene tests consolidados más el nuevo `test_deterministic_task_normalizer.py`.
8. **Documentación**: `gul` tiene `training/README.md` con el flujo completo documentado. `prototype` no tiene README en training.
9. **Todos los IWA builders en `gul` son thin wrappers**: delegan a `iwa_enriched_planner.py` que lee de `trajectory_registry`. En `prototype` cada builder es un módulo manual completo.
10. **El normalizer de `gul` es más grande**: tiene `route_for_web_project_use_case` con route maps para cada uno de los 12 proyectos. El de `prototype` usa `canonical_intent` para resolver use cases no conocidos.

---

## 2. Componentes por branch

### `gul/harvester` — Estructura de archivos clave

```
training/
├── harvester.py                          # orquestador principal (igual al ancestro)
├── format_for_sft.py                     # exportador SFT v2/v3 (solo en gul)
├── _iwa_path.py                          # bootstrap sys.path para imports de IWA
├── autocinema_selector_registry.py       # registry de selectors para autocinema
├── README.md                             # documentación del flujo completo
└── deterministic_harvester/
    ├── __init__.py                       # expone todo incluido trajectory_selectors
    ├── normalizer.py                     # con route_for_web_project_use_case
    ├── selectors.py                      # selectors base (no cambia estructura)
    ├── trajectory_selectors.py           # convierte XPath IWA → candidatos semánticos
    ├── iwa_planned_actions.py            # convierte IWA BaseAction → planned_action dict
    ├── use_case_selectors.py             # verifica cobertura semántica por use case
    ├── builders/
    │   ├── registry.py                   # cubre los 12 proyectos IWA
    │   ├── iwa_enriched_planner.py       # factory genérico para todos los IWA builders
    │   ├── autocinema.py                 # manual (igual que ancestro)
    │   ├── autobooks.py                  # thin wrapper → iwa_enriched_planner
    │   ├── autozone.py                   # thin wrapper
    │   ├── autodining.py                 # thin wrapper
    │   ├── autocrm.py                    # thin wrapper
    │   ├── automail.py                   # thin wrapper
    │   ├── autolodge.py                  # thin wrapper
    │   ├── autodelivery.py               # thin wrapper
    │   ├── autowork.py                   # thin wrapper
    │   ├── autoconnect.py                # thin wrapper
    │   ├── autocalendar.py               # thin wrapper
    │   └── autolist.py                   # thin wrapper
tests/
    ├── test_autobooks_iwa_planners.py
    ├── test_autodining_iwa_planners.py
    ├── test_autozone_iwa_planners.py
    ├── ...                               # un archivo por proyecto IWA
    ├── test_iwa_planned_actions.py
    └── test_semantic_selector_coverage.py
```

### `harvester/prototype` — Estructura de archivos clave

```
training/
├── harvester.py                          # orquestador (igual estructura)
├── use_case_registry.py                  # metadata rica por use case (NUEVO)
├── use_case_intents.py                   # sistema de intents canónicos (NUEVO)
└── deterministic_harvester/
    ├── __init__.py                       # expone autohealth builders
    ├── normalizer.py                     # con _canonical_use_case + canonical_intent
    ├── selectors.py                      # con route_for_use_case multi-project
    ├── builders/
    │   ├── registry.py                   # cubre autocinema + autohealth; fallback por intent
    │   ├── autocinema.py                 # manual
    │   └── autohealth.py                 # NUEVO — manual, pasos a mano por use case
tests/
    ├── test_deterministic_planners.py    # expandido
    ├── test_deterministic_task_normalizer.py  # NUEVO
    ├── test_harvester.py                 # expandido
    └── test_use_case_registry.py         # NUEVO
```

---

## 3. Flujo paso a paso de cada pipeline

### 3.1 `gul/harvester` — Flujo determinístico (IWA-backed)

```
CLI
 └─ focus_use_case.py teacher-harvest
     --project-id autodining
     --use-case SEARCH_RESTAURANT
     --task-cache data/task_cache/autodining_tasks.json
     --deterministic-only
         │
         ▼
cmd_claude_harvest()                         [focus_use_case.py:300]
  → HarvestConfig(deterministic_only=True, web_project_id="autodining")
  → seeds = _resolve_seed_list(...)          [lee seeds del task cache]
         │
         ▼
collect_rows_for_seeds(strategy="code-aware")       [harvester.py]
  → _collect_rows_for_seeds_code_aware()
    → por cada seed: _collect_seed_rows_deterministic_first()
         │
         ▼
_collect_seed_rows_deterministic_first(seed=N)      [harvester.py]
  → load_task_objective(cache_path, use_case, seed)
         │
         ▼
load_task_objective()                        [normalizer.py:1208]
  → load_task_row()  → lee el JSON del cache para ese seed
  → normalize_task_row()
      → _constraint_hints_from_task()       [parsea criteria del task]
      → _resolve_auth_field_values()        [extrae username/password]
      → _pick_value()                       [resuelve valores de campos]
      → route_for_web_project_use_case()    [determina route: /search, /profile, etc.]
  → DeterministicTaskObjective(
        web_project_id="autodining",
        use_case="SEARCH_RESTAURANT",
        seed=N,
        field_values={...},
        entity_filters={...},
        ...
    )
         │
         ▼
build_deterministic_plan(objective)          [planners.py → builders/registry.py]
  → DETERMINISTIC_PLAN_BUILDERS[("autodining", "SEARCH_RESTAURANT")]
  → build_autodining_plan(objective)         [builders/autodining.py]
       → build_iwa_enriched_deterministic_plan("autodining", objective)
             [builders/iwa_enriched_planner.py]
  → planned_actions_for_iwa_use_case_enriched("autodining", "SEARCH_RESTAURANT")
             [trajectory_selectors.py]
       → get_trajectory_map("autodining")["SEARCH_RESTAURANT"]   ← IWA trajectory registry
       → iwa_actions_to_planned_actions(actions, frontend_url)   [iwa_planned_actions.py]
       → por cada action: enrich selectors con _semantic_candidates_from_xpath()
           → XPath "@id='search-btn'" → {type: "cssSelector", value: "#search-btn"}
           → XPath "contains(@class,'btn')" → {type: "className", value: "btn"}
  → DeterministicPlan(
        actions=[navigate, wait, type_text, click, wait, ...],
        source="iwa_p04_autodining"
    )
         │
         ▼
_execute_candidate_attempt(config, bundle)   [harvester.py]
  → run_eval_attempt()                       [focus_pipeline.py]
      → operador ejecuta los pasos en el browser
      → evaluador devuelve score
  → write_candidate() / write_replay_report()
  → si score == 1.0 → es gold
         │
         ▼
[si deterministic_only=True y no es gold → para, no llama a Claude]
         │
         ▼
output:
  data/autodining/SEARCH_RESTAURANT/
  ├── gold/runs/seed_NNNN_deterministic.json
  ├── harvester/claude_runs/seed_NNNN/
  └── task_cache/autodining_seed_NNNN.json
```

### 3.2 `harvester/prototype` — Flujo determinístico (hand-coded + intent fallback)

```
CLI
 └─ focus_use_case.py teacher-harvest
     --project-id autohealth
     --use-case BOOK_APPOINTMENT
     --task-cache data/task_cache/autohealth_tasks.json
     --deterministic-only
         │
         ▼
cmd_claude_harvest()                         [focus_use_case.py:307]
  → HarvestConfig(deterministic_only=True, web_project_id="autohealth")
         │
         ▼
collect_rows_for_seeds(strategy="code-aware")       [harvester.py]
  → _collect_seed_rows_deterministic_first(seed=N)
         │
         ▼
load_task_objective()                        [normalizer.py:849]
  → load_task_row()
  → normalize_task_row()
      → _canonical_use_case(use_case)        ← DIFERENCIA vs gul
           → canonical_intent("BOOK_APPOINTMENT")
           → mapea a intent "SCHEDULE" o similar
      → [misma lógica de constraints/auth/fields que gul]
  → DeterministicTaskObjective(
        web_project_id="autohealth",
        use_case="BOOK_APPOINTMENT",
        ...
    )
         │
         ▼
build_deterministic_plan(objective)          [builders/registry.py]
  → busca DETERMINISTIC_PLAN_BUILDERS[("autohealth", "BOOK_APPOINTMENT")]
  → build_autohealth_plan(objective)         [builders/autohealth.py]
       → pasos escritos a mano:
           navigate("/appointments", objective)
           wait(0.5)
           click(selector=...)
           type_text("doctor_name", ...)
           ...
  → si no encuentra el use case:
       → canonical_intent("BOOK_APPOINTMENT") → "SCHEDULE"
       → _INTENT_TO_AUTOCINEMA_USE_CASE["SCHEDULE"] → "ADD_FILM" (si existe)
       → usa builder de autocinema como fallback   ← DIFERENCIA vs gul
         │
         ▼
[misma ejecución/evaluación que gul]
         │
         ▼
output: data/autohealth/BOOK_APPOINTMENT/gold/...
```

---

## 4. Diferencias técnicas clave

### 4.1 Builder: IWA-backed vs hand-coded

| Aspecto | `gul/harvester` | `harvester/prototype` |
|---|---|---|
| Origen del plan | `autoppia_iwa.trajectory_registry` | Código Python a mano |
| Para agregar proyecto nuevo | Thin wrapper de 30 líneas | Builder completo por use case |
| Mantenimiento | Actualizar IWA trajectories | Actualizar cada builder |
| Falla si IWA trajectories cambia | Sí, directamente | No afecta (son independientes) |
| Riesgo de divergencia IWA-operator | Bajo (sincronizados) | Alto (duplicación) |

**gul — autodining.py (30 líneas):**
```python
_PROJECT = "autodining"
AUTODINING_PLAN_BUILDERS = iwa_enriched_action_builders(_PROJECT)

def build_autodining_plan(objective):
    return build_iwa_enriched_deterministic_plan(_PROJECT, objective, ...)
```

**prototype — autohealth.py (400+ líneas):**
```python
def _build_book_appointment(objective):
    return [
        navigate("/appointments", objective),
        wait(0.5),
        click(xpath_selector("//button[contains(text(),'Book')]")),
        ...  # cada paso de UI escrito a mano
    ]
```

### 4.2 Selector enrichment

**gul** — `trajectory_selectors.py` convierte XPath IWA → múltiples candidatos semánticos:
```python
# Input: XPath de IWA
"@id='search-input'"

# Output: lista de candidatos semánticos
[
    {"type": "cssSelector", "value": "#search-input"},
    {"type": "id", "value": "search-input"},
    # + variantes del variants.json del proyecto
]
```

**prototype** — selectors escritos directamente en `selectors.py`:
```python
def login_username_selectors(project_id="autocinema", seed=None):
    return selector_candidates_for_ids(["username-input", "user-input"], ...)
```

### 4.3 Routing de use cases

**gul — `normalizer.py`**: un dict por proyecto
```python
_ROUTE_BY_USE_CASE_AUTOBOOKS = {"ADD_BOOK": "/profile", ...}
_ROUTE_BY_USE_CASE_AUTODINING = {"SEARCH_RESTAURANT": "/restaurants", ...}

def route_for_web_project_use_case(*, web_project_id, use_case):
    if web_project_id == "autobooks": return _ROUTE_BY_USE_CASE_AUTOBOOKS.get(use_case, "/")
    if web_project_id == "autodining": return _ROUTE_BY_USE_CASE_AUTODINING.get(use_case, "/")
    ...
```

**prototype — `builders/registry.py`**: fallback por intent
```python
def build_registered_plan(objective):
    builder = DETERMINISTIC_PLAN_BUILDERS.get((project_id, use_case))
    if builder is None:
        intent = canonical_intent(use_case)           # "SEARCH_PRESCRIPTION" → "SEARCH"
        mapped = _INTENT_TO_AUTOCINEMA_USE_CASE[intent]  # "SEARCH" → "SEARCH_FILM"
        builder = DETERMINISTIC_PLAN_BUILDERS.get(("autocinema", mapped))
    return builder(objective)
```

### 4.4 Use Case Registry (solo en prototype)

`training/use_case_registry.py` — metadata operacional por use case:
```python
UseCaseSpec(
    name="BOOK_APPOINTMENT",
    project_id="autohealth",
    harvester_hints=(
        "Open the appointments page first.",
        "Fill doctor name and date fields before submitting.",
    ),
    likely_failure_clusters=("page_not_reached", "form_variant"),
    recommended_gold_target=500,
    recommended_holdout_target=50,
)
```

Esta metadata la consume el `claude_guided_harvester.py` para dar hints al modelo durante la colección.

---

## 5. Comandos de cada pipeline

### Pipeline `gul/harvester`

```bash
# 1. Harvest determinístico de un proyecto IWA
python scripts/eval/focus_use_case.py teacher-harvest \
  --project-id autodining \
  --use-case SEARCH_RESTAURANT \
  --task-cache data/task_cache/autodining_tasks.json \
  --deterministic-only \
  --execution-mode operator

# 2. Harvest determinístico de autocinema
python scripts/eval/focus_use_case.py teacher-harvest \
  --project-id autocinema \
  --use-case LOGIN \
  --task-cache data/task_cache/autocinema_50_tasks.json \
  --deterministic-only

# 3. Consolidar gold
python scripts/eval/focus_use_case.py consolidate-gold \
  --project-id autodining --use-case SEARCH_RESTAURANT

# 4. Export SFT
python scripts/eval/focus_use_case.py export-sft \
  --project-id autodining --use-case SEARCH_RESTAURANT

# 5. Validar dataset
python scripts/eval/focus_use_case.py validate-dataset \
  --project-id autodining --use-case SEARCH_RESTAURANT

# 6. Ver cobertura semántica de selectors (solo en gul)
python -c "
from training.deterministic_harvester.use_case_selectors import has_semantic_selector_coverage
print(has_semantic_selector_coverage('autodining', 'SEARCH_RESTAURANT'))
"

# 7. Ver use cases disponibles para un proyecto IWA
python -c "
from training.deterministic_harvester.trajectory_selectors import list_iwa_use_cases
print(list_iwa_use_cases('autodining'))
"
```

### Pipeline `harvester/prototype`

```bash
# 1. Harvest determinístico de autohealth (nuevo proyecto)
python scripts/eval/focus_use_case.py teacher-harvest \
  --project-id autohealth \
  --use-case BOOK_APPOINTMENT \
  --task-cache data/task_cache/autohealth_tasks.json \
  --deterministic-only \
  --execution-mode operator

# 2. Harvest con fallback por intent
python scripts/eval/focus_use_case.py teacher-harvest \
  --project-id autohealth \
  --use-case LOGIN \          # LOGIN existe en autocinema, se resuelve por intent
  --task-cache data/task_cache/autohealth_tasks.json \
  --deterministic-only

# 3. Ver spec de un use case
python -c "
from training.use_case_registry import get_use_case_spec
spec = get_use_case_spec('BOOK_APPOINTMENT', project_id='autohealth')
print(spec.harvester_hints)
print(spec.likely_failure_clusters)
"

# 4. Consolidar, export, validate — igual que gul
python scripts/eval/focus_use_case.py consolidate-gold \
  --project-id autohealth --use-case BOOK_APPOINTMENT

# 5. Ver intent canónico de un use case
python -c "
from training.use_case_intents import canonical_intent
print(canonical_intent('BOOK_APPOINTMENT'))   # → 'SCHEDULE' o similar
print(canonical_intent('SEARCH_PRESCRIPTION')) # → 'SEARCH'
"
```

### Artifacts esperados (ambos pipelines)

```
data/<project_id>/<use_case>/
├── gold/
│   ├── runs/
│   │   └── seed_NNNN_deterministic.json     # reporte del run
│   └── episodes.jsonl                        # gold consolidado (paso consolidate-gold)
├── sft/
│   ├── train.jsonl                           # export SFT train split
│   └── val.jsonl                             # export SFT val split
├── harvester/
│   └── claude_runs/seed_NNNN/
│       ├── brief.json
│       └── attempt_01_brief.json
└── task_cache/
    └── <use_case>_seed_NNNN.json             # task cache por seed
```

---

## 6. Dónde puede fallar cada pipeline (debug trace)

### `gul/harvester` — Puntos de fallo

| Paso | Archivo | Qué puede fallar |
|---|---|---|
| `load_task_objective` | `normalizer.py:1208` | Task cache no tiene ese seed/use case |
| `route_for_web_project_use_case` | `normalizer.py:403` | Proyecto no tiene route map → devuelve `/` siempre |
| `get_trajectory_map("autodining")` | IWA library | IWA no instalado o proyecto no registrado |
| `iwa_actions_to_planned_actions` | `iwa_planned_actions.py` | BaseAction con selector desconocido |
| `_semantic_candidates_from_xpath` | `trajectory_selectors.py` | XPath no parseable → selector vacío |
| `run_eval_attempt` | `focus_pipeline.py` | Timeout de browser / evaluador caído |
| `deterministic_only` check | `focus_use_case.py:341` | RuntimeError si rows no son deterministic |

### `harvester/prototype` — Puntos de fallo

| Paso | Archivo | Qué puede fallar |
|---|---|---|
| `_canonical_use_case` | `normalizer.py:742` | Intent no mapeado → use case pasa raw |
| `build_registered_plan` | `builders/registry.py` | Use case no en builders ni en intent map → ValueError |
| Pasos del autohealth builder | `builders/autohealth.py` | Selector hardcoded no existe en la UI real |
| `get_use_case_spec` | `use_case_registry.py` | Use case no registrado → UseCaseSpec vacío |
| Intent fallback → autocinema builder | `builders/registry.py` | Autocinema builder ejecuta pasos incorrectos para autohealth |

---

## 7. Recomendación práctica

### Qué conservar de cada branch

**De `gul/harvester`:**
- `trajectory_selectors.py` + `iwa_planned_actions.py` — la arquitectura de enriquecimiento semántico es sólida y escalable
- `iwa_enriched_planner.py` — el patrón de factory genérico para todos los proyectos IWA evita duplicación masiva
- `format_for_sft.py` — exportador SFT v2/v3 dedicado
- `README.md` — documentación
- El patrón de un test file por proyecto IWA

**De `harvester/prototype`:**
- `use_case_registry.py` — la metadata de harvesting hints y failure clusters es valiosa para el guided pipeline
- `use_case_intents.py` + `canonical_intent()` — el sistema de intents como fallback de routing es robusto y evita errores silenciosos
- `builders/autohealth.py` — ejemplo de builder manual para proyectos no cubiertos por IWA
- `_canonical_use_case` en el normalizer — normalización de nombres de use cases antes de buscar en el registry

### Unificación sugerida

1. **Primero**: traer `use_case_registry.py` + `use_case_intents.py` a `gul` como metadata layer
2. **Segundo**: agregar el intent fallback de `builders/registry.py` a `gul` para que proyectos no-IWA no rompan
3. **Tercero**: traer `autohealth.py` builder a `gul` como primer ejemplo de builder manual + IWA coexistiendo
4. **No mezclar**: no migrar el patrón manual de autohealth a todos los proyectos IWA — el patrón IWA-backed de `gul` es correcto para ellos
