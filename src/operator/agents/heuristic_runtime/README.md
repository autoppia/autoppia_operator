# Heuristic Runtime Archive

This folder contains the heuristic inference operator that was used to close demo-web use cases quickly.

It is intentionally separated from the clean inference path.

- Clean inference runtime: `src/operator/agents/operator.py`
- Heuristic runtime archive: `src/operator/agents/heuristic_runtime/structured_heuristic_operator.py`

Runtime selection:
- `WEB_AGENT_RUNTIME=structured|structured_inference|operator|clean|clean_model` -> clean model-driven inference
- `WEB_AGENT_RUNTIME=heuristic|heuristic_structured|structured_heuristic` -> archived heuristic inference
