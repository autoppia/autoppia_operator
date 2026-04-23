# Selector Coverage Matrix (Non-Cinema)

This matrix captures baseline selector coverage from IWA trajectories for the non-cinema rollout scope.

Legend:
- `with_attr`: use cases with at least one `attributeValueSelector` step.
- `no_attr`: use cases with no `attributeValueSelector` steps (typically XPath-heavy).

| Project | Total use cases | with_attr | no_attr | Rollout phase |
| --- | ---: | ---: | ---: | --- |
| `autocrm` | 23 | 22 | 1 | 1 |
| `autobooks` | 20 | 17 | 3 | 1 |
| `automail` | 25 | 19 | 6 | 1 |
| `autodelivery` | 20 | 15 | 5 | 1 |
| `autowork` | 36 | 26 | 10 | 1 |
| `autocalendar` | 19 | 6 | 13 | 2 |
| `autolodge` | 19 | 10 | 9 | 2 |
| `autodining` | 20 | 0 | 20 | 3 |
| `autozone` | 14 | 0 | 14 | 3 |
| `autoconnect` | 26 | 4 | 22 | 3 |
| `autolist` | 12 | 0 | 12 | 3 |

## Immediate implications

- Phase 1 projects can be upgraded quickly by prioritizing id/class/text selector helpers and adding fallback only where needed.
- Phase 2 projects need mixed treatment: semantic selectors for core controls, strict fallback for deeply nested views.
- Phase 3 projects require XPath semantic extraction plus curated helper maps to ensure deterministic robustness.
