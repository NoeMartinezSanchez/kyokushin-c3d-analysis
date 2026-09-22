# FASE 1.8F — Task 6: Consolidación controlada del Data Mart 250 Hz

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-21
**Script:** `scripts/14_consolidate_250hz_data_mart.py`
**Salidas:** `output/data_mart/task6_backup/`, `output/data_mart/task6_consolidation/`, Data Mart canónico `athlete_execution_features.csv` (428 filas), `data_mart_summary.csv`

---

## 1. Objetivo

Incorporar al Data Mart las ejecuciones aceptadas **S02–S05 × E01-T01 × 250 Hz** de la cohorte (Tasks 3 y 5) **manteniendo exactamente el contrato estructural existente** (39 columnas, mismas definiciones, mismas unidades). Es una **capa de consolidación y trazabilidad**, no un dataset ML.

## 2. Fuentes utilizadas

| Fuente | Filas | Detalle |
|---|---|---|
| `data_mart/athlete_execution_features.csv` (Mart original) | 18 | B0367 (9, 200 Hz) + B0377 (9, S02/S03/S05) |
| `scaling_validation/cohort_expansion/cohort_golden_path.csv` | 410 | 37 `task3_golden_path` + 373 `cohort_expansion_gp` (10 filas nuevas ya contienen Task 3; no se duplican) |
| `config/athletes/*.yaml` | 34 | validación config ↔ Mart (señal, lado, sampling) |

## 3. Estado del Data Mart antes

18 filas, 2 atletas (B0367/B0377), 3 técnicas (S02/S03/S05), 39 columnas, `source_dataset=feature_readiness_sample`, `mart_version=1.0`.

## 4. Cohorte incorporada

**428 filas finales** = 18 históricas preservadas + **410 nuevas** (S02–S05, E01-T01, 250 Hz). Atletas finales: **34** (B0367 histórico 200 Hz + 33 × 250 Hz). 200 Hz en la capa nueva: **0** (solo B0367 histórico).

## 5. Reglas de inclusión

- S02–S05, E01, T01, `sampling_rate_hz = 250`, `qc_status = accepted`.
- Todas las filas nuevas provienen de `cohort_golden_path` (ejecuciones aceptadas del pipeline, sin recalcular C3D).

## 6. Reglas de exclusión

- S01 (decisión Task 4), cohorte 200 Hz como ampliación (B0368/B0369/B0370 fuera; B0367 solo histórico), rejected/review (no se usa como accepted), técnicas sin evidencia.

## 7. Tratamiento de B0377

B0377 conserva sus **9 filas históricas** (S02/S03/S05) del Mart; **no se duplican** (no aparece en `cohort_golden_path`), no se inventa S04.

## 8. Tratamiento de B0400/B0371/B0380

Consolidados desde `_source=task3_golden_path` (37 filas) de Task 3, **sin reprocesar** sus C3D ni modificar sus features; mapeados al contrato con `source_dataset=task3_golden_path`.

## 9. Tratamiento de B0388/B0401

S04 sin evidencia (NEEDS_VALIDATION, Task 5): no se crearon filas para S04 de estos atletas; quedan `PARTIAL_S02_S05` en la cobertura. No se inventan ni se rellenan valores.

## 10. Tratamiento de S01

S01 permanece fuera del Data Mart (0 filas S01); documentado como pendiente de validación (Task 4).

## 11. Validación del contrato

- **39 columnas, mismo orden** que el histórico; sin columnas nuevas (ni `_source`, ni `signal_used`, etc.).
- `execution_id` con la convención del diccionario `{athlete}|{technique}|{condition}|{trial}|{repetition:03d}` → **único** (428/428).
- **Históricas idénticas al backup** (18/18, valores verificados columna a columna).
- Sin NaN en features obligatorias (11 features, 0 NaN).
- `qc_status=accepted` y `quality_flag=OK` en el 100 %.
- Unidades conservadas del pipeline (m/s, m/s², m, deg) — el mapeo es solo de nombres.

## 12. Duplicados

`data_mart_task6_duplicates.csv` (vacío): 0 colisiones histórico/bloque y 0 duplicados en la suma (`execution_id` único, atletas disjuntos).

## 13. Reconciliación

`source_reconciliation.csv`:

| source | candidate/inserted | duplicate | excluded |
|---|---|---|---|
| feature_readiness_sample | 18 / 18 | 0 | 0 |
| task3_golden_path | 37 / 37 | 0 | 0 |
| cohort_expansion_gp | 373 / 373 | 0 | 0 |
| TOTAL | 428 / 428 | 0 | 0 |

## 14. Cobertura final

`data_mart_task6_coverage.csv` (34 atletas): `COMPLETE_S02_S05` (S02–S05 con filas) salvo B0367 (`HISTORICAL_ONLY`), B0377/B0388/B0401 (`PARTIAL_S02_S05`). Ejecuciones por técnica: **S05=118 · S02=109 · S03=107 · S04=94**. Sin rankings.

## 15. Comparability

`comparability_*` conserva el mapa del contrato (reusado de `07`): duration/time_to_peak → `DIRECTLY_COMPARABLE`; vmax/vmean/amax/path_length → `REQUIRES_NORMALIZATION`; displacement/rom*/snr → `COMPARABLE_WITH_CAVEAT`. No se convierte en decisión de inclusión para ML.

## 16. Limitaciones

- Solo E01-T01 × S02–S05; S01 fuera; 200 Hz no es cohorte nueva.
- Configs aún `provisional`; B0388/B0401-S04 sin filas (documentado).
- Sin normalización, sin ML, sin splits; `data_mart_summary.csv` refleja las cifras consolidadas.

## 17. Estado para ML Dataset v0

El Mart consolidado (428 filas, 39 columnas, contratable, trazable por `source_dataset`) queda **listo como fuente** para la construcción del dataset ML v0 (Task 7). No se construye ahí ningún dataset de entrenamiento.

## 18. Recomendación para Task 7

Construir el dataset ML v0 a partir de este Mart: definir features candidatas, `target_technique`, columnas de identidad/leakage, y plan de validación por atleta (GroupKFold / LOSO). S01 excluido; decisiones de normalización posteriores.

## Reproducibilidad

```bash
.venv\Scripts\python scripts\14_consolidate_250hz_data_mart.py
```

Ejecutar dos veces produce el mismo resultado (determinista). El Mart original queda respaldado en `output/data_mart/task6_backup/`.

## Tests

`pytest tests -q` → 136 previos ajustados al nuevo contrato + nuevos de Task 6. Ver sección 19 de `tests/test_pipeline.py`.