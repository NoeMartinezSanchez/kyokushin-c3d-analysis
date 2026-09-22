# FASE 1.8F — Task 5: Expansión controlada de la cohorte 250 Hz

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-20
**Script:** `scripts/13_phase_1_8f_cohort_expansion.py`
**Salidas:** `output/scaling_validation/cohort_expansion/` (6 CSV + 4 figuras + log) + 29 configs nuevas en `config/athletes/`

---

## 1. Objetivo

Expandir de forma controlada el pipeline validado (auditoría de señales → recomendaciones → config → Golden Path S02–S05) hacia la **cohorte 250 Hz completa**, **sin actualizar el Data Mart** y **sin construir el dataset ML**. Deja la cohorte auditada y procesada para una etapa posterior (ampliación del Data Mart → dataset ML).

## 2. Universo 250 Hz

Descubierto desde el inventario (`scaling_readiness.csv`, `rate_hz == 250`): **33 atletas** a 250 Hz. Excluidos de la cohorte: **B0367 (200 Hz)** y Grupo C (B0368/B0369/B0370, 200 Hz) → `200HZ_REFERENCE`.

## 3. Atletas ya validados (no reprocesados)

- **B0377** → `EXISTING_PARTIAL_DATA`: 9 filas S02/S03/S05 ya en el Data Mart; no se duplican ni se cuentan como expansión.
- **B0400 · B0371 · B0380** → `EXISTING_GOLDEN_PATH`: Golden Path de Task 3 **consolidado** (leído de `output/scaling_validation/phase_1_8f_task3_*.csv`), sin recalcular C3D.

## 4. Atletas pendientes

**29 atletas** (250 Hz, Grupo B, sin config previa): auditados en esta tarea con el flujo Task 1→3.

## 5. Metodología de auditoría

Cada atleta × técnica (S02–S05, E01-T01) × 6 señales (`RTOE/LTOE/RANK/LANK/RHEE/LHEE`): métricas reutilizadas de `09` (`_metrics_of`, `_events_of`, `_velocity`, `_lateral_evidence`), clasificación con la regla existente `snr≥8 ∧ baseline<100 ∧ accepted≥3` (reuso `11._signal_status`). Sin umbrales nuevos.

## 6. Cobertura por técnica (Golden Path, ejecuciones aceptadas S02–S05)

| Técnica | Aceptadas (nuevos + 3 existentes) |
|---|---|
| S02 | 103 |
| S03 | 101 |
| S04 | 94 |
| S05 | 112 |
| **Total** | **410** (373 de los 29 nuevos + 37 de Task 3) |

Las 33 celdas auditaron con archivo E01-T01 presente (0 celdas `INSUFFICIENT_DATA`).

## 7. Señales recomendadas (114 recomendaciones RECOMMENDED de 116)

`RTOE` 97 · `LTOE` 13 · `LANK` 2 · `RANK` 1 · `LHEE` 1.

## 8. Lateralidad

**16 de 114 celdas recomendadas usan lado IZQUIERDO** (10 atletas distintos), p. ej. B0374-S04→LTOE, B0383-S03→LTOE, B0386-S02→LTOE, B0402-S02→LTOE, B0403-S05→LHEE. En el Golden Path corregido, **70 de 373 ejecuciones nuevas usan `movement_side=L`** (LTOE 59, LANK 6, LHEE 5). La lateralidad es específica de `athlete × technique`; no existe una regla global de lado. Se documenta como lateralidad, no como técnica distinta.

## 9. Casos NEEDS_VALIDATION

**S04** en 2 atletas (la técnica problemática consistente con Task 1/3):
- **B0388-S04** (LTOE, snr 29.1, 2 aceptadas), **B0401-S04** (LTOE, snr 24.0, 1 aceptada) → señal marginal (no alcanza `accepted ≥ 3`). No se fuerza configuración: S04 queda **fuera del config** de esos dos atletas y documentado.

## 10. Golden Path (nuevos)

Procesados **29 atletas** (S02–S05 de las técnicas con recomendación `RECOMMENDED`). Resultados en `cohort_golden_path.csv` / `cohort_events.csv` / `cohort_execution_quality.csv`, con columna `_source` (`cohort_expansion_gp` vs `task3_golden_path`).

## 11. Calidad de ejecuciones

| Evento | n |
|---|---|
| Accepted | 410 |
| Rejected | 303 |
| Review | 22 |

QC: `cohort_execution_quality.csv` por ejecución (`quality_flag`), aplicado sobre las aceptadas (373 nuevas + 37 existentes; sin `WARN/INVALID` sistémicos).

## 12. Problemas encontrados

1. **S04**: única técnica con celdas `NEEDS_VALIDATION` (2/29 atletas) y menor nº aceptadas (94); consistente con el patrón ya observado (baselines altos de pierna en S04).
2. **Lateralidad izquierda frecuente** (16 celdas / 10 atletas): refuerza la necesidad de validación por celda; no es un error.
3. Ninguna señal recomendada quedó vacía; ninguna config inválida según el loader de `02` (las 29 configs se leen correctamente).
4. Sin errores sistémicos de segmentación en la cohorte nueva (373 aceptadas nuevas, ratios rejected/review razonables).

## 13. Limitaciones

- Solo **E01-T01** × S02–S05 en esta expansión (no E02/E03/E04, T02; no S01).
- Configs `provisional`; NO han sido validadas más allá del Golden Path E01-T01.
- S01 permanece `NEEDS_VALIDATION` y **fuera** de esta expansión (política Task 4).
- Cohortes 200 Hz (Grupo C) y normalización 200/250 Hz pendientes.
- Sin ML, sin balanceo, sin normalización, sin actualización del Data Mart.

## 14. Estado de la cohorte (no se presenta como "completa")

| Categoría | Atletas |
|---|---|
| A) Config validada (completa) | — (todas `provisional`) |
| B) Auditoría sin config definitiva | — (0 `NEEDS_VALIDATION` a nivel atleta; 2 celdas S04 pendientes) |
| C) Golden Path procesado (config provisional + GP) | **29 nuevos** + B0400/B0371/B0380 (GP Task 3) |
| D) Pendientes de procesar | **0** |

**Estado global:** 33/33 atletas 250 Hz auditados y con Golden Path S02–S05; **27 `READY_FOR_CONFIG` + 2 `PARTIAL_CONFIG` (B0388, B0401)**; 0 `NEEDS_VALIDATION`/`INSUFFICIENT_DATA` a nivel atleta; B0377 `EXISTING_PARTIAL_DATA`; B0367 `200HZ_REFERENCE`.

## 15. Preparación para actualización del Data Mart

La expansión actual demuestra el pipeline escala (410 ejecuciones S02–S05 — 373 nuevas + 37 de Task 3 —, representación por celda auditada). **Antes** de ampliar el Data Mart (tarea posterior): consolidar las 29 configs `provisional`, resolver las 2 celdas S04 `NEEDS_VALIDATION`, y definir el contrato de ampliación (`source_dataset`, `feature_version`, `comparability_*`) para unificar los 3 orígenes (B0377; Task 3; cohort nuevo).

## 16. Siguiente paso (no ejecutado aquí)

**Ampliar el Data Mart de forma controlada** con la cohorte S02–S05 (410 ejecuciones) manteniendo el contrato; luego retomar el dataset ML v0. No se avanza automáticamente; pendiente de revisión.

## Reproducibilidad

```bash
.venv\Scripts\python scripts\13_phase_1_8f_cohort_expansion.py                # 29 atletas
.venv\Scripts\python scripts\13_phase_1_8f_cohort_expansion.py --limit 2      # muestra
```

El script no escribe fuera de `output/scaling_validation/cohort_expansion/`, `config/athletes/` (configs justificadas) y el listado protegido. Data Mart intacto.

## Tests

`pytest tests -q` → 125 previos + nuevos de Task 5 (validación estática de CSVs + determinismo `--limit 2`). Ver sección 18 de `tests/test_pipeline.py`.