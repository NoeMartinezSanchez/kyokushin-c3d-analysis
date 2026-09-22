# TASK 7 — Diseño y auditoría del ML Dataset v0

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-21
**Script:** `scripts/15_build_ml_dataset_v0.py`
**Salidas:** `output/ml_dataset_v0/` (12 CSV + 3 figuras + `ml_validation_strategy.md` + log)
**NO se entrenaron modelos ni se ajustó normalización.**

---

## 1. Objetivo

Diseñar, auditar y documentar el primer dataset preparado para Machine Learning (ML Dataset v0) para **clasificación de técnica** (`y = technique`), usando únicamente ejecuciones aceptadas 250 Hz × S02–S05 × E01 × T01. Documentar leakage, comparabilidad, cobertura por atleta y estrategia de validación agrupada.

## 2. Fuente de datos

`output/data_mart/athlete_execution_features.csv` (428 filas, **39 columnas** — contrato real; no 43). Diccionario: `docs/athlete_data_mart_dictionary.md`. Auditoría reconocida: `docs/phase_1_8f_task6_data_mart_consolidation.md`.

## 3. Universo analizado

Mart consolidado: 428 filas, 34 atletas (33 × 250 Hz + B0367 200 Hz), S02–S05, E01-T01. Tras filtros → **419 filas / 33 atletas** (solo se eliminaron las 9 de B0367 por 200 Hz).

## 4. Filtros (trazabilidad en `population_filter_audit.csv`)

| etapa | antes | después | removidas | razón |
|---|---|---|---|---|
| total_mart | 428 | 428 | 0 | población inicial |
| sampling_250hz | 428 | 419 | 9 | excluye 200 Hz (B0367) |
| condition_E01 | 419 | 419 | 0 | golden path |
| trial_T01 | 419 | 419 | 0 | golden path |
| technique_S02_S05 | 419 | 419 | 0 | S01 fuera |
| qc_accepted | 419 | 419 | 0 | accepted |
| quality_ok_warn | 419 | 419 | 0 | OK/WARN |

## 5. Unidad de observación

1 fila = 1 ejecución aceptada de una técnica (repetición individual). NO agregaciones por atleta.

## 6. Variable objetivo

`technique` (target), clases S02/S03/S04/S05. `class_distribution.csv`: S02=106 (25.3 %), S03=104 (24.8 %), S04=94 (22.4 %), S05=115 (27.4 %); ratio max/min = **1.22 → BALANCED** (sin balancear en esta tarea). Atletas por clase: S02/S03/S05=33, S04=30 (B0377/B0388/B0401 sin S04).

## 7. Features candidatas (11)

`duration_s, time_to_peak_s, vmax, vmean, amax, displacement, path_length, hip_rom, knee_rom, ankle_rom, snr` — evaluadas por unidad, comparabilidad, distribución, missingness (0), outliers (diagnóstico) y correlación (redundancia). Nada se elimina automáticamente.

## 8. Features excluidas y motivo (`feature_audit.csv`, 39 columnas reales)

- **Leakage de individuo/ejecución:** `athlete_id`, `execution_id` (roles `identifier`, no predictors).
- **Config de señal/lateralidad (leakage MEDIUM):** `primary_signal`, `movement_side` (70 filas con lado L concentradas en S04/S03: incluir permitiría aprender "qué señal se configuró").
- **Constantes tras filtro:** `sampling_rate_hz`, `condition`, `trial`, `qc_status`, `quality_flag`.
- **Metadata/versionado:** `event_id`, `repetition`, `source_dataset`, `feature_version`, `segmentation_version`, `units_version`, `mart_version`, `comparability_*` (10).

## 9. Leakage audit

- `athlete_id`/`execution_id`: HIGH → se conservan solo como identifiers en el CSV (nunca predictores).
- `primary_signal`/`movement_side`/`source_dataset`: MEDIUM → excluidos como features.
- Validación futura agrupada por `athlete_id` elimina el leakage grupal de repeticiones (no split fila a fila).

## 10. Comparabilidad (`feature_comparability_audit.csv`)

duration/time_to_peak → `DIRECTLY_COMPARABLE`; vmax/vmean/amax/path_length → `REQUIRES_NORMALIZATION` (`normalization_required=True`, raw permitido en v0, normalización dentro de cada fold en 7B); displacement/rom*/snr → `COMPARABLE_WITH_CAVEAT`. `snr` incluida con caveat (proxy señal/baseline, riesgo documentado).

## 11. Missingness

0 NaN en las 11 features (419/419).

## 12. Distribuciones (`feature_distribution_summary.csv` + figuras)

Percentiles p01–p99 e IQR por feature; figuras `feature_distributions_hist.png` y `feature_by_technique_box.png`. Ejemplos: vmax media 7.45 m/s (p95 10.6); snr muy sesgada (mediana 329, p99 4211).

## 13. Outliers (`outlier_audit.csv`, IQR)

Diagnóstico, NO eliminación. Destacan: time_to_peak 16.5 %, vmean 16.0 %, path_length 15.3 %, duration 14.6 %, snr 13.1 %, vmax 12.9 %; amax 0 %. Se documentan para revisión en 7B.

## 14. Correlaciones (`feature_correlation.csv` + heatmap)

Altas: vmean–path_length **0.86**, vmax–vmean **0.85**, vmax–path_length **0.84**, vmax–amax **0.83**, duration_s–path_length **0.82**. Se identifican como redundancias potenciales; **no** se eliminan con criterio automático.

## 15. Cobertura por atleta (`athlete_coverage.csv`)

33 atletas, 9–18 ejecuciones cada uno; 3 atletas con cobertura parcial (<4 técnicas): **B0377, B0388, B0401** (sin S04) — documentados, no fabricados.

## 16. Desbalance de clases

`imbalance_level = BALANCED` (ratio 1.22). Sin oversampling/undersampling/SMOTE/class weights.

## 17. Estrategia de validación (`ml_validation_strategy.md`)

**Grupo = `athlete_id`; prohibido split fila a fila.** Baseline: **GroupKFold(n_splits=5)** (33 grupos, clase con menos grupos S04=30 ≥5). Documentadas: StratifiedGroupKFold (viable k≤~30 pero estratificación parcial por grupos incompletos) y Leave-One-Group-Out (33 folds). Normalización/scalado solo dentro de cada fold en 7B (evita leakage).

## 18. Limitaciones

- n=33 atletas (varianza alta de métricas).
- Solo 250 Hz × E01-T01 × S02–S05; S01 y 200 Hz fuera.
- 3 atletas sin S04; `snr` con caveat; correlate kinetics redundantes (sin eliminar).
- Configs provisionales (representación auditada por celda).
- Sin validación en E02/E03/E04/T02.

## 19. Qué NO se hizo

No se entrenó; no se ejecutó clustering/PCA/feature selection/tuning; no se normalizó para entrenamiento; no se modificó el Data Mart (la **fuente operacional** es el archivo de 39 columnas; se documentó la discrepancia con cualquier mención de 43), scripts de segmentación, configs ni dashboard.

## 20. Criterios para pasar a Task 7B

- Dataset v0 reproducible (419×14) y manifest acorde.
- Confirmada la estrategia GroupKFold (agrupada) y el historial de features/comparabilidad.
- Task 7B definirá el pipeline de evaluación (scaler per-in-fold, métricas por fold, manejo de outliers/redundancia según evidencia, sin tocar este dataset).