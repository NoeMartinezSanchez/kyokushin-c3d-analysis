# TASK 7B — Baseline ML: clasificación de técnica (S02–S05)

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-21
**Script:** `scripts/16_task7b_baseline_ml.py`
**Salidas:** `output/ml_results/` (11 CSV + `experiment_config.json` + 6 figuras + logs) · Vista ML: `dashboard/app_ml.py`
**NO se hizo tuning, DL, SMOTE, PCA, ni se modificó el ML Dataset v0.**

---

## 1. Objetivo

Determinar si las features biomecánicas de una ejecución permiten clasificar S02–S05 **sobre atletas no vistos** (validación agrupada por `athlete_id`), mediante un **baseline interpretable** (Logistic Regression + StandardScaler per-fold). No busca maximizar una métrica; busca un baseline reproducible y auditable.

## 2. Dataset

`output/ml_dataset_v0/ml_dataset_v0.csv` (**input inmutable**, hash `7c5703a5…` verificado antes/después): **419 filas · 33 atletas · S02–S05 · 250 Hz · E01-T01 · accepted**.

## 3. X

11 features aprobadas en Task 7: `duration_s, time_to_peak_s, vmax, vmean, amax, displacement, path_length, hip_rom, knee_rom, ankle_rom, snr`.

## 4. y

`technique` (clases S02/S03/S04/S05).

## 5. groups

`athlete_id` (unidad de agrupación; prohibido split fila a fila). `fold_assignments.csv`: cada atleta en un único fold (verificado).

## 6. Modelo

`LogisticRegression(solver='lbfgs', multi_class=default multinomial, max_iter=2000, C=1.0, random_state=0)`. (sklearn 1.9: el parámetro `multi_class` fue eliminado; se usa el default multinomial.)

## 7. Preprocesamiento

`StandardScaler` **dentro del Pipeline de cada fold** (ajustado solo con TRAIN; el validation nunca se usa para ajustar).

## 8. Validación

`GroupKFold(n_splits=5)` por `athlete_id`. **Los mismos folds para A y B.** Sin intersección de atletas entre train/validation por fold (asserts internos).

## 9. Métricas

Por fold: accuracy, balanced_accuracy, precision/recall/f1 macro y weighted; por clase (precision/recall/f1/support OOF pooled); OOF con probabilidades; media±std y min–max.

## 10. Baseline A (11 features, con SNR)

| Métrica | media | std | min | max |
|---|---|---|---|---|
| accuracy | 0.629 | 0.079 | 0.532 | 0.744 |
| balanced_accuracy | 0.622 | 0.079 | 0.525 | 0.724 |
| precision_macro | 0.637 | 0.086 | 0.535 | 0.764 |
| recall_macro | 0.622 | 0.079 | 0.525 | 0.724 |
| f1_macro | 0.617 | 0.082 | 0.514 | 0.729 |
| f1_weighted | 0.623 | 0.079 | 0.530 | 0.740 |

## 11. Baseline B (10 features, sin SNR)

| Métrica | media | std | min | max |
|---|---|---|---|---|
| accuracy | 0.651 | 0.059 | 0.570 | 0.733 |
| balanced_accuracy | 0.645 | 0.058 | 0.563 | 0.714 |
| precision_macro | 0.659 | 0.066 | 0.574 | 0.754 |
| recall_macro | 0.645 | 0.058 | 0.563 | 0.714 |
| f1_macro | 0.640 | 0.062 | 0.550 | 0.720 |
| f1_weighted | 0.645 | 0.060 | 0.567 | 0.731 |

## 12. Resultados (descripción, sin ranking)

- Hay **señal predictiva por encima del azar (0.25)**: accuracy media 0.63–0.65 con validación a atletas no vistos.
- **B (sin SNR)** obtuvo medias ligeramente superiores (accuracy 0.651 vs 0.629; f1_macro 0.640 vs 0.617) y **menor desviación** (std accuracy 0.059 vs 0.079). El `snr` tiene el coeficiente |abs| medio más bajo de A (0.184), consistente con un aporte marginal.
- No se declara un "ganador": se describen diferencias observadas.

## 13. Variabilidad por fold

Folds A: f1_macro 0.514–0.729; B: 0.550–0.720. El **fold 3** fue el de mayor métrica en ambos; el **fold 1** el menor en A (0.514). Variabilidad moderada; se muestra en `fold_metrics.png`.

## 14. Errores

A=155 y B=146 de 419 predicciones. Confusión dominante entre patadas: S03→S05 (B=17), S03→S02 (B=18), S04→S02 (B=16), S02→S04 (B=17). **S03 (gedan) y S02 son las de menor recall**; **S05 la de mayor f1 (B: 0.720)**. Patrones en `confusion_matrix.png`/`error_matrix.png`.

## 15. Coeficientes (features escaladas)

Mayores |abs| medio (ambos): `vmax`, `path_length`, `amax`, `hip_rom`, `knee_rom`. `snr` el menor (A: 0.184). Coeficientes por fold×clase en `feature_coefficients.csv`; resumen en `feature_coefficient_summary.csv`. **No se usan para eliminar features.**

## 16. Limitaciones

- n=33 atletas (métricas con varianza; solo 5 folds).
- Baseline lineal: no implica que no exista señal no lineal.
- `movement_side`/`primary_signal` excluidos (leakage de config).
- Severe desbalance no presente (ratio 1.22) pero S03/S04 más difíciles.
- Sin normalización temporal 200/250 (no aplica aquí: todo 250 Hz).

## 17. Qué NO se hizo

Sin tuning, GridSearch, random search, bayésico, AutoML, DL, ensembles, SMOTE/oversampling/undersampling/class weights, PCA, clustering, eliminación automática de outliers, feature selection, scalers alternativos, y **sin modificar any input** (v0, Mart, segmentación, configs, dashboard clásico).

## 18. Criterios para el siguiente experimento

Con la evidencia descrita, el siguiente experimento podría evaluar (en tarea separada): modelos no lineales interpretables con la **misma validación agrupada** y `movement_side` como control; análisis específico de S03/S04 (dificultad); y re-evaluación con más atletas cuando la cohorte 200 Hz esté normalizada. Todo sin tocar `ml_dataset_v0.csv`.