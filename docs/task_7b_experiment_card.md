# Experiment Card — TASK7B_BASELINE_001

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-21
**Estado:** baseline reproducible; sin tuning.

| Campo | Valor |
|---|---|
| Experiment ID | `TASK7B_BASELINE_001` |
| Dataset | ML Dataset v0 (`output/ml_dataset_v0/ml_dataset_v0.csv`, inmutable; hash `7c5703a5…`) |
| Population | 250 Hz / E01 / T01 / S02–S05 / accepted |
| Model | LogisticRegression (solver `lbfgs`, multi_class default multinomial, `max_iter=2000`, `C=1.0`, `random_state=0`) |
| Validation | 5-fold GroupKFold by `athlete_id` |
| Scaling | StandardScaler fitted **per fold** (dentro del Pipeline) |
| Experiments | A = con SNR (11 features) · B = sin SNR (10 features) — mismos folds |
| Target | `technique` (S02/S03/S04/S05) |
| Unit | ejecución (1 fila = 1 repetición) |
| Group | `athlete_id` |
| No tuning | Sí (sin GridSearch/Random/Bayes/AutoML) |
| No data augmentation | Sí (sin SMOTE/oversampling/undersampling) |
| No PCA / clustering / feature selection | Sí |
| Preprocesamiento | StandardScaler; sin normalización temporal (todo 250 Hz) |
| Artefactos | `output/ml_results/` (11 CSV + `experiment_config.json` + 6 figuras) |
| Determinismo | Sin RNG no controlado; byte-identidad entre ejecuciones (verificado) |
| Dashboard | `dashboard/app_ml.py` (presentación read-only de `ml_results/`) |

## Resultados de referencia (media ± std, 5 folds)

| Métrica | A (con SNR) | B (sin SNR) |
|---|---|---|
| accuracy | 0.629 ± 0.079 | 0.651 ± 0.059 |
| balanced_accuracy | 0.622 ± 0.079 | 0.645 ± 0.058 |
| f1_macro | 0.617 ± 0.082 | 0.640 ± 0.062 |
| f1_weighted | 0.623 ± 0.079 | 0.645 ± 0.060 |

Descripción observada (sin ranking): B obtuvo medias ligeramente superiores y menor desviación; `snr` presentó el menor |coef| medio en A (0.184).