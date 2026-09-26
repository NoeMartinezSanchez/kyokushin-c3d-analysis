# Experiment Card — TASK8B_NONLINEAR_002

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-22
**Estado:** experimento controlado; sin tuning.

| Campo | Valor |
|---|---|
| EXPERIMENT ID | `TASK8B_NONLINEAR_002` |
| MODEL | RandomForestClassifier (params fijos) |
| DATASET | ML Dataset v0 (`output/ml_dataset_v0/ml_dataset_v0.csv`, inmutable) |
| N | 419 ejecuciones · 33 atletas |
| FEATURES | 10 (Experimento B de Task 7B): duration_s, time_to_peak_s, vmax, vmean, amax, displacement, path_length, hip_rom, knee_rom, ankle_rom |
| TARGET | `technique` (S02/S03/S04/S05) |
| GROUP | `athlete_id` |
| VALIDATION | GroupKFold(5) por atleta; **folds congelados de Task 7B** (validados contra re-derivación) |
| HYPERPARAMETERS | n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True, class_weight=None |
| RANDOM STATE | 42 |
| PRIMARY METRICS (OOF) | accuracy 0.663 ± 0.115 · f1_macro 0.646 ± 0.129 · balanced_accuracy 0.654 ± 0.124 |
| BASELINE (Task 7B, exp B) | accuracy 0.651 ± 0.059 · f1_macro 0.640 ± 0.062 · balanced_accuracy 0.645 ± 0.058 |
| CHANGES | media OOF ligeramente superior (≈ +1.3 pp accuracy, +0.5 pp f1_macro) pero **std ≈ 2× mayor**; errores 141 (vs 146), 95 compartidos, 51 corregidos, 46 nuevos |
| RESULT | Evidencia descriptiva moderada; **sin ganador declarado** (mayor varianza entre folds) |
| LIMITATIONS | n=33 atletas/5 folds; RF sin calibrar; varianza alta; solo S02–S05 × E01-T01 250 Hz |

## Artefactos
`output/ml_results_task8b/` — `oof_predictions`, `fold_metrics`, `global_metrics`, `metrics_by_technique`, `confusion_matrix`, `feature_importance`, `athlete_metrics`, `error_comparison`, `error_comparison_pairs`, `model_comparison`, `experiment_config.json`, `integrity_hashes.csv` (10/10 PASS), `figures/` (5).

## No tuning / no augmentation / no feature selection / no C3D / no Data Mart
Sí (en todos los casos).