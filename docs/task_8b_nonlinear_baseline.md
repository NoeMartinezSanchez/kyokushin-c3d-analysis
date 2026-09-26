# TASK 8B — Baseline no lineal interpretable (RandomForest) con validación agrupada

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-22
**Script:** `scripts/19_task8b_nonlinear_baseline.py`
**Salidas:** `output/ml_results_task8b/` (10 CSV + `experiment_config.json` + 5 figuras + log)
**Contexto:** comparación directa con el baseline lineal de Task 7B (**Experimento B**, 10 features). Sin tuning, sin ranking, sin causalidad.

---

## 1. Objetivo

Evaluar si una representación **no lineal interpretable** (Random Forest con parámetros fijos) captura relaciones entre las mismas features que el baseline lineal no captura completamente, bajo **exactamente la misma validación agrupada por atleta**.

## 2. Dataset

`output/ml_dataset_v0/ml_dataset_v0.csv` (inmutable): **419 ejecuciones · 33 atletas · S02–S05 · 250 Hz · E01-T01**.

## 3. Features

Las **10 del Experimento B (Task 7B)**: `duration_s, time_to_peak_s, vmax, vmean, amax, displacement, path_length, hip_rom, knee_rom, ankle_rom`. **SNR excluida.** Sin features nuevas.

## 4. Target

`technique` (S02/S03/S04/S05).

## 5. Groups

`athlete_id`.

## 6–7. Modelo y parámetros fijos

`RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1, class_weight=None)` — **sin optimización**.

## 8. Validación

`GroupKFold(5)` por atleta, reutilizando los **folds congelados** de Task 7B (`fold_assignments.csv`, 33 atletas → 6/7/7/6/7) y **validado** contra re-derivar GroupKFold sobre v0 (idéntico mapeo). OOF: cada `execution_id` aparece **exactamente una vez** (419/419). Sin intersección de atletas train/test por fold (asserts internos).

## 9. Resultados globales (OOF, 5 folds)

| métrica | media | std | min | max |
|---|---|---|---|---|
| accuracy | 0.6632 | 0.1145 | 0.551 | 0.837 |
| balanced_accuracy | 0.6540 | 0.1244 | 0.506 | 0.825 |
| precision_macro | 0.6710 | 0.1274 | 0.531 | 0.865 |
| recall_macro | 0.6540 | 0.1244 | 0.506 | 0.825 |
| f1_macro | 0.6457 | 0.1289 | 0.498 | 0.832 |
| f1_weighted | 0.6535 | 0.1193 | 0.528 | 0.836 |

## 10. Resultados por fold

`fold_metrics.csv` (n_train/n_valid, nº atletas y las 6 métricas por fold 1–5). La variabilidad entre folds es evidente (accuracy 0.551–0.837).

## 11. Resultados por técnica (OOF pooled)

| técnica | precision | recall | f1 | support |
|---|---|---|---|---|
| S02 | 0.656 | 0.594 | 0.624 | 106 |
| S03 | 0.620 | 0.596 | 0.608 | 104 |
| S04 | 0.611 | 0.617 | 0.614 | 94 |
| S05 | 0.742 | 0.826 | 0.782 | 115 |

OBSERVACIÓN: S05 es la técnica con mayor f1 y S03 la menor — describe al clasificador en este dataset; no implica dificultad intrínseca.

## 12. Matriz de confusión

`confusion_matrix.csv` (conteos + normalizada por clase verdadera); orden S02–S05.

## 13. Comparación Task 7B vs Task 8B (media ± std sobre OOF)

| métrica | LogReg (7B-B) | RF (8B) |
|---|---|---|
| accuracy | 0.651 ± 0.059 | 0.663 ± 0.115 |
| balanced_accuracy | 0.645 ± 0.058 | 0.654 ± 0.124 |
| f1_macro | 0.640 ± 0.062 | 0.646 ± 0.129 |
| f1_weighted | 0.645 ± 0.060 | 0.653 ± 0.119 |

LENGUAJE PERMITIDO: "El modelo no lineal obtuvo una **accuracy OOF media superior en ~1.3 puntos porcentuales** y f1_macro superior en ~0.5 puntos, **pero con el doble de desviación entre folds** (0.115 vs 0.059)". No se declara un ganador.

## 14. Comparación por técnica

S05 mejora (f1 0.72 → 0.78); S04 estable (0.64 → 0.61); S03 similar (0.61 → 0.61); S02 similar (0.63 → 0.62). Variaciones puntuales sin evidencia de superioridad general.

## 15. Comparación de errores (7B-B vs 8B)

| error_set | n | shared | corregidos por 8B | nuevos en 8B |
|---|---|---|---|---|
| task7b_B | 146 | 95 | 51 | 46 |
| task8b_RF | 141 | 95 | 51 | 46 |

Pares principales (S03→S02, S02→S04, S03→S05, S04→S02, S02→S03, S05→S03): ver `confusion_pairs_comparison.png`; los conjuntos de pares de confusión y su persistencia en `error_comparison_pairs.csv` (**12 pares persistidos** en ambos modelos).

## 16. Comparación con Task 8A

Identificados por `execution_id`: **95 errores compartidos**, 51 corregidos por RF, 46 nuevos. La persistencia de ciertos pares sugiere patrones de confusión estables; se describe sin causalidad.

## 17. Feature importance (RF, media ± std)

`hip_rom (0.171)`, `amax (0.163)`, `knee_rom (0.112)`, `path_length (0.103)`, `vmax (0.098)`, ... `duration_s (0.057)`. OBSERVACIÓN: RF da mayor peso relativo a `hip_rom`/`amax`/`knee_rom`; LogReg a vmax/path_length/amax/hip_rom. **No son escalas comparables**; se describe, no se concluye causalidad.

## 18–21. Atletas / Limitaciones / Interpretación / Recomendación

- Artletas: `athlete_metrics.csv` (sin ranking; sin sujetos con concentración extrema).
- Limitaciones: RF con mayor varianza entre folds; n=33; baseline no calibrado; OOF=B de referencia.
- Interpretación: la media OOF del RF es ligeramente superior pero con más varianza; el solapamiento S05/otras parcial se mantiene. Evidencia insuficiente para afirmar ganador.
- Recomendación Task 9: consolidar con la evidencia de 8A/8B (valorar si se prioriza varianza vs media), y en paralelo normalización 200/250 Hz / Grupo C.

## Reproducibilidad

```bash
.venv\Scripts\python scripts\19_task8b_nonlinear_baseline.py
```

Determinista (random_state=42; folds congelados); `integrity_hashes.csv` 10/10 PASS.