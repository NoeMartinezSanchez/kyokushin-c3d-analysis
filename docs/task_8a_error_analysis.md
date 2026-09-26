# TASK 8A — Análisis biomecánico descriptivo de errores OOF (baseline)

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-22
**Script:** `scripts/18_task8a_error_analysis.py`
**Salidas:** `output/task8a_error_analysis/` (10 CSV + 6 figuras + log)
**Naturaleza:** análisis **post-hoc y descriptivo**; no entrena, no hace tuning, no modifica Task 7B / v0 / Data Mart / dashboard.

---

## 1. Objetivo

Entender, con evidencia descriptiva de las predicciones OOF ya generadas, qué características biomecánicas aparecen asociadas a las ejecuciones que el baseline confunde y qué patrones muestran los principales pares de confusión. No busca mejorar el modelo ni declarar técnicas "más difíciles".

## 2. Dataset utilizado

`output/ml_dataset_v0/ml_dataset_v0.csv` (419 ejecuciones, 33 atletas, S02–S05, 250 Hz · E01-T01) — **solo lectura, sin modificar**.

## 3. Baseline analizado

Task 7B (`TASK7B_BASELINE_001`, congelado): LogisticRegression + StandardScaler per-fold, GroupKFold(5) por atleta. **Referencia para el análisis de errores: Experimento B (10 features, sin SNR).** A se usa solo para comparación por `execution_id`.

## 4. Fuente de predicciones OOF

`output/ml_results/oof_predictions.csv` (read-only): `experiment, execution_id, athlete_id, true_technique, predicted_technique, fold, prob_S02..S05`. Join 1:1 por `execution_id` con v0 (419/419 en cada experimento; verificado).

## 5. Definición de correcto / error

`correct = true==pred` · `error = true!=pred` (descriptivo).

## 6. Features analizadas (Experimento B)

`duration_s, time_to_peak_s, vmax, vmean, amax, displacement, path_length, hip_rom, knee_rom, ankle_rom`. Sin features nuevas, sin transformaciones.

## 7. Resultados generales (OOF)

| Experimento | total | correct | errors | error_pct |
|---|---|---|---|---|
| A | 419 | 264 | 155 | 36.99 % |
| **B (referencia)** | 419 | 273 | **146** | **34.84 %** |

Comparación A vs B por `execution_id`: **144 errores compartidos**; A-unique 11; B-unique 2 (descriptivo; sin declarar ganador).

## 8. Resultados por técnica (OOF, B)

| técnica | total | correct | errors | accuracy | error_prop |
|---|---|---|---|---|---|
| S02 | 106 | 68 | 38 | 0.642 | 0.359 |
| S03 | 104 | 59 | 45 | 0.567 | **0.433** |
| S04 | 94 | 61 | 33 | 0.649 | 0.351 |
| S05 | 115 | 85 | 30 | 0.739 | 0.261 |

OBSERVACIÓN: S03 es la técnica con mayor proporción de error en OOF (0.433) y S05 la menor (0.261). Esto describe al clasificador en este dataset; **no** prueba dificultad biomecánica intrínseca.

## 9. Principales pares de confusión (OOF, B)

| true→pred | n | prop. de errores |
|---|---|---|
| **S03→S02** | 18 | 12.3 % |
| **S02→S04** | 17 | 11.6 % |
| **S03→S05** | 17 | 11.6 % |
| **S04→S02** | 16 | 11.0 % |
| S02→S03 | 14 | 9.6 % |
| S05→S03 | 13 | 8.9 % |

Otros pares con ≥5 observaciones aparecen automáticamente en `confusion_pairs.csv`.

## 10. Análisis biomecánico descriptivo

**Correct vs Error (medianas, B):**

| feature | correct | error | dif. abs. | dif. rel % |
|---|---|---|---|---|
| hip_rom | 118.9 | 107.5 | 11.4 | 9.6 |
| ankle_rom | 61.8 | 57.0 | 4.8 | 7.8 |
| amax | 125.9 | 121.1 | 4.8 | 3.8 |
| knee_rom | 141.5 | 136.9 | 4.6 | 3.2 |
| duration_s | 0.928 | 0.850 | 0.078 | 8.4 |
| time_to_peak_s | 0.304 | 0.274 | 0.030 | 9.9 |
| vmax | 8.20 | 7.71 | 0.49 | 6.0 |

OBSERVACIÓN: las ejecuciones erróneas presentan medianas descriptivamente más bajas en ROM de cadera/tobillo, menor duración, menor vmax y menor tiempo-al-pico.
INTERPRETACIÓN DESCRIPTIVA: las confusiones tienden a concentrarse en ejecuciones con menor magnitud general cinemática.
HIPÓTESIS (requiere validación): ejecuciones de menor velocidad/ROM pueden quedar más cerca en el espacio de features lineales de otras técnicas.

**Perfiles por par (medianas; unidades del pipeline):**
- **S02→S04** (n=17): las S02 confundidas tienen `displacement` mediana 0.189 vs 0.097 de las S02 correctas (+94.9 %), y mayor `amax` (+17 %) y `hip_rom` (+15.8 %) — descriptivamente más cercanas al perfil de un Mawashi (mayor extensión/desplazamiento).
- **S03→S02** (n=18): menor `ankle_rom` (−10.9 %) y mayor `time_to_peak_s` (+9.3 %) que las S03 correctas; y `hip_rom` mayor (+8.0 %).
- **S03→S05** (n=17): ver `confusion_pair_profiles.csv`.
- **S02→S03**, **S05→S03**, **S04→S02**: idem en CSV.

Se usa MEDIANA (robusta). Diferencias pequeñas se reportan tal cual; ninguna se interpreta como causa.

## 11. Análisis de confianza (B)

| grupo | mediana prob_true | mediana margen |
|---|---|---|
| correct | 0.781 | 0.624 |
| error | 0.223 | 0.289 |

OBSERVACIÓN: los errores tienen probabilidad de la clase verdadera baja (mediana 0.223) y margen menor; no obstante, el margen de muchos errores es positivo (≈0.29), por lo que **coexisten errores "de baja" y "de media-alta confianza"**. No se propone umbral de rechazo.

## 12. Análisis por atleta (B) — sin ranking

`athlete_error_summary.csv` (n, correct, errors, error_rate, técnicas presentes). OBSERVACIÓN: **ningún atleta** con ≥20 ejecuciones supera el 50 % de error; los errores no se concentran en un sujeto dominante. Variabilidad entre sujetos documentada como posible fuente de varianza, no como evaluación.

## 13. Análisis por fold (B)

`fold_error_summary.csv` (n, errors, error_rate por fold 1–5; 77–89 ejecuciones por fold). OBSERVACIÓN: los patrones de error se repiten en varios folds (no es un artefacto de un solo fold).

## 14. Limitaciones

- n=33 atletas, 5 folds; clasificador lineal (no informa sobre señal no lineal).
- OOF de B=146 errores: pares con n<10 tienen perfiles con alta variabilidad (p. ej. S02→S05, n=7, con valores extremos en varias features).
- `snr` no analizada en B (excluida del baseline de referencia).
- Solo S02–S05 × E01-T01; sin E02/E04, sin 200 Hz.
- Las diferencias son descriptivas; sin pruebas inferenciales.

## 15. Interpretación

Los errores muestran un **solapamiento descriptivo de la representación de features** entre ciertas técnicas (p. ej. S02 confundida con S04 presenta desplazamientos/ROM descriptivamente mayores). Esto describe solapamiento en el espacio de features del dataset, **no** que las técnicas no se puedan distinguir biomecánicamente (requeriría más evidencia).

## 16. Qué NO puede concluirse

No puede concluirse: causalidad de las features, que una técnica sea biomecánicamente más difícil, que un atleta sea peor, que el modelo "entienda biomecánica", ni que el modelo esté listo para producción.

## 17. Recomendaciones para Task 8B

Con esta evidencia, Task 8B podría: examinar los perfiles por par con n mayor (S03→S02, S02→S04, S03→S05) usando features adicionales ya disponibles (sin re-entrenar), y evaluar si un modelo no lineal con la misma validación agrupada reduce el solapamiento — manteniendo este análisis como referencia congelada.

## Reproducibilidad

```bash
.venv\Scripts\python scripts\18_task8a_error_analysis.py
```

Determinista (sin RNG); `integrity_hashes.csv` verifica md5 antes/después de los 9 archivos protegidos (todo PASS).