# TASK 9 — Capa de Predicción / Inferencia (Demo)

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-28
**Paquete:** `inference/` · **Artefacto:** `output/ml_inference/` · **CLI:** `scripts/21_demo_inference.py`
**Evaluación:** demo reproducible de inferencia; **no** es un sistema de predicción de rendimiento deportivo.

---

## 1. Objetivo

Convertir los resultados de ML ya existentes en una **capa de inferencia reutilizable, independiente del dashboard**, que reciba las características biomecánicas de una ejecución y devuelva una **identificación probabilística de la técnica** junto con el perfil de ejecución. No se entrena ni se modifica nada histórico.

## 2. Modelo utilizado

**RandomForestClassifier de Task 8B** (parámetros fijos: `n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1, class_weight=None`). Se usa como **modelo demostrativo / baseline no lineal** — nunca como "mejor modelo".

> **Decisión documentada (autorizada):** no existía modelo serializado. Se ajustó un modelo sobre el **dataset completo (419)** con los mismos parámetros fijos y se serializó como **artefacto demostrativo**. Sus métricas internas **NO** estiman generalización: las cifras oficiales son las OOF de Task 8B (`output/ml_results_task8b/`). Esto no es tuning.

## 3. Features

Las **10 features del Experimento B** (sin `snr`): `duration_s, time_to_peak_s, vmax, vmean, amax, displacement, path_length, hip_rom, knee_rom, ankle_rom`. `snr` existe en el dataset pero fue excluida del modelo; por eso no forma parte del contrato de entrada.

## 4–6. Target, clases y preprocesamiento

- Target: `technique`; clases `S02/S03/S04/S05`; poblacón 250 Hz · E01-T01 · accepted (ML Dataset v0, inmutable).
- Preprocesamiento en inferencia: **ninguno** (RF no requiere escala); se valida exactitud del contrato.

## 7. Artefacto del modelo

- Ruta: `output/ml_inference/task8b_rf.joblib`
- `artifact_info.json`: `model_version="task8b_random_forest_baseline"`, `inference_version="0.1.0"`, **sha256=`04b859efce45…54c90`**, size 4 460 841 bytes, features/target/clases, md5 de `ml_dataset_v0.csv`, fecha, nota demo.
- Serialización **determinista** (re-build → mismo sha256).

## 8. Interfaz de inferencia

```python
from inference import predict_execution, load_model, model_info

res = predict_execution(features_dict)   # features_dict: las 10 features
# -> {predicted_technique, technique_name, probabilities, confidence, features}
```

Validación estricta (errores controlados `InferenceError`): feature faltante, feature extra (fuera del contrato), NaN, Inf, no-numérico. Sin correcciones silenciosas.

## 9–10. Ejemplo de entrada y salida

Entrada = fila real del v0 (p. ej. una de `S04…`). Salida (CLI): técnica predicha, probabilidades S02–S05 (suma≈1), `confidence` = probabilidad de la clase predicha, perfil biomecánico (vmax, amax, ROM cadera/rodilla/tobillo, duración).

Ejecuciones reales probadas (una por técnica): S02→S02, S03→S03, S04→S04, S05→S05 (predicción correcta; si alguna no coincidiera, sería un resultado válido, sin modificar el modelo).

## 11. Tests

`tests/test_inference.py` (**13 tests**): artefacto existe, hash == info, entrada válida, features faltante/extra/NaN/Inf → error, probs≈1, clase ∈ S02–S05, resultado con 10 features, determinismo (2 llamadas idénticas), modelo carga (n_features_in_=10), v0 intacto (md5), las 4 ejecuciones reales válidas.

## 12. Reproducibilidad

La misma entrada → el mismo resultado (determinista). Rebuild del artefacto → sha256 idéntico (verificado). Documentado en `artifact_info.json`.

## 13. Limitaciones

- Modelo **demostrativo** (no estima generalización; oficiales = OOF 8B).
- Solo las 10 features (sin SNR) y técnica como target.
- `confidence` es la probabilidad del clasificador; **no** es una probabilidad validada de éxito deportivo.
- No incorpora S01, 200 Hz, E02/E03/E04, ni atletas nuevos.

---

> **Esta capa demuestra inferencia sobre patrones biomecánicos; no constituye todavía un sistema de predicción de rendimiento deportivo ni de resultados competitivos.**

## Consumo futuro (etapa 2 — Athlete Performance Dashboard)

`inference/` es independiente del dashboard: la UI (p. ej. Streamlit) importará `predict_execution` y `model_info` y renderizará "Technique Prediction + Biomechanical Profile" por ejecución. No se implementa en esta tarea.