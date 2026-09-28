# TASK 10 — Athlete Performance Dashboard (demo)

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-28
**Comando:** `streamlit run dashboard/app_performance.py`
**Framework:** Streamlit (ya usado en `dashboard/app.py` y `app_ml.py`; app independiente)

---

## 1. Objetivo

Materializar el **flujo visual del producto**: `EXECUTION → BIOMECHANICAL FEATURES → ML INFERENCE → TECHNIQUE PREDICTION → BIOMECHANICAL PROFILE`, orientado a una tablet para la federación. No es un dashboard de notebook: claridad > complejidad.

## 2. Arquitectura

`DATA → INFERENCE → PRESENTATION`, separada:
- `dashboard/performance_data.py` — carga de `ml_dataset_v0/ml_dataset_v0.csv`, selector y extracción de features (sin ML).
- `dashboard/performance_ui.py` — helpers de presentación (agrupación, unidades, barras) sin streamlit (testable).
- `dashboard/app_performance.py` — app Streamlit (glue). La **única** entrada al modelo es `from inference import predict_execution`; **no** importa sklearn ni scripts de entrenamiento.

## 3. Framework

Streamlit 1.64 (existente). No se reemplaza ni modifica `app.py` ni `app_ml.py`.

## 4. Fuentes de datos

- `output/ml_dataset_v0/ml_dataset_v0.csv` (selección de ejecuciones reales).
- `inference/` + `output/ml_inference/task8b_rf.joblib` (inferencia; `model_info()` para metadatos).
- Sin C3D, sin segmentación, sin recomputación de features, sin ML dentro del dashboard.

## 5. Flujo de inferencia

1. Seleccionar Atleta y Ejecución (reales del v0);
2. `features_of(row)` → las 10 features (sin SNR);
3. `predict_execution(features)` → `{predicted_technique, technique_name, probabilities, confidence, features}`;
4. presentar.

## 6. Componentes visuales

- **Header**: «KARATE PERFORMANCE INTELLIGENCE» + subtítulo + badge **DEMO**.
- **Selector Atleta → Ejecución** (grande, tablet); se muestra `athlete_id`, `execution_id` y **Técnica de referencia**.
- **Tarjeta de predicción**: técnica + confidence % (+ técnica legible) con aviso de que `confidence` es la probabilidad del clasificador.
- **Probabilidades S02–S05**: barras horizontales + %.
- **Perfil biomecánico** agrupado (Ejecución / Velocidad y aceleración / Movimiento / Rango de movimiento), con unidades (s, m/s, m/s², m, °).
- **Indicador neutro**: «La predicción coincide/difiere de la técnica de referencia» (sin lenguaje de error/score).
- **«Qué ve el modelo»**: variables que usa el clasificador.
- **Información del modelo** (colapsable): Random Forest, `task8b_random_forest_baseline`, inference 0.1.0, 10 features, clases, dataset y validación GroupKFold.
- **Disclaimer** visible.

## 7. Ejecución local

```bash
streamlit run dashboard/app_performance.py
```

## 8. Ejemplo

Estado inicial: ejecución real S04 (`B0371|S04|E01|T01|001`) → predicción **S04 · 92.0 %** (la usada en Task 9). Perfil mostrado con unidades (p. ej. vmax 10.79 m/s, amax 196.6 m/s², ROM cadera 119.7°).

## 9. Tests

`tests/test_performance.py` — imports, carga del dataset, selección/features, `predict_execution` correcto, resultado completo, ejecución inicial válida y **sin `sklearn` ni imports de `scripts/`** en el dashboard.

## 10. Limitaciones

- Solo demo (modelo demostrativo Task 8B; `confidence` ≠ éxito deportivo).
- Solo ejecuciones del v0 (S02–S05 × E01-T01, 250 Hz); selectores limitados a esos datos.
- Sin Comparison/Coach Insights (etapa 3) ni polish final de tablet (etapa 4).

## 11. Siguiente etapa

**Task 11 — Comparison + Coach Insights**: arquitectura preparada (capa DATA/INFERENCE reutilizable) para añadir comparaciones entre atletas y notas orientadas al entrenador sin alterar la capa de inferencia.