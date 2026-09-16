# FASE 1.8D — Sports Performance Intelligence — Dashboard MVP

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-16
**Framework:** Streamlit (instalado; añadido a `requirements.txt`)
**Código:** `dashboard/app.py`, `dashboard/data.py`, `dashboard/components.py`
**Fuente única de datos:** `output/data_mart/athlete_execution_features.csv` (Data Mart de Fase 1.8C)

---

## 1. Objetivo

Crear el primer dashboard MVP que demuestre la visión **"Sports Performance Intelligence"** sobre el Data Mart ya validado. Es un **prototipo descriptivo**: no lee C3D, no recalcula features, no ejecuta el pipeline, no normaliza, no hace ML.

## 2. Arquitectura

```
C3D → Segmentation → Feature extraction → Athlete Data Mart → [ESTE DASHBOARD] → Dashboard final / ML
```
El dashboard es un **consumidor** del Data Mart. Estructura (mínima, escalable):

- `dashboard/data.py` — capa de datos: carga el Data Mart, valida el contrato, filtra, expone comparabilidad (sin recalcularla), y **no importa ezc3d**.
- `dashboard/components.py` — etiquetas legibles en español (conservando nombres técnicos como tooltip), unit map, y helpers de gráficos Plotly.
- `dashboard/app.py` — app Streamlit: sidebar + 5 vistas.

## 3. Dependencia del Data Mart

- El único archivo leído es `output/data_mart/athlete_execution_features.csv`.
- El dashboard **no modifica** ese archivo (verificado por test `test_dashboard_data_mart_not_modified`).
- La comparabilidad se muestra **tal como está** en el Data Mart (`comparability_*`), sin recalcular lógica.

## 4. Vistas

1. **Descripción del atleta (Overview):** KPIs (atleta, nº ejecuciones, técnicas, condiciones, sampling rate, estado QC/calidad). Sin "score" ni ranking.
2. **Análisis de técnica / ejecución:** por repetición, métricas temporales, cinemáticas, articulares y SNR (box + puntos + tarjetas).
3. **Consistencia entre repeticiones:** media, mediana, desviación estándar y coeficiente de variación (CV). Variabilidad descriptiva, **no** calidad.
4. **Comparación entre atletas:** B0367 vs B0377 para la misma técnica+condición+trial, agrupado por estado de comparabilidad:
   - `DIRECTLY_COMPARABLE` (duration_s, time_to_peak_s) — comparación permitida en el alcance actual.
   - `COMPARABLE_WITH_CAVEAT` (displacement, hip/knee/ankle_rom, snr) — visible con reserva.
   - `REQUIRES_NORMALIZATION` (vmax, vmean, amax, path_length) — aviso "Exploratorio solo — normalización temporal pendiente".
5. **Datos, comparabilidad y cobertura:** estados explicados, cobertura actual, limitaciones y versionado del Data Mart.

## 5. Métricas disponibles

| Grupo | Métricas |
|-------|----------|
| Temporal | duration_s, time_to_peak_s |
| Cinemática | vmax, vmean, amax, displacement, path_length |
| Articular | hip_rom, knee_rom, ankle_rom |
| Calidad | snr, qc_status, quality_flag |

## 6. Reglas de comparabilidad (del Data Mart)

- `DIRECTLY_COMPARABLE`: duration_s, time_to_peak_s.
- `COMPARABLE_WITH_CAVEAT`: displacement (geométrico — Hz solo indirecta vía frames), hip/knee/ankle_rom (join side), snr.
- `REQUIRES_NORMALIZATION`: vmax, vmean, amax, path_length (derivación / discretización).

## 7. Cobertura actual

| Dimensión | Valor |
|-----------|-------|
| Atletas | 2 (B0367, B0377) |
| Técnicas | S02, S03, S05 |
| Condición | E01 |
| Trial | T01 |
| Ejecuciones | 18 |

Limitaciones visibles en la app: S01/S04 no forman parte de las métricas validadas actuales; B0377 mantiene validación pendiente en S01/S04; la normalización temporal 200/250 Hz no está completada; es un prototipo sobre el golden-path del Data Mart, no el dataset completo.

## 8. Idiomas y unidades

- **UI en español**; los nombres técnicos originales se conservan como tooltip/metadato (Step 6).
- Unidades respetadas tal como el Data Mart: segundos (tiempos), metros (desplazamiento), m/s (velocidad), m/s² (aceleración), grados (ROM), SNR adimensional. **Sin conversiones silenciosas.**

## 9. Lenguaje neutral

El dashboard usa términos descriptivos ("valor observado", "variabilidad", "medición disponible", "requiere normalización"). **No** dice "mayor = mejor", ni "atleta superior". Solo presenta mediciones y su estado de comparabilidad.

## 10. Cómo ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
(Abrir el navegador en el puerto que indica Streamlit, normalmente `http://localhost:8501`.)

## 11. Validación / tests

- `pytest tests -q` → **61 passed** (51 previos + 10 del dashboard/Data layer), 0 errores, exit=0.
- Tests del dashboard: carga (18 filas), esquema, sin NaN, atletas, técnicas, comparabilidad preservada, filtros por atleta/técnica, **no accede a C3D**, y **Data Mart no modificado**.
- Smoke test: la app arranca con `AppTest` de Streamlit sin excepciones, renderiza 22 gráficos Plotly, y el hash del CSV del Data Mart permanece idéntico tras la sesión.

## 12. Archivos

- **Creados:** `dashboard/app.py`, `dashboard/data.py`, `dashboard/components.py`, `docs/phase_1_8d_dashboard.md`.
- **Modificados:** `requirements.txt` (+streamlit), `docs/phase_1_8c_data_mart.md` (corrección 43→39 columnas), `tests/test_pipeline.py` (+10 tests).
- **No modificado:** `output/data_mart/athlete_execution_features.csv` (verificado con hash).

## 13. Limitaciones

- Sólo golden-path (18 ejecuciones); sin S01/S04.
- Comparación entre atletas limitada a las técnicas/condiciones presentes en ambos (golden path).
- Las features `REQUIRES_NORMALIZATION` se muestran como exploratorias (200 vs 250 Hz aún sin normalizar).
- Sin ML, sin clustering, sin DTW, sin scores de rendimiento.

## 14. Siguiente fase recomendada

**FASE 1.8E** (cuando se consolide):
1. Ampliar cobertura del Data Mart (T02/E02; resolver S01/S04-B0377) y normalización temporal 200/250 Hz.
2. Incorporar más atletas manteniendo el **mismo contrato** de columnas.
3. Añadir vistas analíticas residuales (p. ej., evolución entre trials) y, posteriormente, un módulo de ML **supervisado por humano** (GroupKFold/LOSO) fuera del dashboard.

---

## Resumen de entregables (formato del prompt)

1. **Archivos creados:** `dashboard/` (app, data, components), `docs/phase_1_8d_dashboard.md`.
2. **Archivos modificados:** `requirements.txt`, `docs/phase_1_8c_data_mart.md` (43→39), `tests/test_pipeline.py`.
3. **Framework elegido:** **Streamlit** (ligero, Python, ideal para MVP analítico; Plotly ya estaba en el proyecto para gráficos). El prompt pedía preferir un framework liviano; Streamlit cumple.
4. **Cómo lanzar:** `streamlit run dashboard/app.py`.
5. **Smoke-test:** OK — AppTest sin excepciones, 22 charts, título "Sports Performance Intelligence" presente.
6. **Tests:** **61 passed**, 0 errores, exit=0.
7. **Data Mart NO modificado:** sí (verificado por hash y test dedicado).
8. **C3D NO leídos por el dashboard:** sí (data.py no importa ezc3d; test dedicado).
9. **Cobertura:** 2 atletas, 3 técnicas (S02/S03/S05), 1 condición (E01), 1 trial (T01), 18 ejecuciones.
10. **Limitaciones:** golden-path solamente; S01/S04 y B0377-S01/S04 pendientes; normalización temporal pendiente; sin ML.
11. **Siguiente fase:** 1.8E — ampliar cobertura + normalización temporal + más atletas sobre el mismo contrato.