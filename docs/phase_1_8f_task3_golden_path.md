# FASE 1.8F — Task 3: Golden Path E01-T01 × S01–S05

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-20
**Script:** `scripts/10_phase_1_8f_task3_golden_path.py`
**Salidas:** `output/scaling_validation/` (4 CSV + 15 figuras de segmentación + 2 resúmenes + log)

---

## 1. Objetivo

Ejecutar **por primera vez** el pipeline de segmentación sobre B0400, B0371 y B0380 usando **exclusivamente** las configuraciones de la Tarea 2, y responder con evidencia:

> ¿Las señales/lateralidades seleccionadas producen segmentaciones razonables en E01-T01?

Es una **validación del pipeline, no optimización**: no se modificó el algoritmo ni se hizo tuning. Resultados extraños se registran, cuantifican y documentan (no se corrigen).

## 2. Alcance

- **Atletas:** B0400 · B0371 · B0380
- **Condición/trial:** únicamente **E01-T01**
- **Técnicas:** únicamente S01–S05
- **Frecuencia:** cohorte 250 Hz
- **Procesado:** 15 C3D (3×5). No se procesaron E02/E03/E04, T02, otros trials ni otros atletas.

## 3. Configuraciones utilizadas (Task 2)

| Athlete | Technique | Signal | Side | Status |
|---------|-----------|--------|------|--------|
| B0400 | S01 | RFIN | R | RECOMMENDED |
| B0400 | S02–S05 | RTOE | R | RECOMMENDED |
| B0371 | S01 | RFIN | R | NEEDS_VALIDATION |
| B0371 | S02–S05 | RTOE | R | RECOMMENDED |
| B0380 | S01 | RFIN | R | NEEDS_VALIDATION |
| B0380 | S02 | LTOE | L | RECOMMENDED |
| B0380 | S03–S05 | RTOE | R | RECOMMENDED |

Sin cambios respecto a Task 2.

## 4. Resultados de segmentación (por atleta × técnica)

| Athlete | Tech | Señal usada | Side | Rate | Candidatos | Aceptadas | Rechazadas | Review | QC OK/WARN/REV/INV | Estado |
|---|---|---|---|---|---|---|---|---|---|---|
| B0400 | S01 | RFIN | R | 250 | 10 | 5 | 4 | 1 | 5/0/0/0 | OBSERVED_OK |
| B0400 | S02 | RTOE | R | 250 | 4 | 3 | 1 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0400 | S03 | RTOE | R | 250 | 4 | 3 | 1 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0400 | S04 | RTOE | R | 250 | 6 | 3 | 3 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0400 | S05 | RTOE | R | 250 | 9 | 3 | 5 | 1 | 3/0/0/0 | OBSERVED_OK |
| B0371 | **S01** | **RTOE** | R | 250 | 8 | 2 | 5 | 1 | 2/0/0/0 | **REVIEW_REQUIRED** |
| B0371 | S02 | RTOE | R | 250 | 4 | 3 | 1 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0371 | S03 | RTOE | R | 250 | 10 | 3 | 7 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0371 | S04 | RTOE | R | 250 | 7 | 3 | 4 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0371 | S05 | RTOE | R | 250 | 3 | 3 | 0 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0380 | **S01** | **RTOE** | R | 250 | 6 | 2 | 3 | 1 | 2/0/0/0 | **REVIEW_REQUIRED** |
| B0380 | **S02** | **LTOE** | L | 250 | 6 | 4 | 2 | 0 | 4/0/0/0 | OBSERVED_OK |
| B0380 | S03 | RTOE | R | 250 | 5 | 3 | 2 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0380 | S04 | RTOE | R | 250 | 5 | 3 | 2 | 0 | 3/0/0/0 | OBSERVED_OK |
| B0380 | S05 | RTOE | R | 250 | 6 | 3 | 3 | 0 | 3/0/0/0 | OBSERVED_OK |

**Totales:** 93 eventos · **46 ejecuciones aceptadas** · 43 rechazadas · 4 review · QC 46/46 cubierto, 0 WARN/REVIEW/INVALID.

Estados descriptivos (solo en este informe/tabla): `OBSERVED_OK`, `REVIEW_REQUIRED`, `INSUFFICIENT_EVIDENCE`. Nada se alteró en el esquema global.

## 5. Features obtenidas (medias de ejecuciones aceptadas, observadas)

| Celda | dur (s) | t2peak (s) | vmax (m/s) | vmean (m/s) | amax (m/s²) | disp (m) | path (m) | hip ROM (°) | knee ROM (°) | ankle ROM (°) | SNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0400-S01 | 0.502 | — | 4.453 | — | — | — | — | — | — | — | 77.2 |
| B0400-S02 | 0.897 | — | 8.079 | — | — | — | — | — | — | — | 850.2 |
| B0400-S03 | 0.953 | — | 7.831 | — | — | — | — | — | — | — | 575.0 |
| B0400-S04 | 0.835 | — | 9.083 | — | — | — | — | — | — | — | 124.7 |
| B0400-S05 | 1.016 | — | 6.063 | — | — | — | — | — | — | — | 1726.8 |
| B0371-S01 | 0.624 | — | 3.997 | — | — | — | — | — | — | — | 260.8 |
| B0371-S02 | 0.940 | — | 10.338 | — | — | — | — | — | — | — | 803.8 |
| B0371-S03 | 0.712 | — | 10.366 | — | — | — | — | — | — | — | 754.2 |
| B0371-S04 | 0.997 | — | 10.855 | — | — | — | — | — | — | — | 571.8 |
| B0371-S05 | 1.009 | — | 9.036 | — | — | — | — | — | — | — | 340.8 |
| B0380-S01 | 0.390 | — | **0.290** | — | — | — | — | — | — | — | 16.9 |
| B0380-S02 | 0.362 | — | 1.945 | — | — | — | — | — | — | — | 28.6 |
| B0380-S03 | 0.996 | — | 7.505 | — | — | — | — | — | — | — | 560.1 |
| B0380-S04 | 1.032 | — | 8.314 | — | — | — | — | — | — | — | 115.6 |
| B0380-S05 | 0.816 | — | 8.292 | — | — | — | — | — | — | — | 269.1 |

Detalle completo (t2peak, vmean, amax, displacement, path_length y ROM por lado `joints_side`) en `phase_1_8f_task3_golden_path.csv` y `phase_1_8f_task3_summary.csv`. **vmax/vmean/amax/path_length son `REQUIRES_NORMALIZATION` (Fase 1.8B): solo se reportan como observadas, no se usan para comparar rendimiento.**

## 6. S01 — caso especial

Comportamiento observado (evidencia del pipeline, sin auto-validación):

- **B0400-S01**: la configuración **RFIN fue usada** (snr 77.2 ≥ 8). 5 aceptadas, vmax medio 4.45 m/s, 4 rechazadas, 1 review. Segmentación plausible de un gyaku-zuki. `OBSERVED_OK`.
- **B0371-S01** y **B0380-S01**: **RFIN NO fue usada**. La puerta de señal del pipeline (`pick_best_signal`, `min_snr=8`) rechaza RFIN porque su SNR está por debajo (audit: 6.6 y 4.4, baseline alto ~1065/1184 mm/s) y cae al respaldo → **RTOE**. Con RTOE (el pie apenas participa en un puño) las "aceptadas" son de baja actividad (vmax medio **3.997** y **0.290** m/s), con 5/1 y 3/1 rejected/review. `REVIEW_REQUIRED`.

**Interpretación honesta:** en los dos casos `NEEDS_VALIDATION`, la señal configurada no puede ni siquiera ser usada por el propio gate del pipeline; la validación adicional (más trials/condiciones) sigue pendiente. **No se cambiaron** RFIN→LFIN, thresholds ni estados. No se declara S01 validado.

## 7. B0380-S02 — caso especial

El pipeline usó **LTOE / lado L** (confirmado; `movement_side=L`, `signal_used=LTOE`, SNR 28.6). 4 ejecuciones aceptadas, 2 rechazadas, vmax medio 1.945 m/s. La hipótesis de la auditoría (RTOE no segmentable por baseline alto) queda validada por comportamiento: con LTOE la segmentación produce ejecuciones plausibles y limpias. `OBSERVED_OK`. **No se usó RTOE** aunque tenga mayor vmax.

## 8. Comparación descriptiva con B0367 / B0377 (E01-T01, ejecuciones aceptadas)

| Technical | B0367 (200 Hz) | B0377 (250 Hz) | B0400 (250 Hz) | B0371 (250 Hz) | B0380 (250 Hz) |
|---|---|---|---|---|---|
| S01 | 3 | 1 | 5 | 2 (RTOE fallback) | 2 (RTOE fallback) |
| S02 | 3 | 3 | 3 | 3 | 4 (LTOE/L) |
| S03 | 3 | 3 | 3 | 3 | 3 |
| S04 | 3 | 3 | 3 | 3 | 3 |
| S05 | 3 | 3 | 3 | 3 | 3 |

Solo descriptivo: no hay "mejor/peor", no hay ranking. Los tres nuevos producen 3 aceptadas en la mayoría de patadas, con dos casos S01 a revisión y B0380-S02 usando lateralidad izquierda. B0367 es 200 Hz (no normalizado), así que las magnitudes no son directamente comparables entre cohortes.

## 9. Casos problemáticos (documentados, no corregidos)

1. **B0371-S01 y B0380-S01**: desviación de señal configurada (RFIN) → RTOE por el gate `min_snr=8`; ejecuciones aceptadas de baja actividad (vmax 0.29–4.0 m/s). Anomalía principal de la tarea.
2. **B0380-S01**: las 2 aceptadas con RTOE tienen vmax medio de **0.29 m/s** (movimiento casi nulo) — detecciones espurias de bajo nivel, coherentes con un pinch azules sin endpoint de pie.
3. **B0400-S05**: 9 candidatos (5 rechazados) — señal con múltiples bandas de actividad; 3 aceptadas salen con separación estable. Se deja documentado en las figuras.
4. **B0400-S01**: 5 aceptadas (vs 3 habituales) con 1 review — revisar en la figura.
5. Ningún WARN/REVIEW/INVALID en QC (`qc_*` acumulado = 0 en los 46 casos; cobertura 100 %).

## 10. Estado de validación

| Estado | Celdas |
|---|---|
| OBSERVED_OK | 13 (todas las patadas de B0400/B0371/B0380 + S01-B0400) |
| REVIEW_REQUIRED | 2 (B0371-S01, B0380-S01) |
| INSUFFICIENT_EVIDENCE | 0 |

15/15 celdas con datos (los 15 archivos existen). 13/15 arrojan segmentación plausible; 2/15 (los S01 con NEEDS_VALIDATION) requieren revisión.

## 11. Limitaciones

- Solo E01-T01; no E01-T02, E02, E03/E04.
- 3 atletas × 5 técnicas, cohorte 250 Hz únicamente.
- No se resolvió 200 vs 250 Hz; no hay normalización.
- No se validó comparabilidad poblacional; sin ML.
- No se tocó el Data Mart (sigue 18 filas, sin B04xx).
- Los resultados con features `REQUIRES_NORMALIZATION` no permiten conclusiones de rendimiento.

## 12. Próximo paso

No se decide automáticamente una Siguiente tarea. Esta evidencia queda lista para revisión (en particular, qué hacer con el gate `min_snr=8` en S01 y si hay que validar S01 con RFIN mediante un umbral dedicado). Se detiene aquí.

## Tests

`pytest tests -q` → 102 previos + ~11 nuevos de Task 3. Ver `tests/test_pipeline.py` sección 16.

## Reproducibilidad

```bash
.venv\Scripts\python scripts\10_phase_1_8f_task3_golden_path.py
```