# FASE 1.8C — Athlete Data Mart

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-14
**Script:** `scripts/07_build_athlete_data_mart.py`
**Salidas:** `output/data_mart/athlete_execution_features.csv`, `output/data_mart/data_mart_summary.csv`
**Doc:** `docs/athlete_data_mart_dictionary.md`

---

## 1. Objetivo

Construir la primera capa de datos analítica estable (Athlete Data Mart) que será consumida por el futuro Dashboard y, posteriormente, por ML. El Data Mart es una **capa de consolidación**: lee outputs ya generados, NO relee C3D.

## 2. Fuente de datos

- Primaria: `output/feature_readiness_sample.csv` (18 ejecuciones golden path).
- Complemento: `output/executions_sample.csv` (B0367) y `output/B0377/executions_sample.csv` (metadata: sampling rate, señal primaria, event_id global).
- Config: `config/athletes/<id>.yaml` → `JOINTS_SIDE` para `movement_side`.

## 3. Alcance

| Dimensión | Valor |
|-----------|-------|
| Atletas | B0367, B0377 |
| Técnicas | S02, S03, S05 |
| Condición | E01 |
| Trial | T01 |
| Filas esperadas | 18 (3 por atleta × 3 técnicas) |

**Excluidos** (aunque existan archivos): S01, S04, E02/E03/E04, T02, otros atletas.

## 4. Arquitectura

```
C3D → Ingest/Audit → Segmentation → Execution Dataset → Feature Dataset
                                                                ↓
                                                   ATHLETE DATA MART  (esta fase)
                                                                ↓
                                                     Dashboard / ML (futuro)
```
El Dashboard leerá **solo** el Data Mart; nunca C3D ni archivos intermedios.

## 5. Esquema

`athlete_execution_features.csv` (39 columnas):

| Grupo | Columnas |
|-------|----------|
| Identidad | athlete_id, execution_id, technique, condition, trial, repetition |
| Adquisición | sampling_rate_hz, primary_signal, movement_side, event_id |
| Temporal | duration_s, time_to_peak_s |
| Cinemática | vmax, vmean, amax, displacement, path_length |
| Articulaciones | hip_rom, knee_rom, ankle_rom |
| Calidad | snr, qc_status, quality_flag |
| Comparabilidad | comparability_duration_s … comparability_snr (11 cols) |
| Metadatos | feature_version, segmentation_version, units_version, mart_version, source_dataset |

## 6-10. Ejecuciones / atletas / técnicas / condiciones / sampling rates

- Ejecuciones: **18** (9 B0367 + 9 B0377).
- Atletas: 2. Técnicas: 3 (S02/S03/S05). Condiciones: 1 (E01).
- Sampling rates: **B0367 = 200 Hz** (9), **B0377 = 250 Hz** (9). No modificados.

## 11. Features

Iguales al Feature Dataset (temporales, cinemática, articulaciones, calidad). Sin NaN (missing_feature_values = 0).

## 12. Estados de comparabilidad

| Feature | Estado |
|---------|--------|
| duration_s | DIRECTLY_COMPARABLE |
| time_to_peak_s | DIRECTLY_COMPARABLE (caveat: relativo al evento, no impacto) |
| vmax, vmean, amax, path_length | REQUIRES_NORMALIZATION |
| displacement | COMPARABLE_WITH_CAVEAT (magnitud geométrica; Hz solo indirecta vía frames start/end) |
| hip_rom, knee_rom, ankle_rom | COMPARABLE_WITH_CAVEAT (depende de movement_side) |
| snr | COMPARABLE_WITH_CAVEAT |

**No se normalizó 200/250 Hz** (fase futura); los raw quedan intactos.

## 13. Validaciones

Script `07` ejecuta validaciones automáticas — **todas OK**:
- execution_id único, sin duplicados.
- athlete/technique/condition/trial dentro del golden path.
- 18 filas; 9 por atleta; 3 por técnica-atleta.
- qc_status = accepted (18/18); quality_flag = OK (18/18).
- sin NaN en features mínimas.
- B0367 = 200 Hz; B0377 = 250 Hz.
- movement_side = R en golden path (derivado de joints_side, no hardcode).

## 14. Problemas encontrados

1. `feature_readiness_sample.csv` no expone `repetition` (está codificado en `execution_id`) → en `07` se extrae del execution_id. Sin impacto en datos.
2. Sin otros problemas: el esquema, trazabilidad y frecuencia salen coherentes.

## 15. Limitaciones

- Solo golden path (18 filas); cobertura completa de B0377 y de los 37 atletas queda para fases futuras.
- S01-B0377 y S04-B0377 siguen `NEEDS_VALIDATION` (no se tocaron en esta fase).
- No se construyó dashboard, no se hizo normalización temporal ni rankings.
- Los `comparability_*` son documentación de estado; no implican valores normalizados.

## 16. Siguiente paso recomendado

**FASE 1.8D — Dashboard prototipo** (reading-only sobre el Data Mart) en cuanto se conserve el schema estable:
1. Definir el contrato de lectura del Data Mart (columnas y unidades).
2. Prototipo de visualización: tablas/gráficas desde `athlete_execution_features.csv` (1 fila = 1 ejecución) sin lectura de C3D.
3. En paralelo, preparar la normalización temporal (200/250 Hz) y resolución de los NEEDS_VALIDATION antes de considerarlo definitivo para ML.

Alternativa si aún no se quiere dashboard: ampliar cobertura del Data Mart (agregar técnicas/condiciones/trials por atleta) manteniendo el mismo esquema.

---

## Resumen ejecutivo (formato solicitado)

- **A. Archivos creados:** `scripts/07_build_athlete_data_mart.py`, `output/data_mart/athlete_execution_features.csv`, `output/data_mart/data_mart_summary.csv`, `docs/athlete_data_mart_dictionary.md`, `docs/phase_1_8c_data_mart.md`.
- **B. Archivos modificados:** ninguno de los outputs históricos; solo se añadió la capa nueva. (Nota: sin cambios a `02`/`06`/`01`.)
- **C. Esquema final:** 39 columnas (identidad, adquisición, temporal, cinemática, articulaciones, calidad, comparabilidad, metadatos). *(Fase 1.8D: corregido — el "43" era un error de documentación; el CSV real tiene 39.)*
- **D. Nº de filas:** 18.
- **E. Nº de atletas:** 2.
- **F. Distribución por técnica:** S02: 6, S03: 6, S05: 6 (3 por atleta).
- **G. Distribución por sampling rate:** 200 Hz → 9 (B0367); 250 Hz → 9 (B0377).
- **H. Estados de comparabilidad:** 2 DIRECTLY_COMPARABLE, 4 COMPARABLE_WITH_CAVEAT (hip/knee/ankle ROM por joints_side + displacement geométrico) + snr caveat, 4 REQUIRES_NORMALIZATION (vmax/vmean/amax/path_length).
- **I. Validaciones:** OK (identidad/calidad/unidades/frecuencia/lado).
- **J. pytest:** 51 passed, 0 errores, exit=0 (regresión de la fase ejecutada al cierre).
- **K. Problemas:** solo el de `repetition` derivado de execution_id (resuelto).
- **L. Recomendación Fase 1.8D:** Dashboard prototipo leyendo solo el Data Mart, más normalización temporal y cierre de NEEDS_VALIDATION.