# Data Dictionary — Athlete Data Mart

**Fuente:** `output/data_mart/athlete_execution_features.csv`
**Versiones:** `feature_version=1.0`, `segmentation_version=1.8.0` (motor), `units_version=1.0`, `mart_version=1.0`
**Fuente de datos:** `feature_readiness_sample` (golden path S02/S03/S05-E01-T01, B0367+B0377)

> Convención: `comparability_*` usa las clases del plan 1.8C: `DIRECTLY_COMPARABLE`, `COMPARABLE_WITH_CAVEAT`, `REQUIRES_NORMALIZATION`, `NEEDS_VALIDATION`, `NOT_AVAILABLE`.

---

## Identidad

### athlete_id
- Descripción: identificador del atleta (código del dataset).
- Unidad: n/a (texto). Fuente: `feature_readiness_sample`.
- Golden path: `B0367`, `B0377`.

### execution_id
- Descripción: identificador estable y único de la ejecución: `{athlete}|{technique}|{condition}|{trial}|{repetition:03d}`.
- Unidad: n/a. Fuente: `feature_readiness_sample`.
- Observación: no puede haber duplicados (validado).

### technique / condition / trial / repetition
- Descripción: técnica (S0X), condición (E0X), trial (T0X) y repetición dentro del archivo.
- Golden path: S02/S03/S05 × E01 × T01, repetition 1..3.
- Fuente: `feature_readiness_sample` / `execution_id`.

## Adquisición

### sampling_rate_hz
- Descripción: frecuencia de muestreo del C3D (leída del archivo, no hardcodeada).
- Unidad: Hz. Fuente: `executions_sample`.
- Golden path: B0367=200, B0377=250. **No normalizado.**

### primary_signal
- Descripción: marcador usado como señal primaria de segmentación para esa técnica.
- Golden path: RTOE (S02/S03/S05). Fuente: `executions_sample.signal_used`.
- Observación: es un **proxy** de la ejecución; NO llamarlo "velocidad de impacto".

### movement_side
- Descripción: lado de las articulaciones derivado de `joints_side` de la config del atleta (R/L).
- Golden path: `R` en todos (S02/S03/S05). Fuente: config `config/athletes/<id>.yaml` (vía `JOINTS_SIDE`).
- Observación: NO hardcodeado; depende de la configuración por atleta/técnica.

### event_id
- Descripción: índice **global** del evento en `segmentation_events.csv` (trazabilidad).
- Unidad: n/a (entero). Fuente: `executions_sample`.

## Variables temporales

### duration_s
- Definición: `(end_frame - start_frame) / sampling_rate`.
- Unidad: segundos. Fuente: `feature_readiness_sample`.
- Comparabilidad: DIRECTLY_COMPARABLE (independiente de Hz).
- Nota: relativa al evento segmentado (start/end del detector).

### time_to_peak_s
- Definición: `(peak_frame - start_frame) / sampling_rate`.
- Unidad: segundos. Fuente: `feature_readiness_sample`.
- Comparabilidad: DIRECTLY_COMPARABLE con caveat — es **relativo al evento segmentado**, NO tiempo hasta impacto real.

## Cinemática (señal primaria)

### vmax
- Definición: máximo de la velocidad 3D de la señal primaria dentro de la ventana.
- Unidad: m/s (conversión mm/s × MM_IN_M). Fuente: `feature_readiness_sample`.
- Comparabilidad: REQUIRES_NORMALIZATION (200 vs 250 Hz).
- Tipo: proxy cinemático; NO "velocidad de impacto".

### vmean
- Definición: media de la velocidad de la señal primaria dentro de la ventana.
- Unidad: m/s. Comparabilidad: REQUIRES_NORMALIZATION.

### amax
- Definición: máximo del valor absoluto de la aceleración, derivada discreta de la velocidad suavizada (Hanning 15).
- Unidad: m/s² (mm/s² × MM_IN_M). Comparabilidad: REQUIRES_NORMALIZATION.
- Nota: derivada de velocidad con suavizado; documentado como derivación, no impacto.

### displacement
- Definición: distancia entre la posición inicial y final del endpoint en la ventana.
- Unidad: m (mm × MM_IN_M). Comparabilidad: COMPARABLE_WITH_CAVEAT (magnitud geométrica independiente de Hz; Hz solo influye indirectamente vía la elección de frames start/end del detector).
- Observación: conserva la implementación histórica del pipeline.

### path_length
- Definición: longitud acumulada de la trayectoria dentro de la ventana (suma de diferenciales).
- Unidad: m. Comparabilidad: REQUIRES_NORMALIZATION (sumatorio por frame; depende de la discretización/sampling).

## Articulaciones

### hip_rom / knee_rom / ankle_rom
- Definición: rango angular (max−min) del ángulo articulado del lado `movement_side`, dentro de la ventana.
- Unidad: **grados** (C3D ya en deg; no se convierte).
- Fuente: `feature_readiness_sample` (de `rom_{movement_side}Hip/Knee/AnkleAngles`).
- Comparabilidad: COMPARABLE_WITH_CAVEAT — comparables entre atletas solo si ambos usan el mismo `movement_side` (golden path: R).

## Calidad

### snr
- Definición: `vmax / (baseline + 1)` de la señal primaria (baseline = mediana primer segundo).
- Unidad: adimensional. Comparabilidad: COMPARABLE_WITH_CAVEAT (baseline en mm/s depende levemente de Hz).
- Nota: alto SNR no implica buena técnica.

### qc_status
- Definición: estado del evento en `segmentation_events.csv` (accepted/review/rejected).
- Golden path: accepted (18/18).
- Nota: NO se convierte review→accepted automáticamente.

### quality_flag
- Definición: bandera de QC por ejecución (OK/WARN) desde `execution_quality.csv`.
- Golden path: OK (18/18).

## Comparabilidad

### comparability_<feature>
- Descripción: estado de comparabilidad de cada feature (una columna por feature).
- Golden path:
  - duration_s, time_to_peak_s → DIRECTLY_COMPARABLE
  - vmax, vmean, amax, path_length → REQUIRES_NORMALIZATION
  - displacement → COMPARABLE_WITH_CAVEAT (magnitud geométrica; Hz solo indirecta vía frames)
  - hip_rom, knee_rom, ankle_rom, snr → COMPARABLE_WITH_CAVEAT
- No implica normalización aplicada; solo documenta el estado.

## Metadatos

### feature_version / segmentation_version / units_version / mart_version
- Descripción: versiones de la capa de features, del motor de segmentación, de las conversiones de unidades y del Data Mart.
- Valores: `1.0` / `1.8.0` (motor) / `1.0` / `1.0`.
- Siguen el patrón de versionado del proyecto (`SEGMENTATION_VERSION`).

### source_dataset
- Descripción: dataset origen del Data Mart.
- Valor: `feature_readiness_sample` (golden path).

---

## Notas de trazabilidad

- El `event_id` del Data Mart corresponde al **índice global** de `segmentation_events.csv` (corregido en FASE 1.8B-1; no es el índice posicional de la aceptada).
- Para "¿de dónde salió esta fila?": `athlete_id + technique + condition + trial + execution_id` → fichero C3D de origen + intervalo `start/peak/end` en frames + señal primaria + método `activity_bands`.