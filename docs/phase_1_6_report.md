# FASE 1.6 — Control de calidad, trazabilidad y validación del pipeline

**Proyecto:** Karate Athlete Performance Intelligence
**Dataset:** Subconjunto local B0367 (26 C3D, 1 atleta)
**Fecha:** 2026-09-13
**Scripts:** `scripts/01_dataset_exploration.py`, `scripts/02_execution_segmentation.py`, `tests/test_pipeline.py`
**Config:** `config/segmentation.yaml`
**Salidas:** `output/executions_sample.csv`, `output/segmentation_events.csv`, `output/execution_quality.csv`, `output/qc_summary.csv`, `output/units_verified.csv`, `output/execution_features_sample.csv`

---

## 1. Problema inicial

El reporte de la Fase 1.5 declaraba "17 filas" en `executions_sample.csv`, pero el resumen enumeraba S01(3) + S02(3) + S04-E01(3) + S04-E02(3) + E04-atacante(3) + E04-defensivas(5) = **20**. La instrucción pedía investigar sin asumir cuál número es correcto.

## 2. Discrepancia 17 vs 20 — resolución

Investigado directamente desde el código y los archivos:

- **`executions_sample.csv` contiene 17 filas**, que son las ejecuciones **aceptadas del atleta B0367** (el participante del dataset).
- Las **3 "ejecuciones de E04-atacante"** que sumaban al "20" corresponden al **oponente B0368**, y van a un fichero distinto: `output/e04_defender_attacker_distances.csv` (3 filas).
- Por tanto: **17 (atleta B0367 en la tabla principal) + 3 (oponente B0368, fichero aparte) = 20**. Ambos números son correctos; el error fue **sumar dos niveles de agregación** en el texto del informe 1.5, sin distinguir "tabla de ejecuciones del atleta" de "análisis de interacción E04".

**Conclusión:** no se invalidó ninguna conclusión de Fase 1.5; sí se detectó un problema de **trazabilidad/documentación** (falta indicar explícitamente de qué fichero sale cada fila y qué sujeto representa). Corregido en esta fase.

## 3. Número real de ejecuciones (reproducible)

Conteo reproducido con el código instrumentado:

| Archivo | Señal | Eventos candidatos | Aceptados | Rechazados | Review |
|---------|-------|---:|---:|---:|---:|
| S01-E01-T01 | RFIN | 9 | 3 | 5 | 1 |
| S02-E01-T01 | RTOE | 3 | 3 | 0 | 0 |
| S04-E01-T01 | RTOE | 3 | 3 | 0 | 0 |
| S04-E02-T01 | RTOE | 6 | 3 | 3 | 0 |
| S04-E04-T01 (defensor B0367) | RTOE | 7 | 5 | 2 | 0 |
| **Total** | | **28** | **17** | **10** | **1** |

- **17 aceptadas** → exactamente las 17 filas de `executions_sample.csv` (verificado con test `test_events_csv_consistent_with_accepted`).
- **10 rechazadas**: `duration_too_short` (5, en S01 y E04), `activity_too_low` (2, en S01), `secondary_peak` (3, en S04-E02).
- **1 review**: S01 a los 8.005 s, razón `low_activity_uncertain`.

## 4. Razones de rechazo detalladas (solo las que produce el algoritmo real)

| Razón | Definición en el algoritmo | Casos |
|-------|------------------------------|-------|
| `activity_too_low` | pico de la banda < `review_peak_ratio` (0.30) × vmax global | 2 (S01: 0.72 s, 7.68 s) |
| `duration_too_short` | banda con duración < `min_dur_s` (0.15 s) | 3 (S01) + 2 (E04) |
| `secondary_peak` | banda absorbida por la anterior (gap < `min_gap_s` y pico menor) — retorno/recuperación | 3 (S04-E02) |
| `low_activity_uncertain` | pico en [0.30, 0.45) × vmax → requiere revisión humana | 1 (S01, 8.005 s) |

## 5. Evento no clasificado a ~8 s en S01 — investigación

El evento a **8.005 s** tiene pico de velocidad RFIN de 2194 mm/s = **32 % del vmax global** (6872). Investigado con múltiples marcadores:

- **Ambas manos** se mueven (LFIN ~1943, RFIN ~2194, RWRB ~1986, RWRA ~1971 mm/s) — no es un golpe con una sola extremidad.
- **Codos y hombros casi no se mueven** (RELB ~1073 mm/s vs 6800–7310 mm/s en los 3 puños reales).
- La **altura de la mano derecha** baja y sube (1386 → 1103 → 1552 mm).
- El torso (C7, TRXO) y las caderas (RHJC/LHJC) muestran solo movimiento leve (~300–400 mm/s).

**Interpretación:** movimiento de **reposicionamiento/reajuste postural y de guardia** tras el asalto — no una cuarta ejecución de Gyaku-Zuki. **No se puede determinar de forma 100 % automática** (el detector automático no distingue "reposicionamiento" de "técnica débil"), por lo que queda marcado como **`review`** con razón `low_activity_uncertain` en `segmentation_events.csv`, y la interpretación por inspección multi-marcador se documenta por separado en este informe.

## 6. Validación visual de la segmentación

Gráficas en `output/phase_validation/` (regeneradas con los mismos parámetros):

| Archivo | Detectados | Aceptados | Review | Observaciones de inspección |
|---------|---:|---:|---:|---|
| S01-E01-T01 | 9 | 3 | 1 | 3 puños (0.72–0.90, 2.06–2.60, 3.50–4.06, 4.94–5.50); picos menores de braceo en 0.72/7.68 rechazados; 8.005 s en review |
| S02-E01-T01 | 3 | 3 | 0 | 3 patadas limpias (~1.76–6.45 s) con pico dominante bien marcado |
| S04-E01-T01 | 3 | 3 | 0 | 3 Mawashi-Geri jodan (~1.66–6.40 s) |
| S04-E02-T01 | 6 | 3 | 0 | 3 patadas; 3 bandas secundarias (retorno tras golpear escudo) absorbidas como secondary_peak |
| S04-E04-T01 | 7 | 5 | 0 | 5 respuestas defensivas del defensor B0367; 2 rechazos por duración (8.885 s, 9.76 s) |

**Alcance de la validación:** validación exploratoria/técnica del pipeline, NO validación científica biomecánica.

## 7. Definición operacional de "una ejecución individual"

Documentada y usada por el pipeline:

```
start (fin del reposo) -> [preparación/chamber] -> aceleración -> pico de velocidad (extensión/impacto) -> desaceleración -> end (retorno al reposo)
```

- `start` = primer frame donde la velocidad suavizada del endpoint supera el nivel de actividad.
- `peak` = frame de velocidad máxima dentro de la banda (NO se asume que equivale al impacto).
- `end` = último frame donde la señal sigue sobre el nivel de actividad.
- `start` NO se considera el inicio biomecánico real del gesto (el "chamber" interno de cada ejecución puede empezar antes); se define operativamente por el método.

## 8. Señales primarias por técnica

| Técnica | Señal | Movimiento | Estado |
|---------|-------|------------|--------|
| S01 | RFIN (mano derecha) | punch | Validada (3/3) |
| S02 | RTOE | kick | Validada (3/3) |
| S03 | **TBD** | kick | Sin validar en esta submuestra |
| S04 | RTOE | kick | Validada (3+3+5) |
| S05 | **TBD** | kick | Sin validar en esta submuestra |

Centralizado en `config/segmentation.yaml`. S03 y S05 quedan como `TBD` (el código no inventa señal: si el config fuera `TBD` sin fallback, la SNR falla y el archivo se documenta).

## 9. Unidades y frecuencia — verificadas desde los C3D

`output/units_verified.csv` (leído directamente de cada C3D representativo):

| Parámetro | Valor verificado |
|-----------|------------------|
| sampling_rate_hz | **200.0** (todos) |
| position_unit | mm |
| angle_unit | deg |
| force_unit | N |
| moment_unit | Nmm |
| power_unit | W |
| analog_used | 0 (sin canales analógicos) |
| fp_used | 0 (sin plataformas de fuerza) |

**Discrepancia confirmada y documentada:** paper = 250 Hz; C3D = 200 Hz. NO se asume; se lee y se reporta la discrepancia.

## 10. Dominancia lateral

**Dependencia del lado derecho identificada en el pipeline:**
- `config/segmentation.yaml` fija RFIN (S01) y RTOE (S02–S05) como señales primarias — determinadas para **B0367**, cuya pierna dominante es la derecha.
- Los fallbacks (`FALLBACK_SIGNAL`) y los ángulos analizados (`joins`) también asumen el lado derecho (R*).

**Riesgo:** un atleta zurdo/dominante izquierdo sería mal segmentado (su técnica real no genera pico en RTOE).

**Estrategia propuesta (sin implementar — fuera de alcance de esta fase):**
1. Añadir a la config un campo por atleta: `dominant_side` (L/R) o `dominant_leg_marker_map` (`S01: LFIN`, `S02: LTOE`, ...).
2. En `pick_best_signal`, si la señal configurada para el lado derecho no tiene SNR suficiente, probar automáticamente el lado izquierdo y documentar el resultado con `[warn]`.
3. Para los 37 atletas: mapear `B0xxx → dominant_side` desde el registro de participantes (o inferir empíricamente por SNR intra-atleta tomando el lado con mayor SNR estable).

## 11. E04 — múltiples sujetos

Verificado y separado en el C3D:

- **B0367 = defensor** (115 marcadores, 0 variables derivadas; rol `defensor`, tipo `defensive_response`).
- **B0368 = atacante** (140 puntos = 115 marcadores + 25 ángulos derivados).

Separación de niveles:

| Nivel | Fichero | Contenido |
|-------|---------|-----------|
| `features individuales` | `executions_sample.csv` (filas E04, `role=defensor`) | vmax ~0.5–1.2 m/s (respuestas defensivas) |
| `interaction features` | `e04_defender_attacker_distances.csv` | min distance, distance at peak, relative velocity, toe→def center |

Test `test_features_finite` confirma que `executions_sample.csv` **NO contiene** las interaction features (`min_dist_m`, etc.). Las interacciones **no se mezclan** en el dataset de clasificación de técnicas.

## 12. DTW — revisión técnica

- **¿Funciona?** Sí, técnicamente (ver Fase 1.5 y `output/dtw_validation/`).
- **Señal:** velocidad del endpoint (RTOE/RFIN) **normalizada por ejecución** (z-score de la propia curva) → sin leakage entre trials.
- **Preprocessing:** suavizado Hanning (15 muestras) + z-score por ventana.
- **Comparación correcta:** misma técnica + misma condición + mismo atleta + trial distinto (S02-E01 T01 vs T02, etc.). **NO se comparan técnicas distintas.**
- **Interpretación:** las distancias DTW (~0.4–1.5) son estables y las ejecuciones de un mismo trial son muy repetibles (vmax de 3 patadas del mismo trial: 10.30/10.37/10.29 m/s).
- **Limitación:** la "similitud 1/(1+d)" necesita un baseline **entre-técnicas** antes de usarse como métrica absoluta; pendiente.

## 13. Features — dataset por niveles

`output/execution_features_sample.csv` (17 filas, 53 columnas) separa explícitamente:

- `metadata`: source_file, athlete_id, technique, condition, trial, role, execution_type, repetition, signal_used, units, segmentation método/versión.
- `segmentation`: start/peak/end frames y tiempos, duración.
- `temporal`: duración, tiempo a pico.
- `kinematic`: vmax/vmean/amax/amean, desplazamiento, path length, ROM endpoint.
- `joint`: ROM y velocidad angular de cadera/rodilla/tobillo/codo/hombro.
- `com`: velocidad y ROM del centro de masa (NaN en E04 defensor — no calculado, correcto).
- `coordination`: `coord_delay_proximal_distal_s` (**NO VALIDADO** en Fase 1.5; se mantiene marcado como tal).

`executions_sample.csv` NO se elimina (sigue siendo la tabla canónica con todas las columnas).

## 14. Control de calidad

`output/execution_quality.csv` cubre el 100 % de las 17 ejecuciones aceptadas. Resumen `output/qc_summary.csv`:

| Métrica | Valor |
|---------|-------|
| total_executions | 17 |
| valid_executions | 17 |
| warn_executions | 0 |
| review_executions | 0 |
| invalid_executions | 0 |
| cobertura | 100 % |
| events_total | 28 |
| events_accepted / rejected / review | 17 / 10 / 1 |

Comprobaciones incluidas por ejecución: NaN/inf en features, duraciones anómalas (fuera de [0.1, 2.0] s → WARN), vmax imposible (>20 m/s), NaN del endpoint (>5 %), frames/markers faltantes vía `get_time`.

## 15. Revisión del código

- `02_execution_segmentation.py` ahora: carga `config/segmentation.yaml`, emite eventos con status/reason (sin duplicar la lógica de segmentación), añade trazabilidad por fila, genera `qc_summary`, `units_verified`, `execution_features_sample`, `segmentation_events`.
- Eliminada una fuente de duplicación (las funciones de validación ahora reutilizan `segment_repetitions` con `return_events=False`).
- `01_dataset_exploration.py` no se modificó (no era necesario para esta fase).
- El pipeline corre desde cero: `python scripts/02_execution_segmentation.py`.

## 16. Tests

`tests/test_pipeline.py` — **11 tests, todos pasan** (`pytest tests -q`):

1. `test_c3d_can_open`
2. `test_sampling_rate` (200 Hz)
3. `test_metadata_parsed`
4. `test_intervals_valid` (start < peak < end)
5. `test_duration_positive`
6. `test_no_overlap_between_executions` (dentro de cada archivo)
7. `test_segment_repetitions_returns_events`
8. `test_features_finite`
9. `test_source_file_exists`
10. `test_unique_execution_id`
11. `test_events_csv_consistent_with_accepted` (aceptados == filas de executions_sample)

## 17. Trazabilidad — "¿De dónde salió exactamente esta fila?"

Cada fila de `executions_sample.csv` identifica sin ambigüedad:

```
source_file + technique + condition + trial + repetition + event_id
  -> start_frame / peak_frame / end_frame
  -> signal_used
  -> segmentation_method = activity_bands, segmentation_version = 1.6.0
  -> unidades (position=mm, velocity mm/s->m/s, angle=deg)
```

Ejemplo (fila real): `2017-01-31-B0367-S04-E02-T01.c3d | S04 | E02 | T01 | rep 1 | event_id 0 | RFIN/RTOE | start 1.525s | peak 1.770s | end 2.250s`.

## 18. Criterio de finalización

Demostrado:

```
C3D -> evento candidato (28) -> accepted/rejected/review (17/10/1)
    -> ejecución individual (17) -> features por nivel -> fila trazable en executions_sample.csv
```

Y respondido: "¿De dónde salió esta fila?" → fichero + frames + señal + método (sección 17).

## 19. Problemas restantes y recomendación para escalar

**Obligatorio antes de escalar (los 37 atletas):**
1. **Dominancia lateral** por atleta (resolver L/R, ver §10) — sin esto un zurdo se segmentaría mal.
2. Validar señales de **S03 y S05** (actualmente TBD).
3. Confirmar con revisión humana el **evento review** de S01 y generalizar el criterio de review a todo el dataset.

**Deseable pero puede esperar:**
1. Baseline DTW entre-técnicas para interpretar la similitud de forma absoluta.
2. Resolver o retirar definitivamente `coord_delay_proximal_distal_s`.
3. Ampliar tests (por ejemplo, tests por técnica, tests de robustez a NaN en el endpoint).

**Información adicional que el pipeline necesita para un nuevo atleta:**
- su ID/participante (para `athlete_prefix`),
- su **lado dominante** (o una sesión de calibración para inferirlo),
- confirmación de captura a 200 Hz y unidades estándar (el script ya lo verifica),
- las condiciones disponibles (E03 presente o no) para decidir si E04 se trata como técnica o como escenario defensivo.