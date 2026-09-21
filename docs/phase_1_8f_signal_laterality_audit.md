# FASE 1.8F — Tarea 1: Auditoría de señal y lateralidad

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-20
**Script:** `scripts/09_signal_laterality_audit.py`
**Salidas:** `output/scaling_selection/phase_1_8f_signal_audit.csv`, `output/scaling_selection/phase_1_8f_signal_recommendations.csv`, `output/scaling_selection/signal_audit/*.png`, `phase_1_8f_signal_audit_log.txt`
**Universo:** B0400 · B0371 · B0380 × S01–S05 × E01-T01 × 250 Hz.

---

## 1. Objetivo

Determinar, **con evidencia de los C3D**, qué marker/señal y qué lateralidad son más apropiados para segmentar cada una de las cinco técnicas en los tres atletas seleccionados en la Tarea 0. Es una **auditoría previa a la creación de configuraciones**: NO se crean `config/athletes/*.yaml`, no se procesa la cohorte, no se toca el Data Mart ni el dashboard, no se ejecuta ML ni se normaliza frecuencia.

La decisión es **multicriterio y determinista** (no "mayor vmax" ni "mayor SNR" a secas): aptitud real de segmentación sobre la señal suavizada con los parámetros activos del pipeline, coherencia lateral por pares anatómicos, establecimiento de candidatos accepted/rejected/review y separación temporal.

## 2. Universo

- Atletas: **B0400** (40 C3D), **B0371** (30 C3D), **B0380** (41 C3D).
- Condición/trial: **E01-T01** únicamente (golden path comparable).
- Técnicas: **S01–S05**. No se analizaron E02/E03/E04, T02 ni S06.
- Los 15 archivos (`3×5`) existen y son **250 Hz** (verificado en `file_inventory.csv`); no hubo sustituciones de trial.

## 3. Metodología

Se reutilizó lógica existente (no se duplicó):

- `02_get_signal` → velocidad 3D del marcador (mm/s).
- `02_segment_repetitions` → eventos `accepted/rejected/review` con el método de bandas de actividad y la config activa (`config/segmentation.yaml`, sin cambios).
- `02_smooth` de `01` → preprocesado idéntico al pipeline.
- `04_lateral_pair_scores` → comparación bilateral L vs R por par anatómico (misma función usada en FASE 1.7 para B0377-S04).
- Regla de idoneidad existente (`04/fase4_signals`): `snr ≥ 8.0 ∧ baseline < 100 mm/s ∧ accepted ≥ 3`.

Métricas por `atleta × técnica × señal candidata` (`phase_1_8f_signal_audit.csv`, 78 filas):
- S01 → `RFIN`, `LFIN`; S02–S05 → `RTOE`, `LTOE`, `RANK`, `LANK`, `RHEE`, `LHEE`.
- `baseline_median`, `baseline_mad` (MAD del primer segundo, complemento diagnóstico robusto; el pipeline no lo usa), `threshold` = `baseline + activity_frac·(vmax−baseline)`, `vmax`, `snr`, `candidate_count`, `accepted/rejected/review_count`, `candidate_separation_s`, `quality_status`, `suitability_status`, `notes`.

Decisión (`phase_1_8f_signal_recommendations.csv`, 15 filas) por reglas transparentes:
1. Para patadas: se elige el lado con señal **apta** (no sólo dominancia de vmax); si ambos lados son aptos, el de mayor `snr`; si ninguno es apto, se marca `NEEDS_VALIDATION` del toe lateralmente dominante o `AMBIGUOUS` si la evidencia es simétrica.
2. Para S01: se elige la mano `RFIN/LFIN` con mejor aptitud (apta primero, luego `snr`, luego baseline); si ninguna es apta → `NEEDS_VALIDATION` (precedente B0377-S01).

Estados: `RECOMMENDED`, `NEEDS_VALIDATION`, `AMBIGUOUS`, `INSUFFICIENT_DATA`. `NEEDS_VALIDATION` coincide con la taxonomía ya usada en `config/athletes/*.yaml`.

## 4. Resultados por atleta

### B0400
- **S01 → RFIN (R)** · `RECOMMENDED`: baseline 69 mm/s, snr 75.4, 5/10 ejecuciones aceptadas, separación 1.45 s. LFIN apto igualmente (baseline 141). Como ambos puños se mueven en Gyaku-Zuki el ratio R/L≈1.04 no decanta por vmax; la aptitud de segmentación de RFIN la hace la señal útil.
- **S02 → RTOE (R)** · `RECOMMENDED`: snr 802.7, baseline 9, 3/4 acc. Lateraliad R clara (toe/ankle/heel ratio 3.6/2.7/2.7). Alternativa RANK.
- **S03 → RTOE (R)** · `RECOMMENDED`: snr 565.7, baseline 14, 3/4 acc. Ratio 3.7/2.8/2.8.
- **S04 → RTOE (R)** · `RECOMMENDED`: snr 119.5, baseline 78, 3/6 acc. Ratio 5.2/3.9/3.8.
- **S05 → RTOE (R)** · `RECOMMENDED`: snr 1750.0, baseline 3, 3/9 acc. Ratio 2.8/2.7/2.7.

### B0371
- **S01 → RFIN (R)** · `NEEDS_VALIDATION`: baseline muy alto **1065 mm/s**, snr 6.6 (mismo patrón que B0377-S01). Se mantiene RFIN como candidato primario, pendiente de umbral robusto con más trials/condiciones. Alternativa LFIN.
- **S02 → RTOE (R)** · `RECOMMENDED`: snr 791.1, baseline 13, 3/4 acc. Ratio 3.8/3.3/3.3.
- **S03 → RTOE (R)** · `RECOMMENDED`: snr 722.2, baseline 14, 3/10 acc.
- **S04 → RTOE (R)** · `RECOMMENDED`: snr 551.2, baseline 19, 3/7 acc. (En este atleta S04 mantiene RTOE; no reproduce el LTOE de B0377.)
- **S05 → RTOE (R)** · `RECOMMENDED`: snr 344.7, baseline 27, 3/3 acc.

### B0380
- **S01 → RFIN (R)** · `NEEDS_VALIDATION`: baseline **1184 mm/s**, snr 4.4 (patrón B0377-S01). Candidato primario RFIN pendiente de umbral.
- **S02 → LTOE (L)** · `RECOMMENDED`: **hallazgo de lateralidad nueva**. Aunque RTOE tiene mayor vmax (9.01 m/s vs 2.58 m/s, ratio 3.49), su baseline es **1857 mm/s** (actividad casi continua, no segmentable: snr 4.8) mientras **LTOE es limpia** (baseline 90, snr 28.3, 4/6 acc, separación 0.99 s). La señal útil para segmentación es la izquierda.
- **S03 → RTOE (R)** · `RECOMMENDED`: snr 567.3, baseline 14, 3/5 acc.
- **S04 → RTOE (R)** · `RECOMMENDED`: snr 116.4, baseline 73, 3/5 acc.
- **S05 → RTOE (R)** · `RECOMMENDED`: snr 229.0, baseline 37, 3/6 acc.

## 5. Comparación entre atletas

| Técnica | B0400 | B0371 | B0380 |
|---|---|---|---|
| S01 | RFIN (R) · RECOMMENDED | RFIN (R) · NEEDS_VALIDATION | RFIN (R) · NEEDS_VALIDATION |
| S02 | RTOE (R) · RECOMMENDED | RTOE (R) · RECOMMENDED | **LTOE (L) · RECOMMENDED** |
| S03 | RTOE (R) · RECOMMENDED | RTOE (R) · RECOMMENDED | RTOE (R) · RECOMMENDED |
| S04 | RTOE (R) · RECOMMENDED | RTOE (R) · RECOMMENDED | RTOE (R) · RECOMMENDED |
| S05 | RTOE (R) · RECOMMENDED | RTOE (R) · RECOMMENDED | RTOE (R) · RECOMMENDED |

Las configuraciones **no** son idénticas entre atletas: S01 difiere en estado y S02-B0380 en lateralidad.

## 6. Hallazgos

1. **S01 sigue siendo la técnica problemática** en 2 de 3 atletas (B0371 y B0380, baseline >1000 mm/s, patrón B0377-S01). Solo B0400 tiene un S01 limpio (RFIN usable). Esto **confirma** que S01 no debe asumirse resuelto y que B0371/B0380 requieren umbral robusto o una segunda condición/trial para validar.
2. **S04 NO vuelve a mostrar lateralidad variable**: en los tres atletas S04→RTOE con evidencia clara. El LTOE de B0377-S04 **no** se generaliza aquí.
3. **S02/S03/S05 mantienen RTOE** en B0400 y B0371.
4. **Nueva lateralidad por técnica (B0380-S02→L IZQUIERDA)**: el hallazgo esporádico de B0377 se repite en otro atleta pero en **otra técnica**. Refuerza la regla de que la lateralidad y la señal deben validarse **por atleta × técnica**, nunca asumirse universales.
5. Caso ambigüedad: ninguno `AMBIGUOUS` ni `INSUFFICIENT_DATA` en este universo (todos los archivos y señales existen).
6. Variabilidad de baseline: B0380-S02-RTOE es un ejemplo extremo de **dominancia de vmax engañosa** (ratio 3.49) con señal inservible; la regla multicriterio la descartó correctamente.

## 7. Decisiones que todavía NO se toman

Esta auditoría:
- **NO crea configuraciones** (`config/athletes/B0400|B0371|B0380.yaml` no se generan ni modifican);
- NO incorpora a los tres atletas al Data Mart;
- NO valida toda la cobertura del atleta (solo golden path E01-T01);
- NO demuestra generalización completa del pipeline;
- NO compara rendimiento deportivo entre atletas.

## Trazabilidad

Cada recomendación es rastreable hasta el C3D:
`atletas/<id>/<fecha>-<id>-<S0X>/<fecha>-<id>-<S0X>-E01-T01.c3d` → `file_inventory.csv` → señal → métricas del audit (baseline/MAD/vmax/snr/eventos/separación) → comparación bilateral `lateral_pair_scores` → `phase_1_8f_signal_recommendations.csv` (con `evidence_summary` en cada fila).

## Reproducibilidad

```bash
.venv\Scripts\python scripts\09_signal_laterality_audit.py
```

Re-ejecutar produce el mismo resultado (cubierto por test de determinismo). Unidades en mm/s (igual que el pipeline); preprocesado con la config activa (ventana ~60 ms a 250 Hz vs ~75 ms a 200 Hz — sin normalizar, documentado).

## Tests

`pytest tests -q` → 83 previos + 10 nuevos (Fase 1.8F Tarea 1), todos pasando.