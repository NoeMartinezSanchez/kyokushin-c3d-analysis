# FASE 1.8F — Tarea 2: Configuraciones explícitas B0400 / B0371 / B0380

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-20
**Archivos creados:** `config/athletes/B0400.yaml`, `config/athletes/B0371.yaml`, `config/athletes/B0380.yaml`
**Sin procesar:** esta tarea solo crea y valida configuración (no se corrió segmentación).

---

## 1. Objetivo

Convertir las recomendaciones de la auditoría de señal/lateralidad (FASE 1.8F Tarea 1) en configuraciones **YAML explícitas por atleta**, compatibles con el esquema existente (`config/athletes/<id>.yaml`, espejo de B0367/B0377) y consumidas por el mismo loader (`02_execution_segmentation.py` → `load_config` + `_build_signals`).

Se escribe **únicamente** configuración: no se procesan C3D, no se construye Data Mart, no se ejecuta ML, no se normaliza frecuencia y no se tocan `segmentation.yaml`, el dashboard ni los baselines.

## 2. Fuente de decisión

Las señales/lateralidades provienen **únicamente** de:

- `docs/phase_1_8f_signal_laterality_audit.md` (auditoría E01-T01)
- `output/scaling_selection/phase_1_8f_signal_recommendations.csv` (**fuente única de verdad**)

Ningún valor se eligió por intuición. La validación cruza el YAML contra ese CSV (test `test_1_8f2_matches_recommendations_csv`): si se cambia B0380-S02 de LTOE a RTOE, o B0371-S01 de `NEEDS_VALIDATION` a `RECOMMENDED`, sin actualizar el audit, el test falla.

## 3. Configuración final

| Athlete | Technique | Signal | Side | Status |
|---------|-----------|--------|------|--------|
| B0400 | S01 | RFIN | R | RECOMMENDED |
| B0400 | S02 | RTOE | R | RECOMMENDED |
| B0400 | S03 | RTOE | R | RECOMMENDED |
| B0400 | S04 | RTOE | R | RECOMMENDED |
| B0400 | S05 | RTOE | R | RECOMMENDED |
| B0371 | S01 | RFIN | R | NEEDS_VALIDATION |
| B0371 | S02 | RTOE | R | RECOMMENDED |
| B0371 | S03 | RTOE | R | RECOMMENDED |
| B0371 | S04 | RTOE | R | RECOMMENDED |
| B0371 | S05 | RTOE | R | RECOMMENDED |
| B0380 | S01 | RFIN | R | NEEDS_VALIDATION |
| B0380 | S02 | **LTOE** | **L** | RECOMMENDED |
| B0380 | S03 | RTOE | R | RECOMMENDED |
| B0380 | S04 | RTOE | R | RECOMMENDED |
| B0380 | S05 | RTOE | R | RECOMMENDED |

Mapeo de campos del esquema real (no se inventó una estructura nueva):
- `primary_signal` → `techniques.<T>.signal`
- `movement_side` → `techniques.<T>.laterality` + `techniques.<T>.joints_side` (consumida por el pipeline para features/signals)
- estado por técnica → `techniques.<T>.thresholds.status` (reutiliza el patrón de B0377-S01)
- `sampling_rate_hz` → `metadata.sampling_rate: 250.0`

## 4. Casos especiales

1. **B0380 S02 → LTOE (izquierda).** La auditoría mostró que RTOE tiene el **mayor vmax** (9.01 m/s vs 2.58, ratio 3.49) pero un **baseline de ~1857 mm/s** (actividad casi continua, snr 4.8, no segmentable), mientras **LTOE es limpia** (baseline 90, snr 28.3, 4/6 ejecuciones aceptadas). La decisión es multicriterio: señal útil para **segmentación**, no la de mayor valor numérico. `joints_side: L` acompaña a `signal: LTOE`.
2. **B0371 S01 → RFIN + NEEDS_VALIDATION.** RFIN existe y es el candidato primario, pero el baseline es alto (~1065 mm/s, snr 6.6), mismo patrón que B0377-S01. Se conserva RFIN y se marca `thresholds.status: NEEDS_VALIDATION`.
3. **B0380 S01 → RFIN + NEEDS_VALIDATION.** Igual que B0371: baseline alto (~1184 mm/s, snr 4.4). Se conserva RFIN pendiente de validar el umbral.

**`NEEDS_VALIDATION` no implica que la señal sea incorrecta**: significa que requiere validación adicional durante el Golden Path.

## 5. Limitaciones

- La auditoría se realizó **solo en E01-T01** de cada técnica.
- **No se ha validado la segmentación completa** de estos atletas (nada ha sido procesado).
- No se evaluó E01-T02.
- No se evaluó E02 (escudo).
- No se evaluaron E03/E04 (atacante/defensor; implican 2 sujetos).
- La generalización completa de **S01** no está resuelta (baselines altos en B0371/B0380).
- La diferencia **200 Hz vs 250 Hz** sigue sin normalizar.

## 6. Próximo paso

**FASE 1.8F Tarea 3 — Golden Path E01-T01 × S01–S05** para B0400, B0371 y B0380, que deberá evaluar:
- número de ejecuciones y QC;
- duración;
- señal primaria usada y lateralidad;
- consistencia respecto a B0367/B0377;
- comportamiento de los casos `NEEDS_VALIDATION` (S01 de B0371/B0380).

Esta tarea **se detiene aquí**; no se avanza a la Tarea 3 automáticamente.

## Tests

`pytest tests -q` → 95 previos + 7 nuevos de la Tarea 2 (total 102), todos pasando. Incluye carga real vía `load_config`/`_build_signals`, coincidencia exacta contra el audit CSV, y protección de B0367/B0377/Data Mart.