# FASE 1.8A — Configuración por atleta y por técnica

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-13
**Alcance:** parametrizar el pipeline por atleta (B0367 baseline + B0377) sin romper la reproducibilidad y preparando el camino al dashboard de 37 atletas y ML.
**Scripts:** `02_execution_segmentation.py` (motor parametrizado), `05_phase_1_8a_validation.py` (validación), `04_athlete_generalization_audit.py` (reutiliza ruta por atleta).
**Config:** `config/segmentation.yaml` (global) + `config/athletes/<id>.yaml` (por atleta).

---

## 1. Objetivo

Convertir las conclusiones de FASE 1.7 en una **arquitectura de configuración por atleta** con un único motor reutilizable, de modo que cualquier atleta nuevo pueda procesarse definiendo únicamente su configuración (señal, lateralidad, joints side, umbrales) y su ruta de datos — sin editar código Python.

```
MOTOR GENERAL (02)
     +
config/segmentation.yaml (global)
     +
config/athletes/<id>.yaml (por atleta y técnica)
     =
PIPELINE REUTILIZABLE (B0367 → 26 ejecuciones; B0377 → 13 controladas)
```

## 2. Arquitectura antes/después

**Antes (FASE 1.5-1.7):**
```
config global (segmentation.yaml) → pipeline
  · DATA_DIR fijo = B0367/
  · señales R*/RTOE en constantes del script
  · joins (ángulos) hardcodeados a lado derecho
  · salidas siempre a output/ raíz
  · sin selección de atleta
```

**Después (FASE 1.8A):**
```
config global + config athletes/<id>.yaml + config técnica
            ↓
set_active_athlete(id)  →  CFG + señales + joints + OUT_ATHLETE
            ↓
motor de segmentación (sin cambios de algoritmo)
            ↓
output/  (B0367, raíz)   |   output/<athlete_id>/  (B0377, subdir)
```

## 3. Parámetros generalizados (salieron del código → config)

| Parámetro | Antes | Ahora |
|-----------|-------|-------|
| Carpeta de datos | `01: DATA_DIR=B0367` | `data_dir_for(athlete)` (helper en `02`) |
| Señal primaria por técnica | constantes `PRIMARY_SIGNAL` | `config/athletes/<id> → techniques.*.signal` |
| Lado de articulaciones (joins/chain) | `R*` hardcoded | `techniques.*.joints_side` (R/L) |
| Lateralidad | implícita (siempre derecha) | `techniques.*.laterality` (per-técnica) |
| Umbrales (opcional) | solo globales | override por técnica (`thresholds`) |
| Rutas de salida | `output/` fijo | `OUT_ATHLETE` (`output/<id>/` para no-baseline) |
| rate/unidades/derivadas/Tarcza | — | se **leen del C3D** (`verify_units_from_c3d`), no se hardcodean |

## 4. Configuración B0367 (baseline)

`config/athletes/B0367.yaml`: reproduce exactamente el comportamiento histórico.

```
S01: RFIN (right, R); S02..S05: RTOE (right, R)
sampling_rate 200 (anotación; la fuente real es el C3D)
status: validated
```

## 5. Configuración B0377

`config/athletes/B0377.yaml` (evidencia de FASE 1.7, no inventada):

| Técnica | signal | laterality | joints_side | umbral |
|---------|--------|-----------|-------------|--------|
| S01 | RFIN | bilateral | R | NEEDS_VALIDATION (baseline alto ~599) |
| S02 | RTOE | right | R | global |
| S03 | RTOE | right | R | global |
| S04 | **LTOE** | **left** | **L** | global |
| S05 | RTOE | right | R | global |

sampling_rate 250 (anotación); status: provisional.

## 6. S01 — problema del baseline (B0377)

`output/phase_1_8a/s01_b0377_threshold_validation.csv`:

| atleta | mediana 1er seg | señal p50 | MAD | med+3MAD | picos robustos | aceptadas pipeline |
|--------|----------------|-----------|-----|----------|----------------|-------------------|
| B0377 T01 | 599 | 754 | 628 | 2638 | 3 | 1 |
| B0377 T02 | 791 | 723 | 573 | 2442 | 3 | 2 |
| B0367 T01 | 79 | 637 | 547 | 2277 | 3 | 3 |

**Diagnóstico:** la mediana del primer segundo en B0377 no es "reposo" (599 vs 78), por eso el umbral heredado fragmenta. Un umbral robusto de la señal completa (`median + 3·MAD`) separa 3 picos reales en ambos atletas. **No se modificó el algoritmo** (regla 1.8A): se documenta la evidencia y S01-B0377 se marca **NEEDS_VALIDATION** en la config. Recomendación futura: reemplazar `baseline = mediana(primer segundo)` por una estadística robusta de la señal en una fase de automatización de umbrales.

## 7. S04 — cambio RTOE → LTOE (B0377)

`output/phase_1_8a/s04_b0377_validation.csv` (S04-E01-T01):

| marker | baseline | vmax | SNR | aceptadas | rechazadas | review |
|--------|---------|------|-----|-----------|------------|--------|
| RTOE | 159 | 4289 | 27 | 2 | 6 | 1 |
| **LTOE** | **53** | **9324** | **173** | **3** | 3 | 1 |
| RANK | 315 | 4069 | 13 | 4 | 10 | 2 |
| LANK | 130 | 7366 | 56 | 3 | 5 | 0 |
| RHEE | 252 | 4529 | 18 | 4 | 14 | 0 |
| LHEE | 159 | 7601 | 48 | 3 | 4 | 0 |

**Conclusión:** LTOE es claramente superior para S04-B0377 (mayor vmax/SNR y 3 ejecuciones estables). Se adopta como **config oficial**. Las features articulares (`joints_side=L`) ahora usan `LHipAngles/LKneeAngles/LAnkleAngles` en esta técnica.

## 8. Validación B0367 (regresión)

`output/phase_1_8a/b0367_regression.csv` + `output/qc_summary.csv`:

- **26 ejecuciones** (coincide con baseline histórico 1.6.1/1.6.2).
- QC: 26/26, cobertura 100 %, 0 errores, mismos intervalos y señales.
- Outputs históricos en `output/` raíz **sin cambios** (salvo regeneración idéntica).

## 9. Validación B0377 (controlada)

`output/phase_1_8a/b0377_config_validation.csv` y `output/B0377/`:

| Técnica | trial | señal | aceptadas | nota |
|---------|-------|-------|-----------|------|
| S01 | E01-T01 | RFIN | 1 | pendiente umbral (NEEDS_VALIDATION) |
| S02 | E01-T01 | RTOE | 3 | ok |
| S03 | E01-T01 | RTOE | 3 | ok |
| S04 | E01-T01 | **LTOE** | 3 | ok (lateralidad izquierda) |
| S05 | E01-T01 | RTOE | 3 | ok |

**Total: 13 ejecuciones, QC 13/13, 250 Hz leído del C3D.** Las salidas viven en `output/B0377/` y no tocan el baseline.

## 10. Tests

`pytest tests -q` → **36 passed** (22 previos + 14 nuevos de FASE 1.8A), incluidos:
- `test_athlete_config_exists_b0367/b0377`
- `test_b0377_s04_uses_ltoe`
- `test_b0377_s01_threshold_is_explicit_or_needs_validation`
- `test_set_active_athlete_resolves_signals`
- `test_outputs_separated_by_athlete`
- `test_b0377_run_leaves_b0367_output_intact`
- `test_data_dir_resolved_by_athlete`
- `test_config_overrides_global`

## 11. Cambios realizados (archivos)

| Archivo | Acción |
|---------|--------|
| `config/athletes/B0367.yaml` | creado |
| `config/athletes/B0377.yaml` | creado |
| `scripts/02_execution_segmentation.py` | refactor: `data_dir_for`, `all_files/find_files(athlete)`, `load_config(athlete)`, `_deep_merge`, `_build_signals`, `JOINTS_SIDE`, `set_active_athlete`, `OUT_ATHLETE`, `main(athlete_id)`, `verify_units_from_c3d(athlete,out)` |
| `scripts/04_athlete_generalization_audit.py` | `all_files_for/discover_athletes` usan `data_dir_for` del motor |
| `scripts/05_phase_1_8a_validation.py` | creado (regresión + S01 + S04 + B0377) |
| `tests/test_pipeline.py` | +14 tests |
| `docs/phase_1_8a_config_audit.md` | creado |
| `docs/phase_1_8a_report.md` | este informe |

## 12. Archivos NO modificados

- `AGENTS.md` (regla intacta)
- `config/segmentation.yaml` (global sin cambios)
- `scripts/01_dataset_exploration.py` (no se tocó; la ruta se resuelve con helper local)
- Outputs históricos de `output/` raíz: regenerados de forma idéntica al baseline (26/26)
- `scripts/03_signal_validation.py` (no se tocó)
- Algoritmo de segmentación `segment_repetitions`, `quality_check`, `qc_summary`, `dtw_dist`, `normalize_` — intactos

## 13. Limitaciones

- **S01-B0377**: la señal RFIN existe y tiene 3 picos reales con umbral robusto, pero el umbral actual del método (mediana del primer segundo) no es válido para este atleta → queda **NEEDS_VALIDATION**. No se forzó un valor.
- **S04-B0377**: LTOE validada solo sobre E01-T01 (y configuración); debe confirmarse con T02 y E02 (pendiente).
- **250 vs 200 Hz**: **no normalizado todavía** (fuera de alcance de 1.8A). El rate se lee y se conserva por archivo.
- **B0377 no completo**: solo se procesó E01-T01 por técnica; E02/E03/E04 y T02 quedan para fase posterior.
- **No se hizo ML** ni clustering.

## 14. Recomendación FASE 1.8B

Con la evidencia, la recomendación **B) lograr configuración por atleta estable** ya está esencialmente hecha en 1.8A. Para **1.8B** recomendamos, en orden:

1. **Resolver S01-B0377**: implementar el umbral robusto (median+MAD) **de forma parametrizable** (sin cambiar el algoritmo por defecto de B0367) y validarlo sobre T01/T02 y E02.
2. **Ampliar cobertura B0377**: procesar E01-T02, E02, y validar S04-LTOE en T02/E02; sumar E03/E04 con manejo de rol/prefijo.
3. **Normalización temporal** (200 vs 250 Hz) si el dashboard requiere comparar entre atletas; documentar el criterio antes de implementar.
4. **Feature extraction generalizada**: confirmar ROM/velocidad angular de `joints_side` en B0377 (S04-L).

No incorporar un tercer atleta ni ML hasta cerrar 1 y 2.

## Resumen ejecutivo

- **Atleta procesado:** B0377 (39 C3D; sub-conjunto E01-T01 por técnica).
- **Señal por técnica:** S02/S03/S05 → RTOE; S04 → **LTOE**; S01 → RFIN (umbral pendiente).
- **Lateralidad:** derecha en S02/S03/S05, **izquierda en S04**, bilateral en S01 — por técnica, no global.
- **Baseline B0367:** intacto, 26 ejecuciones, QC 26/26.
- **B0377 controlado:** 13 ejecuciones con su config, QC 13/13.
- **Generalizabilidad:** el motor ahora procesa por atleta sin editar código; solo requiere `config/athletes/<id>.yaml` + ruta de datos.
- **Pendientes:** S01 umbral (NEEDS_VALIDATION), S04-LTOE en más trials, normalización 200/250 Hz, cobertura completa de B0377.