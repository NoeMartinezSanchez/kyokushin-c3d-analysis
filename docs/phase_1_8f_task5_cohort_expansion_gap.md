# FASE 1.8F — Task 5 (suspendida): Brechas para la expansión de cohorte 250 Hz

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-20
**Estado:** ⏸ **ML v0 detenido por decisión.** Antes de construir el primer dataset ML hay que **expandir de forma controlada** la cohorte procesada. Este documento registra **qué falta** para pasar de los 5 atletas actualmente validados a la **cohorte 250 Hz completa**, con cifras verificadas del inventario (FASE 1.8E) y sin modificar el Data Mart.

---

## 1. Contexto

La Task 5 original (construir `output/ml_v0/`) quedó **suspendida** tras auditar su única fuente (Data Mart): el Mart contiene **18 filas de validación** (B0367 200 Hz + B0377 × S02/S03/S05, sin S04), un subconjunto de validación **no representativo** de la cohorte. Mezclar los 3 atletas del Golden Path con B0377 para "fabricar" un dataset ML (4 atletas, algunas técnicas con 0 filas) no sería un dataset representativo. **No se modifica el Data Mart ni se construye el dataset v0 en esta tarea.**

## 2. Estado actual (los 5 atletas validados)

| Atleta | Frecuencia | Config | Estado | Ejecuciones procesadas | Fuente |
|---|---|---|---|---|---|
| B0367 | 200 Hz | validado | baseline histórico | 26 (QC 26/26) | `output/` (raíz) |
| B0377 | 250 Hz | provisional | 2.º atleta de generalización | 13 controladas (S02/S03/S05 golden) | `output/B0377/` |
| B0400 | 250 Hz | provisional (Task 2) | nuevo | Golden Path 46 totales | `output/scaling_validation/` |
| B0371 | 250 Hz | provisional (Task 2) | nuevo | (incl. 37 filas S02–S05) | `output/scaling_validation/` |
| B0380 | 250 Hz | provisional (Task 2) | nuevo | | `output/scaling_validation/` |

Nota: B0367 es 200 Hz (baseline histórico); los otros 4 son 250 Hz.

## 3. Cohorte objetivo — números verificados (inventario 1.8E)

| Concepto | Valor |
|---|---|
| Atletas totales | **37** |
| — cohorte **250 Hz** | **33** |
| — cohorte **200 Hz** | **4** (B0367 + Grupo C: B0368/B0369/B0370) |
| 250 Hz procesados hasta hoy | **4** (B0377, B0400, B0371, B0380) |
| 250 Hz **elegibles restantes sin procesar** | **29** |
| Grupos/revisiones pendientes | cohorte 200 Hz (Grupo C): decisión de frecuencia/normalización pendiente |

## 4. Matriz de brechas por atleta

Los **29 atletas restantes** (250 Hz, Grupo B, `NOT_CONFIGURED`, sin anomalías, con E01 y cobertura S01–S05) no tienen **ningún** artefacto del pipeline de expansión (verificado):

| Atleta | Config `<id>.yaml` | Auditoría señal/lateralidad | Golden Path E01-T01 | Técnicas (inventario) |
|---|---|---|---|---|
| B0372, B0373, B0374, B0375, B0376, B0378, B0379, B0381, B0382, B0383, B0384, B0385, B0386, B0387, B0388, B0389, B0391, B0392, B0393, B0394, B0395, B0396, B0398, B0399, B0401–B0405 | ❌ | ❌ | ❌ | S01–S05 presentes (inventario) |

Todos los 29 cubren S01–S05 en el inventario; **falta confirmar** la disponibilidad exacta de `E01-T01` por técnica y las condiciones E02/E03/E04 por atleta (depende de `file_inventory` por celda, no verificado ahora).

## 5. Qué falta por atleta (pipeline de incorporación, ya establecido en Task 1→3)

Para cada uno de los 29, replicar el flujo controlado ya probado en B0400/B0371/B0380:

1. **Auditoría de señal/lateralidad** (reutilizar `09_signal_laterality_audit.py` extendido a más atletas): métricas por atleta×técnica×señal (RFIN/LFIN/RTOE/LTOE/…), comparación bilateral, `phase_1_8f_signal_recommendations.csv`.
2. **Recomendación → Config** (`config/athletes/<id>.yaml`, esquema existente; `validation.status: provisional`).
3. **Golden Path controlado** (reutilizar `10_phase_1_8f_task3_golden_path.py`) → ejecuciones accepted/rejected/review, QC, figuras.
4. **Casos especiales por celda** (S01, baselines altos, lateralidades izquierdas) registrados como anomalías, sin corregir el algoritmo.

## 6. Bloqueos transversales

- **S01 (Task 4):** representación inconsistent entre atletas (RFIN no segmentable en B0371/B0380; LFIN no rescata; fallback RTOE débil). **Política adoptada:** S01 queda `NEEDS_VALIDATION` y **fuera del dataset ML**. En la expansión, S01 debe registrarse por celda (config RFIN + estado) pero no bloqueará la incorporación del atleta; su resolución (umbral robusto por atleta/más trials) es tarea dedicada posterior.
- **Lateralidad por técnica:** no es universal (B0380-S02→LTOE, B0377-S04→LTOE). Cada atleta se audita por celda; la lateralidad se registra, nunca se asume.
- **Frecuencia:** cohorte 200 Hz (B0368-B0370) fuera del universo ML v0; normalización temporal 200/250 Hz y decisión sobre el Grupo C siguen pendientes.
- **Contrato del Data Mart:** la expansión del Mart (cuando proceda) debe mantener el **esquema único** (`feature_version`, `comparability_*`, `primary_signal`, `movement_side`, `qc_status`, `source_dataset`), ampliada fila a fila de forma validada, no mezclando artefactos sueltos.
- **Volumen/ventanas:** la auditoría 1.8E tardó ~28 min para 37 atletas; el audit de señales y el golden path son por atleta (re-ejecutables). Para 29 atletas conviene procesar **por lotes** con checkpoint/reanudación similar al de 1.8E.

## 7. Secuencia propuesta (controlada, sin fabricar dataset)

1. **Lote 0 — Preparación (gap-closing):** script de auditoría multi-atleta (extiende `09`) para los 29: coverage por celda (E01-T01 × S01–S05), métricas de señal/lateralidad, recomendaciones. Salida: tabla de readiness por atleta.
2. **Lote 1 — Configs:** generar `config/athletes/<id>.yaml` provisionales (validación automática como `test_1_8f2_*`), lote a lote (p. ej. 5–8 atletas), sin modificar configs existentes.
3. **Lote 2 — Golden Path por lote:** ejecutar `10` por lote; registrar anomalías (S01, baselines, lateralidades) sin corregir algoritmo.
4. **Lote 3 — Revisión consolidada:** QC agregado, consistencia de representación por técnica, decisiones de `NEEDS_VALIDATION`.
5. **Lote 4 — Ampliación controlada del Data Mart** (tarea dedicada, contrato único) → recién entonces **retomar Task 5 (ML v0)** con una cohorte representativa.

## 8. Criterios de aceptación para reanudar ML v0

- ≥ los atletas 250 Hz con config provisional y golden path QC estable para S02–S05 (n esperado tras lote: 29 nuevos, sujeto a disponibilidad real de archivos por celda).
- Cobertura por técnica sin huecos estructurales desconocidos.
- S01 excluido del dataset ML; normalización/no-200 Hz documentada.
- No entrenar modelo todavía; Task 5 se reanuda como dataset construction + audit sobre el Mart expandido.

## 9. Restricciones de esta tarea

- **NO** se modificó `output/data_mart/athlete_execution_features.csv`.
- **NO** se construyó `output/ml_v0/`.
- **NO** se tocó `02`, `segmentation.yaml`, `config/athletes/*`, dashboard, históricos B0367/B0377.
- Sin ML, sin normalización, sin procesar C3D nuevos aquí.

## 10. Siguiente paso (no ejecutado aquí)

Iniciar la **expansión de cohorte** (Lote 0 del §7): auditar señales/lateralidad de los 29 atletas restantes y materializar la tabla de readiness, previa aprobación.