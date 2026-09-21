# FASE 1.8F — Tarea 0: Selección automática de 3 atletas candidatos

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-20
**Script:** `scripts/08_select_1_8f_candidates.py`
**Salidas:** `output/scaling_selection/phase_1_8f_candidate_selection.csv`, `output/scaling_selection/phase_1_8f_log.txt`
**Fuente de verdad:** tablas de inventario de FASE 1.8E (`output/athlete_inventory/`)

---

## 1. Objetivo

Seleccionar, de forma **reproducible y auditable**, 3 atletas para una **prueba controlada de generalización del pipeline** (C3D → repeticiones → segmentación → features → 1 fila = 1 ejecución).

La selección es **estructural**: representa la **variabilidad técnica del dataset** que puede afectar la robustez del pipeline. **NO** es una selección por rendimiento deportivo, velocidad, número de ejecuciones, edad, grado ni ranking. Ningún atleta se califica aquí como "mejor"/"peor".

Esta tarea **solo selecciona y justifica**: no procesa C3D nuevos, no segmenta, no crea configuraciones definitivas, no modifica el Data Mart ni el dashboard, no ejecuta ML.

## 2. Universo analizado

Cifras obtenidas de los inventarios de FASE 1.8E (`output/athlete_inventory/`), no estimadas:

| Concepto | Valor |
|---|---|
| Atletas totales (inventario 1.8E) | **37** |
| — de los cuales a **250 Hz** | **33** |
| — a **200 Hz (cohorte 200 Hz: B0367–B0370)** | **4** |
| — excluidos por ser **B0367/B0377** (yap procesados) | **2** |
| — excluidos por anomalías estructurales | 0 |
| — excluidos por no tener E01 | 0 |
| — excluidos por no cubrir S01–S05 | 1 (B0370, sin S05; ya excluido por 200 Hz) |
| **Candidatos elegibles** | **32** |

Elegibles = Grupo B de `scaling_readiness` (33 incl. B0377), 250 Hz, `NOT_CONFIGURED`, sin anomalías, con E01 y cobertura S01–S05, excluyendo B0367 y B0377 → **32 atletas**.

## 3. Criterios de selección

Solo criterios estructurales, en orden de aplicación:

1. **Frecuencia:** 250 Hz como requisito de inclusión (no diferenciador, todos los elegibles lo cumplen).
2. **Exclusiones fijas:** B0367 (baseline protegido) y B0377 (2.º atleta de generalización ya usado). Cohorte 200 Hz (Grupo C) fuera del universo.
3. **Anomalías:** ningún elegible con `anomaly_flags` no vacío.
4. **Cobertura:** presencia de E01 y de las técnicas S01–S05 (verificado con `coverage_matrix.csv`).
5. **Estado de configuración:** preferir `NOT_CONFIGURED` (todos los elegibles lo cumplen).
6. **Estructura de variables derivadas** (de `derived_variable_inventory.csv`): diferenciar patrones de derivadas dentro de la cohorte 250 Hz.
7. **Variabilidad de markers/estructura punto** (de `athlete_inventory.csv`, rango `n_points_min/max`, y `marker_availability.csv`): probar que el pipeline no depende de una configuración particular de marcadores.
8. **Diversidad:** penalizar candidatos idénticos entre sí; cubrir combinaciones estructurales distintas dentro del Grupo B.
9. **Completitud:** preferir cobertura amplia, pero **no** seleccionar simplemente a los tres con más archivos.

No se usa ningún clúster/ML: la selección es exhaustiva y **determinista** (búsqueda sobre el cojunto de tríos que maximiza la suma de distancias estructurales por pares; empates resueltos por el id menor).

## 4. Los tres candidatos

> Nota de terminología: `rank_internal` (1, 2, 3) es un **orden interno de selección técnica**, no un ranking deportivo.

### B0400 — [derived_structure_variant]

- **Estructura que representa:** única atleta elegible con variable derivadas atípicas en la cohorte 250 Hz: `E01/E02 = 48|74` (filas con 48 y con 74 derivadas) en lugar del patrón modal 74; también único rango de puntos distinto (`148–382` vs `191–382`).
- **Cobertura:** 40 C3D, S01–S05, E01–E04, T01–T02; `signal_markers=7/7`.
- **Qué lo diferencia:** no comparte ni la estructura derivada estándar ni el rango de puntos estándar del resto del Grupo B.
- **Aporte a generalización:** prueba que el pipeline no dependa accidentalmente del esquema de 74 derivadas ni de una cobertura de marcadores particular.

### B0371 — [condition_gap_variant]

- **Estructura que representa:** el único elegible **sin la condición E04** (cubre E01, E02, E03); también es el de menor nº de archivos entre los tres (30).
- **Cobertura:** 30 C3D, S01–S05, E01/E02/E03, T01–T02; `signal_markers=7/7`.
- **Qué lo diferencia:** presenta un hueco de condición que otros atletas no tienen (los también con gap de condición son B0382/B0383, ambos sin E03; B0371 es el sin-E04).
- **Aporte a generalización:** prueba que el pipeline tolera cobertura parcial de condiciones sin romper el contrato ni asumir que una condición siempre existe.

### B0380 — [extra_technique_variant]

- **Estructura que representa:** el único elegible con una técnica adicional fuera de S01–S05 (`S06` presente además de S01–S05).
- **Cobertura:** 41 C3D, S01–S06, E01–E04, T01–T02; `signal_markers=7/7`.
- **Qué lo diferencia:** contiene una etiqueta de técnica no contemplada por el pipeline (S06).
- **Aporte a generalización:** prueba que el pipeline no falla ni sobreinterpreta técnicas no esperadas (robustez ante etiquetas desconocidas).

### Por qué estos tres y no otros

Existen otras variantes estructurales en el Grupo B (p. ej. B0382/B0383 sin E03; B0381/B0393 con T01–T05, 51/44 archivos). El procedimiento elige el trío que maximiza la diversidad de **tipos de variante distintos**, sin redundancia: dentro de una misma clase (p. ej. hueco de condición) solo entra un representante. B0400, B0371 y B0380 aportan tres tipos de variante **distintos y no solapados**.

## 5. Diversidad cubierta

| Dimensión | B0400 | B0371 | B0380 |
|---|---|---|---|
| Frecuencia | 250 Hz | 250 Hz | 250 Hz |
| Técnicas | S01–S05 | S01–S05 | S01–S06 |
| Condiciones | E01–E04 | E01, E02, E03 | E01–E04 |
| Trials | T01–T02 | T01–T02 | T01–T02 |
| Estructura derivada | `48\|74/25` | `74/25` | `74/25` |
| Estructura markers | 7/7; puntos 148–382 | 7/7; puntos 191–382 | 7/7; puntos 191–382 |
| Rol dentro de la selección | variante de estructura derivada | variante de hueco de condición | variante de técnica extra |
| C3D | 40 | 30 | 41 |

## 6. Qué NO demuestra esta selección

Esta selección **NO demuestra**:

- rendimiento deportivo de los atletas;
- calidad técnica del karate / nivel de grado;
- superioridad o inferioridad biomecánica de unos sobre otros;
- representatividad estadística de toda la población (n=3 de una prueba controlada);
- que la cohorte 200 Hz sea procesable ni comparable (queda fuera por decisión de frecuencia).

Representa únicamente una **selección técnica y estructural** para evaluar la generalización del pipeline dentro de la cohorte 250 Hz.

## 7. Siguiente paso

El siguiente paso de FASE 1.8F será **validar señales/lateralidad** de estos atletas y posteriormente **procesar de forma controlada** B0400, B0371 y B0380 (configs explícitas por técnica, validación contra el golden path, contrato del Data Mart sin cambios).

**No se realiza ese paso todavía.**

## Reproducibilidad

```bash
.venv\Scripts\python scripts\08_select_1_8f_candidates.py
# Salidas: output/scaling_selection/phase_1_8f_candidate_selection.csv
#          output/scaling_selection/phase_1_8f_log.txt
```

Re-ejecutar produce exactamente la misma selección (verificación automática por test `test_1_8f_selection_deterministic`).

## Tests

`pytest tests -q` → 74 previos + nuevos de 1.8F (todos pasan), 0 errores.