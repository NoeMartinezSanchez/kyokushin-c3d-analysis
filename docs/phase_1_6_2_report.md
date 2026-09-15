# FASE 1.6.2 — Reconciliación de conteos y cobertura del pipeline

**Proyecto:** Karate Athlete Performance Intelligence
**Subconjunto:** B0367 (solo este atleta)
**Fecha:** 2026-09-13
**Alcance:** solo reconciliar cifras; no se desarrollaron funcionalidades nuevas, no se modificó el algoritmo de segmentación, no se procesaron otros atletas.

---

## 1. Problema reportado

En `docs/phase_1_6_1_report.md`, la **sección 8** afirmaba que RTOE detecta 3 ejecuciones en cada uno de los 7 trials evaluados (S03: E01-T01/T02/E02-T01/T02 → 12; S05: E01-T01/T02/E02-T01 → 9; total 21), mientras la **sección 17** declaraba **26 ejecuciones aceptadas**. Además, la frase *"45 eventos (16 accepted → 26, rejected → 17, review → 2)"* parecía contener una inconsistencia.

Este informe determina — a partir de los C3D, la config, los scripts, CSV y logs — cuáles son los conteos reales y qué causa la diferencia.

---

## 2. Método

Se inspeccionó:

- `config/segmentation.yaml`
- `scripts/02_execution_segmentation.py` (lista `selected`)
- `scripts/03_signal_validation.py` (lista de trials evaluados)
- `output/executions_sample.csv`
- `output/segmentation_events.csv`
- `output/execution_quality.csv`
- `output/qc_summary.csv`
- `output/signal_validation/signal_validation_candidates.csv`
- Los 26 C3D de `B0367` (inventario)

---

## 3. Distinción clave: A) trials de VALIDACIÓN vs B) trials INTEGRADOS

Son dos conjuntos **distintos**:

- **A) Trials usados para VALIDAR la señal (`03_signal_validation.py`)** — evalúa candidatos en todos los trials de S03/S05 con E01/E02 disponibles:
  - S03 → E01-T01, E01-T02, E02-T01, E02-T02 (4 trials)
  - S05 → E01-T01, E01-T02, E02-T01 (3 trials)
  - **Total: 7 trials**, cada uno con RTOE detectando 3 ejecuciones → **21 detecciones** en la tabla de validación.

- **B) Trials INTEGRADOS al pipeline de extracción (`02_execution_segmentation.py`)** — el `selected` del `main()` es una **submuestra fija de 8 archivos**:
  ```
  S01-E01-T01, S02-E01-T01, S03-E01-T01,
  S04-E01-T01, S04-E02-T01, S04-E04-T01,
  S05-E01-T01, S05-E02-T01
  ```
  Es la lista histórica de archivos representativos de las fases 1.5/1.6/1.6.1. **No incluye ningún T02** de S03/S05 ni S03-E02.

**Conclusión:** las cifras 21 (sección 8) y 26 (sección 17) no son comparables directamente porque miden cosas distintas. **No existe un bug de conteo** en el algoritmo; la "diferencia" es: la sección 8 reporta la **validación experimental de la señal** (7 trials), mientras la sección 17 reporta el **pipeline integrado** (submuestra de 8 archivos).

---

## 4. Tabla de reconciliación

| technique | condition | trial | validation_script (03) | segmentation_pipeline (02) | accepted | rejected | review |
|-----------|-----------|-------|:---:|:---:|:---:|:---:|:---:|
| S03 | E01 | T01 | ✅ | ✅ | 3 | 0 | 0 |
| S03 | E01 | T02 | ✅ | ❌ | (3 en 03) | (1 en 03) | 0 |
| S03 | E02 | T01 | ✅ | ❌ | (3 en 03) | (1 en 03) | 0 |
| S03 | E02 | T02 | ✅ | ❌ | (3 en 03) | (3 en 03) | 0 |
| S03 | E04 | T01 | ❌ | ❌ | — | — | — |
| S03 | E04 | T02 | ❌ | ❌ | — | — | — |
| S05 | E01 | T01 | ✅ | ✅ | 3 | 2 | 0 |
| S05 | E01 | T02 | ✅ | ❌ | (3 en 03) | (3 en 03) | 0 |
| S05 | E02 | T01 | ✅ | ✅ | 3 | 5 | 1 |
| S05 | E02 | T02 | — (no existe en submuestra) | ❌ | — | — | — |
| S05 | E04 | T0x | — (no existe en submuestra) | ❌ | — | — | — |
| S01/S02/S04 (trials preexistentes) | E01/E02/E04 | T01 | ❌ | ✅ | 3/3/(5 E04) | (ver 1.6) | 1 |

**Interpretación de la tabla:**
- Las celdas **con paréntesis** son conteos de la **validación (03)**, no integradas al pipeline.
- Las celdas **sin paréntesis** son conteos actualmente **integrados** (en `executions_sample.csv` / `segmentation_events.csv`).
- Los trials S03-E01-T02, S03-E02-T01, S03-E02-T02, S05-E01-T02 están **validados por 03 pero NO integrados** en `02`.

---

## 5. Respuestas a las preguntas del objetivo

### 5.1 ¿Cuántos trials analizó `03_signal_validation.py`?
**7 trials** (S03: 4, S05: 3). Fuente: lista de `for technique, trials in [...]` y `signal_validation_candidates.csv` (contiene 7 combinaciones technique/condition/trial × 6 candidatos = 42 filas).

### 5.2 ¿Cuántos trials procesó `02_execution_segmentation.py`?
**8 archivos** fijos en `selected` (lista hardcodeada de `main()`). Fuente: `executions_sample.csv` (agrupa por technique/condition/trial → 8 grupos) y `segmentation_events.csv` (8 source_file).

### 5.3 ¿Cuántas ejecuciones por trial?
- **S03-E01-T01 (integrado):** 3 aceptadas, 0 rej, 0 review.
- **S05-E01-T01 (integrado):** 3 aceptadas, 2 rej, 0 review.
- **S05-E02-T01 (integrado):** 3 aceptadas, 5 rej, 1 review.
- **Trials validados pero no integrados:** S03-E01-T02 (3 acc), S03-E02-T01 (3 acc), S03-E02-T02 (3 acc), S05-E01-T02 (3 acc) — **validación 03**, no en pipeline.

### 5.4 ¿Aceptadas/rechazadas/review por trial?
Ver tabla de la sección 4. En el **pipeline integrado** (8 archivos): 26 accepted / 17 rejected / 2 review, lo que coincide con `qc_summary.csv` (events_total=45).

### 5.5 ¿Está el pipeline limitado intencionalmente a T01 / submuestra?
**Sí.** La lista `selected` es una muestra fija de 8 archivos representativos heredada de Fase 1.5 (antes solo S01/S02/S04), a la que en 1.6.1 se añadieron S03-E01-T01, S05-E01-T01 y S05-E02-T01. **No se procesa el conjunto completo**.

### 5.6 ¿Está documentada esa limitación?
**Parcialmente.** La Fase 1.5 indicó que se trabajaba "sobre archivos representativos" y el informe 1.6 mencionó la submuestra, **pero no se documentó explícitamente** que `02` procesa solo una lista del `selected` y que eso **difiere** del conjunto de validación (7 trials) de `03`. Esta omisión es la causa de la confusión 21 vs 26.

### 5.7 ¿Corresponde el 26 al conjunto realmente integrado al pipeline?
**Sí.** `executions_sample.csv` contiene exactamente 26 filas = 8 archivos del `selected`. `qc_summary.csv`: total_executions=26, valid=26. Es coherente con `segmentation_events.csv` (26 accepted).

### 5.8 ¿Existe un bug que impide integrar trials validados?
**No hay bug de algoritmo.** Sí hay una **omisión de cobertura**: 4 trials validados por `03` (S03-E01-T02, S03-E02-T01, S03-E02-T02, S05-E01-T02) no están en el `selected` de `02`. Se trata de una decisión de muestreo (no integrados), no de un fallo de segmentación. **Corrección propuesta (futura, no aplicada en esta fase):** ampliar el `selected` o parametrizar la lista de archivos (p. ej., leer todos los E01/E02 de S03/S05) preservando el resto. Se deja documentado; no se modifica el código para no alterar el baseline establecido.

---

## 6. Resumen B0367

| Concepto | Valor |
|----------|-------|
| Ejecuciones previamente existentes (Fase 1.6) | **17** (S01:3, S02:3, S04-E01:3, S04-E02:3, S04-E04:5) |
| Nuevas ejecuciones incorporadas en 1.6.1 (integradas en `02`) | **9** (+3 S03-E01, +3 S05-E01, +3 S05-E02) |
| **Total actual (pipeline integrado)** | **26** |
| Trials validados por `03_signal_validation.py` | **7** (S03: 4, S05: 3) |
| Trials integrados por `02_execution_segmentation.py` | **8** (lista `selected`) |
| Trials validados por `03` pero **NO integrados** en `02` | **4** (S03-E01-T02, S03-E02-T01, S03-E02-T02, S05-E01-T02) |
| Cobertura real del pipeline sobre los 26 C3D | **8/26 archivos = 30.8 %** |
| Cobertura sobre S03+S05 (E01/E02) | 7 trials evaluados / 3 integrados |

---

## 7. Corrección menor aplicada en `docs/phase_1_6_1_report.md`

La frase *"45 eventos (16 accepted → 26, ...)"* contenía un error tipográfico: el **16** era incorrecto (el total de accepted es 26, confirmado por `segmentation_events.csv` y `qc_summary.csv`). Se corrigió a:

> *"45 eventos (accepted → 26, rejected → 17, review → 2)"*, con nota de corrección.

No se modificó ninguna otra sección.

---

## 8. Respuestas finales

### ¿26 ejecuciones es el número correcto del pipeline actual?
**Sí.** Es exactamente lo que produce `02_execution_segmentation.py` con su lista `selected` de 8 archivos, verificado en `executions_sample.csv` (26 filas), `segmentation_events.csv` (26 accepted) y `qc_summary.csv` (26/26). Es el baseline vigente.

### ¿El "21" de la sección 8 fue solo una validación experimental?
**Sí.** El 21 (7 trials × 3 detecciones) corresponde a la **validación de la señal** en `03_signal_validation.py`, no a ejecuciones integradas al pipeline. Es esperable y correcto que 21 ≠ 26, porque son dos conceptos: señal candidata validada vs ejecución incorporada al dataset de extracción.

### ¿Existe algún trial validado que debería estar integrado y no lo está?
**Sí — 4 trials** están validados por `03` con RTOE (3 ejecuciones cada uno) pero no figuran en el `selected` de `02`: **S03-E01-T02, S03-E02-T01, S03-E02-T02, S05-E01-T02**. No es un bug de algoritmo, sino una **decisión de muestreo** (el pipeline procesa una submuestra fija). Se proponen como candidatos a integrar en una fase futura de cobertura completa.

### ¿Qué debemos considerar como baseline B0367 para comparar con el segundo atleta?
**26 ejecuciones en 8 archivos** (lista `selected` actual), con **QC 26/26 y 0 errores**. IMPORTANTE: para comparar atletas de forma justa, la **misma lista de trials** debe aplicarse a cada atleta (misma técnica/condición/trial por participante). La submuestra actual (solo algunos trials por técnica) es suficiente para validar el pipeline, pero **no** representa el dataset completo de cada atleta. Para comparación entre atletas se recomienda procesar, por lo menos, un trial por técnica/condición común (E01 y E02, T01 y T02) de forma consistente entre participantes.

---

## 9. Limitaciones y alcance

- No se modificó el algoritmo de segmentación.
- No se procesaron otros atletas, no se implementó ML, no se creó dashboard.
- No se modificó el AGENTS.md.
- La ampliación de cobertura (integrar 4 trials validados) se deja **documentada y propuesta**, no ejecutada, para mantener intacto el baseline de 26.

---

## 10. Evidencia reproducida

Comandos/funciones usados para verificar (reproducible):

1. Inventario: `all_files()` sobre `B0367/` → 26 C3D.
2. `selected` de `02`: extraído del `main()` (8 archivos).
3. Trials de `03`: `for ... in [("S03", ["E01-T01","E01-T02","E02-T01","E02-T02"]), ("S05", ["E01-T01","E01-T02","E02-T01"])]` → 7.
4. Conteos: `executions_sample.csv` (26), `segmentation_events.csv` (45 eventos → 26/17/2), `qc_summary.csv` (26/26, 100 %).