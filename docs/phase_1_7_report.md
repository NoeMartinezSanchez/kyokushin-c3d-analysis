# FASE 1.7 — Generalización del pipeline a un segundo atleta (B0377)

**Proyecto:** Karate Athlete Performance Intelligence
**Atleta analizado:** B0377 (descubierto en `atletas/B0377/`)
**Baseline de referencia:** B0367 (26 ejecuciones, 8 archivos, QC 26/26)
**Fecha:** 2026-09-13
**Script:** `scripts/04_athlete_generalization_audit.py`
**Salidas:** `output/athlete_generalization/*.csv|png`

**Regla seguida:** no se modificaron `02_execution_segmentation.py`, `AGENTS.md`, ni la config global; no se entrenó ML; no se alteró el baseline de B0367.

---

## 1. Inventario del nuevo atleta (FASE 1)

**Atleta:** B0377 — 39 archivos C3D, fecha única 2017-02-20, **250 Hz** (≠ 200 Hz de B0367).

| Atributo | Valor |
|----------|-------|
| nº C3D | 39 |
| Técnicas | S01–S05 (todas) |
| Condiciones | E01, E02, E03, E04 (**incluye E03 attacker**, ausente en B0367) |
| Trials | T01, T02 |
| Frecuencia | **250 Hz** (verificado en todos) |
| Unidades | mm, deg, N, Nmm, W |
| Puntos por archivo | 119 (E01 sin escudo) a 382 (E03/E04, 2 sujetos) |
| Sujetos vistos | B0377, B0378 (atacante en E03), B0376 (defensor en E04), Tarcza |
| Variables derivadas | 72 en E01/E02; 23 en E03/E04 (solo ángulos del sujeto de interés) |
| Tarcza | 4 marcadores (en E02); 6 en B0367 |
| Canales analógicos | 0 |

**Observación clave de estructura:** B0377 tiene cobertura **completa** (todos los trials T01/T02, incluye E03). S02 tiene 7 archivos (falta S02-E04-T02). Es un conjunto más homogéneo y completo que B0367.

---

## 2. Comparación estructural B0367 vs B0377 (FASE 2)

`output/athlete_generalization/marker_comparison.csv` (230 filas):

- **Marcadores idénticos:** 191 compartidos (misma nomenclatura PlugInGait completa: LFHD…RTOE + clusters + derivadas).
- **Solo en B0367:** 27 (incluye `Tarcza5`, `Tarcza6` y subtipos de cluster que B0377 no tiene).
- **Solo en B0377:** 0 — todos los marcadores de B0377 están en B0367.
- **Compatibilidad de atributos:** 195/230 filas compatibles.

Elementos **idénticos**: reader C3D, unidades, nomenclatura de marcadores, sistema PlugInGait, coordenadas X/Y/Z+residual.
Elementos **compatibles con ajuste**: frecuencia (B0377=250 vs B0367=200), nº de marcadores Tarcza (4 vs 6), nº de variables derivadas (72 vs 103).
Elementos **ausentes en B0377**: los clusters extra y `Tarcza*` adicionales de B0367.
Elementos **nuevos en B0377**: condición E03 (atacante) con dos sujetos.
Elementos **ambiguos**: la asignación de variables derivadas en E03/E04 cambia según quién es el sujeto de interés (ver §6).

---

## 3. Lateralidad de B0377 (FASE 3, sin asumir)

`output/athlete_generalization/lateralality_analysis.csv` + `lateralality_by_technique.png`.

| Par | L vmax (media) | R vmax (media) | ratio R/L |
|-----|------|------|-----|
| toe | 4365 | 6915 | 2.50 |
| ankle | 3896 | 5837 | 2.00 |
| heel | 4107 | 6182 | 1.90 |
| knee | 2288 | 3494 | 2.30 |
| thigh | 1929 | 2991 | 1.90 |

**Decisión agregada: RIGHT** (mediana ratio = 2.0). **Pero la lateralidad NO es uniforme por técnica**:

| Técnica | ratio R/L (toe) | pierna más activa |
|---------|------|------------|
| S01 (puño) | 0.95 | indefinida (bilateral, esperable en Gyaku-Zuki) |
| S02 (Mae-Geri) | 2.68 | **derecha** |
| S03 (Mawashi gedan) | 6.41 | **derecha** |
| S04 (Mawashi jodan) | **0.46** | **izquierda** |
| S05 (Ushiro-Mawashi) | 1.97 | **derecha** |

**Hallazgo de generalización:** B0377 ejecuta el **Mawashi-Geri jodan (S04) con la pierna IZQUIERDA** (LTOE vmax 9513 vs RTOE 4349), mientras las demás técnicas son de pierna derecha. Esto **contradice el supuesto implícito** de B0367 (pierna derecha para todas). La lateralidad se infiere de los datos por técnica, no es universal.

**Conclusión lateralidad B0377:** agregada = RIGHT; **por técnica: S02/S03/S05 → derecha, S04 → izquierda, S01 → bilateral**. La regla del pipeline "una señal por técnica (R*) configurada en YAML" **no es universal**: exige parametrizar por atleta y por técnica.

---

## 4. Compatibilidad de señales (FASE 4)

`output/athlete_generalization/signal_comparison.csv` y `signal_comparison_b0377.png`. Se probaron las señales configuradas para B0367 sobre B0377 (E01-T01):

| técnica | señal (B0367) | ¿existe? | baseline | vmax | SNR | aceptadas | estado |
|---------|--------------|:--:|--:|--:|--:|:--:|--------|
| S01 | RFIN | sí | 599 | 8103 | 13.5 | **1** | requiere parametrización |
| S02 | RTOE | sí | 29 | 7149 | 240 | 3 | **compatible** |
| S03 | RTOE | sí | 73 | 10390 | 140 | 3 | **compatible** |
| S04 | RTOE | sí | 159 | 4289 | 27 | **2** | requiere parametrización |
| S05 | RTOE | sí | 31 | 9385 | 296 | 3 | **compatible** |

**Diagnóstico:**
- **S02, S03, S05** → la señal RTOE de B0367 funciona en B0377 sin cambios (3 aceptadas, SNR alta).
- **S01 (RFIN)**: la mano de B0377 tiene **actividad basal alta (599 mm/s vs ~78 en B0367)** → el detector de bandas con umbral de B0367 fragmenta la señal (detecta +4 golpes, acepta solo 1 de forma poco fiable). La señal existe y hay picos claros (~1.74/3.21/4.74/6.11 s), pero requiere **parametrización de umbrales por atleta**.
- **S04 (RTOE)**: la pierna real es la **IZQUIERDA** (LTOE vmax 9324 vs RTOE 4289) → con RTOE solo se detectan 2 fragmentos cortos. La señal correcta es **LTOE**, no RTOE.

---

## 5. Prueba controlada del pipeline actual (FASE 5)

`output/athlete_generalization/pipeline_b0377_e01_run.csv`. Resultado ejecutando el pipeline tal como está (sin cambios) sobre E01-T01 de cada técnica:

| tag | status | marker | SNR | aceptadas | rechazadas | review |
|-----|--------|--------|----:|:--:|:--:|:--:|
| S01-E01-T01 | processed | RFIN | 14.2 | **1** | 7 | 2 |
| S02-E01-T01 | processed | RTOE | 250 | 3 | 0 | 0 |
| S03-E01-T01 | processed | RTOE | 141 | 3 | 3 | 0 |
| S04-E01-T01 | processed | RTOE | 29 | **2** | 6 | 1 |
| S05-E01-T01 | processed | RTOE | 302 | 3 | 6 | 0 |

**El pipeline NO falla técnicamente** (procesa todos, exit=0), pero **sub-detecta S01 y S04** por las razones de §4 (basal del puño y lateralidad izquierda de S04). **Con LTOE en S04**, la segmentación es limpia: 3 ejecuciones (~1.30/3.28/5.56 s). Es decir, el fallo es **de configuración por atleta**, no del algoritmo.

---

## 6. Matriz de compatibilidad de componentes (FASE 6)

`output/athlete_generalization/pipeline_compatibility.csv`:

| componente | B0367 | B0377 | clase | acción |
|-----------|-------|-------|-------|--------|
| C3D reader (ezc3d) | ok | ok | **A** | sin cambios |
| unidades (mm/deg/N/Nmm/W) | mm,deg | mm,deg | **A** | sin cambios |
| marker mapping (nomenclatura) | PlugInGait | PlugInGait | **A** | sin cambios |
| QC | 26/26 | probado | **A** | sin cambios |
| sampling rate | 200 Hz | **250 Hz** | **B** | ya se lee del archivo; verificar normalización temporal |
| derived variables | 103 (E01/E02) | **72 (E01/E02)** | **B** | verificar disponibilidad por archivo |
| lateralidad | derecha (R*) | **RIGHT agregada; S04=izquierda** | **B** | parametrizar por atleta y por técnica |
| signal selection | RFIN/RTOE en YAML | **S04 → LTOE; S01 → umbrales** | **C** | señal específica de B0367 NO es universal |
| execution segmentation | ok | ok con ajustes | **B** | umbrales por atleta (basal del puño) |
| feature extraction | ok | a probar con señal correcta | **B** | depende de markers/ángulos por archivo |
| E04 múltiples sujetos | B0367 def / B0368 atk | **E04: B0377 def / B0376 atk; E03: B0378 atk / B0377** | **B** | rol definido por prefijo; ya manejado en `02` |
| Tarcza | 6 markers | **4 markers** | **B** | parametrizar nº de marcadores de escudo |
| DTW | probado | pendiente | **B** | requiere misma técnica/condición/trial |

---

## 7. Respuestas a las preguntas de éxito

1. **¿La estructura C3D es compatible con B0367?** **Sí**, casi idéntica: mismos markers (191 comunes), mismos units, mismo PlugInGait. Diferencias menores: 250 Hz (vs 200), 72 derivadas (vs 103), Tarcza 4 (vs 6), y la condición **E03 nueva**.
2. **¿Los markers son comparables?** **Sí**, nomenclatura idéntica; B0377 no introduce marcadores nuevos.
3. **¿La lateralidad puede inferirse?** **Sí — RIGHT agregada, pero por técnica S04 = IZQUIERDA.** Es un resultado cuantitativo, no una suposición.
4. **¿Las señales de B0367 funcionan en B0377?** Parcialmente: **S02/S03/S05 sí (RTOE)**, **S01 y S04 NO** (basal alta del puño y lateralidad izquierda del S04).
5. **¿Qué componentes requieren parametrización por atleta?** Lateralidad, señales por técnica, umbrales de segmentación (basal), nº de marcadores Tarcza, y mapeo E03/E04 al rol.
6. **¿Qué componentes pueden convertirse en reglas generales?** C3D reader, unidades, marker mapping, QC, E04 multiple-subject (por prefijo), y el **método de bandas de segmentación** (con umbrales parametrizados).
7. **¿Qué debemos modificar antes de procesar múltiples atletas?** (a) parametrización de lateralidad/señal/umbrales **por atleta y por técnica** (p. ej. por archivo de config por atleta); (b) usar un **rate por archivo** (ya leído del C3D) y normalizar frecuencias si se mezclan 200/250 Hz; (c) ampliar la prueba a más trials de B0377 antes de concluir.

---

## 8. Hallazgo de generalización (lo más importante)

La configuración de B0367 **NO es universal**:
- El supuesto "pierna derecha para todas las técnicas" **falla**: B0377 ejecuta S04 con la izquierda.
- El umbral de actividad para el puño (S01) debe ajustarse por atleta (basal 599 vs 78 mm/s).
- El nº de marcadores del escudo varía (4 vs 6) y debe leerse del archivo, no suponerse.

Por tanto, el pipeline es **generalizable en su mecánica** (lectura, segmentación por bandas, QC, trazabilidad) pero **NO en su parametrización** (señal/lateralidad/umbrales). **No se puede usar la config de B0367 para B0377 tal cual.**

---

## 9. Tests

`pytest tests -q` → **22 passed** (16 previos + 6 nuevos: `test_new_athlete_dir_exists`, `test_new_athlete_c3d_can_be_inspected`, `test_no_right_laterality_assumed`, `test_unknown_is_valid_status`, `test_no_hardcoded_b0377_signals_in_script02`, `test_b0367_baseline_reproducible`).

Se verificó que el **baseline B0367 permanece intacto** (26 filas, solo B0367) tras la auditoría.

---

## 10. Limitaciones

- La prueba de pipeline sobre B0377 usó solo E01-T01 de cada técnica (submuestra). No se procesaron aún E02/E03/E04 ni T02 de B0377.
- La lateralidad "S04 = izquierda" se infiere por actividad cinemática (vmax LTOE ≫ RTOE) con 1 archivo por técnica; debe confirmarse con T02 y E02.
- No se aplicaron cambios a la config ni a los scripts: es una auditoría, no una integración.
- No se entrenó ML ni se compararon atletas entre sí a nivel de features (fuera de alcance de esta fase).

---

## 11. RECOMENDACIÓN PARA FASE 1.8

**Clasificación: B) generalizar lateralidad** — con acciones previas.

La evidencia muestra que:
1. Las señales S02/S03/S05 (RTOE) ya son compatibles en B0377 (derecha), pero **S04 requiere LTOE** (lateralidad izquierda por técnica).
2. **S01 necesita parametrización de umbrales** (basal ~600 mm/s) independientemente de la lateralidad.

Recomendación concreta:
- **Fase 1.8a (obligatorio antes de escalar):** introducir una **config por atleta** (e.g. `config/athletes/<id>.yaml`) que defina lateralidad señal por técnica y umbrales por atleta, leyéndolos desde el C3D (rate, nº Tarcza, variables derivadas disponibles) en lugar de asumirlos. Verificar S04 con **LTOE** sobre T01/T02 y E02.
- **Fase 1.8b (deseable después):** generalizar la extracción de features con verificación por archivo (que existan ángulos/potencias) y normalización temporal para 200 vs 250 Hz.
- **NO incorporar un tercer atleta** ni entrenar ML hasta que B0377 esté parametrizado y su QC reproduzca una submuestra comparable (misma regla de trials que B0367).

**No se inicia la Fase 1.8 en esta sesión.**

---

## Resumen ejecutivo

- **Atleta analizado:** B0377.
- **Nº de C3D:** 39.
- **Técnicas:** S01–S05.
- **Condiciones:** E01, E02, E03 (nueva vs B0367), E04.
- **Trials:** T01 y T02.
- **Compatibilidad con B0367:** alta en estructura (191 marcadores comunes, mismos units, mismo PlugInGait); difiere en frecuencia (250 vs 200 Hz), nº derivadas (72 vs 103) y Tarcza (4 vs 6).
- **Lateralidad:** RIGHT agregada, pero **S04 = IZQUIERDA** (hallazgo crítico).
- **Señales compatibles:** S02, S03, S05 (RTOE).
- **Señales NO compatibles tal cual:** S01 (basal alta del puño) y S04 (lateralidad izquierda → LTOE).
- **Ejecuciones procesables con el pipeline actual tal cual:** solo S02/S03/S05 (3 por técnica); S01/S04 requieren parametrización.
- **Problemas encontrados:** (1) lateralidad por técnica no uniforme; (2) umbral de basal para puños específico del atleta; (3) frecuencia/distribución de variables derivadas variables; (4) E03 nuevo requiere manejo de rol.
- **Cambios recomendados:** config por atleta con lateralidad/señal/umbrales; leer rate y nº Tarcza del archivo; verificar S04-LTOE sobre más trials; después normalización temporal.
- **Conclusión sobre generalización:** el **núcleo del pipeline (lectura, segmentación por bandas, QC, trazabilidad) se generaliza**; la **parametrización (señal-lateralidad-umbrales) NO se hereda automáticamente** y debe definirse por atleta.