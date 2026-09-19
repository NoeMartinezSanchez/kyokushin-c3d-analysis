# FASE 1.8E — Inventario completo de atletas y homogeneidad del dataset

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-17
**Script:** `scripts/08_athlete_inventory.py`
**Salidas:** `output/athlete_inventory/` (11 CSV)
**Duración real de ejecución:** ~27.7 min primera ejecución (1661.8 s); reanudación con checkpoints ~64 s.

---

## 1. Objetivo

Radiografía estructural completa del dataset antes de escalar el pipeline desde 2 atletas hacia la población total. Es una **auditoría de inventario**, NO procesamiento masivo: no se segmentó, no se construyó Data Mart global, no se crearon configs, no se modificó nada.

## 2. Población detectada (real)

- **37 atletas** descubiertos por el filesystem en `atletas/` (B0367…B0405, sin B0390 ni B0397).
- **1411 archivos C3D** inventariados (coincide con el paper: 37 participantes / 1411 grabaciones).
- Fuente canónica: `atletas/`. **Se excluyó** la copia `B0367/` de la raíz (idéntica a `atletas/B0367/`, 26 archivos) para evitar duplicados.

| Conteo | Valor |
|--------|-------|
| Atletas | 37 |
| C3D | 1411 |
| Archivos por atleta | 16 (B0370) – 51 (B0381); mayoría 39–41 |

## 3. Inventario por atleta y por archivo

- `athlete_inventory.csv` (37 filas): nº archivos, técnicas/condiciones/trials, sampling rates, rango de puntos y de variables derivadas.
- `file_inventory.csv` (1411 filas × 27 cols): nombre, técnica/condición/trial, rate, unidades, nº puntos, frames, duración, variables derivadas, sujetos/prefijos, Tarcza, señales de segmentación disponibles.

## 4. Frecuencias de muestreo (Hipótesis CONFIRMADA)

`sampling_rate_summary.csv` y `sampling_rate_group.csv`:

| Frecuencia | Archivos | Atletas |
|-----------|---------|---------|
| **200 Hz** | 91 | 4 (B0367, B0368, B0369, B0370) |
| **250 Hz** | 1320 | 33 |

La hipótesis inicial (200 Hz → B0367–B0370; 250 Hz → resto) se **confirmó** leyendo cada archivo. No hay más frecuencias. Los 4 atletas de 200 Hz constituyen una **cohorte** especial que coincide con los 4 primeros del rango.

## 5. Unidades

Uniforme: posición **mm**, ángulos **deg**, fuerzas **N**, momentos **N·mm**, potencia **W** en todos los archivos (mismo PlugInGait). Sin anomalías de unidades.

## 6. Homogeneidad de markers

`marker_availability.csv`: las señales de segmentación **RFIN, RTOE, LTOE, RANK, LANK, RHEE, LHEE** están disponibles en **el 100% de las filas atleta×técnica** (37 atletas × 5 técnicas, salvo las combinaciones ausentes). Esto significa que **la señal primaria es estructuralmente viable para todos los atletas** (la idoneidad real dependerá de validación posterior, no solo de existencia).

Diferencias: 191 marcadores comunes; variantes de clusters y Tarcza (6 en cohorte 200 Hz; 4 en la mayoría de 250 Hz; B0367 hasta 409 puntos en algunas grabaciones).

## 7. Homogeneidad de variables derivadas

`derived_variable_inventory.csv`:
- **E01/E02**: 74 derivadas (mayoría); 101 (modelo completo, algunos atletas, p. ej. B0367/B0368); 48 en un subconjunto.
- **E03/E04**: 25 derivadas (solo ángulos del sujeto de interés, consistente con B0367/B0377).

Implicación: el número y tipo de variables derivadas **varía por atleta y condición**; el pipeline debe leer por archivo (ya lo hace vía `verify_units_from_c3d`/`extract_features` con retroceso a NaN).

## 8. Estructura de sujetos / roles (E03/E04)

`role_inventory.csv`:
- **E01**: 368 archivos, 1 sujeto real (atleta).
- **E02**: 370 archivos, 1 sujeto real + Tarcza (escudo, no persona).
- **E03**: 323 archivos, **2 sujetos** (atacante + defensor).
- **E04**: 350 archivos, **2 sujetos**.

**673 de 1411 archivos (47.7 %) son multi-sujeto real** (E03+E04). El prefijo (`subject_prefixes`) identifica al atleta vs oponente; el rol se determina por condición (E03/E04) y por coincidencia del prefijo con el nombre de archivo.

## 9. Cobertura por técnica / condición / trial

- 5 técnicas presentes en casi todos los atletas.
- **S05 ausente en B0370** (único hueco de técnica detectado; explica sus 16 archivos).
- **E03 ausente en 6 atletas** (B0367–B0370 + B0382, B0383 — incluye toda la cohorte 200 Hz).
- **E04 ausente en 3** (B0368, B0370, B0371).

## 10. Estado de configuración

`configuration_gap.csv`:
- **B0367**: config validada (S01=RFIN, S02–S05=RTOE) → **protected baseline**.
- **B0377**: config provisional (S01=RFIN, S02/S03/S05=RTOE, S04=LTOE).
- **35 atletas restantes**: `NOT_CONFIGURED` (sin config automática generada).

## 11. Anomalías estructurales

`anomaly_inventory.csv`: **0 archivos con anomalías** (ningún archivo ilegible, sin rate inesperado, sin frames/points nulos/negativos, sin técnica/condición/trial UNKNOWN, sin duplicados). Estructura del dataset **notablemente limpia**.

## 12. Grupos de readiness de escalado (técnico, no rendimiento)

`scaling_readiness.csv`:
| Grupo | N | Criterio | Atletas |
|-------|---|----------|---------|
| **A** | 1 | validado/protegido | B0367 |
| **B** | 33 | compatible tras configuración (250 Hz sin config + B0377 provisional) | B0371–B0405 (32) + B0377 |
| **C** | 3 | requiere revisión (200 Hz sin config — cohorte diferente al golden path) | B0368, B0369, B0370 |

Esto **no es ranking de rendimiento**: es clasificación de readiness de ingeniería de datos.

## 13. Limitaciones

- Inventario estructural: la idoneidad **real** de cada señal (vmax/SNR/lateralidad) requiere validación por técnica (fases futuras), no solo disponibilidad.
- No se infirió lateralidad ni nivel; solo se marcó estado (`NOT_CONFIGURED`).
- E03/E04 implican 2 sujetos: la asignación de variables derivadas al atleta/protección requiere manejo por prefijo (ya parte del pipeline).
- 200 vs 250 Hz: no normalizado (queda pendiente para 1.8F/1.8G).

## 14. Tests

`pytest tests -q` → **74 passed** (61 previos + 13 nuevos de FASE 1.8E), 0 errores, exit=0. Incluye: script ejecuta, fuente no modificada, IDs únicos, archivos existen, rate válido, hipótesis de frecuencia confirmada, hechos B0367/B0377 representados, config gap, anomalías sin crash, columnas requeridas, sin duplicados, markers disponibles en todos, scaling readiness correcto.

## 15. Recomendación para FASE 1.8F

El dataset está **estructuralmente listo para escalar** bajo condiciones:

1. **A)** Contin~uar con un atleta de 250 Hz (Grupo B) definiendo su config de señales por técnica (siguiente paso natural).
2. **B)** Generalizar la parametrización de señales/lateralidad por atleta (motor ya lo soporta).
3. **C)** Normalizar la cohorte 200 Hz o documentarla como subpoblación separada.
4. **D)** Ampliar el Data Mart/Dashboard a múltiples atletas usando el mismo contrato.

**Recomendación prioritaria (1.8F):** procesar 2–3 atletas adicionales de la cohorte **250 Hz** (Grupo B) con configs explícitas por técnica (identificando RFIN/RTOE y lateraldidad con evidencia), validar sus ejecuciones contra el golden path y **mantener el contrato del Data Mart sin cambios**. La cohorte 200 Hz (B0368–B0370) queda marcada como **Grupo C** hasta resolver decisiones de frecuencia comparativa.

## Resumen ejecutivo

- **Atletas:** 37 detectados; **C3D:** 1411.
- **Frecuencias:** solo 2 (200 Hz ×4 atletas [B0367–B0370]; 250 Hz ×33); hipótesis confirmada.
- **Unidades:** uniformes (mm/deg/N/Nmm/W).
- **Markers señal:** 100% disponibles en todos los atletas.
- **Derivadas:** 25 (E03/E04) / 74–101 (E01/E02); varía por atleta/condición.
- **Multi-sujeto:** 673 archivos (E03 323 + E04 350).
- **Cobertura:** casi completa; excepciones S05-B0370, E03 en 6 atletas, E04 en 3.
- **Config:** 2 atletas configurados (B0367 validado, B0377 provisional); 35 NOT_CONFIGURED.
- **Anomalías:** 0.
- **Readiness:** A=1, B=33, C=3.
- **Tiempo:** 27.7 min primera; ~64 s reanudación (checkpoints).

**Conclusión:** el dataset es **estructuralmente homogéneo** (homogéneo con caveats de frecuencia/cohorte). El pipeline está en condiciones de escalar a la cohorte 250 Hz con configuración explícita por atleta; la cohorte 200 Hz requiere decisiones de comparabilidad antes de incluirse.