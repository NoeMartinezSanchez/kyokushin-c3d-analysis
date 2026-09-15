# FASE 1.5 — Validación del pipeline de ejecuciones individuales

**Proyecto:** Karate Athlete Performance Intelligence
**Dataset:** Subconjunto local B0367 (26 C3D, 1 atleta)
**Fecha:** 2026-09-10
**Script:** `scripts/02_execution_segmentation.py`
**Salidas:** `output/executions_sample.csv`, `output/execution_quality.csv`, `output/phase_validation/`, `output/dtw_validation/`, `output/condition_comparison/`, `output/dtw_validation_results.csv`, `output/e04_defender_attacker_distances.csv`, `output/condition_s04_air_shield.csv`, `output/execution_file_summary.csv`

---

## 1. Objetivo de esta fase

Validar experimentalmente que podemos convertir un C3D con varias repeticiones en un dataset tabular **1 fila = 1 ejecución**:

```
C3D -> identificación de repeticiones -> segmentación -> features -> 1 fila/ejecución
```

No se entrenó ningún modelo, no se modificaron los C3D, no se construyó dashboard, y no se descargaron datos nuevos.

---

## 2. Archivos utilizados

| Archivo | Técnica | Condición | Notas |
|---------|---------|-----------|-------|
| `2017-…-S01-E01-T01.c3d` | Gyaku-Zuki | E01 (aire) | 3 ejecuciones |
| `2017-…-S02-E01-T01.c3d` | Mae-Geri | E01 (aire) | 3 ejecuciones |
| `2017-…-S04-E01-T01.c3d` | Mawashi-Geri jodan | E01 (aire) | 3 ejecuciones |
| `2017-…-S04-E02-T01.c3d` | Mawashi-Geri jodan | E02 (escudo) | 3 ejecuciones |
| `2017-…-S04-E04-T01.c3d` | Mawashi-Geri jodan | E04 (defensor) | 5 eventos defensivos de B0367; el atacante (B0368) ejecuta 3 patadas |

Todos los archivos presentan la estructura esperada. El archivo E04 fue analizado por separado (sección 7) porque involucra dos sujetos.

---

## 3. Elección de la señal (Primer paso — inspección de señales)

**Hallazgo clave:** NO se debe asumir que RTOE es siempre la mejor señal.

| Técnica | Mejor señal | Razón | Limitaciones |
|---------|-------------|-------|--------------|
| S01 Gyaku-Zuki (puño) | **RFIN / RWRB** (mano derecha) | La patada no existe: el endpoint del gesto es el puño. RFIN alcanza vmax ~6.9 m/s con SNR 92. En cambio RTOE tiene una SNR engañosa (560) por su base ~0, pero sus picos corresponden al desplazamiento del pie, no al golpe. | El cuarto movimiento de RFIN (2.2 m/s, t≈8 s) es un braceo final, no una repetición: se filtra por ratio de altura. |
| S02 Mae-Geri | **RTOE** (pie derecho dominante) | Pico de patada claro (~10.3–10.4 m/s), base estable ~15 mm/s, SNR 671. | Cada patada origina 2 picos (extensión + retorno); se fusionan por bandas de actividad (ver §4). |
| S03/S04/S05 (patadas) | **RTOE/RANK/RHEE** (pie derecho) | Mismo patrón que S02. En S04-E01 el mejor endpoint de la muestra resultó RHEE en la exploración Fase 1 y RTOE en esta corrida (SNR 438); ambos son válidos. | RANK puede ser más suave que RTOE; la elección automática por SNR+efectuador resuelve esto. |
| E04 (defensor) | RTOE de **B0368 (atacante)** para detectar la patada; RTOE de B0367 solo muestra movimientos defensivos breves (~1 m/s) | Quien ejecuta la técnica es el atacante. | No son ejecuciones del atleta B0367; se etiquetan como `defensive_response`. |

**Decisión de diseño:** cada técnica tiene una **señal primaria por efecto del gesto** (mano para puño, pie para patadas) y la SNR se usa solo como comprobación (mínimo 8). Esto evita el sesgo de elegir marcadores casi inmóviles con SNR altísimas (pie durante un puño).

---

## 4. Método de segmentación

Implementado en `segment_repetitions()` — **método por bandas de actividad**, más robusto que "buscar todos los máximos":

1. **Preprocess mínimo:** velocidad 3D del endpoint + suavizado Hanning (ventana 15 ≈ 75 ms @200 Hz).
2. **Baseline:** mediana de la velocidad del primer segundo (atleta quieto en kumite-no-kamae).
3. **Nivel de actividad** = `baseline + 0.15 * (vmax - baseline)`.
4. **Bandas:** períodos continuos donde la velocidad suavizada supera el nivel de actividad. Cada banda = una ejecución. Esto **fusiona los subpicos** de la misma ejecución (chamber + extensión + retorno) y separa ejecuciones distintas.
5. **Fusión de bandas contiguas:** si dos bandas están separadas por < 0.3 s y la segunda tiene pico ≤ a la primera, se consideran la misma ejecución (retorno/recuperación).
6. **Filtro de altura:** se descartan bandas cuyo pico < 45 % del máximo global (movimientos menores de ajuste o braceo).
7. **Filtro de duración:** 0.15–2.5 s.

Parámetros: `activity_frac=0.15, smooth_win=15, min_gap_s=0.30, min_peak_ratio=0.45`.

**Importante (no forzar):** para S05 (Ushiro-Mawashi-Geri) la fase 1 detectó solo 3 archivos locales y esta fase no los incluyó por cobertura parcial. No se forzó la segmentación donde no hay datos suficientes.

---

## 5. Resultado de la segmentación

| Archivo | Señal | nº detecciones | nº esperado (paper) | Evaluación |
|---------|-------|:---:|:---:|---|
| S01-E01-T01 | RFIN | 3 | 3 | Coherente |
| S02-E01-T01 | RTOE | 3 | 3 | Coherente |
| S04-E01-T01 | RTOE | 3 | 3 | Coherente |
| S04-E02-T01 | RTOE | 3 | 3 | Coherente |
| S04-E04-T01 (B0367) | RTOE | 5 (defensivos, ~0.5–1.2 m/s) | n/a | NO son ejecuciones de la técnica |
| S04-E04-T01 (B0368) | RTOE | 3 patadas (~10–12 m/s) | 3 | Coherente (rol atacante) |

**Validación visual:** todas las gráficas quedaron en `output/phase_validation/` con start/peak/end marcados y la banda sombreada, permitiendo revisión manual.

**Conclusiones visuales de las gráficas (verificables en PNG):**
- En S01 las 3 replicaciones son distinguibles como tres ráfagas de velocidad de RFIN seguidas de retorno.
- En S02/S04 cada patada aparece como una banda con un pico dominante; el pico secundario (retorno) queda fuera del conteo por el umbral de bandas.
- En S04-E04 la señal del defensor es irregular y corta (~0.17 s por evento), coherente con una respuesta defensiva y no con una técnica.

---

## 6. Tabla de ejecuciones (output/executions_sample.csv)

17 filas, 43 columnas. Estructura por fila:

- **Metadata:** `athlete, technique, technique_name, condition, condition_name, trial, role, execution_type, repetition`
- **Segmentación:** `start_frame, peak_frame, end_frame, start_time_s, peak_time_s, end_time_s`
- **Temporal:** `duration_s, time_to_peak_s`
- **Cinemática:** `vmax_m_s, vmean_m_s, amax_m_s2, amean_m_s2, displacement_m, path_length_m, rom_m`
- **Articular:** `rom_<joint>Angles`, `avg_angvel_<joint>` (cadera/rodilla/tobillo/codo/hombro según técnica)
- **COM:** `com_vmax_m_s, com_rom_m`
- **Coordinación:** `coord_delay_proximal_distal_s` (ver §9)
- **Señal:** `signal_marker, signal_snr`

Ejemplo (valores reales):

```
athlete technique condition trial role repetition duration_s vmax_m_s  displacement_m
B0367   S01       E01       T01   atacante      1       0.540     6.591    0.190
B0367   S02       E01       T01   atacante      1       0.810    10.301    0.050
B0367   S04       E01       T01   atacante      1       0.990     8.993    0.040
B0367   S04       E02       T01   atacante      1       0.725    15.426    0.568
B0367   S04       E04       T01   defensor      1       0.175     0.735    0.086
```

**Unidades (verificadas en parámetros C3D y conservadas):**
- Posiciones/mm; convertidas a **m** en métricas derivadas (`*1e-3`).
- Velocidades m/s, aceleraciones m/s², desplazamiento/trayectoria en m.
- Ángulos en **grados (deg)** — `ANGLE_UNITS=deg` en el C3D; velocidades angulares en deg/s.
- Momentos en **N·mm**, fuerzas en N, potencia en W (parámetros `MOMENT_UNITS/POWER_UNITS/FORCE_UNITS`).
- Frecuencia **200 Hz** (parámetro `POINT.RATE`), NO 250 Hz como el paper → discrepancia documentada. Se mantienen ambas observaciones.

---

## 7. E04 — Análisis defensor (B0367) vs atacante (B0368)

**Verificado en el C3D:**
- **B0367 (defensor, atleta del dataset):** 115 puntos, todos marcadores (0 variables derivadas). Rol = recibe las patadas.
- **B0368 (atacante):** 140 puntos = 115 marcadores + 25 variables derivadas (ángulos) — el modelo PlugInGait se aplicó al atacante, no al defensor.

**No se mezclaron sujetos:** prefijos `B0367:` y `B0368:` separan los marcadores correctamente (vía `get_time` con prefijo).

**Cinemática relativa por ejecución (output/e04_defender_attacker_distances.csv), usando centro pélvico (media de ASIS/PSIS) de cada sujeto:**

| rep | min dist entre sujetos | dist al pico | vel relativa máx | distancia pie atacante→centro defensor (mín) |
|-----|----|----|----|----|
| 1 | 1.028 m | 1.184 m | 1.018 m/s | 0.586 m |
| 2 | 1.037 m | 1.141 m | 0.847 m/s | 0.564 m |
| 3 | 1.186 m | 1.240 m | 0.988 m/s | 0.656 m |

La gráfica `output/phase_validation/e04_defender_vs_attacker.png` muestra velocidad del atacante con sus ejecuciones, altura del centro pélvico de ambos, distancia entre sujetos y velocidad relativa. **Solo descriptivo; no hay modelo de interacción.**

**Limitación documentada:** las variables derivadas (ángulos/COM) solo existen para el atacante; el COM del defensor no está disponible (por eso `com_vmax_m_s` es NaN en filas E04 de B0367).

---

## 8. Repetibilidad T01 vs T02 (DTW) — output/dtw_validation/

Se compararon ejecuciones emparejadas 1:1 de T01 vs T02 para S02-E01, S04-E01 y S04-E02 con **DTW sobre el perfil de velocidad normalizado** (z-score) de cada ejecución.

| base | DTW dist (rep1/2/3) | Euclid (rep1/2/3) | similitud 1/(1+d) |
|------|-----|-----|-----|
| S02-E01 | 0.497 / 1.254 / 0.429 | 0.188 / 0.203 / 0.151 | 0.67 / 0.44 / 0.70 |
| S04-E01 | 0.604 / 0.816 / 0.860 | 0.130 / 0.583 / 0.158 | 0.62 / 0.55 / 0.54 |
| S04-E02 | 0.516 / 0.594 / 0.591 | 0.310 / 0.283 / 0.397 | 0.66 / 0.63 / 0.63 |

**¿Es útil el DTW?** Sí, con una advertencia: la distancia DTW normalizada es **baja y estable** (~0.5–0.9) cuando las curvas son similares en forma, y **mayor que la Euclidiana** (0.5 vs 0.15) porque captura desalineación temporal entre ejecuciones de longitudes distintas. La figura `dtw_S02-E01_rep1.png` muestra el alineamiento temporal (camino DTW) entre T01 y T02. El DTW permite comparar ejecuciones de distinta duración sin re-muestrear, lo cual es exactamente la situación de nuestro dataset (las ejecuciones no alinean en longitud). La métrica "similitud = 1/(1+d)" es una convención útil, pero **no debe usarse todavía como métrica absoluta** sin un baseline entre-técnicas.

**Vista clínica de la repetibilidad:** dentro de un mismo trial, S02 repetición 1 y 3 tienen vmax casi idénticas (10.30/10.37/10.29 m/s) → alta consistencia interna. La repetición 2 (10.37) también consistente. Esto apoya la validez de extraer 1 fila/ejecución.

---

## 9. Comparación E01 (aire) vs E02 (escudo) — S04

A nivel descriptivo en **este atleta (n=1)**, medias de 3 repeticiones T01:

| Variable | E01 (aire) | E02 (escudo) |
|----------|-----------|--------------|
| duración | 0.99 s | 0.80 s |
| vmax | 9.42 m/s | **15.71 m/s** |
| vmean | 5.13 m/s | 6.11 m/s |
| amax | 131 m/s² | **340 m/s²** |
| desplazamiento | 0.10 m | 0.46 m |
| trayectoria (path) | 5.07 m | 4.85 m |
| ROM endpoint | 2.09 m | 2.25 m |
| ROM cadera | 150.4° | 110.9° |
| ROM rodilla | 137.2° | 144.2° |
| ROM tobillo | 41.3° | 56.3° |

**Interpretación cauta (sin sobreinterpretar):** en este atleta, golpear el escudo se asocia con mayor velocidad/aceleración del endpoint, mayor desplazamiento y mayor flexo-extensión de rodilla-tobillo, pero menor duración y menor ROM de cadera. **NO afirmamos que "el escudo mejore el rendimiento"**: la velocidad se mide en el pie (que además debe recorrer ~0.46 m para alcanzar el blanco), el impacto físico puede ampliar la cinemática, y n=1 impide generalizar. Solo describe diferencias observadas.

La gráfica `output/condition_comparison/s04_air_vs_shield_speed.png` superpone el perfil de velocidad de la repetición 1 de cada condición.

---

## 10. Features — cuáles funcionan y cuáles no

### Funcionan bien (valores estables, sin NaN, plausiblemente físicos)
- `duration_s`, `time_to_peak_s`
- `vmax_m_s`, `vmean_m_s`, `amax_m_s2` del endpoint
- `displacement_m`, `path_length_m`, `rom_m`
- `rom_<Articulación>Angles` y `avg_angvel_<Articulación>` (presentes en E01/E02; grados y deg/s)
- `com_vmax_m_s`, `com_rom_m` (disponibles en E01/E02; NaN en E04 defensor, correcto)

### Problemáticas / a descartar o revisar
- **`coord_delay_proximal_distal_s` NO VALIDADO.** El delay (pico de |velocidad angular| de la articulación distal − proximal) resultó inestable: valores negativos y ceros sin patrón consistente. El pico de |vel. angular| del **promedio de las 3 componentes** del ángulo no refleja limpiamente la secuencia proximal-distal. Se recomienda revisar usando componentes específicas (p. ej. flexo-extensión) o sincronización entre picos de velocidad lineal de segmentos (RHJC→RKJC→RAJC) en el dataset completo.
- Las filas E04 (defensor) no deben alimentar métricas de técnica: son `defensive_response`.
- El cuarto pico de RFIN en S01 (braceo final) fue filtrado por la regla `min_peak_ratio=0.45`.

### Lo que NO calculamos por falta de datos
- GRF, momentos reales, EMG, presión plantar, velocidad de impacto instrumentada.

---

## 11. Decisión sobre no leak / ML futuro

- Cada ejecución se procesa de forma **independiente** (ventana local `start..end`): las features no usan información de otras repeticiones ni del resto del archivo.
- La normalización usada en DTW es **por ejecución** (z-score de la propia curva), no global del dataset → sin leakage entre trials.
- `baseline` se calcula sobre el primer segundo de cada archivo (posición de reposo), no sobre información futura.
- El agrupamiento futuro por **participante** (GroupKFold/LOSO) debe usar `athlete` como grupo; este script ya deja `athlete` en cada fila.

---

## 12. Control de calidad (output/execution_quality.csv)

- 17 detecciones, todas `quality_flag=OK`, 0 % NaN en el endpoint de la ventana.
- Umbrales definidos: duración fuera de [0.1, 2.0] s → WARN; vmax > 20 m/s → WARN; NaN > 5 % en endpoint → WARN. (Ninguno se disparó en esta muestra.)

---

## 13. Respuestas formales del informe

**A. ¿Podemos identificar ejecuciones individuales?** Sí, en técnicas de ataque (S01–S04) con 3 repeticiones consistentes por trial.

**B. ¿Cuántas ejecuciones por archivo?** 3 (S01/S02/S04-E01/E02), 3 patadas del atacante en S04-E04, y 5 eventos defensivos de B0367 en S04-E04 (no son técnica).

**C. ¿La segmentación parece correcta visualmente?** Sí, en las 5 figuras de `phase_validation/` se observan bandas con un pico dominante y retornos bien separados.

**D. ¿Qué señal funciona mejor por técnica?** RFIN/RWRB para S01 (puño); RTOE/RHEE para S02/S03/S04/S05; RTOE del atacante para E04.

**E. ¿Qué variables podemos calcular de manera confiable?** Temporal, cinemática del endpoint, ROMs y vel. angulares articulares, COM (E01/E02), métricas del escenario E04.

**F. ¿Qué variables son problemáticas?** `coord_delay_proximal_distal_s` (inestable, NO VALIDADO) y cualquier métrica de técnica en filas de defensor; COM en E04 (no disponible).

**G. ¿DTW parece útil?** Sí, para comparar formas de ejecución de distinta duración; la similitud debe relativizarse a un baseline entre técnicas (pendiente).

**H. ¿Podemos comparar aire vs escudo?** Sí, a nivel descriptivo 1 atleta: E02 mostró mayor vmax/amax/desplazamiento y menor duración que E01 (sin afirmar beneficio).

**I. ¿Podemos analizar atacante vs defensor?** Sí, en el modo exploratorio: distancia inter-sujetos, velocidad relativa y distancia pie→defensor son calculables; no hay modelo todavía.

**J. ¿Qué problemas quedan antes de los 37 atletas?** (a) coordenación proximal-distal sin método robusto; (b) decidir si E04 se compara como técnica o como escenario defensivo (requiere E03 attacker); (c) confirmar el lado dominante por participante (aquí se fijó derecha por evidencia); (d) validación humana de las bandas por técnica.

**K. ¿Qué modificaciones necesita el pipeline antes de escalar?** (1) parametrizar señales por técnica y por pierna dominante por atleta; (2) guardar `rate` y unidades explícitas por fila; (3) añadir baseline DTW inter-técnicas; (4) reporte de quality al 100 % de archivos; (5) resolver el método de coordinación o descartarlo.

---

## 14. Criterio de éxito

| Criterio | Estado |
|----------|--------|
| 1. Lectura correcta de C3D | ✅ verificada (ezc3d, 200 Hz, mm, unidades del C3D) |
| 2. Detección razonable de múltiples ejecuciones | ✅ 3 rep/archivo en técnicas de ataque |
| 3. Visualización de fases | ✅ `phase_validation/` (5 + 1 fig) |
| 4. Extracción de features | ✅ 43 columnas por fila |
| 5. Una fila por ejecución | ✅ `executions_sample.csv` (17 filas) |
| 6. Comparación T01/T02 | ✅ DTW + figuras de alineamiento |
| 7. Comparación E01/E02 | ✅ tablas + figura |
| 8. Documentación de limitaciones | ✅ incluye "NO VALIDADO" para coordenación |

**No ocultamos nada:** la coordenación proximal-distal se reporta como NO VALIDADA y el E04 como escenario defensivo, no como técnica del atleta.