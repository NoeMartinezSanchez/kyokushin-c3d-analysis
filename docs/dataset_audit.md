# Auditoría exploratoria y técnica — Dataset de Motion Capture de Karate Kyokushin

**Proyecto:** Sports Analytics / Machine Learning aplicado a karate
**Dataset:** *Optical motion capture dataset of selected techniques in beginner and advanced Kyokushin karate athletes* (Szczęsna, Błaszczyszyn, Pawlyta — *Scientific Data* 2021, DOI 10.1038/s41597-021-00801-5).
**Origen:** https://doi.org/10.6084/m9.figshare.c.4981073
**Subconjunto local:** participante **B0367**.
**Fecha de auditoría:** 2026-09-06
**Scripts:** `scripts/01_dataset_exploration.py` (reproducible).
**Salidas generadas:** `output/tables/*.csv`, `output/figures/*.png`, `output/exploration_log.txt`.

---

## 1. Resumen ejecutivo

Confirmamos que **los archivos C3D contienen mucha más información de la que sugiere el papel "39 marcadores"**:

- Cada grabación E01/E02 trae **218 puntos** = **39 marcadores anatómicos PlugInGait** + **76 marcadores de cluster** + **103 variables biomecánicas derivadas** (ángulos articulares, potencias, fuerzas, momentos, centros de masa y articulares).
- Las variables derivadas **ya vienen calculadas dentro del C3D** por el pipeline Vicon/PlugInGait (no hay que derivarlas nosotros).
- No hay canales analógicos (sin plataformas de fuerza, sin EMG): las "fuerzas/momentos" del C3D son **estimaciones del modelo PlugInGait**, no medidas directas de reacción en el suelo.

La **lectura** de C3D es **correcta y reproducible** con `ezc3d >= 1.7`. Toda la ruta de análisis del demo es viable sobre este subconjunto para **un atleta (B0367)**, con la limitación de que no permite generalizar a población.

**Hallazgo metodológico importante de la condición "defender" (E04):**
En E04 el archivo contiene **dos sujetos** (B0367 como defensor y B0368 como atacante). El atleta del dataset **defiende** y el oponente ejecuta la técnica. Esto impide comparar la patada del atleta "aire vs escudo vs oponente" de forma limpia sin contar con E03 (attacker). Se documenta como limitante de esta submuestra.

---

## 2. Descripción del dataset

- **Fuente:** movimiento capturado con **Vicon** (10× Vantage V5 + 10× MX-T40), 200 Hz efectivos en estos archivos (el paper dice 250 Hz; ver sección 14).
- **Participantes originales:** 37 atletas Kyokushin (infantiles, jóvenes y adultos), rango de grado 9º kyu a 4º dan.
- **Técnicas grabadas:** Gyaku-Zuki (S01), Mae-Geri (S02), Mawashi-Geri gedan (S03), Mawashi-Geri jodan (S04), Ushiro-Mawashi-Geri (S05).
- **Condiciones:** E01 aire, E02 escudo (shield), E03 atacante, E04 defensor.
- **Trials:** T01/T02 por técnica+condición.
- **Contenido por trial:** 3–4 repeticiones de la técnica (verificado en varios archivos: 3–4 picos de velocidad altos por grabación).
- **Objetivo del demo:** no sustituir la inteligencia deportiva tradicional; explorar biomecánica, cinemática, coordinación, simetría, fases, perfiles de movimiento y ML.

---

## 3. Estructura de archivos (subconjunto local)

```
B0367/
├── 2017-01-31-B0367-S01/  Gyaku-Zuki          (5 archivos)
├── 2017-01-31-B0367-S02/  Mae-Geri            (6 archivos)
├── 2017-01-31-B0367-S03/  Mawashi-Geri gedan  (6 archivos)
├── 2017-01-31-B0367-S04/  Mawashi-Geri jodan  (6 archivos)
└── 2017-01-31-B0367-S05/  Ushiro-Mawashi-Geri (3 archivos)
```

**Resumen de inventario (FASE 1):**

| Atributo          | Valor                        |
|-------------------|------------------------------|
| Total archivos    | **26 C3D**                   |
| Participantes     | 1 (B0367)                    |
| Técnicas (S)      | 5 (S01–S05)                  |
| Condiciones (E)   | 3 (E01, E02, E04)            |
| Trials (T)        | 2 (T01, T02)                 |
| Fecha única       | 2017-01-31                   |
| Tamaño total      | ~0.212 GB                    |
| Trials faltantes  | 14 (incluye todo E03, S05-E02-T02, S05-E04) |

**Cobertura por técnica × condición × trial** (1 = presente, 0 = ausente):

|  | E01 T01 | E01 T02 | E02 T01 | E02 T02 | E04 T01 | E04 T02 |
|--|---------|---------|---------|---------|---------|---------|
| S01 | 1 | 0 | 1 | 1 | 1 | 1 |
| S02 | 1 | 1 | 1 | 1 | 1 | 1 |
| S03 | 1 | 1 | 1 | 1 | 1 | 1 |
| S04 | 1 | 1 | 1 | 1 | 1 | 1 |
| S05 | 1 | 1 | 1 | 0 | 0 | 0 |

> E03 (attacker) no está presente en esta submuestra. S05 tiene solo 3 archivos.

---

## 4. Variables disponibles (dentro del C3D, verificadas)

Por punto (marcador o derivado), el C3D guarda **filas X, Y, Z + residual**, en **mm**, a **200 Hz**.

### 4.1 Puntos de marcadores (en E01/E02)
- **39 anatómicos PlugInGait** (cabeza 4, columna/clavícula/tórax 5, hombros 2, brazos/antebrazos 4, codos 2, muñecas 4, manos 2, pelvis 4, muslos 2, rodillas 2, tibias 2, tobillos 2, talones 2, dedos pies 2).
- **76 marcadores de cluster** (grupos de 4 por segmento: pelvis, fémur, tibia, pie, punta, cabeza, clavícula, torso, húmero, radio, mano).
- En **E02** se añaden **6 marcadores del escudo** (`Tarcza1`–`Tarcza6`).

### 4.2 Variables biomecánicas derivadas (103 en E01/E02)
| Categoría | Cantidad | Ejemplos |
|-----------|----------|----------|
| Ángulos articulares | 30 | LHipAngles, LKneeAngles, LAnkleAngles, LAbsAnkleAngle, RShoulderAngles, LElbowAngles, LWristAngles, LSpineAngles, LThoraxAngles, RHeadAngles, LPelvisAngles, LFootProgressAngles |
| Potencia | 16 | LHipPower, LKneePower, LAnklePower, LWaistPower, LShoulderPower, LElbowPower, LWristPower, LNeckPower |
| Fuerza | 16 | LAnkleForce, LKneeForce, LHipForce, LWaistForce, LShoulderForce, LElbowForce, LWristForce, LNeckForce |
| Momento | 16 | LAnkleMoment, LKneeMoment, LHipMoment, LWaistMoment, LShoulderMoment, LElbowMoment, LWristMoment, LNeckMoment |
| Centros de masa | 15 | CentreOfMass, CentreOfMassFloor, PelvisCOM, LeftFemurCOM, RightTibiaCOM, RightHandCOM, ThoraxCOM, HeadCOM, ... |
| Centros articulares | 10 | LHJC, RHJC, LKJC, RKJC, LAJC, RAJC, LSJC, RSJC, LEJC, REJC, LWJC, RWJC |

> En **E04 (defensor)** solo se calcularon **23 variables derivadas** (ángulos mínimos del atacante B0368); el C3D de E04 trae 397 puntos (2 sujetos × ~198).

### 4.3 Orden de ejes y metadatos
- **X** = eje frontal (izquierda→derecha), **Y** = sagital (adelante→atrás), **Z** = vertical (según paper).
- Unidades: **mm** (POINT.UNITS), frecuencia **200 Hz** (POINT.RATE verificado), escala -0.01.
- **Fabricante:** Vicon. Sin canales analógicos (ANALOG.USED = 0). Sin plataformas de fuerza.
- **Eventos:** 4 eventos genéricos por archivo ("Event"/"General"), no segmentan fases específicas; los `TIMES` sugieren inicio de cada repetición (p. ej. S04-E01-T01: 1.23, 3.15, 4.93, 6.99 s).

---

## 5. Variables derivables (propuestas)

Ver FASE 7 (sección 9) para la matriz completa. Las derivables a partir de las trayectorias y las variables ya presentes incluyen:

- Velocidades y aceleraciones de cualquier marcador (numerical diff + suavizado).
- Velocidades/aces articulares (derivada de los ángulos del C3D).
- ROM articular por fase (rango de ángulos).
- Fases de ejecución (onset, preparación, ejecución, pico, recuperación) — método reproducible implementado.
- Desplazamiento, distancia recorrida (path length), rango de movimiento del endpoint.
- Coordinación proximal-distal (retraso temporal entre picos de segmentos), correlaciones cruzadas entre articulaciones.
- Simetría L/R (comparación marcadores y ángulos izquierdo/derecho).
- Estabilidad (desplazamiento y velocidad del COM).
- Distancia al objetivo Tarcza (en E02) y distancia entre atletas (en E04).

---

## 6. Calidad de datos

- **NaN/missing:** bajo. En archivos representativos E01/E02 ~**0.15–0.20 %** de frames por eje (afectan casi siempre a los mismos marcadores de brazos/piernas en momentos de oclusión). En E02 se mantiene ~0.18 %. La condición E04 también ~0.18 %.
- **Línea base:** velocidad del RTOE en reposo 14–32 mm/s → el archivo empieza con el atleta quieto en kumite-no-kamae.
- **Consistencia:** todos los archivos a 200 Hz, mm, mismo fabricante, misma estructura E01/E02.
- **Excepción estructural:** E04 tiene 2 sujetos y solo 23 derivados (ver sección 4.2).

---

## 7. Visualizaciones disponibles (generadas)

| Archivo | Contenido |
|---------|-----------|
| `f1_distributions.png` | Inventario: archivos por técnica, condición, cobertura |
| `f4_<trial>_xyz.png` | Trayectoria X/Y/Z del RTOE |
| `f4_<trial>_traj3d.png` | Trayectoria 3D del RTOE con fases marcadas |
| `f4_<trial>_phases.png` | Velocidad + aceleración con fases detectadas |
| `f4_<trial>_knee_angle.png` | Ángulo de rodilla |
| `f4_<trial>_com.png` | COM y velocidad del COM |
| `f5_condition_comparison.png` | Comparativa E01/E02/E04 (vel/longitud/ROM/distancias) |
| `f5_condition_speed_overlay.png` | Perfiles de velocidad superpuestos por condición |
| `f6_repeatability_s04_e01.png` | T01 vs T02 (perfil de velocidad) |

---

## 8. Limitaciones

1. **Un solo atleta local (B0367)** → no es posible generalizar ni entrenar clasificación de nivel con validez estadística.
2. **Falta E03 (attacker)** → la comparación "aire/escudo/oponente" queda incompleta; E04 mide al atleta como **defensor**, no como ejecutor.
3. **Fuerzas/momentos son modelo, no medidas** → sin plataformas de fuerza no hay GRF ni momentos de reacción reales.
4. **Frecuencia 200 Hz en archivos** vs 250 Hz citados en el paper → revisar capítulo figshare; los tiempos de impacto tan cortos (~10 ms) apenas se resuelven a 200 Hz.
5. **S05 submuestra pobre** (3 archivos) → repetibilidad limitada para Ushiro-Mawashi-Geri.
6. **No hay analógicos** → sin EMG, sin presión plantar, sin fuerza de impacto instrumentada.
7. **Eventos genéricos** → la segmentación de fases debe derivarse de las señales (método reproducible propuesto), no de eventos Vicon.
8. **Ausencia de mediciones fisiológicas** → imposible hablar de fatiga, carga interna, VO2, etc.

---

## 9. Features propuestas (matriz F7)

### A) Features realmente calculables con este dataset
**Temporal:**
- duración de la ejecución, tiempo hasta velocidad máxima, tiempo hasta aceleración máxima, duración de fases (preparatoria/ejecución/final).

**Cinemática (endpoint y articulaciones):**
- velocidad máxima/media, aceleración máxima, desplazamiento neto, distancia recorrida, rango de movimiento del endpoint.

**Articular:**
- ROM de cadera/rodilla/tobillo/hombro/codo/muñeca (ya presentes como ángulos), velocidad angular, aceleración angular.

**Coordinación:**
- retraso temporal entre picos de segmentos (cadera→rodilla→pie), correlación cruzada entre ángulos articulares, índice de coordinación proximal-distal.

**Estabilidad:**
- desplazamiento del COM, velocidad del COM, oscilación lateral del COM.

**Simetría / calidad:**
- diferencias L/R por articulación, % de frames con NaN (integridad del registro).

### B) Features que requerirían sensores/datos adicionales
- Fuerza de reacción del suelo (GRF) y momentos reales de pie — requiere plataformas de fuerza.
- EMG (activación muscular) — requiere electromiografía.
- Presión plantar distribuida.
- Aceleración/giro por segmento (IMU tipo WIMU).
- Velocidad de impacto real sobre el objetivo — requiere escudo instrumentado.
- Frecuencia cardíaca, lactato, VO2, fatiga — requiere fisiológicos.
- Resultado/precisión del golpe en combate.

---

## 10. Casos de uso (demo "Karate Athlete Performance Intelligence")

Con los datos disponibles se puede mostrar, **sin inventar métricas**:

1. **Athlete Profile:** perfil cinemático básico del atleta (velocidad pico por técnica, ROMs, COM).
2. **Movement Analysis:** fases de ejecución detectadas reproduciblemente, trayectorias 2D/3D, velocidad/aceleración.
3. **Technique Analysis:** comparación entre S01–S05 (distinta cinemática por técnica).
4. **Execution Comparison:** comparación T01 vs T02 (repetibilidad, DTW) y E01 vs E02 (aire vs escudo).
5. **Biomechanical Metrics:** ángulos articulares, potencia/fuerza/momento (modelo), ROM, simetría L/R.
6. **Machine Learning (demo de framework, no productivo):** clustering intra-atleta, anomalías, búsqueda de similitud y clasificación de técnica/condición.
7. **Movement Signature:** vector de features normalizado por ejecución (perfil de forma).

> Para una federación, el demo mostraría el **framework** y la visualización, no cifras concluyentes por atleta (dado n=1).

---

## 11. Problemas de ML posibles (evaluación sin entrenar)

| Problema | Objetivo | Features | Modelo | Leakage | Muestra necesaria | ¿Viable con lo local? |
|----------|----------|----------|--------|---------|-------------------|------------------------|
| Clasificación de técnica | S01–S05 | F7 (temporal, cinemática, articular) | RF / GBM / MLP | medio (trials del mismo atleta) | >=100 ejec/clase | Parcial (demo de framework) |
| Clasificación de condición | E01/E02/E04 | vel pico, altura rodilla, distancia Tarcza, duración | SVM / Logistic | mismo atleta | 30–60/condición | Sí (conceptual) |
| Clasificación de nivel | beginner vs advanced | F7 | ensemble + **GroupKFold/LOSO** | **MUY ALTO** si se mezclan trials del atleta | >=20–50 atletas/grupo | **NO** con B0367 |
| Clustering | patrones de movimiento | F7 + DTW-k | k-means / HDBSCAN | n/a | >=50 ejec | Sí (exploratorio) |
| Anomaly detection | ejecuciones atípicas | F7 / distancias DTW | IsolationForest / OCSVM | n/a | >=30 baseline | Sí (intra-atleta) |
| Similarity search | recuperar ejecuciones similares | curvas normalizadas + DTW | kNN+DTW | n/a | >=100 indexadas | Sí (26 archivos) |

**Riesgo metodológico dominante:** el split debe ser **por participante** (GroupKFold / Leave-One-Subject-Out). Con uno solo (B0367) no se puede entrenar clasificación de nivel legítima. Cualquier clasificador entrenado con splits aleatorios de archivos sobre memorizaría al atleta.

---

## 12. Riesgos metodológicos

1. **Leakage entre trials del mismo atleta** → usar GroupKFold por participante; documentado.
2. **El "modelo PlugInGait" no es ground truth biológico** → fuerzas/momentos son estimaciones; reportar como "estimación del modelo".
3. **200 vs 250 Hz** → resolución temporal; eventos de kick cortos (−10–30 ms) quedan al límite de Nyquist/antialiasing; suavizar con ventanas cortas y reportar.
4. **Normalización espacial/temporal** → necesario antes de comparar entre atletas (z-score de trayectorias y resampleo temporal, como proponen los autores).
5. **E04 no es comparable a E01/E02 a nivel de "misma técnica del mismo atleta"** → en E04 el atleta defiende; para comparación de ejecución se debe acudir a E03 (no presente aquí).
6. **Golpe de preparación ("chamber")** puede confundir la detección de fases → método umbral relativo + inspección visual por técnica.
7. **Pequeños NaN** → interpolar solo para métricas puntuales; reportar cobertura.

---

## 13. Recomendación de siguiente paso

1. **Descargar el dataset completo (37 atletas)** desde figshare. Con eso pasamos de "demo de framework" a experimentos con validez:
   - clasificación de técnica y condición con GroupKFold,
   - clasificación beginner/advanced con LOSO,
   - análisis entre-grupos (edad, grado, experiencia).
2. **Construir un pipeline reproducible en dos etapas:**
   - `02_extract_features.py`: de cada trial → vector de features F7 (A) + metadatos (atleta, técnica, condición, trial).
   - `03_evaluate_models.py`: prototipar modelos con GroupKFold/LOSO.
3. **Primer experimento de ML recomendado:** **clasificación de técnica (S01–S05)** sobre el dataset completo, con features F7 y GroupKFold por participante. Es el más discriminativo, con datos suficientes (1,411 grabaciones), y permite validar el pipeline completo de features+split antes de atacar nivel o condición.
4. **Reportar siempre** la frecuencia real (200 Hz) y la cobertura de datos.

---

## 14. Qué necesitaríamos para datos reales de una federación

Para replicar este tipo de análisis con atletas de una federación mexicana, el mínimo viable:

**Laboratorio / hardware:**
- Sistema de captura óptica (Vicon/Motion Analysis/Qualisys) con >=10 cámaras 200 Hz+ y marcadores PlugInGait Full Body (39 anatómicos + clusters) → mismo protocolo que el dataset.
- **Alternativa escalable:** sistema de IMU/inercial tipo WIMU (1 sensor por segmento: pies, tibias, muslos, pelvis, torso, brazos, manos) para salir del laboratorio hacia el gimnasio o tatami.
- Plataformas de fuerza (para GRF reales) y escudo/pad instrumentado (para velocidad de impacto) si se quieren métricas cinéticas reales.
- Plantas de presión o EMG si se quiere detalle de apoyo/muscular.

**Datos a capturar por atleta:**
- carnet (edad, sexo, peso, talla, grado kyu/dan, años de experiencia, pierna dominante),
- entrevistas/consentimiento acorde a normas éticas (similar al dataset original),
- protocolo estandarizado: calentamiento, kumite-no-kamae, T01/T02 por técnica y condición (aire, escudo, oponente), <=2 min por intento.

**Equipo de datos/ML:**
- pipeline C3D→features (reutilizable del script 01),
- base de datos por atleta con versionado de grabaciones,
- revisión humana (biomecánica + entrenador) de las segmentaciones de fases,
- validación con GroupKFold/LOSO y métricas interpretables para deporte (no solo accuracy).

**Condiciones para un sistema tipo WIMU:**
- sensores inerciales sincronizados (tiempo), calibración de orientación por segmento, fusión magnetómetro-giro-acelerómetro, y un pipeline de inferencia de articulaciones (tipo bracelet/strap solution);
- dataset de referencia para calibrar el mapping IMU→ángulos articulares (idealmente con captura óptica simultánea en una sesión de corrección, como es estándar).

---

### Respuesta ejecutiva

- **A. Qué tenemos realmente:** 26 C3D de 1 atleta (B0367), 5 técnicas, 3 condiciones, 2 trials, ~200 Hz, mm, con 39 marcadores anatómicos + 76 clusters + 103 variables derivadas del modelo PlugInGait (ángulos, potencias, fuerzas, momentos, COM) ya calculadas dentro de cada archivo.
- **B. Qué podemos medir:** cinemática 3D completa (trayectorias, velocidades, aceleraciones), ángulos y velocidades articulares, ROM, fases de ejecución (método reproducible), COM y estabilidad, simetría L/R, coordinación proximal-distal, repetibilidad (DTW), comparación aire vs escudo, y features para ML.
- **C. Qué NO podemos medir:** GRF y momentos reales (sin plataformas), EMG, presión plantar, velocidad de impacto construida en el objetivo, respuesta fisiológica/fatiga, y nada concluyente de nivel del atleta (n=1).
- **D. Qué demo podemos construir:** un **framework completo de "Karate Athlete Performance Intelligence"** — perfil del atleta, análisis de movimiento y técnicas, comparación de ejecuciones, métricas biomecánicas, clustering/anomalías/similitud y un movement signature — sobre datos reales de un atleta, dejando claro su alcance de demo (no de validación estadística).
- **E. Qué datos adicionales necesitaríamos:** el dataset completo (37 atletas) para entrenar; o en campo real, protocolo óptico/IMU tipo WIMU, plataformas de fuerza y escudo instrumentado, más carnet fisiológico-deportivo del atleta y equipo de validación humano.
- **F. Primer experimento de ML recomendado:** **clasificación de técnica (S01–S05)** con features F7 y **GroupKFold por participante**, usando el dataset completo — valida todo el pipeline (extracción, división, métricas) antes de abordar nivel o condición.