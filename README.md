# Karate Athlete Performance Intelligence — Auditoría exploratoria

Proyecto de Sports Analytics / Machine Learning aplicado a karate Kyokushin.

Objetivo del trabajo: convertir grabaciones públicas de **optical motion capture** (Vicon, formato C3D) en un dataset tabular **1 fila = 1 ejecución**, extraer features biomecánicas por ejecución y preparar experimentos de ML (clasificación de técnica, condición y nivel).

Dataset público de Agnieszka Szczęsna, Monika Błaszczyszyn, Magdalena Pawlyta (*Scientific Data* 2021, DOI 10.1038/s41597-021-00801-5, [figshare](https://doi.org/10.6084/m9.figshare.c.4981073)).

Subconjunto local: **participante B0367** (26 archivos C3D) y **B0377** (39 archivos C3D, en `atletas/`).

> **Resumen:** Fases 1, 1.5, 1.6, 1.6.1, 1.6.2, 1.7, **1.8A (config por atleta)**, **1.8B-1 (feature readiness)** y **1.8C (Athlete Data Mart)** completadas. El mismo motor parametrizado procesa B0367 (26 ejecuciones, QC 26/26) y B0377 (13 controladas, QC 13/13) usando `config/athletes/<id>.yaml`. Golden path S02/S03/S05-E01-T01 → **18 ejecuciones con features comparables** en un **Data Mart** (`output/data_mart/athlete_execution_features.csv`) listo para el futuro Dashboard/ML. Baseline B0367 intacto. No se entrenan modelos definitivos ni se convierte el dataset masivamente a CSV.

---

## Datos disponibles para el análisis

Esta sección describe **qué contienen los archivos C3D** (verificado experimentalmente; informe completo en [`docs/dataset_audit.md`](docs/dataset_audit.md)).

### 1. Inventario del subconjunto local

| Atributo | B0367 | B0377 |
|----------|-------|-------|
| Archivos C3D | 26 | 39 |
| Fecha | 2017-01-31 | 2017-02-20 |
| Técnicas | S01–S05 | S01–S05 |
| Condiciones | E01, E02, E04 | E01, E02, **E03**, E04 |
| Trials | T01, T02 | T01, T02 |
| Frecuencia real | **200 Hz** | **250 Hz** |
| Tamaño | ~0.212 GB | — |

Nomenclatura de archivos: `YYYY-MM-DD-CODE-S0X-E0Z-T0J.c3d`.
- S01 = Gyaku-Zuki (puño recto), S02 = Mae-Geri (patada frontal), S03 = Mawashi-Geri gedan, S04 = Mawashi-Geri jodan, S05 = Ushiro-Mawashi-Geri.
- E01 = aire, E02 = escudo (shield), E03 = atacante (ausente en B0367), E04 = defensor.
- T01/T02 = trial 1 / 2 por técnica+condición.

Cobertura en B0367 (técnica × condición × trial; 1 = presente):

| | E01 T01 | E01 T02 | E02 T01 | E02 T02 | E04 T01 | E04 T02 |
|--|---------|---------|---------|---------|---------|---------|
| S01 | 1 | 0 | 1 | 1 | 1 | 1 |
| S02 | 1 | 1 | 1 | 1 | 1 | 1 |
| S03 | 1 | 1 | 1 | 1 | 1 | 1 |
| S04 | 1 | 1 | 1 | 1 | 1 | 1 |
| S05 | 1 | 1 | 1 | 0 | 0 | 0 |

> E03 (attacker) **no** está en B0367. S05 solo tiene 3 archivos (cobertura parcial). B0377 sí es completo (incluye E03).

### 2. Qué hay dentro de cada C3D

Por punto, el C3D guarda **filas X, Y, Z + residual**, en **mm**. Cada grabación E01/E02 trae **218 puntos**:

- **39 marcadores anatómicos PlugInGait** (cabeza, columna/clavícula/tórax, hombros, brazos/antebrazos, codos, muñecas, manos, pelvis, muslos, rodillas, tibias, tobillos, talones, dedos de los pies).
- **76 marcadores de cluster** (grupos de 4 por segmento: pelvis, fémur, tibia, pie, punta, cabeza, clavícula, torso, húmero, radio, mano).
- **103 variables biomecánicas derivadas ya calculadas** por el pipeline Vicon/PlugInGait (no hay que derivarlas nosotros):
  - **30 ángulos articulares** (LHipAngles, LKneeAngles, LElbowAngles, LSpineAngles, RHeadAngles, ...).
  - **16 potencias** (LHipPower, LKneePower, LAnklePower, LShoulderPower, ...).
  - **16 fuerzas + 16 momentos** (estimaciones del modelo, NO medidas de plataformas).
  - **15 centros de masa** (CentreOfMass, PelvisCOM, LeftFemurCOM, HeadCOM, ...).
  - **~10 centros articulares** (LHJC, RHJC, LKJC, RKJC, LAJC, RAJC, ...).
- En **E02** se añaden **6 marcadores del escudo** (`Tarcza1`–`Tarcza6`).

En **E04 (defensor)** el archivo contiene **dos sujetos** (B0367 defensor + B0368 atacante) → **397 puntos** (2 × ~198). El atleta del dataset **defiende**; quien ejecuta la técnica es el oponente. Solo hay 23 variables derivadas en E04.

### 3. Características técnicas verificadas

- **Ejes:** X = frontal (izquierda→derecha), Y = sagital (adelante→atrás), Z = vertical.
- **Unidades:** mm (POINT.UNITS); ángulos en deg; derivadas en N, N·mm, W.
- **Frecuencia:** 200 Hz en B0367 y 250 Hz en B0377 (**el paper cita 250 Hz** — discrepancia documentada).
- **Fabricante:** Vicon. **Sin canales analógicos** (sin plataformas de fuerza, sin EMG): fuerzas/momentos son estimaciones del modelo PlugInGait, no reacción en el suelo.
- **NaN/missing:** bajo (~0.15–0.20 % de frames por eje), por oclusión en brazos/piernas.
- **Línea base:** el archivo empieza con el atleta quieto en kumite-no-kamae (velocidad del pie 14–32 mm/s).
- **Eventos Vicon:** 4 genéricos por archivo ("Event"/"General"), no segmentan fases específicas — la segmentación debe derivarse de las señales.

### 4. Qué se puede medir vs. qué NO

**Sí se puede medir:** cinemática 3D completa (trayectorias, velocidades, aceleraciones de cualquier marcador), ángulos y velocidades articulares, ROM articular, fases de ejecución (método reproducible), COM y estabilidad, simetría L/R, coordinación proximal-distal, repetibilidad T01 vs T02 (DTW), comparación aire vs escudo (E01 vs E02), distancia al escudo (Tarcza) y entre atletas (E04).

**NO se puede medir con este subconjunto:** fuerza de reacción del suelo (GRF) y momentos reales (sin plataformas), EMG, presión plantar, impacto instrumentado sobre el objetivo, respuesta fisiológica/fatiga, ni nada concluyente sobre nivel del atleta (n=1, sin validez estadística).

---

## Resumen de fases realizadas

Qué se ha hecho hasta el momento, con su script, salida y estado. Informes detallados en `docs/`.

| Fase | Objetivo | Script | Salidas | Estado |
|------|----------|--------|---------|--------|
| **1 — Auditoría del dataset** | Inventariar, inspeccionar técnicamente y mapear los 26 C3D de B0367; validar lectura con ezc3d; documentar variables, calidad, visualizaciones, features y riesgos de ML | `01_dataset_exploration.py` | `output/tables/`, `output/figures/`, `output/exploration_log.txt` | ✅ Completada |
| **1.5 — Pipeline de ejecuciones** | Validar el pipeline `C3D → repeticiones → segmentación → features → 1 fila = 1 ejecución` | `02_execution_segmentation.py` | `executions_sample.csv`, `execution_quality.csv`, `phase_validation/`, `dtw_validation/`, `condition_comparison/` | ✅ Completada |
| **1.6 — QC y trazabilidad** | Resolver discrepancia 17 vs 20; clasificar eventos accepted/rejected/review con razón; verificar unidades/frecuencia | `01` + `02` + `tests/test_pipeline.py` | `segmentation_events.csv`, `qc_summary.csv`, `units_verified.csv`, `execution_features_sample.csv` | ✅ Completada |
| **1.6.1 — Señales S03/S05** | Determinar experimentalmente la mejor señal de segmentación para Mawashi-Geri gedan y Ushiro-Mawashi-Geri (criterio ≠ solo SNR) | `03_signal_validation.py` | `output/signal_validation/` | ✅ Completada |
| **1.6.2 — Reconciliación de conteos** | Aclarar diferencia entre trials de validación (7) y trials integrados al pipeline (8) — no hay bug, miden cosas distintas | revisión de config/scripts/CSV | `docs/phase_1_6_2_report.md` | ✅ Completada |
| **1.7 — Generalización a B0377** | Probar el pipeline en un segundo atleta sin tocar el baseline; comparar estructura y lateralidad | `04_athlete_generalization_audit.py` | `output/athlete_generalization/` | ✅ Completada |
| **1.8A — Config por atleta** | Parametrizar señal/lateralidad/joints/rutas por atleta en YAML con un único motor | `02` (motor) + `05_phase_1_8a_validation.py` | `config/athletes/<id>.yaml`, `output/phase_1_8a/` | ✅ Completada (regresión + S01/S04 OK) |
| **1.8B-1 — Feature readiness** | Verificar que el golden path (S02/S03/S05-E01-T01) genera features mínimas comparables entre atletas | `06_feature_readiness.py` | `feature_readiness_sample.csv`, `feature_readiness_audit.csv` | ✅ Completada (18 ejecuciones; corregido bug event_id) |
| **1.8C — Athlete Data Mart** | Consolidar golden path en un Data Mart estable (identidad + metadata + features + comparabilidad + versiones) para el futuro Dashboard/ML | `07_build_athlete_data_mart.py` | `output/data_mart/athlete_execution_features.csv`, `data_mart_summary.csv` | ✅ Completada (18 filas; validaciones OK) |
| **1.8D — Dashboard/cobertura (siguiente)** | Resolver umbral S01-B0377 (baseline alto, marcado `NEEDS_VALIDATION`), ampliar cobertura B0377, normalizar 200/250 Hz, prototipo de Dashboard leyendo solo el Data Mart | — | — | ⏳ Pendiente |

**Hallazgo transversal importante:** la configuración es **por atleta y por técnica** (B0367: S01=RFIN, S02–S05=RTOE, lateralidad derecha; B0377: S04=LTOE, lateralidad izquierda en S04). La selección de señales de B0367 **no es universal**.
Para ML futuro se usará siempre **división por participante (GroupKFold / Leave-One-Subject-Out)**, nunca random split por archivo.

---

## Estructura del proyecto

```
Deporte_sensores/
├── B0367/                      # Datos originales B0367 (NO MODIFICAR)
├── atletas/
│   └── B0377/                  # Segundo atleta (Fase 1.7/1.8A)
├── config/
│   ├── segmentation.yaml       # Config global (umbrales, método)
│   └── athletes/
│       ├── B0367.yaml          # Config por atleta (señal/lateralidad/joints)
│       └── B0377.yaml
├── scripts/
│   ├── 01_dataset_exploration.py    # Fase 1: auditoría del dataset
│   ├── 02_execution_segmentation.py # Fase 1.5/1.6/1.8A: motor por atleta
│   ├── 03_signal_validation.py      # Fase 1.6.1: validación de señales S03/S05
│   ├── 04_athlete_generalization_audit.py  # Fase 1.7: auditoría de generalización
│   ├── 05_phase_1_8a_validation.py  # Fase 1.8A: regresión + S01/S04
│   ├── 06_feature_readiness.py      # Fase 1.8B-1: features mínimas comparables (golden path)
│   └── 07_build_athlete_data_mart.py  # Fase 1.8C: consolida el Data Mart (capas de datos)
├── tests/
│   └── test_pipeline.py        # Tests mínimos (pytest, 51 tests)
├── docs/
│   ├── dataset_audit.md         # Informe Fase 1 (archivos/calidad/features/ML)
│   ├── phase_1_5_report.md      # Informe Fase 1.5
│   ├── phase_1_6_report.md      # Informe Fase 1.6
│   ├── phase_1_6_1_report.md    # Informe Fase 1.6.1 (señales S03/S05)
│   ├── phase_1_6_2_report.md    # Informe Fase 1.6.2 (reconciliación de conteos)
│   ├── phase_1_7_report.md      # Informe Fase 1.7 (generalización a B0377)
│   ├── phase_1_8a_config_audit.md  # Auditoría de config (Fase 1.8A)
│   ├── phase_1_8a_report.md     # Informe Fase 1.8A (config por atleta)
│   ├── phase_1_8b_feature_readiness.md  # Informe Fase 1.8B-1 (features mínimas)
│   ├── phase_1_8c_data_mart.md      # Informe Fase 1.8C (Athlete Data Mart)
│   └── athlete_data_mart_dictionary.md  # Data dictionary del Data Mart
├── output/
│   ├── athlete_generalization/  # Auditoría de generalización a B0377 (Fase 1.7)
│   ├── phase_1_8a/              # Validación de la config por atleta (1.8A)
│   ├── data_mart/              # Athlete Data Mart (Fase 1.8C; capa para Dashboard/ML)
│   │   ├── athlete_execution_features.csv  # 1 fila = 1 ejecución (18)
│   │   └── data_mart_summary.csv           # métricas del Data Mart
│   ├── B0377/                   # Salidas de B0377 (no tocan el baseline)
│   ├── feature_readiness_sample.csv  # golden path features (Fase 1.8B-1)
│   ├── figures/                # Visualizaciones Fase 1 (PNG)
│   ├── tables/                 # Tablas CSV Fase 1
│   ├── executions_sample.csv   # 1 fila = 1 ejecución B0367 (26, trazable)
│   ├── execution_features_sample.csv  # features por nivel conceptual
│   ├── execution_quality.csv   # control de calidad por ejecución
│   ├── segmentation_events.csv # TODOS los eventos (accepted/rejected/review)
│   ├── qc_summary.csv          # resumen total/valid/review/invalid
│   ├── units_verified.csv      # unidades/frecuencia leídas de los C3D
│   ├── phase_validation/       # segmentación validada visualmente
│   ├── signal_validation/      # comparativa de señales S03/S05 (1.6.1)
│   ├── dtw_validation/         # DTW T01 vs T02
│   ├── condition_comparison/   # E01 vs E02
│   └── exploration_log.txt / phase15_log.txt / phase18_*.txt
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.12
- Ver `requirements.txt`.

Instalación (Windows):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Se usa **ezc3d** (≥1.7) para leer los C3D. La librería entrega los puntos como array `(4, n_points, n_frames)` con filas `X, Y, Z, residual` (mm).

---

## Cómo ejecutar la exploración

```bash
# Fase 1 — auditoría del dataset
.venv\Scripts\python scripts\01_dataset_exploration.py

# Fase 1.5/1.6/1.8A — pipeline por atleta (baseline B0367 -> output/)
.venv\Scripts\python scripts/02_execution_segmentation.py

# Procesar OTRO atleta (B0377 -> output/B0377/)
.venv\Scripts\python scripts/02_execution_segmentation.py B0377

# Fase 1.6.1 — validación de señales de S03/S05
.venv\Scripts\python scripts\03_signal_validation.py

# Fase 1.7 — auditoría de generalización (B0367 vs B0377)
.venv\Scripts\python scripts\04_athlete_generalization_audit.py

# Fase 1.8A — validación de la config por atleta (regresión + S01/S04)
.venv\Scripts\python scripts\05_phase_1_8a_validation.py

# Fase 1.8B-1 — features mínimas comparables (golden path S02/S03/S05-E01-T01)
.venv\Scripts\python scripts\06_feature_readiness.py

# Fase 1.8C — construir el Athlete Data Mart (consumirá el futuro Dashboard)
.venv\Scripts\python scripts\07_build_athlete_data_mart.py

# Tests (51)
.venv\Scripts\python -m pytest tests -q
```

El script `01` (auditoría):

1. Inventaría todos los `.c3d` (técnica / condición / trial / tamaño / fechas).
2. Inspecciona técnicamente archivos representativos (frames, Hz, duración, marcadores, unidades, NaN, fabricante, eventos).
3. Construye el mapa de marcadores → regiones anatómicas.
4. Explora señales y detecta **fases de ejecución** con un método reproducible (umbrales relativos sobre el perfil de velocidad del endpoint).
5. Compara condiciones (E01 aire / E02 escudo / E04 defensor) para Mawashi-Geri jodan.
6. Mide repetibilidad T01 vs T02 con **Dynamic Time Warping**.
7. Documenta el feature engineering (calculable vs no calculable).
8. Documenta los problemas de ML viables y riesgos de leakage.

El script `02` (motor parametrizado por atleta, Fase 1.5/1.6/1.8A):

1. Resuelve la **carpeta de datos** por atleta (`data_dir_for`): B0367 en `B0367/`, otros en `atletas/<id>/`.
2. Carga `config/segmentation.yaml` (global) y `config/athletes/<id>.yaml` (por atleta), con merge profundo.
3. Selecciona la **señal efectuadora por técnica** desde la config del atleta (B0367: RFIN/RTOE; B0377: S04=LTOE).
4. Segmenta repeticiones por **bandas de actividad** y clasifica cada evento como accepted/rejected/review con razón (→ `segmentation_events.csv`).
5. Extrae features por ejecución con **`joints_side` por técnica** (L en S04-B0377) → **1 fila = 1 ejecución** trazable (`executions_sample.csv`).
6. Genera control de calidad (`execution_quality.csv`), resumen (`qc_summary.csv`) y verifica unidades/frecuencia desde los C3D (`units_verified.csv`).
7. Escribe **por atleta**: baselines en `output/` raíz; otros atletas en `output/<athlete_id>/` (no tocan el histórico).
8. Las validaciones DTW/condición/E04 solo corren para el baseline (requieren config propia).

Salidas en `output/` (B0367) y `output/<athlete_id>/` (otros atletas).

El script `06` (Feature Readiness, Fase 1.8B-1):

1. Reutiliza el motor `02` y consume los `executions_sample.csv` ya generados por atleta (no relee C3D).
2. Filtra el **golden path** (S02/S03/S05-E01-T01, B0367+B0377) → `feature_readiness_sample.csv`.
3. Mapea `rom_{joints_side}Hip/Knee/AnkleAngles` → `hip_rom/knee_rom/ankle_rom` desde la config del atleta.
4. Emite `feature_readiness_audit.csv` con el estado de comparabilidad de cada feature (sensible a Hz, a lateralidad, directamente comparable).

El script `07` (Athlete Data Mart, Fase 1.8C):

1. Consolida `feature_readiness_sample.csv` + metadata de `executions_sample.csv` en `output/data_mart/athlete_execution_features.csv` (1 fila = 1 ejecución).
2. Añade metadata de adquisición (`sampling_rate_hz`, `primary_signal`, `movement_side`, `event_id` global), comparabilidad por feature, y versionado (`feature_version`, `segmentation_version`, `units_version`, `mart_version`).
3. Valida identidad/calidad/unidades/frecuencia/lado y escribe `data_mart_summary.csv`.
4. Es la **capa que consumirá el futuro Dashboard** (nunca C3D).

---

## Hallazgos clave

- Los C3D contienen 39 marcadores anatómicos PlugInGait + 76 marcadores de cluster + **103 variables derivadas ya calculadas** (ángulos, potencias, fuerzas, momentos, COM).
- No hay canales analógicos (sin plataformas de fuerza): fuerzas/momentos son estimaciones del modelo PlugInGait.
- En la condición **E04 (defensor)** el atleta B0367 defiende; el C3D incluye al oponente B0368 (atacante) que ejecuta la técnica. Esto es una limitante importante para comparar ejecuciones.
- **Dos atletas procesados por el mismo motor:** B0367 (26 ejecuciones, 200 Hz) y B0377 (13 controladas, 250 Hz). La configuración (señal/lateralidad/joints) es **por atleta y por técnica** en `config/athletes/`.
- **B0377 ejecuta el Mawashi-Geri jodan (S04) con la pierna IZQUIERDA** (señal LTOE; joints_side L) — la configuración de B0367 no es universal.
- **S01-B0377** tiene baseline alto (~599 mm/s): 3 picos reales detectables con umbral robusto (median+3·MAD), pero el método actual (mediana del primer segundo) no lo resuelve → marcado `NEEDS_VALIDATION`.
- Frecuencia real **200 Hz (B0367) / 250 Hz (B0377)** verificada en los C3D (el paper cita 250 Hz). Aún no se normaliza temporalmente.
- La clasificación de nivel atleta **NO es válida con un solo participante**: se requiere el dataset completo (37 atletas) y **GroupKFold / Leave-One-Subject-Out**.
- **Pipeline 1 fila = 1 ejecución validado y trazable:** 45 eventos → 26 aceptadas / 17 rechazadas / 2 en review, 100 % cobertura de QC. Cada fila identifica fichero + frames + señal + método.
- En E02 (escudo) la velocidad pico del pie sube a ~15.7 m/s frente a 9.4 m/s en aire (descriptivo, n=1, sin afirmar beneficio).
- La **coordinación proximal-distal** quedó **NO VALIDADA**; pendiente de rediseño.
- **Señal por técnica (B0367):** S01=RFIN, S02=S03=S04=S05=RTOE. **Por técnica (B0377):** S04=LTOE (izquierda), resto RTOE. La selección NO es universal.
- **Pendiente (Fase 1.8D):** resolver umbral S01-B0377, ampliar cobertura B0377 (T02/E02/E03/E04), normalización 200/250 Hz y prototipo de Dashboard leyendo solo el Data Mart.

---

## Próximos pasos sugeridos

1. **Fase 1.8D:** Dashboard prototipo leyendo solo el Data Mart, resolver S01-B0377, ampliar cobertura B0377, validar S04-LTOE en más trials y normalización temporal 200/250 Hz.
2. Descargar el dataset completo desde figshare (37 atletas) cuando se autorice.
3. `08_extract_features.py`: extraer features por ejecución → dataset tabular multiatleta (reutilizando `02` con `config/athletes/<id>.yaml`).
4. `09_evaluate_models.py`: primer experimento = **clasificación de técnica (S01–S05)** con GroupKFold por participante.

Ver informes: [`docs/dataset_audit.md`](docs/dataset_audit.md), [`docs/phase_1_5_report.md`](docs/phase_1_5_report.md), [`docs/phase_1_6_report.md`](docs/phase_1_6_report.md), [`docs/phase_1_6_1_report.md`](docs/phase_1_6_1_report.md), [`docs/phase_1_6_2_report.md`](docs/phase_1_6_2_report.md), [`docs/phase_1_7_report.md`](docs/phase_1_7_report.md), [`docs/phase_1_8a_report.md`](docs/phase_1_8a_report.md), [`docs/phase_1_8b_feature_readiness.md`](docs/phase_1_8b_feature_readiness.md) y [`docs/phase_1_8c_data_mart.md`](docs/phase_1_8c_data_mart.md).