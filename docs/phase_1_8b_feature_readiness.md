# FASE 1.8B-1 — Feature Readiness Audit

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-13
**Alcance (golden path):** B0367 + B0377 × S02/S03/S05 × E01-T01
**Script:** `scripts/06_feature_readiness.py`
**Salidas:** `output/feature_readiness_sample.csv`, `output/feature_readiness_audit.csv`

**Regla de diseño:** pocas features confiables y comparables antes que muchas dudosas. La siguiente capa (Dashboard/ML) NO debe leer C3D: debe consumir Feature Dataset.

---

## 1. ¿Qué pudimos calcular?

Para el golden path (18 ejecuciones: 9 B0367 + 9 B0377), **todas** las features mínimas del plan se calcularon y quedaron en `feature_readiness_sample.csv`:

| Columna | Estado | Nota |
|---------|--------|------|
| `execution_id` | ✅ | formato `{athlete}|{technique}|{condition}|{trial}|{repetition:03d}`; único |
| `duration_s` | ✅ | `(end-start)/rate` (segundos) |
| `time_to_peak_s` | ✅ | `(peak-start)/rate`; medida **relativa al evento segmentado**, NO se asume que es tiempo hasta impacto |
| `vmax` / `vmean` | ✅ | de la señal primaria configurada (RTOE); en m/s; son **proxy**, no "velocidad de impacto" |
| `amax` | ✅ | derivada de la velocidad suavizada (gradiente discreto, ventana Hanning 15); en m/s² |
| `displacement` | ✅ | distancia entre posición inicial y final del endpoint (mm→m) |
| `path_length` | ✅ | longitud acumulada de la trayectoria (mm→m) |
| `hip_rom` / `knee_rom` / `ankle_rom` | ✅ | desde `rom_{joints_side}Hip/Knee/AnkleAngles` (en **grados**); mapeado por config |
| `snr` | ✅ | vmax/(baseline+1) de la señal primaria |
| `qc_status` | ✅ | `status` del evento en `segmentation_events.csv` (ver §bug) |
| `quality_flag` | ✅ | OK/WARN desde `execution_quality.csv` |

## 2. ¿Qué falta?

Nada del conjunto mínimo solicitado. **Fuera de alcance** (por decisión explícita): E02/E03/E04, T02, S01 (NEEDS_VALIDATION), S04 (validación limitada) y otros atletas.

## 3. ¿Son comparables entre B0367 y B0377?

Medias por atleta/técnica (golden path):

| atleta | técnica | duration_s | vmax | amax | hip_rom | knee_rom | ankle_rom |
|--------|---------|-----------:|-----:|-----:|--------:|---------:|----------:|
| B0367 | S02 | 0.83 | 10.32 | 167.9 | 121.8 | 156.5 | 44.5 |
| B0367 | S03 | 0.95 | 10.42 | 164.2 | 110.1 | 151.2 | 38.8 |
| B0367 | S05 | 0.76 | 10.19 | 88.4 | 138.4 | 132.8 | 46.4 |
| B0377 | S02 | 0.99 | 6.89 | 133.0 | 113.5 | 153.7 | 47.2 |
| B0377 | S03 | 0.94 | 9.65 | 177.1 | 86.8 | 142.4 | 62.1 |
| B0377 | S05 | 0.80 | 9.16 | 82.9 | 144.2 | 111.4 | 57.4 |

**Comparabilidad temática:**
- `duration_s`, `time_to_peak_s` → **directamente comparables** (independientes de Hz).
- `hip/knee/ankle_rom` → mismas unidades (deg), mismo lado (R por `joints_side`); comparables solo si ambos atletas usen el mismo lado (aqui R).
- `vmax`, `vmean`, `amax`, `path_length` → **afectadas por sampling rate** (200 vs 250 Hz); las magnitudes se muestran para inspección pero **NO deben compararse formalmente sin normalización temporal**. (Ej. B0367 S02 vmax 10.3 vs B0377 6.9: diferencia grande → requiere verificación, no es conclusión.)
- `displacement` → **magnitud geométrica** (inicial→final), independiente de Hz en su valor; la frecuencia solo influye indirectamente vía la elección de frames start/end del detector → COMPARABLE_WITH_CAVEAT.

## 4. Features dependientes de sampling rate

| Feature | Sensible a Hz | Razón |
|---------|:---:|-------|
| duration_s, time_to_peak_s | NO | se expresa en segundos |
| vmax/vmean/amax | SÍ | cálculo usa `rate` (mm/frame → mm/s → m/s) |
| displacement | NO (indirecta) | geométrico (inicial→final); Hz solo vía frames start/end |
| path_length | SÍ | suma de diferenciales por frame; a mayor Hz más muestras |
| hip/knee/ankle_rom | parcial | grados por ventana; el número de muestras cambia pero el ROM en grados converge |
| snr | parcial | baseline en mm/s depende de Hz; ratio vmax/bl estable |

**Conclusión:** `duration_s` y `time_to_peak_s` son robustos; las cinemáticas necesitan normalización temporal (fase futura) para comparación formal.

## 5. Features dependientes de lateralidad / joints_side

`hip_rom`,`knee_rom`,`ankle_rom` → dependen de `joints_side` (config por técnica). Golden path: R en ambos atletas para S02/S03/S05. B0377/S04 (izquierda) queda documentado pero fuera de este análisis. Verificado que **no hay referencias hardcodeadas a R** en `extract_features` (mapea por `JOINTS_SIDE`).

## 6. Features dependientes de la señal primaria

`vmax`, `vmean`, `amax`, `snr`, `displacement`, `path_length` se calculan sobre **la señal primaria configurada** (RTOE para S02/S03/S05 de ambos). Saludablemente homogéneo en este golden path.

## 7. Unidades consistentes?

| Feature | Unidad final | Conversión |
|---------|--------------|------------|
| duration_s / time_to_peak_s | s | — |
| vmax / vmean | m/s | mm/s × MM_IN_M |
| amax | m/s² | mm/s² × MM_IN_M |
| displacement / path_length | m | mm × MM_IN_M |
| hip/knee/ankle_rom | **deg** | sin conversión (C3D ya en deg) |
| snr | adimensional | — |

Centralizado en `MM_IN_M = 1e-3` en `extract_features`. Sin riesgo de mezclar mm/m ni grados/radianes.

## 8. Problema encontrado y corregido (trazabilidad)

Al validar el `feature_readiness_sample` se observó que `qc_status` variaba (accepted/rejected) para filas **aceptadas**, lo cual era imposible. Causa: **bug en `02_execution_segmentation.py`** — `event_id` en `executions_sample.csv` era el **índice posicional** de la ejecución aceptada (0,1,2...), mientras `segmentation_events.csv` usa el **índice global** de todos los eventos (incluye rechazados). Ambos divergían tras rechazos.

**Corrección aplicada:** en `process_file`, el `event_id` de cada fila ahora se resuelve mediante un mapa `(start,peak,end frame) → event_id global` del registro de eventos. Tras regenerar B0367 y B0377, las 18 ejecuciones del golden path tienen `qc_status=accepted` coherente.

**Impacto:** los `event_id` anteriores (Fases 1.5-1.8A) para archivos **sin rechazos** coincidían por casualidad; con rechazos divergían. El baseline cuantitativo (26 ejecuciones, QC) **no cambia**; solo el `event_id` de trazabilidad queda correcto.

## 9. Problemas restantes antes de escalar a 37 atletas

1. **Normalización temporal** (200 vs 250 Hz) para comparar vmax/amax/path_length formalmente — pendiente.
2. **S01-B0377** umbral (NEEDS_VALIDATION): no bloquea el golden path, pero debe resolverse para cobertura completa.
3. **S04-B0377** (LTOE/izquierda): ROM mapeado a L* ya funciona; falta validar con más trials.
4. **Convención de nomenclatura** por atleta al escalar (carpetas, joints_side por atleta) — ya parametrizada en config.
5. Confirmar que `quality_flag` y `qc_status` se mantengan coherentes si se procesan ejecuciones "review".

## 10. Tabla de estado (Feature | Estado | Problema | Acción futura)

| Feature | Estado | Problema | Acción futura |
|---------|--------|----------|---------------|
| duration_s | READY | — | — |
| time_to_peak_s | READY_WITH_CAVEAT | relativo al evento, no al impacto real | documentar en dashboard |
| vmax | READY_WITH_CAVEAT | sensible a Hz; es proxy | normalización temporal + naming "velocidad endpoint" |
| vmean | READY_WITH_CAVEAT | sensible a Hz | idem |
| amax | READY_WITH_CAVEAT | derivada discreta suavizada | idem + documentar ventana |
| displacement | READY | definición estable | — |
| path_length | READY_WITH_CAVEAT | suma por frame; convergen con más Hz | documentar dependencia de Hz |
| hip_rom | READY | depende de joints_side | mantener config por técnica |
| knee_rom | READY | idem | idem |
| ankle_rom | READY | idem | idem |
| snr | READY_WITH_CAVEAT | baseline en mm/s depende levemente de Hz | documentar |
| qc_status | READY | fundido con event_id (corregido) | monitorear en próximos atletas |
| quality_flag | READY | OK/WARN | enlazar con revisión humana |

## 11. Enlace con la arquitectura futura

```
C3D → Ingest/Audit → Segmentation → Execution Dataset → Feature Dataset
        (01)             (02)              (02)           (06)
                  → Athlete Data Mart → Dashboard → ML
```
`feature_readiness_sample.csv` es el prototipo del **Feature Dataset** (1 fila = 1 ejecución con unidad/definición documentada y comparabilidad conocida).

## 12. Archivos

- **Modificado:** `scripts/02_execution_segmentation.py` (event_id global — corrección trazabilidad).
- **Nuevo:** `scripts/06_feature_readiness.py`, `output/feature_readiness_sample.csv`, `output/feature_readiness_audit.csv`.
- **Regenerado:** `output/executions_sample.csv`, `output/B0377/executions_sample.csv` (con event_id corregido).

## 13. Comandos ejecutados

```bash
.venv\Scripts\python scripts/02_execution_segmentation.py          # B0367
.venv\Scripts\python scripts/02_execution_segmentation.py B0377    # B0377
.venv\Scripts\python scripts/06_feature_readiness.py               # feature readiness
.venv\Scripts\python -m pytest tests -q                            # regresión
```

## 14. Ejecuciones procesadas

**18** ejecuciones del golden path (9 B0367 + 9 B0377), todas `qc_status=accepted`, `quality_flag OK`, sin NaN en features mínimas. Baseline B0367 sigue en 26/26 (incluye S01/S04 fuera del range).

## 15. Criterio de éxito

✅ "Para B0367 y B0377, en S02/S03/S05/E01-T01, existe un conjunto mínimo de features por ejecución almacenable en una tabla estructurada, con unidades y definiciones documentadas, y sabemos cuáles son directamente comparables (duration, time_to_peak, ROM por lado), cuáles son comparables con caveat (displacement, snr) y cuáles requieren normalización temporal antes de una comparación formal (vmax/vmean/amax/path_length)."

## 16. Recomendación para la siguiente fase

1. **FASE 1.8C (siguiente):** crear el **Athlete Data Mart** consumiendo `feature_readiness_sample` por atleta (un único dataset normalizado con `athlete_id` como partición), conservando metadatos de unidades/Hz por fila. No leer C3D en el dashboard.
2. **Antes de ML:** normalización temporal (200/250 Hz) y resolución de los NEEDS_VALIDATION (S01-B0377, S04 extensión).
3. **Dashboard (posterior):** consumir solo Feature Dataset / Data Mart; nunca C3D directo.