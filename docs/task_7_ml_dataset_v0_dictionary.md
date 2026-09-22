# Data Dictionary — ML Dataset v0

**Fuente:** `output/ml_dataset_v0/ml_dataset_v0.csv` (419 filas × 14 columnas)
**Manifest:** `output/ml_dataset_v0/dataset_manifest.csv`

Unidad de observación: **1 ejecución aceptada** (250 Hz × S02–S05 × E01 × T01).
Roles: `identifier` (nunca predictores), `target`, `feature`.

## Identifiers (trazabilidad / agrupación)

### execution_id — role: identifier
- Identificador único de ejecución `{athlete}|{technique}|{condition}|{trial}|{repetition:03d}`. Nunca predictor.

### athlete_id — role: identifier
- Identificador del atleta. **Requerido como grupo para la validación futura (GroupKFold).** Nunca predictor (el objetivo es aprender la técnica, no memorizar individuos).

## Target

### technique — role: target
- Variable objetivo: técnica. Clases: `S02`, `S03`, `S04`, `S05`.

## Features (units y comparabilidad del contrato)

| feature | unit | comparability | normalization_required | nota |
|---|---|---|---|---|
| duration_s | s | DIRECTLY_COMPARABLE | no | duración del evento segmentado |
| time_to_peak_s | s | DIRECTLY_COMPARABLE | no | relativo al evento, no impacto |
| vmax | m/s | REQUIRES_NORMALIZATION | sí | proxy cinemático, no impacto |
| vmean | m/s | REQUIRES_NORMALIZATION | sí | |
| amax | m/s² | REQUIRES_NORMALIZATION | sí | derivada de velocidad suavizada |
| displacement | m | COMPARABLE_WITH_CAVEAT | no | geométrico |
| path_length | m | REQUIRES_NORMALIZATION | sí | sumatorio por frame |
| hip_rom | deg | COMPARABLE_WITH_CAVEAT | no | lado `movement_side` |
| knee_rom | deg | COMPARABLE_WITH_CAVEAT | no | lado `movement_side` |
| ankle_rom | deg | COMPARABLE_WITH_CAVEAT | no | lado `movement_side` |
| snr | dimensionless | COMPARABLE_WITH_CAVEAT | no | proxy señal/baseline; COMENTARIO: advertencia por celda |

**Nota:** los valores raw están permitidos en `ml_dataset_v0`; la normalización se aplicará **dentro de cada fold** en Task 7B (para evitar leakage).

## Exclusiones (documentadas en `feature_audit.csv`)

Excluidas como features: `athlete_id`, `execution_id` (HIGH → metadata), `primary_signal`, `movement_side`, `source_dataset` (config/provenance → leakage medium), `sampling_rate_hz`, `condition`, `trial`, `qc_status`, `quality_flag` (constantes), `event_id`, `repetition` (metadata), `comparability_*`, versiones (metadata/versionado).

## Válido
- Población: 419 ejecuciones, 33 atletas, `qc_status=accepted`, 0 NaN, 0 inf.
- Clases: S02=106, S03=104, S04=94, S05=115 (BALANCED; ratio 1.22).
- Cobertura parcial: B0377, B0388, B0401 (sin S04) — no fabricadas.