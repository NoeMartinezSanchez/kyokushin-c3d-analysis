# Auditoría de configuración — FASE 1.8A

Documento pre-refactor (FASE 1 del plan 1.8A): ubicación de parámetros antes de moverlos a configuración por atleta.

Fecha: 2026-09-13

## Tabla de auditoría

| Parámetro | Actualmente dónde | General o específico | Debe ir a config atleta | Motivo |
|-----------|-------------------|----------------------|--------------------------|--------|
| `DATA_DIR = ROOT/"B0367"` | `01:63` | específico (atleta) | sí (resuelto por `data_dir_for`) | sin esto no se encuentra `atletas/B0377` |
| `signals.*.signal` (RFIN/RTOE…) | `segmentation.yaml` + `02` (PRIMARY_SIGNAL) | específico (técnica/atleta) | sí (override en `config/athletes/<id>.yaml`) | B0377 S04 = LTOE |
| `FALLBACK_SIGNAL` | `02` hardcoded | específico | decisión (heredado por motor) | replicar sin romper B0367 |
| `joins` / `chain` (ángulos R*) | `02:extract_features` hardcoded | específico (lateralidad) | sí (`joints_side`) | S04-B0377 es zurda |
| `selected` (lista trials) | `02:main` hardcoded | específico (atleta) | por atleta (lista controlada) | baseline vs B0377 distinta |
| `params.*` (umbrales) | `segmentation.yaml` | general | queda global; override atleta opcional | umbrales de segmentación son del método |
| `OUT/...` (rutas de salida) | `02` raíz | específico (atleta) | sí (`OUT_ATHLETE`) | no sobrescribir histórico |
| `role` (defensor/atacante) | `02:process_file` | general por condición | mantener | E03/E04 según condición |
| `rate` / unidades / derivadas | se leen del C3D (`verify_units_from_c3d`) | hecho del C3D | **NO** en config (se lee) | fuente real = archivo |

## Decisiones tomadas

1. **Ruta de datos por atleta**: helper local `data_dir_for(athlete_id)` en `02` (no se toca `01`).
2. **Salidas**: `B0367` sigue en `output/` raíz (histórico intacto); otros atletas → `output/<athlete_id>/`.
3. **Config por atleta**: `config/athletes/<id>.yaml` sobreescribe la global vía merge profundo.
4. **Laterality y joints_side**: por técnica en la config del atleta.
5. **FALLBACK_SIGNAL**: heredado por el motor (idéntico al comportamiento pre-1.8A).
6. **NO** se modifica el algoritmo de segmentación, QC, features ni `01`.