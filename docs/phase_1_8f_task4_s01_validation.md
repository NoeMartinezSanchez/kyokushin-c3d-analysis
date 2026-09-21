# FASE 1.8F — Task 4: Validación dirigida de S01 (pre-ML)

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-20
**Script:** `scripts/11_s01_targeted_validation.py`
**Salidas:** `output/scaling_validation/s01_validation/` (2 CSV + 8 figuras + log) · `docs/phase_1_8f_task4_s01_validation.md`

---

## 1. Objetivo

Investigar con evidencia la **inconsistencia de representación de S01** (Gyaku-Zuki) entre atletas, antes de construir el dataset ML. El riesgo: que un modelo aprenda diferencias de **señal usada por el pipeline** en lugar de diferencias reales del movimiento.

Preguntas a responder: (1) ¿RFIN sigue siendo conceptualmente apropiado para S01? (2) ¿RFIN es segmentable en los atletas problemáticos? (3) ¿LFIN aporta evidencia? (4) ¿RTOE como fallback es válido o es detección débil? (5) ¿el problema está en la señal, la lateralidad, el baseline, la segmentación o el criterio de calidad? **Nada se modifica** (algoritmo, gate, señales, configs).

## 2. Contexto del problema

| Atleta | Config S01 | Golden Path Task 3 |
|---|---|---|
| B0367 | RFIN (histórico, 200 Hz) | no aplica |
| B0377 | RFIN (NEEDS_VALIDATION) | no aplica |
| B0400 | RFIN | RFIN usada (snr 77) · 5 acc |
| B0371 | RFIN | **RTOE** usada (gate SNR) · 2 acc |
| B0380 | RFIN | **RTOE** usada (gate SNR) · 2 acc |

## 3. Metodología

- Capa de auditoría read-only sobre **S01 × E01-T01 × {RFIN, LFIN, RTOE}** en 5 atletas.
- Reutiliza `09._velocity/_metrics_of/_events_of` (que usan `02.get_signal`/`segment_repetitions` y `01.smooth`) y `02.load_config` para la señal configurada.
- Regla de idoneidad existente: `snr≥8 ∧ baseline<100 mm/s ∧ accepted≥3` → `segmentable`; `marginal` si hay algo positivo; `not_segmentable` si no; `missing` si el marcador no existe. Sin umbrales nuevos.
- B0367 = referencia histórica (200 Hz): métricas frescas pero **comparación cualitativa**, no de magnitudes.

## 4. Tabla comparativa (S01, E01-T01)

| Athlete | Señal | baseline | MAD | vmax | SNR | threshold | cand | acc | rej | rev | dur media | sep | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0367 | RFIN | 78.6 | 46.3 | 6872 | 86.3 | 1097.7 | 9 | 3 | 5 | 1 | 0.28 | 0.72 | segmentable |
| B0367 | LFIN | 80.8 | 42.5 | 5564 | 68.0 | 903.4 | 8 | 4 | 3 | 1 | 0.60 | 0.73 | segmentable |
| B0367 | RTOE | 4.4 | 2.8 | 3056 | 563.8 | 462.2 | 8 | 5 | 3 | 0 | 0.30 | 0.51 | segmentable |
| B0377 | RFIN | 599.0 | 377.2 | 8103 | 13.5 | 1724.6 | 10 | 1 | 7 | 2 | 0.19 | 0.25 | marginal |
| B0377 | LFIN | 462.7 | 293.2 | 5709 | 12.3 | 1249.6 | 9 | 4 | 4 | 1 | 0.26 | 0.70 | marginal |
| B0377 | RTOE | 10.7 | 6.6 | 3010 | 256.6 | 460.5 | 6 | 4 | 2 | 0 | 0.36 | 0.89 | segmentable |
| B0400 | RFIN | 69.3 | 6.5 | 5305 | 75.4 | 854.6 | 10 | 5 | 4 | 1 | 0.39 | 1.45 | segmentable |
| B0400 | LFIN | 141.0 | 19.8 | 5111 | 36.0 | 886.5 | 11 | 5 | 5 | 1 | 0.33 | 1.43 | marginal |
| B0400 | RTOE | 11.1 | 6.3 | 276 | 22.9 | 50.8 | 7 | 3 | 3 | 1 | 0.22 | 2.21 | segmentable |
| B0371 | RFIN | 1065.0 | 456.7 | 7035 | 6.6 | 1960.5 | 7 | 3 | 3 | 1 | 0.32 | 0.26 | marginal |
| B0371 | LFIN | 1238.7 | 403.1 | 5754 | 4.6 | 1916.0 | 9 | 2 | 6 | 1 | 0.24 | 0.51 | marginal |
| B0371 | RTOE | 15.8 | 5.4 | 4223 | 250.9 | 646.9 | 8 | 2 | 5 | 1 | 0.31 | 0.40 | marginal |
| B0380 | RFIN | 1183.9 | 486.7 | 5174 | 4.4 | 1782.4 | 7 | 3 | 4 | 0 | 0.26 | 0.52 | marginal |
| B0380 | LFIN | 1332.4 | 573.9 | 5036 | 3.8 | 1887.9 | 7 | 0 | 7 | 0 | 0.20 | 0.31 | not_segmentable |
| B0380 | RTOE | 21.5 | 6.6 | 374 | 16.6 | 74.3 | 6 | 2 | 3 | 1 | 0.26 | 0.38 | marginal |

## 5. Análisis por atleta

- **B0367** (referencia histórica): las 3 señales segmentables (RFIN snr 86, LFIN 68, RTOE 564). Baseline de mano bajo (≈79 mm/s, MAD 46). RFIN validado históricamente. **VALIDATED**.
- **B0377** (intermedio): RFIN marginal (**baseline 599, MAD 377**, snr 13.5, **1 acc**); LFIN marginal (4 acc); RTOE segmentable (4 acc). RFIN no alcanza la regla por baseline; aun así un candidato plausible existe. **PROMISING_BUT_INCOMPLETE**.
- **B0400** (referencia positiva): RFIN **segmentable** (baseline 69, MAD 6.5, snr 75, 5 acc, sep 1.45 s); LFIN marginal (baseline 141). El pipeline usó RFIN en Task 3. **VALIDATED**.
- **B0371** (problemático): RFIN marginal con **baseline 1065 y MAD 457** (dispersión ≈43 % del baseline) y **sep 0.26 s** (implausiblemente corta); LFIN marginal (baseline 1238). Golden Path: fallback a **RTOE** con snr 250 pero **solo 2 aceptadas** (dur 0.31 s). **PROMISING_BUT_INCOMPLETE** · `RFIN_NOT_VALIDATED;RTOE_FALLBACK_NOT_VALIDATED`.
- **B0380** (problemático): RFIN marginal (baseline 1184, MAD 487, snr 4.4); **LFIN not_segmentable (0 aceptadas)**; Golden Path: fallback a **RTOE** con 2 aceptadas de **vmax muy baja (~0.29 m/s)** y dur 0.26 s. **PROMISING_BUT_INCOMPLETE** · `RFIN_NOT_VALIDATED;RTOE_FALLBACK_NOT_VALIDATED`.

## 6. Análisis RFIN

- Segmentable y limpia en **B0367 (78.6) y B0400 (69.3)**: baseline bajo, MAD pequeño, snr alto, 3–5 aceptadas con separación estable.
- Márgenes alta baseline + MAD alta en **B0377, B0371, B0380** (462–487 mm/s de MAD ≈ 40 % del baseline): la mano está en **actividad casi continua** en esos E01-T01; el método de bandas no puede separar golpes porque no hay reposo. La regla `baseline<100` es legítimamente fallida → **el gate no es caprichoso aquí**.
- Separaciones implausiblemente cortas en B0371 (0.26 s) y B0380 (0.52 s) refuerzan que RFIN segmenta ruido de guardia/reposicionamiento, no golpes.
- **No se etiqueta `PIPELINE_GATE_LIMITATION`** en B0371/B0380 (baseline ≥100 → problema del lado de la señal, no del umbral).

## 7. Análisis LFIN

- No rescata: en B0371 (baseline 1238) y B0380 (**0 aceptadas**, baseline 1332) es igual o peor que RFIN. En B0400 es negotiable (141). La inconsistencia no es de **lateralidad de la mano**.

## 8. Análisis RTOE (fallback en B0371/B0380)

- RTOE tiene baseline limpio (~16/21 mm/s) y snr alto (251/17) — por eso `pick_best_signal` la eligió como respaldo — pero **solo produce 2 ejecuciones aceptadas** de **duración corta (0.26–0.31 s)** y **vmax baja (~0.29–4.2 m/s)**: el pie apenas participa en un puño. Las detecciones son **débiles y no representan el golpe** → `RTOE_FALLBACK_NOT_VALIDATED`.

## 9. B0400 como referencia

Único atleta nuevo cuyo S01-RFIN funciona de punta a punta (config → pipeline → 5 aceptadas limpias). Confirma que RFIN **puede** ser la señal correcta para S01, pero **no** que lo sea universalmente: n=1 dentro de 250 Hz (B0400) frente a 2 casos no segmentables (B0371/B0380) y 1 marginal (B0377).

## 10. B0371 y B0380 como casos problemáticos

En ambos, la mano (RFIN y LFIN) tiene baseline alto y actividad continua; el pipeline cae a RTOE por el gate de SNR, y RTOE produce detecciones débiles. Evidencia insuficiente para decir si el "golpe real" es separable con otro marker/ventana o si en estos videos E01-T01 el puño no tiene reposo limpio. No se puede distinguir del todo "señal vs segmentación": **señalan problema de representación/satisfación del criterio**, sin cambio posible hoy sin modificar el algoritmo.

## 11. B0377 como caso intermedio

RFIN ya marcado NEEDS_VALIDATION; los números confirman: snr 13.5 (pasa gate) pero baseline 599 → marginal, 1 aceptada. A diferencia de B0371/B0380, RTOE aquí sí es segmentable (4 acc) — B0377 es "casi": la señal pasa SNR pero no es limpia.

## 12. B0367 como referencia histórica

200 Hz, sin fallback, RFIN/LFIN/RTOE segmentables. Comparación cualitativa: magnitud (mm/s) no comparable por la frecuencia no normalizada.

## 13. Implicaciones para representación ML

S01 **no tiene hoy una representación consistente entre atletas**: RFIN en 2/3 nuevos (B0400 OK), fallback RTOE débil en 2, marginal en B0377. Un modelo con S01 captaría **la señal usada** (RFIN vs RTOE) como señal de discriminación espuria. **Conclusión: S01 no debe entrar aún al primer dataset ML.**

## 14. Decisión / recomendación

**Recomendación global determinista: Opción B** — mantener S01 como `NEEDS_VALIDATION` y **excluir S01 del primer dataset ML**, construyendo mésticamente el primer dataset con las técnicas con representación consistente (S02–S05: RTOE en 250 Hz, salvo B0380-S02→LTOE). No se elige por conveniencia: surge de 2/3 atletas nuevos con RFIN no segmentable + fallback RTOE no validado.

Reglas que generaron la decisión (script `11`): si ≥1 atleta nuevo con la señal configurada no segmentable/marginal y ninguna señal alternativa alcanza `segmentable` → **B**; si todas segmentable → A; si alguna alternativa segmentable → C; si no hay suficiente evidencia → D.

## 15. Limitaciones

- n=5, un único archivo E01-T01 por atleta; sin T02/E02/E03/E04.
- B0367 a 200 Hz (no normalizado): comparación solo cualitativa.
- `PROMISING_BUT_INCOMPLETE` en B0371/B0380 refleja que hay *alguna* detección (snr o acc≥1) sin alcanzar la regla completa; no implica que el golpe sea separable. Puede haber sobredetección de guardia/reposicionamiento.
- No se puede distinguir del todo "señal vs segmentación": para los casos de baseline alto se necesitaría una validación con umbral robusto/otra ventana, que **no** es parte de esta tarea.
- Sin normalización, sin ML.

## 16. Siguiente paso (no ejecutado)

Con esta evidencia, el siguiente paso razonable (para una tarea futura separada) sería: mantener S01 fuera del primer dataset ML y construir el dataset con S02–S05; en paralelo, investigar S01 con más trials/condiciones o un umbral dedicado por atleta, sin cambiar el algoritmo global.

## Tests

`pytest tests -q` → 116 previos + ~9 nuevos de Task 4 (ver sección 17 de `tests/test_pipeline.py`).

## Reproducibilidad

```bash
.venv\Scripts\python scripts\11_s01_targeted_validation.py
```