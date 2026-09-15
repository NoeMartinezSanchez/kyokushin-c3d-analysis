# FASE 1.6.1 — Validación de señales de segmentación para S03 y S05

**Proyecto:** Karate Athlete Performance Intelligence
**Subconjunto:** B0367 (solo este atleta)
**Fecha:** 2026-09-13
**Scripts:** `scripts/03_signal_validation.py`, `scripts/02_execution_segmentation.py`
**Config:** `config/segmentation.yaml` (actualizado a versión 1.6.1)
**Salidas:** `output/signal_validation/signal_validation_candidates.csv`, `output/signal_validation/signal_validation_S0X_*.png`

---

## 1. Objetivo

Determinar experimentalmente la mejor señal (marcador) para representar la ejecu­ción temporal de **S03 = Mawashi-Geri gedan** y **S05 = Ushiro-Mawashi-Geri** en B0367, para permitir la segmentación inicio→aceleración→pico→retorno. El criterio NO es únicamente SNR (la Fase 1.5 mostró SNR engañosa con actividad basal ~0).

## 2. Archivos S03 analizados

| Condición | Trial | source_file | rate (Hz) | frames | markers | tarcza |
|-----------|-------|-------------|-----------|--------|---------|--------|
| E01 | T01 | 2017-01-31-B0367-S03-E01-T01.c3d | 200 | 1875 | B0367 (+93 deriv) | 0 |
| E01 | T02 | 2017-01-31-B0367-S03-E01-T02.c3d | 200 | 1775 | B0367 (+93 deriv) | 0 |
| E02 | T01 | 2017-01-31-B0367-S03-E02-T01.c3d | 200 | 1768 | Tarcza+B0367 | 6 |
| E02 | T02 | 2017-01-31-B0367-S03-E02-T02.c3d | 200 | 1883 | Tarcza+B0367 | 6 |

## 3. Archivos S05 analizados

| Condición | Trial | source_file | rate (Hz) | frames | markers | tarcza |
|-----------|-------|-------------|-----------|--------|---------|--------|
| E01 | T01 | 2017-01-31-B0367-S05-E01-T01.c3d | 200 | 2334 | B0367 (+93 deriv) | 0 |
| E01 | T02 | 2017-01-31-B0367-S05-E01-T02.c3d | 200 | 1912 | B0367 (+93 deriv) | 0 |
| E02 | T01 | 2017-01-31-B0367-S05-E02-T01.c3d | 200 | 2370 | Tarcza+B0367 | 6 |
| E02 | T02 | — (ausente en submuestra local) | — | — | — | — |

E04 de S03 existe (con B0368) pero se excluye de esta validación por ser condición de oponente (no comparable a E01/E02 para señal de técnica individual).

## 4. Candidatos evaluados

`RTOE, RANK, RHEE, RKNE, RTHI, RHJC` (extremidad derecha — la pierna dominante de B0367). Se incluyeron rodilla, muslo y cadera como candidatos biomecánicamente relevantes, no solo los tres endpoints.

## 5. Método de evaluación

Se evaluó por archivo: amplitud de movimiento (ROM), vmax, vmean, amax, SNR, actividad basal, nº de picos candidatos (>30 % del rango), separación temporal entre picos, y el resultado de la segmentación del pipeline (aceptadas/rechazadas/review). Después se agrupó por técnica y se verificó la **consistencia entre trials** (criterio decisivo). No se usó SNR como único criterio.

## 6. Resultados cuantitativos

Resumen por técnica (media sobre todos los trials E01+E02 disponibles):

| técnica | marker | vmax (mm/s) | SNR | baseline (mm/s) | picos candidatos | aceptadas | rechazadas | review | sep. media (s) |
|---------|--------|------------|-----|-----------------|------------------|-----------|------------|--------|----------------|
| S03 | **RTOE** | 13378 | **735** | 20.7 | 3.5 | **3.00** | 1.25 | 0.00 | 1.62 |
| S03 | RANK | 10697 | 286 | 40.5 | 4.0 | 3.00 | 0.75 | 0.00 | 1.42 |
| S03 | RHEE | 11130 | 317 | 39.8 | 3.75 | 3.00 | 1.00 | 0.00 | 1.57 |
| S03 | RKNE | 6366 | 38 | 169.6 | 6.0 | 3.75 | 3.50 | 1.75 | 0.66 |
| S03 | RTHI | 4398 | 25 | 178.6 | 6.0 | 3.00 | 2.75 | 1.50 | 0.69 |
| S03 | RHJC | 1960 | 9 | 225.2 | 6.0 | 4.50 | 3.75 | 0.25 | 0.92 |
| S05 | **RTOE** | 10861 | **579** | 22.9 | 4.67 | **3.00** | 3.33 | 0.33 | 1.97 |
| S05 | RANK | 9781 | 335 | 35.9 | 4.33 | 3.00 | 3.67 | 0.33 | 1.97 |
| S05 | RHEE | 10611 | 368 | 37.3 | 4.33 | 3.00 | 4.33 | 0.33 | 1.97 |
| S05 | RKNE | 6410 | 32 | 201.4 | 6.67 | 3.33 | 2.00 | 0.33 | 0.57 |
| S05 | RTHI | 4611 | 20 | 226.3 | 6.67 | 3.33 | 4.33 | 0.00 | 0.57 |
| S05 | RHJC | 2681 | 10 | 261.4 | 7.67 | 5.67 | 3.67 | 1.00 | 0.80 |

Observaciones:
- **RTOE es el candidato con la SNR más alta y la menor actividad basal** en ambas técnicas, y presenta la **menor cantidad de picos espurios** para S03 (3.5 vs 4.0 RANK).
- **RKNE/RTHI/RHJC** tienen actividad basal elevada (170–260 mm/s) y producen demasiados picos (6–8) → poco discriminativos y con separación temporal reducida (0.6–0.9 s); no son adecuados como señal primaria.
- La **amplitud (vmax) de S03-E02 es mayor que S03-E01** (13–16 m/s vs 10 m/s) — esperable al golpear el escudo; la detectabilidad temporal se mantiene estable.

## 7. Resultados visuales

Figuras comparativas RTOE/RANK/RHEE generadas en:

- `output/signal_validation/signal_validation_S03_E01-T01.png`
- `output/signal_validation/signal_validation_S03_E02-T01.png`
- `output/signal_validation/signal_validation_S05_E01-T01.png`
- `output/signal_validation/signal_validation_S05_E02-T01.png`

Cada figura muestra señal suavizada (m/s), umbral de actividad, picos candidatos y bandas de ejecución aceptadas. La inspección confirma que RTOE produce bandas nítidas con un pico dominante por ejecución y separación visible.

## 8. Comparación entre trials (criterio decisivo)

Verificado con RTOE en **todos** los trials disponibles:

| caso | aceptadas | rechazadas | review | duraciones (s) |
|------|-----------|------------|--------|----------------|
| S03-E01-T01 | 3 | 0 | 0 | 1.03 / 0.86 / 0.96 |
| S03-E01-T02 | 3 | 1 | 0 | 0.82 / 0.76 / 1.00 |
| S03-E02-T01 | 3 | 1 | 0 | 0.67 / 0.74 / 0.73 |
| S03-E02-T02 | 3 | 3 | 0 | 0.74 / 0.72 / 0.74 |
| S05-E01-T01 | 3 | 2 | 0 | 0.77 / 0.74 / 0.78 |
| S05-E01-T02 | 3 | 3 | 0 | 0.70 / 0.76 / 0.77 |
| S05-E02-T01 | 3 | 5 | 1 | 0.98 / 1.14 / 1.38 |

**RTOE detecta 3 ejecuciones en los 7 trials evaluados** (consistencia 7/7). Las duraciones son estables dentro de cada técnica. No se declaró validado ningún candidato que funcionara solo en un trial.

## 9. Comparación E01 (aire) vs E02 (escudo)

- En **E01** la señal es muy limpia (0–1 rechazo).
- En **E02** aumentan los rechazos por picos secundarios de *retorno tras el impacto* (S03: 1–3; S05: 5 + 1 review). Esto es comportamentalmente esperable al golpear el escudo (rebote/retorno de la pierna) y NO es un fallo de la señal: el detector los clasifica como `secondary_peak`.
- La **temporalidad** (3 picos por archivo, picos separados ~1.6–2.8 s) se mantiene consistente entre aire y escudo.
- No se exige que las amplitudes sean iguales entre condiciones; se evalúa detectabilidad/temporalidad/consistencia.

## 10. Señal seleccionada para S03

**PRIMARY_SIGNAL = RTOE** | secondary = RHEE
- Mayor SNR (735), menor baseline (20.7 mm/s), menor nº de picos espurios.
- 3/3 ejecuciones en los 4 trials E01/E02, duraciones estables.
- RTOE es el endpoint distal real de la patada gedan; proxy razonable de la fase de extensión.

## 11. Señal seleccionada para S05

**PRIMARY_SIGNAL = RTOE** | secondary = RANK
- Mayor SNR (579), baseline baja (22.9 mm/s).
- 3/3 ejecuciones en los 3 trials disponibles, duraciones estables.
- Ushiro-Mawashi-Geri es una patada circular; RTOE (punta del pie) captura el gesto de la pierna ejecutora.

## 12. Señales descartadas y por qué

| Señal | Motivo |
|-------|--------|
| **RKNE** | Baseline alta (~170–200 mm/s), 6–7 picos candidatos, separación baja (0.57–0.66 s) → poca discriminación; produjo 1.75 review en S03. |
| **RTHI** | Baseline alta (~180–226 mm/s), múltiples picos espurios. |
| **RHJC** | Baseline muy alta (~225–261 mm/s), SNR ~10, 6–8 picos, sobre-segmenta (aceptó 4.5 y 5.7 de media) y generó reviews. |
| **RANK/RHEE** (como primarias) | Funcionan, pero SNR menor y más picos candidatos que RTOE; se mantienen como secundarias de respaldo. |

## 13. Limitaciones

- Validación técnica/exploratoria, NO científico-biomecánica. "Señal con mejor detectabilidad", **no** "inicio biomecánico exacto de la técnica".
- El pico de velocidad de RTOE se usa como **proxy** de la extensión; no debe confundirse con el instante de impacto.
- n=1 (B0367) → las señales son decisión técnica para este atleta, no regla universal.
- S05-E02-T02 y todo E03 no están en la submuestra local.

## 14. Qué quedó NO VALIDADO

- No se validó ningún endpoint **izquierdo**: la pierna dominante de B0367 es la derecha (confirmado por RTOE vs LTOE ~4× en vmax y baseline). Para atletas zurdos la lateralidad debe parametrizarse (regla ya documentada en AGENTS.md).
- La confirmación de con **quién golpea S05-E02** (1 review por retorno) queda como evento `review`, pendiente de revisión humana si se usa ese trial con fines comparativos.

## 15. Cambios en config/segmentation.yaml

- `segmentation_version: 1.6.0 → 1.6.1`
- `signals.S03: signal TBD → RTOE` (+ secondary RHEE)
- `signals.S05: signal TBD → RTOE` (+ secondary RANK)
- Comentario documentando que la selección es específica de B0367 (no universal).
- No se hardcodearon señales en `02_execution_segmentation.py` (el script lee `config/segmentation.yaml`; se añadieron S03/S05 a la lista `selected` de `main()` para integrarlos al pipeline).

## 16. Tests ejecutados

`pytest tests -q` → **16 passed** (11 previos + 5 nuevos):
- `test_s03_signal_defined`
- `test_s05_signal_defined`
- `test_signals_come_from_yaml`
- `test_signals_not_hardcoded_in_script`
- `test_config_loadable`

## 17. Resultado final del pipeline

`.venv\Scripts\python scripts/02_execution_segmentation.py` → exit=0, 0 errores.

- **26 ejecuciones aceptadas** de B0367 (antes 17): +3 S03-E01, +3 S05-E01, +3 S05-E02.
- **45 eventos** (accepted → 26, rejected → 17, review → 2). *(Nota Fase 1.6.2: corregido — el "16" previo era un error tipográfico.)*
- QC: 26/26 válidas, cobertura 100 %, sin NaN/inf en features cinemáticas.
- **Sin regresiones** en S01/S02/S04 (intervalos idénticos a la Fase 1.6).
- Figuras de validación generadas para S03 y S05 en `output/phase_validation/`.

## 18. Resumen ejecutivo

- **A. Señal S03:** RTOE (secundaria RHEE).
- **B. Señal S05:** RTOE (secundaria RANK).
- **C. Candidatos descartados:** RKNE, RTHI, RHJC (baseline alta y sobre-segmentación); RANK/RHEE funcionan pero quedan como secundarias.
- **D. Ejecuciones ahora:** S03 = 3 (E01-T01), S05 = 3 (E01-T01) + 3 (E02-T01). Total pipeline = 26.
- **E. Regresiones en S01/S02/S04:** ninguna (intervalos idénticos).
- **F. NO VALIDADO:** ninguna señal izquierda (lateralidad pendiente de parametrizar); un evento review en S05-E02-T01.
- **G. ¿Listo para otro atleta?** El pipeline es reproducible; sigue pendiente la parametrización de lateralidad/endpoint por atleta y el procesamiento del dataset completo.
- **H. Cambios en AGENTS.md:** ninguno necesario (la regla de "configuración centralizada, no hardcodear señales" ya existe y se cumplió).