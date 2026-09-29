# TASK 11 — Comparación + Sugerencias para el entrenador (Coach Insights)

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-28
**Capa:** `dashboard/performance_analysis.py` · App: `dashboard/app_performance.py`
**Comando:** `streamlit run dashboard/app_performance.py`

---

## 1. Objetivo

Convertir el dashboard de "muestra lo que hizo el modelo" a "información útil para un entrenador": comparar la ejecución seleccionada con un **perfil de referencia observado** de su misma técnica, señalar diferencias descriptivas y ofrecer **posibles áreas de observación** — sin rankings, sin scores y sin causalidad.

## 2. Perfil de referencia (descriptivo)

Por técnica (S02–S05) y para las 10 features del modelo: **mediana (referencia visual)** y **Q1–Q3 (rango central observado)**, calculados sobre el ML Dataset v0 (419 ejecuciones). La referencia es **observada**, no óptima ni normativa.

> "The reference profile represents the observed distribution in the demo dataset. It is not an optimal or normative biomechanical standard."

## 3. Features

Las 10 del modelo (sin SNR): duración, tiempo-al-pico, vmax, vmean, amax, desplazamiento, longitud de trayecto, ROM cadera/rodilla/tobillo.

## 4. Cálculo de diferencias (respecto a la mediana de la técnica)

- `difference_abs = atleta − mediana_referencia`.
- `difference_pct = 100 · (atleta − mediana) / mediana`; **si la mediana es 0 → solo diferencia absoluta** (decisión documentada).
- Unidades originales (sin normalización nueva).

## 5. IQR / estado

Estado descriptivo frente al rango central observado:
`ABOVE / WITHIN / BELOW_REFERENCE_RANGE` (por encima / dentro / por debajo del rango). Sin términos good/bad/poor.

## 6. Reglas descriptivas

- Comparación **solo dentro de la misma técnica** (no S02 vs S04).
- **Diferencias observadas**: 2–4 features priorizadas automáticamente (fuera del rango, por |%Δ| o |Δ|), sin generar score.
- **Coach insights**: separación `Observación → Revisión`, sin causalidad y **sin usar feature importance** como consejo.

## 7. Coach Insights (reglas)

- Observación: "X se observa por encima/debajo de la mediana de referencia (−26 %)".
- Revisión: "Posible área de observación: revisar X durante el movimiento (inspección)".
- Se evita: "mala técnica", "necesita aumentar X", "causará menor rendimiento", "técnica incorrecta".

## 8. Limitaciones

- Referencia descriptiva de un demo (n por técnica ~94–118); sin normativa ni estándar.
- Comparaciones descriptivas; dos ejecuciones ≠ evolución temporal (no se usa "mejorado/empeorado").
- Sin ranking de atletas; sin diagnóstico automático ni recomendaciones de entrenamiento.

## 9. Ejemplos

- Predicción S04 (92 %) → comparación de vmax/amax/ROM de cadera contra la mediana Q1–Q3 de S04; diferencias observadas listadas; sugerencias como "revisar ROM de cadera durante el movimiento".
- Segunda ejecución del mismo atleta/técnica: diferencias absolutas/% entre ejecuciones (sin afirmar mejora).

## 10. Siguiente etapa

**Task 12 — Tablet polish / presentación**: refinar la UI con los datos longitudinales cuando estén disponibles; la capa `performance_analysis` queda lista para futuras sesiones.