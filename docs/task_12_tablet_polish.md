# TASK 12 — Tablet Polish / Presentación de producto

**Proyecto:** Karate Athlete Performance Intelligence
**Fecha:** 2026-09-28
**App:** `dashboard/app_performance.py` (solo PRESENTATION)
**Comando:** `streamlit run dashboard/app_performance.py`

---

## Objetivo

Convertir el dashboard en una **presentación visual de producto** orientada a tablet para la federación. Solo UX/UI; **no** se tocó modelo, dataset, resultados ML, `inference/` ni `performance_analysis.py`.

## Cambios visuales

- **Header**: «KARATE PERFORMANCE INTELLIGENCE» + subtítulo «Análisis biomecánico asistido por inteligencia artificial» + chip **DEMO**.
- **Barra de flujo A–D**: `A · EJECUCIÓN → B · ANÁLISIS → C · COMPARACIÓN → D · ENTRENADOR` (badges; experiencia continua, no oculta secciones).
- **Hero dominante**: técnica grande + nombre + **confianza con barra** + frase «El modelo identifica esta ejecución como S0X». Se eliminó del hero el texto técnico del modelo.
- **Selector compacto** (1 fila): Atleta · Ejecución · Técnica observada · Condición (E01·T01 derivada de `execution_id`).
- **Perfil biomecánico**: tarjetas agrupadas (Ejecución / Velocidad y aceleración / Movimiento / Movilidad articular) con nombres legibles y unidades.
- **Comparación**: conserva la lógica de `performance_analysis`; se muestran mini-barras de proporción (Atleta / Ref. mediana / Rango observado / estado).
- **Diferencias observadas**: bullets con % (desde `observed_differences`).
- **Coach Insights**: «Observaciones para el entrenador» (Observación + Posible área de observación), lenguaje neutro.
- **Ejecución vs ejecución**: tabla compacta (Variable/Actual/Otra/Diferencia) con «se observan diferencias descriptivas» (sin mejoró/empeoró).
- **EVOLUCIÓN DEL ATLETA**: tarjeta «Futura función — seguimiento longitudinal» (sin datos ficticios).
- **Detalles del modelo** (colapsable) con «Qué ve el modelo» + toda la info técnica.
- **Disclaimer**: «Demo tecnológica… predicciones y comparaciones descriptivas; no diagnóstico ni predicción de rendimiento competitivo».

## Cambios visuales (mejoras 1–3, Request speciale)

1. **Barra de flujo con recuadros y flechas**: en lugar de texto plano, cada paso (A–D) es un **recuadro redondeado de color propio** (A `#1f4e79`, B `#2e6fb3`, C `#c98a1b`, D `#5b8db8`) y entre ellos una **flecha corta y ancha `➜`** que apunta al siguiente paso (fluía visual A→B→C→D).
2. **Wireframe GIF por atleta × técnica (galería)**: se reutiliza el generador (script 17 refactorizado → `make_wireframe_gif`) y el nuevo `scripts/22_build_athlete_gallery.py` genera **129 GIF** (33 atletas × S02–S05, omitiendo celdas sin ejecución aceptada) en **`output/gallery/athlete_<id>_<S0X>.gif`** con:
   - **puntos en los marcadores** del mismo color de las líneas (“puntos unidos”);
   - **trayectoria del marcador efectuador en rojo** (señal configurada por atleta×técnica → puño RFIN/LFIN en S01, dedo RTOE/LTOE en patadas);
   - **grid estándar fijo GLOBAL del lote** (dos pasadas) para que ninguna imagen “crezca ni decrezca”.
   En el dashboard, el **GIF se muestra a la derecha** de la zona superior (junto a Atleta › Ejecución › Técnica observada › Condición). Dashboard **no lee C3D** (solo la galería pre-generada).
3. **Foto de técnica con resaltado suave**: en la sección A, a la derecha de la predicción, se muestra la **foto de la técnica observada** (`images/technique_<S0X>_<nombre>.jpg`) con un **pulso suave** (opacidad 0.3→1→0.3 ×3, ~0.9 s) que se dispara al cambiar de ejecución, comunicando “este es el resultado”.

## Arquitectura

`DATA (performance_data) → INFERENCE (inference) → ANALYSIS (performance_analysis) → PRESENTATION (app_performance + performance_ui)`. Task 12 solo tocó PRESENTATION.

## Vistas / secciones

Ejecución (A) → Análisis (B) → Comparación (C) → Entrenador (D, con evolución futura) → Detalles del modelo → Disclaimer.

## Diseño tablet

Texto grande, controles grandes, poco texto por bloque, tarjetas y barras, sin scroll horizontal, sin tablas densas. CSS localizado (`<style>` en la app) documentado.

## Comparación / Coach Insights / Evolución longitudinal futura

Ver Task 11 (`performance_analysis.py`, intacta). La evolución longitudinal se representa solo como concepto («Futura función»).

## Integridad

Antes/después: `ml_dataset_v0/`, `ml_results/`, `ml_results_task8b/`, `ml_inference/`, `inference/` — hashes sin cambios. `performance_analysis.py`, `app.py`, `app_ml.py` y scripts 02–21 intactos.

## Tests

`pytest -k "test_analy or test_performance or test_inference"` (ver README). No se importa sklearn en presentación; lógica de datos/análisis/inferencia sin cambios.

## Limitaciones

- Solo presentación; la app sigue siendo un demo (confianza ≠ éxito deportivo).
- Sin datos longitudinales reales; evolución = concepto.
- La UI adaptada a tablet horizontal (Streamlit responsive); el pulido fino de dispositivo específico queda para etapas posteriores.

## Qué NO se implementó

Rankings, scores, diagnóstico, recomendaciones automáticas, evolución ficticia, nuevos modelos/features/normalización, integraciones, autenticación/API.

## Siguiente etapa

Cuando existan sesiones longitudinales reales: Task 13 podría materializar la evolución del atleta reutilizando `performance_analysis` sin tocar la capa de inferencia.