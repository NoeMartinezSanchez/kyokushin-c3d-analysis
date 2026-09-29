# -*- coding: utf-8 -*-
"""
app_performance.py — Athlete Performance Dashboard (Task 10, demo).

Producto de demostración para la federación: selecciona una ejecución real
del ML Dataset v0, ejecuta la capa de inferencia y muestra la predicción de
técnica + perfil biomecánico. Tablet-first, en castellano.

Flujo:  EXECUTION -> FEATURES -> INFERENCE -> TECHNIQUE PREDICTION -> PROFILE

La única entrada al modelo es `inference.predict_execution`; NO hay sklearn,
NO lectura de C3D, NO segmentación, NO entrenamiento.

Lanzamiento:
    streamlit run dashboard/app_performance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import predict_execution, model_info  # noqa: E402
from inference.schemas import CLASSES, InferenceError  # noqa: E402
from performance_data import (available_athletes, default_execution,  # noqa: E402
                              executions_for, execution_row, features_of, load_v0)
from performance_ui import (GROUPS, MODEL_INPUT_NAMES, FEATURE_UNITS,  # noqa: E402
                            format_metric, metric_card_html, prob_bar)
from performance_analysis import (reference_profile, compare_to_reference,  # noqa: E402
                                  compare_executions, observed_differences,
                                  coach_insights)

st.set_page_config(page_title="KARATE PERFORMANCE INTELLIGENCE",
                   page_icon="🥋", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
h1 {color: #1f4e79;}
[data-testid="stSelectbox"] > div > div {min-height: 3.2rem;}
</style>
""", unsafe_allow_html=True)


def _status_text(status: str) -> str:
    return {
        "ABOVE_REFERENCE_RANGE": "por encima del rango de referencia",
        "BELOW_REFERENCE_RANGE": "por debajo del rango de referencia",
        "WITHIN_REFERENCE_RANGE": "dentro del rango de referencia",
    }.get(status, status)


def _render_comparison(comp: list) -> None:
    key_features = ["vmax", "vmean", "amax", "duration_s", "hip_rom",
                    "knee_rom", "ankle_rom"]
    for item in comp:
        if item["feature"] not in key_features:
            continue
        unit = FEATURE_UNITS.get(item["feature"], "")
        st.markdown(
            f"**{item['label']}** · Atleta "
            f"{format_metric(item['athlete'], unit)} · Ref. mediana "
            f"{format_metric(item['reference_median'], unit)} · "
            f"(rango {format_metric(item['reference_q1'], unit)} – "
            f"{format_metric(item['reference_q3'], unit)})"
            f" — {_status_text(item['status'])}")


def render_prediction_card(res: dict):
    """Tarjeta principal: técnica predicha + confidence."""
    st.markdown("### Technique Prediction")
    st.markdown(
        f"<div style='border:2px solid #1f4e79;border-radius:14px;"
        f"padding:1.1rem 1.3rem;background:#f2f6fb;'>"
        f"<div style='font-size:3rem;font-weight:700;color:#1f4e79;'>"
        f"{res['predicted_technique']}</div>"
        f"<div style='font-size:1.1rem;color:#4a5a6a;'>{res['technique_name']}</div>"
        f"<div style='font-size:1.6rem;font-weight:600;color:#2e6fb3;'>"
        f"{res['confidence'] * 100:.1f} %</div>"
        f"<div style='font-size:.85rem;color:#4a5a6a;'>Confidence</div></div>",
        unsafe_allow_html=True)
    st.caption("Modelo: Random Forest — Demo Baseline")
    st.caption("“Confidence” representa la probabilidad entregada por el "
               "clasificador; no es una probabilidad validada de éxito "
               "deportivo.")


def render_probabilities(res: dict):
    st.markdown("### Probabilidades por técnica")
    for c in CLASSES:
        v = res["probabilities"][c]
        st.markdown(f"**{c}** · {v * 100:5.1f} %")
        st.markdown(f"`{prob_bar(v)}`")
    st.caption("Identificación probabilística de la técnica (clasificador).")


def render_profile(res: dict):
    st.markdown("## Perfil biomecánico")
    cols = st.columns(4)
    for idx, (group_title, items) in enumerate(GROUPS):
        with cols[idx % 4]:
            st.markdown(f"**{group_title}**")
            for f, label, unit in items:
                val = res["features"][f]
                st.markdown(metric_card_html(label, format_metric(val, unit)),
                            unsafe_allow_html=True)


def main():
    df = load_v0()

    c_left, c_right = st.columns([1, 2])
    with c_left:
        athletes = available_athletes()
        sel_athlete = st.selectbox("Atleta", athletes)
    execs = executions_for(sel_athlete)
    default = default_execution() if sel_athlete else None
    if default not in execs:
        default = execs[0] if execs else None
    with c_right:
        sel_exec = st.selectbox("Ejecución", execs,
                                index=execs.index(default) if default in execs else 0,
                                format_func=lambda eid: eid)

    row = execution_row(sel_exec)
    ref_tech = str(row["technique"])

    feats = features_of(row)
    try:
        res = predict_execution(feats)
    except InferenceError as e:
        st.error(f"Error de inferencia: {e}")
        return

    # Identificación
    st.markdown("---")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Atleta", row["athlete_id"])
    i2.metric("Ejecución", row["execution_id"])
    i3.metric("Técnica de referencia", ref_tech)
    i4.metric("Técnica predicha", res["predicted_technique"])

    # Indicador neutro de coincidencia
    if res["predicted_technique"] == ref_tech:
        st.success("La predicción **coincide** con la técnica de referencia.")
    else:
        st.info("La predicción **difiere** de la técnica de referencia "
                "(resultado válido del demo).")

    # Main view
    st.markdown("---")
    col_pred, col_prob, col_prof = st.columns([1, 1, 1])
    with col_pred:
        render_prediction_card(res)
    with col_prob:
        render_probabilities(res)

    st.markdown("---")
    render_profile(res)

    # ================= COMPARACIÓN vs PERFIL DE REFERENCIA =================
    profiles = reference_profile(df)
    ref = profiles.get(ref_tech, {})
    comp = compare_to_reference(res["features"], ref) if ref else []
    st.markdown("## Comparación con el perfil de referencia")
    st.caption(f"Perfil observado en el dataset del demo para la técnica "
               f"{ref_tech} (mediana y rango central Q1–Q3). No es un estándar "
               f"normativo ni óptimo.")
    _render_comparison(comp)

    outside = observed_differences(comp)
    if outside:
        st.markdown("### Diferencias observadas")
        for o in outside:
            st.markdown(f"- **{o['label']}** — {_status_text(o['status'])}")

    insights = coach_insights(outside)
    if insights:
        st.markdown("### Sugerencias para el entrenador")
        for it in insights:
            st.markdown(f"- **Observación:** {it['observation']}")
            st.caption(it["coach_review"])

    # ================= COMPARAR CON OTRA EJECUCIÓN =================
    with st.expander("Comparar con otra ejecución (misma técnica)"):
        same = [e for e in executions_for(str(row["athlete_id"]))
                if e != sel_exec and str(execution_row(e)["technique"]) == ref_tech]
        if same:
            other = st.selectbox("Otra ejecución", same)
            row2 = execution_row(other)
            diffs = compare_executions(feats, features_of(row2))
            st.markdown(f"**{sel_exec}** vs **{other}**")
            st.markdown("Diferencias (ejecución B − ejecución A):")
            for d in diffs:
                pct = (f" ({d['difference_pct']:+.0f} %)"
                       if d["difference_pct"] is not None else "")
                st.markdown(f"- {d['label']}: {d['difference_abs']:+.3f}{pct}")
        else:
            st.caption("No hay otra ejecución de la misma técnica para este atleta.")

    st.markdown("### FUTURE: ATHLETE EVOLUTION")
    st.caption("Con sesiones de entrenamiento longitudinales, este perfil podría "
               "extenderse para monitorizar cómo evolucionan las características "
               "de ejecución a lo largo del tiempo. (Concepto — sin datos aún.)")

    c_what, c_model = st.columns(2)
    with c_what:
        st.markdown("## Qué ve el modelo")
        st.caption("La predicción se genera a partir de características "
                   "biomecánicas de ejecución extraídas del movimiento.")
        st.markdown(" · ".join(MODEL_INPUT_NAMES.values()))
    with c_model:
        with st.expander("Información del modelo"):
            info = model_info()
            st.write(f"**Modelo:** Random Forest")
            st.write(f"**Versión:** {info.get('model_version', 'n/d')}")
            st.write(f"**Versión de inferencia:** {info.get('inference_version', 'n/d')}")
            st.write(f"**Features:** {len(info.get('features', []))}"
                     f" · **Clases:** {', '.join(info.get('classes', []))}")
            st.write("**Dataset:** ML Dataset v0")
            st.write("**Validación:** GroupKFold por atleta")

    st.markdown("---")
    st.caption("Demo basado en datos biomecánicos de movimiento. Este "
               "prototipo demuestra la inferencia de patrones de técnica y "
               "no predice todavía rendimiento competitivo ni resultados "
               "deportivos.")


if __name__ == "__main__":
    main()